using CairoMakie
using ComponentArrays: ComponentArray
using JLD2
using MLUtils
using Optimisers
using Random
using SciMLSensitivity
using Statistics
using Zygote

include("TestNODEModel.jl")

const TRAIN_RESULTS_DIR = joinpath(@__DIR__, "results")
const TRAIN_MODEL_DIR = joinpath(@__DIR__, "model")
const DEFAULT_SEGMENTS_PER_BATCH = 512
const DEFAULT_PRETRAIN_ONLY_EPOCHS = 300
const DEFAULT_INITIAL_LR = 0.01f0
const DEFAULT_LR_DECAY_FACTOR = 0.5f0
const DEFAULT_LR_DECAY_EVERY = 25
const DEFAULT_MIN_LR = 1f-4

function ensure_train_dirs!()
    mkpath(TRAIN_RESULTS_DIR)
    mkpath(TRAIN_MODEL_DIR)
end

function derivative_segment_loss(
    ps,
    model,
    st,
    sample::Array{Float32, 3},
    dt::Float32;
    graph=model.base_graph,
)
    size(sample, 3) == 2 || throw(ArgumentError("window sample must contain exactly two time steps"))
    truth = sample
    segment_times = Float32[0.0f0, dt]
    x0 = truth[:, :, 1]
    sensealg = InterpolatingAdjoint(; autojacvec=ZygoteVJP())
    prediction, _, _ = rollout(model, x0, segment_times, ps, st; graph, sensealg)

    truth_next = truth[:, :, 2]
    pred_next = prediction[:, :, 2]
    return mean(abs.(pred_next .- truth_next))
end

function derivative_batch_loss(
    ps,
    model,
    st,
    batch_sample::Array{Float32, 3},
    batch_graph,
    dt::Float32,
)
    return Float32(derivative_segment_loss(ps, model, st, batch_sample, dt; graph=batch_graph))
end

function collate_merged_scenarios(samples, base_graph)
    return (
        sample = cat(samples...; dims=2),
        graph = disconnected_graph(base_graph, length(samples)),
    )
end

function build_pair_dataset(data::Array{Float32, 4})
    scenario_count = size(data, 4)
    num_steps = size(data, 3)
    observations = Array{Float32, 3}[]
    sizehint!(observations, scenario_count * (num_steps - 1))
    for scenario_idx in 1:scenario_count
        for start_idx in 1:(num_steps - 1)
            push!(observations, copy(@view data[:, :, start_idx:(start_idx + 1), scenario_idx]))
        end
    end
    return observations
end

function plot_pretrain_loss(loss_trace; output_path=joinpath(TRAIN_RESULTS_DIR, "pretrain_loss_curve.png"))
    fig = Figure(size=(1000, 500))
    ax = Axis(fig[1, 1]; xlabel="iteration", ylabel="loss", yscale=log10, title="Pretrain loss")
    if !isempty(loss_trace)
        lines!(ax, 1:length(loss_trace), loss_trace, color=:darkgreen, linewidth=2, label="Pretrain")
        axislegend(ax; position=:rt)
    end
    save(output_path, fig)
    return output_path
end

function compute_normalization_stats(data::Array{Float32, 4})
    x_mean = mean(data; dims=(2, 3, 4))
    x_std = std(data; dims=(2, 3, 4))
    x_std = max.(x_std, 1f-3)
    return Float32.(x_mean), Float32.(x_std)
end

function normalize_data(data::Array{Float32, 4}, x_mean::Array{Float32, 4}, x_std::Array{Float32, 4})
    return (data .- x_mean) ./ x_std
end

function save_pretrain_snapshot!(
    weights_path::String,
    best_flat,
    best_loss::Float32,
    best_epoch::Int,
    x_mean,
    x_std,
)
    @save weights_path best_flat best_loss best_epoch x_mean x_std
end

function scheduled_lr(
    epoch::Int;
    initial_lr::Float32=DEFAULT_INITIAL_LR,
    decay_factor::Float32=DEFAULT_LR_DECAY_FACTOR,
    decay_every::Int=DEFAULT_LR_DECAY_EVERY,
    min_lr::Float32=DEFAULT_MIN_LR,
)
    decay_step = (epoch - 1) ÷ max(decay_every, 1)
    return max(min_lr, initial_lr * (decay_factor ^ decay_step))
end

function pretrain_node(;
    dataset_path=joinpath(@__DIR__, "data", "data.jld2"),
    seed::Int=20260302,
    epochs::Int=DEFAULT_PRETRAIN_ONLY_EPOCHS,
    segments_per_batch::Int=DEFAULT_SEGMENTS_PER_BATCH,
    max_scenarios::Union{Nothing, Int}=nothing,
    initial_lr::Float32=DEFAULT_INITIAL_LR,
    decay_factor::Float32=DEFAULT_LR_DECAY_FACTOR,
    decay_every::Int=DEFAULT_LR_DECAY_EVERY,
    min_lr::Float32=DEFAULT_MIN_LR,
)
    ensure_train_dirs!()
    rng = Random.Xoshiro(seed)
    data, times = load_dataset(dataset_path)
    scenario_count = max_scenarios === nothing ? size(data, 4) : min(size(data, 4), max_scenarios)
    data = Float32.(data[:, :, :, 1:scenario_count])
    x_mean, x_std = compute_normalization_stats(data)
    data = normalize_data(data, x_mean, x_std)
    dt = Float32(times[2] - times[1])
    observations = build_pair_dataset(data)
    model, ps0, st, _ = setup_model(rng)
    flat_ps = ComponentArray(ps0)
    batch_size = min(segments_per_batch, length(observations))

    current_lr = scheduled_lr(1; initial_lr, decay_factor, decay_every, min_lr)
    opt = Optimisers.AdamW(current_lr)
    opt_state = Optimisers.setup(opt, flat_ps)
    loss_trace = Float32[]
    best_loss = Inf32
    best_epoch = 0
    best_flat = deepcopy(flat_ps)
    weights_path = joinpath(TRAIN_MODEL_DIR, "pretrain_model_weights.jld2")

    for epoch in 1:epochs
        next_lr = scheduled_lr(epoch; initial_lr, decay_factor, decay_every, min_lr)
        if next_lr != current_lr
            current_lr = next_lr
            Optimisers.adjust!(opt_state, current_lr)
        end
        collate = samples -> collate_merged_scenarios(samples, model.base_graph)
        loader = DataLoader(observations; batchsize=batch_size, collate, shuffle=true, partial=true, rng=rng)
        epoch_losses = Float32[]

        for batch in loader
            loss, grad = Zygote.withgradient(flat_ps) do x
                derivative_batch_loss(x, model, st, batch.sample, batch.graph, dt)
            end
            opt_state, flat_ps = Optimisers.update(opt_state, flat_ps, grad[1])
            push!(epoch_losses, Float32(loss))
        end

        epoch_loss = mean(epoch_losses)
        push!(loss_trace, epoch_loss)
        if epoch_loss < best_loss
            best_loss = epoch_loss
            best_epoch = epoch
            best_flat = deepcopy(flat_ps)
            save_pretrain_snapshot!(weights_path, best_flat, best_loss, best_epoch, x_mean, x_std)
        end

        if epoch % 5 == 0 || epoch == 1 || epoch == epochs
            println("Pretrain-only epoch $(epoch)/$(epochs): lr=$(round(current_lr, digits=6)) loss=$(round(epoch_loss, digits=6))")
        end
    end

    return (
        loss_trace=loss_trace,
        weights_path=weights_path,
        best_loss=best_loss,
        best_epoch=best_epoch,
        trained_scenarios=scenario_count,
    )
end

function test_pretrained_model(weights_path::String; dataset_path=joinpath(@__DIR__, "data", "data.jld2"))
    return compare_single_trajectory(;
        dataset_path,
        weights_path,
        output_name="pretrained_best_node_rollout_comparison.png",
    )
end

function main()
    epochs = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : DEFAULT_PRETRAIN_ONLY_EPOCHS
    segments_per_batch = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : DEFAULT_SEGMENTS_PER_BATCH
    max_scenarios = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing

    training = pretrain_node(; epochs, segments_per_batch, max_scenarios)
    loss_path = plot_pretrain_loss(training.loss_trace)
    test_result = test_pretrained_model(training.weights_path)

    println("Saved pretrain weights to $(training.weights_path)")
    println("Saved pretrain loss curve to $(loss_path)")
    println("Saved pretrain test plot to $(test_result.output_path)")
    println("Best pretrain loss = $(training.best_loss)")
    println("Best epoch = $(training.best_epoch)")
    println("Trained scenarios = $(training.trained_scenarios)")
    println("MAE by feature = $(test_result.mae_by_feature)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
