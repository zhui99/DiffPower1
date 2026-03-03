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

const MULTISTEP_RESULTS_DIR = joinpath(@__DIR__, "results")
const MULTISTEP_MODEL_DIR = joinpath(@__DIR__, "model")
const DEFAULT_MULTISTEP_BATCH = 512
const DEFAULT_MULTISTEP_EPOCHS = 100
const DEFAULT_PREDICTION_HORIZON = 3
const DEFAULT_INITIAL_LR = 0.01f0
const DEFAULT_LR_DECAY_FACTOR = 0.5f0
const DEFAULT_LR_DECAY_EVERY = 25
const DEFAULT_MIN_LR = 1f-4

function ensure_multistep_dirs!()
    mkpath(MULTISTEP_RESULTS_DIR)
    mkpath(MULTISTEP_MODEL_DIR)
end

function multistep_window_loss(
    ps,
    model,
    st,
    sample::Array{Float32, 3},
    dt::Float32;
    graph=model.base_graph,
)
    horizon = size(sample, 3) - 1
    horizon >= 1 || throw(ArgumentError("window sample must contain at least two time steps"))
    segment_times = Float32.(collect(0:horizon) .* dt)
    x0 = sample[:, :, 1]
    sensealg = InterpolatingAdjoint(; autojacvec=ZygoteVJP())
    prediction, _, _ = rollout(model, x0, segment_times, ps, st; graph, sensealg)
    truth_future = @view sample[:, :, 2:end]
    pred_future = @view prediction[:, :, 2:end]
    return mean(abs.(pred_future .- truth_future))
end

function multistep_batch_loss(
    ps,
    model,
    st,
    batch_sample::Array{Float32, 3},
    batch_graph,
    dt::Float32,
)
    return Float32(multistep_window_loss(ps, model, st, batch_sample, dt; graph=batch_graph))
end

function collate_merged_windows(samples, base_graph)
    return (
        sample = cat(samples...; dims=2),
        graph = disconnected_graph(base_graph, length(samples)),
    )
end

function build_multistep_dataset(data::Array{Float32, 4}, horizon::Int)
    window_len = horizon + 1
    num_steps = size(data, 3)
    num_steps >= window_len || throw(ArgumentError("time dimension $(num_steps) is smaller than required window length $(window_len)"))
    scenario_count = size(data, 4)
    observations = Array{Float32, 3}[]
    sizehint!(observations, scenario_count * (num_steps - horizon))
    for scenario_idx in 1:scenario_count
        for start_idx in 1:(num_steps - horizon)
            stop_idx = start_idx + horizon
            push!(observations, copy(@view data[:, :, start_idx:stop_idx, scenario_idx]))
        end
    end
    return observations
end

function plot_multistep_loss(loss_trace; output_path=joinpath(MULTISTEP_RESULTS_DIR, "multistep_pretrain_loss_curve.png"))
    fig = Figure(size=(1000, 500))
    ax = Axis(fig[1, 1]; xlabel="iteration", ylabel="loss", yscale=log10, title="Multi-step pretrain loss")
    if !isempty(loss_trace)
        lines!(ax, 1:length(loss_trace), loss_trace, color=:darkorange, linewidth=2, label="3-step pretrain")
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

function save_multistep_snapshot!(
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

function multistep_pretrain_node(;
    dataset_path=joinpath(@__DIR__, "data", "data.jld2"),
    seed::Int=20260302,
    epochs::Int=DEFAULT_MULTISTEP_EPOCHS,
    batch_size::Int=DEFAULT_MULTISTEP_BATCH,
    max_scenarios::Union{Nothing, Int}=nothing,
    horizon::Int=DEFAULT_PREDICTION_HORIZON,
    initial_lr::Float32=DEFAULT_INITIAL_LR,
    decay_factor::Float32=DEFAULT_LR_DECAY_FACTOR,
    decay_every::Int=DEFAULT_LR_DECAY_EVERY,
    min_lr::Float32=DEFAULT_MIN_LR,
)
    ensure_multistep_dirs!()
    rng = Random.Xoshiro(seed)
    data, times = load_dataset(dataset_path)
    scenario_count = max_scenarios === nothing ? size(data, 4) : min(size(data, 4), max_scenarios)
    data = Float32.(data[:, :, :, 1:scenario_count])
    x_mean, x_std = compute_normalization_stats(data)
    data = normalize_data(data, x_mean, x_std)
    dt = Float32(times[2] - times[1])
    observations = build_multistep_dataset(data, horizon)
    model, ps0, st, _ = setup_model(rng)
    flat_ps = ComponentArray(ps0)
    effective_batch = min(batch_size, length(observations))

    current_lr = scheduled_lr(1; initial_lr, decay_factor, decay_every, min_lr)
    opt = Optimisers.AdamW(current_lr)
    opt_state = Optimisers.setup(opt, flat_ps)
    loss_trace = Float32[]
    best_loss = Inf32
    best_epoch = 0
    best_flat = deepcopy(flat_ps)
    weights_path = joinpath(MULTISTEP_MODEL_DIR, "multistep_pretrain_model_weights.jld2")

    for epoch in 1:epochs
        next_lr = scheduled_lr(epoch; initial_lr, decay_factor, decay_every, min_lr)
        if next_lr != current_lr
            current_lr = next_lr
            Optimisers.adjust!(opt_state, current_lr)
        end
        collate = samples -> collate_merged_windows(samples, model.base_graph)
        loader = DataLoader(observations; batchsize=effective_batch, collate, shuffle=true, partial=true, rng=rng)
        epoch_losses = Float32[]

        for batch in loader
            loss, grad = Zygote.withgradient(flat_ps) do x
                multistep_batch_loss(x, model, st, batch.sample, batch.graph, dt)
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
            save_multistep_snapshot!(weights_path, best_flat, best_loss, best_epoch, x_mean, x_std)
        end

        if epoch % 5 == 0 || epoch == 1 || epoch == epochs
            println("Multi-step pretrain epoch $(epoch)/$(epochs): lr=$(round(current_lr, digits=6)) loss=$(round(epoch_loss, digits=6))")
        end
    end

    return (
        loss_trace = loss_trace,
        weights_path = weights_path,
        best_loss = best_loss,
        best_epoch = best_epoch,
        trained_scenarios = scenario_count,
        horizon = horizon,
    )
end

function test_multistep_model(weights_path::String; dataset_path=joinpath(@__DIR__, "data", "data.jld2"))
    return compare_single_trajectory(;
        dataset_path,
        weights_path,
        output_name = "multistep_pretrained_best_node_rollout_comparison.png",
    )
end

function main()
    epochs = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : DEFAULT_MULTISTEP_EPOCHS
    batch_size = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : DEFAULT_MULTISTEP_BATCH
    max_scenarios = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing
    horizon = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : DEFAULT_PREDICTION_HORIZON

    training = multistep_pretrain_node(; epochs, batch_size, max_scenarios, horizon)
    loss_path = plot_multistep_loss(training.loss_trace)
    test_result = test_multistep_model(training.weights_path)

    println("Saved multi-step pretrain weights to $(training.weights_path)")
    println("Saved multi-step pretrain loss curve to $(loss_path)")
    println("Saved multi-step pretrain test plot to $(test_result.output_path)")
    println("Best multi-step pretrain loss = $(training.best_loss)")
    println("Best epoch = $(training.best_epoch)")
    println("Prediction horizon = $(training.horizon)")
    println("Trained scenarios = $(training.trained_scenarios)")
    println("MAE by feature = $(test_result.mae_by_feature)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
