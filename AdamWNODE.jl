using CairoMakie
using JLD2
using Optimisers
using Random
using SciMLSensitivity
using Statistics
using Zygote

include("TestNODEModel.jl")

const TRAIN_RESULTS_DIR = joinpath(@__DIR__, "results")
const TRAIN_MODEL_DIR = joinpath(@__DIR__, "model")
const DEFAULT_SEGMENT_LENGTH = 25
const DEFAULT_SEGMENTS_PER_BATCH = 6
const DEFAULT_ADAMW_ONLY_EPOCHS = 300

function ensure_train_dirs!()
    mkpath(TRAIN_RESULTS_DIR)
    mkpath(TRAIN_MODEL_DIR)
end

function build_training_windows(n_times::Int; segment_length::Int=DEFAULT_SEGMENT_LENGTH)
    starts = collect(1:segment_length:(n_times - segment_length + 1))
    if isempty(starts)
        error("segment_length=$(segment_length) is too long for n_times=$(n_times).")
    end
    return starts
end

function build_window_pairs(data, starts; max_scenarios::Union{Nothing, Int}=nothing)
    scenario_count = size(data, 4)
    if max_scenarios !== nothing
        scenario_count = min(scenario_count, max_scenarios)
    end
    return [(scenario_idx, start_idx) for scenario_idx in 1:scenario_count, start_idx in starts] |> vec
end

function segment_loss(
    flat_ps::AbstractVector,
    rebuild_ps,
    model,
    st,
    sample::Array{Float32, 3},
    times,
    start_idx::Int,
    segment_length::Int,
)
    stop_idx = start_idx + segment_length - 1
    local_times = times[start_idx:stop_idx] .- times[start_idx]
    truth = sample[:, :, start_idx:stop_idx]
    x0 = truth[:, :, 1]
    ps = rebuild_ps(flat_ps)
    prediction, _, _ = rollout(model, x0, local_times, ps, st)

    scale = max.(std(truth; dims=3), 1f-3)
    residual = (prediction .- truth) ./ scale
    return mean(abs2, residual)
end

function batch_loss(
    flat_ps::AbstractVector,
    rebuild_ps,
    model,
    st,
    data,
    times,
    window_pairs,
    segment_length::Int,
)
    losses = map(window_pairs) do (scenario_idx, start_idx)
        sample = Float32.(data[:, :, :, scenario_idx])
        segment_loss(flat_ps, rebuild_ps, model, st, sample, times, start_idx, segment_length)
    end
    return mean(losses)
end

function plot_adamw_loss(loss_trace; output_path=joinpath(TRAIN_RESULTS_DIR, "adamw_loss_curve.png"))
    fig = Figure(size=(1000, 500))
    ax = Axis(fig[1, 1]; xlabel="iteration", ylabel="loss", yscale=log10, title="AdamW loss")
    if !isempty(loss_trace)
        lines!(ax, 1:length(loss_trace), loss_trace, color=:steelblue, linewidth=2, label="AdamW")
        axislegend(ax; position=:rt)
    end
    save(output_path, fig)
    return output_path
end

function save_adamw_snapshot!(weights_path::String, best_flat, best_loss::Float32, best_epoch::Int)
    @save weights_path best_flat best_loss best_epoch
end

function load_initial_flat!(
    flat_ps::AbstractVector{Float32},
    init_weights_path::Union{Nothing, String},
)
    if init_weights_path === nothing || !isfile(init_weights_path)
        return false
    end
    bundle = JLD2.load(init_weights_path)
    if haskey(bundle, "best_flat")
        flat_ps .= Float32.(bundle["best_flat"])
        return true
    end
    return false
end

function adamw_train_node(;
    dataset_path=joinpath(@__DIR__, "data", "data.jld2"),
    seed::Int=20260302,
    epochs::Int=DEFAULT_ADAMW_ONLY_EPOCHS,
    segment_length::Int=DEFAULT_SEGMENT_LENGTH,
    segments_per_batch::Int=DEFAULT_SEGMENTS_PER_BATCH,
    max_scenarios::Union{Nothing, Int}=nothing,
    init_weights_path::Union{Nothing, String}=joinpath(TRAIN_MODEL_DIR, "pretrain_model_weights.jld2"),
)
    ensure_train_dirs!()
    rng = Random.Xoshiro(seed)
    data, times = load_dataset(dataset_path)
    model, ps0, st, _ = setup_model(rng)
    flat_ps0, rebuild_ps = Optimisers.destructure(ps0)
    flat_ps = copy(Float32.(flat_ps0))
    initialized_from_pretrain = load_initial_flat!(flat_ps, init_weights_path)

    starts = build_training_windows(length(times); segment_length)
    window_pairs = build_window_pairs(data, starts; max_scenarios)
    batch_size = min(segments_per_batch, length(window_pairs))

    opt = Optimisers.AdamW(0.01f0)
    opt_state = Optimisers.setup(opt, flat_ps)
    loss_trace = Float32[]
    best_loss = Inf32
    best_epoch = 0
    best_flat = copy(flat_ps)
    weights_path = joinpath(TRAIN_MODEL_DIR, "adamw_model_weights.jld2")

    for epoch in 1:epochs
        batch = rand(rng, window_pairs, batch_size)
        loss, grad = Zygote.withgradient(flat_ps) do x
            batch_loss(x, rebuild_ps, model, st, data, times, batch, segment_length)
        end
        opt_state, flat_ps = Optimisers.update(opt_state, flat_ps, grad[1])

        loss32 = Float32(loss)
        push!(loss_trace, loss32)
        if loss32 < best_loss
            best_loss = loss32
            best_epoch = epoch
            best_flat .= flat_ps
            save_adamw_snapshot!(weights_path, best_flat, best_loss, best_epoch)
        end

        println("AdamW-only epoch $(epoch)/$(epochs): loss=$(round(loss, digits=6))")
    end

    return (
        loss_trace=loss_trace,
        weights_path=weights_path,
        best_loss=best_loss,
        best_epoch=best_epoch,
        trained_scenarios=max_scenarios === nothing ? size(data, 4) : min(size(data, 4), max_scenarios),
        initialized_from_pretrain=initialized_from_pretrain,
    )
end

function test_adamw_model(weights_path::String; dataset_path=joinpath(@__DIR__, "data", "data.jld2"))
    return compare_single_trajectory(;
        dataset_path,
        weights_path,
        output_name="adamw_best_node_rollout_comparison.png",
    )
end

function main()
    epochs = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : DEFAULT_ADAMW_ONLY_EPOCHS
    segment_length = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : DEFAULT_SEGMENT_LENGTH
    segments_per_batch = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : DEFAULT_SEGMENTS_PER_BATCH
    max_scenarios = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : nothing
    init_weights_path = length(ARGS) >= 5 ? ARGS[5] : joinpath(TRAIN_MODEL_DIR, "pretrain_model_weights.jld2")

    training = adamw_train_node(; epochs, segment_length, segments_per_batch, max_scenarios, init_weights_path)
    loss_path = plot_adamw_loss(training.loss_trace)
    test_result = test_adamw_model(training.weights_path)

    println("Saved AdamW weights to $(training.weights_path)")
    println("Saved AdamW loss curve to $(loss_path)")
    println("Saved AdamW test plot to $(test_result.output_path)")
    println("Best AdamW loss = $(training.best_loss)")
    println("Best epoch = $(training.best_epoch)")
    println("Trained scenarios = $(training.trained_scenarios)")
    println("Initialized from pretrain = $(training.initialized_from_pretrain)")
    println("MAE by feature = $(test_result.mae_by_feature)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
