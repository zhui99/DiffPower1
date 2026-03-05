using CairoMakie
using ComponentArrays: ComponentArray
using JLD2
using Optimisers
using Random
using Statistics

include("Model.jl")

const TEST_RESULTS_DIR = joinpath(@__DIR__, "results")

function ensure_test_results_dir!()
    mkpath(TEST_RESULTS_DIR)
end

function normalize_state_tensor(data::Array{Float32, 3}, x_mean::Array{Float32, 4}, x_std::Array{Float32, 4})
    μ = dropdims(x_mean; dims=(2, 3, 4))
    σ = dropdims(x_std; dims=(2, 3, 4))
    return (data .- reshape(μ, :, 1, 1)) ./ reshape(σ, :, 1, 1)
end

function denormalize_state_tensor(data::Array{Float32, 3}, x_mean::Array{Float32, 4}, x_std::Array{Float32, 4})
    μ = dropdims(x_mean; dims=(2, 3, 4))
    σ = dropdims(x_std; dims=(2, 3, 4))
    return data .* reshape(σ, :, 1, 1) .+ reshape(μ, :, 1, 1)
end

function load_saved_parameters(weights_path::Union{Nothing, String}; seed::Int=20260302)
    model, ps0, st, _ = setup_model(Random.Xoshiro(seed))
    if weights_path === nothing
        return model, ps0, st, false, nothing, nothing
    end
    isfile(weights_path) || throw(ArgumentError("weights_path=$(weights_path) does not exist"))

    bundle = JLD2.load(weights_path)
    haskey(bundle, "best_flat") || throw(ArgumentError("weights file $(weights_path) does not contain key `best_flat`"))
    stored_ps = Float32.(bundle["best_flat"])
    ps = ComponentArray(ps0)
    ps .= stored_ps
    x_mean = haskey(bundle, "x_mean") ? Float32.(bundle["x_mean"]) : nothing
    x_std = haskey(bundle, "x_std") ? Float32.(bundle["x_std"]) : nothing
    return model, ps, st, true, x_mean, x_std
end

function compare_single_trajectory(;
    dataset_path=joinpath(@__DIR__, "data", "data.jld2"),
    weights_path::Union{Nothing, String}=nothing,
    scenario_idx::Int=1,
    bus_idx::Int=32,
    seed::Int=20260302,
    output_name::String="node_rollout_comparison.png",
)
    data, times = load_dataset(dataset_path)
    n_scenarios = size(data, 4)
    1 <= scenario_idx <= n_scenarios || throw(ArgumentError("scenario_idx=$(scenario_idx) out of range 1:$(n_scenarios)"))
    1 <= bus_idx <= size(data, 2) || throw(ArgumentError("bus_idx=$(bus_idx) out of range 1:$(size(data, 2))"))

    model, ps, st, loaded_weights, x_mean, x_std = load_saved_parameters(weights_path; seed)
    truth = Float32.(data[:, :, :, scenario_idx])
    truth_for_model = if x_mean === nothing || x_std === nothing
        truth
    else
        normalize_state_tensor(truth, x_mean, x_std)
    end
    x0 = truth_for_model[:, :, 1]
    relative_times = times .- first(times)
    prediction, _, _ = rollout(model, x0, relative_times, ps, st)
    prediction = if x_mean === nothing || x_std === nothing
        prediction
    else
        denormalize_state_tensor(prediction, x_mean, x_std)
    end

    feature_names = ["omega_dev", "delta", "V", "P", "Q"]
    mae_by_feature = vec(mean(abs.(prediction[:, bus_idx, :] .- truth[:, bus_idx, :]); dims=2))

    fig = Figure(size=(1200, 900))
    for i in 1:length(feature_names)
        ax = Axis(
            fig[i, 1];
            xlabel=i == length(feature_names) ? "time (s)" : "",
            ylabel=feature_names[i],
            title="Bus $(bus_idx) $(feature_names[i]) | MAE=$(round(mae_by_feature[i], digits=4))",
        )
        lines!(ax, relative_times, vec(truth[i, bus_idx, :]), color=:steelblue, linewidth=2, label="true")
        lines!(ax, relative_times, vec(prediction[i, bus_idx, :]), color=:firebrick, linewidth=2, linestyle=:dash, label="pred")
        axislegend(ax; position=:rb)
    end

    ensure_test_results_dir!()
    output_path = joinpath(TEST_RESULTS_DIR, output_name)
    save(output_path, fig)

    return (
        output_path=output_path,
        prediction_size=size(prediction),
        scenario_idx=scenario_idx,
        bus_idx=bus_idx,
        loaded_weights=loaded_weights,
        weights_path=weights_path,
        time_window=(first(relative_times), last(relative_times)),
        mae_by_feature=mae_by_feature,
    )
end

function main()
    weights_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    println(compare_single_trajectory(; weights_path))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
