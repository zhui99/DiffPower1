using ComponentArrays
using CairoMakie
using DifferentialEquations
using JLD2
using Lux
using Random
using Statistics

include(joinpath(@__DIR__, "Model.jl"))
const TEST_RESULTS_DIR = joinpath(@__DIR__, "results")

function load_flat_data(path::AbstractString)
    bundle = load(path)
    data = Float32.(bundle["data"])   # (state_dim, time_steps, scenarios)
    times = Float32.(bundle["times"])
    return data, times
end

function run_forward_smoke(; dataset_path=joinpath(@__DIR__, "data", "data.jld2"), batch_size::Int=4)
    data, times = load_flat_data(dataset_path)
    state_dim, _, scenarios = size(data)
    n_buses = state_dim ÷ MODEL_FEATURE_DIM

    rng = Random.default_rng()
    model = HeteroBusDynamics(n_buses)
    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps)

    # Single-sample forward
    u_single = data[:, 1, 1]
    du_single, _ = model(u_single, ps, st)
    @assert size(du_single) == size(u_single)
    @assert all(isfinite, du_single)

    # Batch forward (columns are different scenarios at the same time step)
    bs = min(batch_size, scenarios)
    u_batch = reduce(hcat, [data[:, 1, i] for i in 1:bs])
    du_batch, _ = model(u_batch, ps, st)
    @assert size(du_batch) == size(u_batch)
    @assert all(isfinite, du_batch)

    # ODE forward with the exact closure form requested
    function f_ode(u, p, t)
        du, _ = model(u, p, st)
        return du
    end

    tspan = (0.0f0, times[2] - times[1])
    prob_single = ODEProblem{false}(ODEFunction{false}(f_ode), u_single, tspan, ps)
    sol_single = solve(prob_single, Tsit5(); saveat=[tspan[1], tspan[2]])
    @assert length(sol_single.u) == 2
    @assert size(sol_single.u[end]) == size(u_single)
    @assert all(isfinite, sol_single.u[end])

    prob_batch = ODEProblem{false}(ODEFunction{false}(f_ode), u_batch, tspan, ps)
    sol_batch = solve(prob_batch, Tsit5(); saveat=[tspan[1], tspan[2]])
    @assert length(sol_batch.u) == 2
    @assert size(sol_batch.u[end]) == size(u_batch)
    @assert all(isfinite, sol_batch.u[end])

    return (
        data_size = size(data),
        single_input_size = size(u_single),
        single_forward_size = size(du_single),
        batch_input_size = size(u_batch),
        batch_forward_size = size(du_batch),
        single_ode_final_size = size(sol_single.u[end]),
        batch_ode_final_size = size(sol_batch.u[end]),
        dt = tspan[2],
    )
end

function reshape_state_trajectory(flat_traj::AbstractMatrix, n_buses::Int)
    feature_dim = size(flat_traj, 1) ÷ n_buses
    return reshape(flat_traj, feature_dim, n_buses, size(flat_traj, 2))
end

function compare_single_trajectory_plot(;
    dataset_path=joinpath(@__DIR__, "data", "data.jld2"),
    scenario_idx::Int=1,
    bus_idx::Int=32,
    output_name::String="model_forward_comparison.png",
)
    data, times = load_flat_data(dataset_path)
    state_dim, time_steps, scenarios = size(data)
    n_buses = state_dim ÷ MODEL_FEATURE_DIM
    1 <= scenario_idx <= scenarios || throw(ArgumentError("scenario_idx out of range"))
    1 <= bus_idx <= n_buses || throw(ArgumentError("bus_idx out of range"))

    rng = Random.default_rng()
    model = HeteroBusDynamics(n_buses)
    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps)

    x0 = data[:, 1, scenario_idx]
    relative_times = times .- first(times)

    function f_ode(u, p, t)
        du, _ = model(u, p, st)
        return du
    end

    prob = ODEProblem{false}(ODEFunction{false}(f_ode), x0, (relative_times[1], relative_times[end]), ps)
    sol = solve(prob, Tsit5(); saveat=relative_times)
    pred_flat = hcat(sol.u...)
    @assert size(pred_flat) == (state_dim, time_steps)

    truth = reshape_state_trajectory(data[:, :, scenario_idx], n_buses)
    pred = reshape_state_trajectory(pred_flat, n_buses)
    feature_names = ["omega_dev", "delta", "V", "P", "Q"]
    mae_by_feature = vec(mean(abs.(pred[:, bus_idx, :] .- truth[:, bus_idx, :]); dims=2))

    fig = Figure(size=(1200, 900))
    for i in 1:MODEL_FEATURE_DIM
        ax = CairoMakie.Axis(
            fig[i, 1];
            xlabel=i == MODEL_FEATURE_DIM ? "time (s)" : "",
            ylabel=feature_names[i],
            title="Bus $(bus_idx) $(feature_names[i]) | MAE=$(round(mae_by_feature[i], digits=4))",
        )
        lines!(ax, relative_times, vec(truth[i, bus_idx, :]), color=:steelblue, linewidth=2, label="true")
        lines!(ax, relative_times, vec(pred[i, bus_idx, :]), color=:firebrick, linewidth=2, linestyle=:dash, label="pred")
        axislegend(ax; position=:rb)
    end

    mkpath(TEST_RESULTS_DIR)
    output_path = joinpath(TEST_RESULTS_DIR, output_name)
    save(output_path, fig)

    return (
        output_path=output_path,
        scenario_idx=scenario_idx,
        bus_idx=bus_idx,
        prediction_size=size(pred),
        mae_by_feature=mae_by_feature,
    )
end

function main()
    result = run_forward_smoke()
    plot_result = compare_single_trajectory_plot()
    println("Forward smoke passed.")
    println(result)
    println("Trajectory plot saved.")
    println(plot_result)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
