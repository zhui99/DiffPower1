using CairoMakie
using CSV
using DataFrames
using DifferentialEquations
using JLD2
using LinearAlgebra
using PowerDynamics
using Random
using Statistics

include("ieee39.jl")

# Script-level constants for the Phase-1 data pipeline.
const DATA_DIR = joinpath(@__DIR__, "data")
const RESULTS_DIR = joinpath(@__DIR__, "results")
const NETWORK_NAME = "ieee39-inverter"
const FEATURE_NAMES = ["omega_dev", "delta", "V", "P", "Q"]
const FEATURE_COLORS = [:steelblue, :firebrick, :darkgreen, :darkorange, :purple, :brown, :teal, :goldenrod, :magenta, :black]
const DEFAULT_SCENARIOS = 100
const DEFAULT_SEED = 20260228
const TSPAN = (0.0, 5.0)
const DT = 0.01
const LOAD_EVENT_TIME = 0.80
const MAX_ATTEMPTS_FACTOR = 4
const NUM_BUSES = 39

function ensure_dirs!()
    # Keep generated artifacts in stable project-local folders.
    mkpath(DATA_DIR)
    mkpath(RESULTS_DIR)
end

function read_tables()
    # Load the static network description used by feature mapping.
    base = joinpath(@__DIR__, NETWORK_NAME)
    bus_df = CSV.read(joinpath(base, "bus.csv"), DataFrame)
    return bus_df
end

function build_symbol_index(nw, bus_df::DataFrame)
    # Prebuild symbolic indices once so time-step extraction only does state reads.
    nbus = nrow(bus_df)
    u_r = [VIndex(bus, Symbol("busbar₊u_r")) for bus in 1:nbus]
    u_i = [VIndex(bus, Symbol("busbar₊u_i")) for bus in 1:nbus]
    p = [VIndex(bus, Symbol("busbar₊P")) for bus in 1:nbus]
    q = [VIndex(bus, Symbol("busbar₊Q")) for bus in 1:nbus]
    omega = Vector{Any}(undef, nbus)
    fill!(omega, nothing)

    for row in eachrow(bus_df)
        bus = row.bus
        omega[bus] = if row.category in ("ctrld_machine", "ctrld_machine_load")
            VIndex(bus, Symbol("ctrld_gen₊machine₊ω"))
        elseif row.category == "unctrld_machine_load"
            VIndex(bus, Symbol("machine₊ω"))
        elseif row.category == "inverter"
            VIndex(bus, Symbol("inverter₊ω"))
        else
            nothing
        end
    end

    return (; u_r, u_i, p, q, omega)
end

function event_spec(rng::AbstractRNG, bus_df::DataFrame; forced_event_type=nothing)
    # Only generate load-step disturbances; line-fault scenarios are intentionally disabled.
    if !isnothing(forced_event_type) && forced_event_type != "load_step"
        throw(ArgumentError("Unsupported event type: $(forced_event_type). Only load_step is enabled."))
    end

    load_buses = bus_df[(bus_df.bus_type .== "PQ") .& (bus_df.has_load .== true), :bus]
    bus = rand(rng, load_buses)
    dP = clamp(0.35 * randn(rng), -1, 1)
    dQ = 0.0
    return (; event_type="load_step", target=bus, delta=(dP, dQ), fault=(0.0, 0.0))
end

function apply_event!(nw, spec)
    # Mutate the network callbacks to represent the sampled load-step disturbance.
    if spec.event_type != "load_step"
        throw(ArgumentError("Unsupported event type: $(spec.event_type). Only load_step is enabled."))
    end
    set_perturbation!(nw, spec.target, collect(spec.delta), LOAD_EVENT_TIME)
end

function extract_tensor(nw, sol, sample_times, index_map)
    # Extract [omega_dev, delta, V, P, Q] for every bus and time point.
    # P/Q are read directly from busbar states. V and delta are reconstructed
    # from busbar voltage components because the bus model does not expose
    # busbar₊V / busbar₊δ symbolic states.
    nbus = length(index_map.u_r)
    nt = length(sample_times)
    data = Array{Float32}(undef, 5, nbus, nt)

    for (k, t) in enumerate(sample_times)
        # Sample strictly after the disturbance to avoid the event discontinuity itself.
        state = NWState(sol, t)
        voltages = ComplexF64[
            state[index_map.u_r[bus]] + im * state[index_map.u_i[bus]]
            for bus in 1:nbus
        ]

        for bus in 1:nbus
            omega_idx = index_map.omega[bus]
            omega_dev = isnothing(omega_idx) ? 0.0 : state[omega_idx] - 1.0
            data[1, bus, k] = Float32(omega_dev)
            data[2, bus, k] = Float32(angle(voltages[bus]))
            data[3, bus, k] = Float32(abs(voltages[bus]))
            data[4, bus, k] = Float32(state[index_map.p[bus]])
            data[5, bus, k] = Float32(state[index_map.q[bus]])
        end
    end

    return data
end

function flatten_tensor(tensor::Array{Float32, 3})
    # Flatten (feature, bus, time) -> (feature*bus, time) with bus-major ordering.
    nfeature, nbus, nt = size(tensor)
    flat = Array{Float32}(undef, nfeature * nbus, nt)
    for k in 1:nt
        flat[:, k] = vec(@view tensor[:, :, k])
    end
    return flat
end

function unflatten_tensor(flat::Array{Float32, 2}, nfeature::Int, nbus::Int)
    # Restore one scenario from (feature*bus, time) to (feature, bus, time).
    nt = size(flat, 2)
    tensor = Array{Float32}(undef, nfeature, nbus, nt)
    for k in 1:nt
        tensor[:, :, k] = reshape(@view(flat[:, k]), nfeature, nbus)
    end
    return tensor
end

function run_scenario(rng::AbstractRNG, bus_df::DataFrame, sample_times; forced_event_type=nothing)
    # Build a fresh network for each rollout so callbacks do not leak across scenarios.
    nw = get_IEEE39_base(NETWORK_NAME)
    # Silence the power-flow initializer's verbose per-component logging.
    u0 = redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            initialize_from_pf!(nw)
        end
    end
    index_map = build_symbol_index(nw, bus_df)

    spec = event_spec(rng, bus_df; forced_event_type)
    apply_event!(nw, spec)

    prob = ODEProblem(nw, u0, TSPAN)
    sol = solve(
        prob,
        Rodas4P2();
    )

    if !SciMLBase.successful_retcode(sol)
        return nothing
    end

    tensor = extract_tensor(nw, sol, sample_times, index_map)
    if any(!isfinite, tensor)
        return nothing
    end

    return tensor
end

function generator_buses(bus_df::DataFrame)
    # The IEEE-39 setup contains 10 generator buses (30-39).
    return sort(bus_df[bus_df.has_gen .== true, :bus])
end

function save_preview(times, tensor, preview_buses; filename="sample_state_trajectories.png", title_prefix="")
    # Save one quick-look plot showing all generator-bus trajectories per feature.
    fig = CairoMakie.Figure(size=(1400, 900))
    for i in 1:length(FEATURE_NAMES)
        ax = CairoMakie.Axis(
            fig[i, 1];
            xlabel=i == length(FEATURE_NAMES) ? "time (s)" : "",
            ylabel=FEATURE_NAMES[i],
            title="$(title_prefix)generator buses $(FEATURE_NAMES[i])",
        )
        for (j, bus) in enumerate(preview_buses)
            color = FEATURE_COLORS[mod1(j, length(FEATURE_COLORS))]
            CairoMakie.lines!(ax, times, vec(tensor[i, bus, :]), color=color, linewidth=1.8, label="bus $(bus)")
        end
        CairoMakie.axislegend(ax; position=:rb, nbanks=2, labelsize=10)
    end
    CairoMakie.save(joinpath(RESULTS_DIR, filename), fig)
    return nothing
end

function save_reference_plots(; seed=DEFAULT_SEED)
    # Generate one deterministic load-step reference plot.
    ensure_dirs!()
    rng = MersenneTwister(seed)
    bus_df = read_tables()
    times = collect((LOAD_EVENT_TIME + DT):DT:TSPAN[2])
    preview_buses = generator_buses(bus_df)

    load_tensor = run_scenario(rng, bus_df, times; forced_event_type="load_step")
    isnothing(load_tensor) && error("Failed to generate load_step reference plot.")
    save_preview(times, load_tensor, preview_buses; filename="load_step_state_trajectories.png", title_prefix="load_step ")

    return (
        preview_buses=preview_buses,
    )
end

function main()
    # End-to-end entrypoint for dataset generation.
    ensure_dirs!()
    n_scenarios = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : DEFAULT_SCENARIOS
    seed = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : DEFAULT_SEED
    rng = MersenneTwister(seed)
    bus_df = read_tables()
    # Store only the post-disturbance trajectory, not the pre-fault steady state.
    times = collect((LOAD_EVENT_TIME + DT):DT:TSPAN[2])
    preview_buses = generator_buses(bus_df)

    trajectories = Array{Float32, 2}[]

    attempts = 0
    max_attempts = max(n_scenarios * MAX_ATTEMPTS_FACTOR, n_scenarios)
    while length(trajectories) < n_scenarios && attempts < max_attempts
        attempts += 1
        tensor = run_scenario(rng, bus_df, times)
        if isnothing(tensor)
            continue
        end

        push!(trajectories, flatten_tensor(tensor))
    end

    if isempty(trajectories)
        error("No successful scenarios were generated.")
    end
    if length(trajectories) < n_scenarios
        @warn "Generated fewer successful scenarios than requested" requested=n_scenarios generated=length(trajectories)
    end

    data = cat(trajectories...; dims=3)
    preview_tensor = unflatten_tensor(trajectories[1], length(FEATURE_NAMES), NUM_BUSES)
    save_preview(times, preview_tensor, preview_buses)

    data_path = joinpath(DATA_DIR, "data.jld2")
    preview_path = joinpath(RESULTS_DIR, "sample_state_trajectories.png")
    @save data_path data times FEATURE_NAMES NUM_BUSES

    println("Generated $(size(data, 3)) scenarios with flattened shape $(size(data)) (state_dim, time, scenario).")
    println("Saved dataset to $(data_path).")
    println("Saved preview figure to $(preview_path).")
    println("Stored only post-event trajectories from t >= $(LOAD_EVENT_TIME + DT)s.")
    println("Power source: busbar₊P / busbar₊Q.")
    first_feature_rows = 1:length(FEATURE_NAMES):(length(FEATURE_NAMES) * NUM_BUSES)
    omega_values = data[first_feature_rows, :, :]
    println("omega_dev mean/std = $(round(mean(omega_values), digits=6)) / $(round(std(omega_values), digits=6))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
