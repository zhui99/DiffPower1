using CairoMakie
using BenchmarkTools
using ComponentArrays
using JLD2
using Lux
using ManifoldNeuralODEs
using OrdinaryDiffEq: Tsit5, AutoTsit5, Rodas4P2
using Random
using Statistics

include(joinpath(@__DIR__, "Model.jl"))
include(joinpath(@__DIR__, "PowerFlow.jl"))

const TEST_RESULTS_DIR = joinpath(@__DIR__, "results")
const FEATURE_NAMES_DEFAULT = ["omega_dev", "delta", "V", "P", "Q"]
const FEATURE_COLORS = [:steelblue, :firebrick, :darkgreen, :darkorange, :purple, :brown, :teal, :goldenrod, :magenta, :black]
const NUM_BUSES_DEFAULT = 39

Base.@kwdef struct NodeConfig
    mode::Symbol = :plain                # :plain | :projected | :stabilized
    use_analytic_jacobian::Bool = true
    single_factorize::Bool = true        # only for :projected
    gamma::Float32 = 0.01f0              # only for :stabilized
    solver::Symbol = :tsit5              # :tsit5 | :autotsit_rodas4p2
    abstol::Float32 = 1f-3
    reltol::Float32 = 1f-3
    adaptive::Bool = false
    dt::Float32 = 0.01f0
end

function config_name(cfg::NodeConfig)
    solver_tag = cfg.solver == :tsit5 ? "tsit5" : "autotsit_rodas4p2"
    if cfg.mode == :plain
        return "plain_$(solver_tag)"
    elseif cfg.mode == :projected
        proj = cfg.single_factorize ? "single" : "robust"
        jac = cfg.use_analytic_jacobian ? "analytic" : "autodiff"
        return "projected_$(proj)_$(jac)_$(solver_tag)"
    elseif cfg.mode == :stabilized
        jac = cfg.use_analytic_jacobian ? "analytic" : "autodiff"
        return "stabilized_$(jac)_$(solver_tag)"
    end
    throw(ArgumentError("unsupported mode=$(cfg.mode)"))
end

function make_solver(solver_name::Symbol)
    if solver_name == :tsit5
        return Tsit5()
    elseif solver_name == :autotsit_rodas4p2
        return AutoTsit5(Rodas4P2())
    end
    throw(ArgumentError("unsupported solver=$(solver_name)"))
end

function load_flat_data(path::AbstractString)
    bundle = load(path)
    data = Float32.(bundle["data"]) # (state_dim, time_steps, scenarios)
    times = Float32.(bundle["times"])
    return data, times
end

function reshape_state_trajectory(flat_traj::AbstractMatrix, n_buses::Int)
    feature_dim = size(flat_traj, 1) ÷ n_buses
    return reshape(flat_traj, feature_dim, n_buses, size(flat_traj, 2))
end

function flatten_tensor(tensor::Array{Float32,3})
    nfeature, nbus, nt = size(tensor)
    flat = Array{Float32}(undef, nfeature * nbus, nt)
    for k in 1:nt
        flat[:, k] = vec(@view tensor[:, :, k])
    end
    return flat
end

function generator_buses_default(n_buses::Int)
    if n_buses >= 39
        return collect(30:39)
    end
    return collect(max(1, n_buses - 9):n_buses)
end

function nearest_time_indices(reference_times::AbstractVector{<:Real}, query_times::AbstractVector{<:Real})
    idxs = similar(query_times, Int)
    for i in eachindex(query_times)
        t = query_times[i]
        j = searchsortedlast(reference_times, t)
        if j <= 0
            idxs[i] = 1
        elseif j >= length(reference_times)
            idxs[i] = length(reference_times)
        else
            left = reference_times[j]
            right = reference_times[j + 1]
            idxs[i] = abs(t - left) <= abs(right - t) ? j : (j + 1)
        end
    end
    return idxs
end

function solution_to_matrix(sol)
    first_u = sol.u[1]
    if ndims(first_u) == 1
        return hcat(sol.u...)
    end
    stacked = cat(sol.u...; dims=3)
    return stacked[:, 1, :]
end

function make_constraints(n_buses::Int, ybus::AbstractMatrix{<:Complex})
    function constraints(u, t)
        if !all(isfinite, u)
            nrows = 2 * n_buses
            return ndims(u) == 1 ? fill(1f6, nrows) : fill(1f6, nrows, size(u, 2))
        end
        return power_flow(u, t; Ybus=ybus)
    end

    function constraints_jacobian(u, t)
        if !all(isfinite, u)
            nrows = 2 * n_buses
            ncols = size(u, 1)
            return ndims(u) == 1 ?
                zeros(Float32, nrows, ncols) :
                zeros(Float32, nrows, ncols, size(u, 2))
        end
        return pf_jacobian_analytic(u, t; Ybus=ybus)
    end
    return constraints, constraints_jacobian
end

"""
1) build_node: 按配置创建 Node
"""
function build_node(
    cfg::NodeConfig,
    dynamics,
    constraints,
    constraints_jacobian,
    tspan::Tuple{<:Real,<:Real},
    saveat::Vector{Float32},
)
    solver = make_solver(cfg.solver)
    ode_solver_options = (
        abstol=cfg.abstol,
        reltol=cfg.reltol,
        adaptive=cfg.adaptive,
        dt=cfg.dt,
        save_everystep=false,
        save_start=true,
        saveat=saveat,
    )

    if cfg.mode == :plain
        return NeuralODE(dynamics, tspan, solver; ode_solver_options...)
    elseif cfg.mode == :projected
        jac = cfg.use_analytic_jacobian ? constraints_jacobian : nothing
        return ManifoldProjectedNeuralODE(
            dynamics,
            constraints,
            tspan,
            solver;
            manifold_jacobian=jac,
            jacobian_ad=AutoForwardDiff(),
            projection_kwargs=(; abstol=1e-4, single_factorize=cfg.single_factorize),
            ode_solver_options...,
        )
    elseif cfg.mode == :stabilized
        jac = cfg.use_analytic_jacobian ? constraints_jacobian : nothing
        return StabilizedNeuralODE(
            dynamics,
            constraints,
            cfg.gamma,
            tspan,
            solver;
            manifold_jacobian=jac,
            jacobian_ad=AutoForwardDiff(),
            ode_solver_options...,
        )
    end
    throw(ArgumentError("unsupported mode=$(cfg.mode)"))
end

function benchmark_solver_efficiency(;
    mode::Symbol=:projected,
    use_analytic_jacobian::Bool=true,
    single_factorize::Bool=true,
    gamma::Float32=0.01f0,
    window_steps::Int=50,
    scenario_idx::Int=1,
    start_idx::Int=1,
    dataset_path=joinpath(@__DIR__, "data", "data.jld2"),
    output_path=joinpath(TEST_RESULTS_DIR, "rollout_solver_benchmark.jld2"),
)
    configs = [
        NodeConfig(
            mode=mode,
            use_analytic_jacobian=use_analytic_jacobian,
            single_factorize=single_factorize,
            gamma=gamma,
            solver=:tsit5,
        ),
        NodeConfig(
            mode=mode,
            use_analytic_jacobian=use_analytic_jacobian,
            single_factorize=single_factorize,
            gamma=gamma,
            solver=:autotsit_rodas4p2,
        ),
    ]
    return rollout(
        dataset_path=dataset_path,
        output_path=output_path,
        scenario_idx=scenario_idx,
        start_idx=start_idx,
        window_steps=window_steps,
        configs=configs,
    )
end

"""
2) rollout: 运行轨迹预测并保存数据，同时返回每个配置耗时
"""
function rollout(;
    dataset_path=joinpath(@__DIR__, "data", "data.jld2"),
    output_path=joinpath(TEST_RESULTS_DIR, "rollout_data.jld2"),
    scenario_idx::Int=1,
    start_idx::Int=1,
    window_steps::Int=50,
    dt::Union{Nothing,Float32}=nothing,
    configs::Vector{NodeConfig}=[
        NodeConfig(mode=:plain),
        NodeConfig(mode=:projected, use_analytic_jacobian=true, single_factorize=true),
        NodeConfig(mode=:projected, use_analytic_jacobian=false, single_factorize=true),
    ],
)
    data, times = load_flat_data(dataset_path)
    state_dim, total_steps, scenarios = size(data)
    n_buses = state_dim ÷ MODEL_FEATURE_DIM
    1 <= scenario_idx <= scenarios || throw(ArgumentError("scenario_idx out of range"))
    1 <= start_idx <= total_steps || throw(ArgumentError("start_idx out of range"))

    dt_used = isnothing(dt) ? Float32(times[2] - times[1]) : dt
    stop_idx = min(total_steps, start_idx + window_steps - 1)
    stop_idx > start_idx || throw(ArgumentError("window_steps is too small"))
    effective_steps = stop_idx - start_idx + 1
    segment_times = Float32.(collect(0f0:dt_used:(effective_steps - 1) * dt_used))
    tspan = extrema(segment_times)

    truth_flat = Float32.(data[:, start_idx:stop_idx, scenario_idx])
    x0 = reshape(data[:, start_idx, scenario_idx], :, 1)
    truth_tensor = reshape_state_trajectory(truth_flat, n_buses)

    ybus = build_ybus(; num_buses=n_buses)
    constraints, constraints_jacobian = make_constraints(n_buses, ybus)
    dynamics = HeteroBusDynamics(n_buses)

    prediction_map = Dict{String,Any}()
    retcode_map = Dict{String,Any}()

    mkpath(TEST_RESULTS_DIR)
    for cfg in configs
        name = config_name(cfg)
        node = build_node(cfg, dynamics, constraints, constraints_jacobian, tspan, segment_times)

        # 为公平对比，用同一个固定 seed 初始化每个 node 的参数
        rng = Random.Xoshiro(20260306)
        ps, st = Lux.setup(rng, node)
        ps = ComponentArray(ps)

        node(x0, ps, st) # warmup
        sol, _ = @btime $node($x0, $ps, $st) evals=1 samples=1
        pred_flat = solution_to_matrix(sol)
        prediction_map[name] = (
            config=cfg,
            times=Float32.(sol.t),
            flat=Float32.(pred_flat),
        )
        retcode_map[name] = sol.retcode
    end

    labels = String["truth"]
    trajectories = Array{Float32,2}[truth_flat]
    for cfg in configs
        name = config_name(cfg)
        push!(labels, name)
        push!(trajectories, prediction_map[name].flat)
    end
    data_out = cat(trajectories...; dims=3)
    feature_names_out = FEATURE_NAMES_DEFAULT

    bundle = Dict(
        # Keep the same core fields as GenData.jl for reuse.
        "data" => data_out,                       # (state_dim, time, scenario_like)
        "times" => segment_times,
        "FEATURE_NAMES" => feature_names_out,
        "NUM_BUSES" => n_buses,
    )
    save(output_path, bundle)

    return (
        output_path=output_path,
        scenario_idx=scenario_idx,
        start_idx=start_idx,
        stop_idx=stop_idx,
        steps=effective_steps,
        labels=labels,
        retcodes=retcode_map,
    )
end

function save_preview(times, tensor, preview_buses, feature_names; filename, title_prefix="")
    fig = CairoMakie.Figure(size=(1400, 900))
    for i in 1:length(feature_names)
        ax = CairoMakie.Axis(
            fig[i, 1];
            xlabel=i == length(feature_names) ? "time (s)" : "",
            ylabel=feature_names[i],
            title="$(title_prefix)generator buses $(feature_names[i])",
        )
        for (j, bus) in enumerate(preview_buses)
            color = FEATURE_COLORS[mod1(j, length(FEATURE_COLORS))]
            CairoMakie.lines!(ax, times, vec(tensor[i, bus, :]), color=color, linewidth=1.8, label="bus $(bus)")
        end
        CairoMakie.axislegend(ax; position=:rb, nbanks=2, labelsize=10)
    end
    CairoMakie.save(filename, fig)
    return nothing
end

"""
3) plot_from_rollout: 读取 rollout 文件画图（格式与 GenData.jl 一致）
"""
function plot_from_rollout(;
    rollout_path=joinpath(TEST_RESULTS_DIR, "rollout_data.jld2"),
    output_dir=joinpath(TEST_RESULTS_DIR, "rollout_previews"),
)
    bundle = load(rollout_path)
    data = Float32.(bundle["data"])
    times = Float32.(bundle["times"])
    feature_names = haskey(bundle, "FEATURE_NAMES") ? Vector{String}(bundle["FEATURE_NAMES"]) : FEATURE_NAMES_DEFAULT
    n_buses = Int(haskey(bundle, "NUM_BUSES") ? bundle["NUM_BUSES"] : NUM_BUSES_DEFAULT)
    labels = haskey(bundle, "labels") ? Vector{String}(bundle["labels"]) : ["scenario_$(i)" for i in 1:size(data, 3)]
    preview_buses = generator_buses_default(n_buses)

    mkpath(output_dir)
    output_paths = String[]
    for i in 1:size(data, 3)
        tensor = reshape_state_trajectory(data[:, :, i], n_buses)
        label = labels[i]
        safe_label = replace(label, r"[^A-Za-z0-9_\-]" => "_")
        path = joinpath(output_dir, "$(safe_label)_state_trajectories.png")
        save_preview(times, tensor, preview_buses, feature_names; filename=path, title_prefix="$(label) ")
        push!(output_paths, path)
    end

    return (
        output_dir=output_dir,
        output_paths=output_paths,
        labels=labels,
        preview_buses=preview_buses,
    )
end

function main()
    rollout_result = rollout()
    plot_result = plot_from_rollout(; rollout_path=rollout_result.output_path)
    println("Rollout done.")
    println(rollout_result)
    println("Plot done.")
    println(plot_result)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
