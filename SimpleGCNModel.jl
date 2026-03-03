using CSV
using DataFrames
using DifferentialEquations
using GNNLux
using JLD2
using LinearAlgebra
using Lux
using NNlib
using Random
using Zygote

const MODEL_DATA_DIR = joinpath(@__DIR__, "data")
const MODEL_NETWORK_DIR = joinpath(@__DIR__, "ieee39-inverter")
const MODEL_FEATURE_DIM = 5
const MODEL_GCN_HIDDEN_DIM = 32

struct LatentNeuralODE{M,So,T,K} <: Lux.AbstractLuxWrapperLayer{:model}
    model::M
    solver::So
    tspan::T
    kwargs::K
end

function LatentNeuralODE(
    model::Lux.AbstractLuxLayer;
    solver=AutoTsit5(Rodas5P(autodiff=false)),
    tspan=(0.0f0, 1.0f0),
    kwargs...,
)
    return LatentNeuralODE(model, solver, tspan, kwargs)
end

struct SimpleGCNDynamics{C1,C2} <: Lux.AbstractLuxLayer
    conv1::C1
    conv2::C2
end

function SimpleGCNDynamics(feature_dim::Int, hidden_dim::Int)
    return SimpleGCNDynamics(
        GCNConv(feature_dim => hidden_dim),
        GCNConv(hidden_dim => feature_dim),
    )
end

function Lux.initialparameters(rng::AbstractRNG, layer::SimpleGCNDynamics)
    return (
        conv1 = Lux.initialparameters(rng, layer.conv1),
        conv2 = Lux.initialparameters(rng, layer.conv2),
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::SimpleGCNDynamics)
    return (
        conv1 = Lux.initialstates(rng, layer.conv1),
        conv2 = Lux.initialstates(rng, layer.conv2),
    )
end

function (layer::SimpleGCNDynamics)(graph, x::AbstractMatrix, ps, st)
    x, stconv1 = layer.conv1(graph, x, ps.conv1, st.conv1)
    x = NNlib.gelu.(x)
    x, stconv2 = layer.conv2(graph, x, ps.conv2, st.conv2)
    return x, (conv1 = stconv1, conv2 = stconv2)
end

function (n::LatentNeuralODE)(graph, x::AbstractMatrix, ps, st)
    function dudt(u, p, t)
        du, _ = n.model(graph, u, p, st)
        return du
    end

    prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps)
    sol = solve(prob, n.solver; n.kwargs...)
    return sol, st
end

struct PowerSystemNODE{G,D}
    base_graph::G
    dynamics::D
    feature_dim::Int
    num_buses::Int
end

function read_branch_table()
    path = joinpath(MODEL_NETWORK_DIR, "branch.csv")
    return CSV.read(path, DataFrame)
end

function normalized_adjacency(branch_df::DataFrame, num_buses::Int)
    adj = zeros(Float32, num_buses, num_buses)
    for row in eachrow(branch_df)
        src = row.src_bus
        dst = row.dst_bus
        z = row.R + im * row.X
        weight = Float32(abs(inv(z)))
        adj[src, dst] += weight
        adj[dst, src] += weight
    end
    for i in 1:num_buses
        adj[i, i] += 1.0f0
    end
    degree = vec(sum(adj; dims=2))
    inv_sqrt_degree = similar(degree)
    @inbounds for i in eachindex(degree)
        inv_sqrt_degree[i] = degree[i] > 0 ? inv(sqrt(degree[i])) : 0.0f0
    end
    dinv = Diagonal(inv_sqrt_degree)
    return dinv * adj * dinv
end

function build_power_system_node(;
    num_buses::Int=39,
    input_dim::Int=MODEL_FEATURE_DIM,
    hidden_dim::Int=MODEL_GCN_HIDDEN_DIM,
    solver=AutoTsit5(Rodas5P(autodiff=false)),
)
    branch_df = read_branch_table()
    adjacency = normalized_adjacency(branch_df, num_buses)
    base_graph = GNNGraph(Matrix{Float32}(adjacency))
    model = PowerSystemNODE(
        base_graph,
        LatentNeuralODE(
            SimpleGCNDynamics(input_dim, hidden_dim);
            solver,
            tspan=(0.0f0, 1.0f0),
            save_everystep=false,
            save_start=true,
            abstol=1f-8,
            reltol=1f-6,
        ),
        input_dim,
        num_buses,
    )
    return model, adjacency
end

function load_dataset(path::AbstractString=joinpath(MODEL_DATA_DIR, "data.jld2"))
    bundle = load(path)
    data = bundle["data"]
    times = bundle["times"]
    return data, times
end

function disconnected_graph(base_graph, repeat_count::Int)
    repeat_count >= 1 || throw(ArgumentError("repeat_count must be >= 1"))
    return repeat_count == 1 ? base_graph : GNNLux.batch(fill(base_graph, repeat_count))
end

Zygote.@nograd disconnected_graph

function rollout(model::PowerSystemNODE, x0::AbstractMatrix, times, ps, st; graph=model.base_graph, solve_kwargs...)
    x0f = Float32.(x0)
    tspan = Float32.(times)
    local_times = tspan .- first(tspan)
    dynamics_layer = LatentNeuralODE(
        model.dynamics.model;
        solver=model.dynamics.solver,
        tspan=(first(local_times), last(local_times)),
        saveat=local_times,
        save_start=true,
        abstol=1f-8,
        reltol=1f-6,
        solve_kwargs...,
    )
    sol, dyn_st = dynamics_layer(graph, x0f, ps.dynamics, st.dynamics)
    decoded_steps = map(sol.u) do u
        Float32.(u)
    end
    decoded = isempty(decoded_steps) ?
        Array{Float32}(undef, model.feature_dim, size(x0f, 2), 0) :
        cat(decoded_steps...; dims=3)
    updated_state = (dynamics = dyn_st,)
    return decoded, sol, updated_state
end

function setup_model(rng::AbstractRNG=Random.Xoshiro(0); kwargs...)
    model, adjacency = build_power_system_node(; kwargs...)
    dynamics_ps, dynamics_st = LuxCore.setup(rng, model.dynamics)
    ps = (dynamics = dynamics_ps,)
    st = (dynamics = dynamics_st,)
    return model, ps, st, adjacency
end

function smoke_test(; dataset_path=joinpath(MODEL_DATA_DIR, "data.jld2"))
    data, times = load_dataset(dataset_path)
    model, ps, st, adjacency = setup_model()
    x0 = Float32.(data[:, :, 1, 1])
    prediction, sol, _ = rollout(model, x0, times, ps, st)
    return (
        prediction_size = size(prediction),
        latent_steps = length(sol.u),
        adjacency_size = size(adjacency),
    )
end

function merged_batch_smoke_test(; dataset_path=joinpath(MODEL_DATA_DIR, "data.jld2"), batch_size::Int=2)
    data, times = load_dataset(dataset_path)
    batch_size = min(batch_size, size(data, 4))
    model, ps, st, adjacency = setup_model()
    x0 = reduce(hcat, [Float32.(data[:, :, 1, idx]) for idx in 1:batch_size])
    graph = disconnected_graph(model.base_graph, batch_size)
    prediction, sol, _ = rollout(model, x0, times, ps, st; graph)
    return (
        input_size = size(x0),
        prediction_size = size(prediction),
        latent_steps = length(sol.u),
        adjacency_size = size(adjacency),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    println(smoke_test())
end
