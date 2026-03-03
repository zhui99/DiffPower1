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
const MODEL_EMBED_DIM = 4
const MODEL_GCN_HIDDEN_DIM = 32
const MODEL_DECODER_HIDDEN_DIM = 64

# Neural ODE wrapper modeled after the Lux example in main.jl, but operating on
# compressed latent states shaped as (latent_dim, batch_count).
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

struct GraphEncoder{C1,C2,P} <: Lux.AbstractLuxLayer
    base_num_buses::Int
    conv1::C1
    conv2::C2
    projector::P
end

function GraphEncoder(input_dim::Int, gcn_hidden_dim::Int, latent_dim::Int, base_num_buses::Int)
    projector = Chain(
        Dense(gcn_hidden_dim, gcn_hidden_dim),
        x -> NNlib.gelu.(x),
        Dense(gcn_hidden_dim, latent_dim),
    )
    return GraphEncoder(
        base_num_buses,
        GCNConv(input_dim => gcn_hidden_dim),
        GCNConv(gcn_hidden_dim => gcn_hidden_dim),
        projector,
    )
end

function Lux.initialparameters(rng::AbstractRNG, layer::GraphEncoder)
    return (
        conv1 = Lux.initialparameters(rng, layer.conv1),
        conv2 = Lux.initialparameters(rng, layer.conv2),
        projector = Lux.initialparameters(rng, layer.projector),
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::GraphEncoder)
    return (
        conv1 = Lux.initialstates(rng, layer.conv1),
        conv2 = Lux.initialstates(rng, layer.conv2),
        projector = Lux.initialstates(rng, layer.projector),
    )
end

function (layer::GraphEncoder)(graph, x::AbstractMatrix, ps, st)
    x, stconv1 = layer.conv1(graph, x, ps.conv1, st.conv1)
    x = NNlib.gelu.(x)
    x, stconv2 = layer.conv2(graph, x, ps.conv2, st.conv2)
    x = NNlib.gelu.(x)
    total_nodes = size(x, 2)
    total_nodes % layer.base_num_buses == 0 || throw(ArgumentError("total_nodes=$(total_nodes) must be a multiple of base_num_buses=$(layer.base_num_buses)"))
    batch_count = total_nodes ÷ layer.base_num_buses
    pooled = reshape(x, size(x, 1), layer.base_num_buses, batch_count)
    pooled = dropdims(sum(pooled; dims=2) ./ layer.base_num_buses; dims=2)
    latent, stproj = layer.projector(pooled, ps.projector, st.projector)
    return latent, (conv1 = stconv1, conv2 = stconv2, projector = stproj)
end

struct MLPDynamics{D1,D2} <: Lux.AbstractLuxLayer
    dense1::D1
    dense2::D2
end

function MLPDynamics(latent_dim::Int)
    hidden_dim = 2 * latent_dim
    return MLPDynamics(
        Dense(latent_dim, hidden_dim),
        Dense(hidden_dim, latent_dim),
    )
end

function Lux.initialparameters(rng::AbstractRNG, layer::MLPDynamics)
    return (
        dense1 = Lux.initialparameters(rng, layer.dense1),
        dense2 = Lux.initialparameters(rng, layer.dense2),
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::MLPDynamics)
    return (
        dense1 = Lux.initialstates(rng, layer.dense1),
        dense2 = Lux.initialstates(rng, layer.dense2),
    )
end

function (layer::MLPDynamics)(x::AbstractMatrix, ps, st)
    x, stdense1 = layer.dense1(x, ps.dense1, st.dense1)
    x = NNlib.gelu.(x)
    x, stdense2 = layer.dense2(x, ps.dense2, st.dense2)
    return x, (dense1 = stdense1, dense2 = stdense2)
end

struct TransposeGCNDecoder{D1,C1,C2} <: Lux.AbstractLuxLayer
    base_num_buses::Int
    output_dim::Int
    dense1::D1
    conv1::C1
    conv2::C2
end

function TransposeGCNDecoder(latent_dim::Int, hidden_dim::Int, output_dim::Int, base_num_buses::Int)
    return TransposeGCNDecoder(
        base_num_buses,
        output_dim,
        Dense(latent_dim, hidden_dim),
        GCNConv(hidden_dim => hidden_dim),
        GCNConv(hidden_dim => output_dim),
    )
end

function Lux.initialparameters(rng::AbstractRNG, layer::TransposeGCNDecoder)
    return (
        dense1 = Lux.initialparameters(rng, layer.dense1),
        conv1 = Lux.initialparameters(rng, layer.conv1),
        conv2 = Lux.initialparameters(rng, layer.conv2),
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::TransposeGCNDecoder)
    return (
        dense1 = Lux.initialstates(rng, layer.dense1),
        conv1 = Lux.initialstates(rng, layer.conv1),
        conv2 = Lux.initialstates(rng, layer.conv2),
    )
end

function (layer::TransposeGCNDecoder)(graph, x::AbstractMatrix, ps, st)
    expanded, stdense1 = layer.dense1(x, ps.dense1, st.dense1)
    expanded = NNlib.gelu.(expanded)
    batch_count = size(expanded, 2)
    expanded_3d = reshape(expanded, size(expanded, 1), 1, batch_count)
    node_features = reshape(
        repeat(expanded_3d, 1, layer.base_num_buses, 1),
        size(expanded, 1),
        layer.base_num_buses * batch_count,
    )
    node_features, stconv1 = layer.conv1(graph, node_features, ps.conv1, st.conv1)
    node_features = NNlib.gelu.(node_features)
    decoded, stconv2 = layer.conv2(graph, node_features, ps.conv2, st.conv2)
    return decoded, (dense1 = stdense1, conv1 = stconv1, conv2 = stconv2)
end

function (n::LatentNeuralODE)(x::AbstractMatrix, ps, st)
    function dudt(u, p, t)
        du, _ = n.model(u, p, st)
        return du
    end

    prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps)
    sol = solve(prob, n.solver; n.kwargs...)
    return sol, st
end

# Full NODE model used for system identification on post-fault trajectories.
struct PowerSystemNODE{G,E,D,C}
    base_graph::G
    encoder::E
    dynamics::D
    decoder::C
    feature_dim::Int
    num_buses::Int
    latent_dim::Int
    embed_dim::Int
end

function read_branch_table()
    path = joinpath(MODEL_NETWORK_DIR, "branch.csv")
    return CSV.read(path, DataFrame)
end

function normalized_adjacency(branch_df::DataFrame, num_buses::Int)
    # Build a symmetric weighted adjacency and apply GCN normalization.
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
    Dinv = Diagonal(inv_sqrt_degree)
    return Dinv * adj * Dinv
end

function build_encoder(input_dim::Int, gcn_hidden_dim::Int, latent_dim::Int, num_buses::Int)
    return GraphEncoder(input_dim, gcn_hidden_dim, latent_dim, num_buses)
end

function build_dynamics(latent_dim::Int)
    return MLPDynamics(latent_dim)
end

function build_decoder(latent_dim::Int, decoder_hidden_dim::Int, output_dim::Int, num_buses::Int)
    return TransposeGCNDecoder(latent_dim, decoder_hidden_dim, output_dim, num_buses)
end

function build_power_system_node(;
    num_buses::Int=39,
    input_dim::Int=MODEL_FEATURE_DIM,
    latent_dim::Int=16,
    embed_dim::Int=MODEL_EMBED_DIM,
    gcn_hidden_dim::Int=MODEL_GCN_HIDDEN_DIM,
    decoder_hidden_dim::Int=MODEL_DECODER_HIDDEN_DIM,
    solver=AutoTsit5(Rodas5P(autodiff=false)),
)
    branch_df = read_branch_table()
    adjacency = normalized_adjacency(branch_df, num_buses)
    base_graph = GNNGraph(Matrix{Float32}(adjacency))
    encoder_input_dim = input_dim + embed_dim
    model = PowerSystemNODE(
        base_graph,
        build_encoder(encoder_input_dim, gcn_hidden_dim, latent_dim, num_buses),
        LatentNeuralODE(
            build_dynamics(latent_dim);
            solver,
            tspan=(0.0f0, 1.0f0),
            save_everystep=false,
            save_start=true,
            abstol=1f-8,
            reltol=1f-6,
        ),
        build_decoder(latent_dim, decoder_hidden_dim, input_dim, num_buses),
        input_dim,
        num_buses,
        latent_dim,
        embed_dim,
    )
    return model, adjacency
end

function load_dataset(path::AbstractString=joinpath(MODEL_DATA_DIR, "data.jld2"))
    bundle = load(path)
    data = bundle["data"]
    times = bundle["times"]
    return data, times
end

function repeated_bus_embedding(bus_embedding::AbstractMatrix, node_count::Int, base_num_buses::Int)
    node_count % base_num_buses == 0 || throw(ArgumentError("node_count=$(node_count) must be a multiple of base_num_buses=$(base_num_buses)"))
    repeat_count = node_count ÷ base_num_buses
    return repeat(Float32.(bus_embedding), 1, repeat_count)
end

function augment_with_bus_embedding(x::AbstractMatrix, bus_embedding::AbstractMatrix, base_num_buses::Int)
    return vcat(Float32.(x), repeated_bus_embedding(bus_embedding, size(x, 2), base_num_buses))
end

function disconnected_graph(base_graph, repeat_count::Int)
    repeat_count >= 1 || throw(ArgumentError("repeat_count must be >= 1"))
    return repeat_count == 1 ? base_graph : GNNLux.batch(fill(base_graph, repeat_count))
end

Zygote.@nograd disconnected_graph

function rollout(model::PowerSystemNODE, x0::AbstractMatrix, times, ps, st; graph=model.base_graph, solve_kwargs...)
    # Encode the initial physical state, evolve it in latent space using the Neural ODE
    # wrapper, then decode each saved latent step back to the physical feature space.
    x0f = Float32.(x0)
    x0_aug = augment_with_bus_embedding(x0f, ps.bus_embedding, model.num_buses)
    z0, enc_st = model.encoder(graph, x0_aug, ps.encoder, st.encoder)
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
    sol, dyn_st = dynamics_layer(z0, ps.dynamics, st.dynamics)

    decoded_steps = map(sol.u) do zflat
        xhat_raw, _ = model.decoder(graph, zflat, ps.decoder, st.decoder)
        return Float32.(xhat_raw)
    end
    decoded = isempty(decoded_steps) ?
        Array{Float32}(undef, MODEL_FEATURE_DIM, size(x0, 2), 0) :
        cat(decoded_steps...; dims=3)

    updated_state = (encoder = enc_st, dynamics = dyn_st, decoder = st.decoder)
    return decoded, sol, updated_state
end

function setup_model(rng::AbstractRNG=Random.Xoshiro(0); kwargs...)
    model, adjacency = build_power_system_node(; kwargs...)
    encoder_ps, encoder_st = LuxCore.setup(rng, model.encoder)
    dynamics_ps, dynamics_st = LuxCore.setup(rng, model.dynamics)
    decoder_ps, decoder_st = LuxCore.setup(rng, model.decoder)
    bus_embedding = 0.01f0 .* randn(rng, Float32, model.embed_dim, model.num_buses)
    ps = (encoder = encoder_ps, dynamics = dynamics_ps, decoder = decoder_ps, bus_embedding = bus_embedding)
    st = (encoder = encoder_st, dynamics = dynamics_st, decoder = decoder_st)
    return model, ps, st, adjacency
end

function smoke_test(; dataset_path=joinpath(MODEL_DATA_DIR, "data.jld2"))
    # Build the model and run one forward pass on the first scenario.
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