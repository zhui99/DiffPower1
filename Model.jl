using Lux
using NNlib
using Random
const MODEL_FEATURE_DIM = 5
const MODEL_EMBED_DIM = 5
const MODEL_MLP_HIDDEN_DIM = 10

struct HeteroBusDynamics{E,M} <: Lux.AbstractLuxLayer
    n_buses::Int
    state_dim::Int
    embed::E
    mlp::M
end

function HeteroBusDynamics(
    n_buses::Int;
    state_dim::Int=MODEL_FEATURE_DIM,
    embed_dim::Int=MODEL_EMBED_DIM,
    hidden_dim::Int=MODEL_MLP_HIDDEN_DIM,
)
    embed_layer = Lux.Embedding(n_buses => embed_dim)
    mlp = Chain(
        Dense(state_dim + embed_dim => hidden_dim, NNlib.gelu),
        Dense(hidden_dim => state_dim),
    )
    return HeteroBusDynamics(n_buses, state_dim, embed_layer, mlp)
end

function Lux.initialstates(rng::AbstractRNG, layer::HeteroBusDynamics)
    return (
        bus_indices = collect(1:layer.n_buses),
        embed = Lux.initialstates(rng, layer.embed),
        mlp = Lux.initialstates(rng, layer.mlp),
    )
end

function Lux.initialparameters(rng::AbstractRNG, layer::HeteroBusDynamics)
    return (
        embed = Lux.initialparameters(rng, layer.embed),
        mlp = Lux.initialparameters(rng, layer.mlp),
    )
end

function (layer::HeteroBusDynamics)(u_flat, ps, st)
    is_vector = ndims(u_flat) == 1
    u_matrix = is_vector ? reshape(u_flat, :, 1) : u_flat
    ndims(u_matrix) == 2 || throw(ArgumentError("u_flat must be 1D or 2D"))

    expected_dim = layer.state_dim * layer.n_buses
    size(u_matrix, 1) == expected_dim ||
        throw(ArgumentError("expected flattened state rows=$(expected_dim), got $(size(u_matrix, 1))"))

    batch_size = size(u_matrix, 2)
    u_shaped = reshape(u_matrix, layer.state_dim, layer.n_buses, batch_size)

    e, st_embed = layer.embed(st.bus_indices, ps.embed, st.embed)
    e_expanded = repeat(reshape(e, size(e, 1), layer.n_buses, 1), 1, 1, batch_size)
    nn_input = cat(u_shaped, e_expanded; dims=1)
    nn_input = reshape(nn_input, size(nn_input, 1), layer.n_buses * batch_size)

    du_shaped, st_mlp = layer.mlp(nn_input, ps.mlp, st.mlp)
    du_shaped = reshape(du_shaped, layer.state_dim, layer.n_buses, batch_size)
    du_flat = reshape(du_shaped, size(u_matrix))

    st_new = (bus_indices = st.bus_indices, embed = st_embed, mlp = st_mlp)
    return (is_vector ? vec(du_flat) : du_flat), st_new
end
