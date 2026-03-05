using CSV
using DataFrames
using JLD2
using LinearAlgebra
using Lux
using Statistics

const PF_NETWORK_DIR = joinpath(@__DIR__, "ieee39-inverter")
const PF_DATA_PATH = joinpath(@__DIR__, "data", "data.jld2")
const PF_FEATURE_DIM = 5
const PF_NUM_BUSES = 39

function read_branch_table(path::AbstractString=joinpath(PF_NETWORK_DIR, "branch.csv"))
    CSV.read(path, DataFrame)
end

function build_ybus(branch_df::DataFrame, num_buses::Int)
    ybus = zeros(ComplexF32, num_buses, num_buses)
    for row in eachrow(branch_df)
        src = Int(row.src_bus)
        dst = Int(row.dst_bus)
        r_src = (hasproperty(row, :r_src) && !iszero(row.r_src)) ? Float32(row.r_src) : 1.0f0
        r_dst = (hasproperty(row, :r_dst) && !iszero(row.r_dst)) ? Float32(row.r_dst) : 1.0f0
        active = hasproperty(row, :active) ? Float32(row.active) : 1.0f0

        z = ComplexF32(row.R, row.X)
        y_series = inv(z)
        y_shunt_src = ComplexF32(row.G_src, row.B_src)
        y_shunt_dst = ComplexF32(row.G_dst, row.B_dst)

        yss = -(y_series + y_shunt_src) * (r_src^2)
        ysd = y_series * r_src * r_dst
        yds = y_series * r_src * r_dst
        ydd = -(y_series + y_shunt_dst) * (r_dst^2)

        ybus[src, src] += active * yss
        ybus[src, dst] += active * ysd
        ybus[dst, src] += active * yds
        ybus[dst, dst] += active * ydd
    end
    ybus
end

function build_ybus(;
    branch_path::AbstractString=joinpath(PF_NETWORK_DIR, "branch.csv"),
    num_buses::Int=PF_NUM_BUSES,
)
    build_ybus(read_branch_table(branch_path), num_buses)
end

"""
    power_flow(u_flat, t; Ybus)

Power-flow residual for flattened states.

Input:
- `u_flat`: `(5n, B)` or `(5n,)`, state order per bus is `[omega_dev, delta, V, P, Q]`.
- `Ybus`: admittance matrix `(n, n)`.

Output:
- `(2n, B)` residual matrix:
  - first `n` rows: `ΔP = P_inj + P`
  - last  `n` rows: `ΔQ = Q_inj + Q`
  where `P, Q` are stored with opposite sign convention in dataset.
"""
function power_flow(u_flat, t; Ybus)
    u_matrix = ndims(u_flat) == 1 ? reshape(u_flat, :, 1) : u_flat
    ndims(u_matrix) == 2 || throw(ArgumentError("u_flat must be 1D or 2D"))

    n_buses = size(Ybus, 1)
    size(Ybus, 1) == size(Ybus, 2) || throw(ArgumentError("Ybus must be square"))
    size(u_matrix, 1) == PF_FEATURE_DIM * n_buses ||
        throw(ArgumentError("u_flat row size must be 5*n_buses"))

    batch_count = size(u_matrix, 2)
    u_3d = reshape(u_matrix, PF_FEATURE_DIM, n_buses, batch_count)
    δ = @view u_3d[2, :, :]
    V = @view u_3d[3, :, :]
    P = @view u_3d[4, :, :]
    Q = @view u_3d[5, :, :]

    V_complex = V .* cis.(δ)
    I_complex = Ybus * V_complex
    S_calc = V_complex .* conj.(I_complex)

    residuals_p = real.(S_calc) .+ P
    residuals_q = imag.(S_calc) .+ Q
    vcat(residuals_p, residuals_q)
end

"""
    pf_jacobian(u_flat, t; Ybus)

Jacobian of `power_flow` w.r.t. flattened state.

Returns:
- single sample input `(5n,)` -> Jacobian `(2n, 5n)`
- batch input `(5n, B)` -> stacked Jacobian `(2n, 5n, B)`
"""
function pf_jacobian(u_flat::AbstractArray, t; Ybus)
    x = ndims(u_flat) == 1 ? reshape(u_flat, :, 1) :
        ndims(u_flat) == 2 ? u_flat :
        throw(ArgumentError("u_flat must be 1D or 2D"))

    j3 = Lux.batched_jacobian(z -> power_flow(z, t; Ybus=Ybus), AutoForwardDiff(), x)
    return ndims(u_flat) == 1 ? @view(j3[:, :, 1]) : j3
end

function pf_jacobian_analytic_single(u_flat::AbstractVector, t; Ybus)
    n_buses = size(Ybus, 1)
    size(u_flat, 1) == PF_FEATURE_DIM * n_buses ||
        throw(ArgumentError("u_flat row size must be 5*n_buses"))

    state = reshape(u_flat, PF_FEATURE_DIM, n_buses)
    theta = @view state[2, :]
    vm = @view state[3, :]
    T = eltype(u_flat)

    v_cx = vm .* cis.(theta)
    i_cx = Ybus * v_cx

    diag_v = Diagonal(v_cx)
    diag_i_conj = Diagonal(conj.(i_cx))
    diag_v_conj = Diagonal(conj.(v_cx))
    phase_shift = Diagonal(cis.(theta))
    phase_shift_conj = Diagonal(cis.(-theta))

    dS_ddelta = im .* diag_v * diag_i_conj .- im .* diag_v * conj.(Ybus) * diag_v_conj
    dS_dV = phase_shift * diag_i_conj .+ diag_v * conj.(Ybus) * phase_shift_conj

    dP_ddelta = real.(dS_ddelta)
    dQ_ddelta = imag.(dS_ddelta)
    dP_dV = real.(dS_dV)
    dQ_dV = imag.(dS_dV)

    z_nn = zeros(T, n_buses, n_buses)
    i_nn = Diagonal(ones(T, n_buses))

    # Residual definitions: ΔP = P_inj + P, ΔQ = Q_inj + Q
    J_P_3d = cat(z_nn, dP_ddelta, dP_dV, i_nn, z_nn; dims=3)
    J_Q_3d = cat(z_nn, dQ_ddelta, dQ_dV, z_nn, i_nn; dims=3)
    J_P_flat = reshape(permutedims(J_P_3d, (1, 3, 2)), n_buses, PF_FEATURE_DIM * n_buses)
    J_Q_flat = reshape(permutedims(J_Q_3d, (1, 3, 2)), n_buses, PF_FEATURE_DIM * n_buses)
    return vcat(J_P_flat, J_Q_flat)
end

function pf_jacobian_analytic(u_flat::AbstractArray, t; Ybus)
    if ndims(u_flat) == 1
        return pf_jacobian_analytic_single(u_flat, t; Ybus=Ybus)
    elseif ndims(u_flat) == 2
        j_list = map(1:size(u_flat, 2)) do b
            pf_jacobian_analytic_single(@view(u_flat[:, b]), t; Ybus=Ybus)
        end
        return cat(j_list...; dims=3)
    else
        throw(ArgumentError("u_flat must be 1D or 2D"))
    end
end

function compare_pf_jacobian(u_flat::AbstractArray, t; Ybus)
    j_auto = pf_jacobian(u_flat, t; Ybus=Ybus)
    j_analytic = pf_jacobian_analytic(u_flat, t; Ybus=Ybus)
    diff = abs.(j_auto .- j_analytic)
    return (
        auto_size=size(j_auto),
        analytic_size=size(j_analytic),
        max_abs_diff=maximum(diff),
        mean_abs_diff=mean(diff),
    )
end

function load_flat_batch(
    dataset_path::AbstractString=PF_DATA_PATH;
    time_idx::Int=1,
    batch_size::Int=4,
)
    bundle = load(dataset_path)
    data = Float32.(bundle["data"]) # (5n, T, S)
    times = Float32.(bundle["times"])
    ndims(data) == 3 || throw(ArgumentError("expected data shape (5n, T, S)"))
    time_idx <= size(data, 2) || throw(ArgumentError("time_idx out of range"))
    bs = min(batch_size, size(data, 3))
    return data[:, time_idx, 1:bs], times[time_idx]
end

function smoke_test_power_flow(;
    dataset_path::AbstractString=PF_DATA_PATH,
    num_buses::Int=PF_NUM_BUSES,
    time_idx::Int=1,
    batch_size::Int=4,
)
    ybus = build_ybus(; num_buses)
    x_batch, t = load_flat_batch(dataset_path; time_idx, batch_size)
    residual = power_flow(x_batch, t; Ybus=ybus)
    @assert size(residual) == (2 * num_buses, size(x_batch, 2))
    @assert all(isfinite, residual)

    x_single = @view x_batch[:, 1]
    residual_single = power_flow(x_single, t; Ybus=ybus)
    @assert size(residual_single) == (2 * num_buses, 1)
    @assert all(isfinite, residual_single)

    jac_single = pf_jacobian(x_single, t; Ybus=ybus)
    @assert size(jac_single) == (2 * num_buses, PF_FEATURE_DIM * num_buses)
    @assert all(isfinite, jac_single)

    jac_batch = pf_jacobian(x_batch, t; Ybus=ybus)
    @assert size(jac_batch) == (2 * num_buses, PF_FEATURE_DIM * num_buses, size(x_batch, 2))
    @assert all(isfinite, jac_batch)

    jac_cmp_single = compare_pf_jacobian(x_single, t; Ybus=ybus)
    jac_cmp_batch = compare_pf_jacobian(x_batch, t; Ybus=ybus)

    return (
        ybus_size=size(ybus),
        batch_input_size=size(x_batch),
        batch_residual_size=size(residual),
        single_residual_size=size(residual_single),
        jac_single_size=size(jac_single),
        jac_batch_size=size(jac_batch),
        jac_cmp_single=jac_cmp_single,
        jac_cmp_batch=jac_cmp_batch,
        mean_abs_residual=mean(abs, residual),
        max_abs_residual=maximum(abs, residual),
        time=t,
    )
end

function main()
    result = smoke_test_power_flow()
    println(result)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
