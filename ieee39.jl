# IEEE 39-bus network assembly utilities.
# This file converts the CSV description into a PowerDynamics network and
# provides helper perturbations used by data generation and experiments.
using PowerDynamics
using PowerDynamics.Library
using ModelingToolkit
using NetworkDynamics
using DataFrames
using CSV
include("DroopInverter.jl")
include("NeuralModel.jl")

# DATA_DIR = joinpath(@__DIR__, "ieee39-inverter")

function safe_read(path)
    # Missing optional tables are treated as empty so the network builder stays robust.
    if isfile(path)
        return CSV.read(path, DataFrame)
    else
        return DataFrame()
    end
end

function apply_csv_params!(bus, table, bus_index)
    # Apply all non-index columns from one CSV row to the compiled bus defaults.
    if isempty(table)
        return
    end
    row_idx = findfirst(table.bus .== bus_index)
    if isnothing(row_idx)
        return 
    end

    # Apply all parameters except "bus" column
    row = table[row_idx, :]
    for col_name in names(table)
        if col_name != "bus"
            set_default!(bus, Regex(col_name*"\$"), row[col_name])
        end
    end
end

function init_neuralnet!(bus, n_input=1, n_output=1, n_hidden=10, rng=Xoshiro(0))
    # Initialize neural inverter weights with a stable Kaiming scheme.
    init_val = kaiming_uniform(rng, Float64, n_hidden, n_input)
    for i in 1:n_hidden
        for j in 1:n_input
            set_default!(bus, Regex("w1_$(i)_$(j)\$"), init_val[i, j])
        end
    end
    init_val = kaiming_uniform(rng, Float64, n_output, n_hidden)
    for i in 1:n_output
        for j in 1:n_hidden
            set_default!(bus, Regex("w2_$(i)_$(j)\$"), init_val[i, j])
        end
    end 
end

function get_IEEE39_base(name; nn=false, rng=Xoshiro(0))
    # Build the full network from CSV templates and attach the requested inverter model.
    DATA_DIR = joinpath(@__DIR__, name)
    branch_df = safe_read(joinpath(DATA_DIR, "branch.csv"))
    bus_df = safe_read(joinpath(DATA_DIR, "bus.csv"))
    load_df = safe_read(joinpath(DATA_DIR, "load.csv"))
    machine_df = safe_read(joinpath(DATA_DIR, "machine.csv"))
    avr_df = safe_read(joinpath(DATA_DIR, "avr.csv"))
    gov_df = safe_read(joinpath(DATA_DIR, "gov.csv"))
    inverter_df = safe_read(joinpath(DATA_DIR, "inverter.csv"))

    BASE_MVA = 100.0
    BASE_FREQ = 50.0

    load = ZIPLoad(;name=:ZIPLoad)

    uncontrolled_machine = SauerPaiMachine(;
        τ_m_input=false,  ## No external mechanical torque input
        vf_input=false,   ## No external field voltage input
        name=:machine,
    )

    _machine = SauerPaiMachine(;
        name=:machine,
    )
    _avr = AVRTypeI(;
        name=:avr,
        ceiling_function=:quadratic,
    )
    _gov = TGOV1(; name=:gov,)

    controlled_machine = CompositeInjector(
        [_machine, _avr, _gov],
        name=:ctrld_gen
    )
    if nn == false
        inverter = VSGInverter(; name=:inverter)
    else
        inverter = NeuralModel(; name=:inverter)
    end

    @named junction_bus_template = compile_bus(MTKBus())
    strip_defaults!(junction_bus_template)  ## Clear default parameters for manual setting

    @named load_bus_template = compile_bus(MTKBus(load))
    strip_defaults!(load_bus_template)

    @named ctrld_machine_bus_template = compile_bus(
        MTKBus(controlled_machine);
    )
    strip_defaults!(ctrld_machine_bus_template)

    set_default!(ctrld_machine_bus_template, r"S_b$", BASE_MVA)
    set_default!(ctrld_machine_bus_template, r"ω_b$", 2π*BASE_FREQ)

    @named ctrld_machine_load_bus_template = compile_bus(
        MTKBus(controlled_machine, load);
    )
    strip_defaults!(ctrld_machine_load_bus_template)
    set_default!(ctrld_machine_load_bus_template, r"S_b$", BASE_MVA)
    set_default!(ctrld_machine_load_bus_template, r"ω_b$", 2π*BASE_FREQ)
    formula1 = @initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
    set_initformula!(ctrld_machine_load_bus_template, formula1)

    @named unctrld_machine_load_bus_template = compile_bus(
        MTKBus(uncontrolled_machine, load);
    )
    strip_defaults!(unctrld_machine_load_bus_template)
    set_default!(unctrld_machine_load_bus_template, r"S_b$", BASE_MVA)
    set_default!(unctrld_machine_load_bus_template, r"ω_b$", 2π*BASE_FREQ)
    set_initformula!(unctrld_machine_load_bus_template, formula1)

    @named inverter_bus_template = compile_bus(
        MTKBus(inverter);
    )
    # if nn == false
    #      strip_defaults!(inverter_bus_template)
    # end
    formula2 = @initformula :inverter₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
    add_initformula!(inverter_bus_template, formula2)

    # set_default!(inverter_bus_template, r"S_b$", BASE_MVA)
    # set_default!(inverter_bus_template, r"ω_b$", 2π*BASE_FREQ)

    busses = []
    for row in eachrow(bus_df)
        i = row.bus

        # Select the correct compiled bus template from the bus category.
        bus = if row.category == "junction"
            compile_bus(junction_bus_template; vidx=i, name=Symbol("bus$i"))
        elseif row.category == "load"
            compile_bus(load_bus_template; vidx=i, name=Symbol("bus$i"))
        elseif row.category == "ctrld_machine"
            compile_bus(ctrld_machine_bus_template; vidx=i, name=Symbol("bus$i"))
        elseif row.category == "ctrld_machine_load"
            compile_bus(ctrld_machine_load_bus_template; vidx=i, name=Symbol("bus$i"))
        elseif row.category == "unctrld_machine_load"
            compile_bus(unctrld_machine_load_bus_template; vidx=i, name=Symbol("bus$i"))
        elseif row.category == "inverter"
            compile_bus(inverter_bus_template; vidx=i, name=Symbol("bus$i"))
        end

        # Push component-level parameters from the static CSV definitions.
        apply_csv_params!(bus, load_df, i)
        apply_csv_params!(bus, machine_df, i)
        apply_csv_params!(bus, avr_df, i)
        apply_csv_params!(bus, gov_df, i)
        if nn == false
            # apply_csv_params!(bus, inverter_df, i)
        elseif row.category == "inverter"
            init_neuralnet!(bus, 4, 4, 4, rng)
        end

        # Configure the power-flow constraints used by initialize_from_pf!.
        pf_model = if row.bus_type == "PQ"
            pfPQ(P=row.P, Q=row.Q)  ## Load bus: fixed P and Q
        elseif row.bus_type == "PV"
            pfPV(P=row.P, V=row.V)  ## Generator bus: fixed P and V
        elseif row.bus_type == "Slack"
            pfSlack(V=row.V, δ=0)   ## Slack bus: fixed V and angle
        end
        set_pfmodel!(bus, pf_model)

        push!(busses, bus)
    end

    @named piline_template = compile_line(MTKLine(PiLine_fault(;name=:piline)))

    branches = []
    for row in eachrow(branch_df)
        # Create line instance with topology
        line = compile_line(piline_template; src=row.src_bus, dst=row.dst_bus)

        # Apply electrical parameters from CSV data
        for col_name in names(branch_df)
            if col_name ∉ ["src_bus", "dst_bus", "transformer"]
                set_default!(line, Regex(col_name*"\$"), row[col_name])
            end
        end

        push!(branches, line)
    end

    nw = Network(busses, branches)

    # formula1 = @initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
    # formula2 = @initformula :inverter₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
    # set_initformula!(nw[VIndex(31)], formula1)
    # set_initformula!(nw[VIndex(39)], formula1)
    # set_initformula!(nw[VIndex(32)], formula2)
    return nw
end

# mutable struct Perturb
#     P::Float64
#     Q::Float64
#     BUS::Int
#     t::Float64
# end
function set_perturbation!(nw, bus_idx, load_change, t_event)
    # Apply a load-step event to one PQ bus at a prescribed time.
    u0 = NWState(nw) # state is stored in metadata because of mutating init function!
    u0.p[vidxs(nw, :, :ZIPLoad₊KpZ)] .= 0.0
    u0.p[vidxs(nw, :, :ZIPLoad₊KqZ)] .= 0.0
    u0.p[vidxs(nw, :, :ZIPLoad₊KpC)] .= 1.0
    u0.p[vidxs(nw, :, :ZIPLoad₊KqC)] .= 1.0
    set_defaults!(nw, u0.p)
    _change_load = ComponentAffect([], [:ZIPLoad₊Pset, :ZIPLoad₊Qset]) do u, p, ctx
        p[:ZIPLoad₊Pset] += load_change[1]
        p[:ZIPLoad₊Qset] += load_change[2]
    end
    load_change_cb = PresetTimeComponentCallback(t_event, _change_load)
    set_callback!(nw, VIndex(bus_idx), load_change_cb)
end

function set_linetrip!(nw, line_idx, t_event)
    # Apply a short-circuit followed by line disconnection on one branch.
    # Define callback to enable the short circuit first.
    _enable_short = ComponentAffect([], [:piline₊shortcircuit]) do u, p, ctx
        # @info "Short circuit activated on line $(ctx.src)→$(ctx.dst) at t = $(ctx.t)s"
        p[:piline₊shortcircuit] = 1
    end
    shortcircuit_cb = PresetTimeComponentCallback(t_event[1], _enable_short)
    # Then clear the fault by permanently disconnecting the affected line.
    _disable_line = ComponentAffect([], [:piline₊active]) do u, p, ctx
        # @info "Line $(ctx.src)→$(ctx.dst) disconnected at t = $(ctx.t)s"
        p[:piline₊active] = 0
    end
    deactivate_cb = PresetTimeComponentCallback(t_event[2], _disable_line)
    set_callback!(nw, EIndex(line_idx), (shortcircuit_cb, deactivate_cb))
end;
