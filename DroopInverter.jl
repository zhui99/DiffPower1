# Library components for inverter dynamics used by the IEEE 39-bus builder.
# This file defines baseline PI/lead blocks plus droop and VSG inverter models.
using PowerDynamics, NetworkDynamics, ModelingToolkit
using PowerDynamics.Library
using ModelingToolkit: t_nounits as t, D_nounits as Dt

@mtkmodel sPI begin
    # Simple PI controller used in the inverter inner control loops.
    @structural_parameters begin
        Kp
        Ki
    end
    @variables begin
        in(t), [description="Controller input", input=true]
        out(t), [description="Controller output", output=true, guess=0]
        integral(t), [description="Integral state", guess=0]
    end
    @equations begin
        Dt(integral) ~ in
        out ~ Kp * in + Ki * integral
    end
end;

@mtkmodel SimpleLead begin
    # First-order lead/lag style block used to approximate inverter output dynamics.
    @structural_parameters begin
        K # Gain
        T # Time constant
        guess=0
    end
    @variables begin
        in(t), [guess=guess, description="Input signal", input=true]
        out(t), [description="Output signal", output=true]
    end
    @equations begin
        T*Dt(in) ~ K*out - in
    end
end;

@mtkmodel DroopInverter begin
    # Baseline droop-controlled inverter model.
    @components begin
        terminal = Terminal()
        PI_Vd = sPI(Kp=5, Ki=5)
        PI_Vq = sPI(Kp=5, Ki=5)
        PI_Id = sPI(Kp=3, Ki=5)
        PI_Iq = sPI(Kp=3, Ki=5)
        Lead_Ed = SimpleLead(K=1, T=0.01, guess=1)
        Lead_Eq = SimpleLead(K=1, T=0.01, guess=0)
    end

    @parameters begin
        Pset, [description="Active power setpoint [pu]", guess=1]
        Qset, [description="Reactive power setpoint [pu]", guess=0]
        Vset, [description="Voltage magnitude setpoint [pu]", guess=1]
        ω_b=2π*50, [description="Nominal frequency rad/s"]
        Kp=0.033, [description="Active power droop coefficient [pu]"]
        Kq=0.045, [description="Reactive power droop coefficient [pu]"]
        τ_p = 0.01, [description="Active Power filter time constant [s]"]
        τ_q = 0.01, [description="Reactive Power filter time constant [s]"]
        Rf = 0.0, [description="Filter resistance [pu]"]
        Xf = 0.15, [description="Filter reactance [pu]"]
    end

    @variables begin
        Ed(t), [description="d-axis voltage command", guess=1]
        Eq(t), [description="q-axis voltage command", guess=0]
        Id_ref(t), [description="d-axis current reference"]
        Iq_ref(t), [description="q-axis current reference"]
        Id(t), [guess=0, description="d-axis current"]
        Iq(t), [guess=0, description="q-axis current"]
        Vd(t), [guess=0, description="d-axis voltage"]
        Vq(t), [guess=1, description="q-axis voltage"]
        Pe(t), [description="Active power [pu]", guess=1]
        Qe(t), [description="Reactive power [pu]", guess=0]
        Pfilt(t), [description="Filtered active power [pu]", guess=1]
        Qfilt(t), [description="Filtered reactive power [pu]", guess=1]
        ω(t), [description="Frequency [pu]"]
        δ(t), [description="Voltage angle [rad]", guess=0]
        V(t), [description="Voltage magnitude [pu]"]
    end
    # rotation
    begin
        T_to_loc(α)  = [ sin(α) -cos(α);
                         cos(α)  sin(α)]
        T_to_glob(α) = [ sin(α)  cos(α);
                        -cos(α)  sin(α)]
    end

    @equations begin
        ## Transform terminal voltages to dq frame
        [terminal.u_r, terminal.u_i] .~ T_to_glob(δ) * [Vd, Vq] 
        [Id, Iq] .~ T_to_loc(δ)*[terminal.i_r, terminal.i_i]
        ## Power measurement from terminal quantities
        Pe ~ Vd*Id + Vq*Iq
        Qe ~ Vq*Id - Vd*Iq
        ## First-order low-pass filtering
        τ_p * Dt(Pfilt) ~ Pe - Pfilt
        τ_q * Dt(Qfilt) ~ Qe - Qfilt
        ## Droop control equations
        ω ~ 1 - Kp * (Pfilt - Pset)  # Frequency decreases with excess power
        V ~ Vset - Kq * (Qfilt - Qset)  # Voltage decreases with excess reactive power
        Dt(δ) ~ ω_b * (ω - 1)

        # Voltage Loop
        PI_Vd.in ~ V - Vd
        Id_ref ~ PI_Vd.out
        PI_Vq.in ~ 0 - Vq
        Iq_ref ~ PI_Vq.out

        ## Current Loop
        PI_Id.in ~ Id_ref - Id
        Ed ~ PI_Id.out + Vd + Id*Rf - Iq * Xf
        PI_Iq.in ~ Iq_ref - Iq
        Eq ~ PI_Iq.out + Vq + Iq*Rf + Id * Xf

        ## Inverter output dynamics
        Lead_Ed.in ~ Ed
        Lead_Eq.in ~ Eq
        
        Lead_Ed.out ~ Vd + Id*Rf - Iq * Xf
        Lead_Eq.out ~ Vq + Iq*Rf + Id * Xf

    end
end;

@mtkmodel VSGInverter begin
    # Virtual synchronous generator variant used as the default physical inverter.
    @components begin
        terminal = Terminal()
        PI_Vd = sPI(Kp=5, Ki=5)
        PI_Vq = sPI(Kp=5, Ki=5)
        PI_Id = sPI(Kp=3, Ki=5)
        PI_Iq = sPI(Kp=3, Ki=5)
        Lead_Ed = SimpleLead(K=1, T=0.01, guess=1)
        Lead_Eq = SimpleLead(K=1, T=0.01, guess=0)
    end

    @parameters begin
        Pset, [description="Active power setpoint [pu]", guess=1]
        Qset, [description="Reactive power setpoint [pu]", guess=0]
        Vset, [description="Voltage magnitude setpoint [pu]", guess=1]
        ω_b=2π*50, [description="Nominal frequency rad/s"]
        H = 2, [description="Inertia constant [s]"]
        D = 30, [description="Damping coefficient [pu]"]
        Kq=0.045, [description="Reactive power droop coefficient"]
        τ_p = 0.01, [description="Active Power filter time constant [s]"]
        τ_q = 0.01, [description="Reactive Power filter time constant [s]"]
        Rf = 0.0, [description="Filter resistance [pu]"]
        Xf = 0.15, [description="Filter reactance [pu]"]
    end

    @variables begin
        Ed(t), [description="d-axis voltage command", guess=1]
        Eq(t), [description="q-axis voltage command", guess=0]
        Id_ref(t), [description="d-axis current reference"]
        Iq_ref(t), [description="q-axis current reference"]
        Id(t), [guess=0, description="d-axis current"]
        Iq(t), [guess=0, description="q-axis current"]
        Vd(t), [guess=0, description="d-axis voltage"]
        Vq(t), [guess=1, description="q-axis voltage"]
        Pe(t), [description="Active power [pu]", guess=1]
        Qe(t), [description="Reactive power [pu]", guess=0]
        Pfilt(t), [description="Filtered active power [pu]", guess=1]
        Qfilt(t), [description="Filtered reactive power [pu]", guess=1]
        ω(t), [description="Frequency [pu]", guess=1]
        δ(t), [description="Voltage angle [rad]", guess=0]
        V(t), [description="Voltage magnitude [pu]"]
    end
    # rotation
    begin
        T_to_loc(α)  = [ sin(α) -cos(α);
                         cos(α)  sin(α)]
        T_to_glob(α) = [ sin(α)  cos(α);
                        -cos(α)  sin(α)]
    end

    @equations begin
        ## Transform terminal voltages to dq frame
        [terminal.u_r, terminal.u_i] .~ T_to_glob(δ) * [Vd, Vq] 
        [Id, Iq] .~ T_to_loc(δ)*[terminal.i_r, terminal.i_i]
        ## Power measurement from terminal quantities
        Pe ~ Vd*Id + Vq*Iq
        Qe ~ Vq*Id - Vd*Iq
        ## First-order low-pass filtering
        τ_p * Dt(Pfilt) ~ Pe - Pfilt
        τ_q * Dt(Qfilt) ~ Qe - Qfilt
        ## Droop control equations
        2H * Dt(ω) ~ -D * (ω - 1) - (Pfilt - Pset)  # Frequency decreases with excess power
        V ~ Vset - Kq * (Qfilt - Qset)  # Voltage decreases with excess reactive power
        Dt(δ) ~ ω_b * (ω - 1)

        # Voltage Loop
        PI_Vd.in ~ V - Vd
        Id_ref ~ PI_Vd.out
        PI_Vq.in ~ 0 - Vq
        Iq_ref ~ PI_Vq.out

        ## Current Loop
        PI_Id.in ~ Id_ref - Id
        Ed ~ PI_Id.out + Vd + Id*Rf - Iq * Xf
        PI_Iq.in ~ Iq_ref - Iq
        Eq ~ PI_Iq.out + Vq + Iq*Rf + Id * Xf

        ## Inverter output dynamics
        Lead_Ed.in ~ Ed
        Lead_Eq.in ~ Eq
        
        Lead_Ed.out ~ Vd + Id*Rf - Iq * Xf
        Lead_Eq.out ~ Vq + Iq*Rf + Id * Xf

    end
end;
