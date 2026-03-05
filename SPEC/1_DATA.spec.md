Role: Data Engineer
Tools: PowerDynamics.jl, NetworkDynamics.jl, DifferentialEquations.jl, JLD2.jl, CairoMakie.jl

Functional Spec (must match current `GenData.jl`):

1. Environment and Entry
- Script: `GenData.jl`
- Create folders: `./data`, `./results`
- CLI:
  - `julia --project=. GenData.jl`
  - `julia --project=. GenData.jl <n_scenarios> <seed>`
- Defaults:
  - `DEFAULT_SCENARIOS = 100`
  - `DEFAULT_SEED = 20260228`
  - `NETWORK_NAME = "ieee39-inverter"`

2. Simulation Base
- Build IEEE-39 system with `get_IEEE39_base("ieee39-inverter")` from `ieee39.jl`.
- Initialize from power flow: `initialize_from_pf!(nw)`.
- Simulate with:
  - `ODEProblem(nw, u0, (0.0, 5.0))`
  - solver `Rodas5P()`
  - `abstol=1e-8`, `reltol=1e-6`, `maxiters=2_000_000`

3. Scenario Policy
- Only `load_step` is enabled.
- Disturbance time: `LOAD_EVENT_TIME = 0.80`.
- Disturbance bus: random from PQ buses with load (`bus_type == "PQ"` and `has_load == true`).
- Disturbance magnitude:
  - `dP = clamp(0.35 * randn(rng), -0.8, 0.8)`
  - `dQ = 0.0`
- Failed/unstable scenarios are dropped directly:
  - non-success retcode
  - any non-finite extracted feature
- Retry budget:
  - `max_attempts = max(n_scenarios * 4, n_scenarios)`

4. Feature Extraction
- Fixed sampling interval: `DT = 0.01`
- Save only post-event trajectory:
  - `times = (0.80 + 0.01):0.01:5.0`
- Per-bus features (feature_dim = 5):
  - `[omega_dev, delta, V, P, Q]`
  - `omega_dev = ω - 1.0` (machine/inverter category-specific index)
  - `delta, V` reconstructed from `busbar₊u_r`, `busbar₊u_i`
  - `P, Q` read directly from `busbar₊P`, `busbar₊Q`

5. Flattening Rule (critical)
- Raw one-scenario tensor shape before flatten:
  - `(feature, bus, time) = (5, 39, T)`
- Saved one-scenario matrix shape after flatten:
  - `(state_dim, time) = (195, T)`
- Flatten implementation:
  - For each time `k`: `flat[:, k] = vec(tensor[:, :, k])`
- Therefore flatten order is:
  - feature index varies fastest, then bus index (Julia column-major behavior).

6. Dataset Save Contract
- Multi-scenario stacked shape:
  - `data`: `(state_dim, time, scenario)` i.e. `(195, T, N)`
- Saved fields in `data/data.jld2`:
  - `data`
  - `times`
  - `FEATURE_NAMES = ["omega_dev","delta","V","P","Q"]`
  - `NUM_BUSES = 39`
- Do not store:
  - `retcode`, `event_type`, `target`, `delta`, `fault`

7. Preview Output
- Reconstruct first scenario from flattened matrix for visualization.
- Plot generator buses (10 buses, 30-39) for all 5 features.
- Save figure:
  - `results/sample_state_trajectories.png`

8. Reproducibility Checklist
- Run:
  - `julia --project=. GenData.jl 100 20260228`
- Expected log characteristics:
  - reports flattened shape `(195, T, N)`
  - confirms post-event only (`t >= 0.81s`)
  - prints omega statistics
- Validate JLD2 keys:
  - `data`, `times`, `FEATURE_NAMES`, `NUM_BUSES`
