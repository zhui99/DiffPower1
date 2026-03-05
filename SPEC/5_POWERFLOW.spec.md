Role: SciML Researcher / QA
Tools: Julia, CSV.jl, DataFrames.jl, Lux.jl, JLD2.jl

Goal:
实现并验证 IEEE39 系统潮流残差 `power_flow` 与其雅可比（AutoForwardDiff / 解析）的一致性。

Files:
- Script: `PowerFlow.jl`
- Network data: `ieee39-inverter/branch.csv`
- Dataset: `data/data.jld2` (flattened shape `(5n, T, S)`)

State Convention:
- 每个母线状态顺序固定为 `[omega_dev, delta, V, P, Q]`
- `n = 39`, 总状态维度 `5n = 195`
- 输入支持单样本 `(195,)` 和 batch `(195, B)`

Ybus:
- 基于 `PiLine` 参数构建复导纳矩阵
- 使用字段 `R, X, G_src, B_src, G_dst, B_dst, r_src, r_dst, active`

Power Flow Residual:
- 函数: `power_flow(u_flat, t; Ybus)`
- 内部将 `u_flat` reshape 为 `(5, n, B)`
- 计算:
  - `V_complex = V .* cis.(delta)`
  - `I_complex = Ybus * V_complex`
  - `S_calc = V_complex .* conj.(I_complex)`
- 数据符号约定:
  - 数据中的 `P, Q` 与注入定义相反
  - 残差定义为
    - `ΔP = P_inj + P`
    - `ΔQ = Q_inj + Q`
- 输出形状: `(2n, B)`

Jacobian:
1. AutoDiff Jacobian:
   - `pf_jacobian(u_flat, t; Ybus)`
   - 使用 `Lux.batched_jacobian(..., AutoForwardDiff(), ...)`
   - 单样本返回 `(2n, 5n)`，batch 返回 `(2n, 5n, B)`

2. Analytic Jacobian:
   - `pf_jacobian_analytic_single(u_flat, t; Ybus)`
   - 使用复数块公式（`dS_ddelta`, `dS_dV`）
   - 使用 `cat + permutedims + reshape` 完成列交错
   - 列顺序与扁平输入一致: `[ω1,δ1,V1,P1,Q1, ω2,...]`
   - `d(ΔP)/dP = +I`, `d(ΔQ)/dQ = +I`

3. Jacobian Compare:
   - `compare_pf_jacobian(u_flat, t; Ybus)`
   - 统计 `max_abs_diff` 与 `mean_abs_diff`

Smoke Test:
- `smoke_test_power_flow()`:
  - 读取 batch 数据 `(195, B)`
  - 验证残差形状和数值有限性
  - 验证 Jacobian（Auto / Analytic）形状
  - 对比误差统计
- 运行方式:
  - `julia --project=. PowerFlow.jl`

Current Expected Behavior:
- 残差数量级在稳态附近应接近 0（约 `1e-6 ~ 1e-5`）
- 解析雅可比与 AutoForwardDiff 误差应在数值精度量级（约 `1e-4` 最大误差以内）
