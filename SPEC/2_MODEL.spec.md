Role: SciML Researcher
Tools: Lux.jl, DifferentialEquations.jl, ComponentArrays.jl

Functional Spec (Current `Model.jl` Implementation):
1. 状态定义
    设系统有 `n` 个总线，每个总线状态为
    `x_i = [omega_dev, delta, V, P, Q]`，状态维度固定为 `5`。
    全系统扁平化后状态维度为 `5*n`（IEEE39 时为 `195`）。

2. 模型结构
    当前模型使用 `HeteroBusDynamics`（Lux 自定义层）：
    - `Embedding(n_buses => embed_dim)`，默认 `embed_dim=5`，用于学习节点身份异构性。
    - `MLP = Dense(5+embed_dim => 10, gelu) -> Dense(10 => 5)`。
    - 不使用 GCN，不使用 ManifoldProjection，不包含潮流约束项。

3. 前向输入/输出约定
    `HeteroBusDynamics(u_flat, ps, st)` 支持两种输入：
    - 单样本：`u_flat::Vector`，形状 `(5*n,)`
    - Batch：`u_flat::Matrix`，形状 `(5*n, B)`
    前向返回同形状导数：
    - 单样本输出 `(5*n,)`
    - Batch 输出 `(5*n, B)`

4. 内部计算流程
    - 将扁平状态 reshape 为 `(5, n, B)`；
    - 通过 `Embedding(1:n)` 得到 `(embed_dim, n)`；
    - 沿 batch 维复制 embedding 并与状态在特征维拼接；
    - 展平为 `(5+embed_dim, n*B)` 送入 MLP；
    - 输出 reshape 回 `(5*n, B)`（或向量形式）。

5. ODE 封装方式
    使用如下闭包供 `DifferentialEquations.jl` 求解：
    `f_ode(u, p, t) = first(model(u, p, st))`
    可直接对单样本或 batch 初值进行积分。

Batch 说明（当前实现）:
- Batch 通过列拼接表达，即把 `B` 个样本组织为 `(5*n, B)`。
- 同一套参数共享于 batch 全部样本。
- 不构建并图拓扑，不做图重绑。

Validation:
- `TestModelForward.jl` 已验证：
  - 单样本前向可运行；
  - Batch 前向可运行；
  - 单样本/Batch ODE 前向可运行；
  - 可输出“预测轨迹 vs 真实轨迹”对比图。

          
