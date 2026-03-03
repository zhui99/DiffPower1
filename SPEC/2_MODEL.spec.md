Role: SciML Researcher
Tools: Lux.jl, LuxCore.jl, DifferentialEquations.jl, GNNLux.jl, ComponentArrays.jl
Functional Spec:
    Graph Encoder: 输入维度为 `[num_features + embed_dim, num_buses]`。当前实现中 `num_features=5`，并为每个 bus 追加一个可训练 `bus embedding`（默认 `embed_dim=4`），以增强对节点身份和异质性的建模能力。
    GCN Layer: 优先直接使用 `GNNLux.jl` 提供的 `GCNConv` 构建图卷积层，不再手写邻接矩阵乘法逻辑。电网图由 `branch.csv` 构建，并作为运行时输入显式传入 GCN encoder。
    Encoder Compression: Encoder 应先经过 1-2 层 GCN 提取节点特征，再按每个 `39-bus` 子图进行聚合（当前实现为均值聚合），随后使用 MLP 将聚合后的图级特征压缩到低维 latent 空间，以减小 ODE 状态维度。
    Neural ODE (Evolution): 定义 `d_latent/dt = NN(latent, p)`，其中 `NN` 为普通 MLP，而不是图卷积网络。该 MLP 仅在压缩后的 latent 空间中演化，以降低 stiff ODE 的计算成本。求解器使用 `AutoTsit5(Rodas5P(autodiff=false))` 以兼顾效率与刚性系统稳定性。
    Decoder: Decoder 应采用 Transpose-GCN 风格结构：先通过 MLP 将低维 latent 扩展到节点级隐藏特征，再通过 1-2 层图卷积沿拓扑恢复到原始物理空间。当前实现中由于电网图是无向归一化图，解码端可直接使用 `GCNConv` 作为转置式图恢复算子。输出维度保持为 `[num_features, num_buses]`，即仍只恢复物理状态量，不输出 embedding。
    Activation: Encoder 的 GCN 后处理、latent dynamics MLP 与 decoder MLP 均统一使用 `gelu` 激活函数。
    Batch Support:
        模型前向接口统一采用二维输入 `[feature, total_nodes]`。
        多个 `39-bus` 样本组成 batch 时，应先将各样本沿节点维拼接成更大的二维张量，例如 `5 x (39 * B)`。
        同时通过 `GNNLux.batch(...)` 将基础 `39-bus` 图复制并合并为不连通大图，并将该大图作为 `rollout` 的输入拓扑，使每个子图互不连边，从而在一次前向中实现 batch。
        对合并图输入，前向输出应为 `[feature, total_nodes, time]`。
Model Integration:
    使用自定义容器封装 `encoder / dynamics / decoder`，并在 `setup_model` 中通过 `LuxCore.setup` 分别初始化三个子模块的 `ps` 与 `st`。
    `rollout` 内部当前直接使用矩阵型 latent 状态调用 ODE 求解器，不再需要对 latent 做 `vec/reshape` 扁平化转换。
    `rollout` 应支持显式接收 `graph` 输入；单场景默认使用基础 `39-bus` 图，多场景 batch 时由调用方传入预先构造好的不连通并图，而不在模型内部按节点数自动重绑图结构。
    由于 latent 定义已变化，修改 encoder/dynamics/decoder 结构后应视为新模型版本，旧 checkpoint 不保证兼容。
