# Project: IEEE 39-Bus Test Case Frequency Prediction via Neural ODE

## 🤖 Roles & Responsibilities
- **Architect (Lead)**: 负责定义数据结构、模型接口和全局超参数。
- **Data Engineer**: 负责 `PowerDynamics.jl` 仿真、故障场景生成及 JLD2 持久化。
- **SciML Researcher**: 负责 `Lux.jl` 模型构建，结合 GCN 与 Neural ODE，处理 stiff 系统。
- **Optimization Specialist**: 实现 Multiple Shooting 损失函数与 AdamW+LBFGS 二阶段训练策略。
- **QA & Visualization**: 负责绘图、模型推理性能评估及闭环调试。

## 📈 Workflow Pipeline
1.  **Phase 1: [DATA]** - 建立数据管道与仿真基准。
2.  **Phase 2: [MODEL]** - 构建 GCN-NODE 混合架构。
3.  **Phase 3: [TRAIN]** - 实现多段射击训练算法与持久化。
4.  **Phase 4: [VIZ_TEST]** - 可视化、闭环测试与超参数微调。

## 🎯 Success Criteria
- 能够自动生成并读取 `data.jld2`。
- Neural ODE 能够处理 IEEE 39-bus 的 stiff 动态（需使用 Rodas5P）。
- 最终预测误差在测试集上达到指定阈值，并产出三张核心分析图表。

## 🔧 Current Implementation Notes
- 数据集当前仅包含 `load_step` 场景，保存的是故障发生后的状态轨迹，采样步长为 `DT = 0.01s`。
- `NODEModel.jl` 当前使用“GCN 编码器 + 低维 Neural ODE + Transpose-GCN 解码器”的结构：编码器先用 2 层 `GNNLux.jl` 的 `GCNConv` 提取节点特征，再按每个 `39-bus` 子图做聚合，并用 MLP 压缩到低维 latent；`dynamics` 只在该低维 latent 空间中使用普通 MLP 演化；解码器先用 MLP 将 latent 扩展到节点级隐藏特征，再用两层图卷积按拓扑恢复物理状态。
- 模型输入状态为每个母线的 `5` 个物理量 `[omega_dev, delta, V, P, Q]`，并额外拼接每个母线的可训练 `bus embedding`（默认维度 `4`）以表征节点身份。
- 当前前向推理统一使用二维输入 `feature x total_nodes`。多个场景组成 mini-batch 时，会先在节点维拼接成更大的二维张量，并显式构造对应的不连通并图作为 `rollout` 的输入拓扑，而不再在模型内部按节点数自动重绑图结构。
- 当前编码器使用 `gelu` 激活函数，`dynamics` 和解码器内部的 MLP 也统一使用 `gelu`，以保持非线性的一致性。
- `LatentNeuralODE` 当前直接使用矩阵型 latent 状态进行 ODE 积分，不再对 latent 做 `vec/reshape` 扁平化转换。
- `PretrainNODE.jl` 当前采用单步预测预训练：输入 `x(t)`，在固定时间窗 `(0, dt)` 上预测 `x(t+dt)`，损失函数为下一时刻状态的 `MAE`。
- `PretrainNODE.jl` 当前会先将每个场景拆分为所有相邻两帧窗口，形成总样本数为 `场景数 * (时间步 - 1)` 的窗口级数据集；`DataLoader` 直接对这些窗口样本做 batch 采样，而不再在每个 batch 内遍历整段时间序列。
- 预训练敏感性算法当前使用 `InterpolatingAdjoint(; autojacvec=ZygoteVJP())`，原因是其在当前模型上的梯度计算速度显著快于 `QuadratureAdjoint`。
- `PretrainNODE.jl` 当前通过 `MLUtils.DataLoader(collate=...)` 将多个窗口样本沿节点维合并为一个大图样本，并在 `collate` 阶段同步生成对应的不连通拓扑，再把样本和拓扑一起送入训练。
- 由于模型参数结构已从“图空间 GCN dynamics”调整为“压缩 latent 空间 MLP dynamics”，旧的模型权重文件不再兼容当前 `NODEModel.jl`，修改结构后需要重新运行 pretrain。
