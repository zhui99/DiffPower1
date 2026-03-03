Role: QA & Visualization 
EngineerTools: CairoMakie.jl, Statistics.jl
Functional Spec:
    Inference:
        加载 `model_weights.jld2`，对测试集中的单个 `load_step` 场景进行全时段预测。
        当前默认测试脚本仍以单场景推理为主；若引入并图 batch 推理，需在可视化前将合并图输出按每 `39` 个节点拆回单场景。
        当前模型的前向链路为“GCN 编码压缩 -> 低维 latent ODE -> Transpose-GCN 解码恢复”；因此测试权重必须与当前 `NODEModel.jl` 的 latent 定义和 decoder 结构一致，结构更新后应重新生成权重文件再做推理。
        当前 `LatentNeuralODE` 直接使用矩阵型 latent 状态积分，因此测试中看到的每一步 latent 都对应 `latent_dim x batch_count` 的矩阵，而不是扁平向量。
    Plotting:
        `pretrain_loss_curve.png` 或 `loss_curve.png`: 展示当前训练阶段的收敛过程。
        `node_rollout_comparison.png` / `pretrained_best_node_rollout_comparison.png`: 绘制某个关键母线的 True vs Predicted 对比曲线。
        对当前架构，建议重点关注故障初期的恢复段是否平滑，以确认“压缩 latent + Transpose-GCN decoder”没有引入不合理跳变。
        `error_heatmap.png`: 作为后续扩展项，可在 39 节点拓扑图或热力图上展示各节点平均预测误差。
