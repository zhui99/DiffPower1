Role: Optimization Specialist
Tools: Optimization.jl, OptimizationOptimJL.jl, Zygote.jl
Pretrain:
    当前实现不再直接拟合显式时间微分 $\dot x$，而是采用单步预测预训练：
    对每个时间片段，仅取连续两帧 `[x(t), x(t+dt)]`，将 `x(t)` 作为输入，在固定时间窗 `(0, dt)` 上通过 NODE 前向积分预测 `x(t+dt)`。
    预训练损失定义为下一时刻状态的平均绝对误差 `MAE(x_pred(t+dt), x_true(t+dt))`。
    当前相邻采样步长固定为数据集存储的 `dt = times[2] - times[1]`，不再为每个片段单独重建局部时间轴。
    训练数据组织方式已调整为窗口级样本集：应先将每个场景拆成所有相邻两帧窗口，因此预训练样本总数为 `场景数 * (时间步 - 1)`。
    当前前向中实际被 ODE 积分的是“encoder 压缩后的图级 latent”，而不是原始 `39-bus` 节点特征图；因此 pretrain 的主要目标是先让 encoder / latent MLP dynamics / decoder 这条降维-演化-恢复链路对单步预测稳定收敛。
Functional Spec: 
    参考 StiffNODE.jl中训练的部分设置。
    Multiple Shooting:
        实现Multiple Shooting损失函数，将长序列划分为若干子时间段，以增强 ODE 训练的收敛性。
    Two-Stage Training:
        Stage 1: AdamW 优化器，学习率 0.01，进行快速粗调（300 epochs）。
        Stage 2: LBFGS 优化器，利用二阶信息进行精细收敛。
    Sensitivity:
        预训练阶段当前优先使用 `InterpolatingAdjoint(; autojacvec=ZygoteVJP())`。在本项目的单步损失设置下，该方法相较 `QuadratureAdjoint` 有更好的梯度计算吞吐量。
    Mini-batch:
        预训练批处理使用 `MLUtils.DataLoader` 对“窗口级样本”遍历 mini-batch，而不是对整段场景遍历。
        当前实现中应使用 `collate` 自定义批处理行为：将多个两帧窗口样本 `[feature, bus, 2]` 在节点维合并成一个更大的三维样本 `[feature, 39*B, 2]`。
        `collate` 还应同步构造 `GNNLux.batch(...)` 的不连通并图，并在前向时将该拓扑显式传入 `rollout`，而不在模型内部按节点数自动推断。
        合并后的样本在前向时通过不连通并图实现 batch，不再依赖显式四维 `batch_data[:, :, :, scenario_idx]` 遍历；encoder 会先对每个子图做聚合压缩，因此 batch 增大时，ODE 部分的状态维度随“子图个数”线性增长，而不是随“总节点数”线性增长。
        当前 batch 内每个样本已经是单个两帧窗口，因此不再需要在 `derivative_batch_loss` 内遍历全部时间步对。
Persistence:
    训练完成后，将最佳参数 ps 保存至 ./model/model_weights.jld2。
    若修改了 latent 维度、GCN hidden 维度、聚合方式或 decoder 展开方式，应重新训练并覆盖旧权重文件，而不是继续复用旧 checkpoint。
