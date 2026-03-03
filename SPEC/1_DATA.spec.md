Role: Data Engineer
Tools: PowerDynamics.jl, NetworkDynamics.jl, JLD2.jl 
Functional Spec:
    Environment: 创建 ./data 和 ./results 文件夹。
    Simulation Base: 加载 IEEE 39-bus 系统(参考 `ieee39.jl`; `ieee39-inverter` 为模型配置文件，不做修改), 利用 SciML 中的 ODE 求解方法 `Rodas5P()` 进行仿真。
    Scenario Generation: 当前仅保留 `load_step` 故障。
        Load Step: 随机选取一个 PQ 节点引入有功负荷突变；当前实现中 `dQ = 0`，即只保留有功扰动。
    Feature Extraction:
        采样步长固定为 `DT = 0.01s`。
        仅保存故障发生后的状态轨迹，不再保存故障前稳态段。
        每个母线统一提取 `\mathbf{x} = [\omega_{dev}, \delta, V, P, Q]`。
        其中 `P/Q` 直接读取 ODE 结果中的 busbar 量；`delta` 与 `V` 当前由母线电压实部/虚部恢复得到。
    Data Save:
        多次进行系统仿真（每次引入不同 `load_step` 场景），将每个母线的状态变化保存至 `data.jld2`。
        当前数据文件仅保存 `data`、`times` 和 `FEATURE_NAMES`，不再保存 `retcode`、`event_type`、`target` 等场景标签。
        若某次求解失败或状态提取出现非有限值，应直接丢弃该场景并生成下一个样本。
