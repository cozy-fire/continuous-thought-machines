# 实现依据

- 02第4节：固定E_old+KB先采集，之后更新E/g/F；单帧动作条件预测，MSE+0.02 SIGReg。
- SIGReg按计划的ECF公式：256方向、17knots、0..3、B系数、对称梯形权重；两个时间切片共用方向。
- X的原始L2分数用于当前轮top-K；任务间不混合，不跨world版本排名。没有PPO之前只提供X转移接入接口，验证数据明确标为固定策略接口探针。
- fresh与高误差池只存uint8图像、动作、边界、ID和来源版本，无reward/latent。
- 全局checkpoint和阶段恢复仍属于交付5；本项实现分片/manifest原子发布及版本发布接口。
