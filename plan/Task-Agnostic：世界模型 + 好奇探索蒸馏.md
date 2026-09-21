# Task-Agnostic：世界模型 + 好奇探索蒸馏 

在任务无关阶段，训练按 `6` 个 visit 进行。每个 visit 按 `顺序` 依次访问 **2D Maze（mazes-medium）** 和 **MiniGrid 4Rooms**。每次访问一个任务，连续执行 **两轮“世界模型训练 → 好奇心探索 → 知识蒸馏”**。

每轮包含三个阶段：

1. **世界模型训练**  
使用 `[固定CTM（使用视觉编码器快照）作为策略]` 采集转移，并结合 `[好奇心探索阶段的高预测误差数据]`，训练共享 ResNet、`[视觉投影头]` 和动作条件预测器：

    $z_t=g(\operatorname{Pool}(E_\theta(o_t))),\qquad \hat z_{t+1}=F_\psi(z_t,a_t) \\  L_{\mathrm{vision}} = L_{\mathrm{forward}}(\hat z_{t+1},z_{t+1}) + \lambda_{\mathrm{SIG}}L_{\mathrm{SIGReg}}$

    本阶段冻结 CTM 控制器参数。预测器输入为 `[单帧表示]`，训练预算步数还没决定。  
    视觉投影头设置：池化后的 ResNet 特征额外经过MLP得到特征向量$z_t$用于后续动力学预测和SIGReg计算
2. **好奇心探索**  
冻结世界模型的全部参数及运行统计，以固定的预测误差生成 TAPD 式内在奖励：

    $ r_t^i=\operatorname{stopgrad}\!\left[\log(1+\|\hat z_{t+1}-z_{t+1}\|_2)\right]. $

    使用 PPO-clip 更新 Active CTM 及 Actor–Critic，KB 保持冻结。探索数据按上述数据规则保存，供后续世界模型训练使用。训练预算步数还没决定。
3. **知识蒸馏**  
固定世界模型和训练后的教师策略，将 Active 的动作分布蒸馏到 KB，采用 `[蒸馏损失与抗遗忘机制]`，预算步数还没决定。下一轮 Active 按 `[重置]` 处理，并通过 `[KB侧向连接方式：接在每个 tick 的 Synapse 输入或内部状态上]` 使用蒸馏后的知识。只蒸馏动作分布，Critic 按当前奖励目标学习

首次训练时，CTM 随机初始化。任务切换时，各模块及循环状态按`[共享 ResNet 不变；Active 的 Attention、CTM、Actor–Critic 随机初始化]`处理。

全部任务无关 visits 完成后，**冻结共享 ResNet 及其运行统计**，进入后续使用外在奖励的 P&amp;C 阶段，由 KB 和 Active 两套完整 CTM 控制器共同复用该视觉 backbone。



&nbsp;

&nbsp;
