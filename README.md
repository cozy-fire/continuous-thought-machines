# TAPD & CTM跨任务联合训练

本项目基于 [CTM 源项目](https://github.com/SakanaAI/continuous-thought-machines/blob/main/README.md)，在 2D Maze（mazes-medium）和 MiniGrid FourRooms 上研究共享视觉表征与逐任务知识压缩。两种环境统一为 $3\times84\times84$ RGB 观察和五维离散动作；FourRooms 的两个额外动作槽执行等待。策略不接收任务 ID。

## 架构

随机初始化的 ResNet34-2 使用 32 组 GroupNorm，输出 $128\times21\times21$ 空间特征。KB 与 Active 各有独立的空间 cross-attention 和 CTM 控制器；KB 是无 Critic 的策略，Active 是 Actor-Critic。每次观察推进 2 个内部 tick，CTM 保存 40 tick 记忆。每个 tick 先更新 KB，Adapter 将其当前 post activation 经 LayerNorm、线性投影和零初始化门控传给 Active 的 synapse 输入。KB 在 Active 学习时冻结；Active 每轮 X 和每个 P 阶段随机初始化。

## Task-Agnostic：$X\rightarrow C\rightarrow F$

默认执行 4 次 visit，每次按 Maze→FourRooms 顺序访问，每个任务执行 2 轮 $X\rightarrow C\rightarrow F$。每轮 X 恰好 1,000,000 环境步，无外在任务奖励。每个环境槽仅保存当前 episode 最近 20 张原始 RGB 帧；reset 时以初始帧填满窗口。动作后的真实末帧先与窗口比较，再入窗。设余弦相似度为 $s_i$，候选帧与当前帧的间隔为 $d_i\in[1,20]$，唯一 PPO 回报为

$$
q_i=\operatorname{clip}\left(\frac{s_i-0.999}{0.001},0,1\right),\quad
w(d_i)=0.2+0.8\frac{20-d_i}{19},\quad
r_t=-\max_i q_iw(d_i).
$$

X 用循环 PPO-clip 同时更新 Active、Adapter、Critic 和共享视觉编码器 $E$。对有效观察的视觉特征另计算 $z=g(\operatorname{GAP}(E(o)))$，其中 $g$ 是 $128\rightarrow512\rightarrow128$ 的两层 projector；单次反传的目标是

$$
\mathcal L_X=\mathcal L_{\mathrm{PPO}}+0.02\,\mathcal L_{\mathrm{SIGReg}}(z)/N.
$$

SIGReg 通过随机方向的经验特征函数约束 $z$ 的分布；projector 只接受该项梯度。Active 的优化器状态每轮重置，$E+g$ 的 AdamW 状态跨轮保留。X 不使用动作条件预测器、世界模型 MSE 或 high-error replay。

C 冻结完整双列教师与视觉编码器，将教师动作分布蒸馏到 KB，并使用 Online EWC：

$$
\mathcal L_C=\mathbb E_t D_{\mathrm{KL}}(\pi_T\|\pi_{\mathrm{KB}})
+\frac{\lambda_{\mathrm{EWC}}}{2}\sum_i\Omega_i(\theta_i-\theta_i^*)^2.
$$

C/F 在学习窗口前使用 20 张历史观察（40 tick）进行无梯度 burn-in。F 在 KB 自己采样的轨迹上估计策略 Fisher，对权重不做优化：

$$
\widehat F_i=\frac1N\sum_n\left[\partial_{\theta_i}\log\pi_{\mathrm{KB}}(a_n\mid H_n)\right]^2,
\quad\Omega_i\leftarrow\gamma_F\Omega_i+\widehat F_i.
$$

## Progress & Compress：$P\rightarrow C\rightarrow F$

TA 完成后永久冻结共享视觉编码器。P 使用任务成功奖励与循环 PPO 训练新 Active；C/F 与 TA 同义，共进行 3 次完整任务遍历。PPO 的主要目标为

$$
\mathcal L_{\mathrm{PPO}}=-\mathbb E_t\min\!\left(\rho_t\widehat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t\right)
+\frac{c_v}{2}\mathbb E_t(V_t-\widehat R_t)^2-c_H\mathbb E_t\mathcal H(\pi_t).
$$

TA 每个 visit 末评估当前 KB；P&C 每个 visit 末评估当前 KB 和两个任务各自 P 结束时的 Active 快照。最终仅测试 KB。单任务 CTM、顺序 PPO 与无探索蒸馏 P&C 对照复用同 seed 的冻结 TA 视觉权重，因此报告中单列共享视觉预训练成本。运行、配置、schema v2 checkpoint 和 smoke 验证见 [实现说明](tasks/continual_nav/README.md)。
