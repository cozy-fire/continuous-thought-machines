#  CTM跨任务联合训练

基于 [CTM 源项目](https://github.com/SakanaAI/continuous-thought-machines/blob/main/README.md)，本项目以 **On-policy Distillation（OPD）与 Progress & Compress（P&C）** 验证 CTM 在 2D Maze medium 和 MiniGrid FourRooms 间的知识学习、压缩与保留。当前方案从随机初始化开始，直接使用专一教师。

## 统一输入与双列结构

学生只接收 $3\times84\times84$ RGB，不接收任务 ID 或教师专用观察。两任务统一为五维动作：Maze 为上、下、左、右、等待；FourRooms 为左转、右转、前进及两个等待槽。

知识库 **KB** 与活动列 **Active** 各有独立的 ResNet34-2（GroupNorm）、空间 Attention、CTM 和 Actor。编码器输出 $128\times21\times21$ 空间特征；共享 KB 在 Maze 每次观察执行 5 ticks，在 FourRooms 执行 2 ticks，神经元滑动记忆固定为 40 个内部 ticks。计算次数由外部任务上下文选择，不作为网络输入；它同时改变计算深度和历史观察在窗口中的保留范围。Attention 根据神经元同步状态查询空间特征。

每个 tick 先推进 KB，再将其当前 post activation 经过可训练的门控 Adapter 输入 Active 的 synapse；两列使用当前任务相同的 ticks，侧向输入停止向 KB 反传。每个 P 开始时，Active 复制当前 KB 的视觉权重，随机初始化控制器、Actor 和 Adapter；完整旧 KB 冻结。首次 KB 尚无知识时关闭侧向连接。

## Progress：专一教师指导学生轨迹

P 由 **Active 自己采样动作**，教师在学生实际到达的状态提供动作分布。Active 的视觉编码器、CTM、Actor 和 Adapter 一起训练；完整旧 KB 与专一教师冻结。

- **Maze：** 算法教师读取当前完整原生地图，用 BFS 计算当前位置到终点的最短路径，将第一步动作作为 one-hot 目标。每步根据学生实际位置重新求解，不读取数据集答案路线。
- **FourRooms：** 冻结的专一 CTM 教师读取同一环境状态的原生符号观察，连续推进自己的状态。七动作概率映射为学生五动作分布：

$$
p_{\mathrm{expert}}=
\left[p_0,p_1,p_2,\frac{p_3+p_4+p_5+p_6}{2},\frac{p_3+p_4+p_5+p_6}{2}\right].
$$

两任务从 episode 的第一步开始教学，无预热或 burn-in。纯动作蒸馏目标为

$$
\mathcal L_P=\frac1N\sum_{t\in\mathcal V}
D_{\mathrm{KL}}\!\left(p_{\mathrm{expert},t}\Vert\pi_{\mathrm{Active},t}\right).
$$

Maze 的 one-hot KL 等价于最短路径动作的交叉熵。环境奖励仅用于记录和评估，不进入训练目标。

## Compress：压缩完整策略并保留旧知识

C 冻结 P 结束时的完整双列策略，包括当时的旧 KB、两套视觉编码器和 Adapter。环境动作由该双列教师采样；独立 KB 在同一轨迹上学习教师动作分布。KB 的视觉编码器、CTM 和 Actor 都接受蒸馏梯度，不直接用 Active 覆盖 KB。

$$
\mathcal L_C=\frac1N\sum_{t\in\mathcal V}
D_{\mathrm{KL}}\!\left(\pi_{\mathrm{Dual},t}\Vert\pi_{\mathrm{KB},t}\right)
+\frac{250}{2}\sum_j\Omega_j(\theta_j-\theta_j^\star)^2.
$$

Online EWC 同时约束 KB 的视觉与控制器参数。首个 C 无历史约束；后续用累计重要性 $\Omega$ 和最近一次压缩后的参数中心 $\theta^\star$ 保留已学知识。

## Fisher 与跨任务循环

F 由当前 KB 采样轨迹，逐样本计算策略 score 梯度平方，不更新模型参数：

$$
\widehat F_j=\frac1K\sum_{n=1}^{K}
\left[\partial_{\theta_j}\log\pi_{\mathrm{KB}}(a_n\mid H_n)\right]^2,
\qquad
\Omega_j\leftarrow0.3\,\Omega_j+\widehat F_j.
$$

当前进行两次完整任务遍历，每次依次执行 **Maze $P\rightarrow C\rightarrow F$，FourRooms $P\rightarrow C\rightarrow F$**。连续状态在 episode reset 时恢复可学习初态；每个学习窗口保存采集起点状态，更新时重放全部新观察，窗口之间截断梯度而保留采集状态。

P 后评估完整 Active，C 后评估 KB；每个策略都在两个任务的固定面板上测试，最终独立测试 KB。成功率与跨任务遗忘用于判断学习和保留效果。该方案验证教师知识的学习与压缩；是否产生正向迁移仍需进一步对照实验。
