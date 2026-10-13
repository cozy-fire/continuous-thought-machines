# W&B 指标速查表（schema v2）

主方法为 `tapd_ctm_visual_revisit`。以下面板只适用于 v2 新 X；v1 的 W.fit、预测误差和 high-error 池指标见 `v1-W_X_C_F` 分支。训练横轴 `global_env_steps` 是全程已采集的环境步，当前阶段内还加上本阶段已完成步数；评估步数不进入该轴。

| 面板 | 指标 | 含义 |
|---|---|---|
| `ta_X_{task}/*` | `policy_loss`, `value_loss`, `entropy`, `ppo_loss`, `approximate_kl` | 循环 PPO 的策略、价值、熵、合计损失及近似 KL；`ppo_loss` 未含 SIGReg。 |
| `ta_X_{task}/*` | `sigreg`, `loss` | `sigreg` 是除以有效观察数后的正则项；`loss` 为 PPO 加 `0.02×sigreg` 的联合目标。 |
| `ta_X_{task}/*` | `grad_norm`, `visual_grad_norm` | 裁剪前 Active 和 Encoder＋projector 两组的梯度范数。 |
| `ta_X_{task}/*` | `mean_penalty`, `nonzero_penalty_fraction`, `unchanged_pixel_fraction`, `identical_frame_fraction` | 截至该 rollout 的平均负回报绝对值、非零惩罚比例、RGB 元素不变比例和整帧完全相同的转移比例。 |
| `ta_X_{task}/*` | `mean_max_similarity`, `mean_match_similarity`, `similarity_bin_0..4_fraction` | 窗口内最大余弦值、最大综合惩罚对应的余弦值及前者的区间占比；区间边界依次为 0.95、0.99、0.999、0.9995。 |
| `ta_X_{task}/*` | `mean_match_gap`, `gap_1..20_fraction` | 最大综合惩罚对应的时间间隔均值与各间隔占比。 |
| `ta_X_{task}/*` | `action_0..4_fraction` | 该 X 阶段累计采样动作分布，含 FourRooms 两个等待槽。 |
| `ta_X_{task}/*` 与通用字段 | `stage_updates`；`x_joint_updates` | 前者是本轮累计联合优化器更新数，后者是历史行的全程累计 X 联合更新数；均不是环境步数。 |
| `{family}_C_{task}/*` | `kl`, `ewc`, `loss`, `grad_norm` 等 | 蒸馏教师到 KB 及 Online EWC 的分项、合计和梯度。 |
| `{family}_P_{task}/*` | PPO 损失与梯度 | 外在任务奖励下的 Active PPO；Encoder 在 P 永久冻结。 |
| `eval_*` | `success_rate`, `mean_return`, `mean_length` 等 | 固定面板评估，按 TA/P&C、KB/Active、来源任务及被评估任务拆分。 |
| `progress/*` | `global_env_steps`, `explore_steps`, `progress_steps`, `x_joint_updates`, `ppo_optimizer_updates`, `distill_optimizer_updates` | 仅在阶段完成时写入的累计预算与更新次数。 |

本地 `events.jsonl` 和 `metrics.jsonl` 与 W&B 来自同一事件流。每轮 X 的诊断文件位于 `diagnostics/`，至多包含四条完整 episode 的逐帧图像、动作、奖励、匹配相似度、间隔和终止信息。`evaluation/forgetting.json` 分别保存 TA、P&C 的 KB visit 末遗忘矩阵；Active 不进入该矩阵。
