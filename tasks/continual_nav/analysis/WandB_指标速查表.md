# Continual Nav W&B 指标速查表

本文对应当前 `tasks.continual_nav` 的日志实现。代码通过指标键名中的 `/` 让 W&B 分组，不预设固定仪表盘布局；实际页面可能将同组指标拆成多张图。主方法的阶段前缀中，`ta` 是 Task-Agnostic，`pnc` 是 Progress & Compress，`{task}` 是**当前训练任务**（`maze_medium` 或 `fourrooms`）。

| 面板键名模式 | 横轴 | 写入时机 |
|---|---|---|
| `ta_W_{task}/*` | `world_optimizer_updates`：全程累计 W 更新数 | W.fit 每 50 次更新及该轮最后一次；阶段结束还写入一次最终指标。W.collect 不产出 Loss。 |
| `ta_X_{task}/*`、`pnc_P_{task}/*` | `global_env_steps`：全程累计训练环境步 | 每个 PPO rollout 完成后；阶段结束还写入一次最终指标。 |
| `ta_C_{task}/*`、`pnc_C_{task}/*` | `global_env_steps` | 每 10 个蒸馏窗口及阶段最后一个窗口；阶段结束还写入一次最终指标。 |
| `ta_F_{task}/*`、`pnc_F_{task}/*` | `global_env_steps` | Fisher 阶段结束时。 |
| `eval_{family}_{split}_{policy}_{evaluated_task}/*` | `global_env_steps` | visit 末 validation；全程结束后最终策略的 test。 |
| `eval_{family}_{split}_{policy}_performance/*` | `global_env_steps` | 同一次评估结束时。 |
| `progress_{split}_{policy}_{evaluated_task}/*` | `downstream_progress_steps`：全程累计 P 环境步 | P&C 或基线评估结束时。 |
| `progress/*` | `global_env_steps` | 每个训练阶段成功提交时。 |

`global_env_steps` 包含 W.collect、X、C、F、P 的训练交互，不包含 W.fit 的优化器更新或评估交互。阶段进行中，日志使用“此前已提交的训练步数＋本阶段已完成步数”；因此一个 X 阶段的横轴跨度不会从 0 开始。每轮 X 的独立预算由 `exploration.steps_per_round` 指定，不能用 `global_env_steps` 是否超过该值判断超预算。

## W.fit：世界模型与回放

适用于 `ta_W_{task}/*`。损失在当前 batch 上、执行参数更新前计算；`updates` 在每轮 W.fit 重新从 1 计数。

| 指标 | 含义与单位 | 解读提示 |
|---|---|---|
| `loss` | 本次世界模型总损失：`forward_mse + sigreg.lambda × sigreg`。 | 当前完整配置的 `sigreg.lambda=0.02`；不是评估回报。 |
| `forward_mse` | 预测下一帧潜变量与实际下一帧潜变量之间的逐元素均方误差。 | 观察动作条件预测是否改善。 |
| `sigreg` | 当前帧和下一帧潜变量的 SIGReg 损失平均值。 | 原始正则项尚未乘 `sigreg.lambda`。 |
| `grad_norm` | 参数梯度的范数，**裁剪前**取值。 | 可以超过配置的裁剪阈值。 |
| `updates` | 本轮 W.fit 已完成的优化器更新次数。 | 与全程累计的横轴 `world_optimizer_updates` 不同。 |
| `fresh_count`、`high_count` | 当前 batch 从 fresh 池、高误差池抽取的 transition 条数。 | 没有旧高误差池时 `high_count=0`。 |
| `replay_loaded_records` | 当前 W.fit 预加载的 transition 总数。 | shard 模式下该日志字段记为 0，不代表磁盘池为空。 |
| `replay_cache_bytes` | 当前 W.fit 内存 replay 缓存占用字节数。 | shard 模式下为 0；不等于整个进程的内存占用。 |
| `replay_preload_seconds` | 本轮 replay 预加载耗时，单位秒。 | 一次性耗时，每次更新日志重复记录该值。 |
| `replay_sample_seconds` | 从本轮 fit 开始累计的 batch 采样耗时，单位秒。 | 是累计值，不是单次采样耗时。 |
| `fit_seconds` | 从预加载结束到当前更新的累计训练耗时，单位秒。 | 不含 `replay_preload_seconds`；可用更新数差／时间差算稳态吞吐。 |

## X/P：循环 PPO

适用于 `ta_X_{task}/*` 和 `pnc_P_{task}/*`。每条常规日志记录的是**最近一个 rollout 中最后一个 minibatch**的损失，不是整段 rollout 的平均值；一个 rollout 的 `updates` 当前通常为 4。X 使用好奇心奖励，P 使用任务奖励，但当前 W&B 映射没有单独的训练 reward 或 episode return 指标。

| 指标 | 含义 | 解读提示 |
|---|---|---|
| `policy_loss` | PPO-clip 的策略损失，使用归一化优势。 | 可以为负；不能单凭它判断任务成功。 |
| `value_loss` | Critic 的回报目标误差：`0.5 × mean((value − return)^2)`。 | 指标内已有 `0.5`，总损失中还要乘 `ppo.vf_coef`。 |
| `entropy` | 当前动作分布的平均熵，单位 nat。 | 越高通常表示动作分布越分散；总损失中以负号乘熵系数。 |
| `loss` | `policy_loss + ppo.vf_coef × value_loss − ppo.entropy_coef × entropy`。 | 当前配置的系数分别是 `0.25` 和 `0.01`。 |
| `approximate_kl` | 基于新旧动作 log-prob 的近似 KL。 | 用于观察策略更新幅度，并非环境表现。 |
| `max_logprob_difference` | 当前 minibatch 中新旧动作 log-prob 差的最大绝对值。 | 不是 KL；容易受个别样本影响。 |
| `grad_norm` | 参数梯度的范数，裁剪前取值。 | 可超过配置的 `max_grad_norm`。 |
| `updates` | 最近一个 rollout 的 PPO 优化器更新次数。 | 不是全程累计更新数；累计数看 `progress/ppo_optimizer_updates`。 |

## C/F：蒸馏与 Fisher

C 的常规日志每 10 个窗口写一次，但 Loss 只取**最后一个窗口的最后一个 minibatch**；`updates` 也是该窗口的更新数，不是 10 个窗口的合计。

| 面板 | 指标 | 含义与注意事项 |
|---|---|---|
| `ta_C_{task}/*`、`pnc_C_{task}/*` | `kl` | 冻结教师双列策略与学生 KB 的动作分布 KL，按有效学习目标求均值。 |
| 同上 | `ewc` | 已乘 EWC 系数的 KB 参数保护惩罚；首次没有 Fisher 状态时为 0。 |
| 同上 | `loss` | 蒸馏总损失：`kl + ewc`。 |
| 同上 | `grad_norm`、`updates` | 裁剪前梯度范数；最近一个窗口的优化器更新次数，当前通常为 4。 |
| `ta_F_{task}/*`、`pnc_F_{task}/*` | `fisher_sum` | **本次** Fisher 对角估计在全部受保护参数上的和，不是历史在线 Fisher 总量，也不是成功率。 |
| 同上 | `scored_samples` | 本次参与 Fisher 评分的 transition 数；完整配置为 1024。 |

## 固定面板评估

评估键名中的 `{policy}` 为 `kb`、`active_maze_medium`、`active_fourrooms`，基线还可能是 `single`。Active 名称表示**它在何任务的 P 阶段产生**；`{evaluated_task}` 才表示本次评估任务。例如 `eval_pnc_validation_active_fourrooms_maze_medium/success_rate` 是 FourRooms Active 在 Maze 上的成功率。

| 指标 | 含义与单位 | 注意事项 |
|---|---|---|
| `episodes`、`successes`、`failures` | 固定面板 episode 总数、成功数和失败数。 | 正式配置每次评估每任务 200 局。 |
| `success_rate` | `successes / episodes`，范围 `[0,1]`。 | 判断策略完成任务的主指标。 |
| `mean_return` | 所有 episode 的环境奖励总和的平均值。 | 是评估回报，不是 PPO 的训练 Loss。 |
| `mean_length_all` | 所有 episode 的平均环境步数。 | 同时包含成功和失败局。 |
| `mean_length_success` | 仅成功 episode 的平均环境步数。 | 没有成功局时为 `None`，不会写成 W&B 数值点。 |
| `mean_success_path_ratio` | 成功 Maze 局的 `实际步数 / 最短路径步数` 的平均值。 | 越接近 1 路径越短；FourRooms 或没有成功局时不写数值点。 |

每个数值指标还会有 `visit_end_{metric}` 或 `final_test_{metric}` 形式的同值副本，用来区分评估事件。`eval_{family}_{split}_{policy}_performance/*` 则记录评估成本：

| 指标 | 含义 |
|---|---|
| `elapsed_seconds` | 本次完整评估的墙钟耗时，单位秒。 |
| `episodes_per_second` | 本次评估的总 episode 数除以上述耗时。 |
| `transitions` | 本次评估实际执行的环境步数；计入 `progress/eval_steps`，不计入训练预算。 |

`progress_{split}_{policy}_{evaluated_task}/success_rate` 复用相同成功率，但横轴换成累计 P 环境步，方便比较下游学习。TA 每个 visit 末评估 KB；P&C 每个 visit 末分别评估 KB、Maze Active、FourRooms Active，每个策略都评估两个任务。训练全部结束后，仅最终 KB（单列基线则为最终单列策略）进入独立 test 面板。历史 Active 不做最终 test。

## 阶段累计进度与元数据

`progress/*` 仅在阶段提交时更新，以下指标均为**自本次 run 开始的累计值**：

| 指标 | 含义 |
|---|---|
| `global_env_steps` | 所有训练阶段的环境步总数；不包括评估。 |
| `world_collect_steps`、`explore_steps`、`progress_steps` | 分别为 W.collect、X、P 的环境步数。 |
| `compress_steps`、`fisher_steps` | 分别为 C、F 的环境步数。 |
| `eval_steps` | 评估环境步数，独立于训练预算。 |
| `world_optimizer_updates`、`ppo_optimizer_updates`、`distill_optimizer_updates` | 分别为 W、PPO、蒸馏的累计优化器更新数。 |
| `policy_internal_ticks`、`eval_internal_ticks` | 训练／评估期间控制器内部 synapse 前向的 batch 加权调用数；包含同一步可能执行的多次模型计算，不等于环境步。 |

W&B 历史行还含 `stage`、`event_type`、`global_env_steps`、`world_optimizer_updates`、`downstream_progress_steps` 等通用字段；评估行另有 `evaluation_event`、`evaluation_id`、`evaluation_visit`、`evaluation_policy`。这些字段可用于筛选，不应误当独立的训练 Loss。代码仍支持手工视觉漂移事件的 `drift_{task}/mean_kl`，但当前自动训练流程**不触发**该面板。TA 与 P&C 的遗忘矩阵写入 `evaluation/forgetting.json`，没有单独映射成 W&B 标量。

实现依据：[W&B 键名与横轴](../wandb_logging.py)、[阶段调度和事件](../train.py)、[世界模型损失](../learning/world.py)、[PPO 损失](../learning/ppo.py)、[蒸馏损失](../learning/distill.py)、[评估指标](metrics.py)。
