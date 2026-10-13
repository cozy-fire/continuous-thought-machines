# 当前 W&B 指标筛选清单

> 2026-10-05：P/C/F训练期间不执行 validation/test，默认命令在全部训练完成后独立评估；以下白名单中的评估行仅解释历史日志，当前训练不会产生这些曲线。各阶段通过保存的 P/C/F 权重在本地独立评估，输出 JSON。Maze P 只记录整体/逐步 KL 等训练指标，不计算 `agreement`；FourRooms P 和 C 保留训练一致率。

## 已执行的最终筛选（2026-10-01）

本节为当前 W&B 上传契约；后文原始清单仅作为字段解释，已被本节白名单覆盖。其他字段仍记录在本地 JSONL，不再上传 W&B。

| 范围 | 保留的指标 | 口径 |
|---|---|---|
| P/C 学习 | `kl`、`total_loss`、`agreement`、`student_entropy`、`grad_norm` | 保持阶段累计、有效样本加权均值 |
| C 正则 | `ewc` | 含系数的 EWC，P 不上传 |
| P/C 行为 | `action_fractions/0`…`/4` | 两次训练日志之间的动作次数 / 该区间全部 transition 数 |
| P/C 行为 | `episode_mean_return` | 该区间结束的完整 episode 的 return 均值，含成功和超时 |
| P/C 行为 | `episode_success_rate` | 该区间成功结束 episode 数 / 该区间全部结束 episode 数 |
| P/C/F 进度 | `global_env_steps`、`stage_env_steps` | 保持原训练预算口径；F 只上传这两项 |
| validation / final test | 本地独立评估 JSON，保留成功率、return、length、每集轨迹摘要和阶段权重 SHA | 训练不执行、不上传；历史曲线不删除 |

区间是配置 `logging.interval_windows` 对应的连续 rollout 区间，阶段尾不足间隔也发布。完整 episode 归入其结束时所在区间；其 return 可包含此前区间的奖励，未结束 episode 不参与均值。没有 episode 结束时，两条 episode 指标为本地 null，W&B 不上传该点，不能理解为成功率 0。每阶段环境 reset 并重新建立统计，不跨阶段保留未结束轨迹。C 的行为属于冻结双列教师。

没有修改命名空间和默认横轴：W&B Step 仍是实际日志调用次数；由于无关事件不再上传，其数值不再能与旧版本 Step 直接对齐。可在图表中选择同命名空间 `global_env_steps` 作为横轴。没有删除历史 run 中已存在的旧曲线。

---

依据 2026-10-01 本地 `wandb_logging.py`、`runner.py`、`learning/` 和 `evaluate.py`。本清单供用户选择，尚未删除或改变任何日志。详细口径见同目录 `WandB_指标速查表.md`。

## 阅读规则

- 路径为 `family/phase/task/policy_type/active_source/evaluation_task/split/指标名`，缺失身份字段显示 `none`，显式 Python None 显示 `None`。
- 默认横轴 Step 是 `run.log()` 次数，不是环境步。visit 不在路径中，同任务同阶段跨 visit 共用曲线。
- P/C 学习指标为当前阶段按有效样本加权的累计均值；行为与耗时也从阶段起点累计，不代表最近窗口。
- P 行为属于 Active；C 行为属于冻结双列教师。C 后 validation 才衡量 KB 表现。
- 表中的建议仅供筛选，尚未执行。“保留本地”表示建议保留事件审计数据，不必生成 W&B 图。

## P/C 学习

| 指标 | 含义 | 当前出现范围 | 建议 |
|---|---|---|---|
| `kl` | 教师到学生的 KL，nat | P/C | 保留 |
| `total_loss` | P 为 KL；C 为 KL+EWC | P/C | 保留；P 与 kl 重复 |
| `agreement` | 教师与学生 argmax 动作一致率 | P/C | 保留 |
| `student_entropy` | 学生动作分布熵 | P/C | 保留 |
| `expert_entropy` | 教师动作分布熵；Maze P 固定为 0 | P/C | 可选 |
| `ewc` | 含系数的完整 KB EWC 惩罚 | P/C，P 为 0 | 仅 C 保留 |
| `ewc_encoder` | EWC 的视觉部分 | P/C | 可选，诊断视觉遗忘 |
| `ewc_controller_actor` | EWC 的 CTM+Actor 部分 | P/C | 可选 |
| `grad_norm` | 全部训练参数裁剪前梯度范数 | P/C | 保留 |
| `encoder_grad_norm` | Encoder 裁剪前梯度范数 | P/C | 可选 |
| `controller_grad_norm` | CTM 裁剪前梯度范数 | P/C | 可选 |
| `actor_grad_norm` | Actor 裁剪前梯度范数 | P/C | 可选 |
| `adapter_grad_norm` | Adapter 裁剪前梯度范数；C 为 0 | P/C | 可选，仅 P 有意义 |
| `controller_actor_grad_norm` | CTM 与 Actor 合并范数 | P/C | 建议隐藏，与分项重复 |

## 训练行为

| 指标 | 含义 | 当前出现范围 | 建议 |
|---|---|---|---|
| `action_counts/0`…`/4` | 五动作累计次数，五条曲线 | P/C | 改为区间动作比例更直观 |
| `reward_sum` | 累计环境奖励；不参与训练 Loss | P/C | 改为区间 episode mean return |
| `successes` | 累计成功 episode 数 | P/C | 改为区间 episode success rate |
| `timeouts` | 累计超时 episode 数 | P/C | 与成功数结合计算区间成功率 |
| `displacement_rate` | FourRooms 真正位移步数/总步数 | P/C FourRooms | 保留，建议区间口径 |
| `turn_rate` | FourRooms 真正转向步数/总步数 | P/C FourRooms | 保留，建议区间口径 |
| `mean_distance_to_goal` | 动作前 BFS 距离均值，含 reset 后新起点 | P Maze | 可选，不能单独证明趋近目标 |

区间统计是建议新增的替代口径，当前代码没有这些新区间曲线。区间 episode return 应按 episode 完成归属并保存跨窗口的未结束 episode 状态，不能简单用区间 reward_sum 除以区间成功数。

## 预算与进度

| 指标 | 含义 | 当前出现范围 | 建议 |
|---|---|---|---|
| `global_env_steps` | 全 run 训练环境步，评估另计 | 训练/评估/边界 | 保留并作为横轴候选 |
| `stage_env_steps` | 当前阶段环境步 | P/C/F | 保留 |
| `transitions` | 当前阶段 transition 数，与 stage_env_steps 重复 | P/C | 保留本地 |
| `windows` | 累计 rollout 窗口数 | P/C | 保留本地 |
| `eligible_target_steps` | 有效教学样本数，当前等于 transitions | P/C | 保留本地，做完整性检查 |
| `optimizer_updates` | training 事件为阶段更新数；stage_complete 为全局数 | P/C/阶段完成 | 保留本地或统一命名后保留 |
| `global_optimizer_updates` | run 全局 Adam 更新数 | P/C | 可选 |
| `empty_minibatches` | 无目标而跳过的 minibatch 数，当前应为 0 | P | 保留本地，异常时告警 |
| `next_transition_id` | 下一条全局 transition ID | P | 保留本地 |
| `visit` | visit 编号，不是质量指标 | 阶段事件 | 元数据，建议放入路径 |
| `next_index` | 已提交边界后的阶段索引 | 阶段完成/finalized | 保留本地 |
| `next_uncommitted_index` | 中断后待重做阶段索引 | run_interrupted | 保留本地 |
| `scored_samples` | Fisher 唯一评分样本数，不是 optimizer 更新数 | F | 保留本地 |

## 性能

| 指标 | 含义 | 当前出现范围 | 建议 |
|---|---|---|---|
| `transitions_per_second` | 阶段累计环境步/墙钟秒 | P | 保留，建议 C 也补齐及增加区间值 |
| `elapsed_seconds` | 阶段累计墙钟时间 | P/C/F | 可选 |
| `student_forward_seconds` | P 采集推理；C 独立 KB 窗口重放 | P/C | 性能诊断时保留 |
| `expert_forward_seconds` | P 专一教师查询；C 双列教师推理 | P/C | 性能诊断时保留 |
| `env_step_seconds` | 同步各槽环境执行耗时 | P/C | 性能诊断时保留 |
| `learner_seconds` | 完整窗口更新调用耗时 | P | 可选，与学习分项有重叠 |
| `learner_forward_seconds` | 有梯度重放与 Loss 计算 | P/C | 性能诊断时保留 |
| `learner_backward_seconds` | 反传耗时 | P/C | 性能诊断时保留 |
| `learner_optimizer_seconds` | 梯度裁剪与 Adam 更新耗时 | P/C | 性能诊断时保留 |
| `scoring_seconds` | Fisher 逐样本梯度评分耗时 | F | 可选 |
| `peak_allocated_bytes` | 本次 runner 调用累计 CUDA allocated 峰值，含评估 | memory | 保留；建议显示 GiB |
| `maps` | 内存地图缓存条数 | map_cache | 保留本地 |
| `cache_bytes` | 内存地图缓存字节数 | map_cache | 保留本地 |
| `load_seconds` | 内存地图预加载耗时 | map_cache | 保留本地 |

分项不是互不重叠的纯 kernel 时间，不能全部相加当总耗时。编译捕获/CPU 提交空隙可能进入 GPU event 区间。

## validation / final test

| 指标 | 含义 | 建议 |
|---|---|---|
| `success_rate` | 固定面板成功率 | 必须保留 |
| `mean_return` | 全部 episode 平均环境 return | 保留 |
| `mean_length` | 全部 episode 平均长度 | 保留，与成功率一起看 |
| `episodes` | 本次面板 episode 数 | 保留本地，完整性检查 |
| `environment_steps` | 当前任务评估实际环境步 | 保留本地，不计训练预算 |
| `action_counts/0`…`/4` | 贪心评估动作累计次数 | 改为评估动作比例更直观 |
| `displacement_rate` | FourRooms 面板真实位移率 | 保留 |
| `turn_rate` | FourRooms 面板真实转向率 | 保留 |
| `evaluation_seconds` | 本次整个双任务评估总耗时，在两任务路径重复写入 | 可选，建议每次评估只记录一次 |

## 自动上传的元数据

| 字段 | 上传方式 | 建议 |
|---|---|---|
| `time_unix`、`seed`、`schema_version` | 数字，生成标量图 | 保留本地或 run config，不生成指标图 |
| `method`、`sequence_protocol`、`timing_mode`、`event` | 字符串记录 | 保留本地或元数据 |
| `family`、`phase`、`task`、`policy_type` | 字符串记录，同时参与路径 | 保留身份，避免重复作为指标 |
| `stage`、`attempt_id`、`evaluation_id` | 字符串记录 | 保留本地，追踪恢复与评估 |
| `active_source`、`evaluation_task`、`split` | 非空字符串记录，同时参与路径 | 保留评估身份 |
| `note` | 中断说明字符串 | 保留本地 |

`statistics`、`selected_ids`、逐 episode `results`、布尔值和 None 不展开上传。评估报告顶层 `episodes_per_second`、`backend`、`num_envs` 也未自动上传。修改图表白名单不等于删除本地审计信息；后续需用户确认具体筛选范围。
