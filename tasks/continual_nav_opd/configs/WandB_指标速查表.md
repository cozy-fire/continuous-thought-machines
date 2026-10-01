# v3 OPD：W&B 路径与指标速查表

> 2026-10-01 更新：本文后续为原完整事件字段的详细说明，不再表示全部上传 W&B。当前上传白名单和新增区间行为指标以同目录 `WandB_指标筛选清单.md` 的“已执行的最终筛选”为准。本地完整事件字段仍保留。W&B 不再上传元数据图、累计动作次数/奖励/成功数量及性能图；validation/final test 的原评估指标全部保留。

记录日期：2026-09-30。依据提交 `ae0e3e7` 的实际日志实现；后续日志代码改变时应同步修订本表。本表描述字段语义，不将 smoke 结果视为正式训练效果证明。

## 1. 为什么标题和路径中出现 none

`EventLogger.emit()` 固定用七个字段拼接命名空间，再追加指标名：

```text
family/phase/task/policy_type/active_source/evaluation_task/split/metric
```

| 路径段 | 含义 | 当前 P 训练事件示例 |
|---|---|---|
| `family` | 训练流程家族 | `pnc` |
| `phase` | 阶段：P 蒸馏专一教师、C 压缩、F Fisher、final 最终收尾 | `P` |
| `task` | 当前训练阶段所属任务 | `maze_medium` |
| `policy_type` | 所描述的学习策略：Active 或 KB | `active` |
| `active_source` | 被评估 Active 的来源任务；当前训练事件未提供此字段 | 缺失 → `none` |
| `evaluation_task` | 当前评估的任务；训练事件不提供 | 缺失 → `none` |
| `split` | 固定面板类型：validation/test；训练事件不提供 | 缺失 → `none` |

因此，截图中的 `pnc/P/maze_medium/active/none/none/none/total_loss` 是 P 训练 Loss 的正常日志键。三个 `none` 是路径占位符，不表示没有教师、没有 Active、没有视觉输入，也不是 Loss 为 NaN。

具体实现使用 `str(event.get(k, 'none'))`：**字段缺失得到小写 `none`，字段显式为 Python `None` 得到大写 `None`。** 日志中的数值 `None` 则不会上传为标量曲线。这三种情况要区分。

| 事件 | 当前命名空间示例 | 说明 |
|---|---|---|
| Maze P 训练 | `pnc/P/maze_medium/active/none/none/none` | 三个评估相关字段未提供 |
| FourRooms C 训练 | `pnc/C/fourrooms/kb/none/none/none` | `kb` 是学习对象；环境动作仍由 C 教师采样 |
| Maze Active 在 FourRooms 上 validation | `pnc/P/maze_medium/active/maze_medium/fourrooms/validation` | 来源任务与评估任务均完整 |
| Maze C 后 KB 在 FourRooms 上 validation | `pnc/C/maze_medium/kb/None/fourrooms/validation` | `active_source` 显式为 `None`，因为评估 KB |
| 最终 KB 的 FourRooms test | `pnc/final/none/kb/none/fourrooms/test` | 最终事件没有单一来源训练任务 |
| F | `pnc/F/maze_medium/kb/none/none/none` | KB Fisher 估计 |
| 阶段完成或显存事件 | `pnc/P/maze_medium/none/none/none/none` | 这些事件当前未传 `policy_type` |

这属于日志命名与展示问题，不影响训练计算。本次文档留档不改日志代码，不修改正在运行的训练。

## 2. 横轴、visit 与统计范围

| 项目 | 当前口径 | 阅读时的注意事项 |
|---|---|---|
| W&B 默认横轴 `Step` | 每次 `run.log()` 的日志记录步 | 不是环境步、optimizer 更新数或 CTM tick；训练、评估、显存等事件都会占用日志步 |
| `visit` | 从 0 开始的 visit 编号；正式配置为 v0/v1 | 作为普通数值被上传，所以会出现平坦的 `visit` 图；不是训练质量指标 |
| 路径中的 visit | **当前没有 visit 这一段** | 同任务同阶段的 v0/v1 共用日志键；阶段累计值会在下一 visit 归零，看起来可能突然下降 |
| `stage_env_steps` / `transitions` | 当前 P/C 阶段从起点累计的训练环境 transition 数 | 下一阶段重新计数；每槽一次动作各算一条 transition |
| `global_env_steps` | 整个 run 的训练环境步累计 | 跨阶段连续；不包含 validation/test；适合比较整体预算进度 |
| Loss、熵、动作一致率、梯度范数 | 当前 P/C 阶段从起点累计、按有效教学样本数加权的均值 | 不是最新窗口或最后一个 minibatch 的值；早期数据会使曲线变平滑、变化滞后 |
| 动作数、成功数、超时数、秒数 | 当前阶段累计量 | 要观察近期策略，需取相邻记录的差值 |
| 评估指标 | 当前策略、当前任务固定面板的整组结果 | 不与训练阶段的累计行为统计混用 |

当前配置 `learning_steps=50` 指每槽每窗口最多 50 次新观察/环境动作。16 槽时完整窗口产生 800 条 transition；1 epoch、4 个环境 minibatch，对应 4 次 Adam 更新。正式配置每 10 个窗口记录一次训练日志，即通常每 8,000 环境步记录一次；最后不足记录间隔时仍记录阶段末结果。

两任务都从首步教学，没有教师预热、burn-in 或额外历史动作预算。`eligible_target_steps` 应与当前阶段 `transitions` 相等。

对于阶段累计均值 `m` 和对应样本数 `n`，相邻两次记录间的均值可还原为 `(m₂n₂−m₁n₁)/(n₂−n₁)`。动作近期频率使用 `(count₂−count₁)/(transitions₂−transitions₁)`。仅在同一 stage/attempt 内计算，不能跨 visit 或阶段重执行边界作差。

## 3. P/C 学习指标

P 的目标分布来自专一教师：Maze 为当前位置 BFS 最短路径首动作的 one-hot，FourRooms 为冻结原生教师映射后的五动作分布。C 的目标来自 P 完成时的完整冻结双列快照，包括当时旧 KB、两套 Encoder 和 Adapter。

| 指标 | 含义与公式/单位 | 解读与边界 |
|---|---|---|
| `kl` | 学习重放时的 `KL(p_teacher || π_student)`，有效观察均值，单位 nat | P 学生为 Active，C 学生为 KB；是每次 minibatch 更新前计算的 Loss，再累计平均。Maze one-hot 下等于正确动作的负对数概率 |
| `total_loss` | P：`kl`；C：`kl + ewc` | 无 PPO、Critic、任务奖励项、熵奖励或视觉特征匹配项；日志中的均值可有少量浮点舍入差异 |
| `agreement` | `argmax(p_teacher) == argmax(π_student)` 的比例，范围 0–1 | 比较学习重放的贪心动作，不是实际采样动作一致率；也不是任务成功率。并列最大概率按 argmax 的首个索引处理 |
| `expert_entropy` | 目标分布熵 `−Σ p log p`，单位 nat | Maze one-hot 正常为 0；C 描述双列教师，不能与 P 的专一教师混为一谈 |
| `student_entropy` | 学生五动作分布熵 `−Σ π log π`，单位 nat | 范围 0–ln(5)≈1.609；较低表示动作分布更集中。集中本身不证明正确或塌缩，要结合一致率和成功率 |
| `ewc` | C 的 Online EWC 惩罚 `250/2 × Σ F_j(θ_j−θ*_j)²` | 已包含系数；覆盖完整 KB Encoder、CTM、Actor。P 恒为 0；首个 C 没有历史 Fisher 时为 0 |
| `ewc_encoder` | 上述惩罚中参数名以 `encoder.` 开头的部分 | 衡量视觉参数偏离历史中心的加权代价，不是视觉蒸馏误差 |
| `ewc_controller_actor` | 上述惩罚中其余 KB 参数部分 | CTM 与 Actor 的保留约束；两项之和等于 `ewc` |
| `grad_norm` | 本次反传全部可训练参数梯度的 L2 范数 | **裁剪前**；P 覆盖 Active Encoder/CTM/Actor/Adapter，C 覆盖完整 KB。图上是 minibatch 范数的有效样本加权均值，不是平均梯度的范数 |
| `encoder_grad_norm` | 学生 Encoder 参数梯度 L2 范数 | P 为 Active Encoder；C 为 KB Encoder；裁剪前，统计方式同上 |
| `controller_grad_norm` | 学生 CTM controller 参数梯度 L2 范数 | 不包括 Encoder、Actor、外部 Adapter |
| `actor_grad_norm` | 学生 Actor 参数梯度 L2 范数 | 输出五动作分布的 Actor；没有 Critic |
| `adapter_grad_norm` | 双列 Adapter 梯度 L2 范数 | C 无 Adapter，记录 0。首任务 P 尚未开启侧向连接时也正常为 0；zero gate 下部分投影首步零梯度是预期行为 |
| `controller_actor_grad_norm` | 每个 minibatch 内 `sqrt(controller_norm² + actor_norm²)`，之后累计平均 | 不包括 Encoder 或 Adapter；不能直接用两个已累计均值代入公式得到完全相同的数值 |

全体可训练参数梯度裁剪上限为 0.5；因此 `grad_norm > 0.5` 是裁剪前的正常观测，不表示裁剪失败。

## 4. 环境行为与训练进度

**P 行为来自当前 Active 的随机采样；C 行为来自冻结双列教师的随机采样。** 因此，C 的动作分布、环境回报、成功数和移动率描述的是教师轨迹，不能证明正在学习的 KB 已掌握该策略。KB 的真实表现要看 C 后的固定面板评估。

| 指标 | 含义 | 范围/注意事项 |
|---|---|---|
| `windows` | 当前阶段累计采集并更新的窗口数 | 每个完整窗口为每槽 50 步；尾窗口可更短 |
| `transitions` | 当前 P/C 阶段累计环境 transition 数 | 通常与 `stage_env_steps` 相同 |
| `stage_env_steps` | 当前阶段已用训练预算 | F 事件给阶段全部采集步数；评估不增加它 |
| `global_env_steps` | 已完成阶段预算加当前阶段已采集步数 | 当前阶段中途日志不代表这些步数已提交可恢复 checkpoint |
| `eligible_target_steps` | 当前 P/C 阶段累计用于 Loss 的真实教学观察数 | 当前无预热方案应等于 `transitions`；padding 不计入 |
| `optimizer_updates` | `training` 事件中为当前 P/C 阶段累计 Adam 更新数 | `stage_complete` 事件中的同名字段是 run 全局累计数，两类事件路径不同；F 不更新优化器 |
| `global_optimizer_updates` | `training` 事件中的 run 全局累计更新数 | 前序阶段更新数加当前阶段更新数；适合核对总训练量 |
| `empty_minibatches` | P 中累计没有有效教学目标而跳过更新的 minibatch 数 | 当前 P 全观察教学应为 0；C 当前不发布此标量 |
| `next_transition_id` | P 日志中下一条 transition 的全局 ID | 与采样唯一标识有关，不是额外训练步；当前 C 训练事件不发布此字段 |
| `action_counts/0` … `/4` | 当前阶段五种实际采样动作的累计次数 | Logger 将 `action_counts` 列表展开为五条 W&B 曲线；总和应等于该阶段 transition 数 |
| `reward_sum` | 当前阶段所有槽的环境奖励总和 | 仅诊断，不进入 KL Loss；不是平均 episode return。Maze 成功才奖励 `1−0.9×步数/300`；FourRooms 使用原生成功回报 |
| `successes` | 当前阶段成功终止 episode 的累计数 | 是数量，不是成功率；尚未结束的 episode 不计入 |
| `timeouts` | 当前阶段达到时间上限且未成功的 episode 累计数 | 两任务当前上限 300 步；恰在末步成功只算成功，不重复算超时 |
| `displacement_rate` | FourRooms 中 agent 位置真正发生变化的 transition 数 / transition 总数 | P/C 为阶段累计比例；包含 forward 被墙阻挡等情况。Maze 当前记录 JSON `null`，不生成 W&B 标量 |
| `turn_rate` | FourRooms 中朝向真正发生变化的 transition 数 / transition 总数 | 旋转并不等于位移；Maze 同样无此标量 |
| `mean_distance_to_goal` | Maze P 中动作前各状态的 BFS 最短路径距离累计平均，单位为网格移动步数 | 随机 reset 带来新的起点距离；不是最终剩余距离。FourRooms 为 `null`；C 未实现该字段 |
| `elapsed_seconds` | 当前 P/C 阶段内采集、更新及日志回调等累计墙钟时间 | 不包括随后固定面板评估；C 的冻结教师创建/初始 hash 等在计时起点前，不包含在此值中 |
| `transitions_per_second` | P 中 `transitions / elapsed_seconds` | 当前 P 阶段累计平均吞吐，不是最新窗口吞吐；C 当前未发布此字段，可用上述两列自行计算 |

| 动作索引 | Maze | FourRooms |
|---|---|---|
| 0 | up | turn_left |
| 1 | down | turn_right |
| 2 | left | forward |
| 3 | right | wait，映射 sentinel 9 |
| 4 | wait | wait，映射 sentinel 9 |

## 5. 耗时与显存

以下 P/C 秒数是当前阶段累计分项；不是每步平均值。比较近期吞吐需取差值。阶段秒数之和不能直接当作整个 run 的预计耗时，还需加入初始化、评估、checkpoint、Fisher 等开销。

| 指标 | 计时对象 | 注意事项 |
|---|---|---|
| `student_forward_seconds` | P：采集时 Active 双列推理；C：在教师完整采集后、更新前重放独立 KB 的整个窗口 | 与有梯度学习前向不同；C 保留独立 KB 初态和原采集 CNN batch 形状 |
| `expert_forward_seconds` | P：Maze BFS 或原生 FourRooms 教师；C：冻结完整双列教师 | Maze BFS 为 CPU 墙钟；其余 GPU 计时受 `timing_mode` 影响 |
| `env_step_seconds` | 同步向量环境执行各槽动作的 CPU 墙钟 | 16 个训练槽仍逐槽 step；subprocess 并行仅用于评估 |
| `learner_seconds` | P：完整 `update_window` 调用的墙钟，包括学习重放、反传、优化及统计开销 | 当前 C 不发布这一总项 |
| `learner_forward_seconds` | 有梯度序列重放和 Loss/EWC 计算 | 学习视觉 microbatch 使用配置值；与采集视觉分块不是同一口径 |
| `learner_backward_seconds` | `loss.backward()` | 包含 Encoder、CTM、Actor，以及 P 中可训练 Adapter 的反传 |
| `learner_optimizer_seconds` | 全体训练参数梯度裁剪与 `optimizer.step()` | 不含此前的梯度范数计算、有限性检查和指标传输 |
| `scoring_seconds` | F 逐样本重放、求 log-policy score 梯度平方的同步墙钟 | F 事件发布；不含前面的环境采集 |
| `elapsed_seconds`（F） | F 整体墙钟，包括采集、评分和 Online Fisher 更新等 | F 没有 P/C 细分的常规耗时曲线 |
| `peak_allocated_bytes` | CUDA 下阶段提交前读取的 PyTorch 已分配显存峰值，单位 byte | 除以 `2^30` 转 GiB；峰值会包含评估快照等开销。计数器在本次 runner 调用开始时重置，**不是每个阶段重新重置**；恢复后重新起算，也不是 nvidia-smi 当前占用 |

默认 `timing_mode=events`：GPU 分项用当前 stream 的 CUDA event 时间区间，可能包含 CPU 提交工作之间的空隙，并非纯 kernel 时间；CPU 环境/BFS 用墙钟。`synchronized` 模式在计时前后同步，开销更大。不同模式分项不能直接作提速对比；端到端墙钟和同负载比较更有意义。

## 6. 固定面板评估指标

P 后评估完整 Active，C 后评估 KB；每个策略分别评估两个任务。正式 validation/test 每任务 200 集，使用固定面板、argmax 动作，不采用训练时的随机采样。最终 test 只评估最终 KB。训练成功数量不能替代这些结果。

| W&B 指标 | 含义 | 统计范围/单位 |
|---|---|---|
| `episodes` | 本次被评估任务的完整 episode 数 | 正式每任务 200；不是训练 episode 数 |
| `success_rate` | 成功 episode 数 / `episodes` | 范围 0–1，固定面板策略表现的主要指标 |
| `mean_return` | 各 episode 环境奖励总和的平均值 | 含失败集；不是成功集条件均值 |
| `mean_length` | 各 episode 实际动作步数的平均值 | 含成功与超时；长度变短需结合成功率理解 |
| `environment_steps` | 本次被评估任务的 episode 长度总和 | 评估步数，**不计入训练预算** |
| `action_counts/0` … `/4` | 本任务全部评估 episode 的动作累计数 | 描述贪心策略，与 P/C 的随机采样动作分布不同 |
| `displacement_rate` | FourRooms 面板的实际位移步数 / 总评估步数 | 全面板按步加权；Maze 为 `null` |
| `turn_rate` | FourRooms 面板的实际转向步数 / 总评估步数 | 全面板按步加权；Maze 为 `null` |
| `evaluation_seconds` | 该策略本次整个双任务评估的总墙钟 | 同一个总时长被写入两个任务的日志命名空间；不是该任务单独的耗时，不能把两条相加 |

评估 JSON 顶层还保存 `episodes_per_second`、`elapsed_seconds`、`environment_steps`、`backend`、`num_envs`，其中环境步数是两个任务总和。这些顶层性能字段**当前没有被 Runner 自动展开上传 W&B**；不要把报告字段当作已存在的在线曲线。

评估 JSON 的 `tasks[task].results` 保存逐 episode：`panel_index`、`return`、`length`、`action_counts`、`moves`、`turns`、`shortest_path`、`success`、`terminated`、`truncated`。`shortest_path` 目前只计算 Maze 初始位置到目标的最短网格移动步数；FourRooms 为 `null`，未计算最短路径。逐 episode 列表当前不作为 W&B 标量上传。

## 7. Fisher、阶段完成与元数据

| 字段 | 所在事件/文件 | 含义与边界 |
|---|---|---|
| `scored_samples` | `fisher` | F 从真实轨迹中选出的唯一 transition 评分数；正式 1,024，smoke 8。逐样本梯度平方后平均，不是优化器更新次数 |
| `selected_ids` | 本地 `stage_complete.statistics` | 被选择的 transition ID 列表；不是 W&B 标量曲线。checkpoint 的 Fisher 状态保存估计结果、中心和计数，不保存这个列表 |
| `next_index` | `stage_complete` / `finalized` | 已提交边界后的下一个阶段索引，12 表示所有训练阶段完成；完整结束还需 `finalized=true` 和最终产物校验 |
| `next_uncommitted_index` | `run_interrupted` | 异常时下一未提交阶段的索引；恢复整段重做该阶段，不续接中途窗口 |
| `visit` | 常规阶段事件 | 当前 visit 编号；不在路径中，因而不能自动隔离两轮曲线 |
| `schema_version` | 所有事件 | 当前为 3；属于协议标识，不是性能指标 |
| `seed` | 所有事件 | 训练随机 seed，当前正式 run 为 0 |
| `time_unix` | 所有事件 | 写入事件时的 Unix 秒时间戳；不是阶段耗时 |
| `method` | 所有事件 | `ctm_pnc_opd` |
| `sequence_protocol` | 所有事件 | `rollout_state_v1`，表示保存 rollout 初态后重放全部新观察的时序契约 |
| `timing_mode` | 所有事件 | `events` 或 `synchronized` |
| `event` | 所有事件 | `training` / `fisher` / `evaluation` / `memory` / `stage_complete` / `final_test` / `finalized` / `run_interrupted` |
| `stage` | 常规阶段事件 | 如 `pnc/v0/maze_medium/P`，完整阶段身份，含 visit |
| `attempt_id` | 常规阶段事件 | 本次阶段执行的唯一标识；中断后重做会改变，避免将两次累计统计混算 |
| `evaluation_id` | validation 事件与评估报告 | 本次评估唯一标识；最终报告也有 `final_test` 标识，但最终 W&B 事件当前未传该字段 |
| `statistics` | 本地 `stage_complete` 事件 | 完整阶段统计字典；Logger 当前不递归展开字典到 W&B |

Logger 当前把所有顶层数字都上传为标量，把字符串也上传为记录；所以 `visit`、`seed`、`schema_version` 等也会自动出现图。这些是元数据，阅读训练性能时可以隐藏。布尔值不作为标量上传；只有 `action_counts` 列表被显式展开，其余列表、字典和 `None` 值不展开。

`events.jsonl` 与 `metrics.jsonl` 当前写入同一份事件内容。W&B 展示的标量来自这份事件；完整报告、逐 episode 结果和 checkpoint 状态需查本地文件。

## 8. 当前展示限制与后续整理方向

1. `none` 来源明确，不必为此重启训练。当前命名同时服务训练/评估，导致训练路径冗长；未来可按事件类型生成不同路径，保留完整身份在事件记录中。
2. 建议阅读时以 `global_env_steps` 为横轴。但代码目前未调用 `define_metric` 设置专用 step metric，默认仍是 W&B `Step`；需在图表设置中选择已记录的同命名空间环境步字段。
3. v0/v1 共用同名指标；若未来要求严格分开曲线，应在命名空间加入 visit 或显式定义独立系列，不能只依赖平坦的 `visit` 元数据图。
4. 训练曲线为累计均值。更及时的诊断可另加窗口/区间指标，但当前不能把累计曲线解释成近期均值。
5. C 的训练行为属于教师，C 后 validation 才是 KB 学习成果；低 KL、低熵、较高动作一致率都不能单独证明任务成功。
6. 正式 run 受源码哈希约束。本轮只添加说明文档；若以后修改日志代码，必须单独处理运行进程及合法恢复边界，不能在运行中直接替换受检源码。

## 9. 源码对应位置

| 逻辑 | 源文件 |
|---|---|
| 路径拼接、标量/列表上传 | `tasks/continual_nav_opd/wandb_logging.py`：`EventLogger.emit` |
| P/C/F 事件、全局计数、评估展开、显存峰值 | `tasks/continual_nav_opd/runner.py`：`run` |
| KL、熵、一致率、梯度范数、样本加权 | `tasks/continual_nav_opd/learning/sequence.py`：`distribution_metrics`、`update_window` |
| P 的阶段累计统计 | `tasks/continual_nav_opd/learning/progress.py`：`run_progress` |
| C 的阶段累计统计 | `tasks/continual_nav_opd/learning/distill.py`：`run_compress_stage` |
| P 轨迹、动作前教师标签、Maze 距离 | `tasks/continual_nav_opd/data/progress.py`：`ProgressCollector.collect` |
| C 教师轨迹、独立 KB 重放 | `tasks/continual_nav_opd/data/compress.py`：`CompressCollector.collect` |
| EWC、逐样本 Fisher 与在线衰减 | `tasks/continual_nav_opd/learning/fisher.py` |
| 固定面板、逐 episode 汇总与评估报告 | `tasks/continual_nav_opd/evaluate.py`：`evaluate_policy` |
| GPU events/CPU 墙钟计时口径 | `tasks/continual_nav_opd/timing.py`：`WindowTimer` |
