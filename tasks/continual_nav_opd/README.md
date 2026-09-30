# CTM P&C 在策略动作蒸馏

方法名为 `ctm_pnc_opd`，checkpoint schema 为 v3，序列协议为 `rollout_state_v1`。本包与旧 v2 Runner 独立，没有 TA、PPO、Critic、世界模型或内在奖励。两个任务分别由 Maze 最短路径算法教师和 FourRooms 神经教师提供教学目标，学生使用统一 RGB 与五动作接口。

## 01：配置与阶段表

从仓库根目录，在 `ctm` Python 环境运行：

```powershell
python -m unittest discover -s tests/continual_nav_opd -p test_config_schedule.py
python -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/full.yaml --dry-run
```

`full.yaml` 定义正式预算，RTX5090 配置只覆盖执行设置。两者均展开为 12 阶段、50,976,384 环境步；smoke 为 12 阶段、1,664 环境步。未知字段、身份不兼容或预算无法整除环境槽数时明确失败。`dry-run` 不加载数据或教师，也不创建训练 run 或连接 W&B。实际训练必须提供 `--seed` 和 `--run-dir`。

唯一序列协议是 `rollout_state_v1`：在本窗口第一张新观察处理前，保存学习策略当前状态的 `detach().clone()`，随后重放本窗口全部新观察。采集状态跨窗口保留，仅在真正的 episode 边界按槽 reset。没有教师预热、历史观察 burn-in 或首步标签屏蔽。

## 02：环境与专一教师

接口包括 `envs.MazeEnv`、`envs.FourRoomsEnv`、`SyncVectorEnv`、`teachers.MazeTeacher` 和 `FourRoomsTeacher`。**两任务均从第一个动作前观察起教学，reset 后首步也立即教学。** Maze 在 CPU 从实际当前位置即时执行 BFS，不缓存跨查询路径；FourRooms 严格加载指定原生七动作权重，每槽独立维持可学习初态展开后的连续状态。其原生动作 3–6 的概率总和平均分给两个学生等待动作。

```powershell
python -m unittest discover -s tests/continual_nav_opd -p test_teachers_envs.py
python -m tasks.continual_nav_opd.inspect_teachers --config tasks/continual_nav_opd/configs/full.yaml --device cuda:0 --output-dir scientific-evidence/continual_nav_opd/new_teacher_audit
```

配置中的数据和 checkpoint 相对路径均相对于仓库根目录解析。数据、教师权重和验证产物不进入 Git。检查工具拒绝覆盖已有输出目录；完整评估固定面板，记录每集结果、耗时、原生调用一致性和教师权重不变检查，全部完成后才发布 `report.json`。`sha256.json` 校验生成文件，不包含它自身。随机五动作代理只验证接口，不代表学生表现。02 的环境 step 同步执行，生产 serial/subprocess 评估器由 06 实现。

`ObservationPair` 将学生 RGB 与教师观察分字段保存。`EnvStep.transition_next` 是真实动作后／终止帧，`next_obs` 是结束槽 reset 后的下一次输入，`next_episode_start` 标记需要 reset 的槽。下一次教师调用使用 `next_obs`；不得把 Maze 终止帧当成新的可行动状态查询。

教师接口：

- `MazeTeacher.predict(observations, contexts)`：输入 NumPy `uint8 [B,3,19,19]` 和每槽地图哈希／episode ID。返回 NumPy `action: int64[B]`、`probabilities: float32[B,5]`、`valid_target: bool[B]`、`distance_to_goal: int64[B]`。`save_descriptor(path)` 以不覆盖模式写入规范 JSON；`source_snapshot_id` 为描述文件的 SHA-256。
- `FourRoomsTeacher(checkpoint, device)`：加载指定可信原生 checkpoint。`initial_state(B)` 返回独立的 `CTMState`；`predict(observations, state, episode_start)` 接收 NumPy `uint8 [B,7,7,3]` 原生符号观察及 NumPy `bool[B]` 标志，返回已 detach 的设备 Tensor：`native_probabilities: [B,7]`、`probabilities: [B,5]`、`valid_target: bool[B]` 和新 `CTMState`。下一次观察继续使用新 state，窗口切换不得 reset。`source_snapshot_id` 是完整权重 SHA-256，`metadata` 保存原始 args 及严格加载协议。
- `SyncVectorEnv.reset()`：返回堆叠双观察和每槽 info。`step(actions)` 返回 `EnvStep`；结束槽的 info 保留终止身份并增加 `reset_info`。重复使用同一个环境实例或非法动作批量会失败。环境元信息和教师专用观察不能进入学生 RGB 输入。

## 03：独立视觉双列模型

```python
from tasks.continual_nav_opd.models import (
    StandalonePolicy, DualPolicy, detach_clone_state, frozen_copy,
    save_snapshot, load_snapshot,
)

kb = StandalonePolicy(config).to(device)  # Trainable encoder/controller/actor, no critic.
dual = DualPolicy(config, kb, kb_ready=False)  # First P; live KB remains independent and trainable.
state = dual.initial_state(batch_size)
initial_state = detach_clone_state(state)  # Save BEFORE the rollout's first new image.
logits, state = dual.step(rgb_u8, state, episode_start)
output = dual.sequence(rollout_rgb_u8, initial_state, rollout_episode_start, valid_mask)
```

图像、state 和 mask Tensor 必须位于策略设备。`step` 输入 `uint8 [B,3,84,84]`；`sequence` 只接收当前窗口的 `[L,B,3,84,84]` 新观察，`1 <= L <= 50`，返回 `PolicySequenceOutput(logits=[L,B,5], state=...)`。重放使用传入起点，不重新初始化或隐式 detach。仅有效观察进入视觉 microbatch，随后按原时间顺序推进 CTM，每张观察 2 ticks。padding 输出零 logits，不改变任何列的 state。`select_state` 和 `detach_clone_state` 为 minibatch 与窗口起点建立独立存储。

每列独立拥有 ResNet34-2／GroupNorm32，以及可学习 `start_pre`／`start_post` `[512,40]` trace，分别对应原 CTM 的 `start_trace`／`start_activated_trace` 角色。学习列在有效 reset 位置恢复初态时保留初态参数梯度；从第一张有效图起即可训练 Encoder，没有 TA 冻结开关或历史图像前缀。完整 KB 参数前缀固定为 `encoder/controller/actor`，Dual 为 `kb/active/adapter`。

`DualPolicy(config, live_kb, kb_ready=...)` 创建完整旧 KB 的独立冻结副本，仅将其 Encoder 权重复制到新 Active；Active Controller／Actor 和零 gate Adapter 新初始化。传入的 live KB 模式及梯度不受影响。每 tick 先用旧 KB 自身特征推进，再将该 tick 的新激活 detach 后输入 Adapter，作用于 Active 同 tick 的 synapse 路径。`kb_ready=false` 关闭侧向连接；gate 初值 0 时，gate 可获得梯度，projection／norm 首步梯度可以为 0。`train()` 不解除旧 KB 冻结。每个新 P 由调用方新建 Dual 和优化器。

`frozen_copy(policy)` 独立复制全部参数和 buffer、清除梯度，并在父模块调用 `train()` 后保持 eval。`save_snapshot(policy, new_path)` 导出完整 v3 推理策略、两套 Encoder、P 当时的旧 KB、Adapter、`kb_ready` 和配置身份。`load_snapshot(path, device)` 校验身份／配置、严格加载所有键，返回存储独立的冻结推理模型，不替换为较新的 KB。这是推理导出，**不是阶段 resume checkpoint**；优化器／RNG／源码／引用完整性及阶段原子提交由 06 实现。C 学生是另一份可训练 `StandalonePolicy`，不能直接把冻结教师切成 train 充当学生。

```powershell
python -m unittest discover -s tests/continual_nav_opd -p test_models.py
```

## 04：Progress 学生轨迹蒸馏

`ProgressCollector(envs, student, expert, action_rng)` reset 环境并建立学生双列 state；FourRooms 教师维护另一份独立 state。`collect(length, start_transition_id)` 只保存本窗口新观察及起点，返回 `CollectedWindow`：其中 `batch` 是 `SequenceBatch`，其余字段为诊断用的采集 logits、reward、终止信息、耗时和 Maze 距离，不进入 Loss。`length` 的单位是每槽新环境步，上限 50。

每次学生先读取 RGB，再由教师读取同一动作前状态；动作始终从学生五维概率采样。Maze 标签为 BFS 第一条边的 one-hot，FourRooms 为映射后的五维软目标。两任务全部真实观察均有目标，终止后改用新 episode 的 reset 观察。缺失目标、非法概率或非有限 Loss／梯度明确失败。

`run_progress(envs, student, expert, config, steps, action_rng, minibatch_rng, start_transition_id=0, on_window=None)` 的 `steps` 是所有槽合计环境步，必须为正数且整除槽数。调用方传入按 03 新建的可训练 `DualPolicy`。`action_rng` 为独立 `torch.Generator`；采样概率移动到该 Generator 的设备。`minibatch_rng` 为独立 `numpy.random.Generator`，仅打乱环境槽，不打乱序列时间。

P 内新建统一 Adam，参数恰为 Active Encoder／Controller／Actor 和 Adapter，学习率 `1e-4`、betas `(0.9,0.999)`、eps `1e-5`、weight decay 0，统一梯度裁剪 0.5。正式每窗口每槽最多 50 步，1 epoch、4 个环境 minibatch；尾窗口缩短以精确达到预算。smoke 使用 2 槽、1 minibatch，每任务 256 步，窗口合计步数为 100／100／56。

`update_window` 从每组独立克隆的 `initial_state` 重放全部新观察，只按 `loss_mask` 计算 `KL(expert || student)`。零教师概率贡献 0，不产生 NaN；无熵项、任务 reward、价值或 PPO 项。空 padding 组不执行 optimizer step，任一真实观察缺少标签则报错。更新后 collector 保留并 detach 采集结束 state，不用学习器重放结束 state 覆盖，也不按新参数重建历史。

返回 `ProgressResult` 包含 `transitions`、`optimizer_updates`、`eligible_target_steps`、`next_transition_id`、`windows`、`statistics` 和完整冻结 P 结束 `policy`，可供 C 使用。成功 P 必须满足 `eligible_target_steps == transitions == steps`。环境生命周期由调用方管理，异常时调用方也需关闭环境。

`on_window` 每 `logging.interval_windows` 个窗口及阶段末接收累计统计：有效样本加权 KL／动作一致率／教师与学生熵、裁剪前梯度范数及分项、动作计数、教学步／环境步／更新数、真实 reward 和结束计数。耗时分别记录环境 step、教师查询、学生采集 forward、学习 forward（视觉编码＋时序 CTM）、backward 和裁剪／optimizer；CUDA 边界同步计时。`learner_seconds` 包含整体学习耗时，与各分项有包含关系，不可相加当总耗时。Runner 将同一统计写入本地事件与 W&B；本地验证关闭 W&B。

```powershell
python -m unittest discover -s tests/continual_nav_opd -p test_progress.py
python -m tasks.continual_nav_opd.verify_stage --stage 04 --device cuda:0 --output-dir scientific-evidence/continual_nav_opd/new_progress_audit
python -m unittest discover -s tests/continual_nav_opd -p 'test_*.py'
```

`verify_stage --stage 04` 只运行真实两任务 P 的 smoke 预算，各 256 步、256 有效教学步、3 窗口、3 次更新，不执行 C/F 或正式训练。它保存每任务完整 `active.pt`、`events.jsonl`、实际累计 KL 曲线 `kl.png`、计数及冻结检查，重新加载并比较输出，最后保存报告与大小／SHA 清单。输出目录必须不存在。两个任务分别从随机 KB 开始验证，不能据此声称完成知识迁移；短 smoke 的 KL 或成功数也不证明正式学习效果。

## 05：完整 KB 压缩与 Fisher

`run_compress_stage(envs, dual_teacher, kb, fisher, config, steps, action_rng, minibatch_rng, start_transition_id=0, on_window=None)` 使用独立冻结完整 Dual 教师采样动作，同时推进教师与 KB 学生各自的状态。窗口保存 KB 自己的起点；KB 从当前参数开始，不复制 Active。学习所有 `encoder/controller/actor` 参数，目标为教师概率到 KB 的 KL，加完整视觉 EWC。首个 C 没有历史 Fisher，EWC 为 0；后续 EWC 分别记录视觉与 Controller＋Actor 的贡献。

`run_fisher_stage` 在当前 KB 自身轨迹上精确采集配置预算，用独立 NumPy RNG 无放回选评分 ID。每个样本从其窗口起点重放全部新观察，对采样动作的 log probability 求梯度，先平方再平均。所有 KB 参数必须入字典，包括未使用梯度对应的 0；不平方 batch 平均梯度。F 没有 optimizer step，参数不变，完成后清除梯度。Online Fisher 为 `0.3 * previous + current`，中心替换为本次 KB 的独立 CPU 拷贝。

```powershell
python -m unittest discover -s tests/continual_nav_opd -p test_compress_fisher.py
python -m tasks.continual_nav_opd.verify_stage --stage 05 --device cuda:0 --output-dir scientific-evidence/continual_nav_opd/new_compress_audit
```

05 验证使用实际教师，在两任务上顺序执行 P→C→F，第二 C 使用第一 F 的完整视觉约束；同时检查各任务 P Active 和 C KB 在两任务固定小面板上的表现。测试成功仅说明工程正确。

## 06：训练、评估、恢复与可视化

完整流程只有 P→C→F，没有 TA 输入。每个 P 创建新 Active 和 Adam；每个 C 创建新 Adam；跨阶段保留 KB、Fisher、参数中心、预算和独立 RNG。正式配置为 2 visits、12 阶段；smoke 每个 P 为 256 步／3 更新，C 为 128 步／2 更新，F 为 32 步／8 个评分样本，总 1,664 步／20 次优化器更新。

```powershell
# Small verification budget; output directory must not exist.
python -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/smoke.yaml --device cuda:0 --seed 0 --run-dir scientific-evidence/continual_nav_opd/new_smoke --wandb-mode disabled

# Formal command, to execute only when explicitly authorized.
python -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/rtx5090_32gb.yaml --seed 0 --run-dir runs/continual_nav_opd/new_seed0 --wandb-mode online

# Exact same config, seed and device override as the original run.
python -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/smoke.yaml --device cuda:0 --seed 0 --run-dir scientific-evidence/continual_nav_opd/new_smoke --wandb-mode disabled --resume
```

`--max-stages N` 只在完整阶段提交后停止，适合检查边界恢复。`--device` 是显式配置覆盖，进入配置哈希；恢复必须保持一致。各训练槽同步 step；评估支持 `serial` 或显式 `spawn` 的 `subprocess`，仅环境进入 worker，模型和完整双列状态始终在主进程。

P 结束评估完整 Active，C 结束评估 KB，两者均覆盖两任务；评估不改训练模式、参数或 RNG。F 不额外评估。visit 报告引用已完成结果；KB 遗忘记录不混入 Active。最终 F 完成后执行 KB 独立 test，再导出 `exports/final.pt` 并标记 `finalized`。评估步独立统计，不加入训练预算。

每阶段必须完成训练和必要评估，随后原子写 `.pt`、`.complete.json`，最后更新 `checkpoints/latest.json`。恢复检查完整配置、源码、算法教师、复制到 run 内的 FourRooms 权重、地图 manifest、引用文件的 SHA／大小、预算／更新计数和时序协议。P 的完整快照包含当时旧 KB、两套 Encoder、Actor 和 Adapter；恢复不改接新 KB。缺失或损坏的文件、旧 schema、旧序列协议都拒绝。

窗口与 Adam 状态不做阶段中途恢复。未提交阶段从上一合法边界完整重做；残留文件用独立 attempt 标识隔离，未提交评估不进入汇总。最后 F 已提交但 test 中断时，只重做 finalization；已 finalized 的 resume 不重复 test。`events.jsonl` 可含被中断 attempt，确认已提交进度以 `latest.json` 和 complete marker 为准。

本地事件与 W&B 使用相同加权口径，W&B 名空间区分阶段、任务、策略、Active 来源、评估任务和 split。记录 KL、一致率、熵、EWC 分项、梯度、动作、真实 reward、FourRooms 位移／转向率、采集／学习／Fisher／评估耗时和显存。`disabled` 不导入 W&B、不要求登录；`online` 使用已登录环境，不在文件中保存凭据。

```powershell
python -m tasks.continual_nav_opd.evaluate --checkpoint scientific-evidence/continual_nav_opd/new_smoke/checkpoints/latest.json --policy kb --split validation --tasks maze_medium fourrooms --backend serial --num-envs 2 --device cuda:0 --output scientific-evidence/continual_nav_opd/new_eval.json
python -m tasks.continual_nav_opd.evaluate --checkpoint scientific-evidence/continual_nav_opd/new_smoke/checkpoints/latest.json --policy active --active-key pnc/v0/maze_medium/P --split validation --device cuda:0 --output scientific-evidence/continual_nav_opd/new_active_eval.json
python -m tasks.continual_nav_opd.visualize --checkpoint scientific-evidence/continual_nav_opd/new_smoke/exports/final.pt --policy kb --task fourrooms --panel-index 0 --device cuda:0 --output-dir scientific-evidence/continual_nav_opd/new_visualization
python -m tasks.continual_nav_opd.visualize --checkpoint scientific-evidence/continual_nav_opd/new_smoke/checkpoints/latest.json --policy active --active-key pnc/v0/maze_medium/P --task maze_medium --panel-index 0 --teacher-diagnostics --device cuda:0 --output-dir scientific-evidence/continual_nav_opd/new_teaching_visualization
```

独立评估输出必须不存在。读取推理 `.pt` 时需要其 SHA 侧车、metadata 及 metadata 侧车，以及 metadata 引用的地图 manifest。`--policy active` 拒绝单列权重。可视化的全局地图只用于展示，推理仍使用局部／规定 RGB；输出 `behavior.gif`、`steps.csv`、`reward.png` 和 `trajectory.json`，包含动作、每步 reward、累计 reward、终止信息。GIF 含初始帧，帧数必须为动作数＋1。只有显式 `--teacher-diagnostics` 的 Active 展示查询专一教师；纯 KB 部署不伪造教学目标。

## 07：集成验收与性能交接

`python -m tasks.continual_nav_opd.verify_smoke --device cuda:0 --output-dir <新目录>` 实际运行两任务、两 visits 的12阶段，包含首P边界恢复、最终幂等恢复、完整产物校验、serial/subprocess对照和两任务Active媒体核验。独立单元测试仍需另行执行。

`python -m tasks.continual_nav_opd.profile_training --device cuda:0 --num-envs 2 --microbatch 8 --minibatches 1 --output-dir <另一新目录>` 使用真实教师测量采集、学习与逐样本Fisher，记录视觉/CTM前向、视觉反向、优化器和显存。同步诊断存在计时开销；本机结果不能代表RTX5090实测。正式启动、恢复、输入产物和异常边界见 [运行手册](RUNBOOK.md)。
