# CTM P&C 在策略动作蒸馏

方法名为 `ctm_pnc_opd`，checkpoint schema 为 v4，序列协议为 `maze_onpolicy5_v1`。本包与旧 v2 Runner 独立，没有 TA、PPO、Critic、世界模型或内在奖励。两个任务分别由 Maze 最短路径算法教师和 FourRooms 神经教师提供教学目标，学生使用统一 RGB 与五动作接口。

本次课程实现的实测结果、验证范围及复现命令见 [Maze v4 验收记录](VERIFICATION_MAZE_V4.md)。

## 01：配置与阶段表

从仓库根目录，在 `ctm` Python 环境运行：

```powershell
python -m unittest discover -s tests/continual_nav_opd -p test_config_schedule.py
python -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/full.yaml --dry-run
```

`full.yaml` 与 `remote_config.yaml` 展开为两个 visits、12 个 P/C/F 阶段。Maze 每次 P 在内部依次训练 10/40/160/640/2560/10240/44488 张地图；前三池各 2000 次、后四池各 1500 次 Adam，共 12000 次/visit、24000 次/两 visits。跳过 40960。Maze P 不接受环境步预算；`TaskBudget.progress_steps` 只供 FourRooms 使用。

`maze_progress.pool_sizes/pool_updates` 是课程的配置来源。每次更新为 100 条独立五步序列，microbatch5。正式 P 的最多有效决策数为 6000000/visit，终止 mask 使实际数减少。所有阶段合计最多 26736384 个训练决策：P=21600000、C=5120000、F=16384；`budget_summary.env_steps_are_upper_bounds` 明确标识容量。smoke 显式用 2/4 张地图、各 1 次更新，仍保持 100/5 批次，两 visits 共 18 次 Adam、最多 3152 个决策。

FourRooms 每 visit 的 P 决策预算为 4800000、C 为 1280000；Maze C 同为 1280000。`remote_config.yaml` 与 `local32_8gb.yaml` 使用 32 槽、50 步观察窗口、4 个更新分组，每组 400 个决策：两任务每 visit 的 P 均为 12000 次 Adam、C 均为 3200 次，两 visits 合计 60800 次。Maze P 的每次更新最多 500 个有效决策，因此对齐的是更新次数，决策数并不相同。默认 `full.yaml` 仍用 8 槽，每组 100 个决策；FourRooms P 为 48000 次/visit、两任务 C 各 12800 次/visit，Maze P 仍为 12000 次/visit，合计 171200 次，不宣称与 Maze 对齐。F 每阶段仍为 4096 个决策、1024 个评分样本，不执行 Adam 更新。

Maze ticks=5、M=40 固定；FourRooms ticks=2，采样、50 步窗口、Loss 与优化器保持现状。dry-run 不加载地图或教师、不创建 run；它输出容量、Maze Adam 预算和完整阶段表。新预算产生不同配置哈希，旧预算训练产物不能用于新预算续训；checkpoint schema 仍为 v4，没有兼容迁移。schema v3、旧序列协议和旧 Maze 环境步字段值直接拒绝。

## 02：环境与专一教师

接口包括 `envs.MazeEnv`、`envs.FourRoomsEnv`、`SyncVectorEnv`、`teachers.MazeTeacher` 和 `FourRoomsTeacher`。**两任务均从第一个动作前观察起教学，reset 后首步也立即教学。** Maze C/F 与评估仍在 CPU 从当前位置查询 BFS；Maze P 使用一次反向 BFS 构造的等价状态表；FourRooms 严格加载指定原生七动作权重，每槽独立维持可学习初态展开后的连续状态。其原生动作 3–6 的概率总和平均分给两个学生等待动作。

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
logits, state = dual.step(rgb_u8, state, episode_start, task="maze_medium")
output = dual.sequence(rollout_rgb_u8, initial_state, rollout_episode_start, valid_mask,
                       task="maze_medium")
```

图像、state 和 mask Tensor 必须位于策略设备。`step` 输入 `uint8 [B,3,84,84]`；`sequence` 只接收当前窗口的 `[L,B,3,84,84]` 新观察，`1 <= L <= 50`，返回 `PolicySequenceOutput(logits=[L,B,5], state=...)`。两入口必须显式传入 `task`，一次调用只能处理同一任务；未知任务立即报错。任务仅选择递归次数，不进入 Encoder、Attention、CTM 或 Actor 的输入。重放使用传入起点，不重新初始化或隐式 detach。仅有效观察进入视觉 microbatch，随后按原时间顺序推进 CTM：Maze 每张观察 5 ticks，FourRooms 2 ticks。padding 输出零 logits，不改变任何列的 state。`select_state` 和 `detach_clone_state` 为 minibatch 与窗口起点建立独立存储。

KB 在两个任务间共享全部权重和相同参数形状。P 双列、C 双列教师与 KB 学生、F、validation/test 和可视化均按当前环境任务选择同一预算；跨任务评估按评估任务选择，不能沿用 Active 来源任务的 ticks。切换任务创建独立 episode 状态。40 tick 窗口在 FourRooms 约覆盖 20 次观察，在 Maze 约覆盖 8 次观察；更早的信息只能经递归间接延续。Maze P 的 5 个决策展开 25 tick/列；C/F 的原 50 观察窗口展开 250 tick/列，记忆窗口不截断计算图。

每列独立拥有 ResNet34-2／GroupNorm32，以及可学习 `start_pre`／`start_post` `[512,40]` trace，分别对应原 CTM 的 `start_trace`／`start_activated_trace` 角色。学习列在有效 reset 位置恢复初态时保留初态参数梯度；从第一张有效图起即可训练 Encoder，没有 TA 冻结开关或历史图像前缀。完整 KB 参数前缀固定为 `encoder/controller/actor`，Dual 为 `kb/active/adapter`。

`DualPolicy(config, live_kb, kb_ready=...)` 创建完整旧 KB 的独立冻结副本，仅将其 Encoder 权重复制到新 Active；Active Controller／Actor 和零 gate Adapter 新初始化。传入的 live KB 模式及梯度不受影响。每 tick 先用旧 KB 自身特征推进，再将该 tick 的新激活 detach 后输入 Adapter，作用于 Active 同 tick 的 synapse 路径。`kb_ready=false` 关闭侧向连接；gate 初值 0 时，gate 可获得梯度，projection／norm 首步梯度可以为 0。`train()` 不解除旧 KB 冻结。每个新 P 由调用方新建 Dual 和优化器。

`frozen_copy(policy)` 独立复制全部参数和 buffer、清除梯度，并在父模块调用 `train()` 后保持 eval。`save_snapshot(policy, new_path)` 导出完整 v4 推理策略、两套 Encoder、P 当时的旧 KB、Adapter、`kb_ready` 和配置身份。`load_snapshot(path, device)` 校验身份／配置、严格加载所有键，返回存储独立的冻结推理模型，不替换为较新的 KB。这是推理导出，**不是阶段 resume checkpoint**；优化器／RNG／源码／引用完整性及阶段原子提交由 06 实现。C 学生是另一份可训练 `StandalonePolicy`，不能直接把冻结教师切成 train 充当学生。

```powershell
python -m unittest discover -s tests/continual_nav_opd -p test_models.py
```

快照的哈希配置保存完整 `ticks_by_task` 映射，重载后仍可按两个任务各自的预算推理。旧 `ctm.ticks: 2` 配置和产物不自动迁移；阶段恢复同时检查配置、带 ticks 的阶段表及源码哈希。该改动不增加任何神经网络参数。

## 04：Maze 五步 on-policy 课程与 FourRooms Progress

Maze P 使用 `data/maze_curriculum.py` 和 `learning/maze_progress.py`，不使用连续环境槽的 `ProgressCollector`。地图池按 manifest 训练划分的固定去重顺序取嵌套前缀，验证、测试及固定面板不变。起点在当前池全部非终止可通行位置中均匀有放回采样；并非先均匀选地图。每条序列恢复可学习 `start_pre/start_post`，五步内不 reset、不 detach；到达目标的动作计入 Loss，之后 mask，不在同一序列内 reset 新 episode。

先以未更新的当前策略无梯度批量采集 100 条序列。每一决策批量执行 GPU 前向，学生按 softmax 概率用独立 Torch RNG 抽样动作，BFS 标签仅提供监督。保存动作前 RGB、标签、动作和有效 mask。采集结束后分 20 组，每组 5 条，从**有梯度的可学习初态**重放相同观察；不重新抽样动作、不推进环境、不使用采集初态的无梯度副本。

每组有效决策 one-hot CE 求和并反传；CE 等于教师 one-hot 到学生的 KL。全部组结束后按整个更新的有效决策总数归一化梯度，检查有限值、clip0.5，Adam 一次。没有策略梯度、reward Loss 或熵正则。池扩增不重建 Active、Adapter 或 Adam。新 visit 的 P 仍按原规则从当前 KB 新建 Dual、Active/Adam 与侧向连接。

每个 run 一次加载全部 50000 张 19×19 RGB 原图，数组只读、P/C/F 共享；独立本地评估另行加载对应面板。P 的紧凑表仅包含训练前缀：int32 状态/转移索引、标签、距离、终止标志；不保存数百万 Python 状态字典，也不缓存全部状态的 84×84 图片。每张地图从 goal 反向 BFS，按上/下/左/右选最短动作，动态批量渲染的 nearest 缩放与原 PIL 逐像素一致。五个时间步依次推进，独立序列批量并行，无逐序列进程或异步旧策略采样。

FourRooms 保留 `ProgressCollector` / `run_progress` / `update_window`：50 新观察/窗口，保存 detached 当前 rollout 起点，跨窗口继续状态，真实 episode 边界 reset。学生概率采样，教师连续窗口、软 KL、预算和 minibatch 设置保持原状。Maze C/F 使用原教师与原窗口路径。

Maze 日志包含地图池、池内/P 累计 Adam 更新、有效决策数、整体与逐步 KL、初始窗口/视觉梯度、采集/重放耗时和缓存字节数。Maze P 不计算或记录动作一致率，地图池边界只提交 checkpoint。整个训练流程不运行 validation/test，闭环成功率后续用阶段权重在本地评估。Maze P 图表主轴为 `global_optimizer_updates`；KL 不证明完整 rollout 成功。

```powershell
python -m unittest tests.continual_nav_opd.test_maze_curriculum -v
python -m tasks.continual_nav_opd.verify_maze_progress --device cuda:0 --output-dir <新验证目录>
```

## 05：完整 KB 压缩与 Fisher

`run_compress_stage(envs, dual_teacher, kb, fisher, config, steps, action_rng, minibatch_rng, start_transition_id=0, on_window=None)` 使用独立冻结完整 Dual 教师采样动作，同时推进教师与 KB 学生各自的状态。窗口保存 KB 自己的起点；KB 从当前参数开始，不复制 Active。学习所有 `encoder/controller/actor` 参数，目标为教师概率到 KB 的 KL，加完整视觉 EWC。首个 C 没有历史 Fisher，EWC 为 0；后续 EWC 分别记录视觉与 Controller＋Actor 的贡献。

`run_fisher_stage` 在当前 KB 自身轨迹上精确采集配置预算，用独立 NumPy RNG 无放回选评分 ID。每个样本从其窗口起点重放全部新观察，对采样动作的 log probability 求梯度，先平方再平均。所有 KB 参数必须入字典，包括未使用梯度对应的 0；不平方 batch 平均梯度。F 没有 optimizer step，参数不变，完成后清除梯度。Online Fisher 为 `0.3 * previous + current`，中心替换为本次 KB 的独立 CPU 拷贝。

```powershell
python -m unittest discover -s tests/continual_nav_opd -p test_compress_fisher.py
python -m tasks.continual_nav_opd.verify_stage --stage 05 --device cuda:0 --output-dir scientific-evidence/continual_nav_opd/new_compress_audit
```

05 验证使用实际教师，在两任务上顺序执行 P→C→F，第二 C 使用第一 F 的完整视觉约束；同时检查各任务 P Active 和 C KB 在两任务固定小面板上的表现。测试成功仅说明工程正确。

## 06：训练、评估、恢复与可视化

完整流程只有 P→C→F，没有 TA 输入。每个 P 创建新 Active 和 Adam；每个 C 创建新 Adam；跨阶段保留 KB、Fisher、参数中心、预算和独立 RNG。正式配置为 2 visits、12 阶段；smoke 的 Maze P 为 2 个小地图池各 1 更新，FourRooms P 为 256 步/3 更新，C 为 128 步/2 更新，F 为 32 步/8 个评分样本；总最多 3152 决策/18 次更新。

```powershell
# Small verification budget; output directory must not exist.
python -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/smoke.yaml --device cuda:0 --seed 0 --run-dir scientific-evidence/continual_nav_opd/new_smoke --wandb-mode disabled

# Formal command, to execute only when explicitly authorized.
python -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/remote_config.yaml --seed 0 --run-dir runs/continual_nav_opd/new_seed0 --wandb-mode online

# Exact same config, seed and device override as the original run.
python -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/smoke.yaml --device cuda:0 --seed 0 --run-dir scientific-evidence/continual_nav_opd/new_smoke --wandb-mode disabled --resume
```

`--max-stages N` 只在完整阶段提交后停止，未完成全部训练时不触发评估。`--device` 是显式配置覆盖，进入配置哈希；恢复必须保持一致。各训练槽同步 step；评估支持 `serial` 或显式 `spawn` 的 `subprocess`，仅环境进入 worker，模型和完整双列状态始终在主进程。

Runner 只执行 P→C→F。每个 P 保存完整 Active 双列推理快照，每个 C/F 保存该阶段的独立 KB 快照，共 12 份；F 不更新权重，但仍保存明确的阶段产物。最后 F 提交后导出 `exports/final.pt` 和带 SHA 的 `exports/stages.json` 权重索引，随后标记 `finalized`。训练中不运行评估。默认 `train` 命令在全部训练 finalized 后启动独立评估进程：依次对12份阶段权重执行两任务 validation，再对最终 KB 执行 test。默认输出位于同级 `<run-dir>_evaluation`，可用 `--evaluation-dir` 指定；`--skip-evaluation` 只训练和导出。评估不写训练目录或训练 W&B，恢复 checkpoint、教师/地图身份和训练日志仍保留。

独立评估入口也可在本地调用或重试；匹配本次计划且 SHA/身份/面板数量验证通过的已完成报告会跳过，不重跑训练。评估失败时命令返回失败，训练仍保持 finalized；优先独立重试评估，不必调用训练 resume。

```powershell
python -m tasks.continual_nav_opd.evaluate_stages --stage-index <local-run>/exports/stages.json --output-dir <local-run>_evaluation --device cuda:0
# To defer evaluation to another machine, append --skip-evaluation to the train command.
```

每阶段完成训练并保存带身份和 SHA 的推理权重后，原子写恢复 `.pt`、`.complete.json`，最后更新 `checkpoints/latest.json`。恢复检查完整配置、源码、教师、地图 manifest、全部阶段权重与 metadata 的 SHA／大小、预算／更新计数和时序协议。新边界身份为 `v4_training_boundary`，旧依赖评估的 `v4_stage_boundary` 不迁移。P 的快照保存当时旧 KB、两套 Encoder、Actor 和 Adapter；C/F 快照保存当时 KB，评估不会改接后来的权重。

每个 Maze 池完成后原子保存完整 Dual、Adam、下一个池索引、有效决策/更新计数和全部 RNG。latest 最后更新。中断重做未完成池，已完成池不重复训练；完整 P 保存推理快照后发布给 C。FourRooms 与 C/F 仍不做窗口或 Adam 中途恢复，未提交阶段完整重做；残留文件用独立 attempt 隔离，以已提交快照列表为准。最后 F 已提交但导出中断时，只重做 finalization，不加载全地图或重做 F；已 finalized 的 resume 不改写产物。进度以 `latest.json` 和 complete marker 为准。

训练事件与 W&B 保留 KL、FourRooms P/C 一致率、熵、EWC、梯度、动作、训练 reward、FourRooms 位移／转向率、采集／学习／Fisher 耗时和显存；这些训练统计不构成固定面板评估。`disabled` 不导入 W&B；`online` 使用已登录环境，不保存凭据。独立评估只写指定的新 JSON，保留两任务固定面板、argmax、连续窗口及 serial/subprocess 后端，不改权重或 RNG。

```powershell
python -m tasks.continual_nav_opd.evaluate --checkpoint scientific-evidence/continual_nav_opd/new_smoke/exports/stages.json --policy kb --stage-key pnc/v0/maze_medium/C --split validation --tasks maze_medium fourrooms --backend serial --num-envs 2 --device cuda:0 --output scientific-evidence/continual_nav_opd/new_eval.json
python -m tasks.continual_nav_opd.evaluate --checkpoint scientific-evidence/continual_nav_opd/new_smoke/exports/stages.json --policy active --stage-key pnc/v0/maze_medium/P --split validation --device cuda:0 --output scientific-evidence/continual_nav_opd/new_active_eval.json
python -m tasks.continual_nav_opd.evaluate --checkpoint scientific-evidence/continual_nav_opd/new_smoke/exports/final.pt --policy kb --split test --device cuda:0 --output scientific-evidence/continual_nav_opd/new_final_test.json
python -m tasks.continual_nav_opd.visualize --checkpoint scientific-evidence/continual_nav_opd/new_smoke/exports/final.pt --policy kb --task fourrooms --panel-index 0 --device cuda:0 --output-dir scientific-evidence/continual_nav_opd/new_visualization
python -m tasks.continual_nav_opd.visualize --checkpoint scientific-evidence/continual_nav_opd/new_smoke/exports/stages.json --policy active --stage-key pnc/v0/maze_medium/P --task maze_medium --panel-index 0 --teacher-diagnostics --device cuda:0 --output-dir scientific-evidence/continual_nav_opd/new_teaching_visualization
```

独立单报告评估输出必须不存在。`--stage-key` 可选择任意 P/C/F；索引由训练完成时生成，也可直接传阶段 `.pt`。`.pt` 需携带 SHA 侧车、metadata 及其侧车、引用的地图 manifest。索引或直接权重加载不依赖恢复 checkpoint、训练源码哈希、教师权重或 W&B，可复制到本地评估；保持相对布局和地图路径。评估 JSON 记录阶段身份与真实权重 SHA。`latest.json` 仍受严格续训源码校验，未完成 run 可直接使用已提交阶段 `.pt`。`--policy active` 拒绝单列权重。可视化仍用局部 RGB，输出 GIF/CSV/reward/trajectory；只有显式 Active `--teacher-diagnostics` 才加载教师。

## 07：集成验收与性能交接

`python -m tasks.continual_nav_opd.verify_smoke --device cuda:0 --output-dir <新目录>` 先运行两任务、两 visits 的12阶段，核验零训练评估、阶段权重与恢复；完成后独立评估全部阶段，比较最终 KB serial/subprocess，检查两任务 Active 媒体。离线评估前后训练目录 SHA 必须一致。独立单元测试仍需另行执行。最新验证见 [训练/评估分离验收](VERIFICATION_TRAINING_ONLY.md)。

`python -m tasks.continual_nav_opd.profile_training --device cuda:0 --num-envs 2 --microbatch 8 --minibatches 1 --output-dir <另一新目录>` 使用真实教师测量采集、学习与逐样本Fisher，记录视觉/CTM前向、视觉反向、优化器和显存。同步诊断存在计时开销；本机结果不能代表RTX5090实测。正式启动、恢复、输入产物和异常边界见 [运行手册](RUNBOOK.md)。

## 本地 32 槽与环境序列 microbatch

`configs/local32_8gb.yaml` 继承 `remote_config.yaml` 的全部预算、FourRooms/C/F 的32 个采集槽、50 张新观察/窗口和4个环境 minibatches；仅设置 `ctm_compile: default` 和 `sequence_microbatch_envs: 1`。Maze P不使用这个选项。后者的单位是**环境槽**，与 `encoder_microbatch_images` 的图片单位不同。每个原更新组仍含8槽×50观察=400条样本；分8次完整序列反传后，统一裁剪并执行一次Adam更新，每窗口仍4次更新。此选项仅作用于原窗口学习路径；Maze P 固定 microbatch5，五步完整递归。原路径序列保留50张观察，不截断时间轴；Maze每张5ticks，FourRooms每张2ticks，M=40内部ticks。

`sequence_microbatch_envs: 0` 保持整组重放。非零时，KL梯度按各分块有效样本数/整组有效样本数累积；完整KB的EWC每次Adam更新只加入一次。采集状态、窗口起点和教师标签语义保持不变。GroupNorm不使用跨样本统计，但浮点求和顺序会变化，不承诺逐位数值相同。

每个完成的训练日志事件会立即刷新到stdout，同时保留完整本地JSONL与既有W&B图表。默认首窗口打印一次，之后每10窗口及阶段末打印；编译和未完成的窗口不会打印训练Loss。Windows本地编译设置 `TORCHINDUCTOR_COMPILE_THREADS=1`，避免编译子进程使用不受支持的 `pass_fds`。
