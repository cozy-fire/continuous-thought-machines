# TAPD × CTM 项目交接规则

本文件给接手本仓库的 Agent 使用，记录**长期有效的操作边界和核验方法**，不记录某次训练的 PID、SSH 端口、W&B run、剩余时间或阶段进度。接手时先读用户最新指令，再查实际代码、配置、Git 和运行实例；不要把本文件当作当前状态报告。

> 文件名 `AGENT.md` 按用户要求保留。Codex 自动发现的是 `AGENTS.md`，根目录另有一个极短入口指向本文件。其他 Agent 若不自动加载入口，须显式读取本文件。

## 1. 决策优先级与仓库边界

1. 用户本次明确要求优先；然后是实际代码与解析后的配置；再是本文件、[项目 README](README.md) 和 [任务 README](tasks/continual_nav/README.md)。`plan/` 中的早期计划和 `analysis/` 中的历史报告不能覆盖当前实现。发现不一致时，先查源码和运行产物，再告知用户。
2. 当前主方法在 `main`：`tapd_ctm_visual_revisit`，checkpoint schema v2。旧 W.collect/W.fit/X 及预测误差好奇心实现留在 `v1-W_X_C_F` 分支。不要把 v1 的术语、replay/high_error、checkpoint 或视觉权重带入 v2；不要把整体方案的效果差异归因于单一机制。
3. Git `origin` 应指向 `cozy-fire/continuous-thought-machines`；SakanaAI 的仓库是上游参考，不是本项目的推送目标。操作前执行 `git remote -v`、`git status --short`、`git branch --show-current`。Windows 若遇 dubious ownership，使用**单条命令**的 `git -c safe.directory=<本次仓库绝对路径>`，不要修改全局 Git 安全配置。
4. `runs/`、`data/*`、`scientific-evidence/`、`plan/`、`research/` 被忽略；不要把训练权重、地图数据、临时日志、凭据或 W&B key 加入提交。用户只要求只读分析时，不停止进程、不编辑文件、不改变远端状态。停止、重启、删产物、删 W&B run、切分支、推送等操作要以当前用户授权为准；处理目标用明确的 run 路径或 PID，避免通配式批量操作。
5. 远端实例、GPU、SSH 端口、仓库目录、`ctm` 环境目录和登录态都可能换。每次重新确认，不能复用旧对话里的连接信息。不要把密码或 API key 写入代码、文档、命令输出或训练日志。

## 2. 真正的入口和配置来源

| 目的 | 入口／来源 | 交接时要注意 |
|---|---|---|
| 阶段顺序、visit 边界、方法名 | `tasks/continual_nav/schedule.py` | `expand_stages` 是阶段表的准绳；一次 TA visit 完成任务顺序中**每个任务的全部 round**，不是每轮就切任务。 |
| 正式预算 | `tasks/continual_nav/configs/full.yaml` | `rtx5090_32gb.yaml` 继承它，覆盖训练/评估槽数并显式写出 minibatch（当前 minibatch 数与 full 同值）；`smoke.yaml` 是小预算测试。`config.py` 的 dataclass 默认值不是正式实验预算。 |
| 配置检查和预算计算 | `tasks/continual_nav/config.py` | 始终运行 `python -m tasks.continual_nav.config --config <yaml>`，核对 `config_hash`、`ta_rounds`、`shared_ta_steps`、`main_steps`、`x_joint_updates`。所有阶段预算必须能被 `training.num_envs` 整除。 |
| 训练、日志、恢复 | `tasks/continual_nav/train.py`、`checkpoint.py`、`wandb_logging.py` | 从仓库根目录以 `python -m ...` 调用；`--seed` 和 `--run-dir` 必填。 |
| 环境、数据与动作 | `envs/`、`data/manifest.py`、`data/rollout.py` | 地图数据不在 Git；先验证目录和 manifest，不要重造固定评估面板。 |
| 模型与损失 | `models/`、`learning/` | 修改梯度边界、时序或 reward 前，先读对应实现和测试。 |
| 固定面板评估 | `evaluate.py`、`envs/evaluation.py` | 训练环境同步逐槽 `step()`；**只有评估**的 subprocess 后端使用 `spawn` 并行环境。 |
| 验证 | `tests/continual_nav/`、`verify_smoke.py` | 测试输出目录必须是新目录；正式预算修改通常先做配置解析和针对性验证，不用跑完整正式训练当测试。 |

最短的无训练副作用检查：

```bash
python -m tasks.continual_nav.config --config tasks/continual_nav/configs/rtx5090_32gb.yaml
python -m tasks.continual_nav.train --config tasks/continual_nav/configs/rtx5090_32gb.yaml --method tapd_ctm_visual_revisit --dry-run
```

`--dry-run` 会读取迷宫地图并生成阶段表，但不开始训练。当前正式配置为 4 个 TA visits、每任务每 visit 2 轮、每轮 X 500,000 环境步；**不要抄旧数字开展新运行**，以命令解析结果为准。`rtx5090_32gb.yaml` 是配置名称，不证明实际 GPU 为 RTX 5090；用 `nvidia-smi` 核对型号和显存。

## 3. 不可混淆的方法契约

- 两任务为 `maze_medium` 与 `fourrooms`，输入统一为 `uint8 [3,84,84]` RGB，策略不接收任务 ID。Maze 的 0/1/2/3/4 为上/下/左/右/等待；FourRooms 的 0/1/2 为左转/右转/前进，3/4 是环境中的等待映射。不得把动作索引直接当作两任务同义动作。
- 视觉骨干为 ResNet34-2 + GroupNorm32，输出 `[B,128,21,21]`。KB 是无 critic 的单列策略；Active 是带 critic 的双列策略，Adapter 接 KB **当前 tick** 的 post activation。每观察推进 2 tick，CTM 记忆 40 tick。X 的 KB 冻结，Active 每轮重新初始化；共享 Encoder/projector 权重及视觉优化器状态跨 X 轮延续。
- v2 TA 每轮是 `X→C→F`，**没有 W 阶段、动作条件预测器或 high_error replay**。X 每槽只保留当前 episode 最近 20 张原始 RGB 帧；reset 用初始帧填窗。动作后必须用真实 `transition_next_obs` 先算回报再入窗，不能把自动 reset 的图片当上一 episode 的末帧。回报仅为像素重访惩罚，不混入环境任务成功奖励。X 的 PPO 与 SIGReg 一次反传更新 Encoder、projector 和 Active；KB 保持冻结。
- C 用冻结的双列教师及 Encoder、20 张观察的无梯度 burn-in、KL 蒸馏和 Online EWC 更新 KB；F 在 KB 自身轨迹上估计 Fisher，不训练策略。TA 结束才导出并永久冻结共享视觉。P&C 每任务执行 `P→C→F`；当前代码的 P 用任务成功奖励训练 Active/Adapter/Critic，**Encoder 仍冻结**。若用户以后要求 P 更新 Encoder，那是新的算法和 checkpoint 契约变更，不能悄悄在运行中切换。
- TA 每个完整 visit 末只评估当前 KB；P&C 每个完整 visit 末评估 KB、Maze Active、FourRooms Active，均覆盖两个任务。P 完成保存的 Active 快照带**当时的旧 KB、Adapter 和视觉引用**；不能把它接到后来 C 更新过的 KB。最终 test 仅用最终 KB（单列基线用最终单列策略）。评估环境步独立计数，不属于训练预算。
- 对照方法名见 `schedule.METHODS`。除主方法外都要求同 seed 的 schema-v2 `exports/vision_final.pt`，且结构与 Maze manifest 哈希匹配；`single_task_ctm_shared_vision` 还必须传 `--task`。对照报告必须计入或明确列出共享视觉预训练成本，不能称为“完全没有预训练”。

## 4. 新训练、续训与代码更新

在**当前**实例上先找仓库和环境、确认数据存在、读取 Git 状态与运行进程，确认 W&B 使用环境内版本：

```bash
git remote -v
git status --short
nvidia-smi
<ctm-python> -c 'import wandb; print(wandb.__version__, wandb.__file__)'
```

远端曾发生用户级 `~/.local` 的 `wandb` 覆盖 `ctm` 环境版本，导致 `ImportError: cannot import name 'Imports'`。遇到此类情况，用 `PYTHONNOUSERSITE=1 <ctm-python> ...` 并确认 `wandb.__file__` 位于环境目录；不要以修改训练算法掩盖包污染，也不要在文档里保存登录 key。

新运行必须选**不存在**的 `--run-dir`，因为 Runner 会以 `exist_ok=False` 创建。先验证配置、方法、seed、地图数据和实际 GPU，再在后台启动并将 stdout/stderr 重定向到**运行目录之外**的 launch log，保存精确 PID；使用 `nohup` 或 `start_new_session=True`，并检查 SSH 断开后进程仍在。示意：

```bash
PYTHONNOUSERSITE=1 <ctm-python> -u -m tasks.continual_nav.train \
  --config tasks/continual_nav/configs/rtx5090_32gb.yaml \
  --method tapd_ctm_visual_revisit --seed <seed> \
  --run-dir <new-run-dir> --wandb-mode online
```

后台机制、日志路径和 PID 保存由执行者补齐。运行创建后核对 `resolved_config.yaml`、`provenance.json`、`events.jsonl` 和实际进程；只有 W&B 未禁用时才应有 `wandb_run.json`，在线 URL 还须结合同步日志核对。只看到进程存活或本地日志初始化，不等于训练正常推进。

停止时先用完整命令行确认**唯一目标 PID**，对该 PID 发送 `TERM`，再确认它退出；不要用宽泛 `pkill -f python`。旧 run 产物默认保留。远端 `git pull --ff-only` 前检查工作区是否干净并避免与运行中代码交叉修改；有未提交改动时先审阅，不要 `reset --hard` 或覆盖。若用户要改配置后重训，停止旧进程、提交或同步配置、用**新 run 目录从头运行**；不能对变更后的配置强行 `--resume`。

同一 run 的合法续训：

```bash
PYTHONNOUSERSITE=1 <ctm-python> -m tasks.continual_nav.train \
  --config <与原 run 完全相同的配置> --method <原方法> \
  --seed <原 seed> --run-dir <原 run-dir> --resume --wandb-mode online
```

`--resume` 只读取 `checkpoints/latest.json` 指向的完整 `.complete.json` 标记；配置哈希、阶段表、方法/seed、受检源码哈希与引用文件 SHA-256 都必须匹配。受检源码范围由 `checkpoint.source_manifest()` 定义：`tasks/continual_nav/**/*.py`、`models/resnet.py` 和 `models/modules.py`，不是全仓库。X/C/F/P 中途没有可续的 minibatch 或窗口；中断后从上一**已提交阶段边界**重做未提交阶段。`--max-stages` 只在完整阶段结束处停。v1 checkpoint/视觉 artifact 不能用于 v2。切换实例时，若要续训，复制完整 run 树及对应地图数据，连同 manifest、checkpoint 标记、exports、diagnostics 和 `.sha256.json` sidecar 一起核对；只拷贝一个 `.pt` 不足以恢复。

## 5. 观察曲线与判定证据

`events.jsonl` 是逐事件原始记录，`metrics.jsonl` 是供画图的子集，W&B 由同一事件生成。详细字段看 [W&B 指标速查表](tasks/continual_nav/analysis/WandB_指标速查表.md)。读曲线时必须同时看 `stage`、方法、seed、`global_env_steps`、阶段内 `consumed` 和完整 visit；同一 `ta_X_<task>` 面板会串接多个 round/visit。`exploration.steps_per_round` 是**每一轮 X**的预算，`global_env_steps` 跨 X/C/F/P 累加，不能拿全局轴直接与单轮预算比较。

X 中的 `mean_penalty`、动作频率、整帧不变率、匹配间隔等是**当前 X 阶段从起点累计**的统计；比较早晚行为须用 `累计值×累计步数` 作差还原区间，不能把曲线点当作当前 rollout 的均值。`ppo_loss` 是 `policy_loss + vf_coef × value_loss − entropy_coef × entropy`，尚未加 SIGReg；`value_loss` 为未裁剪 MSE。当前实现每轮 PPO 日志只给**最后一个 minibatch** 的损失，不是整轮 minibatch 平均，毛刺不可直接判成训练发散。`grad_norm` 为裁剪**之前**的范数。诊断时同时看 KL、entropy、惩罚、动作分布和 episode/timeout 周期。

Maze 画面只随代理位置变化，`1 − identical_frame_fraction` 可作为**确实发生位移的比例**；这不等于到达新位置、走出局部循环或达到目标。FourRooms 的原地旋转也会改变局部 RGB 图，不能用该指标直接算位移率；当前通用日志没有 FourRooms 的独立位移率，留档图片和动作只能辅助判断。若要严格统计 FourRooms 位移，须额外记录环境位置或使用经验证的状态轨迹。`nonzero_penalty_fraction` 是 20 帧窗口的重访信号，不是成功率。X 结束才写 `diagnostics/`，每轮最多四条**选择性的**完整训练 episode；可用逐帧图、动作、reward、间隔检查实际轨迹，但不能把它们当作固定评估样本。正式任务表现以 visit 末 validation 和最终 test 的固定面板成功率为准；评估尚未执行时明确写“未知”。

手动固定面板评估可用 `python -m tasks.continual_nav.evaluate --checkpoint <run>/checkpoints/latest.json --split validation --output <新文件>`，按需传 `--backend subprocess --num-envs 16`。输出路径必须不存在。`--policy active` 针对具备双列状态的完整阶段 checkpoint，不能拿最终 `exports/final.pt` 冒充历史 Active 快照。`evaluation.interval_steps`、`drift_episodes` 是兼容字段，不会触发自动阶段中途评估。训练环境的 16 槽是同步逐槽执行；不要把评估的 subprocess 加速说成训练环境并行加速。

## 6. 修改、测试、交付的最低标准

1. 先明确用户要求是只读分析、改代码、续训还是从头重训；只读任务不产生新训练、不改 Git。涉及核心机制时先定位数据/状态/梯度边界和对应测试，关键逻辑写准确英文注释；配置或文档的小改动保持局部，不附带重构。
2. 改正式预算时改 `full.yaml`，检查硬件 profile 的继承关系和 README 中对应数字；不要误改 `smoke.yaml` 或只改 `config.py` 默认值。配置变更后运行 config CLI、必要的 `--dry-run`，核对预算和阶段表。模型、奖励、采样、恢复或评估逻辑变更时运行 `python -m unittest discover -s tests/continual_nav -p 'test_*.py'` 及新目录的 `python -m tasks.continual_nav.verify_smoke --output-dir <新目录> --device <cpu或cuda:N>`；正式训练不是测试替代品。
3. 远端训练前确认代码提交/工作区、`resolved_config.yaml`、数据 manifest、环境包与 GPU；缩小预算的 smoke 成功后再启动大预算。相同代码、配置和 seed 才有可比较的续训；比较实验时记录被改变的变量和共享视觉训练成本。
4. 提交前查看 `git diff --check`、`git status --short`，只提交本次授权的源码/文档。不要顺手清理历史产物或用户文件。推送后核对本地、`origin/main`、远端 checkout 的 commit ID；远端工作区不干净时先处理冲突来源，不要覆盖。
5. 汇报必须区分“读到源码”“通过测试”“观察到进程仍运行”“完成完整训练”“固定面板实际结果”。训练完成要同时核对最后完整阶段 checkpoint、`finalized`、`next_index == len(stages)`、预算计数、`evaluation/final_test.json`、`exports/final.pt`、模型/地图可加载及进程退出状态。`final.pt` **不在 checkpoint 的引用校验列表里**，必须单独调用 `checkpoint.load_artifact(<run>/exports/final.pt)` 检查它的 `.sha256.json` sidecar，再用 `evaluate.load_for_evaluation(...)` 检查模型和地图；`checkpoint.load` 应传入原配置哈希及当前受检源码哈希。后台启动器最好记录退出码；若未记录，不能凭进程消失声称退出码为 0。仅凭 W&B 曲线停止或文件存在不能宣称完成。若某项证据尚未取得，直接写清边界。
