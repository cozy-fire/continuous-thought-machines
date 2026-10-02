# CTM P&C OPD 运行交接

## 环境与输入检查

从仓库根目录执行。Windows 使用 `conda activate ctm`，设置 `PYTHONNOUSERSITE=1`；本地已验证解释器为 `D:\conda\envs\ctm\python.exe`。远端必须重新确认解释器、Torch/CUDA、`nvidia-smi`、可用内存与磁盘。不要以 profile 名称代替硬件核验。

必须具备 `data/mazes/medium` 的原始 Maze 数据，以及 `logs/rl/MiniGrid-FourRooms-v0/run1/ctm_2_M40_100M_e001/checkpoint.pt`。Maze 教师为实时 BFS，没有 Large 教师权重依赖。FourRooms 教师严格加载原始配置及全部权重键；不得静默随机初始化。新 run 会复制教师、生成地图 manifest、记录教师描述与 SHA。不要使用 v2 的 TA/视觉产物。

```powershell
python -c "import sys,torch,wandb; print(sys.executable,torch.__version__,torch.version.cuda,wandb.__file__); print(torch.cuda.get_device_name(0))"
python -m tasks.continual_nav_opd.config --config tasks/continual_nav_opd/configs/remote_config.yaml
python -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/remote_config.yaml --dry-run
wandb login
```

`wandb login` 使用交互式登录，不把凭据放进命令、代码或文档。工程验收默认禁用 W&B；通过本地日志与 mock 验证不代表在线服务已测试。正式运行前确认本环境能正常登录并初始化 online run。

## 预算与正式启动

基础与 remote_config 配置均为两个 visits、每 visit Maze P/C/F 后 FourRooms P/C/F，12 阶段。Maze 每 P=3,200,000、C=1,280,000；FourRooms 每 P=15,000,000、C=6,000,000；每 F=4,096 环境步、1,024 个唯一评分样本。总环境步=50,976,384，不包含评估。remote_config 使用32训练槽、4环境 minibatches、视觉 microbatch=200图片；基础配置8槽。remote_config 每完整窗口1,600环境步，每次更新400条样本；总更新127,400次，是旧16槽/4 minibatch的254,800次的一半。相较64槽/8 minibatch的127,408次，仅消除了FourRooms P尾窗口多出的8次更新。两任务均从第一观察开始教学，没有预热、burn-in、TA或PPO。

下面命令仅作为交接说明，需用户授权后在目标机器执行。`run-dir` 必须不存在：

```bash
PYTHONNOUSERSITE=1 python -u -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/remote_config.yaml --seed 0 --run-dir runs/continual_nav_opd/seed0_first_formal --wandb-mode online
```

后台部署应把 stdout/stderr 放在 run 目录之外，用 `nohup` 或独立会话启动，核验精确 PID、完整命令、W&B URL和事件推进。不要同时启动本机性能测试污染正式计时。RTX5090上的实际吞吐和microbatch200显存上限必须在该硬件实测。

## 验收与计时

```bash
python -m unittest discover -s tests/continual_nav_opd -p 'test_*.py'
python -m unittest discover -s tests/continual_nav -p 'test_*.py'
python -m tasks.continual_nav_opd.verify_smoke --device cuda:0 --output-dir scientific-evidence/continual_nav_opd/new_stage07_smoke
python -m tasks.continual_nav_opd.profile_training --device cuda:0 --num-envs 2 --microbatch 8 --minibatches 1 --output-dir scientific-evidence/continual_nav_opd/new_profile
```

验收入口真实运行两个 visits，首个 P 后停止并恢复，检查全部12阶段、1,664步、20更新、完整Fisher、8份双任务validation、2份visit报告、final test、最终权重。检查 finalized 恢复不改写产物，实测 serial/subprocess 的逐episode结果，渲染两个 v0 Active 的配套旧 KB。保存环境、源码、配置、SHA清单和报告；失败非零退出，不发布通过。

性能入口不运行正式预算；每任务 P/C 各2个完整窗口，F按smoke的32步/8样本测量。参数 `--num-envs`、`--minibatches`、`--microbatch` 必须满足配置校验。报告包含真实教师、环境、视觉前向、CTM tick、视觉反向、整段反向/优化器及Fisher计时、启动和首窗口开销、allocated/reserved显存。诊断同步及反向hook会增加开销，嵌套计时不能直接相加。CTM/Actor/状态图的反向残差包含未单独计时操作，不能称为纯CTM反向。吞吐不能仅根据显存占用判断。

## 合法恢复与部署产物

```bash
PYTHONNOUSERSITE=1 python -u -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/remote_config.yaml --seed 0 --run-dir runs/continual_nav_opd/seed0_first_formal --resume --wandb-mode online
```

只恢复 `checkpoints/latest.json` 指向的已提交阶段边界。配置、方法/schema/时序协议、受检源码、seed、教师、地图和引用文件必须匹配。中断重做未提交的整个 P/C/F及对应评估；不恢复窗口或Adam中途状态。不能修改预算或源码后强行续训。最终 F 已提交但 finalization 中断时，只重做 finalization；已 finalized 不重复 test。

跨机器复制完整run及配套地图，保留引用相对路径，并核验文件数、大小和SHA。只有推理时可使用完整 `.pt` 加 `.sha256.json`、`.metadata.json`、metadata侧车及其引用地图manifest；单独一个 `.pt` 不构成可校验部署产物。Active保存完整旧KB、独立视觉和Adapter，不得改接后来的KB。

## 评估与行为查看

每个 P 后评估完整Active，每个C后评估KB，均用两个任务各200固定validation episode。F后visit汇总复用结果，不重跑；最终KB独立test每任务200集。评估环境步另计。

```bash
python -m tasks.continual_nav_opd.evaluate --checkpoint <run>/checkpoints/latest.json --policy kb --split test --backend subprocess --num-envs 16 --device cuda:0 --output <new-report.json>
python -m tasks.continual_nav_opd.visualize --checkpoint <run>/checkpoints/latest.json --policy active --active-key pnc/v0/maze_medium/P --task maze_medium --panel-index 0 --device cuda:0 --teacher-diagnostics --output-dir <new-media-dir>
python -m tasks.continual_nav_opd.visualize --checkpoint <run>/exports/final.pt --policy kb --task fourrooms --panel-index 0 --device cuda:0 --output-dir <another-new-media-dir>
```

输出目录/报告必须不存在。全局图仅展示，不进入学生输入。`steps.csv`与`trajectory.json`记录每步动作/reward，`behavior.gif`含初始帧，`reward.png`表示同一轨迹。

## 异常诊断与边界

判断已提交进度看latest和complete标记，不以未提交事件或残留snapshot为准。KL按有效样本加权；FourRooms真实位移率与转向率辅助发现转圈，动作一致率不能代替成功率。低显存不证明GPU未充分计算；观察采集、学习、教师、环境、Fisher与评估分项时间。

若出现无标签、非有限Loss/梯度、损坏引用或源码不匹配，记录失败证据，不跳过校验、不换教师、不清理历史run。OOM时记录具体槽数/minibatches/microbatch/allocated/reserved；配置改动用新run验收，不能强行恢复。smoke成功只证明工程流程，不证明正式策略有效。停止、删除、推送和远端正式训练单独遵循用户授权。

## E1/E2 执行与计时口径

完整有效观察窗口使用 dense 重放；含 padding 的窗口仍走逐槽掩码路径。CTM 按时间顺序推进并重置完整状态，可学习初始状态仍参与梯度。`DualPolicy.kb_ready` 保留原持久化键，推理读取生命周期内的 Python 缓存；需要变更时调用 `set_kb_ready`，禁止直接写 buffer。加载快照时自动刷新缓存，冻结快照禁止修改。

C 先以完整冻结双列教师采集全部观察与动作，再在任何优化器更新前从独立 KB 的窗口初态重放完整窗口。视觉编码可以合并图片，CTM 顺序不变；重放终态只接续 KB 采集，不使用教师状态或学习重放终态。参数、预算、Adam、精度及采样 RNG 不变。

C 采集的 Encoder 分块固定为 `min(num_envs, encoder_microbatch_images)`，保持旧逐步推理的实际 CNN batch 形状；窗口图片一次上传，特征提前计算后按时间推进 CTM。未采用增大采集 CNN batch 的初版：其浮点误差在完整窗口中放大，隐藏状态超出容差。P/C 学习重放仍使用配置规定的视觉 microbatch。

默认 `--timing-mode events`：CUDA 分项为当前 stream 的事件区间，包括主机提交工作造成的空隙，不代表纯 kernel 时间；CPU 环境/BFS仍为墙钟。窗口结束统一解析事件；动作所需 CPU 下载仍会等待。`--timing-mode synchronized` 可诊断前后同步的墙钟区间，增加开销。事件携带 `timing_mode`；不同模式的分项不直接作为提速比较。性能诊断入口 `profile_training` 固定使用 synchronized，外层窗口/阶段墙钟用于同负载比较。CPU运行两种模式均使用墙钟。

受检源码和任务执行配置改变，旧开发 run 不允许续训。新完整 v3 inference artifact 在哈希配置中保存两任务 ticks，加载时恢复该映射；缺少 `ticks_by_task` 的旧全局 `ticks: 2` 产物明确拒绝，读取旧实验需使用其对应源码版本。不要重写旧 run 的源码哈希绕过检查。

## 按任务执行 ticks

`full.yaml` 中 `ctm.ticks_by_task.maze_medium: 5`、`ctm.ticks_by_task.fourrooms: 2` 是统一来源；remote 和 smoke 继承它。`memory_length: 40` 固定为内部 tick 数。训练环境步、visit、窗口长度、minibatch、Adam 更新次数及 Loss 均未改变。KB 权重在任务间共享；两列内部逐 tick 对齐，不为任务创建不同 KB。

P 采样及重放、C 双列教师与 KB 学生、F 使用环境任务的 ticks。每个任务报告记录 `ticks` 与 `memory_ticks`；跨任务评估和可视化使用评估任务的 ticks，与 Active 来源任务无关。执行预算记录在本地事件及 W&B run 配置中，不新增 ticks 曲线。直接调用 `step`／`sequence` 必须传 `task=`；不要临时修改 `controller.ticks`，该属性已移除。任务切换使用新的环境和状态，不能将一任务的 live trace 直接移交另一任务。

较长 ticks 会增大递归反传图；50 张观察在 Maze 中展开 250 tick／列。2-tick 或 75-tick 性能与显存数据不能直接用于预测当前 5-tick 负载。正式远端 profile 的显存可行性须在目标 GPU 实测，不能把小预算 smoke 作为大 batch 的保证。

## E3、内存地图与 CTM 编译

正式配置默认 `optimization.e3_cache: true`、`environment.map_cache: memory`。E3 对同一观察、同一列的全部 tick 复用 Attention 的 token/K/V；Q 每 tick 重算。同步衰减权重只在一次序列 forward 内共享，保留梯度，不跨 Adam 更新。缓存不会绕过任务 ticks，教师、Loss、初态、reset 和参数键保持原定义。

Runner 启动时校验并解码训练地图和固定 validation/test 面板，保存连续 `uint8` CPU 数组，各训练槽共享。reset 只读取内存，不再打开 PNG。评估子进程接收解码后的 CPU 图片与起终点，不读取地图文件。缓存不进入 checkpoint；结束或异常退出释放，恢复时重新校验加载。地图加载失败直接报错，不退回磁盘。`map_cache` 事件记录 `maps`、`cache_bytes`、`load_seconds`。独立工具的并行槽通过弱引用池共享缓存，没有全局强引用长期保留图片。

教师模型在阶段入口加载后常驻 GPU，窗口训练不读取权重文件。checkpoint、源码/引用完整性校验和日志仍保留磁盘操作；不能为了省 I/O 跳过这些边界。需要磁盘对照时设置 `environment.map_cache: disk`。

`optimization.ctm_compile` 支持 `disabled`、`default` 和 `reduce-overhead`。基础 `full.yaml` 默认为 `disabled`；与 GPU 型号无关的 `remote_config.yaml` 默认使用 `reduce-overhead`，使用 32 个训练槽、4 个环境 minibatch 及视觉 microbatch 200，预算继承基础配置。只编译 CTM tick 的张量核心，Encoder、Python 校验、状态容器和计时不在编译区域。编译失败直接抛出错误，禁止通过 `suppress_errors=True` 静默切回 eager。编译后的 callable 不附着在模型上，完整双列快照仍拥有独立旧 KB 存储与原参数命名。

编译性能比较必须分开首次编译与热身后的窗口耗时。不同 batch 形状、冻结/反传模式或输入 stride 可能触发重新编译；单个 kernel 编译成功不代表完整 P/C/F 可训练。完整验证入口：

```bash
python -m tasks.continual_nav_opd.verify_smoke --device cuda:0 --ctm-compile default --output-dir <new-directory>
```

Windows 本地编译需要匹配 PyTorch 的 `triton-windows`、`PYTHONUTF8=1`，并把 `TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR` 指向可写目录。本轮测试环境是 PyTorch 2.13/CUDA 12.6 与 triton-windows 3.7.1.post27；不能直接据此断言 Linux RTX5090 的提速。推理产物按原始配置字段校验哈希，再补可选运行选项默认值；两任务 ticks 映射是必需字段，不做旧全局 ticks 迁移。训练恢复仍受当前源码与配置校验限制。

## 本地 8 GiB GPU 的32槽入口

在conda的ctm环境及仓库根目录执行以下命令。此入口仍是完整正式预算，不是本轮连续窗口诊断的短预算；只有用户明确要求完整本地训练时才启动。

```powershell
$env:TORCHINDUCTOR_COMPILE_THREADS='1'
python -u -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/local32_8gb.yaml --device cuda:0 --seed 0 --run-dir <new-run-directory>
```

CLI具体参数以 `python -m tasks.continual_nav_opd.train --help` 为准。新profile使用default编译，避免reduce-overhead的CUDA Graph树管理；不修改remote_config。环境序列microbatch=1仍完整重放50观察、按原8槽组累积梯度，每窗口4次更新，不通过减少ticks、窗口长度或增加Adam步数降低显存。配置/源码哈希发生变化时必须创建新run，不绕过旧checkpoint恢复检查。

后台日志应能看到首个完成窗口的 `event=training` JSON，包含环境步、KL、动作一致率、更新次数和耗时；随后按日志间隔及阶段末打印。首窗口含编译开销，因此不能以尚未出现Loss判断进程失败。若长时间无日志，分别检查采集、编译及学习，而不是只看进程存活。
