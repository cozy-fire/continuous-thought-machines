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

两个 visits，每 visit Maze P内部完整课程→C→F，再 FourRooms P→C→F。Maze 池为 10/40/160/640/2560/10240/44488，前三池2000、后四池1500次 Adam：12000/visit。100条独立5决策序列/更新，microbatch5；T=5、M=40。每个池均使用当前权重的学生概率采样。Maze C=1280000；FourRooms P=4800000、C=1280000；F=4096决策、1024评分样本不变。实际有效决策因终止mask变化，最多26736384：P=21600000、C=5120000、F=16384。

以32槽、50步观察窗口、4个更新分组为对齐基准：`remote_config.yaml` 和 `local32_8gb.yaml` 中，两任务每 visit 的 P 各12000次Adam、C各3200次，两 visits 合计60800次，其中Maze P为24000次。FourRooms/C每组400个决策，Maze P每次最多500个有效决策；更新次数对齐不代表决策数或耗时相等。`full.yaml` 的默认8槽不变，FourRooms P为48000次/visit、两任务C各12800次/visit，Maze P仍12000次/visit，合计171200次，不能套用32槽的对齐结论。新预算改变配置哈希，必须使用新run，不恢复旧预算产物；schema与训练算法不变。

按此前A4000、32槽及 `local32_8gb.yaml` 运行配置的短测时外推，新预算名义训练时长约74小时（3.1天），不含训练后的独立评估。这是历史吞吐外推，未运行新预算完整训练；新实例需重新测时，策略效果由后续固定面板评估判断。

全部50000张地图一次加载为只读CPU缓存，44488张训练地图的状态表与采样索引不包含验证/测试数据。启动日志检查 `map_cache` / `maze_state_table` 的地图数、非终止位置数、加载/构建耗时及实际数组字节占用。配置错误或解码失败直接终止，不退回磁盘。正式最后池必须覆盖整个固定训练划分，smoke显式用2/4张小池。

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

验收入口先真实运行两个 visits：首池及首 P 后中断恢复，核验12阶段、最多3152决策、18更新、完整Fisher、12份阶段权重及最终权重/索引，确认训练没有评估事件或目录。训练 finalized 后才单独评估全部阶段、比较最终 KB 的 serial/subprocess 逐episode结果、渲染两个 v0 Active，并核验整个训练目录 SHA 未变化。默认训练 CLI 同样在 finalized 后调用独立评估进程；工程验收另外比较后端和媒体，不在正式启动时重复这些检查。

性能入口不运行正式预算；每任务 P/C 各2个完整窗口，F按smoke的32步/8样本测量。参数 `--num-envs`、`--minibatches`、`--microbatch` 必须满足配置校验。报告包含真实教师、环境、视觉前向、CTM tick、视觉反向、整段反向/优化器及Fisher计时、启动和首窗口开销、allocated/reserved显存。诊断同步及反向hook会增加开销，嵌套计时不能直接相加。CTM/Actor/状态图的反向残差包含未单独计时操作，不能称为纯CTM反向。吞吐不能仅根据显存占用判断。

## 合法恢复与部署产物

```bash
PYTHONNOUSERSITE=1 python -u -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/remote_config.yaml --seed 0 --run-dir runs/continual_nav_opd/seed0_first_formal --resume --wandb-mode online
```

只恢复 `checkpoints/latest.json` 指向的已提交Maze池/其他阶段边界。配置、方法/schema/时序协议、受检源码、seed、教师、地图和引用文件必须匹配。边界类型为 `v4_training_boundary`，旧评估耦合边界不迁移。Maze P 每池保存 Dual/Adam/课程计数及 RNG，重做未完成池；末池已保存但权重导出中断时恢复完整 P 后重新导出。FourRooms和C/F重做未提交阶段，不恢复窗口。最终 F 已提交但 finalization 中断时，只重做导出，不加载全地图；已 finalized 不改写产物。

跨机器复制完整run及配套地图，保留引用相对路径，并核验文件数、大小和SHA。只有推理时可使用完整 `.pt` 加 `.sha256.json`、`.metadata.json`、metadata侧车及其引用地图manifest；单独一个 `.pt` 不构成可校验部署产物。Active保存完整旧KB、独立视觉和Adapter，不得改接后来的KB。

## 评估与行为查看

训练阶段内不评估，只保存可独立推理的权重。`snapshots/` 保存 P 的完整 Active/旧KB/Adapter 和每个 C/F 的当时 KB；`exports/stages.json` 索引全部12阶段，`exports/final.pt` 保存最终 KB。默认 `train` 在全部训练 finalized 后调用独立 `evaluate_stages`：12份阶段权重各在两任务做 validation，随后最终 KB 在两任务做 test。正式配置每任务各200固定 episode；预算和面板没有改变。默认评估输出为同级 `<run-dir>_evaluation`，`--evaluation-dir` 可指定 run 之外的目录。`--skip-evaluation` 只训练，之后在本地用同一独立入口评估。`--max-stages` 停在未完成训练时不评估。

评估失败不撤销训练 finalized。独立入口核验计划、权重、metadata 和结果 SHA，重试时跳过已完成且匹配的报告。成功后生成 `summary.json`；失败记录 `status.json` 的当前报告与错误。默认训练命令会传播评估失败的退出码，训练成功与评估成功须分别检查。已 finalized 的训练 resume 不改写训练产物，但默认会继续未完成的评估；只检查训练恢复时传 `--skip-evaluation`。

```bash
python -m tasks.continual_nav_opd.evaluate_stages --stage-index <local-run>/exports/stages.json --output-dir <local-run>_evaluation --device cuda:0
```

```bash
python -m tasks.continual_nav_opd.evaluate --checkpoint <local-run>/exports/stages.json --policy kb --stage-key pnc/v0/maze_medium/C --split validation --backend subprocess --num-envs 16 --device cuda:0 --output <new-stage-report.json>
python -m tasks.continual_nav_opd.evaluate --checkpoint <local-run>/exports/final.pt --policy kb --split test --backend subprocess --num-envs 16 --device cuda:0 --output <new-final-report.json>
python -m tasks.continual_nav_opd.visualize --checkpoint <local-run>/exports/stages.json --policy active --stage-key pnc/v0/maze_medium/P --task maze_medium --panel-index 0 --device cuda:0 --teacher-diagnostics --output-dir <new-media-dir>
python -m tasks.continual_nav_opd.visualize --checkpoint <run>/exports/final.pt --policy kb --task fourrooms --panel-index 0 --device cuda:0 --output-dir <another-new-media-dir>
```

输出目录/报告必须不存在。阶段索引和直接推理权重不要求训练源码哈希或恢复 checkpoint；复制权重、全部侧车、manifest 并保持相对布局，本地须有对应地图。`--policy active` 用于 P，`--policy kb` 用于 C/F；`--stage-key` 通用选择阶段，评估 JSON 保存真实权重 SHA。不加载教师做普通固定面板评估。全局图仅展示，不进入学生输入，GIF/CSV/reward 对应同一轨迹。

## 异常诊断与边界

判断已提交进度看latest和complete标记，不以未提交事件或残留snapshot为准。KL按有效样本加权；FourRooms真实位移率与转向率辅助发现转圈，动作一致率不能代替成功率。低显存不证明GPU未充分计算；观察采集、学习、教师、环境、Fisher与评估分项时间。

若出现无标签、非有限Loss/梯度、损坏引用或源码不匹配，记录失败证据，不跳过校验、不换教师、不清理历史run。OOM时记录具体槽数/minibatches/microbatch/allocated/reserved；配置改动用新run验收，不能强行恢复。smoke成功只证明工程流程，不证明正式策略有效。停止、删除、推送和远端正式训练单独遵循用户授权。

## E1/E2 执行与计时口径

完整有效观察窗口使用 dense 重放；含 padding 的窗口仍走逐槽掩码路径。CTM 按时间顺序推进并重置完整状态，可学习初始状态仍参与梯度。`DualPolicy.kb_ready` 保留原持久化键，推理读取生命周期内的 Python 缓存；需要变更时调用 `set_kb_ready`，禁止直接写 buffer。加载快照时自动刷新缓存，冻结快照禁止修改。

C 先以完整冻结双列教师采集全部观察与动作，再在任何优化器更新前从独立 KB 的窗口初态重放完整窗口。视觉编码可以合并图片，CTM 顺序不变；重放终态只接续 KB 采集，不使用教师状态或学习重放终态。参数、预算、Adam、精度及采样 RNG 不变。

C 采集的 Encoder 分块固定为 `min(num_envs, encoder_microbatch_images)`，保持旧逐步推理的实际 CNN batch 形状；窗口图片一次上传，特征提前计算后按时间推进 CTM。未采用增大采集 CNN batch 的初版：其浮点误差在完整窗口中放大，隐藏状态超出容差。P/C 学习重放仍使用配置规定的视觉 microbatch。

默认 `--timing-mode events`：CUDA 分项为当前 stream 的事件区间，包括主机提交工作造成的空隙，不代表纯 kernel 时间；CPU 环境/BFS仍为墙钟。窗口结束统一解析事件；动作所需 CPU 下载仍会等待。`--timing-mode synchronized` 可诊断前后同步的墙钟区间，增加开销。事件携带 `timing_mode`；不同模式的分项不直接作为提速比较。性能诊断入口 `profile_training` 固定使用 synchronized，外层窗口/阶段墙钟用于同负载比较。CPU运行两种模式均使用墙钟。

受检源码和任务执行配置改变，旧开发 run 不允许续训。新完整 v4 inference artifact 在哈希配置中保存两任务 ticks，加载时恢复该映射；缺少 `ticks_by_task` 的旧全局 `ticks: 2` 产物明确拒绝，读取旧实验需使用其对应源码版本。不要重写旧 run 的源码哈希绕过检查。

## 按任务执行 ticks

`full.yaml` 中 `ctm.ticks_by_task.maze_medium: 5`、`ctm.ticks_by_task.fourrooms: 2` 是统一来源；remote 和 smoke 继承它。`memory_length: 40` 固定为内部 tick 数。FourRooms与Maze C/F的窗口、预算、Loss未改变；Maze P按独立五步课程协议执行。KB 权重在任务间共享；两列内部逐 tick 对齐，不为任务创建不同 KB。

P 采样及重放、C 双列教师与 KB 学生、F 使用环境任务的 ticks。每个任务报告记录 `ticks` 与 `memory_ticks`；跨任务评估和可视化使用评估任务的 ticks，与 Active 来源任务无关。执行预算记录在本地事件及 W&B run 配置中，不新增 ticks 曲线。直接调用 `step`／`sequence` 必须传 `task=`；不要临时修改 `controller.ticks`，该属性已移除。任务切换使用新的环境和状态，不能将一任务的 live trace 直接移交另一任务。

较长 ticks 会增大递归反传图；50 张观察在 Maze 中展开 250 tick／列。2-tick 或 75-tick 性能与显存数据不能直接用于预测当前 5-tick 负载。正式远端 profile 的显存可行性须在目标 GPU 实测，不能把小预算 smoke 作为大 batch 的保证。

## E3、内存地图与 CTM 编译

正式配置默认 `optimization.e3_cache: true`、`environment.map_cache: memory`。E3 对同一观察、同一列的全部 tick 复用 Attention 的 token/K/V；Q 每 tick 重算。同步衰减权重只在一次序列 forward 内共享，保留梯度，不跨 Adam 更新。缓存不会绕过任务 ticks，教师、Loss、初态、reset 和参数键保持原定义。

Runner 启动时校验并解码完整训练、validation/test 划分，保存连续 `uint8` CPU 数组，各训练槽共享。reset 只读取内存，不再打开 PNG。评估子进程接收解码后的 CPU 图片与起终点，不读取地图文件。缓存不进入 checkpoint；结束或异常退出释放，恢复时重新校验加载。地图加载失败直接报错，不退回磁盘。`map_cache` 事件记录 `maps`、`cache_bytes`、`load_seconds`。独立工具的并行槽通过弱引用池共享缓存，没有全局强引用长期保留图片。

教师模型在阶段入口加载后常驻 GPU，窗口训练不读取权重文件。checkpoint、源码/引用完整性校验和日志仍保留磁盘操作；不能为了省 I/O 跳过这些边界。v4配置只接受memory缓存；验证入口的串行参考复用同一内存地图。

`optimization.ctm_compile` 支持 `disabled`、`default` 和 `reduce-overhead`。基础 `full.yaml` 默认为 `disabled`；与 GPU 型号无关的 `remote_config.yaml` 默认使用 `reduce-overhead`，使用 32 个训练槽、4 个环境 minibatch 及视觉 microbatch 200，预算继承基础配置。只编译 CTM tick 的张量核心，Encoder、Python 校验、状态容器和计时不在编译区域。编译失败直接抛出错误，禁止通过 `suppress_errors=True` 静默切回 eager。编译后的 callable 不附着在模型上，完整双列快照仍拥有独立旧 KB 存储与原参数命名。

编译性能比较必须分开首次编译与热身后的窗口耗时。不同 batch 形状、冻结/反传模式或输入 stride 可能触发重新编译；单个 kernel 编译成功不代表完整 P/C/F 可训练。完整验证入口：

```bash
python -m tasks.continual_nav_opd.verify_smoke --device cuda:0 --ctm-compile default --output-dir <new-directory>
```

Windows 本地编译需要匹配 PyTorch 的 `triton-windows`、`PYTHONUTF8=1`，并把 `TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR` 指向可写目录。本轮测试环境是 PyTorch 2.13/CUDA 12.6 与 triton-windows 3.7.1.post27；不能直接据此断言 Linux RTX5090 的提速。v4推理产物校验当前配置哈希和schema，拒绝旧产物；两任务 ticks 映射是必需字段，不做旧全局 ticks 迁移。训练恢复仍受当前源码与配置校验限制。

## 本地 8 GiB GPU 的32槽入口

在conda的ctm环境及仓库根目录执行以下命令。此入口仍是完整正式预算，不是本轮连续窗口诊断的短预算；只有用户明确要求完整本地训练时才启动。

```powershell
$env:TORCHINDUCTOR_COMPILE_THREADS='1'
python -u -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/local32_8gb.yaml --device cuda:0 --seed 0 --run-dir <new-run-directory>
```

CLI具体参数以 `python -m tasks.continual_nav_opd.train --help` 为准。新profile使用default编译，避免reduce-overhead的CUDA Graph树管理；不修改remote_config。环境序列microbatch=1仍完整重放50观察、按原8槽组累积梯度，每窗口4次更新，不通过减少ticks、窗口长度或增加Adam步数降低显存。配置/源码哈希发生变化时必须创建新run，不绕过旧checkpoint恢复检查。

后台日志应能看到首个完成窗口的 `event=training` JSON，包含环境步、KL、更新次数和耗时；FourRooms P 和 C 另保留动作一致率。Maze P 不计算一致率，也不在地图池边界额外评估；随后按日志间隔及阶段末打印。首窗口含编译开销，因此不能以尚未出现Loss判断进程失败。若长时间无日志，分别检查采集、编译及学习，而不是只看进程存活。

## Maze批量采集验证

`python -m tasks.continual_nav_opd.verify_maze_progress --device cuda:0 --output-dir <新目录>` 加载完整缓存并构建完整训练状态表；比较100批量采集与固定相同动作的逐序列前向参考，核验BFS/PIL观察，记录GPU耗时/峰值显存、CPU RSS与数组占用，并只执行一次小验证Adam。它不是正式训练。

`verify_smoke` 运行两任务两visits的缩小预算，模拟首池中断，核验恢复后不重复已完成池、18次累计Adam、实际有效决策计数、完整Fisher和固定面板。不得根据这些短验证声称策略学习效果或目标远端GPU提速。
