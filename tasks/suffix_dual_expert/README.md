# suffix_dual_expert

单列CTM随机初始化 → Maze严格前缀/末5步教学 → Maze Fisher → 双专家Compress → 单列最终KB。每阶段只执行一次，训练内不评估，Compress后不新增F。

本目录是唯一维护的跨任务实现。只复用根目录 `models.resnet`、`models.modules` 的通用层（及其 `models.utils` 依赖），其余依赖为PyTorch、NumPy、Pillow、PyYAML、Gymnasium和MiniGrid。训练、恢复、评估和可视化均使用本目录接口。

## 输入与模型

学生接受 `uint8 [B,3,84,84]`，内部除以255，输出五动作logits和显式CTM递归状态。ResNet34-2/GroupNorm32、D512、输入特征D128、M40、NLM hidden16、四头空间RoPE；Maze每决策5 ticks、FourRooms每决策2 ticks。任务只确定ticks，不作为神经输入或任务专属head。

Maze medium为19×19全图，蓝色答案路径在观察前擦除，红色为当前agent、绿色为goal；动作0..4为上/下/左/右/等待。FourRooms为原MiniGrid局部RGB视野，tile8后最近邻缩放至84×84；动作0/1/2为转左/转右/前进，3/4映射native no-op6。每episode上限300决策。

## 配置与资产

唯一默认配置为 [config.yaml](config.yaml)，未列字段使用 `Config` 默认值；`--dry-run` 输出解析后配置、完整课程与精确预算。小验证通过修改同一配置中的显式预算字段完成，不改变正式默认值。

| 阶段 | 默认预算 |
|---|---|
| Maze长度 | 5/10/15/20/25/30；每长度地图池5/10/20/40/80/160/320 |
| 5步 | 前两池各2000，其余五池各1000，合计9000 Adam |
| 其余每长度 | 前两池各200，其余五池各500，每长度2900 Adam |
| Maze合计 | 42个池边界、23500 Adam，同一Adam跨所有池/长度保留 |
| Maze F | 4096在线决策、1024个不重复score，无Adam |
| Compress | 16+16槽×50决策×800窗口对，1280000决策、每任务640000；3200 fresh Adam |
| 独立评估 | Maze前后各800、FourRooms教师/最终KB各800，共3200 episodes，仅持续递归 |

默认FP32、禁TF32、确定性CUDA、compile关闭、seed0；Adam lr1e-4、betas0.9/0.999、eps1e-5、weight_decay0，统一clip0.5。Maze与Compress的Adam计数分别记录。

默认保留 `checkpoint_encoder: true`，Maze默认microbatch25，Compress默认microbatch4；有效batch、Loss归一化、统一clip和Adam次数不变。整段观察先按 `encoder_chunk: 32` 批量编码，再顺序执行CTM；不会detach前缀或缩短BPTT。编码器使用GroupNorm且无随机层，批量执行及激活重算保留同一数学目标，但浮点累积顺序不同，不能保证与旧microbatch逐位相同。

显存较少时可显式降低 `suffix_microbatch`（须整除100）和 `compress_microbatch`（须整除每任务四条的组）。显存充足时可在新run显式关闭 `checkpoint_encoder`。本地50步Compress测时未显示关闭重算的稳定收益，因此默认保留较低显存组合。配置在启动时解析并绑定checkpoint；训练中不自动试探OOM或更改批次。

[config.a4000.yaml](config.a4000.yaml)仅覆盖执行参数，继承相同默认课程与预算：Maze microbatch20且关闭重算。它是16GiB设备的候选配置，30步峰值约12.3GiB为本地micro5的线性显存外推，尚未在A4000实测；不能把配置文件存在当作目标GPU验收通过。只在新run选择，已有run恢复保持原配置与源码身份。

运行需提供三类输入资产，Git源码不包含被忽略的教师权重和地图：

- `data/mazes/medium/`：训练集按数值路径排序、SHA去重，取前320张组成嵌套池。
- `assets/maze_test800.json`：固定未见800图面板的顺序和SHA，启动检查所有测试图与训练图无重叠。
- `assets/fourrooms_teacher.pt`：独立的完整v0 Dual文件，SHA256 `d76c748579b2c1eeaa089ae0d41570d091490d355d970cb7628c08dbc5f0461c`。含普通配置与张量，用 `weights_only=True` 加载，保留内部旧KB、Active、Adapter。不接入新学生KB。

当前本地资产已复制到本目录；`assets/provenance.json`仅记录来源和SHA，不用于定位历史目录。更换存放位置通过配置显式指定；不得把旧模型类或旧环境对象写入新checkpoint。

## 教学与Compress

每Maze更新固定权重采集100条接受序列，候选批100/上限100000，起点均匀覆盖当前池非终止可通行状态。学生Argmax真实推进；前L−5步严格匹配BFS单标签（平局顺序上/下/左/右），前缀结束于goal的候选剔除。末5步有效CE按全批有效决策归一化，microbatch25累积梯度，统一clip后一次Adam。前缀和可学习初态保持梯度连接，序列间恢复初态；5步无筛选前缀。候选耗尽为真实失败，不自动放宽或重启。

Maze训练后的同一学生直接成为KB。F用Softmax动作采样，按独立真实递归窗口为每个选定动作重放log-prob；先平方单个score的梯度再平均，完整覆盖encoder/controller/actor。F不改策略，只保存Fisher和theta-star。

Compress分别冻结Maze KB副本与完整FourRooms Dual。两个专家分别Softmax在线驱动各自环境，教师与学生、两任务之间均维护独立递归状态。每次窗口采集后把每任务16条连续序列打乱槽顺序分四组，每组各四条序列、microbatch4，不打乱时间步；允许显式使用microbatch1/2/4。每个Adam目标为0.5×Maze平均KL + 0.5×FourRooms平均KL + 250/2×Fisher加权参数偏移平方；EWC仅一次，所有项共同clip后更新。窗口内完整BPTT、窗口间截断，episode边界恢复可学习初态。Fisher与两个教师固定不变。

EWC默认启用，开关为 `ewc_enabled: true`。在所用YAML中设 `ewc_enabled: false`，Compress只优化两任务各0.5的KL，不计算EWC或其梯度、不把Fisher上传GPU；日志显示 `ewc=0`、`ewc_applications=0`。`ewc_lambda`保留为启用时的系数，默认250。关闭EWC仍执行原Maze F并保存Fisher，训练预算和边界保持不变。开关写入解析配置、协议、指标和summary，恢复须保持相同开关。

## 命令与恢复

从仓库根目录运行：

```powershell
& D:/conda/envs/ctm/python.exe -X utf8 -m tasks.suffix_dual_expert.train --config tasks/suffix_dual_expert/config.yaml --dry-run
& D:/conda/envs/ctm/python.exe -X utf8 -m tasks.suffix_dual_expert.train --config tasks/suffix_dual_expert/config.yaml --out runs/suffix_dual_expert_seed0
& D:/conda/envs/ctm/python.exe -X utf8 -m tasks.suffix_dual_expert.train --config tasks/suffix_dual_expert/config.yaml --out runs/suffix_dual_expert_seed0 --resume
& D:/conda/envs/ctm/python.exe -X utf8 -m tasks.suffix_dual_expert.evaluate --run runs/suffix_dual_expert_seed0
```

训练创建新run，不覆盖旧目录；恢复必须显式 `--resume`。可用 `--stop-after-boundaries N` 在第N个新提交边界停下，供恢复验证使用；它不改变协议预算。

Maze只从完整地图池边界恢复；F只从其开始/完成边界恢复；Compress每10个完整窗口及最终窗口提交。保存学生、阶段计数、当前Adam、完整RNG、Fisher及两任务的环境/学生/专家递归状态。环境保存seed、位置、方向、步数和RNG等普通状态，不pickle对象。新恢复文件SHA与latest指针原子提交后才删除上一恢复文件；未提交更新不计入恢复后的指标日志。六个长度最终权重、F边界KB和最终KB单独保留。

恢复要求配置、受检源码、数据面板与教师身份完全匹配，且Adam步数、已完成课程与窗口计数一致。推理快照不能用于续训。

### 同一F完成边界的EWC消融

F阶段不受EWC开关影响，计算、预算、RNG及保存方式不变。使用现有完整边界暂停父run：`--stop-after-boundaries N`，N为配置的Maze阶段数加1（当前默认42+1=43）。父run此时必须为 `phase=compress, windows=0`，尚未有C优化器或采集器状态。

```powershell
& D:/conda/envs/ctm/python.exe -X utf8 -m tasks.suffix_dual_expert.train --config tasks/suffix_dual_expert/config.yaml --out runs/maze_f_source --stop-after-boundaries 43
& D:/conda/envs/ctm/python.exe -X utf8 -m tasks.suffix_dual_expert.train --config tasks/suffix_dual_expert/config.yaml --from-fisher runs/maze_f_source --out runs/compress_ewc_on --ewc on
& D:/conda/envs/ctm/python.exe -X utf8 -m tasks.suffix_dual_expert.train --config tasks/suffix_dual_expert/config.yaml --from-fisher runs/maze_f_source --out runs/compress_ewc_off --ewc off
```

`--from-fisher`只新建Compress分支，不重新执行Maze或F。两组复制同一KB、Fisher、RNG和既有推理快照，各自使用fresh Adam、窗口从0开始。仅允许 `ewc_enabled`不同；系数、其余配置、源码、地图、教师及依赖身份仍须匹配。父run不被修改；两份 `compress_origin.json`、protocol和summary绑定相同来源边界SHA，并明确Maze/F为复用预算。

父run保持停在F完成边界，直到两组分支创建；原latest保存机制会删除被后续边界取代的恢复文件，本次不改变F保存方式。不能从C中途或最终边界使用此入口，也不能用单个推理权重代替完整F边界。加 `--dry-run`可只读核验来源，不创建分支。

分支创建后独立续训：用自己的目录、原配置与原EWC开关执行 `--resume`，不再传 `--from-fisher`。例如关闭组使用 `--out runs/compress_ewc_off --resume --ewc off`。普通resume仍不允许切换开关。使用其他配置时，父run与两个分支必须选择同一份配置，并据其实际Maze阶段数设置暂停点。

## 独立测试

只在训练 `summary.json` complete后运行，所有模型eval/inference模式，无Adam、前缀筛选或周期恢复。Maze前后使用同800图/起点/顺序，FourRooms使用同seeds3000000..3000799，Argmax/max300/FP32/batch32。输出逐episode结果、动作轨迹、Wilson区间、差pp、仅前/后成功数、配对bootstrap和两项McNemar-Holm；核验内存权重和快照文件未变化。

单独复测某个长度边界：

```powershell
& D:/conda/envs/ctm/python.exe -X utf8 -m tasks.suffix_dual_expert.evaluate --checkpoint runs/suffix_dual_expert_seed0/snapshots/maze_length25.pt --out runs/suffix_dual_expert_seed0/maze25_test800
```

`--episodes N`仅用于更小诊断子集，默认始终800，不能把小验证结果作为策略性能。低KL、非零EWC、流程通过或不显著差异不证明泛化、保留或等价。

## 验证入口

```powershell
& D:/conda/envs/ctm/python.exe -X utf8 -m unittest tasks.suffix_dual_expert.tests.test_core -v
& D:/conda/envs/ctm/python.exe -X utf8 -m tasks.suffix_dual_expert.tests.verify_gpu --out scientific-evidence/suffix_dual_expert_local_check
```

核心测试与GPU验证都屏蔽旧包导入。迁移golden夹具已经在冻结参考版本生成，后续验证只读取夹具和独立教师资产。GPU验证只运行明确的小验证预算，检查真实320图缓存下100序列/配置microbatch完整30步重放，以及完整16+16槽50步窗口、两任务非零梯度、EWC调用、教师/Fisher固定、下一窗口与Adam逐位恢复和8个诊断episode。它不启动正式训练。

执行参数测量入口（固定轨迹/目标，不采集正式训练数据）：

```powershell
& D:/conda/envs/ctm/python.exe -X utf8 -m tasks.suffix_dual_expert.tests.benchmark_replay --out scientific-evidence/suffix_replay_benchmark
& D:/conda/envs/ctm/python.exe -X utf8 -m tasks.suffix_dual_expert.tests.benchmark_compress --out scientific-evidence/suffix_compress_benchmark
```

Replay测量包含320图缓存、30步/100序列、前向与反向，clip/Adam单独计时；Compress测量完整50步/16+16槽学生更新，排除教师与环境采集。两者均验证全部参数梯度和Adam结果的浮点容差，不能把组件加速比当作整段训练加速比。实际优化证据见[优化验收记录](reports/engineering/optimization/verification.md)。

2026-10-11初版验收通过：11项核心测试、冻结参考模型/环境比对、两个旧包实际不存在的隔离副本dry-run，以及完整30步/C窗口/恢复与8个诊断episode。初版源码证据见[原验收记录](reports/engineering/implementation/verification.md)。同日执行优化后重新通过13项核心测试、完整30步/两个真实C窗口/逐位恢复及8个诊断episode；当前micro25/4、保留重算，30步/C峰值预留5366/2204MiB。组件测时中Maze完整重放约1.42倍、C学生更新约3.10倍，设备为RTX4060 Laptop，不是A4000或完整训练加速比。详见[当前优化验收记录](reports/engineering/optimization/verification.md)。正式预算训练及完整测试尚未执行。

## 可视化

注意力热力图、神经元联合UMAP、按动作汇总，以及Maze历史窗口填满后切换FourRooms的配对对照，统一放在 [visualize/](visualize/README.md)。默认权重路径由其独立 `config.yaml` 指定；两种模式共享代码，均为冻结推理。
