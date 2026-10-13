# suffix_dual_expert 实现验收

2026-10-11，本地实现与运行验证通过。正式训练未启动，两个旧任务目录未删除。本记录证明实现和恢复流程可运行，不证明新方案的泛化或知识保留效果。

## 已实现协议

随机单列CTM → Maze严格前缀课程 → 同一网络的Maze Fisher → 冻结Maze KB副本与完整FourRooms Dual的联合Compress。无P、任务重访或Compress后F。

默认解析结果：42个Maze阶段、23,500次Maze Adam；F为4,096决策/1,024评分；C为1,280,000决策、每任务640,000、800窗口对/3,200次fresh Adam；独立持续递归测试默认共3,200 episodes。

## 实际检查

| 检查 | 结果与范围 | 证据 |
|---|---|---|
| 核心测试 | 11项通过，0失败、0跳过；六种长度、前缀及初态梯度、终止mask、采样耗尽无更新、联合KL/EWC目标、Fisher、环境/RNG、配对统计及参考推理 | `implementation_integrity.json`；`tasks/suffix_dual_expert/tests/test_core.py` |
| 激活重算 | CPU上所有参数梯度与关闭重算逐位相同；不缩短BPTT | 核心测试 `test_activation_recompute_has_identical_gradients` |
| 模型迁移 | 完整Dual教师和单列CTM的固定多步/episode重置参考输出最大绝对差均为0；两张地图的320个非终止状态BFS标签匹配FIFO参考 | `golden_parity.json`；新包内两份NPZ夹具 |
| 环境迁移 | Maze 100步与FourRooms 1,800步、共8个环境episode；五动作的像素、位置/方向、终止标志一致 | `environment_parity.json` |
| 环境恢复 | FourRooms常规动作恢复和接近300步超时时恢复通过 | 核心测试；`isolated_structure_verified.json` |
| 删除旧包边界 | 临时独立副本只有新包、允许复用的根模型层、独立教师与1,120张真实地图；两个旧包实际不存在，最新源码dry-run成功，默认预算一致 | `isolated_dry_run.json`；`implementation_integrity.json` |
| 真实30步GPU重放 | 320张训练地图缓存、100序列/micro10、完整30步BPTT，一次Adam；encoder/controller/actor梯度均有限非零；峰值预留2,788 MiB | `gpu_validation_v2/replay30.json` |
| 真实联合C窗口 | 每个验证run执行完整16+16槽×50步×2窗口对，共3,200决策/8次Adam；两任务实测梯度非零、EWC每Adam一次且后续非零；两个教师和Fisher不变；峰值预留1,070 MiB | `gpu_validation_v2/verification.json`；direct/resumed的metrics与summary |
| 恢复一致性 | 在Maze池、F及第一个C窗口边界暂停并恢复；下一Maze更新、第二完整C窗口、全部最终学生参数、Adam状态和保留快照与不中断运行逐位一致 | `gpu_validation_v2/verification.json`；direct/resumed checkpoints |
| 独立评估入口 | Maze前/后与FourRooms教师/最终KB各2个episode，共8个；结果及配对统计输出成功，内存权重/快照未更新 | `gpu_validation_v2/direct/evaluation/summary.json` |
| 静态与差异检查 | 新包13个Python文件语法检查通过；AST未发现旧包或历史runtime导入；检查源码、教师、地图面板、夹具SHA及Git差异 | `implementation_integrity.json` |

GPU为NVIDIA GeForce RTX 4060 Laptop GPU，FP32、确定性CUDA、TF32关闭。完整30步验证使用BFS路径检验重放与显存，不是严格筛选后采集的正式训练批次。GPU全流程验证缩小为两个5步池更新、F64决策/2评分、C两个完整窗口；正式默认配置未更改。

首次未启用CNN激活重算的长序列验证因实际显存竞争而严重变慢，已结束该验证进程。当前默认使用非重入CNN激活重算，梯度一致性及完整长序列验证通过；实际吞吐和可用显存仍取决于运行设备和其他GPU程序。

## 产物与使用

实现和命令见仓库 `tasks/suffix_dual_expert/README.md`、唯一配置 `config.yaml`。教师已独立复制到新目录，原SHA保留；来源JSON仅为审计信息，不在运行时定位旧目录。恢复只读已提交边界，推理快照不用于续训。

一次性迁移参考工具保留在本证据目录，不属于新包的运行依赖。临时隔离副本与空的中断验证目录已清理；保留实际验收JSON、诊断run与来源核验记录。现有历史实现、权重和实验结果保持原状。

本次没有执行23,500次Maze正式训练、1,024次正式Fisher评分、800个正式C窗口或3,200次正式测试。正式方案效果需要按同一配置完成训练和独立评估后判断。
