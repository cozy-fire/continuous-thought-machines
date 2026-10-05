# Maze 五步 on-policy 课程实现验收

验证日期：2026-10-05。实现位于 `main` 工作区；没有执行正式训练，也没有修改 `v3-opd`。

## 实现契约

- 仅替换 Maze P：地图池为 10/40/160/640/2560/10240/44488，更新预算为 2000/2000/2000/1500/1500/1500/1500。每次 P 为 12000 次 Adam，两次 visit 为 24000 次。
- 每更新采集 100 条独立五步序列，学生按 softmax 概率抽样动作。起点在当前训练前缀全部非终止可通行位置中均匀采样，五步内不 reset、不 detach；到达目标的动作有效，之后 mask。
- 固定权重采集后，按 20 组、每组 5 条重放。从可学习初态建立梯度路径，按实际有效决策总数归一化 CE 梯度，clip0.5，Adam 一次。
- 每池原子保存完整 Dual、Adam、进度和全部 RNG。恢复跳过已完成池，重做未提交池；只有完整 P 发布供 C 使用的 Active 快照。schema v4 直接拒绝旧产物。
- 全地图只读缓存与紧凑训练状态表共享给 P/C/F 和评估；验证、测试索引与训练表隔离。FourRooms 及 Maze C/F 的采样、窗口、Loss 和预算保持原路径。
- 后续调整：删除 Maze P 的整体和逐步动作一致率统计，保留训练 KL。原实现的一致率复用训练 logits，没有额外前向；地图池边界始终只提交 checkpoint。完整 P/C 阶段末及最终 test 的闭环评估保持原流程。删除统计后的 13 项针对性回归通过，包括梯度/Adam/恢复及评估/日志检查。以下完整集成与 GPU 数值来自删除统计前的测量，不作为本次删减的提速证明。

## 验证结果

OPD 单元测试 **97 项全部通过**。覆盖课程预算、嵌套前缀和均匀起点，BFS/转移/观察等价，终止 mask，概率采样，采集与重放分离，第五步到较早状态和可学习初态的梯度，分组梯度与完整批次参考一致，单次 Adam，课程边界权重/Adam/RNG 恢复及旧 schema 拒绝。新增的有限 logits、非有限 Loss 边界验证了 Adam 不会执行。

两任务、两次 visit 的缩小预算 P→C→F 集成检查通过：**12 阶段、18 次 Adam、3137 个实际有效训练决策**，配置容量为 3152。包含首个 Maze 池完成后的故障注入和恢复、最终快照导出、完成后幂等恢复、serial/subprocess 固定面板一致，以及两任务 GIF/CSV/图像核验。该集成检查后增加的 Loss 有限性保护另经最终单元测试和 GPU 更新复核。

完整数据检查确认 50000 张缓存地图，训练/验证/测试分别为 **44488/512/5000** 张，训练非终止位置 **7118080** 个，划分间 SHA 无交集。真实地图上的 5632 个位置与原 BFS 教师一致，128 张动态观察与原 PIL nearest 输出逐像素一致；小地图测试另逐状态核验了全部五动作与 MazeEnv 的转移。

## 本地 GPU 实测

硬件为 **NVIDIA GeForce RTX 4060 Laptop GPU**，PyTorch 2.13.0+cu126、FP32 eager、E3 开启。100 条序列使用相同起点和固定动作，每轮批量与串行各测 3 次，全部轨迹和 mask 一致。首次测量与最终复核均通过，详细 JSON 保存在验证目录：

| 项目 | 首次测量 | 最终复核 |
|---|---:|---:|
| 批量采集中位耗时 | 0.858 秒 | 0.995 秒 |
| 串行采集中位耗时 | 8.098 秒 | 38.515 秒 |
| 批量采集峰值 allocated 显存 | 369.6 MiB | 369.6 MiB |
| 串行采集峰值 allocated 显存 | 111.6 MiB | 111.6 MiB |
| 一次真实更新的重放耗时 | 4.372 秒 | 12.635 秒 |
| 重放峰值 allocated 显存 | 561.8 MiB | 561.8 MiB |
| 全地图加载 / 状态表构建 | 57.98 / 13.94 秒 | 266.31 / 64.72 秒 |

全地图 RGB 数组为 51.64 MiB，紧凑状态表（含缩放索引）为 232.59 MiB，两者合计 **284.23 MiB**。首次报告的状态表字节数未计入 672 字节缩放索引，最终报告已补齐。最终复核与单元测试部分重叠；两轮耗时差异明显，不据此承诺稳定加速倍数。

两轮真实更新均包含 496 个有效决策，所有 Adam step 均为 1，初始窗口和视觉梯度非零，梯度及权重有限。最终复核的地图加载和状态表构建 RSS 增量分别约 86.19/232.97 MiB；数组字节数与进程 RSS 是不同口径。allocated 包含模型及其临时张量；reserved 受同一进程内 CUDA 缓存分配器影响，不能当作隔离进程之间的显存比较。

这些结果验证工程行为和本机测量，不证明训练收敛、未见地图泛化或其他 GPU 上的吞吐。在线 W&B 未执行；配置、事件和横轴接口已通过单元测试。

## 复现

在仓库根目录、`ctm` Python 环境运行，GPU 检查的输出目录必须不存在：

```powershell
python -m tasks.continual_nav_opd.train --config tasks/continual_nav_opd/configs/full.yaml --dry-run
python -m unittest discover -s tests/continual_nav_opd -v
python -m tasks.continual_nav_opd.verify_maze_progress --device cuda:0 --output-dir scientific-evidence/maze_v4_new_gpu
python -m tasks.continual_nav_opd.verify_smoke --device cuda:0 --output-dir scientific-evidence/maze_v4_new_integration
```

本次工程证据位于 `scientific-evidence/maze_v4_implementation/`：单元测试日志 `unit_tests_final.log`，首次 GPU 测量 `gpu_benchmark/report.json`，最终 GPU 复核 `gpu_benchmark_final/report.json`，集成结果 `integration/report.json`。数据、日志和权重不进入 Git；此文件仅保留实现契约和验收结论。
