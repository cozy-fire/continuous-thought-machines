# PNC 训练与本地评估分离

实现日期：2026-10-05。本次仅修改并验证代码，没有启动正式训练，没有修改 `v3-opd`。

## 当前契约

- Runner 仅执行 P→C→F，不调用阶段 validation、visit 汇总或最终 test，不创建评估目录、不产生训练评估事件。
- 12 个阶段分别保存推理权重：P 为完整 Dual（当时旧 KB、Active、Adapter），C/F 为当时独立 KB。F 不更新策略，但有单独阶段快照。
- 最终导出 `exports/final.pt` 与带 SHA 的 `exports/stages.json`，保存成功即可提交 `finalized`，不依赖评估。
- 默认 `train` 在上述全部训练结束后启动独立 `evaluate_stages` 进程，先验证12份阶段权重、再测试最终 KB；评估不会插入训练阶段。`--skip-evaluation` 可推迟评估，`--evaluation-dir` 指定训练目录之外的输出。失败不撤销 finalized，独立重试只补做未完成报告。
- 仍保留训练日志、地图/教师身份与恢复 checkpoint。Maze 每池的 Adam/RNG 恢复和 FourRooms/C/F 的阶段恢复保持原规则。
- 新边界类型 `v4_training_boundary` 拒绝旧评估耦合边界。阶段权重和索引核验 SHA、阶段身份、完整配置与固定 manifest。
- 本地评估选择 `--stage-key pnc/v0/maze_medium/C` 等任意 P/C/F。索引或直接 `.pt` 不要求训练源码、恢复 checkpoint、教师权重或 W&B；复制推理权重、侧车、manifest 并保留布局，使用本地对应地图。
- 原固定 validation/test 面板、argmax、连续状态及 serial/subprocess 后端保留。评估 JSON 标识实际阶段与 checkpoint SHA，不更新权重或训练目录。

## 验证

最终 OPD 单元测试 **99 项全部通过**（420.69 秒）。本次新增故障验证覆盖阶段快照导出中断、最终导出中断及不重复 F/地图加载、历史 C/F 权重不被后来 KB 替换、索引身份错配、旧边界拒绝。正式 dry-run 显示 `evaluation_in_training=false` 和 `stage_snapshot_count=12`；训练预算保持原值。

本地 RTX 4060 Laptop GPU、FP32 eager 的两任务/两 visits 验收通过：**12 阶段、18 次 Adam、3137 个有效决策、12 份阶段权重、零训练评估**。首个 Maze 池后模拟中断并恢复，首 P 后再恢复；完成后的 resume 不改写产物。完整 Fisher 覆盖 86 个参数。

训练 finalized 后独立加载并评估全部12阶段，每阶段两个任务各2个固定 validation episode；最终 KB 的 test serial/subprocess 逐 episode 结果一致。两个 v0 Active 的 GIF、CSV 和 PNG 验证通过。离线评估前后，整个训练目录的文件路径、大小与 SHA 清单完全一致。

本次训练验证仍加载完整50000张地图。训练提交完成后，验证驱动交接给独立后处理；后处理共享一次完整核验的真实 manifest，仍校验阶段权重/身份及实际面板地图 SHA，避免各阶段重复扫描全部地图。验收原始记录保存这一交接原因，不把驱动切换当作训练失败。临时训练权重在全部验收后清理，仅留报告、日志、元数据与离线结果；不保留本次验证权重作实验产物。

本次原始证据在忽略目录 `scientific-evidence/pnc_training_only_implementation/`：`unit_tests.log`、`dry_run.json`、`gpu_integration.log`、`gpu_offline_verification.log` 及集成 `report.json` / SHA 清单。工程验证不证明学习效果或提速比例，W&B online 未实测。

## 本地评估命令

在仓库根目录、ctm Python 中执行，输出文件必须不存在：

```bash
python -m tasks.continual_nav_opd.evaluate --checkpoint <local-run>/exports/stages.json --policy kb --stage-key pnc/v0/maze_medium/C --split validation --device cuda:0 --output <new-stage-report.json>
python -m tasks.continual_nav_opd.evaluate --checkpoint <local-run>/exports/stages.json --policy active --stage-key pnc/v0/maze_medium/P --split validation --device cuda:0 --output <new-active-report.json>
python -m tasks.continual_nav_opd.evaluate --checkpoint <local-run>/exports/final.pt --policy kb --split test --device cuda:0 --output <new-final-report.json>
python -m tasks.continual_nav_opd.evaluate_stages --stage-index <local-run>/exports/stages.json --output-dir <local-run>_evaluation --device cuda:0
```

## 默认训练后评估验收（2026-10-06）

新增默认衔接：`train` 在全部训练 finalized 后启动独立 `evaluate_stages` 进程，输出默认在同级 `<run-dir>_evaluation`。阶段内不评估，未完成的 `--max-stages` 不评估；`--skip-evaluation` 只训练。独立入口用带 SHA 的计划和报告续做未完成工作，既有结果身份不匹配或损坏时拒绝复用；评估失败不改变训练 finalized。

OPD 单元测试 **106 项全部通过**（519.38 秒）。新增7项覆盖默认启动顺序、跳过/未完成训练、独立进程失败传播、全部阶段与最终 test、报告中断续做、结果身份/计划校验、manifest 只校验一次、复用同一面板缓存，以及训练和已完成评估文件不改写。正式 dry-run 显示 `evaluation_in_training=false`、`evaluation_after_training=true`，训练预算保持原值。

真实 RTX 4060 Laptop GPU、PyTorch 2.13.0+cu126 上使用518张原始地图的临时副本、正常512张验证划分和两 visits 的 smoke 预算：12阶段、18次 Adam、3149有效训练决策。默认 CLI 已在训练完成后启动独立评估。验证会话在首份报告之后中断，原进程已退出；之后用独立入口复用首份报告、完成其余报告。12份阶段权重的两任务 validation 与最终 KB 的两任务 test 均完成，每任务每面板1 episode（共26 episodes）。独立评估退出码0；再执行默认训练 CLI 的 finalized resume，退出码0，全部训练文件与已完成评估报告的SHA清单不变。

原新训练 CLI 的最终退出码没有取得，不将会话中断报告为训练失败或完整命令退出成功。恢复期间旧验证文件的Python路径校验遇到沙箱访问限制，通过已授权权限通道完成验证，没有修改源代码绕过限制。结果证明默认衔接、独立续做和幂等恢复，不证明正式训练表现或速度。证据在 `scientific-evidence/pnc_post_training_evaluation/`；验证完成后清理临时训练权重、地图副本和一次性驱动，仅留报告、日志、元数据与SHA清单。
