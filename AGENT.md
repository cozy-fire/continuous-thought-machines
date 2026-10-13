# CTM 项目交接规则

本文件维护当前方案、长期边界与入口，不记录实例、PID、进度或ETA。先读用户最新指令，再核对实际代码、解析配置和对应实验记录。

## 当前方案

唯一维护的跨任务方案为 [suffix_dual_expert](tasks/suffix_dual_expert/README.md)：随机单列CTM或视觉LSTM严格前缀课程 → Maze Fisher → 双专家Compress → 独立配对评估。`models/ctm`与`models/lstm`区分学生架构，`models/common`及根训练流程共用；外部FourRooms教师固定为完整CTM Dual。模型、教学Loss、任务状态、预算、checkpoint和EWC消融契约均以该目录代码及说明为准。

当前代码不依赖已删除的旧方案包。上游通用模型和原始任务作为基础组件或参考保留。实验报告与图像位于 [方案报告目录](tasks/suffix_dual_expert/reports/README.md)，旧接口文档位于 [文档归档](docs/archive/README.md)，历史实验harness可能使用独立冻结实现和协议；不得以当前默认配置改写其身份或结果。

## 信息来源

| 信息 | 权威来源 |
|---|---|
| 项目定位和导航 | [根README](README.md) |
| 模型、训练、恢复、评估与验收入口 | [方案说明](tasks/suffix_dual_expert/README.md)及对应代码 |
| 参数与精确预算 | `tasks/suffix_dual_expert/config.py`、所选YAML及 `train --dry-run` 输出 |
| 可视化输入和对照协议 | [可视化说明](tasks/suffix_dual_expert/visualize/README.md)及生成的summary |
| 实验来源和结论 | [实验索引](EXPERIMENTS.md)与各实验protocol、日志、summary、核验记录 |

文档与代码冲突时核对实际解析结果；历史验收不能证明当前训练已完成。硬件profile只改变其明确覆盖的执行参数，不能混用测时、显存或吞吐。

## 执行边界

- Git目标为 `cozy-fire/continuous-thought-machines`；SakanaAI仓库是上游参考。修改前核对remote、branch和status，保留用户改动。Windows安全目录例外只用单条 `git -c safe.directory=<绝对路径>`，不改全局配置。
- 新训练先dry-run核验配置、地图、测试面板、教师和GPU，使用独立run。配置、受检源码或协议变化时不能绕过校验续训。
- 恢复仅用已提交完整边界，核对Adam、课程和窗口计数；推理快照不是恢复checkpoint。完整Dual教师保留其内部旧KB、Active和Adapter，不接入后来更新的KB。F完成边界分支遵循方案的 `--from-fisher` 契约。
- 训练不执行阶段内评估。训练complete与独立评估通过分别核验；可视化为冻结推理和保存特征分析，不更新权重。
- 远端实例与权限从用户当前授权及身份记录取得。凭据不打印或保存。停止、重启、下载、删除和归还按本次授权，不继承历史实例权限。
- 只读监视失败重试一次，区分网络、读取与真实训练失败；同步失败不把旧图当作最新。通知频率按用户要求。
- 本地 `scientific-evidence/`、`runs/`、地图、权重、冻结源码、报告和来源引用可能是复现依赖。清理前核对范围；换端口不能证明资产无用。归档记录不作为当前连接或启动配置。

## 验证与交付

文档改动检查链接和diff；删除旧模块后运行 `tasks.suffix_dual_expert.tests.test_core`，核验dry-run及评估/可视化CLI入口。核心逻辑修改再执行相称的测试和必要小验证，记录实际设备与配置，不用正式训练代替验证。

结论区分源码检查、流程验证、训练完成、固定面板评估和可视化观察。KL、EWC非零或UMAP邻近不能证明闭环成功、知识保留或因果功能。正式结论必须有对应协议要求的退出、summary、预算、配对结果和权重不变证据。

不主动提交或推送。用户当前明确指令优先于历史文档和默认约定。
