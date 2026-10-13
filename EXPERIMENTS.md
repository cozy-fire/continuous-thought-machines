# 实验报告索引

唯一维护实现为 [suffix_dual_expert](tasks/suffix_dual_expert/README.md)。本页仅索引方案目录中的报告和对应图像；历史设置与当前默认配置分开读取。

## 实验报告

| 实验 | 报告 | 对应图像 |
|---|---|---|
| 历史Maze ticks、独立状态课程、on-policy五步与teacher-forcing | [历史报告汇总](tasks/suffix_dual_expert/reports/maze_history/README.md) | 各报告目录中的训练曲线及Word内嵌图 |
| 独立10/15/20步Maze及5→10迁移 | [10步](tasks/suffix_dual_expert/reports/sequence_evaluations/length10_test200/测试报告.md)、[15步](tasks/suffix_dual_expert/reports/sequence_evaluations/length15_test200/测试报告.md)、[20步](tasks/suffix_dual_expert/reports/sequence_evaluations/length20_test200/测试报告.md)、[5→10](tasks/suffix_dual_expert/reports/sequence_evaluations/transfer5_to10_test200/测试报告.md) | [独立序列KL](tasks/suffix_dual_expert/reports/sequence_evaluations/independent_KL_curves.png)、[5→10 KL](tasks/suffix_dual_expert/reports/sequence_evaluations/transfer5_to10_KL_curves.png) |
| 严格前缀10/15/20/25与800图评估 | [实验汇总Word](tasks/suffix_dual_expert/reports/maze_compress/实验汇总报告.docx)、[800图分析](tasks/suffix_dual_expert/reports/prefix/test800/analysis-report.md)、[BFS分层分析](tasks/suffix_dual_expert/reports/prefix/test800/path_length_analysis.md) | [连续KL](tasks/suffix_dual_expert/reports/prefix/prefix_10_15_20_25_KL_curves.png)、[评估图说明](tasks/suffix_dual_expert/reports/prefix/test800/figure-catalog.md) |
| 独立四来源无EWC Compress | [A4000远端结果](tasks/suffix_dual_expert/reports/maze_compress/remote_four_sources/analysis-report.md)、[RTX4060本地15/20复测](tasks/suffix_dual_expert/reports/maze_compress/local_retest15_20/analysis-report.md) | 各报告的figures目录；[训练KL](tasks/suffix_dual_expert/reports/maze_compress/remote_four_sources/KL_curves.png) |
| 旧完整FourRooms教师单任务Compress | [配对结果](tasks/suffix_dual_expert/reports/teacher_compress/completed_analysis.md) | [KL/EWC曲线](tasks/suffix_dual_expert/reports/teacher_compress/KL_EWC_curves.png) |
| 双专家在线Compress | [实验汇总Word](tasks/suffix_dual_expert/reports/maze_compress/实验汇总报告.docx)中的双专家结果 | [两任务KL/EWC曲线](tasks/suffix_dual_expert/reports/dual_teacher_compress/KL_EWC_curves.png) |
| Maze历史填满→FourRooms切换对照 | [结果分析](tasks/suffix_dual_expert/reports/maze_to_fourrooms/analysis-report.md)、[统计附录](tasks/suffix_dual_expert/reports/maze_to_fourrooms/stats-appendix.md) | [图说明](tasks/suffix_dual_expert/reports/maze_to_fourrooms/figure-catalog.md) |

## 工程验证

[初版验收](tasks/suffix_dual_expert/reports/engineering/implementation/verification.md)与[执行优化验收](tasks/suffix_dual_expert/reports/engineering/optimization/verification.md)记录组件验证和测时，不代表正式训练性能或目标远端设备验收。当前入口见[方案说明](tasks/suffix_dual_expert/README.md)和[可视化说明](tasks/suffix_dual_expert/visualize/README.md)。

## 收录范围

报告统一放置于 `tasks/suffix_dual_expert/reports/`，只收录Markdown、Word报告及对应图像。原始日志、指标、轨迹、协议JSON、校验记录、冻结源码、权重和地图保留在本地忽略目录，不纳入提交。被取消的主线不作为完成实验列出，原审计资产仍保留。

报告结论须遵守原设备、来源、面板、预算与统计限制；不同设备的复测不合并，低KL、EWC非零或UMAP邻近不能证明闭环知识保留。新增报告按同一范围收录，不链接本地数据目录。
