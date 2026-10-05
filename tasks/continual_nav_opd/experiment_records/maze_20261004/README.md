# Maze 实验报告与结论

|实验|主要结论|
|---|---|
|[不同ticks配置](01_ticks_sweep/README.md)|5 ticks可出现拟合信号，ticks与效果不单调|
|[独立状态T5扩增](02_independent_T5_curriculum/README.md)|训练一致率99.39%，未见状态69.5%，完整rollout成功率16%/15.5%|
|[On-policy五步序列](03_sequence5_T5_curriculum/README.md)|未见200图重置/持续成功率17%/16%，动作一致率49.35%/27.21%|
|[Teacher-forcing五步序列](04_teacher_forcing_sequence5_T5_curriculum/README.md)|训练五步一致率99.13%，未见200图成功率16.5%/13.5%，动作一致率50.30%/13.01%|

所有实验已完成。本目录仅保留Markdown及Word报告和结论；日志、原始指标、轨迹、图像、动画和校验清单已移除。Word内嵌图表属于报告内容。原始实验与本地备份位于被Git忽略的scientific-evidence，本次清理范围仅为experiment_records。

单seed、预算和地图池不同；训练一致率不能替代闭环成功率，也不能据此宣称某训练方式普遍更优。
