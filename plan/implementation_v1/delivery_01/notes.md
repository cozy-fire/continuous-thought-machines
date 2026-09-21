# 依据与发现

- 计划依据：00第6节、01第2/7节、02配置表/预算/manifest规则、03 E01–E08。
- 当前源码：MiniGrid 3.1.0 step(9)报错；action6无动作；成功和第300步超时可同时置True，适配器需成功优先。
- RGBImgPartialObsWrapper(tile_size=8)产生56×56局部RGB；最近邻转84×84。
- 本地Maze train/0=45000 PNG，test/0=5000 PNG，抽查19×19与五色编码匹配。
- 原仓库tests/conftest.py导入大量模型与任务，不改旧文件；本轮用unittest discover独立执行。
- 原有未跟踪文件.idea/、AGENTS.md、featurize_release.sh、research/、scientific-evidence/均保留。

## 验证结果

- 24项unittest通过，包含7×7视野内的墙后遮挡不泄漏测试。
- 正式manifest：train44488/validation512/test5000；固定panel200/200，drift32。
- 每环境610个实际transitions，成功0、timeout2；本轮验证接口，不评估学习效果。
- Python3.12.14、NumPy2.5.3、torch2.13.0+cu126、Gymnasium1.3.0、MiniGrid3.1.0、Pillow12.3.0、PyYAML6.0.3。
- schema1初次落地；补足factory keyword配置参数、Python lambda_字段名、训练seed与eval布局seed区别、render返回HWC。未改变动作、输入、奖励或算法决策。
