# 实现依据

- PPO：按环境分组，不打散时间；起点state detach；超时使用真实final frame与未reset trace bootstrap；GAE在成功/超时均截断。
- C：冻结teacher全动作log-probs；学生独立initial state，10观察无梯度burn-in，窗口learning目标只使用一次。teacher collect trace跨batch延续。
- F：KB自身采样，无reward/critic；独立RNG从所有真实transition无放回选点；梯度逐样本平方平均。参数名/shape严格匹配。
- EWC：250/2乘加权平方和；F_new=phase_decay*F_old+F_current；center更新为当前KB参数副本。

- API落实：F采集/选点、当前F估计、Online EWC状态生成拆为3个函数；01契约和README已同步。PPOBatch新增可选真实next图像尾字段，旧人工batch可省略，无持久checkpoint格式更改。
- 验证完成：62项unit tests；真实Maze/FourRooms X40/C40/F20及单列P40。learning_report和unit_tests日志保存于scientific-evidence/continual_nav/delivery_04_20260921。
- 正式模型效果/GPU/full预算尚未验证；当前probe的world随机冻结，完整W→X→C→F调度仍由后续交付实现。
