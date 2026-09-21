# 交付2：视觉、RoPE、CTM与双列策略

按用户要求串行，仅实现主计划第6节第2项。继续应用karpathy-guidelines、daily-coding、grill-me的先查代码原则及planning-with-files；不创建子agent、不进入训练器实现。

- [x] 核对模型契约、SuperLinear/ResNet实际代码及第1项公共类型。
- [x] 实现视觉、二维RoPE、同步滑窗、CTM和独立/单列/双列策略。
- [x] 验证M01–M08、序列padding/reset、梯度与快照隔离。
- [x] 接真实两环境输入做模型forward/backward验证，回归第1项测试。
- [x] 文档/API更新并交付；不自动提交或推送。

工作区已有未提交.gitignore（/plan/排除）改动，保留。原模型文件不改。

结果：新增15项模型测试，加第1项24项共39项通过。真实Maze/FourRooms各50步、100transitions，CPU float32逐步与重放logits/value/state误差均0；冻结E/KB无梯度且hash不变。停在第2项。
