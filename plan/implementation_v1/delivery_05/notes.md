# 交付5实现记录

- 02 §§9–11规定四类方法、模型初始随机流、全量成本和阶段边界恢复。
- W.collect与W.fit为独立阶段；X边界必须保存旧KB和训练后Active依赖，C之后才更新学生KB。
- 单列基线没有虚构KB/world/Fisher；vision artifact须匹配seed和架构，报告共享TA来源成本。
- 恢复以原子完成标记与checkpoint哈希为准，不能使用半写文件或猜日志。各阶段起点重置环境，因此无需保存Gym内部state。
- 评估不消耗训练RNG；所有评估使用固定manifest面板，成功长度无样本时null。
- C开始前必须将teacher的旧KB与live学生KB断开；否则阶段所有权检查会发现同一参数既被当作冻结教师又被当作学生训练。
- 控制器tick hooks使用WeakSet记住对象，不能用可能复用的Python id；评估副本记录到eval_internal_ticks。
- 视觉artifact复制原始字节，不能重新torch.save后声称保留原文件hash。
- final-test/导出完成标志写入checkpoint；文件存在本身不足以判定finalize已提交。
- 恢复验证源码hash，因此测试或真实run正在执行时不能修改生产源码；本次最终72项测试在源码稳定后完整通过。
- JSONL是尝试历史，checkpoint才是正式状态；突然断电下的最后vector部分交互与中断评估成本不承诺逐步精确记录。
