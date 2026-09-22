# 交付5：阶段调度、checkpoint、恢复、评估和条件基线

仅实现交付5，不自动提交推送，不启动full训练；交付6的完整760步smoke另行执行。

- [x] 四种method的确定性阶段展开、独立RNG和预算。
- [x] 阶段边界原子checkpoint、来源/数据/replay校验、模型重建和视觉导出。
- [x] 固定面板独立评估、指标、W同轨迹漂移、阶段评估事件。
- [x] 单阶段执行衔接、train/evaluate CLI、条件基线视觉来源与成本。
- [x] 边界恢复/失败/单列基线/评估隔离测试，有限真实环境验证。
- [x] 接口文档、交付报告及计划状态。

依照已读取的karpathy-guidelines、daily-coding、grill-me、planning-with-files执行；先查代码可回答的问题。参数、损失和预算保持02协议。

完成记录：72项单元/边界测试通过；真实W.collect/W.fit后新进程恢复X/C，120训练transitions，自动eval 8400与独立CLI eval 1200；产物核验通过。下一阶段F尚未执行，完整760步smoke仍属交付6。详见[交付报告](交付报告.md)。
