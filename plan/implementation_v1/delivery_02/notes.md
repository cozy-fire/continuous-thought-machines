# 核查与实现约定

- 本地SuperLinear实际einsum为BDM,MHD->BDH，保留每神经元权重/bias及除以可学习T的语义。
- ResNet使用本地prepare_resnet_backbone('resnet34-2')，无额外输入通道转换。
- sequence接口显式接收fmap序列与episode_start/valid_mask；不隐式detach时序图。图像编码由外层每观察一次完成。
- 模型层只提供视觉冻结、只读快照等局部接口；全局phase状态机和优化器重置属于交付5。

## 证据

- 单模型测试15项通过；全套39项通过。首次测试进程在用户继续消息后session ID失效，重新运行并持久保存日志，不将丢失会话视为成功依据。
- `scientific-evidence/continual_nav/delivery_02_20260921/all_tests.log`记录完整回归。
- `model_report.json`记录真实像素50步/任务，CPU float32，三类最大差异均0。
- 真实输入action_sync.decay梯度L1=2.184026698159869e-6，out_sync.decay=0.10626165568828583；所有约定Active分组和Adapter有有限非零梯度。
- 视觉与KB的完整state_dict hash在forward/backward前后相同；合成测试还验证一次Active optimizer.step后冻结tensor不变。
- 无GPU验证、无正式训练或学习效果声明。
