# 测试

当前跨任务方案的测试位于 `tasks/suffix_dual_expert/tests/`。从仓库根目录执行：

```powershell
python -m unittest tasks.suffix_dual_expert.tests.test_core tasks.suffix_dual_expert.tests.lstm.test_policy -v
python -m tasks.suffix_dual_expert.train --config tasks/suffix_dual_expert/config.yaml --dry-run
python -m tasks.suffix_dual_expert.evaluate --help
python -m tasks.suffix_dual_expert.visualize --help
```

`tests/common/`测试共享严格前缀/BPTT/终止mask、双专家KL与一次EWC、Fisher、配置预算、完整边界/RNG恢复和配对统计；`tests/ctm/`测试CTM与冻结golden夹具；`tests/lstm/`测试视觉LSTM的hidden query、持续状态、梯度、序列重放和架构身份。旧`test_core`入口保留CTM/common兼容。GPU小验证与执行参数测量入口见[方案说明](../tasks/suffix_dual_expert/README.md)。这些验证不是正式训练或总体性能结论。

原有 `tests/tests.py` 和 `tests/test_data.py` 保留为上游CTM通用模型/原始任务测试，独立于当前跨任务方案：

```powershell
python -m pytest tests/tests.py
```

其中golden测试针对原始parity、qamnist及rl任务；不要把其结果作为跨任务训练或Compress保留证据。
