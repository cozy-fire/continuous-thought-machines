# 测试

当前跨任务方案的测试位于 `tasks/suffix_dual_expert/tests/`。从仓库根目录执行：

```powershell
python -m unittest tasks.suffix_dual_expert.tests.test_core -v
python -m tasks.suffix_dual_expert.train --config tasks/suffix_dual_expert/config.yaml --dry-run
python -m tasks.suffix_dual_expert.evaluate --help
python -m tasks.suffix_dual_expert.visualize --help
```

核心测试覆盖严格前缀/BPTT/终止mask、双专家KL与一次EWC、Fisher、配置预算、完整边界/RNG恢复、配对统计和冻结golden夹具。GPU小验证与执行参数测量入口见[方案说明](../tasks/suffix_dual_expert/README.md)。这些验证不是正式训练或总体性能结论。

原有 `tests/tests.py` 和 `tests/test_data.py` 保留为上游CTM通用模型/原始任务测试，独立于当前跨任务方案：

```powershell
python -m pytest tests/tests.py
```

其中golden测试针对原始parity、qamnist及rl任务；不要把其结果作为跨任务训练或Compress保留证据。
