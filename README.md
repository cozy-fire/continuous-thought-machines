# CTM跨任务学习

当前唯一维护方案是 [suffix_dual_expert](tasks/suffix_dual_expert/README.md)：随机单列CTM或视觉LSTM → Maze medium严格前缀课程 → Maze Fisher → Maze与FourRooms双专家Compress → 单列KB独立配对评估。两种学生的模型目录分开，训练与Loss流程共用。

基于[CTM上游实现](https://github.com/SakanaAI/continuous-thought-machines)。输入为RGB `3×84×84`，输出为五动作；Maze每决策5个内部tick，FourRooms为2个，CTM历史窗口40。任务仅选择tick数，不作为神经输入或任务专属head。训练中不评估；策略性能、泛化和保留以独立固定面板结果判断。

## 阅读入口

| 目的 | 入口 |
|---|---|
| 接手与操作边界 | [AGENT.md](AGENT.md) |
| 模型、流程、配置、训练、恢复及评估 | [方案说明](tasks/suffix_dual_expert/README.md) |
| 注意力、神经元UMAP及Maze→FourRooms切换对照 | [可视化](tasks/suffix_dual_expert/visualize/README.md) |
| 核心测试与上游测试的边界 | [测试说明](tests/README.md) |
| 实验结果与历史证据定位 | [实验索引](EXPERIMENTS.md) |

## 使用

Python环境安装仓库依赖：`pip install -r requirements.txt`。运行前需按方案说明提供Maze地图、固定测试面板和完整FourRooms教师权重；这些本地资产不随Git克隆取得。

以下命令从仓库根目录运行：

```powershell
# 只读检查配置、资产、课程与精确预算
python -m tasks.suffix_dual_expert.train --config tasks/suffix_dual_expert/configs/ctm.yaml --dry-run

# 新训练；使用独立输出目录
python -m tasks.suffix_dual_expert.train --config tasks/suffix_dual_expert/configs/ctm.yaml --out runs/suffix_dual_expert/ctm/seed0

# 训练complete后单独评估
python -m tasks.suffix_dual_expert.evaluate --run runs/suffix_dual_expert/ctm/seed0

# 可视化；默认权重路径在visualize/config.yaml中配置
python -m tasks.suffix_dual_expert.visualize
```

预算和执行参数以 `tasks/suffix_dual_expert/config.py` 与解析后的YAML为准，不在根文档重复。恢复只使用已提交完整边界；推理快照不能代替恢复checkpoint。

`scientific-evidence/`、`runs/`、数据和权重保留为本地实验资产。实验报告与对应图像集中于 [方案报告目录](tasks/suffix_dual_expert/reports/README.md)；旧接口文档保留在 [历史文档](docs/archive/README.md)，不作为当前配置或执行入口。仓库保留上游CTM通用模型与原始任务参考；当前跨任务维护入口只有 `suffix_dual_expert`。
