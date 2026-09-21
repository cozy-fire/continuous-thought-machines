# PNN 后续工作证据记录

核查日期：2026-09-10。

## 种子论文
- Progressive Neural Networks，Andrei A. Rusu et al.，2016。
- https://arxiv.org/abs/1606.04671
- 已确认：冻结旧列，新增任务列，通过横向连接复用历史特征；实验包含 Atari 和 3D maze。

## 已核实证据

| 工作 | 原始来源及位置 | 关系与证据限制 |
|---|---|---|
| PNN | https://arxiv.org/html/1606.04671v4 ，§2 / Limitations | 冻结旧列；横向迁移；原文明确指出参数增长和测试时 task label 问题。2022 是修订日期，原文首发 2016。 |
| P&C | https://proceedings.mlr.press/v80/schwarz18a/schwarz18a.pdf ，§3–5、Table 2 | 直接 PNN 后续；active column + knowledge base、蒸馏 + online EWC。Table 2 一轮 PNN 86.50±0.9 / 108,000K，P&C 70.32±3.3 / 659K；五轮 P&C 82.84±1.4。不得将五轮与一轮当相同数据预算比较。 |
| Sim-to-Real | https://proceedings.mlr.press/v78/rusu17a/rusu17a.pdf ，§3、§4.2、Fig.5 | PNN 直接应用。MuJoCo → real Jaco；64×64 RGB；9 个自由度离散速度策略；真实目标到达任务约 60,000 步/4 小时。不是 zero-shot transfer，也不是复杂通用操作。 |
| DEN | https://arxiv.org/pdf/1708.01547 ，§3–4、Fig.3 | 扩展网络改进路线：选择性再训练、按需扩展、分裂、时间戳。CIFAR-100 的指标为 AUROC，且有二分类子任务，不是统一百类准确率。 |
| Learn to Grow | https://proceedings.mlr.press/v97/li19m/li19m.pdf ，§3–4 | 按层搜索 reuse/adaptation/new，复用参数可固定或正则微调；后者不是结构零遗忘。PNN 是对比与讨论对象，不写成原模型直接补丁。 |
| BPNN | https://proceedings.mlr.press/v202/schnaus23a/schnaus23a.pdf ，§2.4、§4.2、Table 1 | 原文明示 Bayesian re-interpretation of PNN。Table 1 平均 BPNN 97.3±1.2，所列较强 PNN 96.7±1.2；不宣称显著性。不是解决模型扩展问题的压缩方法。 |
| Progressive Prompts | https://arxiv.org/html/2301.12314v1 ，§1、§2.3、§3、§5 | 原文明说 inspired by progressive networks。基础模型和旧 prompts 冻结，拼接新 prompt；训练和推理都已知 task identity。ICLR 2023 官方论文目录确认收录。 |
| DER | https://arxiv.org/pdf/2103.16788 ，§1–4；CVPR 2021 官方页面 | 冻结旧特征提取器、新增特征维度，通道掩码剪枝和统一分类器；本次所读全文未找到 Progressive 字符串，不宣称直接继承 PNN，作为结构同类列入。 |
| FOSTER | https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850393.pdf ，§4–5、Table 1 | DER 后续扩展/压缩方向。CIFAR-100 B0 10 steps、2000 exemplars，表内 DER 69.74、FOSTER 72.90。两数相减为 3.16 个百分点，而正文/Improvement 行写 3.06，保留数值冲突，不沿用该提升数字。 |
| MEMO | https://arxiv.org/html/2205.13218v2 ，方法及总存储比较 | 共享通用浅层、扩展专用深层；强调模型与样本总预算。ICLR 2023 的正式论文 PDF： https://openreview.net/pdf?id=S07feAlQHgM 。 |
| ProgLoRA | https://aclanthology.org/2025.findings-acl.143/ 及 PDF | ACL 2025 Findings。新增 LoRA block，任务分配/回忆。未核实 PNN 直接继承，单列现代相邻路线。 |
| PCLR | https://proceedings.iclr.cc/paper_files/paper/2026/hash/5a5acfd0876c940d81619c1dc60e7748-Abstract-Conference.html 及论文 Table 1 | ICLR 2026。LoRA rank pool + compression/integration/learning。LLaVA-1.5-7B / CoIN，Avg.ACC 62.19、Forgetting 3.39；同表 ProgLoRA 59.09 / 7.53。只代表该论文评测，非全领域排名。未核到 PNN 直接继承。 |

## 官方代码入口
- DEN：https://github.com/jaehong31/DEN （README 确认原论文作者和 ICLR 2018）
- BPNN：https://github.com/DLR-RM/BPNN
- Progressive Prompts：https://github.com/arazd/ProgressivePrompts
- DER：https://github.com/Rhyssiyan/DER-ClassIL.pytorch
- FOSTER：https://github.com/G-U-N/ECCV22-FOSTER
- MEMO：https://github.com/wangkiw/ICLR23-MEMO
- ProgLoRA：https://github.com/ku-nlp/ProgLoRA
- PCLR：https://github.com/SII-HITclearlove777/PCLR
- P&C、Sim-to-Real、Learn to Grow：本次未核到作者官方完整实现。不能将第三方复现标为官方。

## 筛选与核查边界
- CPG、PathNet、2026 DYNcPNN 为检索候选，未进入最终详读清单；不声称覆盖所有引用工作。
- 不使用动态引用量和 stars 给论文排名；看方法关系、正式来源、实验内容及代码可用性。
- 研究使用论文原文、会议/出版社记录及作者仓库；未运行算法或复现实验。
- PDF 截图调用仅返回引用项，没有可见图像；数值依据可提取的原文表格和上下文核对，不声称已视觉复核。
