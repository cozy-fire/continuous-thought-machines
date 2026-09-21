# Progressive Neural Networks 后续工作调研

检索与核查日期：2026-09-10。种子论文：[Progressive Neural Networks，Rusu et al.，2016](https://arxiv.org/abs/1606.04671)。

## 结论与阅读优先级

如果希望理解“PNN 后来发展成了什么”，优先阅读 **Progress & Compress、Sim-to-Real Robot Learning、Learn to Grow、Bayesian Progressive Neural Networks、Progressive Prompts**。它们分别回答容量增长、真实机器人迁移、结构选择、不确定性与语言模型参数效率问题。

如果要做视觉类增量实验，再读 **DER、FOSTER、MEMO**。如果关注当代多模态模型，再补 **ProgLoRA、PCLR**；后两篇在机制上相邻，但本次未核到其直接继承 PNN 的证据。

“比较好”按方法关系清楚、问题有价值、正式发表与实验依据、代码可用性筛选，不代表所有论文在同一基准下的性能排名。本报告精选 9 篇核心/结构相关工作，另加 2 篇现代延伸；不是完整引用清单。

## 1. 原论文留下的核心问题

PNN 为新任务增加网络列，冻结旧列，并通过横向连接读取过去的特征。这样可以保住旧任务的既有计算路径，并让新任务利用旧知识。原文 §2 的 Limitations 已指出两个问题：**参数随任务增长；测试时选择哪一列需要任务标签**。旧列冻结也意味着新任务学到的知识不会自动改善旧列。[原文 §2](https://arxiv.org/html/1606.04671v4)

后续文献可以按下列问题理解：

| 研究问题 | 代表工作 | 与 PNN 的关系 |
|---|---|---|
| 能不能保留迁移能力，同时固定容量？ | Progress & Compress | 直接延续并改造 |
| 能不能把仿真学到的技能转移到实体机器人？ | Sim-to-Real Robot Learning | 原作者直接应用 |
| 到底需要增加多少、哪些参数？ | DEN、Learn to Grow | 明确讨论 PNN 局限的结构扩展路线 |
| 能不能表示知识的不确定性？ | BPNN | 原文明示贝叶斯重解释 |
| 大语言模型能不能只扩展少量参数？ | Progressive Prompts | 原文明示受 PNN 启发 |
| 不提供任务标签，如何统一识别所有类别？ | DER、FOSTER、MEMO | 类增量学习中的结构相关路线，非原始 PNN 的直接替换 |
| 多模态 LoRA 扩展后如何控制容量？ | ProgLoRA、PCLR | 机制相邻的现代延伸，非已证实的直接传承 |

## 2. 最值得读的直接后续与结构改进

### 2.1 Progress & Compress: A Scalable Framework for Continual Learning

**Schwarz et al.，ICML 2018。优先级最高。**

保留一列学习新任务的 active column 和一个 knowledge base：先借助旧知识学新任务，再把新能力蒸馏进知识库，并用 online EWC 保护已有参数。其价值是把持续扩列改成固定容量的“学习—巩固”循环。实验覆盖 Omniglot、Atari 和 3D 迷宫。

它不是无代价地超过 PNN：Table 2 的 Omniglot 单轮结果为 PNN **86.50% / 108M 参数**、P&C **70.32% / 659K 参数**。这是明显的容量与保留性能权衡；P&C 也不再有原 PNN 的严格结构零遗忘保证。论文另报五轮 P&C 82.84%，不能与单轮结果视作相同训练预算。

[正式论文](https://proceedings.mlr.press/v80/schwarz18a.html) · [方法和实验 PDF，§3–5、Table 2](https://proceedings.mlr.press/v80/schwarz18a/schwarz18a.pdf)

### 2.2 Sim-to-Real Robot Learning from Pixels with Progressive Nets

**Rusu et al.，CoRL 2017；预印本 2016。机器人方向优先读。**

先在 MuJoCo 训练一列视觉控制网络，再冻结它，为真实 Jaco 机械臂增加新列，通过横向连接复用仿真特征。输入为 RGB 图像，控制输出为各自由度的离散速度选择。

§4.2 的真实目标到达实验每次约 **60,000 步、4 小时**，progressive 方法优于文中直接微调和从头训练对照。它有真实硬件证据，但需要真机继续训练；任务是相对简单的 reaching，不是零样本迁移或通用复杂操作。

[正式论文](https://proceedings.mlr.press/v78/rusu17a.html) · [PDF，§3–4.2、Fig.5](https://proceedings.mlr.press/v78/rusu17a/rusu17a.pdf)

### 2.3 Lifelong Learning with Dynamically Expandable Networks（DEN）

**Yoon et al.，ICLR 2018。理解“按需扩容”的经典。**

新任务先尝试选择性再训练；能力不足时增加必要神经元，并通过分裂/复制和时间戳减轻旧单元的语义漂移。它针对 PNN 每任务都分配整列的问题，改为更细粒度的共享与增长。

实验覆盖 MNIST 变体、CIFAR-100 和 AWA，并报告容量曲线。需要注意其二分类子任务和 **AUROC** 评测，不能把数值直接当成现代 CIFAR-100 统一分类准确率。阈值、正则强度及扩展规则也增加了实现复杂度。

[论文](https://arxiv.org/abs/1708.01547) · [原文 §3–4、Fig.3](https://arxiv.org/pdf/1708.01547) · [作者代码](https://github.com/jaehong31/DEN)

### 2.4 Learn to Grow: A Continual Structure Learning Framework for Overcoming Catastrophic Forgetting

**Li et al.，ICML 2019。适合研究模块如何复用。**

对每层搜索三种选择：**复用旧层、增加适配器、新建层**，把“新任务应该借用哪些结构”变为可学习决策。实验包括 Permuted MNIST、Split CIFAR-100 和 Visual Domain Decathlon，并分析搜索到的结构。

其可取之处是给出任务关系对应的结构解释，而不只是决定模型宽度。代价是额外结构搜索；复用层既可冻结，也可正则微调，后者不能称为严格零遗忘。[正式论文](https://proceedings.mlr.press/v97/li19m.html) · [原文 §3–4](https://proceedings.mlr.press/v97/li19m/li19m.pdf)

### 2.5 Learning Expressive Priors for Generalization and Uncertainty Estimation in Neural Networks（含 BPNN）

**Schnaus et al.，ICML 2023。直接扩展 PNN 的概率建模工作。**

论文 §2.4 提出 Bayesian Progressive Neural Networks：保留 PNN 结构，同时为权重建立后验，并把学得的分布作为后续学习的先验，结合 Laplace 近似和 PAC-Bayes 目标。

它研究的是小数据泛化和不确定性，而不是解决扩列的容量成本。Table 1 的持续物体识别结果中，BPNN 平均准确率 **97.3±1.2**，较强 PNN 对照为 **96.7±1.2**；不能仅凭该差值宣称统计显著。它适合“模型何时不确定、旧知识该如何作为先验”的问题。

[正式论文](https://proceedings.mlr.press/v202/schnaus23a.html) · [PDF，§2.4、§4.2](https://proceedings.mlr.press/v202/schnaus23a/schnaus23a.pdf) · [官方代码](https://github.com/DLR-RM/BPNN)

### 2.6 Progressive Prompts: Continual Learning for Language Models

**Razdaibiedina et al.，ICLR 2023。PNN 向语言模型迁移最清楚的一篇。**

原文明确写明受 progressive networks 启发：冻结语言模型和旧 soft prompts，新任务只训练新 prompt，并把它与历史 prompts 拼接。扩展单位从完整网络变为一段可训练输入。实验使用 BERT/T5，包括 15 个文本分类任务的长序列。

关键边界：§2.3 明确假设训练和推理时都知道任务身份；prompt 长度也会随任务增加。因此它不能直接证明无需路由的通用终身语言模型已经实现。

[论文及原文 §1–3、§5](https://arxiv.org/html/2301.12314v1) · [ICLR 2023 官方目录](https://iclr.cc/virtual/2023/papers.html) · [官方代码](https://github.com/arazd/ProgressivePrompts)

## 3. 做视觉类增量学习时值得读的三篇

这一组在“保留旧结构、增加新表示”上与 PNN 相通，但评价问题不同：模型需要在所有已见类别中统一预测，且使用历史样本缓存。尤其是 DER，本次阅读的全文没有核实到对 PNN 的直接方法继承，不能画成确定的作者传承链。

| 工作 | 主要改进 | 为什么值得读 | 主要边界 |
|---|---|---|---|
| **DER，Yan et al.，CVPR 2021** | 冻结旧提取器，增加新特征，配合剪枝和分类器训练 | 是动态表征扩展在类增量学习中的重要代表；实验含 CIFAR-100 与 ImageNet | 仍有模型增长与旧类/新类分类偏置；这里的 DER 不是 Dark Experience Replay |
| **FOSTER，Wang et al.，ECCV 2022** | 新增模块拟合残差，再蒸馏回单 backbone | 直接研究扩展方法的成本，提供“先增加可塑性、再压缩”的实现 | 保留单 backbone 不等于训练峰值显存恒定，也不等于无历史样本 |
| **MEMO，Zhou et al.，ICLR 2023** | 共享通用浅层，只扩展专用深层 | 不仅改架构，还把模型大小与 exemplar 存储纳入统一预算 | 更节省而非固定容量；后部模块仍会增加 |

来源与代码：

- DER：[CVPR 论文](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_DER_Dynamically_Expandable_Representation_for_Class_Incremental_Learning_CVPR_2021_paper.html) · [官方代码](https://github.com/Rhyssiyan/DER-ClassIL.pytorch)
- FOSTER：[ECCV 论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850393.pdf) · [官方代码](https://github.com/G-U-N/ECCV22-FOSTER)
- MEMO：[论文](https://arxiv.org/abs/2205.13218) · [ICLR 正式 PDF](https://openreview.net/pdf?id=S07feAlQHgM) · [官方代码](https://github.com/wangkiw/ICLR23-MEMO)

FOSTER 的 Table 1 给出同一设置下的例子：CIFAR-100、B0、10 个增量阶段、2000 个缓存样本，DER 平均增量准确率 **69.74%**，FOSTER **72.90%**；后者每阶段结束仅保留一个 backbone。原文所写提升幅度与这两个表内数值相减不一致，故这里只保留原始数值，不转述提升幅度。[Table 1 与 §5.1](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850393.pdf)

## 4. 2025—2026 年现代延伸阅读

以下两篇处理同类“扩展、复用、压缩”问题，但本次未核到它们直接基于 2016 PNN 的证据。

**ProgLoRA：Progressive LoRA for Multimodal Continual Instruction Tuning，Yu et al.，ACL 2025 Findings。** 每个新任务增加 LoRA block，通过 task-aware allocation 和 task recall 利用已有能力。它是观察逐任务结构扩展如何应用于多模态指令微调的合适入口；正式归属是 Findings，不能标成 ACL 主会论文。[论文](https://aclanthology.org/2025.findings-acl.143/) · [官方代码](https://github.com/ku-nlp/ProgLoRA)

**PCLR：Progressively Compressed LoRA for Multimodal Continual Instruction Tuning，Meng et al.，ICLR 2026。** 把 LoRA 拆为 rank 向量池，并循环压缩旧参数、整合相似知识、利用释放容量学习新任务。它比简单增加专家更贴近固定预算问题。论文 Table 1 在 LLaVA-1.5-7B / CoIN 下报告平均准确率 **62.19**、遗忘指标 **3.39**，同表 ProgLoRA 为 **59.09 / 7.53**。这只是该论文设置下的对照结果。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5a5acfd0876c940d81619c1dc60e7748-Abstract-Conference.html) · [官方代码](https://github.com/SII-HITclearlove777/PCLR)

## 5. 阅读顺序与比较规则

- **理解原论文后续：** PNN → P&C → Learn to Grow → BPNN → Progressive Prompts。
- **机器人与强化学习：** 先读 Sim-to-Real 的 §4.2，再读 P&C 的进展/压缩机制和 RL 实验。
- **视觉持续学习实验：** DER → FOSTER → MEMO，重点对齐总存储预算。
- **语言/多模态适配：** Progressive Prompts → ProgLoRA → PCLR；此顺序是阅读建议，不代表已经证实的逐篇引用传承。

比较结果前至少统一：测试时任务身份是否已知、是否允许历史样本、总参数和缓存预算、是否有额外预训练、任务顺序，以及使用的是最终准确率还是平均增量准确率。冻结旧参数只能保住相应计算路径，新增分类头、路由和类别竞争仍可能使系统整体表现下降。

本次进行了文献和作者仓库核查，没有运行代码或复现实验。P&C、Sim-to-Real、Learn to Grow 的官方完整实现未在本次核查中确认；其他条目提供了已打开核对的作者代码入口。详细证据位置和未纳入的候选见同目录 notes.md。
