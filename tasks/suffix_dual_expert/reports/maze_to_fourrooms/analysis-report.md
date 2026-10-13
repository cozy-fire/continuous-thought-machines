# Maze历史窗口填满后切换FourRooms

重置初态和继承Maze历史两组均10/10与10/10成功。继承组平均多用27.1步，中位差19.0步，7例变慢、3例变快。

## 目标与设置

使用已完成双专家Compress的单列最终KB，SHA `2646ed09c50b5b5107e41924738719653c7a35289873de9ca1e3c6ea0a98604f`；这是同时学习Maze和FourRooms的KB，不是此前只训练Maze的25步Active。参数冻结，Argmax、FP32、Maze ticks5 / FourRooms ticks2、历史40tick、300步FourRooms上限，无额外训练。

Maze来自同一未见200图面板的map142，BFS92步、此前完整闭环96步成功。精确复现前48个动作后切换，已经执行240个内部tick；原初始化的40槽全部替换为Maze计算，切换时仍距Maze目标48步。没有提前终止成功、没有向FourRooms携带环境位置或像素，仅保留完整pre/post递归状态。

将同一Maze状态独立复制给此前已核验的10个成功长路FourRooms案例。每例两条件从同一seed、位置、朝向开始：fresh第一步episode_start=true；maze_history从第一步起始终false。环境是新FourRooms环境，但模型状态不清空。只加载同一个KB、同一个actor；task参数切换为fourrooms只决定每输入2tick，没有显式任务embedding。此实验不证明模型自行决定何时切换内部计算预算。

## 逐例闭环结果

|FourRooms seed|最短动作路|重置状态步数|继承Maze步数|增加步数|首动作一致|相同视觉前20步动作分歧|
|---|---|---|---|---|---|---|
|3000081|33|111|229|118|False|1|
|3000107|33|41|117|76|False|5|
|3000001|30|48|41|-7|False|5|
|3000009|30|36|37|1|False|2|
|3000039|30|195|266|71|False|3|
|3000105|30|53|56|3|False|4|
|3000008|29|35|83|48|False|5|
|3000056|29|81|116|35|False|4|
|3000141|29|75|61|-14|False|3|
|3000035|28|128|68|-60|False|2|

## 历史干扰与适应边界

首个FourRooms动作有10/10例与fresh不同：fresh均输出0（左转），继承Maze历史均输出3（FourRooms中为no-op）。继承组第2动作开始均已输出有效FourRooms动作，随后最终全部完成，并非等待20步才开始动作。

相同视觉探针使用fresh闭环的前20张观测，分别从两初态处理，不让两个环境轨迹分叉；这样动作概率和状态差异来自初始历史差异。原Maze历史槽每动作移出2格，第20动作后没有直接保留的旧槽，但旧历史可以经新pre/post递归继续影响后续计算。第20步post状态RMS差异仍为0.173–0.183，动作分布TV最大为0.485，因此不能把20步称为彻底清除记忆。

此次结果支持“在这一Maze历史条件和这10个FourRooms案例上，无需重置也能最终完成任务”；不支持“没有历史干扰”“切换立刻准确”“知识完全不受污染”或普遍跨任务保证。闭环不同首动作可能引发探索路径差异，最终步数不能全归因于前20步窗口本身。仅一个Maze历史，未比较同任务已成熟FourRooms历史，因此不能分离跨任务污染与一般非初始状态效应。

## 验证

源KB SHA核验、严格加载与关键模型/环境源码SHA均匹配此前核验。Maze前48动作/位置逐项复现，fresh FourRooms各例成功和步数复现旧记录。两条件所有动作在独立同seed环境重放，位置、方向、done和success匹配；保留历史条件前20动作逐步断言旧pre/post槽确实原样移位。全程logits/pre/post有限、模型hash和checkpointSHA前后不变。state和真实输入/世界图、动作概率完整保存。

图表见 `figures/paired_steps.png`、`figures/matched_input_history.png`、`figures/trajectories.png`。
