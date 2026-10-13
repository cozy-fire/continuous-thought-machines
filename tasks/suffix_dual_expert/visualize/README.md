# 可视化

两种模式共用同一套代码：逐tick注意力与神经元UMAP，以及Maze历史→FourRooms的配对对照。仅单列 `Policy` 推理快照，不接收Dual教师或训练恢复checkpoint；无训练、梯度或权重更新，不依赖两个旧continual目录或实验脚本。

从仓库根目录运行：

```powershell
# 第一次：采集20个固定案例、生成页面并启动本地服务
python -m tasks.suffix_dual_expert.visualize

# 第二次：同样权重的Maze完整历史切换到10个FourRooms
python -m tasks.suffix_dual_expert.visualize --config tasks/suffix_dual_expert/visualize/switch.yaml

# 已生成后只打开服务，不重新推理
python -m tasks.suffix_dual_expert.visualize --serve-only
python -m tasks.suffix_dual_expert.visualize --config tasks/suffix_dual_expert/visualize/switch.yaml --serve-only
```

默认地址分别为 `http://127.0.0.1:8768/`、`http://127.0.0.1:8769/`。`--no-serve` 仅生成文件；`--reuse` 在配置、案例和权重相符时复用采集数据，重新生成页面。首次运行要求输出目录不存在，避免覆盖旧结果。两种模式分别输出到 `runs/visualize_baseline` 和 `runs/visualize_switch`。

修改 [config.yaml](config.yaml) 的 `checkpoint` 指定默认权重，也可临时使用 `--checkpoint <路径>`。权重、地图和输出路径相对仓库根目录，绝对路径也可用；`cases` 相对本目录。 [switch.yaml](switch.yaml) 继承默认配置，只覆盖模式、输出与端口。CUDA可改为 `device: cpu`。

[cases.json](cases.json) 保留之前的10张Maze与10个FourRooms（原先按成功结果挑选的长路径案例）；换权重不重新筛选成功结果。切换模式使用文件内第一张Maze，默认48动作后切换，保存整个pre/post状态。若Maze提前终止则明确报错，不伪造切换。需保证至少8个Maze动作填满40tick窗口。对照分支使用全新初态；相同输入探针重放fresh分支前若干观测，不执行探针预测动作。

注意力只叠加实际四head均值的最高89/441区域；FourRooms全图显示原生局部视野与遮挡。UMAP默认一个点一个神经元，按案例等量采样后z-score/cosine拟合；所有分支共享同权重的fresh参考布局与P1–P99颜色尺度。支持1/2/3维、手动应用参数和按动作汇总。旧Maze槽在20个FourRooms动作后移除，但递归影响可能继续存在。

输出保留原始NPZ、配置/案例/权重SHA及 `summary.json`；服务只使用保存特征在CPU上重新拟合UMAP。成功案例子集、注意力和UMAP均不能单独证明总体成功率、知识保留或因果功能。

结构：`collect.py` 采集；`page.py` 打包；`server.py` 本地调参服务；`assets/` 共用页面与交互。依赖原模块已有环境，加 `umap-learn`（及其Numba/Scikit-learn依赖）。
