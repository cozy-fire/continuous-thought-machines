# 图目录

success_rates.png：固定800图全部已核验长度成功率，误差条Wilson95%区间；reset周期不同须保留限制，不能宣称等价。

path_length_breakdown.png：持续递归按BFS距离的探索性成功率，各层样本数与区间见stats.json；路线长不是唯一难度，无分层显著性结论。

path_length_breakdown_compare.png：上图持续递归、下图reset，每图比较全部已核验长度在相同BFS四层的成功率与Wilson95%区间；横轴标每层地图数，线段仅引导阅读。path_length_L10/15/20.png沿用原两模式图样式，各自显示同样分组。数据来源path_length_stats.json。

当前合并图包括10/15/20/25：总体图用于判断25步是否超过既有20步，地图抽样Wilson区间不代表训练seed；观察25递归与20差值很小，正式配对检验未建立提升。BFS分层图显示25步的变化分布于不同路径层，结合24项Holm统计，不把折线或单层数值当因果或通用能力证据。
