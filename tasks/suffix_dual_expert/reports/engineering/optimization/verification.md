# Optimization verification

已实现整段Maze CNN批量编码，并将默认Maze microbatch从10增至25、Compress从1增至4。完整前缀BPTT、100条接受序列、有效后缀归一化、统一clip、每次一个Adam及每组一次EWC均保持。

## 实测组件性能

设备：RTX4060 Laptop 8GiB；FP32/禁TF32/确定性CUDA，无compile。每设置预热后两次测量，使用CUDA同步计时。

| 组件 | 原执行 | 新默认 | 组件加速 | 新峰值预留 |
|---|---:|---:|---:|---:|
| 30步/100轨迹完整前向反向，含320图缓存 | 11.483秒，逐步CNN/micro10/重算 | 8.109秒，批量CNN/micro25/重算 | 1.416倍，耗时降低29.4% | 5248MiB |
| 50步/16+16槽学生更新，四次联合Adam/EWC | 29.337秒，micro1/重算 | 9.462秒，micro4/重算 | 3.100倍，耗时降低67.7% | 1782MiB |

Replay计时仅前向/反向，clip/Adam另计；排除主机梯度对照耗时。轨迹为固定BFS路径参考，不是正式严格筛选接受轨迹。Compress用固定合成目标和episode重置，不包含教师前向、环境推进和写盘。不同GPU/长度/阶段会改变比例，不能把这些组件比直接乘到A4000总训练时间。

单独批量CNN、保持micro10/重算为11.679秒，没有测得加速。关闭重算时micro2/5重放为38.212/17.462秒，反而较慢；micro10因保守显存预估跳过，未发生OOM。Compress4关闭重算为9.499秒，与开启的9.462秒无稳定收益，峰值预留升至4340MiB，因此默认保留重算。

原始记录：`replay_benchmark_v2/benchmark.json`、`compress_benchmark/benchmark.json`。初次过严Adam容差失败的`replay_benchmark/benchmark.json`保留审计；不作为通过证据。

## 数值与协议

- 当前配置13项核心测试全部通过，19.182秒，未跳过。硬标签CE、全参数梯度、初态梯度、终止mask、不同microbatch的一次Adam以及每组EWC一次均检查。
- 完整30步CUDA对照，全部参数梯度容差rtol5e-4/atol3e-6；观察到最大绝对差约5.3e-7。一次Adam后参数最大绝对差约1.53e-6，检查容差rtol1e-5/atol5e-6。浮点执行顺序不同，不声称旧/新参数逐位一致。
- 完整50步联合更新检查每次clip前的全部参数梯度及四次Adam后的参数；micro1/2/4、重算开关均通过。梯度容差rtol1e-3/atol5e-6，最终参数rtol2e-5/atol1e-6；每窗口4次EWC/4次Adam。
- 优化前后的预算均为42 Maze阶段/23500 Maze Adam、4096/1024 Fisher、800 C窗口/3200 fresh Adam/128万决策、3200独立评估episodes。
- 默认配置文件与Config解析结果相同；A4000候选覆盖配置继承相同预算。

## 显存与配置

默认 `suffix_microbatch=25, compress_microbatch=4, checkpoint_encoder=true, encoder_chunk=32`。`config.a4000.yaml`提供Maze20/关闭重算的显式候选，预计约12.3GiB allocated来自本地micro5的线性显存外推，尚未在A4000验证。推理、冻结专家不重算；没有新增自动OOM重试、动态批次改变或放宽筛选。

更改配置或源码不能绕过既有身份校验恢复旧run。数学目标保持，但优化前后不要求逐位相同；同一优化配置的直接运行与恢复仍须逐位一致。

## 综合验收

当前默认配置综合GPU验收已通过，记录为 `gpu_validation/verification.json`，绑定当前源码SHA与完整解析配置：

- 真实320图缓存、完整30步/100序列/micro25重放；encoder/controller/actor梯度有限且非零，峰值预留5366MiB（约5.24GiB）。
- 两个真实16+16槽50步双专家窗口、8次联合Adam；两任务梯度非零、每窗口4次EWC、EWC非零，峰值预留2204MiB。小流程的Maze/F预算明确缩小，不计入正式预算。
- Maze池/F/C窗口完整边界的直接与恢复运行：学生、Adam及三个推理快照逐位一致。
- 完整FourRooms教师、Maze冻结副本及Fisher不变。
- 8个诊断episode完成，策略及快照权重不变；不作为泛化或保留结论。
- 旧包导入被屏蔽；13核心测试、AST/导入检查和diff检查通过。统一证据SHA及当前源身份见 `optimization_integrity.json`。

未启动正式训练，未SSH、提交或修改旧包。优化前的11项测试和时长估计保持历史身份，不冒充当前配置验收。
