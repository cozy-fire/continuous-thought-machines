# Continual navigation：schema v2

主方法 `tapd_ctm_visual_revisit` 使用 Task-Agnostic 的 `X→C→F` 和后续 `P→C→F`。`configs/full.yaml` 是正式预算，`configs/rtx5090_32gb.yaml` 保持相同环境步与 visit 数并调整吞吐设置，`configs/smoke.yaml` 是缩小预算的全流程验证。`python -m tasks.continual_nav.config --config <yaml>` 输出预算；`python -m tasks.continual_nav.train --config <yaml> --dry-run` 输出完整阶段表。

`ppo.encoder_microbatch_images` 控制一次 ResNet 前向最多处理的图片数，不改变 PPO minibatch、优化器更新次数或训练预算。`full.yaml` 设为 32；RTX 5090 配置设为 200，覆盖当前每个 50 步×4 槽的完整 PPO minibatch。更改此值会改变配置哈希，已有 run 不能据此直接续训。

## 关键契约

- 一次 TA visit 完成两个任务各自的全部 round；每个 round 依次执行 X、C、F。正式配置为 4 visits、每任务每 visit 2 rounds、每轮 X 1,000,000 环境步。X 的奖励仅来自当前 episode 原始像素重访惩罚。`transition_next_obs` 是奖励查询帧，自动 reset 帧只用于下一 episode 的窗口初始化。
- X 的 Encoder、projector 和 Active 用 PPO＋SIGReg 联合更新；KB 冻结。每轮 Active 及其优化器随机重置；视觉权重及 AdamW 状态跨 X 轮延续。X 阶段完整结束后才生成可恢复 checkpoint。X 的 episode 窗口不持久化，最多四条完成的 episode 写入 `diagnostics/`，包含逐帧图片、动作、回报、窗口最大相似度、最大综合惩罚对应的相似度与间隔。
- C 使用冻结的双列教师、20 张观察的 burn-in 和 Online EWC 蒸馏 KB。F 使用 KB 自身轨迹估计 Fisher。TA 结束后永久冻结 Encoder。P&C 的 P 只更新 Active/Adapter/Critic，C/F 仍只更新 KB 或 Fisher。
- TA 每 visit 末评估 KB；P&C 每 visit 末依次评估 KB 与两个任务各自 P 阶段的原始 Active 快照。Active 快照包含当时的旧 KB，并引用冻结视觉 artifact。最终 test 只用 KB。评估步数单独计数，不计入训练预算。
- 训练 checkpoint marker/payload、视觉 artifact 和 X episode artifact 使用 schema v2。v1 训练 checkpoint 和视觉 artifact 不可导入 v2；历史实现保存在 `v1-W_X_C_F` 分支。`--resume` 只从已提交的完整阶段边界恢复；未提交阶段重跑。配置哈希、代码源文件哈希及所有引用文件完整性均需匹配。

## 命令

在仓库根目录、已激活 `ctm` 环境运行：

```powershell
python -m tasks.continual_nav.train --config tasks/continual_nav/configs/rtx5090_32gb.yaml --method tapd_ctm_visual_revisit --seed 0 --run-dir runs/continual_nav/v2-seed0 --wandb-mode online
python -m tasks.continual_nav.train --config tasks/continual_nav/configs/rtx5090_32gb.yaml --method tapd_ctm_visual_revisit --seed 0 --run-dir runs/continual_nav/v2-seed0 --resume --wandb-mode online
python -m tasks.continual_nav.verify_smoke --output-dir runs/continual_nav/v2-smoke --device cpu
```

主方法运行结束后，`exports/vision_final.pt` 可供共享视觉基线导入。所有训练与评估标量写入本地 `events.jsonl`、`metrics.jsonl` 和 W&B。X 的 `ta_X_{task}/*` 面板包含 PPO、SIGReg、两组梯度范数、平均惩罚、非零惩罚率、RGB 元素不变率、整帧不变率、动作频率、相似度区间及间隔频率。`progress/*` 记录阶段完成的累计环境步与更新次数；评估面板按 TA/P&C、KB/Active 和任务分开。

本实现仍使用同步训练环境适配器；评估可通过 `evaluation.backend=subprocess` 和 `evaluation.num_envs` 并行环境执行。`evaluation.interval_steps` 与 `evaluation.drift_episodes` 是保留的旧配置字段，不触发自动阶段中途评估。
