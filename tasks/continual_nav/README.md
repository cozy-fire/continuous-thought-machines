# Continual navigation — deliveries 1–3

本目录当前交付配置、公共数据类、数据划分、环境、CTM模型、世界模型/SIGReg及原始图像回放；PPO、蒸馏与全局阶段调度尚未实现。关键 Python 逻辑采用英文注释。

## 运行与验证

从 `continuous-thought-machines` 仓库根目录执行。以下命令使用现有 `ctm` 环境；本交付没有安装或升级依赖。

```powershell
& 'D:/conda/envs/ctm/python.exe' -m unittest discover -s tests/continual_nav -p 'test_*.py' -v
& 'D:/conda/envs/ctm/python.exe' -m tasks.continual_nav.config --config tasks/continual_nav/configs/full.yaml
& 'D:/conda/envs/ctm/python.exe' -m tasks.continual_nav.config --config tasks/continual_nav/configs/smoke.yaml
& 'D:/conda/envs/ctm/python.exe' -m tasks.continual_nav.verify_envs --config tasks/continual_nav/configs/full.yaml --output-dir scientific-evidence/continual_nav/delivery_01_new_run
```

最后一条命令需要本地 `data/mazes/medium/{train,test}/0/*.png`；可用 `--maze-root` 指定其他根目录。输出目录必须不存在。命令仅做环境验证，固定执行两任务各610次transition，不运行模型、优化器或完整训练smoke。

新增依赖清单为本目录 `requirements.txt`；测试使用标准库 `unittest`。旧 `tests/conftest.py` 依赖多个既有任务，独立unittest发现不会加载它。

## 配置接口

- `load_config(path) -> Config`：full必须完整；smoke通过`extends: full.yaml`明确覆盖。未知字段、缺失字段、重复YAML键、错误类型、非有限浮点与不兼容参数均报错。
- `save_config(config,path)`：保存无继承的完整配置；`config_hash(config)`提供稳定SHA-256。
- YAML中的`lambda`在Python中为`lambda_`；其余字段同名。所有相对数据路径相对**调用时工作目录**，`extends`相对**配置文件目录**。
- full使用`cuda:0`，环境验证不初始化CUDA；smoke默认`cpu`用于本地验证，模型结构/float32/ticks/M/损失不变。
- v1的图像、动作数和模型结构锁定为计划值；修改结构须同步升级契约。预算、设备与数值超参数受显式校验。
- `budget_summary`只计算预算，无任务调度副作用。主方法full=37,922,880 transitions/seed，smoke=760；不含评估与重跑。

## 数据与环境接口

```python
from tasks.continual_nav.config import load_config
from tasks.continual_nav.data.manifest import build_manifest, MazeManifest
from tasks.continual_nav.envs import build_env, VectorEnvAdapter

config = load_config('tasks/continual_nav/configs/full.yaml')
e = config.evaluation
manifest = build_manifest(config.environment.maze_root,
    validation_count=e.maze_validation_hashes,
    validation_episodes=e.validation_episodes,
    test_episodes=e.test_episodes, drift_episodes=e.drift_episodes)
# build_manifest validates split separation; load(path, root) also verifies file hashes.
env = build_env('maze_medium', 'train', 0, config=config, maze_manifest=manifest)
obs, info = env.reset()
next_obs, reward, terminated, truncated, info = env.step(4)
frame = env.render()  # HWC uint8 84x84x3, identical to the policy observation.
env.close()
```

- 观察是连续内存uint8 `[3,84,84]`；不包含任务名称、mission或direction数值。环境不做`/255`，该转换属于后续模型入口。
- `render()`返回当前观察的HWC副本；FourRooms不使用原生全局render。调用前必须reset。
- 单环境结束后必须显式reset；`VectorEnvAdapter`在同一步自动reset，`EnvStep.transition_next_obs`保留真实终止帧，`next_obs`返回新episode首帧。终止槽位的`info`保留旧episode字段，另含`reset_info`。
- 训练`seed`为采样流种子。Maze逐episode均匀抽取当前split文件；FourRooms从`[0,1000000)`采样原生episode seed。显式`reset(seed=s)`重建流；自动reset传None以继续流。
- FourRooms评估构造后从固定panel首项开始，后续reset依次循环；`reset(seed=2000005)`可直接选择validation panel内的布局。构造参数seed不改变固定评估面板。
- Maze评估factory只从固定panel抽样；后续评估器需以每个panel entry创建单图`MazeEnv(root,(entry,))`，实现每图恰评估一次。当前交付不实现评估器。
- manifest的`sha256`是原始PNG文件字节哈希。validation选择train排序后前512个不同hash，所有同hash副本同时移出train；排序键为`(sha256,relative_path)`。仅扫描`train/0`和`test/0`。manifest保存相对路径；载入校验文件存在、内容及split隔离。
- 世界模型回放和Fisher接口没有reward字段；CTM/双列state为显式类型。所有batch的时间维在batch维之前，具体shape见`contracts.py`英文docstring。

## 交付2：模型接口

`models/vision.py`复用本地resnet34-2；`rope.py`实现投影后Q/K的二维轴向旋转；`ctm.py`实现两类有限同步窗口及NLM；`policy.py`实现独立KB、单列Actor–Critic与双列侧向连接。

```python
import torch
from tasks.continual_nav.models import (
    VisionEncoder, encode_obs, StandalonePolicy, SingleActorCritic, DualPolicy,
    detach_state, frozen_copy,
)

encoder = VisionEncoder(config)  # Frozen/eval by default, including BatchNorm.
kb = StandalonePolicy(config)   # No critic and no encoder inside KB.
policy = DualPolicy(config, kb, kb_ready=False)  # Random Active, independent of KB.
state = policy.initial_state(batch=2)
images = torch.zeros(2, 3, 84, 84, dtype=torch.uint8)
with torch.no_grad():
    fmap = encode_obs(images, encoder)  # Once per observation batch.
    result = policy.step(fmap, state, torch.ones(2, dtype=torch.bool))
    state = result.state
# Actions/log_prob are constructed outside policy.step from result.logits.
teacher = frozen_copy(policy)  # Copies parameters/buffers, including the old KB.
```

- 三类策略均提供`initial_state(B,device=None)`和`sequence(fmaps,state,episode_start,valid_mask=None)`。`fmaps`为float32 `[L,B,128,21,21]`；两种mask为bool `[L,B]`。
- `StandalonePolicy.step`返回`(logits,CTMState)`；单列Actor–Critic与双列的`step`返回`PolicyOutput`。`sequence`统一返回`PolicySequenceOutput`，其`logits`为`[L,B,5]`，`value`仅Actor–Critic有`[L,B]`，KB为None。
- 每观察推进2ticks；两个独立同步向量各528维，M=20。序列内部不自动detach；rollout边界由调用者使用`detach_state`。padding输出置零、state保持原值，不能当有效策略目标。
- `DualPolicy`持有冻结KB、随机Active和Adapter；KB先推进每tick，Active再使用同tick的KB激活。`kb_ready=False`强制禁用侧连，首次压缩完成后由阶段控制器调用`policy.kb_ready.fill_(True)`。
- `VisionEncoder.set_world_training(True)`只由未来W.fit调用。结束后`freeze()`锁定参数、梯度与BN；TA结束调用`freeze(permanent=True)`，之后禁止重新开启视觉训练。单纯父模块`train()`不会打开冻结BN。
- 构造新的DualPolicy即新建Active/Adapter，不复制KB控制器；优化器清空属于未来阶段状态机的责任。
- `frozen_copy(module)`使用deepcopy隔离参数与buffers。快照采样state需调用快照自身`initial_state`创建；运行trace不存于模型内部。裸`state_dict`序列化可用，阶段边界checkpoint尚未实现。
- `Controller.tick`只返回当前tick的新state与激活；`SpatialAttention(...,return_weights=True)`可用于诊断，默认不保存历史Attention权重。

真实环境像素验证（无优化器训练循环，需新输出目录或尚无`model_report.json`的目录）：

```powershell
& 'D:/conda/envs/ctm/python.exe' -m tasks.continual_nav.verify_models --config tasks/continual_nav/configs/full.yaml --output-dir scientific-evidence/continual_nav/model_check_new_run --device cpu --steps 50
```

可用`--maze-manifest`复用交付1保存的JSON（加载仍核验文件hash）；不提供时按配置生成manifest。该命令将两种环境图像交给共享E和双列策略，比较50步逐步采集/序列重放，并用合成loss检查梯度，不评价任务成功率，也不是完整训练smoke。

## 交付3：世界模型与回放接口

`models/world.py`持有共享E、GAP→128维projector g和动作条件predictor F。两帧拼成`[2B,3,84,84]`，只调用一次E；两端都有梯度，不使用stop-gradient。F接收5维policy action one-hot，包括两个不同的wait槽位，预测下一帧latent本身。

- `WorldModel.transition(obs,next_obs,actions)`接收uint8 `[B,3,84,84]`与int64 `[B]`，返回`WorldPrediction(z,z_next,z_pred,error)`；前三项为`[B,128]`，error为未平方L2 `[B]`。
- `prediction_losses`返回`(total,mse,sigreg)`。MSE对batch和特征求均值；SIGReg对两个时间切片分别计算后平均，共用256个单位随机方向、17个积分节点。总损失为`mse+0.02*sigreg`。
- `world.curiosity(...)`只允许冻结模式，返回`(log1p(error),error)`，两者均detach。没有外在奖励混合或奖励归一化。
- `make_world_optimizer(world,config)`创建仅包含E/g/F的AdamW。调用者跨轮保留同一个optimizer；`fit_world_model(...,task,replay_rng,sigreg_rng)`执行配置规定的更新次数，返回最后一次更新的loss/梯度范数/样本配额。成功和失败都会冻结E/g/F及BN；失败不推进版本，但可能已经完成部分参数更新，不能原地当作完整轮继续发布。
- 成功fit后仅调用一次`world.commit_fit()`，将encoder/world版本各加1。它是内存版本操作；生产流程必须由交付5保存完整阶段checkpoint后才能进入X。TA结束仍需`encoder.freeze(permanent=True)`。已有旧视觉state_dict不含新增`encoder_version`字段，严格加载时需要显式迁移，不允许静默忽略。

`SnapshotCollector(E,standalone_KB,config,world_version=...)`深拷贝固定E_old和KB，使用独立CPU action RNG进行categorical采样；不读取外在奖励。`collect(envs,writer,steps,action_rng,start_transition_id)`重置环境和trace，精确采集steps条实际transition并发布manifest，返回`CollectionResult`。steps必须整除num_envs。调用者维护全局递增transition_id；episode_id在单个source_stage内跨槽位唯一，全局身份由source_stage和episode_id共同确定。writer的snapshot_id必须与collector一致。

`data/replay.py`的数据流：

1. `ShardedWriter`只接收同task、同encoder/world版本的`Transition`；每片最多2048条，图像复制为uint8。W不存error；X额外存原始L2排序分数，均不存reward或latent。达到精确预算后`finish()`原子发布manifest，记录各分片SHA-256。
2. `TransitionStore(manifest)`核验分片哈希，`take(indices)`按请求顺序返回记录，支持重复索引；内存只缓存一个解压分片。采样后的图像每次由当前E/g重新编码。
3. `ReplayBank.begin(task,encoder_version,world_version,source_stage,expected_count)`创建当前X的builder。逐条`offer(record,error,source="X")`保留top-K；同分时transition_id较小者优先。像素存磁盘memmap，heap只持分数和slot。必须提交本轮全部expected_count条后才能`finish()`，整个新池替换该任务上轮池；另一任务不变。异常时调用`abort()`释放本builder的临时文件。
4. `sample_world_batch(fresh,high,task,batch_size,high_fraction,rng)`从各池均匀有放回采样。full为128+128，smoke为4+4；空高误差池全部fresh。返回CPU `WorldBatch`，先fresh后high，含来源mask和ID；不能跨任务。
5. replay的`latest.json`原子切换不等于全局checkpoint。旧版本暂留，生产调度器只能在阶段checkpoint提交后调用`bank.prune(task,protected_manifests=...)`，保留恢复所引用的版本。单一writer，不支持并发写同一task池。

真实环境小预算验证（CPU；必须使用新的输出目录）：

```powershell
& 'D:/conda/envs/ctm/python.exe' -m tasks.continual_nav.verify_world --config tasks/continual_nav/configs/smoke.yaml --maze-manifest scientific-evidence/continual_nav/delivery_01_20260921/maze_splits.json --output-dir scientific-evidence/continual_nav/world_check_new_run
```

每任务连续验证两轮，每轮W采集40条并更新2次，再用固定随机Active采集40条检查冻结好奇心和top-K。合计320条、8次更新；这是接口验证，没有PPO学习、压缩或任务成功率评估，也不支持恢复。命令拒绝大预算配置。

## 后续衔接

交付4实现PPO、蒸馏与Fisher/EWC；交付5实现阶段状态机、完整checkpoint、恢复及评估衔接。`PhaseKey`只是阶段身份数据类，不会执行任何训练。交付3没有启动完整训练。
