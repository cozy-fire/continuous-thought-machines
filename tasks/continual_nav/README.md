# Continual navigation — deliveries 1–6

本目录已实现配置、数据契约、环境、CTM、世界模型/SIGReg、原始图像回放、循环PPO、循环蒸馏与Fisher/EWC。交付5新增全局阶段调度、阶段边界checkpoint、固定面板评估及三类共享视觉条件基线入口。关键 Python 逻辑采用英文注释。

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
- `budget_summary`只计算预算，无任务调度副作用。主方法full=32,290,112 transitions/seed，smoke=760；不含评估与重跑。

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
- Maze评估factory只从固定panel抽样；评估器以每个panel entry创建单图`MazeEnv(root,(entry,))`，实现每图恰评估一次。对应实现为`evaluate.panel_envs`。
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
- `VisionEncoder.set_world_training(True)`只由W.fit调用。结束后`freeze()`锁定参数、梯度与BN；TA结束调用`freeze(permanent=True)`，之后禁止重新开启视觉训练。单纯父模块`train()`不会打开冻结BN。
- 构造新的DualPolicy即新建Active/Adapter，不复制KB控制器；阶段调度器在每次X/P开始时重建对应优化器。
- `frozen_copy(module)`使用deepcopy隔离参数与buffers。快照采样state需调用快照自身`initial_state`创建；运行trace不存于模型内部。裸`state_dict`序列化及交付5的阶段边界checkpoint均可用。
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

交付5已实现阶段状态机、完整checkpoint、恢复及评估衔接；交付6的完整760步GPU smoke及独立边界恢复已通过。`PhaseKey`只是阶段身份数据类，不会执行任何训练。当前没有启动完整训练。

## 交付4：循环学习接口

这些接口执行单个阶段或单个batch，不实现跨任务调度和恢复。图像batch与目标默认存CPU，运行state和模型在同一设备；动作、minibatch、Fisher抽样各使用独立CPU `torch.Generator`。交付4最初验证设备为CPU；后续已通过RTX 4060 Laptop上的有限GPU验证。

### X/P：PPO

```python
from tasks.continual_nav.learning.ppo import run_ppo_stage

# Caller creates a fresh random Active for each X/P; the helper does not reset weights.
policy = DualPolicy(config, kb, kb_ready=kb_ready)
result = run_ppo_stage(
    envs, policy, encoder, config, phase="X",
    steps=config.exploration.steps_per_round, world=world,
    high_error=builder, action_rng=action_rng, minibatch_rng=minibatch_rng,
    start_transition_id=next_transition_id,
)
next_transition_id = result["next_transition_id"]
```

- `PPOCollector(envs,policy,encoder,config,phase=...,world=None,high_error=None,start_transition_id=0)`在阶段入口重置环境/trace。X强制共享同一个world.encoder，冻结E/g/F；P不允许world或high_error，永久冻结E。单列P可传`SingleActorCritic`。
- `collect(steps,action_rng=...)`返回PPOBatch，steps必须为num_envs的正整数倍且不超过一个rollout。保存真实`transition_next_obs`。尾rollout直接缩短时间维；学习器也支持valid_mask。
- 每个动作后用真实next frame和动作后的trace做独立bootstrap前向，丢弃其新trace，正式collect state只推进当前观察。成功不bootstrap，超时bootstrap但截断GAE；该额外前向不计环境交互。X只存当时算出的log1p(L2)奖励，不访问reward_ext；原始L2交给builder。P仅使用reward_ext。
- `compute_gae(reward,value,bootstrap,terminated,truncated,valid,gamma,gae_lambda)`返回原始advantages和returns；padding不传播GAE。policy loss仅使用归一化优势副本，不改变returns。
- `make_ppo_optimizer(policy,encoder,config)`每阶段新建Adam，仅含Active/Adapter/Critic，或单列Controller/Actor/Critic。
- `train_ppo(batch,policy,encoder,optimizer,config,rng=...)`按环境随机分组，时间顺序不变；起点state detach，序列内部反传，episode reset截断。验证optimizer所有权，返回最后minibatch的loss/entropy/KL/梯度范数及总updates。
- `set_stage_learning_rate(...,completed_rollouts,total_rollouts)`以已完成rollout比例设置学习率。stage helper在每轮更新前设置，阶段完成后设0；两个rollout的更新学习率为1e-4、5e-5，结束为0。
- 每次更新后继续使用采集末尾的detached trace；不改用新参数重新计算的trace。这是计划规定的截断近似。
- `run_ppo_stage`精确消耗steps，重建optimizer但不初始化policy，成功后发布完整X池，返回transitions/updates/rollouts/next_transition_id/pool_manifest/last等字段。builder失败清理由调用者在finally执行abort；失败阶段不允许直接继续，应由最近已提交checkpoint恢复。

### C：冻结教师与序列蒸馏

`SequenceCollector(envs,teacher,encoder,config,mode="C",start_transition_id=...)`立即深拷贝双列teacher，包括旧KB、Active、Adapter；之后学生KB更新不能改变该快照。共享E固定版本并冻结，teacher采集trace跨窗口持续运行。

`collect(steps,action_rng=...)`每次收集至多`num_envs*learning_steps`条新转移，返回SequenceBatch。obs/masks为`[U+L,B,...]`，U=10，L=配置learning_steps；前缀从上个窗口已有观察取得，不足左padding，尾窗口右padding。全动作teacher_log_probs为`[L,B,5]`，在采集时保存；前缀不新增交互、不重复计目标。

`make_distill_optimizer(kb,encoder,config)`每C新建仅包含学生KB的Adam。`train_distill(batch,kb,encoder,ewc,optimizer,config,rng=...)`按环境分组，从学生自己的initial_state开始，以no_grad推进10个观察（20ticks），detach后对learning步反传。loss为`KL(teacher||student)+lambda/2*sum(F*(theta-center)^2)`。无value、entropy或hidden-state loss。首次ewc=None时惩罚为0，后续立即启用。

`run_compress_stage(envs,teacher,kb,encoder,config,ewc,steps=...,action_rng=...,minibatch_rng=...,start_transition_id=...)`先创建teacher快照，再启用学生训练；每收集一个窗口即训练一次，不在C结束后批量重复。返回transitions/updates/windows/next_transition_id/teacher_snapshot_id/last/kb_ready=True。返回的readiness须由调度器保存并用于下一次DualPolicy构造，helper不修改其他policy对象的buffer。

### F：无奖励Fisher与EWC状态

1. `collect_fisher(envs,kb,encoder,config,action_rng=...,fisher_rng=...,start_transition_id=...)`只接受无Critic的StandalonePolicy。固定快照采集配置规定的4096条自身策略转移，以独立Fisher RNG无放回抽取1024个ID；smoke为20/8。返回FisherSequenceBatch列表，不访问或保存reward。F数据仅保留本阶段CPU窗口，不上GPU堆叠完整轨迹。
2. `estimate_fisher(kb,encoder,sequences,config)`检查选中ID唯一且满足预算，按原序列重放，每次只保留一个环境窗口的计算图。burn-in与C一致；对每个选中log-prob分别autograd.grad，将梯度平方累加后除以计分点数。返回CPU float32 `dict[name,Tensor]`，无梯度参数保留全零键。结束冻结KB，不执行optimizer.step，不改变E/KB权重或BN。
3. `update_online_fisher(kb,current,previous,decay=...,sample_count=...,stage_key=...,encoder_version=...)`返回FisherState；首次直接用current，后续`decay*previous+current`，center始终替换为当前KB的独立副本。调用者按TA/P&C明确传入0.1/0.3，不由函数猜阶段。
4. `ewc_penalty(kb,state,coefficient)`严格核验参数名、shape、float32、有限性和非负Fisher，保护范围仅KB Controller+Actor。sample_count记录本次估计数；completed_compressions累计。

Fisher是在截断序列上的策略梯度平方均值，不是精确全轨迹Fisher；不做张量min–max归一化或epsilon floor。C/F的左padding不推进state，10帧burn-in不意味着精确恢复无限历史。

真实两环境验证（新输出目录）：

```powershell
& 'D:/conda/envs/ctm/python.exe' -m tasks.continual_nav.verify_learning --config tasks/continual_nav/configs/smoke.yaml --maze-manifest scientific-evidence/continual_nav/delivery_01_20260921/maze_splits.json --output-dir scientific-evidence/continual_nav/learning_check_new_run
```

每任务X40/C40/F20，再单列P40，共240次交互、6次PPO更新、4次蒸馏更新、16个Fisher计分点。此验证的世界模型为随机初始化后冻结，用于检验奖励/梯度/回放接口，未先执行W.fit，不是完整760步smoke或成功率评估。输出learning_report.json。日志中的PPO loss/KL是optimizer.step前的值；单minibatch首epoch的ratio为1、归一化policy loss接近0是正常情况，不代表梯度为0。

## 交付5：运行、恢复与评估

`schedule.expand_stages`生成完整有序阶段表；`train.Runner`每次`run_next()`提交一个阶段。W.collect和W.fit分别占一个阶段，init不计入`--max-stages`。阶段入口检查可训练参数所有权；每个X/P新建随机Active及Adapter，C使用独立旧KB教师快照，F更新在线Fisher。单列基线跨P保留权重、重建优化器。TA最后一次F后永久冻结并导出E。

从仓库根目录执行以下命令。训练命令必须显式提供seed与新run目录；不自动启动配置中的其他seed。`--max-stages`只在完整阶段后停止。

```powershell
# Inspect data/configuration/schedule without constructing or training models.
& 'D:/conda/envs/ctm/python.exe' -m tasks.continual_nav.train --config tasks/continual_nav/configs/smoke.yaml --dry-run

# Only W.collect and W.fit; this is not the complete 760-transition smoke.
& 'D:/conda/envs/ctm/python.exe' -m tasks.continual_nav.train --config tasks/continual_nav/configs/smoke.yaml --method tapd_ctm --seed 0 --run-dir runs/continual_nav/example --max-stages 2

# Continue from the committed boundary for one additional stage (X here).
& 'D:/conda/envs/ctm/python.exe' -m tasks.continual_nav.train --config tasks/continual_nav/configs/smoke.yaml --method tapd_ctm --seed 0 --run-dir runs/continual_nav/example --resume --max-stages 1

# Evaluate a complete checkpoint on CPU; output must not already exist.
& 'D:/conda/envs/ctm/python.exe' -m tasks.continual_nav.evaluate --checkpoint runs/continual_nav/example/checkpoints/latest.json --policy kb --split validation --output runs/continual_nav/example/evaluation/manual_validation.json
```

`--maze-manifest <path>`可复用固定split；加载仍核验原图SHA-256、split隔离、validation独立hash数量和面板大小。训练首次运行保存所用面板，恢复使用run内的副本。dry-run只验证配置/数据/预算，**不验证基线视觉artifact可用性或设备可训练性**。

### 四种方法与视觉来源

| `--method` | 实例及流程 | 新运行额外参数 |
|---|---|---|
| `tapd_ctm` | TA的W.collect/W.fit/X/C/F，随后P/C/F | 禁止传视觉artifact |
| `single_task_ctm_shared_vision` | 独立SingleActorCritic，同一任务连续P段 | `--task maze_medium`或`fourrooms`；`--vision-checkpoint` |
| `sequential_ppo_shared_vision` | 一个SingleActorCritic，任务间保留参数 | `--vision-checkpoint` |
| `pnc_without_exploration_distill_shared_vision` | 随机KB与空Fisher开始P/C/F | `--vision-checkpoint` |

视觉输入必须是**同seed主方法完成TA后**的`exports/vision_final.pt`及同目录的`.pt.sha256.json`侧文件；不是任意ResNet权重。导入验证seed、方法、模型结构、固定数据/面板hash和永久冻结标记，原字节复制到基线run的`exports/shared_vision.pt`。基线不载入主方法控制器、g/F或Fisher；恢复时已保存视觉引用，无需重复传`--vision-checkpoint`。

full每seed共享TA为11,265,536 transitions；主方法总计32,290,112；条件P&C新增21,024,576；顺序PPO新增15,000,000；单任务每个策略新增7,500,000。成本记录同时保存共享TA来源成本、是否本run实付、已提交总步数及Progress步数。评估、中断重跑另计。名称始终保留`shared_vision`，不称完全无预训练对照。

### 保存和恢复契约

- `checkpoints/<stage编码>__<uuid>.pt`保存模型与buffers、世界模型AdamW、Fisher/center、独立RNG、阶段前缀、计数器和依赖hash。E/world版本为模型state_dict中的buffer。单列的KB/dual/world/Fisher字段为null；空replay引用为`{}`。
- 提交顺序是临时文件→原子替换pt→`.complete.json`→原子切换`latest.json`。只有latest指向的完整边界用于CLI恢复；不依据日志、文件时间或孤立pt猜进度。单run只支持一个writer。
- 恢复必须使用相同config、method、seed、task、源码manifest及所需数据。缺失/修改的manifest、分片、地图或视觉文件直接报错。stage boundary重置环境/trace，所以不保存Gym运行时；不支持阶段内部逐步续训。
- W.fit完成提交后清理fresh；X新池提交后清理该任务旧池。保留审计manifest和checkpoint，但**旧checkpoint可能因依赖已清理而不能恢复**。需长期保留某个旧恢复点时，应在下一阶段开始前复制完整run目录。
- `attempts/*.json`记录每次阶段尝试。`confirmed_env_steps`仅在一次vector step完整返回后递增并落盘，不能保证突然断电或vector内部部分失败时最后若干步的精确计数。`committed`区分正式预算与未提交的额外交互；异常后重新构造Runner并`--resume`，不要在失败对象上原地继续。
- `--max-stages`不是暂停运行中的阶段。用户中断W.fit/X/P/C/F时，恢复会从上一个边界重执行整个未完成阶段；完整结束后`finalized=True`防止再次执行final-test。

### 评估、日志与导出

`evaluate.py`按固定面板逐episode执行argmax；每episode使用策略自己的learned initial state。模型深拷贝冻结，训练RNG和原模型mode/参数不变。自动评估使用模型当前设备；独立evaluate CLI在CPU加载。主方法/P&C默认评估独立KB，单列自动加载SingleActorCritic。`--policy active`只用于存在双列状态的X/P边界；C之后不再保留Active。

W.fit前后评估KB小面板，并沿旧视觉驱动的**同一条轨迹**计算新旧视觉下策略KL。X/P开始、每配置interval和结束评估Active；C开始/结束评估KB；F不重复评估。interval在PPO rollout完成后检查，full的100,000恰好整除400条rollout；其他兼容配置若不整除，事件会延后至首个越过阈值的rollout边界。

成功率、原始return、全部episode长度、成功episode长度及Maze成功路径/最短路比均保留。无成功时成功长度和路径比是null。`analysis.metrics.forgetting`按after-C矩阵计算历史最好成绩减当前成绩；`seed_summary`保留每seed值、均值和样本标准差，单seed标准差为null。该函数不自动启动或汇集其他seed实验。

| 文件 | 用途 |
|---|---|
| `resolved_config.yaml`、`provenance.json` | 完整配置、源码hash、阶段表、RNG派生种子、软件/设备版本 |
| `manifests/` | Maze/evaluation固定面板及清理前的数据审计manifest |
| `events.jsonl` | 阶段进入时参数所有权、更新、评估与完成事件 |
| `metrics.jsonl`、`wandb_run.json`、`wandb/` | 本地审计指标、W&B运行标识和SDK缓存；X/P每PPO rollout、W.fit每50次更新、TA及P&C的C每10个训练窗口记录曲线，阶段末尾另记汇总；Fisher统计与评估/漂移 |
| `attempts/` | 包含失败尝试的已确认训练交互记录 |
| `exports/vision_final.pt` | 主方法TA完成后的冻结视觉及共享成本 |
| `exports/final.pt` | 全流程结束的E+KB或E+SingleActorCritic、配置、固定manifest、成本 |
| `evaluation/final_test.json`、`forgetting.json` | 最终测试及after-C原始矩阵/遗忘量 |

JSONL是追加的尝试日志，恢复不会删除失败尝试中已经写出的曲线点；判断正式完成状态以checkpoint为准。`policy_internal_ticks`统计实际控制器样本ticks，包括双列、bootstrap、burn-in、重放；`eval_internal_ticks`独立统计自动评估。两者都不是环境transition数。最终checkpoint的`eval_steps`包含已完成评估；被中断的评估没有逐步成本journal，不应把它当作包含所有失败尝试的总成本。

W和C的间隔由`world.log_interval_updates`、`distill.log_interval_windows`配置，阶段最后一次更新始终记录；smoke将两者设为1。full配置的`training.remote_logging: true`在Runner启动时创建W&B在线run；smoke配置为false，保留本地JSONL且不连接W&B。运行前安装`tasks/continual_nav/requirements.txt`并完成`wandb login`；可用`WANDB_PROJECT`和`WANDB_ENTITY`指定目标项目及账号，未指定项目时为`tapd-ctm-continual-nav`。`--wandb-mode offline`可以在不上传的情况下验证W&B日志，`--wandb-mode disabled`仅用于本地测试。

W&B的`ta_W_<task>/*`使用累计世界模型优化器更新数作横轴；X/C/P及评估以累计训练环境步为横轴，P阶段另有以下游Progress步数为横轴的`progress_<task>/success_rate`。`W.collect`没有Loss，F只记录Fisher统计。`metrics.jsonl`保留完整阶段/尝试事件，W&B曲线也可能包含未提交尝试的点；正式恢复点仍以checkpoint为准。W&B run的ID和URL写入`wandb_run.json`。从旧TensorBoard运行恢复时使用原运行目录中的`resolved_config.yaml`和`--wandb-mode online`；仅精确匹配迁移前源码的checkpoint可跨越这次日志迁移。

推理导出附SHA-256侧文件，不含replay/optimizer，可通过evaluate CLI直接加载；仍要求配置指向的Maze原图存在且hash相符。完整运行目录较大，checkpoint未自动裁剪，评估深拷贝也有额外内存开销。交付5已验证CPU有限阶段及恢复接口，后续GPU有限W/X/C/F与单列P也已通过；交付6完整760步GPU smoke也已通过。首次seed 0正式长训练已在首轮W.collect完成后按用户要求停止，未完成的W.fit没有提交。

GPU有限验证使用RTX 4060 Laptop 8 GiB、torch 2.13.0+cu126：两任务模型检查24条转移，W/X/C/F主链140条训练转移及新进程恢复，独立FourRooms单列P40条转移均通过。Attention/Adapter梯度、冻结参数和Fisher检查通过；这不是完整760步smoke或full预算显存验证。

## 交付6：完整集成smoke

`verify_smoke.py`执行标准smoke的全部4轮TA和2个下游P/C/F访问。验证器只允许仓库smoke配置，唯一允许覆盖的是device；使用实际地图和固定面板，不mock评估，不修改奖励或训练算法。必须从仓库根目录执行，并提供尚不存在的输出目录：

```powershell
& 'D:/conda/envs/ctm/python.exe' -m tasks.continual_nav.verify_smoke --config tasks/continual_nav/configs/smoke.yaml --device cuda:0 --maze-manifest scientific-evidence/continual_nav/delivery_01_20260921/maze_splits.json --output-dir scientific-evidence/continual_nav/smoke_new
```

没有GPU时可显式改为`--device cpu`；不会自动回退。运行时间还包括固定面板评估、两个恢复子进程和导出复核，不能只用760训练步估算耗时。

验证器使用`AuditedRunner`观察原有事件：记录每阶段参数数量与tensor hash、Active重建、实际PPO梯度组、C教师隔离、Fisher累积与中心，以及各任务回放来源。模块参数数量可能重叠（world包含共享E），不能跨模块求和当作模型总参数量。

首次W.fit后和首次X后分别复制完整run依赖。主run完成后，两个独立Python子进程各恢复并执行下一阶段，将模型/优化器/Fisher、随机流、计数和回放原图/动作/ID与主run对应边界比较。CPU浮点要求精确一致；GPU浮点使用`atol=1e-5, rtol=1e-4`，整数/布尔、RNG、回放内容和计数仍精确比较。输出实际最大误差，不把容差通过称为逐位一致。

主run计760训练转移，world更新8次、PPO更新12次、蒸馏更新12次，Fisher共6次×8计分点。恢复探针额外训练80步，其评估另计。最终推理artifact重新加载并复核固定test结果；两任务各保存至多16步RGB轨迹。三类条件基线（单任务分别构造两份）仅初始化，检查同一主方法视觉artifact的hash、冻结状态和独立控制器，不额外训练基线。

输出目录包含：

- `run/`：主run全部标准产物，另有逐阶段`smoke_audit.json`。
- `recovery_W/`、`recovery_X/`：保留原依赖的恢复分支；各自`.json`结果与`.log`日志位于输出根目录。
- `expected_after_X.pt`、`expected_after_C.pt`：比较用状态，不是正式推理或训练入口。
- `trajectories/{task}.npz`：uint8观察`[T+1,3,84,84]`和int64动作`[T]`；`manifest.json`记录动作、奖励、结束标记及文件hash。内容是真实最终KB执行，不是生成视频或答案路径。
- `baselines/`：共享视觉基线初始化checkpoint。
- `smoke_report.json`与`smoke_report.md`：配置、来源、阶段审计、恢复容差和分项成本。报告保留在实验产物内。

只有验证器正常结束且报告`status=passed`才算完成。主run训练结束但恢复比较失败，仍不算整体验收通过。测试不要求成功率门槛，不因小预算成功率低而扩大预算。完整smoke不等于正式训练，也不证明full batch适配本机显存。

2026-09-23验收：76项CPU测试通过；RTX 4060 Laptop完整760步GPU主run、W/X独立恢复（最大张量误差均0）、最终导出复核与共享视觉基线初始化均通过。产物在`scientific-evidence/continual_nav/delivery_06_20260922/full_smoke_verified/`。其中`smoke_report.md/json`为验收报告，`acceptance_summary.json`记录最终checkpoint严格计数及此前失败验证的额外成本。验收JSON中的policy_internal_ticks另含保存末尾RGB轨迹的前向计算；正式主run计数以checkpoint为准。
