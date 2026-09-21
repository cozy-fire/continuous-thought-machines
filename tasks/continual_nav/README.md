# Continual navigation — delivery 1

本目录当前交付配置、公共数据类、数据划分和环境；模型与学习器在后续交付实现。关键 Python 逻辑采用英文注释。

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

## 后续衔接

交付2使用`Config`、`CTMState`、`DualState`和`PolicyOutput`实现模型；不得重新定义这些类型。阶段状态机、checkpoint、PPO、蒸馏、Fisher估计与高误差replay尚未实现。`PhaseKey`只是阶段身份数据类，不会执行任何训练。
