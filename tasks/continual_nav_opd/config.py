"""Strict typed configuration, with no model/data/remote side effects."""
from __future__ import annotations

import argparse
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
METHOD = "ctm_pnc_opd"
TASKS = ("maze_medium", "fourrooms")


@dataclass(frozen=True)
class ObservationConfig:
    shape: tuple[int, int, int]
    scale: str
    actions: int

@dataclass(frozen=True)
class EnvironmentConfig:
    max_steps: int
    reward: str
    maze_root: str
    tile_size: int
    map_cache: str = "disk"

@dataclass(frozen=True)
class VisionConfig:
    backbone: str
    pretrained: bool
    norm: str

@dataclass(frozen=True)
class TaskTicks:
    maze_medium: int
    fourrooms: int

    def for_task(self, task: str) -> int:
        if task not in TASKS:
            raise ValueError(f"unknown CTM execution task: {task!r}")
        return getattr(self, task)


@dataclass(frozen=True)
class CTMConfig:
    d_model: int
    d_input: int
    ticks_by_task: TaskTicks
    memory_length: int
    synapse: str
    nlm_hidden: int
    deep_nlm: bool
    nlm_layernorm: bool
    neuron_selection: str
    n_out: int
    n_action: int
    dropout: float

@dataclass(frozen=True)
class AttentionConfig:
    heads: int
    rope_theta: float
    query_position: tuple[int, int]

@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    num_envs: int
    device: str
    precision: str
    vector_backend: str
    remote_logging: bool

@dataclass(frozen=True)
class TaskBudget:
    compress_steps: int
    progress_steps: int = 0  # FourRooms only; Maze P is counted in optimizer updates.

@dataclass(frozen=True)
class TaskBudgets:
    maze_medium: TaskBudget
    fourrooms: TaskBudget

@dataclass(frozen=True)
class PNCConfig:
    visits: int
    task_budgets: TaskBudgets

@dataclass(frozen=True)
class MazeProgressConfig:
    pool_sizes: tuple[int, ...]
    pool_updates: tuple[int, ...]
    sequences_per_update: int
    microbatch_sequences: int
    decisions: int

@dataclass(frozen=True)
class FisherConfig:
    collect_steps: int
    scored_samples: int

@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float

@dataclass(frozen=True)
class OptimizationConfig:
    learning_steps: int
    minibatches: int
    update_epochs: int
    encoder_microbatch_images: int
    optimizer: OptimizerConfig
    lr_schedule: str
    max_grad_norm: float
    e3_cache: bool = False
    ctm_compile: str = "disabled"
    # Zero replays the full optimizer group; nonzero splits only independent
    # environment sequences and accumulates gradients before one Adam step.
    sequence_microbatch_envs: int = 0

@dataclass(frozen=True)
class DistillConfig:
    temperature: float

@dataclass(frozen=True)
class MazeTeacherConfig:
    type: str
    algorithm_version: str
    tie_break_order: tuple[int, int, int, int]
    target_distribution: str

@dataclass(frozen=True)
class FourRoomsTeacherConfig:
    checkpoint: str

@dataclass(frozen=True)
class TeachersConfig:
    maze_medium: MazeTeacherConfig
    fourrooms: FourRoomsTeacherConfig

@dataclass(frozen=True)
class EWCConfig:
    lambda_: float
    decay: float

@dataclass(frozen=True)
class EvaluationConfig:
    backend: str
    num_envs: int
    validation_episodes: int
    test_episodes: int
    maze_validation_hashes: int
    fourrooms_train_seed_stop: int
    fourrooms_validation_seed_start: int
    fourrooms_test_seed_start: int

@dataclass(frozen=True)
class LoggingConfig:
    interval_windows: int

@dataclass(frozen=True)
class Config:
    method: str
    schema_version: int
    sequence_protocol: str
    task_order: tuple[str, str]
    observation: ObservationConfig
    environment: EnvironmentConfig
    vision: VisionConfig
    ctm: CTMConfig
    attention: AttentionConfig
    training: TrainingConfig
    pnc: PNCConfig
    maze_progress: MazeProgressConfig
    fisher: FisherConfig
    optimization: OptimizationConfig
    distill: DistillConfig
    teachers: TeachersConfig
    ewc: EWCConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig


class StrictLoader(yaml.SafeLoader):
    pass

def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"invalid/duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result

StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def _read(path: Path, chain=()):
    path = path.resolve(strict=True)
    if path in chain:
        raise ValueError("cyclic config inheritance")
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=StrictLoader)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str) or not parent:
        raise ValueError("extends must name a YAML file")
    return _merge(_read(path.parent/parent, (*chain, path)), raw)


def _convert(raw, kind, location):
    if is_dataclass(kind):
        hints = get_type_hints(kind)
        names = {f.name: ("lambda" if f.name == "lambda_" else f.name) for f in fields(kind)}
        required = {names[f.name] for f in fields(kind) if f.default is MISSING}
        if not isinstance(raw, dict) or not required <= set(raw) or not set(raw) <= set(names.values()):
            raise ValueError(f"{location}: missing or unknown fields; expected {sorted(names.values())}")
        return kind(**{name: _convert(raw[key], hints[name], f"{location}.{key}") for name, key in names.items() if key in raw})
    if get_origin(kind) is tuple:
        types = get_args(kind)
        if len(types) == 2 and types[1] is Ellipsis:
            if not isinstance(raw, (tuple, list)) or not raw:
                raise ValueError(f"{location}: expected a nonempty sequence")
            return tuple(_convert(v, types[0], location) for v in raw)
        if not isinstance(raw, (tuple, list)) or len(raw) != len(types):
            raise ValueError(f"{location}: wrong tuple length")
        return tuple(_convert(v, t, location) for v, t in zip(raw, types))
    if kind is float and type(raw) in (int, float) and math.isfinite(raw):
        return float(raw)
    if kind in (int, bool, str) and type(raw) is kind:
        return raw
    raise ValueError(f"{location}: invalid {kind.__name__} value")


def parse_config(raw: dict) -> Config:
    config = _convert(raw, Config, "config")
    validate_config(config)
    return config


def validate_config(c: Config) -> None:
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    require((c.method, c.schema_version, c.sequence_protocol) == (METHOD, 4, "maze_onpolicy5_v1"), "unsupported method/schema/sequence protocol")
    require(c.task_order == TASKS, "task_order must be maze_medium, fourrooms")
    require(c.observation == ObservationConfig((3,84,84), "uint8_to_float_div255", 5), "invalid student interface")
    require(c.environment.max_steps == 300 and c.environment.reward == "success_time_discount" and c.environment.tile_size == 8 and bool(c.environment.maze_root), "invalid environment")
    require(c.environment.map_cache == "memory", "v4 requires the shared immutable map cache")
    require(c.vision == VisionConfig("resnet34-2", False, "groupnorm32"), "invalid visual architecture")
    require(c.ctm == CTMConfig(512,128,c.ctm.ticks_by_task,40,"two_linear_glu_blocks",16,True,False,"first-last",32,32,0.0), "invalid CTM architecture")
    require(all(type(c.ctm.ticks_by_task.for_task(task)) is int and c.ctm.ticks_by_task.for_task(task) > 0
                for task in TASKS), "task ticks must be positive integers")
    require(c.ctm.ticks_by_task.maze_medium == 5, "Maze requires ticks=5")
    m = c.maze_progress
    require(len(m.pool_sizes) == len(m.pool_updates) and min(m.pool_sizes) > 0 and min(m.pool_updates) > 0
            and all(a < b for a,b in zip(m.pool_sizes,m.pool_sizes[1:])), "invalid Maze curriculum")
    require(m.decisions == 5 and m.sequences_per_update == 100 and m.microbatch_sequences == 5,
            "Maze requires 100 independent five-decision sequences, microbatch5")
    require(c.attention == AttentionConfig(4,10000.0,(10,10)), "invalid attention")
    t, opt = c.training, c.optimization
    require(t.seed >= 0 and t.num_envs > 0 and (t.device == "cpu" or re_cuda_device(t.device)), "invalid training seed/slots/device")
    require(t.precision == "float32" and t.vector_backend == "sync", "invalid precision/backend")
    require(c.pnc.visits > 0 and c.fisher.collect_steps > 0 and 0 < c.fisher.scored_samples <= c.fisher.collect_steps, "invalid visits/Fisher")
    for task in TASKS:
        budget = getattr(c.pnc.task_budgets, task)
        require(budget.compress_steps > 0 and budget.compress_steps % t.num_envs == 0, f"{task}: C budget must be positive/divisible by slots")
        require((budget.progress_steps == 0 if task == 'maze_medium' else budget.progress_steps > 0 and budget.progress_steps % t.num_envs == 0),
                f"{task}: P budget must use the task's counting protocol")
    require(c.fisher.collect_steps % t.num_envs == 0, "F budget must divide by slots")
    require(0 < opt.learning_steps <= 50 and opt.minibatches > 0 and t.num_envs % opt.minibatches == 0 and opt.update_epochs == 1 and opt.encoder_microbatch_images > 0, "invalid sequence/minibatch settings")
    require(type(opt.e3_cache) is bool and opt.ctm_compile in ('disabled','default','reduce-overhead'), "invalid E3/CTM compile settings")
    require(type(opt.sequence_microbatch_envs) is int and 0 <= opt.sequence_microbatch_envs <= t.num_envs,
            "sequence microbatch must be 0 (disabled) or a positive environment-slot count within num_envs")
    require(opt.optimizer == OptimizerConfig("Adam",1e-4,(0.9,0.999),1e-5,0.0) and opt.lr_schedule == "constant" and opt.max_grad_norm == 0.5, "invalid Adam/clipping settings")
    require(c.distill.temperature == 1 and c.ewc == EWCConfig(250.0,0.3), "invalid distillation/EWC")
    require(c.teachers.maze_medium == MazeTeacherConfig("shortest_path_bfs","bfs_grid_v1",(0,1,2,3),"one_hot") and bool(c.teachers.fourrooms.checkpoint), "invalid teacher protocol")
    e = c.evaluation
    require(e.backend in ("serial","subprocess") and min(e.num_envs,e.validation_episodes,e.test_episodes)>0 and e.maze_validation_hashes == 512 and e.validation_episodes <= 512 and e.test_episodes <= 1000000, "invalid evaluation")
    require((e.fourrooms_train_seed_stop,e.fourrooms_validation_seed_start,e.fourrooms_test_seed_start)==(1000000,2000000,3000000), "invalid seed partitions")
    require(c.logging.interval_windows > 0, "invalid logging interval")


def re_cuda_device(value: str) -> bool:
    return value.startswith("cuda:") and value[5:].isdigit()


def resolved_dict(config: Config) -> dict:
    raw = asdict(config)
    raw["ewc"]["lambda"] = raw["ewc"].pop("lambda_")
    return raw


def config_hash(config: Config) -> str:
    return raw_config_hash(resolved_dict(config))


def raw_config_hash(raw: dict) -> str:
    # Historical inference artifacts hash their original fields, before new defaults.
    # Stage-boundary restoration still requires the current config AND source hashes.
    return hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def load_config(path: str | Path) -> Config:
    path = Path(path)
    return parse_config(_read(path if path.is_absolute() else REPO_ROOT/path))


def budget_summary(config: Config) -> dict:
    values = {phase:0 for phase in ("P","C","F")}
    for task in config.task_order:
        budget = getattr(config.pnc.task_budgets,task)
        steps = sum(config.maze_progress.pool_updates)*config.maze_progress.sequences_per_update*5 if task == 'maze_medium' else budget.progress_steps
        values["P"] += steps*config.pnc.visits
        values["C"] += budget.compress_steps*config.pnc.visits
        values["F"] += config.fisher.collect_steps*config.pnc.visits
    return dict(**values,total=sum(values.values()),stage_count=6*config.pnc.visits,
                env_steps_are_upper_bounds=True,maze_P_updates_per_visit=sum(config.maze_progress.pool_updates),
                maze_P_updates=sum(config.maze_progress.pool_updates)*config.pnc.visits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(dict(method=config.method,schema=config.schema_version,sequence_protocol=config.sequence_protocol,
                          ticks_by_task=asdict(config.ctm.ticks_by_task),memory_ticks=config.ctm.memory_length,
                          config_hash=config_hash(config),budget=budget_summary(config)),indent=2))

if __name__ == "__main__":
    main()
