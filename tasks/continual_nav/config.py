"""Strict, versioned configuration and budget arithmetic (no training side effects)."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import types
from typing import get_args, get_origin, get_type_hints

import yaml

METHODS = ("tapd_ctm_visual_revisit", "single_task_ctm_shared_vision",
           "sequential_ppo_shared_vision", "pnc_without_exploration_distill_shared_vision")


@dataclass(frozen=True)
class ObservationConfig:
    shape: tuple[int, ...] = (3, 84, 84)
    scale: str = "uint8_to_float_div255"
    actions: int = 5


@dataclass(frozen=True)
class EnvironmentConfig:
    max_steps: int = 300
    reward: str = "success_time_discount"
    maze_root: str = "data/mazes/medium"
    tile_size: int = 8


@dataclass(frozen=True)
class VisionConfig:
    backbone: str = "resnet34-2"
    pretrained: bool = False
    norm: str = "groupnorm32"


@dataclass(frozen=True)
class CTMConfig:
    d_model: int = 512
    d_input: int = 128
    ticks: int = 2
    memory_length: int = 40
    synapse: str = "two_linear_glu_blocks"
    nlm_hidden: int = 16
    deep_nlm: bool = True
    nlm_layernorm: bool = False
    neuron_selection: str = "first-last"
    n_out: int = 32
    n_action: int = 32
    dropout: float = 0.0


@dataclass(frozen=True)
class AttentionConfig:
    heads: int = 4
    rope_theta: float = 10000.0
    query_position: tuple[int, int] = (10, 10)


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    num_envs: int = 8
    device: str = "cuda:0"
    precision: str = "float32"
    vector_backend: str = "sync"
    remote_logging: bool = False


@dataclass(frozen=True)
class AgnosticConfig:
    visits: int = 4
    rounds_per_task: int = 2


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "Adam"
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-5
    weight_decay: float = 0.0


@dataclass(frozen=True)
class SIGRegConfig:
    enabled: bool = True
    lambda_: float = 0.02
    directions: int = 256
    knots: int = 17
    t_max: float = 3.0


@dataclass(frozen=True)
class ExplorationConfig:
    steps_per_round: int = 200000
    history_length: int = 20
    similarity_threshold: float = 0.999
    min_gap_weight: float = 0.2
    projector_hidden: int = 512
    projection_dim: int = 128
    visual_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(
        name="AdamW", eps=1e-8, weight_decay=1e-4))
    visual_max_grad_norm: float = 1.0
    diagnostic_episodes: int = 4


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 50
    num_minibatches: int = 4
    update_epochs: int = 1
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    vf_coef: float = 0.25
    entropy_coef: float = 0.01
    clip_value_loss: bool = False
    normalize_advantage: bool = True
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    lr_schedule: str = "linear_per_stage"


@dataclass(frozen=True)
class DistillConfig:
    agnostic_steps_per_round: int = 300000
    learning_steps: int = 50
    log_interval_windows: int = 10
    burnin_ticks: int = 40
    burnin_env_obs: int = 20
    minibatches: int = 4
    update_epochs: int = 1
    temperature: float = 1.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    max_grad_norm: float = 0.5
    lr_schedule: str = "constant"


@dataclass(frozen=True)
class PNCConfig:
    visits: int = 3
    progress_steps: int = 2500000
    compress_steps: int = 1000000


@dataclass(frozen=True)
class EWCConfig:
    lambda_: float = 250.0
    agnostic_decay: float = 0.1
    pnc_decay: float = 0.3


@dataclass(frozen=True)
class FisherConfig:
    collect_steps: int = 4096
    scored_samples: int = 1024


@dataclass(frozen=True)
class EvaluationConfig:
    backend: str = "subprocess"
    num_envs: int = 16
    validation_episodes: int = 200
    test_episodes: int = 200
    drift_episodes: int = 32
    interval_steps: int = 100000
    maze_validation_hashes: int = 512
    fourrooms_train_seed_stop: int = 1000000
    fourrooms_validation_seed_start: int = 2000000
    fourrooms_test_seed_start: int = 3000000


@dataclass(frozen=True)
class Config:
    schema_version: int = 2
    task_order: tuple[str, ...] = ("maze_medium", "fourrooms")
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    ctm: CTMConfig = field(default_factory=CTMConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    agnostic: AgnosticConfig = field(default_factory=AgnosticConfig)
    sigreg: SIGRegConfig = field(default_factory=SIGRegConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    pnc: PNCConfig = field(default_factory=PNCConfig)
    ewc: EWCConfig = field(default_factory=EWCConfig)
    fisher: FisherConfig = field(default_factory=FisherConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


class StrictLoader(yaml.SafeLoader):
    """Duplicate keys are errors: a second value must not silently win."""


def _mapping(loader: StrictLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"non-string or duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _read(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"cyclic config inheritance: {path}")
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=StrictLoader)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str):
        raise ValueError("extends must be a path string")
    base = _read(path.parent / parent, (*seen, path))

    def merge(target: dict, override: dict) -> None:
        for key, value in override.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                merge(target[key], value)
            else:
                target[key] = value
    merge(base, raw)
    return base


def _parse(annotation: type, value: object, path: str) -> object:
    if is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected mapping")
        hints = get_type_hints(annotation)
        names = {f.name.rstrip("_"): f.name for f in fields(annotation)}
        if set(value) != set(names):
            raise ValueError(f"{path}: unknown={set(value)-set(names)}, missing={set(names)-set(value)}")
        return annotation(**{name: _parse(hints[name], value[key], f"{path}.{key}")
                             for key, name in names.items()})
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is tuple:
        if not isinstance(value, (tuple, list)) or (args[-1] is not Ellipsis and len(value) != len(args)):
            raise ValueError(f"{path}: invalid sequence")
        return tuple(_parse(args[0] if args[-1] is Ellipsis else args[i], v, f"{path}[{i}]")
                     for i, v in enumerate(value))
    if origin is types.UnionType and type(None) in args:
        return None if value is None else _parse(args[0], value, path)
    if annotation is float and type(value) in (int, float) and math.isfinite(value):
        return float(value)
    if annotation in (int, bool, str) and type(value) is annotation:
        return value
    raise ValueError(f"{path}: expected {annotation}, got {value!r}")


def resolved_dict(config: Config) -> dict:
    def convert(obj: object) -> object:
        if isinstance(obj, dict):
            return {k.rstrip("_"): convert(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [convert(v) for v in obj]
        return obj
    return convert(asdict(config))


def validate_config(c: Config) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    # Structural choices are fixed for v2; changing them requires a new contract.
    fixed = ("schema_version", "task_order", "observation", "vision", "ctm", "attention")
    defaults = Config()
    for name in fixed:
        require(getattr(c, name) == getattr(defaults, name), f"unsupported v2 {name}")
    require(c.environment.max_steps == 300 and c.environment.tile_size == 8
            and c.environment.reward == "success_time_discount", "unsupported environment semantics")
    require(bool(c.environment.maze_root), "maze_root cannot be empty")
    require(c.training.precision == "float32" and c.training.vector_backend == "sync",
            "v2 requires float32 and synchronous environments")
    require(c.training.device == "cpu" or (c.training.device.startswith("cuda:")
            and c.training.device[5:].isdigit()), "device must be cpu or an explicit cuda:N")
    require(bool(c.training.seeds) and len(set(c.training.seeds)) == len(c.training.seeds)
            and min(c.training.seeds) >= 0, "training seeds must be distinct nonnegative integers")
    positives = {
        "num_envs": c.training.num_envs, "TA visits": c.agnostic.visits,
        "rounds_per_task": c.agnostic.rounds_per_task, "P&C visits": c.pnc.visits,
        "history length": c.exploration.history_length,
        "diagnostic episodes": c.exploration.diagnostic_episodes,
        "rollout_steps": c.ppo.rollout_steps, "ppo minibatches": c.ppo.num_minibatches,
        "learning_steps": c.distill.learning_steps, "distill minibatches": c.distill.minibatches,
        "distill log interval": c.distill.log_interval_windows,
        "Fisher samples": c.fisher.scored_samples, "eval interval": c.evaluation.interval_steps,
        "validation episodes": c.evaluation.validation_episodes, "test episodes": c.evaluation.test_episodes,
        "drift episodes": c.evaluation.drift_episodes, "evaluation num_envs": c.evaluation.num_envs,
    }
    for name, value in positives.items():
        require(value > 0, f"{name} must be positive")
    n = c.training.num_envs
    require(n % c.ppo.num_minibatches == 0 and n % c.distill.minibatches == 0,
            "num_envs must be divisible by both minibatch counts")
    budgets = (c.exploration.steps_per_round,
               c.distill.agnostic_steps_per_round, c.pnc.progress_steps,
               c.pnc.compress_steps, c.fisher.collect_steps)
    require(all(b > 0 and b % n == 0 for b in budgets),
            "every transition budget must be positive and divisible by num_envs")
    require(c.fisher.scored_samples <= c.fisher.collect_steps, "too many Fisher scored samples")
    require(c.distill.burnin_ticks == c.ctm.memory_length
            and c.distill.burnin_env_obs * c.ctm.ticks == c.distill.burnin_ticks,
            "burn-in must equal M ticks (20 observations at 2 ticks)")
    require(c.ppo.update_epochs == c.distill.update_epochs == 1, "v2 uses one update epoch")
    require(c.exploration.history_length == 20 and c.exploration.projector_hidden == 512
            and c.exploration.projection_dim == 128 and c.exploration.diagnostic_episodes == 4,
            "unsupported visual revisit structure")
    require(0 <= c.exploration.similarity_threshold < 1 and 0 < c.exploration.min_gap_weight <= 1,
            "invalid visual revisit penalty")
    require(c.evaluation.backend in ("serial", "subprocess"), "unsupported evaluation backend")
    require(c.sigreg.enabled and c.sigreg.directions > 0 and c.sigreg.knots >= 2
            and c.sigreg.t_max > 0 and c.sigreg.lambda_ >= 0, "invalid SIGReg configuration")
    for optimizer, expected in ((c.exploration.visual_optimizer, "AdamW"), (c.ppo.optimizer, "Adam"),
                                (c.distill.optimizer, "Adam")):
        require(optimizer.name == expected and optimizer.lr > 0 and optimizer.eps > 0
                and optimizer.weight_decay >= 0 and all(0 <= b < 1 for b in optimizer.betas),
                "invalid optimizer configuration")
    require(c.distill.lr_schedule == "constant" and c.ppo.lr_schedule == "linear_per_stage",
            "unsupported learning-rate schedule")
    require(0 <= c.ppo.gamma <= 1 and 0 <= c.ppo.gae_lambda <= 1 and 0 < c.ppo.clip_coef < 1,
            "invalid PPO gamma/lambda/clip")
    require(not c.ppo.clip_value_loss and c.ppo.normalize_advantage and c.ppo.target_kl is None,
            "unsupported PPO variant")
    require(min(c.exploration.visual_max_grad_norm, c.ppo.max_grad_norm, c.distill.max_grad_norm) > 0
            and c.ppo.vf_coef >= 0 and c.ppo.entropy_coef >= 0 and c.distill.temperature == 1,
            "invalid loss/gradient configuration")
    require(c.ewc.lambda_ >= 0 and 0 <= c.ewc.agnostic_decay <= 1 and 0 <= c.ewc.pnc_decay <= 1,
            "invalid EWC configuration")
    ev = c.evaluation
    require(ev.maze_validation_hashes == 512 and ev.validation_episodes <= 512
            and ev.drift_episodes <= ev.validation_episodes, "invalid validation panel sizes")
    require((ev.fourrooms_train_seed_stop, ev.fourrooms_validation_seed_start,
             ev.fourrooms_test_seed_start) == (1000000, 2000000, 3000000), "v1 seed partitions are fixed")
    require(ev.validation_episodes <= 1000000 and ev.test_episodes <= 1000000, "seed panels overlap")


def load_config(path: str | Path) -> Config:
    c = _parse(Config, _read(Path(path)), "config")
    validate_config(c)
    return c


def config_hash(config: Config) -> str:
    return hashlib.sha256(json.dumps(resolved_dict(config), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def save_config(config: Config, path: str | Path) -> None:
    validate_config(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(resolved_dict(config), sort_keys=False), encoding="utf-8")


def config_from_dict(value: dict) -> Config:
    """Reconstruct the same strict, fully resolved configuration from a checkpoint."""
    config = _parse(Config, value, "config")
    validate_config(config)
    return config


def budget_summary(c: Config) -> dict[str, int]:
    """Count real transitions, not ticks, replays or optimizer updates."""
    rounds = c.agnostic.visits * len(c.task_order) * c.agnostic.rounds_per_task
    segments = c.pnc.visits * len(c.task_order)
    ta = rounds * (c.exploration.steps_per_round
                   + c.distill.agnostic_steps_per_round + c.fisher.collect_steps)
    pnc = segments * (c.pnc.progress_steps + c.pnc.compress_steps + c.fisher.collect_steps)
    ppo = segments * c.pnc.progress_steps
    return dict(ta_rounds=rounds, pnc_segments=segments, shared_ta_steps=ta,
                downstream_pnc_steps=pnc, main_steps=ta+pnc, sequential_ppo_steps=ppo,
                single_task_steps=c.pnc.visits*c.pnc.progress_steps,
                all_methods_steps=ta+2*pnc+2*ppo,
                x_joint_updates=rounds*math.ceil(c.exploration.steps_per_round/(c.training.num_envs*c.ppo.rollout_steps))*c.ppo.num_minibatches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate config and print budgets; never starts training.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional resolved YAML output")
    args = parser.parse_args()
    c = load_config(args.config)
    if args.output:
        save_config(c, args.output)
    print(json.dumps({"config_hash": config_hash(c), "budgets": budget_summary(c)}, indent=2))


if __name__ == "__main__":
    main()
