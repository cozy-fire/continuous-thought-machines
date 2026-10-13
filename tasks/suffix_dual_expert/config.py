"""One explicit protocol; no task registry or legacy configuration classes."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json

import yaml

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
TEACHER_SHA = 'd76c748579b2c1eeaa089ae0d41570d091490d355d970cb7628c08dbc5f0461c'
TICKS = {'maze_medium': 5, 'fourrooms': 2}
MODEL_SPEC = {
    'observation': {'shape': [3, 84, 84], 'scale': 'uint8_to_float_div255', 'actions': 5},
    'vision': {'backbone': 'resnet34-2', 'pretrained': False, 'norm': 'groupnorm32'},
    'ctm': {'d_model': 512, 'd_input': 128, 'ticks_by_task': TICKS, 'memory_length': 40,
            'synapse': 'two_linear_glu_blocks', 'nlm_hidden': 16, 'deep_nlm': True,
            'nlm_layernorm': False, 'neuron_selection': 'first-last', 'n_out': 32,
            'n_action': 32, 'dropout': 0.0},
    'attention': {'heads': 4, 'rope_theta': 10000.0, 'query_position': [10, 10]},
}


@dataclass(frozen=True)
class Config:
    maze_root: str = 'data/mazes/medium'
    teacher_checkpoint: str = 'tasks/suffix_dual_expert/assets/fourrooms_teacher.pt'
    teacher_sha256: str = TEACHER_SHA
    test_manifest: str = 'tasks/suffix_dual_expert/assets/maze_test800.json'
    lengths: tuple[int, ...] = (5, 10, 15, 20, 25, 30)
    pool_sizes: tuple[int, ...] = (5, 10, 20, 40, 80, 160, 320)
    first_updates: tuple[int, ...] = (2000, 2000, 1000, 1000, 1000, 1000, 1000)
    suffix_updates: tuple[int, ...] = (200, 200, 500, 500, 500, 500, 500)
    sequences: int = 100
    suffix_microbatch: int = 25
    candidate_batch: int = 100
    max_candidates: int = 100000
    encoder_chunk: int = 32
    checkpoint_encoder: bool = True
    fisher_steps: int = 4096
    fisher_samples: int = 1024
    slots_per_task: int = 16
    window: int = 50
    groups: int = 4
    compress_microbatch: int = 4
    compress_windows: int = 800
    save_windows: int = 10
    ewc_enabled: bool = True
    ewc_lambda: float = 250.0
    lr: float = 0.0001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 0.00001
    clip: float = 0.5
    seed: int = 0
    device: str = 'cuda:0'
    eval_maze: int = 800
    eval_fourrooms: int = 800
    eval_batch: int = 32

    def __post_init__(self) -> None:
        if not self.lengths or any(v not in (5, 10, 15, 20, 25, 30) for v in self.lengths):
            raise ValueError('lengths must be an ordered subset of 5/10/15/20/25/30')
        if tuple(sorted(set(self.lengths))) != self.lengths or self.lengths[0] != 5:
            raise ValueError('curriculum must start at 5 and increase')
        if not self.pool_sizes or tuple(sorted(set(self.pool_sizes))) != self.pool_sizes or self.pool_sizes[0] < 1:
            raise ValueError('map pools must be positive and increasing')
        if any(len(v) != len(self.pool_sizes) or any(n < 1 for n in v)
               for v in (self.first_updates, self.suffix_updates)):
            raise ValueError('each pool needs a positive update budget')
        if min(self.sequences, self.suffix_microbatch, self.candidate_batch, self.encoder_chunk) < 1:
            raise ValueError('batch sizes must be positive')
        if self.sequences % self.suffix_microbatch or self.max_candidates < self.sequences:
            raise ValueError('invalid suffix microbatch or candidate cap')
        if self.groups != 4 or self.slots_per_task < 4 or self.slots_per_task % 4:
            raise ValueError('four joint groups require a multiple of four slots per task')
        if self.compress_microbatch < 1 or (self.slots_per_task//self.groups) % self.compress_microbatch or not 1 <= self.window <= 50:
            raise ValueError('Compress microbatch must divide the task group size; window must be 1..50')
        if min(self.compress_windows, self.save_windows, self.fisher_samples, self.eval_batch) < 1:
            raise ValueError('window/sample counts must be positive')
        if self.fisher_steps % (2 * self.slots_per_task) or not 0 < self.fisher_samples <= self.fisher_steps:
            raise ValueError('Fisher decisions must divide into slots and cover scored samples')
        if not 0 < self.eval_maze <= 800 or not 0 < self.eval_fourrooms <= 800:
            raise ValueError('fixed evaluation panels support 1..800 episodes')
        if min(self.lr, self.eps, self.clip, self.ewc_lambda) <= 0:
            raise ValueError('invalid optimizer/EWC parameters')
        if not isinstance(self.checkpoint_encoder,bool):
            raise ValueError('checkpoint_encoder must be boolean')
        if not isinstance(self.ewc_enabled,bool):
            raise ValueError('ewc_enabled must be boolean')

    def schedule(self) -> list[dict]:
        return [dict(index=i, length=length, maps=maps, updates=updates)
                for i, (length, maps, updates) in enumerate(
                    (length, maps, updates) for length in self.lengths
                    for maps, updates in zip(self.pool_sizes,
                        self.first_updates if length == 5 else self.suffix_updates))]

    def budget(self) -> dict:
        return dict(maze_stages=len(self.schedule()),
                    maze_adam=sum(s['updates'] for s in self.schedule()),
                    fisher_decisions=self.fisher_steps, fisher_scores=self.fisher_samples,
                    compress_windows=self.compress_windows,
                    compress_adam=self.groups * self.compress_windows,
                    compress_decisions=2 * self.slots_per_task * self.window * self.compress_windows,
                    decisions_per_task=self.slots_per_task * self.window * self.compress_windows,
                    evaluation_episodes=2 * (self.eval_maze + self.eval_fourrooms))

    def resolved(self) -> dict:
        return json.loads(json.dumps(asdict(self)))


def path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def load_config(file: str | Path | None = None) -> Config:
    raw = yaml.safe_load(Path(file).read_text(encoding='utf-8')) if file else {}
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(Config)}:
        raise ValueError('unknown configuration field')
    for name in ('lengths', 'pool_sizes', 'first_updates', 'suffix_updates', 'betas'):
        if name in raw:
            raw[name] = tuple(raw[name])
    return Config(**raw)
