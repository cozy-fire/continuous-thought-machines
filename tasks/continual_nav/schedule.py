"""Deterministic method schedules and independently restorable random streams."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import random

import numpy as np
import torch

from .config import Config, validate_config
from .contracts import PhaseKey, TASKS

METHODS = ("tapd_ctm_visual_revisit", "single_task_ctm_shared_vision", "sequential_ppo_shared_vision",
           "pnc_without_exploration_distill_shared_vision")
STREAMS = ("model_init", "policy_action", "env", "ppo_shuffle", "fisher", "sigreg", "evaluation")


@dataclass(frozen=True)
class Stage:
    ordinal: int
    key: PhaseKey
    transitions: int

    def record(self) -> dict:
        return dict(ordinal=self.ordinal, key=str(self.key), identity=asdict(self.key),
                    transitions=self.transitions)


def expand_stages(config: Config, method: str, task: str | None = None) -> list[Stage]:
    validate_config(config)
    if method not in METHODS or (method == METHODS[1] and task not in TASKS) or (method != METHODS[1] and task is not None):
        raise ValueError("invalid method/run-task combination")
    stages = [Stage(0, PhaseKey("init"), 0)]
    def add(key, steps):
        stages.append(Stage(len(stages), key, steps))
    if method == METHODS[0]:
        for visit in range(config.agnostic.visits):
            for current_task in config.task_order:
                for round_index in range(config.agnostic.rounds_per_task):
                    for phase, count in (("X", config.exploration.steps_per_round),
                        ("C", config.distill.agnostic_steps_per_round), ("F", config.fisher.collect_steps)):
                        add(PhaseKey("ta", current_task, visit, round_index, phase=phase), count)
    for visit in range(config.pnc.visits):
        for current_task in ((task,) if method == METHODS[1] else config.task_order):
            if method in (METHODS[1], METHODS[2]):
                family = "single" if method == METHODS[1] else "seq"
                add(PhaseKey(family, current_task, visit=None if family == "single" else visit,
                             segment=visit if family == "single" else None, phase="P"), config.pnc.progress_steps)
            else:
                for phase, count in (("P", config.pnc.progress_steps), ("C", config.pnc.compress_steps),
                                     ("F", config.fisher.collect_steps)):
                    add(PhaseKey("pnc", current_task, visit, phase=phase), count)
    return stages


def derive_seed(seed: int, method: str, task: str | None, stream: str, ordinal: int = 0) -> int:
    if seed < 0 or method not in METHODS or stream not in STREAMS or ordinal < 0:
        raise ValueError("invalid deterministic RNG identity")
    task_code = TASKS.index(task) if task is not None else 2
    return int(np.random.SeedSequence([seed, METHODS.index(method), task_code, STREAMS.index(stream)+1, ordinal])
               .generate_state(1, dtype=np.uint64)[0]) % (2**63-1)


def visit_identity(stage: Stage) -> tuple[str, int | None]:
    key = stage.key
    return key.family, key.segment if key.family == "single" else key.visit


def ends_visit(stages: list[Stage], index: int) -> bool:
    return stages[index].key.family != "init" and (
        index+1 == len(stages) or visit_identity(stages[index]) != visit_identity(stages[index+1]))


class RandomStreams:
    def __init__(self, seed: int, method: str, task: str | None):
        self.identity = (seed, method, task)
        self.seeds = {name: derive_seed(seed, method, task, name) for name in STREAMS}
        self.torch = {name: torch.Generator().manual_seed(self.seeds[name])
                      for name in ("policy_action", "ppo_shuffle", "fisher", "sigreg", "evaluation")}
        self.numpy = {name: np.random.default_rng(self.seeds[name]) for name in ("env",)}

    def state_dict(self) -> dict:
        return dict(seeds=self.seeds, torch={k: g.get_state() for k, g in self.torch.items()},
                    numpy={k: g.bit_generator.state for k, g in self.numpy.items()})

    def load_state_dict(self, state: dict) -> None:
        if state["seeds"] != self.seeds or set(state["torch"]) != set(self.torch) or set(state["numpy"]) != set(self.numpy):
            raise ValueError("RNG stream identity mismatch")
        for name, generator in self.torch.items():
            generator.set_state(state["torch"][name].cpu())
        for name, generator in self.numpy.items():
            generator.bit_generator.state = state["numpy"][name]

    @contextmanager
    def model_initialization(self, ordinal: int):
        with isolated_rng():
            torch.manual_seed(derive_seed(*self.identity, "model_init", ordinal))
            yield


@contextmanager
def isolated_rng():
    """Evaluation/model construction cannot perturb process-global training RNGs."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
