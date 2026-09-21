"""Shared data layouts. Time precedes batch; images are RGB uint8 CHW.

These records carry data only. They do not implement model or learning logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

TaskKey: TypeAlias = Literal["maze_medium", "fourrooms"]
Split: TypeAlias = Literal["train", "validation", "test"]
ImageArray: TypeAlias = NDArray[np.uint8]
TASKS: tuple[TaskKey, ...] = ("maze_medium", "fourrooms")
OBS_SHAPE = (3, 84, 84)
NUM_ACTIONS = 5


@dataclass
class CTMState:
    """pre/post: float32 [B,D,M], oldest tick first."""
    pre: Tensor
    post: Tensor
    kind: Literal["single"] = field(default="single", init=False)


@dataclass
class DualState:
    kb: CTMState
    active: CTMState
    kind: Literal["dual"] = field(default="dual", init=False)


PolicyState: TypeAlias = CTMState | DualState


@dataclass
class PolicyOutput:
    logits: Tensor  # [B,5]
    value: Tensor  # [B]; standalone KB has a separate logits/state interface.
    state: PolicyState


@dataclass
class EnvStep:
    next_obs: ImageArray  # [B,3,84,84]; reset observation for finished slots.
    transition_next_obs: ImageArray  # Real post-action frame, never reset data.
    reward_ext: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    next_episode_start: NDArray[np.bool_]
    info: list[dict[str, object]]


@dataclass(frozen=True)
class Transition:
    """World-model replay record; deliberately has no reward or answer label."""
    obs: ImageArray
    transition_next_obs: ImageArray
    action: int
    terminated: bool
    truncated: bool
    episode_start: bool
    episode_id: int
    episode_step: int
    task_key: TaskKey
    encoder_version: int
    world_model_version: int
    transition_id: int


@dataclass
class PPOBatch:
    """Tensor fields are [L,B], except obs [L,B,3,84,84]."""
    obs: Tensor
    actions: Tensor
    old_logprob: Tensor
    old_value: Tensor
    reward: Tensor
    advantages: Tensor
    returns: Tensor
    episode_start: Tensor
    terminated: Tensor
    truncated: Tensor
    initial_state: PolicyState
    bootstrap_values: Tensor
    valid_mask: Tensor


@dataclass
class SequenceBatch:
    """Masks/obs cover [U+L,B]; teacher_log_probs covers [L,B,5]."""
    obs: Tensor
    episode_start: Tensor
    valid_mask: Tensor
    burnin_mask: Tensor
    loss_mask: Tensor
    teacher_log_probs: Tensor
    teacher_snapshot_id: str
    encoder_version: int


@dataclass
class FisherSequenceBatch:
    """All fields cover [U+L,B]; score_mask selects unique learning steps.

    No reward, value, advantage or teacher target may enter this interface.
    """
    obs: Tensor
    episode_start: Tensor
    valid_mask: Tensor
    burnin_mask: Tensor
    actions: Tensor
    score_mask: Tensor
    transition_ids: Tensor


@dataclass
class FisherState:
    importance: dict[str, Tensor]
    theta_star: dict[str, Tensor]
    completed_compressions: int
    sample_count: int
    stage_key: str
    encoder_version: int


@dataclass(frozen=True)
class PhaseKey:
    """Identity only; execution and checkpoint state machines are delivery 5."""
    family: Literal["init", "ta", "pnc", "single", "seq"]
    task: TaskKey | None = None
    visit: int | None = None
    round: int | None = None
    segment: int | None = None
    phase: Literal["W", "X", "C", "F", "P"] | None = None
    subphase: Literal["collect", "fit"] | None = None

    def __post_init__(self) -> None:
        if self.family == "init":
            if any(x is not None for x in (self.task, self.visit, self.round,
                                          self.segment, self.phase, self.subphase)):
                raise ValueError("init cannot contain task/stage fields")
            return
        if self.task not in TASKS:
            raise ValueError("phase requires a known task")
        if self.family not in ("ta", "pnc", "single", "seq"):
            raise ValueError("unknown phase family")
        index = self.segment if self.family == "single" else self.visit
        if type(index) is not int or index < 0:
            raise ValueError("phase index must be a nonnegative integer")
        if self.family == "single" and self.visit is not None:
            raise ValueError("single uses segment, not visit")
        if self.family != "single" and self.segment is not None:
            raise ValueError("segment is exclusive to single")
        allowed = {"ta": ("W", "X", "C", "F"), "pnc": ("P", "C", "F"),
                   "single": ("P",), "seq": ("P",)}[self.family]
        if self.phase not in allowed:
            raise ValueError("phase is not valid for this family")
        if self.family == "ta":
            if type(self.round) is not int or self.round < 0:
                raise ValueError("TA requires a nonnegative round")
        elif self.round is not None:
            raise ValueError("round is exclusive to TA")
        if self.subphase is not None and (self.phase != "W" or self.subphase not in ("collect", "fit")):
            raise ValueError("only W accepts collect/fit subphases")

    def __str__(self) -> str:
        if self.family == "init":
            return "init"
        if self.family == "single":
            key = f"single/{self.task}/s{self.segment}/{self.phase}"
        else:
            key = f"{self.family}/v{self.visit}/{self.task}"
            if self.family == "ta":
                key += f"/r{self.round}"
            key += f"/{self.phase}"
        return key + (f"/{self.subphase}" if self.subphase else "")
