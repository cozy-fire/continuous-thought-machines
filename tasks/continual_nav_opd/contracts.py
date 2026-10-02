"""Time-major interfaces: current rollout only, no history/burn-in prefix."""
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

TaskKey: TypeAlias = Literal["maze_medium", "fourrooms"]
Split: TypeAlias = Literal["train", "validation", "test"]
Phase: TypeAlias = Literal["P", "C", "F"]
NUM_ACTIONS = 5
STUDENT_SHAPE = (3,84,84)

@dataclass(frozen=True)
class ObservationPair:
    student_rgb: NDArray[np.uint8]  # [B,3,84,84]; environment instances omit B.
    teacher_obs: NDArray[np.uint8]  # Maze [B,3,19,19], FourRooms [B,7,7,3].

@dataclass(frozen=True)
class EnvStep:
    transition_next: ObservationPair  # True post-action/terminal image.
    next_obs: ObservationPair  # Auto-reset image for finished slots.
    reward_ext: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    next_episode_start: NDArray[np.bool_]
    info: list[dict[str,object]]

@dataclass
class CTMState:
    pre: Tensor  # [B,512,40]; oldest tick first.
    post: Tensor

@dataclass
class DualState:
    kb: CTMState
    active: CTMState

PolicyState: TypeAlias = CTMState | DualState

@dataclass
class PolicySequenceOutput:
    logits: Tensor  # [L,B,5], no critic.
    state: PolicyState

@dataclass
class SequenceBatch:
    obs: Tensor  # uint8 [L,B,3,84,84], 1<=L<=50, only new observations.
    episode_start: Tensor  # bool [L,B].
    valid_mask: Tensor
    target_mask: Tensor
    teacher_probs: Tensor  # float32 [L,B,5].
    actions: Tensor  # int64 [L,B].
    transition_ids: Tensor  # Only padding uses -1.
    initial_state: PolicyState  # Detached independent copy BEFORE first observation.
    source_snapshot_id: str
    task: TaskKey  # External execution context only; never encoded as a model input.
    ticks: int  # Actual collection budget, checked before gradient replay.

    @property
    def loss_mask(self):
        return self.valid_mask & self.target_mask

@dataclass
class FisherState:
    importance: dict[str,Tensor]  # Complete KB encoder/controller/actor names.
    theta_star: dict[str,Tensor]
    completed_compressions: int
    sample_count: int
    stage_key: str
