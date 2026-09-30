"""Explicit copy, slot selection and differentiable reset of finite CTM traces."""
from __future__ import annotations
import torch
from torch import Tensor
from ..contracts import CTMState, DualState, PolicyState


def detach_clone_state(state: PolicyState) -> PolicyState:
    """Capture a rollout origin without storage aliases or the previous window's graph."""
    if isinstance(state, DualState):
        return DualState(detach_clone_state(state.kb), detach_clone_state(state.active))
    if not isinstance(state, CTMState):
        raise TypeError("expected an explicit CTMState or DualState")
    return CTMState(state.pre.detach().clone(), state.post.detach().clone())


def select_state(state: PolicyState, indices: Tensor) -> PolicyState:
    if isinstance(state, DualState):
        return DualState(select_state(state.kb, indices), select_state(state.active, indices))
    return CTMState(state.pre.index_select(0, indices), state.post.index_select(0, indices))


def replace_slots(state: PolicyState, indices: Tensor, replacement: PolicyState) -> PolicyState:
    if isinstance(state, DualState) and isinstance(replacement, DualState):
        return DualState(replace_slots(state.kb, indices, replacement.kb),
                         replace_slots(state.active, indices, replacement.active))
    if not isinstance(state, CTMState) or not isinstance(replacement, CTMState):
        raise TypeError("replacement must match the state type")
    # Out-of-place index_copy preserves BPTT and never writes into a saved initial_state.
    return CTMState(state.pre.index_copy(0, indices, replacement.pre),
                    state.post.index_copy(0, indices, replacement.post))


def reset_state(state: PolicyState, mask: Tensor, initial: PolicyState) -> PolicyState:
    if isinstance(state, DualState) and isinstance(initial, DualState):
        return DualState(reset_state(state.kb, mask, initial.kb),
                         reset_state(state.active, mask, initial.active))
    if not isinstance(state, CTMState) or not isinstance(initial, CTMState):
        raise TypeError("initial state must match the state type")
    if mask.dtype != torch.bool or mask.shape != (state.pre.shape[0],) or mask.device != state.pre.device:
        raise ValueError("episode_start must be bool [B] on the state device")
    if state.pre.shape != initial.pre.shape or state.post.shape != initial.post.shape:
        raise ValueError("reset state shapes must match")
    # Learned initial parameters retain gradients at valid episode boundaries.
    return CTMState(torch.where(mask[:, None, None], initial.pre, state.pre),
                    torch.where(mask[:, None, None], initial.post, state.post))
