"""Batch-first recurrent states; explicit types in new tensor-only boundaries."""
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class CTMState:
    pre: Tensor
    post: Tensor


@dataclass
class LSTMState:
    h: Tensor  # [batch, layers, hidden]; native LSTM transposes at its boundary.
    c: Tensor


@dataclass
class DualState:
    kb: CTMState
    active: CTMState


State = CTMState | LSTMState | DualState


def state_map(state: State, operation) -> State:
    if isinstance(state, DualState):
        return DualState(state_map(state.kb,operation),state_map(state.active,operation))
    if isinstance(state, CTMState):
        return CTMState(operation(state.pre),operation(state.post))
    if isinstance(state, LSTMState):
        return LSTMState(operation(state.h),operation(state.c))
    raise TypeError('unknown recurrent state type')


def detached(state: State) -> State:
    return state_map(state,lambda t:t.detach().clone())


def select(state: State, indices: Tensor) -> State:
    return state_map(state,lambda t:t.index_select(0,indices.to(t.device)))


def pack_state(state: State) -> dict:
    def cpu(t):
        return t.detach().cpu().clone()
    if isinstance(state,DualState):
        return dict(kind='ctm_dual',kb=pack_state(state.kb),active=pack_state(state.active))
    if isinstance(state,CTMState):
        return dict(kind='ctm',pre=cpu(state.pre),post=cpu(state.post))
    if isinstance(state,LSTMState):
        return dict(kind='lstm',h=cpu(state.h),c=cpu(state.c))
    raise TypeError('unknown recurrent state type')


def unpack_state(value: dict, device: str | torch.device, expected_kind: str | None = None) -> State:
    # Untagged states are historical CTM tooling only, never a new LSTM state.
    kind = value.get('kind', 'ctm_dual' if 'kb' in value else 'ctm')
    if expected_kind is not None and kind != expected_kind:
        raise ValueError('recurrent state type mismatch')
    if kind == 'ctm_dual' and set(value) in ({'kind','kb','active'},{'kb','active'}):
        return DualState(unpack_state(value['kb'],device,'ctm'),unpack_state(value['active'],device,'ctm'))
    if kind == 'ctm' and set(value) in ({'kind','pre','post'},{'pre','post'}):
        return CTMState(value['pre'].clone().to(device),value['post'].clone().to(device))
    if kind == 'lstm' and set(value) == {'kind','h','c'}:
        return LSTMState(value['h'].clone().to(device),value['c'].clone().to(device))
    raise ValueError('unknown/malformed recurrent state')


def reset(state: CTMState | LSTMState, starts: Tensor, initial: CTMState | LSTMState):
    if type(state) is not type(initial):
        raise ValueError('recurrent state type mismatch')
    old = (state.pre,state.post) if isinstance(state,CTMState) else (state.h,state.c)
    new = (initial.pre,initial.post) if isinstance(initial,CTMState) else (initial.h,initial.c)
    if (starts.shape != (old[0].shape[0],) or starts.dtype != torch.bool
        or any(a.shape != b.shape for a,b in zip(old,new))):
        raise ValueError('invalid recurrent state or episode_start bool [B]')
    mask = starts[:,None,None]
    return type(state)(*(torch.where(mask,b,a) for a,b in zip(old,new)))
