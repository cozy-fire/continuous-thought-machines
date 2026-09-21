"""Finite-window CTM with separate action-query and output synchrony."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from models.modules import SuperLinear, Squeeze
from ..config import Config, validate_config
from ..contracts import CTMState, DualState, PolicyState
from .rope import SpatialAttention


def reset_state(state: PolicyState, mask: Tensor, initial: PolicyState) -> PolicyState:
    """Replace selected slots without mutating another column or breaking valid history."""
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError("reset mask must be bool [B]")
    if isinstance(state, DualState) and isinstance(initial, DualState):
        return DualState(reset_state(state.kb, mask, initial.kb),
                         reset_state(state.active, mask, initial.active))
    if not isinstance(state, CTMState) or not isinstance(initial, CTMState):
        raise TypeError("state and initial must have the same explicit state type")
    if state.pre.shape != initial.pre.shape or state.post.shape != initial.post.shape or state.pre.shape[0] != mask.numel():
        raise ValueError("reset state shapes do not match")
    select = mask[:, None, None]
    return CTMState(torch.where(select, initial.pre, state.pre), torch.where(select, initial.post, state.post))


def detach_state(state: PolicyState) -> PolicyState:
    if isinstance(state, DualState):
        return DualState(detach_state(state.kb), detach_state(state.active))
    return CTMState(state.pre.detach(), state.post.detach())


class WindowSynchrony(nn.Module):
    def __init__(self, neurons: int, memory: int, start: int):
        super().__init__()
        pairs = torch.triu_indices(neurons, neurons)
        self.register_buffer("left", pairs[0] + start)
        self.register_buffer("right", pairs[1] + start)
        self.register_buffer("ages", torch.arange(memory-1, -1, -1, dtype=torch.float32))
        self.decay = nn.Parameter(torch.zeros(pairs.shape[1]))
        self.output_dim = pairs.shape[1]

    def forward(self, post: Tensor) -> Tensor:
        # Recompute from the finite window; age zero is the newest tick.
        weights = torch.exp(-self.decay.clamp(0, 4)[:, None] * self.ages[None, :])
        products = post[:, self.left, :] * post[:, self.right, :]
        return (products * weights).sum(-1) / weights.sum(-1).sqrt()


class Controller(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        validate_config(config)
        c = config.ctm
        self.d_model, self.memory, self.ticks = c.d_model, c.memory_length, c.ticks
        scale = math.sqrt(1 / (c.d_model + c.memory_length))
        self.start_pre = nn.Parameter(torch.empty(c.d_model, c.memory_length).uniform_(-scale, scale))
        self.start_post = nn.Parameter(torch.empty(c.d_model, c.memory_length).uniform_(-scale, scale))
        self.out_sync = WindowSynchrony(c.n_out, c.memory_length, 0)
        self.action_sync = WindowSynchrony(c.n_action, c.memory_length, c.d_model-c.n_action)
        self.attention = SpatialAttention(self.action_sync.output_dim, c.d_input,
                                          config.attention.heads, config.attention.rope_theta,
                                          config.attention.query_position)
        self.synapse = nn.Sequential(nn.Linear(c.d_input+c.d_model, 2*c.d_model), nn.GLU(),
                                     nn.LayerNorm(c.d_model), nn.Linear(c.d_model, 2*c.d_model),
                                     nn.GLU(), nn.LayerNorm(c.d_model))
        self.nlm = nn.Sequential(SuperLinear(c.memory_length, 2*c.nlm_hidden, c.d_model), nn.GLU(),
                                SuperLinear(c.nlm_hidden, 2, c.d_model), nn.GLU(), Squeeze(-1))

    def initial_state(self, batch: int, device: torch.device | str | None = None) -> CTMState:
        if batch < 1:
            raise ValueError("batch must be positive")
        if device is not None and torch.device(device) != self.start_pre.device:
            raise ValueError("move the controller to the requested device before creating state")
        # Clone the expanded view so callers cannot alias trainable initial storage.
        return CTMState(self.start_pre.unsqueeze(0).expand(batch, -1, -1).clone(),
                        self.start_post.unsqueeze(0).expand(batch, -1, -1).clone())

    def tick(self, fmap: Tensor, state: CTMState, lateral: Tensor | None = None) -> tuple[CTMState, Tensor]:
        if fmap.ndim != 4 or tuple(fmap.shape[1:]) != (128, 21, 21):
            raise ValueError("fmap must be [B,128,21,21]")
        expected = (fmap.shape[0], self.d_model, self.memory)
        if state.pre.shape != expected or state.post.shape != expected:
            raise ValueError("invalid CTM state shape")
        if fmap.dtype != torch.float32 or state.pre.dtype != torch.float32 or state.post.dtype != torch.float32:
            raise ValueError("v1 CTM uses float32 features and state")
        attended = self.attention(fmap, self.action_sync(state.post))
        previous = state.post[:, :, -1]
        if lateral is not None:
            if lateral.shape != previous.shape:
                raise ValueError("lateral input must be [B,D]")
            previous = previous + lateral
        new_pre = self.synapse(torch.cat((attended, previous), dim=-1))
        pre = torch.cat((state.pre[:, :, 1:], new_pre.unsqueeze(-1)), dim=-1)
        new_post = self.nlm(pre)
        post = torch.cat((state.post[:, :, 1:], new_post.unsqueeze(-1)), dim=-1)
        return CTMState(pre, post), new_post

    def readout(self, state: CTMState) -> Tensor:
        return self.out_sync(state.post)
