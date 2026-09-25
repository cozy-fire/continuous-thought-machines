"""Independent KB, single-column baselines and KB-to-Active tick-level laterals."""
from __future__ import annotations

from copy import deepcopy
from typing import TypeVar

import torch
from torch import Tensor, nn

from ..config import Config
from ..contracts import CTMState, DualState, PolicyOutput, PolicySequenceOutput, PolicyState
from .ctm import Controller, reset_state


def _head(input_dim: int, output_dim: int) -> nn.Sequential:
    layers = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64),
                           nn.ReLU(), nn.Linear(64, output_dim))
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
    return layers


class StandalonePolicy(nn.Module):
    """Knowledge-base policy. Deliberately contains no critic, encoder or adapter."""
    def __init__(self, config: Config):
        super().__init__()
        self.controller = Controller(config)
        self.actor = _head(self.controller.out_sync.output_dim, config.observation.actions)

    def initial_state(self, batch: int, device: torch.device | str | None = None) -> CTMState:
        return self.controller.initial_state(batch, device)

    def _advance(self, fmap: Tensor, state: CTMState, episode_start: Tensor) -> CTMState:
        if not isinstance(state, CTMState):
            raise TypeError("standalone policy requires CTMState")
        state = reset_state(state, episode_start, self.initial_state(fmap.shape[0]))
        for _ in range(self.controller.ticks):
            state, _ = self.controller.tick(fmap, state)
        return state

    def step(self, fmap: Tensor, state: CTMState, episode_start: Tensor) -> tuple[Tensor, CTMState]:
        state = self._advance(fmap, state, episode_start)
        return self.actor(self.controller.readout(state)), state

    def sequence(self, fmaps: Tensor, state: CTMState, episode_start: Tensor,
                 valid_mask: Tensor | None = None) -> PolicySequenceOutput:
        return _sequence(self, fmaps, state, episode_start, valid_mask)


class SingleActorCritic(StandalonePolicy):
    """Used for Active and single-column PPO; the critic is never stored in KB."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.critic = _head(self.controller.out_sync.output_dim, 1)

    def step(self, fmap: Tensor, state: CTMState, episode_start: Tensor) -> PolicyOutput:
        state = self._advance(fmap, state, episode_start)
        sync = self.controller.readout(state)
        return PolicyOutput(self.actor(sync), self.critic(sync).squeeze(-1), state)


class LateralAdapter(nn.Module):
    def __init__(self, width: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, width, bias=False)
        nn.init.orthogonal_(self.projection.weight, gain=1.0)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, kb_activation: Tensor) -> Tensor:
        # Detaching KB is independent of adapter autograd; its gate/weights still learn.
        return self.gate.tanh() * self.projection(self.norm(kb_activation.detach()))


class DualPolicy(nn.Module):
    def __init__(self, config: Config, kb: StandalonePolicy, *, kb_ready: bool = False):
        super().__init__()
        if type(kb) is not StandalonePolicy:
            raise TypeError("KB must be StandalonePolicy without a critic")
        self.kb = kb
        self.kb.requires_grad_(False).eval()
        for parameter in self.kb.parameters():
            parameter.grad = None
        # Active starts from its own random initialization, never a copy of KB.
        self.active = SingleActorCritic(config)
        self.adapter = LateralAdapter(config.ctm.d_model)
        self.register_buffer("kb_ready", torch.tensor(kb_ready, dtype=torch.bool))

    def train(self, mode: bool = True) -> DualPolicy:
        super().train(mode)
        self.kb.eval()
        return self

    def initial_state(self, batch: int, device: torch.device | str | None = None) -> DualState:
        with torch.no_grad():
            kb = self.kb.initial_state(batch, device)
        return DualState(kb, self.active.initial_state(batch, device))

    def step(self, fmap: Tensor, state: DualState, episode_start: Tensor) -> PolicyOutput:
        if not isinstance(state, DualState):
            raise TypeError("dual policy requires DualState")
        state = reset_state(state, episode_start, self.initial_state(fmap.shape[0]))
        kb_state, active_state = state.kb, state.active
        ready = bool(self.kb_ready.item())
        for _ in range(self.active.controller.ticks):
            # KB must advance first: Active receives this tick's new KB activation.
            with torch.no_grad():
                kb_state, kb_new = self.kb.controller.tick(fmap, kb_state)
            lateral = self.adapter(kb_new) if ready else None
            active_state, _ = self.active.controller.tick(fmap, active_state, lateral)
        sync = self.active.controller.readout(active_state)
        return PolicyOutput(self.active.actor(sync), self.active.critic(sync).squeeze(-1),
                            DualState(kb_state, active_state))

    def sequence(self, fmaps: Tensor, state: DualState, episode_start: Tensor,
                 valid_mask: Tensor | None = None) -> PolicySequenceOutput:
        return _sequence(self, fmaps, state, episode_start, valid_mask)


def _sequence(policy: StandalonePolicy | DualPolicy, fmaps: Tensor, state: PolicyState,
              episode_start: Tensor, valid_mask: Tensor | None) -> PolicySequenceOutput:
    if fmaps.ndim != 5 or tuple(fmaps.shape[2:]) != (128, 21, 21) or fmaps.shape[0] == 0:
        raise ValueError("sequence features must be nonempty [L,B,128,21,21]")
    shape = fmaps.shape[:2]
    if episode_start.shape != shape or episode_start.dtype != torch.bool:
        raise ValueError("episode_start must be bool [L,B]")
    if valid_mask is None:
        valid_mask = torch.ones_like(episode_start)
    if valid_mask.shape != shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool [L,B]")
    logits, values = [], []
    for t in range(shape[0]):
        valid = valid_mask[t]
        # Mask inputs before computing; padding may contain arbitrary stored values.
        features = torch.where(valid[:, None, None, None], fmaps[t], 0.0)
        output = policy.step(features, state, episode_start[t] & valid)
        if isinstance(output, PolicyOutput):
            step_logits, proposed_state = output.logits, output.state
            values.append(torch.where(valid, output.value, 0.0))
        else:
            step_logits, proposed_state = output
        # Padding neither resets nor advances trace. No implicit detach occurs here.
        state = reset_state(proposed_state, ~valid, state)
        logits.append(torch.where(valid[:, None], step_logits, 0.0))
    return PolicySequenceOutput(torch.stack(logits), torch.stack(values) if values else None, state)


ModuleT = TypeVar("ModuleT", bound=nn.Module)


def frozen_copy(module: ModuleT) -> ModuleT:
    """Storage-isolated snapshot; the caller creates fresh recurrent sampling state."""
    snapshot = deepcopy(module)
    snapshot.requires_grad_(False).eval()
    for parameter in snapshot.parameters():
        parameter.grad = None
    # Vision snapshots must also lock their forward/autograd policy.
    from .vision import VisionEncoder
    if isinstance(snapshot, VisionEncoder):
        snapshot.freeze()
    return snapshot
