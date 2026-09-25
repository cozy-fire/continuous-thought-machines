"""Explicit parameter ownership and time-preserving recurrent helpers."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import OptimizerConfig
from ..contracts import CTMState, DualState, PolicyState
from ..models import DualPolicy, SingleActorCritic, StandalonePolicy, VisionEncoder, detach_state, encode_obs


def select_state(state: PolicyState, indices: Tensor) -> PolicyState:
    if isinstance(state, DualState):
        return DualState(select_state(state.kb, indices), select_state(state.active, indices))
    return CTMState(state.pre[indices].detach(), state.post[indices].detach())


def freeze(module: nn.Module) -> None:
    module.requires_grad_(False).eval()
    for p in module.parameters():
        p.grad = None


def prepare_ppo(policy: SingleActorCritic | DualPolicy, encoder: VisionEncoder,
                *, phase: str = "P") -> list[nn.Parameter]:
    if not isinstance(policy, (SingleActorCritic, DualPolicy)):
        raise TypeError("PPO requires an actor-critic")
    if phase == "X":
        encoder.set_x_training(True)
    elif phase == "P":
        encoder.freeze()
    else:
        raise ValueError("invalid PPO phase")
    policy.requires_grad_(True).train()
    if isinstance(policy, DualPolicy):
        freeze(policy.kb)
    return [p for p in policy.parameters() if p.requires_grad]


def prepare_kb(kb: StandalonePolicy, encoder: VisionEncoder) -> list[nn.Parameter]:
    if type(kb) is not StandalonePolicy:
        raise TypeError("distillation/Fisher require standalone KB without a critic")
    encoder.freeze()
    kb.requires_grad_(True).train()
    return list(kb.parameters())


def check_optimizer(optimizer: torch.optim.Optimizer, parameters: list[nn.Parameter]) -> None:
    actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
    if len(actual) != len(set(actual)) or set(actual) != {id(p) for p in parameters}:
        raise ValueError("optimizer parameter ownership does not match this phase")


def encode_sequence(obs: Tensor, encoder: VisionEncoder, *, trainable: bool = False,
                    max_images_per_forward: int = 32) -> Tensor:
    if obs.ndim != 5 or not obs.shape[0] or not obs.shape[1] or max_images_per_forward < 1:
        raise ValueError("visual sequence must be nonempty [L,B,C,H,W] with a positive chunk size")
    length, batch_size = obs.shape[:2]
    frames = obs.reshape(length*batch_size, *obs.shape[2:])
    device = next(encoder.parameters()).device
    with torch.set_grad_enabled(trainable):
        # GroupNorm has per-image statistics, so contiguous chunks preserve each
        # frame's features while bounding retained ResNet activations in X.
        features = torch.cat([encode_obs(chunk.to(device), encoder)
                              for chunk in frames.split(max_images_per_forward)], dim=0)
    return features.reshape(length, batch_size, *features.shape[1:])


def environment_groups(batch_size: int, count: int, rng: torch.Generator) -> tuple[Tensor, ...]:
    if count < 1 or batch_size % count or rng.device.type != "cpu":
        raise ValueError("environment groups require divisible batch and a CPU RNG")
    return torch.randperm(batch_size, generator=rng).chunk(count)


def adam(parameters: list[nn.Parameter], settings: OptimizerConfig) -> torch.optim.Adam:
    return torch.optim.Adam(parameters, lr=settings.lr, betas=settings.betas,
                            eps=settings.eps, weight_decay=settings.weight_decay)


def sequence_logits(kb: StandalonePolicy, features: Tensor, episode_start: Tensor,
                    valid: Tensor, burnin: Tensor, prefix: int) -> Tensor:
    if not 0 < prefix < len(features) or any(x.shape != features.shape[:2] for x in (episode_start, valid, burnin)):
        raise ValueError("invalid burn-in sequence layout")
    expected = torch.zeros_like(burnin)
    expected[:prefix] = True
    if not torch.equal(expected, burnin):
        raise ValueError("burn-in must mark exactly the fixed prefix, including left padding")
    with torch.no_grad():
        state = kb.sequence(features[:prefix], kb.initial_state(features.shape[1]),
                            episode_start[:prefix], valid[:prefix]).state
    # The student owns its initial trace; teacher traces are never used here.
    return kb.sequence(features[prefix:], detach_state(state), episode_start[prefix:], valid[prefix:]).logits
