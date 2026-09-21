"""Reward-free per-transition policy Fisher and strict accumulated Online EWC."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ..config import Config
from ..contracts import FisherSequenceBatch, FisherState
from ..models import StandalonePolicy, VisionEncoder
from .common import encode_sequence, freeze, prepare_kb, sequence_logits


def validate_fisher(kb: StandalonePolicy, state: FisherState) -> None:
    parameters = dict(kb.named_parameters())
    if set(parameters) != set(state.importance) or set(parameters) != set(state.theta_star):
        raise ValueError("Fisher parameter names do not match KB")
    for name, p in parameters.items():
        f, center = state.importance[name], state.theta_star[name]
        if f.shape != p.shape or center.shape != p.shape or f.dtype != torch.float32 or center.dtype != torch.float32:
            raise ValueError(f"incompatible Fisher shape/dtype: {name}")
        if not bool(torch.isfinite(f).all() and torch.isfinite(center).all() and (f >= 0).all()):
            raise ValueError(f"non-finite/negative Fisher state: {name}")


def ewc_penalty(kb: StandalonePolicy, state: FisherState | None, coefficient: float) -> Tensor:
    if type(kb) is not StandalonePolicy:
        raise TypeError("EWC protects standalone KB only")
    if coefficient < 0:
        raise ValueError("negative EWC coefficient")
    device = next(kb.parameters()).device
    if state is None:
        return torch.zeros((), device=device)
    validate_fisher(kb, state)
    penalty = torch.zeros((), device=device)
    for name, p in kb.named_parameters():
        penalty = penalty+(state.importance[name].to(device).detach()*(p-state.theta_star[name].to(device).detach()).square()).sum()
    return (coefficient/2)*penalty


def squared_score_sum(log_probs: Tensor, named_parameters: dict[str, nn.Parameter]) -> dict[str, Tensor]:
    """Square each sample gradient BEFORE summing; never square a batch mean."""
    result = {name: torch.zeros_like(p, device="cpu", dtype=torch.float32) for name, p in named_parameters.items()}
    values = log_probs.reshape(-1)
    for i, log_prob in enumerate(values):
        gradients = torch.autograd.grad(log_prob, tuple(named_parameters.values()),
                                        retain_graph=i+1 < len(values), allow_unused=True)
        for (name, _), gradient in zip(named_parameters.items(), gradients):
            if gradient is not None:
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError("non-finite Fisher gradient")
                result[name].add_(gradient.detach().float().cpu().square())
    return result


def estimate_fisher(kb: StandalonePolicy, encoder: VisionEncoder,
                    sequences: Sequence[FisherSequenceBatch], config: Config) -> dict[str, Tensor]:
    ids = []
    for batch in sequences:
        if bool((batch.score_mask & (~batch.valid_mask | batch.burnin_mask)).any()):
            raise ValueError("Fisher may score only valid learning transitions")
        ids.extend(batch.transition_ids[batch.score_mask].tolist())
    if len(ids) != config.fisher.scored_samples or len(ids) != len(set(ids)) or any(i < 0 for i in ids):
        raise ValueError("Fisher scored IDs must be unique and meet the exact budget")
    prepare_kb(kb, encoder)
    kb.eval()
    kb.zero_grad(set_to_none=True)
    parameters = dict(kb.named_parameters())
    result = {name: torch.zeros_like(p, device="cpu", dtype=torch.float32) for name, p in parameters.items()}
    device, u = next(kb.parameters()).device, config.distill.burnin_env_obs
    try:
        for batch in sequences:
            # One environment graph at a time bounds recurrent Fisher memory.
            for slot in range(batch.obs.shape[1]):
                mask = batch.score_mask[u:, slot].to(device)
                if not bool(mask.any()):
                    continue
                features = encode_sequence(batch.obs[:, slot:slot+1], encoder)
                logits = sequence_logits(kb, features, batch.episode_start[:, slot:slot+1].to(device),
                    batch.valid_mask[:, slot:slot+1].to(device), batch.burnin_mask[:, slot:slot+1].to(device), u)
                log_probs = logits[:, 0].log_softmax(-1).gather(-1, batch.actions[u:, slot:slot+1].to(device)).squeeze(-1)
                sums = squared_score_sum(log_probs[mask], parameters)
                for name in result:
                    result[name].add_(sums[name])
    finally:
        # F computes gradients with autograd.grad but never changes weights or BN.
        freeze(kb)
    return {name: value/len(ids) for name, value in result.items()}


def update_online_fisher(kb: StandalonePolicy, current: dict[str, Tensor], previous: FisherState | None,
                         *, decay: float, sample_count: int, stage_key: str, encoder_version: int) -> FisherState:
    if type(kb) is not StandalonePolicy or not 0 <= decay <= 1 or sample_count < 1 or not stage_key or encoder_version < 0:
        raise ValueError("invalid Online EWC update")
    center = {name: p.detach().float().cpu().clone() for name, p in kb.named_parameters()}
    state = FisherState({name: f.detach().cpu().clone() for name, f in current.items()}, center,
                        1, sample_count, stage_key, encoder_version)
    validate_fisher(kb, state)
    if previous is not None:
        validate_fisher(kb, previous)
        state.importance = {name: decay*previous.importance[name].detach().cpu()+f for name, f in state.importance.items()}
        state.completed_compressions = previous.completed_compressions+1
    # Always replace the center with the current compressed KB, not the old center.
    return state
