"""Teacher-to-student action KL after detached burn-in, with Online EWC."""
from __future__ import annotations

from collections.abc import Callable
import torch
from torch import Tensor

from ..config import Config
from ..contracts import FisherState, SequenceBatch
from ..envs import VectorEnvAdapter
from ..models import DualPolicy, StandalonePolicy, VisionEncoder
from .common import adam, check_optimizer, encode_sequence, environment_groups, prepare_kb, sequence_logits
from .fisher import ewc_penalty


def action_kl(student_logits: Tensor, teacher_log_probs: Tensor, valid: Tensor) -> Tensor:
    if student_logits.shape != teacher_log_probs.shape or valid.shape != student_logits.shape[:2] or not bool(valid.any()):
        raise ValueError("invalid action-KL layout/empty learning targets")
    teacher = teacher_log_probs[valid].detach()
    student = student_logits[valid].log_softmax(-1)
    return (teacher.exp()*(teacher-student)).sum(-1).mean()


def make_distill_optimizer(kb: StandalonePolicy, encoder: VisionEncoder, config: Config) -> torch.optim.Adam:
    return adam(prepare_kb(kb, encoder), config.distill.optimizer)


def train_distill(batch: SequenceBatch, kb: StandalonePolicy, encoder: VisionEncoder,
                  ewc: FisherState | None, optimizer: torch.optim.Optimizer, config: Config,
                  *, rng: torch.Generator) -> dict:
    if batch.encoder_version != int(encoder.encoder_version):
        raise ValueError("teacher targets belong to a different encoder version")
    if not torch.equal(batch.loss_mask, batch.valid_mask & ~batch.burnin_mask):
        raise ValueError("learning mask must exclude exactly padding and burn-in")
    parameters = prepare_kb(kb, encoder)
    check_optimizer(optimizer, parameters)
    device, u = next(kb.parameters()).device, config.distill.burnin_env_obs
    updates, metrics = 0, {}
    for _ in range(config.distill.update_epochs):
        for indices in environment_groups(batch.obs.shape[1], config.distill.minibatches, rng):
            features = encode_sequence(batch.obs[:, indices], encoder)
            logits = sequence_logits(kb, features, batch.episode_start[:, indices].to(device),
                batch.valid_mask[:, indices].to(device), batch.burnin_mask[:, indices].to(device), u)
            kl = action_kl(logits, batch.teacher_log_probs[:, indices].to(device), batch.loss_mask[u:, indices].to(device))
            penalty = ewc_penalty(kb, ewc, config.ewc.lambda_)
            loss = kl+penalty
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite distillation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, config.distill.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            metrics = dict(loss=loss.item(), kl=kl.item(), ewc=penalty.item(), grad_norm=float(norm))
    return {**metrics, "updates": updates}


def run_compress_stage(envs: VectorEnvAdapter, teacher: DualPolicy, kb: StandalonePolicy,
                        encoder: VisionEncoder, config: Config,
                        ewc: FisherState | None, *, steps: int, action_rng: torch.Generator,
                        minibatch_rng: torch.Generator, start_transition_id: int = 0,
                        on_window: Callable[[int, int, dict], None] | None = None) -> dict:
    from ..data.rollout import SequenceCollector
    if steps < 1 or steps % envs.num_envs:
        raise ValueError("compression budget must be a positive multiple of num_envs")
    # Snapshot before the optimizer enables or changes the live student's parameters.
    collector = SequenceCollector(envs, teacher, encoder, config, mode="C", start_transition_id=start_transition_id)
    optimizer = make_distill_optimizer(kb, encoder, config)
    consumed, updates, windows = 0, 0, 0
    metrics = {}
    while consumed < steps:
        size = min(envs.num_envs*config.distill.learning_steps, steps-consumed)
        batch = collector.collect(size, action_rng=action_rng)
        metrics = train_distill(batch, kb, encoder, ewc, optimizer, config, rng=minibatch_rng)
        consumed += size
        updates += metrics["updates"]
        windows += 1
        if on_window is not None and (windows % config.distill.log_interval_windows == 0 or consumed == steps):
            on_window(consumed, updates, metrics)
    return dict(transitions=consumed, updates=updates, windows=windows, last=metrics,
                teacher_snapshot_id=collector.snapshot_id, next_transition_id=collector.next_transition_id,
                kb_ready=True)
