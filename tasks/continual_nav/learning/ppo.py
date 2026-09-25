"""Recurrent PPO-clip with separate timeout bootstrap and episode-boundary masks."""
from __future__ import annotations

import torch
import math
from collections.abc import Callable
from torch import Tensor
from torch.distributions import Categorical

from ..config import Config
from ..contracts import PPOBatch
from ..envs import VectorEnvAdapter
from ..models import DualPolicy, SingleActorCritic, VisionEncoder, SigregProjector, SIGReg
from .common import adam, check_optimizer, encode_sequence, environment_groups, prepare_ppo, select_state


@torch.no_grad()
def compute_gae(reward: Tensor, value: Tensor, bootstrap: Tensor, terminated: Tensor,
                truncated: Tensor, valid: Tensor, gamma: float, gae_lambda: float) -> tuple[Tensor, Tensor]:
    if any(x.shape != reward.shape for x in (value, bootstrap, terminated, truncated, valid)) or reward.ndim != 2:
        raise ValueError("GAE fields must all be [L,B]")
    advantages = torch.zeros_like(value)
    carry = torch.zeros_like(value[0])
    for t in reversed(range(len(reward))):
        delta = reward[t] + gamma*(~terminated[t])*bootstrap[t] - value[t]
        # Timeout bootstraps its final frame, but never propagates reset-episode GAE.
        carry = torch.where(valid[t], delta + gamma*gae_lambda*(~(terminated[t] | truncated[t]))*carry, 0.)
        advantages[t] = carry
    return advantages, torch.where(valid, advantages+value, 0.)


def ppo_loss(logits: Tensor, value: Tensor, actions: Tensor, old_logprob: Tensor,
             advantages: Tensor, returns: Tensor, valid: Tensor, config: Config) -> tuple[Tensor, dict]:
    if not bool(valid.any()):
        raise ValueError("PPO minibatch contains no valid transitions")
    distribution = Categorical(logits=logits[valid])
    new_logprob = distribution.log_prob(actions[valid])
    old = old_logprob[valid].detach()
    adv = advantages[valid].detach()
    if config.ppo.normalize_advantage:
        adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    ratio = (new_logprob-old).exp()
    policy_loss = torch.maximum(-adv*ratio, -adv*ratio.clamp(1-config.ppo.clip_coef, 1+config.ppo.clip_coef)).mean()
    value_loss = 0.5*(value[valid]-returns[valid].detach()).square().mean()
    entropy = distribution.entropy().mean()
    total = policy_loss+config.ppo.vf_coef*value_loss-config.ppo.entropy_coef*entropy
    return total, dict(policy_loss=policy_loss.item(), value_loss=value_loss.item(), entropy=entropy.item(),
                       max_logprob_difference=(new_logprob-old).abs().max().item(),
                       approximate_kl=((ratio-1)-(new_logprob-old)).mean().item())


def make_ppo_optimizer(policy: SingleActorCritic | DualPolicy, encoder: VisionEncoder,
                       config: Config) -> torch.optim.Adam:
    return adam(prepare_ppo(policy, encoder, phase="P"), config.ppo.optimizer)


def _visual_parameters(encoder: VisionEncoder, projector: SigregProjector) -> dict[str, Tensor]:
    return {**{f"encoder.{name}": p for name, p in encoder.named_parameters()},
            **{f"projector.{name}": p for name, p in projector.named_parameters()}}


def make_x_optimizer(policy: DualPolicy, encoder: VisionEncoder, projector: SigregProjector,
                     config: Config, visual_state: dict | None) -> torch.optim.AdamW:
    policy_parameters = prepare_ppo(policy, encoder, phase="X")
    projector.requires_grad_(True).train()
    p, v = config.ppo.optimizer, config.exploration.visual_optimizer
    optimizer = torch.optim.AdamW([
        dict(name="policy", params=policy_parameters, lr=p.lr, betas=p.betas, eps=p.eps, weight_decay=p.weight_decay),
        dict(name="visual", params=list(_visual_parameters(encoder, projector).values()), lr=v.lr,
             betas=v.betas, eps=v.eps, weight_decay=v.weight_decay)])
    if visual_state:
        names = _visual_parameters(encoder, projector)
        if set(visual_state) != set(names):
            raise ValueError("visual optimizer state does not match the shared encoder/projector")
        for name, parameter in names.items():
            optimizer.state[parameter] = {key: value.to(parameter.device) if isinstance(value, Tensor) else value
                                          for key, value in visual_state[name].items()}
    return optimizer


def visual_optimizer_state(optimizer: torch.optim.Optimizer, encoder: VisionEncoder,
                           projector: SigregProjector) -> dict:
    return {name: {key: value.detach().cpu().clone() if isinstance(value, Tensor) else value
                   for key, value in optimizer.state[parameter].items()}
            for name, parameter in _visual_parameters(encoder, projector).items()}


def set_stage_learning_rate(optimizer: torch.optim.Optimizer, config: Config, *,
                            completed_rollouts: int, total_rollouts: int) -> float:
    if total_rollouts < 1 or not 0 <= completed_rollouts <= total_rollouts:
        raise ValueError("invalid PPO stage progress")
    lr = config.ppo.optimizer.lr*(1-completed_rollouts/total_rollouts)
    for group in optimizer.param_groups:
        if group.get("name") != "visual":
            group["lr"] = lr
    return lr


def train_ppo(batch: PPOBatch, policy: SingleActorCritic | DualPolicy, encoder: VisionEncoder,
              optimizer: torch.optim.Optimizer, config: Config, *, rng: torch.Generator,
              phase: str = "P", projector: SigregProjector | None = None,
              regularizer: SIGReg | None = None, sigreg_rng: torch.Generator | None = None) -> dict:
    policy_parameters = prepare_ppo(policy, encoder, phase=phase)
    if phase == "X":
        if projector is None or regularizer is None or sigreg_rng is None:
            raise ValueError("X joint PPO requires projector, SIGReg and its RNG")
        projector.requires_grad_(True).train()
        visual_parameters = list(_visual_parameters(encoder, projector).values())
    else:
        visual_parameters = []
    check_optimizer(optimizer, policy_parameters+visual_parameters)
    device = next(policy.parameters()).device
    updates, metrics = 0, {}
    for _ in range(config.ppo.update_epochs):
        for cpu_indices in environment_groups(batch.obs.shape[1], config.ppo.num_minibatches, rng):
            indices = cpu_indices.to(device)
            features = encode_sequence(batch.obs[:, cpu_indices], encoder, trainable=phase == "X",
                                       max_images_per_forward=config.ppo.encoder_microbatch_images)
            def field(name):
                return getattr(batch, name)[:, cpu_indices].to(device)
            output = policy.sequence(features, select_state(batch.initial_state, indices),
                                     field("episode_start"), field("valid_mask"))
            ppo_total, metrics = ppo_loss(output.logits, output.value, field("actions"), field("old_logprob"),
                                    field("advantages"), field("returns"), field("valid_mask"), config)
            if phase == "X":
                valid_features = features[field("valid_mask")]
                latent = projector(valid_features)
                directions = regularizer.sample_directions(latent.shape[-1], sigreg_rng, device)
                sigreg = regularizer(latent, directions) / len(latent)
                loss = ppo_total + config.sigreg.lambda_ * sigreg
            else:
                sigreg = None
                loss = ppo_total
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy_parameters, config.ppo.max_grad_norm,
                                                   error_if_nonfinite=True)
            if visual_parameters:
                visual_norm = torch.nn.utils.clip_grad_norm_(visual_parameters,
                    config.exploration.visual_max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            metrics.update(loss=loss.item(), grad_norm=float(norm))
            if sigreg is not None:
                metrics.update(ppo_loss=ppo_total.item(), sigreg=sigreg.item(), visual_grad_norm=float(visual_norm))
    return {**metrics, "updates": updates}


def run_ppo_stage(envs: VectorEnvAdapter, policy: SingleActorCritic | DualPolicy, encoder: VisionEncoder, config: Config,
                   *, phase: str, steps: int, action_rng: torch.Generator, minibatch_rng: torch.Generator,
                   projector: SigregProjector | None = None, regularizer: SIGReg | None = None,
                   sigreg_rng: torch.Generator | None = None, visual_state: dict | None = None,
                   start_transition_id: int = 0, on_rollout: Callable[[int, dict], None] | None = None) -> dict:
    """One uninterrupted X/P stage; global scheduling/checkpointing belongs to delivery 5."""
    from ..data.rollout import PPOCollector
    if steps < 1 or steps % envs.num_envs:
        raise ValueError("stage budget must be a positive multiple of num_envs")
    collector = PPOCollector(envs, policy, encoder, config, phase=phase,
                             start_transition_id=start_transition_id)
    optimizer = (make_x_optimizer(policy, encoder, projector, config, visual_state) if phase == "X"
                 else make_ppo_optimizer(policy, encoder, config))
    capacity = envs.num_envs*config.ppo.rollout_steps
    rollouts, consumed, updates = math.ceil(steps/capacity), 0, 0
    metrics = {}
    for index in range(rollouts):
        set_stage_learning_rate(optimizer, config, completed_rollouts=index, total_rollouts=rollouts)
        size = min(capacity, steps-consumed)
        batch = collector.collect(size, action_rng=action_rng)
        metrics = train_ppo(batch, policy, encoder, optimizer, config, rng=minibatch_rng,
                            phase=phase, projector=projector, regularizer=regularizer, sigreg_rng=sigreg_rng)
        consumed += size
        updates += metrics["updates"]
        if on_rollout is not None:
            on_rollout(consumed, {**metrics, "stage_updates": updates, **collector.revisit_metrics()})
        # Keep the detached collect trace, not a trace recomputed with updated weights.
    set_stage_learning_rate(optimizer, config, completed_rollouts=rollouts, total_rollouts=rollouts)
    if phase == "X":
        encoder.freeze()
        projector.requires_grad_(False).eval()
    diagnostic = collector.revisit_metrics()
    return dict(transitions=consumed, updates=updates, rollouts=rollouts, final_lr=optimizer.param_groups[0]["lr"],
                next_transition_id=collector.next_transition_id, last=metrics,
                visual_state=visual_optimizer_state(optimizer, encoder, projector) if phase == "X" else None,
                revisit=diagnostic, diagnostic_episodes=collector.diagnostics.selected() if phase == "X" else [])
