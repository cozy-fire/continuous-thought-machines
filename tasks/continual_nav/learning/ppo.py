"""Recurrent PPO-clip with separate timeout bootstrap and episode-boundary masks."""
from __future__ import annotations

import torch
import math
from torch import Tensor
from torch.distributions import Categorical

from ..config import Config
from ..contracts import PPOBatch
from ..data.replay import HighErrorBuilder
from ..envs import VectorEnvAdapter
from ..models import DualPolicy, SingleActorCritic, VisionEncoder, WorldModel
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
    return adam(prepare_ppo(policy, encoder), config.ppo.optimizer)


def set_stage_learning_rate(optimizer: torch.optim.Optimizer, config: Config, *,
                            completed_rollouts: int, total_rollouts: int) -> float:
    if total_rollouts < 1 or not 0 <= completed_rollouts <= total_rollouts:
        raise ValueError("invalid PPO stage progress")
    lr = config.ppo.optimizer.lr*(1-completed_rollouts/total_rollouts)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_ppo(batch: PPOBatch, policy: SingleActorCritic | DualPolicy, encoder: VisionEncoder,
              optimizer: torch.optim.Optimizer, config: Config, *, rng: torch.Generator) -> dict:
    parameters = prepare_ppo(policy, encoder)
    check_optimizer(optimizer, parameters)
    device = next(policy.parameters()).device
    updates, metrics = 0, {}
    for _ in range(config.ppo.update_epochs):
        for cpu_indices in environment_groups(batch.obs.shape[1], config.ppo.num_minibatches, rng):
            indices = cpu_indices.to(device)
            features = encode_sequence(batch.obs[:, cpu_indices], encoder)
            def field(name):
                return getattr(batch, name)[:, cpu_indices].to(device)
            output = policy.sequence(features, select_state(batch.initial_state, indices),
                                     field("episode_start"), field("valid_mask"))
            loss, metrics = ppo_loss(output.logits, output.value, field("actions"), field("old_logprob"),
                                    field("advantages"), field("returns"), field("valid_mask"), config)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, config.ppo.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            metrics.update(loss=loss.item(), grad_norm=float(norm))
    return {**metrics, "updates": updates}


def run_ppo_stage(envs: VectorEnvAdapter, policy: SingleActorCritic | DualPolicy, encoder: VisionEncoder, config: Config,
                   *, phase: str, steps: int, action_rng: torch.Generator, minibatch_rng: torch.Generator,
                   world: WorldModel | None = None, high_error: HighErrorBuilder | None = None,
                   start_transition_id: int = 0) -> dict:
    """One uninterrupted X/P stage; global scheduling/checkpointing belongs to delivery 5."""
    from ..data.rollout import PPOCollector
    if steps < 1 or steps % envs.num_envs:
        raise ValueError("stage budget must be a positive multiple of num_envs")
    if high_error is not None and high_error.expected_count != steps:
        raise ValueError("X pool and PPO stage budgets must match")
    collector = PPOCollector(envs, policy, encoder, config, phase=phase, world=world,
                             high_error=high_error, start_transition_id=start_transition_id)
    optimizer = make_ppo_optimizer(policy, encoder, config)
    capacity = envs.num_envs*config.ppo.rollout_steps
    rollouts, consumed, updates = math.ceil(steps/capacity), 0, 0
    metrics = {}
    for index in range(rollouts):
        set_stage_learning_rate(optimizer, config, completed_rollouts=index, total_rollouts=rollouts)
        size = min(capacity, steps-consumed)
        batch = collector.collect(size, action_rng=action_rng)
        metrics = train_ppo(batch, policy, encoder, optimizer, config, rng=minibatch_rng)
        consumed += size
        updates += metrics["updates"]
        # Keep the detached collect trace, not a trace recomputed with updated weights.
    set_stage_learning_rate(optimizer, config, completed_rollouts=rollouts, total_rollouts=rollouts)
    pool = high_error.finish() if high_error is not None else None
    return dict(transitions=consumed, updates=updates, rollouts=rollouts, final_lr=optimizer.param_groups[0]["lr"],
                next_transition_id=collector.next_transition_id, last=metrics,
                pool_manifest=str(pool.manifest) if pool is not None else None)
