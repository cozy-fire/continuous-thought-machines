"""Stage-local streaming collectors. Recurrent sampling state survives learner updates."""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical

from ..config import Config
from ..contracts import FisherSequenceBatch, PPOBatch, PolicyOutput, SequenceBatch, Transition
from ..envs import VectorEnvAdapter
from ..models import DualPolicy, SingleActorCritic, StandalonePolicy, VisionEncoder, WorldModel, detach_state, encode_obs, frozen_copy
from ..learning.common import prepare_ppo
from ..learning.ppo import compute_gae
from .replay import HighErrorBuilder


def sample_actions(logits: Tensor, rng: torch.Generator) -> Tensor:
    if rng.device.type != "cpu":
        raise ValueError("action sampling requires an explicit CPU generator")
    return torch.multinomial(logits.softmax(-1).cpu(), 1, generator=rng).squeeze(-1)


class PPOCollector:
    def __init__(self, envs: VectorEnvAdapter, policy: SingleActorCritic | DualPolicy,
                 encoder: VisionEncoder, config: Config, *, phase: str,
                 world: WorldModel | None = None, high_error: HighErrorBuilder | None = None,
                 start_transition_id: int = 0):
        if phase not in ("X", "P") or start_transition_id < 0:
            raise ValueError("invalid PPO phase/transition identity")
        if phase == "X" and (world is None or world.encoder is not encoder):
            raise ValueError("X requires the world model owning the same shared encoder")
        if phase == "P" and (world is not None or high_error is not None):
            raise ValueError("P cannot call the world model or populate the X pool")
        self.envs, self.policy, self.encoder, self.config = envs, policy, encoder, config
        self.phase, self.world, self.high_error = phase, world, high_error
        self.device = next(policy.parameters()).device
        if world is not None:
            world.freeze()
        if phase == "P":
            encoder.freeze(permanent=True)
        prepare_ppo(policy, encoder)
        self.obs, infos = envs.reset()
        if high_error is not None and (any(i["task_key"] != high_error.task for i in infos)
            or high_error.encoder_version != int(encoder.encoder_version)
            or high_error.world_version != int(world.world_model_version)):
            raise ValueError("X builder task/version mismatch")
        self.starts = np.ones(envs.num_envs, bool)
        self.state = detach_state(policy.initial_state(envs.num_envs))
        self.next_transition_id = start_transition_id

    @torch.no_grad()
    def collect(self, steps: int, *, action_rng: torch.Generator) -> PPOBatch:
        b = self.envs.num_envs
        if steps < 1 or steps % b or steps > b*self.config.ppo.rollout_steps:
            raise ValueError("rollout requires an exact positive budget no larger than rollout_steps")
        initial = detach_state(self.state)
        rows = {name: [] for name in ("obs", "next", "action", "logprob", "value", "reward", "start", "term", "trunc", "bootstrap")}
        for _ in range(steps//b):
            obs = torch.from_numpy(self.obs.copy())
            output = self.policy.step(encode_obs(obs.to(self.device), self.encoder), self.state,
                                      torch.from_numpy(self.starts).to(self.device))
            actions = sample_actions(output.logits, action_rng)
            result = self.envs.step(actions.numpy())
            final = torch.from_numpy(result.transition_next_obs.copy())
            # Bootstrap from a functional state fork. Never assign its advanced trace
            # back to the collector, and never reset it before a timeout final frame.
            bootstrap = self.policy.step(encode_obs(final.to(self.device), self.encoder), output.state,
                                         torch.zeros(b, dtype=torch.bool, device=self.device)).value
            if self.phase == "X":
                reward, errors = self.world.curiosity(obs.to(self.device), final.to(self.device), actions.to(self.device))
                if self.high_error is not None:
                    for slot, info in enumerate(result.info):
                        self.high_error.offer(Transition(self.obs[slot], result.transition_next_obs[slot], int(actions[slot]),
                            bool(result.terminated[slot]), bool(result.truncated[slot]), bool(self.starts[slot]),
                            int(info["episode_id"])*b+slot, int(info["episode_step"]), info["task_key"],
                            int(self.encoder.encoder_version), int(self.world.world_model_version),
                            self.next_transition_id+slot), float(errors[slot]))
            else:
                reward = torch.from_numpy(result.reward_ext.copy())
            values = (obs, final, actions, Categorical(logits=output.logits).log_prob(actions.to(self.device)).cpu(),
                      output.value.cpu(), reward.cpu(), torch.from_numpy(self.starts.copy()),
                      torch.from_numpy(result.terminated.copy()), torch.from_numpy(result.truncated.copy()), bootstrap.cpu())
            for key, value in zip(rows, values):
                rows[key].append(value)
            self.next_transition_id += b
            self.obs, self.starts = result.next_obs, result.next_episode_start
            self.state = detach_state(output.state)
        x = {name: torch.stack(value) for name, value in rows.items()}
        valid = torch.ones_like(x["term"])
        adv, returns = compute_gae(x["reward"], x["value"], x["bootstrap"], x["term"], x["trunc"], valid,
                                   self.config.ppo.gamma, self.config.ppo.gae_lambda)
        return PPOBatch(x["obs"], x["action"], x["logprob"], x["value"], x["reward"], adv, returns,
                        x["start"], x["term"], x["trunc"], initial, x["bootstrap"], valid, x["next"])


class SequenceCollector:
    """Own a frozen policy snapshot; only U old image batches persist across windows."""
    def __init__(self, envs: VectorEnvAdapter, policy: StandalonePolicy | DualPolicy,
                 encoder: VisionEncoder, config: Config, *, mode: str, start_transition_id: int = 0):
        if mode not in ("C", "F") or (mode == "F" and type(policy) is not StandalonePolicy):
            raise TypeError("F collects only standalone KB; mode must be C or F")
        if mode == "C" and not isinstance(policy, DualPolicy):
            raise TypeError("C requires a dual teacher")
        if start_transition_id < 0:
            raise ValueError("negative transition identity")
        self.envs, self.encoder, self.config, self.mode = envs, encoder, config, mode
        encoder.freeze()
        self.policy = frozen_copy(policy)
        self.device = next(self.policy.parameters()).device
        digest = hashlib.sha256()
        for name, value in self.policy.state_dict().items():
            digest.update(name.encode())
            digest.update(value.cpu().contiguous().numpy().tobytes())
        self.snapshot_id = digest.hexdigest()
        self.encoder_version = int(encoder.encoder_version)
        self.obs, _ = envs.reset()
        self.starts = np.ones(envs.num_envs, bool)
        self.state = self.policy.initial_state(envs.num_envs)
        self.history = deque(maxlen=config.distill.burnin_env_obs)
        self.next_transition_id = start_transition_id

    @torch.no_grad()
    def collect(self, steps: int, *, action_rng: torch.Generator) -> SequenceBatch | FisherSequenceBatch:
        b, u, length = self.envs.num_envs, self.config.distill.burnin_env_obs, self.config.distill.learning_steps
        if steps < 1 or steps % b or steps > b*length:
            raise ValueError("invalid exact sequence-window budget")
        if int(self.encoder.encoder_version) != self.encoder_version or self.encoder._world_training:
            raise RuntimeError("sequence collection requires a fixed visual encoder")
        obs = torch.zeros(u+length, b, 3, 84, 84, dtype=torch.uint8)
        starts = torch.zeros(u+length, b, dtype=torch.bool)
        valid = torch.zeros_like(starts)
        burnin = torch.zeros_like(starts); burnin[:u] = True
        actions = torch.zeros(u+length, b, dtype=torch.int64)
        ids = torch.full_like(actions, -1)
        targets = torch.zeros(length, b, 5)
        for i, (old_obs, old_starts) in enumerate(self.history, start=u-len(self.history)):
            obs[i], starts[i], valid[i] = old_obs, old_starts, True
        for t in range(steps//b):
            images, flags = torch.from_numpy(self.obs.copy()), torch.from_numpy(self.starts.copy())
            output = self.policy.step(encode_obs(images.to(self.device), self.encoder), self.state, flags.to(self.device))
            if isinstance(output, PolicyOutput):
                logits, self.state = output.logits, output.state
            else:
                logits, self.state = output
            sampled = sample_actions(logits, action_rng)
            result = self.envs.step(sampled.numpy())
            # Both C and F deliberately avoid accessing reward_ext.
            obs[u+t], starts[u+t], valid[u+t], actions[u+t] = images, flags, True, sampled
            ids[u+t] = torch.arange(self.next_transition_id, self.next_transition_id+b)
            targets[t] = logits.log_softmax(-1).cpu()
            self.next_transition_id += b
            self.history.append((images, flags))
            self.obs, self.starts = result.next_obs, result.next_episode_start
        if self.mode == "F":
            return FisherSequenceBatch(obs, starts, valid, burnin, actions, torch.zeros_like(valid), ids)
        return SequenceBatch(obs, starts, valid, burnin, valid & ~burnin, targets,
                             self.snapshot_id, self.encoder_version)


def collect_fisher(envs: VectorEnvAdapter, kb: StandalonePolicy, encoder: VisionEncoder, config: Config,
                   *, action_rng: torch.Generator, fisher_rng: torch.Generator,
                   start_transition_id: int = 0) -> list[FisherSequenceBatch]:
    collector = SequenceCollector(envs, kb, encoder, config, mode="F", start_transition_id=start_transition_id)
    remaining, batches = config.fisher.collect_steps, []
    while remaining:
        steps = min(remaining, envs.num_envs*config.distill.learning_steps)
        batches.append(collector.collect(steps, action_rng=action_rng))
        remaining -= steps
    if fisher_rng.device.type != "cpu" or config.fisher.scored_samples > config.fisher.collect_steps:
        raise ValueError("invalid Fisher selection budget/RNG")
    # IDs count only new learning transitions; burn-in can never be selected twice.
    selected = torch.randperm(config.fisher.collect_steps, generator=fisher_rng)[:config.fisher.scored_samples]+start_transition_id
    for batch in batches:
        batch.score_mask = torch.isin(batch.transition_ids, selected) & batch.valid_mask & ~batch.burnin_mask
    return batches
