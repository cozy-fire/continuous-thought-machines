"""Stage-local streaming collectors. Recurrent sampling state survives learner updates."""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical

from ..config import Config
from ..contracts import FisherSequenceBatch, PPOBatch, PolicyOutput, SequenceBatch
from ..envs import VectorEnvAdapter
from ..models import DualPolicy, SingleActorCritic, StandalonePolicy, VisionEncoder, detach_state, encode_obs, frozen_copy
from ..learning.common import prepare_ppo
from ..learning.ppo import compute_gae
from ..learning.revisit import VisualRevisit


def sample_actions(logits: Tensor, rng: torch.Generator) -> Tensor:
    if rng.device.type != "cpu":
        raise ValueError("action sampling requires an explicit CPU generator")
    return torch.multinomial(logits.softmax(-1).cpu(), 1, generator=rng).squeeze(-1)


class PPOCollector:
    def __init__(self, envs: VectorEnvAdapter, policy: SingleActorCritic | DualPolicy,
                 encoder: VisionEncoder, config: Config, *, phase: str,
                 start_transition_id: int = 0):
        if phase not in ("X", "P") or start_transition_id < 0:
            raise ValueError("invalid PPO phase/transition identity")
        self.envs, self.policy, self.encoder, self.config = envs, policy, encoder, config
        self.phase = phase
        self.device = next(policy.parameters()).device
        if phase == "P":
            encoder.freeze(permanent=True)
        prepare_ppo(policy, encoder, phase=phase)
        self.obs, infos = envs.reset()
        self.revisit = VisualRevisit(config, self.device) if phase == "X" else None
        if self.revisit is not None:
            self.revisit.reset(self.obs, infos)
        self.revisit_statistics = dict(reward_sum=0., nonzero=0, unchanged_pixels=0., identical_frames=0,
            max_similarity_sum=0., match_similarity_sum=0., gap_sum=0., count=0,
            actions=np.zeros(5, dtype=np.int64), similarity_bins=np.zeros(5, dtype=np.int64),
            gaps=np.zeros(config.exploration.history_length, dtype=np.int64))
        self.diagnostics = EpisodeDiagnostics(self.obs, infos, start_transition_id,
            config.exploration.steps_per_round) if phase == "X" else None
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
                reward, max_similarity, matched_similarity, gap = self.revisit.score(result.transition_next_obs, result.info,
                    result.next_episode_start, result.next_obs)
                self.diagnostics.record(actions.numpy(), result.transition_next_obs, reward.numpy(),
                    max_similarity.numpy(), matched_similarity.numpy(), gap.numpy(), result.terminated, result.truncated,
                    result.info, result.next_obs, self.next_transition_id)
                stats = self.revisit_statistics
                stats["reward_sum"] += float(reward.sum())
                stats["nonzero"] += int((reward < 0).sum())
                equality = self.obs == result.transition_next_obs
                stats["unchanged_pixels"] += float(equality.mean(axis=(1, 2, 3)).sum())
                stats["identical_frames"] += int(np.count_nonzero(equality.all(axis=(1, 2, 3))))
                stats["max_similarity_sum"] += float(max_similarity.sum())
                stats["match_similarity_sum"] += float(matched_similarity.sum())
                stats["gap_sum"] += float(gap.sum())
                stats["count"] += b
                stats["actions"] += np.bincount(actions.numpy(), minlength=5)
                stats["similarity_bins"] += np.bincount(np.searchsorted(
                    [.95, .99, .999, .9995], max_similarity.numpy(), side="right"), minlength=5)
                stats["gaps"] += np.bincount(gap.numpy(), minlength=self.config.exploration.history_length+1)[1:]
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

    def revisit_metrics(self) -> dict[str, float]:
        if self.phase != "X":
            return {}
        stats = self.revisit_statistics
        count = stats["count"]
        if not count:
            return {}
        metrics = dict(mean_penalty=-stats["reward_sum"]/count,
            nonzero_penalty_fraction=stats["nonzero"]/count,
            unchanged_pixel_fraction=stats["unchanged_pixels"]/count,
            identical_frame_fraction=stats["identical_frames"]/count,
            mean_max_similarity=stats["max_similarity_sum"]/count,
            mean_match_similarity=stats["match_similarity_sum"]/count,
            mean_match_gap=stats["gap_sum"]/count)
        metrics.update({f"action_{i}_fraction": float(v/count) for i, v in enumerate(stats["actions"])})
        metrics.update({f"similarity_bin_{i}_fraction": float(v/count) for i, v in enumerate(stats["similarity_bins"])})
        metrics.update({f"gap_{i+1}_fraction": float(v/count) for i, v in enumerate(stats["gaps"])})
        return metrics


class EpisodeDiagnostics:
    """Retain only selected complete episodes and current per-slot traces."""
    def __init__(self, images: np.ndarray, infos: list[dict], start_id: int, budget: int):
        self.start_id, self.halfway = start_id, start_id + budget // 2
        self.current = [self._new(images[i], infos[i], start_id + i) for i in range(len(images))]
        self.first = self.midpoint = self.last = self.most_penalized = None

    @staticmethod
    def _new(image: np.ndarray, info: dict, first_id: int) -> dict:
        return dict(first_transition_id=first_id, task=info["task_key"], episode_id=int(info["episode_id"]),
                    map_identity=info.get("map_sha256", info.get("episode_seed")),
                    images=[image.copy()], actions=[], rewards=[], max_similarities=[], matched_similarities=[], gaps=[],
                    terminated=False, truncated=False, success=False)

    def record(self, actions: np.ndarray, finals: np.ndarray, rewards: np.ndarray,
               max_similarities: np.ndarray, matched_similarities: np.ndarray, gaps: np.ndarray, terminated: np.ndarray,
               truncated: np.ndarray, infos: list[dict], resets: np.ndarray, base_id: int) -> None:
        for slot, episode in enumerate(self.current):
            episode["images"].append(finals[slot].copy())
            episode["actions"].append(int(actions[slot]))
            episode["rewards"].append(float(rewards[slot]))
            episode["max_similarities"].append(float(max_similarities[slot]))
            episode["matched_similarities"].append(float(matched_similarities[slot]))
            episode["gaps"].append(int(gaps[slot]))
            if terminated[slot] or truncated[slot]:
                episode["terminated"] = bool(terminated[slot])
                episode["truncated"] = bool(truncated[slot])
                episode["success"] = bool(infos[slot].get("success", False))
                episode["last_transition_id"] = base_id + slot
                episode["nonzero_penalties"] = sum(value < 0 for value in episode["rewards"])
                if self.first is None or episode["last_transition_id"] < self.first["last_transition_id"]:
                    self.first = episode
                if episode["last_transition_id"] >= self.halfway and (self.midpoint is None or
                        episode["last_transition_id"] < self.midpoint["last_transition_id"]):
                    self.midpoint = episode
                if self.last is None or episode["last_transition_id"] > self.last["last_transition_id"]:
                    self.last = episode
                if self.most_penalized is None or (episode["nonzero_penalties"], -episode["first_transition_id"]) > (
                        self.most_penalized["nonzero_penalties"], -self.most_penalized["first_transition_id"]):
                    self.most_penalized = episode
                self.current[slot] = self._new(resets[slot], infos[slot]["reset_info"], base_id + len(self.current) + slot)

    def selected(self) -> list[dict]:
        unique = {}
        for label, episode in (("first", self.first), ("after_halfway", self.midpoint),
                               ("last", self.last), ("most_penalized", self.most_penalized)):
            if episode is not None:
                key = (episode["first_transition_id"], episode["last_transition_id"])
                if key not in unique:
                    item = {**episode, "selection": [label], "images": torch.from_numpy(np.stack(episode["images"]))}
                    unique[key] = item
                else:
                    unique[key]["selection"].append(label)
        return list(unique.values())


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
        if int(self.encoder.encoder_version) != self.encoder_version or self.encoder._x_training:
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
