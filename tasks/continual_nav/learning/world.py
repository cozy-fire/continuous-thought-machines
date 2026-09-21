"""Frozen W collection and offline MSE/SIGReg updates, without external rewards."""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import Tensor

from ..config import Config, config_hash
from ..contracts import CollectionResult, TaskKey, Transition, WorldPrediction
from ..data.replay import ShardedWriter, TransitionStore, sample_world_batch
from ..envs.vector import VectorEnvAdapter
from ..models.policy import StandalonePolicy, frozen_copy
from ..models.sigreg import SIGReg
from ..models.vision import VisionEncoder, encode_obs
from ..models.world import WorldModel


class SnapshotCollector:
    """Own E_old/KB copies; callers may update live E/KB without affecting this collector."""
    def __init__(self, encoder: VisionEncoder, kb: StandalonePolicy, config: Config,
                 *, world_version: int):
        if type(kb) is not StandalonePolicy:
            raise TypeError("W collection uses standalone KB, not Active or a critic")
        self.encoder, self.kb = frozen_copy(encoder), frozen_copy(kb)
        self.encoder_version = int(self.encoder.encoder_version)
        self.world_version = world_version
        self.config_hash = config_hash(config)
        self.device = next(self.encoder.parameters()).device
        if next(self.kb.parameters()).device != self.device:
            raise ValueError("collector E and KB must be on the same device")
        digest = hashlib.sha256(self.config_hash.encode())
        for prefix, module in (("E", self.encoder), ("KB", self.kb)):
            for name, value in module.state_dict().items():
                digest.update((prefix+name).encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.snapshot_id = digest.hexdigest()

    @torch.no_grad()
    def collect(self, envs: VectorEnvAdapter, writer: ShardedWriter, *, steps: int,
                action_rng: torch.Generator, start_transition_id: int = 0) -> CollectionResult:
        if steps <= 0 or steps % envs.num_envs or start_transition_id < 0:
            raise ValueError("collection budget must be a positive multiple of num_envs")
        if action_rng.device.type != "cpu":
            raise ValueError("collection requires a separate CPU action generator")
        if (writer.source != "W" or writer.expected_count != steps or writer.count != 0
                or writer.encoder_version != self.encoder_version or writer.world_version != self.world_version
                or writer.header["snapshot_id"] != self.snapshot_id):
            raise ValueError("fresh writer does not match the fixed snapshot/budget")
        obs, infos = envs.reset()
        if any(info["task_key"] != writer.task for info in infos):
            raise ValueError("W collection must use one current task")
        state = self.kb.initial_state(envs.num_envs)
        episode_start = np.ones(envs.num_envs, dtype=np.bool_)
        next_id = start_transition_id
        for _ in range(steps // envs.num_envs):
            fmap = encode_obs(torch.from_numpy(obs).to(self.device), self.encoder)
            logits, state = self.kb.step(fmap, state, torch.from_numpy(episode_start).to(self.device))
            # multinomial with one draw is categorical sampling with an explicit RNG.
            actions = torch.multinomial(logits.softmax(-1).cpu(), 1, generator=action_rng).squeeze(-1).numpy()
            result = envs.step(actions)
            for slot, info in enumerate(result.info):
                if info["task_key"] != writer.task:
                    raise ValueError("environment changed task during W collection")
                # Episode IDs are unique across vector slots within this source stage.
                record = Transition(obs[slot], result.transition_next_obs[slot], int(actions[slot]),
                    bool(result.terminated[slot]), bool(result.truncated[slot]), bool(episode_start[slot]),
                    int(info["episode_id"])*envs.num_envs+slot, int(info["episode_step"]), writer.task,
                    self.encoder_version, self.world_version, next_id)
                writer.append(record)
                next_id += 1
            # Deliberately never access result.reward_ext, including for logging.
            obs, episode_start = result.next_obs, result.next_episode_start
        manifest = writer.finish()
        return CollectionResult(steps, next_id, str(manifest), self.snapshot_id)


def prediction_losses(prediction: WorldPrediction, regularizer: SIGReg, directions: Tensor,
                      coefficient: float) -> tuple[Tensor, Tensor, Tensor]:
    forward = (prediction.z_pred-prediction.z_next).square().mean()
    # Two marginal constraints share the same directions, not a merged time batch.
    sigreg = 0.5*(regularizer(prediction.z, directions)+regularizer(prediction.z_next, directions))
    return forward+coefficient*sigreg, forward, sigreg


def make_world_optimizer(world: WorldModel, config: Config) -> torch.optim.AdamW:
    o = config.world.optimizer
    # Keep this optimizer across W rounds; do not rebuild its moment estimates in fit().
    return torch.optim.AdamW(world.parameters(), lr=o.lr, betas=o.betas, eps=o.eps,
                             weight_decay=o.weight_decay)


def fit_world_model(world: WorldModel, regularizer: SIGReg, fresh: TransitionStore,
                    high: TransitionStore | None, optimizer: torch.optim.Optimizer, config: Config,
                    *, task: TaskKey, replay_rng: np.random.Generator,
                    sigreg_rng: torch.Generator) -> dict[str, float | int]:
    if world._fit_complete:
        raise RuntimeError("publish or discard the previous completed fit before starting another")
    if fresh.encoder_version != int(world.encoder.encoder_version) or fresh.world_version != int(world.world_model_version):
        raise ValueError("fresh data must come from the current pre-fit snapshot")
    if high is not None and high.world_version > int(world.world_model_version):
        raise ValueError("high-error pool cannot come from a future world version")
    expected = {id(p) for p in world.parameters()}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("W optimizer must own exactly E/g/F, with no duplicates or controller parameters")
    device = next(world.parameters()).device
    regularizer.to(device)
    world.set_fit_mode()
    completed = False
    metrics: dict[str, float | int] = {}
    try:
        for update in range(config.world.updates_per_round):
            batch = sample_world_batch(fresh, high, task=task, batch_size=config.world.batch_size,
                                      high_fraction=config.replay.high_error_fraction, rng=replay_rng)
            optimizer.zero_grad(set_to_none=True)
            prediction = world.transition(batch.obs.to(device), batch.next_obs.to(device), batch.actions.to(device))
            directions = regularizer.sample_directions(prediction.z.shape[1], sigreg_rng, device)
            loss, mse, sigreg = prediction_losses(prediction, regularizer, directions, config.sigreg.lambda_)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite world loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(world.parameters(), config.world.max_grad_norm,
                                                      error_if_nonfinite=True)
            optimizer.step()
            metrics = dict(updates=update+1, loss=loss.item(), forward_mse=mse.item(), sigreg=sigreg.item(),
                           grad_norm=float(grad_norm), high_count=int(batch.from_high_error.sum()),
                           fresh_count=int((~batch.from_high_error).sum()))
        completed = True
    finally:
        # A failure never publishes versions and never leaves BN in training mode.
        world.freeze()
        world._fit_complete = completed
    return metrics
