"""Synchronous, same-step autoreset with explicit terminal observations."""
from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np

from ..contracts import EnvStep, ImageArray
from .common import validate_action


class VectorEnvAdapter:
    def __init__(self, envs: Sequence[gym.Env]):
        if not envs or len({id(e) for e in envs}) != len(envs):
            raise ValueError("vector slots require distinct, nonempty environments")
        self.envs = list(envs)
        self.num_envs = len(self.envs)
        self._ready = False

    def reset(self, seeds: Sequence[int] | None = None) -> tuple[ImageArray, list[dict]]:
        if seeds is not None and len(seeds) != self.num_envs:
            raise ValueError("one reset seed is required per environment")
        pairs = [env.reset(seed=None if seeds is None else seeds[i]) for i, env in enumerate(self.envs)]
        self._ready = True
        return np.stack([p[0] for p in pairs]), [p[1] for p in pairs]

    def step(self, actions: Sequence[int]) -> EnvStep:
        if not self._ready:
            raise RuntimeError("reset is required before vector step")
        if np.shape(actions) != (self.num_envs,):
            raise ValueError("actions must have shape [num_envs]")
        # Validate the whole batch before advancing any environment.
        actions = [validate_action(a) for a in actions]
        next_obs, final_obs, rewards, terminated, truncated, infos = [], [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, term, trunc, info = env.step(action)
            final_obs.append(obs.copy())
            info = dict(info)
            if term or trunc:
                # Preserve the real target before reset, even if the environment reuses buffers.
                obs, reset_info = env.reset()
                info["reset_info"] = reset_info
            next_obs.append(obs.copy())
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            infos.append(info)
        term = np.asarray(terminated, dtype=np.bool_)
        trunc = np.asarray(truncated, dtype=np.bool_)
        return EnvStep(np.stack(next_obs), np.stack(final_obs), np.asarray(rewards, dtype=np.float32),
                       term, trunc, term | trunc, infos)

    def close(self) -> None:
        for env in self.envs:
            env.close()
        self._ready = False
