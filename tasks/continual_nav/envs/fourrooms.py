"""Five-action, partial-RGB adaptation of native MiniGrid FourRooms."""
from __future__ import annotations

import gymnasium as gym
import minigrid  # Registers MiniGrid-FourRooms-v0 with Gymnasium.
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper

from ..contracts import ImageArray, OBS_SHAPE, Split
from .common import pixels, validate_action


class FourRoomsEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
    render_mode = "rgb_array"

    def __init__(self, *, split: Split = "train", seed: int = 0,
                 evaluation_seeds: tuple[int, ...] = ()):
        super().__init__()
        if split not in ("train", "validation", "test"):
            raise ValueError(f"unknown split: {split}")
        if split != "train" and not evaluation_seeds:
            raise ValueError("evaluation requires an explicit fixed seed panel")
        self.split, self.evaluation_seeds = split, evaluation_seeds
        native = gym.make("MiniGrid-FourRooms-v0", max_steps=300, render_mode="rgb_array")
        self.native = RGBImgPartialObsWrapper(native, tile_size=8)
        self.observation_space = gym.spaces.Box(0, 255, OBS_SHAPE, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(5)
        self.max_steps = 300
        self.episode_id = -1
        self._pending_seed = seed
        self._panel_index = 0
        self._done = True
        self._obs: ImageArray | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options:
            raise ValueError("FourRooms reset options are not supported")
        super().reset(seed=seed if seed is not None else self._pending_seed)
        self._pending_seed = None
        if self.split == "train":
            # The RNG seed controls a stream of episode seeds in the training partition.
            self.episode_seed = int(self.np_random.integers(0, 1000000))
        else:
            # Explicit eval seeds identify layouts; autoreset walks the fixed panel.
            if seed is not None:
                if seed not in self.evaluation_seeds:
                    raise ValueError("reset seed is outside the evaluation panel")
                self._panel_index = self.evaluation_seeds.index(seed)
            self.episode_seed = self.evaluation_seeds[self._panel_index]
            self._panel_index = (self._panel_index + 1) % len(self.evaluation_seeds)
        raw, _ = self.native.reset(seed=self.episode_seed)
        self.episode_id += 1
        self._done = False
        self._obs = pixels(raw["image"])
        return self._obs.copy(), self._info(False)

    def _info(self, success: bool) -> dict[str, object]:
        return dict(task_key="fourrooms", episode_id=self.episode_id,
                    episode_step=int(self.native.unwrapped.step_count),
                    episode_seed=self.episode_seed, success=success)

    def step(self, action: int):
        action = validate_action(action)
        if self._done:
            raise RuntimeError("reset is required before step")
        mapped = action if action < 3 else 9
        # 9 is our sentinel only. MiniGrid rejects it; native action 6 is a no-op.
        native_action = 6 if mapped == 9 else mapped
        raw, reward, terminated, truncated, _ = self.native.step(native_action)
        success = bool(terminated)
        truncated = bool(truncated and not success)
        self._done = success or truncated
        self._obs = pixels(raw["image"])
        info = self._info(success)
        info.update(policy_action=action, mapped_action=mapped, native_action=native_action)
        return self._obs.copy(), float(reward), success, truncated, info

    def render(self) -> ImageArray:
        # Never use native.render(): that would expose a global view to callers.
        if self._obs is None:
            raise RuntimeError("reset is required before render")
        return np.ascontiguousarray(self._obs.transpose(1, 2, 0))

    def close(self) -> None:
        self.native.close()
