"""RL dynamics over the original medium Maze maps, without solution-path pixels."""
from __future__ import annotations

from collections import deque
from io import BytesIO
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

from ..contracts import ImageArray, OBS_SHAPE
from ..data.manifest import MazeEntry, read_entry
from .common import pixels, validate_action

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))


def load_map(root: Path, entry: MazeEntry) -> tuple[ImageArray, tuple[int, int], tuple[int, int]]:
    with Image.open(BytesIO(read_entry(root, entry))) as image:
        rgb = np.array(image.convert("RGB"))
    if rgb.shape != (19, 19, 3):
        raise ValueError(f"expected a 19x19 RGB maze: {entry.path}")
    allowed = {(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)}
    if any(tuple(c) not in allowed for c in np.unique(rgb.reshape(-1, 3), axis=0)):
        raise ValueError(f"unknown maze color: {entry.path}")
    starts = np.argwhere(np.all(rgb == (255, 0, 0), axis=-1))
    goals = np.argwhere(np.all(rgb == (0, 255, 0), axis=-1))
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError(f"maze must contain exactly one start and goal: {entry.path}")
    start, goal = tuple(map(int, starts[0])), tuple(map(int, goals[0]))
    # Blue pixels encode the supervised answer. Erase them before any observation.
    rgb[np.all(rgb == (0, 0, 255), axis=-1)] = (255, 255, 255)
    rgb[start] = (255, 255, 255)
    walls = np.all(rgb == 0, axis=-1)
    queue, seen = deque([start]), {start}
    while queue:
        row, col = queue.popleft()
        for dr, dc in MOVES[:4]:
            pos = (row + dr, col + dc)
            if 0 <= pos[0] < 19 and 0 <= pos[1] < 19 and not walls[pos] and pos not in seen:
                seen.add(pos)
                queue.append(pos)
    if goal not in seen:
        raise ValueError(f"unreachable maze goal: {entry.path}")
    return rgb, start, goal


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
    render_mode = "rgb_array"

    def __init__(self, root: str | Path, entries: tuple[MazeEntry, ...], *, seed: int = 0):
        super().__init__()
        if not entries:
            raise ValueError("MazeEnv needs a nonempty manifest split")
        self.root, self.entries = Path(root), entries
        self.observation_space = gym.spaces.Box(0, 255, OBS_SHAPE, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(5)
        self.max_steps = 300
        self._pending_seed = seed
        self._done = True
        self.episode_id = -1
        self._obs: ImageArray | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options:
            raise ValueError("Maze reset options are not supported")
        super().reset(seed=seed if seed is not None else self._pending_seed)
        self._pending_seed = None
        self.entry = self.entries[int(self.np_random.integers(len(self.entries)))]
        self.base_rgb, self.agent_pos, self.goal_pos = load_map(self.root, self.entry)
        self.walls = np.all(self.base_rgb == 0, axis=-1)
        self.step_count = 0
        self.episode_id += 1
        self._done = False
        return self._observe(), self._info(False)

    def _observe(self) -> ImageArray:
        image = self.base_rgb.copy()
        image[self.agent_pos] = (255, 0, 0)
        self._obs = pixels(image)
        return self._obs.copy()

    def _info(self, success: bool) -> dict[str, object]:
        return dict(task_key="maze_medium", episode_id=self.episode_id,
                    episode_step=self.step_count, success=success, map_path=self.entry.path,
                    map_sha256=self.entry.sha256)

    def step(self, action: int):
        action = validate_action(action)
        if self._done:
            raise RuntimeError("reset is required before step")
        self.step_count += 1
        dr, dc = MOVES[action]
        pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
        if 0 <= pos[0] < 19 and 0 <= pos[1] < 19 and not self.walls[pos]:
            self.agent_pos = pos
        success = self.agent_pos == self.goal_pos
        # A success on step 300 terminates; it is not also a time-limit truncation.
        truncated = self.step_count >= self.max_steps and not success
        self._done = success or truncated
        reward = 1.0 - 0.9 * self.step_count / self.max_steps if success else 0.0
        return self._observe(), reward, success, truncated, self._info(success)

    def render(self) -> ImageArray:
        """Return exactly the policy view in HWC layout, never an answer overlay."""
        if self._obs is None:
            raise RuntimeError("reset is required before render")
        return np.ascontiguousarray(self._obs.transpose(1, 2, 0))
