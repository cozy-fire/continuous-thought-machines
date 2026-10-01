"""Paired observations of one environment state; only RGB enters the student."""
from pathlib import Path

import numpy as np

from tasks.continual_nav.envs.maze import MazeEnv as RGBMazeEnv
from tasks.continual_nav.envs.fourrooms import FourRoomsEnv as RGBFourRoomsEnv
from tasks.continual_nav.envs.common import pixels, validate_action
from ..contracts import ObservationPair, EnvStep
from .map_cache import MazeMapCache, shared_map_cache


class MazeEnv(RGBMazeEnv):
    def __init__(self, root, entries, *, seed=0, map_cache=None, map_data=None):
        super().__init__(root, entries, seed=seed)
        self.map_cache, self.map_data = map_cache, map_data

    def reset(self, **kwargs):
        try:
            if self.map_cache is not None or self.map_data is not None:
                if set(kwargs)-{'seed','options'}:
                    raise TypeError('unexpected Maze reset arguments')
                if kwargs.get('options'):
                    raise ValueError('Maze reset options are not supported')
                # Preserve Gym's seed initialization and exactly one map-selection draw.
                import gymnasium as gym
                seed = kwargs.get('seed')
                gym.Env.reset(self, seed=seed if seed is not None else self._pending_seed)
                self._pending_seed = None
                self.entry = self.entries[int(self.np_random.integers(len(self.entries)))]
                data = self.map_data if self.map_data is not None else self.map_cache.get(self.root, self.entry)
                self.base_rgb, self.agent_pos, self.goal_pos = data
                self.walls = np.all(self.base_rgb == 0, axis=-1)
                self.step_count = 0
                self.episode_id += 1
                self._done = False
                return self._observe(), self._info(False)
            return super().reset(**kwargs)
        except (ValueError, OSError) as exc:
            entry = getattr(self, "entry", None)
            raise ValueError(f"map={getattr(entry, 'sha256', 'unknown')} "
                             f"episode={self.episode_id + 1}: {exc}") from exc

    def _observe(self):
        image = self.base_rgb.copy()
        image[self.agent_pos] = (255, 0, 0)
        self._obs = pixels(image)
        return ObservationPair(self._obs.copy(), np.ascontiguousarray(image.transpose(2, 0, 1)))


class FourRoomsEnv(RGBFourRoomsEnv):
    def _pair(self, rgb):
        # gen_obs reads the SAME underlying state as the RGB wrapper, with no step/reset.
        symbols = self.native.unwrapped.gen_obs()["image"].copy()
        return ObservationPair(rgb, symbols)

    def reset(self, **kwargs):
        rgb, info = super().reset(**kwargs)
        info.update(agent_pos=tuple(map(int, self.native.unwrapped.agent_pos)),
                    agent_dir=int(self.native.unwrapped.agent_dir))
        return self._pair(rgb), info

    def step(self, action):
        native = self.native.unwrapped
        before_pos, before_dir = tuple(map(int, native.agent_pos)), int(native.agent_dir)
        rgb, reward, terminated, truncated, info = super().step(action)
        info.update(agent_pos_before=before_pos, agent_dir_before=before_dir,
                    agent_pos_after=tuple(map(int, native.agent_pos)),
                    agent_dir_after=int(native.agent_dir))
        return self._pair(rgb), reward, terminated, truncated, info


def stack_observations(observations):
    if not observations:
        raise ValueError("cannot stack an empty observation batch")
    return ObservationPair(np.stack([o.student_rgb for o in observations]),
                           np.stack([o.teacher_obs for o in observations]))


class SyncVectorEnv:
    """Explicit terminal/reset separation; slots must belong to the same task."""
    def __init__(self, envs):
        self.envs = tuple(envs)
        if not self.envs or len({type(e) for e in self.envs}) != 1:
            raise ValueError("vector requires nonempty homogeneous environments")
        if len({id(e) for e in self.envs}) != len(self.envs):
            raise ValueError("environment slots must be independent instances")

    def reset(self):
        values = [env.reset() for env in self.envs]
        return stack_observations([o for o, _ in values]), [info for _, info in values]

    def step(self, actions):
        if len(actions) != len(self.envs):
            raise ValueError("action count must equal slot count")
        # Validate the entire batch before advancing any environment.
        actions = [validate_action(action) for action in actions]
        terminal, following, rewards, terms, truncs, infos = [], [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, term, trunc, info = env.step(action)
            terminal.append(obs)
            # Never overwrite a true terminal observation with the automatic reset image.
            if term or trunc:
                next_obs, reset_info = env.reset()
                info = {**info, "reset_info": reset_info}
            else:
                next_obs = obs
            following.append(next_obs)
            rewards.append(reward)
            terms.append(term)
            truncs.append(trunc)
            infos.append(info)
        terminated, truncated = np.asarray(terms, dtype=bool), np.asarray(truncs, dtype=bool)
        return EnvStep(stack_observations(terminal), stack_observations(following),
                       np.asarray(rewards, dtype=np.float32), terminated, truncated,
                       terminated | truncated, infos)

    def close(self):
        for env in self.envs:
            env.close()


def make_env(config, task, split, *, manifest=None, seed=0, evaluation_seeds=(), map_cache=None):
    if task == "maze_medium":
        if manifest is None:
            raise ValueError("Maze requires its verified manifest")
        from ..config import REPO_ROOT
        root = Path(config.environment.maze_root)
        root = root if root.is_absolute() else REPO_ROOT / root
        entries = manifest.entries(split, panel=split != "train")
        if config.environment.map_cache == 'memory' and map_cache is None:
            map_cache = shared_map_cache(root, entries)
        return MazeEnv(root, entries, seed=seed, map_cache=map_cache)
    if task == "fourrooms":
        return FourRoomsEnv(split=split, seed=seed, evaluation_seeds=tuple(evaluation_seeds))
    raise ValueError(f"unknown task: {task}")
