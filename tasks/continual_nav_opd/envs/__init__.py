"""Paired observations of one environment state; only RGB enters the student."""
from pathlib import Path

import numpy as np

from tasks.continual_nav.envs.maze import MazeEnv as RGBMazeEnv
from tasks.continual_nav.envs.fourrooms import FourRoomsEnv as RGBFourRoomsEnv
from tasks.continual_nav.envs.common import pixels, validate_action
from ..contracts import ObservationPair, EnvStep


class MazeEnv(RGBMazeEnv):
    def reset(self, **kwargs):
        try:
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


def make_env(config, task, split, *, manifest=None, seed=0, evaluation_seeds=()):
    if task == "maze_medium":
        if manifest is None:
            raise ValueError("Maze requires its verified manifest")
        from ..config import REPO_ROOT
        root = Path(config.environment.maze_root)
        return MazeEnv(root if root.is_absolute() else REPO_ROOT / root,
                       manifest.entries(split, panel=split != "train"), seed=seed)
    if task == "fourrooms":
        return FourRoomsEnv(split=split, seed=seed, evaluation_seeds=tuple(evaluation_seeds))
    raise ValueError(f"unknown task: {task}")
