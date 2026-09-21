"""Explicit environment construction; task names never enter observations."""
from pathlib import Path

from ..config import Config, validate_config
from ..contracts import Split, TaskKey
from ..data.manifest import MazeManifest
from .fourrooms import FourRoomsEnv
from .maze import MazeEnv
from .vector import VectorEnvAdapter


def build_env(task_key: TaskKey, split: Split, seed: int, *, config: Config,
              maze_manifest: MazeManifest | None = None,
              maze_root: str | Path | None = None) -> MazeEnv | FourRoomsEnv:
    """Relative maze_root is resolved from the caller's working directory.

    Evaluation uses fixed panels. Maze samples panel maps; evaluators can build
    one-entry MazeEnv instances to visit each fixed map exactly once.
    """
    validate_config(config)
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown split: {split}")
    if task_key == "maze_medium":
        if maze_manifest is None:
            raise ValueError("Maze requires a verified split manifest")
        entries = maze_manifest.entries(split, panel=split != "train")
        return MazeEnv(maze_root or config.environment.maze_root, entries, seed=seed)
    if task_key == "fourrooms":
        ev = config.evaluation
        panel = ()
        if split != "train":
            start = ev.fourrooms_validation_seed_start if split == "validation" else ev.fourrooms_test_seed_start
            count = ev.validation_episodes if split == "validation" else ev.test_episodes
            panel = tuple(range(start, start + count))
        return FourRoomsEnv(split=split, seed=seed, evaluation_seeds=panel)
    raise ValueError(f"unknown task: {task_key}")


__all__ = ["build_env", "MazeEnv", "FourRoomsEnv", "VectorEnvAdapter"]
