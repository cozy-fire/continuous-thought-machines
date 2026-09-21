"""Delivery-1 verification against local maps and real MiniGrid, without a policy."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from PIL import Image

from .config import budget_summary, config_hash, load_config, save_config
from .contracts import OBS_SHAPE
from .data.manifest import build_manifest
from .envs import VectorEnvAdapter, build_env


def verify(config_path: Path, output_dir: Path, maze_root: Path | None = None) -> dict:
    config = load_config(config_path)
    root = (maze_root or Path(config.environment.maze_root)).resolve()
    ev = config.evaluation
    manifest = build_manifest(root, validation_count=ev.maze_validation_hashes,
                              validation_episodes=ev.validation_episodes,
                              test_episodes=ev.test_episodes, drift_episodes=ev.drift_episodes)
    # Keep evidence from previous invocations; require a new output directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "maze_splits.json"
    manifest.save(manifest_path)
    save_config(config, output_dir / "resolved_config.yaml")
    report = {
        "scope": "delivery_1_environment_only_no_training",
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "torch", "gymnasium", "minigrid", "Pillow", "PyYAML")},
        "config_hash": config_hash(config), "maze_root": str(root),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "map_counts": {s: len(manifest.entries(s)) for s in ("train", "validation", "test")},
        "panel_counts": {"validation": len(manifest.validation_panel),
                         "test": len(manifest.test_panel), "drift": len(manifest.drift_panel)},
        "budgets": budget_summary(config), "tasks": {},
    }
    for task in config.task_order:
        envs = [build_env(task, "train", seed, config=config, maze_manifest=manifest, maze_root=root)
                for seed in (0, 1)]
        vector = VectorEnvAdapter(envs)
        terminated_count = truncated_count = 0
        try:
            obs, _ = vector.reset()
            assert obs.shape == (2, *OBS_SHAPE) and obs.dtype == np.uint8
            for i, env in enumerate(envs):
                frame = env.render()
                np.testing.assert_array_equal(frame, obs[i].transpose(1, 2, 0))
                Image.fromarray(frame).save(output_dir / f"{task}_reset_{i}.png")
            # Five actions are exercised first; waiting then guarantees a timeout.
            for step in range(305):
                result = vector.step([step if step < 5 else 4] * 2)
                assert result.next_obs.shape == result.transition_next_obs.shape == (2, *OBS_SHAPE)
                assert result.next_obs.dtype == result.transition_next_obs.dtype == np.uint8
                assert result.reward_ext.dtype == np.float32
                assert not np.any(result.terminated & result.truncated)
                np.testing.assert_array_equal(result.next_episode_start, result.terminated | result.truncated)
                terminated_count += int(result.terminated.sum())
                truncated_count += int(result.truncated.sum())
                for i in range(2):
                    np.testing.assert_array_equal(envs[i].render(), result.next_obs[i].transpose(1, 2, 0))
                    if result.next_episode_start[i]:
                        assert result.info[i]["reset_info"]["episode_step"] == 0
                if step == 299:
                    Image.fromarray(result.transition_next_obs[0].transpose(1, 2, 0)).save(
                        output_dir / f"{task}_transition_at_step300.png")
            assert truncated_count >= 2
            report["tasks"][task] = dict(transitions=610, terminated=terminated_count,
                                          truncated=truncated_count, observation_shape=list(obs.shape),
                                          reset_step_render="passed")
        finally:
            vector.close()
    report["status"] = "passed"
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--maze-root", type=Path)
    args = parser.parse_args()
    report = verify(args.config, args.output_dir, args.maze_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
