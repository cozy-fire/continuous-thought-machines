"""Fixed-panel argmax evaluation and same-trajectory visual drift, isolated from training."""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import torch

from .analysis.metrics import episode_metrics
from .checkpoint import atomic_json, load, load_artifact, verify_reference
from .config import Config, config_from_dict
from .contracts import PolicyOutput
from .data.manifest import MazeManifest
from .envs import FourRoomsEnv, MazeEnv
from .models import DualPolicy, SingleActorCritic, StandalonePolicy, VisionEncoder, encode_obs, frozen_copy
from .schedule import isolated_rng


def shortest_path(env: MazeEnv) -> int:
    queue, seen = deque([(env.agent_pos, 0)]), {env.agent_pos}
    while queue:
        pos, distance = queue.popleft()
        if pos == env.goal_pos:
            return distance
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = pos[0]+dr, pos[1]+dc
            if 0 <= nxt[0] < 19 and 0 <= nxt[1] < 19 and not env.walls[nxt] and nxt not in seen:
                seen.add(nxt); queue.append((nxt, distance+1))
    raise ValueError("unreachable evaluation maze")


def panel_envs(task: str, split: str, manifest: MazeManifest, config: Config, *, drift: bool = False):
    if split not in ("validation", "test") or (drift and split != "validation"):
        raise ValueError("invalid evaluation panel")
    count = config.evaluation.drift_episodes if drift else (
        config.evaluation.validation_episodes if split == "validation" else config.evaluation.test_episodes)
    if task == "maze_medium":
        panel = manifest.drift_panel if drift else manifest.entries(split, panel=True)
        if len(panel) < count:
            raise ValueError("insufficient fixed Maze evaluation panel")
        for entry in panel[:count]:
            yield MazeEnv(config.environment.maze_root, (entry,))
    elif task == "fourrooms":
        start = config.evaluation.fourrooms_validation_seed_start if split == "validation" else config.evaluation.fourrooms_test_seed_start
        for seed in range(start, start+count):
            yield FourRoomsEnv(split=split, evaluation_seeds=(seed,))
    else:
        raise ValueError("unknown evaluation task")


def _step(policy, encoder, obs, state, start):
    device = next(encoder.parameters()).device
    image = torch.from_numpy(obs).unsqueeze(0).to(device)
    result = policy.step(encode_obs(image, encoder), state, torch.tensor([start], device=device))
    return (result.logits, result.state) if isinstance(result, PolicyOutput) else result


@torch.no_grad()
def evaluate_policy(policy, encoder: VisionEncoder, manifest: MazeManifest, config: Config,
                    *, split: str = "validation", drift: bool = False) -> dict:
    with isolated_rng():
        policy, encoder = frozen_copy(policy), frozen_copy(encoder)
        results, total = {}, 0
        for task in config.task_order:
            episodes = []
            for env in panel_envs(task, split, manifest, config, drift=drift):
                try:
                    obs, _ = env.reset()
                    distance = shortest_path(env) if task == "maze_medium" else None
                    state, reward_sum, length = policy.initial_state(1), 0., 0
                    while True:
                        logits, state = _step(policy, encoder, obs, state, length == 0)
                        obs, reward, term, trunc, info = env.step(int(logits.argmax(-1).item()))
                        reward_sum += reward; length += 1; total += 1
                        if term or trunc:
                            episodes.append(dict(success=bool(info["success"]), length=length,
                                                 **{"return": reward_sum}, shortest_path=distance))
                            break
                finally:
                    env.close()
            results[task] = {**episode_metrics(episodes), "raw_episodes": episodes}
        return dict(split=split, drift_panel=drift, tasks=results, transitions=total)


@torch.no_grad()
def visual_drift(kb: StandalonePolicy, old_encoder: VisionEncoder, new_encoder: VisionEncoder,
                 manifest: MazeManifest, config: Config) -> dict:
    with isolated_rng():
        policy, old, new = frozen_copy(kb), frozen_copy(old_encoder), frozen_copy(new_encoder)
        results, transitions = {}, 0
        for task in config.task_order:
            total_kl, count = 0., 0
            for env in panel_envs(task, "validation", manifest, config, drift=True):
                try:
                    obs, _ = env.reset()
                    old_state, new_state = policy.initial_state(1), policy.initial_state(1)
                    start = True
                    while True:
                        old_logits, old_state = _step(policy, old, obs, old_state, start)
                        new_logits, new_state = _step(policy, new, obs, new_state, start)
                        lp, lq = old_logits.log_softmax(-1), new_logits.log_softmax(-1)
                        total_kl += (lp.exp()*(lp-lq)).sum().item(); count += 1
                        # Both recurrent policies follow the exact old-vision behavior trajectory.
                        obs, _, term, trunc, _ = env.step(int(old_logits.argmax(-1).item()))
                        start = False
                        if term or trunc:
                            break
                finally:
                    env.close()
            results[task] = dict(mean_kl=total_kl/count, observations=count)
            transitions += count
        return dict(tasks=results, transitions=transitions)


def load_for_evaluation(path: Path, policy_choice: str = "kb"):
    with isolated_rng():
        if path.parent.name == "checkpoints":
            root = path.resolve().parent.parent
            marker = None if path.name == "latest.json" else (path.with_suffix(".complete.json") if path.suffix == ".pt" else path)
            payload = load(root, marker)
            config = config_from_dict(payload["config"])
            manifest = MazeManifest.load(verify_reference(root, payload["manifests"]["maze"]), config.environment.maze_root)
            encoder = VisionEncoder(config)
            encoder.load_state_dict(payload["models"]["encoder"])
            if payload["policy_kind"] == "single":
                policy = SingleActorCritic(config)
                policy.load_state_dict(payload["models"]["single"])
            elif policy_choice == "active":
                if payload["models"]["dual"] is None:
                    raise ValueError("this boundary has no Active policy")
                policy = DualPolicy(config, StandalonePolicy(config))
                policy.load_state_dict(payload["models"]["dual"])
            else:
                policy = StandalonePolicy(config)
                policy.load_state_dict(payload["models"]["kb"])
        else:
            payload = load_artifact(path)
            if payload["kind"] != "inference" or policy_choice == "active":
                raise ValueError("expected an inference artifact; Active requires a complete stage checkpoint")
            config = config_from_dict(payload["config"])
            # Exports embed the fixed split manifest, but still verify current map bytes.
            from .data.manifest import MazeEntry, verify_manifest
            raw = payload["maze_manifest"]
            manifest = MazeManifest(**{key: tuple(MazeEntry(**x) for x in value) if key != "schema_version" else value
                                       for key, value in raw.items()})
            verify_manifest(manifest, config.environment.maze_root)
            encoder = VisionEncoder(config); encoder.load_state_dict(payload["encoder"])
            policy = SingleActorCritic(config) if payload["policy_kind"] == "single" else StandalonePolicy(config)
            policy.load_state_dict(payload["policy"])
        encoder.freeze()
        return policy, encoder, manifest, config, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("kb", "active"), default="kb")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    policy, encoder, manifest, config, payload = load_for_evaluation(args.checkpoint, args.policy)
    result = evaluate_policy(policy, encoder, manifest, config, split=args.split)
    result.update(method=payload["method"], seed=payload["seed"], policy_kind=payload["policy_kind"],
                  costs=payload.get("costs"), training_task=payload.get("task"))
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
