"""Fixed-panel argmax evaluation and same-trajectory visual drift, isolated from training."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .analysis.metrics import episode_metrics
from .checkpoint import atomic_json, load, load_artifact, verify_reference
from .config import Config, config_from_dict
from .contracts import CTMState, DualState, PolicyOutput, PolicyState
from .data.manifest import MazeManifest
from .envs.evaluation import EvaluationPool, shortest_path
from .models import DualPolicy, SingleActorCritic, StandalonePolicy, VisionEncoder, encode_obs, frozen_copy
from .schedule import isolated_rng


def panel_specs(task: str, split: str, manifest: MazeManifest, config: Config, *, drift: bool = False):
    if split not in ("validation", "test") or (drift and split != "validation"):
        raise ValueError("invalid evaluation panel")
    count = config.evaluation.drift_episodes if drift else (
        config.evaluation.validation_episodes if split == "validation" else config.evaluation.test_episodes)
    if task == "maze_medium":
        panel = manifest.drift_panel if drift else manifest.entries(split, panel=True)
        if len(panel) < count:
            raise ValueError("insufficient fixed Maze evaluation panel")
        for entry in panel[:count]:
            yield task, split, config.environment.maze_root, entry
    elif task == "fourrooms":
        start = config.evaluation.fourrooms_validation_seed_start if split == "validation" else config.evaluation.fourrooms_test_seed_start
        for seed in range(start, start+count):
            yield task, split, None, seed
    else:
        raise ValueError("unknown evaluation task")


def panel_envs(task: str, split: str, manifest: MazeManifest, config: Config, *, drift: bool = False):
    from .envs.evaluation import create_panel_env
    for spec in panel_specs(task, split, manifest, config, drift=drift):
        yield create_panel_env(spec)


def _select_state(state: PolicyState, indices: torch.Tensor) -> PolicyState:
    if isinstance(state, DualState):
        return DualState(_select_state(state.kb, indices), _select_state(state.active, indices))
    return CTMState(state.pre[indices], state.post[indices])


def _assign_state(state: PolicyState, indices: torch.Tensor, value: PolicyState) -> None:
    if isinstance(state, DualState):
        _assign_state(state.kb, indices, value.kb)
        _assign_state(state.active, indices, value.active)
    else:
        state.pre[indices] = value.pre
        state.post[indices] = value.post


def _step(policy, encoder, obs, state, start):
    device = next(encoder.parameters()).device
    image = torch.from_numpy(obs).unsqueeze(0).to(device)
    result = policy.step(encode_obs(image, encoder), state, torch.tensor([start], device=device))
    return (result.logits, result.state) if isinstance(result, PolicyOutput) else result


@torch.no_grad()
def evaluate_policy(policy, encoder: VisionEncoder, manifest: MazeManifest, config: Config,
                    *, split: str = "validation", drift: bool = False,
                    tasks: tuple[str, ...] | None = None) -> dict:
    selected = config.task_order if tasks is None else tasks
    if not selected or len(set(selected)) != len(selected) or any(t not in config.task_order for t in selected):
        raise ValueError("invalid evaluation task subset")
    started = perf_counter()
    with isolated_rng():
        policy, encoder = frozen_copy(policy), frozen_copy(encoder)
        device = next(encoder.parameters()).device
        results, total = {}, 0
        panels = {task: list(panel_specs(task, split, manifest, config, drift=drift)) for task in selected}
        size = min(config.evaluation.num_envs, max(map(len, panels.values())))
        with EvaluationPool(size, config.evaluation.backend) as pool:
            for task, specs in panels.items():
                episodes = [None] * len(specs)
                state = policy.initial_state(size)
                # Slot identities are stable; panel identities advance only on completion.
                active = {slot: slot for slot in range(min(size, len(specs)))}
                next_index = len(active)
                obs, distances, lengths, returns = {}, {}, {}, {}
                reset = pool.exchange("reset", {slot: specs[index] for slot, index in active.items()})
                for slot, (frame, distance) in reset.items():
                    obs[slot], distances[slot], lengths[slot], returns[slot] = frame, distance, 0, 0.
                while active:
                    slots = sorted(active)
                    indices = torch.tensor(slots, device=device)
                    images = torch.from_numpy(np.stack([obs[slot] for slot in slots])).to(device)
                    starts = torch.tensor([lengths[slot] == 0 for slot in slots], device=device)
                    output = policy.step(encode_obs(images, encoder), _select_state(state, indices), starts)
                    logits, proposed = (output.logits, output.state) if isinstance(output, PolicyOutput) else output
                    _assign_state(state, indices, proposed)
                    actions = logits.argmax(-1).cpu().tolist()
                    steps = pool.exchange("step", dict(zip(slots, actions)))
                    resets = {}
                    for slot, (frame, reward, term, trunc, info) in steps.items():
                        obs[slot] = frame
                        returns[slot] += reward
                        lengths[slot] += 1
                        total += 1
                        if term or trunc:
                            index = active.pop(slot)
                            episodes[index] = dict(success=bool(info["success"]), length=lengths[slot],
                                **{"return": returns[slot]}, shortest_path=distances[slot])
                            if next_index < len(specs):
                                active[slot] = next_index
                                resets[slot] = specs[next_index]
                                next_index += 1
                    for slot, (frame, distance) in pool.exchange("reset", resets).items():
                        obs[slot], distances[slot], lengths[slot], returns[slot] = frame, distance, 0, 0.
                        # episode_start resets both recurrent columns to learned initial state.
                    if not active and any(episode is None for episode in episodes):
                        raise RuntimeError("incomplete evaluation panel")
                results[task] = {**episode_metrics(episodes), "raw_episodes": episodes}
        elapsed = perf_counter()-started
        return dict(split=split, drift_panel=drift, tasks=results, transitions=total,
                    elapsed_seconds=elapsed, episodes_per_second=sum(map(len, panels.values()))/elapsed)


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
    parser.add_argument("--backend", choices=("serial", "subprocess"))
    parser.add_argument("--num-envs", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    policy, encoder, manifest, config, payload = load_for_evaluation(args.checkpoint, args.policy)
    config = replace(config, evaluation=replace(config.evaluation,
        backend=args.backend or config.evaluation.backend,
        num_envs=args.num_envs if args.num_envs is not None else config.evaluation.num_envs))
    from .config import validate_config
    validate_config(config)
    result = evaluate_policy(policy, encoder, manifest, config, split=args.split)
    result.update(method=payload["method"], seed=payload["seed"], policy_kind=payload["policy_kind"],
                  costs=payload.get("costs"), training_task=payload.get("task"))
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
