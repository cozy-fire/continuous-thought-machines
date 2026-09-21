"""Bounded real-environment W/replay verification; X is a probe without PPO updates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import config_hash, load_config
from .contracts import Transition
from .data.manifest import MazeManifest
from .data.replay import ReplayBank, ShardedWriter, TransitionStore
from .envs import VectorEnvAdapter, build_env
from .learning.world import SnapshotCollector, fit_world_model, make_world_optimizer
from .models import DualPolicy, SIGReg, StandalonePolicy, VisionEncoder, WorldModel, encode_obs
from .verify_models import tensor_hash


@torch.no_grad()
def exploration_probe(world, policy, envs, builder, steps, rng, next_id):
    """Exercise reward/top-K interfaces with a fixed random Active, not trained exploration."""
    obs, _ = envs.reset()
    starts = np.ones(envs.num_envs, bool)
    state = policy.initial_state(envs.num_envs)
    before = tensor_hash(world)
    first = None
    for _ in range(steps // envs.num_envs):
        images = torch.from_numpy(obs)
        output = policy.step(encode_obs(images, world.encoder), state, torch.from_numpy(starts))
        state = output.state
        actions = torch.multinomial(output.logits.softmax(-1), 1, generator=rng).squeeze(-1)
        result = envs.step(actions.numpy())
        next_images = torch.from_numpy(result.transition_next_obs)
        reward, error = world.curiosity(images, next_images, actions)
        assert torch.isfinite(reward).all() and not reward.requires_grad
        if first is None:
            first = (images.clone(), next_images.clone(), actions.clone(), error.clone())
        for slot, info in enumerate(result.info):
            builder.offer(Transition(obs[slot], result.transition_next_obs[slot], int(actions[slot]),
                bool(result.terminated[slot]), bool(result.truncated[slot]), bool(starts[slot]),
                int(info["episode_id"])*envs.num_envs+slot, int(info["episode_step"]), builder.task,
                builder.encoder_version, builder.world_version, next_id), float(error[slot]))
            next_id += 1
        obs, starts = result.next_obs, result.next_episode_start
    assert tensor_hash(world) == before
    torch.testing.assert_close(world.curiosity(*first[:3])[1], first[3], atol=0, rtol=0)
    return next_id


def verify(config_path: Path, output_dir: Path, manifest_path: Path) -> dict:
    config = load_config(config_path)
    # This command must stay bounded even if accidentally given full.yaml.
    if config.world.collect_steps_per_round > 1000 or config.world.updates_per_round > 10 or config.exploration.steps_per_round > 1000:
        raise ValueError("verification requires small smoke budgets")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = MazeManifest.load(manifest_path, Path(config.environment.maze_root).resolve())
    torch.manual_seed(17)
    world = WorldModel(config, VisionEncoder(config))
    kb = StandalonePolicy(config)
    kb_before = tensor_hash(kb)
    optimizer = make_world_optimizer(world, config)
    regularizer = SIGReg(config.sigreg)
    bank = ReplayBank(output_dir / "replay", capacity=config.replay.high_error_capacity_per_task,
                      shard_size=config.replay.shard_size)
    replay_rng = np.random.default_rng(21)
    sigreg_rng = torch.Generator().manual_seed(22)
    action_rng = torch.Generator().manual_seed(23)
    next_id, rounds = 0, []
    for task_index, task in enumerate(config.task_order):
        for local_round in range(2):
            label = f"{task}_{local_round}"
            envs = VectorEnvAdapter([build_env(task, "train", 100*task_index+i, config=config,
                                     maze_manifest=manifest) for i in range(config.training.num_envs)])
            builder = None
            try:
                snapshot = SnapshotCollector(world.encoder, kb, config, world_version=int(world.world_model_version))
                snapshot_before = tensor_hash(snapshot.encoder), tensor_hash(snapshot.kb)
                writer = ShardedWriter(output_dir / label, task=task, encoder_version=snapshot.encoder_version,
                    world_version=snapshot.world_version, source="W", source_stage=label,
                    expected_count=config.world.collect_steps_per_round, snapshot_id=snapshot.snapshot_id,
                    shard_size=config.replay.shard_size)
                collected = snapshot.collect(envs, writer, steps=config.world.collect_steps_per_round,
                                              action_rng=action_rng, start_transition_id=next_id)
                next_id = collected.next_transition_id
                before = [tensor_hash(m) for m in (world.encoder, world.projector, world.predictor)]
                metrics = fit_world_model(world, regularizer, TransitionStore(collected.manifest), bank.latest(task),
                    optimizer, config, task=task, replay_rng=replay_rng, sigreg_rng=sigreg_rng)
                assert all(tensor_hash(m) != h for m, h in zip((world.encoder, world.projector, world.predictor), before))
                assert snapshot_before == (tensor_hash(snapshot.encoder), tensor_hash(snapshot.kb))
                ev, wv = world.commit_fit()
                # In-memory version commit is sufficient only for this non-resumable probe.
                # The production scheduler must persist a phase checkpoint before entering X.
                policy = DualPolicy(config, kb, kb_ready=False).eval()
                builder = bank.begin(task, encoder_version=ev, world_version=wv, source_stage=label+"_X_probe",
                                     expected_count=config.exploration.steps_per_round)
                next_id = exploration_probe(world, policy, envs, builder, config.exploration.steps_per_round,
                                             action_rng, next_id)
                pool = builder.finish()
                assert len(pool) == min(config.replay.high_error_capacity_per_task, config.exploration.steps_per_round)
                assert tensor_hash(kb) == kb_before
                rounds.append(dict(task=task, round=local_round, collected=collected.transitions,
                    x_probe_transitions=config.exploration.steps_per_round, encoder_version=ev, world_version=wv,
                    pool_size=len(pool), fresh_manifest=collected.manifest, pool_manifest=str(pool.manifest), **metrics))
                print(f"verified {label}: world={wv}, high_count={metrics['high_count']}", flush=True)
            finally:
                if builder is not None:
                    builder.abort()
                envs.close()
    assert [r["high_count"] for r in rounds] == [0, config.world.batch_size//2, 0, config.world.batch_size//2]
    assert all(int(s["step"]) == 4*config.world.updates_per_round for s in optimizer.state.values())
    report = dict(status="passed", scope="delivery_3_W_and_fixed_Active_X_interface_probe_no_PPO",
        device="cpu", torch=torch.__version__, config_hash=config_hash(config),
        environment_transitions=next_id, optimizer_updates=4*config.world.updates_per_round,
        snapshot_isolated=True, frozen_world_unchanged=True, kb_unchanged=True, rounds=rounds)
    (output_dir / "world_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maze-manifest", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(verify(args.config, args.output_dir, args.maze_manifest), indent=2))


if __name__ == "__main__":
    main()
