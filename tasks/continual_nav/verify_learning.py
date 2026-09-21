"""Bounded delivery-4 X/C/F and single-column P verification on real pixel environments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import torch

from .config import config_hash, load_config
from .data.manifest import MazeManifest
from .data.replay import ReplayBank
from .data.rollout import collect_fisher
from .envs import VectorEnvAdapter, build_env
from .learning.distill import run_compress_stage
from .learning.fisher import estimate_fisher, update_online_fisher
from .learning.ppo import run_ppo_stage
from .models import DualPolicy, SingleActorCritic, StandalonePolicy, VisionEncoder, WorldModel, frozen_copy
from .verify_models import tensor_hash


def verify(config_path: Path, output_dir: Path, manifest_path: Path) -> dict:
    config = load_config(config_path)
    budgets = (config.exploration.steps_per_round, config.distill.agnostic_steps_per_round,
               config.fisher.collect_steps, config.pnc.progress_steps)
    if max(budgets) > 100 or config.training.num_envs > 2 or config.fisher.scored_samples > 16:
        raise ValueError("learning verification requires small smoke budgets")
    manifest = MazeManifest.load(manifest_path, Path(config.environment.maze_root).resolve())
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(31)
    encoder = VisionEncoder(config)
    world, kb = WorldModel(config, encoder), StandalonePolicy(config)
    world_hash = tensor_hash(world)
    bank = ReplayBank(output_dir / "replay", capacity=config.replay.high_error_capacity_per_task)
    action_rng = torch.Generator().manual_seed(32)
    minibatch_rng = torch.Generator().manual_seed(33)
    fisher_rng = torch.Generator().manual_seed(34)
    previous, next_id, reports = None, 0, []
    for task_index, task in enumerate(config.task_order):
        envs = VectorEnvAdapter([build_env(task, "train", 31+100*task_index+i, config=config,
                                  maze_manifest=manifest) for i in range(config.training.num_envs)])
        builder = None
        try:
            policy = DualPolicy(config, kb, kb_ready=previous is not None)
            kb_before, active_before = tensor_hash(kb), tensor_hash(policy.active)
            builder = bank.begin(task, encoder_version=0, world_version=0, source_stage=f"verify/{task}/X",
                                  expected_count=config.exploration.steps_per_round)
            x = run_ppo_stage(envs, policy, encoder, config, phase="X", steps=config.exploration.steps_per_round,
                world=world, high_error=builder, action_rng=action_rng, minibatch_rng=minibatch_rng,
                start_transition_id=next_id)
            next_id = x["next_transition_id"]
            assert tensor_hash(kb) == kb_before and tensor_hash(world) == world_hash
            assert tensor_hash(policy.active) != active_before
            assert len(bank.latest(task)) == config.replay.high_error_capacity_per_task
            teacher = frozen_copy(policy)
            teacher_hash = tensor_hash(teacher)
            c = run_compress_stage(envs, teacher, kb, encoder, config, previous,
                steps=config.distill.agnostic_steps_per_round, action_rng=action_rng,
                minibatch_rng=minibatch_rng, start_transition_id=next_id)
            next_id = c["next_transition_id"]
            assert tensor_hash(kb) != kb_before and tensor_hash(teacher) == teacher_hash
            assert tensor_hash(world) == world_hash
            kb_after = tensor_hash(kb)
            sequences = collect_fisher(envs, kb, encoder, config, action_rng=action_rng,
                                       fisher_rng=fisher_rng, start_transition_id=next_id)
            next_id += config.fisher.collect_steps
            current = estimate_fisher(kb, encoder, sequences, config)
            state = update_online_fisher(kb, current, previous, decay=config.ewc.agnostic_decay,
                sample_count=config.fisher.scored_samples, stage_key=f"verify/{task}/F", encoder_version=0)
            if previous is not None:
                for name, importance in state.importance.items():
                    torch.testing.assert_close(importance, config.ewc.agnostic_decay*previous.importance[name]+current[name])
                assert c["last"]["ewc"] > 0
            assert tensor_hash(kb) == kb_after and tensor_hash(world) == world_hash
            assert all(p.grad is None for p in kb.parameters())
            fisher_total = sum(f.sum().item() for f in current.values())
            assert fisher_total > 0
            reports.append(dict(task=task, X=x, C=c, fisher_samples=config.fisher.scored_samples,
                                fisher_current_sum=fisher_total, completed_compressions=state.completed_compressions,
                                teacher_unchanged=True, world_unchanged=True, fisher_kb_unchanged=True))
            previous = state
            print(f"verified {task}: X={x['updates']} updates, C={c['updates']} updates, F={state.sample_count} scores", flush=True)
        finally:
            if builder is not None:
                builder.abort()
            envs.close()
    # Exercise the permanent-vision single-column baseline without g/F or EWC.
    encoder.freeze(permanent=True)
    permanent_hash = tensor_hash(encoder)
    policy = SingleActorCritic(config)
    before = tensor_hash(policy)
    envs = VectorEnvAdapter([build_env("fourrooms", "train", 200+i, config=config)
                             for i in range(config.training.num_envs)])
    try:
        p = run_ppo_stage(envs, policy, encoder, config, phase="P", steps=config.pnc.progress_steps,
                          action_rng=action_rng, minibatch_rng=minibatch_rng, start_transition_id=next_id)
    finally:
        envs.close()
    assert tensor_hash(encoder) == permanent_hash and tensor_hash(policy) != before
    report = dict(status="passed", scope="delivery_4_real_pixel_learning_not_full_schedule",
        world_initialization="random_frozen_no_W_fit_in_this_probe", device="cpu", python=platform.python_version(),
        torch=torch.__version__, config_hash=config_hash(config), environment_transitions=p["next_transition_id"],
        ppo_updates=sum(r["X"]["updates"] for r in reports)+p["updates"],
        distill_updates=sum(r["C"]["updates"] for r in reports), fisher_scored_samples=2*config.fisher.scored_samples,
        rounds=reports, single_column_P=p, permanent_encoder_unchanged=True)
    (output_dir / "learning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maze-manifest", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(verify(args.config, args.output_dir, args.maze_manifest), indent=2))


if __name__ == "__main__":
    main()
