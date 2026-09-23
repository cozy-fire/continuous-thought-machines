"""Run the exact 760-transition integration smoke and audit independent boundary recovery."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from . import checkpoint as ck
from .config import Config, config_hash, load_config, resolved_dict, save_config
from .data.replay import TransitionStore
from .evaluate import evaluate_policy, load_for_evaluation, panel_envs
from .models import encode_obs
from .schedule import METHODS, expand_stages, isolated_rng
from .train import Runner


def require_smoke(config: Config) -> None:
    """Allow only the published smoke recipe, with an explicit device override."""
    expected = load_config(Path(__file__).parent / "configs/smoke.yaml")
    expected = replace(expected, training=replace(expected.training, device=config.training.device))
    if resolved_dict(config) != resolved_dict(expected):
        raise ValueError("verification requires the exact smoke configuration except device")


def state_hash(module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        # Permanent-freeze bookkeeping changes at the final TA boundary, not visual weights.
        if name.endswith("permanently_frozen"):
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compare_tree(left, right, *, atol: float, rtol: float, path: str = "root") -> float:
    """Compare numerical state without silently accepting missing keys or changed RNGs."""
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or left.shape != right.shape:
            raise AssertionError(f"tensor contract mismatch at {path}")
        floating = left.is_floating_point()
        torch.testing.assert_close(left, right, atol=atol if floating else 0,
                                   rtol=rtol if floating else 0, msg=lambda msg: f"{path}: {msg}")
        return float((left-right).abs().max()) if floating and left.numel() else 0.
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"dictionary keys mismatch at {path}")
        return max((compare_tree(value, right[key], atol=atol, rtol=rtol, path=f"{path}.{key}")
                    for key, value in left.items()), default=0.)
    if isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"sequence mismatch at {path}")
        return max((compare_tree(a, b, atol=atol, rtol=rtol, path=f"{path}[{i}]")
                    for i, (a, b) in enumerate(zip(left, right))), default=0.)
    if type(left) is not type(right) or left != right:
        raise AssertionError(f"value mismatch at {path}: {left!r} != {right!r}")
    return 0.


def pool_fingerprint(root: Path, ref: dict) -> dict:
    store = TransitionStore(ck.verify_reference(root, ref, replay=True))
    digest = hashlib.sha256()
    ids = []
    for record in store.take(list(range(len(store)))):
        raw = asdict(record)
        digest.update(raw.pop("obs").tobytes())
        digest.update(raw.pop("transition_next_obs").tobytes())
        digest.update(json.dumps(raw, sort_keys=True).encode())
        ids.append(record.transition_id)
    return dict(task=store.task, count=len(store), content_sha256=digest.hexdigest(), ids=ids,
                encoder_version=store.encoder_version, world_version=store.world_version)


def comparison_state(root: Path) -> dict:
    payload = ck.load(root, expected_sources=ck.source_manifest())
    keys = ("models", "world_optimizer", "fisher", "rng", "counters", "kb_ready", "completed", "next_index")
    return {**{key: payload[key] for key in keys},
            "pools": {task: pool_fingerprint(root, ref) for task, ref in payload["pools"].items()}}


class AuditedRunner(Runner):
    """Observe existing stage events; never replace rewards, losses, RNGs or evaluations."""
    def __init__(self, *args, **kwargs):
        self.audit_rows, self.current_audit, self.teacher = [], None, None
        self.previous_active = None
        super().__init__(*args, **kwargs)

    def _event(self, value):
        super()._event(value)
        event = value["type"]
        if event == "stage_enter":
            stage = self.stages[self.next_index]
            self.current_audit = dict(stage=value["stage"], phase=stage.key.phase, family=stage.key.family,
                task=stage.key.task, encoder_version=int(self.encoder.encoder_version),
                kb_ready=bool(self.kb_ready), encoder_before=state_hash(self.encoder),
                world_before=state_hash(self.world), kb_before=state_hash(self.kb),
                trainable_counts={name: sum(p.numel() for p in module.parameters() if p.requires_grad)
                    for name, module in (("encoder", self.encoder), ("world", self.world),
                                         ("kb", self.kb), ("policy", self.policy)) if module is not None},
                pool_before={task: pool_fingerprint(self.root, ref) for task, ref in self.pools.items()},
                fisher_before_sum=sum(x.sum().item() for x in self.fisher.importance.values()) if self.fisher else 0.)
            if stage.key.phase in ("X", "P"):
                initial = state_hash(self.policy.active.controller)
                if initial == state_hash(self.kb.controller) or initial == self.previous_active:
                    raise AssertionError("Active must be independently reinitialized")
                self.current_audit["active_initial_hash"] = initial
                self.current_audit["adapter_initial_gate"] = self.policy.adapter.gate.item()
                assert self.policy.adapter.gate.item() == 0
            if stage.key.phase == "C":
                self.teacher = self.policy
                self.current_audit["teacher_before"] = state_hash(self.teacher)
                teacher_ptrs = {p.data_ptr() for p in self.teacher.parameters()}
                assert not teacher_ptrs.intersection(p.data_ptr() for p in self.kb.parameters())
        elif event == "ppo_update":
            groups = ("attention.q", "attention.k", "attention.v", "attention.output", "synapse", "nlm")
            norms = {}
            for group in groups:
                gradients = [p.grad for name, p in self.policy.active.controller.named_parameters() if name.startswith(group)]
                assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
                norms[group] = sum(g.abs().sum().item() for g in gradients)
                assert norms[group] > 0, group
            norms["adapter"] = sum(p.grad.abs().sum().item() for p in self.policy.adapter.parameters() if p.grad is not None)
            self.current_audit.setdefault("gradient_groups", []).append(norms)
        elif event == "learner_update":
            assert all(math.isfinite(x) for x in value["metrics"].values())
            self.current_audit["metrics"] = value["metrics"]
        elif event == "stage_complete":
            row = self.current_audit
            row.update(encoder_after=state_hash(self.encoder), world_after=state_hash(self.world),
                       kb_after=state_hash(self.kb), counters=dict(self.counters),
                       encoder_version_after=int(self.encoder.encoder_version),
                       permanently_frozen=bool(self.encoder.permanently_frozen))
            if not value["stage"].endswith("W/fit"):
                assert row["encoder_before"] == row["encoder_after"]
                assert row["world_before"] == row["world_after"]
            else:
                assert row["encoder_before"] != row["encoder_after"]
                expected_high = 4 if row["task"] in row["pool_before"] else 0
                assert row["metrics"]["high_count"] == expected_high
            if row["phase"] != "C":
                assert row["kb_before"] == row["kb_after"]
            else:
                assert row["kb_before"] != row["kb_after"]
                assert row["teacher_before"] == state_hash(self.teacher)
                row["teacher_unchanged_and_storage_isolated"] = True
                self.teacher = None
            if row["phase"] == "X":
                pool = pool_fingerprint(self.root, self.pools[row["task"]])
                assert pool["task"] == row["task"] and pool["count"] == 20
                for other, before in row["pool_before"].items():
                    if other != row["task"]:
                        assert before == pool_fingerprint(self.root, self.pools[other])
                row["new_pool"] = pool
            if row["phase"] in ("X", "P"):
                self.previous_active = state_hash(self.policy.active.controller)
            if row["phase"] == "F":
                assert self.fisher.sample_count == 8
                total = sum(x.sum().item() for x in self.fisher.importance.values())
                decay = self.config.ewc.agnostic_decay if row["family"] == "ta" else self.config.ewc.pnc_decay
                assert math.isclose(total, decay*row["fisher_before_sum"]+row["metrics"]["fisher_sum"], rel_tol=1e-5, abs_tol=1e-6)
                for name, param in self.kb.named_parameters():
                    torch.testing.assert_close(param.detach().cpu(), self.fisher.theta_star[name], atol=0, rtol=0)
                row["fisher_total"] = total
                row["fisher_completed_compressions"] = self.fisher.completed_compressions
            self.audit_rows.append(row)
            ck.atomic_json(self.root / "smoke_audit.json", self.audit_rows)


@torch.no_grad()
def record_trajectories(runner: Runner, output: Path) -> dict:
    """Save actual uint8 observations and policy actions, without advancing training RNGs."""
    result = {}
    with isolated_rng():
        for task in runner.config.task_order:
            env = next(panel_envs(task, "validation", runner.manifest, runner.config))
            images, actions, rows = [], [], []
            try:
                obs, _ = env.reset()
                state = runner.kb.initial_state(1)
                for step in range(16):
                    logits, state = runner.kb.step(encode_obs(torch.from_numpy(obs)[None].to(runner.device), runner.encoder),
                        state, torch.tensor([step == 0], device=runner.device))
                    action = int(logits.argmax(-1))
                    images.append(obs.copy()); actions.append(action)
                    obs, reward, terminated, truncated, info = env.step(action)
                    rows.append(dict(action=action, reward=reward, terminated=terminated, truncated=truncated))
                    if terminated or truncated:
                        break
                images.append(obs.copy())
            finally:
                env.close()
            path = output / f"{task}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, observations=np.stack(images), actions=np.asarray(actions, dtype=np.int64))
            result[task] = dict(transitions=len(actions), frames=len(images), file=path.name,
                                sha256=ck.file_hash(path), steps=rows)
    ck.atomic_json(output / "manifest.json", result)
    return result


def resume_probe(root: Path, expected: Path, output: Path) -> None:
    config = load_config(root / "resolved_config.yaml")
    require_smoke(config)
    runner = AuditedRunner(config, METHODS[0], 0, root, resume=True)
    before = dict(runner.counters)
    runner.run_next()
    actual, reference = comparison_state(root), torch.load(expected, weights_only=True)
    atol, rtol = (0., 0.) if config.training.device == "cpu" else (1e-5, 1e-4)
    error = compare_tree(reference, actual, atol=atol, rtol=rtol)
    ck.atomic_json(output, dict(status="passed", stage=runner.completed[-1], max_abs_error=error,
        atol=atol, rtol=rtol, counters=runner.counters, RNG_and_replay_exact=True,
        additional_training_steps=runner.counters["global_env_steps"]-before["global_env_steps"],
        additional_evaluation_steps=runner.counters["eval_steps"]-before["eval_steps"]))


def verify(config: Config, output: Path, manifest: Path) -> None:
    require_smoke(config)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_config(config, output / "gpu_or_cpu_smoke.yaml")
    runner = AuditedRunner(config, METHODS[0], 0, output / "run", manifest_path=manifest)
    for _ in range(1, len(runner.stages)):
        runner.run_next()
        key = runner.completed[-1]
        print(f"committed {key}: {runner.counters['global_env_steps']} transitions", flush=True)
        if key == "ta/v0/maze_medium/r0/W/fit":
            shutil.copytree(runner.root, output / "recovery_W")
        elif key == "ta/v0/maze_medium/r0/X":
            ck.atomic_torch(output / "expected_after_X.pt", comparison_state(runner.root))
            shutil.copytree(runner.root, output / "recovery_X")
        elif key == "ta/v0/maze_medium/r0/C":
            ck.atomic_torch(output / "expected_after_C.pt", comparison_state(runner.root))
    assert runner.finalized and runner.counters["global_env_steps"] == 760
    expected = dict(world_collect_steps=160, explore_steps=160, compress_steps=240, fisher_steps=120,
                    progress_steps=80, world_optimizer_updates=8, ppo_optimizer_updates=12, distill_optimizer_updates=12)
    for name, value in expected.items():
        assert runner.counters[name] == value, name
    assert runner.fisher.completed_compressions == 6 and int(runner.encoder.encoder_version) == 4
    for label, expected_name in (("W", "X"), ("X", "C")):
        with (output / f"recovery_{label}.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "tasks.continual_nav.verify_smoke", "--resume-probe",
                str(output / f"recovery_{label}"), "--expected", str(output / f"expected_after_{expected_name}.pt"),
                "--output-dir", str(output / f"recovery_{label}.json")], check=True, stdout=stream, stderr=subprocess.STDOUT)
        print(f"verified independent recovery from {label}", flush=True)
    policy, encoder, fixed, restored_config, _ = load_for_evaluation(runner.root / "exports/final.pt")
    assert state_hash(policy) == state_hash(runner.kb) and state_hash(encoder) == state_hash(runner.encoder)
    test = evaluate_policy(policy.to(runner.device), encoder.to(runner.device), fixed, restored_config, split="test")
    original = json.loads((runner.root / "evaluation/final_test.json").read_text())
    assert test["tasks"] == original["tasks"]
    trajectories = record_trajectories(runner, output / "trajectories")
    baselines = []
    for method, task in ((METHODS[1], "maze_medium"), (METHODS[1], "fourrooms"), (METHODS[2], None), (METHODS[3], None)):
        baseline = Runner(config, method, 0, output / "baselines" / (method+"_"+str(task)), task=task,
            manifest_path=runner.root / "manifests/maze_splits.json", vision_checkpoint=runner.root / "exports/vision_final.pt")
        assert state_hash(baseline.encoder) == state_hash(runner.encoder)
        assert baseline.world is None and baseline.fisher is None
        assert baseline.vision_source["sha256"] == runner.vision_export["sha256"]
        controller = baseline.policy.controller if baseline.single else baseline.kb.controller
        assert state_hash(controller) != state_hash(runner.kb.controller)
        baselines.append(dict(method=method, task=task, vision_hash=baseline.vision_source["sha256"],
                              costs=baseline.costs(), initialized_only=True))
    recovery = {label: json.loads((output / f"recovery_{label}.json").read_text()) for label in ("W", "X")}
    report = dict(status="passed", config_hash=config_hash(config), device=str(runner.device),
        provenance=json.loads((runner.root / "provenance.json").read_text()), counters=runner.counters,
        stage_audit=runner.audit_rows, recovery=recovery, baselines=baselines, trajectories=trajectories,
        costs=dict(main_training=760, recovery_training=sum(x["additional_training_steps"] for x in recovery.values()),
            recovery_evaluation=sum(x["additional_evaluation_steps"] for x in recovery.values()),
            main_evaluation=runner.counters["eval_steps"], exported_policy_recheck=test["transitions"],
            trajectory_steps=sum(x["transitions"] for x in trajectories.values())),
        final_export_matches=True, fixed_test_recheck_matches=True, baselines_trained=False)
    ck.atomic_json(output / "smoke_report.json", report)
    (output / "smoke_report.md").write_text("# Complete continual navigation smoke\n\n"
        "Status: passed. Main training: 760 transitions; world updates: 8.\n\n"
        "Four TA rounds and two downstream P/C/F visits completed. W and X boundaries were resumed in separate processes.\n\n"
        "See smoke_report.json for configuration, provenance, each phase audit, recovery tolerance, input/RNG comparisons, "
        "fixed-test results, shared-vision baseline initialization and separate interaction costs.\n\n"
        "This is a correctness smoke, not evidence of learned task performance or full-budget GPU capacity.\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/smoke.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maze-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-probe", type=Path)
    parser.add_argument("--expected", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.resume_probe is not None:
        if args.expected is None:
            parser.error("--resume-probe requires --expected")
        resume_probe(args.resume_probe, args.expected, args.output_dir)
    else:
        if args.maze_manifest is None:
            parser.error("full smoke requires --maze-manifest")
        config = load_config(args.config)
        config = replace(config, training=replace(config.training, device=args.device))
        verify(config, args.output_dir, args.maze_manifest)


if __name__ == "__main__":
    main()
