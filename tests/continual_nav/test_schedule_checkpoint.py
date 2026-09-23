"""Stage boundaries, committed dependencies and single/dual reconstruction."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from fixtures import map_file
from tasks.continual_nav import checkpoint as ck
from tasks.continual_nav.config import Config, load_config, resolved_dict
from tasks.continual_nav.data.manifest import build_manifest
from tasks.continual_nav.schedule import METHODS, RandomStreams, derive_seed, expand_stages
from tasks.continual_nav.train import Runner
from tasks.continual_nav.evaluate import load_for_evaluation
from tasks.continual_nav.verify_models import tensor_hash


def setUpModule():
    global _threads
    _threads = torch.get_num_threads(); torch.set_num_threads(2)


def tearDownModule():
    torch.set_num_threads(_threads)


def fake_evaluate(*args, **kwargs):
    return dict(transitions=2, tasks={task: dict(success_rate=0.) for task in ("maze_medium", "fourrooms")})


class ScheduleTests(unittest.TestCase):
    def test_atomic_json_retries_transient_windows_replace_denial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.json"
            path.write_text('{"step": 0}', encoding="utf-8")
            replace = ck.os.replace
            calls = 0

            def transient_denial(source, destination):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise PermissionError("temporarily locked")
                replace(source, destination)

            with patch.object(ck.os, "name", "nt"), patch.object(ck.os, "replace", side_effect=transient_denial):
                ck.atomic_json(path, {"step": 1})
            self.assertEqual(calls, 3)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"step": 1})

    def test_all_schedules_and_seed_formula(self):
        full = Config()
        main = expand_stages(full, METHODS[0])
        self.assertEqual(sum(s.transitions for s in main), 32290112)
        self.assertEqual(sum(s.key.phase == "F" for s in main), 22)
        self.assertEqual(sum(s.key.subphase == "collect" for s in main), 16)
        self.assertEqual(sum(s.world_updates for s in main), 80000)
        self.assertEqual([s.ordinal for s in main], list(range(len(main))))
        smoke = load_config("tasks/continual_nav/configs/smoke.yaml")
        self.assertEqual([sum(s.transitions for s in expand_stages(smoke, method, "maze_medium" if method == METHODS[1] else None))
                          for method in METHODS], [760, 40, 80, 200])
        self.assertNotEqual(derive_seed(0, METHODS[0], None, "env"), derive_seed(0, METHODS[1], "maze_medium", "env"))
        with self.assertRaises(ValueError):
            expand_stages(smoke, METHODS[1])

    def test_rng_roundtrip_and_initialization_isolation(self):
        streams = RandomStreams(0, METHODS[0], None)
        state = deepcopy(streams.state_dict())
        values = torch.rand(3, generator=streams.torch["policy_action"])
        seeds = streams.numpy["env"].integers(100, size=4)
        streams.load_state_dict(state)
        torch.testing.assert_close(values, torch.rand(3, generator=streams.torch["policy_action"]))
        self.assertEqual(seeds.tolist(), streams.numpy["env"].integers(100, size=4).tolist())
        before = torch.get_rng_state().clone()
        with streams.model_initialization(2):
            first = torch.rand(4)
        with streams.model_initialization(2):
            second = torch.rand(4)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        for i in range(5): map_file(self.data, i)
        for i in range(2): map_file(self.data, i, "test")
        self.manifest = build_manifest(self.data, validation_count=2, validation_episodes=2, test_episodes=2, drift_episodes=2)
        base = load_config("tasks/continual_nav/configs/smoke.yaml")
        self.c = replace(base, environment=replace(base.environment, maze_root=str(self.data)),
                         world=replace(base.world, collect_steps_per_round=4, updates_per_round=1, batch_size=2),
                         exploration=replace(base.exploration, steps_per_round=4),
                         distill=replace(base.distill, agnostic_steps_per_round=4),
                         pnc=replace(base.pnc, progress_steps=4, compress_steps=4),
                         fisher=replace(base.fisher, collect_steps=4, scored_samples=2))
        self.patches = [patch("tasks.continual_nav.train.prepare_manifest", return_value=self.manifest),
                        patch("tasks.continual_nav.train.evaluate_policy", side_effect=fake_evaluate),
                        patch("tasks.continual_nav.train.visual_drift", return_value=dict(transitions=2, tasks={}))]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)

    def runner(self, name, **kwargs):
        return Runner(self.c, kwargs.pop("method", METHODS[0]), 0, self.root / name, **kwargs)

    def compare(self, left, right):
        self.assertEqual(left.counters, right.counters)
        self.assertEqual(tensor_hash(left.encoder), tensor_hash(right.encoder))
        if left.kb is not None:
            self.assertEqual(tensor_hash(left.kb), tensor_hash(right.kb))
        if left.policy is not None:
            self.assertEqual(tensor_hash(left.policy), tensor_hash(right.policy))
        for name in left.rng.torch:
            torch.testing.assert_close(left.rng.torch[name].get_state(), right.rng.torch[name].get_state(), atol=0, rtol=0)

    def test_W_collect_W_fit_X_C_boundaries_resume_exactly(self):
        original = self.runner("a")
        original.run_next()  # W.collect
        shutil.copytree(original.root, self.root / "b")
        restored = self.runner("b", resume=True)
        self.compare(original, restored)
        for _ in range(3):  # W.fit, X, C
            original.run_next(); restored.run_next()
            self.compare(original, restored)
            restored = self.runner("b", resume=True)
            self.compare(original, restored)
        self.assertTrue(restored.kb_ready)
        self.assertIsNone(restored.policy)
        self.assertTrue(all(int(s["step"]) == 1 for s in restored.world_optimizer.state.values()))
        logged = [json.loads(line) for line in (original.root / "metrics.jsonl").read_text().splitlines()]
        world = next(row for row in logged if row["type"] == "world_update")
        distill = next(row for row in logged if row["type"] == "distill_update")
        self.assertEqual(world["world_optimizer_updates"], 1)
        self.assertEqual(distill["consumed"], 4)
        events = EventAccumulator(str(original.root / "tensorboard")).Reload()
        self.assertEqual(events.Scalars(world["stage"]+"/curve/loss")[-1].step, 1)
        self.assertEqual(events.Scalars(distill["stage"]+"/curve/loss")[-1].step, 12)

    def vision(self):
        origin = self.runner("source")
        origin.encoder.freeze(permanent=True)
        path = self.root / "vision.pt"
        ck.save_artifact(path, dict(kind="vision", method=METHODS[0], seed=0, config=resolved_dict(self.c),
            encoder=origin.encoder.state_dict(), shared_ta_steps=1234,
            maze_manifest_hash=origin.manifests["maze"]["sha256"]))
        return path

    def test_single_baseline_resume_without_phantom_KB(self):
        vision = self.vision()
        a = self.runner("a", method=METHODS[2], vision_checkpoint=vision)
        initial_encoder = tensor_hash(a.encoder)
        self.assertIsNone(a.kb); self.assertIsNone(a.world)
        a.run_next()
        shutil.copytree(a.root, self.root / "b")
        b = self.runner("b", method=METHODS[2], resume=True)
        self.compare(a, b)
        a.run_next(); b.run_next()
        self.compare(a, b)
        self.assertEqual(tensor_hash(a.encoder), initial_encoder)
        payload = ck.load(a.root)
        self.assertTrue(payload["finalized"])
        self.assertIsNone(payload["models"]["kb"])
        self.assertIsNone(payload["world_optimizer"])
        self.assertEqual(a.costs()["shared_visual_TA_generation_steps"], 1234)
        self.assertEqual(a.vision_source["sha256"], ck.file_hash(vision))
        for path in (a.root / "exports/final.pt", a.root / "checkpoints/latest.json"):
            policy, encoder, _, _, _ = load_for_evaluation(path)
            self.assertEqual(tensor_hash(policy), tensor_hash(a.policy))
            self.assertEqual(tensor_hash(encoder), tensor_hash(a.encoder))

    def test_conditional_pnc_resumes_P_C_F_and_preserves_shared_encoder(self):
        vision = self.vision()
        a = self.runner("a", method=METHODS[3], vision_checkpoint=vision)
        initial_encoder = tensor_hash(a.encoder)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            a.finalize()
        a.run_next()  # P leaves Active available for compression and separate evaluation.
        policy, _, _, _, _ = load_for_evaluation(a.root / "checkpoints/latest.json", "active")
        self.assertEqual(tensor_hash(policy), tensor_hash(a.policy))
        shutil.copytree(a.root, self.root / "b")
        b = self.runner("b", method=METHODS[3], resume=True)
        for _ in range(2):  # C and F, including reconstruction after C discarded Active.
            a.run_next(); b.run_next()
            self.compare(a, b)
            b = self.runner("b", method=METHODS[3], resume=True)
        self.assertEqual(tensor_hash(a.encoder), initial_encoder)
        self.assertIsNone(b.world)
        self.assertIsNotNone(b.fisher)
        self.assertEqual(b.fisher.completed_compressions, 1)
        for name in a.fisher.importance:
            torch.testing.assert_close(a.fisher.importance[name], b.fisher.importance[name], atol=0, rtol=0)
        logged = [json.loads(line) for line in (a.root / "metrics.jsonl").read_text().splitlines()]
        self.assertTrue(any(row["type"] == "ppo_update" and row["stage"].endswith("/P") for row in logged))
        self.assertTrue(any(row["type"] == "distill_update" and row["stage"].endswith("/C") for row in logged))
        events = EventAccumulator(str(a.root / "tensorboard")).Reload()
        self.assertEqual(events.Scalars("pnc/v0/maze_medium/P/policy_loss")[-1].step, 4)
        self.assertEqual(events.Scalars("pnc/v0/maze_medium/C/curve/loss")[-1].step, 8)

    def test_final_TA_boundary_exports_frozen_vision_and_enters_P(self):
        run = self.runner("a")
        # Synthetic pre-boundary fixture: this is a lifecycle test, not a full TA run.
        run.next_index = max(s.ordinal for s in run.stages if s.key.family == "ta")
        run.completed = [str(s.key) for s in run.stages[:run.next_index]]
        run.counters["global_env_steps"] = sum(s.transitions for s in run.stages[:run.next_index])
        run.kb_ready = True
        run.run_next()
        self.assertTrue(bool(run.encoder.permanently_frozen))
        self.assertEqual(run.stages[run.next_index].key.phase, "P")
        artifact = ck.load_artifact(run.root / "exports/vision_final.pt")
        self.assertEqual(artifact["shared_ta_steps"], run.counters["global_env_steps"])
        restored = self.runner("a", resume=True)
        self.compare(run, restored)
        baseline = self.runner("single", method=METHODS[1], task="maze_medium",
                               vision_checkpoint=run.root / "exports/vision_final.pt")
        self.assertEqual(tensor_hash(baseline.encoder), tensor_hash(run.encoder))
        baseline.run_next()
        self.assertTrue(baseline.finalized)
        before = tensor_hash(restored.encoder)
        restored.run_next()
        self.assertEqual(tensor_hash(restored.encoder), before)

    def test_missing_corrupt_uncommitted_dependencies_and_config_rejected(self):
        run = self.runner("a")
        run.run_next()
        with self.assertRaisesRegex(ValueError, "configuration"):
            ck.load(run.root, expected_config_hash="wrong")
        with self.assertRaisesRegex(ValueError, "source"):
            ck.load(run.root, expected_sources={})
        temp = run.root / "checkpoints/half_written.pt"
        temp.write_bytes(b"incomplete")
        self.assertEqual(ck.load(run.root)["current_stage"], str(run.stages[1].key))
        manifest = ck.contained(run.root, run.fresh["path"])
        shard = manifest.parent / "shard_000000.npz"
        shard.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            self.runner("a", resume=True)
        shard.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            ck.load(run.root)

    def test_failure_keeps_last_commit_and_records_extra_interactions(self):
        run = self.runner("a")
        original = run.marker
        from tasks.continual_nav.learning.world import SnapshotCollector
        real_collect = SnapshotCollector.collect
        def failing(collector, *args, **kwargs):
            real_collect(collector, *args, **kwargs)
            raise RuntimeError("injected failure after collection")
        with patch.object(SnapshotCollector, "collect", failing):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                run.run_next()
        self.assertEqual(ck.load(run.root)["current_stage"], "init")
        journal = json.loads(next((run.root / "attempts").glob("*.json")).read_text())
        self.assertEqual(journal["confirmed_env_steps"], 4)
        self.assertFalse(journal["committed"])
        resumed = self.runner("a", resume=True)
        resumed.run_next()
        self.assertEqual(resumed.counters["global_env_steps"], 4)
        self.assertNotEqual(resumed.marker, original)


if __name__ == "__main__":
    unittest.main()
