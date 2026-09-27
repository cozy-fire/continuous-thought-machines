"""New X→C→F schedule and fail-closed schema-v2 checkpoints."""
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
import json
import unittest

from tasks.continual_nav import checkpoint as ck
from tasks.continual_nav.config import load_config, resolved_dict
from tasks.continual_nav.schedule import METHODS, RandomStreams, ends_visit, expand_stages, select_stages
from tasks.continual_nav.train import _validate_pnc_budget_resume


class ScheduleCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("tasks/continual_nav/configs/smoke.yaml")

    def test_ta_rounds_visits_and_baselines(self):
        stages = expand_stages(self.config, METHODS[0])
        self.assertEqual([s.key.phase for s in stages if s.key.family == "ta"], ["X", "C", "F"]*4)
        self.assertFalse(any("/W" in str(s.key) for s in stages))
        self.assertEqual([str(s.key) for i, s in enumerate(stages) if ends_visit(stages, i)],
                         ["ta/v0/fourrooms/r1/F", "pnc/v0/fourrooms/F"])
        self.assertEqual(sum(s.transitions for s in stages), 600)
        self.assertEqual(len(expand_stages(self.config, METHODS[1], "maze_medium")), 2)
        handoff_stages = select_stages(self.config, METHODS[0], phase_mode="pnc")
        self.assertEqual([s.key.family for s in handoff_stages], ["init"]+["pnc"]*6)
        self.assertEqual([str(s.key) for s in handoff_stages[1:]],
                         [f"pnc/v0/{task}/{phase}" for task in ("maze_medium", "fourrooms")
                          for phase in ("P", "C", "F")])
        ta_stages = select_stages(self.config, METHODS[0], phase_mode="ta")
        self.assertEqual([s.key.family for s in ta_stages], ["init"]+["ta"]*12)

    def test_rng_state_roundtrip(self):
        a = RandomStreams(0, METHODS[0], None)
        state = a.state_dict()
        first = a.numpy["env"].integers(1000)
        a.load_state_dict(state)
        self.assertEqual(first, a.numpy["env"].integers(1000))

    def test_v1_marker_rejected_before_payload(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "old.complete.json"
            marker.write_text(json.dumps({"schema_version": 1, "stage_key": "init", "checkpoint": {}}))
            with self.assertRaisesRegex(ValueError, "complete checkpoint"):
                ck.load(root, marker)

    def test_pnc_budget_resume_preserves_committed_prefix_and_changes_future(self):
        old = load_config("tasks/continual_nav/configs/full.yaml")
        new = replace(old, pnc=replace(old.pnc, visits=6, progress_steps=250000, compress_steps=100000))
        old_stages = select_stages(old, METHODS[0], phase_mode="pnc")
        new_stages = select_stages(new, METHODS[0], phase_mode="pnc")
        next_index = next(i + 1 for i, stage in enumerate(old_stages)
                          if str(stage.key) == "pnc/v0/fourrooms/F")
        payload = dict(config=resolved_dict(old), stages=[stage.record() for stage in old_stages],
                       next_index=next_index,
                       sources={"tasks/continual_nav/train.py": "old", "same.py": "same"}, phase_mode="pnc")

        resumed = _validate_pnc_budget_resume(payload, new, new_stages,
            {"tasks/continual_nav/train.py": "new", "same.py": "same"})

        self.assertEqual(len(resumed), 1 + 6 * len(new.task_order) * 3)
        self.assertEqual([stage.transitions for stage in resumed[:next_index]],
                         [stage.transitions for stage in old_stages[:next_index]])
        self.assertEqual(str(resumed[next_index].key), "pnc/v1/maze_medium/P")
        self.assertEqual(resumed[next_index].transitions, 250000)
        self.assertEqual(resumed[next_index + 1].transitions, 100000)

    def test_pnc_budget_resume_rejects_other_config_changes(self):
        old = load_config("tasks/continual_nav/configs/full.yaml")
        changed = replace(old, pnc=replace(old.pnc, visits=6, progress_steps=250000, compress_steps=100000),
                          training=replace(old.training, num_envs=old.training.num_envs * 2))
        stages = select_stages(changed, METHODS[0], phase_mode="pnc")
        payload = dict(config=resolved_dict(old), stages=[stage.record() for stage in
                       select_stages(old, METHODS[0], phase_mode="pnc")], next_index=7,
                       sources={"tasks/continual_nav/train.py": "old"}, phase_mode="pnc")
        with self.assertRaisesRegex(ValueError, "non-P&C configuration"):
            _validate_pnc_budget_resume(payload, changed, stages,
                                        {"tasks/continual_nav/train.py": "new"})


if __name__ == "__main__":
    unittest.main()
