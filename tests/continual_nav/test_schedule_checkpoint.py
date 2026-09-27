"""New X→C→F schedule and fail-closed schema-v2 checkpoints."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from tasks.continual_nav import checkpoint as ck
from tasks.continual_nav.config import load_config
from tasks.continual_nav.schedule import METHODS, RandomStreams, ends_visit, expand_stages, select_stages


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


if __name__ == "__main__":
    unittest.main()
