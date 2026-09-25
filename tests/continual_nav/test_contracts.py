from dataclasses import fields
import unittest

import torch

from tasks.continual_nav.contracts import (CTMState, DualState, FisherSequenceBatch,
                                         PhaseKey, PolicyOutput)


class ContractTests(unittest.TestCase):
    def test_explicit_policy_state_tags(self):
        a = CTMState(torch.zeros(2, 512, 40), torch.zeros(2, 512, 40))
        b = CTMState(torch.ones(2, 512, 40), torch.ones(2, 512, 40))
        single = PolicyOutput(torch.zeros(2, 5), torch.zeros(2), a)
        dual = PolicyOutput(torch.zeros(2, 5), torch.zeros(2), DualState(a, b))
        self.assertEqual(single.state.kind, "single")
        self.assertEqual(dual.state.kind, "dual")
        self.assertNotEqual(dual.state.kb.post.data_ptr(), dual.state.active.post.data_ptr())

    def test_reward_free_records(self):
        for cls in (FisherSequenceBatch,):
            names = {f.name for f in fields(cls)}
            self.assertFalse(names & {"reward", "reward_ext", "value", "advantage", "teacher_log_probs"})
        self.assertIn("score_mask", {f.name for f in fields(FisherSequenceBatch)})

    def test_phase_identity(self):
        self.assertEqual(str(PhaseKey("init")), "init")
        self.assertEqual(str(PhaseKey("ta", "maze_medium", 0, 1, phase="X")),
                         "ta/v0/maze_medium/r1/X")
        self.assertEqual(str(PhaseKey("pnc", "fourrooms", 2, phase="C")), "pnc/v2/fourrooms/C")
        self.assertEqual(str(PhaseKey("single", "maze_medium", segment=2, phase="P")),
                         "single/maze_medium/s2/P")
        invalid = [dict(family="init", task="maze_medium"), dict(family="pnc", task="maze_medium", visit=0, phase="X"),
                   dict(family="ta", task="maze_medium", visit=0, phase="W", round=0),
                   dict(family="seq", task="fourrooms", visit=-1, phase="P"),
                   dict(family="seq", task="fourrooms", visit=0, phase="C")]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PhaseKey(**kwargs)
