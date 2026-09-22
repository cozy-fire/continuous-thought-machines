"""Fixed-panel metrics and RNG/model isolation; independent of full training runs."""
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tasks.continual_nav.analysis.metrics import episode_metrics, forgetting, seed_summary
from tasks.continual_nav.config import load_config
from tasks.continual_nav.evaluate import evaluate_policy, visual_drift
from tasks.continual_nav.models import StandalonePolicy, VisionEncoder
from tasks.continual_nav.verify_models import tensor_hash


class ShortPanelEnv:
    def reset(self):
        self.length = 0
        # Deliberately consume global RNGs to verify that evaluation restores them.
        random.random(); np.random.rand(); torch.rand(1)
        return np.zeros((3, 84, 84), np.uint8), {}
    def step(self, action):
        self.length += 1
        return np.full((3, 84, 84), self.length, np.uint8), 0., False, self.length == 2, dict(success=False)
    def close(self):
        pass


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads(); torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.threads)
        self.c = load_config(Path("tasks/continual_nav/configs/smoke.yaml"))

    def test_metrics_no_success_and_sample_std(self):
        metrics = episode_metrics([dict(success=False, length=300, **{"return": 0.})])
        self.assertIsNone(metrics["mean_length_success"])
        self.assertEqual(metrics["failures"], 1)
        self.assertEqual(forgetting([dict(a=.5), dict(a=.8), dict(a=.2)])[-1]["a"], .8-.2)
        summary = seed_summary({0: 1., 1: 2., 2: 3., 3: 4.})
        self.assertAlmostEqual(summary["sample_std"], (5/3)**.5)
        self.assertEqual(summary["mean"], 2.5)

    def test_evaluation_and_drift_preserve_parameters_modes_and_global_RNG(self):
        encoder, kb = VisionEncoder(self.c), StandalonePolicy(self.c)
        kb.train()
        original = tensor_hash(encoder), tensor_hash(kb)
        ts, ns, ps = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
        def panels(*args, **kwargs):
            return iter([ShortPanelEnv(), ShortPanelEnv()])
        with patch("tasks.continual_nav.evaluate.panel_envs", side_effect=panels), patch("tasks.continual_nav.evaluate.shortest_path", return_value=2):
            result = evaluate_policy(kb, encoder, None, self.c)
            drift = visual_drift(kb, encoder, encoder, None, self.c)
        self.assertEqual(result["transitions"], 8)
        self.assertTrue(all(r["mean_length_success"] is None for r in result["tasks"].values()))
        self.assertTrue(all(r["mean_kl"] == 0 for r in drift["tasks"].values()))
        self.assertEqual(original, (tensor_hash(encoder), tensor_hash(kb)))
        self.assertTrue(kb.training)
        torch.testing.assert_close(ts, torch.get_rng_state(), atol=0, rtol=0)
        self.assertEqual(ps, random.getstate())
        np.testing.assert_array_equal(ns[1], np.random.get_state()[1])


if __name__ == "__main__":
    unittest.main()
