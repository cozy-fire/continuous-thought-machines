"""Fixed-panel metrics and RNG/model isolation; independent of full training runs."""
from pathlib import Path
from dataclasses import replace
from functools import partial
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tasks.continual_nav.analysis.metrics import episode_metrics, forgetting, seed_summary
from tasks.continual_nav.config import load_config
from tasks.continual_nav.contracts import CTMState, DualState, PolicyOutput
from tasks.continual_nav.evaluate import evaluate_policy, visual_drift
from tasks.continual_nav.envs.evaluation import EvaluationPool
from tasks.continual_nav.models import StandalonePolicy, SingleActorCritic, DualPolicy, VisionEncoder
from tasks.continual_nav.verify_models import tensor_hash


class ShortPanelEnv:
    def __init__(self, index=0):
        self.limit = 1+index % 4
        self.success = index % 2 == 0
    def reset(self):
        self.length = 0
        # Deliberately consume global RNGs to verify that evaluation restores them.
        random.random(); np.random.rand(); torch.rand(1)
        return np.zeros((3, 84, 84), np.uint8), {}
    def step(self, action):
        self.length += 1
        done = self.length == self.limit
        return (np.full((3, 84, 84), self.length, np.uint8), float(action * 5**self.length),
                done and self.success, done and not self.success, dict(success=done and self.success))
    def close(self):
        pass


def short_factory(spec):
    if spec[3] == -1:
        raise RuntimeError("injected worker failure")
    return ShortPanelEnv(spec[3])


def short_specs(*args, **kwargs):
    return [("fourrooms", "validation", None, i) for i in range(5)]


class CountingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, obs):
        return (obs[:, :1, :1, :1]*255).round()


class CountingPolicy(torch.nn.Module):
    """Require both traces to match the environment's exact episode-local clock."""
    def initial_state(self, size):
        def column(value):
            x = torch.full((size, 1, 1), float(value))
            return CTMState(x.clone(), x.clone())
        return DualState(column(5), column(9))

    def step(self, features, state, starts):
        from tasks.continual_nav.models.ctm import reset_state
        state = reset_state(state, starts, self.initial_state(len(starts)))
        clock = features[:, 0, 0, 0]
        for column, initial in ((state.kb, 5), (state.active, 9)):
            torch.testing.assert_close(column.pre[:, 0, 0], clock+initial, atol=0, rtol=0)
            torch.testing.assert_close(column.post[:, 0, 0], clock+initial, atol=0, rtol=0)
            column.pre += 1
            column.post += 1
        return PolicyOutput(torch.zeros((len(starts), 5)), torch.zeros(len(starts)), state)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads(); torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.threads)
        base = load_config(Path("tasks/continual_nav/configs/smoke.yaml"))
        self.c = replace(base, evaluation=replace(base.evaluation, backend="serial"))

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
        with patch("tasks.continual_nav.evaluate.panel_envs", side_effect=panels), \
             patch("tasks.continual_nav.evaluate.panel_specs", side_effect=short_specs), \
             patch("tasks.continual_nav.evaluate.EvaluationPool", partial(EvaluationPool, factory=short_factory)), \
             patch("tasks.continual_nav.evaluate.shortest_path", return_value=2):
            result = evaluate_policy(kb, encoder, None, self.c)
            drift = visual_drift(kb, encoder, encoder, None, self.c)
        self.assertEqual(result["transitions"], 22)
        self.assertTrue(all(r["episodes"] == 5 for r in result["tasks"].values()))
        self.assertTrue(all(r["mean_kl"] == 0 for r in drift["tasks"].values()))
        self.assertEqual(original, (tensor_hash(encoder), tensor_hash(kb)))
        self.assertTrue(kb.training)
        torch.testing.assert_close(ts, torch.get_rng_state(), atol=0, rtol=0)
        self.assertEqual(ps, random.getstate())
        np.testing.assert_array_equal(ns[1], np.random.get_state()[1])

    def test_batch_and_subprocess_match_serial_for_all_policy_types(self):
        encoder = VisionEncoder(self.c)
        policies = [StandalonePolicy(self.c), SingleActorCritic(self.c),
                    DualPolicy(self.c, StandalonePolicy(self.c), kb_ready=True)]
        for policy in policies:
            with patch("tasks.continual_nav.evaluate.panel_specs", side_effect=short_specs), \
                 patch("tasks.continual_nav.evaluate.EvaluationPool", partial(EvaluationPool, factory=short_factory)):
                reports = []
                for backend, size in (("serial", 1), ("serial", 3), ("subprocess", 3)):
                    config = replace(self.c, evaluation=replace(self.c.evaluation, backend=backend, num_envs=size))
                    reports.append(evaluate_policy(policy, encoder, None, config, tasks=("fourrooms",)))
                self.assertEqual(reports[0]["tasks"], reports[1]["tasks"])
                self.assertEqual(reports[0]["tasks"], reports[2]["tasks"])
                self.assertEqual(reports[2]["transitions"], 11)
                raw = reports[2]["tasks"]["fourrooms"]["raw_episodes"]
                self.assertEqual([row["length"] for row in raw], [1, 2, 3, 4, 1])
                self.assertEqual([row["success"] for row in raw], [True, False, True, False, True])

    def test_worker_failure_closes_every_process(self):
        with EvaluationPool(2, "subprocess", factory=short_factory) as pool:
            processes = list(pool.processes)
            with self.assertRaisesRegex(RuntimeError, "injected worker failure"):
                pool.exchange("reset", {0: ("fourrooms", "validation", None, 0),
                                        1: ("fourrooms", "validation", None, -1)})
            self.assertTrue(all(not process.is_alive() for process in processes))

    def test_recycled_slots_and_tail_preserve_each_columns_episode_clock(self):
        config = replace(self.c, evaluation=replace(self.c.evaluation, num_envs=3))
        with patch("tasks.continual_nav.evaluate.panel_specs", side_effect=short_specs), \
             patch("tasks.continual_nav.evaluate.EvaluationPool", partial(EvaluationPool, factory=short_factory)):
            result = evaluate_policy(CountingPolicy(), CountingEncoder(), None, config, tasks=("fourrooms",))
        self.assertEqual(result["transitions"], 11)


if __name__ == "__main__":
    unittest.main()
