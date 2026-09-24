from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from tasks.continual_nav.config import (Config, budget_summary, config_hash,
                                       load_config, save_config, validate_config)

CONFIGS = Path(__file__).resolve().parents[2] / "tasks" / "continual_nav" / "configs"


class ConfigTests(unittest.TestCase):
    def test_full_and_smoke_budgets(self):
        self.assertEqual(Config().agnostic.visits, 4)
        full = load_config(CONFIGS / "full.yaml")
        smoke = load_config(CONFIGS / "smoke.yaml")
        self.assertEqual(full.agnostic.visits, 4)
        self.assertTrue(full.training.remote_logging)
        self.assertFalse(smoke.training.remote_logging)
        self.assertEqual((full.world.log_interval_updates, full.distill.log_interval_windows), (50, 10))
        self.assertEqual((smoke.world.log_interval_updates, smoke.distill.log_interval_windows), (1, 1))
        self.assertEqual(budget_summary(full)["main_steps"], 25250112)
        self.assertEqual(budget_summary(full)["all_methods_steps"], 76274688)
        self.assertEqual(budget_summary(full)["world_updates"], 1600000)
        self.assertEqual(budget_summary(smoke)["main_steps"], 760)
        self.assertEqual(budget_summary(smoke)["world_updates"], 8)
        rtx = load_config(CONFIGS / "rtx5090_32gb.yaml")
        self.assertEqual(budget_summary(rtx)["main_steps"], budget_summary(full)["main_steps"])
        self.assertEqual((rtx.agnostic, rtx.pnc, rtx.world.collect_steps_per_round,
                          rtx.exploration.steps_per_round, rtx.distill.agnostic_steps_per_round),
                         (full.agnostic, full.pnc, full.world.collect_steps_per_round,
                          full.exploration.steps_per_round, full.distill.agnostic_steps_per_round))
        self.assertEqual((rtx.world.batch_size, rtx.world.updates_per_round), (512, 20000))
        self.assertEqual(rtx.world.batch_size * rtx.world.updates_per_round, 10240000)
        self.assertEqual(budget_summary(rtx)["world_updates"], 320000)
        for config in (full, rtx):
            self.assertEqual(config.replay.fit_cache, "memory")
            self.assertEqual((config.evaluation.backend, config.evaluation.num_envs), ("subprocess", 16))
        self.assertEqual(smoke.evaluation.num_envs, 2)
        for name in ("vision", "ctm", "attention", "observation", "sigreg"):
            self.assertEqual(getattr(full, name), getattr(smoke, name))
        self.assertEqual(smoke.distill.burnin_env_obs, 10)

    def test_task_agnostic_visits_are_configurable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.yaml"
            path.write_text("extends: " + str(CONFIGS / "full.yaml") + "\nagnostic:\n  visits: 7\n")
            self.assertEqual(load_config(path).agnostic.visits, 7)

    def test_resolved_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolved.yaml"
            c = load_config(CONFIGS / "smoke.yaml")
            save_config(c, path)
            self.assertEqual(load_config(path), c)
            self.assertEqual(config_hash(load_config(path)), config_hash(c))

    def test_reject_bad_yaml_fields_types_and_inheritance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_config(Config(), root / "base.yaml")
            cases = ["training:\n  num_env: 2\n", "training:\n  num_envs: true\n",
                     "ppo:\n  gamma: .nan\n", "training:\n  seeds: [0, '1']\n",
                     "training: null\n", "ppo:\n  gamma: 0.9\n  gamma: 0.8\n",
                     "sigreg:\n  directions: 1.5\n"]
            for content in cases:
                with self.subTest(content=content):
                    (root / "bad.yaml").write_text("extends: base.yaml\n" + content)
                    with self.assertRaises(ValueError):
                        load_config(root / "bad.yaml")
            (root / "cycle.yaml").write_text("extends: cycle.yaml\n")
            with self.assertRaisesRegex(ValueError, "cyclic"):
                load_config(root / "cycle.yaml")
            (root / "empty.yaml").write_text("schema_version: 1\n")
            with self.assertRaisesRegex(ValueError, "missing"):
                load_config(root / "empty.yaml")

    def test_reject_incompatible_contract_and_budgets(self):
        c = Config()
        invalid = [replace(c, observation=replace(c.observation, actions=3)),
                   replace(c, ctm=replace(c.ctm, ticks=3)),
                   replace(c, attention=replace(c.attention, heads=3)),
                   replace(c, distill=replace(c.distill, burnin_env_obs=20)),
                   replace(c, training=replace(c.training, num_envs=3)),
                   replace(c, world=replace(c.world, collect_steps_per_round=100001)),
                   replace(c, world=replace(c.world, batch_size=3)),
                   replace(c, world=replace(c.world, log_interval_updates=0)),
                   replace(c, distill=replace(c.distill, log_interval_windows=0)),
                   replace(c, fisher=replace(c.fisher, scored_samples=4097)),
                   replace(c, replay=replace(c.replay, fit_cache="unknown")),
                   replace(c, evaluation=replace(c.evaluation, backend="unknown")),
                   replace(c, evaluation=replace(c.evaluation, num_envs=0)),
                   replace(c, environment=replace(c.environment, max_steps=301)),
                   replace(c, evaluation=replace(c.evaluation, drift_episodes=201))]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                validate_config(item)
