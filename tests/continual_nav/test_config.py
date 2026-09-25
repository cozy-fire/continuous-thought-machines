"""Schema-v2 budgets and strict configuration parsing."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tasks.continual_nav.config import Config, budget_summary, config_hash, load_config, save_config, validate_config

CONFIGS = Path(__file__).resolve().parents[2] / "tasks/continual_nav/configs"


class ConfigTests(unittest.TestCase):
    def test_full_smoke_and_5090_budgets(self):
        full = load_config(CONFIGS / "full.yaml")
        smoke = load_config(CONFIGS / "smoke.yaml")
        gpu = load_config(CONFIGS / "rtx5090_32gb.yaml")
        self.assertEqual((full.schema_version, full.agnostic.visits, full.exploration.steps_per_round), (2, 4, 1000000))
        self.assertEqual(budget_summary(full)["ta_rounds"], 16)
        self.assertEqual(budget_summary(full)["shared_ta_steps"], 16*(1000000+150000+4096))
        self.assertEqual(budget_summary(full)["main_steps"], budget_summary(gpu)["main_steps"])
        self.assertEqual(budget_summary(smoke)["main_steps"], 600)
        self.assertEqual((full.ctm.memory_length, full.distill.burnin_env_obs), (40, 20))
        self.assertEqual((gpu.evaluation.backend, gpu.evaluation.num_envs), ("subprocess", 16))
        self.assertEqual((full.ppo.encoder_microbatch_images, gpu.ppo.encoder_microbatch_images), (32, 200))

    def test_visits_and_roundtrip(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            save_config(replace(Config(), agnostic=replace(Config().agnostic, visits=7)), path)
            c = load_config(path)
            self.assertEqual(c.agnostic.visits, 7)
            self.assertEqual(config_hash(c), config_hash(load_config(path)))

    def test_reject_v1_and_invalid_v2(self):
        c = Config()
        for changed in (replace(c, schema_version=1),
                        replace(c, ctm=replace(c.ctm, memory_length=20)),
                        replace(c, distill=replace(c.distill, burnin_env_obs=10)),
                        replace(c, exploration=replace(c.exploration, steps_per_round=200001)),
                        replace(c, ppo=replace(c.ppo, encoder_microbatch_images=0)),
                        replace(c, exploration=replace(c.exploration, similarity_threshold=1.0))):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_config(changed)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.yaml"
            path.write_text("extends: " + str(CONFIGS / "full.yaml") + "\nworld:\n  updates_per_round: 1\n")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
