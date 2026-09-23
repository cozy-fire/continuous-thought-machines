"""Fail-closed smoke scope and numerical/replay comparison contracts."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from tasks.continual_nav.config import Config, load_config
from tasks.continual_nav.verify_smoke import compare_tree, pool_fingerprint, require_smoke, state_hash
from tasks.continual_nav.checkpoint import reference
from test_replay import fresh_store


class SmokeTests(unittest.TestCase):
    def test_only_exact_smoke_with_device_override_is_allowed(self):
        config = load_config("tasks/continual_nav/configs/smoke.yaml")
        require_smoke(config)
        require_smoke(replace(config, training=replace(config.training, device="cuda:0")))
        for changed in (Config(), replace(config, exploration=replace(config.exploration, steps_per_round=20))):
            with self.assertRaises(ValueError):
                require_smoke(changed)

    def test_float_tolerance_does_not_relax_integer_RNG_or_structure(self):
        reference = dict(weights=torch.tensor([1.]), rng=torch.tensor([2], dtype=torch.uint8), count=40)
        close = dict(weights=torch.tensor([1.000001]), rng=reference["rng"].clone(), count=40)
        self.assertGreater(compare_tree(reference, close, atol=1e-5, rtol=1e-4), 0)
        for changed in ({**close, "count": 41}, {**close, "rng": torch.tensor([3], dtype=torch.uint8)},
                        {**close, "extra": 1}, {**close, "weights": torch.tensor([float("nan")])}):
            with self.assertRaises(AssertionError):
                compare_tree(reference, changed, atol=1e-5, rtol=1e-4)
        with self.assertRaises(AssertionError):
            compare_tree(reference, close, atol=0, rtol=0)

    def test_hash_excludes_freeze_flag_but_not_parameters_or_BN(self):
        module = torch.nn.BatchNorm1d(2)
        module.register_buffer("permanently_frozen", torch.tensor(False))
        initial = state_hash(module)
        module.permanently_frozen.fill_(True)
        self.assertEqual(state_hash(module), initial)
        module.running_mean.add_(1)
        self.assertNotEqual(state_hash(module), initial)

    def test_pool_fingerprint_uses_real_transition_frames_and_ignores_storage_UUID(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = fresh_store(root / "a"), fresh_store(root / "b")
            a = pool_fingerprint(root, reference(root, first.manifest))
            b = pool_fingerprint(root, reference(root, second.manifest))
            self.assertEqual(a, b)
            self.assertEqual(a["ids"], [0, 1, 2])
            changed = fresh_store(root / "c", version=1)
            self.assertNotEqual(a, pool_fingerprint(root, reference(root, changed.manifest)))


if __name__ == "__main__":
    unittest.main()
