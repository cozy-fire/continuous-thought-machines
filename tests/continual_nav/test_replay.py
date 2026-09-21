"""Raw-pixel integrity, deterministic top-K replacement and sampling contracts."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from tasks.continual_nav.contracts import Transition
from tasks.continual_nav.data.replay import ReplayBank, ShardedWriter, TransitionStore, sample_world_batch


def record(i, task="maze_medium", version=0):
    return Transition(np.full((3, 84, 84), i % 256, np.uint8),
                      np.full((3, 84, 84), (i+1) % 256, np.uint8), i % 5,
                      False, False, True, i, 1, task, version, version, i)


def fresh_store(path, count=3, version=0):
    writer = ShardedWriter(path, task="maze_medium", encoder_version=version,
        world_version=version, source="W", source_stage="test", expected_count=count, shard_size=2)
    for i in range(count):
        writer.append(record(i, version=version))
    return TransitionStore(writer.finish())


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bank = ReplayBank(self.root / "bank", capacity=3, shard_size=2)

    def tearDown(self):
        self.tmp.cleanup()

    def pool(self, ids, scores, task="maze_medium", version=0):
        builder = self.bank.begin(task, encoder_version=version, world_version=version,
                                  source_stage=f"X{version}", expected_count=len(ids))
        try:
            for i, score in zip(ids, scores):
                builder.offer(record(i, task, version), score)
            return builder.finish()
        finally:
            builder.abort()

    def test_shards_preserve_bytes_order_duplicates_and_metadata(self):
        store = fresh_store(self.root / "fresh", 5)
        self.assertEqual([s["count"] for s in store.header["shards"]], [2, 2, 1])
        records = store.take([4, 0, 4, 2])
        self.assertEqual([r.transition_id for r in records], [4, 0, 4, 2])
        for r in records:
            np.testing.assert_array_equal(r.obs, record(r.transition_id).obs)
            np.testing.assert_array_equal(r.transition_next_obs, record(r.transition_id).transition_next_obs)
        self.assertTrue(np.isnan(store.error(0)))
        self.assertEqual(set(store._cache), {"obs", "next_obs", "metadata"})
        self.assertLessEqual(len(store._cache["obs"]), 2)

    def test_incomplete_manifest_and_owning_image_bytes(self):
        w = ShardedWriter(self.root / "fresh", task="maze_medium", encoder_version=0,
            world_version=0, source="W", source_stage="W", expected_count=2)
        r = record(1)
        w.append(r)
        r.obs.fill(99)
        with self.assertRaises(RuntimeError):
            w.finish()
        self.assertFalse((w.directory / "manifest.json").exists())
        with self.assertRaises(ValueError):
            w.append(record(1))
        w.append(record(2))
        self.assertTrue((TransitionStore(w.finish()).take([0])[0].obs == 1).all())
        with self.assertRaises(RuntimeError):
            w.append(record(3))

    def test_corrupt_shard_rejected(self):
        store = fresh_store(self.root / "fresh")
        shard = store.manifest.parent / store.header["shards"][0]["path"]
        with shard.open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaises(ValueError):
            TransitionStore(store.manifest)

    def test_topk_ties_and_whole_round_replacement_task_isolation(self):
        old = self.pool([8, 2, 6, 1, 9], [5, 5, 9, 5, 0])
        self.assertEqual([r.transition_id for r in old.take(range(3))], [6, 1, 2])
        other = self.pool([30], [100], "fourrooms")
        new = self.pool([12, 13], [0.1, 0.2], version=1)
        self.assertEqual([r.transition_id for r in new.take(range(2))], [13, 12])
        self.assertEqual(self.bank.latest("fourrooms").manifest, other.manifest)
        self.assertEqual(self.bank.latest("maze_medium").manifest, new.manifest)
        self.bank.prune("maze_medium", protected_manifests=(old.manifest,))
        self.assertTrue(old.manifest.exists())
        self.bank.prune("maze_medium")
        self.assertFalse(old.manifest.exists())
        self.assertTrue(new.manifest.exists())

    def test_invalid_or_incomplete_X_cannot_replace_pool(self):
        old = self.pool([1], [2])
        b = self.bank.begin("maze_medium", encoder_version=1, world_version=1,
                            source_stage="X1", expected_count=2)
        try:
            for r, score, source in [(record(2), 1, "X"), (record(2, version=1), 1, "W"),
                                      (record(2, version=1), float("nan"), "X")]:
                with self.assertRaises(ValueError):
                    b.offer(r, score, source=source)
            b.offer(record(2, version=1), 1)
            with self.assertRaises(ValueError):
                b.offer(record(2, version=1), 1)
            with self.assertRaises(RuntimeError):
                b.finish()
            self.assertEqual(self.bank.latest("maze_medium").manifest, old.manifest)
        finally:
            b.abort()
        self.assertFalse(b.directory.exists())

    def test_sampler_ratio_replacement_rng_and_cross_task(self):
        fresh = fresh_store(self.root / "fresh", 1)
        high = self.pool([50], [1])
        def sample(pool, seed=7):
            return sample_world_batch(fresh, pool, task="maze_medium", batch_size=8,
                                      high_fraction=0.5, rng=np.random.default_rng(seed))
        self.assertEqual(sample(None).transition_ids.tolist(), [0]*8)
        mixed = sample(high)
        self.assertEqual(mixed.transition_ids.tolist(), [0]*4+[50]*4)
        self.assertEqual(mixed.from_high_error.tolist(), [False]*4+[True]*4)
        self.assertEqual(mixed.obs.dtype, torch.uint8)
        torch.testing.assert_close(mixed.obs, sample(high).obs)
        full = sample_world_batch(fresh, high, task="maze_medium", batch_size=256,
                                  high_fraction=0.5, rng=np.random.default_rng(7))
        self.assertEqual(int(full.from_high_error.sum()), 128)
        self.assertEqual(full.transition_ids.tolist(), [0]*128+[50]*128)
        with self.assertRaises(ValueError):
            sample(self.pool([20], [1], "fourrooms"))


if __name__ == "__main__":
    unittest.main()
