"""Exact raw-pixel revisit reward and episode isolation tests."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from tasks.continual_nav.config import Config
from tasks.continual_nav.learning.revisit import VisualRevisit
from tasks.continual_nav.data.rollout import EpisodeDiagnostics
from tasks.continual_nav import checkpoint as ck


def frame(pixel: int) -> np.ndarray:
    image = np.zeros((1, 3, 84, 84), dtype=np.uint8)
    image.reshape(1, -1)[0, pixel] = 255
    return image


def identity(episode: int = 0, map_hash: str = "a") -> dict:
    return dict(task_key="maze_medium", episode_id=episode, map_sha256=map_hash)


class RevisitTests(unittest.TestCase):
    def setUp(self):
        self.reward = VisualRevisit(Config(), torch.device("cpu"))

    def test_first_unchanged_and_zero_norm(self):
        initial = frame(0)
        self.reward.reset(initial, [identity()])
        reward, similarity, matched, gap = self.reward.score(initial, [identity()], np.array([False]), initial)
        self.assertAlmostEqual(reward.item(), -1, places=6)
        self.assertEqual(gap.item(), 1)
        self.assertAlmostEqual(similarity.item(), 1, places=6)
        self.assertAlmostEqual(matched.item(), 1, places=6)
        self.reward.reset(np.zeros_like(initial), [identity()])
        reward, similarity, _, _ = self.reward.score(initial, [identity()], np.array([False]), initial)
        self.assertEqual((reward.item(), similarity.item()), (0, 0))

    def test_long_gap_and_strongest_weighted_match(self):
        initial = frame(0)
        self.reward.reset(initial, [identity()])
        for step in range(1, 20):
            reward, _, _, _ = self.reward.score(frame(step), [identity()], np.array([False]), initial)
            self.assertEqual(reward.item(), 0)
        reward, _, _, gap = self.reward.score(initial, [identity()], np.array([False]), initial)
        self.assertAlmostEqual(reward.item(), -0.2, places=6)
        self.assertEqual(gap.item(), 20)
        reward, _, _, gap = self.reward.score(initial, [identity()], np.array([False]), initial)
        self.assertAlmostEqual(reward.item(), -1, places=6)
        self.assertEqual(gap.item(), 1)

    def test_maximum_similarity_is_separate_from_weighted_match(self):
        initial = np.full((1, 3, 84, 84), 255, dtype=np.uint8)
        near = initial.copy(); near.reshape(1, -1)[0, 0] = 0
        self.reward.reset(initial, [identity()])
        for _ in range(19):
            self.reward.score(near, [identity()], np.array([False]), initial)
        reward, maximum, matched, gap = self.reward.score(initial, [identity()], np.array([False]), initial)
        self.assertAlmostEqual(maximum.item(), 1, places=6)
        self.assertLess(matched.item(), maximum.item())
        self.assertEqual(gap.item(), 1)
        self.assertLess(reward.item(), -0.2)

    def test_real_terminal_frame_then_reset_and_identity_guard(self):
        old, new = frame(0), frame(1)
        self.reward.reset(old, [identity()])
        info = {**identity(), "reset_info": identity(1, "b")}
        reward, _, _, _ = self.reward.score(old, [info], np.array([True]), new)
        self.assertAlmostEqual(reward.item(), -1, places=6)
        reward, _, _, _ = self.reward.score(new, [identity(1, "b")], np.array([False]), new)
        self.assertAlmostEqual(reward.item(), -1, places=6)
        with self.assertRaises(ValueError):
            self.reward.score(new, [identity(2, "c")], np.array([False]), new)

    def test_diagnostics_select_complete_episodes_without_duplicates(self):
        trace = EpisodeDiagnostics(frame(0), [identity()], 100, 4)
        for step in range(4):
            info = {**identity(step), "reset_info": identity(step+1)}
            trace.record(np.array([step % 5]), frame(step+1), np.array([-1. if step == 1 else 0.]),
                         np.array([1.]), np.array([1.]), np.array([1]), np.array([True]), np.array([False]),
                         [info], frame(step+2), 100+step)
        selected = trace.selected()
        self.assertEqual(len(selected), 4)
        self.assertEqual(sorted(row["last_transition_id"] for row in selected), [100, 101, 102, 103])
        self.assertEqual(next(row for row in selected if "most_penalized" in row["selection"])["last_transition_id"], 101)
        self.assertTrue(all(len(row["images"]) == len(row["actions"])+1 for row in selected))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.pt"
            ck.save_artifact(path, {"episodes": selected})
            self.assertEqual(len(ck.load_artifact(path)["episodes"]), 4)


if __name__ == "__main__":
    unittest.main()
