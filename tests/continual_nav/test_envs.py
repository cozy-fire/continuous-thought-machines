import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from minigrid.core.world_object import Goal, Wall

from fixtures import map_file
from tasks.continual_nav.config import Config
from tasks.continual_nav.data.manifest import MazeEntry
from tasks.continual_nav.envs import MazeEnv, FourRoomsEnv, VectorEnvAdapter, build_env
from tasks.continual_nav.envs.common import pixels


class MazeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.entry = map_file(self.root)
        self.env = MazeEnv(self.root, (self.entry,))
        self.addCleanup(self.env.close)

    def test_pixels_actions_path_erasure_and_success(self):
        obs, info = self.env.reset(seed=5)
        self.assertEqual(obs.shape, (3, 84, 84))
        self.assertEqual(obs.dtype, np.uint8)
        self.assertTrue(obs.flags.c_contiguous)
        self.assertFalse(np.any(np.all(obs.transpose(1, 2, 0) == (0, 0, 255), axis=-1)))
        self.assertEqual(info["episode_step"], 0)
        np.testing.assert_array_equal(self.env.render(), obs.transpose(1, 2, 0))
        # Up, down and left hit walls; wait is also a counted no-op.
        for a in (0, 1, 2, 4):
            result = self.env.step(a)
            self.assertEqual(self.env.agent_pos, (1, 1))
            self.assertEqual(result[1:4], (0.0, False, False))
        moved, _, _, _, _ = self.env.step(3)
        self.assertEqual(self.env.agent_pos, (1, 2))
        expected = self.env.base_rgb.copy()
        expected[1, 2] = (255, 0, 0)
        np.testing.assert_array_equal(moved, pixels(expected))
        final, reward, term, trunc, info = self.env.step(3)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertAlmostEqual(reward, 1 - 0.9 * 6 / 300)
        self.assertTrue(info["success"])
        self.assertEqual(self.env.goal_pos, (1, 3))
        self.assertFalse(np.any(np.all(final.transpose(1, 2, 0) == (0, 255, 0), axis=-1)))
        with self.assertRaises(RuntimeError):
            self.env.step(4)

    def test_timeout_and_last_step_success(self):
        self.env.reset()
        self.env.step_count = 299
        self.assertEqual(self.env.step(4)[1:4], (0.0, False, True))
        self.env.reset()
        self.env.step(3)
        self.env.step_count = 299
        _, reward, term, trunc, _ = self.env.step(3)
        self.assertEqual((term, trunc), (True, False))
        self.assertAlmostEqual(reward, 0.1)

    def test_vertical_moves_and_out_of_bounds(self):
        p = self.root / self.entry.path
        rgb = np.zeros((19, 19, 3), dtype=np.uint8)
        rgb[0:3, 0] = 255
        rgb[0, 0] = (255, 0, 0)
        rgb[2, 0] = (0, 255, 0)
        Image.fromarray(rgb).save(p)
        entry = MazeEntry(self.entry.path, hashlib.sha256(p.read_bytes()).hexdigest())
        env = MazeEnv(self.root, (entry,))
        env.reset()
        for a in (0, 2):
            env.step(a)
            self.assertEqual(env.agent_pos, (0, 0))
        env.step(1)
        self.assertEqual(env.agent_pos, (1, 0))
        env.step(0)
        self.assertEqual(env.agent_pos, (0, 0))

    def test_invalid_map_and_actions_fail_without_advancing(self):
        self.env.reset()
        for action in (-1, 5, 9, 1.0, True):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.env.step(action)
        self.assertEqual(self.env.step_count, 0)
        p = self.root / self.entry.path
        rgb = np.array(Image.open(p))
        rgb[1, 2] = 0
        Image.fromarray(rgb).save(p)
        entry = MazeEntry(self.entry.path, hashlib.sha256(p.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, "unreachable"):
            MazeEnv(self.root, (entry,)).reset()

    def test_seeded_map_sampling_and_observation_ownership(self):
        entries = tuple(map_file(self.root, i) for i in range(5))
        a, b = MazeEnv(self.root, entries, seed=17), MazeEnv(self.root, entries, seed=17)
        for _ in range(8):
            oa, ia = a.reset()
            ob, ib = b.reset()
            self.assertEqual(ia["map_sha256"], ib["map_sha256"])
            np.testing.assert_array_equal(oa, ob)
        oa[:] = 0
        self.assertTrue(a.render().any())


class FourRoomsTests(unittest.TestCase):
    def setUp(self):
        self.env = FourRoomsEnv(seed=9)
        self.addCleanup(self.env.close)
        self.env.reset()

    def test_partial_rgb_and_wait_mapping(self):
        native = self.env.native.unwrapped
        pos, direction, count = tuple(native.agent_pos), native.agent_dir, native.step_count
        with patch.object(self.env.native, "step", wraps=self.env.native.step) as step:
            for action in (3, 4):
                obs, reward, term, trunc, info = self.env.step(action)
                self.assertEqual(info["mapped_action"], 9)
                self.assertEqual(info["policy_action"], action)
                self.assertEqual(step.call_args.args, (6,))
                self.assertEqual(tuple(native.agent_pos), pos)
                self.assertEqual(native.agent_dir, direction)
                self.assertEqual((reward, term, trunc), (0.0, False, False))
                self.assertEqual(obs.shape, (3, 84, 84))
                self.assertEqual(obs.dtype, np.uint8)
                raw = native.get_frame(tile_size=8, agent_pov=True)
                self.assertEqual(raw.shape, (56, 56, 3))
                np.testing.assert_array_equal(obs, pixels(raw))
                np.testing.assert_array_equal(self.env.render(), obs.transpose(1, 2, 0))
        self.assertEqual(native.step_count, count + 2)

    def test_native_turns_and_forward(self):
        native = self.env.native.unwrapped
        native.agent_pos, native.agent_dir = (2, 2), 0
        native.grid.set(3, 2, None)
        self.env.step(0)
        self.assertEqual(native.agent_dir, 3)
        self.env.step(1)
        self.assertEqual(native.agent_dir, 0)
        self.env.step(2)
        self.assertEqual(tuple(native.agent_pos), (3, 2))

    def test_success_at_timeout_and_regular_timeout(self):
        native = self.env.native.unwrapped
        native.agent_pos, native.agent_dir = (2, 2), 0
        native.grid.set(3, 2, Goal())
        native.step_count = 299
        _, reward, term, trunc, info = self.env.step(2)
        self.assertAlmostEqual(reward, 0.1)
        self.assertEqual((term, trunc, info["success"]), (True, False, True))
        self.env.reset()
        native.step_count = 299
        self.assertEqual(self.env.step(4)[1:4], (0.0, False, True))

    def test_observation_does_not_expose_remote_global_cells(self):
        native = self.env.native.unwrapped
        native.agent_pos, native.agent_dir = (15, 15), 0
        before = self.env.step(3)[0]
        native.grid.set(1, 1, Wall())
        after = self.env.step(3)[0]
        np.testing.assert_array_equal(before, after)

    def test_wall_occludes_cells_inside_the_view_rectangle(self):
        native = self.env.native.unwrapped
        native.agent_pos, native.agent_dir = (2, 2), 0
        native.grid.vert_wall(4, 0, 19)
        native.grid.set(5, 2, None)
        before = self.env.step(3)[0]
        # This cell is inside the 7x7 crop, but behind an opaque wall.
        native.grid.set(5, 2, Goal())
        after = self.env.step(3)[0]
        self.assertFalse(native.see_through_walls)
        np.testing.assert_array_equal(before, after)

    def test_seed_partitions_and_reproducibility(self):
        a = build_env("fourrooms", "train", 52, config=Config())
        b = build_env("fourrooms", "train", 52, config=Config())
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        for _ in range(3):
            oa, ia = a.reset()
            ob, ib = b.reset()
            self.assertEqual(ia["episode_seed"], ib["episode_seed"])
            self.assertTrue(0 <= ia["episode_seed"] < 1000000)
            np.testing.assert_array_equal(oa, ob)
        validation = build_env("fourrooms", "validation", 0, config=Config())
        self.addCleanup(validation.close)
        self.assertEqual(validation.reset()[1]["episode_seed"], 2000000)
        self.assertEqual(validation.reset()[1]["episode_seed"], 2000001)
        with self.assertRaises(ValueError):
            validation.reset(seed=0)
        self.assertEqual(validation.reset(seed=2000005)[1]["episode_seed"], 2000005)


class VectorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        entry = map_file(root)
        self.a, self.b = MazeEnv(root, (entry,)), MazeEnv(root, (entry,))
        self.vector = VectorEnvAdapter([self.a, self.b])
        self.addCleanup(self.vector.close)
        self.vector.reset([0, 1])

    def test_terminal_and_reset_frames_are_separate(self):
        self.vector.step([3, 4])
        result = self.vector.step([3, 4])
        self.assertEqual(result.next_obs.shape, (2, 3, 84, 84))
        self.assertEqual(result.reward_ext.dtype, np.float32)
        self.assertEqual(result.terminated.tolist(), [True, False])
        self.assertEqual(result.next_episode_start.tolist(), [True, False])
        self.assertFalse(np.array_equal(result.transition_next_obs[0], result.next_obs[0]))
        np.testing.assert_array_equal(result.transition_next_obs[1], result.next_obs[1])
        self.assertEqual(result.info[0]["episode_step"], 2)
        self.assertEqual(result.info[0]["reset_info"]["episode_step"], 0)
        self.assertEqual(self.b.step_count, 2)
        saved = result.transition_next_obs.copy()
        self.vector.step([3, 4])
        np.testing.assert_array_equal(saved, result.transition_next_obs)

    def test_timeout_final_frame_and_atomic_action_validation(self):
        self.vector.step([3, 4])
        self.a.step_count = 299
        result = self.vector.step([4, 4])
        self.assertEqual(result.truncated.tolist(), [True, False])
        self.assertFalse(np.array_equal(result.transition_next_obs[0], result.next_obs[0]))
        old = (self.a.step_count, self.b.step_count)
        with self.assertRaises(ValueError):
            self.vector.step([4, 9])
        self.assertEqual(old, (self.a.step_count, self.b.step_count))

    def test_duplicate_environment_slots_rejected(self):
        with self.assertRaises(ValueError):
            VectorEnvAdapter([self.a, self.a])
