"""Protocol tests use real environments and the selected native teacher weights."""
import ast
import hashlib
import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from tasks.continual_nav_opd.config import load_config, REPO_ROOT
from tasks.continual_nav_opd.contracts import CTMState
from tasks.continual_nav_opd.data import MazeEntry
from tasks.continual_nav_opd.envs import MazeEnv, FourRoomsEnv, SyncVectorEnv
from tasks.continual_nav_opd.teachers import (
    MazeTeacher, FourRoomsTeacher, shortest_path, map_probabilities,
)


def maze_image(start=(1, 1), goal=(2, 2)):
    image = np.zeros((19, 19, 3), dtype=np.uint8)
    image[1:4, 1:4] = 255
    image[start], image[goal] = (255, 0, 0), (0, 255, 0)
    return image


class MazeTests(unittest.TestCase):
    def target(self, image):
        return MazeTeacher().predict(image.transpose(2, 0, 1)[None],
                                     [{"map_sha256": "test-map", "episode_id": 7}])

    def test_four_directions_and_fifo(self):
        for action, goal in enumerate(((1, 2), (3, 2), (2, 1), (2, 3))):
            result = self.target(maze_image((2, 2), goal))
            self.assertEqual(result.action.tolist(), [action])
            self.assertEqual(result.distance_to_goal.tolist(), [1])
            np.testing.assert_array_equal(result.probabilities, np.eye(5, dtype=np.float32)[[action]])
            self.assertTrue(result.valid_target.all())
        self.assertEqual(self.target(maze_image()).action[0], 1)  # down before right
        self.assertEqual(shortest_path(np.zeros((3, 3), dtype=bool), (1, 1), (1, 1)), [])

    def test_detour_dead_end_and_replanning(self):
        image = maze_image((1, 1), (3, 3))
        image[2, 1] = 0
        result = self.target(image)
        self.assertEqual(result.action[0], 3)
        self.assertEqual(result.distance_to_goal[0], 4)
        image[1, 1] = 255
        image[1, 2] = (255, 0, 0)
        self.assertEqual(self.target(image).distance_to_goal[0], 3)
        walls = np.ones((3, 3), dtype=bool)
        with self.assertRaises(ValueError):
            shortest_path(walls, (0, 0), (2, 2))

    def test_bad_colors_positions_and_unreachable(self):
        cases = []
        for color in ((0, 0, 255), (11, 12, 13), (255, 0, 0), (0, 255, 0)):
            image = maze_image()
            image[3, 3] = color
            cases.append(image)
        image = maze_image()
        image[1, 1] = 255
        cases.append(image)
        image = maze_image()
        image[2, 2] = 255
        cases.append(image)
        image = maze_image()
        image[1, 2] = image[2, 1] = 0
        cases.append(image)
        for image in cases:
            with self.assertRaisesRegex(ValueError, "map=test-map episode=7"):
                self.target(image)

    def test_purity_and_descriptor(self):
        image = maze_image()
        saved = image.copy()
        py, np_state, rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        first, second = self.target(image), self.target(image)
        np.testing.assert_array_equal(first.probabilities, second.probabilities)
        np.testing.assert_array_equal(image, saved)
        self.assertEqual(random.getstate(), py)
        self.assertEqual(np.random.get_state()[0], np_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        self.assertEqual(np.random.get_state()[2:], np_state[2:])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        teacher = MazeTeacher()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "teacher.json"
            teacher.save_descriptor(path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), teacher.source_snapshot_id)
            with self.assertRaises(FileExistsError):
                teacher.save_descriptor(path)

    def test_maze_terminal_autoreset_and_batch_validation(self):
        from tasks.continual_nav.envs.maze import MazeEnv as OldEnv
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "train/0/one.png"
            path.parent.mkdir(parents=True)
            image = maze_image((1, 1), (1, 2))
            image[3, 3] = (0, 0, 255)  # Answer pixels must disappear in both views.
            Image.fromarray(image).save(path)
            entry = MazeEntry("train/0/one.png", hashlib.sha256(path.read_bytes()).hexdigest())
            envs = [MazeEnv(root, (entry,), seed=i) for i in range(2)]
            vector = SyncVectorEnv(envs)
            old = OldEnv(root, (entry,), seed=0)
            try:
                obs, _ = vector.reset()
                np.testing.assert_array_equal(obs.student_rgb[0], old.reset()[0])
                self.assertEqual(obs.teacher_obs.shape, (2, 3, 19, 19))
                self.assertFalse(np.any(np.all(obs.teacher_obs[0].transpose(1, 2, 0) == (0, 0, 255), axis=-1)))
                with self.assertRaises(ValueError):
                    vector.step([4, 9])
                self.assertEqual(envs[0].step_count, 0)
                step = vector.step([3, 4])
                np.testing.assert_array_equal(step.transition_next.student_rgb[0], old.step(3)[0])
                self.assertEqual(step.terminated.tolist(), [True, False])
                self.assertEqual(step.next_episode_start.tolist(), [True, False])
                self.assertNotEqual(step.info[0]["episode_id"], step.info[0]["reset_info"]["episode_id"])
                with self.assertRaises(ValueError):
                    MazeTeacher().predict(step.transition_next.teacher_obs[:1], step.info[:1])
                self.assertTrue(MazeTeacher().predict(step.next_obs.teacher_obs, step.info).valid_target.all())
                self.assertEqual(envs[1].episode_id, 0)
            finally:
                vector.close()
                old.close()


class FourRoomsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        config = load_config("tasks/continual_nav_opd/configs/full.yaml")
        cls.teacher = FourRoomsTeacher(REPO_ROOT / config.teachers.fourrooms.checkpoint)

    def test_native_mapping_zero_mass(self):
        probs = torch.tensor([[0., 0., 0., .1, .2, .3, .4], [1., 0., 0., 0., 0., 0., 0.]])
        target = map_probabilities(probs)
        torch.testing.assert_close(target, torch.tensor([[0., 0., 0., .5, .5], [1., 0., 0., 0., 0.]]))
        with self.assertRaises(ValueError):
            map_probabilities(probs * 2)

    def test_same_state_rgb_symbols_dynamics_and_noops(self):
        from tasks.continual_nav.envs.fourrooms import FourRoomsEnv as OldEnv
        paired = FourRoomsEnv(split="validation", evaluation_seeds=(2000000,))
        old = OldEnv(split="validation", evaluation_seeds=(2000000,))
        try:
            obs, _ = paired.reset()
            np.testing.assert_array_equal(obs.student_rgb, old.reset()[0])
            np.testing.assert_array_equal(obs.teacher_obs, paired.native.unwrapped.gen_obs()["image"])
            for action in (0, 1, 2, 3, 4):
                new, reward, term, trunc, info = paired.step(action)
                previous = old.step(action)
                np.testing.assert_array_equal(new.student_rgb, previous[0])
                self.assertEqual((reward, term, trunc), previous[1:4])
                if action >= 3:
                    self.assertEqual((info["mapped_action"], info["native_action"]), (9, 6))
                    self.assertEqual(info["agent_pos_before"], info["agent_pos_after"])
                    self.assertEqual(info["agent_dir_before"], info["agent_dir_after"])
            positions = []
            for action in range(3, 7):
                paired.reset()
                before = paired.native.unwrapped.gen_obs()["image"].copy()
                paired.native.step(action)
                np.testing.assert_array_equal(before, paired.native.unwrapped.gen_obs()["image"])
                positions.append(tuple(paired.native.unwrapped.agent_pos))
                self.assertEqual(paired.native.unwrapped.step_count, 1)
            self.assertEqual(len(set(positions)), 1)
        finally:
            paired.close()
            old.close()

    def test_original_agent_parity_from_first_step_and_reset(self):
        # Execute the unchanged native Agent definition without importing its plotting/UMAP stack.
        from models.ctm_rl import ContinuousThoughtMachineRL
        path = REPO_ROOT / "tasks/rl/train.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                       and node.name in ("layer_init", "Agent")]
        namespace = {"torch": torch, "nn": torch.nn, "np": np,
                     "ContinuousThoughtMachineRL": ContinuousThoughtMachineRL,
                     "Categorical": torch.distributions.Categorical}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
        Agent = namespace["Agent"]
        teacher = self.teacher
        with torch.random.fork_rng(devices=[]):
            reference = Agent(7, teacher.args, torch.device("cpu"))
            reference.load_state_dict(teacher.model.state_dict(), strict=True)
        reference.eval().requires_grad_(False)
        envs = [FourRoomsEnv(split="validation", evaluation_seeds=(2000000+i,)) for i in range(2)]
        try:
            observations = np.stack([env.reset()[0].teacher_obs for env in envs])
            state, original = teacher.initial_state(2), reference.get_initial_state(2)
            before = teacher.parameter_hash()
            for step in range(5):
                starts = np.array([step == 0 or step == 3, step == 0], dtype=bool)
                output = teacher.predict(observations, state, starts)
                with torch.no_grad():
                    states = reference._get_ctm_hidden_states(original, torch.tensor(starts, dtype=torch.float32), 2)
                    synch, original = reference.recurrent_model(torch.as_tensor(observations), states)
                    probs = torch.softmax(reference.actor(synch), -1)
                torch.testing.assert_close(output.native_probabilities, probs, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(output.state.pre, original[0], atol=1e-5, rtol=1e-4)
                self.assertTrue(output.valid_target.all())  # No prefix masking, even a short episode.
                self.assertFalse(output.probabilities.requires_grad)
                state = output.state
                observations = np.stack([env.step(step % 5)[0].teacher_obs for env in envs])
            self.assertEqual(teacher.parameter_hash(), before)
            self.assertFalse(teacher.model.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.model.parameters()))
        finally:
            for env in envs:
                env.close()

    def test_state_and_observation_rejection(self):
        teacher = self.teacher
        obs = np.zeros((1, 7, 7, 3), dtype=np.uint8)
        for bad in (obs.astype(np.float32), obs.transpose(0, 3, 1, 2), obs[:0]):
            with self.assertRaises(ValueError):
                teacher.predict(bad, teacher.initial_state(1), np.array([True]))
        with self.assertRaises(ValueError):
            teacher.predict(obs, teacher.initial_state(2), np.array([True]))

    def test_actual_short_episode_and_slot_autoreset_labels(self):
        from minigrid.core.world_object import Goal
        from tasks.continual_nav.envs.common import pixels
        vector = SyncVectorEnv([FourRoomsEnv(seed=i) for i in range(2)])
        try:
            vector.reset()
            for env in vector.envs:
                native = env.native.unwrapped
                native.agent_pos, native.agent_dir = (2, 2), 0
                native.grid.set(3, 2, Goal())
            obs = np.stack([env._pair(pixels(env.native.unwrapped.get_frame(tile_size=8, agent_pov=True))).teacher_obs
                            for env in vector.envs])
            teacher = self.teacher
            target = teacher.predict(obs, teacher.initial_state(2), np.ones(2, dtype=bool))
            self.assertTrue(target.valid_target.all())
            step = vector.step([2, 4])
            self.assertEqual(step.terminated.tolist(), [True, False])
            self.assertEqual(step.info[0]['episode_step'], 1)
            after = teacher.predict(step.next_obs.teacher_obs, target.state, step.next_episode_start)
            fresh = teacher.predict(step.next_obs.teacher_obs[:1], teacher.initial_state(1), np.array([True]))
            torch.testing.assert_close(after.probabilities[:1], fresh.probabilities, atol=1e-5, rtol=1e-4)
            self.assertTrue(after.valid_target.all())
            self.assertFalse(np.array_equal(step.transition_next.teacher_obs[0], step.next_obs.teacher_obs[0]))
        finally:
            vector.close()

    def test_strict_checkpoint_and_loader_rng(self):
        teacher = self.teacher
        rng = torch.get_rng_state().clone()
        copy = FourRoomsTeacher(teacher.checkpoint)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(copy.parameter_hash(), teacher.parameter_hash())
        state = dict(teacher.model.state_dict())
        state.pop("actor.0.bias")
        with patch("torch.load", return_value={"args": teacher.args, "model_state_dict": state}):
            with self.assertRaises(RuntimeError):
                FourRoomsTeacher(teacher.checkpoint)


if __name__ == "__main__":
    unittest.main()
