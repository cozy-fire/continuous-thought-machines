"""Independent numerical checks and world/snapshot phase isolation."""
from dataclasses import replace
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tasks.continual_nav.config import Config, load_config
from tasks.continual_nav.data.replay import ShardedWriter, TransitionStore
from tasks.continual_nav.learning.world import SnapshotCollector, fit_world_model, make_world_optimizer, prediction_losses
from tasks.continual_nav.models import SIGReg, StandalonePolicy, VisionEncoder, WorldModel, frozen_copy
from tasks.continual_nav.verify_models import tensor_hash
from tests.continual_nav.test_replay import fresh_store


def setUpModule():
    global _threads
    _threads = torch.get_num_threads()
    torch.set_num_threads(2)


def tearDownModule():
    torch.set_num_threads(_threads)


class WorldTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.c = load_config(Path("tasks/continual_nav/configs/smoke.yaml"))

    def test_sigreg_matches_independent_scalar_quadrature(self):
        cfg = replace(self.c.sigreg, directions=2, knots=5)
        reg = SIGReg(cfg)
        z = torch.tensor([[0.2, -0.3], [1.1, 0.5], [-0.8, 0.4]], requires_grad=True)
        a = torch.eye(2)
        expected = 0.0
        dt = cfg.t_max/(cfg.knots-1)
        for d in range(2):
            for k in range(cfg.knots):
                t = k*dt
                phi = math.exp(-t*t/2)
                real = sum(math.cos(float(row[d])*t) for row in z.detach())/3
                imag = sum(math.sin(float(row[d])*t) for row in z.detach())/3
                expected += 3/2*phi*dt*(1 if k in (0, cfg.knots-1) else 2)*((real-phi)**2+imag**2)
        loss = reg(z, a)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().sum().item(), 0)

    def test_sigreg_gaussian_vs_collapse_and_explicit_rng(self):
        reg = SIGReg(self.c.sigreg)
        rng = torch.Generator().manual_seed(5)
        state = rng.get_state()
        a = reg.sample_directions(128, rng, torch.device("cpu"))
        rng.set_state(state)
        torch.testing.assert_close(a, reg.sample_directions(128, rng, torch.device("cpu")), atol=0, rtol=0)
        torch.testing.assert_close(a.norm(dim=0), torch.ones(256))
        gaussian = reg(torch.randn(4096, 128), a).item()
        collapsed = reg(torch.zeros(4096, 128), a).item()
        self.assertLess(gaussian, collapsed/100)

    def test_joint_encoding_target_gradients_action_onehot_and_reward(self):
        world = WorldModel(self.c, VisionEncoder(self.c))
        obs = torch.randint(256, (2, 3, 84, 84), dtype=torch.uint8)
        nxt = torch.randint(256, obs.shape, dtype=torch.uint8)
        seen, inputs = [], []
        h1 = world.encoder.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape[0]))
        h2 = world.predictor.register_forward_pre_hook(lambda _, args: inputs.append(args[0].detach()))
        world.set_fit_mode()
        pred = world.transition(obs, nxt, torch.tensor([3, 4]))
        h1.remove(); h2.remove()
        self.assertEqual(seen, [4])
        torch.testing.assert_close(inputs[0][:, -5:], torch.eye(5)[[3, 4]])
        pred.z_next.square().mean().backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in world.encoder.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in world.predictor.parameters()))
        world.zero_grad(set_to_none=True)
        pred = world.transition(obs, nxt, torch.tensor([3, 4]))
        reg = SIGReg(self.c.sigreg)
        directions = reg.sample_directions(128, torch.Generator().manual_seed(6), torch.device("cpu"))
        with patch.object(reg, "forward", wraps=reg.forward) as spy:
            loss, mse, _ = prediction_losses(pred, reg, directions, self.c.sigreg.lambda_)
            self.assertIs(spy.call_args_list[0].args[1], spy.call_args_list[1].args[1])
        torch.testing.assert_close(mse, (pred.z_pred-pred.z_next).square().mean())
        loss.backward()
        for module in (world.encoder, world.projector, world.predictor):
            grads = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        snapshot = frozen_copy(world)
        self.assertFalse(snapshot._fit_mode)
        world.freeze()
        before = tensor_hash(world)
        world.train()  # Parent train() must not reactivate a frozen world or its BN.
        reward, error = world.curiosity(obs, nxt, torch.tensor([3, 4]))
        p = world.transition(obs, nxt, torch.tensor([3, 4]))
        torch.testing.assert_close(error, (p.z_pred-p.z_next).square().sum(-1).sqrt())
        torch.testing.assert_close(reward, error.log1p())
        self.assertFalse(reward.requires_grad)
        self.assertEqual(tensor_hash(world), before)
        self.assertTrue(all(p.grad is None for p in world.parameters()))
        with self.assertRaises(ValueError):
            world.transition(obs, nxt, torch.tensor([9, 4]))

    def test_fit_versions_optimizer_persistence_and_failure_freeze(self):
        c = replace(self.c, world=replace(self.c.world, batch_size=2))
        world = WorldModel(c, VisionEncoder(c))
        reg = SIGReg(c.sigreg)
        optimizer = make_world_optimizer(world, c)
        kb = StandalonePolicy(c)
        before = tensor_hash(kb)
        with TemporaryDirectory() as tmp:
            for version in range(2):
                fresh = fresh_store(Path(tmp)/str(version), version=version)
                hashes = [tensor_hash(m) for m in (world.encoder, world.projector, world.predictor)]
                logged = []
                result = fit_world_model(world, reg, fresh, None, optimizer, c, task="maze_medium",
                    replay_rng=np.random.default_rng(8), sigreg_rng=torch.Generator().manual_seed(9),
                    on_update=lambda step, metrics: logged.append((step, dict(metrics))))
                self.assertEqual(result["updates"], 2)
                self.assertIsNone(fresh._memory)
                self.assertGreater(result["replay_cache_bytes"], 0)
                self.assertEqual([step for step, _ in logged], [1, 2])
                self.assertEqual(logged[-1][1], result)
                self.assertEqual(int(world.world_model_version), version)
                self.assertTrue(all(tensor_hash(m) != h for m, h in zip((world.encoder, world.projector, world.predictor), hashes)))
                self.assertEqual(world.commit_fit(), (version+1, version+1))
                with self.assertRaises(RuntimeError):
                    world.commit_fit()
            self.assertTrue(all(int(state["step"]) == 4 for state in optimizer.state.values()))
            fresh = fresh_store(Path(tmp)/"failed", version=2)
            with patch("tasks.continual_nav.learning.world.prediction_losses", return_value=(torch.tensor(float("nan")), None, None)):
                with self.assertRaises(FloatingPointError):
                    fit_world_model(world, reg, fresh, None, optimizer, c, task="maze_medium",
                        replay_rng=np.random.default_rng(8), sigreg_rng=torch.Generator().manual_seed(9))
            self.assertEqual(int(world.world_model_version), 2)
            self.assertIsNone(fresh._memory)
            self.assertFalse(world.training)
            with self.assertRaises(RuntimeError):
                world.commit_fit()
        self.assertEqual(before, tensor_hash(kb))

    def test_collector_snapshot_true_terminal_frames_exact_budget_no_reward(self):
        class EnvProbe:
            num_envs = 2
            def reset(self):
                self.step_id = 0
                return np.zeros((2, 3, 84, 84), np.uint8), [{"task_key": "maze_medium"}]*2
            def step(self, actions):
                self.step_id += 1
                class Result(SimpleNamespace):
                    @property
                    def reward_ext(self):
                        raise AssertionError("W must never read external rewards")
                return Result(next_obs=np.zeros((2, 3, 84, 84), np.uint8),
                    transition_next_obs=np.full((2, 3, 84, 84), 255, np.uint8),
                    terminated=np.ones(2, bool), truncated=np.zeros(2, bool),
                    next_episode_start=np.ones(2, bool),
                    info=[dict(task_key="maze_medium", episode_id=self.step_id-1, episode_step=1)]*2)
        encoder, kb = VisionEncoder(self.c), StandalonePolicy(self.c)
        collector = SnapshotCollector(encoder, kb, self.c, world_version=0)
        snapshot_hashes = tensor_hash(collector.encoder), tensor_hash(collector.kb)
        with torch.no_grad():
            next(encoder.parameters()).add_(1)
            next(kb.parameters()).add_(1)
        self.assertEqual(snapshot_hashes, (tensor_hash(collector.encoder), tensor_hash(collector.kb)))
        with TemporaryDirectory() as tmp:
            sequences = []
            for run in range(2):
                writer = ShardedWriter(Path(tmp)/str(run), task="maze_medium", encoder_version=0,
                    world_version=0, source="W", source_stage="probe", expected_count=4,
                    snapshot_id=collector.snapshot_id)
                result = collector.collect(EnvProbe(), writer, steps=4, action_rng=torch.Generator().manual_seed(8), start_transition_id=20)
                self.assertEqual(result.next_transition_id, 24)
                records = TransitionStore(result.manifest).take(range(4))
                self.assertTrue(all((r.obs == 0).all() and (r.transition_next_obs == 255).all() for r in records))
                self.assertEqual([r.episode_id for r in records], [0, 1, 2, 3])
                sequences.append([r.action for r in records])
            self.assertEqual(sequences[0], sequences[1])
        self.assertEqual(snapshot_hashes, (tensor_hash(collector.encoder), tensor_hash(collector.kb)))


if __name__ == "__main__":
    unittest.main()
