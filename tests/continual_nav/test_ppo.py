"""PPO numerical reference, final-frame bootstrap and recurrent update isolation."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from tasks.continual_nav.config import load_config
from tasks.continual_nav.data.rollout import PPOCollector
from tasks.continual_nav.learning.common import encode_sequence
from tasks.continual_nav.learning.ppo import compute_gae, make_ppo_optimizer, make_x_optimizer, ppo_loss, run_ppo_stage, set_stage_learning_rate, train_ppo
from tasks.continual_nav.models import DualPolicy, SIGReg, SigregProjector, SingleActorCritic, StandalonePolicy, VisionEncoder, encode_obs
from tasks.continual_nav.verify_models import tensor_hash


def setUpModule():
    global _threads
    _threads = torch.get_num_threads()
    torch.set_num_threads(2)


def tearDownModule():
    torch.set_num_threads(_threads)


class PixelProbe:
    """Two slots, explicit final/reset frames, optional forbidden reward access."""
    num_envs = 2
    def __init__(self, reward=0.5, forbid_reward=False):
        self.reward, self.forbid_reward = reward, forbid_reward
    def reset(self):
        self.t = 0
        self.actions = []
        return np.zeros((2, 3, 84, 84), np.uint8), [dict(task_key="maze_medium", episode_id=0,
            map_sha256="probe") for _ in range(2)]
    def step(self, actions):
        self.t += 1
        self.actions.append(np.asarray(actions).copy())
        final = np.full((2, 3, 84, 84), 30+self.t, np.uint8)
        done = self.t % 2 == 0
        forbid, reward = self.forbid_reward, self.reward
        class Result(SimpleNamespace):
            @property
            def reward_ext(self):
                if forbid:
                    raise AssertionError("reward access forbidden")
                return np.full(2, reward, np.float32)
        infos = [dict(task_key="maze_medium", episode_id=(self.t-1)//2,
            episode_step=(self.t-1)%2+1, map_sha256="probe") for _ in range(2)]
        if done:
            for info in infos:
                info["reset_info"] = dict(task_key="maze_medium", episode_id=self.t//2,
                    map_sha256="probe")
        return Result(next_obs=np.zeros_like(final) if done else final.copy(), transition_next_obs=final,
                      terminated=np.array([done, False]), truncated=np.array([False, done]),
                      next_episode_start=np.array([done, done]),
                      info=infos)


class PPOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.c = load_config(Path("tasks/continual_nav/configs/smoke.yaml"))

    def test_gae_ordinary_success_timeout_and_padding(self):
        reward = torch.tensor([[1., 1., 1.], [2., 2., 2.], [99., 99., 99.]])
        value = torch.ones_like(reward)*0.5
        bootstrap = torch.ones_like(reward)*3
        term = torch.zeros_like(reward, dtype=torch.bool); term[0, 1] = True
        trunc = torch.zeros_like(term); trunc[0, 2] = True
        valid = torch.ones_like(term); valid[2] = False
        a, ret = compute_gae(reward, value, bootstrap, term, trunc, valid, 0.9, 0.8)
        delta1 = 2+0.9*3-0.5
        torch.testing.assert_close(a[0], torch.tensor([1+2.7-0.5+0.72*delta1, 0.5, 3.2]))
        torch.testing.assert_close(a[1], torch.full((3,), delta1))
        self.assertTrue((a[2] == 0).all())
        torch.testing.assert_close(ret[0], a[0]+0.5)

    def test_clipped_loss_population_normalization_and_target_detach(self):
        c = replace(self.c, ppo=replace(self.c.ppo, normalize_advantage=False))
        logits = torch.tensor([[[0.4, -0.4]], [[-0.5, 0.5]]], requires_grad=True)
        values = torch.tensor([[0.2], [-0.1]], requires_grad=True)
        actions = torch.zeros(2, 1, dtype=torch.long)
        old = torch.tensor([[-0.8], [-0.5]], requires_grad=True)
        adv = torch.tensor([[1.], [-2.]], requires_grad=True)
        returns = torch.tensor([[1.], [2.]], requires_grad=True)
        valid = torch.ones(2, 1, dtype=torch.bool)
        loss, _ = ppo_loss(logits, values, actions, old, adv, returns, valid, c)
        distribution = torch.distributions.Categorical(logits=logits)
        ratio = (distribution.log_prob(actions)-old.detach()).exp()
        expected = torch.maximum(-adv.detach()*ratio, -adv.detach()*ratio.clamp(.9, 1.1)).mean()
        expected += .25*.5*(values-returns.detach()).square().mean()-.01*distribution.entropy().mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(old.grad); self.assertIsNone(adv.grad); self.assertIsNone(returns.grad)
        _, metrics = ppo_loss(logits[:1], values[:1], actions[:1], old[:1], adv[:1], returns[:1], valid[:1], self.c)
        self.assertEqual(metrics["policy_loss"], 0)

    def test_real_policy_replay_ratio_final_bootstrap_and_frozen_groups(self):
        encoder = VisionEncoder(self.c)
        policy = DualPolicy(self.c, StandalonePolicy(self.c), kb_ready=True)
        with torch.no_grad():
            policy.adapter.gate.fill_(0.3)
        collector = PPOCollector(PixelProbe(), policy, encoder, self.c, phase="P")
        optimizer = make_ppo_optimizer(policy, encoder, self.c)
        before = tensor_hash(encoder), tensor_hash(policy.kb)
        active_before = tensor_hash(policy.active)
        for _ in range(2):
            batch = collector.collect(6, action_rng=torch.Generator().manual_seed(4))
            with torch.no_grad():
                output = policy.sequence(encode_sequence(batch.obs, encoder), batch.initial_state, batch.episode_start)
                logprob = output.logits.log_softmax(-1).gather(-1, batch.actions[..., None]).squeeze(-1)
                torch.testing.assert_close(logprob, batch.old_logprob, atol=1e-5, rtol=0)
                torch.testing.assert_close(output.value, batch.old_value, atol=1e-5, rtol=0)
                torch.testing.assert_close(output.state.active.post, collector.state.active.post, atol=1e-5, rtol=0)
                # Time index 1 is a true final frame, not the zero reset frame at index 2.
                if bool(batch.truncated[1, 1]):
                    state = policy.sequence(encode_sequence(batch.obs[:2], encoder), batch.initial_state, batch.episode_start[:2]).state
                    expected = policy.step(encode_obs(batch.transition_next_obs[1], encoder), state, torch.zeros(2, dtype=torch.bool)).value
                    torch.testing.assert_close(expected, batch.bootstrap_values[1])
            collect_state = collector.state.active.post.clone()
            metrics = train_ppo(batch, policy, encoder, optimizer, self.c, rng=torch.Generator().manual_seed(5))
            self.assertLess(metrics["max_logprob_difference"], 1e-5)
            torch.testing.assert_close(collect_state, collector.state.active.post, atol=0, rtol=0)
        self.assertEqual(before, (tensor_hash(encoder), tensor_hash(policy.kb)))
        self.assertNotEqual(active_before, tensor_hash(policy.active))
        self.assertGreater(policy.adapter.projection.weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.kb.parameters()))

    def test_X_uses_visual_reward_and_joint_gradients(self):
        encoder = VisionEncoder(self.c)
        projector = SigregProjector(self.c)
        policy = DualPolicy(self.c, StandalonePolicy(self.c))
        before = tuple(tensor_hash(module) for module in (encoder, projector, policy.active, policy.kb))
        collector = PPOCollector(PixelProbe(forbid_reward=True), policy, encoder, self.c, phase="X")
        batch = collector.collect(4, action_rng=torch.Generator().manual_seed(7))
        self.assertTrue((batch.reward <= 0).all())
        optimizer = make_x_optimizer(policy, encoder, projector, self.c, None)
        train_ppo(batch, policy, encoder, optimizer, self.c, rng=torch.Generator().manual_seed(8),
                  phase="X", projector=projector, regularizer=SIGReg(self.c.sigreg),
                  sigreg_rng=torch.Generator().manual_seed(9))
        after = tuple(tensor_hash(module) for module in (encoder, projector, policy.active, policy.kb))
        self.assertTrue(all(a != b for a, b in zip(before[:3], after[:3])))
        self.assertEqual(before[3], after[3])

    def test_stage_exact_tail_and_learning_rate_restart(self):
        encoder, policy = VisionEncoder(self.c), SingleActorCritic(self.c)
        result = run_ppo_stage(PixelProbe(), policy, encoder, self.c, phase="P", steps=22,
            action_rng=torch.Generator().manual_seed(1), minibatch_rng=torch.Generator().manual_seed(2))
        self.assertEqual((result["transitions"], result["rollouts"], result["updates"]), (22, 2, 2))
        self.assertEqual(result["final_lr"], 0)
        opt = make_ppo_optimizer(policy, encoder, self.c)
        self.assertEqual(set_stage_learning_rate(opt, self.c, completed_rollouts=0, total_rollouts=2), 1e-4)
        self.assertEqual(len(opt.state), 0)
        with self.assertRaises(ValueError):
            train_ppo(PPOCollector(PixelProbe(), policy, encoder, self.c, phase="P").collect(2, action_rng=torch.Generator()),
                      policy, encoder, torch.optim.Adam(encoder.parameters()), self.c, rng=torch.Generator())


if __name__ == "__main__":
    unittest.main()
