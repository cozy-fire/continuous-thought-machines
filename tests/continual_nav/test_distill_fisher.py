"""Sequence KL, detached burn-in and per-sample reward-free Fisher references."""
from dataclasses import replace
from pathlib import Path
import math
import unittest

import torch

from tasks.continual_nav.config import load_config
from tasks.continual_nav.data.rollout import SequenceCollector, collect_fisher
from tasks.continual_nav.learning.common import sequence_logits
from tasks.continual_nav.learning.distill import action_kl, make_distill_optimizer, run_compress_stage, train_distill
from tasks.continual_nav.learning.fisher import estimate_fisher, ewc_penalty, squared_score_sum, update_online_fisher
from tasks.continual_nav.models import DualPolicy, StandalonePolicy, VisionEncoder, detach_state
from tasks.continual_nav.verify_models import tensor_hash
from tests.continual_nav.test_ppo import PixelProbe


def setUpModule():
    global _threads
    _threads = torch.get_num_threads()
    torch.set_num_threads(2)


def tearDownModule():
    torch.set_num_threads(_threads)


class DistillFisherTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        self.c = load_config(Path("tasks/continual_nav/configs/smoke.yaml"))

    def test_KL_direction_and_teacher_detach(self):
        teacher = torch.tensor([[[.8, .2]]]).log().requires_grad_()
        student = torch.tensor([[[.3, .7]]]).log().requires_grad_()
        loss = action_kl(student, teacher, torch.ones(1, 1, dtype=torch.bool))
        self.assertAlmostEqual(loss.item(), .8*math.log(.8/.3)+.2*math.log(.2/.7), places=6)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.abs().sum(), 0)

    def test_burnin_twenty_ticks_no_gradient_padding_and_learning_gradient(self):
        kb = StandalonePolicy(self.c)
        features = torch.randn(13, 2, 128, 21, 21, requires_grad=True)
        valid = torch.ones(13, 2, dtype=torch.bool); valid[:3, 0] = False
        starts = torch.zeros_like(valid); starts[3, 0] = True
        burnin = torch.zeros_like(valid); burnin[:10] = True
        logits = sequence_logits(kb, features, starts, valid, burnin, 10)
        with torch.no_grad():
            manual = kb.initial_state(2)
            for t in range(10):
                output = kb.sequence(features[t:t+1], manual, starts[t:t+1], valid[t:t+1])
                manual = output.state
        expected = kb.sequence(features[10:], detach_state(manual), starts[10:], valid[10:]).logits
        torch.testing.assert_close(logits, expected)
        logits[-1, :, 0].sum().backward()
        self.assertEqual(features.grad[:10].abs().sum(), 0)
        self.assertGreater(features.grad[10:].abs().sum(), 0)
        self.assertEqual(kb.controller.ticks*10, 20)

    def test_streaming_teacher_isolation_history_tail_and_student_update(self):
        encoder, kb = VisionEncoder(self.c), StandalonePolicy(self.c)
        dual = DualPolicy(self.c, kb, kb_ready=True)
        collector = SequenceCollector(PixelProbe(forbid_reward=True), dual, encoder, self.c, mode="C")
        teacher_hash, visual_hash = tensor_hash(collector.policy), tensor_hash(encoder)
        optimizer = make_distill_optimizer(kb, encoder, self.c)
        first = collector.collect(20, action_rng=torch.Generator().manual_seed(1))
        self.assertEqual(int(first.loss_mask.sum()), 20)
        self.assertFalse(first.valid_mask[:10].any())
        before = tensor_hash(kb)
        teacher_state = collector.state.active.post.clone()
        train_distill(first, kb, encoder, None, optimizer, self.c, rng=torch.Generator().manual_seed(2))
        self.assertNotEqual(before, tensor_hash(kb))
        torch.testing.assert_close(teacher_state, collector.state.active.post, atol=0, rtol=0)
        second = collector.collect(2, action_rng=torch.Generator().manual_seed(3))
        torch.testing.assert_close(second.obs[:10], first.obs[10:])
        self.assertTrue(second.valid_mask[:11].all())
        self.assertFalse(second.valid_mask[11:].any())
        self.assertEqual(int(second.loss_mask.sum()), 2)
        train_distill(second, kb, encoder, None, optimizer, self.c, rng=torch.Generator().manual_seed(4))
        self.assertEqual(teacher_hash, tensor_hash(collector.policy))
        self.assertEqual(visual_hash, tensor_hash(encoder))
        self.assertTrue(all(p.grad is None for p in collector.policy.parameters()))

    def test_compression_exact_tail_budget(self):
        encoder, kb = VisionEncoder(self.c), StandalonePolicy(self.c)
        logged = []
        result = run_compress_stage(PixelProbe(forbid_reward=True), DualPolicy(self.c, kb), kb, encoder, self.c, None,
            steps=22, action_rng=torch.Generator().manual_seed(1), minibatch_rng=torch.Generator().manual_seed(2),
            start_transition_id=100, on_window=lambda consumed, updates, metrics:
                logged.append((consumed, updates, dict(metrics))))
        self.assertEqual((result["transitions"], result["updates"], result["next_transition_id"]), (22, 2, 122))
        self.assertEqual([(consumed, updates) for consumed, updates, _ in logged], [(20, 1), (22, 2)])
        self.assertEqual(logged[-1][2], result["last"])
        self.assertTrue(result["kb_ready"])

    def test_opposite_sample_gradients_do_not_cancel_Fisher(self):
        parameter = torch.nn.Parameter(torch.tensor(0.))
        unused = torch.nn.Parameter(torch.tensor(3.))
        log_probs = torch.stack([torch.nn.functional.logsigmoid(parameter), torch.nn.functional.logsigmoid(-parameter)])
        sums = squared_score_sum(log_probs, {"p": parameter, "unused": unused})
        self.assertAlmostEqual(float(sums["p"]/2), .25)
        self.assertEqual(float(sums["unused"]), 0)
        self.assertIsNone(parameter.grad)

    def test_Fisher_collection_unique_tail_and_no_reward_or_parameter_update(self):
        c = replace(self.c, fisher=replace(self.c.fisher, collect_steps=22))
        encoder, kb = VisionEncoder(c), StandalonePolicy(c)
        before = tensor_hash(encoder), tensor_hash(kb)
        batches = collect_fisher(PixelProbe(forbid_reward=True), kb, encoder, c,
            action_rng=torch.Generator().manual_seed(5), fisher_rng=torch.Generator().manual_seed(6), start_transition_id=70)
        ids = torch.cat([batch.transition_ids[batch.score_mask] for batch in batches])
        self.assertEqual(ids.numel(), 8)
        self.assertEqual(ids.unique().numel(), 8)
        self.assertTrue(((ids >= 70) & (ids < 92)).all())
        self.assertEqual(sum(int((b.valid_mask & ~b.burnin_mask).sum()) for b in batches), 22)
        self.assertTrue(all(not bool((b.score_mask & b.burnin_mask).any()) for b in batches))
        current = estimate_fisher(kb, encoder, batches, c)
        self.assertEqual(before, (tensor_hash(encoder), tensor_hash(kb)))
        self.assertEqual(set(current), set(dict(kb.named_parameters())))
        self.assertGreater(sum(float(f.sum()) for f in current.values()), 0)
        self.assertTrue(all(bool(torch.isfinite(f).all() and (f >= 0).all()) for f in current.values()))
        self.assertTrue(all(p.grad is None for p in kb.parameters()))
        repeated = collect_fisher(PixelProbe(reward=999), kb, encoder, c,
            action_rng=torch.Generator().manual_seed(5), fisher_rng=torch.Generator().manual_seed(6), start_transition_id=70)
        for b, r in zip(batches, repeated):
            torch.testing.assert_close(b.actions, r.actions)
            torch.testing.assert_close(b.score_mask, r.score_mask)
        duplicate = [batches[0], batches[0]]
        with self.assertRaises(ValueError):
            estimate_fisher(kb, encoder, duplicate, c)

    def test_online_accumulation_center_penalty_and_strict_names(self):
        kb = StandalonePolicy(self.c)
        current = {name: torch.ones_like(p)*.5 for name, p in kb.named_parameters()}
        old = update_online_fisher(kb, current, None, decay=.1, sample_count=8, stage_key="F0", encoder_version=0)
        self.assertEqual(float(ewc_penalty(kb, old, 250).detach()), 0)
        p = next(kb.parameters())
        with torch.no_grad():
            p.flatten()[0].add_(.1)
        self.assertAlmostEqual(float(ewc_penalty(kb, old, 250).detach()), 250/2*.5*.1**2, places=5)
        new = update_online_fisher(kb, current, old, decay=.3, sample_count=8, stage_key="F1", encoder_version=1)
        self.assertEqual(new.completed_compressions, 2)
        for name, value in new.importance.items():
            torch.testing.assert_close(value, torch.full_like(value, .65))
            torch.testing.assert_close(new.theta_star[name], dict(kb.named_parameters())[name])
        self.assertEqual(float(ewc_penalty(kb, new, 250).detach()), 0)
        new.importance.pop(next(iter(new.importance)))
        with self.assertRaises(ValueError):
            ewc_penalty(kb, new, 250)


if __name__ == "__main__":
    unittest.main()
