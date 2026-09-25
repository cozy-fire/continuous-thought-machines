"""Numerical, recurrent and gradient tests for the delivery-2 model contract."""
import io
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from tasks.continual_nav.config import Config
from tasks.continual_nav.contracts import CTMState, DualState
from tasks.continual_nav.models import (Controller, DualPolicy, SingleActorCritic, StandalonePolicy,
    VisionEncoder, detach_state, encode_obs, frozen_copy, reset_state)
from tasks.continual_nav.models.ctm import WindowSynchrony
from tasks.continual_nav.models.rope import SpatialAttention, SpatialRoPE, axial_angles, rotate_pairs


def setUpModule():
    global _threads
    _threads = torch.get_num_threads()
    torch.set_num_threads(2)


def tearDownModule():
    torch.set_num_threads(_threads)


def state_close(test, a, b):
    if isinstance(a, DualState):
        state_close(test, a.kb, b.kb)
        state_close(test, a.active, b.active)
    else:
        torch.testing.assert_close(a.pre, b.pre, atol=1e-5, rtol=0)
        torch.testing.assert_close(a.post, b.post, atol=1e-5, rtol=0)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.c = Config()

    def test_visual_shape_preprocessing_and_frozen_buffers(self):
        encoder = VisionEncoder(self.c)
        images = torch.randint(0, 256, (2, 3, 84, 84), dtype=torch.uint8)
        before = {n: x.clone() for n, x in encoder.state_dict().items()}
        seen = []
        hook = encoder.backbone.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
        encoder.train()  # A parent train() cannot silently enable frozen BN.
        fmap = encode_obs(images, encoder)
        hook.remove()
        self.assertEqual(fmap.shape, (2, 128, 21, 21))
        torch.testing.assert_close(seen[0], images.float()/255, atol=0, rtol=0)
        self.assertFalse(fmap.requires_grad)
        for n, tensor in encoder.state_dict().items():
            torch.testing.assert_close(tensor, before[n], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            encode_obs(images.float(), encoder)

    def test_visual_x_training_then_permanent_freeze(self):
        encoder = VisionEncoder(self.c)
        images = torch.randint(0, 256, (2, 3, 84, 84), dtype=torch.uint8)
        self.assertTrue(any(isinstance(m, nn.GroupNorm) for m in encoder.modules()))
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in encoder.modules()))
        encoder.set_x_training(True)
        encode_obs(images, encoder).square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters()))
        encoder.freeze(permanent=True)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in encoder.parameters()))
        with self.assertRaises(RuntimeError):
            encoder.set_x_training(True)
        saved = io.BytesIO()
        torch.save(encoder.state_dict(), saved)
        saved.seek(0)
        restored = VisionEncoder(self.c)
        restored.load_state_dict(torch.load(saved, weights_only=True))
        with self.assertRaises(RuntimeError):
            restored.set_x_training(True)

    def test_rope_coordinates_norm_and_zero_identity(self):
        rope = SpatialRoPE()
        self.assertEqual(rope.key_coords[22].tolist(), [1, 1])
        self.assertEqual(rope.query_coords.tolist(), [[10, 10]])
        x = torch.randn(2, 4, 441, 32)
        rotated = rotate_pairs(x, rope.key_cos, rope.key_sin)
        torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1), atol=1e-5, rtol=1e-6)
        angles = axial_angles(torch.zeros(441, 2), 32, 10000)
        torch.testing.assert_close(rotate_pairs(x, angles.cos(), angles.sin()), x, atol=0, rtol=0)
        expected = 2 * 10000 ** (-2*3/16)
        self.assertAlmostEqual(axial_angles(torch.tensor([[2., 7.]]), 32, 10000)[0, 3].item(), expected, places=6)
        self.assertFalse(any(p.requires_grad for p in rope.parameters()))

    def test_attention_matches_explicit_project_rotate_attend(self):
        attention = SpatialAttention(528)
        fmap, sync = torch.randn(2, 128, 21, 21), torch.randn(2, 528)
        out, weights = attention(fmap, sync, return_weights=True)
        tokens = attention.token_input(fmap.flatten(2).transpose(1, 2))
        query = attention.query_input(sync).unsqueeze(1)
        def split(x):
            return x.reshape(2, -1, 4, 32).transpose(1, 2)
        q, k, v = split(attention.q(query)), split(attention.k(tokens)), split(attention.v(tokens))
        q, k = attention.rope(q, k)
        expected_weights = torch.softmax(q @ k.transpose(-1, -2)/math.sqrt(32), -1)
        # V remains unrotated in the independent attention reconstruction.
        expected = attention.output((expected_weights @ v).transpose(1, 2).reshape(2, 128))
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(out, expected)
        self.assertEqual(weights.shape, (2, 4, 1, 441))

    def test_synchrony_matches_scalar_window_formula(self):
        sync = WindowSynchrony(3, 4, 2)
        post = torch.randn(2, 7, 4)
        with torch.no_grad():
            sync.decay.copy_(torch.tensor([-1., 0., 0.2, 1., 4., 5.]))
        expected = torch.zeros(2, 6)
        for b in range(2):
            k = 0
            for i in range(2, 5):
                for j in range(i, 5):
                    weights = [math.exp(-(3-u)*min(4, max(0, sync.decay[k].item()))) for u in range(4)]
                    expected[b, k] = sum(weights[u]*post[b, i, u]*post[b, j, u] for u in range(4))/math.sqrt(sum(weights))
                    k += 1
        torch.testing.assert_close(sync(post), expected, atol=1e-6, rtol=1e-6)
        controller = Controller(self.c)
        self.assertLess(controller.out_sync.right.max(), controller.action_sync.left.min())
        self.assertNotEqual(controller.out_sync.decay.data_ptr(), controller.action_sync.decay.data_ptr())

    def test_tick_window_shift_and_nlm_neuron_independence(self):
        controller = Controller(self.c)
        state = controller.initial_state(2)
        new, activation = controller.tick(torch.randn(2, 128, 21, 21), state)
        torch.testing.assert_close(new.pre[:, :, :-1], state.pre[:, :, 1:], atol=0, rtol=0)
        torch.testing.assert_close(new.post[:, :, :-1], state.post[:, :, 1:], atol=0, rtol=0)
        torch.testing.assert_close(activation, new.post[:, :, -1], atol=0, rtol=0)
        trace = torch.randn(2, 512, self.c.ctm.memory_length)
        original = controller.nlm(trace)
        changed = trace.clone()
        changed[:, 13] += 2
        difference = controller.nlm(changed) - original
        self.assertGreater(difference[:, 13].abs().sum(), 0)
        difference[:, 13] = 0
        self.assertEqual(difference.abs().sum(), 0)

    def test_two_ticks_and_reset_only_selected_slots(self):
        policy = SingleActorCritic(self.c)
        state = policy.initial_state(2)
        self.assertNotEqual(state.pre.data_ptr(), policy.controller.start_pre.data_ptr())
        perturbed = CTMState(state.pre+2, state.post-3)
        reset = reset_state(perturbed, torch.tensor([True, False]), state)
        torch.testing.assert_close(reset.pre[0], state.pre[0])
        torch.testing.assert_close(reset.post[1], perturbed.post[1])
        fmap = torch.randn(2, 128, 21, 21)
        with patch.object(policy.controller, "tick", wraps=policy.controller.tick) as tick:
            policy.step(fmap, state, torch.zeros(2, dtype=torch.bool))
            self.assertEqual(tick.call_count, 2)

    def test_step_sequence_and_batch_parity_for_all_policies(self):
        kb = StandalonePolicy(self.c)
        policies = [kb, SingleActorCritic(self.c), DualPolicy(self.c, StandalonePolicy(self.c), kb_ready=True)]
        fmaps = torch.randn(4, 2, 128, 21, 21)
        starts = torch.tensor([[True, True], [False, False], [True, False], [False, True]])
        for policy in policies:
            with self.subTest(policy=type(policy).__name__), torch.no_grad():
                sequence = policy.sequence(fmaps, policy.initial_state(2), starts)
                state = policy.initial_state(2)
                for t in range(4):
                    result = policy.step(fmaps[t], state, starts[t])
                    if isinstance(result, tuple):
                        logits, state = result
                        self.assertIsNone(sequence.value)
                    else:
                        logits, state = result.logits, result.state
                        torch.testing.assert_close(result.value, sequence.value[t], atol=1e-5, rtol=0)
                    torch.testing.assert_close(logits, sequence.logits[t], atol=1e-5, rtol=0)
                state_close(self, state, sequence.state)
                for b in range(2):
                    one = policy.sequence(fmaps[:, b:b+1], policy.initial_state(1), starts[:, b:b+1])
                    torch.testing.assert_close(one.logits[:, 0], sequence.logits[:, b], atol=1e-5, rtol=0)

    def test_padding_does_not_advance_reset_or_receive_gradient(self):
        policy = DualPolicy(self.c, StandalonePolicy(self.c), kb_ready=True)
        fmaps = torch.randn(4, 2, 128, 21, 21, requires_grad=True)
        valid = torch.tensor([[False, True], [True, True], [True, False], [False, True]])
        starts = torch.ones(4, 2, dtype=torch.bool)
        result = policy.sequence(fmaps, policy.initial_state(2), starts, valid)
        self.assertEqual(result.logits[~valid].abs().sum(), 0)
        for b in range(2):
            indices = valid[:, b].nonzero().squeeze(-1)
            expected = policy.sequence(fmaps[indices, b:b+1], policy.initial_state(1), starts[indices, b:b+1])
            torch.testing.assert_close(result.logits[indices, b], expected.logits[:, 0], atol=1e-5, rtol=0)
            for name in ("kb", "active"):
                torch.testing.assert_close(getattr(result.state, name).post[b:b+1],
                                           getattr(expected.state, name).post, atol=1e-5, rtol=0)
        (result.logits.square().sum()+result.value.square().sum()).backward()
        self.assertEqual(fmaps.grad[~valid].abs().sum(), 0)
        self.assertGreater(fmaps.grad[valid].abs().sum(), 0)

    def test_lateral_order_gate_and_readiness(self):
        dual = DualPolicy(self.c, StandalonePolicy(self.c), kb_ready=True)
        features = torch.randn(2, 128, 21, 21)
        starts = torch.zeros(2, dtype=torch.bool)
        state = dual.initial_state(2)
        with torch.no_grad():
            plain = dual.active.step(features, state.active, starts)
            gated = dual.step(features, state, starts)
            torch.testing.assert_close(plain.logits, gated.logits, atol=0, rtol=0)
            dual.adapter.gate.fill_(0.7)
        events = []
        kb_tick, active_tick = dual.kb.controller.tick, dual.active.controller.tick
        def tick_kb(*args, **kwargs):
            out = kb_tick(*args, **kwargs)
            events.append(("kb", out[1].clone()))
            return out
        def tick_active(fmap, state, lateral):
            self.assertEqual(events[-1][0], "kb")
            torch.testing.assert_close(lateral, dual.adapter(events[-1][1]), atol=0, rtol=0)
            events.append(("active", None))
            return active_tick(fmap, state, lateral)
        with patch.object(dual.kb.controller, "tick", tick_kb), patch.object(dual.active.controller, "tick", tick_active):
            connected = dual.step(features, state, starts)
        self.assertEqual([x[0] for x in events], ["kb", "active", "kb", "active"])
        self.assertGreater((plain.logits-connected.logits).abs().sum(), 0)
        dual.kb_ready.fill_(False)
        disabled = dual.step(features, state, starts)
        torch.testing.assert_close(disabled.logits, plain.logits, atol=0, rtol=0)

    def test_gradient_routes_and_frozen_parameter_storage(self):
        encoder = VisionEncoder(self.c)
        dual = DualPolicy(self.c, StandalonePolicy(self.c), kb_ready=True)
        dual.train()
        with torch.no_grad():
            dual.adapter.gate.fill_(0.4)
        frozen_before = {"encoder."+n: p.clone() for n, p in encoder.state_dict().items()}
        frozen_before.update({"kb."+n: p.clone() for n, p in dual.kb.state_dict().items()})
        images = torch.randint(0, 256, (4, 3, 84, 84), dtype=torch.uint8)
        fmaps = encode_obs(images, encoder).reshape(2, 2, 128, 21, 21)
        result = dual.sequence(fmaps, dual.initial_state(2), torch.zeros(2, 2, dtype=torch.bool))
        loss = -result.logits.log_softmax(-1)[..., 2].mean() + (result.value-1).square().mean()
        loss.backward()
        groups = ("controller.attention.query_input", "controller.attention.token_input",
                  "controller.attention.q", "controller.attention.k", "controller.attention.v",
                  "controller.attention.output", "controller.synapse", "controller.nlm",
                  "controller.out_sync.decay", "controller.action_sync.decay", "actor", "critic",
                  "controller.start_pre", "controller.start_post")
        for group in groups:
            grads = [p.grad for n, p in dual.active.named_parameters() if n.startswith(group)]
            self.assertTrue(grads and all(g is not None and torch.isfinite(g).all() for g in grads), group)
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0, group)
        self.assertGreater(dual.adapter.projection.weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in dual.kb.parameters()))
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        trainable = [p for p in dual.parameters() if p.requires_grad]
        torch.optim.Adam(trainable, lr=1e-4).step()
        for prefix, module in (("encoder.", encoder), ("kb.", dual.kb)):
            for name, value in module.state_dict().items():
                torch.testing.assert_close(value, frozen_before[prefix+name], atol=0, rtol=0)

    def test_initial_zero_gate_still_receives_gradient(self):
        dual = DualPolicy(self.c, StandalonePolicy(self.c), kb_ready=True)
        out = dual.step(torch.randn(2, 128, 21, 21), dual.initial_state(2), torch.zeros(2, dtype=torch.bool))
        out.logits[:, 0].sum().backward()
        self.assertGreater(dual.adapter.gate.grad.abs(), 0)
        self.assertEqual(dual.adapter.projection.weight.grad.abs().sum(), 0)

    def test_recurrent_gradient_detach_and_episode_boundary(self):
        policy = SingleActorCritic(self.c)
        features = torch.randn(2, 2, 128, 21, 21, requires_grad=True)
        first = policy.step(features[0], policy.initial_state(2), torch.zeros(2, dtype=torch.bool))
        second = policy.step(features[1], first.state, torch.tensor([True, False]))
        second.logits[:, 0].sum().backward()
        self.assertEqual(features.grad[0, 0].abs().sum(), 0)
        self.assertGreater(features.grad[0, 1].abs().sum(), 0)
        policy.zero_grad(set_to_none=True)
        features2 = torch.randn(2, 2, 128, 21, 21, requires_grad=True)
        first = policy.step(features2[0], policy.initial_state(2), torch.zeros(2, dtype=torch.bool))
        second = policy.step(features2[1], detach_state(first.state), torch.zeros(2, dtype=torch.bool))
        second.logits[:, 0].sum().backward()
        self.assertEqual(features2.grad[0].abs().sum(), 0)

    def test_frozen_snapshot_is_storage_isolated(self):
        live = DualPolicy(self.c, StandalonePolicy(self.c), kb_ready=True)
        with torch.no_grad():
            live.adapter.gate.fill_(0.4)
        teacher = frozen_copy(live)
        live_storage = {x.data_ptr() for x in live.state_dict().values()}
        self.assertFalse(live_storage & {x.data_ptr() for x in teacher.state_dict().values()})
        features = torch.randn(2, 128, 21, 21)
        starts = torch.zeros(2, dtype=torch.bool)
        before = teacher.step(features, teacher.initial_state(2), starts).logits.clone()
        with torch.no_grad():
            for parameter in live.parameters():
                parameter.add_(0.2)
        after = teacher.step(features, teacher.initial_state(2), starts).logits
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        self.assertFalse(after.requires_grad)
        encoder = VisionEncoder(self.c)
        encoder.set_x_training(True)
        snapshot = frozen_copy(encoder)
        obs = torch.randint(0, 256, (1, 3, 84, 84), dtype=torch.uint8)
        old = encode_obs(obs, snapshot)
        with torch.no_grad():
            next(encoder.parameters()).add_(1)
        torch.testing.assert_close(encode_obs(obs, snapshot), old, atol=0, rtol=0)

    def test_state_dict_roundtrip_and_random_active_initialization(self):
        dual = DualPolicy(self.c, StandalonePolicy(self.c), kb_ready=True)
        self.assertFalse(torch.equal(dual.kb.controller.start_pre, dual.active.controller.start_pre))
        self.assertNotEqual(dual.kb.actor[0].weight.data_ptr(), dual.active.actor[0].weight.data_ptr())
        self.assertFalse(any("critic" in n for n in dual.kb.state_dict()))
        restored = DualPolicy(self.c, StandalonePolicy(self.c))
        restored.load_state_dict(dual.state_dict())
        self.assertTrue(restored.kb_ready)
        features = torch.randn(2, 128, 21, 21)
        starts = torch.zeros(2, dtype=torch.bool)
        with torch.no_grad():
            out = dual.step(features, dual.initial_state(2), starts)
            other = restored.step(features, restored.initial_state(2), starts)
        torch.testing.assert_close(out.logits, other.logits, atol=0, rtol=0)
        state_close(self, out.state, other.state)
