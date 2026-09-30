"""Full-size v3 policies: ownership, same-tick causality and whole-rollout replay."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from tasks.continual_nav_opd.config import load_config
from tasks.continual_nav_opd.contracts import CTMState, DualState
from tasks.continual_nav_opd.models import (
    StandalonePolicy, DualPolicy, frozen_copy, detach_clone_state,
    select_state, save_snapshot, load_snapshot,
)


def tensors(state):
    if isinstance(state, DualState):
        return (*tensors(state.kb), *tensors(state.active))
    return state.pre, state.post


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.config = load_config("tasks/continual_nav_opd/configs/smoke.yaml")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(37)
        self.kb = StandalonePolicy(self.config)
        self.rgb = torch.randint(0, 256, (2, 3, 84, 84), dtype=torch.uint8)

    def assertStateClose(self, first, second):
        for a, b in zip(tensors(first), tensors(second)):
            torch.testing.assert_close(a, b, atol=3e-5, rtol=1e-4)

    def assertStorageIndependent(self, first, second):
        for name, value in first.state_dict().items():
            self.assertNotEqual(value.data_ptr(), second.state_dict()[name].data_ptr(), name)

    def assertGrad(self, module):
        values = [p.grad for p in module.parameters() if p.grad is not None]
        self.assertTrue(values)
        self.assertTrue(all(torch.isfinite(g).all() for g in values))
        self.assertGreater(sum(float(g.abs().sum()) for g in values), 0)

    def test_shape_ticks_learned_state_and_parameter_namespace(self):
        self.assertEqual({name.split('.')[0] for name, _ in self.kb.named_parameters()},
                         {'encoder', 'controller', 'actor'})
        self.assertFalse(any('critic' in name for name, _ in self.kb.named_parameters()))
        norms = [m for m in self.kb.encoder.modules() if isinstance(m, nn.GroupNorm)]
        self.assertTrue(norms)
        self.assertTrue(all(m.num_groups == 32 for m in norms))
        self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.kb.modules()))
        initial = self.kb.initial_state(2)
        self.assertEqual(initial.pre.shape, (2,512,40))
        self.assertTrue(initial.pre.requires_grad and initial.post.requires_grad)
        self.assertNotEqual(initial.pre.data_ptr(), self.kb.controller.start_pre.data_ptr())
        with torch.no_grad():
            fmap = self.kb.encoder(self.rgb)
            self.assertEqual(fmap.shape, (2,128,21,21))
            with patch.object(self.kb.controller, 'tick', wraps=self.kb.controller.tick) as tick:
                logits, final = self.kb.step(self.rgb, initial, torch.ones(2, dtype=torch.bool))
                self.assertEqual(tick.call_count, 2)
            self.assertEqual(logits.shape, (2,5))
            torch.testing.assert_close(final.pre[:,:,:-2], initial.pre[:,:,2:])
            torch.testing.assert_close(final.post[:,:,:-2], initial.post[:,:,2:])

    def test_p_initialization_and_frozen_owner(self):
        dual = DualPolicy(self.config, self.kb, kb_ready=True)
        self.assertStorageIndependent(self.kb, dual.kb)
        self.assertStorageIndependent(dual.kb.encoder, dual.active.encoder)
        for key, value in dual.kb.encoder.state_dict().items():
            torch.testing.assert_close(value, dual.active.encoder.state_dict()[key], atol=0, rtol=0)
        self.assertFalse(torch.equal(dual.kb.controller.synapse[0].weight, dual.active.controller.synapse[0].weight))
        self.assertFalse(torch.equal(dual.kb.actor[0].weight, dual.active.actor[0].weight))
        self.assertTrue(all(p.requires_grad for p in self.kb.parameters()))
        dual.train()
        self.assertFalse(dual.kb.training)
        self.assertFalse(any(p.requires_grad for p in dual.kb.parameters()))
        self.assertTrue(dual.active.training and all(p.requires_grad for p in dual.active.parameters()))
        snapshot = frozen_copy(dual)
        snapshot.train()
        self.assertFalse(any(m.training for m in snapshot.modules()))
        self.assertFalse(any(p.requires_grad for p in snapshot.parameters()))

    def test_gradient_routes_zero_and_nonzero_gate(self):
        dual = DualPolicy(self.config, self.kb, kb_ready=True)
        before = {n: t.clone() for n, t in dual.kb.state_dict().items()}
        for gate in (0., .3):
            dual.zero_grad(set_to_none=True)
            with torch.no_grad():
                dual.adapter.gate.fill_(gate)
            output = dual.sequence(self.rgb[:1].unsqueeze(0), detach_clone_state(dual.initial_state(1)),
                                   torch.ones(1,1,dtype=torch.bool))
            F.cross_entropy(output.logits.flatten(0,1), torch.tensor([3])).backward()
            self.assertGrad(dual.active.encoder)
            self.assertGrad(dual.active.controller)
            self.assertGreater(float(dual.active.controller.out_sync.decay.grad.abs().sum()), 0)
            self.assertGreater(float(dual.active.controller.action_sync.decay.grad.abs().sum()), 0)
            self.assertGrad(dual.active.actor)
            self.assertGreater(float(dual.adapter.gate.grad.abs()), 0)
            self.assertFalse(any(p.grad is not None for p in dual.kb.parameters()))
            self.assertGreater(float(dual.active.controller.start_pre.grad.abs().sum()),0)
            self.assertGreater(float(dual.active.controller.start_post.grad.abs().sum()),0)
            if gate == 0:
                self.assertEqual(float(dual.adapter.projection.weight.grad.abs().sum()), 0)
            else:
                self.assertGrad(dual.adapter.projection)
                self.assertGrad(dual.adapter.norm)
        optimizer = torch.optim.Adam([p for p in dual.parameters() if p.requires_grad], lr=1e-4)
        optimizer.step()
        for name, value in dual.kb.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_current_tick_and_two_independent_visual_paths(self):
        dual = DualPolicy(self.config, self.kb, kb_ready=True)
        kb_outputs, adapter_inputs, active_inputs = [], [], []
        handles = [
            dual.adapter.register_forward_pre_hook(lambda module,args: adapter_inputs.append(args[0].clone())),
            dual.kb.encoder.register_forward_hook(lambda m,a,o: kb_outputs.append(o.detach().clone())),
            dual.active.encoder.register_forward_hook(lambda m,a,o: active_inputs.append(o.detach().clone())),
        ]
        new_posts, seen_features = [], []
        original = dual.kb.controller.tick
        def tick(fmap,state,lateral=None):
            seen_features.append(fmap)
            result = original(fmap,state,lateral)
            new_posts.append(result[1].clone())
            return result
        try:
            with torch.no_grad():
                # Make the visual paths distinguishable before inspecting their consumers.
                next(dual.active.encoder.parameters()).add_(.1)
                with patch.object(dual.kb.controller,'tick',side_effect=tick):
                    dual.step(self.rgb, dual.initial_state(2), torch.ones(2,dtype=torch.bool))
            self.assertEqual(len(new_posts),2)
            for received, produced in zip(adapter_inputs,new_posts):
                torch.testing.assert_close(received,produced,atol=0,rtol=0)
            for fmap in seen_features:
                torch.testing.assert_close(fmap,kb_outputs[0],atol=0,rtol=0)
            self.assertFalse(torch.equal(kb_outputs[0],active_inputs[0]))
        finally:
            for handle in handles:
                handle.remove()

    def test_unready_kb_cannot_influence_active(self):
        dual = DualPolicy(self.config, self.kb, kb_ready=False)
        with torch.no_grad(), patch.object(dual.adapter,'forward',side_effect=AssertionError('unready lateral')):
            initial = dual.initial_state(2)
            logits, state = dual.step(self.rgb,initial,torch.ones(2,dtype=torch.bool))
            own_logits, own_state = dual.active.step(self.rgb,initial.active,torch.ones(2,dtype=torch.bool))
            torch.testing.assert_close(logits,own_logits,atol=0,rtol=0)
            self.assertStateClose(state.active,own_state)

    def test_full_window_replay_and_cross_window_state(self):
        observations = torch.stack([self.rgb, self.rgb.flip(-1), self.rgb.flip(-2)])
        starts = torch.tensor([[True,True],[False,False],[True,False]])
        for policy in (self.kb,DualPolicy(self.config,self.kb,kb_ready=True)):
            origin = detach_clone_state(policy.initial_state(2))
            state = origin
            with torch.no_grad():
                collected=[]
                for rgb, reset in zip(observations,starts):
                    logits,state=policy.step(rgb,state,reset)
                    collected.append(logits)
                replay=policy.sequence(observations,origin,starts)
                torch.testing.assert_close(replay.logits,torch.stack(collected),atol=3e-5,rtol=1e-4)
                self.assertStateClose(replay.state,state)
                first=policy.sequence(observations[:2],origin,starts[:2])
                saved=detach_clone_state(first.state)
                second=policy.sequence(observations[2:],saved,starts[2:])
                self.assertStateClose(second.state,replay.state)
                torch.testing.assert_close(second.logits,replay.logits[2:],atol=3e-5,rtol=1e-4)
                # Slot 0 reset equals an independent fresh episode; slot 1 carries prior history.
                single_logits,single_state=policy.step(observations[2,:1],policy.initial_state(1),torch.ones(1,dtype=torch.bool))
                torch.testing.assert_close(second.logits[0,:1],single_logits,atol=3e-5,rtol=1e-4)
                self.assertStateClose(select_state(second.state,torch.tensor([0])),single_state)

    def test_padding_neither_encodes_nor_resets_and_microbatch_equivalence(self):
        observations = torch.stack([self.rgb,self.rgb.flip(-1)])
        starts = torch.ones(2,2,dtype=torch.bool)
        valid = torch.tensor([[True,False],[False,False]])
        for policy in (self.kb,DualPolicy(self.config,self.kb,kb_ready=True)):
            origin=detach_clone_state(policy.initial_state(2))
            with torch.no_grad():
                output=policy.sequence(observations,origin,starts,valid)
                expected_logits,expected_state=policy.step(observations[0,:1],select_state(origin,torch.tensor([0])),torch.ones(1,dtype=torch.bool))
                self.assertStateClose(select_state(output.state,torch.tensor([0])),expected_state)
                self.assertStateClose(select_state(output.state,torch.tensor([1])),select_state(origin,torch.tensor([1])))
                torch.testing.assert_close(output.logits[0,:1],expected_logits,atol=3e-5,rtol=1e-4)
                self.assertEqual(float(output.logits[~valid].abs().sum()),0.)
                with patch.object(policy,'_encode',side_effect=AssertionError('padding encoder')):
                    empty=policy.sequence(observations,origin,starts,torch.zeros_like(valid))
                self.assertStateClose(empty.state,origin)
        config=replace(self.config,optimization=replace(self.config.optimization,encoder_microbatch_images=1))
        micro=StandalonePolicy(config)
        micro.load_state_dict(self.kb.state_dict(),strict=True)
        with torch.no_grad():
            a=self.kb.sequence(observations, self.kb.initial_state(2), starts)
            b=micro.sequence(observations, micro.initial_state(2), starts)
            torch.testing.assert_close(a.logits,b.logits,atol=3e-5,rtol=1e-4)
            self.assertStateClose(a.state,b.state)

    def test_origin_detached_independent_and_standalone_visual_learning(self):
        _, live=self.kb.step(self.rgb,self.kb.initial_state(2),torch.ones(2,dtype=torch.bool))
        saved=detach_clone_state(live)
        for original,copied in zip(tensors(live),tensors(saved)):
            self.assertFalse(copied.requires_grad)
            self.assertIsNone(copied.grad_fn)
            self.assertNotEqual(original.data_ptr(),copied.data_ptr())
        old_saved=detach_clone_state(saved)
        self.kb.zero_grad(set_to_none=True)
        output=self.kb.sequence(self.rgb.unsqueeze(0),saved,torch.zeros(1,2,dtype=torch.bool))
        F.cross_entropy(output.logits.flatten(0,1),torch.tensor([0,1])).backward()
        self.assertGrad(self.kb.encoder)
        self.assertGrad(self.kb.controller)
        self.assertGrad(self.kb.actor)
        self.assertStateClose(saved,old_saved)
        self.assertTrue(all(x.grad is None for x in tensors(saved)))

    def test_complete_snapshot_restoration_and_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            for index,policy in enumerate((self.kb,DualPolicy(self.config,self.kb,kb_ready=True))):
                snapshot=frozen_copy(policy)
                self.assertStorageIndependent(policy,snapshot)
                path=Path(folder)/f'{index}.pt'
                save_snapshot(snapshot,path)
                rng=torch.get_rng_state().clone()
                restored=load_snapshot(path)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertStorageIndependent(snapshot,restored)
                if isinstance(restored,DualPolicy):
                    self.assertTrue(restored.kb_ready.item())
                    self.assertStorageIndependent(restored.kb.encoder,restored.active.encoder)
                restored.train()
                self.assertFalse(any(m.training for m in restored.modules()))
                self.assertFalse(any(p.requires_grad for p in restored.parameters()))
                with torch.no_grad():
                    first,_=snapshot.step(self.rgb,snapshot.initial_state(2),torch.ones(2,dtype=torch.bool))
                    actual,_=restored.step(self.rgb,restored.initial_state(2),torch.ones(2,dtype=torch.bool))
                    torch.testing.assert_close(actual,first,atol=0,rtol=0)
                    next(policy.parameters()).add_(1)
                    after,_=snapshot.step(self.rgb,snapshot.initial_state(2),torch.ones(2,dtype=torch.bool))
                    torch.testing.assert_close(after,first,atol=0,rtol=0)
                with self.assertRaises(FileExistsError):
                    save_snapshot(snapshot,path)
            bad=Path(folder)/'bad.pt'
            torch.save({'schema_version':2},bad)
            with self.assertRaises(ValueError):
                load_snapshot(bad)

    def test_invalid_shapes_masks_and_state_types(self):
        state=self.kb.initial_state(2)
        for rgb in (self.rgb.float(), self.rgb[:0], self.rgb[:,:,:83]):
            with self.assertRaises(ValueError):
                self.kb.step(rgb,state,torch.ones(2,dtype=torch.bool))
        with self.assertRaises(ValueError):
            self.kb.sequence(self.rgb.unsqueeze(0),state,torch.ones(1,2))
        with self.assertRaises(TypeError):
            self.kb.sequence(self.rgb.unsqueeze(0),DualState(state,state),torch.ones(1,2,dtype=torch.bool))


if __name__=='__main__':
    unittest.main()
