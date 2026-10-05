"""Task budgets change recurrence, never weights, state shapes or model inputs."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tasks.continual_nav_opd.config import load_config, resolved_dict, parse_config, config_hash
from tasks.continual_nav_opd.models import StandalonePolicy, DualPolicy, save_snapshot, load_snapshot, detach_clone_state
from tasks.continual_nav_opd.learning.fisher import policy_hash, estimate_fisher
from tasks.continual_nav_opd.learning.sequence import update_window
from tests.continual_nav_opd import test_compress_fisher as fixtures
from tasks.continual_nav_opd.data.compress import CompressCollector


class TaskTickModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.config = load_config('tasks/continual_nav_opd/configs/smoke.yaml')

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(81)
        self.kb = StandalonePolicy(self.config)
        self.rgb = torch.randint(256, (1,3,84,84), dtype=torch.uint8)

    def test_shared_kb_counts_and_memory_shapes(self):
        pointers = {n:p.data_ptr() for n,p in self.kb.named_parameters()}
        before = policy_hash(self.kb)
        for task, count in (('maze_medium',5), ('fourrooms',2)):
            with torch.no_grad(), patch.object(self.kb.controller,'tick',wraps=self.kb.controller.tick) as calls:
                logits, state = self.kb.step(self.rgb,self.kb.initial_state(1),torch.ones(1,dtype=torch.bool),task=task)
            self.assertEqual(calls.call_count,count)
            self.assertEqual(state.pre.shape,(1,512,40))
            self.assertEqual(state.post.shape,(1,512,40))
            self.assertEqual(logits.shape,(1,5))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertEqual(pointers,{n:p.data_ptr() for n,p in self.kb.named_parameters()})
        self.assertEqual(before,policy_hash(self.kb))
        self.assertFalse(hasattr(self.kb.controller,'ticks'))

    def test_dual_same_tick_laterals_for_each_task(self):
        dual = DualPolicy(self.config,self.kb,kb_ready=True)
        for task, count in (('maze_medium',5), ('fourrooms',2)):
            produced, received = [], []
            original = dual.kb.controller.tick
            def record(*args,**kwargs):
                output = original(*args,**kwargs)
                produced.append(output[1].detach().clone())
                return output
            handle = dual.adapter.register_forward_pre_hook(lambda module,args: received.append(args[0].detach().clone()))
            try:
                with torch.no_grad(), patch.object(dual.kb.controller,'tick',side_effect=record) as kb_ticks, \
                     patch.object(dual.active.controller,'tick',wraps=dual.active.controller.tick) as active_ticks:
                    dual.step(self.rgb,dual.initial_state(1),torch.ones(1,dtype=torch.bool),task=task)
                self.assertEqual((kb_ticks.call_count,active_ticks.call_count,len(received)),(count,count,count))
                for a,b in zip(produced,received):
                    torch.testing.assert_close(a,b,atol=0,rtol=0)
            finally:
                handle.remove()

    def test_sampling_replay_reset_and_gradients_both_budgets(self):
        dual = DualPolicy(self.config,self.kb,kb_ready=True)
        with torch.no_grad(): dual.adapter.gate.fill_(.3)
        before = policy_hash(dual.kb)
        images = torch.stack([self.rgb,self.rgb.flip(-1),self.rgb.flip(-2)])
        starts = torch.tensor([[True],[False],[True]])
        for task, ticks in (('maze_medium',5), ('fourrooms',2)):
            origin = detach_clone_state(dual.initial_state(1))
            state = detach_clone_state(origin)
            with torch.no_grad():
                collected=[]
                for rgb,reset in zip(images,starts):
                    logits,state=dual.step(rgb,state,reset,task=task)
                    collected.append(logits)
                # Keep CNN batch shape equal to collection when checking long recurrences.
                replay=dual.sequence(images,origin,starts,task=task,encoder_chunk_images=1)
                torch.testing.assert_close(replay.logits,torch.stack(collected),atol=3e-5,rtol=1e-4)
                torch.testing.assert_close(replay.state.active.post,state.active.post,atol=3e-5,rtol=1e-4)
                fresh,_=dual.step(images[-1],dual.initial_state(1),torch.ones(1,dtype=torch.bool),task=task)
                torch.testing.assert_close(replay.logits[-1],fresh,atol=3e-5,rtol=1e-4)
            dual.zero_grad(set_to_none=True)
            with patch.object(dual.active.controller,'tick',wraps=dual.active.controller.tick) as calls:
                output=dual.sequence(images[:1],origin,starts[:1],task=task)
                output.logits.log_softmax(-1)[...,3].neg().mean().backward()
            self.assertEqual(calls.call_count,ticks)
            for module in (dual.active.encoder,dual.active.controller,dual.active.actor,dual.adapter):
                gradients=[p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0)
            self.assertFalse(any(p.grad is not None for p in dual.kb.parameters()))
        self.assertEqual(before,policy_hash(dual.kb))

    def test_snapshots_persist_both_budgets_and_reject_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            for index,policy in enumerate((self.kb,DualPolicy(self.config,self.kb,kb_ready=True))):
                path=Path(directory)/f'{index}.pt'
                save_snapshot(policy,path)
                restored=load_snapshot(path)
                self.assertEqual(restored.config.ctm.ticks_by_task,self.config.ctm.ticks_by_task)
                for task in self.config.task_order:
                    with torch.no_grad():
                        a,_=policy.step(self.rgb,policy.initial_state(1),torch.ones(1,dtype=torch.bool),task=task)
                        b,_=restored.step(self.rgb,restored.initial_state(1),torch.ones(1,dtype=torch.bool),task=task)
                    torch.testing.assert_close(a,b,atol=0,rtol=0)
                artifact=torch.load(path,weights_only=True)
                del artifact['config']['ctm']['ticks_by_task']
                artifact['config']['ctm']['ticks']=2
                torch.save(artifact,path)
                with self.assertRaisesRegex(ValueError,'missing or unknown fields'):
                    load_snapshot(path)

    def test_explicit_task_and_strict_config(self):
        with self.assertRaises(TypeError):
            self.kb.step(self.rgb,self.kb.initial_state(1),torch.ones(1,dtype=torch.bool))
        with patch.object(self.kb,'_encode',side_effect=AssertionError('invalid task encoded')):
            with self.assertRaisesRegex(ValueError,'unknown CTM execution task'):
                self.kb.step(self.rgb,self.kb.initial_state(1),torch.ones(1,dtype=torch.bool),task='unknown')
        raw=resolved_dict(self.config)
        del raw['ctm']['ticks_by_task']['fourrooms']
        with self.assertRaises(ValueError): parse_config(raw)
        changed=replace(self.config,ctm=replace(self.config.ctm,ticks_by_task=replace(self.config.ctm.ticks_by_task,fourrooms=3)))
        self.assertNotEqual(config_hash(changed),config_hash(self.config))
        other=StandalonePolicy(changed)
        self.assertEqual({n:p.shape for n,p in other.named_parameters()},{n:p.shape for n,p in self.kb.named_parameters()})


class TaskTickReplayTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.CompressFisherTests.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.CompressFisherTests.tearDownClass.__func__)
    setUp = fixtures.CompressFisherTests.setUp

    def test_wrong_collected_budget_rejected_before_update_or_fisher(self):
        batch=CompressCollector(self.envs,self.teacher,self.kb,self.rng,'fixture').collect(1,0).batch
        self.assertEqual((batch.task,batch.ticks),('maze_medium',5))
        optimizer=torch.optim.Adam(self.kb.parameters(),lr=1e-4)
        for bad in (replace(batch,ticks=2),replace(batch,task='fourrooms')):
            with patch.object(optimizer,'step',side_effect=AssertionError('wrong-budget update')):
                with self.assertRaisesRegex(ValueError,'execution budget mismatch'):
                    update_window(self.kb,bad,optimizer,self.config,np.random.default_rng(1))
            with self.assertRaisesRegex(ValueError,'execution budget mismatch'):
                estimate_fisher(self.kb,[bad],[0])


if __name__=='__main__': unittest.main()
