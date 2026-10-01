"""C state isolation and per-sample full-visual Online Fisher."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from tasks.continual_nav_opd.config import load_config
from tasks.continual_nav_opd.data import MazeEntry
from tasks.continual_nav_opd.envs import MazeEnv,SyncVectorEnv
from tasks.continual_nav_opd.models import StandalonePolicy,DualPolicy,frozen_copy,detach_clone_state
from tasks.continual_nav_opd.data.compress import CompressCollector
from tasks.continual_nav_opd.learning.distill import run_compress_stage
from tasks.continual_nav_opd.learning.fisher import estimate_fisher,update_online_fisher,ewc_loss,policy_hash,run_fisher_stage
from tasks.continual_nav_opd.learning.sequence import update_window


class CompressFisherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads(); torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)
    def setUp(self):
        torch.manual_seed(9)
        self.config=load_config('tasks/continual_nav_opd/configs/smoke.yaml')
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name); path=root/'train/0/x.png'; path.parent.mkdir(parents=True)
        rgb=np.zeros((19,19,3),dtype=np.uint8); rgb[1,1:18]=255; rgb[1,1]=(255,0,0); rgb[1,17]=(0,255,0)
        Image.fromarray(rgb).save(path)
        entry=MazeEntry('train/0/x.png',hashlib.sha256(path.read_bytes()).hexdigest())
        self.envs=SyncVectorEnv([MazeEnv(root,(entry,),seed=i) for i in range(2)]); self.addCleanup(self.envs.close)
        self.kb=StandalonePolicy(self.config); self.teacher=frozen_copy(DualPolicy(self.config,self.kb,kb_ready=True))
        self.rng=torch.Generator().manual_seed(33)

    def test_independent_origins_full_replay_and_continuity(self):
        collector=CompressCollector(self.envs,self.teacher,self.kb,self.rng,policy_hash(self.teacher))
        window=collector.collect(2,10)
        with torch.no_grad(): out=self.kb.sequence(window.batch.obs,window.batch.initial_state,window.batch.episode_start)
        torch.testing.assert_close(out.logits,window.student_logits,atol=2e-5,rtol=1e-5)
        self.assertNotEqual(collector.state.pre.data_ptr(),collector.teacher_state.kb.pre.data_ptr())
        previous=detach_clone_state(collector.state); second=collector.collect(1,14)
        torch.testing.assert_close(second.batch.initial_state.pre,previous.pre,atol=0,rtol=0)
        self.assertFalse(second.batch.episode_start.any())
        live=detach_clone_state(collector.state)
        opt=torch.optim.Adam(self.kb.parameters(),lr=1e-4)
        update_window(self.kb,second.batch,opt,self.config,np.random.default_rng(4))
        collector.detach_live_state(); torch.testing.assert_close(live.pre,collector.state.pre,atol=0,rtol=0)

    def test_c_changes_visual_and_controller_but_not_teacher(self):
        before=policy_hash(self.teacher); visual=next(self.kb.encoder.parameters()).detach().clone()
        control=next(self.kb.actor.parameters()).detach().clone()
        events=[]
        result=run_compress_stage(self.envs,self.teacher,self.kb,None,self.config,4,self.rng,np.random.default_rng(2),on_window=events.append)
        self.assertEqual(len(events),1)
        self.assertAlmostEqual(sum(events[0]['action_fractions']),1.)
        self.assertIn('episode_mean_return',events[0])
        self.assertIn('episode_success_rate',events[0])
        self.assertEqual((result.transitions,result.optimizer_updates,result.next_transition_id),(4,1,4))
        self.assertTrue(result.kb_ready)
        self.assertFalse(torch.equal(visual,next(self.kb.encoder.parameters())))
        self.assertFalse(torch.equal(control,next(self.kb.actor.parameters())))
        self.assertEqual(before,policy_hash(self.teacher)); self.assertEqual(result.statistics['ewc'],0)
        self.assertGreater(result.statistics['encoder_grad_norm'],0)

    def test_fisher_is_average_of_individual_squares_not_batch_square(self):
        window=CompressCollector(self.envs,self.teacher,self.kb,self.rng,'fixture').collect(1,0).batch
        # Force different score actions so cancellation makes batch-gradient-square incorrect.
        window.actions[0]=torch.tensor([0,1])
        before=policy_hash(self.kb)
        result=estimate_fisher(self.kb,[window],[0,1])
        out=self.kb.sequence(window.obs,window.initial_state,window.episode_start)
        param=self.kb.actor[-1].bias
        score=out.logits.log_softmax(-1)[0]
        a=torch.autograd.grad(score[0,0],param,retain_graph=True)[0]
        b=torch.autograd.grad(score[1,1],param)[0]
        torch.testing.assert_close(result['actor.4.bias'],(a.square()+b.square())/2,atol=1e-6,rtol=1e-5)
        self.assertGreater(float((result['actor.4.bias']-((a+b)/2).square()).abs().max()),.01)
        self.assertEqual(before,policy_hash(self.kb))
        self.assertEqual(set(result),set(dict(self.kb.named_parameters())))
        self.assertGreater(sum(float(v.sum()) for k,v in result.items() if k.startswith('encoder.')),0)
        with self.assertRaises(ValueError): estimate_fisher(self.kb,[window],[0,0])
        with self.assertRaises(ValueError): estimate_fisher(self.kb,[window],[-1])

    def test_online_decay_center_and_ewc_validation(self):
        current={n:torch.full_like(p,.1) for n,p in self.kb.named_parameters()}
        first=update_online_fisher(self.kb,current,None,2,'first')
        loss,parts=ewc_loss(self.kb,first,250); self.assertEqual(float(loss.detach()),0.)
        with torch.no_grad(): next(self.kb.encoder.parameters()).add_(.001)
        loss,parts=ewc_loss(self.kb,first,250)
        self.assertGreater(float(parts['ewc_encoder'].detach()),0)
        second=update_online_fisher(self.kb,current,first,2,'second')
        for n in current:
            torch.testing.assert_close(second.importance[n],current[n]*1.3)
            torch.testing.assert_close(second.theta_star[n],dict(self.kb.named_parameters())[n])
        self.assertEqual(second.completed_compressions,2)
        bad=current.copy(); bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError): update_online_fisher(self.kb,bad,first,2,'bad')
        bad=current.copy(); bad[next(iter(bad))]=torch.tensor([-1.])
        with self.assertRaises(ValueError): update_online_fisher(self.kb,bad,None,2,'bad')

    def test_f_exact_budget_unique_scores_and_no_optimizer(self):
        before=policy_hash(self.kb)
        with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('F optimizer')):
            result=run_fisher_stage(self.envs,self.kb,None,self.config,32,self.rng,np.random.default_rng(5),11,'F')
        self.assertEqual((result['transitions'],result['scored_samples'],result['optimizer_updates']),(32,8,0))
        self.assertEqual(len(set(result['selected_ids'])),8); self.assertEqual(result['next_transition_id'],43)
        self.assertEqual(before,policy_hash(self.kb))
        self.assertTrue(all(p.grad is None for p in self.kb.parameters()))


if __name__=='__main__': unittest.main()
