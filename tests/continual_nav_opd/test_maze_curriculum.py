"""Independent dynamics/oracle, sampling, masking and gradient references."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from tasks.continual_nav_opd.config import load_config
from tasks.continual_nav_opd.contracts import CTMState, PolicySequenceOutput
from tasks.continual_nav_opd.data import MazeEntry
from tasks.continual_nav_opd.data.maze_curriculum import MazeStateTable, collect_sequences
from tasks.continual_nav_opd.envs.map_cache import MazeMapCache
from tasks.continual_nav_opd.learning.maze_progress import replay_sequences, run_maze_progress
from tasks.continual_nav_opd.models.ctm import Controller
from tasks.continual_nav_opd.teachers import shortest_path
from tasks.continual_nav.envs.maze import MazeEnv, MOVES
from tasks.continual_nav.envs.common import pixels


class TinyPolicy(nn.Module):
    """A small differentiable recurrence for the full-100 gradient reference."""
    def __init__(self):
        super().__init__()
        self.config = load_config('tasks/continual_nav_opd/configs/smoke.yaml')
        self.active = nn.Module()
        self.active.controller = nn.Module()
        self.active.controller.start_pre = nn.Parameter(torch.tensor([[.2]]))
        self.active.controller.start_post = nn.Parameter(torch.tensor([[.1]]))
        self.active.encoder = nn.Linear(3,1)
        self.active.actor = nn.Linear(1,5)
        self.adapter = nn.Linear(1,1)
        self._frozen = False

    def initial_state(self,batch):
        c = self.active.controller
        return CTMState(c.start_pre.expand(batch,1).unsqueeze(-1).clone(),c.start_post.expand(batch,1).unsqueeze(-1).clone())

    def step(self,rgb,state,starts,*,task):
        if bool(starts.any()): raise AssertionError('unexpected internal reset')
        feature = self.active.encoder(rgb.float().mean((-2,-1))/255).unsqueeze(-1)
        post = (state.post*.7+state.pre*.1+feature).tanh()
        return self.active.actor(post[...,0]),CTMState(state.post,post)

    def sequence(self,rgb,state,starts,valid,*,task):
        out = []
        for t in range(len(rgb)):
            logits,proposed = self.step(rgb[t],state,starts[t],task=task)
            mask=valid[t,:,None,None]
            state=CTMState(torch.where(mask,proposed.pre,state.pre),torch.where(mask,proposed.post,state.post))
            out.append(logits)
        return PolicySequenceOutput(torch.stack(out),state)


class MazeCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entries = []
        for i in range(3):
            rgb = np.zeros((19,19,3),dtype=np.uint8)
            rgb[1:4,1:4+i] = 255
            rgb[1,1]=(255,0,0)
            rgb[3,3+i]=(0,255,0)
            path=self.root/'train/0'/f'{i}.png'
            path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(rgb).save(path)
            self.entries.append(MazeEntry(f'train/0/{i}.png',hashlib.sha256(path.read_bytes()).hexdigest()))
        self.cache = MazeMapCache(self.root,self.entries)
        self.table = MazeStateTable(self.cache,self.entries[:2])

    def test_every_label_transition_and_pixel_matches_existing_environment(self):
        table = self.table
        for m,entry in enumerate(table.entries):
            env = MazeEnv(self.root,(entry,))
            env.reset()
            for sid in range(table.offsets[m],table.offsets[m+1]):
                pos = tuple(map(int,divmod(int(table.cells[sid]),19)))
                image=env.base_rgb.copy(); image[pos]=(255,0,0)
                np.testing.assert_array_equal(table.observations([sid])[0],pixels(image))
                if table.terminal[sid]: continue
                path=shortest_path(env.walls,pos,env.goal_pos)
                self.assertEqual(int(table.distances[sid]),len(path))
                self.assertEqual(int(table.labels[sid]),path[0])
                for a in range(5):
                    env.agent_pos=pos; env.step_count=0; env._done=False
                    _,_,done,_,_=env.step(a)
                    target=int(table.transitions[sid,a])
                    self.assertEqual(tuple(map(int,divmod(int(table.cells[target]),19))),env.agent_pos)
                    self.assertEqual(bool(table.terminal[target]),done)
        self.assertFalse(self.cache.images.flags.writeable)
        self.assertEqual(table.transitions.dtype,np.int32)

    def test_uniform_state_sampling_prefix_and_no_evaluation_leak(self):
        t=self.table; rng=np.random.default_rng(1)
        ids=t.sample(2,100000,rng)
        eligible=t.start_ids[:t.start_offsets[2]]
        counts=np.bincount(ids,minlength=len(t.cells))[eligible]
        self.assertLess(float(np.max(np.abs(counts-counts.mean()))/counts.mean()),.08)
        self.assertFalse(t.terminal[ids].any())
        self.assertTrue((t.map_ids[t.sample(1,1000,rng)]==0).all())
        self.assertNotIn(self.entries[2],t.entries)
        self.assertEqual(t.start_offsets[1],len(t.start_ids[t.map_ids[t.start_ids]==0]))

    def test_probabilistic_actions_serial_reference_terminal_mask_and_pure_replay(self):
        t=self.table; policy=TinyPolicy()
        original_weights=policy.active.actor.weight.detach().clone()
        with torch.no_grad():
            policy.active.actor.weight.zero_(); policy.active.actor.bias.zero_()
        # Goal-reaching actions are valid, then no new episode is started.
        adjacent=np.flatnonzero(t.distances==1)[0]
        starts=np.full(100,adjacent,dtype=np.int32)
        fixed=np.full((5,100),int(t.labels[adjacent]),dtype=np.int64)
        args=(t,policy,2,torch.Generator().manual_seed(1),np.random.default_rng(2))
        batch=collect_sequences(*args,starts=starts,fixed_actions=fixed)
        serial=collect_sequences(*args,starts=starts,fixed_actions=fixed,forward_batch=1)
        for name in ('obs','labels','actions','valid'): torch.testing.assert_close(getattr(batch,name),getattr(serial,name),atol=0,rtol=0)
        self.assertEqual(int(batch.valid.sum()),100)
        self.assertTrue(batch.valid[0].all()); self.assertFalse(batch.valid[1:].any())
        self.assertEqual(int(batch.terminated.sum()),100)
        sampled=collect_sequences(*args)
        self.assertGreater(len(torch.unique(sampled.actions[sampled.valid])),1)
        with torch.no_grad(): policy.active.actor.weight.copy_(original_weights)
        sampled=collect_sequences(*args)
        before=args[3].get_state().clone()
        optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
        with patch.object(t,'observations',side_effect=AssertionError('replay stepped environment')), \
             patch.object(torch.Tensor,'argmax',side_effect=AssertionError('Maze P computed agreement')):
            metrics=replay_sequences(policy,sampled,optimizer)
        self.assertTrue(torch.equal(before,args[3].get_state()))
        self.assertGreater(metrics['initial_window_grad_norm'],0)
        self.assertNotIn('agreement',metrics)
        self.assertNotIn('step_agreement',metrics)
        self.assertEqual(len(metrics['step_kl']),5)
        self.assertEqual({float(x['step']) for x in optimizer.state.values()},{1.})

    def test_microbatch_gradient_and_adam_equal_independent_full_batch_reference(self):
        torch.manual_seed(2)
        grouped=TinyPolicy(); full=deepcopy(grouped)
        batch=collect_sequences(self.table,grouped,2,torch.Generator().manual_seed(4),np.random.default_rng(3))
        opt1=torch.optim.Adam(grouped.parameters(),lr=1e-4,eps=1e-5)
        opt2=torch.optim.Adam(full.parameters(),lr=1e-4,eps=1e-5)
        with patch.object(opt1,'step',wraps=opt1.step) as call:
            replay_sequences(grouped,batch,opt1)
        self.assertEqual(call.call_count,1)
        out=full.sequence(batch.obs,full.initial_state(100),torch.zeros_like(batch.valid),batch.valid,task='maze_medium')
        F.cross_entropy(out.logits[batch.valid],batch.labels[batch.valid]).backward()
        torch.nn.utils.clip_grad_norm_(full.parameters(),.5)
        opt2.step()
        for a,b in zip(grouped.parameters(),full.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=2e-5) if a.grad is not None else self.assertIsNone(b.grad)
            torch.testing.assert_close(a,b,atol=1e-7,rtol=2e-5)

    def test_finite_logits_with_overflowing_loss_stop_before_adam(self):
        student=TinyPolicy()
        batch=collect_sequences(self.table,student,2,torch.Generator().manual_seed(1),np.random.default_rng(2))
        batch.labels.zero_()
        logits=torch.full((5,5,5),3e38,requires_grad=True)
        with torch.no_grad(): logits[...,0]=-3e38
        optimizer=torch.optim.Adam(student.parameters())
        with patch.object(student,'sequence',return_value=PolicySequenceOutput(logits,None)), \
             patch.object(optimizer,'step',side_effect=AssertionError('nonfinite update')):
            with self.assertRaisesRegex(ValueError,'nonfinite Maze loss'):
                replay_sequences(student,batch,optimizer)

    def test_fifth_decision_gradient_reaches_first_state_and_learned_initial_window(self):
        config=load_config('tasks/continual_nav_opd/configs/smoke.yaml')
        controller=Controller(config)
        state=controller.initial_state(1)
        first=None
        for decision in range(5):
            for _ in range(5): state,_=controller.tick(torch.rand(1,128,21,21),state)
            if decision==0:
                first=state
                first.pre.retain_grad(); first.post.retain_grad()
        controller.readout(state).square().mean().backward()
        self.assertGreater(float(first.post.grad.abs().sum()),0)
        self.assertGreater(float(controller.start_pre.grad.abs().sum()),0)
        self.assertGreater(float(controller.start_post.grad.abs().sum()),0)

    def test_pool_resume_restores_weights_adam_and_rng_without_repeating_completed_pool(self):
        config=load_config('tasks/continual_nav_opd/configs/smoke.yaml')
        config=replace(config,maze_progress=replace(config.maze_progress,pool_sizes=(1,2)))
        torch.manual_seed(7); student=TinyPolicy(); student.config=config; initial=deepcopy(student)
        action=torch.Generator().manual_seed(8); rng=np.random.default_rng(9)
        saved={}
        def boundary(progress):
            saved.update(deepcopy(progress)); saved['action']=action.get_state(); saved['rng']=deepcopy(rng.bit_generator.state)
            if progress['next_pool']==1: raise RuntimeError('power loss')
        with self.assertRaisesRegex(RuntimeError,'power loss'):
            run_maze_progress(self.table,student,config,action,rng,on_pool=boundary)
        restored=TinyPolicy(); restored.config=config; restored.load_state_dict(saved['dual'])
        action.set_state(saved['action']); rng.bit_generator.state=saved['rng']
        result=run_maze_progress(self.table,restored,config,action,rng,resume=saved)
        reference=run_maze_progress(self.table,initial,config,torch.Generator().manual_seed(8),np.random.default_rng(9))
        self.assertEqual(result.optimizer_updates,2)
        self.assertNotIn('agreement',saved['sums'])
        self.assertNotIn('agreement',result.statistics)
        self.assertEqual(result.transitions,reference.transitions)
        for a,b in zip(restored.parameters(),initial.parameters()): torch.testing.assert_close(a,b,atol=0,rtol=0)


if __name__=='__main__': unittest.main()
