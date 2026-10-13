from dataclasses import replace
from copy import deepcopy
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from tasks.suffix_dual_expert.config import Config, path, PACKAGE, ROOT,load_config
from tasks.suffix_dual_expert.model import State, detached, select, load_teacher, Policy
from tasks.suffix_dual_expert.suffix import candidates, replay, update, PrefixSamplingExhausted
from tasks.suffix_dual_expert.compress import Window, kl_divergence, update_pair
from tasks.suffix_dual_expert.fisher import estimate, ewc_loss, validate_fisher
from tasks.suffix_dual_expert.checkpoint import (capture_rng, restore_rng, atomic_torch, model_hash,
    atomic_json,tensor_state,tensor_hash,save_boundary,load_boundary,snapshot)
from tasks.suffix_dual_expert.train import fork_compress_boundary
from tasks.suffix_dual_expert.fisher import fisher_hash
from tasks.suffix_dual_expert.envs import file_sha
from tasks.suffix_dual_expert.envs import MapCache, MazeEnv, FourRoomsEnv, training_entries
from tasks.suffix_dual_expert.evaluate import paired, holm, wilson


class ForbidLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('tasks.continual_nav'):
            raise ImportError('legacy package forbidden: '+fullname)


sys.meta_path.insert(0,ForbidLegacy())


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Parameter(torch.tensor(.1))
        self.weight = nn.Parameter(torch.tensor(.02))
        self.bias = nn.Parameter(torch.tensor([5.,0.,0.,0.,0.]))
        self.history = []

    def initial_state(self,batch):
        x = self.initial.expand(batch,1,1).clone()
        return State(x,x.clone())

    def step(self,rgb,state,starts,*,task):
        initial = self.initial_state(len(rgb))
        prev = torch.where(starts[:,None,None],initial.post,state.post)
        x = .9*prev+self.weight+rgb.float().flatten(1).mean(1)[:,None,None]/25500
        if torch.is_grad_enabled():
            x.retain_grad(); self.history.append(x)
        return self.bias + x.flatten(1)*torch.arange(5,device=x.device),State(x,x)

    def sequence(self,images,state,starts,*,task):
        values=[]
        for rgb,mask in zip(images,starts):
            logits,state=self.step(rgb,state,mask,task=task);values.append(logits)
        return torch.stack(values),state


class CTMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_teacher_golden_fixture(self):
        fixture=PACKAGE/'tests/fixtures/teacher_golden.npz'
        if not fixture.exists():self.skipTest('migration fixture not yet generated')
        value=np.load(fixture)
        teacher=load_teacher(path(Config().teacher_checkpoint),'cpu')
        state=teacher.initial_state(value['rgb'].shape[1]);outputs=[]
        for rgb,starts in zip(value['rgb'],value['starts']):
            logits,state=teacher.step(torch.from_numpy(rgb),state,torch.from_numpy(starts),task='fourrooms')
            outputs.append(logits)
        np.testing.assert_allclose(torch.stack(outputs).numpy(),value['logits'],rtol=1e-5,atol=1e-6)
        for name,tensor in [('kb_pre',state.kb.pre),('kb_post',state.kb.post),('active_pre',state.active.pre),('active_post',state.active.post)]:
            np.testing.assert_allclose(tensor.numpy(),value[name],rtol=1e-5,atol=1e-6)

    def test_student_golden_fixture(self):
        fixture=PACKAGE/'tests/fixtures/student_golden.npz'
        if not fixture.exists():self.skipTest('migration fixture not yet generated')
        value=np.load(fixture)
        artifact=torch.load(path(Config().teacher_checkpoint),map_location='cpu',weights_only=True)
        model=Policy();model.load_state_dict({n[7:]:v for n,v in artifact['state_dict'].items() if n.startswith('active.')})
        with torch.no_grad():
            logits,state=model.sequence(torch.from_numpy(value['rgb']),model.initial_state(value['rgb'].shape[1]),
                                        torch.from_numpy(value['starts']),task='maze_medium')
        np.testing.assert_allclose(logits.numpy(),value['logits'],rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(state.pre.numpy(),value['pre'],rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(state.post.numpy(),value['post'],rtol=1e-5,atol=1e-6)

    def test_activation_recompute_has_identical_gradients(self):
        torch.manual_seed(17)
        normal=Policy(checkpoint_encoder=False);recomputed=Policy(checkpoint_encoder=True)
        recomputed.load_state_dict(normal.state_dict())
        rgb=torch.randint(0,255,(2,1,3,84,84),dtype=torch.uint8)
        starts=torch.tensor([[True],[False]])
        for model in (normal,recomputed):
            logits,_=model.sequence(rgb,model.initial_state(1),starts,task='maze_medium')
            logits.square().sum().backward()
        for (name,a),(_,b) in zip(normal.named_parameters(),recomputed.named_parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0,msg=name)

    def test_batched_suffix_matches_stepwise_loss_gradients_and_adam(self):
        torch.manual_seed(19)
        images=torch.randint(0,255,(12,3,84,84),dtype=torch.uint8)
        labels=torch.arange(12)%5
        ids=torch.arange(20).reshape(10,2)%12
        mask=torch.tensor([[True,True],[True,True],[True,False],[True,False],[False,False]])
        reference=Policy(checkpoint_encoder=False)
        initial={n:v.clone() for n,v in reference.state_dict().items()}
        state=reference.initial_state(2);total=torch.zeros(())
        for t in range(10):
            logits,state=reference.step(images[ids[t]],state,torch.full((2,),t==0),task='maze_medium')
            if t>=5:
                total=total+(torch.nn.functional.cross_entropy(logits,labels[ids[t]],reduction='none')*mask[t-5]).sum()
        (total/mask.sum()).backward()
        expected={n:p.grad.clone() for n,p in reference.named_parameters()}
        opt=torch.optim.Adam(reference.parameters(),lr=1e-4,eps=1e-5)
        torch.nn.utils.clip_grad_norm_(reference.parameters(),.5);opt.step()
        for microbatch,recompute in ((1,False),(2,False),(2,True)):
            model=Policy(encoder_chunk=8,checkpoint_encoder=recompute);model.load_state_dict(initial)
            actual=0.
            for first in range(0,2,microbatch):
                loss=replay(model,images,labels,ids[:,first:first+microbatch],mask[:,first:first+microbatch])
                (loss/mask.sum()).backward();actual+=float(loss.detach())
            self.assertAlmostEqual(actual,float(total.detach()),places=4)
            for name,p in model.named_parameters():
                torch.testing.assert_close(p.grad,expected[name],rtol=3e-4,atol=2e-6,msg=name)
            self.assertGreater(float(model.controller.start_pre.grad.abs().sum()),0)
            self.assertGreater(float(model.controller.start_post.grad.abs().sum()),0)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,eps=1e-5)
            torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
            for (name,p),(_,q) in zip(model.named_parameters(),reference.named_parameters()):
                torch.testing.assert_close(p,q,rtol=1e-5,atol=3e-7,msg=name)
