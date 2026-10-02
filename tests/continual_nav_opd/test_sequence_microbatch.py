"""Accumulate full environment sequences without changing Adam or EWC boundaries."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from tasks.continual_nav_opd.config import load_config,validate_config
from tasks.continual_nav_opd.contracts import CTMState,SequenceBatch,PolicySequenceOutput
from tasks.continual_nav_opd.models import StandalonePolicy,DualPolicy,detach_clone_state
from tasks.continual_nav_opd.learning.sequence import update_window


class TinyPolicy(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        self.encoder=nn.Linear(1,2);self.controller=nn.Linear(2,2);self.actor=nn.Linear(2,5)
        self.calls=[]
    def sequence(self,images,state,starts,valid=None,*,task):
        self.calls.append((images.shape[0],images.shape[1]))
        features=images.float().mean((2,3,4),keepdim=False).unsqueeze(-1)/255
        return PolicySequenceOutput(self.actor(self.controller(self.encoder(features))),state)


class SequenceMicrobatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def config(self,slots=4,minibatches=1,micro=0):
        c=load_config('tasks/continual_nav_opd/configs/smoke.yaml')
        return replace(c,training=replace(c.training,num_envs=slots),
                       optimization=replace(c.optimization,minibatches=minibatches,
                                            sequence_microbatch_envs=micro,encoder_microbatch_images=2))
    def batch(self,policy,length=3,slots=4,valid=None):
        valid=torch.ones(length,slots,dtype=torch.bool) if valid is None else valid
        images=torch.randint(256,(length,slots,3,84,84),dtype=torch.uint8)
        targets=torch.rand(length,slots,5);targets=targets/targets.sum(-1,keepdim=True)
        starts=torch.zeros(length,slots,dtype=torch.bool);starts[0]=True
        if length>1:starts[1,0]=True
        origin=detach_clone_state(policy.initial_state(slots)) if hasattr(policy,'initial_state') else CTMState(torch.zeros(slots,1,1),torch.zeros(slots,1,1))
        return SequenceBatch(images,starts,valid,valid.clone(),targets,torch.zeros(length,slots,dtype=torch.long),
                             torch.arange(length*slots).reshape(length,slots),origin,'fixture','fourrooms',2)
    def optimizer(self,policy):
        return torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=1e-4,eps=1e-5)
    def compare(self,dual=False):
        torch.manual_seed(12)
        full=self.config();small=self.config(micro=1)
        a=DualPolicy(full,StandalonePolicy(full),kb_ready=True) if dual else StandalonePolicy(full)
        b=DualPolicy(small,StandalonePolicy(small),kb_ready=True) if dual else StandalonePolicy(small)
        if dual:
            with torch.no_grad():a.adapter.gate.fill_(.3)
        b.load_state_dict(a.state_dict())
        valid=torch.tensor([[True,True,True,False],[True,False,True,False],[True,False,False,False]])
        batch=self.batch(a,valid=valid)
        oa,ob=self.optimizer(a),self.optimizer(b)
        calls=[0,0]
        def regularizer(policy,index):
            def compute():
                calls[index]+=1
                actor=policy.active.actor if dual else policy.actor
                value=.025*actor[-1].weight.square().sum()
                return value,{'ewc_controller_actor':value}
            return compute
        before={n:v.clone() for n,v in a.kb.state_dict().items()} if dual else {}
        ma=update_window(a,batch,oa,full,np.random.default_rng(5),regularizer(a,0))
        mb=update_window(b,batch,ob,small,np.random.default_rng(5),regularizer(b,1))
        self.assertEqual(calls,[1,1])
        self.assertEqual((ma['optimizer_updates'],mb['optimizer_updates'],mb['eligible_target_steps']),(1,1,6))
        for key in ('kl','total_loss','ewc','ewc_controller_actor','agreement','student_entropy','grad_norm'):
            self.assertAlmostEqual(ma[key],mb[key],delta=2e-5,msg=key)
        for (name,x),(_,y) in zip(a.named_parameters(),b.named_parameters()):
            torch.testing.assert_close(x,y,atol=2e-5,rtol=2e-4,msg=name)
            if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,atol=2e-5,rtol=2e-3,msg=name)
        if dual:
            for name,value in b.kb.state_dict().items():torch.testing.assert_close(value,before[name],atol=0,rtol=0)
    def test_real_kb_unequal_counts_resets_and_ewc_once(self):self.compare()
    def test_real_dual_preserves_frozen_kb_and_full_gradient(self):self.compare(dual=True)
    def test_32_slots_400_observations_one_step_per_original_group(self):
        torch.manual_seed(3)
        config=self.config(slots=32,minibatches=4,micro=1)
        model=TinyPolicy(config);batch=self.batch(model,length=50,slots=32)
        optimizer=self.optimizer(model)
        with patch.object(optimizer,'step',wraps=optimizer.step) as steps:
            metrics=update_window(model,batch,optimizer,config,np.random.default_rng(3))
        self.assertEqual(steps.call_count,4)
        self.assertEqual(metrics['eligible_target_steps'],1600)
        self.assertEqual(model.calls,[(50,1)]*32)
    def test_empty_chunk_does_not_apply_extra_regularizer_or_step(self):
        config=self.config(micro=1);model=TinyPolicy(config)
        valid=torch.zeros(3,4,dtype=torch.bool);valid[:,2]=True
        batch=self.batch(model,valid=valid);optimizer=self.optimizer(model)
        calls=[]
        def regularizer():
            calls.append(1);v=model.actor.weight.square().sum()*.001
            return v,{'ewc_controller_actor':v}
        result=update_window(model,batch,optimizer,config,np.random.default_rng(1),regularizer)
        self.assertEqual((len(calls),result['optimizer_updates'],result['eligible_target_steps']),(1,1,3))
        empty=replace(batch,valid_mask=torch.zeros_like(valid),target_mask=torch.zeros_like(valid))
        with patch.object(optimizer,'step',side_effect=AssertionError('empty step')):
            self.assertEqual(update_window(model,empty,optimizer,config,np.random.default_rng(1),regularizer)['optimizer_updates'],0)
    def test_invalid_microbatch(self):
        for value in (-1,True,5):
            with self.assertRaises(ValueError):validate_config(self.config(micro=value))


if __name__=='__main__':unittest.main()
