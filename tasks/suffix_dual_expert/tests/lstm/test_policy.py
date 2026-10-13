"""Online LSTM state, attention, BPTT and checkpoint contracts."""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from ...config import Config, PACKAGE, ROOT, load_config
from ...models import build_policy, frozen_copy, CTMPolicy, LSTMPolicy, load_teacher
from ...models.common.state import CTMState, LSTMState, DualState, detached, select, pack_state, unpack_state
from ...suffix import replay
from ...compress import Window, update_pair, Collector
from ...fisher import ewc_loss
from ...checkpoint import snapshot, atomic_torch, load_boundary, save_boundary, source_hashes
from ...evaluate import load_policy


class SmallEncoder(nn.Module):
    """Only for temporal/gradient tests; real ResNet tested separately."""
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3,128)

    def forward(self,rgb):
        return self.projection(rgb.float().mean((-2,-1))/255.)[:,:,None,None].expand(-1,-1,21,21)


def small_policy():
    model = build_policy(Config(model_type='lstm',lstm_hidden=16,lstm_layers=2,checkpoint_encoder=False))
    model.encoder = SmallEncoder()
    return model


class LSTMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)

    def test_configs_and_independent_teacher_spec(self):
        ctm = load_config(PACKAGE/'configs/ctm.yaml')
        lstm = load_config(PACKAGE/'configs/lstm.yaml')
        self.assertEqual(ctm.budget(),lstm.budget())
        self.assertEqual(ctm.resolved(),Config().resolved())
        self.assertEqual(lstm.model_spec()['lstm']['ticks_by_task'],{'maze_medium':5,'fourrooms':2})
        self.assertNotIn('ctm',lstm.model_spec())
        self.assertEqual(lstm.model_spec()['vision'],ctm.model_spec()['vision'])
        self.assertEqual(lstm.model_spec()['attention'],ctm.model_spec()['attention'])
        altered=lstm.model_spec();altered['vision']['norm']='changed'
        self.assertEqual(lstm.model_spec()['vision']['norm'],'groupnorm32')
        self.assertEqual(load_config(PACKAGE/'configs/lstm.a4000.yaml').budget(),lstm.budget())
        for changes in ({'model_type':'other'},{'lstm_layers':0},{'lstm_hidden':False}):
            with self.assertRaises(ValueError): replace(lstm,**changes)
        files=source_hashes()
        for name in ('common/vision.py','ctm/policy.py','ctm/teacher.py','lstm/policy.py'):
            self.assertIn('tasks/suffix_dual_expert/models/'+name,files)

    def test_real_visual_forward_and_recompute(self):
        config=Config(model_type='lstm')
        normal=build_policy(replace(config,checkpoint_encoder=False))
        recomputed=build_policy(config);recomputed.load_state_dict(normal.state_dict())
        self.assertIsInstance(normal,LSTMPolicy)
        self.assertEqual(normal.controller.lstm.num_layers,2)
        self.assertEqual(normal.controller.lstm.hidden_size,512)
        self.assertEqual(normal.controller.attention.query_input.in_features,512)
        rgb=torch.randint(0,256,(2,1,3,84,84),dtype=torch.uint8)
        starts=torch.tensor([[True],[False]])
        results=[]
        for model in (normal,recomputed):
            logits,state=model.sequence(rgb,model.initial_state(1),starts,task='fourrooms')
            self.assertEqual(logits.shape,(2,1,5));self.assertEqual(state.h.shape,(1,2,512))
            self.assertTrue(torch.isfinite(logits).all())
            logits.square().sum().backward();results.append(logits.detach())
        torch.testing.assert_close(*results,rtol=0,atol=0)
        for (name,a),(_,b) in zip(normal.named_parameters(),recomputed.named_parameters()):
            self.assertIsNotNone(a.grad,name)
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0,msg=name)
        self.assertGreater(float(normal.encoder.backbone.conv1.weight.grad.abs().sum()),0)

    def test_query_ticks_continuous_state_episode_reset_and_sequence(self):
        model=small_policy()
        rgb=torch.randint(0,256,(3,2,3,84,84),dtype=torch.uint8)
        starts=torch.tensor([[True,True],[False,False],[True,False]])
        for task,ticks in (('maze_medium',5),('fourrooms',2)):
            with torch.no_grad():
                initial=model.initial_state(2);queries=[]
                handle=model.controller.attention.query_input.register_forward_pre_hook(
                    lambda m,args:queries.append(args[0].clone()))
                first,state=model.step(rgb[0],initial,starts[0],task=task);handle.remove()
                self.assertEqual(len(queries),ticks)
                torch.testing.assert_close(queries[0],initial.h[:,-1],rtol=0,atol=0)
                self.assertFalse(torch.equal(queries[0],queries[1]))
                continued,_=model.step(rgb[1],state,starts[1],task=task)
                fresh,_=model.step(rgb[1],model.initial_state(2),torch.ones(2,dtype=torch.bool),task=task)
                self.assertFalse(torch.equal(continued,fresh))
                values=[first,continued]
                _,state=model.step(rgb[1],state,starts[1],task=task)
                last,state=model.step(rgb[2],state,starts[2],task=task);values.append(last)
                batched,batched_state=model.sequence(rgb,model.initial_state(2),starts,task=task)
                torch.testing.assert_close(batched,torch.stack(values),rtol=2e-5,atol=1e-6)
                torch.testing.assert_close(batched_state.h,state.h,rtol=2e-5,atol=1e-6)
                reset_logit,_=model.step(rgb[2,:1],model.initial_state(1),torch.ones(1,dtype=torch.bool),task=task)
                torch.testing.assert_close(last[:1],reset_logit,rtol=2e-5,atol=1e-6)
        frozen=frozen_copy(model);frozen.train()
        self.assertFalse(frozen.training);self.assertTrue(all(not p.requires_grad for p in frozen.parameters()))
        with self.assertRaises(ValueError):model.step(rgb[0].float(),model.initial_state(2),starts[0],task='fourrooms')
        with self.assertRaises(ValueError):model.step(rgb[0],CTMState(torch.zeros(2,16,40),torch.zeros(2,16,40)),starts[0],task='fourrooms')

    def test_prefix_gradients_and_microbatch_objective(self):
        model=small_policy();weights=deepcopy(model.state_dict())
        images=torch.randint(0,256,(10,3,84,84),dtype=torch.uint8)
        labels=torch.arange(10)%5;ids=torch.arange(20).reshape(10,2)%10
        mask=torch.ones(5,2,dtype=torch.bool);mask[-2:,1]=False
        features=[]
        def hook(m,args,value):
            value.retain_grad();features.append(value)
        handle=model.encoder.register_forward_hook(hook)
        loss=replay(model,images,labels,ids,mask)/mask.sum();loss.backward();handle.remove()
        self.assertGreater(float(features[0].grad[:10].abs().sum()),0)  # prefix, not just the taught suffix.
        self.assertGreater(float(model.controller.start_hidden.grad.abs().sum()),0)
        self.assertGreater(float(model.controller.start_cell.grad.abs().sum()),0)
        expected={n:p.grad.clone() for n,p in model.named_parameters()}
        for micro in (1,2):
            other=small_policy();other.load_state_dict(weights)
            actual=0.
            for first in range(0,2,micro):
                part=replay(other,images,labels,ids[:,first:first+micro],mask[:,first:first+micro])/mask.sum()
                part.backward();actual+=float(part.detach())
            self.assertAlmostEqual(actual,float(loss.detach()),places=5)
            for name,p in other.named_parameters():
                torch.testing.assert_close(p.grad,expected[name],rtol=5e-4,atol=2e-6,msg=name)

    def test_state_selection_serialisation_and_wrong_collector_type(self):
        model=small_policy();state=model.initial_state(3)
        subset=select(state,torch.tensor([2,0]));self.assertEqual(subset.h.shape,(2,2,16))
        torch.testing.assert_close(subset.h,state.h[[2,0]])
        copied=detached(state);self.assertFalse(copied.h.requires_grad)
        self.assertNotEqual(copied.h.data_ptr(),state.h.data_ptr())
        restored=unpack_state(pack_state(copied),'cpu','lstm')
        torch.testing.assert_close(restored.h,state.h);torch.testing.assert_close(restored.c,state.c)
        with self.assertRaisesRegex(ValueError,'state type'):unpack_state(pack_state(copied),'cpu','ctm')
        with self.assertRaises(ValueError):unpack_state(dict(kind='unknown'),'cpu')
        legacy=unpack_state(dict(pre=torch.zeros(1,1,40),post=torch.ones(1,1,40)),'cpu')
        self.assertIsInstance(legacy,CTMState)
        collector=object.__new__(Collector);collector.state=copied
        collector.teacher_state=DualState(legacy,legacy);collector.device='cpu'
        with self.assertRaisesRegex(ValueError,'state type'):
            collector.restore(dict(state=pack_state(legacy),teacher_state=pack_state(collector.teacher_state)))

    def test_snapshot_legacy_and_architecture_rejection(self):
        (ROOT/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as directory:
            file=Path(directory)/'snapshot.pt';config=Config(model_type='lstm',lstm_hidden=16)
            model=build_policy(config);snapshot(file,model,config.resolved())
            restored=load_policy(file,'cpu',config)
            self.assertIsInstance(restored,LSTMPolicy)
            for n,v in model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[n],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'architecture'):load_policy(file,'cpu',Config())
            artifact=torch.load(file,weights_only=True);artifact['model_type']='ctm';atomic_torch(file,artifact)
            with self.assertRaisesRegex(ValueError,'architecture'):load_policy(file,'cpu')
            ctm=build_policy(Config());legacy=dict(schema='suffix_dual_expert_policy_v1',model='single',
                config={k:v for k,v in Config().resolved().items() if k not in ('model_type','lstm_hidden','lstm_layers')},
                state_dict=ctm.state_dict())
            atomic_torch(file,legacy);self.assertIsInstance(load_policy(file,'cpu'),CTMPolicy)
            legacy['config']['model_type']='lstm';atomic_torch(file,legacy)
            with self.assertRaisesRegex(ValueError,'CTM only'):load_policy(file,'cpu')
            identity=dict(config=config.resolved())
            boundary=dict(schema='suffix_dual_expert_v2',identity=identity,model_type='lstm',model_spec=config.model_spec(),
                phase='maze',maze_updates=0,windows=0)
            save_boundary(Path(directory),boundary);load_boundary(Path(directory),identity)
            with self.assertRaisesRegex(ValueError,'resume config'):
                load_boundary(Path(directory),dict(config=replace(config,model_type='ctm').resolved()))
            boundary['model_type']='ctm';save_boundary(Path(directory),boundary)
            with self.assertRaisesRegex(ValueError,'architecture'):load_boundary(Path(directory),identity)
            boundary['schema']='suffix_dual_expert_v1';save_boundary(Path(directory),boundary)
            with self.assertRaisesRegex(ValueError,'resume config'):load_boundary(Path(directory),identity)

    def test_joint_loss_ewc_switch_for_lstm(self):
        model=small_policy();weights=deepcopy(model.state_dict())
        batches=[]
        for task in ('maze_medium','fourrooms'):
            rgb=torch.randint(0,256,(3,4,3,84,84),dtype=torch.uint8)
            starts=torch.zeros(3,4,dtype=torch.bool);starts[0]=True
            batches.append(Window(rgb,starts,torch.tensor([.1,.2,.3,.2,.2]).expand(3,4,5),
                                  detached(model.initial_state(4)),task))
        fisher=dict(importance={n:torch.ones_like(p)*.001 for n,p in model.named_parameters()},
                    theta_star={n:p.detach().clone()-.01 for n,p in model.named_parameters()},samples=1)
        for enabled in (True,False):
            model.load_state_dict(weights)
            config=Config(model_type='lstm',slots_per_task=4,compress_microbatch=1,ewc_enabled=enabled)
            opt=torch.optim.Adam(model.parameters(),lr=config.lr,eps=config.eps)
            with patch('tasks.suffix_dual_expert.compress.ewc_loss',wraps=ewc_loss) as calls:
                stats=update_pair(model,batches,opt,fisher if enabled else None,np.random.default_rng(39),config,diagnostics=True)
            self.assertEqual(calls.call_count,4 if enabled else 0)
            self.assertEqual(stats['adam_updates'],4)
            self.assertGreater(stats['maze_gradient_norm'],0);self.assertGreater(stats['fourrooms_gradient_norm'],0)
            self.assertGreater(stats['ewc'],0) if enabled else self.assertEqual(stats['ewc'],0)
            self.assertTrue(all(int(v['step'])==4 for v in opt.state.values()))


if __name__=='__main__':
    unittest.main()
