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


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_budget_and_legacy_block(self):
        config=Config();stages=config.schedule()
        self.assertEqual(config.budget()['maze_adam'],23500)
        self.assertEqual(len(stages),42)
        self.assertEqual(sum(s['updates'] for s in stages[:7]),9000)
        self.assertEqual([s['updates'] for s in stages[7:14]],[200,200,500,500,500,500,500])
        self.assertEqual(config.budget()['compress_adam'],3200)
        self.assertEqual(config.budget()['evaluation_episodes'],3200)
        self.assertEqual(load_config(PACKAGE/'config.yaml').resolved(),config.resolved())
        self.assertEqual(load_config(PACKAGE/'config.a4000.yaml').budget(),config.budget())
        with self.assertRaises(ImportError):
            __import__('tasks.continual_nav_opd')

    def test_all_lengths_prefix_bptt_and_terminal_masks(self):
        images=torch.ones(2,1,1,1,dtype=torch.uint8)
        labels=torch.zeros(2,dtype=torch.long)
        transitions=torch.zeros(2,5,dtype=torch.long)
        for length in (5,10,15,20,25,30):
            model=Toy()
            ids,_,mask=candidates(model,images,labels,transitions,torch.tensor([0,1]),length)
            self.assertEqual(ids.shape,(length,2));self.assertEqual(int(mask.sum()),10)
            replay(model,images,labels,ids,mask).backward()
            self.assertGreater(float(model.history[0].grad.abs().sum()),0)
            self.assertGreater(float(model.initial.grad.abs()),0)
        transitions.fill_(-1)
        ids,_,mask=candidates(Toy(),images,labels,transitions,torch.tensor([0]),5)
        self.assertEqual(mask[:,0].tolist(),[True,False,False,False,False])
        ids,_,_=candidates(Toy(),images,labels,transitions,torch.tensor([0]),10)
        self.assertEqual(ids.shape[1],0)

    def test_exhausted_does_not_update(self):
        config=replace(Config(),sequences=2,suffix_microbatch=1,candidate_batch=2,max_candidates=4)
        model=Toy();optimizer=torch.optim.Adam(model.parameters())
        before=model_hash(model)
        tables=(torch.zeros(1,1,1,1,dtype=torch.uint8),torch.ones(1,dtype=torch.long),torch.zeros(1,5,dtype=torch.long))
        with self.assertRaises(PrefixSamplingExhausted) as caught:
            update(model,optimizer,tables,np.random.default_rng(0),1,10,config)
        self.assertEqual(caught.exception.candidates,4)
        self.assertEqual(before,model_hash(model));self.assertFalse(optimizer.state)

    def test_microbatch_matches_joint_objective_ewc_once(self):
        config=replace(Config(),slots_per_task=8,window=3,compress_microbatch=1)
        torch.manual_seed(7)
        model=Toy();reference=Toy();reference.load_state_dict(model.state_dict())
        batches=[]
        for task,length in (('maze_medium',3),('fourrooms',4)):
            images=torch.randint(0,255,(length,8,1,1,1),dtype=torch.uint8)
            masks=torch.zeros(length,8,dtype=torch.bool);masks[0]=True
            probs=torch.randn(length,8,5).softmax(-1)
            batches.append(Window(images,masks,probs,detached(model.initial_state(8)),task))
        fisher=dict(importance={n:torch.full_like(p,.03) for n,p in model.named_parameters()},
                    theta_star={n:p.detach().clone()-.01 for n,p in model.named_parameters()},samples=4)
        actual=torch.optim.Adam(model.parameters(),lr=config.lr,betas=config.betas,eps=config.eps)
        expected=torch.optim.Adam(reference.parameters(),lr=config.lr,betas=config.betas,eps=config.eps)
        with patch('tasks.suffix_dual_expert.compress.ewc_loss',wraps=ewc_loss) as calls:
            statistics=update_pair(model,batches,actual,fisher,np.random.default_rng(11),config)
            self.assertEqual(calls.call_count,4)
        rng=np.random.default_rng(11)
        groups=[np.split(rng.permutation(8),4) for _ in batches]
        for i in range(4):
            expected.zero_grad(set_to_none=True);loss=ewc_loss(reference,fisher,config.ewc_lambda)
            for k,batch in enumerate(batches):
                indices=torch.as_tensor(groups[k][i])
                logits,_=reference.sequence(batch.obs[:,indices],select(batch.initial,indices),batch.starts[:,indices],task=batch.task)
                loss=loss+.5*kl_divergence(logits,batch.targets[:,indices]).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),config.clip);expected.step()
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=1e-7)
        self.assertEqual(statistics['adam_updates'],statistics['ewc_applications'])

    def test_disabled_ewc_matches_pure_joint_kl(self):
        self.assertTrue(Config().ewc_enabled)
        for invalid in ('false',0,None):
            with self.assertRaises(ValueError):replace(Config(),ewc_enabled=invalid)
        config=replace(Config(),ewc_enabled=False,window=3)
        self.assertEqual(config.budget(),replace(config,ewc_enabled=True).budget())
        (ROOT/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as directory:
            file=Path(directory)/'off.yaml';file.write_text('ewc_enabled: false\n',encoding='utf-8')
            self.assertFalse(load_config(file).ewc_enabled)
        torch.manual_seed(31)
        model=Toy();reference=Toy();reference.load_state_dict(model.state_dict())
        batches=[]
        for task in ('maze_medium','fourrooms'):
            images=torch.randint(0,255,(3,16,1,1,1),dtype=torch.uint8)
            masks=torch.zeros(3,16,dtype=torch.bool);masks[0]=True;masks[2,::3]=True
            targets=torch.randn(3,16,5).softmax(-1)
            batches.append(Window(images,masks,targets,detached(model.initial_state(16)),task))
        actual=torch.optim.Adam(model.parameters(),lr=config.lr,betas=config.betas,eps=config.eps)
        expected=torch.optim.Adam(reference.parameters(),lr=config.lr,betas=config.betas,eps=config.eps)
        with patch('tasks.suffix_dual_expert.compress.ewc_loss',side_effect=AssertionError('EWC must not execute')) as calls:
            stats=update_pair(model,batches,actual,None,np.random.default_rng(37),config)
        calls.assert_not_called()
        self.assertFalse(stats['ewc_enabled']);self.assertEqual(stats['ewc'],0.)
        self.assertEqual(stats['ewc_applications'],0);self.assertEqual(stats['adam_updates'],4)
        self.assertAlmostEqual(stats['total_loss'],.5*(stats['maze_kl']+stats['fourrooms_kl']),places=6)
        rng=np.random.default_rng(37);groups=[np.split(rng.permutation(16),4) for _ in batches]
        for group in range(4):
            expected.zero_grad(set_to_none=True);loss=torch.zeros(())
            for i,batch in enumerate(batches):
                indices=torch.as_tensor(groups[i][group])
                logits,_=reference.sequence(batch.obs[:,indices],select(batch.initial,indices),batch.starts[:,indices],task=batch.task)
                loss=loss+.5*kl_divergence(logits,batch.targets[:,indices]).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),config.clip);expected.step()
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=1e-7)
        with self.assertRaisesRegex(ValueError,'requires Fisher'):
            update_pair(model,batches,actual,None,np.random.default_rng(37),replace(config,ewc_enabled=True))

    def test_fisher_is_nonnegative_without_weight_updates(self):
        class FakeEnvs:
            envs=list(range(8));task='maze_medium'
            def reset(self):return np.zeros((8,1,1,1),dtype=np.uint8)
            def step(self,actions):return self.reset(),np.zeros(8,dtype=bool)
        config=replace(Config(),slots_per_task=4,window=2,fisher_steps=16,fisher_samples=4,compress_microbatch=1)
        model=Toy();before=model_hash(model)
        value,stats=estimate(model,FakeEnvs(),config,torch.Generator().manual_seed(2),np.random.default_rng(3))
        validate_fisher(model,value)
        self.assertEqual(before,model_hash(model));self.assertEqual(stats['optimizer_updates'],0)
        self.assertEqual(stats['decisions'],16)
        self.assertGreater(sum(float(v.sum()) for v in value['importance'].values()),0)
        without,_=estimate(model,FakeEnvs(),replace(config,ewc_enabled=False),torch.Generator().manual_seed(2),np.random.default_rng(3))
        self.assertEqual(fisher_hash(value),fisher_hash(without))

    def test_f_completed_compress_forks_only_allow_ewc_flag(self):
        config=replace(Config(),device='cpu',lengths=(5,),pool_sizes=(2,),first_updates=(1,),suffix_updates=(1,),
                       slots_per_task=4,compress_microbatch=1,window=1,fisher_steps=8,fisher_samples=1)
        identity=dict(config=config.resolved(),source={'test':'same'},teacher_sha256='same')
        model=Toy();fisher=dict(importance={n:torch.ones_like(p) for n,p in model.named_parameters()},
                               theta_star={n:p.detach().clone() for n,p in model.named_parameters()},samples=1)
        streams={'action':torch.Generator().manual_seed(43),'prefix':np.random.default_rng(47),
                 'mini':np.random.default_rng(49),'fisher':np.random.default_rng(51)}
        (ROOT/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as directory:
            base=Path(directory);parent=base/'parent';parent.mkdir();(parent/'snapshots').mkdir()
            anchor=snapshot(parent/'snapshots/maze_before_compress.pt',model,config.resolved())
            original=dict(schema='suffix_dual_expert_v2',identity=identity,model_type=config.model_type,
                model_spec=config.model_spec(),phase='compress',windows=0,
                stage_index=1,maze_updates=1,completed_stages=config.schedule(),optimizer=None,collectors=None,
                student=tensor_state(model),fisher=fisher,rng=capture_rng(streams),snapshots={'maze_before_compress':anchor},
                fisher_statistics=dict(decisions=8,scores=1,optimizer_updates=0,weights_unchanged=True))
            atomic_json(parent/'protocol.json',dict(identity=identity));save_boundary(parent,original)
            before={str(p.relative_to(parent)):file_sha(p) for p in parent.rglob('*') if p.is_file()}
            inspected=fork_compress_boundary(config,None,parent,identity,Toy())
            self.assertEqual(inspected['compress_origin']['maze_anchor_sha256'],anchor['sha256'])
            self.assertEqual(before,{str(p.relative_to(parent)):file_sha(p) for p in parent.rglob('*') if p.is_file()})
            branches=[]
            for enabled in (True,False):
                branch_config=replace(config,ewc_enabled=enabled)
                branch_identity=dict(identity,config=branch_config.resolved())
                out=base/('on' if enabled else 'off')
                branch=fork_compress_boundary(branch_config,out,parent,branch_identity,Toy())
                branches.append(branch)
                self.assertIsNone(branch['optimizer']);self.assertIsNone(branch['collectors'])
                self.assertEqual(branch['windows'],0)
                self.assertEqual(tensor_hash(branch['student']),tensor_hash(original['student']))
                self.assertEqual(fisher_hash(branch['fisher']),fisher_hash(fisher))
                self.assertTrue(torch.equal(branch['rng']['torch'],original['rng']['torch']))
                self.assertTrue(torch.equal(branch['rng']['streams']['action'],original['rng']['streams']['action']))
                for name in ('python','numpy'):self.assertEqual(branch['rng'][name],original['rng'][name])
                for name in ('prefix','mini','fisher'):self.assertEqual(branch['rng']['streams'][name],original['rng']['streams'][name])
                self.assertEqual(load_boundary(out,branch_identity)['compress_origin'],branch['compress_origin'])
                self.assertEqual(file_sha(out/'snapshots/maze_before_compress.pt'),anchor['sha256'])
            self.assertEqual(branches[0]['compress_origin']['boundary_sha256'],branches[1]['compress_origin']['boundary_sha256'])
            self.assertEqual(before,{str(p.relative_to(parent)):file_sha(p) for p in parent.rglob('*') if p.is_file()})
            with self.assertRaisesRegex(ValueError,'resume config'):
                load_boundary(base/'off',identity)
            for changed in (replace(config,lr=config.lr*2),replace(config,ewc_lambda=500.)):
                with self.assertRaisesRegex(ValueError,'resume config'):
                    fork_compress_boundary(changed,base/'invalid',parent,dict(identity,config=changed.resolved()),Toy())
                self.assertFalse((base/'invalid').exists())
            with self.assertRaisesRegex(ValueError,'resume config'):
                fork_compress_boundary(config,base/'bad_source',parent,dict(identity,source={'test':'changed'}),Toy())
            with self.assertRaises(FileExistsError):
                fork_compress_boundary(config,base/'on',parent,identity,Toy())
            for mutation in ({'phase':'fisher'},{'windows':1},{'optimizer':{'state':{}}},{'collectors':[]}):
                bad=deepcopy(original);bad.update(mutation);save_boundary(parent,bad)
                with self.assertRaisesRegex(ValueError,'F-completed'):
                    fork_compress_boundary(config,base/'bad_phase',parent,identity,Toy())
                self.assertFalse((base/'bad_phase').exists())

    def test_tensor_only_rng_checkpoint(self):
        streams={'action':torch.Generator().manual_seed(1),'prefix':np.random.default_rng(2)}
        state=capture_rng(streams)
        (ROOT/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as directory:
            file=Path(directory)/'rng.pt';atomic_torch(file,state)
            loaded=torch.load(file,weights_only=True)
        a=torch.rand(5,generator=streams['action']);b=streams['prefix'].integers(100,size=5)
        restore_rng(loaded,streams)
        self.assertTrue(torch.equal(a,torch.rand(5,generator=streams['action'])))
        np.testing.assert_array_equal(b,streams['prefix'].integers(100,size=5))

    def test_environment_restore_and_bfs_transition_table(self):
        config=Config();entries=training_entries(path(config.maze_root),2);cache=MapCache(path(config.maze_root),entries)
        images,labels,transitions,map_ids=cache.tables()
        env=MazeEnv(cache,entries[:1],0);env.reset()
        cells=[tuple(map(int,p)) for p in np.argwhere(~env.walls) if tuple(p)!=env.goal]
        for i,pos in enumerate(cells):
            for action in range(5):
                env.position=pos;env.steps=0;env.done=False
                obs,done,success=env.step(action)
                following=transitions[i,action]
                self.assertEqual(success,following==-1)
                if following>=0:np.testing.assert_array_equal(obs,images[following])
            distance=cache.maps[entries[0]['sha256']][4]
            env.position=pos;env.steps=0;env.done=False
            env.step(int(labels[i]));self.assertEqual(distance[env.position],distance[pos]-1)
        room=FourRoomsEnv(3)
        try:
            room.reset()
            for action in (0,1,2,3,4):
                saved=room.capture();expected=room.step(action)[0]
                room.restore(saved);np.testing.assert_array_equal(room.step(action)[0],expected)
        finally:room.close()

    def test_statistics_are_paired(self):
        def result(successes):
            episodes=[dict(identity=i,initial_position=[i,0],success=v) for i,v in enumerate(successes)]
            return dict(episodes=episodes,success_rate=sum(successes)/len(successes))
        comparisons={'maze':paired(result([True,True,False,False]),result([True,False,True,True])),
                     'rooms':paired(result([False,False]),result([True,True]))}
        holm(comparisons)
        self.assertEqual(comparisons['maze']['difference_pp'],25.)
        self.assertEqual(comparisons['maze']['before_only'],1)
        self.assertEqual(comparisons['maze']['after_only'],2)
        self.assertLessEqual(wilson(0,800)[0],1e-15)
        with self.assertRaises(ValueError):
            paired(result([True]),dict(episodes=[dict(identity=9,initial_position=[0,0],success=True)],success_rate=1.))

    def test_compress_microbatch_preserves_joint_updates(self):
        torch.manual_seed(23)
        initial=Toy().state_dict();results=[]
        for microbatch in (1,2,4):
            model=Toy();model.load_state_dict(initial)
            config=replace(Config(),compress_microbatch=microbatch)
            batches=[]
            for task in ('maze_medium','fourrooms'):
                images=torch.arange(48,dtype=torch.uint8).reshape(3,16,1,1,1)
                masks=torch.zeros(3,16,dtype=torch.bool);masks[0]=True;masks[2,::3]=True
                targets=torch.tensor([.1,.2,.3,.2,.2]).expand(3,16,5)
                batches.append(Window(images,masks,targets,detached(model.initial_state(16)),task))
            fisher=dict(importance={n:torch.ones_like(p)*.01 for n,p in model.named_parameters()},
                        theta_star={n:p.detach().clone()-.01 for n,p in model.named_parameters()},samples=4)
            optimizer=torch.optim.Adam(model.parameters(),lr=config.lr,eps=config.eps)
            with patch('tasks.suffix_dual_expert.compress.ewc_loss',wraps=ewc_loss) as calls:
                stats=update_pair(model,batches,optimizer,fisher,np.random.default_rng(29),config)
            self.assertEqual(calls.call_count,4);self.assertEqual(stats['adam_updates'],4)
            results.append(({n:p.detach().clone() for n,p in model.named_parameters()},stats))
        for weights,stats in results[1:]:
            for name,tensor in weights.items():
                torch.testing.assert_close(tensor,results[0][0][name],rtol=0,atol=1e-7)
            for key in ('maze_kl','fourrooms_kl','ewc','total_loss'):
                self.assertAlmostEqual(stats[key],results[0][1][key],places=6)
        with self.assertRaises(ValueError):replace(Config(),compress_microbatch=3)


if __name__=='__main__':
    unittest.main()
