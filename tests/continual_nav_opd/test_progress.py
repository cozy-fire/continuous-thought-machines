"""Current-rollout OPD: true first-step labels, student sampling and state ownership."""
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from tasks.continual_nav_opd.config import load_config
from tasks.continual_nav_opd.data import MazeEntry
from tasks.continual_nav_opd.data.progress import ProgressCollector
from tasks.continual_nav_opd.envs import MazeEnv,SyncVectorEnv
from tasks.continual_nav_opd.models import StandalonePolicy,DualPolicy,detach_clone_state
from tasks.continual_nav_opd.teachers import MazeTeacher
from tasks.continual_nav_opd.teachers.maze import MazeTarget
from tasks.continual_nav_opd.learning.progress import run_progress
from tasks.continual_nav_opd.learning.sequence import distribution_metrics,update_window


def state_tensors(state):
    return (state.kb.pre,state.kb.post,state.active.pre,state.active.post)


class ForcedStudent(DualPolicy):
    """Controlled student distribution, intentionally different from its teacher."""
    def step(self,rgb,state,episode_start,*,task):
        _,state=super().step(rgb,state,episode_start, task=task)
        logits=torch.full((len(rgb),5),-1000.,device=rgb.device)
        logits[:,4]=0.  # Student waits; BFS never waits.
        return logits,state


class RecordingTeacher(MazeTeacher):
    def __init__(self):
        self.calls=[]
    def predict(self,observations,contexts):
        self.calls.append([(c['episode_id'],c['episode_step']) for c in contexts])
        return super().predict(observations,contexts)


class TimeoutMaze(MazeEnv):
    def reset(self,**kwargs):
        obs,info=super().reset(**kwargs)
        # Controlled boundary fixture: next counted action reaches the original 300-step limit.
        if self._pending_seed is None:
            self.step_count=299
        return obs,self._info(False)


class ProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads=torch.get_num_threads()
        torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
    def setUp(self):
        torch.manual_seed(14)
        self.config=load_config('tasks/continual_nav_opd/configs/smoke.yaml')
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        path=self.root/'train/0/long.png'
        path.parent.mkdir(parents=True)
        rgb=np.zeros((19,19,3),dtype=np.uint8)
        rgb[1,1:18]=255
        rgb[1,1]=(255,0,0)
        rgb[1,17]=(0,255,0)
        Image.fromarray(rgb).save(path)
        self.entry=MazeEntry('train/0/long.png',hashlib.sha256(path.read_bytes()).hexdigest())
        self.student=DualPolicy(self.config,StandalonePolicy(self.config),kb_ready=False)
        self.envs=SyncVectorEnv([MazeEnv(self.root,(self.entry,),seed=i) for i in range(2)])
        self.addCleanup(self.envs.close)
        self.rng=torch.Generator().manual_seed(80)
        self.shuffle=np.random.default_rng(81)

    def optimizer(self,student=None):
        student=student or self.student
        opt=self.config.optimization.optimizer
        return torch.optim.Adam([p for p in student.parameters() if p.requires_grad],lr=opt.lr,
                                betas=opt.betas,eps=opt.eps,weight_decay=opt.weight_decay)

    def test_kl_direction_zeros_onehot_and_invalid_distributions(self):
        logits=torch.tensor([[1.,2.,-1.,.5,-.5]],requires_grad=True)
        onehot=torch.eye(5)[[3]]
        result=distribution_metrics(logits,onehot)
        torch.testing.assert_close(result['kl'].mean(),F.cross_entropy(logits,torch.tensor([3])))
        result['kl'].sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        p=torch.tensor([[.7,.1,.1,.05,.05]])
        measured=distribution_metrics(logits,p)['kl']
        expected=(p*(p.log()-logits.log_softmax(-1))).sum(-1)
        torch.testing.assert_close(measured,expected)
        q=logits.softmax(-1).detach()
        self.assertGreater(float((measured-(q*(q.log()-p.log())).sum(-1)).abs().detach()),.01)
        for bad in (p*2,p*float('nan'),p-1):
            with self.assertRaises(ValueError): distribution_metrics(logits,bad)

    def test_actions_from_student_and_first_step_targets(self):
        student=ForcedStudent(self.config,StandalonePolicy(self.config),kb_ready=False)
        expert=RecordingTeacher()
        collector=ProgressCollector(self.envs,student,expert,self.rng)
        first=collector.collect(2,17)
        self.assertEqual(expert.calls[0],[(0,0),(0,0)])
        self.assertTrue(first.batch.target_mask.all())
        self.assertTrue((first.batch.actions==4).all())
        self.assertTrue((first.batch.teacher_probs.argmax(-1)==3).all())
        self.assertEqual(first.batch.transition_ids.tolist(),[[17,18],[19,20]])
        self.assertEqual(first.batch.obs.shape,(2,2,3,84,84))
        self.assertFalse(first.batch.teacher_probs.requires_grad)
        self.assertFalse(hasattr(first.batch,'burnin_mask'))

    def test_actual_timeout_reset_labels_and_terminal_separation(self):
        envs=SyncVectorEnv([TimeoutMaze(self.root,(self.entry,),seed=i) for i in range(2)])
        self.addCleanup(envs.close)
        expert=RecordingTeacher()
        collector=ProgressCollector(envs,self.student,expert,self.rng)
        batch=collector.collect(3,0)
        self.assertTrue(batch.truncated.all())
        self.assertTrue(batch.batch.episode_start.all())
        self.assertTrue(batch.batch.target_mask.all())
        self.assertEqual([c[0][0] for c in expert.calls],[0,1,2])
        self.assertEqual(batch.info[-1][0]['reset_info']['episode_id'],3)

    def test_replay_matches_collection_and_live_state_survives_update(self):
        collector=ProgressCollector(self.envs,self.student,MazeTeacher(),self.rng)
        first=collector.collect(2,0)
        with torch.no_grad():
            replay=self.student.sequence(first.batch.obs,first.batch.initial_state,first.batch.episode_start, task='maze_medium')
        torch.testing.assert_close(replay.logits,first.student_logits,atol=2e-5,rtol=1e-5)
        self.assertEqual(int((replay.logits.argmax(-1)!=first.student_logits.argmax(-1)).sum()),0)
        for saved,live,played in zip(state_tensors(first.batch.initial_state),state_tensors(collector.state),state_tensors(replay.state)):
            self.assertNotEqual(saved.data_ptr(),live.data_ptr())
            self.assertFalse(saved.requires_grad)
            torch.testing.assert_close(live,played,atol=2e-5,rtol=1e-5)
        second=collector.collect(2,4)
        self.assertFalse(second.batch.episode_start.any())
        live=detach_clone_state(collector.state)
        update_window(self.student,second.batch,self.optimizer(),self.config,self.shuffle)
        collector.detach_live_state()
        for a,b in zip(state_tensors(live),state_tensors(collector.state)):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            self.assertFalse(b.requires_grad)

    def test_fixed_trajectory_kl_decreases_and_encoder_receives_gradient(self):
        window=ProgressCollector(self.envs,self.student,MazeTeacher(),self.rng).collect(1,0)
        batch=window.batch
        optimizer=self.optimizer()
        def loss():
            with torch.no_grad():
                output=self.student.sequence(batch.obs,batch.initial_state,batch.episode_start, task='maze_medium')
                return float(distribution_metrics(output.logits,batch.teacher_probs)['kl'].mean())
        initial=loss()
        old={n:t.clone() for n,t in self.student.kb.state_dict().items()}
        visual_before=next(self.student.active.encoder.parameters()).detach().clone()
        for _ in range(12):
            metrics=update_window(self.student,batch,optimizer,self.config,self.shuffle)
        self.assertLess(loss(),initial-1e-6)
        self.assertGreater(metrics['encoder_grad_norm'],0)
        self.assertFalse(torch.equal(visual_before,next(self.student.active.encoder.parameters())))
        for name,value in self.student.kb.state_dict().items():
            torch.testing.assert_close(value,old[name],atol=0,rtol=0)

    def test_padding_skip_and_weighted_minibatches(self):
        config=replace(self.config,optimization=replace(self.config.optimization,minibatches=2))
        student=DualPolicy(config,StandalonePolicy(config))
        window=ProgressCollector(self.envs,student,MazeTeacher(),self.rng).collect(2,0)
        valid=torch.tensor([[True,False],[True,False]])
        batch=replace(window.batch,valid_mask=valid,target_mask=valid)
        optimizer=self.optimizer(student)
        with torch.no_grad():
            output=student.sequence(batch.obs,batch.initial_state,batch.episode_start,batch.valid_mask, task='maze_medium')
            expected=float(distribution_metrics(output.logits[valid],batch.teacher_probs[valid])['kl'].mean())
        metrics=update_window(student,batch,optimizer,config,self.shuffle)
        self.assertEqual((metrics['optimizer_updates'],metrics['empty_minibatches'],metrics['eligible_target_steps']),(1,1,2))
        self.assertAlmostEqual(metrics['kl'],expected,places=6)
        empty=replace(batch,valid_mask=torch.zeros_like(valid),target_mask=torch.zeros_like(valid))
        with patch.object(optimizer,'step',side_effect=AssertionError('empty optimizer')):
            metrics=update_window(student,empty,optimizer,config,self.shuffle)
        self.assertEqual((metrics['optimizer_updates'],metrics['eligible_target_steps']),(0,0))

    def test_missing_real_target_fails_before_optimizer(self):
        window=ProgressCollector(self.envs,self.student,MazeTeacher(),self.rng).collect(1,0)
        bad=replace(window.batch,target_mask=torch.zeros_like(window.batch.target_mask))
        optimizer=self.optimizer()
        with patch.object(optimizer,'step',side_effect=AssertionError('invalid optimizer')):
            with self.assertRaises(ValueError): update_window(self.student,bad,optimizer,self.config,self.shuffle)
        class InvalidTeacher(MazeTeacher):
            def predict(self,obs,contexts):
                target=super().predict(obs,contexts)
                return MazeTarget(target.action,target.probabilities,np.zeros(len(obs),dtype=bool),target.distance_to_goal)
        with self.assertRaisesRegex(ValueError,'every real observation'):
            run_progress(self.envs,self.student,InvalidTeacher(),self.config,2,self.rng,self.shuffle)
        with self.assertRaises(ValueError):
            run_progress(self.envs,self.student,MazeTeacher(),self.config,0,self.rng,self.shuffle)

    def test_exact_tail_budget_and_cumulative_logging(self):
        config=replace(self.config,optimization=replace(self.config.optimization,learning_steps=2))
        student=DualPolicy(config,StandalonePolicy(config))
        events=[]
        result=run_progress(self.envs,student,MazeTeacher(),config,6,self.rng,self.shuffle,
                            start_transition_id=50,on_window=events.append)
        self.assertEqual((result.transitions,result.eligible_target_steps,result.optimizer_updates,result.windows),(6,6,2,2))
        self.assertEqual(result.next_transition_id,56)
        self.assertEqual([e['transitions'] for e in events],[4,6])
        for event in events:
            self.assertAlmostEqual(sum(event['action_fractions']),1.)
            self.assertIn('episode_mean_return',event)
            self.assertIn('episode_success_rate',event)
        self.assertEqual(sum(result.statistics['action_counts']),6)
        self.assertFalse(any(p.requires_grad for p in result.policy.parameters()))
        with self.assertRaises(ValueError): run_progress(self.envs,student,MazeTeacher(),config,3,self.rng,self.shuffle)

    def test_nonfinite_gradient_stops_before_optimizer_step(self):
        window=ProgressCollector(self.envs,self.student,MazeTeacher(),self.rng).collect(1,0)
        handle=self.student.active.actor[-1].bias.register_hook(lambda grad: grad*float('nan'))
        optimizer=self.optimizer()
        try:
            with patch.object(optimizer,'step',side_effect=AssertionError('nonfinite optimizer')):
                with self.assertRaisesRegex(ValueError,'nonfinite distillation gradient'):
                    update_window(self.student,window.batch,optimizer,self.config,self.shuffle)
        finally:
            handle.remove()

    def test_fourrooms_real_single_step_episode_labels_and_expert_state(self):
        from minigrid.core.world_object import Goal
        from tasks.continual_nav.envs.common import pixels
        from tasks.continual_nav_opd.envs import FourRoomsEnv
        from tasks.continual_nav_opd.teachers import FourRoomsTeacher
        from tasks.continual_nav_opd.config import REPO_ROOT
        class OneStepRooms(FourRoomsEnv):
            def reset(self,**kwargs):
                _,info=super().reset(**kwargs)
                native=self.native.unwrapped
                native.agent_pos,native.agent_dir=(2,2),0
                native.grid.set(3,2,Goal())
                self._obs=pixels(native.get_frame(tile_size=8,agent_pov=True))
                return self._pair(self._obs.copy()),info
        class ForwardStudent(DualPolicy):
            def step(self,rgb,state,episode_start,*,task):
                _,state=super().step(rgb,state,episode_start, task=task)
                logits=torch.full((len(rgb),5),-1000.)
                logits[:,2]=0.
                return logits,state
        envs=SyncVectorEnv([OneStepRooms(seed=i) for i in range(2)])
        self.addCleanup(envs.close)
        student=ForwardStudent(self.config,StandalonePolicy(self.config))
        expert=FourRoomsTeacher(REPO_ROOT/self.config.teachers.fourrooms.checkpoint)
        before=expert.parameter_hash()
        collector=ProgressCollector(envs,student,expert,self.rng)
        with patch.object(expert,'predict',wraps=expert.predict) as calls:
            first=collector.collect(2,0)
            second=collector.collect(1,4)
            self.assertEqual(calls.call_count,3)  # One forward per action-before batch, no extra prefix.
            self.assertTrue(all(call.args[2].all() for call in calls.call_args_list))
        self.assertTrue(first.terminated.all() and second.terminated.all())
        self.assertTrue(first.batch.target_mask.all() and second.batch.target_mask.all())
        self.assertTrue(first.batch.episode_start.all())
        self.assertEqual(second.batch.transition_ids.tolist(),[[4,5]])
        self.assertEqual(expert.parameter_hash(),before)


if __name__=='__main__':
    unittest.main()
