"""Stage fault injection, strict identities/references and independent Active recovery."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import json
import hashlib
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from tasks.continual_nav_opd.config import load_config
from tasks.continual_nav_opd.models import frozen_copy,load_snapshot
from tasks.continual_nav_opd.checkpoint import load_boundary,save_boundary,verify_reference,reference,atomic_json
from tasks.continual_nav_opd.learning.fisher import update_online_fisher,policy_hash
from tasks.continual_nav_opd.runner import run
from tasks.continual_nav_opd.data import MazeEntry,MazeManifest
from dataclasses import replace


class Interrupted(RuntimeError): pass


def fake_progress(envs,dual,expert,config,steps,action,shuffle,start,on_window):
    with torch.no_grad(): dual.active.actor[-1].bias.add_(torch.rand(5)*.01)
    torch.rand(2,generator=action); shuffle.permutation(2)
    return SimpleNamespace(policy=frozen_copy(dual),transitions=steps,optimizer_updates=3,eligible_target_steps=steps,next_transition_id=start+steps,statistics={})


def fake_compress(envs,teacher,kb,fisher,config,steps,action,shuffle,start,on_window):
    with torch.no_grad(): kb.actor[-1].bias.add_(torch.rand(5)*.01)
    torch.rand(2,generator=action); shuffle.permutation(2)
    return SimpleNamespace(kb_ready=True,transitions=steps,optimizer_updates=2,next_transition_id=start+steps,statistics={})


def fake_fisher(envs,kb,previous,config,steps,action,rng,start,key):
    current={n:torch.zeros_like(p,device='cpu') for n,p in kb.named_parameters()}
    rng.choice(steps,8,replace=False)
    state=update_online_fisher(kb,current,previous,8,key)
    return {'fisher':state,'transitions':steps,'next_transition_id':start+steps,'scored_samples':8,'selected_ids':list(range(start,start+8)),'statistics':{}}


def fake_eval(policy,config,manifest,split='validation',identity=None,**kwargs):
    result={'episodes':2,'success_rate':0.,'environment_steps':4,'results':[{},{}]}
    return {**identity,'split':split,'tasks':{t:result.copy() for t in config.task_order},'elapsed_seconds':.01,'environment_steps':8,'episodes_per_second':400.}


class CheckpointScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.threads=torch.get_num_threads(); torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)
    def setUp(self):
        self.config=load_config('tasks/continual_nav_opd/configs/smoke.yaml')
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name)/'run'
        # Scheduling fault injection needs tiny VERIFIED maps, not a full dataset rescan per resume.
        maze_root=Path(self.temp.name)/'mazes'; entries=[]
        for i in range(5):
            path=maze_root/('train' if i<3 else 'test')/'0'/f'{i}.png'; path.parent.mkdir(parents=True,exist_ok=True)
            rgb=np.zeros((19,19,3),dtype=np.uint8); rgb[1,1:18]=255; rgb[1,1]=(255,0,0); rgb[1,17-i]=(0,255,0)
            Image.fromarray(rgb).save(path)
            entries.append(MazeEntry(path.relative_to(maze_root).as_posix(),hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest=MazeManifest((entries[0],),tuple(entries[1:3]),tuple(entries[3:]),tuple(entries[1:3]),tuple(entries[3:]),tuple(entries[1:3]))
        self.config=replace(self.config,environment=replace(self.config.environment,maze_root=str(maze_root)))
        stack=ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch('tasks.continual_nav_opd.runner.build_manifest',return_value=manifest))
        for name,target in [('run_progress',fake_progress),('run_compress_stage',fake_compress),('run_fisher_stage',fake_fisher),('evaluate_policy',fake_eval)]:
            stack.enter_context(patch('tasks.continual_nav_opd.runner.'+name,side_effect=target))
        stack.enter_context(patch('tasks.continual_nav_opd.runner.FourRoomsTeacher',return_value=object()))

    def load(self): return load_boundary(self.root/'checkpoints/latest.json',self.config,0)[0]

    def test_complete_schedule_and_idempotent_finalization(self):
        result=run(self.config,0,self.root)
        self.assertEqual((result['next_index'],result['global_env_steps'],result['next_transition_id'],result['optimizer_updates']),(12,1664,1664,20))
        self.assertTrue(result['finalized']); self.assertEqual(len(result['evaluations']),8); self.assertEqual(len(result['active_snapshots']),4)
        self.assertEqual(len(result['visit_reports']),2); self.assertEqual(result['fisher'].completed_compressions,4)
        with patch('tasks.continual_nav_opd.runner.evaluate_policy',side_effect=AssertionError('duplicate test')):
            resumed=run(self.config,0,self.root,resume=True)
        self.assertTrue(resumed['finalized'])

    def test_uncommitted_p_c_f_and_evaluation_rerun_only_current_stage(self):
        for phase,index in [('P',0),('C',1),('F',2)]:
            def stop(point,stage):
                if stage is not None and stage.phase==phase and point=='training_complete': raise Interrupted(phase)
            with self.assertRaises(Interrupted): run(self.config,0,self.root,resume=self.root.exists(),hook=stop)
            self.assertEqual(self.load()['next_index'],index)
            resumed=run(self.config,0,self.root,resume=True,max_stages=1)
            self.assertEqual(resumed['next_index'],index+1)

    def test_evaluation_failure_does_not_publish_boundary(self):
        def stop(point,stage):
            if point=='evaluation_complete': raise Interrupted('evaluation')
        with self.assertRaises(Interrupted): run(self.config,0,self.root,hook=stop)
        self.assertEqual(self.load()['next_index'],0)
        first=run(self.config,0,self.root,resume=True,max_stages=1)
        self.assertEqual(first['next_index'],1); self.assertEqual(len(first['evaluations']),1)

    def test_final_test_interruption_resumes_without_repeating_f(self):
        def stop(point,stage):
            if point=='final_evaluation_complete': raise Interrupted('test')
        with self.assertRaises(Interrupted): run(self.config,0,self.root,hook=stop)
        self.assertEqual(self.load()['next_index'],12); self.assertFalse(self.load()['finalized'])
        with patch('tasks.continual_nav_opd.runner.run_fisher_stage',side_effect=AssertionError('duplicate F')):
            result=run(self.config,0,self.root,resume=True)
        self.assertTrue(result['finalized'])

    def test_checkpoint_identity_missing_fields_source_and_sha_rejected(self):
        run(self.config,0,self.root,max_stages=1); payload=self.load()
        for key,value in [('schema_version',2),('sequence_protocol','old'),('config_hash','bad'),('teacher_identity',{'maze':'old_neural','fourrooms':'bad'})]:
            bad=payload.copy(); bad[key]=value; save_boundary(self.root,bad)
            with self.assertRaises(ValueError): self.load()
        bad=payload.copy(); del bad['rng']; save_boundary(self.root,bad)
        with self.assertRaises(ValueError): self.load()
        save_boundary(self.root,payload)
        with patch('tasks.continual_nav_opd.checkpoint.source_manifest',return_value={}):
            with self.assertRaises(ValueError): self.load()
        ref=payload['active_snapshots'][next(iter(payload['active_snapshots']))]
        path=verify_reference(self.root,ref)
        with path.open('ab') as stream: stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'SHA/size'): self.load()

    def test_original_active_old_kb_survives_later_compress_and_resume(self):
        p=run(self.config,0,self.root,max_stages=1)
        ref=p['active_snapshots']['pnc/v0/maze_medium/P']; original=load_snapshot(verify_reference(self.root,ref))
        before=policy_hash(original)
        run(self.config,0,self.root,resume=True,max_stages=2)
        current=self.load(); restored=load_snapshot(verify_reference(self.root,current['active_snapshots']['pnc/v0/maze_medium/P']))
        self.assertEqual(before,policy_hash(restored))
        self.assertNotEqual(restored.kb.actor[-1].bias.data_ptr(),current['kb']['actor.4.bias'].data_ptr())
        self.assertFalse(torch.equal(restored.kb.actor[-1].bias,current['kb']['actor.4.bias']))


if __name__=='__main__': unittest.main()
