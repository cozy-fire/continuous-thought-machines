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
from tasks.continual_nav_opd.models import frozen_copy,load_snapshot,DualPolicy,StandalonePolicy
from tasks.continual_nav_opd.checkpoint import load_boundary,save_boundary,verify_reference,reference,atomic_json,seal_artifact
from tasks.continual_nav_opd.evaluate import load_for_evaluation
from tasks.continual_nav_opd.learning.fisher import update_online_fisher,policy_hash
from tasks.continual_nav_opd.runner import run
from tasks.continual_nav_opd.data import MazeEntry,MazeManifest
from dataclasses import replace


class Interrupted(RuntimeError): pass


def fake_progress(envs,dual,expert,config,steps,action,shuffle,start,on_window):
    with torch.no_grad(): dual.active.actor[-1].bias.add_(torch.rand(5)*.01)
    torch.rand(2,generator=action); shuffle.permutation(2)
    return SimpleNamespace(policy=frozen_copy(dual),transitions=steps,optimizer_updates=3,eligible_target_steps=steps,next_transition_id=start+steps,statistics={})


def fake_maze_progress(table,dual,config,action,shuffle,start,on_window,on_pool,resume):
    if resume:
        return SimpleNamespace(policy=frozen_copy(dual),transitions=resume['transitions'],optimizer_updates=resume['updates'],eligible_target_steps=resume['transitions'],next_transition_id=start+resume['transitions'],statistics={})
    with torch.no_grad(): dual.active.actor[-1].bias.add_(torch.rand(5)*.01)
    torch.rand(2,generator=action); shuffle.permutation(2)
    on_pool(dict(next_pool=1,transitions=200,updates=1,sums={},timing={},dual={n:p.detach().cpu().clone() for n,p in dual.state_dict().items()},
                 optimizer={'state':{0:{'step':torch.tensor(1.)}},'param_groups':[]}))
    return SimpleNamespace(policy=frozen_copy(dual),transitions=200,optimizer_updates=1,eligible_target_steps=200,next_transition_id=start+200,statistics={})


def fake_compress(envs,teacher,kb,fisher,config,steps,action,shuffle,start,on_window):
    with torch.no_grad(): kb.actor[-1].bias.add_(torch.rand(5)*.01)
    torch.rand(2,generator=action); shuffle.permutation(2)
    return SimpleNamespace(kb_ready=True,transitions=steps,optimizer_updates=2,next_transition_id=start+steps,statistics={})


def fake_fisher(envs,kb,previous,config,steps,action,rng,start,key):
    current={n:torch.zeros_like(p,device='cpu') for n,p in kb.named_parameters()}
    rng.choice(steps,8,replace=False)
    state=update_online_fisher(kb,current,previous,8,key)
    return {'fisher':state,'transitions':steps,'next_transition_id':start+steps,'scored_samples':8,'selected_ids':list(range(start,start+8)),'statistics':{}}


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
        self.config=replace(self.config,environment=replace(self.config.environment,maze_root=str(maze_root)),
                            maze_progress=replace(self.config.maze_progress,pool_sizes=(1,),pool_updates=(1,)))
        stack=ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch('tasks.continual_nav_opd.runner.build_manifest',return_value=manifest))
        for name,target in [('run_maze_progress',fake_maze_progress),('run_progress',fake_progress),('run_compress_stage',fake_compress),('run_fisher_stage',fake_fisher)]:
            stack.enter_context(patch('tasks.continual_nav_opd.runner.'+name,side_effect=target))
        stack.enter_context(patch('tasks.continual_nav_opd.evaluate.evaluate_policy',side_effect=AssertionError('training invoked evaluation')))
        stack.enter_context(patch('tasks.continual_nav_opd.runner.FourRoomsTeacher',return_value=object()))

    def load(self): return load_boundary(self.root/'checkpoints/latest.json',self.config,0)[0]

    def test_complete_schedule_and_idempotent_finalization(self):
        result=run(self.config,0,self.root)
        self.assertEqual((result['next_index'],result['global_env_steps'],result['next_transition_id'],result['optimizer_updates']),(12,1552,1552,16))
        self.assertTrue(result['finalized']); self.assertEqual(len(result['stage_snapshots']),12)
        self.assertFalse({'evaluations','visit_reports','active_snapshots'} & set(result))
        self.assertFalse((self.root/'evaluation').exists())
        events=[json.loads(s) for s in (self.root/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertFalse(any(e['event'] in ('evaluation','final_test') or e.get('split') in ('test','validation') for e in events))
        self.assertEqual(result['fisher'].completed_compressions,4)
        index=json.loads((self.root/'exports/stages.json').read_text())
        self.assertEqual([s['key'] for s in index['stages']],[s['key'] for s in result['stages']])
        for stage in index['stages']:
            policy=load_snapshot(verify_reference(self.root,stage['reference']))
            self.assertIsInstance(policy,DualPolicy if stage['phase']=='P' else StandalonePolicy)
        before=(self.root/'events.jsonl').read_bytes()
        resumed=run(self.config,0,self.root,resume=True)
        self.assertTrue(resumed['finalized'])
        self.assertEqual(before,(self.root/'events.jsonl').read_bytes())

    def test_uncommitted_p_c_f_rerun_only_current_stage(self):
        for phase,index in [('P',0),('C',1),('F',2)]:
            def stop(point,stage):
                if stage is not None and stage.phase==phase and point=='training_complete': raise Interrupted(phase)
            with self.assertRaises(Interrupted): run(self.config,0,self.root,resume=self.root.exists(),hook=stop)
            self.assertEqual(self.load()['next_index'],index)
            resumed=run(self.config,0,self.root,resume=True,max_stages=1)
            self.assertEqual(resumed['next_index'],index+1)

    def test_snapshot_failure_does_not_publish_boundary(self):
        def stop(point,stage):
            if point=='snapshot_complete': raise Interrupted('snapshot')
        with self.assertRaises(Interrupted): run(self.config,0,self.root,hook=stop)
        self.assertEqual(self.load()['next_index'],0)
        first=run(self.config,0,self.root,resume=True,max_stages=1)
        self.assertEqual(first['next_index'],1); self.assertEqual(len(first['stage_snapshots']),1)

    def test_final_export_interruption_resumes_without_repeating_f_or_loading_maps(self):
        def stop(point,stage):
            if point=='final_export_complete': raise Interrupted('export')
        with self.assertRaises(Interrupted): run(self.config,0,self.root,hook=stop)
        self.assertEqual(self.load()['next_index'],12); self.assertFalse(self.load()['finalized'])
        with patch('tasks.continual_nav_opd.runner.run_fisher_stage',side_effect=AssertionError('duplicate F')), \
             patch('tasks.continual_nav_opd.runner.MazeMapCache',side_effect=AssertionError('final export loaded maps')):
            result=run(self.config,0,self.root,resume=True)
        self.assertTrue(result['finalized'])

    def test_checkpoint_identity_missing_fields_source_and_sha_rejected(self):
        run(self.config,0,self.root,max_stages=1); payload=self.load()
        for key,value in [('artifact_type','v4_stage_boundary'),('schema_version',2),('sequence_protocol','old'),('config_hash','bad'),('teacher_identity',{'maze':'old_neural','fourrooms':'bad'})]:
            bad=payload.copy(); bad[key]=value; save_boundary(self.root,bad)
            with self.assertRaises(ValueError): self.load()
        bad=payload.copy(); del bad['rng']; save_boundary(self.root,bad)
        with self.assertRaises(ValueError): self.load()
        save_boundary(self.root,payload)
        with patch('tasks.continual_nav_opd.checkpoint.source_manifest',return_value={}):
            with self.assertRaises(ValueError): self.load()
        bad=payload.copy(); bad['stage_snapshots']={}; save_boundary(self.root,bad)
        with self.assertRaisesRegex(ValueError,'completed-stage snapshots'): self.load()
        save_boundary(self.root,payload)
        ref=payload['stage_snapshots'][next(iter(payload['stage_snapshots']))]
        path=verify_reference(self.root,ref)
        with path.open('ab') as stream: stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'SHA/size'): self.load()

    def test_original_active_old_kb_survives_later_compress_and_resume(self):
        p=run(self.config,0,self.root,max_stages=1)
        ref=p['stage_snapshots']['pnc/v0/maze_medium/P']; original=load_snapshot(verify_reference(self.root,ref))
        before=policy_hash(original)
        run(self.config,0,self.root,resume=True,max_stages=2)
        current=self.load(); restored=load_snapshot(verify_reference(self.root,current['stage_snapshots']['pnc/v0/maze_medium/P']))
        self.assertEqual(before,policy_hash(restored))
        self.assertNotEqual(restored.kb.actor[-1].bias.data_ptr(),current['kb']['actor.4.bias'].data_ptr())
        self.assertFalse(torch.equal(restored.kb.actor[-1].bias,current['kb']['actor.4.bias']))

    def test_task_ticks_change_rejects_boundary_restore(self):
        run(self.config,0,self.root,max_stages=1)
        changed=replace(self.config,ctm=replace(self.config.ctm,ticks_by_task=replace(self.config.ctm.ticks_by_task,fourrooms=3)))
        with self.assertRaisesRegex(ValueError,'config/source/seed/stages mismatch'):
            load_boundary(self.root/'checkpoints/latest.json',changed,0)

    def test_pool_committed_before_p_snapshot_and_invalid_adam_rejected(self):
        def stop(point,stage):
            if point=='pool_complete': raise Interrupted('pool')
        with self.assertRaises(Interrupted): run(self.config,0,self.root,hook=stop)
        saved=self.load()
        self.assertEqual(saved['next_index'],0)
        self.assertEqual(saved['maze_progress']['next_pool'],1)
        self.assertEqual(saved['stage_snapshots'],{})
        bad=__import__('copy').deepcopy(saved)
        bad['maze_progress']['optimizer']['state'][0]['step']=torch.tensor(2.)
        save_boundary(self.root,bad)
        with self.assertRaisesRegex(ValueError,'Adam steps'): self.load()
        save_boundary(self.root,saved)
        result=run(self.config,0,self.root,resume=True,max_stages=1)
        self.assertIsNone(result['maze_progress'])
        self.assertEqual(result['stage_counters']['pnc/v0/maze_medium/P']['optimizer_updates'],1)
        self.assertEqual(result['global_env_steps'],200)

    def test_offline_index_and_direct_snapshot_keep_historical_c_f_weights_without_resume(self):
        early=run(self.config,0,self.root,max_stages=2)
        key='pnc/v0/maze_medium/C'; path=verify_reference(self.root,early['stage_snapshots'][key])
        original_hash=policy_hash(load_snapshot(path))
        final=run(self.config,0,self.root,resume=True)
        self.assertNotEqual(original_hash,policy_hash(load_snapshot(self.root/'exports/final.pt')))
        with patch('tasks.continual_nav_opd.evaluate.load_boundary',side_effect=AssertionError('offline requires resume source')):
            identity={}
            historical,config,manifest,root=load_for_evaluation(self.root/'exports/stages.json','kb',stage_key=key,identity=identity)
            self.assertEqual(policy_hash(historical),original_hash)
            self.assertEqual(identity['stage'],key); self.assertEqual(identity['checkpoint'],early['stage_snapshots'][key])
            self.assertEqual(config,self.config); self.assertEqual(root,self.root)
            self.assertEqual(len(manifest.entries('test',panel=True)),2)
            self.assertEqual(policy_hash(load_for_evaluation(path,'kb',stage_key=key)[0]),original_hash)
            fkey='pnc/v0/maze_medium/F'
            self.assertEqual(policy_hash(load_for_evaluation(self.root/'exports/stages.json','kb',stage_key=fkey)[0]),original_hash)
        for checkpoint in (path,self.root/'exports/stages.json',self.root/'checkpoints/latest.json'):
            with self.assertRaises(ValueError): load_for_evaluation(checkpoint,'active',stage_key=key)
        with self.assertRaisesRegex(ValueError,'stage snapshot identity'):
            load_for_evaluation(path,'kb',stage_key='pnc/v1/maze_medium/C')
        with self.assertRaisesRegex(ValueError,'stage-key'):
            load_for_evaluation(self.root/'exports/stages.json','kb')
        self.assertEqual(policy_hash(load_for_evaluation(self.root/'checkpoints/latest.json','kb',stage_key=key)[0]),original_hash)

    def test_stage_metadata_and_final_index_tampering_rejected(self):
        result=run(self.config,0,self.root)
        index=self.root/'exports/stages.json'; value=json.loads(index.read_text())
        value['stages'][1]['reference']=value['stages'][-1]['reference']
        atomic_json(index,value); seal_artifact(index)
        result['references']['stage_index']=reference(self.root,index)
        result['references']['stage_index_sidecar']=reference(self.root,str(index)+'.sha256.json')
        save_boundary(self.root,result)
        with self.assertRaisesRegex(ValueError,'final stage index'): self.load()
        with self.assertRaisesRegex(ValueError,'stage snapshot identity'):
            load_for_evaluation(index,'kb',stage_key=value['stages'][1]['key'])


if __name__=='__main__': unittest.main()
