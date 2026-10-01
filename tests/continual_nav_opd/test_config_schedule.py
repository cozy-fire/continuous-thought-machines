"""Stage01 strict parsing, budget invariants and current-rollout contracts."""
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tasks.continual_nav_opd.config import load_config, parse_config, resolved_dict, config_hash, budget_summary
from tasks.continual_nav_opd.schedule import expand_stages
from tasks.continual_nav_opd.contracts import SequenceBatch, CTMState, DualState, FisherState

class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.full=load_config('tasks/continual_nav_opd/configs/full.yaml')

    def test_profiles_budgets_and_stage_order(self):
        for profile,total,envs in [('full',50976384,8),('remote_config',50976384,64),('smoke',1664,2)]:
            c=load_config(f'tasks/continual_nav_opd/configs/{profile}.yaml')
            stages=expand_stages(c)
            self.assertEqual(len(stages),12)
            self.assertEqual(sum(s.env_steps for s in stages),total)
            self.assertEqual(budget_summary(c)['total'],total)
            self.assertEqual(c.training.num_envs,envs)
            self.assertEqual(c.optimization.ctm_compile, "reduce-overhead" if profile == "remote_config" else "disabled")
            self.assertEqual([(s.visit,s.task,s.phase) for s in stages],[(v,t,p) for v in range(2) for t in ('maze_medium','fourrooms') for p in ('P','C','F')])
        self.assertEqual(budget_summary(self.full),dict(P=36400000,C=14560000,F=16384,total=50976384,stage_count=12))

    def test_remote_update_budget(self):
        from math import ceil
        c=load_config('tasks/continual_nav_opd/configs/remote_config.yaml')
        self.assertEqual(c.optimization.minibatches,8)
        self.assertEqual(c.optimization.encoder_microbatch_images,200)
        self.assertEqual(c.optimization.learning_steps,50)
        updates=sum(ceil(s.env_steps/(c.training.num_envs*c.optimization.learning_steps))*
                    c.optimization.minibatches for s in expand_stages(c) if s.phase in ('P','C'))
        self.assertEqual(updates,127408)

    def test_strict_fields_types_and_protocol(self):
        for path,value in [('training.num_envs',3),('pnc.visits',0),('optimization.minibatches',3),('optimization.learning_steps',0),('schema_version',2),('sequence_protocol','other'),('distill.temperature',2),('teachers.maze_medium.type','neural'),('teachers.maze_medium.tie_break_order',[3,2,1,0]),('training.seed',True),('optimization.optimizer.lr',float('nan'))]:
            raw=deepcopy(resolved_dict(self.full)); owner=raw
            parts=path.split('.')
            for key in parts[:-1]: owner=owner[key]
            owner[parts[-1]]=value
            with self.subTest(path=path),self.assertRaises(ValueError): parse_config(raw)
        for path in ['optimization.burnin_env_obs','teachers.fourrooms.warmup_env_steps','teachers.maze_medium.checkpoint']:
            raw=deepcopy(resolved_dict(self.full)); owner=raw; parts=path.split('.')
            for key in parts[:-1]: owner=owner[key]
            owner[parts[-1]]=0
            with self.assertRaises(ValueError): parse_config(raw)
        raw=resolved_dict(self.full); del raw['pnc']['task_budgets']['maze_medium']
        with self.assertRaises(ValueError):parse_config(raw)
        raw=resolved_dict(self.full);raw['task_order']=['maze_medium','maze_medium']
        with self.assertRaises(ValueError):parse_config(raw)

    def test_roundtrip_hash_and_sequence_fields(self):
        self.assertEqual(config_hash(parse_config(resolved_dict(self.full))),config_hash(self.full))
        names={f.name for f in fields(SequenceBatch)}
        self.assertNotIn('burnin_mask',names)
        self.assertIn('initial_state',names)
        import torch
        shape=(2,1)
        initial=DualState(CTMState(torch.zeros(1,512,40),torch.zeros(1,512,40)),
                          CTMState(torch.zeros(1,512,40),torch.zeros(1,512,40)))
        batch=SequenceBatch(torch.zeros(2,1,3,84,84,dtype=torch.uint8),
                            torch.tensor([[True],[False]]),torch.tensor([[True],[False]]),
                            torch.ones(shape,dtype=torch.bool),torch.zeros(2,1,5),
                            torch.zeros(shape,dtype=torch.int64),torch.tensor([[0],[-1]]),initial,'test')
        self.assertEqual(batch.loss_mask.tolist(),[[True],[False]])
        self.assertEqual(batch.obs.shape,(2,1,3,84,84))
        self.assertEqual(batch.teacher_probs.shape,(2,1,5))
        self.assertEqual(batch.initial_state.active.pre.shape,(1,512,40))
        self.assertFalse(batch.initial_state.active.pre.requires_grad)
        self.assertEqual({f.name for f in fields(FisherState)},
                         {'importance','theta_star','completed_compressions','sample_count','stage_key'})

    def test_duplicate_yaml_and_inheritance_cycle(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'a.yaml';path.write_text('method: x\nmethod: y\n')
            with self.assertRaises(ValueError):load_config(path)
            path.write_text('extends: a.yaml\n')
            with self.assertRaises(ValueError):load_config(path)

    def test_dry_run_does_not_load_teacher_and_real_run_fails(self):
        raw=resolved_dict(self.full); raw['teachers']['fourrooms']['checkpoint']='nonexistent.pt'
        import yaml
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'full.yaml';path.write_text(yaml.safe_dump(raw))
            cmd=[sys.executable,'-m','tasks.continual_nav_opd.train','--config',str(path)]
            result=subprocess.run([*cmd,'--dry-run'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('50976384',result.stdout)
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)

if __name__=='__main__':unittest.main()
