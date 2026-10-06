"""Default CLI ordering, independent retry and post-training artifact isolation."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from tasks.continual_nav_opd.checkpoint import seal_artifact
from tasks.continual_nav_opd.evaluate_stages import evaluate_stages
from tasks.continual_nav_opd import train
from tasks.continual_nav_opd.verify_smoke import file_inventory
from tasks.continual_nav_opd.runner import run
import test_checkpoint_schedule as checkpoint_tests


def fake_evaluation(policy,config,manifest,*,split,backend,num_envs,identity,map_cache):
    count=getattr(config.evaluation,split+'_episodes')
    return {**identity,'split':split,'backend':backend,'num_envs':num_envs,
            'tasks':{task:{'episodes':count,'success_rate':0.,'results':[{'success':False}]*count}
                     for task in config.task_order}}


class PostTrainingEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.threads=torch.get_num_threads(); torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)
    def setUp(self):
        # Reuse tiny verified maps and mocked learning; real runner seals the full 12-stage inventory.
        checkpoint_tests.CheckpointScheduleTests.setUp(self)
        run(self.config,0,self.root)
        self.output=self.root.with_name('evaluation')

    def test_all_stages_then_final_test_isolated_and_idempotent(self):
        before=file_inventory(self.root)
        from tasks.continual_nav_opd.data import MazeManifest
        original=MazeManifest.load
        with patch('tasks.continual_nav_opd.evaluate_stages.evaluate_policy',side_effect=fake_evaluation) as evaluator, \
             patch.object(MazeManifest,'load',side_effect=original) as manifest_loader:
            summary=evaluate_stages(self.root/'exports/stages.json',self.output)
        self.assertEqual(evaluator.call_count,13)
        self.assertEqual(manifest_loader.call_count,1)
        calls=evaluator.call_args_list
        self.assertEqual([call.kwargs['split'] for call in calls],['validation']*12+['test'])
        self.assertEqual([call.kwargs['identity']['phase'] for call in calls],['P','C','F']*4+['final'])
        self.assertEqual(len({id(call.kwargs['map_cache']) for call in calls}),1)
        self.assertEqual(summary['stage_evaluations'],12)
        self.assertEqual(len(summary['reports']),13)
        self.assertEqual(before,file_inventory(self.root))
        output_before=file_inventory(self.output)
        with patch('tasks.continual_nav_opd.evaluate_stages.evaluate_policy',side_effect=AssertionError('duplicate evaluation')):
            self.assertEqual(evaluate_stages(self.root/'exports/stages.json',self.output),summary)
        self.assertEqual(output_before,file_inventory(self.output))

    def test_failed_evaluation_retries_only_unfinished_reports(self):
        count=0
        def interrupted(*args,**kwargs):
            nonlocal count
            count+=1
            if count==3: raise RuntimeError('evaluation failed')
            return fake_evaluation(*args,**kwargs)
        with patch('tasks.continual_nav_opd.evaluate_stages.evaluate_policy',side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,'evaluation failed'):
                evaluate_stages(self.root/'exports/stages.json',self.output)
        self.assertEqual(json.loads((self.output/'status.json').read_text())['completed'],2)
        first=(self.output/'000_maze_medium_P_validation.json').read_bytes()
        with patch('tasks.continual_nav_opd.evaluate_stages.evaluate_policy',side_effect=fake_evaluation) as evaluator:
            result=evaluate_stages(self.root/'exports/stages.json',self.output)
        self.assertEqual(evaluator.call_count,11)
        self.assertEqual(first,(self.output/'000_maze_medium_P_validation.json').read_bytes())
        self.assertEqual(result['status'],'complete')
        self.assertTrue(json.loads((self.root/'checkpoints/latest.json').read_text())['complete'])

    def test_plan_report_corruption_and_output_in_run_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside the training run'):
            evaluate_stages(self.root/'exports/stages.json',self.root/'evaluation')
        with patch('tasks.continual_nav_opd.evaluate_stages.evaluate_policy',side_effect=fake_evaluation):
            evaluate_stages(self.root/'exports/stages.json',self.output)
        with self.assertRaisesRegex(ValueError,'different plan'):
            evaluate_stages(self.root/'exports/stages.json',self.output,num_envs=3)
        report=self.output/'000_maze_medium_P_validation.json'
        raw=json.loads(report.read_text()); raw['stage']='wrong'; report.write_text(json.dumps(raw)); seal_artifact(report)
        with self.assertRaisesRegex(ValueError,'report identity'):
            evaluate_stages(self.root/'exports/stages.json',self.output)


class TrainingCLIOrderTests(unittest.TestCase):
    def invoke(self,extra=(),*,finalized=True,failure=None):
        events=[]
        result={'next_index':12 if finalized else 1,'finalized':finalized,'global_env_steps':10,'optimizer_updates':1}
        def training(*args,**kwargs): events.append('training'); return result
        def evaluation(*args,**kwargs):
            self.assertEqual(events,['training'])
            events.append('evaluation')
            self.command=args[0]
            if failure: raise failure
        self.command=None
        with tempfile.TemporaryDirectory() as directory:
            self.run_root=Path(directory)/'run'
            argv=['train','--config','tasks/continual_nav_opd/configs/smoke.yaml',
                  '--seed','0','--run-dir',str(self.run_root),*extra]
            with patch('sys.argv',argv),patch('tasks.continual_nav_opd.runner.run',side_effect=training), \
                 patch('tasks.continual_nav_opd.train.subprocess.run',side_effect=evaluation),redirect_stdout(StringIO()):
                train.main()
        return events

    def test_default_launches_independent_process_only_after_finalized_training(self):
        self.assertEqual(self.invoke(),['training','evaluation'])
        self.assertEqual(self.command[1:3],['-m','tasks.continual_nav_opd.evaluate_stages'])
        self.assertIn(str(self.run_root/'exports/stages.json'),self.command)
        self.assertIn(str(self.run_root.with_name('run_evaluation')),self.command)
        self.assertEqual(self.invoke(finalized=False),['training'])
        self.assertEqual(self.invoke(['--skip-evaluation']),['training'])

    def test_resume_and_explicit_output_trigger_the_same_post_training_step(self):
        self.assertEqual(self.invoke(['--resume','--evaluation-dir','scientific-evidence/new_eval']),['training','evaluation'])
        self.assertIn(str(Path('scientific-evidence/new_eval').resolve()),self.command)

    def test_evaluation_failure_is_visible_without_rerunning_training(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke(failure=subprocess.CalledProcessError(1,['evaluate_stages']))

    def test_dry_run_describes_post_training_default_and_skip(self):
        for extra,expected in [([],True),(['--skip-evaluation'],False)]:
            output=StringIO()
            with patch('sys.argv',['train','--config','tasks/continual_nav_opd/configs/smoke.yaml','--dry-run',*extra]), \
                 patch('tasks.continual_nav_opd.train.subprocess.run',side_effect=AssertionError('dry-run evaluation')), \
                 redirect_stdout(output):
                train.main()
            report=json.loads(output.getvalue())
            self.assertEqual(report['evaluation_after_training'],expected)
            self.assertFalse(report['evaluation_in_training'])


if __name__=='__main__': unittest.main()
