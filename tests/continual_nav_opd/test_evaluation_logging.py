"""Panel refill, state reset, worker cleanup and policy/RNG evaluation isolation."""
from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
import torch
from tasks.continual_nav_opd.config import load_config
from tasks.continual_nav_opd.contracts import ObservationPair,CTMState,DualState
from tasks.continual_nav_opd.models import StandalonePolicy,DualPolicy
from tasks.continual_nav_opd.models.ctm import Controller
from tasks.continual_nav_opd.evaluate import evaluate_policy
from tasks.continual_nav_opd.wandb_logging import EventLogger


class TinyPanelEnv:
    """Controlled episode lengths include early success, timeout, refill and tail batches."""
    def __init__(self,spec): self.seed=spec[3]; self.length=1+(self.seed%3); self.step_count=0
    def observation(self):
        return ObservationPair(np.zeros((3,84,84),dtype=np.uint8),np.zeros((7,7,3),dtype=np.uint8))
    def reset(self): return self.observation(),{}
    def step(self,action):
        self.step_count+=1; end=self.step_count==self.length; success=end and self.seed%2==0
        info={'success':success,'agent_pos_before':(0,0),'agent_pos_after':(self.step_count,0),
              'agent_dir_before':0,'agent_dir_after':1}
        return self.observation(),float(success),success,end and not success,info
    def close(self): pass


def panel_factory(spec): return TinyPanelEnv(spec)
def failing_factory(spec): raise RuntimeError('controlled worker failure')


class CountingPolicy(StandalonePolicy):
    def step(self,rgb,state,episode_start,*,task):
        self.config.ctm.ticks_by_task.for_task(task)
        pre=state.pre.clone(); pre[episode_start]=0; pre=pre+1
        logits=torch.zeros(len(rgb),5); logits[torch.arange(len(rgb)),(pre[:,0,0].long()-1)%5]=1
        return logits,CTMState(pre,pre.clone())


class CountingDual(DualPolicy):
    def step(self,rgb,state,episode_start,*,task):
        self.config.ctm.ticks_by_task.for_task(task)
        pre=state.active.pre.clone(); pre[episode_start]=0; pre=pre+1
        logits=torch.zeros(len(rgb),5); logits[torch.arange(len(rgb)),(pre[:,0,0].long()-1)%5]=1
        return logits,DualState(CTMState(pre.clone(),pre.clone()),CTMState(pre,pre.clone()))


class EvaluationLoggingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.threads=torch.get_num_threads(); torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)
    def setUp(self):
        self.config=load_config('tasks/continual_nav_opd/configs/smoke.yaml')
        self.config=replace(self.config,evaluation=replace(self.config.evaluation,validation_episodes=5))

    def test_serial_subprocess_refill_order_and_tail(self):
        for dual in (False,True):
            policy=CountingDual(self.config,StandalonePolicy(self.config),kb_ready=True) if dual else CountingPolicy(self.config)
            serial=evaluate_policy(policy,self.config,None,tasks=['fourrooms'],backend='serial',num_envs=2,factory=panel_factory)
            parallel=evaluate_policy(policy,self.config,None,tasks=['fourrooms'],backend='subprocess',num_envs=2,factory=panel_factory)
            a=serial['tasks']['fourrooms']; b=parallel['tasks']['fourrooms']
            self.assertEqual(a,b); self.assertEqual([r['panel_index'] for r in a['results']],list(range(5)))
            self.assertEqual(a['episodes'],5); self.assertEqual(a['environment_steps'],sum(1+(2000000+i)%3 for i in range(5)))
            self.assertEqual(a['displacement_rate'],1.); self.assertEqual(a['turn_rate'],1.)
            self.assertTrue(any(r['truncated'] for r in a['results']))
            expected=[sum(r['length']>i for r in a['results']) for i in range(5)]
            self.assertEqual(a['action_counts'],expected)  # Every new episode restarts its action counter.

    def test_model_mode_parameters_and_training_rng_unchanged(self):
        policy=CountingDual(self.config,StandalonePolicy(self.config),kb_ready=True).train()
        before={n:v.clone() for n,v in policy.state_dict().items()}; modes=[m.training for m in policy.modules()]
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        expected=(random.random(),np.random.random(),torch.rand(3))
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        evaluate_policy(policy,self.config,None,tasks=['fourrooms'],backend='serial',factory=panel_factory)
        actual=(random.random(),np.random.random(),torch.rand(3))
        self.assertEqual(expected[:2],actual[:2]); torch.testing.assert_close(expected[2],actual[2],atol=0,rtol=0)
        self.assertEqual(modes,[m.training for m in policy.modules()])
        for n,v in policy.state_dict().items(): torch.testing.assert_close(v,before[n],atol=0,rtol=0)

    def test_cross_task_shared_kb_selects_evaluation_budget(self):
        policy=StandalonePolicy(self.config)
        batches=[]; counts={'maze_medium':0,'fourrooms':0}; context=[None]
        original_step,original_tick=StandalonePolicy.step,Controller.tick
        def step(instance,rgb,state,episode_start,*,task):
            context[0]=task; batches.append(task)
            return original_step(instance,rgb,state,episode_start,task=task)
        def tick(instance,*args,**kwargs):
            counts[context[0]]+=1
            return original_tick(instance,*args,**kwargs)
        def small_panels(config,manifest,task,split,map_cache):
            return [(task,split,None,i) for i in range(3)]
        before={n:v.clone() for n,v in policy.state_dict().items()}
        with patch('tasks.continual_nav_opd.evaluate.panels',side_effect=small_panels), \
             patch('tasks.continual_nav.envs.evaluation.shortest_path',return_value=1), \
             patch.object(StandalonePolicy,'step',step),patch.object(Controller,'tick',tick):
            report=evaluate_policy(policy,self.config,None,backend='serial',num_envs=2,factory=panel_factory)
        for task in self.config.task_order:
            ticks=self.config.ctm.ticks_by_task.for_task(task)
            self.assertEqual(counts[task],ticks*batches.count(task))
            self.assertEqual(report['tasks'][task]['ticks'],ticks)
            self.assertEqual(report['tasks'][task]['memory_ticks'],40)
            self.assertEqual([r['panel_index'] for r in report['tasks'][task]['results']],[0,1,2])
        for name,value in policy.state_dict().items():
            torch.testing.assert_close(value,before[name],atol=0,rtol=0)

    def test_worker_failure_propagates_and_leaves_no_children(self):
        previous={p.pid for p in mp.active_children()}
        with self.assertRaisesRegex(RuntimeError,'controlled worker failure'):
            evaluate_policy(CountingPolicy(self.config),self.config,None,tasks=['fourrooms'],backend='subprocess',factory=failing_factory)
        self.assertEqual(previous,{p.pid for p in mp.active_children()})

    def test_disabled_logger_and_policy_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            logger=EventLogger(directory,self.config,0,'disabled')
            logger.emit({'family':'pnc','phase':'P','task':'fourrooms','policy_type':'active','active_source':'fourrooms',
                         'kl':.2,'eligible_target_steps':2})
            logger.close()
            root=Path(directory); self.assertFalse((root/'wandb_run.json').exists())
            event=json.loads((root/'events.jsonl').read_text())
            self.assertEqual(event['kl'],.2); self.assertEqual(event['active_source'],'fourrooms')
            self.assertEqual(event['method'],'ctm_pnc_opd'); self.assertEqual(event['schema_version'],3)
            self.assertEqual((root/'events.jsonl').read_bytes(),(root/'metrics.jsonl').read_bytes())

    def test_wandb_keeps_two_tasks_and_active_sources_separate(self):
        from types import SimpleNamespace
        logged=[]
        run=SimpleNamespace(id='test',url='https://example.invalid/test',project='test',log=logged.append,finish=Mock())
        module=SimpleNamespace(init=Mock(return_value=run))
        with tempfile.TemporaryDirectory() as directory,patch.dict('sys.modules',{'wandb':module}):
            logger=EventLogger(directory,self.config,0,'online')
            for source,task in [('maze_medium','fourrooms'),('fourrooms','fourrooms'),('maze_medium','maze_medium')]:
                logger.emit({'family':'pnc','phase':'P','task':source,'policy_type':'active','active_source':source,
                             'evaluation_task':task,'split':'validation','success_rate':.25,'kl':.125})
            logger.close()
            namespaces=[next(k for k in event if k.endswith('/success_rate')) for event in logged]
            self.assertEqual(len(set(namespaces)),3)
            self.assertFalse(any(k.endswith('/kl') for event in logged for k in event))
            self.assertEqual(module.init.call_args.kwargs['config']['sequence_protocol'],'rollout_state_v1')
            run.finish.assert_called_once_with(exit_code=0)


if __name__=='__main__': unittest.main()
