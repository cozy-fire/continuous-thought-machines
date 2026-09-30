"""E1 dense replay and E2 independent rollout equivalence, including reset boundaries."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import torch
from tasks.continual_nav_opd.models.policy import _sequence
from tasks.continual_nav_opd.models import StandalonePolicy, DualPolicy, detach_clone_state
from tasks.continual_nav_opd.config import load_config
from tasks.continual_nav_opd.timing import WindowTimer, timing_mode, current_timing_mode
from tests.continual_nav_opd.test_models import tensors
from tests.continual_nav_opd import test_compress_fisher as fixtures
from tasks.continual_nav_opd.data.compress import CompressCollector


class DenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_dense_matches_generic_logits_states_and_all_gradients(self):
        torch.manual_seed(81)
        config = load_config('tasks/continual_nav_opd/configs/smoke.yaml')
        kb = StandalonePolicy(config)
        for ready in (None, False, True):
            with self.subTest(ready=ready):
                dense = deepcopy(kb) if ready is None else DualPolicy(config, kb, kb_ready=ready)
                if ready:
                    dense.adapter.gate.data.fill_(.3)
                generic = deepcopy(dense)
                rgb = torch.randint(256, (3,2,3,84,84), dtype=torch.uint8)
                starts = torch.tensor([[True,True],[False,True],[True,False]])
                valid = torch.ones(3,2,dtype=torch.bool)
                a = _sequence(dense,rgb,detach_clone_state(dense.initial_state(2)),starts,valid)
                b = _sequence(generic,rgb,detach_clone_state(generic.initial_state(2)),starts,valid,force_generic=True)
                torch.testing.assert_close(a.logits,b.logits,atol=2e-6,rtol=1e-5)
                for x,y in zip(tensors(a.state),tensors(b.state)):
                    torch.testing.assert_close(x,y,atol=2e-6,rtol=1e-5)
                a.logits.square().sum().backward(); b.logits.square().sum().backward()
                for (name,x),(_,y) in zip(dense.named_parameters(),generic.named_parameters()):
                    self.assertEqual(x.grad is None,y.grad is None,name)
                    if x.grad is not None:
                        torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-4,msg=name)

    def test_ready_cache_setter_load_and_no_observation_scalar_read(self):
        config=load_config('tasks/continual_nav_opd/configs/smoke.yaml')
        dual=DualPolicy(config,StandalonePolicy(config))
        dual.set_kb_ready(True)
        self.assertTrue(dual._kb_ready_enabled)
        other=DualPolicy(config,StandalonePolicy(config))
        other.load_state_dict(dual.state_dict(),strict=True)
        self.assertTrue(other._kb_ready_enabled)
        rgb=torch.zeros(2,3,84,84,dtype=torch.uint8)
        state=other.initial_state(2)
        with torch.no_grad(), patch.object(torch.Tensor,'item',side_effect=AssertionError('observation scalar read')):
            other.step(rgb,state,torch.ones(2,dtype=torch.bool))

    def test_timing_scope_and_cpu_accumulation(self):
        self.assertEqual(current_timing_mode(),'events')
        with timing_mode('synchronized'):
            self.assertEqual(current_timing_mode(),'synchronized')
            timer=WindowTimer('cpu')
            with timer.measure('work'): pass
            self.assertGreaterEqual(timer.finish()['work'],0)
        self.assertEqual(current_timing_mode(),'events')
        with self.assertRaises(ValueError):
            with timing_mode('invalid'): pass

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_events_match_synchronized_numerics(self):
        results=[]
        for mode in ('events','synchronized'):
            with timing_mode(mode):
                timer=WindowTimer('cuda:0')
                with timer.measure('work'):
                    result=torch.arange(128,device='cuda').square().sum()
                timing=timer.finish()
                self.assertGreaterEqual(timing['work'],0)
                results.append(result.cpu())
        torch.testing.assert_close(*results,atol=0,rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_full_collection_window_preserves_cnn_chunk_and_state(self):
        config=load_config('tasks/continual_nav_opd/configs/smoke.yaml')
        torch.manual_seed(0)
        kb=StandalonePolicy(config).cuda()
        rgb=torch.randint(256,(50,2,3,84,84),dtype=torch.uint8,device='cuda')
        starts=torch.zeros(50,2,dtype=torch.bool,device='cuda')
        starts[0]=True; starts[17,0]=True; starts[31,1]=True
        origin=detach_clone_state(kb.initial_state(2))
        state=detach_clone_state(origin); outputs=[]
        with torch.no_grad():
            for image,reset in zip(rgb,starts):
                output,state=kb.step(image,state,reset); outputs.append(output)
            with patch.object(kb.encoder,'forward',wraps=kb.encoder.forward) as calls:
                batched=kb.sequence(rgb,origin,starts,encoder_chunk_images=2)
                self.assertEqual(calls.call_count,50)
                self.assertTrue(all(len(call.args[0])==2 for call in calls.call_args_list))
        torch.testing.assert_close(torch.stack(outputs),batched.logits,atol=3e-5,rtol=1e-4)
        for a,b in zip(tensors(state),tensors(batched.state)):
            torch.testing.assert_close(a,b,atol=3e-5,rtol=1e-4)


class CompressReplayTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.CompressFisherTests.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.CompressFisherTests.tearDownClass.__func__)
    setUp = fixtures.CompressFisherTests.setUp

    def test_invalid_teacher_stops_before_environment_action(self):
        collector=CompressCollector(self.envs,self.teacher,self.kb,self.rng,'fixture')
        rng_before=self.rng.get_state().clone()
        with patch.object(self.teacher,'step',return_value=(torch.full((2,5),float('nan')),collector.teacher_state)), \
                patch.object(self.envs,'step',side_effect=AssertionError('invalid environment action')):
            with self.assertRaisesRegex(ValueError,'finite'):
                collector.collect(1,0)
        self.assertTrue(torch.equal(rng_before,self.rng.get_state()))

    def test_invalid_student_stops_before_expert_and_environment(self):
        from tasks.continual_nav_opd.data.progress import ProgressCollector
        from tasks.continual_nav_opd.teachers import MazeTeacher
        student=DualPolicy(self.config,self.kb)
        expert=MazeTeacher()
        collector=ProgressCollector(self.envs,student,expert,self.rng)
        with patch.object(student,'step',return_value=(torch.full((2,5),float('nan')),collector.state)), \
                patch.object(expert,'predict',side_effect=AssertionError('invalid expert query')), \
                patch.object(self.envs,'step',side_effect=AssertionError('invalid environment action')):
            with self.assertRaisesRegex(ValueError,'nonfinite sampling'):
                collector.collect(1,0)
    def test_batch_replay_matches_stepwise_origin_across_resets_and_tail(self):
        reference_kb=deepcopy(self.kb)
        collector=CompressCollector(self.envs,self.teacher,self.kb,self.rng,'fixture')
        for length in (3,1):
            # Force one slot reset to exercise complete learned-state replacement.
            collector.starts[1]=True
            with patch.object(self.kb,'step',side_effect=AssertionError('per-step KB inference')):
                window=collector.collect(length,0)
            state=detach_clone_state(window.batch.initial_state)
            logits=[]
            with torch.no_grad():
                for rgb,starts in zip(window.batch.obs,window.batch.episode_start):
                    output,state=reference_kb.step(rgb,state,starts)
                    logits.append(output)
            torch.testing.assert_close(torch.stack(logits),window.student_logits,atol=3e-6,rtol=1e-5)
            for a,b in zip(tensors(state),tensors(collector.state)):
                torch.testing.assert_close(a,b,atol=3e-6,rtol=1e-5)


if __name__=='__main__': unittest.main()
