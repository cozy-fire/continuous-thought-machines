"""Acceptance/profiling tools must fail honestly and restore instrumentation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tasks.continual_nav_opd.profile_training import Timings
from tasks.continual_nav_opd.verify_smoke import file_inventory, audit_run, verify


class Stage07ToolsTests(unittest.TestCase):
    def test_inventory_tracks_bytes_and_relative_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'nested').mkdir(); (root/'nested/a.bin').write_bytes(b'abc')
            before = file_inventory(root)
            self.assertEqual(before['nested/a.bin']['size'], 3)
            (root/'nested/a.bin').write_bytes(b'abd')
            self.assertNotEqual(before, file_inventory(root))

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                verify(directory, 'cpu')

    def test_partial_boundary_rejected(self):
        with patch('tasks.continual_nav_opd.verify_smoke.load_boundary', return_value=({'finalized': False}, None)), \
             patch('tasks.continual_nav_opd.verify_smoke.expand_stages', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                audit_run(Path('.'), None, 0)

    def test_timer_restores_context_after_exception(self):
        timer = Timings('cpu')
        def fail():
            raise RuntimeError('intentional')
        with self.assertRaises(RuntimeError):
            timer.measure(fail, 'window', 'P/learning')()
        self.assertEqual(timer.context, 'initialization')
        self.assertEqual(timer.report()['P/learning/window']['calls'], 1)

    def test_instrumentation_restores_original_methods(self):
        from tasks.continual_nav_opd.models import VisionEncoder
        original = VisionEncoder.forward
        with self.assertRaises(RuntimeError):
            with Timings('cpu').instrument():
                self.assertIsNot(VisionEncoder.forward, original)
                raise RuntimeError('intentional')
        self.assertIs(VisionEncoder.forward, original)

    def test_encoder_timer_waits_for_all_parameter_gradients(self):
        from types import SimpleNamespace
        import torch
        module = torch.nn.Linear(2, 3)
        timer = Timings('cpu'); timer.context = 'P/learning'
        forward = timer.measure(type(module).forward, 'encoder_forward')
        with timer.encoder_backward(SimpleNamespace(encoder=module)):
            # Two image microbatches share parameters; one loss/gradient pass.
            a = forward(module, torch.ones(1, 2))
            b = forward(module, torch.ones(1, 2))
            (a.sum()+b.sum()).backward()
        self.assertEqual(timer.report()['P/learning/encoder_backward']['calls'], 1)
        self.assertEqual(timer.backward_pending, {})
        self.assertTrue(torch.equal(module.weight.grad, torch.full_like(module.weight, 2.)))


if __name__ == '__main__':
    unittest.main()
