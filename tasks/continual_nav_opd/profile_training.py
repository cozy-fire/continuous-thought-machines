"""Synchronized, real-teacher component timings without changing training semantics."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .checkpoint import atomic_json, source_manifest
from .timing import timing_mode
from .config import REPO_ROOT, load_config, resolved_dict
from .data import build_manifest
from .data.progress import ProgressCollector
from .data.compress import CompressCollector
from .envs import SyncVectorEnv, make_env
from .models import StandalonePolicy, DualPolicy, VisionEncoder, Controller
from .teachers import MazeTeacher, FourRoomsTeacher
from .learning.progress import run_progress
from .learning.distill import run_compress_stage
from .learning.fisher import run_fisher_stage


class Timings:
    def __init__(self, device):
        self.device = torch.device(device)
        self.context = 'initialization'
        self.samples = {}
        self.backward_pending = {}

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def measure(self, function, name, context=None):
        def measured(*args, **kwargs):
            old = self.context
            if context:
                self.context = context
            key = f'{self.context}/{name}'
            self.sync(); start = time.perf_counter()
            try:
                result = function(*args, **kwargs)
                if name == 'encoder_forward' and self.context in ('P/learning', 'C/learning') and result.requires_grad:
                    module = args[0]
                    def begin_backward(gradient):
                        self.sync()
                        self.backward_pending.setdefault(id(module), time.perf_counter())
                    result.register_hook(begin_backward)
                return result
            finally:
                self.sync()
                seconds = time.perf_counter()-start
                self.samples.setdefault(key, []).append(seconds)
                self.context = old
        return measured

    @contextmanager
    def instrument(self):
        from contextlib import ExitStack
        from .learning import progress, distill, fisher
        from .teachers import maze
        # Timers only wrap existing calls. No extra inference, labels or RNG draws occur.
        targets = [(VisionEncoder, 'forward', 'encoder_forward', None),
                   (Controller, 'tick', 'ctm_tick', None),
                   (ProgressCollector, 'collect', 'window', 'P/collection'),
                   (CompressCollector, 'collect', 'window', 'C/collection'),
                   (progress, 'update_window', 'window', 'P/learning'),
                   (distill, 'update_window', 'window', 'C/learning'),
                   (fisher, 'estimate_fisher', 'score_all', 'F/scoring'),
                   (torch.autograd, 'grad', 'per_sample_backward', None),
                   (maze, 'parse_observation', 'maze_parse', None),
                   (maze, 'shortest_path', 'maze_bfs', None)]
        with ExitStack() as stack:
            for owner, attribute, name, context in targets:
                stack.enter_context(patch.object(owner, attribute,
                    self.measure(getattr(owner, attribute), name, context)))
            yield

    @contextmanager
    def encoder_backward(self, *policies):
        handles = []
        for policy in policies:
            encoder = policy.encoder
            def finish(grads, module=encoder):
                self.sync()
                begin = self.backward_pending.pop(id(module), None)
                if begin is not None:
                    self.samples.setdefault(f'{self.context}/encoder_backward', []).append(time.perf_counter()-begin)
            # uint8 input has no gradient. Module backward hooks would finish BEFORE
            # parameter gradients; wait for ALL encoder parameter gradients instead.
            handles.append(torch.autograd.graph.register_multi_grad_hook(tuple(encoder.parameters()), finish))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def report(self):
        return {key: {'calls': len(values), 'total_seconds': sum(values),
                      'first_seconds': values[0], 'remaining_seconds': sum(values[1:]),
                      'mean_seconds': sum(values)/len(values)} for key, values in self.samples.items()}


def profile(output_dir, device='cuda:0', slots=2, microbatch=8, minibatches=1):
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(0)
    config = load_config(REPO_ROOT/'tasks/continual_nav_opd/configs/smoke.yaml')
    # Two full 50-observation windows: measured workload is explicit, never a formal run.
    steps = 100*slots
    config = replace(config, training=replace(config.training, device=device, num_envs=slots),
                     optimization=replace(config.optimization, encoder_microbatch_images=microbatch, minibatches=minibatches))
    from .config import validate_config
    validate_config(config)
    atomic_json(output/'resolved_config.json', resolved_dict(config))
    atomic_json(output/'source.json', source_manifest())
    timer = Timings(device); initialization = time.perf_counter()
    manifest = build_manifest(REPO_ROOT/config.environment.maze_root, validation_count=512,
                              validation_episodes=2, test_episodes=2, drift_episodes=2)
    if device.startswith('cuda'):
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    kb = StandalonePolicy(config).to(device)
    initialization = time.perf_counter()-initialization
    reports = []; fisher = None
    try:
        for task_index, task in enumerate(config.task_order):
            task_init = time.perf_counter()
            envs = SyncVectorEnv([make_env(config, task, 'train', manifest=manifest, seed=task_index*100+i) for i in range(slots)])
            dual = DualPolicy(config, kb, kb_ready=task_index > 0)
            expert = MazeTeacher() if task == 'maze_medium' else FourRoomsTeacher(REPO_ROOT/config.teachers.fourrooms.checkpoint, device=device)
            task_init = time.perf_counter()-task_init
            timer.samples = {}
            action = torch.Generator().manual_seed(100+task_index)
            try:
                with timing_mode('synchronized'), timer.instrument():
                    # Leaf multi-grad hooks support backward(), not autograd.grad().
                    # Remove them before F's exact per-sample autograd.grad calls.
                    with timer.encoder_backward(dual.active, kb):
                        p = run_progress(envs, dual, expert, config, steps, action, np.random.default_rng(101+task_index))
                        c = run_compress_stage(envs, p.policy, kb, fisher, config, steps, action, np.random.default_rng(102+task_index))
                    timer.context = 'F/collection'
                    f = run_fisher_stage(envs, kb, fisher, config, config.fisher.collect_steps, action,
                                         np.random.default_rng(103+task_index), stage_key=task+'/F')
                    fisher = f['fisher']
                reports.append({'task': task, 'initialization_seconds': task_init,
                                'P': {'steps': p.transitions, 'updates': p.optimizer_updates, **p.statistics},
                                'C': {'steps': c.transitions, 'updates': c.optimizer_updates, **c.statistics},
                                'F': {'steps': f['transitions'], 'samples': f['scored_samples'], **f['statistics']},
                                'components': timer.report()})
                del dual, expert, p
            finally:
                envs.close()
        report = {'status': 'passed', 'device': device, 'timing_mode': 'synchronized',
                  'gpu': torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
                  'slots': slots, 'learning_observations_per_slot': 50, 'minibatches': minibatches,
                  'microbatch_images': microbatch, 'precision': 'float32', 'initialization_seconds': initialization,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else None,
                  'results': reports,
                  'limits': 'Synchronized diagnostic timing adds overhead; nested totals must not be summed. '
                            'P/C each have two full windows; first vs remaining distinguishes startup. '
                            'F uses 8 real per-sample gradients, not the formal 1024. No RTX5090 speed claim.'}
        atomic_json(output/'report.json', report)
        print(str(output/'report.json'), flush=True)
        return report
    except Exception as exc:
        atomic_json(output/'failure.json', {'error': repr(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True); parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=2); parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--minibatches', type=int, default=1)
    args = parser.parse_args()
    profile(args.output_dir, args.device, args.num_envs, args.microbatch, args.minibatches)


if __name__ == '__main__':
    main()
