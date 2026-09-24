"""Bounded local replay/fit/evaluation comparison; never changes production budgets."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from unittest.mock import patch

import numpy as np
import psutil
import torch

from .checkpoint import atomic_json
from .data.replay import TransitionStore, sample_world_batch
from .evaluate import evaluate_policy, load_for_evaluation
from .envs.evaluation import EvaluationPool
from .learning.world import fit_world_model, make_world_optimizer
from .models import SIGReg, VisionEncoder, WorldModel


class PeakRSS:
    """Sample aggregate resident bytes, including spawned evaluation workers."""
    def __enter__(self):
        self.stop = Event()
        self.peak = 0
        self.process = psutil.Process()
        self.thread = Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def _watch(self):
        while not self.stop.is_set():
            total = 0
            try:
                processes = [self.process, *self.process.children(recursive=True)]
            except psutil.Error:
                processes = [self.process]
            for process in processes:
                try:
                    total += process.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, total)
            self.stop.wait(.02)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


class TracedPool(EvaluationPool):
    """Observe parent-side actions without changing worker execution or policy inputs."""
    def __init__(self, *args, traces: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.traces, self.panels = traces, {}

    def exchange(self, command, requests):
        if command == "reset":
            for slot, spec in requests.items():
                key = repr(spec)
                self.panels[slot] = key
                self.traces[key] = []
        elif command == "step":
            for slot, action in requests.items():
                self.traces[self.panels[slot]].append(action)
        return super().exchange(command, requests)


def benchmark(args) -> dict:
    torch.set_num_threads(2)
    # This comparison process uses deterministic convolutions to separate replay
    # changes from CUDA backward nondeterminism. Production training flags are untouched.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    policy, encoder, manifest, config, _ = load_for_evaluation(args.checkpoint)
    device = torch.device(args.device)
    policy.to(device); encoder.to(device)
    config = replace(config, world=replace(config.world, batch_size=args.batch_size,
                                          updates_per_round=args.updates, log_interval_updates=1))
    fresh = TransitionStore(args.fresh)
    high = TransitionStore(args.high) if args.high else None
    torch.manual_seed(73)
    original = WorldModel(config, VisionEncoder(config)).to(device)
    original.encoder.encoder_version.fill_(fresh.encoder_version)
    original.world_model_version.fill_(fresh.world_version)
    replay, fits, evaluations = {}, {}, {}
    sampled_ids, weights = {}, {}
    for mode in ("shard", "memory"):
        fresh.release()
        if high is not None:
            high.release()
        with PeakRSS() as rss:
            start = perf_counter()
            cache_bytes = 0
            if mode == "memory":
                cache_bytes = fresh.preload() + (high.preload() if high is not None else 0)
            preload = perf_counter()-start
            rng = np.random.default_rng(91)
            samples, times = [], []
            for _ in range(args.samples):
                start = perf_counter()
                batch = sample_world_batch(fresh, high, task=fresh.task, batch_size=args.batch_size,
                                          high_fraction=config.replay.high_error_fraction, rng=rng)
                times.append(perf_counter()-start)
                samples.append(batch.transition_ids.tolist())
        sampled_ids[mode] = samples
        replay[mode] = dict(preload_seconds=preload, cache_bytes=cache_bytes,
            sampling_seconds=sum(times), median_batch_seconds=float(np.median(times)),
            process_tree_peak_rss_bytes=rss.peak)
        fresh.release()
        if high is not None:
            high.release()
        world = deepcopy(original)
        optimizer = make_world_optimizer(world, config)
        variant = replace(config, replay=replace(config.replay, fit_cache=mode))
        logs = []
        with PeakRSS() as rss:
            metrics = fit_world_model(world, SIGReg(config.sigreg), fresh, high, optimizer, variant,
                task=fresh.task, replay_rng=np.random.default_rng(91),
                sigreg_rng=torch.Generator().manual_seed(92),
                on_update=lambda _, value: logs.append(dict(value)))
        # Exclude two warm-up optimizer updates from the steady-state throughput.
        elapsed = logs[-1]["fit_seconds"]-logs[1]["fit_seconds"]
        fits[mode] = dict(metrics=metrics, steady_updates_per_second=(args.updates-2)/elapsed,
                          process_tree_peak_rss_bytes=rss.peak)
        weights[mode] = {key: value.detach().cpu().clone() for key, value in world.state_dict().items()}
        del world, optimizer
    max_error = 0.
    for key, reference in weights["shard"].items():
        actual = weights["memory"][key]
        torch.testing.assert_close(actual, reference, atol=1e-5 if device.type == "cuda" else 0.,
                                   rtol=1e-4 if device.type == "cuda" else 0.)
        if actual.is_floating_point():
            max_error = max(max_error, (actual-reference).abs().max().item())
    del original, weights
    traces = {}
    for backend, slots in (("serial", 1), ("serial", args.num_envs), ("subprocess", args.num_envs)):
        variant = replace(config, evaluation=replace(config.evaluation, backend=backend, num_envs=slots))
        key = f"{backend}_{slots}"
        traces[key] = {}
        with PeakRSS() as rss, patch("tasks.continual_nav.evaluate.EvaluationPool",
                                    partial(TracedPool, traces=traces[key])):
            report = evaluate_policy(policy, encoder, manifest, variant)
        evaluations[key] = dict(report=report, process_tree_peak_rss_bytes=rss.peak)
    baseline = evaluations["serial_1"]["report"]["tasks"]
    return dict(device=str(device), cudnn_deterministic=True,
        records=dict(fresh=len(fresh), high=len(high) if high else 0),
        batch_size=args.batch_size, updates=args.updates, sample_batches=args.samples,
        replay=replay, fits=fits, evaluations=evaluations, fit_max_abs_error=max_error,
        replay_ids_exact=sampled_ids["memory"] == sampled_ids["shard"],
        evaluation_episode_results_exact={key: item["report"]["tasks"] == baseline for key, item in evaluations.items()},
        argmax_differing_episodes={key: [panel for panel, actions in values.items()
                                      if actions != traces["serial_1"][panel]] for key, values in traces.items()},
        action_traces=traces,
        notes="RSS is sampled process-tree resident memory, not a sum of unique physical pages. "
              "Actions are recorded separately: equal episode metrics alone do not establish equal trajectories.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--high", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.updates < 3 or min(args.samples, args.num_envs, args.batch_size) < 1:
        parser.error("output must be new; updates >= 3 and all sizes positive")
    atomic_json(args.output, benchmark(args))


if __name__ == "__main__":
    main()
