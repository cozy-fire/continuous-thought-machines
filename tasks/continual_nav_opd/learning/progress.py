"""Exact-budget P with one fresh Adam and valid-sample-weighted stage logging."""
from dataclasses import dataclass
import time
from typing import Callable
import numpy as np
import torch
from ..config import Config, validate_config
from ..data.progress import ProgressCollector
from ..models import DualPolicy, frozen_copy
from .sequence import update_window


@dataclass
class ProgressResult:
    transitions: int
    optimizer_updates: int
    eligible_target_steps: int
    next_transition_id: int
    windows: int
    statistics: dict
    policy: DualPolicy  # Complete frozen P-end teacher, with its own old KB and both encoders.


def run_progress(envs, student: DualPolicy, expert, config: Config, steps: int,
                 action_rng: torch.Generator, minibatch_rng: np.random.Generator,
                 start_transition_id: int = 0, on_window: Callable[[dict], None] | None = None) -> ProgressResult:
    validate_config(config)
    slots = len(envs.envs)
    if steps <= 0 or steps % slots or slots != config.training.num_envs:
        raise ValueError("P budget must be positive, divisible by configured environment slots")
    if student.config != config or student._frozen:
        raise ValueError("P student must be a trainable policy with matching config")
    student.train()
    parameters = [p for p in student.parameters() if p.requires_grad]
    expected = list(student.active.parameters()) + list(student.adapter.parameters())
    if {id(p) for p in parameters} != {id(p) for p in expected} or len(parameters) != len(expected):
        raise ValueError("P optimizer must include exactly all Active and Adapter parameters")
    opt = config.optimization.optimizer
    optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=opt.betas, eps=opt.eps, weight_decay=opt.weight_decay)
    collector = ProgressCollector(envs,student,expert,action_rng)
    consumed, updates, targets, windows, empty = 0, 0, 0, 0, 0
    sums, action_counts = {}, np.zeros(5,dtype=np.int64)
    total_reward, successes, timeouts, distance_sum = 0., 0, 0, 0.
    moves, turns = 0, 0
    timing = {"student_forward_seconds":0.,"expert_forward_seconds":0.,"env_step_seconds":0.,"learner_seconds":0.,
              "learner_forward_seconds":0.,"learner_backward_seconds":0.,"learner_optimizer_seconds":0.}
    started = time.perf_counter()
    while consumed < steps:
        length = min(config.optimization.learning_steps,(steps-consumed)//slots)
        window = collector.collect(length,start_transition_id+consumed)
        # Collector data is one current rollout only; reward/terminal metadata never enter KL.
        begin = time.perf_counter()
        metrics = update_window(student,window.batch,optimizer,config,minibatch_rng)
        timing["learner_seconds"] += time.perf_counter()-begin
        collector.detach_live_state()
        transitions = length*slots
        if metrics["eligible_target_steps"] != transitions:
            raise ValueError("P cannot complete with missing real targets")
        consumed += transitions
        updates += metrics["optimizer_updates"]
        targets += metrics["eligible_target_steps"]
        empty += metrics["empty_minibatches"]
        windows += 1
        for key,value in metrics.items():
            if key.endswith('_seconds'):
                timing[key] += value
            elif key not in ("eligible_target_steps","optimizer_updates","empty_minibatches"):
                sums[key] = sums.get(key,0.) + value*transitions
        action_counts += torch.bincount(window.batch.actions.flatten(),minlength=5).numpy()
        total_reward += float(window.rewards.sum())
        successes += int(window.terminated.sum())
        timeouts += int(window.truncated.sum())
        if collector.task == 'fourrooms':
            for infos in window.info:
                moves += sum(info['agent_pos_before'] != info['agent_pos_after'] for info in infos)
                turns += sum(info['agent_dir_before'] != info['agent_dir_after'] for info in infos)
        if window.distances is not None:
            distance_sum += float(window.distances.sum())
        for key in window.timing:
            timing[key] += window.timing[key]
        if on_window is not None and (windows % config.logging.interval_windows == 0 or consumed == steps):
            elapsed = time.perf_counter()-started
            on_window({"method":config.method,"schema_version":3,"sequence_protocol":config.sequence_protocol,
                       "phase":"P","task":collector.task,"policy_type":"active",
                       "windows":windows,"transitions":consumed,"stage_env_steps":consumed,
                       "eligible_target_steps":targets,"optimizer_updates":updates,"empty_minibatches":empty,
                       "next_transition_id":start_transition_id+consumed,
                       **{key:value/targets for key,value in sums.items()},
                       "action_counts":action_counts.tolist(),"reward_sum":total_reward,
                       "successes":successes,"timeouts":timeouts,"elapsed_seconds":elapsed,
                       "displacement_rate":moves/consumed if collector.task=='fourrooms' else None,
                       "turn_rate":turns/consumed if collector.task=='fourrooms' else None,
                       "transitions_per_second":consumed/elapsed,**timing,
                       "mean_distance_to_goal":distance_sum/targets if window.distances is not None else None})
        # Drop references before allocating the next rollout/graph; there is no replay/history pool.
        del window
    if targets == 0 or targets != consumed:
        raise ValueError("P ended without complete real teaching targets")
    elapsed = time.perf_counter()-started
    statistics = {**{key:value/targets for key,value in sums.items()},**timing,
                  "elapsed_seconds":elapsed,"action_counts":action_counts.tolist(),"reward_sum":total_reward,
                  "successes":successes,"timeouts":timeouts,"empty_minibatches":empty,
                  "displacement_rate":moves/consumed if collector.task=='fourrooms' else None,
                  "turn_rate":turns/consumed if collector.task=='fourrooms' else None,
                  "mean_distance_to_goal":distance_sum/targets if collector.task=="maze_medium" else None,
                  "transitions_per_second":consumed/elapsed}
    return ProgressResult(consumed,updates,targets,start_transition_id+consumed,windows,statistics,frozen_copy(student))
