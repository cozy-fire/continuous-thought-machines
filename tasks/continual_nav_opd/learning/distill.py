"""Teacher-driven complete-KB distillation with one fresh Adam per C stage."""
from dataclasses import dataclass
import time
import numpy as np
import torch
from ..config import validate_config
from ..contracts import FisherState
from ..models import frozen_copy
from ..data.compress import CompressCollector
from .sequence import update_window
from .behavior import IntervalBehavior
from .fisher import validate_fisher, ewc_loss, policy_hash


@dataclass
class CompressResult:
    transitions: int
    optimizer_updates: int
    next_transition_id: int
    teacher_snapshot_id: str
    kb_ready: bool
    statistics: dict


def run_compress_stage(envs,dual_teacher,kb,fisher,config,steps,action_rng,minibatch_rng,start_transition_id=0,on_window=None):
    validate_config(config)
    slots=len(envs.envs)
    if steps<=0 or steps%slots or slots!=config.training.num_envs: raise ValueError('invalid C budget')
    if kb.config!=config or dual_teacher.config!=config: raise ValueError('C config mismatch')
    if fisher is not None:
        validate_fisher(kb,fisher)
        device=next(kb.parameters()).device
        # Immutable EWC constants are transferred ONCE, not once per parameter/minibatch.
        fisher=FisherState({n:v.to(device) for n,v in fisher.importance.items()},
                          {n:v.to(device) for n,v in fisher.theta_star.items()},
                          fisher.completed_compressions,fisher.sample_count,fisher.stage_key)
    # A new deep copy owns old KB/Active/encoders; no live KB storage is referenced.
    teacher=frozen_copy(dual_teacher); identity=policy_hash(teacher)
    kb.requires_grad_(True).train()
    opt=config.optimization.optimizer
    optimizer=torch.optim.Adam(kb.parameters(),lr=opt.lr,betas=opt.betas,eps=opt.eps,weight_decay=opt.weight_decay)
    collector=CompressCollector(envs,teacher,kb,action_rng,identity)
    behavior=IntervalBehavior(slots)
    consumed=updates=windows=0; sums={}; counts=np.zeros(5,dtype=np.int64)
    reward=0.; successes=timeouts=moves=turns=0; started=time.perf_counter()
    while consumed<steps:
        window=collector.collect(min(config.optimization.learning_steps,(steps-consumed)//slots),start_transition_id+consumed)
        behavior.add(window)
        metrics=update_window(kb,window.batch,optimizer,config,minibatch_rng,lambda:ewc_loss(kb,fisher,config.ewc.lambda_))
        collector.detach_live_state()
        size=window.batch.valid_mask.numel()
        if metrics['eligible_target_steps']!=size: raise ValueError('missing C targets')
        consumed+=size; updates+=metrics['optimizer_updates']; windows+=1
        for key,value in metrics.items():
            if key not in ('optimizer_updates','eligible_target_steps','empty_minibatches'):
                sums[key]=sums.get(key,0.)+value*(1 if key.endswith('_seconds') else size)
        for key,value in window.timing.items(): sums[key]=sums.get(key,0.)+value
        counts+=torch.bincount(window.batch.actions.flatten(),minlength=5).numpy()
        reward+=float(window.rewards.sum()); successes+=int(window.terminated.sum()); timeouts+=int(window.truncated.sum())
        if collector.task=='fourrooms':
            for infos in window.info:
                moves+=sum(info['agent_pos_before']!=info['agent_pos_after'] for info in infos)
                turns+=sum(info['agent_dir_before']!=info['agent_dir_after'] for info in infos)
        if on_window and (windows==1 or windows%config.logging.interval_windows==0 or consumed==steps):
            on_window({'phase':'C','policy_type':'kb','task':collector.task,'stage_env_steps':consumed,'transitions':consumed,
                       'ticks':collector.ticks,'memory_ticks':config.ctm.memory_length,
                       'optimizer_updates':updates,'eligible_target_steps':consumed,'windows':windows,
                       **{k:v if k.endswith('_seconds') else v/consumed for k,v in sums.items()},
                       'action_counts':counts.tolist(),'reward_sum':reward,'successes':successes,'timeouts':timeouts,
                       **behavior.flush(),
                       'displacement_rate':moves/consumed if collector.task=='fourrooms' else None,
                       'turn_rate':turns/consumed if collector.task=='fourrooms' else None,
                       'elapsed_seconds':time.perf_counter()-started})
        del window
    if identity!=policy_hash(teacher): raise RuntimeError('C modified frozen teacher')
    stats={k:v if k.endswith('_seconds') else v/consumed for k,v in sums.items()}
    stats.update(elapsed_seconds=time.perf_counter()-started,action_counts=counts.tolist(),reward_sum=reward,successes=successes,timeouts=timeouts)
    stats.update(ticks=collector.ticks,memory_ticks=config.ctm.memory_length)
    stats.update(displacement_rate=moves/consumed if collector.task=='fourrooms' else None,turn_rate=turns/consumed if collector.task=='fourrooms' else None)
    return CompressResult(consumed,updates,start_transition_id+consumed,identity,True,stats)
