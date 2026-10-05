"""Maze-only five-decision replay, one Adam step per 100 collected sequences."""
import time
import torch
import torch.nn.functional as F
from ..config import validate_config
from ..data.maze_curriculum import collect_sequences
from ..models import frozen_copy
from .progress import ProgressResult


def _norm(parameters):
    parts = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(parts).sum().sqrt()) if parts else 0.


def replay_sequences(student, batch, optimizer, *, microbatch=5):
    """Rebuild learned initial states; replay reads data and never samples/steps.

    Every group contributes a SUM. Normalize once by the actual valid decisions,
    including goal-reaching actions, before clipping and the single Adam step.
    """
    device = next(student.parameters()).device
    count = int(batch.valid.sum())
    if count < 100 or batch.valid.shape != (5,100):
        raise ValueError('invalid independent Maze sequence batch')
    parameters = [p for p in student.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    sums = torch.zeros(2,device=device)
    step_sums = torch.zeros(5,device=device)
    step_counts = batch.valid.sum(1).tolist()
    begin = time.perf_counter()
    for low in range(0,100,microbatch):
        rgb = batch.obs[:,low:low+microbatch].to(device)
        mask = batch.valid[:,low:low+microbatch].to(device)
        labels = batch.labels[:,low:low+microbatch].to(device)
        # Fresh differentiable learned initial state, never the collector copy.
        initial = student.initial_state(rgb.shape[1])
        starts = torch.zeros_like(mask)
        output = student.sequence(rgb,initial,starts,mask,task='maze_medium')
        logits = output.logits
        if not torch.isfinite(logits).all(): raise ValueError('nonfinite Maze replay logits')
        ce = F.cross_entropy(logits.flatten(0,1),labels.flatten(),reduction='none').reshape_as(mask)
        loss = ce[mask].sum()
        if not torch.isfinite(loss): raise ValueError('nonfinite Maze loss')
        loss.backward()
        with torch.no_grad():
            entropy = -(logits.softmax(-1)*logits.log_softmax(-1)).sum(-1)
            sums += torch.stack((ce[mask].sum(),entropy[mask].sum()))
            step_sums += (ce*mask).sum(1)
        del output,initial,logits,loss,rgb
    for p in parameters:
        if p.grad is not None:
            p.grad.div_(count)
            if not torch.isfinite(p.grad).all(): raise ValueError('nonfinite Maze gradient')
    grad_norm = _norm(parameters)
    initial_grad = _norm([p for name,p in student.active.controller.named_parameters() if 'start' in name])
    visual_grad = _norm(student.active.encoder.parameters())
    torch.nn.utils.clip_grad_norm_(parameters,student.config.optimization.max_grad_norm,error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError('nonfinite Maze weights after Adam')
    values = (sums/count).tolist()
    per_step = step_sums.cpu().tolist()
    return dict(kl=values[0],total_loss=values[0],student_entropy=values[1],
                grad_norm=grad_norm,initial_window_grad_norm=initial_grad,visual_grad_norm=visual_grad,
                step_kl=[x/n if n else None for x,n in zip(per_step,step_counts)],
                step_valid_decisions=step_counts,eligible_target_steps=count,optimizer_updates=1,
                replay_seconds=time.perf_counter()-begin)


def run_maze_progress(table, student, config, action_rng, start_rng, start_transition_id=0,
                      on_window=None, on_pool=None, resume=None):
    validate_config(config)
    if student.config != config or student._frozen:
        raise ValueError('Maze P needs a trainable policy with matching configuration')
    student.train()
    parameters = [p for p in student.parameters() if p.requires_grad]
    expected = list(student.active.parameters())+list(student.adapter.parameters())
    if {id(p) for p in parameters} != {id(p) for p in expected}:
        raise ValueError('Maze P must train exactly Active and Adapter')
    opt = config.optimization.optimizer
    optimizer = torch.optim.Adam(parameters,lr=opt.lr,betas=opt.betas,eps=opt.eps,weight_decay=opt.weight_decay)
    if resume is not None: optimizer.load_state_dict(resume['optimizer'])
    consumed = resume['transitions'] if resume else 0
    updates = resume['updates'] if resume else 0
    sums = dict(resume['sums']) if resume else {}
    timing = dict(resume['timing']) if resume else {'collect_seconds':0.,'replay_seconds':0.}
    next_pool = resume['next_pool'] if resume else 0
    m = config.maze_progress
    if resume and not 0 <= next_pool <= len(m.pool_sizes): raise ValueError('invalid pool resume index')
    if m.pool_sizes[-1] > len(table.entries): raise ValueError('curriculum exceeds the training manifest')
    for pool_index in range(next_pool,len(m.pool_sizes)):
        pool_maps,budget = m.pool_sizes[pool_index],m.pool_updates[pool_index]
        for local_update in range(1,budget+1):
            batch = collect_sequences(table,student,pool_maps,action_rng,start_rng)
            metrics = replay_sequences(student,batch,optimizer,microbatch=m.microbatch_sequences)
            valid = metrics['eligible_target_steps']
            consumed += valid
            updates += 1
            for key in ('kl','total_loss','student_entropy'):
                sums[key] = sums.get(key,0.)+metrics[key]*valid
            timing['collect_seconds'] += batch.collect_seconds
            timing['replay_seconds'] += metrics['replay_seconds']
            if on_window and (local_update==1 or local_update % config.logging.interval_windows==0 or local_update==budget):
                on_window({**metrics,'policy_type':'active','pool_maps':pool_maps,'pool_index':pool_index,
                    'pool_optimizer_updates':local_update,'pool_update_budget':budget,'optimizer_updates':updates,
                    'stage_env_steps':consumed,'transitions':consumed,'eligible_target_steps':consumed,'update_valid_decisions':valid,
                    'next_transition_id':start_transition_id+consumed,'collect_seconds':batch.collect_seconds,
                    'replay_seconds':metrics['replay_seconds'],'state_table_bytes':table.table_bytes,
                    'sampled_action_counts':torch.bincount(batch.actions[batch.valid],minlength=5).tolist(),
                    'sequence_goal_reaches':int(batch.terminated.sum())})
            del batch
        progress = dict(next_pool=pool_index+1,transitions=consumed,updates=updates,sums=sums.copy(),timing=timing.copy(),
                        dual={n:p.detach().cpu().clone() for n,p in student.state_dict().items()},
                        optimizer=optimizer.state_dict())
        if on_pool: on_pool(progress)
    statistics = {**{k:v/consumed for k,v in sums.items()},**timing,'pool_sizes':list(m.pool_sizes),
                  'pool_updates':list(m.pool_updates),'valid_decisions':consumed,'optimizer_updates':updates}
    return ProgressResult(consumed,updates,consumed,start_transition_id+consumed,updates,statistics,frozen_copy(student))
