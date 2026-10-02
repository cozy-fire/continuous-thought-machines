"""Complete visual/controller/actor Online EWC and per-transition score Fisher."""
import hashlib
import time
import numpy as np
import torch
from ..contracts import FisherState, SequenceBatch, CTMState
from ..models import StandalonePolicy, detach_clone_state, select_state
from ..data.progress import synchronize


def policy_hash(policy):
    digest=hashlib.sha256()
    for name,value in sorted(policy.state_dict().items()):
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_fisher(kb,state):
    parameters=dict(kb.named_parameters())
    if not isinstance(state,FisherState) or set(state.importance)!=set(parameters) or set(state.theta_star)!=set(parameters):
        raise ValueError('Fisher must cover exactly the complete KB parameter names')
    if state.completed_compressions < 1 or state.sample_count < 1 or not state.stage_key:
        raise ValueError('invalid Fisher counters')
    for name,p in parameters.items():
        for value in (state.importance[name],state.theta_star[name]):
            if value.shape!=p.shape or value.dtype!=torch.float32 or value.device.type!='cpu' or not torch.isfinite(value).all():
                raise ValueError('invalid Fisher shape/dtype/value: '+name)
        if (state.importance[name]<0).any():
            raise ValueError('negative Fisher: '+name)


def ewc_loss(kb,state,coefficient):
    zero=next(kb.parameters()).new_zeros(())
    parts={'ewc_encoder':zero,'ewc_controller_actor':zero}
    if state is None:
        return zero,parts
    # validate_fisher is called at stage entry; transfer constants once in a stage closure.
    for name,p in kb.named_parameters():
        value=(state.importance[name].to(p.device)*(p-state.theta_star[name].to(p.device)).square()).sum()*coefficient/2
        key='ewc_encoder' if name.startswith('encoder.') else 'ewc_controller_actor'
        parts[key]=parts[key]+value
    return sum(parts.values()),parts


def estimate_fisher(kb,windows,selected_ids):
    ids=[int(i) for i in selected_ids]
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError('Fisher requires unique nonempty transition IDs')
    lookup={}
    for batch in windows:
        if batch.ticks != kb.config.ctm.ticks_by_task.for_task(batch.task):
            raise ValueError('Fisher rollout execution budget mismatch')
        for t,b in batch.valid_mask.nonzero().tolist():
            key=int(batch.transition_ids[t,b])
            if key < 0 or key in lookup: raise ValueError('invalid or duplicate Fisher transition ID')
            lookup[key]=(batch,t,b)
    if any(i not in lookup for i in ids): raise ValueError('Fisher ID is not a real transition')
    if type(kb) is not StandalonePolicy or kb._frozen: raise ValueError('F requires trainable complete KB')
    device=next(kb.parameters()).device
    parameters=dict(kb.named_parameters())
    accum={name:torch.zeros_like(p,device='cpu',dtype=torch.float32) for name,p in parameters.items()}
    before=policy_hash(kb); previous_mode=kb.training
    kb.eval()
    try:
        for key in ids:
            batch,t,b=lookup[key]
            indices=torch.tensor([b],device=device)
            origin=detach_clone_state(select_state(batch.initial_state,indices.to(batch.initial_state.pre.device)))
            origin=CTMState(origin.pre.to(device),origin.post.to(device))
            # Rebuild the full causal window for EACH sample; square BEFORE averaging.
            output=kb.sequence(batch.obs[:,b:b+1].to(device),origin,batch.episode_start[:,b:b+1].to(device),batch.valid_mask[:,b:b+1].to(device),task=batch.task)
            score=output.logits[t,0].log_softmax(-1)[int(batch.actions[t,b])]
            grads=torch.autograd.grad(score,tuple(parameters.values()),allow_unused=True)
            for (name,_),grad in zip(parameters.items(),grads):
                if grad is not None:
                    value=grad.detach().float().cpu().square()
                    if not torch.isfinite(value).all(): raise ValueError('nonfinite Fisher gradient: '+name)
                    accum[name].add_(value/len(ids))
            del output,score,grads
    finally:
        kb.zero_grad(set_to_none=True); kb.train(previous_mode)
    if policy_hash(kb)!=before: raise RuntimeError('F modified KB parameters')
    return accum


def update_online_fisher(kb,current,previous,sample_count,stage_key,decay=.3):
    if previous is not None: validate_fisher(kb,previous)
    parameters=dict(kb.named_parameters())
    if set(current)!=set(parameters): raise ValueError('incomplete current Fisher')
    if decay!=.3: raise ValueError('unsupported Online Fisher decay')
    for name,value in current.items():
        if value.shape!=parameters[name].shape or value.dtype!=torch.float32 or not torch.isfinite(value).all() or (value<0).any():
            raise ValueError('invalid current Fisher: '+name)
    state=FisherState({name:value.detach().cpu().float().clone()+(decay*previous.importance[name] if previous else 0.)
                       for name,value in current.items()},
                      {name:p.detach().cpu().float().clone() for name,p in parameters.items()},
                      (previous.completed_compressions if previous else 0)+1,sample_count,stage_key)
    validate_fisher(kb,state)
    return state


def run_fisher_stage(envs,kb,previous,config,steps,action_rng,fisher_rng,start_transition_id=0,stage_key='F'):
    slots=len(envs.envs)
    if steps<=0 or steps%slots or slots!=config.training.num_envs or config.fisher.scored_samples>steps:
        raise ValueError('invalid F budget')
    if kb.config != config: raise ValueError('F config mismatch')
    started=time.perf_counter(); obs,infos=envs.reset(); device=next(kb.parameters()).device
    tasks={info['task_key'] for info in infos}
    if len(tasks)!=1: raise ValueError('F requires one task per rollout')
    task=tasks.pop(); ticks=config.ctm.ticks_by_task.for_task(task)
    state=detach_clone_state(kb.initial_state(slots)); starts=np.ones(slots,dtype=bool)
    windows=[]; consumed=0; before=policy_hash(kb)
    while consumed<steps:
        length=min(config.optimization.learning_steps,(steps-consumed)//slots)
        initial=detach_clone_state(state); initial=CTMState(initial.pre.cpu(),initial.post.cpu())
        images,reset_masks,actions=[],[],[]
        with torch.no_grad():
            for _ in range(length):
                rgb=torch.from_numpy(obs.student_rgb.copy()); reset=torch.from_numpy(starts.copy())
                logits,state=kb.step(rgb.to(device),state,reset.to(device),task=task)
                if not torch.isfinite(logits).all(): raise ValueError('nonfinite F sampling logits')
                action=torch.multinomial(logits.softmax(-1).to(action_rng.device),1,generator=action_rng).squeeze(-1).cpu()
                transition=envs.step(action.numpy()); obs,starts=transition.next_obs,transition.next_episode_start
                images.append(rgb); reset_masks.append(reset); actions.append(action)
        valid=torch.ones(length,slots,dtype=torch.bool)
        windows.append(SequenceBatch(torch.stack(images),torch.stack(reset_masks),valid,torch.zeros_like(valid),
                       torch.zeros(length,slots,5),torch.stack(actions),
                       torch.arange(start_transition_id+consumed,start_transition_id+consumed+length*slots).reshape(length,slots),initial,before,task,ticks))
        state=detach_clone_state(state); consumed+=length*slots
    selected=fisher_rng.choice(np.arange(start_transition_id,start_transition_id+steps),config.fisher.scored_samples,replace=False)
    synchronize(device); score_start=time.perf_counter()
    current=estimate_fisher(kb,windows,selected)
    synchronize(device); score_seconds=time.perf_counter()-score_start
    result=update_online_fisher(kb,current,previous,len(selected),stage_key,config.ewc.decay)
    if before!=policy_hash(kb): raise RuntimeError('F changed KB')
    return {'fisher':result,'transitions':consumed,'optimizer_updates':0,'next_transition_id':start_transition_id+consumed,
            'scored_samples':len(selected),'selected_ids':selected.tolist(),'source_snapshot_id':before,
            'statistics':{'task':task,'ticks':ticks,'memory_ticks':config.ctm.memory_length,
                          'scoring_seconds':score_seconds,'elapsed_seconds':time.perf_counter()-started}}
