"""Single-task sampled score Fisher; square individual gradients before averaging."""
import time
import numpy as np
import torch
from .config import Config
from .models import VisualPolicy as Policy, detached, select
from .envs import BatchEnv
from .checkpoint import model_hash, tensor_hash


def fisher_hash(value: dict) -> str:
    return tensor_hash(value['importance'])+':'+tensor_hash(value['theta_star'])


def validate_fisher(policy: Policy, value: dict) -> None:
    parameters = dict(policy.named_parameters())
    if set(value['importance']) != set(parameters) or set(value['theta_star']) != set(parameters) or value['samples'] < 1:
        raise ValueError('Fisher must cover all student parameters')
    for name,param in parameters.items():
        for tensor in (value['importance'][name],value['theta_star'][name]):
            if tensor.shape != param.shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError('invalid Fisher tensor: '+name)
        if (value['importance'][name] < 0).any():
            raise ValueError('negative importance')


def ewc_loss(policy: Policy, fisher: dict, coefficient: float) -> torch.Tensor:
    return sum((fisher['importance'][name] * (p-fisher['theta_star'][name]).square()).sum()
               for name,p in policy.named_parameters()) * (coefficient/2)


def on_device(fisher: dict, device: str) -> dict:
    return dict(importance={n:v.to(device) for n,v in fisher['importance'].items()},
                theta_star={n:v.to(device) for n,v in fisher['theta_star'].items()},samples=fisher['samples'])


def estimate(policy: Policy, envs: BatchEnv, config: Config, action_rng: torch.Generator,
             score_rng: np.random.Generator) -> tuple[dict,dict]:
    begin = time.perf_counter()
    before = model_hash(policy)
    device = next(policy.parameters()).device
    slots = len(envs.envs)
    obs = envs.reset(); starts = np.ones(slots,dtype=bool)
    state = detached(policy.initial_state(slots))
    windows,consumed = [],0
    while consumed < config.fisher_steps:
        length = min(config.window,(config.fisher_steps-consumed)//slots)
        initial = detached(state)
        images,masks,actions = [],[],[]
        with torch.no_grad():
            for _ in range(length):
                rgb = torch.from_numpy(obs.copy()).to(device)
                mask = torch.from_numpy(starts.copy()).to(device)
                logits,state = policy.step(rgb,state,mask,task=envs.task)
                if not torch.isfinite(logits).all():
                    raise ValueError('nonfinite Fisher rollout')
                action = torch.multinomial(logits.softmax(-1).cpu(),1,generator=action_rng).squeeze(-1)
                images.append(rgb.cpu()); masks.append(mask.cpu()); actions.append(action)
                obs,starts = envs.step(action.numpy())
        windows.append((consumed,torch.stack(images),torch.stack(masks),torch.stack(actions),initial))
        state = detached(state); consumed += length*slots
    chosen = score_rng.choice(config.fisher_steps,config.fisher_samples,replace=False)
    parameters = dict(policy.named_parameters())
    importance = {n:torch.zeros_like(p,device='cpu') for n,p in parameters.items()}
    previous_mode = policy.training
    policy.eval()
    try:
        for key in chosen:
            offset,images,masks,actions,initial = next(w for w in windows if w[0] <= key < w[0]+w[1].shape[0]*slots)
            t,b = divmod(int(key)-offset,slots)
            origin = select(initial,torch.tensor([b],device=device))
            logits,_ = policy.sequence(images[:,b:b+1].to(device),origin,masks[:,b:b+1].to(device),task=envs.task)
            score = logits[t,0].log_softmax(-1)[int(actions[t,b])]
            gradients = torch.autograd.grad(score,tuple(parameters.values()),allow_unused=True)
            for (name,_),gradient in zip(parameters.items(),gradients):
                if gradient is not None:
                    squared = gradient.detach().cpu().float().square()
                    if not torch.isfinite(squared).all():
                        raise ValueError('nonfinite Fisher score gradient')
                    importance[name] += squared/config.fisher_samples
            del logits,score,gradients
    finally:
        policy.zero_grad(set_to_none=True); policy.train(previous_mode)
    if model_hash(policy) != before:
        raise RuntimeError('Fisher stage changed policy weights')
    fisher = dict(importance=importance,theta_star={n:p.detach().cpu().clone() for n,p in parameters.items()},
                  samples=config.fisher_samples)
    validate_fisher(policy,fisher)
    return fisher,dict(decisions=consumed,scores=len(chosen),optimizer_updates=0,weights_unchanged=True,
                       selected_ids=chosen.tolist(),elapsed_seconds=time.perf_counter()-begin)
