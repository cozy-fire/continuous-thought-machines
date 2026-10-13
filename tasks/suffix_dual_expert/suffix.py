"""Fixed-weight strict-prefix collection and suffix-only full BPTT."""
from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from .config import Config
from .models import VisualPolicy as Policy, select


class PrefixSamplingExhausted(RuntimeError):
    def __init__(self, candidates: int, accepted: int, required: int):
        self.candidates, self.accepted, self.required = candidates, accepted, required
        super().__init__(f'{accepted}/{required} accepted after {candidates} candidates; no Adam update')


@dataclass
class AcceptedBatch:
    ids: Tensor
    actions: Tensor
    mask: Tensor
    candidates: int
    passed: int


@torch.no_grad()
def candidates(policy: Policy, images: Tensor, labels: Tensor, transitions: Tensor,
               initial: Tensor, length: int) -> tuple[Tensor,Tensor,Tensor]:
    if length not in (5,10,15,20,25,30):
        raise ValueError('unsupported prediction length')
    prefix = length-5
    ids = initial.clone()
    state = policy.initial_state(len(ids))
    alive = torch.ones(len(ids), dtype=torch.bool, device=ids.device)
    starts = torch.ones_like(alive)
    observations, actions, masks = [], [], []
    for t in range(length):
        observations.append(ids.clone())
        logits, state = policy.step(images[ids], state, starts, task='maze_medium')
        if not torch.isfinite(logits).all():
            raise ValueError('nonfinite collection logits')
        action = logits.argmax(-1); actions.append(action.clone())
        following = transitions[ids,action]
        starts = torch.zeros_like(alive)
        if t < prefix:
            alive = alive & (action == labels[ids]) & (following != -1)
            if not alive.any():
                empty = torch.empty(length,0,dtype=torch.long,device=ids.device)
                return empty, empty.clone(), torch.empty(5,0,dtype=torch.bool,device=ids.device)
            if t == prefix-1:
                keep = alive.nonzero().flatten()
                observations = [x.index_select(0,keep) for x in observations]
                actions = [x.index_select(0,keep) for x in actions]
                state = select(state,keep)
                ids = following.index_select(0,keep)
                alive = torch.ones(len(keep),dtype=torch.bool,device=ids.device)
                starts = torch.zeros_like(alive)
            else:
                ids = torch.where(alive,following,ids)
        else:
            masks.append(alive.clone())
            alive = alive & (following != -1)
            ids = torch.where(alive,following,ids)
    return torch.stack(observations), torch.stack(actions), torch.stack(masks)


def collect(policy: Policy, tables: tuple[Tensor,Tensor,Tensor], rng: np.random.Generator,
            pool_states: int, length: int, config: Config) -> AcceptedBatch:
    images,labels,transitions = tables
    ids,actions,masks = [],[],[]
    count = accepted = passed = 0
    while accepted < config.sequences and count < config.max_candidates:
        n = min(config.candidate_batch,config.max_candidates-count)
        initial = torch.as_tensor(rng.integers(pool_states,size=n),device=images.device)
        batch,a,mask = candidates(policy,images,labels,transitions,initial,length)
        count += n; passed += batch.shape[1]
        take = min(config.sequences-accepted,batch.shape[1])
        if take:
            ids.append(batch[:,:take]); actions.append(a[:,:take]); masks.append(mask[:,:take])
            accepted += take
    if accepted != config.sequences:
        raise PrefixSamplingExhausted(count,accepted,config.sequences)
    return AcceptedBatch(torch.cat(ids,1),torch.cat(actions,1),torch.cat(masks,1),count,passed)


def replay(policy: Policy, images: Tensor, labels: Tensor, ids: Tensor, mask: Tensor) -> Tensor:
    length,batch = ids.shape
    starts = torch.zeros(length,batch,dtype=torch.bool,device=images.device)
    starts[0] = True
    # Batch the independent CNN evaluations across time. Model recursion and its
    # prefix gradient remain connected through the complete sequence.
    logits,_ = policy.sequence(images[ids],policy.initial_state(batch),starts,task='maze_medium')
    losses = F.cross_entropy(logits[-5:].flatten(0,1),labels[ids[-5:]].flatten(),reduction='none')
    return (losses.reshape(5,batch)*mask).sum()


def update(policy: Policy, optimizer: torch.optim.Optimizer, tables: tuple[Tensor,Tensor,Tensor],
           rng: np.random.Generator, pool_states: int, length: int, config: Config) -> dict:
    begin = time.perf_counter()
    batch = collect(policy,tables,rng,pool_states,length,config)
    collected = time.perf_counter()-begin
    valid = int(batch.mask.sum())
    if valid < 1:
        raise ValueError('no eligible suffix decisions')
    replay_begin = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    summed = 0.
    for first in range(0,config.sequences,config.suffix_microbatch):
        stop = first+config.suffix_microbatch
        loss = replay(policy,tables[0],tables[1],batch.ids[:,first:stop],batch.mask[:,first:stop])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite suffix loss')
        (loss/valid).backward(); summed += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),config.clip,error_if_nonfinite=True)
    optimizer.step()
    return dict(kl=summed/valid,valid_suffix_decisions=valid,candidates=batch.candidates,
                accepted=config.sequences,prefix_passed=batch.passed,prefix_pass_rate=batch.passed/batch.candidates,
                grad_norm=float(norm),collection_seconds=collected,replay_update_seconds=time.perf_counter()-replay_begin,
                update_seconds=time.perf_counter()-begin,suffix_microbatch=config.suffix_microbatch,
                encoder_chunk=config.encoder_chunk,checkpoint_encoder=config.checkpoint_encoder)
