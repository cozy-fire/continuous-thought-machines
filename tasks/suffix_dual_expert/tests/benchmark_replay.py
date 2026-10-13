"""Bounded CUDA replay comparison; does not collect or run a formal curriculum."""
import argparse
from dataclasses import replace
import gc
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .test_core import ForbidLegacy  # Installs the independent-runtime import guard.
from ..config import Config,path
from ..envs import MapCache,training_entries
from ..model import Policy
from ..suffix import replay
from ..checkpoint import seed_everything,atomic_json,source_hashes
from ..train import adam


def stepwise(policy,images,labels,ids,mask):
    length,batch=ids.shape;state=policy.initial_state(batch)
    starts=torch.ones(batch,dtype=torch.bool,device=ids.device);loss=torch.zeros((),device=ids.device)
    for t in range(length):
        logits,state=policy.step(images[ids[t]],state,starts,task='maze_medium')
        starts=torch.zeros_like(starts)
        if t>=length-5:
            loss=loss+(F.cross_entropy(logits,labels[ids[t]],reduction='none')*mask[t-(length-5)]).sum()
    return loss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    if args.repeats<1:parser.error('repeats must be positive')
    args.out.mkdir(parents=True,exist_ok=False)
    seed_everything(0)
    config=Config()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    device=config.device
    cache=MapCache(path(config.maze_root),training_entries(path(config.maze_root),320))
    arrays=cache.tables();tables=tuple(torch.from_numpy(v).to(device) for v in arrays[:3])
    distance=np.array([cache.maps[e['sha256']][4][tuple(p)] for e in cache.entries
                      for p in np.argwhere(~cache.maps[e['sha256']][1]) if tuple(p)!=cache.maps[e['sha256']][3]])
    current=torch.as_tensor(np.flatnonzero(distance>=30)[:100],device=device)
    assert len(current)==100
    positions=[]
    for _ in range(30):
        positions.append(current.clone());current=tables[2][current,tables[1][current]]
    ids=torch.stack(positions);mask=torch.ones(5,100,dtype=torch.bool,device=device)
    mask[3:,::7]=False
    del cache,arrays,distance
    reference=Policy().to(device);initial={n:v.detach().cpu().clone() for n,v in reference.state_dict().items()}
    del reference;gc.collect();torch.cuda.empty_cache()
    valid=int(mask.sum());baseline=None;results=[]
    record=dict(gpu=torch.cuda.get_device_name(),total_mib=torch.cuda.get_device_properties(device).total_memory/1024**2,
                source=source_hashes(),length=30,sequences=100,training_maps=320,valid_suffix_decisions=valid,
                scope='fixed BFS reference trajectories; timed full forward/backward, verified clip/one Adam; not formal collection',results=results)
    variants=[('stepwise',10,True),('batched',10,True),('batched',2,False),('batched',5,False),
              ('batched',10,False),('batched',20,True),('batched',25,True),('batched',20,False),('batched',25,False)]
    for method,microbatch,recompute in variants:
        setting=dict(method=method,microbatch=microbatch,checkpoint_encoder=recompute,encoder_chunk=config.encoder_chunk)
        # Bound non-recomputed growth before attempting larger batches. A skipped
        # setting is not an OOM result and is not recommended as measured.
        previous=[r for r in results if r.get('status')=='passed' and not r['checkpoint_encoder']]
        if not recompute and previous:
            last=previous[-1]
            fixed=last['resident_mib']
            estimate=fixed+(last['peak_allocated_mib']-fixed)*microbatch/last['microbatch']
            free=torch.cuda.mem_get_info(device)[0]/1024**2
            if estimate-fixed>free-768:
                setting.update(status='skipped_memory_estimate',estimated_peak_mib=estimate,free_mib=free)
                results.append(setting);atomic_json(args.out/'benchmark.json',record);print(setting,flush=True);continue
        policy=Policy(config.encoder_chunk,recompute).to(device)
        policy.load_state_dict(initial)
        function=stepwise if method=='stepwise' else replay
        # Warm kernels at the actual execution shapes; retain no warmup gradients.
        (function(policy,tables[0],tables[1],ids[:,:microbatch],mask[:,:microbatch])/valid).backward()
        policy.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        optimizer=adam(policy,replace(config,suffix_microbatch=microbatch,checkpoint_encoder=recompute))
        times=[];optimizer_times=[];peak=reserved=gradient_error=adam_error=0.;resident=torch.cuda.memory_allocated()/1024**2
        for trial in range(args.repeats):
            policy.load_state_dict(initial);optimizer.state.clear();optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();loss_value=0.
            for first in range(0,100,microbatch):
                loss=function(policy,tables[0],tables[1],ids[:,first:first+microbatch],mask[:,first:first+microbatch])/valid
                loss.backward();loss_value+=float(loss.detach())
            torch.cuda.synchronize();gradient_seconds=time.perf_counter()-begin
            gradients={n:p.grad.detach().cpu().clone() for n,p in policy.named_parameters()}
            if baseline is not None:
                for name,value in gradients.items():
                    gradient_error=max(gradient_error,float((value-baseline['gradients'][name]).abs().max()))
                    torch.testing.assert_close(value,baseline['gradients'][name],rtol=5e-4,atol=3e-6,msg=name)
                assert abs(loss_value-baseline['loss'])<1e-5
            torch.cuda.synchronize();optimizer_begin=time.perf_counter()
            norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),config.clip,error_if_nonfinite=True);optimizer.step()
            torch.cuda.synchronize();optimizer_times.append(time.perf_counter()-optimizer_begin)
            times.append(gradient_seconds)  # Excludes host-side gradient parity checks.
            if baseline is None:
                baseline=dict(gradients=gradients,loss=loss_value,weights={n:p.detach().cpu().clone() for n,p in policy.named_parameters()})
            else:
                for name,p in policy.named_parameters():
                    actual=p.detach().cpu();error=float((actual-baseline['weights'][name]).abs().max())
                    adam_error=max(adam_error,error)
                    torch.testing.assert_close(actual,baseline['weights'][name],rtol=1e-5,atol=5e-6,msg=f'{name}; max_abs={error}')
            peak=max(peak,torch.cuda.max_memory_allocated()/1024**2)
            reserved=max(reserved,torch.cuda.max_memory_reserved()/1024**2)
            print('REPLAY',setting,'trial',trial+1,'seconds',round(times[-1],3),'peak',round(peak),flush=True)
        setting.update(status='passed',gradient_seconds=times,median_gradient_seconds=statistics.median(times),
                       clip_adam_seconds=optimizer_times,
                       peak_allocated_mib=peak,peak_reserved_mib=reserved,resident_mib=resident,
                       loss=loss_value,grad_norm=float(norm),gradient_and_adam_parity=True)
        setting.update(max_abs_gradient_difference=gradient_error,max_abs_adam_parameter_difference=adam_error)
        results.append(setting);atomic_json(args.out/'benchmark.json',record)
        del policy,optimizer,gradients,loss;gc.collect();torch.cuda.empty_cache()
    reference_seconds=results[0]['median_gradient_seconds']
    for result in results:
        if result.get('status')=='passed':result['speedup_vs_stepwise']=reference_seconds/result['median_gradient_seconds']
    record['status']='passed';atomic_json(args.out/'benchmark.json',record)


if __name__=='__main__':main()
