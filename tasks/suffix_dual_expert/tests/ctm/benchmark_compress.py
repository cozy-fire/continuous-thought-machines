"""Full paired 50-step student replay benchmark, without online data collection."""
import argparse
from dataclasses import replace
import gc
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import torch
from ..common.test_core import ForbidLegacy
from ...config import Config
from ...model import Policy,detached
from ...compress import Window,update_pair
from ...fisher import ewc_loss
from ...checkpoint import atomic_json,seed_everything,source_hashes
from ...train import adam


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    if args.repeats<1:parser.error('repeats must be positive')
    args.out.mkdir(parents=True,exist_ok=False);seed_everything(0)
    config=Config();device=config.device
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    origin=Policy().to(device)
    initial={n:v.detach().cpu().clone() for n,v in origin.state_dict().items()}
    batches=[]
    for task in ('maze_medium','fourrooms'):
        images=torch.randint(0,256,(50,16,3,84,84),dtype=torch.uint8)
        starts=torch.zeros(50,16,dtype=torch.bool);starts[0]=True;starts[17,::3]=True;starts[39,::4]=True
        targets=torch.randn(50,16,5).softmax(-1)
        batches.append(Window(images,starts,targets,detached(origin.initial_state(16)),task))
    fisher=dict(importance={n:torch.ones_like(p)*.01 for n,p in origin.named_parameters()},
                theta_star={n:p.detach().clone()-.001 for n,p in origin.named_parameters()},samples=1024)
    del origin;gc.collect();torch.cuda.empty_cache()
    record=dict(model_type='ctm',gpu=torch.cuda.get_device_name(),source=source_hashes(),slots_per_task=16,window=50,
                scope='synthetic fixed targets with episode resets; timed student replay/EWC/4 Adam; excludes experts/environment',results=[])
    reference=None
    for microbatch,recompute in ((1,True),(2,True),(4,True),(1,False),(2,False),(4,False)):
        setting=dict(compress_microbatch=microbatch,checkpoint_encoder=recompute,encoder_chunk=config.encoder_chunk)
        policy=Policy(config.encoder_chunk,recompute).to(device);policy.load_state_dict(initial)
        cfg=replace(config,compress_microbatch=microbatch,checkpoint_encoder=recompute)
        optimizer=adam(policy,cfg);captured=[];clip=torch.nn.utils.clip_grad_norm_
        def measured(parameters,*args,**kwargs):
            captured.append({n:p.grad.detach().cpu().clone() for n,p in policy.named_parameters()})
            return clip(parameters,*args,**kwargs)
        # First full window warms kernels and checks every joint update's gradient.
        with patch('torch.nn.utils.clip_grad_norm_',side_effect=measured),patch(
            'tasks.suffix_dual_expert.compress.ewc_loss',wraps=ewc_loss) as ewc:
            stats=update_pair(policy,batches,optimizer,fisher,np.random.default_rng(31),cfg)
            assert ewc.call_count==4 and stats['adam_updates']==4
        if reference is None:
            reference=dict(gradients=captured,weights={n:p.detach().cpu().clone() for n,p in policy.named_parameters()})
        else:
            for actual,expected in zip(captured,reference['gradients']):
                for name,tensor in actual.items():torch.testing.assert_close(tensor,expected[name],rtol=1e-3,atol=5e-6,msg=name)
            for name,p in policy.named_parameters():
                torch.testing.assert_close(p.detach().cpu(),reference['weights'][name],rtol=2e-5,atol=1e-6,msg=name)
        del captured
        times=[];peak=reserved=0.
        for trial in range(args.repeats):
            policy.load_state_dict(initial);optimizer.state.clear();optimizer.zero_grad(set_to_none=True)
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            begin=time.perf_counter()
            stats=update_pair(policy,batches,optimizer,fisher,np.random.default_rng(31),cfg)
            torch.cuda.synchronize();times.append(time.perf_counter()-begin)
            peak=max(peak,torch.cuda.max_memory_allocated()/1024**2);reserved=max(reserved,torch.cuda.max_memory_reserved()/1024**2)
            print('COMPRESS',setting,'trial',trial+1,'seconds',round(times[-1],3),'peak',round(peak),flush=True)
        setting.update(status='passed',seconds=times,median_seconds=statistics.median(times),peak_allocated_mib=peak,
                       peak_reserved_mib=reserved,all_four_gradients_and_adam_parity=True,ewc_applications=4)
        record['results'].append(setting);atomic_json(args.out/'benchmark.json',record)
        del optimizer,policy;gc.collect();torch.cuda.empty_cache()
    for result in record['results']:result['speedup_vs_micro1_recompute']=record['results'][0]['median_seconds']/result['median_seconds']
    record['status']='passed';atomic_json(args.out/'benchmark.json',record)


if __name__=='__main__':main()
