"""Bounded local verification, explicitly not the formal training budget."""
import argparse
from dataclasses import replace
import gc
import importlib.abc
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch


class ForbidLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith('tasks.continual_nav'):
            raise ImportError('legacy package forbidden: '+fullname)


sys.meta_path.insert(0,ForbidLegacy())
from tasks.suffix_dual_expert.config import Config,path,load_config
from tasks.suffix_dual_expert.envs import training_entries,MapCache
from tasks.suffix_dual_expert.models import build_policy
from tasks.suffix_dual_expert.suffix import replay
from tasks.suffix_dual_expert.checkpoint import atomic_json,seed_everything,tensor_hash,load_boundary,source_hashes
from tasks.suffix_dual_expert.train import run,preflight,adam
from tasks.suffix_dual_expert.compress import update_pair
from tasks.suffix_dual_expert.evaluate import evaluate


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--config',type=Path);args=parser.parse_args()
    output=args.out.resolve();output.mkdir(parents=True,exist_ok=False)
    config=load_config(args.config);seed_everything(0)
    if not torch.cuda.is_available():raise RuntimeError('real CUDA checks require a GPU')
    cache=MapCache(path(config.maze_root),training_entries(path(config.maze_root),320))
    arrays=cache.tables();tables=tuple(torch.from_numpy(v).cuda() for v in arrays[:3])
    starts=np.flatnonzero(np.array([cache.maps[e['sha256']][4][tuple(p)]
          for e in cache.entries for p in np.argwhere(~cache.maps[e['sha256']][1])
          if tuple(p)!=cache.maps[e['sha256']][3]])>=30)[:100]
    assert len(starts)==100
    current=torch.as_tensor(starts,device='cuda');ids=[]
    for _ in range(30):
        ids.append(current.clone());current=tables[2][current,tables[1][current]]
    ids=torch.stack(ids);mask=torch.ones(5,100,dtype=torch.bool,device='cuda')
    policy=build_policy(config).cuda().train();optimizer=adam(policy,config)
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    for first in range(0,100,config.suffix_microbatch):
        stop=first+config.suffix_microbatch
        (replay(policy,tables[0],tables[1],ids[:,first:stop],mask[:,first:stop])/500).backward()
        print('REPLAY_MICROBATCH',first//config.suffix_microbatch+1,'/',100//config.suffix_microbatch,flush=True)
    norms={name:float(torch.stack([p.grad.square().sum() for p in module.parameters() if p.grad is not None]).sum().sqrt())
           for name,module in [('encoder',policy.encoder),('controller',policy.controller),('actor',policy.actor)]}
    assert all(n>0 and np.isfinite(n) for n in norms.values())
    torch.nn.utils.clip_grad_norm_(policy.parameters(),.5,error_if_nonfinite=True);optimizer.step()
    replay_peak=torch.cuda.max_memory_reserved()/1024**2
    del policy,optimizer,tables,arrays,cache,ids,current;gc.collect();torch.cuda.empty_cache()
    atomic_json(output/'replay30.json',dict(status='passed',sequences=100,microbatch=config.suffix_microbatch,training_maps=320,
        checkpoint_encoder=config.checkpoint_encoder,encoder_chunk=config.encoder_chunk,
        gradients=norms,peak_reserved_mib=replay_peak,scope='BFS-path replay correctness and memory check; not accepted formal trajectories'))
    print('REPLAY30',replay_peak,norms,flush=True)
    tiny=replace(config,lengths=(5,),pool_sizes=(2,5),first_updates=(1,1),suffix_updates=(1,1),
                 fisher_steps=64,fisher_samples=2,compress_windows=2,save_windows=1,eval_maze=2,eval_fourrooms=2)
    torch.cuda.reset_peak_memory_stats()
    original=update_pair
    def measured(*args,**kwargs):return original(*args,**kwargs,diagnostics=True)
    with patch('tasks.suffix_dual_expert.train.update_pair',side_effect=measured):
        direct=run(tiny,output/'direct')
        assert run(tiny,output/'resumed',stop_after_boundaries=1)['status']=='paused_at_boundary'
        assert run(tiny,output/'resumed',resume=True,stop_after_boundaries=2)['status']=='paused_at_boundary'
        assert run(tiny,output/'resumed',resume=True,stop_after_boundaries=1)['status']=='paused_at_boundary'
        resumed=run(tiny,output/'resumed',resume=True)
    identity,_,_=preflight(tiny)
    a=load_boundary(output/'direct',identity);b=load_boundary(output/'resumed',identity)
    assert tensor_hash(a['student'])==tensor_hash(b['student'])
    assert all(torch.equal(a['student'][n],b['student'][n]) for n in a['student'])
    for name in ('maze_length5','maze_before_compress','final_kb'):
        assert direct['snapshots'][name]['sha256']==resumed['snapshots'][name]['sha256']
    for pa,pb in zip(a['optimizer']['state'].values(),b['optimizer']['state'].values()):
        assert all(torch.equal(pa[k],pb[k]) for k in pa)
    rows=[json.loads(line) for line in (output/'direct/metrics.jsonl').read_text().splitlines()]
    compress=[r for r in rows if r['phase']=='compress']
    assert all(r['maze_gradient_norm']>0 and r['fourrooms_gradient_norm']>0 and r['adam_updates']==4
               and r['ewc_applications']==(4 if config.ewc_enabled else 0) and r['ewc_enabled']==config.ewc_enabled for r in compress)
    assert any(r['ewc']>0 for r in compress) if config.ewc_enabled else all(r['ewc']==0 for r in compress)
    mixed_peak=torch.cuda.max_memory_reserved()/1024**2
    result=evaluate(output/'direct',episodes=2)
    assert result['episodes']==8 and result['weights_unchanged']
    record=dict(status='passed',model_type=config.model_type,gpu=torch.cuda.get_device_name(),formal_training_started=False,
        source=source_hashes(),config=config.resolved(),
        replay30_peak_reserved_mib=replay_peak,mixed_peak_reserved_mib=mixed_peak,
        validation_budget=tiny.budget(),resume_student_bitwise_equal=True,resume_adam_bitwise_equal=True,
        frozen_snapshots_equal=True,legacy_imports_blocked=True,teacher_unchanged=direct['teacher_unchanged'],
        maze_teacher_unchanged=direct['maze_teacher_unchanged'],fisher_unchanged=direct['fisher_unchanged'],
        task_gradients=[{k:r[k] for k in ('maze_gradient_norm','fourrooms_gradient_norm','ewc')} for r in compress],
        diagnostic_evaluation_episodes=8,scope='local correctness only; no performance conclusions')
    atomic_json(output/'verification.json',record);print('VERIFICATION_PASSED',json.dumps(record),flush=True)


if __name__=='__main__':main()
