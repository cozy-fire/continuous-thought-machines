"""Small real-model on/off forks from one F boundary; not formal training."""
import argparse
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json

import torch
from .test_core import ForbidLegacy
from ..config import Config
from ..train import run,preflight
from ..fisher import estimate,fisher_hash
from ..checkpoint import load_boundary,tensor_hash,source_hashes,atomic_json
from ..envs import file_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    config=replace(Config(),device='cpu',lengths=(5,),pool_sizes=(2,),first_updates=(1,),suffix_updates=(1,),
        sequences=2,suffix_microbatch=1,candidate_batch=2,slots_per_task=4,compress_microbatch=1,window=1,
        fisher_steps=8,fisher_samples=1,compress_windows=2,save_windows=1,eval_maze=1,eval_fourrooms=1)
    parent=out/'parent';identity,_,_=preflight(config)
    with patch('tasks.suffix_dual_expert.train.estimate',wraps=estimate) as calls:
        assert run(config,parent,stop_after_boundaries=2)['status']=='paused_at_boundary'
        boundary=load_boundary(parent,identity)
        assert boundary['phase']=='compress' and boundary['windows']==0 and boundary['optimizer'] is None
        before={str(p.relative_to(parent)):file_sha(p) for p in parent.rglob('*') if p.is_file()}
        records={};boundaries={}
        for enabled in (True,False):
            name='on' if enabled else 'off';cfg=replace(config,ewc_enabled=enabled)
            dest=out/name;branch_identity,_,_=preflight(cfg)
            assert run(cfg,dest,from_fisher=parent,stop_after_boundaries=1)['status']=='paused_at_boundary'
            saved=load_boundary(dest,branch_identity);boundaries[name]=saved
            assert saved['windows']==1 and all(int(s['step'])==4 for s in saved['optimizer']['state'].values())
            rows=[json.loads(s) for s in (dest/'metrics.jsonl').read_text().splitlines()]
            assert len(rows)==1 and rows[0]['phase']=='compress' and rows[0]['ewc_enabled']==enabled
            records[name]=dict(summary=run(cfg,dest,resume=True),origin=saved['compress_origin'])
            direct=run(cfg,out/(name+'_direct'),from_fisher=parent)
            expected=load_boundary(out/(name+'_direct'),branch_identity)
            resumed=load_boundary(dest,branch_identity)
            assert tensor_hash(expected['student'])==tensor_hash(resumed['student'])
            for a,b in zip(expected['optimizer']['state'].values(),resumed['optimizer']['state'].values()):
                assert all(torch.equal(a[k],b[k]) for k in a)
            assert direct['snapshots']['final_kb']['sha256']==records[name]['summary']['snapshots']['final_kb']['sha256']
            assert records[name]['summary']['teacher_unchanged'] and records[name]['summary']['maze_teacher_unchanged']
            assert records[name]['summary']['fisher_unchanged']
            assert fisher_hash(resumed['fisher'])==fisher_hash(boundary['fisher'])
            initial_ref=json.loads((dest/'compress_origin.json').read_text())
            assert initial_ref['fresh_compress_adam'] and initial_ref['student_sha256']==tensor_hash(boundary['student'])
        assert calls.call_count==1  # Branches do not enter/recompute F.
    assert before=={str(p.relative_to(parent)):file_sha(p) for p in parent.rglob('*') if p.is_file()}
    assert records['on']['origin']['boundary_sha256']==records['off']['origin']['boundary_sha256']
    assert records['on']['origin']['fisher_sha256']==records['off']['origin']['fisher_sha256']
    assert torch.equal(boundaries['on']['rng']['streams']['action'],boundaries['off']['rng']['streams']['action'])
    assert boundaries['on']['rng']['streams']['mini']==boundaries['off']['rng']['streams']['mini']
    for a,b in zip(boundaries['on']['collectors'],boundaries['off']['collectors']):
        assert a['envs']==b['envs']
    for name,applications in (('on',4),('off',0)):
        rows=[json.loads(s) for s in (out/name/'metrics.jsonl').read_text().splitlines()]
        assert all(r['ewc_applications']==applications and r['adam_updates']==4 for r in rows)
        if name=='off':assert all(r['ewc']==0 for r in rows)
    atomic_json(out/'verification.json',dict(status='passed',source=source_hashes(),device='cpu',
        formal_training_started=False,fisher_executions=1,parent_unchanged=True,same_initial_kb_fisher_rng=True,
        fresh_adam_each_branch=True,resume_student_adam_bitwise_equal=True,same_first_window_expert_trajectories=True,
        small_budget=config.budget(),origins={name:v['origin'] for name,v in records.items()}))
    print('COMPRESS_FORK_VERIFIED',flush=True)


if __name__=='__main__':main()
