"""Small real-model on/off forks from one F boundary; not formal training."""
import argparse
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json

import torch
from .test_core import ForbidLegacy
from ...config import Config
from ...train import run,preflight,fork_compress_boundary
from ...fisher import estimate,fisher_hash
from ...checkpoint import load_boundary,tensor_hash,source_hashes,atomic_json
from ...envs import file_sha
from ...compress import update_pair
from ...evaluate import evaluate
from ...models import build_policy


def main(default_model_type='ctm'):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--model-type',choices=('ctm','lstm'),default=default_model_type)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--full-window',action='store_true',help='Real 16+16 slots, 50-step window, micro4')
    args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    config=replace(Config(),model_type=args.model_type,device=args.device,lengths=(5,),pool_sizes=(2,),first_updates=(1,),suffix_updates=(1,),
        sequences=2,suffix_microbatch=1,candidate_batch=2,slots_per_task=4,compress_microbatch=1,window=1,
        fisher_steps=8,fisher_samples=1,compress_windows=2,save_windows=1,eval_maze=1,eval_fourrooms=1)
    if args.full_window:
        config=replace(config,slots_per_task=16,compress_microbatch=4,window=50,fisher_steps=64,fisher_samples=2)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('requested CUDA unavailable')
        torch.cuda.reset_peak_memory_stats()
    def measured(*a,**k):return update_pair(*a,**k,diagnostics=True)
    parent=out/'parent';identity,_,_=preflight(config)
    with patch('tasks.suffix_dual_expert.train.estimate',wraps=estimate) as calls, patch('tasks.suffix_dual_expert.train.update_pair',side_effect=measured):
        assert run(config,parent,stop_after_boundaries=2)['status']=='paused_at_boundary'
        boundary=load_boundary(parent,identity)
        assert boundary['phase']=='compress' and boundary['windows']==0 and boundary['optimizer'] is None
        before={str(p.relative_to(parent)):file_sha(p) for p in parent.rglob('*') if p.is_file()}
        wrong=replace(config,model_type='ctm' if config.model_type=='lstm' else 'lstm')
        wrong_identity,_,_=preflight(wrong)
        try:
            fork_compress_boundary(wrong,out/'wrong_architecture',parent,wrong_identity,build_policy(wrong))
        except ValueError as error:
            assert 'resume config' in str(error)
        else:raise AssertionError('cross-model F fork accepted')
        assert not (out/'wrong_architecture').exists()
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
            assert records[name]['summary']['model_type']==args.model_type
            for row in [json.loads(s) for s in (dest/'metrics.jsonl').read_text().splitlines()]:
                assert row['maze_gradient_norm']>0 and row['fourrooms_gradient_norm']>0
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
    evaluations={name:evaluate(out/name,device=args.device,episodes=1) for name in ('on','off')}
    assert all(v['episodes']==4 and v['weights_unchanged'] and v['model_type']==args.model_type for v in evaluations.values())
    atomic_json(out/'verification.json',dict(status='passed',source=source_hashes(),device=args.device,
        model_type=args.model_type,model_spec=config.model_spec(),full_window=args.full_window,
        gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2 if args.device.startswith('cuda') else None,
        cross_model_fork_rejected=True,diagnostic_evaluation_episodes=8,weights_unchanged=True,
        formal_training_started=False,fisher_executions=1,parent_unchanged=True,same_initial_kb_fisher_rng=True,
        fresh_adam_each_branch=True,resume_student_adam_bitwise_equal=True,same_first_window_expert_trajectories=True,
        small_budget=config.budget(),origins={name:v['origin'] for name,v in records.items()}))
    print('COMPRESS_FORK_VERIFIED',flush=True)


if __name__=='__main__':main()
