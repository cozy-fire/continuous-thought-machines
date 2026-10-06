"""Independent, resumable validation of saved stages and final KB test."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .checkpoint import (atomic_json, load_stage_index, reference, seal_artifact,
                         verify_artifact, verify_reference)
from .config import REPO_ROOT, config_hash
from .envs import MazeMapCache
from .evaluate import evaluate_policy, load_for_evaluation
from .schedule import expand_stages


def _read_report(path: Path, expected: dict, config, backend: str, num_envs: int) -> dict:
    report=json.loads(verify_artifact(path).read_text(encoding='utf-8'))
    if any(report.get(k)!=v for k,v in expected.items()) or (
            report.get('backend'),report.get('num_envs'))!=(backend,num_envs):
        raise ValueError('evaluation report identity mismatch: '+path.name)
    count=getattr(config.evaluation,expected['split']+'_episodes')
    if set(report.get('tasks',{}))!=set(config.task_order) or any(
            task.get('episodes')!=count or len(task.get('results',[]))!=count
            for task in report['tasks'].values()):
        raise ValueError('incomplete evaluation report: '+path.name)
    return report


def evaluate_stages(stage_index, output_dir, device='cpu', backend=None, num_envs=None) -> dict:
    """Never writes training files; retries skip only sealed, matching reports."""
    stage_index=Path(stage_index).resolve()
    index,root=load_stage_index(stage_index)
    output=Path(output_dir).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('evaluation directory must be outside the training run')
    manifest_cache={}
    first=index['stages'][0]
    policy,config,manifest,_=load_for_evaluation(stage_index,first['policy_type'],device,first['key'],
                                                manifest_cache=manifest_cache)
    expected_stages=[asdict(stage) for stage in expand_stages(config)]
    if config_hash(config)!=index['config_hash'] or [
            {k:stage[k] for k in expected_stages[0]} for stage in index['stages']]!=expected_stages:
        raise ValueError('evaluation requires the complete training stage index')
    del policy
    backend=config.evaluation.backend if backend is None else backend
    num_envs=config.evaluation.num_envs if num_envs is None else num_envs
    if backend not in ('serial','subprocess') or type(num_envs) is not int or num_envs<1:
        raise ValueError('invalid evaluation backend/slot count')
    final=root/'exports/final.pt'
    stage_metadata={}
    for stage in index['stages']:
        path=verify_artifact(verify_reference(root,stage['reference']))
        metadata_path=verify_artifact(str(path)+'.metadata.json')
        stage_metadata[stage['key']]=reference(root,metadata_path)
    plan={'artifact_type':'v4_post_training_evaluation','stage_index':reference(root,stage_index),
          'final_policy':reference(root,verify_artifact(final)),
          'final_metadata':reference(root,verify_artifact(str(final)+'.metadata.json')),
          'config_hash':index['config_hash'],'manifest':index['manifest'],'stage_metadata':stage_metadata,
          'backend':backend,'num_envs':num_envs}
    if output.exists():
        saved=json.loads(verify_artifact(output/'plan.json').read_text(encoding='utf-8'))
        if saved!=plan: raise ValueError('evaluation directory belongs to a different plan')
    else:
        output.mkdir(parents=True,exist_ok=False)
        atomic_json(output/'plan.json',plan); seal_artifact(output/'plan.json')
    jobs=[(f'{i:03d}_{stage["task"]}_{stage["phase"]}_validation.json',stage_index,
           stage['policy_type'],stage['key'],{'stage':stage['key'],'phase':stage['phase'],
           'policy_type':stage['policy_type'],'checkpoint':stage['reference'],'split':'validation'})
          for i,stage in enumerate(index['stages'])]
    jobs.append(('final_test.json',final,'kb',None,{'stage':None,'phase':'final','policy_type':'kb',
                 'checkpoint':plan['final_policy'],'split':'test'}))
    reports={}; completed={}; cache=None; current=None
    try:
        for name,path,kind,key,expected in jobs:
            current=name
            if (output/name).exists():
                report=_read_report(output/name,expected,config,backend,num_envs)
            else:
                atomic_json(output/'status.json',{'status':'evaluating','current':name,'completed':len(reports)})
                identity={'evaluation_id':Path(name).stem}
                policy,loaded_config,loaded_manifest,_=load_for_evaluation(
                    path,kind,device,key,identity=identity,manifest_cache=manifest_cache)
                if config_hash(loaded_config)!=index['config_hash'] or loaded_manifest is not manifest:
                    raise ValueError('evaluation snapshot config/manifest mismatch')
                if cache is None:
                    maze_root=Path(config.environment.maze_root)
                    if not maze_root.is_absolute(): maze_root=REPO_ROOT/maze_root
                    cache=MazeMapCache(maze_root,manifest.validation_panel+manifest.test_panel)
                report=evaluate_policy(policy,config,manifest,split=expected['split'],backend=backend,
                                       num_envs=num_envs,identity=identity,map_cache=cache)
                del policy
                atomic_json(output/name,report); seal_artifact(output/name)
                report=_read_report(output/name,expected,config,backend,num_envs)
                print(json.dumps({'event':'post_training_evaluation','report':name,'completed':len(reports)+1,
                                  'total':len(jobs)}),flush=True)
            reports[name]=reference(output,output/name)
            completed[name]={task:stats['success_rate'] for task,stats in report['tasks'].items()}
        summary={'status':'complete','plan':reference(output,output/'plan.json'),
                 'reports':reports,'success_rates':completed,'stage_evaluations':len(index['stages']),
                 'final_test':True}
        if (output/'summary.json').exists():
            previous=json.loads(verify_artifact(output/'summary.json').read_text(encoding='utf-8'))
            if previous!=summary: raise ValueError('evaluation summary mismatch')
            for ref in previous['reports'].values(): verify_reference(output,ref)
        else:
            atomic_json(output/'summary.json',summary); seal_artifact(output/'summary.json')
        status={'status':'complete','completed':len(jobs),'summary':reference(output,output/'summary.json')}
        if not (output/'status.json').exists() or json.loads((output/'status.json').read_text(encoding='utf-8'))!=status:
            atomic_json(output/'status.json',status)
        return summary
    except Exception as exc:
        atomic_json(output/'status.json',{'status':'failed','current':current,'completed':len(reports),'error':repr(exc)})
        raise
    finally:
        if cache is not None: cache.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-index',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--backend',choices=['serial','subprocess'])
    parser.add_argument('--num-envs',type=int)
    args=parser.parse_args()
    torch.set_num_threads(2)
    report=evaluate_stages(args.stage_index,args.output_dir,args.device,args.backend,args.num_envs)
    print(json.dumps({'event':'evaluation_complete','output_dir':str(Path(args.output_dir).resolve()),
                      'stage_evaluations':report['stage_evaluations'],'final_test':report['final_test']}),flush=True)


if __name__=='__main__': main()
