"""Offline fixed-panel stage inference; spawned workers contain environments only."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .envs.evaluation import EvaluationPool
from .config import REPO_ROOT,load_config,config_hash
from .data import MazeManifest
from .envs import MazeEnv,FourRoomsEnv,MazeMapCache
from .models import StandalonePolicy,DualPolicy,frozen_copy,select_state,replace_slots,load_snapshot
from .checkpoint import isolated_rng,atomic_json,verify_artifact,load_boundary,verify_reference,reference,load_stage_index


def create_panel_env(spec):
    task,split,root,entry=spec[:4]
    if task=='maze_medium': return MazeEnv(root,(entry,),map_data=spec[4] if len(spec)==5 else None)
    if task=='fourrooms': return FourRoomsEnv(split=split,evaluation_seeds=(entry,))
    raise ValueError('unknown panel task')


def panels(config,manifest,task,split,map_cache=None):
    if split not in ('validation','test'): raise ValueError('evaluation split must be validation/test')
    if task=='maze_medium':
        root=Path(config.environment.maze_root)
        if not root.is_absolute(): root=REPO_ROOT/root
        entries=manifest.entries(split,panel=True)
    elif task=='fourrooms':
        root=None
        count=getattr(config.evaluation,split+'_episodes')
        start=getattr(config.evaluation,'fourrooms_'+split+'_seed_start')
        entries=range(start,start+count)
    else: raise ValueError('unknown panel task')
    if task=='maze_medium' and config.environment.map_cache=='memory':
        cache=map_cache if map_cache is not None else MazeMapCache(root,entries)
        # Only decoded CPU pixels/positions cross the worker pipe, never models or CUDA.
        return [(task,split,root,entry,cache.get(root,entry)) for entry in entries]
    return [(task,split,root,entry) for entry in entries]


def evaluate_policy(policy,config,manifest,split='validation',tasks=None,backend=None,num_envs=None,identity=None,*,factory=create_panel_env,map_cache=None):
    if policy.config != config: raise ValueError('evaluation policy/config execution budget mismatch')
    tasks=tuple(config.task_order if tasks is None else tasks)
    if not tasks or len(set(tasks))!=len(tasks) or not set(tasks)<=set(config.task_order): raise ValueError('invalid task subset')
    backend=config.evaluation.backend if backend is None else backend
    num_envs=config.evaluation.num_envs if num_envs is None else num_envs
    if type(num_envs) is not int or num_envs<1 or backend not in ('serial','subprocess'):
        raise ValueError('invalid evaluation backend/slot count')
    identity=identity or {}
    started=time.perf_counter(); reports={}
    # Evaluation owns a deep frozen snapshot and restores EVERY caller random stream.
    with isolated_rng(),torch.no_grad():
        snapshot=frozen_copy(policy); device=next(snapshot.parameters()).device
        for task in tasks:
            specs=panels(config,manifest,task,split,map_cache); size=min(num_envs,len(specs)); results=[None]*len(specs)
            state=snapshot.initial_state(size); active={}; next_index=0
            with EvaluationPool(size,backend,factory=factory) as pool:
                def assign(slots):
                    nonlocal next_index,state
                    requests={}
                    for slot in slots:
                        if next_index<len(specs):
                            requests[slot]=specs[next_index]
                            active[slot]={'panel_index':next_index,'return':0.,'length':0,'action_counts':[0]*5,'moves':0,'turns':0,'start':True}
                            next_index+=1
                    for slot,(obs,distance) in pool.exchange('reset',requests).items():
                        active[slot].update(obs=obs,shortest_path=distance)
                assign(range(size))
                while active:
                    slots=sorted(active); indices=torch.tensor(slots,device=device)
                    rgb=torch.from_numpy(np.stack([active[s]['obs'].student_rgb for s in slots])).to(device)
                    starts=torch.tensor([active[s]['start'] for s in slots],dtype=torch.bool,device=device)
                    # Cross-task inference keeps shared weights and resets task-local state.
                    logits,proposed=snapshot.step(rgb,select_state(state,indices),starts,task=task)
                    if not torch.isfinite(logits).all(): raise ValueError('nonfinite evaluation logits')
                    state=replace_slots(state,indices,proposed)
                    actions=logits.argmax(-1).cpu().tolist()
                    finished=[]
                    for slot,(obs,reward,term,trunc,info) in pool.exchange('step',dict(zip(slots,actions))).items():
                        item=active[slot]; action=actions[slots.index(slot)]
                        item['return']+=float(reward); item['length']+=1; item['action_counts'][action]+=1
                        if task=='fourrooms':
                            item['moves']+=info['agent_pos_before']!=info['agent_pos_after']
                            item['turns']+=info['agent_dir_before']!=info['agent_dir_after']
                        item.update(obs=obs,start=False)
                        if term or trunc:
                            index=item['panel_index']
                            results[index]={k:v for k,v in item.items() if k not in ('obs','start')}
                            results[index].update(success=bool(info['success']),terminated=bool(term),truncated=bool(trunc))
                            del active[slot]; finished.append(slot)
                    # Reset flags replace both columns; unused tail slots are never inferred.
                    assign(finished)
            if any(result is None for result in results): raise RuntimeError('incomplete evaluation panel')
            steps=sum(r['length'] for r in results)
            reports[task]={'ticks':config.ctm.ticks_by_task.for_task(task),'memory_ticks':config.ctm.memory_length,
                           'episodes':len(results),'success_rate':float(np.mean([r['success'] for r in results])),
                           'mean_return':float(np.mean([r['return'] for r in results])),
                           'mean_length':float(np.mean([r['length'] for r in results])),
                           'environment_steps':steps,'action_counts':np.sum([r['action_counts'] for r in results],axis=0).tolist(),
                           'displacement_rate':sum(r['moves'] for r in results)/steps if task=='fourrooms' else None,
                           'turn_rate':sum(r['turns'] for r in results)/steps if task=='fourrooms' else None,'results':results}
    elapsed=time.perf_counter()-started
    return {**identity,'split':split,'backend':backend,'num_envs':num_envs,'tasks':reports,'elapsed_seconds':elapsed,
            'environment_steps':sum(r['environment_steps'] for r in reports.values()),
            'episodes_per_second':sum(r['episodes'] for r in reports.values())/elapsed}


def load_for_evaluation(path,policy_type='kb',device='cpu',stage_key=None,*,identity=None,manifest_cache=None):
    path=Path(path).resolve()
    if policy_type not in ('active','kb'): raise ValueError('unknown policy type')
    expected_ref=None; expected_config=None; expected_manifest=None
    if path.name=='latest.json':
        root=path.parent.parent; config=load_config(root/'resolved_config.yaml')
        provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
        payload,_=load_boundary(path,config,provenance['seed'])
        keys=[s['key'] for s in payload['stages'][:payload['next_index']]
              if (s['phase']=='P')==(policy_type=='active')]
        stage_key=stage_key or (keys[-1] if keys else None)
        if stage_key not in payload['stage_snapshots']: raise ValueError('complete stage snapshot unavailable')
        expected_ref=payload['stage_snapshots'][stage_key]
        expected_config=payload['config_hash']; expected_manifest=payload['references']['manifest']
        path=verify_reference(root,expected_ref)
    elif path.suffix=='.json':
        index,root=load_stage_index(path)
        if stage_key is None: raise ValueError('--stage-key is required for a stage index')
        items=[s for s in index['stages'] if s['key']==stage_key]
        if len(items)!=1: raise ValueError('stage not found in index')
        item=items[0]
        if item['policy_type']!=policy_type: raise ValueError('stage/policy type mismatch')
        expected_ref=item['reference']; expected_config=index['config_hash']; expected_manifest=index['manifest']
        path=verify_reference(root,expected_ref)
    policy=load_snapshot(verify_artifact(path),device); config=policy.config
    metadata=json.loads(verify_artifact(str(path)+'.metadata.json').read_text(encoding='utf-8'))
    metadata_root=(path.parent/metadata['run_root']).resolve()
    if expected_ref is not None and metadata_root!=root: raise ValueError('snapshot/index run root mismatch')
    root=metadata_root
    if (stage_key is not None and metadata.get('stage')!=stage_key) or metadata.get('policy_type')!=policy_type:
        raise ValueError('stage snapshot identity mismatch')
    if expected_config is not None and (expected_config!=config_hash(config) or metadata['manifest']!=expected_manifest):
        raise ValueError('snapshot/index config or manifest mismatch')
    manifest=verify_reference(root,metadata['manifest'])
    if policy_type=='active' and not isinstance(policy,DualPolicy): raise ValueError('artifact does not contain a complete Active dual policy')
    if policy_type=='kb' and not isinstance(policy,StandalonePolicy): raise ValueError('artifact does not contain a KB policy')
    if identity is not None:
        identity.update({k:metadata[k] for k in ('stage','visit','task','phase','policy_type','global_env_steps','optimizer_updates')})
        identity.update(checkpoint=reference(root,path),active_source=metadata['task'] if policy_type=='active' else None)
    maze_root=Path(config.environment.maze_root)
    maze_root=(maze_root if maze_root.is_absolute() else REPO_ROOT/maze_root).resolve()
    # A suite owns this cache for one invocation. References are still checked for every snapshot.
    key=(str(manifest),metadata['manifest']['sha256'],str(maze_root))
    if manifest_cache is None:
        loaded_manifest=MazeManifest.load(manifest,maze_root)
    else:
        if key not in manifest_cache: manifest_cache[key]=MazeManifest.load(manifest,maze_root)
        loaded_manifest=manifest_cache[key]
    return policy,config,loaded_manifest,root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True); parser.add_argument('--policy',choices=['kb','active'],default='kb')
    parser.add_argument('--stage-key',help='P, C or F key from exports/stages.json'); parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--tasks',nargs='+'); parser.add_argument('--backend',choices=['serial','subprocess'])
    parser.add_argument('--num-envs',type=int); parser.add_argument('--device',default='cpu'); parser.add_argument('--output',required=True)
    args=parser.parse_args(); output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(2)
    identity={'evaluation_id':output.stem}
    policy,config,manifest,_=load_for_evaluation(args.checkpoint,args.policy,args.device,args.stage_key,identity=identity)
    report=evaluate_policy(policy,config,manifest,args.split,args.tasks,args.backend,args.num_envs,
                           identity)
    atomic_json(output,report)


if __name__=='__main__': main()
