"""P/C/F stage commits and isolated evaluation; an interrupted stage is rerun whole."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import uuid
import numpy as np
import torch
import yaml
from .config import REPO_ROOT,resolved_dict,config_hash
from .schedule import expand_stages
from .data import build_manifest,MazeManifest
from .envs import SyncVectorEnv,make_env
from .models import StandalonePolicy,DualPolicy,save_snapshot,load_snapshot
from .teachers import MazeTeacher,FourRoomsTeacher
from .checkpoint import (atomic_json,reference,verify_reference,source_manifest,rng_state,restore_rng,
                         save_boundary,load_boundary,seal_artifact,verify_artifact)
from .learning.progress import run_progress
from .learning.distill import run_compress_stage
from .learning.fisher import run_fisher_stage,validate_fisher
from .evaluate import evaluate_policy
from .wandb_logging import EventLogger


def export_policy(policy,path,root,manifest_ref):
    save_snapshot(policy,path); seal_artifact(path)
    # Metadata links the inference artifact to fixed panels without external TA files.
    import os
    metadata=Path(str(path)+'.metadata.json')
    atomic_json(metadata,{'run_root':os.path.relpath(root,path.parent),'manifest':manifest_ref})
    seal_artifact(metadata)
    return reference(root,path)


def run(config,seed,run_dir,resume=False,wandb_mode='disabled',max_stages=None,hook=None):
    if seed<0 or (max_stages is not None and max_stages<1): raise ValueError('invalid seed/max-stages')
    root=Path(run_dir).resolve(); stages=expand_stages(config); device=torch.device(config.training.device)
    if device.type=='cuda': torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)
    action=torch.Generator().manual_seed(seed+101); minibatch=np.random.default_rng(seed+102); fisher_rng=np.random.default_rng(seed+103)
    def trigger(point,stage=None):
        if hook: hook(point,stage)
    if resume:
        payload,_=load_boundary(root/'checkpoints/latest.json',config,seed)
        manifest=MazeManifest.load(verify_reference(root,payload['references']['manifest']),REPO_ROOT/config.environment.maze_root)
        kb=StandalonePolicy(config).to(device); kb.load_state_dict(payload['kb'],strict=True)
        if payload['fisher'] is not None: validate_fisher(kb,payload['fisher'])
        restore_rng(payload['rng'],action,minibatch,fisher_rng)
    else:
        root.mkdir(parents=True,exist_ok=False)
        for name in ('teachers','checkpoints','snapshots','evaluation','exports','diagnostics'): (root/name).mkdir()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        config_path=root/'resolved_config.yaml'; config_path.write_text(yaml.safe_dump(resolved_dict(config),sort_keys=False),encoding='utf-8')
        manifest=build_manifest(REPO_ROOT/config.environment.maze_root,validation_count=config.evaluation.maze_validation_hashes,
                                validation_episodes=config.evaluation.validation_episodes,test_episodes=config.evaluation.test_episodes,
                                drift_episodes=min(32,config.evaluation.validation_episodes))
        manifest.save(root/'maze_manifest.json'); MazeTeacher().save_descriptor(root/'teachers/maze_teacher.json')
        shutil.copyfile(REPO_ROOT/config.teachers.fourrooms.checkpoint,root/'teachers/fourrooms.pt')
        # Load copied teacher once BEFORE the initial commit to verify native keys/config.
        native=FourRoomsTeacher(root/'teachers/fourrooms.pt',device='cpu'); del native
        source=source_manifest()
        atomic_json(root/'provenance.json',{'seed':seed,'method':config.method,'schema_version':3,'sequence_protocol':config.sequence_protocol,
                    'config_hash':config_hash(config),'source':source,'hardware':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'})
        refs={name:reference(root,path) for name,path in {'config':config_path,'provenance':root/'provenance.json',
              'manifest':root/'maze_manifest.json','maze_teacher':root/'teachers/maze_teacher.json','fourrooms_teacher':root/'teachers/fourrooms.pt'}.items()}
        kb=StandalonePolicy(config).to(device)
        payload={'artifact_type':'v3_stage_boundary','schema_version':3,'method':config.method,'sequence_protocol':config.sequence_protocol,
                 'seed':seed,'config_hash':config_hash(config),'source':source,'stages':[asdict(s) for s in stages],
                 'next_index':0,'kb':kb.state_dict(),'kb_ready':False,'fisher':None,'global_env_steps':0,'optimizer_updates':0,'next_transition_id':0,
                 'rng':rng_state(action,minibatch,fisher_rng),'references':refs,'active_snapshots':{},'evaluations':{},'visit_reports':{},'finalized':False,
                 'teacher_identity':{'maze':MazeTeacher().source_snapshot_id,'fourrooms':refs['fourrooms_teacher']['sha256']},'data_hash':refs['manifest']['sha256']}
        save_boundary(root,payload)
    logger=EventLogger(root,config,seed,wandb_mode,resume)
    failed=True; completed=0; committed_index=payload['next_index']
    try:
        if payload['finalized']: failed=False; return payload
        for index in range(payload['next_index'],len(stages)):
            stage=stages[index]; attempt=uuid.uuid4().hex; trigger('stage_start',stage)
            identity={'family':stage.family,'visit':stage.visit,'stage':stage.key,'phase':stage.phase,'task':stage.task,'attempt_id':attempt}
            before_steps=payload['global_env_steps']; before_updates=payload['optimizer_updates']
            def log_window(event):
                logger.emit({**identity,**event,'event':'training','global_env_steps':before_steps+event['stage_env_steps'],
                             'global_optimizer_updates':before_updates+event['optimizer_updates']})
            envs=SyncVectorEnv([make_env(config,stage.task,'train',manifest=manifest,seed=seed+index*1000+slot)
                               for slot in range(config.training.num_envs)])
            policy=None
            try:
                if stage.phase=='P':
                    dual=DualPolicy(config,kb,kb_ready=payload['kb_ready'])
                    expert=MazeTeacher() if stage.task=='maze_medium' else FourRoomsTeacher(root/'teachers/fourrooms.pt',device=device)
                    result=run_progress(envs,dual,expert,config,stage.env_steps,action,minibatch,payload['next_transition_id'],log_window)
                    if result.eligible_target_steps!=stage.env_steps: raise ValueError('P target budget mismatch')
                    policy=result.policy; del dual,expert
                    path=root/'snapshots'/f'{index:03d}_{attempt}_active.pt'
                    ref=export_policy(policy,path,root,payload['references']['manifest'])
                    payload['active_snapshots'][stage.key]=ref
                    payload['references'][stage.key+'/sidecar']=reference(root,str(path)+'.sha256.json')
                    payload['references'][stage.key+'/metadata']=reference(root,str(path)+'.metadata.json')
                    payload['references'][stage.key+'/metadata_sidecar']=reference(root,str(path)+'.metadata.json.sha256.json')
                    transitions,updates,next_id=result.transitions,result.optimizer_updates,result.next_transition_id
                    statistics=result.statistics
                elif stage.phase=='C':
                    pkey=stages[index-1].key
                    teacher=load_snapshot(verify_artifact(verify_reference(root,payload['active_snapshots'][pkey])),device)
                    result=run_compress_stage(envs,teacher,kb,payload['fisher'],config,stage.env_steps,action,minibatch,payload['next_transition_id'],log_window)
                    payload['kb_ready']=result.kb_ready; transitions,updates,next_id=result.transitions,result.optimizer_updates,result.next_transition_id
                    statistics=result.statistics; policy=kb; del teacher
                else:
                    result=run_fisher_stage(envs,kb,payload['fisher'],config,stage.env_steps,action,fisher_rng,payload['next_transition_id'],stage.key)
                    payload['fisher']=result['fisher']; transitions,updates,next_id=result['transitions'],0,result['next_transition_id']
                    statistics={**result['statistics'],'scored_samples':result['scored_samples'],'selected_ids':result['selected_ids']}
                    logger.emit({**identity,'event':'fisher','policy_type':'kb','global_env_steps':before_steps+stage.env_steps,
                                 'stage_env_steps':stage.env_steps,'scored_samples':result['scored_samples'],**result['statistics']})
            finally: envs.close()
            if transitions!=stage.env_steps: raise ValueError('stage budget mismatch')
            payload['global_env_steps']+=transitions; payload['optimizer_updates']+=updates; payload['next_transition_id']=next_id
            trigger('training_complete',stage)
            if policy is not None:
                trigger('before_evaluation',stage)
                eid=f'{index:03d}_{attempt}_validation'
                report=evaluate_policy(policy,config,manifest,identity={**identity,'evaluation_id':eid,
                         'policy_type':'active' if stage.phase=='P' else 'kb','active_source':stage.task if stage.phase=='P' else None,
                         'global_env_steps':payload['global_env_steps']})
                path=root/'evaluation'/f'{eid}.json'; atomic_json(path,report)
                payload['evaluations'][stage.key]={'reference':reference(root,path),'policy_type':report['policy_type'],'visit':stage.visit,
                                                   'task':stage.task,'success_rates':{task:r['success_rate'] for task,r in report['tasks'].items()}}
                for task,metrics in report['tasks'].items():
                    logger.emit({**identity,'event':'evaluation','evaluation_id':eid,'evaluation_task':task,'policy_type':report['policy_type'],
                                 'active_source':report['active_source'],'split':'validation','global_env_steps':payload['global_env_steps'],
                                 **{k:v for k,v in metrics.items() if k!='results'},'evaluation_seconds':report['elapsed_seconds']})
                trigger('evaluation_complete',stage)
            payload['next_index']=index+1
            if index+1==len(stages) or stages[index+1].visit!=stage.visit:
                chosen={s.key:payload['evaluations'][s.key] for s in stages[:index+1] if s.visit==stage.visit and s.phase in ('P','C')}
                last_c=next(s.key for s in reversed(stages[:index+1]) if s.phase=='C')
                visit_path=root/'evaluation'/f'visit_{stage.visit}_{attempt}.json'
                atomic_json(visit_path,{'visit':stage.visit,'policies':chosen,'final_kb':payload['evaluations'][last_c],
                                       'kb_only_forgetting_rows':[v for v in payload['evaluations'].values() if v['policy_type']=='kb']})
                payload['visit_reports'][str(stage.visit)]=reference(root,visit_path)
            payload['kb']={n:p.detach().cpu().clone() for n,p in kb.state_dict().items()}
            payload['rng']=rng_state(action,minibatch,fisher_rng)
            if device.type=='cuda':
                logger.emit({**identity,'event':'memory','peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                             'global_env_steps':payload['global_env_steps']})
            trigger('before_commit',stage); save_boundary(root,payload)
            committed_index=payload['next_index']
            logger.emit({**identity,'event':'stage_complete','global_env_steps':payload['global_env_steps'],'optimizer_updates':payload['optimizer_updates'],
                         'next_index':payload['next_index'],'statistics':statistics})
            completed+=1; del result,policy
            if max_stages is not None and completed>=max_stages and payload['next_index']<len(stages):
                failed=False; return payload
        trigger('before_finalization')
        report=evaluate_policy(kb,config,manifest,split='test',identity={'family':'pnc','visit':config.pnc.visits-1,'phase':'final',
                    'evaluation_id':'final_test','policy_type':'kb','active_source':None,'global_env_steps':payload['global_env_steps']})
        trigger('final_evaluation_complete')
        atomic_json(root/'evaluation/final_test.json',report)
        for task,metrics in report['tasks'].items():
            logger.emit({'event':'final_test','family':'pnc','phase':'final','policy_type':'kb','evaluation_task':task,
                         'split':'test','global_env_steps':payload['global_env_steps'],
                         **{k:v for k,v in metrics.items() if k!='results'},'evaluation_seconds':report['elapsed_seconds']})
        final=root/'exports/final.pt'
        # Only uncommitted finalization files may be replaced after a failed final attempt.
        if final.exists(): final.unlink()
        export_policy(kb,final,root,payload['references']['manifest']); verify_artifact(final)
        for name,path in {'final_test':root/'evaluation/final_test.json','final_policy':final,'final_sidecar':Path(str(final)+'.sha256.json'),
                          'final_metadata':Path(str(final)+'.metadata.json'),'final_metadata_sidecar':Path(str(final)+'.metadata.json.sha256.json')}.items():
            payload['references'][name]=reference(root,path)
        payload['finalized']=True; payload['rng']=rng_state(action,minibatch,fisher_rng)
        trigger('before_final_commit'); save_boundary(root,payload)
        logger.emit({'event':'finalized','family':'pnc','phase':'final','policy_type':'kb','global_env_steps':payload['global_env_steps'],'next_index':payload['next_index']})
        failed=False; return payload
    finally:
        if failed:
            logger.emit({'event':'run_interrupted','family':'pnc','next_uncommitted_index':committed_index,
                         'note':'Resume uses latest committed boundary; partial stage data is not reused.'})
        logger.close(failed)
