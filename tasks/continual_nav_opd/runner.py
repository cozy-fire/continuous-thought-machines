"""Maze-pool and P/C/F commits; resume reruns only the uncommitted pool/stage."""
from dataclasses import asdict
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
from .envs import SyncVectorEnv,make_env,MazeMapCache
from .models import StandalonePolicy,DualPolicy,save_snapshot,load_snapshot
from .teachers import MazeTeacher,FourRoomsTeacher
from .checkpoint import (atomic_json,reference,verify_reference,source_manifest,rng_state,restore_rng,
                         save_boundary,load_boundary,seal_artifact,verify_artifact)
from .learning.progress import run_progress
from .data.maze_curriculum import MazeStateTable
from .learning.maze_progress import run_maze_progress
from .learning.distill import run_compress_stage
from .learning.fisher import run_fisher_stage,validate_fisher
from .wandb_logging import EventLogger


def export_policy(policy,path,root,manifest_ref,identity):
    save_snapshot(policy,path); seal_artifact(path)
    # Metadata links the inference artifact to fixed panels without external TA files.
    import os
    metadata=Path(str(path)+'.metadata.json')
    atomic_json(metadata,{'run_root':os.path.relpath(root,path.parent),'manifest':manifest_ref,**identity})
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
        for name in ('teachers','checkpoints','snapshots','exports','diagnostics'): (root/name).mkdir()
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
        atomic_json(root/'provenance.json',{'seed':seed,'method':config.method,'schema_version':4,'sequence_protocol':config.sequence_protocol,
                    'config_hash':config_hash(config),'source':source,'hardware':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'})
        refs={name:reference(root,path) for name,path in {'config':config_path,'provenance':root/'provenance.json',
              'manifest':root/'maze_manifest.json','maze_teacher':root/'teachers/maze_teacher.json','fourrooms_teacher':root/'teachers/fourrooms.pt'}.items()}
        kb=StandalonePolicy(config).to(device)
        payload={'artifact_type':'v4_training_boundary','schema_version':4,'method':config.method,'sequence_protocol':config.sequence_protocol,
                 'seed':seed,'config_hash':config_hash(config),'source':source,'stages':[asdict(s) for s in stages],
                 'next_index':0,'kb':kb.state_dict(),'kb_ready':False,'fisher':None,'global_env_steps':0,'optimizer_updates':0,'next_transition_id':0,
                 'rng':rng_state(action,minibatch,fisher_rng),'references':refs,'stage_snapshots':{},'finalized':False,
                 'teacher_identity':{'maze':MazeTeacher().source_snapshot_id,'fourrooms':refs['fourrooms_teacher']['sha256']},'data_hash':refs['manifest']['sha256'],
                 'stage_counters':{},'maze_progress':None}
        save_boundary(root,payload)
    logger=EventLogger(root,config,seed,wandb_mode,resume)
    failed=True; completed=0; committed_index=payload['next_index']
    map_cache=None
    try:
        if payload['finalized']: failed=False; return payload
        if payload['next_index']<len(stages) and config.environment.map_cache=='memory':
            entries=manifest.entries('train')+manifest.entries('validation')+manifest.entries('test')
            map_cache=MazeMapCache(REPO_ROOT/config.environment.maze_root,entries)
            logger.emit({'event':'map_cache','family':'pnc','maps':len(map_cache.indices),
                         'cache_bytes':map_cache.images.nbytes,'load_seconds':map_cache.load_seconds,
                         'process_rss_bytes':map_cache.process_rss_bytes,'rss_delta_bytes':map_cache.rss_delta_bytes})
        if config.maze_progress.pool_sizes[-1] > len(manifest.train):
            raise ValueError('Maze curriculum exceeds distinct training manifest')
        if config.maze_progress.pool_sizes[-1] == 44488 and len(manifest.train) != 44488:
            raise ValueError('formal curriculum must cover the entire fixed training split')
        maze_table=None
        if payload['next_index']<len(stages):
            maze_table=MazeStateTable(map_cache,manifest.train[:config.maze_progress.pool_sizes[-1]])
            logger.emit({'event':'maze_state_table','maps':len(maze_table.entries),'nonterminal_positions':len(maze_table.start_ids),
                         'table_bytes':maze_table.table_bytes,'build_seconds':maze_table.build_seconds,
                         'process_rss_bytes':maze_table.process_rss_bytes,'rss_delta_bytes':maze_table.rss_delta_bytes,
                         'total_cache_bytes':map_cache.images.nbytes+maze_table.table_bytes})
        for index in range(payload['next_index'],len(stages)):
            stage=stages[index]; attempt=uuid.uuid4().hex; trigger('stage_start',stage)
            identity={'family':stage.family,'visit':stage.visit,'stage':stage.key,'phase':stage.phase,'task':stage.task,'attempt_id':attempt,
                      'ticks':stage.ticks,'memory_ticks':config.ctm.memory_length}
            before_steps=payload['global_env_steps']; before_updates=payload['optimizer_updates']
            def log_window(event):
                logger.emit({**identity,**event,'event':'training','global_env_steps':before_steps+event['stage_env_steps'],
                             'global_optimizer_updates':before_updates+event['optimizer_updates']})
            maze_p=stage.task=='maze_medium' and stage.phase=='P'
            envs=None if maze_p else SyncVectorEnv([make_env(config,stage.task,'train',manifest=manifest,seed=seed+index*1000+slot,map_cache=map_cache)
                               for slot in range(config.training.num_envs)])
            policy=None
            try:
                if stage.phase=='P':
                    dual=DualPolicy(config,kb,kb_ready=payload['kb_ready'])
                    if maze_p:
                        partial=payload['maze_progress']
                        if partial is not None:
                            dual.load_state_dict(partial['dual'],strict=True)
                            # Construction initializes trainable modules. Undo its RNG use.
                            restore_rng(payload['rng'],action,minibatch,fisher_rng)
                        def commit_pool(progress):
                            payload['maze_progress']={**progress,'stage':stage.key}
                            payload['rng']=rng_state(action,minibatch,fisher_rng)
                            save_boundary(root,payload)
                            logger.emit({**identity,'event':'maze_pool_complete','next_pool':progress['next_pool'],
                                         'optimizer_updates':progress['updates'],'valid_decisions':progress['transitions']})
                            trigger('pool_complete',stage)
                        result=run_maze_progress(maze_table,dual,config,action,minibatch,payload['next_transition_id'],log_window,commit_pool,partial)
                        if result.optimizer_updates!=stage.optimizer_updates: raise ValueError('Maze P update budget mismatch')
                    else:
                        expert=FourRoomsTeacher(root/'teachers/fourrooms.pt',device=device)
                        result=run_progress(envs,dual,expert,config,stage.env_steps,action,minibatch,payload['next_transition_id'],log_window)
                        if result.eligible_target_steps!=stage.env_steps: raise ValueError('P target budget mismatch')
                        del expert
                    policy=result.policy; del dual
                    transitions,updates,next_id=result.transitions,result.optimizer_updates,result.next_transition_id
                    statistics=result.statistics
                elif stage.phase=='C':
                    pkey=stages[index-1].key
                    teacher=load_snapshot(verify_artifact(verify_reference(root,payload['stage_snapshots'][pkey])),device)
                    result=run_compress_stage(envs,teacher,kb,payload['fisher'],config,stage.env_steps,action,minibatch,payload['next_transition_id'],log_window)
                    payload['kb_ready']=result.kb_ready; transitions,updates,next_id=result.transitions,result.optimizer_updates,result.next_transition_id
                    statistics=result.statistics; policy=kb; del teacher
                else:
                    result=run_fisher_stage(envs,kb,payload['fisher'],config,stage.env_steps,action,fisher_rng,payload['next_transition_id'],stage.key)
                    payload['fisher']=result['fisher']; transitions,updates,next_id=result['transitions'],0,result['next_transition_id']
                    statistics={**result['statistics'],'scored_samples':result['scored_samples'],'selected_ids':result['selected_ids']}
                    policy=kb
                    logger.emit({**identity,'event':'fisher','policy_type':'kb','global_env_steps':before_steps+stage.env_steps,
                                 'stage_env_steps':stage.env_steps,'scored_samples':result['scored_samples'],**result['statistics']})
            finally:
                if envs is not None: envs.close()
            if not maze_p and transitions!=stage.env_steps: raise ValueError('stage budget mismatch')
            payload['stage_counters'][stage.key]={'transitions':transitions,'optimizer_updates':updates}
            payload['global_env_steps']+=transitions; payload['optimizer_updates']+=updates; payload['next_transition_id']=next_id
            trigger('training_complete',stage)
            # Every phase publishes its own immutable policy, independent of later KB updates.
            policy_type='active' if stage.phase=='P' else 'kb'
            path=root/'snapshots'/f'{index:03d}_{attempt}_{policy_type}.pt'
            payload['stage_snapshots'][stage.key]=export_policy(policy,path,root,payload['references']['manifest'],
                {'stage':stage.key,'visit':stage.visit,'task':stage.task,'phase':stage.phase,'policy_type':policy_type,
                 'stage_counters':payload['stage_counters'][stage.key],'global_env_steps':payload['global_env_steps'],
                 'optimizer_updates':payload['optimizer_updates']})
            for suffix,name in [('.sha256.json','sidecar'),('.metadata.json','metadata'),('.metadata.json.sha256.json','metadata_sidecar')]:
                payload['references'][stage.key+'/'+name]=reference(root,str(path)+suffix)
            trigger('snapshot_complete',stage)
            payload['next_index']=index+1
            payload['maze_progress']=None
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
        final=root/'exports/final.pt'
        # Only uncommitted finalization files may be replaced after a failed final attempt.
        if final.exists(): final.unlink()
        export_policy(kb,final,root,payload['references']['manifest'],
                      {'stage':None,'visit':config.pnc.visits-1,'task':None,'phase':'final','policy_type':'kb',
                       'global_env_steps':payload['global_env_steps'],'optimizer_updates':payload['optimizer_updates']})
        verify_artifact(final)
        # The index is only sealed at finalization; earlier stage commits never reference a mutable index.
        stage_index=root/'exports/stages.json'
        atomic_json(stage_index,{'artifact_type':'v4_stage_index','schema_version':4,'run_root':'..',
                    'config_hash':payload['config_hash'],'manifest':payload['references']['manifest'],
                    'stages':[{**asdict(s),'policy_type':'active' if s.phase=='P' else 'kb',
                               'counters':payload['stage_counters'][s.key],'reference':payload['stage_snapshots'][s.key]} for s in stages]})
        seal_artifact(stage_index)
        trigger('final_export_complete')
        for name,path in {'final_policy':final,'final_sidecar':Path(str(final)+'.sha256.json'),
                          'stage_index':stage_index,'stage_index_sidecar':Path(str(stage_index)+'.sha256.json'),
                          'final_metadata':Path(str(final)+'.metadata.json'),'final_metadata_sidecar':Path(str(final)+'.metadata.json.sha256.json')}.items():
            payload['references'][name]=reference(root,path)
        payload['finalized']=True; payload['rng']=rng_state(action,minibatch,fisher_rng)
        trigger('before_final_commit'); save_boundary(root,payload)
        logger.emit({'event':'finalized','family':'pnc','phase':'final','policy_type':'kb','global_env_steps':payload['global_env_steps'],'next_index':payload['next_index']})
        failed=False; return payload
    finally:
        # Release decoded maps on both normal completion and interrupted stages.
        if map_cache is not None: map_cache.close()
        if failed:
            logger.emit({'event':'run_interrupted','family':'pnc','next_uncommitted_index':committed_index,
                         'note':'Resume uses the last committed Maze pool or other task stage boundary.'})
        logger.close(failed)
