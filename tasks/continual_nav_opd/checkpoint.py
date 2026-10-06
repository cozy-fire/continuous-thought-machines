"""Atomic Maze-pool/other-stage boundaries with strict v4 identity and SHA references."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch
from .config import REPO_ROOT,config_hash
from .schedule import expand_stages
from .teachers import MazeTeacher
from .teachers.fourrooms import file_sha256


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def reference(root,path):
    root=Path(root).resolve(); path=Path(path).resolve(strict=True)
    if not path.is_relative_to(root): raise ValueError('artifact reference escapes run')
    return {'path':path.relative_to(root).as_posix(),'size':path.stat().st_size,'sha256':file_sha256(path)}


def verify_reference(root,ref):
    if not isinstance(ref,dict) or set(ref)!={'path','size','sha256'}: raise ValueError('invalid artifact reference')
    root=Path(root).resolve(); path=(root/ref['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file(): raise ValueError('missing/unsafe artifact reference')
    if path.stat().st_size!=ref['size'] or file_sha256(path)!=ref['sha256']: raise ValueError('artifact SHA/size mismatch: '+ref['path'])
    return path


def source_manifest():
    paths=list((REPO_ROOT/'tasks/continual_nav_opd').rglob('*.py'))
    paths += list((REPO_ROOT/'tasks/continual_nav/envs').glob('*.py'))
    paths += [REPO_ROOT/'tasks/continual_nav/data/manifest.py']
    paths += [REPO_ROOT/'tasks/continual_nav/contracts.py']
    paths += [REPO_ROOT/'models'/name for name in ('ctm.py','ctm_rl.py','resnet.py','modules.py','utils.py','constants.py')]
    return {p.relative_to(REPO_ROOT).as_posix():file_sha256(p) for p in sorted(paths)}


def rng_state(action=None,minibatch=None,fisher=None):
    state={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
           'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}
    if action is not None: state.update(action=action.get_state(),minibatch=minibatch.bit_generator.state,fisher=fisher.bit_generator.state)
    return state


def restore_rng(state,action=None,minibatch=None,fisher=None):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])
    if action is not None:
        action.set_state(state['action']); minibatch.bit_generator.state=state['minibatch']; fisher.bit_generator.state=state['fisher']


@contextmanager
def isolated_rng():
    state=rng_state()
    try: yield
    finally: restore_rng(state)


def seal_artifact(path):
    path=Path(path)
    atomic_json(str(path)+'.sha256.json',{'size':path.stat().st_size,'sha256':file_sha256(path)})


def verify_artifact(path):
    path=Path(path); sidecar=Path(str(path)+'.sha256.json')
    if not path.is_file() or not sidecar.is_file(): raise ValueError('missing artifact/sidecar')
    value=json.loads(sidecar.read_text(encoding='utf-8'))
    if value!={'size':path.stat().st_size,'sha256':file_sha256(path)}: raise ValueError('artifact integrity mismatch')
    return path


def save_boundary(root,payload):
    root=Path(root); directory=root/'checkpoints'; directory.mkdir(exist_ok=True)
    name=f"boundary_{payload['next_index']:03d}_{uuid.uuid4().hex}.pt"
    path=directory/name; temporary=Path(str(path)+'.tmp')
    try:
        with temporary.open('xb') as stream:
            torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)
    marker={'schema_version':4,'checkpoint':reference(root,path),'next_index':payload['next_index'],'finalized':payload['finalized']}
    complete=Path(str(path)+'.complete.json'); atomic_json(complete,marker)
    # latest changes LAST; a crash can only leave an unreferenced complete boundary.
    atomic_json(directory/'latest.json',{'complete':reference(root,complete)})
    return path


REQUIRED={'artifact_type','schema_version','method','sequence_protocol','seed','config_hash','source','stages',
          'next_index','kb','kb_ready','fisher','global_env_steps','optimizer_updates','next_transition_id','rng',
          'references','stage_snapshots','finalized','teacher_identity','data_hash','stage_counters','maze_progress'}


def load_stage_index(path):
    """Portable inference inventory; does not require training source or resume checkpoints."""
    path=verify_artifact(Path(path).resolve())
    index=json.loads(path.read_text(encoding='utf-8'))
    if (index.get('artifact_type'),index.get('schema_version'))!=('v4_stage_index',4):
        raise ValueError('incompatible stage index')
    if not isinstance(index.get('stages'),list) or not index['stages']:
        raise ValueError('empty stage index')
    keys=[s['key'] for s in index['stages']]
    if len(set(keys))!=len(keys): raise ValueError('duplicate stage index keys')
    root=(path.parent/index['run_root']).resolve()
    verify_reference(root,index['manifest'])
    return index,root


def load_boundary(latest,config,seed):
    latest=Path(latest).resolve(); root=latest.parent.parent
    index=json.loads(latest.read_text(encoding='utf-8'))
    if set(index)!={'complete'}: raise ValueError('invalid latest index')
    complete=verify_reference(root,index['complete']); marker=json.loads(complete.read_text(encoding='utf-8'))
    if set(marker)!={'schema_version','checkpoint','next_index','finalized'} or marker['schema_version']!=4:
        raise ValueError('invalid complete marker')
    path=verify_reference(root,marker['checkpoint'])
    # Stage payload is trusted local torch serialization, guarded by file SHA, never v2.
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(payload,dict) or not REQUIRED<=set(payload): raise ValueError('missing checkpoint fields')
    if (payload['artifact_type'],payload['schema_version'],payload['method'],payload['sequence_protocol'])!=('v4_training_boundary',4,'ctm_pnc_opd','maze_onpolicy5_v1'):
        raise ValueError('incompatible checkpoint identity')
    if {'evaluations','visit_reports','active_snapshots'} & set(payload):
        raise ValueError('incompatible evaluation-coupled checkpoint')
    stages=[asdict(s) for s in expand_stages(config)]
    if payload['seed']!=seed or payload['config_hash']!=config_hash(config) or payload['source']!=source_manifest() or payload['stages']!=stages:
        raise ValueError('checkpoint config/source/seed/stages mismatch')
    if not 0<=payload['next_index']<=len(stages) or marker.get('next_index')!=payload['next_index'] or marker.get('finalized')!=payload['finalized']:
        raise ValueError('invalid checkpoint boundary')
    completed=stages[:payload['next_index']]
    if set(payload['stage_counters']) != {s['key'] for s in completed}:
        raise ValueError('invalid completed-stage counters')
    expected=expected_updates=0
    for stage in completed:
        counter=payload['stage_counters'][stage['key']]
        updates=stage['optimizer_updates'] or (0 if stage['phase']=='F' else
            ((stage['env_steps']//config.training.num_envs+config.optimization.learning_steps-1)//config.optimization.learning_steps)*config.optimization.minibatches)
        steps=counter['transitions']
        if counter['optimizer_updates']!=updates or (not stage['optimizer_updates'] and steps!=stage['env_steps']) or (
                stage['optimizer_updates'] and not updates*100 <= steps <= updates*500):
            raise ValueError('checkpoint stage budget mismatch')
        expected+=steps
        expected_updates+=updates
    if payload['global_env_steps']!=expected or payload['next_transition_id']!=expected or (payload['finalized'] and payload['next_index']!=len(stages)):
        raise ValueError('checkpoint budget/index mismatch')
    if payload['optimizer_updates']!=expected_updates: raise ValueError('checkpoint optimizer update count mismatch')
    partial=payload['maze_progress']
    if partial is not None:
        if payload['next_index']==len(stages): raise ValueError('unexpected Maze progress after schedule')
        stage=stages[payload['next_index']]
        if not stage['optimizer_updates'] or partial['stage']!=stage['key'] or not 1<=partial['next_pool']<=len(config.maze_progress.pool_sizes):
            raise ValueError('invalid Maze pool boundary')
        updates=sum(config.maze_progress.pool_updates[:partial['next_pool']])
        if partial['updates']!=updates or not updates*100<=partial['transitions']<=updates*500:
            raise ValueError('invalid Maze pool counters')
        if not partial['optimizer']['state'] or any(float(v['step'])!=updates for v in partial['optimizer']['state'].values()):
            raise ValueError('invalid Maze Adam steps')
        if any(v.is_floating_point() and not torch.isfinite(v).all() for v in partial['dual'].values()):
            raise ValueError('nonfinite Maze boundary weights')
    for ref in payload['references'].values(): verify_reference(root,ref)
    if not {'config','provenance','manifest','maze_teacher','fourrooms_teacher'}<=set(payload['references']):
        raise ValueError('missing required artifact references')
    if not {'python','numpy','torch','cuda','action','minibatch','fisher'}<=set(payload['rng']):
        raise ValueError('missing RNG streams')
    if type(payload['finalized']) is not bool or type(payload['kb_ready']) is not bool or payload['optimizer_updates']<0:
        raise ValueError('invalid checkpoint counters/flags')
    completed=stages[:payload['next_index']]
    if set(payload['stage_snapshots'])!={s['key'] for s in completed}:
        raise ValueError('missing completed-stage snapshots')
    if payload['kb_ready']!=any(s['phase']=='C' for s in completed): raise ValueError('invalid KB readiness')
    fcount=sum(s['phase']=='F' for s in completed)
    if (payload['fisher'] is None)!=(fcount==0) or (payload['fisher'] is not None and payload['fisher'].completed_compressions!=fcount):
        raise ValueError('Fisher stage counter mismatch')
    required_final={'final_policy','final_sidecar','final_metadata','final_metadata_sidecar','stage_index','stage_index_sidecar'}
    if payload['finalized'] and not required_final<=set(payload['references']):
        raise ValueError('finalized checkpoint is missing final artifacts')
    if payload['teacher_identity']['maze']!=MazeTeacher().source_snapshot_id:
        raise ValueError('incompatible Maze algorithm identity')
    descriptor=verify_reference(root,payload['references']['maze_teacher'])
    if descriptor.read_bytes()!=MazeTeacher().descriptor_bytes(): raise ValueError('Maze descriptor/source mismatch')
    if payload['teacher_identity']['fourrooms']!=payload['references']['fourrooms_teacher']['sha256'] or payload['data_hash']!=payload['references']['manifest']['sha256']:
        raise ValueError('teacher/data identity mismatch')
    for stage in completed:
        key=stage['key']; path=verify_reference(root,payload['stage_snapshots'][key])
        # References seal both weights and identity metadata. Every phase must be independently deployable.
        for suffix,name in [('.sha256.json','sidecar'),('.metadata.json','metadata'),('.metadata.json.sha256.json','metadata_sidecar')]:
            ref=payload['references'].get(key+'/'+name)
            if ref is None or verify_reference(root,ref)!=Path(str(path)+suffix):
                raise ValueError('missing/mismatched stage snapshot metadata')
        metadata=json.loads(Path(str(path)+'.metadata.json').read_text(encoding='utf-8'))
        expected_identity={k:stage[k] for k in ('visit','task','phase')}
        expected_identity.update(stage=key,policy_type='active' if stage['phase']=='P' else 'kb',
                                 stage_counters=payload['stage_counters'][key],manifest=payload['references']['manifest'])
        if any(metadata.get(k)!=v for k,v in expected_identity.items()) or (path.parent/metadata['run_root']).resolve()!=root:
            raise ValueError('stage snapshot identity mismatch')
    if payload['finalized']:
        index,index_root=load_stage_index(verify_reference(root,payload['references']['stage_index']))
        expected_index=[{**s,'policy_type':'active' if s['phase']=='P' else 'kb',
                         'counters':payload['stage_counters'][s['key']],'reference':payload['stage_snapshots'][s['key']]} for s in stages]
        if index_root!=root or index['config_hash']!=payload['config_hash'] or index['manifest']!=payload['references']['manifest'] or index['stages']!=expected_index:
            raise ValueError('final stage index mismatch')
    return payload,root
