"""Atomic committed boundaries; only dictionaries, scalars and tensors on disk."""
from pathlib import Path
import hashlib
import json
import os
import random
import uuid

import numpy as np
import torch
from .config import PACKAGE, ROOT
from .envs import file_sha


def atomic_json(file: Path, value: dict | list) -> None:
    file.parent.mkdir(parents=True,exist_ok=True)
    temporary = file.with_name(file.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,file)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(file: Path, value: dict) -> None:
    file.parent.mkdir(parents=True,exist_ok=True)
    temporary = file.with_name(file.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:
            torch.save(value,stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,file)
    finally:
        temporary.unlink(missing_ok=True)


def tensor_state(model) -> dict:
    return {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}


def tensor_hash(values: dict) -> str:
    digest = hashlib.sha256()
    for name,value in sorted(values.items()):
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_hash(model) -> str:
    return tensor_hash(model.state_dict())


def source_hashes() -> dict:
    files = list(PACKAGE.glob('*.py'))
    files += [ROOT/'models'/name for name in ('resnet.py','modules.py','utils.py')]
    return {p.relative_to(ROOT).as_posix():file_sha(p) for p in sorted(files)}


def seed_everything(seed: int) -> None:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def capture_rng(streams: dict) -> dict:
    legacy = np.random.get_state()
    result = dict(python=random.getstate(),numpy=[legacy[0],legacy[1].tolist(),*legacy[2:]],
                  torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
    result['streams'] = {name: value.get_state() if isinstance(value,torch.Generator)
                         else value.bit_generator.state for name,value in streams.items()}
    return result


def restore_rng(value: dict, streams: dict) -> None:
    random.setstate(value['python'])
    legacy = value['numpy']
    np.random.set_state((legacy[0],np.asarray(legacy[1],dtype=np.uint32),*legacy[2:]))
    torch.set_rng_state(value['torch'])
    if value['cuda']:
        if len(value['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from the boundary')
        torch.cuda.set_rng_state_all(value['cuda'])
    for name,stream in streams.items():
        saved = value['streams'][name]
        if isinstance(stream,torch.Generator):
            stream.set_state(saved)
        else:
            stream.bit_generator.state = saved


def save_boundary(run: Path, payload: dict) -> Path:
    directory = run/'checkpoints'
    directory.mkdir(exist_ok=True)
    latest = directory/'latest.json'
    previous = json.loads(latest.read_text()) if latest.exists() else None
    file = directory/f'boundary_{uuid.uuid4().hex}.pt'
    atomic_torch(file,payload)
    atomic_json(latest,dict(path=file.name,sha256=file_sha(file),phase=payload['phase'],
                           maze_updates=payload['maze_updates'],windows=payload['windows']))
    # Commit the new pointer before removing the superseded recovery file.
    if previous:
        old = (directory/previous['path']).resolve()
        if old.is_relative_to(directory.resolve()) and old != file.resolve():
            old.unlink(missing_ok=True)
    return file


def load_boundary(run: Path, identity: dict) -> dict:
    directory = run/'checkpoints'
    ref = json.loads((directory/'latest.json').read_text())
    file = (directory/ref['path']).resolve()
    if not file.is_relative_to(directory.resolve()) or file_sha(file) != ref['sha256']:
        raise ValueError('invalid checkpoint reference or SHA')
    payload = torch.load(file,map_location='cpu',weights_only=True)
    if payload.get('identity') != identity or payload.get('schema') != 'suffix_dual_expert_v1':
        raise ValueError('resume config/source/data/teacher mismatch')
    if payload['phase'] != ref['phase'] or payload['maze_updates'] != ref['maze_updates'] or payload['windows'] != ref['windows']:
        raise ValueError('boundary marker counters differ')
    return payload


def snapshot(file: Path, model, config: dict, **metadata) -> dict:
    atomic_torch(file,dict(schema='suffix_dual_expert_policy_v1',model='single',
                           config=config,state_dict=tensor_state(model),**metadata))
    return {'path':file.name,'sha256':file_sha(file)}


def trim_metrics(file: Path, maze_updates: int, windows: int) -> None:
    if not file.exists():
        return
    lines = []
    for line in file.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break  # An interrupted trailing write was not a committed update.
        if row['phase'] == 'maze' and row['maze_update'] <= maze_updates or row['phase'] == 'compress' and row['window'] <= windows:
            lines.append(line+'\n')
    temporary = file.with_suffix('.tmp')
    temporary.write_text(''.join(lines),encoding='utf-8'); os.replace(temporary,file)
