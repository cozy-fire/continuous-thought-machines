"""Real two-visit acceptance, boundary resume, inference and media verification."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
from PIL import Image

from .checkpoint import atomic_json, load_boundary, verify_reference
from .config import REPO_ROOT, load_config
from .evaluate import evaluate_policy, load_for_evaluation
from .runner import run
from .schedule import expand_stages
from .visualize import visualize


def file_inventory(root: Path) -> dict:
    """Hash files incrementally; never materialize a checkpoint in a second buffer."""
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'inventory.json':
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            result[path.relative_to(root).as_posix()] = {'size': path.stat().st_size, 'sha256': digest.hexdigest()}
    return result


def audit_run(root: Path, config, seed: int) -> dict:
    payload, _ = load_boundary(root/'checkpoints/latest.json', config, seed)
    stages = expand_stages(config)
    if not payload['finalized'] or payload['next_index'] != 12 or len(stages) != 12:
        raise ValueError('incomplete two-visit smoke')
    actual=sum(x['transitions'] for x in payload['stage_counters'].values())
    expected_updates=sum(s.optimizer_updates or (0 if s.phase=='F' else
        ((s.env_steps//config.training.num_envs+config.optimization.learning_steps-1)//config.optimization.learning_steps)*config.optimization.minibatches)
        for s in stages)
    if payload['global_env_steps'] != actual or payload['optimizer_updates'] != expected_updates:
        raise ValueError('unexpected smoke budget or update count')
    events = [json.loads(line) for line in (root/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    completed = [e for e in events if e['event'] == 'stage_complete']
    if len(completed) != 12 or [e['stage'] for e in completed] != [s.key for s in stages]:
        raise ValueError('missing, duplicate or out-of-order stages')
    for stage, event in zip(stages, completed):
        if event.get('ticks') != config.ctm.ticks_by_task.for_task(stage.task) or event.get('memory_ticks') != 40:
            raise ValueError('stage execution budget is wrong')
        stats = event['statistics']
        if stage.phase == 'F':
            if stats['scored_samples'] != 8 or len(set(stats['selected_ids'])) != 8:
                raise ValueError('Fisher IDs are incomplete/nonunique')
        else:
            windows = [e for e in events if e['event'] == 'training' and e['stage'] == stage.key]
            expected_steps=payload['stage_counters'][stage.key]['transitions']
            if not windows or windows[-1]['eligible_target_steps'] != expected_steps:
                raise ValueError('first-step teaching or final target count is wrong')
            if windows[-1]['optimizer_updates'] != (stage.optimizer_updates or (3 if stage.phase == 'P' else 2)):
                raise ValueError('unexpected stage optimizer count')
    if (root/'evaluation').exists() or {'evaluations','visit_reports'} & set(payload) or any(
            e['event'] in ('evaluation','final_test') or e.get('split') in ('validation','test') for e in events):
        raise ValueError('training must not run evaluation')
    if len(payload['stage_snapshots'])!=len(stages): raise ValueError('missing stage inference weights')
    policy, _, _, _ = load_for_evaluation(root/'exports/final.pt', device=config.training.device)
    if set(payload['fisher'].importance) != set(dict(policy.named_parameters())):
        raise ValueError('Fisher does not cover the complete model')
    return {'stages': 12, 'transitions': actual, 'updates': expected_updates, 'stage_snapshots':12,'evaluations_in_training':0,
            'ticks_by_task': asdict(config.ctm.ticks_by_task),
            'memory_ticks': config.ctm.memory_length,
            'fisher_parameters': len(payload['fisher'].importance), 'finalized': True}


def verify(output_dir, device='cuda:0', seed=0, compile_mode='disabled'):
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    config = load_config(REPO_ROOT/'tasks/continual_nav_opd/configs/smoke.yaml')
    config = replace(config, training=replace(config.training, device=device),
                     optimization=replace(config.optimization,ctm_compile=compile_mode))
    start = time.perf_counter()
    commands = [('git_commit', ['git', '-c', f'safe.directory={REPO_ROOT.as_posix()}', 'rev-parse', 'HEAD']),
                ('git_dirty', ['git', '-c', f'safe.directory={REPO_ROOT.as_posix()}', 'status', '--short'])]
    environment = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                   'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'device': device,
                   'gpu': torch.cuda.get_device_name(device) if device.startswith('cuda') else None}
    for name, command in commands:
        value = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        environment[name] = value.stdout.strip()
    atomic_json(root/'environment.json', environment)
    try:
        peaks = []
        def capture_peak():
            if device.startswith('cuda'):
                peaks.append((torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)))
        # Exercise a REAL P boundary resume. Already committed pools/stages must not run again.
        class PoolInterruption(RuntimeError): pass
        def stop_after_pool(point,stage):
            if point=='pool_complete': raise PoolInterruption('intentional pool-boundary power loss')
        try:
            run(config,seed,root/'run',hook=stop_after_pool)
        except PoolInterruption:
            partial,_=load_boundary(root/'run/checkpoints/latest.json',config,seed)
            if partial['maze_progress']['next_pool']!=1 or partial['next_index']!=0:
                raise ValueError('pool boundary not committed')
        else:
            raise ValueError('pool fault was not injected')
        run(config, seed, root/'run', resume=True, max_stages=1)
        capture_peak()
        run(config, seed, root/'run', resume=True)
        capture_peak()
        report = audit_run(root/'run', config, seed)
        before = file_inventory(root/'run')
        run(config, seed, root/'run', resume=True)
        if before != file_inventory(root/'run'):
            raise ValueError('finalized resume rewrote committed artifacts')
        training_seconds=time.perf_counter()-start
        # Local evaluation is a separate invocation after training has finalized. Never writes into run.
        offline=root/'offline_evaluation'; offline.mkdir()
        stage_reports={}
        for stage_index,stage in enumerate(expand_stages(config)):
            identity={}
            kind='active' if stage.phase=='P' else 'kb'
            snapshot,_,manifest,_=load_for_evaluation(root/'run/exports/stages.json',kind,device,stage.key,identity=identity)
            evaluation=evaluate_policy(snapshot,config,manifest,identity=identity,backend='serial')
            if any(m['episodes']!=config.evaluation.validation_episodes for m in evaluation['tasks'].values()):
                raise ValueError('incomplete offline panel')
            name=f'{stage_index:03d}.json'
            atomic_json(offline/name,evaluation)
            stage_reports[stage.key]={task:m['success_rate'] for task,m in evaluation['tasks'].items()}
            del snapshot
        policy, _, manifest, _ = load_for_evaluation(root/'run/exports/final.pt', device=device)
        serial = evaluate_policy(policy, config, manifest, split='test',backend='serial')
        parallel = evaluate_policy(policy, config, manifest, split='test',backend='subprocess')
        for task in config.task_order:
            if serial['tasks'][task]['results'] != parallel['tasks'][task]['results']:
                raise ValueError('serial/subprocess argmax episode divergence; inspect separately')
        atomic_json(offline/'serial.json', serial)
        atomic_json(offline/'subprocess.json', parallel)
        for task in config.task_order:
            destination = root/f'visualization_{task}'
            key = f'pnc/v0/{task}/P'
            visualize(root/'run/exports/stages.json', 'active', task, 0, destination,
                      device=device, stage_key=key, teacher_diagnostics=True)
            rows = list(csv.DictReader((destination/'steps.csv').open(encoding='utf-8', newline='')))
            trajectory = json.loads((destination/'trajectory.json').read_text(encoding='utf-8'))
            with Image.open(destination/'behavior.gif') as image:
                if image.n_frames != len(rows)+1 or trajectory['actions'] != len(rows):
                    raise ValueError('media/CSV length mismatch')
                image.seek(image.n_frames-1); image.load()
            with Image.open(destination/'reward.png') as image:
                image.verify()
        capture_peak()
        if before!=file_inventory(root/'run'): raise ValueError('offline evaluation changed training artifacts')
        # run() resets CUDA peak counters on every resume, including a finalized no-op.
        # Preserve each real training segment's peaks BEFORE testing idempotent resume.
        report.update(status='passed', elapsed_seconds=time.perf_counter()-start,training_seconds=training_seconds,
                      offline_validation=stage_reports,offline_evaluation_preserved_training=True,
                      peak_allocated_bytes=max(p[0] for p in peaks) if peaks else None,
                      peak_reserved_bytes=max(p[1] for p in peaks) if peaks else None,
                      limits='Engineering smoke only; no learning-success or RTX5090 throughput claim. W&B online not tested.')
        atomic_json(root/'report.json', report)
        atomic_json(root/'inventory.json', file_inventory(root))
        print(json.dumps(report), flush=True)
        return report
    except Exception as exc:
        atomic_json(root/'failure.json', {'error': repr(exc), 'elapsed_seconds': time.perf_counter()-start})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ctm-compile',choices=['disabled','default','reduce-overhead'],default='disabled')
    args = parser.parse_args()
    verify(args.output_dir, args.device, args.seed,args.ctm_compile)


if __name__ == '__main__':
    main()
