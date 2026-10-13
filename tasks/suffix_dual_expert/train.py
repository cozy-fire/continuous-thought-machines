"""Sequential Maze suffix -> Maze Fisher -> mixed Compress, with explicit resume."""
import argparse
from dataclasses import replace
import datetime
import json
from pathlib import Path
import time
import shutil
from importlib.metadata import version

import numpy as np
import torch
from .config import Config, MODEL_SPEC, load_config, path
from .envs import MapCache, MazeEnv, FourRoomsEnv, BatchEnv, training_entries, evaluation_entries, file_sha
from .model import Policy, frozen_copy, load_teacher
from .suffix import update as suffix_update
from .fisher import estimate, validate_fisher, fisher_hash, on_device
from .compress import Collector, update_pair
from .checkpoint import (atomic_json, tensor_state, model_hash, seed_everything, capture_rng,
                         restore_rng, source_hashes, save_boundary, load_boundary, snapshot, trim_metrics, tensor_hash)


def adam(policy: Policy, config: Config) -> torch.optim.Adam:
    return torch.optim.Adam(policy.parameters(),lr=config.lr,betas=config.betas,eps=config.eps,weight_decay=0.)


def preflight(config: Config) -> tuple[dict,list[dict],MapCache]:
    teacher = path(config.teacher_checkpoint)
    if file_sha(teacher) != config.teacher_sha256:
        raise ValueError('FourRooms teacher SHA mismatch')
    artifact = torch.load(teacher,map_location='cpu',weights_only=True)
    teacher_config = json.loads(json.dumps(artifact['config']))
    if artifact.get('policy_type') != 'dual' or not bool(artifact['state_dict'].get('kb_ready',False)) or any(
        teacher_config.get(k) != v for k,v in MODEL_SPEC.items()):
        raise ValueError('incompatible complete Dual teacher')
    entries = training_entries(path(config.maze_root),config.pool_sizes[-1])
    tests = evaluation_entries(path(config.test_manifest),config.eval_maze,entries)
    cache = MapCache(path(config.maze_root),entries)
    MapCache(path(config.maze_root),tests)  # Decode and verify the fixed test identities too.
    identity = dict(config=config.resolved(),source=source_hashes(),train_maps=entries,
                    test_manifest_sha256=file_sha(path(config.test_manifest)),teacher_sha256=file_sha(teacher),
                    dependencies={name:version(name) for name in ('torch','numpy','Pillow','PyYAML','gymnasium','minigrid')})
    return identity,entries,cache


def verify_counters(saved: dict, config: Config) -> None:
    schedule = config.schedule()
    stage = saved['stage_index']
    if not 0 <= stage <= len(schedule) or saved['maze_updates'] != sum(s['updates'] for s in schedule[:stage]):
        raise ValueError('Maze boundary counters disagree with the curriculum')
    if saved['completed_stages'] != schedule[:stage]:
        raise ValueError('committed Maze stage inventory mismatch')
    if not 0 <= saved['windows'] <= config.compress_windows:
        raise ValueError('invalid Compress window count')
    if saved['phase'] not in ('maze','fisher','compress','complete'):
        raise ValueError('unknown boundary phase')
    if saved['phase'] != 'maze' and stage != len(schedule):
        raise ValueError('later phase before Maze completion')
    if saved['phase'] in ('maze','fisher') and saved['windows']:
        raise ValueError('Compress counters before Compress')
    if saved['phase'] == 'complete' and saved['windows'] != config.compress_windows:
        raise ValueError('incomplete final window budget')
    if saved['optimizer'] is not None:
        expected = saved['maze_updates'] if saved['phase'] == 'maze' else config.groups*saved['windows']
        steps = [int(v['step']) for v in saved['optimizer']['state'].values()]
        if expected and (not steps or any(n != expected for n in steps)):
            raise ValueError('Adam steps disagree with committed updates')


def fork_compress_boundary(config: Config, output: Path | None, source: Path, identity: dict, student: Policy) -> dict:
    """Copy an existing F-completed boundary; only the C-stage EWC flag may differ."""
    source = source.resolve()
    marker = source/'checkpoints/latest.json'
    ref = json.loads(marker.read_text())
    protocol = json.loads((source/'protocol.json').read_text())
    original_ewc = protocol['identity']['config']['ewc_enabled']
    if not isinstance(original_ewc,bool):
        raise ValueError('source EWC flag must be boolean')
    expected = dict(identity,config=dict(identity['config'],ewc_enabled=original_ewc))
    saved = load_boundary(source,expected)
    if json.loads(marker.read_text()) != ref:
        raise ValueError('source boundary advanced while reading; pause it at F completion')
    if saved['phase'] != 'compress' or saved['windows'] != 0 or saved['optimizer'] is not None or saved['collectors'] is not None:
        raise ValueError('fork requires F-completed boundary before the first Compress update')
    verify_counters(saved,config)
    statistics = saved['fisher_statistics']
    if (statistics['decisions'] != config.fisher_steps or statistics['scores'] != config.fisher_samples
        or statistics['optimizer_updates'] != 0 or not statistics['weights_unchanged']
        or saved['fisher']['samples'] != config.fisher_samples):
        raise ValueError('source Fisher stage is not complete')
    student.load_state_dict(saved['student'],strict=True)
    validate_fisher(student,saved['fisher'])
    files = []
    snapshot_root = (source/'snapshots').resolve()
    for snapshot_ref in saved['snapshots'].values():
        file = (snapshot_root/snapshot_ref['path']).resolve()
        if file.parent != snapshot_root or file_sha(file) != snapshot_ref['sha256']:
            raise ValueError('source snapshot path or SHA mismatch')
        files.append((file,snapshot_ref))
    anchor = saved['snapshots']['maze_before_compress']
    anchor_weights = torch.load(snapshot_root/anchor['path'],map_location='cpu',weights_only=True)['state_dict']
    if tensor_hash(anchor_weights) != tensor_hash(saved['student']):
        raise ValueError('source student differs from the F-completed KB snapshot')
    origin = dict(source_run=str(source),boundary_sha256=ref['sha256'],boundary_file=ref['path'],
                  source_ewc_enabled=original_ewc,ewc_enabled=config.ewc_enabled,
                  student_sha256=model_hash(student),fisher_sha256=fisher_hash(saved['fisher']),
                  maze_anchor_sha256=anchor['sha256'],reused_maze_adam=saved['maze_updates'],
                  reused_fisher_decisions=statistics['decisions'],fresh_compress_adam=True)
    saved['identity'] = identity
    saved['compress_origin'] = origin
    if output is None:
        return saved  # Read-only dry-run; no source/destination artifacts change.
    output.mkdir(parents=True,exist_ok=False)
    (output/'snapshots').mkdir()
    for file,snapshot_ref in files:
        destination = output/'snapshots'/snapshot_ref['path']
        shutil.copyfile(file,destination)
        if file_sha(destination) != snapshot_ref['sha256']:
            raise ValueError('copied snapshot SHA mismatch')
    save_boundary(output,saved)
    atomic_json(output/'compress_origin.json',origin)
    return saved


def run(config: Config, output: Path, resume: bool = False, stop_after_boundaries: int | None = None,
        from_fisher: Path | None = None) -> dict:
    if resume and from_fisher is not None:
        raise ValueError('resume and from_fisher are mutually exclusive')
    identity,entries,cache = preflight(config)
    if config.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by this configuration')
    output = output.resolve()
    seed_everything(config.seed)
    student = Policy(config.encoder_chunk,config.checkpoint_encoder).to(config.device).train()
    streams = dict(prefix=np.random.default_rng(config.seed),action=torch.Generator().manual_seed(config.seed+101),
                   mini=np.random.default_rng(config.seed+102),fisher=np.random.default_rng(config.seed+103))
    if resume:
        saved = load_boundary(output,identity)
        verify_counters(saved,config)
        student.load_state_dict(saved['student'],strict=True)
        trim_metrics(output/'metrics.jsonl',saved['maze_updates'],saved['windows'])
        restore_rng(saved['rng'],streams)
    else:
        if from_fisher is not None:
            saved = fork_compress_boundary(config,output,from_fisher,identity,student)
        else:
            output.mkdir(parents=True,exist_ok=False)
            (output/'snapshots').mkdir()
            saved = dict(schema='suffix_dual_expert_v1',identity=identity,phase='maze',stage_index=0,
                         maze_updates=0,windows=0,completed_stages=[],suffix_decisions=0,candidates=0,
                         fisher=None,fisher_statistics=None,snapshots={},collectors=None,optimizer=None)
        atomic_json(output/'protocol.json',dict(schema='suffix_dual_expert_v1',identity=identity,model=MODEL_SPEC,
                    stages=config.schedule(),budget=config.budget(),prefix='argmax_equals_single_bfs_label',
                    suffix='last5_full_bptt',compress=('0.5_mean_kl_each_task_plus_ewc_once'
                        if config.ewc_enabled else '0.5_mean_kl_each_task'),ewc_enabled=config.ewc_enabled,
                    ewc_lambda=config.ewc_lambda if config.ewc_enabled else 0.,
                    evaluation='independent_recurrent_argmax_max300',formal= config.resolved() == Config().resolved(),
                    compress_origin=saved.get('compress_origin')))
    committed = 0

    def commit(optimizer=None, collectors=None) -> bool:
        nonlocal committed
        saved.update(student=tensor_state(student),optimizer=optimizer.state_dict() if optimizer is not None else None,
                     collectors=[c.capture() for c in collectors] if collectors else None,rng=capture_rng(streams))
        save_boundary(output,saved)
        committed += 1
        paused = stop_after_boundaries is not None and committed >= stop_after_boundaries and saved['phase'] != 'complete'
        atomic_json(output/'status.json',dict(status='paused_at_boundary' if paused else 'training',
                    phase=saved['phase'],maze_updates=saved['maze_updates'],windows=saved['windows']))
        return paused

    if not resume and from_fisher is None:
        optimizer = adam(student,config)
        saved.update(student=tensor_state(student),optimizer=optimizer.state_dict(),rng=capture_rng(streams))
        save_boundary(output,saved)  # Initial committed boundary permits retrying the first pool.
    try:
        with (output/'metrics.jsonl').open('a',encoding='utf-8') as log:
            if saved['phase'] == 'maze':
                optimizer = adam(student,config)
                optimizer.load_state_dict(saved['optimizer'])
                images,labels,transitions,map_ids = cache.tables()
                tables = tuple(torch.from_numpy(v).to(config.device) for v in (images,labels,transitions))
                del images,labels,transitions
                for stage in config.schedule()[saved['stage_index']:]:
                    pool_states = int((map_ids < stage['maps']).sum())
                    for update in range(1,stage['updates']+1):
                        stats = suffix_update(student,optimizer,tables,streams['prefix'],pool_states,stage['length'],config)
                        saved['maze_updates'] += 1
                        saved['suffix_decisions'] += stats['valid_suffix_decisions']; saved['candidates'] += stats['candidates']
                        row = dict(phase='maze',**stage,pool_update=update,maze_update=saved['maze_updates'],**stats)
                        log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
                        if update == 1 or update % 50 == 0:
                            print('MAZE',json.dumps(row),flush=True)
                    saved['stage_index'] += 1; saved['completed_stages'].append(stage)
                    if stage['maps'] == config.pool_sizes[-1]:
                        name = f"maze_length{stage['length']}.pt"
                        saved['snapshots'][f"maze_length{stage['length']}"] = snapshot(output/'snapshots'/name,student,
                            config.resolved(),prediction_length=stage['length'],maze_adam_updates=saved['maze_updates'])
                    saved['phase'] = 'fisher' if saved['stage_index'] == len(config.schedule()) else 'maze'
                    if commit(optimizer if saved['phase'] == 'maze' else None):
                        return {'status':'paused_at_boundary'}
                del tables,optimizer
            if saved['phase'] == 'fisher':
                envs = BatchEnv([MazeEnv(cache,entries,config.seed+1000+i) for i in range(2*config.slots_per_task)])
                try:
                    saved['fisher'],saved['fisher_statistics'] = estimate(student,envs,config,streams['action'],streams['fisher'])
                finally:
                    envs.close()
                saved['snapshots']['maze_before_compress'] = snapshot(output/'snapshots/maze_before_compress.pt',student,
                    config.resolved(),prediction_length=config.lengths[-1],maze_adam_updates=saved['maze_updates'])
                saved['phase'] = 'compress'
                print('FISHER',json.dumps(saved['fisher_statistics']),flush=True)
                if commit():
                    return {'status':'paused_at_boundary'}
            if saved['phase'] == 'compress':
                validate_fisher(student,saved['fisher'])
                anchor = saved['snapshots']['maze_before_compress']
                source = output/'snapshots'/anchor['path']
                if file_sha(source) != anchor['sha256']:
                    raise ValueError('Maze expert snapshot identity mismatch')
                maze_teacher = frozen_copy(student)
                maze_teacher.load_state_dict(torch.load(source,map_location='cpu',weights_only=True)['state_dict'],strict=True)
                four_teacher = load_teacher(path(config.teacher_checkpoint),config.device,config.encoder_chunk)
                expected_teacher,expected_maze,expected_fisher = model_hash(four_teacher),model_hash(maze_teacher),fisher_hash(saved['fisher'])
                gpu_fisher = on_device(saved['fisher'],config.device) if config.ewc_enabled else None
                optimizer = adam(student,config)
                if saved['optimizer'] is not None:
                    optimizer.load_state_dict(saved['optimizer'])
                collectors = [Collector(BatchEnv([MazeEnv(cache,entries,config.seed+1000+i) for i in range(config.slots_per_task)]),
                                       maze_teacher,student,streams['action']),
                              Collector(BatchEnv([FourRoomsEnv(config.seed+3000+i) for i in range(config.slots_per_task)]),
                                       four_teacher,student,streams['action'])]
                if saved['collectors'] is not None:
                    for collector,value in zip(collectors,saved['collectors']):
                        collector.restore(value)
                # Construction consumes RNG; the committed boundary defines the next window.
                restore_rng(saved['rng'],streams)
                try:
                    for index in range(saved['windows'],config.compress_windows):
                        begin = time.perf_counter()
                        batches = [c.collect(config.window) for c in collectors]
                        stats = update_pair(student,batches,optimizer,gpu_fisher,streams['mini'],config)
                        for collector in collectors:
                            collector.truncate()
                        saved['windows'] = index+1
                        row = dict(phase='compress',window=index+1,compress_adam=(index+1)*config.groups,
                                   decisions=(index+1)*2*config.slots_per_task*config.window,
                                   maze_decisions=(index+1)*config.slots_per_task*config.window,
                                   fourrooms_decisions=(index+1)*config.slots_per_task*config.window,
                                   compress_microbatch=config.compress_microbatch,
                                   checkpoint_encoder=student.checkpoint_encoder,encoder_chunk=config.encoder_chunk,
                                   elapsed_seconds=time.perf_counter()-begin,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),**stats)
                        log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
                        del batches
                        if (index+1) % config.save_windows == 0 or index+1 == config.compress_windows:
                            print('COMPRESS',json.dumps(row),flush=True)
                            if commit(optimizer,collectors):
                                return {'status':'paused_at_boundary'}
                    unchanged = (model_hash(four_teacher)==expected_teacher and model_hash(maze_teacher)==expected_maze
                                 and fisher_hash(saved['fisher'])==expected_fisher)
                    if not unchanged:
                        raise RuntimeError('Compress changed a teacher or the fixed Fisher')
                    saved['snapshots']['final_kb'] = snapshot(output/'snapshots/final_kb.pt',student,config.resolved(),
                        prediction_length=config.lengths[-1],maze_adam_updates=saved['maze_updates'],
                        compress_adam_updates=config.groups*saved['windows'])
                    saved['teacher_unchanged'] = True; saved['maze_teacher_unchanged'] = True; saved['fisher_unchanged'] = True
                    saved['phase'] = 'complete'
                    commit(optimizer,collectors)
                finally:
                    for collector in collectors:
                        collector.envs.close()
            verify_counters(saved,config)
            summary = dict(status='complete',budget=config.budget(),maze_adam_updates=saved['maze_updates'],
                           compress_origin=saved.get('compress_origin'),
                           ewc_enabled=config.ewc_enabled,ewc_lambda=config.ewc_lambda if config.ewc_enabled else 0.,
                           compress_adam_updates=config.groups*saved['windows'],windows=saved['windows'],
                           valid_suffix_decisions=saved['suffix_decisions'],candidate_sequences=saved['candidates'],
                           completed_stages=saved['completed_stages'],fisher=saved['fisher_statistics'],
                           snapshots=saved['snapshots'],teacher_unchanged=saved['teacher_unchanged'],
                           maze_teacher_unchanged=saved['maze_teacher_unchanged'],fisher_unchanged=saved['fisher_unchanged'])
            atomic_json(output/'summary.json',summary)
            atomic_json(output/'status.json',dict(status='complete',phase='complete'))
            print('ALL_COMPLETE',json.dumps(config.budget()),flush=True)
            return summary
    except BaseException as error:
        atomic_json(output/'status.json',dict(status='failed',phase=saved['phase'],error=repr(error),
                    maze_updates=saved['maze_updates'],windows=saved['windows']))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--out',type=Path)
    start = parser.add_mutually_exclusive_group()
    start.add_argument('--resume',action='store_true')
    start.add_argument('--from-fisher',type=Path,help='New Compress run from a source run paused at F completion')
    parser.add_argument('--ewc',choices=('on','off'),help='Compress EWC override; ordinary resume must retain its setting')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--stop-after-boundaries',type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.ewc is not None:
        config = replace(config,ewc_enabled=args.ewc=='on')
    if args.dry_run:
        identity,entries,_ = preflight(config)
        origin = None
        if args.from_fisher is not None:
            origin = fork_compress_boundary(config,None,args.from_fisher,identity,
                Policy(config.encoder_chunk,config.checkpoint_encoder))['compress_origin']
        print(json.dumps(dict(config=config.resolved(),stages=config.schedule(),budget=config.budget(),
                              teacher_sha256=identity['teacher_sha256'],training_maps=len(entries),compress_origin=origin),indent=2))
        return
    if args.out is None or args.stop_after_boundaries is not None and args.stop_after_boundaries < 1:
        parser.error('--out required; boundary count must be positive')
    run(config,args.out,args.resume,args.stop_after_boundaries,from_fisher=args.from_fisher)


if __name__ == '__main__':
    main()
