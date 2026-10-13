"""Independent recurrent-only paired testing, never learning or prefix filtering."""
import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from .config import Config, path
from .envs import MapCache, MazeEnv, FourRoomsEnv, evaluation_entries, file_sha
from .models import VisualPolicy as Policy, build_policy, load_teacher, state_kind
from .checkpoint import atomic_json, model_hash, seed_everything


def policy_config(artifact: dict) -> Config:
    schema = artifact.get('schema')
    if schema not in ('suffix_dual_expert_policy_v1','suffix_dual_expert_policy_v2') or artifact.get('model') != 'single':
        raise ValueError('not a single-column inference snapshot')
    config = Config(**{k:tuple(v) if k in ('lengths','pool_sizes','first_updates','suffix_updates','betas') else v
                       for k,v in artifact['config'].items()})
    if schema == 'suffix_dual_expert_policy_v1':
        if config.model_type != 'ctm' or artifact.get('model_type','ctm') != 'ctm':
            raise ValueError('legacy snapshots support CTM only')
    elif artifact.get('model_type') != config.model_type or artifact.get('model_spec') != config.model_spec():
        raise ValueError('snapshot model architecture/spec mismatch')
    return config


def load_policy(file: Path, device: str, expected_config: Config | None = None) -> Policy:
    artifact = torch.load(file,map_location='cpu',weights_only=True)
    config = policy_config(artifact)
    if expected_config is not None and (config.model_type != expected_config.model_type
                                        or config.model_spec() != expected_config.model_spec()):
        raise ValueError('evaluation model architecture/spec mismatch')
    model = build_policy(config)
    model.load_state_dict(artifact['state_dict'],strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite policy weights')
    return model.to(device).eval().requires_grad_(False)


def wilson(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    p = successes/count; d = 1+z*z/count
    center = (p+z*z/(2*count))/d
    margin = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/d
    return [center-margin,center+margin]


def paired(left: dict, right: dict, seed: int = 0) -> dict:
    a,b = left['episodes'],right['episodes']
    if len(a) != len(b) or [(e['identity'],e['initial_position'],e.get('initial_direction')) for e in a] != [
        (e['identity'],e['initial_position'],e.get('initial_direction')) for e in b]:
        raise ValueError('paired identities or initial conditions differ')
    delta = np.asarray([int(y['success'])-int(x['success']) for x,y in zip(a,b)],dtype=np.int8)
    before_only,after_only = int((delta == -1).sum()),int((delta == 1).sum())
    discordant = before_only+after_only
    p = min(1.,2*sum(math.comb(discordant,k) for k in range(min(before_only,after_only)+1))/(1<<discordant)) if discordant else 1.
    rng = np.random.default_rng(seed)
    samples = np.concatenate([delta[rng.integers(len(delta),size=(1000,len(delta)))].mean(1) for _ in range(20)])
    return dict(before_rate=left['success_rate'],after_rate=right['success_rate'],
                difference_pp=100*float(delta.mean()),before_only=before_only,after_only=after_only,
                paired_bootstrap95_pp=(100*np.quantile(samples,[.025,.975])).tolist(),
                mcnemar_exact_p=p)


def holm(comparisons: dict) -> None:
    ordered = sorted(comparisons,key=lambda k:comparisons[k]['mcnemar_exact_p'])
    previous = 0.
    for i,name in enumerate(ordered):
        adjusted = min(1.,max(previous,(len(ordered)-i)*comparisons[name]['mcnemar_exact_p']))
        comparisons[name]['mcnemar_holm_p'] = adjusted
        previous = adjusted


def rollout(policy, task: str, panel: list, cache: MapCache | None, output: Path, batch: int) -> dict:
    device = next(policy.parameters()).device
    before = model_hash(policy)
    episodes = []
    output.mkdir(parents=True,exist_ok=False)
    with (output/'trajectories.csv').open('x',newline='',encoding='utf-8') as stream,torch.inference_mode():
        writer = csv.writer(stream)
        writer.writerow(['episode','step','action','position_before','position_after','done','success'])
        for first in range(0,len(panel),batch):
            portion = panel[first:first+batch]
            envs = [MazeEnv(cache,[entry],0) if task == 'maze_medium' else FourRoomsEnv(0,int(entry)) for entry in portion]
            try:
                images = [env.reset() for env in envs]
                initial = [list(env.position) for env in envs]
                directions = [None if task == 'maze_medium' else int(env.native.unwrapped.agent_dir) for env in envs]
                state = policy.initial_state(len(envs))
                done = np.zeros(len(envs),dtype=bool); success = done.copy(); steps = np.zeros(len(envs),dtype=int)
                for step in range(300):
                    rgb = torch.from_numpy(np.stack(images)).to(device)
                    starts = torch.full((len(envs),),step==0,dtype=torch.bool,device=device)
                    logits,state = policy.step(rgb,state,starts,task=task)
                    if not torch.isfinite(logits).all():
                        raise ValueError('nonfinite evaluation logits')
                    actions = logits.argmax(-1).cpu().tolist()
                    for i,env in enumerate(envs):
                        if done[i]:
                            continue
                        old = list(env.position)
                        images[i],done[i],success[i] = env.step(actions[i])
                        steps[i] += 1
                        writer.writerow([first+i,step+1,actions[i],old,list(env.position),bool(done[i]),bool(success[i])])
                    if done.all():
                        break
                for i,entry in enumerate(portion):
                    episodes.append(dict(panel_index=first+i,identity=entry['sha256'] if task=='maze_medium' else entry,
                        initial_position=initial[i],initial_direction=directions[i],success=bool(success[i]),steps=int(steps[i])))
            finally:
                for env in envs:
                    env.close()
            atomic_json(output/'status.json',dict(status='evaluating',episodes=len(episodes),task=task))
            print('EVALUATE',task,first+len(portion),'/',len(panel),flush=True)
    if model_hash(policy) != before:
        raise RuntimeError('evaluation changed weights')
    successes = sum(e['success'] for e in episodes)
    result = dict(model_type=state_kind(policy).replace('_dual',''),
                  episodes=episodes,episode_count=len(episodes),successes=successes,
                  success_rate=successes/len(episodes),wilson95=wilson(successes,len(episodes)),weights_unchanged=True)
    atomic_json(output/'summary.json',result)
    return result


def evaluate(run: Path, device: str | None = None, episodes: int | None = None) -> dict:
    run = run.resolve()
    training = json.loads((run/'summary.json').read_text())
    protocol = json.loads((run/'protocol.json').read_text())
    if training['status'] != 'complete' or not all(training[k] for k in ('teacher_unchanged','maze_teacher_unchanged','fisher_unchanged')):
        raise ValueError('training is not verified complete')
    raw = protocol['identity']['config']
    config = Config(**{k:tuple(v) if k in ('lengths','pool_sizes','first_updates','suffix_updates','betas') else v for k,v in raw.items()})
    if protocol.get('schema') == 'suffix_dual_expert_v2' and (
        protocol.get('model_type') != config.model_type or protocol.get('model') != config.model_spec()
        or training.get('model_type') != config.model_type or training.get('model_spec') != config.model_spec()):
        raise ValueError('training model architecture/spec mismatch')
    if training['budget'] != config.budget() or training['maze_adam_updates'] != config.budget()['maze_adam'] or training['compress_adam_updates'] != config.budget()['compress_adam']:
        raise ValueError('training budget mismatch')
    device = device or config.device
    seed_everything(config.seed)
    maze_count,rooms_count = config.eval_maze,config.eval_fourrooms
    if episodes is not None:
        if not 1 <= episodes <= min(maze_count,rooms_count):
            raise ValueError('episode override must be a smaller diagnostic subset')
        maze_count = rooms_count = episodes
    manifest = path(config.test_manifest)
    if file_sha(manifest) != protocol['identity']['test_manifest_sha256']:
        raise ValueError('test manifest changed since training')
    entries = evaluation_entries(manifest,maze_count,protocol['identity']['train_maps'])
    cache = MapCache(path(config.maze_root),entries)
    seeds = list(range(3000000,3000000+rooms_count))
    output = run/'evaluation'
    output.mkdir(exist_ok=False)
    begin = time.perf_counter()
    hashes = {}
    for name in ('maze_before_compress','final_kb'):
        ref = training['snapshots'][name]; file = run/'snapshots'/ref['path']
        if file_sha(file) != ref['sha256']:
            raise ValueError('snapshot SHA mismatch')
        hashes[name] = ref['sha256']
    if file_sha(path(config.teacher_checkpoint)) != config.teacher_sha256:
        raise ValueError('teacher changed since training')
    atomic_json(output/'protocol.json',dict(model_type=config.model_type,model_spec=config.model_spec(),
        maps=entries,fourrooms_seeds=seeds,checkpoint_sha256=hashes,
        teacher_sha256=config.teacher_sha256,actions='argmax',recurrent=True,prefix_filter=False,max_steps=300,
        precision='FP32',batch=config.eval_batch,diagnostic_subset=episodes is not None))
    try:
        before = load_policy(run/'snapshots'/training['snapshots']['maze_before_compress']['path'],device,config)
        maze_before = rollout(before,'maze_medium',entries,cache,output/'maze_before',config.eval_batch)
        del before
        after = load_policy(run/'snapshots'/training['snapshots']['final_kb']['path'],device,config)
        maze_after = rollout(after,'maze_medium',entries,cache,output/'maze_after',config.eval_batch)
        rooms_after = rollout(after,'fourrooms',seeds,None,output/'fourrooms_after',config.eval_batch)
        del after
        teacher = load_teacher(path(config.teacher_checkpoint),device,config.encoder_chunk)
        rooms_teacher = rollout(teacher,'fourrooms',seeds,None,output/'fourrooms_teacher',config.eval_batch)
        comparisons = dict(maze_recurrent=paired(maze_before,maze_after),fourrooms=paired(rooms_teacher,rooms_after))
        holm(comparisons)
        summary = dict(status='complete',model_type=config.model_type,model_spec=config.model_spec(),
                       episodes=2*(maze_count+rooms_count),comparison=comparisons,
                       weights_unchanged=True,elapsed_seconds=time.perf_counter()-begin,
                       maze_before=maze_before['success_rate'],maze_after=maze_after['success_rate'],
                       fourrooms_teacher=rooms_teacher['success_rate'],fourrooms_after=rooms_after['success_rate'])
        if any(file_sha(run/'snapshots'/training['snapshots'][name]['path']) != digest for name,digest in hashes.items()):
            raise RuntimeError('snapshot file changed during evaluation')
        atomic_json(output/'summary.json',summary)
        return summary
    except BaseException as error:
        atomic_json(output/'status.json',dict(status='failed',error=repr(error)))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--device')
    parser.add_argument('--episodes',type=int,help='Smaller diagnostic subset, never the default test')
    parser.add_argument('--checkpoint',type=Path,help='Individually test a saved Maze length boundary')
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if args.checkpoint:
        if args.run or not args.out:
            parser.error('individual checkpoint testing requires --out and no --run')
        artifact = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        config = policy_config(artifact)
        seed_everything(config.seed)
        count = args.episodes or config.eval_maze
        if not 1 <= count <= config.eval_maze:
            parser.error('individual episode count outside configured test panel')
        from .envs import training_entries
        train = training_entries(path(config.maze_root),config.pool_sizes[-1])
        entries = evaluation_entries(path(config.test_manifest),count,train)
        cache = MapCache(path(config.maze_root),entries)
        policy = load_policy(args.checkpoint,args.device or config.device)
        rollout(policy,'maze_medium',entries,cache,args.out,config.eval_batch)
    elif args.run:
        print(json.dumps(evaluate(args.run,args.device,args.episodes),indent=2))
    else:
        parser.error('--run or --checkpoint required')


if __name__ == '__main__':
    main()
