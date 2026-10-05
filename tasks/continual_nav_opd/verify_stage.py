"""Small, real-expert stage verification; never launch the formal training budget."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .config import REPO_ROOT,load_config,resolved_dict,config_hash
from .data import build_manifest
from .envs import SyncVectorEnv,make_env,MazeMapCache
from .data.maze_curriculum import MazeStateTable
from .learning.maze_progress import run_maze_progress
from .models import StandalonePolicy,DualPolicy,save_snapshot,load_snapshot
from .teachers import MazeTeacher,FourRoomsTeacher
from .teachers.fourrooms import file_sha256
from .learning.progress import run_progress


def write_json(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)


def state_hash(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['04','05'])
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    if args.stage=='05':
        from .verify_compress import verify
        verify(args.device,args.output_dir)
        return
    config=load_config('tasks/continual_nav_opd/configs/smoke.yaml')
    config=replace(config,training=replace(config.training,device=args.device))
    device=torch.device(args.device)
    output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    try:
        if device.type=='cuda':
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)
        manifest=build_manifest(REPO_ROOT/config.environment.maze_root,
                                validation_count=config.evaluation.maze_validation_hashes,
                                validation_episodes=config.evaluation.validation_episodes,
                                test_episodes=config.evaluation.test_episodes,drift_episodes=2)
        manifest.save(output/'maze_manifest.json')
        cache=MazeMapCache(REPO_ROOT/config.environment.maze_root,manifest.train+manifest.validation+manifest.test)
        table=MazeStateTable(cache,manifest.train[:config.maze_progress.pool_sizes[-1]])
        write_json(output/'resolved_config.json',resolved_dict(config))
        results={}
        for index,task in enumerate(config.task_order):
            folder=output/task
            folder.mkdir()
            torch.manual_seed(config.training.seed+index)
            kb=StandalonePolicy(config).to(device)
            student=DualPolicy(config,kb,kb_ready=False)
            expert=MazeTeacher() if task=='maze_medium' else FourRoomsTeacher(
                REPO_ROOT/config.teachers.fourrooms.checkpoint,device)
            if isinstance(expert,MazeTeacher):
                expert.save_descriptor(folder/'maze_teacher.json')
            else:
                write_json(folder/'fourrooms_teacher.json',expert.metadata)
            before_kb, before_active=state_hash(student.kb),state_hash(student.active)
            before_expert=expert.parameter_hash() if isinstance(expert,FourRoomsTeacher) else expert.source_snapshot_id
            envs=SyncVectorEnv([make_env(config,task,'train',manifest=manifest,seed=config.training.seed+i,map_cache=cache)
                                for i in range(config.training.num_envs)])
            events=[]
            def on_window(event):
                events.append(event)
                print(json.dumps({'task':task,'transitions':event['transitions'],
                                  'updates':event['optimizer_updates'],'kl':event['kl']}),flush=True)
            try:
                if task=='maze_medium':
                    result=run_maze_progress(table,student,config,torch.Generator().manual_seed(100+index),
                                             np.random.default_rng(200+index),on_window=on_window)
                    assert result.optimizer_updates==2 and result.eligible_target_steps==result.transitions
                else:
                    result=run_progress(envs,student,expert,config,
                                    getattr(config.pnc.task_budgets,task).progress_steps,
                                    torch.Generator(device='cpu').manual_seed(100+index),
                                    np.random.default_rng(200+index),on_window=on_window)
                    assert (result.transitions,result.eligible_target_steps,result.windows,result.optimizer_updates)==(256,256,3,3)
                    assert result.next_transition_id==256
                assert before_kb==state_hash(student.kb)
                assert before_active!=state_hash(student.active)
                after_expert=expert.parameter_hash() if isinstance(expert,FourRoomsTeacher) else expert.source_snapshot_id
                assert before_expert==after_expert
                path=folder/'active.pt'
                save_snapshot(result.policy,path)
                restored=load_snapshot(path,device)
                obs,_=envs.reset()
                rgb=torch.from_numpy(obs.student_rgb).to(device)
                with torch.no_grad():
                    first,_=result.policy.step(rgb,result.policy.initial_state(config.training.num_envs),
                                               torch.ones(config.training.num_envs,dtype=torch.bool,device=device),task=task)
                    actual,_=restored.step(rgb,restored.initial_state(config.training.num_envs),
                                          torch.ones(config.training.num_envs,dtype=torch.bool,device=device),task=task)
                torch.testing.assert_close(first,actual,atol=2e-5,rtol=1e-5)
                with (folder/'events.jsonl').open('x',encoding='utf-8') as stream:
                    for event in events:
                        stream.write(json.dumps(event,allow_nan=False)+'\n')
                results[task]={'transitions':result.transitions,'eligible_target_steps':result.eligible_target_steps,
                               'windows':result.windows,'optimizer_updates':result.optimizer_updates,
                               'next_transition_id':result.next_transition_id,'statistics':result.statistics,
                               'old_kb_unchanged':True,'expert_unchanged':True,'active_changed':True,
                               'snapshot_max_logit_difference':float((first-actual).abs().max()),
                               'expert_source_snapshot_id':expert.source_snapshot_id}
                write_json(folder/'report.json',results[task])
            finally:
                envs.close()
            del restored,result,student,kb,expert,envs
        # Curves summarize actual training windows; a short smoke cannot establish learned success.
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        for task in config.task_order:
            events=[json.loads(line) for line in (output/task/'events.jsonl').read_text().splitlines()]
            fig,ax=plt.subplots()
            x='optimizer_updates' if task=='maze_medium' else 'transitions'
            ax.plot([e[x] for e in events],[e['kl'] for e in events],marker='o')
            ax.set(xlabel='Adam updates' if task=='maze_medium' else 'Training environment transitions',ylabel='Valid-sample KL',title=task+' P smoke')
            fig.savefig(output/task/'kl.png',dpi=150,bbox_inches='tight')
            plt.close(fig)
        write_json(output/'report.json',{'status':'passed','stage':'04','method':config.method,'schema_version':4,
                   'sequence_protocol':config.sequence_protocol,'config_hash':config_hash(config),
                   'device':str(device),'hardware':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                   'torch':torch.__version__,'results':results,
                   'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                   'wandb_mode':'disabled','limits':'Real expert P-only smoke; no Compress, formal training, or learning-success claim.'})
        source_paths=list((REPO_ROOT/'tasks/continual_nav_opd').rglob('*.py'))
        source_paths += [REPO_ROOT/'models'/name for name in ('ctm.py','ctm_rl.py','modules.py','resnet.py','utils.py','constants.py')]
        source_paths += list((REPO_ROOT/'tasks/continual_nav/envs').glob('*.py'))
        source_paths += [REPO_ROOT/'tasks/continual_nav/data/manifest.py']
        write_json(output/'source_manifest.json',{p.relative_to(REPO_ROOT).as_posix():file_sha256(p) for p in sorted(source_paths)})
        write_json(output/'sha256.json',{p.relative_to(output).as_posix():{'size':p.stat().st_size,'sha256':file_sha256(p)}
                                         for p in sorted(output.rglob('*')) if p.is_file()})
        print('Stage 04 real-expert smoke passed: '+str(output),flush=True)
    except Exception as exc:
        write_json(output/'failure.json',{'status':'failed','error':repr(exc)})
        raise


if __name__=='__main__':
    main()
