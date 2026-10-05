"""Real P->C->F on both tasks, including the second compression's visual EWC."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .config import REPO_ROOT,load_config,resolved_dict
from .data import build_manifest
from .envs import SyncVectorEnv,make_env,MazeMapCache
from .data.maze_curriculum import MazeStateTable
from .learning.maze_progress import run_maze_progress
from .teachers import MazeTeacher,FourRoomsTeacher
from .models import StandalonePolicy,DualPolicy,save_snapshot
from .learning.progress import run_progress
from .learning.distill import run_compress_stage
from .learning.fisher import run_fisher_stage,policy_hash
from .verify_stage import write_json
from .evaluate import evaluate_policy


def verify(device,output_dir):
    output=Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(0)
    config=load_config('tasks/continual_nav_opd/configs/smoke.yaml')
    config=replace(config,training=replace(config.training,device=device))
    if device.startswith('cuda'): torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)
    manifest=build_manifest(REPO_ROOT/config.environment.maze_root,validation_count=512,validation_episodes=2,test_episodes=2,drift_episodes=2)
    cache=MazeMapCache(REPO_ROOT/config.environment.maze_root,manifest.train+manifest.validation+manifest.test)
    table=MazeStateTable(cache,manifest.train[:config.maze_progress.pool_sizes[-1]])
    kb=StandalonePolicy(config).to(device); fisher=None; ready=False; consumed=0; reports=[]
    try:
        for index,task in enumerate(config.task_order):
            envs=SyncVectorEnv([make_env(config,task,'train',manifest=manifest,seed=index*10+i,map_cache=cache) for i in range(2)])
            rng=torch.Generator().manual_seed(4+index)
            expert=MazeTeacher() if task=='maze_medium' else FourRoomsTeacher(REPO_ROOT/config.teachers.fourrooms.checkpoint,device=device)
            try:
                dual=DualPolicy(config,kb,kb_ready=ready)
                p=(run_maze_progress(table,dual,config,rng,np.random.default_rng(index),consumed) if task=='maze_medium' else
                   run_progress(envs,dual,expert,config,256,rng,np.random.default_rng(index),consumed)); consumed=p.next_transition_id
                teacher_hash=policy_hash(p.policy); before=policy_hash(kb)
                c=run_compress_stage(envs,p.policy,kb,fisher,config,128,rng,np.random.default_rng(index+4),consumed); consumed=c.next_transition_id
                assert before!=policy_hash(kb) and teacher_hash==policy_hash(p.policy)
                active_report=evaluate_policy(p.policy,config,manifest,identity={'policy_type':'active','active_source':task,'evaluation_id':task+'_P'})
                kb_report=evaluate_policy(kb,config,manifest,identity={'policy_type':'kb','evaluation_id':task+'_C'})
                before=policy_hash(kb)
                f=run_fisher_stage(envs,kb,fisher,config,32,rng,np.random.default_rng(index+8),consumed,task+'/F'); consumed=f['next_transition_id']
                assert before==policy_hash(kb); fisher=f['fisher']; ready=c.kb_ready
                reports.append({'task':task,'P':{'steps':p.transitions,'updates':p.optimizer_updates,'targets':p.eligible_target_steps},
                                'C':{'steps':c.transitions,'updates':c.optimizer_updates,**c.statistics},
                                'F':{'steps':f['transitions'],'scored_samples':f['scored_samples'],'unique_ids':len(set(f['selected_ids'])),**f['statistics']},
                                'Active_validation':active_report,'KB_validation':kb_report,
                                'teacher_unchanged':True,'F_parameters_unchanged':True,'full_parameter_count':len(fisher.importance)})
                del dual,p,expert
            finally: envs.close()
        save_snapshot(kb,output/'kb.pt'); torch.save(fisher,output/'fisher.pt')
        write_json(output/'resolved_config.json',resolved_dict(config))
        write_json(output/'report.json',{'status':'passed','stage':'05','transitions':consumed,'results':reports,
                   'completed_compressions':fisher.completed_compressions,'device':device,
                   'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
                   'limits':'Short engineering smoke, not evidence of learned navigation.'})
        print('Stage 05 passed: '+str(output),flush=True)
    except Exception as exc:
        write_json(output/'failure.json',{'error':repr(exc)}); raise
