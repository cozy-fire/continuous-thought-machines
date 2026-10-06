"""Global behavior display with strictly local RGB policy inputs and optional P labels."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from .evaluate import load_for_evaluation,panels,create_panel_env
from .checkpoint import atomic_json,isolated_rng
from .teachers import MazeTeacher,FourRoomsTeacher


def visualize(checkpoint,policy_type,task,panel_index,output_dir,split='validation',device='cpu',stage_key=None,teacher_diagnostics=False):
    output=Path(output_dir).resolve()
    if output.exists(): raise FileExistsError(output)
    identity={}
    policy,config,manifest,root=load_for_evaluation(checkpoint,policy_type,device,stage_key,identity=identity)
    specs=panels(config,manifest,task,split)
    if not 0<=panel_index<len(specs): raise ValueError('panel-index outside fixed panel')
    if teacher_diagnostics and policy_type!='active': raise ValueError('P teaching diagnostics require an Active policy')
    output.mkdir(parents=True,exist_ok=False); env=create_panel_env(specs[panel_index]); rows=[]; frames=[]
    action_names=['up','down','left','right','wait'] if task=='maze_medium' else ['turn_left','turn_right','forward','wait3','wait4']
    with isolated_rng(),torch.no_grad():
        try:
            obs,info=env.reset(); state=policy.initial_state(1); teacher=None; teacher_state=None
            if teacher_diagnostics:
                teacher=MazeTeacher() if task=='maze_medium' else FourRoomsTeacher(root/'teachers/fourrooms.pt',device=device)
                if task=='fourrooms': teacher_state=teacher.initial_state(1)
            def frame(caption):
                raw=env.render() if task=='maze_medium' else env.native.unwrapped.get_frame(tile_size=24,agent_pov=False)
                image=Image.fromarray(raw).resize((504,504),Image.Resampling.NEAREST).convert('RGB')
                canvas=Image.new('RGB',(504,560),'white'); canvas.paste(image,(0,0)); ImageDraw.Draw(canvas).text((8,514),caption,fill='black')
                return canvas
            frames.append(frame('Initial observation; no action or warmup'))
            total=0.; step=0; target=None
            while True:
                start=np.asarray([step==0],dtype=bool); distance=None
                if teacher is not None:
                    if task=='maze_medium':
                        target=teacher.predict(obs.teacher_obs[None],[info]); probs=target.probabilities[0].tolist(); distance=int(target.distance_to_goal[0])
                    else:
                        target=teacher.predict(obs.teacher_obs[None],teacher_state,start); teacher_state=target.state; probs=target.probabilities[0].cpu().tolist()
                else: probs=None
                rgb=torch.from_numpy(obs.student_rgb[None]).to(device)
                logits,state=policy.step(rgb,state,torch.tensor(start,device=device),task=task)
                action=int(logits.argmax(-1)[0]); obs,reward,term,trunc,info=env.step(action); total+=reward; step+=1
                row={'step':step,'action':action,'action_name':action_names[action],'reward':reward,'cumulative_reward':total,
                     'terminated':term,'truncated':trunc,'success':bool(info['success']),
                     'target_mask':True if teacher else None,'teacher_probs':probs,
                     'teacher_action':int(np.argmax(probs)) if probs is not None else None,'distance_to_goal':distance}
                rows.append(row); frames.append(frame(f'Step {step}: {action_names[action]}   reward={reward:.4f}\nReturn={total:.4f}'))
                if term or trunc: break
            frames[0].save(output/'behavior.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=False)
            with (output/'steps.csv').open('x',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            fig,ax=plt.subplots(); ax.plot([r['step'] for r in rows],[r['reward'] for r in rows])
            ax.set(xlabel='Environment step',ylabel='Environment reward'); fig.savefig(output/'reward.png',dpi=150); plt.close(fig)
            with Image.open(output/'behavior.gif') as media:
                if media.n_frames!=len(rows)+1: raise RuntimeError('GIF frames differ from action count+1')
                media.seek(media.n_frames-1); media.load()
            with Image.open(output/'reward.png') as media: media.verify()
            atomic_json(output/'trajectory.json',{'task':task,'split':split,'panel_index':panel_index,'checkpoint':str(Path(checkpoint).resolve()),
                        'ticks':config.ctm.ticks_by_task.for_task(task),'memory_ticks':config.ctm.memory_length,
                        'policy_type':policy_type,'stage_key':identity['stage'],'snapshot':identity['checkpoint'],'teacher_diagnostics':teacher_diagnostics,
                        'frames':len(frames),'actions':len(rows),'success':bool(info['success']),'return':total,'rows':rows})
        finally: env.close()
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True); parser.add_argument('--policy',choices=['kb','active'],default='kb')
    parser.add_argument('--stage-key'); parser.add_argument('--task',choices=['maze_medium','fourrooms'],required=True)
    parser.add_argument('--panel-index',type=int,required=True); parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--device',default='cpu'); parser.add_argument('--output-dir',required=True); parser.add_argument('--teacher-diagnostics',action='store_true')
    args=parser.parse_args(); torch.set_num_threads(2)
    visualize(args.checkpoint,args.policy,args.task,args.panel_index,args.output_dir,args.split,args.device,args.stage_key,args.teacher_diagnostics)


if __name__=='__main__': main()
