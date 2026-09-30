"""Frozen dual actions and an independently advancing KB rollout origin."""
import numpy as np
import torch
from ..contracts import SequenceBatch
from ..models import DualPolicy, StandalonePolicy, detach_clone_state
from .progress import CollectedWindow, validate_targets
from ..timing import WindowTimer


class CompressCollector:
    def __init__(self, envs, teacher, kb, action_rng, source_snapshot_id):
        if not isinstance(teacher, DualPolicy) or not teacher._frozen:
            raise ValueError('C requires a complete frozen dual teacher')
        if not isinstance(kb, StandalonePolicy) or kb._frozen:
            raise ValueError('C requires an independently trainable complete KB')
        if any(a.data_ptr() == b.data_ptr() for a,b in zip(kb.parameters(),teacher.kb.parameters())):
            raise ValueError('C teacher and KB share parameter storage')
        self.envs,self.teacher,self.kb,self.action_rng=envs,teacher,kb,action_rng
        self.device=next(kb.parameters()).device
        self.num_envs=len(envs.envs)
        self.obs,infos=envs.reset()
        self.task=infos[0]['task_key']
        self.starts=np.ones(self.num_envs,dtype=bool)
        self.state=detach_clone_state(kb.initial_state(self.num_envs))
        self.teacher_state=detach_clone_state(teacher.initial_state(self.num_envs))
        self.source_snapshot_id=source_snapshot_id

    @torch.no_grad()
    def collect(self,length,start_transition_id):
        if not 1 <= length <= 50 or start_transition_id < 0:
            raise ValueError('invalid C rollout length/ID')
        initial=detach_clone_state(self.state)
        images,starts,targets,actions,rewards,terms,truncs,infos=[],[],[],[],[],[],[],[]
        timer = WindowTimer(self.device)
        for _ in range(length):
            rgb=torch.from_numpy(self.obs.student_rgb.copy())
            reset=torch.from_numpy(self.starts.copy())
            with timer.measure('expert_forward_seconds'):
                teacher_logits,self.teacher_state=self.teacher.step(rgb.to(self.device),self.teacher_state,reset.to(self.device))
                probs=teacher_logits.softmax(-1).detach().cpu()
            validate_targets(probs,torch.ones(self.num_envs,dtype=torch.bool),self.num_envs)
            action=torch.multinomial(probs.to(self.action_rng.device),1,generator=self.action_rng).squeeze(-1).cpu()
            with timer.measure('env_step_seconds', cpu=True):
                transition=self.envs.step(action.numpy())
            images.append(rgb); starts.append(reset); targets.append(probs); actions.append(action)
            rewards.append(torch.from_numpy(transition.reward_ext.copy()))
            terms.append(torch.from_numpy(transition.terminated.copy())); truncs.append(torch.from_numpy(transition.truncated.copy()))
            infos.append(transition.info)
            self.obs,self.starts=transition.next_obs,transition.next_episode_start
        # Only the frozen teacher drives actions. Replay the independent KB BEFORE updates,
        # with its own saved origin and reset flags; never substitute teacher or learner traces.
        image_stack, start_stack = torch.stack(images), torch.stack(starts)
        with timer.measure('student_forward_seconds'):
            # Preserve collection's effective CNN batch shape. Larger chunks changed CUDA
            # convolution roundoff enough to amplify through recurrent state and Adam.
            chunk = min(self.num_envs, self.kb.config.optimization.encoder_microbatch_images)
            output = self.kb.sequence(image_stack.to(self.device), initial, start_stack.to(self.device),
                                      encoder_chunk_images=chunk)
            self.state = output.state
            kb_logits = output.logits.detach().cpu()
        timing = timer.finish()
        valid=torch.ones(length,self.num_envs,dtype=torch.bool)
        batch=SequenceBatch(image_stack,start_stack,valid,valid.clone(),torch.stack(targets),
                            torch.stack(actions),torch.arange(start_transition_id,start_transition_id+length*self.num_envs).reshape(length,self.num_envs),
                            initial,self.source_snapshot_id)
        return CollectedWindow(batch,kb_logits,torch.stack(rewards),torch.stack(terms),torch.stack(truncs),infos,timing,None)

    def detach_live_state(self):
        self.state=detach_clone_state(self.state)
        self.teacher_state=detach_clone_state(self.teacher_state)
