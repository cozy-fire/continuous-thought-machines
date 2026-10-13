"""Two independently recurrent experts, one joint student update per group."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor
from .config import Config
from .envs import BatchEnv
from .models import VisualPolicy as Policy, DualTeacher, State, detached, select, pack_state, unpack_state
from .models.common.state import state_map
from .fisher import ewc_loss


@dataclass
class Window:
    obs: Tensor
    starts: Tensor
    targets: Tensor
    initial: State
    task: str


class Collector:
    def __init__(self, envs: BatchEnv, teacher: Policy | DualTeacher, student: Policy,
                 action_rng: torch.Generator):
        if any(p.requires_grad for p in teacher.parameters()):
            raise ValueError('expert must be frozen')
        if {p.data_ptr() for p in teacher.parameters()} & {p.data_ptr() for p in student.parameters()}:
            raise ValueError('teacher and student share storage')
        self.envs,self.teacher,self.student,self.rng = envs,teacher,student,action_rng
        self.device = next(student.parameters()).device
        self.slots = len(envs.envs)
        self.obs = envs.reset(); self.starts = np.ones(self.slots,dtype=bool)
        self.state = detached(student.initial_state(self.slots))
        self.teacher_state = detached(teacher.initial_state(self.slots))

    @torch.no_grad()
    def collect(self, length: int) -> Window:
        initial = detached(self.state)
        images,masks,targets = [],[],[]
        for _ in range(length):
            rgb = torch.from_numpy(self.obs.copy())
            mask = torch.from_numpy(self.starts.copy())
            logits,self.teacher_state = self.teacher.step(rgb.to(self.device),self.teacher_state,
                                                         mask.to(self.device),task=self.envs.task)
            if not torch.isfinite(logits).all():
                raise ValueError('nonfinite expert logits')
            probs = logits.softmax(-1).cpu()
            action = torch.multinomial(probs,1,generator=self.rng).squeeze(-1).numpy()
            images.append(rgb); masks.append(mask); targets.append(probs)
            self.obs,self.starts = self.envs.step(action)
        obs,starts = torch.stack(images),torch.stack(masks)
        _,self.state = self.student.sequence(obs.to(self.device),initial,starts.to(self.device),
                                             task=self.envs.task,encoder_chunk=min(self.slots,self.student.encoder_chunk))
        return Window(obs,starts,torch.stack(targets),initial,self.envs.task)

    def truncate(self) -> None:
        self.state,self.teacher_state = detached(self.state),detached(self.teacher_state)

    def capture(self) -> dict:
        return dict(envs=self.envs.capture(),starts=self.starts.tolist(),
                    state=pack_state(self.state),teacher_state=pack_state(self.teacher_state))

    def restore(self, value: dict) -> None:
        student_kind = pack_state(self.state)['kind']
        teacher_kind = pack_state(self.teacher_state)['kind']
        student_state = unpack_state(value['state'],self.device,student_kind)
        teacher_state = unpack_state(value['teacher_state'],self.device,teacher_kind)
        # Check dimensions before changing the environment or collector state.
        expected,actual = [],[]
        for state,shapes in ((self.state,expected),(self.teacher_state,expected),
                             (student_state,actual),(teacher_state,actual)):
            state_map(state,lambda t:shapes.append((t.shape,t.dtype)) or t)
        if expected != actual:
            raise ValueError('recurrent state shape/dtype mismatch')
        self.obs = self.envs.restore(value['envs'])
        self.starts = np.asarray(value['starts'],dtype=bool)
        self.state,self.teacher_state = student_state,teacher_state


def kl_divergence(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.shape != targets.shape or not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError('nonfinite/malformed KL inputs')
    if (targets < 0).any() or not torch.allclose(targets.sum(-1),torch.ones_like(targets[...,0]),atol=1e-5,rtol=1e-4):
        raise ValueError('invalid expert distribution')
    return (torch.special.xlogy(targets,targets)-targets*logits.log_softmax(-1)).sum(-1)


def update_pair(student: Policy, batches: list[Window], optimizer: torch.optim.Optimizer,
                fisher: dict | None, rng: np.random.Generator, config: Config, diagnostics: bool = False) -> dict:
    if len(batches) != 2 or [b.task for b in batches] != ['maze_medium','fourrooms']:
        raise ValueError('joint update needs Maze and FourRooms')
    slots = batches[0].obs.shape[1]
    if slots != batches[1].obs.shape[1] or slots % config.groups:
        raise ValueError('invalid paired slot count')
    if config.ewc_enabled and fisher is None:
        raise ValueError('enabled EWC requires Fisher')
    groups = [np.split(rng.permutation(slots),config.groups) for _ in batches]
    device = next(student.parameters()).device
    sums = dict(maze_kl=0.,fourrooms_kl=0.,ewc=0.,total_loss=0.,adam_updates=0,ewc_applications=0,
                ewc_enabled=config.ewc_enabled)
    if diagnostics:
        sums.update(maze_gradient_norm=0.,fourrooms_gradient_norm=0.)
    for group in range(config.groups):
        optimizer.zero_grad(set_to_none=True)
        task_kl = []
        for i,batch in enumerate(batches):
            previous = [p.grad.detach().clone() if p.grad is not None else None for p in student.parameters()] if diagnostics else None
            indices = torch.as_tensor(groups[i][group],dtype=torch.long)
            denominator = batch.obs.shape[0]*len(indices)
            summed = 0.
            for chunk in indices.split(config.compress_microbatch):
                initial = select(batch.initial,chunk)
                logits,_ = student.sequence(batch.obs.index_select(1,chunk).to(device),initial,
                                             batch.starts.index_select(1,chunk).to(device),task=batch.task)
                targets = batch.targets.index_select(1,chunk).to(device)
                loss_sum = kl_divergence(logits,targets).sum()
                (.5*loss_sum/denominator).backward()
                summed += float(loss_sum.detach())
            task_kl.append(summed/denominator)
            if diagnostics:
                delta = [(p.grad-(old if old is not None else 0)).square().sum()
                         for p,old in zip(student.parameters(),previous) if p.grad is not None]
                name = 'maze_gradient_norm' if i == 0 else 'fourrooms_gradient_norm'
                sums[name] += float(torch.stack(delta).sum().sqrt())/config.groups
        penalty_value = 0.
        if config.ewc_enabled:
            penalty = ewc_loss(student,fisher,config.ewc_lambda)
            if not torch.isfinite(penalty):
                raise ValueError('nonfinite EWC')
            penalty.backward()
            penalty_value = float(penalty.detach())
            sums['ewc_applications'] += 1
        torch.nn.utils.clip_grad_norm_(student.parameters(),config.clip,error_if_nonfinite=True)
        optimizer.step()
        sums['maze_kl'] += task_kl[0]/config.groups
        sums['fourrooms_kl'] += task_kl[1]/config.groups
        sums['ewc'] += penalty_value/config.groups
        sums['total_loss'] += (.5*sum(task_kl)+penalty_value)/config.groups
        sums['adam_updates'] += 1
    return sums
