"""Student-driven, current-window-only collection with action-before expert labels."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor
from ..contracts import SequenceBatch
from ..envs import SyncVectorEnv
from ..models import DualPolicy, detach_clone_state
from ..teachers import MazeTeacher, FourRoomsTeacher
from ..timing import WindowTimer


def validate_targets(probabilities: Tensor, valid: Tensor, batch: int) -> None:
    if probabilities.shape != (batch, 5) or probabilities.dtype != torch.float32:
        raise ValueError("expert probabilities must be float32 [B,5]")
    if valid.shape != (batch,) or valid.dtype != torch.bool or not valid.all():
        raise ValueError("every real observation needs a valid target, including the first/reset step")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any() or not torch.allclose(
            probabilities.sum(-1), torch.ones(batch, device=probabilities.device), atol=1e-5, rtol=1e-4):
        raise ValueError("expert distribution must be finite, nonnegative and normalized")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class CollectedWindow:
    batch: SequenceBatch
    student_logits: Tensor  # Detached CPU [L,B,5], diagnostic only, not PPO data.
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    info: list[list[dict]]
    timing: dict[str, float]
    distances: Tensor | None


class ProgressCollector:
    def __init__(self, envs: SyncVectorEnv, student: DualPolicy,
                 expert: MazeTeacher | FourRoomsTeacher, action_rng: torch.Generator):
        if not isinstance(student, DualPolicy) or student._frozen:
            raise ValueError("P requires a trainable DualPolicy")
        self.envs, self.student, self.expert, self.action_rng = envs, student, expert, action_rng
        self.device = next(student.parameters()).device
        self.num_envs = len(envs.envs)
        self.obs, self.contexts = envs.reset()
        tasks = {info["task_key"] for info in self.contexts}
        expected = "maze_medium" if isinstance(expert, MazeTeacher) else "fourrooms"
        if tasks != {expected} or not isinstance(expert, (MazeTeacher, FourRoomsTeacher)):
            raise ValueError("all environment slots must match the selected expert")
        self.task = expected
        self.ticks = student.config.ctm.ticks_by_task.for_task(self.task)
        self.episode_start = np.ones(self.num_envs, dtype=bool)
        self.state = detach_clone_state(student.initial_state(self.num_envs))
        self.expert_state = expert.initial_state(self.num_envs) if isinstance(expert, FourRoomsTeacher) else None
        self.source_snapshot_id = expert.source_snapshot_id

    @torch.no_grad()
    def collect(self, length: int, start_transition_id: int) -> CollectedWindow:
        if not 1 <= length <= 50 or start_transition_id < 0:
            raise ValueError("window length must be 1..50 and transition ID nonnegative")
        # Save before ANY new observation; copying breaks storage aliases and prior-window graphs.
        initial = detach_clone_state(self.state)
        images, starts, targets, actions, logits_list = [], [], [], [], []
        rewards, terms, truncs, infos, distances = [], [], [], [], []
        timer = WindowTimer(self.device)
        for _ in range(length):
            rgb = torch.from_numpy(self.obs.student_rgb.copy())
            reset = torch.from_numpy(self.episode_start.copy())
            with timer.measure('student_forward_seconds'):
                logits, self.state = self.student.step(rgb.to(self.device), self.state, reset.to(self.device), task=self.task)
                logits_cpu = logits.detach().cpu()
            # Check the already-downloaded diagnostic tensor before executing an action.
            if not torch.isfinite(logits_cpu).all():
                raise ValueError("nonfinite sampling logits")
            with timer.measure('expert_forward_seconds', cpu=isinstance(self.expert, MazeTeacher)):
                if isinstance(self.expert, MazeTeacher):
                    output = self.expert.predict(self.obs.teacher_obs, self.contexts)
                    probabilities = torch.from_numpy(output.probabilities.copy())
                    valid = torch.from_numpy(output.valid_target.copy())
                    distances.append(torch.from_numpy(output.distance_to_goal.copy()))
                else:
                    # Teacher state advances once per real action-before observation, with no warmup.
                    output = self.expert.predict(self.obs.teacher_obs, self.expert_state, self.episode_start)
                    self.expert_state = output.state
                    probabilities, valid = output.probabilities.detach().cpu(), output.valid_target.detach().cpu()
            validate_targets(probabilities, valid, self.num_envs)
            # Only STUDENT probabilities reach the action sampler; never substitute expert actions.
            sampling_probs = logits.softmax(-1).to(self.action_rng.device)
            action = torch.multinomial(sampling_probs, 1, generator=self.action_rng).squeeze(-1).cpu()
            with timer.measure('env_step_seconds', cpu=True):
                transition = self.envs.step(action.numpy())
            images.append(rgb)
            starts.append(reset)
            targets.append(probabilities.clone())
            actions.append(action)
            logits_list.append(logits_cpu)
            rewards.append(torch.from_numpy(transition.reward_ext.copy()))
            terms.append(torch.from_numpy(transition.terminated.copy()))
            truncs.append(torch.from_numpy(transition.truncated.copy()))
            infos.append(transition.info)
            # Terminal images remain diagnostic data, never the next episode's action label.
            self.obs, self.episode_start = transition.next_obs, transition.next_episode_start
            self.contexts = [info["reset_info"] if reset_slot else info
                             for info, reset_slot in zip(transition.info, self.episode_start)]
        shape = (length, self.num_envs)
        batch = SequenceBatch(torch.stack(images), torch.stack(starts), torch.ones(shape, dtype=torch.bool),
                              torch.ones(shape, dtype=torch.bool), torch.stack(targets), torch.stack(actions),
                              torch.arange(start_transition_id, start_transition_id + length*self.num_envs).reshape(shape),
                              initial, self.source_snapshot_id, self.task, self.ticks)
        return CollectedWindow(batch, torch.stack(logits_list), torch.stack(rewards), torch.stack(terms),
                               torch.stack(truncs), infos, timer.finish(), torch.stack(distances) if distances else None)

    def detach_live_state(self) -> None:
        # Continue from COLLECTION history, never from the learner's replayed final state.
        self.state = detach_clone_state(self.state)
