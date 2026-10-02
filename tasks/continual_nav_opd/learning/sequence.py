"""Shared sequence loss/update utilities; only shuffle environment slots."""
import math
import numpy as np
import torch
from torch import Tensor
from ..contracts import SequenceBatch
from ..models import detach_clone_state, select_state
from ..timing import WindowTimer


def distribution_metrics(logits: Tensor, targets: Tensor, *, check_inputs: bool = True) -> dict[str, Tensor]:
    if logits.shape != targets.shape or logits.shape[-1] != 5:
        raise ValueError("logits and targets must match [...,5]")
    targets = targets.detach()
    if check_inputs and (not torch.isfinite(logits).all() or not torch.isfinite(targets).all() or (targets < 0).any() or not torch.allclose(
            targets.sum(-1), torch.ones_like(targets[...,0]), atol=1e-5, rtol=1e-4)):
        raise ValueError("nonfinite logits or invalid target distribution")
    log_probs = logits.log_softmax(-1)
    # xlogy(0,0)=0: exact one-hot labels must not create 0*log(0) NaNs.
    target_log_terms = torch.special.xlogy(targets, targets)
    return {"kl": (target_log_terms - targets*log_probs).sum(-1),
            "agreement": (targets.argmax(-1) == logits.argmax(-1)).float(),
            "expert_entropy": -target_log_terms.sum(-1),
            "student_entropy": -(log_probs.exp()*log_probs).sum(-1)}


def gradient_norm_tensor(parameters, device):
    values = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return torch.stack(values).sum().sqrt() if values else torch.zeros((), device=device)


def gradient_norm(parameters) -> float:
    parameters = list(parameters)
    device = parameters[0].device if parameters else torch.device('cpu')
    return float(gradient_norm_tensor(parameters, device))


def update_window(student, batch: SequenceBatch, optimizer, config, minibatch_rng: np.random.Generator, regularizer=None) -> dict:
    if (batch.valid_mask & ~batch.target_mask).any():
        raise ValueError("a real observation is missing its required target")
    if student.config != config or batch.ticks != config.ctm.ticks_by_task.for_task(batch.task):
        raise ValueError('rollout/model execution budget mismatch')
    slots = batch.obs.shape[1]
    if slots % config.optimization.minibatches:
        raise ValueError("environment slots must divide into minibatches")
    device = next(student.parameters()).device
    totals = {key:0. for key in ("kl","agreement","expert_entropy","student_entropy",
                                "grad_norm","encoder_grad_norm","controller_grad_norm","actor_grad_norm","adapter_grad_norm",
                                "controller_actor_grad_norm",
                                "ewc","ewc_encoder","ewc_controller_actor","total_loss")}
    targets, updates, empty = 0, 0, 0
    timer = WindowTimer(device)
    for group in np.split(minibatch_rng.permutation(slots), config.optimization.minibatches):
        indices = torch.as_tensor(group, dtype=torch.int64)
        cpu_mask = batch.loss_mask.index_select(1, indices).cpu()
        count = int(cpu_mask.sum())
        mask = cpu_mask.to(device)
        if count == 0:
            empty += 1
            continue
        # Each minibatch owns a cloned origin; no prior graph or writable state is shared.
        state = detach_clone_state(select_state(batch.initial_state, indices.to(device)))
        optimizer.zero_grad(set_to_none=True)
        valid = batch.valid_mask.index_select(1, indices).cpu()
        target_cpu = batch.teacher_probs.index_select(1, indices).cpu()[cpu_mask]
        # Validate fixed labels on CPU before allocating a replay graph.
        if not torch.isfinite(target_cpu).all() or (target_cpu < 0).any() or not torch.allclose(
                target_cpu.sum(-1), torch.ones(count), atol=1e-5, rtol=1e-4):
            raise ValueError("nonfinite logits or invalid target distribution")
        with timer.measure('learner_forward_seconds'):
            output = student.sequence(batch.obs.index_select(1,indices).to(device), state,
                                      batch.episode_start.index_select(1,indices).to(device),
                                      None if bool(valid.all()) else valid.to(device), task=batch.task)
            selected_logits = output.logits[mask]
            metrics = distribution_metrics(selected_logits, target_cpu.to(device), check_inputs=False)
            loss = metrics["kl"].mean()
            penalty, parts = regularizer() if regularizer else (loss.new_zeros(()), {})
            loss = loss + penalty
            metrics.update(ewc=penalty.expand(count),total_loss=loss.expand(count),
                           **{key:parts.get(key,loss.new_zeros(())).expand(count) for key in ('ewc_encoder','ewc_controller_actor')})
        with timer.measure('learner_backward_seconds'):
            loss.backward()
        trainable = [p for p in student.parameters() if p.requires_grad]
        active = student.active if hasattr(student,"active") else student
        gradients = {"grad_norm":gradient_norm_tensor(trainable, device),
                     "encoder_grad_norm":gradient_norm_tensor(active.encoder.parameters(), device),
                     "controller_grad_norm":gradient_norm_tensor(active.controller.parameters(), device),
                     "actor_grad_norm":gradient_norm_tensor(active.actor.parameters(), device),
                     "adapter_grad_norm":gradient_norm_tensor(student.adapter.parameters(), device) if hasattr(student,"adapter") else loss.new_zeros(())}
        # One device transfer carries sample sums, pre-clip norms and finite guards.
        keys, grad_keys = list(metrics), list(gradients)
        packed = torch.stack([metrics[k].detach().sum() for k in keys] +
                             [gradients[k] for k in grad_keys] + [torch.isfinite(selected_logits).all().float()]).cpu().tolist()
        values, norms = packed[:len(keys)], packed[len(keys):-1]
        if packed[-1] != 1 or not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite distillation loss or logits")
        if not all(math.isfinite(value) for value in norms):
            raise ValueError("nonfinite distillation gradient")
        gradients_cpu = dict(zip(grad_keys, norms))
        gradients_cpu['controller_actor_grad_norm'] = math.hypot(gradients_cpu['controller_grad_norm'], gradients_cpu['actor_grad_norm'])
        with timer.measure('learner_optimizer_seconds'):
            # The packed pre-clip total norm was checked above; avoid a duplicate GPU scalar guard.
            torch.nn.utils.clip_grad_norm_(trainable, config.optimization.max_grad_norm, error_if_nonfinite=False)
            optimizer.step()
        for key, value in zip(keys, values):
            totals[key] += value
        for key, value in gradients_cpu.items():
            totals[key] += value*count
        targets += count
        updates += 1
    return {**{k:v/targets if targets else 0. for k,v in totals.items()}, **timer.finish(),
            "eligible_target_steps":targets,"optimizer_updates":updates,"empty_minibatches":empty}
