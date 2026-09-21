"""Real-pixel model verification only: no PPO, reward learning or optimizer loop."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import torch
from torch import nn

from .config import config_hash, load_config
from .data.manifest import MazeManifest, build_manifest
from .envs import VectorEnvAdapter, build_env
from .models import DualPolicy, StandalonePolicy, VisionEncoder, encode_obs


def tensor_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify(config_path: Path, output_dir: Path, *, maze_manifest: Path | None = None,
           maze_root: Path | None = None, device: str = "cpu", steps: int = 12) -> dict:
    if steps < 1:
        raise ValueError("verification needs at least one observation")
    config = load_config(config_path)
    root = (maze_root or Path(config.environment.maze_root)).resolve()
    if maze_manifest is not None:
        manifest = MazeManifest.load(maze_manifest, root)
    else:
        e = config.evaluation
        manifest = build_manifest(root, validation_count=e.maze_validation_hashes,
                                  validation_episodes=e.validation_episodes,
                                  test_episodes=e.test_episodes, drift_episodes=e.drift_episodes)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "model_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    torch.manual_seed(17)
    encoder = VisionEncoder(config).to(device)
    policy = DualPolicy(config, StandalonePolicy(config), kb_ready=True).to(device)
    # Exercise nonzero lateral projection gradients, separately from the zero-gate tests.
    with torch.no_grad():
        policy.adapter.gate.fill_(0.4)
    policy.train()
    encoder_before, kb_before = tensor_hash(encoder), tensor_hash(policy.kb)
    vector = VectorEnvAdapter([build_env(task, "train", i, config=config,
                                maze_manifest=manifest, maze_root=root)
                               for i, task in enumerate(config.task_order)])
    batch = len(config.task_order)
    features, starts, collected_logits, collected_values = [], [], [], []
    visual_calls = []
    hook = encoder.register_forward_hook(lambda *_: visual_calls.append(1))
    try:
        obs, _ = vector.reset()
        episode_start = torch.ones(batch, dtype=torch.bool, device=device)
        with torch.no_grad():
            state = policy.initial_state(batch)
            for _ in range(steps):
                fmap = encode_obs(torch.from_numpy(obs).to(device), encoder)
                output = policy.step(fmap, state, episode_start)
                features.append(fmap)
                starts.append(episode_start)
                collected_logits.append(output.logits)
                collected_values.append(output.value)
                state = output.state
                # Argmax is only a bounded interface probe, not a performance evaluation.
                result = vector.step(output.logits.argmax(-1).cpu().numpy())
                obs = result.next_obs
                episode_start = torch.from_numpy(result.next_episode_start).to(device)
    finally:
        hook.remove()
        vector.close()
    assert len(visual_calls) == steps  # Shared fmap once per observation batch, not per tick/column.
    replay = policy.sequence(torch.stack(features), policy.initial_state(batch), torch.stack(starts))
    logits = torch.stack(collected_logits)
    values = torch.stack(collected_values)
    logit_error = (replay.logits-logits).abs().max().item()
    value_error = (replay.value-values).abs().max().item()
    torch.testing.assert_close(replay.logits, logits, atol=1e-5, rtol=0)
    torch.testing.assert_close(replay.value, values, atol=1e-5, rtol=0)
    state_error = 0.0
    for column in ("kb", "active"):
        for trace in ("pre", "post"):
            replay_trace = getattr(getattr(replay.state, column), trace)
            collected_trace = getattr(getattr(state, column), trace)
            torch.testing.assert_close(replay_trace, collected_trace, atol=1e-5, rtol=0)
            state_error = max(state_error, (replay_trace-collected_trace).abs().max().item())
    loss = -replay.logits.log_softmax(-1)[..., 2].mean() + (replay.value-1).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    groups = ("active.controller.attention.query_input", "active.controller.attention.token_input",
              "active.controller.attention.q", "active.controller.attention.k", "active.controller.attention.v",
              "active.controller.attention.output", "active.controller.synapse", "active.controller.nlm",
              "active.controller.action_sync.decay", "active.controller.out_sync.decay",
              "active.actor", "active.critic", "adapter")
    gradient_l1 = {}
    for group in groups:
        grads = [p.grad for n, p in policy.named_parameters() if n.startswith(group)]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), group
        gradient_l1[group] = sum(g.abs().sum().item() for g in grads)
        assert gradient_l1[group] > 0, group
    assert all(p.grad is None for p in encoder.parameters())
    assert all(p.grad is None for p in policy.kb.parameters())
    assert tensor_hash(encoder) == encoder_before and tensor_hash(policy.kb) == kb_before
    report = dict(status="passed", scope="delivery_2_model_only_no_training", python=platform.python_version(),
                  torch=torch.__version__, device=device, config_hash=config_hash(config),
                  task_order=list(config.task_order), observations_per_task=steps,
                  environment_transitions=steps*batch, visual_batch_forwards=len(visual_calls),
                  ticks_per_column_per_environment_step=config.ctm.ticks,
                  fmap_shape=list(features[0].shape), max_logit_error=logit_error,
                  max_value_error=value_error, max_state_error=state_error,
                  synthetic_loss=loss.item(), gradient_l1=gradient_l1,
                  encoder_hash=encoder_before, kb_hash=kb_before,
                  frozen_tensors_unchanged=True, frozen_gradients_absent=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maze-manifest", type=Path)
    parser.add_argument("--maze-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    # Bound CPU thread oversubscription for these small recurrent verification batches.
    torch.set_num_threads(2)
    report = verify(args.config, args.output_dir, maze_manifest=args.maze_manifest,
                    maze_root=args.maze_root, device=args.device, steps=args.steps)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
