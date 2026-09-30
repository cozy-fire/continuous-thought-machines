"""Fixed-image FP32 output/gradient and memory probe; not a throughput benchmark."""
import argparse
from dataclasses import replace
from pathlib import Path
import time

import torch
from .checkpoint import atomic_json, source_manifest
from .config import REPO_ROOT, load_config
from .data import build_manifest
from .envs import make_env
from .models import StandalonePolicy, DualPolicy


def probe(output_dir, device='cuda:0'):
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(0)
    config = load_config(REPO_ROOT/'tasks/continual_nav_opd/configs/smoke.yaml')
    config = replace(config, training=replace(config.training, device=device))
    manifest = build_manifest(REPO_ROOT/config.environment.maze_root, validation_count=512,
                              validation_episodes=2, test_episodes=2, drift_episodes=2)
    env = make_env(config, 'maze_medium', 'train', manifest=manifest, seed=0)
    try:
        observation, _ = env.reset()
    finally:
        env.close()
    # Repeat a real formatted map, with no additional environment actions or teacher labels.
    images = torch.from_numpy(observation.student_rgb).unsqueeze(0).unsqueeze(0).repeat(50, 4, 1, 1, 1).to(device)
    starts = torch.zeros(50, 4, dtype=torch.bool, device=device); starts[0] = True
    dual = DualPolicy(config, StandalonePolicy(config).to(device), kb_ready=True)
    reference_logits = reference_grads = None; results = []
    try:
        for microbatch in (8, 200):
            updated = replace(config, optimization=replace(config.optimization, encoder_microbatch_images=microbatch))
            for policy in (dual, dual.kb, dual.active):
                policy.config = updated
            dual.zero_grad(set_to_none=True)
            if device.startswith('cuda'):
                torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            prediction = dual.sequence(images, dual.initial_state(4), starts)
            loss = prediction.logits.log_softmax(-1)[..., 0].neg().mean()
            loss.backward()
            if device.startswith('cuda'):
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-start
            logits = prediction.logits.detach().cpu()
            grads = {n: p.grad.detach().cpu().clone() for n, p in dual.named_parameters() if p.grad is not None}
            item = {'microbatch_images': microbatch, 'elapsed_seconds': elapsed,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else None}
            if reference_logits is None:
                reference_logits, reference_grads = logits, grads
            else:
                torch.testing.assert_close(logits, reference_logits, atol=3e-5, rtol=1e-4)
                if set(grads) != set(reference_grads):
                    raise ValueError('microbatch changed gradient routes')
                # Elementwise relative error is unstable for near-zero gradients.
                # Report absolute error, and test the full-vector relative L2 error.
                squared_error = sum(float((g.double()-reference_grads[n].double()).square().sum()) for n,g in grads.items())
                squared_reference = sum(float(g.double().square().sum()) for g in reference_grads.values())
                relative_error = (squared_error/max(squared_reference, 1e-30))**.5
                item.update(max_logit_difference=float((logits-reference_logits).abs().max()),
                            max_gradient_difference=max(float((g-reference_grads[n]).abs().max()) for n,g in grads.items()),
                            gradient_relative_l2_error=relative_error, gradient_relative_l2_tolerance=1e-3,
                            argmax_disagreements=int((logits.argmax(-1)!=reference_logits.argmax(-1)).sum()))
                if relative_error > 1e-3:
                    atomic_json(output/'numerical_divergence.json', {'status': 'numerical_divergence',
                        'results': results+[item], 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                        'matmul_precision': torch.get_float32_matmul_precision(),
                        'limits': 'Fixed-image memory/numerical probe, not a failed training smoke. No equivalence claim.'})
                    raise ValueError('microbatch gradient relative L2 error exceeds 1e-3')
            results.append(item); del prediction, loss
        report = {'status': 'passed', 'device': device, 'source': source_manifest(), 'slots': 4,
                  'observations_per_slot': 50, 'results': results,
                  'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'matmul_precision': torch.get_float32_matmul_precision(),
                  'limits': 'Repeated real Maze image, fixed synthetic supervision; numerical/memory probe only. '
                            'Not a real-teacher training throughput estimate or a RTX5090 measurement.'}
        atomic_json(output/'report.json', report); print(str(output/'report.json'), flush=True)
        return report
    except Exception as exc:
        atomic_json(output/'failure.json', {'error': repr(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); probe(args.output_dir, args.device)


if __name__ == '__main__':
    main()
