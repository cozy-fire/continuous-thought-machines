"""Frozen single-column inference for ordinary and task-switch visualizations."""
import math
import json
from pathlib import Path

import numpy as np
import torch

from ..checkpoint import model_hash, seed_everything
from ..config import TICKS
from ..envs import MapCache, MazeEnv, FourRoomsEnv, file_sha
from ..model import Policy, detached, rotate


def collect(config: dict, root: Path, out: Path, panel: list) -> dict:
    seed_everything(0)
    device = config['device']
    checkpoint = root / config['checkpoint']
    checkpoint_hash = file_sha(checkpoint)
    policy = Policy(checkpoint_encoder=False).to(device).eval().requires_grad_(False)
    artifact = torch.load(checkpoint, map_location='cpu', weights_only=True)
    policy.load_state_dict(artifact['state_dict'], strict=True)
    del artifact
    before = model_hash(policy)
    entries = [c['entry'] for c in panel if c['task'] == 'maze_medium']
    cache = MapCache(root / config['maze_root'], entries)
    attention = policy.controller.attention
    keys, posts, heads = {}, [], []

    def key_hook(module, args, value):
        k = value.reshape(1, 441, 4, 32).transpose(1, 2)
        keys['k'] = rotate(k, attention.rope.key_cos, attention.rope.key_sin)

    def query_hook(module, args, value):
        q = value.reshape(1, 1, 4, 32).transpose(1, 2)
        q = rotate(q, attention.rope.query_cos, attention.rope.query_sin)
        weights = (q @ keys['k'].transpose(-2, -1) / math.sqrt(32)).softmax(-1)
        heads.append(weights[0, :, 0, :].mean(0).cpu().numpy().copy())

    hooks = [attention.k.register_forward_hook(key_hook),
             attention.q.register_forward_hook(query_hook),
             policy.controller.nlm.register_forward_hook(
                 lambda m, i, o: posts.append(o[0].cpu().numpy().copy()))]

    def forward(obs, state, task, start):
        logits, state = policy.step(torch.from_numpy(obs[None]).to(device), state,
                                   torch.tensor([start], device=device), task=task)
        if not torch.isfinite(logits).all():
            raise ValueError('nonfinite logits')
        return logits, state

    def responses():
        p, a = np.stack(posts), np.stack(heads)
        if not np.isfinite(p).all() or not np.isfinite(a).all():
            raise ValueError('nonfinite activation or attention')
        if not np.allclose(a.sum(1), 1, atol=1e-6):
            raise ValueError('invalid attention normalization')
        return dict(post=p, attention_head_mean=a,
                    attention_top20_indices=np.argsort(-a, axis=1, kind='stable')[:, :89].astype(np.uint16))

    def rollout(case, ident, warm=None, limit=None, prefix=False):
        env = MazeEnv(cache, [case['entry']], 0) if case['task'] == 'maze_medium' else FourRoomsEnv(case['seed'], evaluation_seed=case['seed'])
        obs = env.reset()
        if list(env.position) != case['initial_position']:
            raise ValueError('case initial position differs from environment')
        state = policy.initial_state(1) if warm is None else detached(warm)
        posts.clear(); heads.clear()
        images, worlds, actions, positions, probabilities = [], [], [], [], []
        bounds, visible, directions = [], [], []
        success = done = False
        try:
            for step in range(limit or config['max_actions']):
                images.append(obs.copy()); positions.append(list(env.position))
                if case['task'] == 'maze_medium':
                    worlds.append(obs.transpose(1, 2, 0).copy())
                else:
                    native = env.native.unwrapped
                    worlds.append(native.get_frame(tile_size=12, highlight=False))
                    _, mask = native.gen_obs_grid()
                    world_mask = np.zeros((native.height, native.width), bool)
                    for y in range(native.height):
                        for x in range(native.width):
                            v = native.relative_coords(x, y)
                            if v is not None:
                                world_mask[y, x] = mask[v[0], v[1]]
                    bounds.append(native.get_view_exts()); visible.append(world_mask)
                    directions.append(native.agent_dir)
                logits, state = forward(obs, state, case['task'], warm is None and step == 0)
                if warm is not None and step < 20:
                    left = max(0, 40-2*(step+1))
                    if not torch.equal(state.pre[:, :, :left], warm.pre[:, :, 2*(step+1):]) or not torch.equal(state.post[:, :, :left], warm.post[:, :, 2*(step+1):]):
                        raise ValueError('Maze history shift differs from the task-switch protocol')
                actions.append(int(logits.argmax())); probabilities.append(logits.softmax(-1)[0].cpu().numpy())
                obs, done, success = env.step(actions[-1])
                if done:
                    break
            if prefix and done:
                raise ValueError('Maze terminated before the requested switch; choose another case/boundary')
            data = dict(**responses(), observations=np.stack(images), worlds=np.stack(worlds),
                        actions=np.asarray(actions), positions=np.asarray(positions), probabilities=np.stack(probabilities),
                        pre_final=state.pre.cpu().numpy(), post_final=state.post.cpu().numpy())
            if visible:
                data.update(view_bounds=np.asarray(bounds), view_visible=np.stack(visible), view_directions=np.asarray(directions))
            np.savez_compressed(out / (ident + '.npz'), **data)
            meta = dict(case, id=ident, steps=len(actions), ticks=len(posts), ticks_per_action=TICKS[case['task']],
                        success=bool(success), final_position=list(env.position), truncated=not done)
            if prefix:
                meta['remaining_bfs'] = int(cache.maps[case['entry']['sha256']][4][env.position])
            print(ident, meta['steps'], 'actions', 'success=', success, flush=True)
            return meta, detached(state), data
        finally:
            env.close()

    baseline, switched, results, probes = [], [], [], []
    try:
        with torch.inference_mode():
            fresh_data = []
            counts = {'maze_medium': 0, 'fourrooms': 0}
            for case in panel:
                counts[case['task']] += 1
                ident = f"{case['task']}_{counts[case['task']]:02d}"
                meta, _, data = rollout(case, ident)
                baseline.append(meta); fresh_data.append(data)
            if config['mode'] == 'switch':
                maze = next(c for c in panel if c['task'] == 'maze_medium')
                n = config['switch_after_actions']
                if n * TICKS['maze_medium'] < policy.controller.start_pre.shape[-1]:
                    raise ValueError('Maze history must fill the CTM window before switching')
                prefix, warm, _ = rollout(maze, 'maze_prefix', limit=n, prefix=True)
                prefix.update(condition='prefix', pair=None)
                switched.append(prefix)
                for j, (case, fresh, data) in enumerate((c, m, d) for c, m, d in zip(panel, baseline, fresh_data) if c['task'] == 'fourrooms'):
                    ident = f'room{j+1:02d}'
                    np.savez_compressed(out / (ident + '_fresh.npz'), **data)
                    fresh = dict(fresh, id=ident+'_fresh', condition='fresh', pair=j)
                    history, _, _ = rollout(case, ident+'_maze_history', warm=warm)
                    history.update(condition='maze_history', pair=j)
                    switched.extend([fresh, history])
                    count = min(config['probe_actions'], fresh['steps'])
                    captured = {}
                    for mode in ['fresh', 'maze_history']:
                        state = policy.initial_state(1) if mode == 'fresh' else detached(warm)
                        posts.clear(); heads.clear(); probs=[]; logprobs=[]; pres=[]; post_states=[]
                        for k, obs in enumerate(data['observations'][:count]):
                            logits, state = forward(obs, state, 'fourrooms', mode == 'fresh' and k == 0)
                            probs.append(logits.softmax(-1)[0].cpu().numpy())
                            logprobs.append(logits.log_softmax(-1)[0].cpu().numpy())
                            pres.append(state.pre.cpu().numpy().copy()); post_states.append(state.post.cpu().numpy().copy())
                        captured[mode] = dict(**responses(), probabilities=np.stack(probs), logprob=np.stack(logprobs), pre=np.stack(pres), history=np.stack(post_states))
                    metrics=[]
                    a, b = captured['fresh'], captured['maze_history']
                    for k, (p, q) in enumerate(zip(a['probabilities'], b['probabilities'])):
                        # Stable KL from saved logits is preferable to log(underflowed softmax).
                        metrics.append(dict(step=k+1, old_maze_slots=max(0, 40-2*(k+1)),
                            fresh_action=int(p.argmax()), warm_action=int(q.argmax()),
                            total_variation=float(np.abs(p-q).sum()/2),
                            kl_fresh_to_warm=float(np.sum(p*(a['logprob'][k]-b['logprob'][k]))),
                            pre_rms=float(np.sqrt(np.mean((a['pre'][k]-b['pre'][k])**2))),
                            post_rms=float(np.sqrt(np.mean((a['history'][k]-b['history'][k])**2)))))
                    np.savez_compressed(out / (ident+'_probe.npz'), observations=data['observations'][:count],
                        **{mode+'_'+key: values[key] for mode, values in captured.items() for key in ['post','attention_head_mean','attention_top20_indices']})
                    probes.append(dict(file=ident+'_probe.npz', steps=count))
                    results.append(dict(case=case, conditions={'fresh':fresh, 'maze_history':history}, matched_visual_probe=metrics))
        if model_hash(policy) != before or file_sha(checkpoint) != checkpoint_hash:
            raise ValueError('weights changed during inference')
    finally:
        for hook in hooks:
            hook.remove()
    summary = dict(status='complete', config=config, checkpoint_sha256=checkpoint_hash, model_hash=before,
                   weights_unchanged=True, reference_cases=baseline, cases=switched or baseline, probes=probes,
                   results=results, prefix=switched[0] if switched else None)
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary
