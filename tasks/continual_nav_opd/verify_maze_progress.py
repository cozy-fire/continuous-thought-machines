"""Real-cache/GPU Maze P engineering verification; never a formal training run."""
import argparse
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from .checkpoint import atomic_json, source_manifest
from .config import REPO_ROOT, load_config
from .data import build_manifest
from .data.maze_curriculum import MazeStateTable, collect_sequences
from .envs import MazeMapCache
from .learning.maze_progress import replay_sequences
from .models import StandalonePolicy, DualPolicy
from .teachers import shortest_path
from tasks.continual_nav.envs.common import pixels


def verify(output_dir,device='cuda:0'):
    root=Path(output_dir).resolve(); root.mkdir(parents=True,exist_ok=False)
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('this acceptance benchmark requires a real CUDA GPU')
    torch.set_num_threads(2); torch.manual_seed(0)
    torch.cuda.set_device(device)
    config=load_config('tasks/continual_nav_opd/configs/full.yaml')
    config=replace(config,training=replace(config.training,device=device))
    process=psutil.Process(); baseline=process.memory_info().rss
    started=time.perf_counter()
    try:
        manifest=build_manifest(REPO_ROOT/config.environment.maze_root)
        splits=[{e.sha256 for e in manifest.entries(s)} for s in ('train','validation','test')]
        if any(splits[a]&splits[b] for a,b in ((0,1),(0,2),(1,2))): raise ValueError('split leakage')
        cache=MazeMapCache(REPO_ROOT/config.environment.maze_root,manifest.train+manifest.validation+manifest.test)
        map_rss=process.memory_info().rss
        table=MazeStateTable(cache,manifest.train)
        table_rss=process.memory_info().rss
        if len(manifest.train)!=44488 or len(table.start_ids)!=7118080: raise ValueError('unexpected full train split')
        # All positions in 32 real maps plus randomly chosen positions in later maps.
        rng=np.random.default_rng(17)
        ids=np.r_[table.start_ids[:table.start_offsets[32]],table.sample(44488,512,rng)]
        for sid in ids:
            m=int(table.map_ids[sid]); cid=table.cache_indices[m]
            pos=tuple(map(int,divmod(int(table.cells[sid]),19)))
            path=shortest_path(np.all(cache.images[cid]==0,axis=-1),pos,cache.positions[cid][1])
            if path[0]!=table.labels[sid] or len(path)!=table.distances[sid]: raise ValueError('BFS oracle mismatch')
        rgb=table.observations(ids[-128:])
        for sid,actual in zip(ids[-128:],rgb):
            image=cache.images[table.cache_indices[table.map_ids[sid]]].copy()
            row,col=divmod(int(table.cells[sid]),19); image[row,col]=(255,0,0)
            np.testing.assert_array_equal(actual,pixels(image))
        kb=StandalonePolicy(config).to(device); dual=DualPolicy(config,kb,kb_ready=True)
        dual.train(); starts=table.sample(44488,100,rng)
        fixed=rng.integers(0,5,size=(5,100))
        # Warm up GPU kernels; measured serial and batched inputs/actions are identical.
        collect_sequences(table,dual,44488,torch.Generator().manual_seed(1),rng,starts=starts,fixed_actions=fixed)
        samples={}; results={}
        for name,batch_size in (('batched',100),('serial',1)):
            measurements=[]
            for _ in range(3):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                begin=time.perf_counter()
                data=collect_sequences(table,dual,44488,torch.Generator().manual_seed(1),rng,
                                       starts=starts,fixed_actions=fixed,forward_batch=batch_size)
                torch.cuda.synchronize()
                measurements.append({'seconds':time.perf_counter()-begin,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            samples[name]=measurements; results[name]=data
        for name in ('obs','labels','actions','valid'):
            torch.testing.assert_close(getattr(results['batched'],name),getattr(results['serial'],name),atol=0,rtol=0)
        for name in ('state_ids','next_ids','terminated'):
            np.testing.assert_array_equal(getattr(results['batched'],name),getattr(results['serial'],name))
        # Actual student-probability collection followed by exactly one real update.
        batch=collect_sequences(table,dual,44488,torch.Generator().manual_seed(19),rng)
        opt=config.optimization.optimizer
        optimizer=torch.optim.Adam([p for p in dual.parameters() if p.requires_grad],lr=opt.lr,eps=opt.eps,betas=opt.betas)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        metrics=replay_sequences(dual,batch,optimizer)
        torch.cuda.synchronize()
        if {float(v['step']) for v in optimizer.state.values()}!={1.}: raise ValueError('Adam count mismatch')
        if metrics['initial_window_grad_norm']<=0 or metrics['visual_grad_norm']<=0: raise ValueError('missing real gradients')
        report={'status':'passed','device':device,'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__,
                'split_maps':[len(x) for x in splits],'total_maps':len(cache.indices),'nonterminal_training_positions':len(table.start_ids),
                'pool_sizes':list(config.maze_progress.pool_sizes),'pool_updates':list(config.maze_progress.pool_updates),
                'updates_per_P':sum(config.maze_progress.pool_updates),'pool_nonterminal_positions':[int(table.start_offsets[n]) for n in config.maze_progress.pool_sizes],
                'map_array_bytes':cache.images.nbytes,'state_table_bytes':table.table_bytes,'total_array_bytes':cache.images.nbytes+table.table_bytes,
                'map_load_seconds':cache.load_seconds,'table_build_seconds':table.build_seconds,
                'map_rss_delta_bytes':map_rss-baseline,'table_rss_delta_bytes':table_rss-map_rss,
                'process_rss_bytes':process.memory_info().rss,'collection_samples':samples,
                'median_collection_seconds':{name:float(np.median([v['seconds'] for v in rows])) for name,rows in samples.items()},
                'real_update':metrics,'replay_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'replay_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'verified_real_BFS_positions':len(ids),'pixel_exact_observations':128,
                'fixed_action_trajectories_equal':True,'elapsed_seconds':time.perf_counter()-started,
                'limits':'Engineering checks on this local GPU; one verification update only, no formal training or learning-success claim.'}
        atomic_json(root/'source.json',source_manifest()); atomic_json(root/'report.json',report)
        cache.close()
        print(__import__('json').dumps(report),flush=True)
        return report
    except Exception as exc:
        atomic_json(root/'failure.json',{'error':repr(exc),'elapsed_seconds':time.perf_counter()-started})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0'); parser.add_argument('--output-dir',required=True)
    args=parser.parse_args(); verify(args.output_dir,args.device)

if __name__=='__main__': main()
