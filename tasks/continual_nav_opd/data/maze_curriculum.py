"""Compact Maze P dynamics and synchronous student-probability collection.

Only training-manifest entries enter this table. The shared cache also owns the
other splits, but no evaluation entry can be sampled through a state index.
"""
from collections import deque
from dataclasses import dataclass
import time

import numpy as np
import psutil
import torch
from tasks.continual_nav.envs.maze import MOVES
from ..models.state import select_state, replace_slots


class MazeStateTable:
    def __init__(self, cache, entries):
        resident_before = psutil.Process().memory_info().rss
        entries = tuple(entries)
        if not entries or len({e.sha256 for e in entries}) != len(entries):
            raise ValueError('Maze P needs distinct ordered training maps')
        self.cache = cache
        self.entries = entries
        self.cache_indices = np.array([cache.indices[e] for e in entries], dtype=np.int32)
        counts = np.count_nonzero(np.any(cache.images[self.cache_indices] != 0, axis=-1), axis=(1,2))
        self.offsets = np.r_[0, np.cumsum(counts)].astype(np.int32)
        self.start_offsets = (self.offsets - np.arange(len(entries)+1)).astype(np.int32)
        count = int(self.offsets[-1])
        self.map_ids = np.empty(count, dtype=np.int32)
        self.cells = np.empty(count, dtype=np.int16)
        self.labels = np.empty(count, dtype=np.uint8)
        self.distances = np.empty(count, dtype=np.int16)
        self.transitions = np.empty((count,5), dtype=np.int32)
        self.terminal = np.zeros(count, dtype=np.bool_)
        self.start_ids = np.empty(int(self.start_offsets[-1]), dtype=np.int32)
        started = time.perf_counter()
        for m, cache_id in enumerate(self.cache_indices):
            walls = np.all(cache.images[cache_id] == 0, axis=-1)
            goal = cache.positions[cache_id][1]
            distance = np.full((19,19), -1, dtype=np.int16)
            distance[goal] = 0
            queue = deque([goal])
            while queue:
                row,col = queue.popleft()
                for dr,dc in MOVES[:4]:
                    y,x = row+dr,col+dc
                    if 0 <= y < 19 and 0 <= x < 19 and not walls[y,x] and distance[y,x] < 0:
                        distance[y,x] = distance[row,col]+1
                        queue.append((y,x))
            cells = np.flatnonzero(~walls)
            if np.any(distance.flat[cells] < 0):
                raise ValueError('all traversable training positions must reach the goal')
            begin,end = self.offsets[m:m+2]
            ids = np.arange(begin,end,dtype=np.int32)
            lookup = np.full((19,19), -1, dtype=np.int32)
            lookup.flat[cells] = ids
            rows,cols = cells//19,cells%19
            self.map_ids[begin:end] = m
            self.cells[begin:end] = cells
            self.distances[begin:end] = distance.flat[cells]
            self.labels[begin:end] = 4
            is_goal = self.distances[begin:end] == 0
            self.terminal[begin:end] = is_goal
            self.start_ids[self.start_offsets[m]:self.start_offsets[m+1]] = ids[~is_goal]
            for a,(dr,dc) in enumerate(MOVES):
                y,x = rows+dr,cols+dc
                inside = (y>=0)&(y<19)&(x>=0)&(x<19)
                next_ids = lookup[np.clip(y,0,18),np.clip(x,0,18)]
                next_ids = np.where(inside & (next_ids >= 0), next_ids, ids)
                self.transitions[begin:end,a] = next_ids
            # The first descending neighbor is the same tie break as forward BFS.
            for a in reversed(range(4)):
                next_ids = self.transitions[begin:end,a]
                chosen = self.distances[begin:end]-1 == distance.flat[self.cells[next_ids]]
                self.labels[begin:end][chosen & ~is_goal] = a
        for value in self.__dict__.values():
            if isinstance(value,np.ndarray): value.flags.writeable = False
        self.build_seconds = time.perf_counter()-started
        # PIL NEAREST samples the center of each destination pixel.
        self.resize_indices = np.floor((np.arange(84)+0.5)*19/84).astype(np.intp)
        self.resize_indices.flags.writeable = False
        self.table_bytes = sum(v.nbytes for v in self.__dict__.values() if isinstance(v,np.ndarray))
        self.process_rss_bytes = psutil.Process().memory_info().rss
        self.rss_delta_bytes = self.process_rss_bytes - resident_before

    def sample(self, pool_maps, count, rng):
        if not 0 < pool_maps <= len(self.entries):
            raise ValueError('pool exceeds training prefix')
        return self.start_ids[rng.integers(int(self.start_offsets[pool_maps]),size=count)]

    def observations(self, ids):
        ids = np.asarray(ids,dtype=np.int32)
        small = self.cache.images[self.cache_indices[self.map_ids[ids]]].copy()
        cells = self.cells[ids]
        small[np.arange(len(ids)),cells//19,cells%19] = (255,0,0)
        indices = self.resize_indices
        rgb = small[:,indices[:,None],indices[None,:],:].transpose(0,3,1,2).copy()
        return rgb


@dataclass
class MazeSequences:
    obs: torch.Tensor  # CPU uint8 [5,100,3,84,84], action-before observations.
    labels: torch.Tensor
    actions: torch.Tensor
    valid: torch.Tensor
    state_ids: np.ndarray
    next_ids: np.ndarray
    terminated: np.ndarray
    collect_seconds: float


@torch.no_grad()
def collect_sequences(table, student, pool_maps, action_rng, start_rng, *,
                      starts=None, fixed_actions=None, forward_batch=100):
    """One unchanged policy, fresh learned windows, and no episode auto-reset.

    starts/fixed_actions/forward_batch expose the serial verification reference;
    production always uses one batch of 100 independent sequences per decision.
    """
    device = next(student.parameters()).device
    ids = table.sample(pool_maps,100,start_rng).copy() if starts is None else np.array(starts,dtype=np.int32,copy=True)
    if ids.shape != (100,) or np.any((ids<0)|(ids>=len(table.cells))):
        raise ValueError('invalid collection state indices')
    if np.any(table.terminal[ids]) or np.any(table.map_ids[ids]>=pool_maps):
        raise ValueError('collection needs 100 nonterminal starts')
    shape = (5,100)
    obs = np.zeros((*shape,3,84,84),dtype=np.uint8)
    labels = np.zeros(shape,dtype=np.int64)
    actions = np.zeros(shape,dtype=np.int64)
    valid = np.zeros(shape,dtype=np.bool_)
    state_ids = np.full(shape,-1,dtype=np.int32)
    next_ids = np.full(shape,-1,dtype=np.int32)
    terminated = np.zeros(shape,dtype=np.bool_)
    live = np.ones(100,dtype=np.bool_)
    state = student.initial_state(100)
    begin = time.perf_counter()
    for t in range(5):
        slots = np.flatnonzero(live)
        if not len(slots): break
        rgb = table.observations(ids[slots])
        outputs = []
        for low in range(0,len(slots),forward_batch):
            part = slots[low:low+forward_batch]
            indices = torch.as_tensor(part,device=device)
            logits,proposed = student.step(torch.from_numpy(rgb[low:low+forward_batch]).to(device),
                select_state(state,indices), torch.zeros(len(part),dtype=torch.bool,device=device),task='maze_medium')
            state = replace_slots(state,indices,proposed)
            outputs.append(logits.cpu())
        logits = torch.cat(outputs)
        if not torch.isfinite(logits).all(): raise ValueError('nonfinite Maze collector logits')
        chosen = (torch.multinomial(logits.softmax(-1),1,generator=action_rng).squeeze(-1).numpy()
                  if fixed_actions is None else np.asarray(fixed_actions[t,slots]))
        if np.any((chosen < 0)|(chosen > 4)): raise ValueError('invalid collected action')
        obs[t,slots] = rgb
        labels[t,slots] = table.labels[ids[slots]]
        actions[t,slots] = chosen
        valid[t,slots] = True
        state_ids[t,slots] = ids[slots]
        ids[slots] = table.transitions[ids[slots],chosen]
        next_ids[t,slots] = ids[slots]
        terminated[t,slots] = table.terminal[ids[slots]]
        live[slots] = ~terminated[t,slots]
    return MazeSequences(torch.from_numpy(obs),torch.from_numpy(labels),torch.from_numpy(actions),
                         torch.from_numpy(valid),state_ids,next_ids,terminated,time.perf_counter()-begin)
