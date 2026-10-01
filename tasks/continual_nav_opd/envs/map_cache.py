"""Validated immutable maps, owned by one run and never included in checkpoints."""
from pathlib import Path
import time
from weakref import WeakValueDictionary
import numpy as np
from tasks.continual_nav.envs.maze import load_map

_shared = WeakValueDictionary()


def shared_map_cache(root, entries):
    """Tools share one pool across live slots; no global strong reference retains it."""
    key = (str(Path(root).resolve()), tuple(entries))
    cache = _shared.get(key)
    if cache is None or cache.images is None:
        cache = MazeMapCache(root, entries)
        _shared[key] = cache
    return cache


class MazeMapCache:
    def __init__(self, root, entries):
        self.root = Path(root).resolve()
        entries = tuple(dict.fromkeys(entries))
        self.indices = {entry: i for i, entry in enumerate(entries)}
        self.images = np.empty((len(entries), 19, 19, 3), dtype=np.uint8)
        self.positions = []
        started = time.perf_counter()
        # Preallocate once: loading never holds two copies of the complete pool.
        for i, entry in enumerate(entries):
            image, start, goal = load_map(self.root, entry)
            self.images[i] = image
            self.positions.append((start, goal))
        self.images.flags.writeable = False
        self.load_seconds = time.perf_counter() - started

    def get(self, root, entry):
        if Path(root).resolve() != self.root:
            raise ValueError('map cache root mismatch')
        index = self.indices[entry]
        start, goal = self.positions[index]
        return self.images[index], start, goal

    def close(self):
        self.images = None
        self.indices.clear()
        self.positions.clear()
