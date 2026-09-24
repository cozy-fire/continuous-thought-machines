"""Bounded raw-pixel storage, per-task top-K replacement, and uniform W sampling."""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import replace
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import shutil
import uuid
from typing import cast

import numpy as np
import torch

from ..contracts import OBS_SHAPE, TASKS, TaskKey, Transition, WorldBatch

META = np.dtype([("action", "<i8"), ("terminated", "?"), ("truncated", "?"),
                 ("episode_start", "?"), ("episode_id", "<i8"), ("episode_step", "<i8"),
                 ("transition_id", "<i8"), ("error", "<f8")])


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate(record: Transition, task: TaskKey, encoder_version: int, world_version: int) -> None:
    if record.task_key != task or record.encoder_version != encoder_version or record.world_model_version != world_version:
        raise ValueError("mixed task or version in one transition store")
    for image in (record.obs, record.transition_next_obs):
        if image.shape != OBS_SHAPE or image.dtype != np.uint8:
            raise ValueError("replay requires raw uint8 [3,84,84] images")
    if type(record.action) is not int or not 0 <= record.action < 5:
        raise ValueError("invalid policy action in replay")
    if record.terminated and record.truncated:
        raise ValueError("success and timeout cannot both be true")
    if min(record.transition_id, record.episode_id) < 0 or record.episode_step < 1:
        raise ValueError("invalid transition/episode identity")


def _row(record: Transition, error: float | None) -> tuple:
    return (record.action, record.terminated, record.truncated, record.episode_start,
            record.episode_id, record.episode_step, record.transition_id,
            np.nan if error is None else error)


class ShardedWriter:
    """At most shard_size image pairs are buffered; manifest publication is atomic."""
    def __init__(self, directory: str | Path, *, task: TaskKey, encoder_version: int,
                 world_version: int, source: str, source_stage: str, expected_count: int,
                 shard_size: int = 2048, snapshot_id: str | None = None):
        if task not in TASKS or source not in ("W", "X") or expected_count < 0 or not 1 <= shard_size <= 2048:
            raise ValueError("invalid transition-store metadata")
        if min(encoder_version, world_version) < 0 or not source_stage:
            raise ValueError("invalid source version/stage")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.task, self.encoder_version, self.world_version = task, encoder_version, world_version
        self.source, self.expected_count, self.shard_size = source, expected_count, shard_size
        self.header = dict(schema_version=1, task=task, encoder_version=encoder_version,
                           world_version=world_version, source=source, source_stage=source_stage,
                           snapshot_id=snapshot_id, expected_count=expected_count)
        self.count = 0
        self._records: list[Transition] = []
        self._errors: list[float | None] = []
        self._shards: list[dict] = []
        self._closed = False
        self._ids: set[int] = set()

    def append(self, record: Transition, *, error: float | None = None) -> None:
        if self._closed or self.count >= self.expected_count:
            raise RuntimeError("closed store or exceeded exact transition budget")
        _validate(record, self.task, self.encoder_version, self.world_version)
        if record.transition_id in self._ids:
            raise ValueError("duplicate transition_id")
        if self.source == "X" and (error is None or not math.isfinite(error) or error < 0):
            raise ValueError("X records require a finite nonnegative raw L2 score")
        if self.source == "W" and error is not None:
            raise ValueError("W fresh data has no curiosity scores")
        # Own the bytes: vector environments may recycle their observation arrays.
        self._records.append(replace(record, obs=record.obs.copy(),
                                     transition_next_obs=record.transition_next_obs.copy()))
        self._errors.append(error)
        self._ids.add(record.transition_id)
        self.count += 1
        if len(self._records) == self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._records:
            return
        name = f"shard_{len(self._shards):06d}.npz"
        path = self.directory / name
        temporary = path.with_suffix(".tmp")
        metadata = np.array([_row(r, e) for r, e in zip(self._records, self._errors)], dtype=META)
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, obs=np.stack([r.obs for r in self._records]),
                                next_obs=np.stack([r.transition_next_obs for r in self._records]),
                                metadata=metadata)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        self._shards.append(dict(path=name, count=len(self._records), sha256=file_hash(path)))
        self._records.clear()
        self._errors.clear()

    def finish(self) -> Path:
        if self._closed or self.count != self.expected_count:
            raise RuntimeError("cannot publish an incomplete or already closed store")
        self._flush()
        manifest = self.directory / "manifest.json"
        _atomic_json(manifest, {**self.header, "count": self.count, "shards": self._shards})
        self._closed = True
        return manifest


class TransitionStore:
    """Validated disk store with an optional, explicitly scoped uint8 memory cache."""
    def __init__(self, manifest: str | Path):
        self.manifest = Path(manifest).resolve()
        self.header = json.loads(self.manifest.read_text(encoding="utf-8"))
        h = self.header
        if h["schema_version"] != 1 or h["task"] not in TASKS or h["source"] not in ("W", "X"):
            raise ValueError("unsupported replay manifest")
        self.task, self.source, self.count = h["task"], h["source"], h["count"]
        self.encoder_version, self.world_version = h["encoder_version"], h["world_version"]
        self._ends, total = [], 0
        for shard in h["shards"]:
            relative = Path(shard["path"])
            if relative.name != shard["path"] or not 1 <= shard["count"] <= 2048:
                raise ValueError("invalid shard manifest entry")
            path = (self.manifest.parent / relative).resolve()
            if path.parent != self.manifest.parent or file_hash(path) != shard["sha256"]:
                raise ValueError("missing, escaping or corrupt replay shard")
            total += shard["count"]
            self._ends.append(total)
        if total != self.count or total != h["expected_count"]:
            raise ValueError("replay count mismatch")
        self._cache_index = -1
        self._cache: dict[str, np.ndarray] = {}
        self._memory: dict[str, np.ndarray] | None = None

    def preload(self) -> int:
        """Decode each shard once without holding a second full-pool copy."""
        if self._memory is not None:
            return sum(a.nbytes for a in self._memory.values())
        arrays = dict(obs=np.empty((self.count, *OBS_SHAPE), dtype=np.uint8),
                      next_obs=np.empty((self.count, *OBS_SHAPE), dtype=np.uint8),
                      metadata=np.empty(self.count, dtype=META))
        start = 0
        try:
            for index, end in enumerate(self._ends):
                data = self._load(index)
                for name in arrays:
                    arrays[name][start:end] = data[name]
                start = end
            # Publish only a completely loaded cache; failures cannot expose partial data.
            self._memory = arrays
        finally:
            self._cache_index, self._cache = -1, {}
        return sum(a.nbytes for a in arrays.values())

    def release(self) -> None:
        self._memory = None
        self._cache_index, self._cache = -1, {}

    def take_arrays(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Return owning batch arrays in request order, including repeated indices."""
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.count):
            raise IndexError("invalid replay indices")
        if self._memory is not None:
            return {name: array[indices] for name, array in self._memory.items()}
        result = dict(obs=np.empty((len(indices), *OBS_SHAPE), dtype=np.uint8),
                      next_obs=np.empty((len(indices), *OBS_SHAPE), dtype=np.uint8),
                      metadata=np.empty(len(indices), dtype=META))
        shards = np.searchsorted(self._ends, indices, side="right")
        for shard in np.unique(shards):
            slots = np.flatnonzero(shards == shard)
            start = 0 if shard == 0 else self._ends[shard-1]
            data = self._load(int(shard))
            for name in result:
                result[name][slots] = data[name][indices[slots]-start]
        return result

    def __len__(self) -> int:
        return self.count

    def _load(self, index: int) -> dict[str, np.ndarray]:
        if index != self._cache_index:
            shard = self.header["shards"][index]
            with np.load(self.manifest.parent / shard["path"], allow_pickle=False) as data:
                if set(data.files) != {"obs", "next_obs", "metadata"}:
                    raise ValueError("unexpected replay payload fields")
                arrays = {name: data[name] for name in data.files}
            size = shard["count"]
            for name in ("obs", "next_obs"):
                if arrays[name].shape != (size, *OBS_SHAPE) or arrays[name].dtype != np.uint8:
                    raise ValueError("invalid replay image array")
            if arrays["metadata"].shape != (size,) or arrays["metadata"].dtype != META:
                raise ValueError("invalid replay metadata array")
            self._cache_index, self._cache = index, arrays
        return self._cache

    def take(self, indices: list[int] | np.ndarray) -> list[Transition]:
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        result: list[Transition | None] = [None] * len(indices)
        for output_index, index in enumerate(indices):
            if not 0 <= index < self.count:
                raise IndexError(index)
            shard = bisect_right(self._ends, int(index))
            start = 0 if shard == 0 else self._ends[shard-1]
            groups[shard].append((output_index, int(index)-start))
        # Group random indices by shard to avoid decompressing the same shard per sample.
        for shard, requests in groups.items():
            data = self._load(shard)
            for output_index, i in requests:
                m = data["metadata"][i]
                result[output_index] = Transition(data["obs"][i].copy(), data["next_obs"][i].copy(),
                    int(m["action"]), bool(m["terminated"]), bool(m["truncated"]), bool(m["episode_start"]),
                    int(m["episode_id"]), int(m["episode_step"]), self.task, self.encoder_version,
                    self.world_version, int(m["transition_id"]))
        return cast(list[Transition], result)

    def error(self, index: int) -> float:
        if not 0 <= index < self.count:
            raise IndexError(index)
        shard = bisect_right(self._ends, index)
        start = 0 if shard == 0 else self._ends[shard-1]
        return float(self._load(shard)["metadata"][index-start]["error"])


def _remove_child(path: Path, parent: Path) -> None:
    # Cleanup is limited to directories created immediately below this owned parent.
    path, parent = path.resolve(), parent.resolve()
    if path.parent != parent or not path.is_dir():
        raise ValueError("refusing cleanup outside the owned replay directory")
    shutil.rmtree(path)


class HighErrorBuilder:
    """One X round. Heap stores scores/slot IDs; pixel capacity lives in a disk memmap."""
    def __init__(self, bank: ReplayBank, task: TaskKey, *, encoder_version: int,
                 world_version: int, source_stage: str, expected_count: int):
        if task not in TASKS or expected_count <= 0 or min(encoder_version, world_version) < 0 or not source_stage:
            raise ValueError("invalid exploration round")
        self.bank, self.task = bank, task
        self.encoder_version, self.world_version = encoder_version, world_version
        self.source_stage, self.expected_count = source_stage, expected_count
        self.capacity = min(bank.capacity, expected_count)
        self._pending_parent = bank.root / task / "pending"
        self.directory = self._pending_parent / uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self._pixels = np.lib.format.open_memmap(self.directory / "pixels.npy", mode="w+",
                            dtype=np.uint8, shape=(self.capacity, 2, *OBS_SHAPE))
        self._meta = np.zeros(self.capacity, dtype=META)
        self._heap: list[tuple[float, int, int]] = []
        self._ids: set[int] = set()
        self._closed = False

    def offer(self, record: Transition, error: float, *, source: str = "X") -> None:
        if self._closed or len(self._ids) >= self.expected_count:
            raise RuntimeError("closed builder or exceeded exploration budget")
        _validate(record, self.task, self.encoder_version, self.world_version)
        if source != "X" or not math.isfinite(error) or error < 0:
            raise ValueError("only X transitions with finite raw L2 scores may enter top-K")
        if record.transition_id in self._ids:
            raise ValueError("duplicate exploration transition_id")
        self._ids.add(record.transition_id)
        key = (float(error), -record.transition_id)
        if len(self._heap) < self.capacity:
            slot = len(self._heap)
            heapq.heappush(self._heap, (*key, slot))
        elif key > self._heap[0][:2]:
            slot = self._heap[0][2]
            heapq.heapreplace(self._heap, (*key, slot))
        else:
            return
        self._pixels[slot, 0] = record.obs
        self._pixels[slot, 1] = record.transition_next_obs
        self._meta[slot] = _row(record, error)

    def finish(self) -> TransitionStore:
        if self._closed or len(self._ids) != self.expected_count:
            raise RuntimeError("cannot publish incomplete exploration top-K")
        version_dir = self.bank.root / self.task / "versions" / uuid.uuid4().hex
        writer = ShardedWriter(version_dir, task=self.task, encoder_version=self.encoder_version,
                               world_version=self.world_version, source="X", source_stage=self.source_stage,
                               expected_count=len(self._heap), shard_size=self.bank.shard_size)
        # Largest error first; equal errors use ascending global transition ID.
        for score, negative_id, slot in sorted(self._heap, reverse=True):
            m = self._meta[slot]
            record = Transition(self._pixels[slot, 0], self._pixels[slot, 1], int(m["action"]),
                bool(m["terminated"]), bool(m["truncated"]), bool(m["episode_start"]),
                int(m["episode_id"]), int(m["episode_step"]), self.task, self.encoder_version,
                self.world_version, -negative_id)
            writer.append(record, error=score)
        manifest = writer.finish()
        store = TransitionStore(manifest)
        self.bank._publish(self.task, manifest)
        self.abort()  # Release only this builder's scratch space after successful publication.
        return store

    def abort(self) -> None:
        if not self._closed:
            self._pixels.flush()
            self._pixels._mmap.close()
            self._closed = True
            _remove_child(self.directory, self._pending_parent)


class ReplayBank:
    def __init__(self, root: str | Path, *, capacity: int = 50000, shard_size: int = 2048):
        if capacity <= 0 or not 1 <= shard_size <= 2048:
            raise ValueError("invalid replay capacity/shard size")
        self.root, self.capacity, self.shard_size = Path(root).resolve(), capacity, shard_size
        self.root.mkdir(parents=True, exist_ok=True)

    def begin(self, task: TaskKey, *, encoder_version: int, world_version: int,
              source_stage: str, expected_count: int) -> HighErrorBuilder:
        return HighErrorBuilder(self, task, encoder_version=encoder_version, world_version=world_version,
                                source_stage=source_stage, expected_count=expected_count)

    def _publish(self, task: TaskKey, manifest: Path) -> None:
        task_dir = self.root / task
        _atomic_json(task_dir / "latest.json", dict(manifest=manifest.relative_to(task_dir).as_posix(),
                                                    sha256=file_hash(manifest)))

    def latest(self, task: TaskKey) -> TransitionStore | None:
        if task not in TASKS:
            raise ValueError("unknown replay task")
        task_dir = self.root / task
        pointer = task_dir / "latest.json"
        if not pointer.exists():
            return None
        value = json.loads(pointer.read_text(encoding="utf-8"))
        path = (task_dir / value["manifest"]).resolve()
        if not path.is_relative_to(task_dir / "versions") or file_hash(path) != value["sha256"]:
            raise ValueError("invalid latest replay pointer")
        store = TransitionStore(path)
        if store.task != task or store.source != "X":
            raise ValueError("latest pool has wrong task/source")
        return store

    def prune(self, task: TaskKey, *, protected_manifests: tuple[Path, ...] = ()) -> None:
        """Call only after a phase checkpoint commits; retain its referenced old pools."""
        current = self.latest(task)
        if current is None:
            return
        parent = self.root / task / "versions"
        keep = {current.manifest.parent, *(Path(p).resolve().parent for p in protected_manifests)}
        for child in parent.iterdir():
            if child.is_dir() and child.resolve() not in keep:
                _remove_child(child, parent)


def sample_world_batch(fresh: TransitionStore, high: TransitionStore | None, *, task: TaskKey,
                       batch_size: int, high_fraction: float, rng: np.random.Generator) -> WorldBatch:
    if fresh.task != task or fresh.source != "W" or len(fresh) == 0:
        raise ValueError("fresh store must be nonempty W data for the current task")
    if high is not None and (high.task != task or high.source != "X"):
        raise ValueError("cross-task or non-X high-error sampling is forbidden")
    if batch_size < 1 or not 0 <= high_fraction < 1 or not float(batch_size*high_fraction).is_integer():
        raise ValueError("invalid mixed batch allocation")
    high_count = int(batch_size*high_fraction) if high is not None and len(high) else 0
    fresh_count = batch_size-high_count
    # Both pools are uniform with replacement; scores only determine membership.
    arrays = fresh.take_arrays(rng.integers(len(fresh), size=fresh_count))
    if high_count:
        extra = high.take_arrays(rng.integers(len(high), size=high_count))
        arrays = {name: np.concatenate((value, extra[name])) for name, value in arrays.items()}
    return WorldBatch(torch.from_numpy(arrays["obs"]),
                      torch.from_numpy(arrays["next_obs"]),
                      torch.from_numpy(arrays["metadata"]["action"].copy()),
                      torch.from_numpy(arrays["metadata"]["transition_id"].copy()),
                      torch.arange(batch_size) >= fresh_count, task)
