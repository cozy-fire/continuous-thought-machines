"""Deterministic Maze splits, excluding duplicated files outside split/0."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from ..contracts import Split


@dataclass(frozen=True)
class MazeEntry:
    path: str  # POSIX path relative to maze_root; never an absolute machine path.
    sha256: str


@dataclass(frozen=True)
class MazeManifest:
    train: tuple[MazeEntry, ...]
    validation: tuple[MazeEntry, ...]
    test: tuple[MazeEntry, ...]
    validation_panel: tuple[MazeEntry, ...]
    test_panel: tuple[MazeEntry, ...]
    drift_panel: tuple[MazeEntry, ...]
    schema_version: int = 1

    def entries(self, split: Split, *, panel: bool = False) -> tuple[MazeEntry, ...]:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"unknown split: {split}")
        if panel and split == "train":
            raise ValueError("train has no evaluation panel")
        return getattr(self, f"{split}_panel" if panel else split)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, root: str | Path) -> MazeManifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        names = ("train", "validation", "test", "validation_panel", "test_panel", "drift_panel")
        if set(raw) != {*names, "schema_version"} or raw["schema_version"] != 1:
            raise ValueError("unsupported manifest schema")
        manifest = cls(**{name: tuple(MazeEntry(**item) for item in raw[name]) for name in names})
        verify_manifest(manifest, root)
        return manifest


def entry_path(root: str | Path, entry: MazeEntry) -> Path:
    root = Path(root).resolve()
    relative = Path(entry.path)
    if relative.is_absolute() or len(relative.parts) != 3 or relative.parts[:2] not in (
            ("train", "0"), ("test", "0")) or relative.suffix != ".png":
        raise ValueError(f"invalid maze path: {entry.path}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"maze path escapes root: {entry.path}")
    return path


def read_entry(root: str | Path, entry: MazeEntry) -> bytes:
    data = entry_path(root, entry).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.sha256:
        raise ValueError(f"maze content changed: {entry.path}")
    return data


def _unique(entries: tuple[MazeEntry, ...]) -> tuple[MazeEntry, ...]:
    # Input is sorted by (hash,path), so each representative is deterministic.
    seen = set()
    result = []
    for entry in entries:
        if entry.sha256 not in seen:
            seen.add(entry.sha256)
            result.append(entry)
    return tuple(result)


def build_manifest(root: str | Path, *, validation_count: int = 512,
                   validation_episodes: int = 200, test_episodes: int = 200,
                   drift_episodes: int = 32) -> MazeManifest:
    root = Path(root).resolve()
    if min(validation_count, validation_episodes, test_episodes, drift_episodes) <= 0:
        raise ValueError("manifest counts must be positive")
    if not drift_episodes <= validation_episodes <= validation_count:
        raise ValueError("invalid panel sizes")

    def scan(split: str) -> tuple[MazeEntry, ...]:
        # Do not recurse: split-root PNGs may duplicate the opposite split.
        entries = [MazeEntry(p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in (root / split / "0").glob("*.png")]
        if not entries:
            raise ValueError(f"no mazes in {root / split / '0'}")
        return tuple(sorted(entries, key=lambda e: (e.sha256, e.path)))

    train_all, test = scan("train"), scan("test")
    if {e.sha256 for e in train_all} & {e.sha256 for e in test}:
        raise ValueError("train/test content hash leakage")
    train_unique, test_unique = _unique(train_all), _unique(test)
    if len(train_unique) <= validation_count or len(test_unique) < test_episodes:
        raise ValueError("insufficient distinct maps; refusing to shrink the requested splits")
    validation_hashes = {e.sha256 for e in train_unique[:validation_count]}
    # Move every duplicate of a selected hash, not just one representative.
    validation = tuple(e for e in train_all if e.sha256 in validation_hashes)
    train = tuple(e for e in train_all if e.sha256 not in validation_hashes)
    panel = train_unique[:validation_episodes]
    return MazeManifest(train, validation, test, panel, test_unique[:test_episodes], panel[:drift_episodes])


def verify_manifest(manifest: MazeManifest, root: str | Path) -> None:
    hash_sets = []
    for split in ("train", "validation", "test"):
        entries = manifest.entries(split)
        if not entries or len({e.path for e in entries}) != len(entries):
            raise ValueError(f"empty/duplicate manifest entries: {split}")
        expected_parent = "test" if split == "test" else "train"
        for entry in entries:
            if Path(entry.path).parts[0] != expected_parent:
                raise ValueError(f"wrong source directory for {split}: {entry.path}")
            read_entry(root, entry)
        hash_sets.append({e.sha256 for e in entries})
    if any(hash_sets[i] & hash_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("manifest split leakage")
    for panel, source in ((manifest.validation_panel, manifest.validation),
                          (manifest.test_panel, manifest.test),
                          (manifest.drift_panel, manifest.validation_panel)):
        if not panel or not set(panel) <= set(source) or len({e.sha256 for e in panel}) != len(panel):
            raise ValueError("invalid evaluation panel")
