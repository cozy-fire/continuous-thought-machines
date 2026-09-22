"""Hash-checked stage commits. Only a completed marker makes a checkpoint recoverable."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import torch

from .data.replay import TransitionStore, file_hash

ROOT = Path(__file__).resolve().parents[2]


def source_manifest() -> dict[str, str]:
    paths = list((ROOT / "tasks/continual_nav").rglob("*.py"))
    paths += [ROOT / "models" / name for name in ("resnet.py", "modules.py")]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)}


def json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("checkpoint reference escapes run directory")
    return path


def reference(root: Path, path: Path) -> dict:
    return dict(path=path.resolve().relative_to(root.resolve()).as_posix(), sha256=file_hash(path))


def verify_reference(root: Path, ref: dict, *, replay: bool = False) -> Path:
    path = contained(root, ref["path"])
    if not path.is_file() or file_hash(path) != ref["sha256"]:
        raise ValueError(f"missing/corrupt checkpoint dependency: {path}")
    if replay:
        TransitionStore(path)  # Validate every shard, not only its manifest.
    return path


def commit(root: Path, stage_key: str, payload: dict) -> Path:
    name = stage_key.replace("/", "__")+"__"+uuid.uuid4().hex
    path = root / "checkpoints" / (name+".pt")
    marker = path.with_suffix(".complete.json")
    atomic_torch(path, payload)
    atomic_json(marker, dict(schema_version=1, stage_key=stage_key, checkpoint=reference(root, path)))
    # A crash before this final switch leaves the previous committed boundary intact.
    atomic_json(root / "checkpoints/latest.json", reference(root, marker))
    return marker


def load(root: Path, marker: Path | None = None, *, expected_config_hash: str | None = None,
         expected_sources: dict | None = None) -> dict:
    root = root.resolve()
    if marker is None:
        marker = verify_reference(root, json.loads((root / "checkpoints/latest.json").read_text(encoding="utf-8")))
    else:
        marker = contained(root, marker.resolve().relative_to(root).as_posix())
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record["schema_version"] != 1 or not marker.name.endswith(".complete.json"):
        raise ValueError("only complete checkpoint markers may be resumed")
    path = verify_reference(root, record["checkpoint"])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"schema_version", "config", "config_hash", "sources", "method", "policy_kind", "seed", "task",
                "current_stage", "next_index", "completed", "stages", "counters", "rng", "models", "world_optimizer",
                "fisher", "kb_ready", "fresh", "pools", "manifests", "vision_source", "vision_export", "evaluations", "finalized"}
    if not required <= payload.keys() or payload["schema_version"] != 1 or payload["current_stage"] != record["stage_key"]:
        raise ValueError("incomplete/inconsistent checkpoint payload")
    if expected_config_hash is not None and payload["config_hash"] != expected_config_hash:
        raise ValueError("checkpoint configuration mismatch")
    if expected_sources is not None and payload["sources"] != expected_sources:
        raise ValueError("checkpoint source manifest mismatch")
    for ref in payload["manifests"].values():
        verify_reference(root, ref)
    if payload["fresh"] is not None:
        verify_reference(root, payload["fresh"], replay=True)
    for ref in payload["pools"].values():
        verify_reference(root, ref, replay=True)
    for ref in (payload["vision_source"], payload["vision_export"]):
        if ref is not None:
            verify_reference(root, ref)
    return payload


def save_artifact(path: Path, payload: dict) -> None:
    atomic_torch(path, payload)
    atomic_json(path.with_suffix(path.suffix+".sha256.json"), {"sha256": file_hash(path)})


def load_artifact(path: Path) -> dict:
    expected = json.loads(path.with_suffix(path.suffix+".sha256.json").read_text(encoding="utf-8"))["sha256"]
    if file_hash(path) != expected:
        raise ValueError("artifact hash mismatch")
    return torch.load(path, map_location="cpu", weights_only=True)
