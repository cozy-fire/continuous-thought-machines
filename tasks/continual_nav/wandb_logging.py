"""W&B curves derived from the same audited events written to local JSONL."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .checkpoint import atomic_json
from .config import Config, config_hash, resolved_dict
from .contracts import TASKS


EVENTS = {"stage_enter", "stage_complete", "world_update", "ppo_update", "distill_update",
          "learner_update", "evaluation", "visual_drift"}


def scalar_payload(event: dict, counters: dict[str, int]) -> tuple[dict, dict[str, str]]:
    """Return one W&B history row and the X axis for each metric section."""
    if event["type"] not in EVENTS:
        return {}, {}
    stage = event["stage"]
    parts = stage.split("/")
    task = next((part for part in parts if part in TASKS), "all")
    phase = parts[-2] if parts[-1] in ("collect", "fit") else parts[-1]
    global_steps = event.get("global_training_steps", counters["global_env_steps"] +
                             event.get("consumed", event.get("stage_training_steps", 0)))
    row = dict(stage=stage, event_type=event["type"], global_env_steps=global_steps,
               world_optimizer_updates=event.get("world_optimizer_updates", counters["world_optimizer_updates"]),
               downstream_progress_steps=event["downstream_progress_steps"])
    axes: dict[str, str] = {}

    if event.get("metrics"):
        prefix = f"{parts[0]}_{phase}_{task}"
        axes[prefix] = "world_optimizer_updates" if phase == "W" else "global_env_steps"
        for name, metric in event["metrics"].items():
            if isinstance(metric, (int, float)):
                row[f"{prefix}/{name}"] = metric

    if event["type"] == "evaluation":
        row["evaluation_event"] = event["event"]
        identity = event.get("policy_kind", "kb")
        if event.get("source_task"):
            identity += "_"+event["source_task"]
        row.update(evaluation_id=event.get("evaluation_id"), evaluation_visit=event.get("visit"),
                   evaluation_policy=identity)
        split = event.get("split", "validation")
        for evaluated_task, metrics in event["report"]["tasks"].items():
            prefix = f"eval_{parts[0]}_{split}_{identity}_{evaluated_task}"
            axes[prefix] = "global_env_steps"
            for name, metric in metrics.items():
                if isinstance(metric, (int, float)):
                    row[f"{prefix}/{name}"] = metric
                    row[f"{prefix}/{event['event']}_{name}"] = metric
            if parts[0] in ("pnc", "single", "seq"):
                progress_prefix = f"progress_{split}_{identity}_{evaluated_task}"
                axes[progress_prefix] = "downstream_progress_steps"
                row[f"{progress_prefix}/success_rate"] = metrics["success_rate"]
        prefix = f"eval_{parts[0]}_{split}_{identity}_performance"
        axes[prefix] = "global_env_steps"
        for name in ("elapsed_seconds", "episodes_per_second", "transitions"):
            if name in event["report"]:
                row[f"{prefix}/{name}"] = event["report"][name]

    if event["type"] == "visual_drift":
        for evaluated_task, metrics in event["report"]["tasks"].items():
            prefix = f"drift_{evaluated_task}"
            axes[prefix] = "global_env_steps"
            row[f"{prefix}/mean_kl"] = metrics["mean_kl"]

    if event["type"] == "stage_complete":
        axes["progress"] = "global_env_steps"
        for name, count in event["counters"].items():
            row[f"progress/{name}"] = count

    return row, axes


class WandbLogger:
    def __init__(self, root: Path, config: Config, method: str, seed: int, task: str | None,
                 *, mode: str, resume: bool):
        if mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb mode must be online, offline, or disabled")
        self.run = None
        self._axes: set[str] = set()
        if mode == "disabled":
            return

        import wandb

        project = os.environ.get("WANDB_PROJECT", "tapd-ctm-continual-nav")
        run_id = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
        previous_data_dir = os.environ.get("WANDB_DATA_DIR")
        if previous_data_dir is None:
            os.environ["WANDB_DATA_DIR"] = str(root / "wandb_data")
        try:
            self.run = wandb.init(project=project, entity=os.environ.get("WANDB_ENTITY") or None,
                                  id=run_id, resume="allow" if resume and mode == "online" else None,
                                  mode=mode, dir=str(root),
                                  name=f"{method}-seed{seed}-{task or 'cross-task'}-{root.name}",
                                  group=method, job_type="continual-nav",
                                  config=dict(method=method, seed=seed, task=task, config_hash=config_hash(config),
                                              resolved=resolved_dict(config), logging_backend="wandb", logging_mode=mode))
        finally:
            if previous_data_dir is None:
                os.environ.pop("WANDB_DATA_DIR", None)
        atomic_json(root / "wandb_run.json", dict(project=project, entity=self.run.entity,
                    id=self.run.id, mode=mode, url=self.run.url))

    def log(self, event: dict, counters: dict[str, int]) -> None:
        if self.run is None:
            return
        row, axes = scalar_payload(event, counters)
        for prefix, axis in axes.items():
            if prefix not in self._axes:
                # W has no environment steps; all other curves use training transitions.
                self.run.define_metric(f"{prefix}/*", step_metric=axis)
                self._axes.add(prefix)
        if row:
            self.run.log(row)

    def finish(self, *, exit_code: int = 0) -> None:
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
            import wandb
            wandb.teardown(exit_code=exit_code)
            self.run = None
