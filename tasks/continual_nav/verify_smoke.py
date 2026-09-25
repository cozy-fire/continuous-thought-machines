"""Exercise a complete schema-v2 smoke run and a committed-boundary resume."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from . import checkpoint as ck
from .config import budget_summary, load_config
from .schedule import METHODS, expand_stages
from .train import Runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(Path(__file__).parent / "configs/smoke.yaml")
    config = replace(config, training=replace(config.training, device=args.device))
    if args.device.startswith("cuda:"):
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.reset_peak_memory_stats()
    root = args.output_dir.resolve()
    if root.exists():
        raise FileExistsError("smoke output directory must be new")
    stages = expand_stages(config, METHODS[0])
    assert all(stage.key.phase != "W" for stage in stages)
    runner = Runner(config, METHODS[0], 0, root, wandb_mode="disabled")
    try:
        for _ in range(3):
            runner.run_next()
        assert runner.completed[-1].endswith("/F")
        first = ck.load(root, expected_sources=runner.sources)
        assert first["counters"]["explore_steps"] == config.exploration.steps_per_round
        assert first["visual_optimizer_state"] and first["diagnostics"]
    finally:
        runner.close_logging()
    runner = Runner(config, METHODS[0], 0, root, resume=True, wandb_mode="disabled")
    try:
        while runner.next_index < len(stages):
            runner.run_next()
        runner.finalize()
        assert runner.counters["global_env_steps"] == budget_summary(config)["main_steps"]
        assert runner.counters["explore_steps"] == budget_summary(config)["ta_rounds"] * config.exploration.steps_per_round
        assert runner.counters["x_joint_updates"] == budget_summary(config)["x_joint_updates"]
        assert len([row for row in runner.evaluations if row["family"] == "ta" and row["event"] == "visit_end"]) == 1
        assert len([row for row in runner.evaluations if row["family"] == "pnc" and row["event"] == "visit_end"]) == 3
        assert runner.encoder.permanently_frozen.item()
        assert runner.finalized
    finally:
        runner.close_logging()
    final = ck.load(root, expected_sources=ck.source_manifest())
    assert final["schema_version"] == 2 and final["next_index"] == len(stages)
    report = dict(stages=len(stages)-1, counters=final["counters"], evaluations=len(final["evaluations"]),
                  diagnostics=len(final["diagnostics"]), device=args.device,
                  peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated()
                                             if args.device.startswith("cuda:") else None),
                  peak_cuda_reserved_bytes=(torch.cuda.max_memory_reserved()
                                            if args.device.startswith("cuda:") else None))
    (root / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
