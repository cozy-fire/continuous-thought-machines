"""Metrics over actual episodes; failed episodes never masquerade as zero-length successes."""
from __future__ import annotations

import statistics


def episode_metrics(episodes: list[dict]) -> dict:
    if not episodes:
        raise ValueError("evaluation requires at least one episode")
    successes = [r for r in episodes if r["success"]]
    ratios = [r["length"]/r["shortest_path"] for r in successes if r.get("shortest_path") is not None]
    return dict(episodes=len(episodes), successes=len(successes), failures=len(episodes)-len(successes),
                success_rate=len(successes)/len(episodes), mean_return=statistics.mean(r["return"] for r in episodes),
                mean_length_all=statistics.mean(r["length"] for r in episodes),
                mean_length_success=statistics.mean(r["length"] for r in successes) if successes else None,
                mean_success_path_ratio=statistics.mean(ratios) if ratios else None)


def forgetting(matrix: list[dict[str, float]]) -> list[dict[str, float]]:
    best, rows = {}, []
    for scores in matrix:
        for task, value in scores.items():
            best[task] = max(best.get(task, value), value)
        rows.append({task: best[task]-value for task, value in scores.items()})
    return rows


def seed_summary(values: dict[int, float]) -> dict:
    if not values:
        raise ValueError("no seed measurements")
    return dict(per_seed=values, mean=statistics.mean(values.values()),
                sample_std=statistics.stdev(values.values()) if len(values) > 1 else None)
