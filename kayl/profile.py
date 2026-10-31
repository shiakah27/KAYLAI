"""Calibration profile: what KAYL learned from the calibration data.

For each (task type, configuration): accuracy, compute per success, average latency and p95
latency. Built from results/calibration_sweep.jsonl only, so it never sees test data."""

import json
import math
from collections import defaultdict
from pathlib import Path

from kayl import metrics

SWEEP_RESULTS = Path("results/calibration_sweep.jsonl")


def p95(values: list[float]) -> float | None:
    """95th percentile, nearest-rank method."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)]


def build_profile(rows: list[dict]) -> dict:
    """rows have config, task_type, correct, compute_flops and latency_s.
    Returns {task_type: {config_name: stats}}."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task_type"], r["config"])].append(r)
    profile = defaultdict(dict)
    for (task_type, name), rs in groups.items():
        n_ok = sum(bool(r["correct"]) for r in rs)
        lat = [r["latency_s"] for r in rs if r.get("latency_s") is not None]
        profile[task_type][name] = {
            "n": len(rs),
            "correct": n_ok,
            "accuracy": n_ok / len(rs),
            "compute_per_success": metrics.per_success([r["compute_flops"] for r in rs], n_ok),
            "avg_latency_s": sum(lat) / len(lat) if lat else None,
            "p95_latency_s": p95(lat),
        }
    return dict(profile)


def load_profile(path: str | Path = SWEEP_RESULTS) -> dict:
    """Build the profile from the saved calibration results."""
    with open(path, encoding="utf-8") as f:
        return build_profile([json.loads(line) for line in f])
