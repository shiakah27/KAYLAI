# -*- coding: utf-8 -*-
"""ملف المعايرة (calibration profile): ماذا تعلّم كَيْل من بيانات المعايرة.

لكل (نوع مهمة، إعداد): الدقة، والحوسبة لكل إجابة صحيحة، ومتوسط الزمن، والزمن p95.
يُبنى من results/calibration_sweep.jsonl (ناتج run_sweep.py)، أي من المعايرة فقط."""

import json
import math
from collections import defaultdict
from pathlib import Path

from kayl import metrics

SWEEP_RESULTS = Path("results/calibration_sweep.jsonl")


def p95(values: list[float]) -> float | None:
    """الزمن الذي تقع تحته 95% من الاستدعاءات (طريقة nearest-rank، بسيطة وواضحة)."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)]


def build_profile(rows: list[dict]) -> dict:
    """rows: صفوف فيها config, task_type, correct, compute_flops, latency_s.
    يرجع {task_type: {config_name: {...}}}."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task_type"], r["config"])].append(r)
    profile = defaultdict(dict)
    for (task_type, name), rs in groups.items():
        n_ok = sum(bool(r["correct"]) for r in rs)
        lat = [r["latency_s"] for r in rs if r.get("latency_s") is not None]
        profile[task_type][name] = {
            "n": len(rs),
            "accuracy": n_ok / len(rs),
            "compute_per_success": metrics.per_success([r["compute_flops"] for r in rs], n_ok),
            "avg_latency_s": sum(lat) / len(lat) if lat else None,
            "p95_latency_s": p95(lat),
        }
    return dict(profile)


def load_profile(path: str | Path = SWEEP_RESULTS) -> dict:
    """يبني الملف من نتائج المعايرة المحفوظة."""
    with open(path, encoding="utf-8") as f:
        return build_profile([json.loads(line) for line in f])
