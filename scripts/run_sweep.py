"""Run every fixed configuration on every calibration item (KAYL's learning data).

- The configurations are the `sweep` list in config.yaml: model + effort + thinking language
  (+ format example for allam).
- Each model runs in its own thread, because Groq's limits are per model.
- Every response is cached. If a run stops (network, daily limit), rerun the same command:
  it resumes from the cache without repeating calls.
- Writes results/calibration_sweep.jsonl and results/calibration_sweep.md, and logs every
  call to SQLite. Nothing is written to results/ if any call failed.

The committed sweep was run with gpt-oss on Groq directly. To replay it with no API key:
    python -m scripts.run_sweep --from-cache --route groq
"""

import argparse
import json
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kayl import metrics, runner
from kayl.client import DailyLimitError, call_model
from kayl.config import load_config
from kayl.data import load_tasks
from kayl.grading import OK, grade
from kayl.logdb import connect, log_call

OUT = Path("results")
TYPES = ["math", "knowledge", "intent"]


def run_model(cfg: dict, configs: list[dict], tasks: list[dict]) -> tuple[list[dict], str | None]:
    """Every configuration of one model, in order. Returns (rows, reason for stopping or None)."""
    rows = []
    for c in configs:
        for i, task in enumerate(tasks):
            try:
                r = call_model(cfg, task, c["model"], c["effort"], c["think_lang"],
                               format_example=c.get("format_example"))
            except DailyLimitError as e:
                return rows, f"{c['name']}: daily limit after {i}/{len(tasks)} — {e}"
            rows.append({"config": c["name"], "task": task, "result": r, "grade": grade(task, r.final)})
        n_ok = sum(x["grade"].correct for x in rows if x["config"] == c["name"])
        print(f"done {c['name']:14s} {n_ok}/{len(tasks)} correct", flush=True)
    return rows, None


def summarize(rows: list[dict], configs: list[dict]) -> dict:
    """Per (type, configuration): accuracy, compute per success, average latency, format failures."""
    table = {}
    for t in TYPES:
        for c in configs:
            mine = [x for x in rows if x["config"] == c["name"] and x["task"]["type"] == t]
            if not mine:
                continue
            res = [x["result"] for x in mine]
            n_ok = sum(x["grade"].correct for x in mine)
            lat = [r.latency_s for r in res if r.latency_s is not None]
            table[(t, c["name"])] = {
                "n": len(mine), "correct": n_ok,
                "compute_per_success": metrics.per_success([r.compute_flops for r in res], n_ok),
                "cost_per_success": metrics.per_success([r.cost_usd for r in res], n_ok),
                "avg_latency_s": sum(lat) / len(lat) if lat else None,
                "format_failures": sum(x["grade"].reason not in (OK, "wrong") for x in mine),
                "truncated": sum(r.finish_reason == "length" for r in res),
                "think_lang_followed": sum(r.think_lang_followed for r in res),
                "avg_input_tokens": sum(r.input_tokens for r in res) / len(res),
                "avg_output_tokens": sum(r.output_tokens for r in res) / len(res),
                "errors": sum(r.error is not None for r in res),
            }
    return table


def markdown(table: dict, configs: list[dict]) -> str:
    lines = []
    for t in TYPES:
        lines += [f"\n### {t}\n",
                  "| configuration | accuracy | compute / success (TFLOPs, est.) | avg latency (s) "
                  "| format failures | truncated | think-lang followed | avg in / out tokens |",
                  "|---|---|---|---|---|---|---|---|"]
        for c in configs:
            s = table.get((t, c["name"]))
            if not s:
                continue
            comp = "n/a" if s["compute_per_success"] is None else f"{s['compute_per_success'] / 1e12:.2f}"
            lat = "n/a" if s["avg_latency_s"] is None else f"{s['avg_latency_s']:.2f}"
            name = f"{c['name']} — {c['note']}" if c.get("note") else c["name"]
            lines.append(f"| {name} | {s['correct']}/{s['n']} ({100 * s['correct'] / s['n']:.0f}%) "
                         f"| {comp} | {lat} | {s['format_failures']} | {s['truncated']} "
                         f"| {s['think_lang_followed']}/{s['n']} "
                         f"| {s['avg_input_tokens']:.0f} / {s['avg_output_tokens']:.0f} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    runner.add_run_args(p)
    args = p.parse_args()

    cfg = runner.apply_run_args(load_config(), args)
    configs = cfg["sweep"]
    tasks = load_tasks(cfg, "calibration")
    by_model = defaultdict(list)
    for c in configs:
        by_model[c["model"]].append(c)
    print(f"{len(configs)} configurations × {len(tasks)} calibration items "
          f"= {len(configs) * len(tasks)} calls ({len(by_model)} models in parallel)", flush=True)

    rows, stops = [], []
    with ThreadPoolExecutor(max_workers=len(by_model)) as pool:
        futures = [pool.submit(run_model, cfg, cs, tasks) for cs in by_model.values()]
        for f in futures:
            r, stop = f.result()
            rows += r
            if stop:
                stops.append(stop)

    run_id = uuid.uuid4().hex[:8]
    conn = connect(cfg["paths"]["db"])
    for x in rows:
        log_call(conn, x["result"], run_id=run_id, config_name=x["config"], correct=x["grade"].correct)
    runner.check_no_failures(runner.failed(x["result"] for x in rows), stops)

    OUT.mkdir(exist_ok=True)
    with (OUT / "calibration_sweep.jsonl").open("w", encoding="utf-8") as f:
        for x in rows:
            d = x["result"].to_dict()
            d.pop("content"), d.pop("reasoning")  # full text stays in the cache; numbers only here
            f.write(json.dumps({"config": x["config"], "grade_reason": x["grade"].reason,
                                "correct": x["grade"].correct, **d}, ensure_ascii=False) + "\n")

    table = summarize(rows, configs)
    md = (f"# Calibration sweep (run_id {run_id})\n\n"
          f"All {len(tasks)} calibration items, temperature 0, one run. Compute is an ESTIMATE "
          f"(2 × active params × API tokens). Latency of cached calls is the original call's latency.\n"
          + markdown(table, configs))
    (OUT / "calibration_sweep.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    runner.run(main)
