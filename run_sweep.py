# -*- coding: utf-8 -*-
"""
كَيْل — الخطوة 5: تشغيل كل الإعدادات الثابتة على كل أسئلة المعايرة.

ماذا يفعل:
  - الإعدادات في config.yaml (sweep): نموذج + مستوى تفكير + لغة تفكير (+ مثال الشكل لـ allam).
  - يشغّل كل نموذج في خيط (thread) مستقل، لأن حدود Groq لكل نموذج على حدة.
  - كل رد محفوظ في الكاش: إذا توقف التشغيل (انقطاع، أو الحد اليومي)، أعد نفس الأمر
    وسيكمل من حيث توقف بدون استدعاءات مكررة.
  - يحفظ كل النتائج في results/calibration_sweep.jsonl وجدولًا في results/calibration_sweep.md،
    ويسجل كل استدعاء في SQLite.

التشغيل:
    python run_sweep.py
"""

import json
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kayl import metrics
from kayl.client import DailyLimitError, call_model
from kayl.config import load_config
from kayl.grading import OK, grade
from kayl.logdb import connect, log_call
from run_eval import load_tasks

OUT = Path("results")
TYPES = ["math", "knowledge", "intent"]


def run_model(cfg: dict, configs: list[dict], tasks: list[dict]) -> tuple[list[dict], str | None]:
    """كل إعدادات نموذج واحد، بالتسلسل. يرجع (النتائج، سبب التوقف إن وُجد)."""
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
    """لكل (نوع، إعداد): الدقة، الحوسبة لكل نجاح، متوسط الزمن، وأعطال الشكل."""
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
            lines.append(f"| {c['name']} | {s['correct']}/{s['n']} ({100 * s['correct'] / s['n']:.0f}%) "
                         f"| {comp} | {lat} | {s['format_failures']} | {s['truncated']} "
                         f"| {s['think_lang_followed']}/{s['n']} "
                         f"| {s['avg_input_tokens']:.0f} / {s['avg_output_tokens']:.0f} |")
    return "\n".join(lines)


def main():
    cfg = load_config()
    configs = cfg["sweep"]
    tasks = load_tasks(cfg, "calibration", n_per_type=10_000)  # كل أسئلة المعايرة
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

    OUT.mkdir(exist_ok=True)
    with (OUT / "calibration_sweep.jsonl").open("w", encoding="utf-8") as f:
        for x in rows:
            d = x["result"].to_dict()
            d.pop("content"), d.pop("reasoning")  # النص الكامل في الكاش؛ هنا الأرقام فقط
            f.write(json.dumps({"config": x["config"], "grade_reason": x["grade"].reason,
                                "correct": x["grade"].correct, **d}, ensure_ascii=False) + "\n")

    table = summarize(rows, configs)
    md = (f"# Calibration sweep (run_id {run_id})\n\n"
          f"All {len(tasks)} calibration items, temperature 0, one run. Compute is an ESTIMATE "
          f"(2 × active params × API tokens). Latency of cached calls is the original call's latency.\n"
          + markdown(table, configs))
    if stops:
        md += "\n\n**INCOMPLETE** — stopped by daily limits:\n" + "\n".join(f"- {s}" for s in stops)
    (OUT / "calibration_sweep.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
