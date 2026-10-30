# -*- coding: utf-8 -*-
"""
كَيْل — تشغيل تقييم بسيط: نفس الإعدادات الثابتة (نموذج، مستوى تفكير، لغة تفكير) على عدد من المهام.

ماذا يفعل:
  يأخذ أول N مهمة من كل نوع (بترتيب الملف، فالنتيجة ثابتة)، ويستدعي كل نموذج،
  ويصحح الجواب، ويسجل كل استدعاء في SQLite، ويطبع جدول النتائج.

مجموعة الاختبار محمية: لا تُستخدم إلا مع --final-eval (للتقييم النهائي فقط).

التشغيل (مثال):
    python run_eval.py --models allam-2-7b openai/gpt-oss-20b --effort low --n-per-type 2
"""

import argparse
import json
import uuid
from collections import defaultdict
from pathlib import Path

from kayl import metrics
from kayl.client import call_model
from kayl.config import effort_options, load_config
from kayl.grading import grade
from kayl.logdb import connect, log_call

SPLITS = {"calibration": "calibration.jsonl", "test": "test.jsonl", "robustness_en": "robustness_en.jsonl"}


def load_tasks(cfg: dict, split: str, n_per_type: int) -> list[dict]:
    """أول N مهمة من كل نوع، بترتيب الملف."""
    path = Path(cfg["paths"]["data"]) / SPLITS[split]
    by_type = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        task = json.loads(line)
        if len(by_type[task["type"]]) < n_per_type:
            by_type[task["type"]].append(task)
    return [t for tasks in by_type.values() for t in tasks]


def fmt_flops(x: float | None) -> str:
    return "n/a" if x is None else f"{x / 1e12:.2f} TFLOPs"


def fmt_usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.6f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="calibration", choices=SPLITS)
    p.add_argument("--final-eval", action="store_true", help="مطلوب لاستخدام مجموعة الاختبار")
    p.add_argument("--models", nargs="+", default=["allam-2-7b", "openai/gpt-oss-20b"])
    p.add_argument("--effort", default="low", help="لنماذج gpt-oss فقط")
    p.add_argument("--think-lang", default="en", choices=["en", "ar"])
    p.add_argument("--n-per-type", type=int, default=2)
    p.add_argument("--config-name", default="fixed")
    args = p.parse_args()

    if args.split == "test" and not args.final_eval:
        raise SystemExit("مجموعة الاختبار للتقييم النهائي فقط. أضف --final-eval إذا كان هذا هو المقصود.")

    cfg = load_config()
    tasks = load_tasks(cfg, args.split, args.n_per_type)
    conn = connect(cfg["paths"]["db"])
    run_id = uuid.uuid4().hex[:8]
    rows = []

    for model in args.models:
        effort = args.effort if effort_options(cfg, model) != [None] else None
        for task in tasks:
            r = call_model(cfg, task, model, effort, args.think_lang)
            g = grade(task, r.final)
            log_call(conn, r, run_id=run_id, config_name=args.config_name, correct=g.correct)
            rows.append((r, g, task))

    print(f"\nrun_id={run_id}  split={args.split}  think_lang={args.think_lang}  n={len(rows)} calls\n")
    print("| task | type | model | effort | gold | FINAL | correct | reason | in | out | latency s "
          "| compute | cost | reasoning lang | followed | ANSWER ar |")
    print("|" + "---|" * 16)
    for r, g, task in rows:
        final = (r.final or "—").replace("|", "/")[:40]
        print(f"| {r.task_id} | {r.task_type} | {r.model} | {r.effort or '—'} | {task['answer']} | {final} "
              f"| {'✅' if g.correct else '❌'} | {g.reason} | {r.input_tokens} | {r.output_tokens} "
              f"| {r.latency_s:.2f} | {fmt_flops(r.compute_flops)} | {fmt_usd(r.cost_usd)} "
              f"| {r.reasoning_lang} | {'yes' if r.think_lang_followed else 'no'} "
              f"| {'yes' if r.answer_in_arabic else 'no'} |" if r.latency_s is not None else
              f"| {r.task_id} | {r.task_type} | {r.model} | — | {task['answer']} | — | ❌ | {r.error} |")

    print("\n| model | correct | compute / success | cost / success | think-lang followed | ANSWER in Arabic |")
    print("|---|---|---|---|---|---|")
    for model in args.models:
        mine = [(r, g) for r, g, _ in rows if r.model == model]
        n_ok = sum(g.correct for _, g in mine)
        compute = metrics.per_success([r.compute_flops for r, _ in mine], n_ok)
        cost = metrics.per_success([r.cost_usd for r, _ in mine], n_ok)
        followed = sum(r.think_lang_followed for r, _ in mine)
        arabic = sum(r.answer_in_arabic for r, _ in mine)
        print(f"| {model} | {n_ok}/{len(mine)} | {fmt_flops(compute)} | {fmt_usd(cost)} "
              f"| {followed}/{len(mine)} | {arabic}/{len(mine)} |")


if __name__ == "__main__":
    main()
