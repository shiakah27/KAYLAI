# -*- coding: utf-8 -*-
"""
كَيْل — التقييم النهائي على مجموعة الاختبار المحجوبة (تشغيل واحد، temperature 0).

التجارب (بالترتيب):
  B  أقل توكنز: allam-2-7b
  D  وكيل كَيْل كاملًا (اختيار + تحقق + تصعيد + انتقال عند الحد)
  A  النموذج الأكبر: gpt-oss-120b بالتفكير الافتراضي (medium) = baseline A
  C  أرخص نموذج بسعر مؤكد: gpt-oss-20b medium
  E  مؤشر صعوبة بسيط (طول السؤال): الطويل -> 120b-medium، القصير -> 20b-medium
  ablations:
     D-verify  : بدون تحقق/تصعيد (المحاولة الأولى فقط)
     D-model   : بدون مفتاح النموذج (120b فقط)
     D-effort  : بدون مفتاح التفكير (medium فقط، بلا رفع للتفكير)
     D-thinklang: مطابق لـ D لأن التفكير بالعربية أُوقف (لا تشغيل)

كل شيء من الكاش إن وُجد؛ إذا توقف التشغيل أعد نفس الأمر.

التشغيل:
    python run_final.py --final-eval
"""

import argparse
import copy
import json
import statistics
import uuid
from pathlib import Path

import litellm

from kayl import metrics
from kayl.agent import run_agent
from kayl.client import DailyLimitError, call_model
from kayl.config import accuracy_floor, load_config
from kayl.grading import grade
from kayl.logdb import connect, log_call
from kayl.profile import load_profile

OUT = Path("results")
TYPES = ["math", "knowledge", "intent"]
SWEEP = {}


def load(cfg: dict, name: str) -> list[dict]:
    return [json.loads(l) for l in Path(cfg["paths"]["data"], name).open(encoding="utf-8")]


def conf(name: str) -> dict:
    c = SWEEP[name]
    return {"model": c["model"], "effort": c["effort"], "think_lang": c["think_lang"],
            "format_example": c.get("format_example")}


def question_tokens(task: dict) -> int:
    """مؤشر الصعوبة لـ E: عدد توكنز السؤال (مع الخيارات)، بنفس العدّاد لكل المهام."""
    text = task["question"] + " " + " ".join(map(str, task.get("choices", [])))
    return litellm.token_counter(model="groq/openai/gpt-oss-20b", text=text)


def fixed_record(exp: str, task: dict, name: str, r) -> dict:
    g = grade(task, r.final)
    return {"experiment": exp, "task_id": task["id"], "task_type": task["type"], "correct": g.correct,
            "grade_reason": g.reason, "configs": [name], "final": r.final,
            "compute_flops": r.compute_flops, "cost_usd": r.cost_usd, "latency_s": r.latency_s or 0.0,
            "escalations": 0, "fallbacks": [], "providers": [r.provider], "errors": [r.error] if r.error else [],
            "cached": [r.cached], "_results": [r]}


def agent_record(exp: str, task: dict, out: dict) -> dict:
    g = grade(task, out["final"])
    results = out.pop("_results")
    return {"experiment": exp, "task_id": task["id"], "task_type": task["type"], "correct": g.correct,
            "grade_reason": g.reason, "configs": [a["config"] for a in out["attempts"]], "final": out["final"],
            "verified": out["verified"], "compute_flops": out["total_compute_flops"],
            "cost_usd": out["total_cost_usd"], "latency_s": out["total_latency_s"],
            "escalations": out["escalations"], "fallbacks": out["fallbacks"],
            "providers": [r.provider for r in results], "errors": [r.error for r in results if r.error],
            "cached": [r.cached for r in results], "decisions": out["decisions"], "_results": results}


def experiments(cfg: dict, profile: dict, threshold: float):
    """(اسم التجربة، دالة تحل مهمة وترجع سجلًا)."""
    def fixed(name):
        return lambda exp, task: fixed_record(exp, task, name, call_model(cfg, task, **conf(name)))

    def router_e(exp, task):
        name = "120b-medium" if question_tokens(task) > threshold else "20b-medium"
        return fixed_record(exp, task, name, call_model(cfg, task, **conf(name)))

    def agent(variant_cfg):
        return lambda exp, task: agent_record(exp, task, run_agent(task, variant_cfg, profile))

    no_verify = copy.deepcopy(cfg); no_verify["agent"]["max_escalations"] = 0
    no_model = copy.deepcopy(cfg)
    no_model["sweep"] = [c for c in cfg["sweep"] if c["model"] == "openai/gpt-oss-120b"]
    no_model["agent"]["escalate_model"] = False
    no_effort = copy.deepcopy(cfg)
    no_effort["sweep"] = [c for c in cfg["sweep"] if c["effort"] in ("medium", None)]
    no_effort["agent"]["escalate_effort"] = False

    return [
        ("B allam (fewest tokens)", fixed("allam")),
        ("D KAYL", agent(cfg)),
        ("A 120b-medium (largest)", fixed("120b-medium")),
        ("C 20b-medium (cheapest price)", fixed("20b-medium")),
        ("E question-length proxy", router_e),
        ("D minus verify/escalate", agent(no_verify)),
        ("D minus model key (120b only)", agent(no_model)),
        ("D minus effort key (medium only)", agent(no_effort)),
    ]


def summarize(records: list[dict], cfg: dict, profile: dict) -> list[dict]:
    rows = []
    for t in TYPES + ["all"]:
        rs = [r for r in records if t == "all" or r["task_type"] == t]
        if not rs:
            continue
        n_ok = sum(r["correct"] for r in rs)
        lat = [r["latency_s"] for r in rs]
        p95 = sorted(lat)[max(0, -(-95 * len(lat) // 100) - 1)]
        row = {"type": t, "n": len(rs), "correct": n_ok,
               "compute_per_success": metrics.per_success([r["compute_flops"] for r in rs], n_ok),
               "cost_per_success": metrics.per_success([r["cost_usd"] for r in rs], n_ok),
               "avg_latency_s": statistics.mean(lat), "p95_latency_s": p95,
               "escalations": sum(r["escalations"] for r in rs),
               "fallbacks": sum(len(r["fallbacks"]) for r in rs),
               "errors": sum(len(r["errors"]) for r in rs)}
        if t != "all":
            floor, _ = accuracy_floor(cfg, profile, t)
            row["meets_floors"] = (n_ok / len(rs) >= floor - 1e-9) and p95 <= cfg["floors"]["latency_p95_s"]
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--final-eval", action="store_true")
    p.add_argument("--only", nargs="*", help="run only experiments whose name starts with these letters")
    args = p.parse_args()
    if not args.final_eval:
        raise SystemExit("هذا هو التقييم النهائي على مجموعة الاختبار. أضف --final-eval إذا كان هذا هو المقصود.")

    cfg, profile = load_config(), load_profile()
    SWEEP.update({c["name"]: c for c in cfg["sweep"]})
    tasks = load(cfg, "test.jsonl")
    threshold = statistics.median(question_tokens(t) for t in load(cfg, "calibration.jsonl"))
    conn, run_id = connect(cfg["paths"]["db"]), uuid.uuid4().hex[:8]
    print(f"final test: {len(tasks)} held-out items, run_id={run_id}, "
          f"E threshold={threshold:.0f} question tokens (calibration median)", flush=True)

    all_records, stopped = [], {}
    for exp, solve in experiments(cfg, profile, threshold):
        if args.only and not any(exp.startswith(o) for o in args.only):
            continue
        records = []
        for i, task in enumerate(tasks, 1):
            try:
                rec = solve(exp, task)
            except DailyLimitError as e:
                stopped[exp] = f"daily limit after {i - 1}/{len(tasks)}: {str(e)[:150]}"
                break
            for j, r in enumerate(rec.pop("_results")):
                log_call(conn, r, run_id=run_id, config_name=f"final:{exp}",
                         correct=rec["correct"] if j == len(rec["configs"]) - 1 else False, escalated=j > 0)
            records.append(rec)
            if i % 10 == 0 or i == len(tasks):
                ok = sum(r["correct"] for r in records)
                new = sum(not c for r in records for c in r["cached"])
                print(f"  [{exp}] {i}/{len(tasks)} done, {ok} correct, {new} new API calls", flush=True)
        all_records += records
        if exp in stopped:
            print(f"  [{exp}] STOPPED: {stopped[exp]}", flush=True)

    OUT.mkdir(exist_ok=True)
    with (OUT / "final_test.jsonl").open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    summary = {exp: summarize([r for r in all_records if r["experiment"] == exp], cfg, profile)
               for exp in dict.fromkeys(r["experiment"] for r in all_records)}
    (OUT / "final_test_summary.json").write_text(
        json.dumps({"run_id": run_id, "threshold_E": threshold, "stopped": stopped, "summary": summary},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsaved results/final_test.jsonl and results/final_test_summary.json")
    for exp, rows in summary.items():
        a = rows[-1]
        comp = "n/a" if a["compute_per_success"] is None else f"{a['compute_per_success'] / 1e12:.2f}"
        print(f"{exp:36s} {a['correct']}/{a['n']}  compute/success {comp} TFLOPs  "
              f"avg {a['avg_latency_s']:.2f}s  esc {a['escalations']}  fb {a['fallbacks']}  err {a['errors']}")


if __name__ == "__main__":
    main()
