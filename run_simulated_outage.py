# -*- coding: utf-8 -*-
"""
كَيْل — انقطاع مُحاكى (SIMULATED): ماذا يفعل الوكيل إذا كان gpt-oss-20b غير متاح؟

ماذا يفعل:
  - يأخذ نفس المسائل الخمس من run_fallback_drill.py (مواقع 60–64، ليست في المعايرة ولا الاختبار).
  - يضع gpt-oss-20b في simulated_outage.unavailable_models (في الذاكرة فقط؛ config.yaml لا يتغير).
  - يشغّل الوكيل: يختار 20b كالعادة (حسب المعايرة)، فيرفض العميل الاستدعاء،
    فينتقل الوكيل تلقائيًا لأفضل إعداد على نموذج آخر ويسجل السبب.
  - يحفظ النتيجة في results/robustness_simulated_outage.json، ويسجلها في SQLite باسم
    "kayl-agent:SIMULATED-outage".

هذا انقطاع مُحاكى، وليس انقطاعًا حقيقيًا من Groq.

التشغيل:
    python run_simulated_outage.py
"""

import copy
import json
import uuid
from pathlib import Path

from kayl.agent import run_agent
from kayl.config import load_config
from kayl.grading import grade
from kayl.logdb import connect, log_call
from kayl.profile import load_profile
from run_fallback_drill import POOL_SLICE, drill_tasks

OUTAGE_MODEL = "openai/gpt-oss-20b"
OUT = Path("results/robustness_simulated_outage.json")


def main():
    cfg = copy.deepcopy(load_config())
    cfg["simulated_outage"]["unavailable_models"] = [OUTAGE_MODEL]
    profile, tasks = load_profile(), drill_tasks()
    conn, run_id = connect(cfg["paths"]["db"]), uuid.uuid4().hex[:8]
    print(f"SIMULATED outage: {OUTAGE_MODEL} marked unavailable; {len(tasks)} tasks (pool positions "
          f"{POOL_SLICE[0]}–{POOL_SLICE[1] - 1}, not in calibration or test)\n")

    records = []
    for task in tasks:
        out = run_agent(task, cfg, profile)
        g = grade(task, out["final"])
        for i, r in enumerate(out.pop("_results")):
            log_call(conn, r, run_id=run_id, config_name="kayl-agent:SIMULATED-outage",
                     correct=g.correct if i == len(out["attempts"]) - 1 else False, escalated=i > 0)
        records.append({**out, "gold": task["answer"], "correct": g.correct})
        print(f"{task['id']}: final={out['final']} gold={task['answer']} correct={g.correct}")
        for d in out["decisions"]:
            print(f"   [{d['step']}] {d['reason']}")

    summary = {"label": "SIMULATED outage (config flag), not a real provider outage",
               "unavailable_models": [OUTAGE_MODEL], "run_id": run_id, "items": len(records),
               "correct": sum(r["correct"] for r in records),
               "tasks_with_fallback": sum(bool(r["fallbacks"]) for r in records),
               "fallback_targets": sorted({f["to"] for r in records for f in r["fallbacks"]}),
               "total_compute_flops": sum(r["total_compute_flops"] for r in records),
               "records": records}
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nSIMULATED: {summary['correct']}/{len(records)} correct, fallback in "
          f"{summary['tasks_with_fallback']}/{len(records)} tasks -> {summary['fallback_targets']}  ({OUT})")


if __name__ == "__main__":
    main()
