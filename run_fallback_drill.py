# -*- coding: utf-8 -*-
"""
كَيْل — حالة متانة حقيقية: ماذا يفعل الوكيل عندما يصل نموذج إلى حده (في الدقيقة أو اليومي)؟

ماذا يفعل:
  - يأخذ 5 مسائل Arabic-GSM8K من نفس الخلط (بذرة 42) لكن من المواقع 60–64،
    أي مسائل ليست في المعايرة ولا في الاختبار (لم يرها كَيْل، ولا تمس مجموعة الاختبار).
  - يشغّل وكيل كَيْل عليها باستدعاءات حقيقية. إذا رفض Groq أحد النماذج بسبب الحد،
    يجب أن ينتقل الوكيل تلقائيًا لنموذج آخر ويسجل السبب.
  - يحفظ النتيجة في results/robustness_fallback_drill.json ويسجل كل محاولة في SQLite.

التشغيل:
    python run_fallback_drill.py
"""

import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kayl.agent import run_agent
from kayl.config import load_config
from kayl.grading import grade
from kayl.logdb import connect, log_call
from kayl.profile import load_profile
from prepare_data import GSM8K_AR, SEED, final_answer

POOL_SLICE = (60, 65)  # المعايرة = 0–29، الاختبار = 30–59
OUT = Path("results/robustness_fallback_drill.json")
RIYADH = timezone(timedelta(hours=3))


def drill_tasks() -> list[dict]:
    """نفس خطوات build_math في prepare_data.py، لكن نأخذ المواقع 60–64."""
    from datasets import load_dataset
    ds = load_dataset(GSM8K_AR, split="main_test")
    items = [{"id": f"gsm8k-ar-{i}", "type": "math", "question": row["question"],
              "answer": final_answer(row["answer"])}
             for i, row in enumerate(ds) if final_answer(row["answer"]) is not None]
    random.Random(SEED).shuffle(items)
    return items[POOL_SLICE[0]:POOL_SLICE[1]]


def main():
    cfg, profile = load_config(), load_profile()
    tasks = drill_tasks()
    used = {json.loads(l)["id"] for f in ("calibration.jsonl", "test.jsonl")
            for l in Path(cfg["paths"]["data"], f).open(encoding="utf-8")}
    assert not used & {t["id"] for t in tasks}, "drill items must not be in calibration or test"

    conn, run_id = connect(cfg["paths"]["db"]), uuid.uuid4().hex[:8]
    started = datetime.now(RIYADH)
    records = []
    for task in tasks:
        out = run_agent(task, cfg, profile)
        g = grade(task, out["final"])
        for i, r in enumerate(out.pop("_results")):
            log_call(conn, r, run_id=run_id, config_name="kayl-agent:fallback-drill",
                     correct=g.correct if i == len(out["attempts"]) - 1 else False, escalated=i > 0)
        records.append({**out, "gold": task["answer"], "correct": g.correct})
        print(f"{task['id']}: final={out['final']} gold={task['answer']} correct={g.correct} "
              f"fallbacks={len(out['fallbacks'])}")
        for d in out["decisions"]:
            print(f"   [{d['step']}] {d['reason']}")

    summary = {
        "run_id": run_id, "started_riyadh": started.isoformat(timespec="seconds"),
        "items": len(records), "pool_positions": list(range(*POOL_SLICE)),
        "correct": sum(r["correct"] for r in records),
        "tasks_with_fallback": sum(bool(r["fallbacks"]) for r in records),
        "verified": sum(r["verified"] for r in records),
        "total_compute_flops": sum(r["total_compute_flops"] for r in records),
        "records": records,
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n{summary['correct']}/{len(records)} correct, fallback used in "
          f"{summary['tasks_with_fallback']}/{len(records)} tasks -> {OUT}")


if __name__ == "__main__":
    main()
