"""Robustness drill with real calls: what does the agent do when a model hits its rate or daily limit?

- Takes 5 Arabic-GSM8K problems from the same seed-42 shuffle, positions 60-64, which are in
  neither calibration nor test (kayl/data.py: drill_tasks).
- Runs the KAYL agent on them with real calls. If Groq refuses a model because of a limit,
  the agent must fall back to another model and record why.
- Writes results/robustness_fallback_drill.json and logs every attempt to SQLite.

The committed drill was run with gpt-oss on Groq directly. To replay it with no API key:
    python -m scripts.run_fallback_drill --from-cache --route groq
"""

import argparse
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kayl import runner
from kayl.agent import run_agent
from kayl.config import load_config
from kayl.data import DRILL_SLICE, drill_tasks
from kayl.grading import grade
from kayl.logdb import connect, log_call
from kayl.profile import load_profile

OUT = Path("results/robustness_fallback_drill.json")
RIYADH = timezone(timedelta(hours=3))


def main():
    p = argparse.ArgumentParser()
    runner.add_run_args(p)
    cfg = runner.apply_run_args(load_config(), p.parse_args())
    profile, tasks = load_profile(), drill_tasks()
    used = {json.loads(l)["id"] for f in ("calibration.jsonl", "test.jsonl")
            for l in Path(cfg["paths"]["data"], f).open(encoding="utf-8")}
    assert not used & {t["id"] for t in tasks}, "drill items must not be in calibration or test"

    conn, run_id = connect(cfg["paths"]["db"]), uuid.uuid4().hex[:8]
    started = datetime.now(RIYADH)
    records, errors = [], []
    for task in tasks:
        out = run_agent(task, cfg, profile)
        g = grade(task, out["final"])
        results = out.pop("_results")
        errors += runner.failed(results)
        for i, r in enumerate(results):
            log_call(conn, r, run_id=run_id, config_name="kayl-agent:fallback-drill",
                     correct=g.correct if i == len(out["attempts"]) - 1 else False, escalated=i > 0)
        records.append({**out, "gold": task["answer"], "correct": g.correct})
        print(f"{task['id']}: final={out['final']} gold={task['answer']} correct={g.correct} "
              f"fallbacks={len(out['fallbacks'])}")
        for d in out["decisions"]:
            print(f"   [{d['step']}] {d['reason']}")
    runner.check_no_failures(errors)

    summary = {
        "run_id": run_id, "started_riyadh": started.isoformat(timespec="seconds"),
        "items": len(records), "pool_positions": list(range(*DRILL_SLICE)),
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
    runner.run(main)
