"""SIMULATED outage: what does the agent do when gpt-oss-20b is unavailable?

- Uses the same 5 drill problems as run_fallback_drill.py (positions 60-64, in neither
  calibration nor test).
- Lists gpt-oss-20b in simulated_outage.unavailable_models, in memory only (config.yaml is
  not changed).
- The agent picks 20b as usual (from calibration), the client refuses the call, and the agent
  falls back to the best configuration on another model and records why.
- Writes results/robustness_simulated_outage.json and logs to SQLite as
  "kayl-agent:SIMULATED-outage".

This is a simulated outage, not a real one from Groq.

The committed run used the routing in config.yaml (OpenRouter pinned to Groq). To replay it
with no API key:
    python -m scripts.run_simulated_outage --from-cache
"""

import argparse
import json
import uuid
from pathlib import Path

from kayl import runner
from kayl.agent import run_agent
from kayl.config import load_config
from kayl.data import DRILL_SLICE, drill_tasks
from kayl.grading import grade
from kayl.logdb import connect, log_call
from kayl.profile import load_profile

OUTAGE_MODEL = "openai/gpt-oss-20b"
OUT = Path("results/robustness_simulated_outage.json")


def main():
    p = argparse.ArgumentParser()
    runner.add_run_args(p)
    cfg = runner.apply_run_args(load_config(), p.parse_args())
    cfg["simulated_outage"]["unavailable_models"] = [OUTAGE_MODEL]
    profile, tasks = load_profile(), drill_tasks()
    conn, run_id = connect(cfg["paths"]["db"]), uuid.uuid4().hex[:8]
    print(f"SIMULATED outage: {OUTAGE_MODEL} marked unavailable; {len(tasks)} tasks (pool positions "
          f"{DRILL_SLICE[0]}–{DRILL_SLICE[1] - 1}, not in calibration or test)\n")

    records, errors = [], []
    for task in tasks:
        out = run_agent(task, cfg, profile)
        g = grade(task, out["final"])
        results = out.pop("_results")
        errors += runner.failed(results)
        for i, r in enumerate(results):
            log_call(conn, r, run_id=run_id, config_name="kayl-agent:SIMULATED-outage",
                     correct=g.correct if i == len(out["attempts"]) - 1 else False, escalated=i > 0)
        records.append({**out, "gold": task["answer"], "correct": g.correct})
        print(f"{task['id']}: final={out['final']} gold={task['answer']} correct={g.correct}")
        for d in out["decisions"]:
            print(f"   [{d['step']}] {d['reason']}")
    runner.check_no_failures(errors)

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
    runner.run(main)
