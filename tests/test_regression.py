"""Regression tests on a fixed small set, replayed from the committed cache (no API calls).

The KAYL agent must reproduce the recorded final-test answers and compute exactly, and the
run scripts' safety checks must refuse incomplete runs."""

import copy
import json
from pathlib import Path

import pytest

from kayl import runner
from kayl.agent import run_agent
from kayl.client import CacheMissError, CallResult, call_model
from kayl.config import load_config
from kayl.data import load_tasks
from kayl.profile import load_profile

CFG = copy.deepcopy(load_config())
CFG["cache_only"] = True
N_PER_TYPE = 3


def recorded_kayl() -> dict:
    with Path("results/final_test.jsonl").open(encoding="utf-8") as f:
        return {r["task_id"]: r for r in map(json.loads, f) if r["experiment"] == "D KAYL"}


@pytest.mark.parametrize("task", load_tasks(CFG, "test", N_PER_TYPE), ids=lambda t: t["id"])
def test_kayl_replays_recorded_final_test(task):
    want = recorded_kayl()[task["id"]]
    out = run_agent(task, CFG, load_profile())
    assert out["final"] == want["final"]
    assert [a["config"] for a in out["attempts"]] == want["configs"]
    assert out["total_compute_flops"] == want["compute_flops"]
    assert all(a["cached"] for a in out["attempts"])


def test_cache_only_raises_on_miss():
    task = {"id": "never-recorded", "type": "math", "question": "كم ١٢٣٤٥ + ٦٧٨٩٠؟ (not in the cache)",
            "answer": "80235"}
    with pytest.raises(CacheMissError):
        call_model(CFG, task, "openai/gpt-oss-20b", "low", "en")


def test_failed_calls_block_writing_results():
    ok = CallResult(task_id="a", task_type="math", model="m", effort=None, think_lang="en")
    skipped = CallResult(task_id="b", task_type="math", model="m", effort=None, think_lang="en",
                         skipped=True, error="context_too_long")
    bad = CallResult(task_id="c", task_type="math", model="m", effort=None, think_lang="en",
                     error="AuthenticationError: no key")
    assert runner.failed([ok, skipped]) == []
    runner.check_no_failures([])
    with pytest.raises(SystemExit):
        runner.check_no_failures(runner.failed([ok, bad]))
    with pytest.raises(SystemExit):
        runner.check_no_failures([], stopped=["daily limit"])
