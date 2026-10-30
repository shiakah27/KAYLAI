# -*- coding: utf-8 -*-
"""اختبارات وكيل كَيْل بنموذج مزيف وملف معايرة مزيف: لا استدعاءات API إطلاقًا."""

import pytest

from kayl import metrics
from kayl.agent import infer_type, next_config, run_agent
from kayl.client import CallResult, DailyLimitError
from kayl.config import accuracy_floor, load_config, max_output_tokens, model_info
from kayl.grading import validate
from kayl.profile import build_profile, p95

CFG = load_config()
MATH = {"id": "m1", "type": "math", "question": "كم ٢ + ٣؟", "answer": "5"}
KNOW = {"id": "k1", "type": "knowledge", "question": "ما عاصمة السعودية؟",
        "choices": ["جدة", "الرياض", "مكة", "الدمام"], "answer": "B"}


def stats(acc, tflops, p95_s=3.0, n=30):
    return {"n": n, "correct": round(acc * n), "accuracy": acc, "compute_per_success": tflops * 1e12,
            "avg_latency_s": p95_s / 2, "p95_latency_s": p95_s}


# allam: الأرخص لكن دقته تحت الحد. 20b-low: الأرخص بين ما يحقق الحد.
PROFILE = {"math": {
    "allam": stats(0.60, 1.0),
    "20b-low": stats(0.90, 2.0),
    "20b-medium": stats(0.93, 3.0),
    "120b-low": stats(0.97, 4.0),
}}


def good_reply(conf_model, effort, think_lang, final="5", answer="الجواب خمسة"):
    return CallResult(task_id="t", task_type="math", model=conf_model, effort=effort, think_lang=think_lang,
                      final=final, answer_text=answer, answer_in_arabic=True, finish_reason="stop",
                      input_tokens=100, output_tokens=50, latency_s=1.0,
                      compute_flops=metrics.call_compute_flops(model_info(CFG, conf_model), 100, 50),
                      cost_usd=metrics.call_cost_usd(model_info(CFG, conf_model), 100, 50))


class FakeModel:
    """يرجع ردًا حسب (النموذج، مستوى التفكير)؛ الافتراضي رد صحيح الشكل."""

    def __init__(self, replies=None):
        self.replies = replies or {}
        self.calls = []

    def __call__(self, cfg, task, model, effort, think_lang, format_example=None):
        self.calls.append((model, effort, think_lang))
        make = self.replies.get((model, effort))
        return make(model, effort, think_lang) if make else good_reply(model, effort, think_lang)


def no_final(model, effort, lang):
    return good_reply(model, effort, lang, final=None)


def steps(out):
    return [d["step"] for d in out["decisions"]]


# ---------- choose ----------

def test_picks_lowest_compute_that_meets_floors():
    fake = FakeModel()
    out = run_agent(MATH, CFG, PROFILE, fake)
    assert fake.calls == [("openai/gpt-oss-20b", "low", "en")]
    choose = out["decisions"][1]["reason"]
    assert "chose 20b-low" in choose and "lowest estimated compute" in choose and "2.00 TFLOPs" in choose


def test_latency_floor_excludes_slow_configs():
    profile = {"math": {**PROFILE["math"], "20b-low": stats(0.90, 2.0, p95_s=40.0)}}
    fake = FakeModel()
    run_agent(MATH, CFG, profile, fake)
    assert fake.calls[0] == ("openai/gpt-oss-20b", "medium", "en")


def test_no_config_meets_floors_picks_most_accurate():
    profile = {"math": {"allam": stats(0.5, 1.0), "20b-low": stats(0.7, 2.0), "120b-low": stats(0.75, 4.0)}}
    out = run_agent(MATH, CFG, profile, FakeModel())
    assert out["attempts"][0]["config"] == "120b-low"
    assert "no configuration met the floors" in out["decisions"][1]["reason"]


def test_allam_skipped_when_task_does_not_fit():
    profile = {"math": {"allam": stats(0.95, 1.0), "20b-low": stats(0.90, 2.0)}}
    long_task = {**MATH, "question": "كلمة " * 4000}
    out = run_agent(long_task, CFG, profile, FakeModel())
    assert out["attempts"][0]["config"] == "20b-low"
    assert "allam skipped" in out["decisions"][1]["reason"] and "does not fit" in out["decisions"][1]["reason"]


def test_allam_chosen_when_it_fits_and_is_cheapest():
    profile = {"math": {"allam": stats(0.95, 1.0), "20b-low": stats(0.90, 2.0)}}
    out = run_agent(MATH, CFG, profile, FakeModel())
    assert out["attempts"][0]["model"] == "allam-2-7b" and out["attempts"][0]["effort"] is None


def test_measurement_only_config_is_never_chosen():
    profile = {"math": {"allam-noex": stats(0.99, 0.5), "20b-low": stats(0.90, 2.0)}}
    out = run_agent(MATH, CFG, profile, FakeModel())
    assert out["attempts"][0]["config"] == "20b-low"


def test_no_profile_uses_fallback():
    fake = FakeModel()
    out = run_agent(MATH, CFG, {}, fake)
    assert fake.calls[0] == (CFG["fallback_model"], "medium", "en")
    assert "no usable calibration data" in out["decisions"][1]["reason"]


# ---------- verify + escalate ----------

def test_verified_first_try_no_escalation():
    out = run_agent(MATH, CFG, PROFILE, FakeModel())
    assert out["verified"] and out["escalations"] == 0
    assert steps(out) == ["estimate", "choose", "solve", "verify", "finish"]


def test_escalate_effort_first():
    fake = FakeModel({("openai/gpt-oss-20b", "low"): no_final})
    out = run_agent(MATH, CFG, PROFILE, fake)
    assert fake.calls == [("openai/gpt-oss-20b", "low", "en"), ("openai/gpt-oss-20b", "medium", "en")]
    assert out["verified"] and out["escalations"] == 1
    esc = next(d["reason"] for d in out["decisions"] if d["step"] == "escalate")
    assert "raise effort low → medium" in esc and "no FINAL line" in esc


def test_at_high_effort_move_to_bigger_model():
    profile = {"math": {"20b-high": stats(0.95, 2.0)}}
    fake = FakeModel({("openai/gpt-oss-20b", "high"): no_final})
    out = run_agent(MATH, CFG, profile, fake)
    assert fake.calls[1] == ("openai/gpt-oss-120b", "high", "en")
    assert "move to the bigger model 120b" in out["decisions"][4]["reason"]


def test_allam_escalates_to_20b_low():
    profile = {"math": {"allam": stats(0.95, 1.0)}}
    fake = FakeModel({("allam-2-7b", None): no_final})
    run_agent(MATH, CFG, profile, fake)
    assert fake.calls[1] == ("openai/gpt-oss-20b", "low", "en")


def test_at_most_two_escalations():
    always_bad = {(m, e): no_final for m in CFG["models"] for e in (None, "low", "medium", "high")}
    fake = FakeModel(always_bad)
    out = run_agent(MATH, CFG, PROFILE, fake)
    assert len(fake.calls) == 3 and out["escalations"] == 2
    assert not out["verified"]
    assert "no verified answer after 2 escalations" in out["decisions"][-1]["reason"]


def test_largest_model_highest_effort_cannot_escalate():
    conf, why = next_config(CFG, {"model": "openai/gpt-oss-120b", "effort": "high", "think_lang": "en"})
    assert conf is None and "largest" in why


@pytest.mark.parametrize("bad, expected", [
    (lambda m, e, l: good_reply(m, e, l, final="5 أو 6"), "more than one number"),
    (lambda m, e, l: CallResult(task_id="t", task_type="math", model=m, effort=e, think_lang=l, final="5",
                                answer_text="five", answer_in_arabic=False, finish_reason="stop"), "not in Arabic"),
    (lambda m, e, l: CallResult(task_id="t", task_type="math", model=m, effort=e, think_lang=l,
                                final=None, finish_reason="length"), "cut off"),
    (lambda m, e, l: CallResult(task_id="t", task_type="math", model=m, effort=e, think_lang=l,
                                error="APIConnectionError: connection reset"), "did not succeed"),
])
def test_verify_failure_reasons(bad, expected):
    out = run_agent(MATH, CFG, PROFILE, FakeModel({("openai/gpt-oss-20b", "low"): bad}))
    assert expected in out["attempts"][0]["verify_reason"]
    assert out["attempts"][0]["verified"] is False and out["escalations"] == 1


def test_totals_include_failed_attempts():
    out = run_agent(MATH, CFG, PROFILE, FakeModel({("openai/gpt-oss-20b", "low"): no_final}))
    assert out["total_compute_flops"] == pytest.approx(sum(a["compute_flops"] for a in out["attempts"]))
    assert len(out["attempts"]) == 2


def test_cost_is_none_if_any_attempt_used_unpriced_model():
    profile = {"math": {"allam": stats(0.95, 1.0)}}
    out = run_agent(MATH, CFG, profile, FakeModel({("allam-2-7b", None): no_final}))
    assert out["total_cost_usd"] is None


def test_every_decision_has_a_reason():
    out = run_agent(MATH, CFG, PROFILE, FakeModel({("openai/gpt-oss-20b", "low"): no_final}))
    assert all(d["reason"].strip() for d in out["decisions"])
    assert steps(out) == ["estimate", "choose", "solve", "verify", "escalate", "solve", "verify", "finish"]


# ---------- estimate / validate / profile ----------

def test_infer_type_without_type_field():
    assert infer_type({"question": "x"})[0] == "math"
    assert infer_type({"question": "x", "choices": ["جدة", "الرياض"]})[0] == "knowledge"
    assert infer_type({"question": "x", "choices": ["renew_passport", "other"]})[0] == "intent"
    assert "given by the task" in infer_type(MATH)[1]


def test_knowledge_task_end_to_end():
    profile = {"knowledge": {"20b-low": stats(0.9, 2.0)}}
    fake = FakeModel({("openai/gpt-oss-20b", "low"): lambda m, e, l: good_reply(m, e, l, final="B")})
    out = run_agent(KNOW, CFG, profile, fake)
    assert out["verified"] and out["final"] == "B"


def test_validate_without_gold():
    assert validate(MATH, "5") == (True, "ok")
    assert validate(MATH, "5 or 6")[0] is False
    assert validate(KNOW, "E")[0] is False      # 4 خيارات فقط
    assert validate({"type": "intent", "choices": ["other"]}, "Other") == (True, "ok")


def test_build_profile_and_p95():
    rows = [{"task_type": "math", "config": "x", "correct": i < 9, "compute_flops": 1e12,
             "latency_s": float(i + 1)} for i in range(10)]
    p = build_profile(rows)["math"]["x"]
    assert p["accuracy"] == 0.9 and p["compute_per_success"] == pytest.approx(10e12 / 9)
    assert p["p95_latency_s"] == 10.0 and p["avg_latency_s"] == 5.5
    assert p95([1.0, 2.0, 3.0]) == 3.0 and p95([]) is None


def test_config_with_too_few_items_is_ignored():
    profile = {"math": {"120b-high": stats(1.0, 0.5, n=2), "20b-low": stats(0.90, 2.0)}}
    out = run_agent(MATH, CFG, profile, FakeModel())
    assert out["attempts"][0]["config"] == "20b-low"
    assert "measured on only 2 items" in out["decisions"][1]["reason"]


# ---------- floors from baseline A ----------

def test_floor_is_baseline_a_minus_one_question():
    profile = {"math": {"120b-medium": stats(28 / 30, 5.0)}}
    floor, why = accuracy_floor(CFG, profile, "math")
    assert floor == pytest.approx(27 / 30) and "28/30 minus 1 question" in why


def test_config_one_question_below_baseline_counts():
    # baseline 28/30 -> floor 27/30: 20b-low (27/30) يحقق الحد، allam (26/30) لا
    profile = {"math": {"120b-medium": stats(28 / 30, 5.0), "20b-low": stats(27 / 30, 2.0),
                        "allam": stats(26 / 30, 1.0)}}
    out = run_agent(MATH, CFG, profile, FakeModel())
    assert out["attempts"][0]["config"] == "20b-low"


def test_fixed_floor_when_no_baseline():
    floor, why = accuracy_floor(CFG, {"math": {}}, "math")
    assert floor == CFG["floors"]["accuracy"] and "fixed" in why


def test_high_effort_cap_only_for_120b():
    assert max_output_tokens(CFG, "openai/gpt-oss-120b", "high") == 8192
    assert max_output_tokens(CFG, "openai/gpt-oss-120b", "medium") == 2048
    assert max_output_tokens(CFG, "openai/gpt-oss-20b", "high") == 2048


# ---------- fallback on rate / daily limits ----------

def daily_limited(model, effort, lang):
    raise DailyLimitError("tokens per day (TPD): Limit 200000")


def rate_limited(model, effort, lang):
    return CallResult(task_id="t", task_type="math", model=model, effort=effort, think_lang=lang,
                      error="RateLimitError: limit")


@pytest.mark.parametrize("limited, words", [(daily_limited, "daily token limit"),
                                            (rate_limited, "rate limit even after retries")])
def test_fallback_to_next_best_config_on_other_model(limited, words):
    fake = FakeModel({("openai/gpt-oss-20b", "low"): limited})
    out = run_agent(MATH, CFG, PROFILE, fake)
    # 20b غير متاح -> أفضل إعداد على نموذج آخر يحقق الحد: 120b-low
    assert fake.calls == [("openai/gpt-oss-20b", "low", "en"), ("openai/gpt-oss-120b", "low", "en")]
    assert out["verified"] and out["escalations"] == 0  # الانتقال ليس تصعيدًا
    fb = out["fallbacks"][0]
    assert fb["from"] == "20b-low" and fb["to"] == "120b-low" and words in fb["reason"]
    assert steps(out) == ["estimate", "choose", "fallback", "solve", "verify", "finish"]


def test_escalation_skips_unavailable_model():
    # 20b وصل حده، ثم 120b-low فشل في التحقق -> رفع التفكير على 120b (وليس العودة لـ 20b)
    fake = FakeModel({("openai/gpt-oss-20b", "low"): daily_limited, ("openai/gpt-oss-120b", "low"): no_final})
    out = run_agent(MATH, CFG, PROFILE, fake)
    assert fake.calls[-1] == ("openai/gpt-oss-120b", "medium", "en")
    assert all(m != "openai/gpt-oss-20b" for m, _, _ in fake.calls[1:])


def test_all_models_limited():
    every = {(m, e): daily_limited for m in CFG["models"] for e in (None, "low", "medium", "high")}
    out = run_agent(MATH, CFG, PROFILE, FakeModel(every))
    assert not out["verified"]
    assert any("no other model is available" in d["reason"] for d in out["decisions"])
