# -*- coding: utf-8 -*-
"""اختبارات العميل بنموذج مزيف (بدون استدعاءات API): الكاش، إعادة المحاولة، تجاوز السياق."""

import litellm
import pytest

import kayl.client as client
from kayl.cache import DiskCache
from kayl.config import load_config

CFG = load_config()
TASK = {"id": "t1", "type": "math", "question": "كم ٢ + ٣؟", "answer": "5"}


class FakeResponse:
    def __init__(self, content, reasoning=None):
        self._d = {
            "choices": [{"finish_reason": "stop",
                         "message": {"content": content, "reasoning": reasoning}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40,
                      "completion_tokens_details": {"reasoning_tokens": 25}},
        }

    def model_dump(self):
        return self._d


@pytest.fixture
def fake_api(monkeypatch):
    """يستبدل litellm.completion ويعدّ الاستدعاءات."""
    calls = []

    def fake_completion(model, messages, **params):
        calls.append({"model": model, "params": params})
        return FakeResponse("ANSWER: الجواب خمسة\nFINAL: 5", reasoning="2 + 3 = 5, easy.")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(client.time, "sleep", lambda s: None)
    return calls


def test_call_parses_and_measures(fake_api, tmp_path):
    r = client.call_model(CFG, TASK, "openai/gpt-oss-20b", "low", "en", cache=DiskCache(tmp_path))
    assert r.final == "5" and r.answer_text == "الجواب خمسة"
    assert r.input_tokens == 100 and r.output_tokens == 40 and r.reasoning_tokens == 25
    assert r.reasoning_lang == "en" and r.think_lang_followed and r.answer_in_arabic
    assert r.compute_flops == pytest.approx(2 * 3.6e9 * 140)
    assert r.cost_usd == pytest.approx((100 * 0.075 + 40 * 0.30) / 1e6)
    assert fake_api[0]["params"]["reasoning_effort"] == "low"


def test_think_lang_not_followed(fake_api, tmp_path):
    r = client.call_model(CFG, TASK, "openai/gpt-oss-20b", "low", "ar", cache=DiskCache(tmp_path))
    assert r.reasoning_lang == "en" and not r.think_lang_followed


def test_second_call_comes_from_cache(fake_api, tmp_path):
    cache = DiskCache(tmp_path)
    first = client.call_model(CFG, TASK, "openai/gpt-oss-20b", "low", "en", cache=cache)
    second = client.call_model(CFG, TASK, "openai/gpt-oss-20b", "low", "en", cache=cache)
    assert len(fake_api) == 1
    assert not first.cached and second.cached
    assert second.final == first.final and second.latency_s == first.latency_s


def test_different_effort_is_a_different_cache_entry(fake_api, tmp_path):
    cache = DiskCache(tmp_path)
    client.call_model(CFG, TASK, "openai/gpt-oss-20b", "low", "en", cache=cache)
    client.call_model(CFG, TASK, "openai/gpt-oss-20b", "high", "en", cache=cache)
    assert len(fake_api) == 2


def test_allam_gets_no_reasoning_effort(fake_api, tmp_path):
    client.call_model(CFG, TASK, "allam-2-7b", None, "en", cache=DiskCache(tmp_path))
    assert "reasoning_effort" not in fake_api[0]["params"]
    assert fake_api[0]["params"]["max_tokens"] == 1024


def test_effort_on_allam_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        client.call_model(CFG, TASK, "allam-2-7b", "low", "en", cache=DiskCache(tmp_path))


def test_task_too_long_for_allam_is_skipped(fake_api, tmp_path):
    long_task = {**TASK, "question": "كلمة " * 5000}
    r = client.call_model(CFG, long_task, "allam-2-7b", None, "en", cache=DiskCache(tmp_path))
    assert r.skipped and r.error == "context_too_long"
    assert fake_api == []  # لم نستدعِ النموذج أصلًا


def test_retry_on_rate_limit(monkeypatch, tmp_path):
    attempts = []

    def flaky(model, messages, **params):
        attempts.append(1)
        if len(attempts) < 3:
            raise litellm.RateLimitError("Please try again in 1.5s", llm_provider="groq", model=model)
        return FakeResponse("ANSWER: خمسة\nFINAL: 5")

    waits = []
    monkeypatch.setattr(litellm, "completion", flaky)
    monkeypatch.setattr(client.time, "sleep", waits.append)
    r = client.call_model(CFG, TASK, "openai/gpt-oss-20b", "low", "en", cache=DiskCache(tmp_path))
    assert r.final == "5" and len(attempts) == 3
    assert waits == [2.0, 2.0]  # 1.5s كما قال Groq + 0.5 هامش


def test_error_after_all_retries_is_recorded(monkeypatch, tmp_path):
    def always_limited(model, messages, **params):
        raise litellm.RateLimitError("limit", llm_provider="groq", model=model)

    monkeypatch.setattr(litellm, "completion", always_limited)
    monkeypatch.setattr(client.time, "sleep", lambda s: None)
    r = client.call_model(CFG, TASK, "openai/gpt-oss-20b", "low", "en", cache=DiskCache(tmp_path))
    assert r.error.startswith("RateLimitError") and r.final is None


def test_wait_parses_minutes_and_ms():
    cfg = {"base_wait_s": 2, "max_wait_s": 600}
    assert client._wait_seconds(Exception("Please try again in 1m26.4s"), 0, cfg) == pytest.approx(86.9)
    assert client._wait_seconds(Exception("Please try again in 547ms"), 0, cfg) == pytest.approx(1.047)
    assert client._wait_seconds(Exception("no hint"), 2, cfg) == 8


def test_daily_limit_stops_instead_of_waiting(monkeypatch, tmp_path):
    def daily(model, messages, **params):
        raise litellm.RateLimitError("Rate limit reached on tokens per day (TPD): Limit 200000",
                                     llm_provider="groq", model=model)

    monkeypatch.setattr(litellm, "completion", daily)
    with pytest.raises(client.DailyLimitError):
        client.call_model(CFG, TASK, "openai/gpt-oss-20b", "low", "en", cache=DiskCache(tmp_path))


def test_gpt_oss_routed_via_openrouter_pinned_to_groq(fake_api, tmp_path):
    r = client.call_model(CFG, TASK, "openai/gpt-oss-20b", "low", "en", cache=DiskCache(tmp_path))
    assert fake_api[0]["model"] == "openrouter/openai/gpt-oss-20b"
    assert fake_api[0]["params"]["extra_body"] == {"provider": {"only": ["groq"], "allow_fallbacks": False}}
    assert r.route == "openrouter" and r.error is None


def test_allam_stays_on_groq(fake_api, tmp_path):
    r = client.call_model(CFG, TASK, "allam-2-7b", None, "en", cache=DiskCache(tmp_path))
    assert fake_api[0]["model"] == "groq/allam-2-7b" and "extra_body" not in fake_api[0]["params"]
    assert r.route == "groq" and r.provider == "groq"


def test_other_provider_is_rejected(monkeypatch, tmp_path):
    def other(model, messages, **params):
        resp = FakeResponse("ANSWER: خمسة\nFINAL: 5")
        resp._d["provider"] = "SomeOtherCloud"
        return resp

    monkeypatch.setattr(litellm, "completion", other)
    r = client.call_model(CFG, TASK, "openai/gpt-oss-20b", "low", "en", cache=DiskCache(tmp_path))
    assert r.provider == "SomeOtherCloud" and r.error == "wrong_provider: SomeOtherCloud"


def test_simulated_outage_blocks_model_even_from_cache(fake_api, tmp_path):
    import copy
    cache = DiskCache(tmp_path)
    client.call_model(CFG, TASK, "openai/gpt-oss-20b", "low", "en", cache=cache)  # now cached
    cfg = copy.deepcopy(CFG); cfg["simulated_outage"]["unavailable_models"] = ["openai/gpt-oss-20b"]
    with pytest.raises(client.ModelUnavailableError, match="SIMULATED"):
        client.call_model(cfg, TASK, "openai/gpt-oss-20b", "low", "en", cache=cache)
    assert len(fake_api) == 1


def test_simulated_outage_is_off_by_default():
    assert CFG["simulated_outage"]["unavailable_models"] == []
