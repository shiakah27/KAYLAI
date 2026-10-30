# -*- coding: utf-8 -*-
"""عميل النماذج: استدعاء واحد لمهمة واحدة بإعدادات محددة (نموذج، مستوى تفكير، لغة تفكير).

ماذا يفعل بالترتيب:
  1. يبني الرسائل (kayl/prompts.py).
  2. يتأكد أن المهمة تتسع لسياق النموذج، وإلا يرجع نتيجة "skipped" بدون استدعاء.
  3. يبحث في الكاش؛ إن لم يجد، يستدعي Groq عبر LiteLLM مع إعادة المحاولة عند حد الطلبات.
  4. يقرأ سطر FINAL وسطر ANSWER وقسم التفكير.
  5. يتحقق: هل فكّر النموذج باللغة المطلوبة؟ وهل جملة ANSWER بالعربية؟
  6. يحسب الحوسبة التقديرية والتكلفة من أعداد التوكنز التي يرجعها الـ API.

لا يقرر هذا الملف أي نموذج نستخدم ولا متى نصعّد؛ هذا عمل الموجّه (router)."""

import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from kayl import metrics
from kayl.cache import DiskCache, cache_key
from kayl.config import fits_context, max_output_tokens, model_info
from kayl.language import detect_lang, strip_copied
from kayl.parsing import parse_answer, parse_final, parse_thinking
from kayl.prompts import build_messages, has_reasoning_channel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class CallResult:
    """نتيجة استدعاء واحد. كل الأرقام من الـ API (ما عدا est_prompt_tokens)."""
    task_id: str
    task_type: str
    model: str
    effort: str | None
    think_lang: str
    content: str | None = None          # النص المرئي من النموذج
    reasoning: str | None = None        # نص التفكير (قناة gpt-oss أو قسم THINKING)
    answer_text: str | None = None      # جملة ANSWER العربية
    final: str | None = None            # قيمة سطر FINAL (None = لم يلتزم بالشكل)
    input_tokens: int = 0
    output_tokens: int = 0              # يشمل توكنز التفكير
    reasoning_tokens: int | None = None
    latency_s: float | None = None
    cost_usd: float | None = None       # None = سعر النموذج غير مؤكد
    compute_flops: float = 0.0          # تقدير
    reasoning_lang: str = "none"        # ar / en / mixed / none
    think_lang_followed: bool = False
    answer_in_arabic: bool = False
    finish_reason: str | None = None
    cached: bool = False
    skipped: bool = False               # لم نستدعِ النموذج (المهمة لا تتسع لسياقه)
    error: str | None = None
    est_prompt_tokens: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# أخطاء مؤقتة نعيد المحاولة عندها
def _retryable_errors():
    import litellm
    return (litellm.RateLimitError, litellm.APIConnectionError, litellm.Timeout,
            litellm.ServiceUnavailableError, litellm.InternalServerError)


def _wait_seconds(error: Exception, attempt: int, retry_cfg: dict) -> float:
    """كم ننتظر قبل المحاولة التالية: ما يقوله Groq ("try again in 1.2s") إن وُجد،
    وإلا انتظار يتضاعف (2، 4، 8...) حتى حد أقصى."""
    m = re.search(r"try again in ([\d.]+)(ms|s)", str(error))
    if m:
        wait = float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)
        return min(wait + 0.5, retry_cfg["max_wait_s"])
    return min(retry_cfg["base_wait_s"] * 2 ** attempt, retry_cfg["max_wait_s"])


def _request(model_id: str, params: dict, messages: list[dict], retry_cfg: dict) -> tuple[dict, float]:
    """يستدعي النموذج مع إعادة المحاولة. يرجع (الرد الخام كقاموس، زمن المحاولة الناجحة)."""
    import litellm
    retryable = _retryable_errors()
    for attempt in range(retry_cfg["max_attempts"]):
        try:
            start = time.perf_counter()
            resp = litellm.completion(model=model_id, messages=messages, **params)
            return resp.model_dump(), time.perf_counter() - start
        except retryable as e:
            if attempt == retry_cfg["max_attempts"] - 1:
                raise
            wait = _wait_seconds(e, attempt, retry_cfg)
            print(f"  retry {attempt + 1}: {type(e).__name__}, waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def call_model(cfg: dict, task: dict, model: str, effort: str | None, think_lang: str,
               cache: DiskCache | None = None, format_example: bool | None = None) -> CallResult:
    """استدعاء واحد لمهمة واحدة. لا يرمي استثناء: الأخطاء تُسجل في result.error.
    format_example: None = القيمة من config.yaml (prompts.format_example)."""
    import litellm

    info = model_info(cfg, model)
    if effort is not None and not info.get("supports_reasoning_effort"):
        raise ValueError(f"{model} does not support reasoning_effort")
    result = CallResult(task_id=task["id"], task_type=task["type"], model=model,
                        effort=effort, think_lang=think_lang)

    if format_example is None:
        format_example = cfg["prompts"]["format_example"]
    messages = build_messages(task, think_lang, info, with_example=format_example)
    result.est_prompt_tokens = litellm.token_counter(model=info["litellm_id"], messages=messages)
    if not fits_context(cfg, model, result.est_prompt_tokens):
        result.skipped, result.error = True, "context_too_long"
        return result

    params = {"temperature": cfg["generation"]["temperature"],
              "max_tokens": max_output_tokens(cfg, model)}
    if effort is not None:
        params["reasoning_effort"] = effort

    cache = cache or DiskCache(Path(cfg["paths"]["cache"]) / "llm")
    key = cache_key(info["litellm_id"], params, messages)
    record = cache.get(key)
    if record is not None:
        result.cached = True
    else:
        try:
            raw, latency = _request(info["litellm_id"], params, messages, cfg["retries"])
        except Exception as e:  # بعد كل المحاولات: نسجل الخطأ ولا نوقف التجربة
            result.error = f"{type(e).__name__}: {str(e)[:300]}"
            return result
        record = {"response": raw, "latency_s": latency,
                  "created": datetime.now(timezone.utc).isoformat()}
        cache.put(key, record)

    _fill_from_response(result, record, info, messages[-1]["content"])
    return result


def _fill_from_response(result: CallResult, record: dict, info: dict, source_text: str) -> None:
    """يملأ النتيجة من الرد الخام (من الـ API أو من الكاش، بنفس الطريقة).
    source_text = نص السؤال كما أرسلناه، لنحذف ما نسخه النموذج منه."""
    raw = record["response"]
    choice = raw["choices"][0]
    message = choice["message"]
    usage = raw["usage"]

    result.content = message.get("content")
    result.finish_reason = choice.get("finish_reason")
    result.latency_s = record["latency_s"]  # زمن الاستدعاء الأصلي، حتى عند القراءة من الكاش

    result.input_tokens = usage["prompt_tokens"]
    result.output_tokens = usage["completion_tokens"]
    details = usage.get("completion_tokens_details") or {}
    result.reasoning_tokens = details.get("reasoning_tokens")

    if has_reasoning_channel(info):
        result.reasoning = message.get("reasoning") or message.get("reasoning_content")
    else:
        result.reasoning = parse_thinking(result.content)
    result.answer_text = parse_answer(result.content)
    result.final = parse_final(result.content)

    # نحذف ما نُسخ من السؤال قبل كشف اللغة (انظر kayl/language.py)
    result.reasoning_lang = detect_lang(strip_copied(result.reasoning, source_text))
    result.think_lang_followed = result.reasoning_lang == result.think_lang
    result.answer_in_arabic = detect_lang(result.answer_text) == "ar"

    result.compute_flops = metrics.call_compute_flops(info, result.input_tokens, result.output_tokens)
    result.cost_usd = metrics.call_cost_usd(info, result.input_tokens, result.output_tokens)
