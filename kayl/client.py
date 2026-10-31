"""Model client: one call for one task with one configuration (model, effort, thinking language).

Steps:
  1. Build the messages (kayl/prompts.py).
  2. Check that the task fits the model's context window; if not, return a "skipped" result
     without calling the model.
  3. Look in the disk cache; on a miss, call the model through LiteLLM, retrying on rate limits.
  4. Parse the FINAL line, the ANSWER line and the reasoning.
  5. Check whether the model reasoned in the requested language and answered in Arabic.
  6. Compute estimated compute and dollar cost from the token counts returned by the API.

Choosing the model and deciding when to escalate is the agent's job (kayl/agent.py)."""

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
    """The result of one call. Every number comes from the API, except est_prompt_tokens."""
    task_id: str
    task_type: str
    model: str
    effort: str | None
    think_lang: str
    content: str | None = None          # visible text of the reply
    reasoning: str | None = None        # gpt-oss reasoning channel, or the THINKING section
    answer_text: str | None = None      # the Arabic ANSWER sentence
    final: str | None = None            # the FINAL value (None = the format was not followed)
    input_tokens: int = 0
    output_tokens: int = 0              # includes reasoning tokens
    reasoning_tokens: int | None = None
    latency_s: float | None = None
    cost_usd: float | None = None       # None = the model's price is not verified
    compute_flops: float = 0.0          # estimate
    reasoning_lang: str = "none"        # ar / en / mixed / none
    think_lang_followed: bool = False
    answer_in_arabic: bool = False
    finish_reason: str | None = None
    provider: str | None = None         # who actually answered: groq, or what OpenRouter reports
    route: str | None = None            # groq or openrouter
    cached: bool = False
    skipped: bool = False               # not called: the task does not fit the context window
    error: str | None = None
    est_prompt_tokens: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _retryable_errors():
    import litellm
    return (litellm.RateLimitError, litellm.APIConnectionError, litellm.Timeout,
            litellm.ServiceUnavailableError, litellm.InternalServerError)


class DailyLimitError(RuntimeError):
    """The model's daily token limit is reached. Waiting a minute does not help, so the run
    stops; rerunning later resumes from the cache."""


class ModelUnavailableError(RuntimeError):
    """The model is listed in simulated_outage in the config (a SIMULATED outage for robustness tests)."""


class CacheMissError(RuntimeError):
    """A cache-only run needed a response that was never recorded."""


def _wait_seconds(error: Exception, attempt: int, retry_cfg: dict) -> float:
    """Seconds to wait before the next attempt: what Groq asks for ("try again in 1.2s" or
    "1m26.4s") when present, otherwise an exponential backoff up to a cap."""
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", str(error))
    if m:
        minutes = int(m.group(1) or 0)
        wait = minutes * 60 + float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
        return min(wait + 0.5, retry_cfg["max_wait_s"])
    return min(retry_cfg["base_wait_s"] * 2 ** attempt, retry_cfg["max_wait_s"])


def _request(model_id: str, params: dict, messages: list[dict], retry_cfg: dict) -> tuple[dict, float]:
    """Call the model with retries. Returns (raw response as a dict, latency of the successful attempt)."""
    import litellm
    retryable = _retryable_errors()
    for attempt in range(retry_cfg["max_attempts"]):
        try:
            start = time.perf_counter()
            resp = litellm.completion(model=model_id, messages=messages, **params)
            return resp.model_dump(), time.perf_counter() - start
        except retryable as e:
            if isinstance(e, litellm.RateLimitError) and "per day" in str(e):
                raise DailyLimitError(str(e)[:300]) from e
            if attempt == retry_cfg["max_attempts"] - 1:
                raise
            wait = _wait_seconds(e, attempt, retry_cfg)
            print(f"  retry {attempt + 1}: {type(e).__name__}, waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def resolve_route(cfg: dict, model: str) -> tuple[str, str, dict]:
    """Where a call goes: (route, model_id, extra params).
    gpt-oss goes through OpenRouter pinned to Groq when the config says so; everything else
    goes to Groq directly."""
    routing = cfg.get("routing") or {}
    or_id = (routing.get("openrouter_ids") or {}).get(model)
    if routing.get("gpt_oss") == "openrouter" and or_id:
        return "openrouter", or_id, {"extra_body": {"provider": routing["openrouter_provider"]}}
    return "groq", model_info(cfg, model)["litellm_id"], {}


def call_model(cfg: dict, task: dict, model: str, effort: str | None, think_lang: str,
               cache: DiskCache | None = None, format_example: bool | None = None) -> CallResult:
    """One call for one task. Call errors are stored in result.error instead of raised, except
    DailyLimitError, ModelUnavailableError and CacheMissError, which the caller must handle.
    format_example: None means use prompts.format_example from the config."""
    import litellm

    if model in (cfg.get("simulated_outage") or {}).get("unavailable_models", []):
        raise ModelUnavailableError(f"{model} is marked unavailable (SIMULATED outage)")
    info = model_info(cfg, model)
    if effort is not None and not info.get("supports_reasoning_effort"):
        raise ValueError(f"{model} does not support reasoning_effort")
    result = CallResult(task_id=task["id"], task_type=task["type"], model=model,
                        effort=effort, think_lang=think_lang)

    if format_example is None:
        format_example = cfg["prompts"]["format_example"]
    messages = build_messages(task, think_lang, info, with_example=format_example)
    result.est_prompt_tokens = litellm.token_counter(model=info["litellm_id"], messages=messages)
    if not fits_context(cfg, model, result.est_prompt_tokens, effort):
        result.skipped, result.error = True, "context_too_long"
        return result

    params = {"temperature": cfg["generation"]["temperature"],
              "max_tokens": max_output_tokens(cfg, model, effort)}
    if effort is not None:
        params["reasoning_effort"] = effort
    route, model_id, extra = resolve_route(cfg, model)
    params.update(extra)
    result.route = route

    cache = cache or DiskCache(Path(cfg["paths"]["cache"]) / "llm")
    key = cache_key(model_id, params, messages)
    record = cache.get(key)
    if record is not None:
        result.cached = True
    elif cfg.get("cache_only"):
        raise CacheMissError(f"cache miss in --from-cache mode: task {task['id']}, {model_id}, "
                             f"effort {effort}, route {route}")
    else:
        try:
            raw, latency = _request(model_id, params, messages, cfg["retries"])
        except DailyLimitError:
            raise
        except Exception as e:  # after all retries: record the error and let the run continue
            result.error = f"{type(e).__name__}: {str(e)[:300]}"
            return result
        record = {"response": raw, "latency_s": latency,
                  "created": datetime.now(timezone.utc).isoformat()}
        cache.put(key, record)

    _fill_from_response(result, record, info, messages[-1]["content"])
    result.provider = record["response"].get("provider") or "groq"
    if route == "openrouter" and result.provider.lower() != "groq":
        # Any other provider would run on different hardware, so its numbers are not comparable.
        result.error = f"wrong_provider: {result.provider}"
    return result


def _fill_from_response(result: CallResult, record: dict, info: dict, source_text: str) -> None:
    """Fill the result from a raw response, the same way for a live call and a cached one.
    source_text is the question as sent, so text the model copied from it can be ignored."""
    raw = record["response"]
    choice = raw["choices"][0]
    message = choice["message"]
    usage = raw["usage"]

    result.content = message.get("content")
    result.finish_reason = choice.get("finish_reason")
    result.latency_s = record["latency_s"]  # the original call's latency, also when read from cache

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

    # Quoted or copied question text would make English reasoning look mixed (see kayl/language.py).
    result.reasoning_lang = detect_lang(strip_copied(result.reasoning, source_text))
    result.think_lang_followed = result.reasoning_lang == result.think_lang
    result.answer_in_arabic = detect_lang(result.answer_text) == "ar"

    result.compute_flops = metrics.call_compute_flops(info, result.input_tokens, result.output_tokens)
    result.cost_usd = metrics.call_cost_usd(info, result.input_tokens, result.output_tokens)
