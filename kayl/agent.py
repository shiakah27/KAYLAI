# -*- coding: utf-8 -*-
"""وكيل كَيْل: رسم LangGraph بسيط من خمس خطوات لكل مهمة.

    estimate -> choose -> solve -> verify --(نجح)--> finish
                            ^          |
                            |      (فشل، وتبقى محاولة)
                            +-- escalate

  1. estimate : ما نوع المهمة؟ وما الذي تعلمناه عنها من المعايرة (calibration profile)؟
  2. choose   : من الإعدادات التي تحقق حدَّي الدقة والزمن لهذا النوع، نختار الأقل حوسبة
                لكل إجابة صحيحة (والتعادل بالزمن). نتجاوز allam إذا لم تتسع المهمة لسياقه.
  3. solve    : استدعاء النموذج بالإعداد المختار.
  4. verify   : بدون معرفة الجواب الصحيح: هل يوجد FINAL؟ هل هو صالح لنوع المهمة؟
                هل جملة ANSWER بالعربية؟ هل انقطع الجواب؟
  5. escalate : عند الفشل: نرفع مستوى التفكير أولًا، ثم ننتقل لنموذج أكبر. حدّ أقصى: تصعيدان.

كل قرار يُحفظ مع سبب قصير يفهمه الإنسان (decisions)، لتعرضه الواجهة."""

import re
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from kayl.client import CallResult, call_model
from kayl.config import fits_context, model_info
from kayl.grading import AMBIGUOUS, INVALID, NO_FINAL, validate
from kayl.prompts import build_messages

TYPES = ("math", "knowledge", "intent")
SIZE_ORDER = ("small", "medium", "large")
LABEL_RE = re.compile(r"^[a-z]+(_[a-z]+)*$")

VERIFY_MESSAGES = {
    NO_FINAL: "no FINAL line in the reply",
    AMBIGUOUS: "FINAL contains more than one number",
    INVALID: "FINAL is not a valid answer for this task type",
}


class AgentState(TypedDict, total=False):
    task: dict
    task_type: str
    current: dict          # الإعداد الحالي: name, model, effort, think_lang, format_example
    attempts: list[dict]   # كل محاولة: الإعداد + CallResult + نتيجة التحقق
    decisions: list[dict]  # [{"step": ..., "reason": ...}]
    escalations: int
    verified: bool


# ---------- helpers ----------

def _decide(state: AgentState, step: str, reason: str) -> list[dict]:
    return state.get("decisions", []) + [{"step": step, "reason": reason}]


def _tflops(x: float | None) -> str:
    return "n/a" if x is None else f"{x / 1e12:.2f} TFLOPs"


def _short(model: str) -> str:
    return {"allam-2-7b": "allam", "openai/gpt-oss-20b": "20b", "openai/gpt-oss-120b": "120b"}.get(model, model)


def infer_type(task: dict) -> tuple[str, str]:
    """نوع المهمة وسبب ذلك. نستخدم حقل type إن وُجد، وإلا نستنتج من شكل المهمة."""
    if task.get("type") in TYPES:
        return task["type"], f"task type '{task['type']}' is given by the task"
    choices = task.get("choices") or []
    if choices and all(LABEL_RE.match(str(c)) for c in choices):
        return "intent", f"inferred 'intent': the task lists {len(choices)} labels"
    if choices:
        return "knowledge", f"inferred 'knowledge': the task has {len(choices)} answer options"
    return "math", "inferred 'math': no options or labels were given"


def task_fits(cfg: dict, task: dict, conf: dict) -> tuple[bool, int]:
    """هل تتسع المهمة لسياق نموذج هذا الإعداد؟ (نفس الفحص الذي يجريه العميل)."""
    import litellm
    info = model_info(cfg, conf["model"])
    fmt = conf.get("format_example", cfg["prompts"]["format_example"])
    msgs = build_messages(task, conf["think_lang"], info, with_example=fmt)
    est = litellm.token_counter(model=info["litellm_id"], messages=msgs)
    return fits_context(cfg, conf["model"], est), est


def next_config(cfg: dict, current: dict) -> tuple[dict | None, str]:
    """الخطوة التالية في سلم التصعيد: رفع التفكير أولًا، ثم نموذج أكبر (بنفس مستوى التفكير)."""
    info = model_info(cfg, current["model"])
    efforts = cfg["reasoning_efforts"]
    if info.get("supports_reasoning_effort") and current["effort"] != efforts[-1]:
        new_effort = efforts[efforts.index(current["effort"]) + 1]
        conf = {**current, "effort": new_effort}
        return conf, f"raise effort {current['effort']} → {new_effort} on {_short(current['model'])}"
    size = SIZE_ORDER.index(info["size"])
    bigger = [m for m, i in cfg["models"].items() if SIZE_ORDER.index(i["size"]) == size + 1]
    if not bigger:
        return None, "already at the largest model with the highest effort"
    model = bigger[0]
    effort = None
    if model_info(cfg, model).get("supports_reasoning_effort"):
        effort = current["effort"] or efforts[0]  # لا ننزل في مستوى التفكير
    conf = {"model": model, "effort": effort, "think_lang": current["think_lang"],
            "format_example": cfg["prompts"]["format_example"]}
    return conf, f"move to the bigger model {_short(model)} (effort {effort or 'n/a'})"


def _name(cfg: dict, conf: dict) -> str:
    """اسم الإعداد من قائمة sweep إن وُجد، وإلا اسم مركب."""
    for c in cfg["sweep"]:
        if (c["model"], c["effort"], c["think_lang"]) == (conf["model"], conf["effort"], conf["think_lang"]) \
                and c.get("format_example", cfg["prompts"]["format_example"]) == \
                conf.get("format_example", cfg["prompts"]["format_example"]):
            return c["name"]
    parts = [_short(conf["model"]), conf["effort"], conf["think_lang"] if conf["think_lang"] != "en" else None]
    return "-".join(p for p in parts if p)


# ---------- graph ----------

def build_agent(cfg: dict, profile: dict, call_fn: Callable[..., CallResult] = call_model):
    """يبني رسم الوكيل. call_fn قابل للاستبدال (الاختبارات تمرر نموذجًا مزيفًا)."""
    floors = cfg["floors"]
    max_escalations = cfg["agent"]["max_escalations"]

    def estimate(state: AgentState) -> dict:
        task_type, why = infer_type(state["task"])
        measured = profile.get(task_type, {})
        n = max((s["n"] for s in measured.values()), default=0)
        why += (f"; calibration profile has {len(measured)} configurations measured on {n} items"
                if measured else "; no calibration profile for this type")
        return {"task_type": task_type, "attempts": [], "escalations": 0,
                "decisions": _decide(state, "estimate", why)}

    def choose(state: AgentState) -> dict:
        task, measured = state["task"], profile.get(state["task_type"], {})
        notes, eligible = [], []
        for c in cfg["sweep"]:
            stats = measured.get(c["name"])
            if stats is None:
                continue
            if c.get("format_example", cfg["prompts"]["format_example"]) != cfg["prompts"]["format_example"]:
                continue  # إعدادات القياس فقط (مثل allam-noex) ليست خيارًا للوكيل
            if stats["n"] < cfg["agent"]["min_calibration_n"]:
                notes.append(f"{c['name']} ignored: measured on only {stats['n']} items")
                continue
            ok, est = task_fits(cfg, task, c)
            if not ok:
                notes.append(f"{c['name']} skipped: the task (~{est} tokens) does not fit its "
                             f"{model_info(cfg, c['model'])['context_window']:,}-token context")
                continue
            eligible.append((c, stats))

        meets = [(c, s) for c, s in eligible
                 if s["accuracy"] >= floors["accuracy"]
                 and s["p95_latency_s"] is not None and s["p95_latency_s"] <= floors["latency_p95_s"]]
        floor_text = f"accuracy ≥ {floors['accuracy']:.0%}, p95 latency ≤ {floors['latency_p95_s']} s"

        if meets:
            c, s = min(meets, key=lambda cs: (cs[1]["compute_per_success"] or float("inf"),
                                              cs[1]["avg_latency_s"] or float("inf")))
            why = (f"chose {c['name']}: lowest estimated compute per success ({_tflops(s['compute_per_success'])}) "
                   f"among {len(meets)} configurations meeting the floors ({floor_text}); "
                   f"calibration accuracy {s['accuracy']:.0%} on {s['n']} items")
        elif eligible:
            c, s = max(eligible, key=lambda cs: (cs[1]["accuracy"], -(cs[1]["compute_per_success"] or float("inf"))))
            why = (f"no configuration met the floors ({floor_text}); chose the most accurate one, "
                   f"{c['name']} ({s['accuracy']:.0%} on {s['n']} items)")
        else:
            c = {"name": None, "model": cfg["fallback_model"], "effort": "medium", "think_lang": "en"}
            why = f"no calibration data for this task type; using the fallback model {_short(c['model'])} at medium effort"
        if notes:
            why += ". " + "; ".join(notes)
        current = {k: c.get(k) for k in ("model", "effort", "think_lang", "format_example")}
        if current["format_example"] is None:
            current["format_example"] = cfg["prompts"]["format_example"]
        current["name"] = c.get("name") or _name(cfg, current)
        return {"current": current, "decisions": _decide(state, "choose", why)}

    def solve(state: AgentState) -> dict:
        c = state["current"]
        r = call_fn(cfg, state["task"], c["model"], c["effort"], c["think_lang"],
                    format_example=c["format_example"])
        lang = {"en": "English", "ar": "Arabic"}[c["think_lang"]]
        why = (f"solved with {c['name']} ({_short(c['model'])}, effort {c['effort'] or 'n/a'}, "
               f"thinking in {lang}): {r.input_tokens} in / {r.output_tokens} out tokens, "
               f"{_tflops(r.compute_flops)}" + (" (from cache)" if r.cached else ""))
        attempts = state["attempts"] + [{"config": c["name"], **c, "result": r}]
        return {"attempts": attempts, "decisions": _decide(state, "solve", why)}

    def verify(state: AgentState) -> dict:
        task, attempt = state["task"], state["attempts"][-1]
        r: CallResult = attempt["result"]
        if r.error:
            ok, why = False, f"failed: the call did not succeed ({r.error[:80]})"
        elif r.finish_reason == "length":
            ok, why = False, "failed: the reply was cut off at the token limit"
        else:
            valid, reason = validate(task, r.final)
            if not valid:
                ok, why = False, f"failed: {VERIFY_MESSAGES[reason]}"
            elif not r.answer_in_arabic:
                ok, why = False, "failed: the ANSWER line is missing or not in Arabic"
            else:
                ok, why = True, f"passed: FINAL '{r.final}' is valid for {task_type_label(state)} and the ANSWER is in Arabic"
        attempts = state["attempts"][:-1] + [{**attempt, "verified": ok, "verify_reason": why}]
        return {"verified": ok, "attempts": attempts, "decisions": _decide(state, "verify", why)}

    def task_type_label(state: AgentState) -> str:
        return {"math": "a math task (one number)", "knowledge": "a multiple-choice task (an option letter)",
                "intent": "intent classification (a known label)"}[state["task_type"]]

    def route(state: AgentState) -> str:
        if state["verified"]:
            return "finish"
        if state["escalations"] >= max_escalations:
            return "finish"
        conf, _ = next_config(cfg, state["current"])
        return "escalate" if conf else "finish"

    def escalate(state: AgentState) -> dict:
        conf, step = next_config(cfg, state["current"])
        conf["name"] = _name(cfg, conf)
        last = state["attempts"][-1]["verify_reason"].removeprefix("failed: ")
        why = f"escalation {state['escalations'] + 1}/{max_escalations}: {step}, because {last}"
        return {"current": conf, "escalations": state["escalations"] + 1,
                "decisions": _decide(state, "escalate", why)}

    def finish(state: AgentState) -> dict:
        last = state["attempts"][-1]
        if state["verified"]:
            why = f"accepted the answer from {last['config']} after {state['escalations']} escalation(s)"
        elif state["escalations"] >= max_escalations:
            why = f"no verified answer after {max_escalations} escalations; returning the last answer, marked unverified"
        else:
            why = "no verified answer and no bigger configuration left; returning the last answer, marked unverified"
        return {"decisions": _decide(state, "finish", why)}

    g = StateGraph(AgentState)
    for name, fn in [("estimate", estimate), ("choose", choose), ("solve", solve),
                     ("verify", verify), ("escalate", escalate), ("finish", finish)]:
        g.add_node(name, fn)
    g.add_edge(START, "estimate")
    g.add_edge("estimate", "choose")
    g.add_edge("choose", "solve")
    g.add_edge("solve", "verify")
    g.add_conditional_edges("verify", route, {"finish": "finish", "escalate": "escalate"})
    g.add_edge("escalate", "solve")
    g.add_edge("finish", END)
    return g.compile()


def run_agent(task: dict, cfg: dict, profile: dict, call_fn: Callable[..., CallResult] = call_model) -> dict:
    """يحل مهمة واحدة ويرجع الجواب، وكل المحاولات، وكل القرارات مع أسبابها، والمجاميع."""
    state = build_agent(cfg, profile, call_fn).invoke({"task": task})
    results = [a["result"] for a in state["attempts"]]
    last = results[-1]
    costs = [r.cost_usd for r in results]
    return {
        "task_id": task.get("id"), "task_type": state["task_type"],
        "final": last.final, "answer_text": last.answer_text,
        "verified": state["verified"], "escalations": state["escalations"],
        "decisions": state["decisions"],
        "attempts": [{"config": a["config"], "model": a["model"], "effort": a["effort"],
                      "think_lang": a["think_lang"], "final": a["result"].final,
                      "verified": a["verified"], "verify_reason": a["verify_reason"],
                      "input_tokens": a["result"].input_tokens, "output_tokens": a["result"].output_tokens,
                      "compute_flops": a["result"].compute_flops, "cost_usd": a["result"].cost_usd,
                      "latency_s": a["result"].latency_s} for a in state["attempts"]],
        # المجاميع تشمل المحاولات الفاشلة (هكذا نحسب الحوسبة لكل إجابة صحيحة)
        "total_compute_flops": sum(r.compute_flops for r in results),
        "total_cost_usd": None if any(c is None for c in costs) else sum(costs),
        "total_latency_s": sum(r.latency_s or 0 for r in results),
    }
