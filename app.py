# -*- coding: utf-8 -*-
"""
كَيْل — واجهة العرض (Streamlit).

ماذا تعرض:
  - مسار القرار خطوة بخطوة: فهم المهمة -> اختيار النموذج والتفكير -> الجواب -> التحقق -> التصعيد عند الحاجة.
  - بطاقة السبب (جملة واحدة من أرقام المعايرة)، وبطاقات الأرقام، وأشرطة الحوسبة مقابل "كل شيء للأكبر".
  - تبويب النتائج: جدول الاختبار النهائي من results/final_test_summary.json.
  - وضع الإعادة (Replay): يعرض تشغيلات حقيقية مسجلة من الكاش، بدون أي استدعاء API.

التشغيل:
    streamlit run app.py
"""

import copy
import html
import json
import re
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import litellm
import streamlit as st

from kayl.agent import infer_type, run_agent
from kayl.client import call_model
from kayl.config import accuracy_floor, load_config
from kayl.grading import normalize_label
from kayl.profile import load_profile

st.set_page_config(page_title="KAYL — Intelligence Without Waste", page_icon="◆", layout="wide",
                   initial_sidebar_state="collapsed")

LARGEST = ("openai/gpt-oss-120b", "medium")  # "All to largest" = baseline A
PRETTY_MODEL = {"allam-2-7b": "ALLaM-2-7B", "openai/gpt-oss-20b": "gpt-oss-20b", "openai/gpt-oss-120b": "gpt-oss-120b"}
TYPE_NAMES = {"math": "math", "knowledge": "multiple-choice knowledge", "intent": "service-request intent"}
EXAMPLES = {  # من بيانات المعايرة (وليس الاختبار)
    "math": ("Math", "gsm8k-ar-1107"),
    "knowledge": ("Knowledge", "arabicmmlu-11638"),
    "intent": ("Saudi-dialect request", "intent-6"),
}
REPLAYS = [  # (key, title, source, task id, label, overrides)
    ("math", "Math question", "calibration", "gsm8k-ar-1107", "Replay of a recorded live run", {"route": "groq"}),
    ("knowledge", "Knowledge question", "calibration", "arabicmmlu-11638", "Replay of a recorded live run",
     {"route": "groq"}),
    ("intent", "Saudi-dialect request", "calibration", "intent-6", "Replay of a recorded live run",
     {"route": "groq"}),
    ("escalation", "Escalation case", "drill", "gsm8k-ar-707", "Replay of a recorded live run",
     {"route": "groq"}),
    ("outage", "Outage — SIMULATED", "drill", "gsm8k-ar-1242",
     "Replay of a recorded live run · SIMULATED outage (gpt-oss-20b marked unavailable)",
     {"route": "openrouter", "unavailable": ["openai/gpt-oss-20b"]}),
]
PIPELINE = [("understand", "Understand task"), ("choose", "Choose model & effort"), ("answer", "Answer"),
            ("verify", "Verify"), ("escalate", "Escalate only if needed")]
STEP_OF = {"estimate": "understand", "choose": "choose", "fallback": "choose", "solve": "answer",
           "verify": "verify", "escalate": "escalate"}

# ---------------------------------------------------------------- theme

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&family=Noto+Kufi+Arabic:wght@400;600&display=swap');
:root{
  --bg:#041A23; --glow:#0F4846; --lime:#C8F63E; --mint:#00CF90; --teal1:#009294; --teal2:#007C92;
  --text:#FFFFFF; --muted:#A9BDB4; --line:rgba(255,255,255,0.12); --card:rgba(255,255,255,0.035);
}
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="collapsedControl"] {display:none !important;}
.stApp{background:radial-gradient(circle at 100% 0%, var(--glow) 0%, rgba(4,26,35,0) 42%), var(--bg); color:var(--text);
  font-family:'Inter',sans-serif;}
.block-container{padding-top:2.2rem; padding-bottom:3rem; max-width:1180px;}
h1,h2,h3,.k-title{font-family:'Space Grotesk',sans-serif; text-transform:uppercase; letter-spacing:.14em; color:var(--text);}
p, label, .stMarkdown {color:var(--text);}
.k-hero{display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; margin-bottom:1.4rem;
  border-bottom:1px solid var(--line); padding-bottom:1.2rem;}
.k-hero .t{font-family:'Space Grotesk'; text-transform:uppercase; letter-spacing:.18em; font-size:2rem; font-weight:700; line-height:1.15;}
.k-hero .t span{color:var(--lime);}
.k-hero .s{color:var(--muted); margin-top:.35rem; font-size:.95rem;}
.k-hero .ar{font-family:'Noto Kufi Arabic'; direction:rtl; font-size:2.6rem; color:var(--lime); font-weight:600;}
.k-label{font-family:'JetBrains Mono'; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted);}
.k-card{background:var(--card); border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem;}
.k-ar{font-family:'Noto Kufi Arabic'; direction:rtl; text-align:right; unicode-bidi:plaintext;}
.k-banner{border:1px solid var(--line); border-radius:999px; padding:.45rem 1rem; display:inline-block;
  font-family:'JetBrains Mono'; font-size:.78rem; letter-spacing:.06em; color:var(--lime); margin:.2rem 0 1rem;}
.k-banner.sim{color:#041A23; background:var(--lime); border-color:var(--lime);}
.k-pipe{display:grid; grid-template-columns:repeat(5,1fr); gap:.6rem; margin:.4rem 0 1rem;}
.k-step{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:.8rem .85rem; min-height:118px;
  transition:all .3s ease; opacity:.45;}
.k-step .n{font-family:'JetBrains Mono'; font-size:.7rem; color:var(--muted); letter-spacing:.1em;}
.k-step .h{font-family:'Space Grotesk'; text-transform:uppercase; letter-spacing:.08em; font-size:.82rem; margin:.25rem 0 .45rem;}
.k-step .r{font-size:.8rem; color:var(--muted); line-height:1.35;}
.k-step.active{opacity:1; border-color:var(--lime); box-shadow:0 0 0 1px var(--lime), 0 0 24px rgba(200,246,62,.18);}
.k-step.active .h{color:var(--lime);}
.k-step.done{opacity:1; border-color:rgba(0,207,144,.55);}
.k-step.done .h{color:var(--mint);}
.k-step.fail{opacity:1; border-color:#E8735A;}
.k-step.fail .h{color:#E8735A;}
.k-step.skip{opacity:.7;}
.k-reason{border-left:3px solid var(--lime); padding:.9rem 1.1rem; background:var(--card); border-radius:0 14px 14px 0;
  font-size:1.02rem; line-height:1.55; margin:.4rem 0 1rem;}
.k-metrics{display:grid; grid-template-columns:repeat(4,1fr); gap:.6rem; margin-bottom:1rem;}
.k-metric .v{font-family:'JetBrains Mono'; font-size:1.55rem; font-weight:600; margin-top:.25rem;}
.k-metric .u{font-family:'JetBrains Mono'; font-size:.78rem; color:var(--muted);}
.k-badge{display:inline-block; border-radius:999px; padding:.15rem .7rem; font-family:'JetBrains Mono'; font-size:.85rem; font-weight:600;}
.k-badge.yes{background:rgba(0,207,144,.15); color:var(--mint); border:1px solid var(--mint);}
.k-badge.no{background:rgba(232,115,90,.12); color:#E8735A; border:1px solid #E8735A;}
.k-bars .row{display:grid; grid-template-columns:150px 1fr 110px; align-items:center; gap:.8rem; margin:.45rem 0;}
.k-bars .lab{font-family:'JetBrains Mono'; font-size:.78rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em;}
.k-bars .track{background:rgba(255,255,255,.06); border-radius:999px; height:16px; overflow:hidden;}
.k-bars .fill{height:100%; border-radius:999px;}
.k-bars .fill.big{background:#5E6E6A;}
.k-bars .fill.kayl{background:linear-gradient(90deg,var(--lime),var(--mint));}
.k-bars .val{font-family:'JetBrains Mono'; font-size:.85rem; text-align:right;}
.k-save{font-family:'Space Grotesk'; text-transform:uppercase; letter-spacing:.1em; color:var(--lime); font-size:1.05rem; margin-top:.4rem;}
.k-answer{font-family:'Noto Kufi Arabic'; direction:rtl; text-align:right; font-size:1.15rem; line-height:1.8;}
.k-foot{color:var(--muted); font-size:.8rem; margin-top:1.4rem; border-top:1px solid var(--line); padding-top:.9rem;
  display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap;}
.k-teal{background:linear-gradient(120deg,var(--teal1),var(--teal2)); border-radius:18px; padding:1.2rem 1.4rem; margin-bottom:1rem;}
.k-teal .big{font-family:'JetBrains Mono'; font-size:1.9rem; font-weight:600;}
.k-teal .grid{display:grid; grid-template-columns:repeat(3,1fr); gap:1rem;}
.k-table{width:100%; border-collapse:collapse; font-size:.86rem;}
.k-table th{font-family:'JetBrains Mono'; font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
  text-align:left; border-bottom:1px solid var(--line); padding:.55rem .5rem;}
.k-table td{border-bottom:1px solid var(--line); padding:.55rem .5rem; font-family:'JetBrains Mono';}
.k-table td:first-child{font-family:'Inter'; font-weight:500;}
.k-table tr.kayl td{color:var(--lime);}
/* Streamlit widgets */
.stButton > button{border-radius:999px !important; background:var(--lime) !important; color:#041A23 !important; border:none !important;
  font-family:'JetBrains Mono' !important; font-weight:600 !important; letter-spacing:.04em; padding:.45rem 1.2rem !important;}
.stButton > button:hover{filter:brightness(1.07); box-shadow:0 0 18px rgba(200,246,62,.25);}
.stButton > button[kind="secondary"]{background:transparent !important; color:var(--lime) !important; border:1px solid var(--lime) !important;}
textarea{direction:rtl; text-align:right; font-family:'Noto Kufi Arabic' !important; background:rgba(255,255,255,.04) !important;
  color:var(--text) !important; border:1px solid var(--line) !important; border-radius:14px !important;}
[data-baseweb="select"] > div, [data-baseweb="radio"]{background:rgba(255,255,255,.04) !important; border-color:var(--line) !important; color:var(--text) !important;}
.stTabs [data-baseweb="tab-list"]{gap:.4rem; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{font-family:'Space Grotesk'; text-transform:uppercase; letter-spacing:.12em; color:var(--muted);}
.stTabs [aria-selected="true"]{color:var(--lime) !important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--lime) !important;}
[data-testid="stWidgetLabel"] p{color:var(--muted) !important; font-family:'JetBrains Mono'; font-size:.75rem; letter-spacing:.08em; text-transform:uppercase;}
@media (max-width: 900px){ .k-pipe{grid-template-columns:1fr 1fr;} .k-metrics{grid-template-columns:1fr 1fr;} .k-teal .grid{grid-template-columns:1fr;} }
"""

FOOTNOTE = "Compute is an estimate (2 × active parameters × tokens). Energy not measured."


def esc(x) -> str:
    return html.escape(str(x))


AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
INTENT_AR = {
    "renew_national_id": "تجديد الهوية الوطنية", "renew_passport": "تجديد جواز السفر",
    "traffic_violation_inquiry": "الاستعلام عن المخالفات المرورية", "vehicle_ownership_transfer": "نقل ملكية مركبة",
    "book_medical_appointment": "حجز موعد طبي", "report_utility_outage": "الإبلاغ عن انقطاع الكهرباء أو الماء",
    "social_support_inquiry": "الاستفسار عن الدعم الحكومي", "file_complaint": "تقديم شكوى",
    "other": "خارج الخدمات المتاحة",
}


def ar_num(text) -> str:
    """للعرض فقط (صندوق السؤال وصندوق الإجابة): أرقام عربية مشرقية مع ٫ و٬ و٪.
    لا نلمس الأرقام الملتصقة بحروف لاتينية (مثل gpt-oss-20b). البيانات وما يُرسل للنموذج لا يتغيران."""
    def conv(m):
        t = re.sub(r"(?<=\d)\.(?=\d)", "٫", m.group(0))
        return re.sub(r"(?<=\d),(?=\d{3})", "٬", t).translate(AR_DIGITS)
    return re.sub(r"(?<![A-Za-z\-\d])\d[\d.,]*(?![\dA-Za-z])", conv, str(text)).replace("%", "٪")


# ---------------------------------------------------------------- data

@st.cache_resource
def base_config() -> dict:
    return load_config()


@st.cache_resource
def profile() -> dict:
    return load_profile()


@st.cache_data
def calibration_tasks() -> dict:
    return {json.loads(l)["id"]: json.loads(l) for l in Path("data/calibration.jsonl").open(encoding="utf-8")}


@st.cache_data(show_spinner=False)
def drill_tasks() -> dict:
    from run_fallback_drill import drill_tasks as load
    return {t["id"]: t for t in load()}


@st.cache_data
def final_summary() -> dict | None:
    p = Path("results/final_test_summary.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def intent_labels() -> list[str]:
    return next(t["choices"] for t in calibration_tasks().values() if t["type"] == "intent")


def case_config(route: str | None = None, unavailable: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(base_config())
    if route:
        cfg["routing"]["gpt_oss"] = route
    cfg["simulated_outage"]["unavailable_models"] = list(unavailable or [])
    return cfg


class ReplayMiss(RuntimeError):
    """في وضع الإعادة لا نستدعي أي نموذج: كل ما ليس في الكاش يُعتبر غير مسجل."""


@contextmanager
def no_api_calls():
    original = litellm.completion

    def refuse(*args, **kwargs):
        raise ReplayMiss("replay mode: this call was not recorded")

    litellm.completion = refuse
    try:
        yield
    finally:
        litellm.completion = original


# ---------------------------------------------------------------- task from free text

OPTION_RE = re.compile(r"^\s*([A-Ea-e]|[أبجده])[\).:\-]\s*(.+)$")


def build_task(text: str, kind: str) -> tuple[dict, str]:
    """يبني مهمة من النص الحر. يرجع (المهمة، كيف عرفنا نوعها)."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    options = [OPTION_RE.match(l) for l in lines]
    if kind in ("Auto", "Knowledge") and sum(bool(m) for m in options) >= 2:
        question = "\n".join(l for l, m in zip(lines, options) if not m)
        return {"id": "live", "question": question, "choices": [m.group(2) for m in options if m]}, "options A–E found"
    if kind == "Math" or (kind == "Auto" and re.search(r"[0-9٠-٩]", text)):
        return {"id": "live", "question": text.strip()}, "contains numbers" if kind == "Auto" else "selected"
    return {"id": "live", "question": text.strip(), "choices": intent_labels()}, \
        "no numbers or options: treated as a service request" if kind == "Auto" else "selected"


# ---------------------------------------------------------------- run

def run_case(task: dict, cfg: dict, replay: bool) -> dict:
    """يشغّل كَيْل، ثم "كل شيء للأكبر" على نفس المهمة للمقارنة."""
    with (no_api_calls() if replay else nullcontext()):
        out = run_agent(task, cfg, profile())
        big_cfg = case_config(cfg["routing"]["gpt_oss"])  # المقارنة بدون انقطاع مُحاكى
        big = call_model(big_cfg, {**task, "type": out["task_type"]}, *LARGEST, "en")
    out.pop("_results", None)
    stats = profile().get(out["task_type"], {}).get("120b-medium")
    if big.error or big.skipped:
        avg = stats["compute_per_success"] * stats["correct"] / stats["n"] if stats else None
        largest = {"compute": avg, "source": "calibration average for this task type (not recorded for this task)"}
    else:
        largest = {"compute": big.compute_flops, "source": "same task, gpt-oss-120b · medium"
                   + (" (recorded)" if big.cached else " (live call)")}
    missing = [a for a in out["attempts"] if a["verify_reason"] and "not recorded" in a["verify_reason"]]
    return {"out": out, "largest": largest, "missing": bool(missing)}


# ---------------------------------------------------------------- rendering

def config_label(cfg: dict, name: str | None, model: str | None = None, effort: str | None = None) -> str:
    sweep = {c["name"]: c for c in cfg["sweep"]}
    if name in sweep:
        model, effort = sweep[name]["model"], sweep[name]["effort"]
    return f"{PRETTY_MODEL.get(model, model)} · {effort + ' effort' if effort else 'no effort control'}"


def display_final(task_type: str, final: str | None) -> str:
    """قيمة FINAL للعرض فقط: جواب الرياضيات بأرقام عربية مشرقية؛ الحرف واسم النية كما هما."""
    if not final:
        return "—"
    return ar_num(final) if task_type == "math" else final


def display_reason(task_type: str, reason: str) -> str:
    """في سجل القرارات: الجواب المقتبس (FINAL '...') بأرقام عربية للعرض فقط؛ باقي النص الإنجليزي كما هو."""
    return re.sub(r"FINAL '([^']*)'", lambda m: f"FINAL '{display_final(task_type, m.group(1))}'", reason)


def step_result(d: dict, out: dict, cfg: dict, attempt_no: int) -> tuple[str, str]:
    """(النص القصير تحت الخطوة، الحالة)."""
    step, why = d["step"], d["reason"]
    if step == "estimate":
        return f"{TYPE_NAMES[out['task_type']].capitalize()} task · calibration profile loaded", "done"
    if step == "choose":
        a = out["attempts"][0]
        first = out["fallbacks"][0]["from"] if out["fallbacks"] else a["config"]
        return config_label(cfg, first), "done"
    if step == "fallback":
        return "Fallback → " + config_label(cfg, out["fallbacks"][0]["to"]) + \
            (" · SIMULATED outage" if "SIMULATED" in why else ""), "fail"
    if step == "solve":
        a = out["attempts"][attempt_no]
        return f"FINAL {display_final(out['task_type'], a['final'])} · {a['input_tokens']} in / {a['output_tokens']} out tokens", "done"
    if step == "verify":
        ok = why.startswith("passed")
        return ("Passed: valid format, Arabic answer" if ok else why.removeprefix("failed: ").capitalize()), \
            ("done" if ok else "fail")
    if step == "escalate":
        return why.split(": ", 1)[1].split(", because")[0].capitalize(), "active"
    return why, "done"


def pipeline_html(states: dict, results: dict) -> str:
    cells = []
    for i, (key, title) in enumerate(PIPELINE, 1):
        st_ = states.get(key, "")
        res = "<br>".join(esc(r) for r in results.get(key, [])) or "&nbsp;"
        cells.append(f'<div class="k-step {st_}"><div class="n">0{i}</div><div class="h">{esc(title)}</div>'
                     f'<div class="r">{res}</div></div>')
    return f'<div class="k-pipe">{"".join(cells)}</div>'


def animate_pipeline(placeholder, out: dict, cfg: dict, animate: bool):
    states, results, attempt_no = {}, {}, -1
    events = [d for d in out["decisions"] if d["step"] in STEP_OF]
    for d in events:
        key = STEP_OF[d["step"]]
        if d["step"] == "solve":
            attempt_no += 1
        text, state = step_result(d, out, cfg, attempt_no)
        if d["step"] == "solve" and attempt_no > 0:
            text = f"Attempt {attempt_no + 1}: " + text
        if animate:
            states[key] = "active"
            results.setdefault(key, []).append(text)
            placeholder.markdown(pipeline_html(states, results), unsafe_allow_html=True)
            time.sleep(0.55)
        else:
            results.setdefault(key, []).append(text)
        states[key] = state if state != "active" else "done"
    if "escalate" not in results:
        results["escalate"] = ["Not needed — verification passed"] if out["verified"] else ["No step left"]
        states["escalate"] = "skip"
    placeholder.markdown(pipeline_html(states, results), unsafe_allow_html=True)


def reason_sentence(out: dict, cfg: dict) -> str:
    t, prof = out["task_type"], profile()
    first = out["fallbacks"][0]["from"] if out["fallbacks"] else out["attempts"][0]["config"]
    s = prof.get(t, {}).get(first)
    choose = next(d["reason"] for d in out["decisions"] if d["step"] == "choose")
    if s and choose.startswith("chose"):
        floor, _ = accuracy_floor(cfg, prof, t)
        text = (f"For {TYPE_NAMES[t]} tasks, KAYL chose {config_label(cfg, first)} because on {s['n']} calibration "
                f"questions it scored {s['correct']}/{s['n']}, meeting the {floor:.0%} accuracy floor, with the lowest "
                f"estimated compute per correct answer ({s['compute_per_success'] / 1e12:.2f} TFLOPs).")
    else:
        text = choose[0].upper() + choose[1:] + "."
    for fb in out["fallbacks"]:
        text += (f" {PRETTY_MODEL['openai/gpt-oss-20b'] if '20b' in fb['from'] else fb['from']} was unavailable"
                 f"{' (SIMULATED outage)' if 'SIMULATED' in fb['reason'] else ''}, so it switched to "
                 f"{config_label(cfg, fb['to'])}, the next cheapest configuration that meets the floors.")
    for d in out["decisions"]:
        if d["step"] == "escalate":
            step, because = d["reason"].split(": ", 1)[1].split(", because ")
            text += f" Verification failed ({because}), so it chose to {step}."
    return text


def results_view(r: dict, cfg: dict, animate: bool):
    out, largest = r["out"], r["largest"]
    st.markdown('<div class="k-label">Decision pipeline</div>', unsafe_allow_html=True)
    animate_pipeline(st.empty(), out, cfg, animate)
    st.markdown(f'<div class="k-label">Why this choice</div><div class="k-reason">{esc(reason_sentence(out, cfg))}</div>',
                unsafe_allow_html=True)
    if r["missing"]:
        st.markdown('<div class="k-banner">Part of this run was not recorded, so replay shows it as missing</div>',
                    unsafe_allow_html=True)

    tokens = sum(a["input_tokens"] + a["output_tokens"] for a in out["attempts"])
    latency = sum(a["latency_s"] or 0 for a in out["attempts"])
    compute = out["total_compute_flops"]
    badge = '<span class="k-badge yes">YES</span>' if out["verified"] else '<span class="k-badge no">NO</span>'
    st.markdown(f"""<div class="k-metrics">
      <div class="k-card k-metric"><div class="k-label">Tokens</div><div class="v">{tokens:,}</div><div class="u">input + output, from the API</div></div>
      <div class="k-card k-metric"><div class="k-label">Latency</div><div class="v">{latency:.2f}<span class="u"> s</span></div><div class="u">{len(out['attempts'])} call(s)</div></div>
      <div class="k-card k-metric"><div class="k-label">Est. compute</div><div class="v">{compute / 1e12:.2f}<span class="u"> TFLOPs</span></div><div class="u">all attempts</div></div>
      <div class="k-card k-metric"><div class="k-label">Verified</div><div class="v">{badge}</div><div class="u">{len(out['fallbacks'])} fallback · {out['escalations']} escalation</div></div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown('<div class="k-label">Compute for this task</div>', unsafe_allow_html=True)
        big = largest["compute"]
        if big:
            top = max(big, compute)
            saved = 1 - compute / big
            save_txt = f"{saved:.0%} less compute than all-to-largest" if saved >= 0 else \
                f"{-saved:.0%} more compute than all-to-largest"
            st.markdown(f"""<div class="k-card k-bars">
              <div class="row"><div class="lab">All to largest</div><div class="track"><div class="fill big" style="width:{100 * big / top:.1f}%"></div></div><div class="val">{big / 1e12:.2f} TF</div></div>
              <div class="row"><div class="lab">KAYL</div><div class="track"><div class="fill kayl" style="width:{100 * compute / top:.1f}%"></div></div><div class="val">{compute / 1e12:.2f} TF</div></div>
              <div class="k-save">{esc(save_txt)}</div>
              <div class="u" style="color:var(--muted);font-size:.75rem;margin-top:.3rem">All to largest: {esc(largest['source'])}</div>
            </div>""", unsafe_allow_html=True)
    with c2:
        if out["task_type"] == "intent":
            # طلبات الخدمة: النية المتوقعة فقط (مع اسم عربي)، بدون رد النموذج الحر. الرد المسجل لا يُعدَّل.
            code = normalize_label(out["final"]) if out["final"] else None
            st.markdown('<div class="k-label">Predicted intent</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="k-card"><div class="k-answer" dir="rtl" style="unicode-bidi:isolate">'
                        f'{esc(INTENT_AR.get(code, "—"))}</div>'
                        f'<div class="k-label" style="margin-top:.6rem">{esc(code or "—")}</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="k-label">Answer</div>', unsafe_allow_html=True)
            answer = ar_num(out["answer_text"]) if out["answer_text"] else "—"
            final = ar_num(out["final"]) if out["final"] and out["task_type"] == "math" else (out["final"] or "—")
            st.markdown(f'<div class="k-card"><div class="k-answer" dir="rtl" style="unicode-bidi:isolate">{esc(answer)}</div>'
                        f'<div class="k-label" style="margin-top:.6rem">FINAL: <bdi dir="rtl">{esc(final)}</bdi></div></div>',
                        unsafe_allow_html=True)

    with st.expander("Every decision, with its reason"):
        for d in out["decisions"]:
            st.markdown(f'<div class="k-label">{esc(d["step"])}</div>'
                        f'<div style="margin-bottom:.5rem">{esc(display_reason(out["task_type"], d["reason"]))}</div>',
                        unsafe_allow_html=True)


def results_tab():
    s = final_summary()
    if not s:
        st.markdown('<div class="k-card">No final test results found.</div>', unsafe_allow_html=True)
        return
    summ = s["summary"]
    kayl, big = summ["D KAYL"][-1], summ["A 120b-medium (largest)"][-1]
    saved = 1 - kayl["compute_per_success"] / big["compute_per_success"]
    st.markdown(f"""<div class="k-teal"><div class="k-label" style="color:#E6FFF6">Held-out test · 90 questions · one run · temperature 0</div>
      <div class="grid" style="margin-top:.6rem">
        <div><div class="big">{saved:.0%}</div>less compute per correct answer than all-to-largest</div>
        <div><div class="big">{kayl['correct']}/{kayl['n']}</div>KAYL correct (all-to-largest: {big['correct']}/{big['n']})</div>
        <div><div class="big">1 / 3</div>task types where KAYL met its own accuracy floor (missed math and intent by 1–2 questions)</div>
      </div></div>""", unsafe_allow_html=True)

    rows = []
    for exp, rs in summ.items():
        a = rs[-1]
        per = [x for x in rs if x["type"] != "all"]
        comp = "n/a" if a["compute_per_success"] is None else f"{a['compute_per_success'] / 1e12:.2f}"
        cost = "n/a" if a["cost_per_success"] is None else f"${a['cost_per_success']:.6f}"
        cls = ' class="kayl"' if exp == "D KAYL" else ""
        rows.append(f"<tr{cls}><td>{esc(exp)}</td><td>{a['correct']}/{a['n']}</td><td>{comp}</td><td>{cost}</td>"
                    f"<td>{a['avg_latency_s']:.2f}</td><td>{sum(x['meets_floors'] for x in per)}/3</td>"
                    f"<td>{' · '.join(str(x['correct']) for x in per)}</td></tr>")
    st.markdown(f"""<div class="k-card"><table class="k-table">
      <tr><th>Experiment</th><th>Correct</th><th>Compute / success (TFLOPs, est.)</th><th>Cost / success</th>
      <th>Avg latency (s)</th><th>Floors met</th><th>Math · Knowledge · Intent (/30)</th></tr>{''.join(rows)}
    </table></div>""", unsafe_allow_html=True)
    st.markdown('<div class="k-label" style="margin-top:.8rem">Notes</div>'
                '<div style="color:var(--muted);font-size:.85rem;line-height:1.6">'
                "KAYL's choices were made on calibration data only and were not changed after the test. "
                "120b-low (the “minus model key” ablation) met all floors with about 31% less compute than all-to-largest; "
                "this is an observation on the test set, not a choice. The verifier checks answer format, not correctness. "
                "E is a simple difficulty proxy (question length), not RouteLLM. Final-test gpt-oss calls went through "
                "OpenRouter pinned to the Groq provider. Full details: RESULTS.md.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- page

def set_example_text(task: dict) -> str:
    """نص المثال كما يظهر في صندوق الإدخال (مع الخيارات للمعرفة)."""
    text = task["question"]
    if task["type"] == "knowledge":
        text += "\n" + "\n".join(f"{'ABCDE'[i]}) {c}" for i, c in enumerate(task["choices"]))
    return text.strip()


KIND_OF = {"math": "Math", "knowledge": "Knowledge", "intent": "Intent"}


def set_example(key: str):
    """زر المثال: يملأ النص، ويجعل قائمة نوع المهمة تطابق نوع المثال."""
    _, tid = EXAMPLES[key]
    task = calibration_tasks()[tid]
    st.session_state["prompt"] = set_example_text(task)
    st.session_state["example"] = tid
    st.session_state["kind"] = KIND_OF[task["type"]]


def sync_kind():
    """عند تعديل النص: قائمة نوع المهمة تطابق النوع الذي يكتشفه كَيْل من النص."""
    text = st.session_state.get("prompt", "")
    if text.strip():
        task, _ = build_task(text, "Auto")
        st.session_state["kind"] = KIND_OF[infer_type(task)[0]]


def main():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.markdown("""<div class="k-hero"><div><div class="t">KAYL — <span>Intelligence</span> Without Waste</div>
      <div class="s">The right measure of AI for every Arabic task: the smallest model and the least thinking that still meets the accuracy floor.</div></div>
      <div class="ar">كَيْل</div></div>""", unsafe_allow_html=True)

    demo, results = st.tabs(["Demo", "Results"])
    with demo:
        replay = st.toggle("Replay mode — recorded live runs, no API calls", value=True)
        cfg, task, label, simulated = None, None, None, False
        if replay:
            names = {k: title for k, title, *_ in REPLAYS}
            key = st.radio("Recorded run", list(names), format_func=names.get, horizontal=True)
            _, title, source, tid, label, ov = next(r for r in REPLAYS if r[0] == key)
            try:
                task = (calibration_tasks() if source == "calibration" else drill_tasks())[tid]
            except Exception as e:  # بيانات التمرين تحتاج نسخة Hugging Face المحلية
                st.markdown(f'<div class="k-card">This recording needs the local dataset cache ({esc(e)}).</div>',
                            unsafe_allow_html=True)
                task = None
            cfg, simulated = case_config(ov["route"], ov.get("unavailable")), bool(ov.get("unavailable"))
            if task:
                st.markdown(f'<div class="k-card k-ar" dir="rtl" style="margin:.4rem 0 .8rem;unicode-bidi:isolate">'
                            f'{esc(ar_num(task["question"]))}</div>',
                            unsafe_allow_html=True)
            go = st.button("▶  Play recording", disabled=task is None)
        else:
            st.markdown('<div class="k-banner">Live mode calls the models: gpt-oss via OpenRouter pinned to Groq, ALLaM on Groq</div>',
                        unsafe_allow_html=True)
            b1, b2, b3, _ = st.columns([1, 1, 1.4, 2])
            b1.button("Math", on_click=set_example, args=("math",), type="secondary")
            b2.button("Knowledge", on_click=set_example, args=("knowledge",), type="secondary")
            b3.button("Saudi-dialect request", on_click=set_example, args=("intent",), type="secondary")
            text = st.text_area("Arabic task", key="prompt", height=120, on_change=sync_kind,
                                placeholder="اكتب سؤالًا بالعربية… (للاختيار من متعدد: سطر لكل خيار مثل A) …)")
            if (text or "").strip():  # نسخة للعرض فقط بأرقام عربية مشرقية؛ النموذج يستلم النص كما كُتب
                lines = []
                for line in text.strip().splitlines():
                    m = OPTION_RE.match(line)  # حرف الخيار (A) …) معزول حتى لا تنقلب القوس من اليمين لليسار
                    lines.append(f'<bdi dir="ltr">{esc(m.group(1))})</bdi> {esc(ar_num(m.group(2)))}' if m
                                 else esc(ar_num(line)))
                st.markdown(f'<div class="k-card k-ar" dir="rtl" style="margin:.2rem 0 .8rem;unicode-bidi:isolate">'
                            f'{"<br>".join(lines)}</div>', unsafe_allow_html=True)
            st.session_state.setdefault("kind", "Auto")
            kind = st.selectbox("Task type", ["Auto", "Math", "Knowledge", "Intent"], key="kind")
            go = st.button("Run KAYL", disabled=not (text or "").strip())
            if go:
                tid = st.session_state.get("example")
                example = calibration_tasks().get(tid)
                if example and set_example_text(example) == text.strip():
                    task = example
                else:
                    task, _ = build_task(text, kind)  # الوكيل نفسه يكتب كيف عرف نوع المهمة (estimate)
                cfg, label = case_config(), "Live run"

        if go and task:
            if label:
                st.markdown(f'<div class="k-banner {"sim" if simulated else ""}">{esc(label)}</div>',
                            unsafe_allow_html=True)
            with st.spinner("Running KAYL…"):
                st.session_state["last"] = (run_case(task, cfg, replay), cfg, label, simulated)
            results_view(st.session_state["last"][0], cfg, animate=True)
        elif st.session_state.get("last"):
            r, cfg_last, label_last, sim_last = st.session_state["last"]
            st.markdown(f'<div class="k-banner {"sim" if sim_last else ""}">{esc(label_last)}</div>',
                        unsafe_allow_html=True)
            results_view(r, cfg_last, animate=False)

    with results:
        results_tab()

    st.markdown(f'<div class="k-foot"><div>{esc(FOOTNOTE)}</div><div>KAYL — research prototype</div></div>',
                unsafe_allow_html=True)


main()
