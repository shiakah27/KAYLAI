# -*- coding: utf-8 -*-
"""
كَيْل — واجهة العرض (Streamlit).

ماذا تعرض:
  - مسار القرار خطوة بخطوة: فهم المهمة -> اختيار النموذج والتفكير -> الإجابة -> التحقق -> التصعيد عند الحاجة.
  - بطاقة السبب (جملة واحدة من أرقام المعايرة)، وبطاقات الأرقام، وأشرطة الحوسبة مقابل "كل شيء للأكبر".
  - تبويب النتائج: جدول الاختبار النهائي من results/final_test_summary.json.
  - وضع الإعادة (Replay): يعرض تشغيلات حقيقية مسجلة من الكاش، بدون أي استدعاء API.

قواعد العرض:
  - الأرقام داخل النص العربي تُعرض بالأرقام العربية المشرقية (٠١٢...) للعرض فقط؛
    البيانات وما يُرسل للنموذج لا يتغيران.
  - في مهام النية نعرض النية المتوقعة فقط (مع اسم عربي)، ولا نعرض رد النموذج الحر.
    ردود النموذج المسجلة لا تُعدَّل أبدًا.

التشغيل:
    streamlit run app.py
    (للعرض المباشر لتسجيل: http://localhost:8501/?case=intent&autoplay=1)
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

from kayl.agent import run_agent
from kayl.client import call_model
from kayl.config import accuracy_floor, load_config
from kayl.grading import normalize_label
from kayl.profile import load_profile

st.set_page_config(page_title="KAYL — Intelligence Without Waste", page_icon="◆", layout="wide",
                   initial_sidebar_state="collapsed")

LARGEST = ("openai/gpt-oss-120b", "medium")  # "All to largest" = baseline A
PRETTY_MODEL = {"allam-2-7b": "ALLaM-2-7B", "openai/gpt-oss-20b": "gpt-oss-20b", "openai/gpt-oss-120b": "gpt-oss-120b"}
TYPE_EN = {"math": "math", "knowledge": "multiple-choice knowledge", "intent": "service-request intent"}
TYPE_AR = {"math": "مسألة رياضيات", "knowledge": "سؤال معرفة (اختيار من متعدد)", "intent": "طلب خدمة (تصنيف النية)"}
TYPE_AR_FOR = {"math": "لمسائل الرياضيات", "knowledge": "لأسئلة المعرفة", "intent": "لطلبات الخدمة"}
EFFORT_AR = {"low": "منخفض", "medium": "متوسط", "high": "عالٍ"}
INTENT_AR = {
    "renew_national_id": "تجديد الهوية الوطنية",
    "renew_passport": "تجديد جواز السفر",
    "traffic_violation_inquiry": "الاستعلام عن المخالفات المرورية",
    "vehicle_ownership_transfer": "نقل ملكية مركبة",
    "book_medical_appointment": "حجز موعد طبي",
    "report_utility_outage": "الإبلاغ عن انقطاع الكهرباء أو الماء",
    "social_support_inquiry": "الاستفسار عن الدعم الحكومي",
    "file_complaint": "تقديم شكوى",
    "other": "خارج الخدمات المتاحة",
}
EXAMPLES = {  # من بيانات المعايرة (وليس الاختبار)
    "math": ("رياضيات · Math", "gsm8k-ar-1107"),
    "knowledge": ("معرفة · Knowledge", "arabicmmlu-11638"),
    "intent": ("طلب باللهجة السعودية · Saudi-dialect request", "intent-6"),
}
RECORDED = ("إعادة لتشغيل حقيقي مسجّل", "Replay of a recorded live run")
REPLAYS = [  # (key, Arabic title, English title, source, task id, simulated?, overrides)
    ("math", "مسألة رياضيات", "Math question", "calibration", "gsm8k-ar-1107", False, {"route": "groq"}),
    ("knowledge", "سؤال معرفة", "Knowledge question", "calibration", "arabicmmlu-11638", False, {"route": "groq"}),
    ("intent", "طلب باللهجة السعودية", "Saudi-dialect request", "calibration", "intent-6", False, {"route": "groq"}),
    ("escalation", "حالة تصعيد", "Escalation case", "drill", "gsm8k-ar-707", False, {"route": "groq"}),
    ("outage", "انقطاع مُحاكى", "Outage — SIMULATED", "drill", "gsm8k-ar-1242", True,
     {"route": "openrouter", "unavailable": ["openai/gpt-oss-20b"]}),
]
PIPELINE = [("understand", "فهم المهمة", "Understand task"),
            ("choose", "اختيار النموذج ومستوى التفكير", "Choose model & effort"),
            ("answer", "الإجابة", "Answer"),
            ("verify", "التحقق", "Verify"),
            ("escalate", "التصعيد عند الحاجة فقط", "Escalate only if needed")]
STEP_OF = {"estimate": "understand", "choose": "choose", "fallback": "choose", "solve": "answer",
           "verify": "verify", "escalate": "escalate"}
VERIFY_AR = [  # (جزء من سبب الفشل بالإنجليزية، النص العربي)
    ("did not succeed", "فشل الاستدعاء"),
    ("cut off", "انقطع الرد عند حد التوكنز"),
    ("no FINAL line", "لا يوجد سطر FINAL في الرد"),
    ("more than one number", "سطر FINAL فيه أكثر من رقم"),
    ("not a valid answer", "الجواب ليس صالحًا لنوع المهمة"),
    ("not in Arabic", "سطر الجواب غير موجود أو ليس بالعربية"),
]
FOOTNOTE_EN = "Compute is an estimate (2 × active parameters × tokens). Energy not measured."
FOOTNOTE_AR = "الحوسبة تقدير (٢ × المعاملات النشطة × التوكنز). الطاقة لم تُقَس."
DIGITS_NOTE = ("الأرقام داخل النص العربي معروضة بالأرقام العربية المشرقية للعرض فقط",
               "Digits shown in Arabic-Indic form for display.")

# ---------------------------------------------------------------- theme

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&family=Noto+Kufi+Arabic:wght@400;600;700&display=swap');
:root{
  --bg:#041A23; --glow:#0F4846; --lime:#C8F63E; --mint:#00CF90; --teal1:#009294; --teal2:#007C92;
  --text:#FFFFFF; --muted:#A9BDB4; --line:rgba(255,255,255,0.12); --glass:rgba(8,40,48,0.42); --bad:#E8735A;
}
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="collapsedControl"] {display:none !important;}
.stApp{color:var(--text); font-family:'Noto Kufi Arabic','Inter',sans-serif;
  background:
    radial-gradient(55% 45% at 30% 6%, rgba(15,72,70,.95) 0%, rgba(15,72,70,0) 70%),
    radial-gradient(35% 30% at 82% 14%, rgba(0,207,144,.20) 0%, rgba(0,207,144,0) 70%),
    radial-gradient(40% 35% at 10% 60%, rgba(0,146,148,.12) 0%, rgba(0,146,148,0) 70%),
    var(--bg);}
.block-container{padding-top:1.6rem; padding-bottom:3rem; max-width:1200px; direction:rtl;}
p, label, .stMarkdown{color:var(--text);}

/* bilingual labels: Arabic main, small English underneath */
.k-l{margin:0 0 .45rem;}
.k-l .ar{display:block; font-family:'Noto Kufi Arabic'; font-weight:600; font-size:.95rem; color:var(--text);}
.k-l .en{display:block; font-family:'JetBrains Mono'; font-size:.64rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); margin-top:.1rem;}
.k-title .ar{font-size:1.15rem;}
bdi.ltr{direction:ltr; unicode-bidi:isolate; font-family:'JetBrains Mono'; font-size:.92em;}

/* hero */
.k-hero{position:relative; overflow:hidden; border-radius:30px; padding:3.1rem 2.8rem 2.7rem; margin-bottom:1.6rem;
  border:1px solid var(--line); direction:ltr; min-height:300px;
  background:radial-gradient(70% 120% at 18% 20%, rgba(15,72,70,.9), rgba(15,72,70,0) 65%),
             radial-gradient(45% 80% at 85% 75%, rgba(0,207,144,.22), rgba(0,207,144,0) 70%), rgba(4,26,35,.55);
  animation:kFade .8s ease both;}
.k-hero .copy{position:relative; z-index:2; max-width:62%;}
.k-hero .eyebrow{font-family:'JetBrains Mono'; font-size:.72rem; letter-spacing:.22em; text-transform:uppercase; color:var(--lime);}
.k-hero .words{display:flex; align-items:baseline; gap:1.4rem; flex-wrap:wrap; margin:.4rem 0 .6rem;}
.k-grad{background:linear-gradient(100deg,#FFFFFF 0%,#00CF90 55%,#C8F63E 100%); -webkit-background-clip:text; background-clip:text;
  color:transparent; filter:drop-shadow(0 0 26px rgba(200,246,62,.28));}
.k-hero .kayl{font-family:'Space Grotesk'; font-weight:700; font-size:5.4rem; line-height:1; letter-spacing:.14em; text-transform:uppercase;}
.k-hero .kayl-ar{font-family:'Noto Kufi Arabic'; font-weight:700; font-size:4.3rem; line-height:1.2; direction:rtl;}
.k-hero .tag{font-family:'Space Grotesk'; text-transform:uppercase; letter-spacing:.2em; font-size:1rem; color:var(--text);}
.k-hero .tag-ar{font-family:'Noto Kufi Arabic'; direction:rtl; font-size:1.2rem; color:var(--mint); margin-top:.2rem;}
.k-hero .sub{color:var(--muted); font-family:'Inter'; font-size:.92rem; margin-top:.9rem; max-width:560px;}
.k-shape{position:absolute; right:-30px; top:50%; width:460px; height:380px; transform:translateY(-50%); z-index:1; pointer-events:none;
  -webkit-mask-image:radial-gradient(75% 75% at 60% 50%, #000 55%, transparent 100%); mask-image:radial-gradient(75% 75% at 60% 50%, #000 55%, transparent 100%);}
.k-shape i{position:absolute; border-radius:42px; filter:blur(1.2px); box-shadow:0 0 60px rgba(0,207,144,.18); animation:kFloat 9s ease-in-out infinite alternate;}
.k-shape i:nth-child(1){width:320px; height:190px; right:90px; top:10px; background:linear-gradient(135deg,#0F4846,#009294); transform:rotate(-16deg); opacity:.9;}
.k-shape i:nth-child(2){width:300px; height:170px; right:55px; top:95px; background:linear-gradient(135deg,#007C92,#00CF90); transform:rotate(-9deg); opacity:.8; animation-delay:-2s;}
.k-shape i:nth-child(3){width:260px; height:150px; right:25px; top:180px; background:linear-gradient(135deg,#00CF90,#C8F63E); transform:rotate(-3deg); opacity:.75; animation-delay:-4s;}
.k-shape i:nth-child(4){width:150px; height:150px; right:150px; top:230px; border-radius:50%; background:radial-gradient(circle,#C8F63E 0%,rgba(200,246,62,0) 70%); filter:blur(18px); opacity:.55;}
@keyframes kFloat{from{translate:0 0} to{translate:-10px 8px}}

/* glass cards */
.k-card{background:var(--glass); -webkit-backdrop-filter:blur(14px); backdrop-filter:blur(14px); border:1px solid var(--line);
  border-radius:18px; padding:1rem 1.15rem;}
.k-ar{font-family:'Noto Kufi Arabic'; direction:rtl; text-align:right; unicode-bidi:isolate;}
.k-question{font-size:1.12rem; line-height:1.9; margin:.3rem 0 .5rem;}
.k-options{margin-top:.4rem; color:var(--muted);}
.k-note{color:var(--muted); font-size:.72rem; margin:.2rem 0 1rem;}
.k-note .en{font-family:'JetBrains Mono'; font-size:.64rem; letter-spacing:.06em; margin-right:.5rem;}
.k-banner{border:1px solid var(--line); border-radius:999px; padding:.4rem 1.1rem; display:inline-flex; gap:.7rem; align-items:center;
  font-size:.85rem; color:var(--lime); margin:.3rem 0 1rem; background:var(--glass); backdrop-filter:blur(10px); animation:kFade .4s ease both;}
.k-banner .en{font-family:'JetBrains Mono'; font-size:.7rem; letter-spacing:.06em; color:var(--muted);}
.k-banner.sim{color:#041A23; background:var(--lime); border-color:var(--lime);}
.k-banner.sim .en{color:#0F4846;}

/* pipeline */
.k-pipe{display:grid; grid-template-columns:repeat(5,1fr); gap:.65rem; margin:.4rem 0 1.1rem;}
.k-step{background:var(--glass); backdrop-filter:blur(14px); border:1px solid var(--line); border-radius:16px; padding:.85rem .9rem; min-height:140px; opacity:.32;}
.k-step .n{font-family:'JetBrains Mono'; font-size:.68rem; color:var(--muted); letter-spacing:.12em;}
.k-step .h .ar{display:block; font-family:'Noto Kufi Arabic'; font-weight:700; font-size:.92rem; margin-top:.2rem;}
.k-step .h .en{display:block; font-family:'Space Grotesk'; text-transform:uppercase; letter-spacing:.12em; font-size:.62rem; color:var(--muted); margin-bottom:.5rem;}
.k-step .r{font-size:.8rem; line-height:1.55;}
.k-step .r .en{display:block; font-family:'Inter'; font-size:.7rem; color:var(--muted); direction:ltr; text-align:right; margin-bottom:.35rem;}
.k-step.active,.k-step.done,.k-step.fail,.k-step.skip{animation:kFade .5s ease both;}
.k-step.active{opacity:1; border-color:var(--lime); box-shadow:0 0 0 1px var(--lime), 0 0 30px rgba(200,246,62,.2);}
.k-step.active .h .ar{color:var(--lime);}
.k-step.done{opacity:1; border-color:rgba(0,207,144,.55);}
.k-step.done .h .ar{color:var(--mint);}
.k-step.fail{opacity:1; border-color:var(--bad);}
.k-step.fail .h .ar{color:var(--bad);}
.k-step.skip{opacity:.75;}
@keyframes kFade{from{opacity:0; transform:translateY(10px)} to{opacity:1; transform:none}}

.k-reason{border-right:3px solid var(--lime); padding:1rem 1.2rem; background:var(--glass); backdrop-filter:blur(14px);
  border-radius:16px 0 0 16px; margin:.4rem 0 1.1rem; animation:kFade .6s ease both;}
.k-reason .ar{font-size:1.02rem; line-height:1.95;}
.k-reason .en{font-family:'Inter'; font-size:.8rem; color:var(--muted); direction:ltr; text-align:left; margin-top:.55rem; line-height:1.55;}
.k-metrics{display:grid; grid-template-columns:repeat(4,1fr); gap:.65rem; margin-bottom:1.1rem;}
.k-metric{animation:kFade .6s ease both;}
.k-metric .v{font-family:'JetBrains Mono'; font-size:1.6rem; font-weight:600; direction:ltr; text-align:right;}
.k-metric .u{font-size:.74rem; color:var(--muted); margin-top:.15rem;}
.k-badge{display:inline-block; border-radius:999px; padding:.15rem .8rem; font-size:.85rem; font-weight:600;}
.k-badge.yes{background:rgba(0,207,144,.14); color:var(--mint); border:1px solid var(--mint);}
.k-badge.no{background:rgba(232,115,90,.12); color:var(--bad); border:1px solid var(--bad);}

/* compute bars */
.k-bars .row{display:grid; grid-template-columns:140px 1fr 90px; align-items:center; gap:.8rem; margin:.55rem 0;}
.k-bars .track{background:rgba(255,255,255,.06); border-radius:999px; height:18px; overflow:hidden;}
.k-bars .fill{height:100%; border-radius:999px; animation:kGrow 1.2s cubic-bezier(.2,.8,.2,1) both;}
.k-bars .fill.big{background:#5E6E6A;}
.k-bars .fill.kayl{background:linear-gradient(270deg,var(--lime),var(--mint)); box-shadow:0 0 18px rgba(200,246,62,.3); animation-delay:.25s;}
.k-bars .val{font-family:'JetBrains Mono'; font-size:.85rem; direction:ltr; text-align:left;}
@keyframes kGrow{from{width:0}}
.k-save .ar{font-family:'Noto Kufi Arabic'; font-weight:700; color:var(--lime); font-size:1.05rem;}
.k-save .en{font-family:'Space Grotesk'; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); font-size:.7rem;}
.k-src{color:var(--muted); font-size:.72rem; margin-top:.45rem;}
.k-answer{font-family:'Noto Kufi Arabic'; direction:rtl; text-align:right; font-size:1.12rem; line-height:1.9;}
.k-intent{font-family:'Noto Kufi Arabic'; font-size:1.45rem; font-weight:700; color:var(--mint);}
.k-code{font-family:'JetBrains Mono'; font-size:.78rem; color:var(--muted); direction:ltr; unicode-bidi:isolate;}

/* results */
.k-teal{background:linear-gradient(120deg,var(--teal1),var(--teal2)); border-radius:22px; padding:1.3rem 1.5rem; margin-bottom:1.1rem; animation:kFade .6s ease both;}
.k-teal .grid{display:grid; grid-template-columns:repeat(3,1fr); gap:1.2rem; margin-top:.7rem;}
.k-teal .big{font-family:'Noto Kufi Arabic'; font-size:2rem; font-weight:700;}
.k-teal .t .en{display:block; font-family:'Inter'; font-size:.74rem; color:#DDF7EE; direction:ltr; text-align:right;}
.k-table{width:100%; border-collapse:collapse; font-size:.84rem; direction:ltr;}
.k-table th{text-align:left; border-bottom:1px solid var(--line); padding:.55rem .5rem; vertical-align:bottom;}
.k-table th .ar{display:block; font-family:'Noto Kufi Arabic'; font-size:.78rem; color:var(--text); direction:rtl; text-align:left;}
.k-table th .en{display:block; font-family:'JetBrains Mono'; font-size:.62rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);}
.k-table td{border-bottom:1px solid var(--line); padding:.55rem .5rem; font-family:'JetBrains Mono';}
.k-table td:first-child{font-family:'Inter'; font-weight:500;}
.k-table tr.kayl td{color:var(--lime);}
.k-foot{color:var(--muted); font-size:.78rem; margin-top:1.6rem; border-top:1px solid var(--line); padding-top:.9rem;
  display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap;}
.k-foot .en{direction:ltr; font-family:'Inter';}

/* Streamlit widgets */
.stButton > button{border-radius:999px !important; background:var(--lime) !important; color:#041A23 !important; border:none !important;
  font-family:'Noto Kufi Arabic','JetBrains Mono' !important; font-weight:600 !important; padding:.45rem 1.3rem !important;
  transition:transform .15s ease, box-shadow .15s ease;}
.stButton > button:hover{transform:translateY(-1px); box-shadow:0 0 22px rgba(200,246,62,.3);}
.stButton > button[kind="secondary"]{background:transparent !important; color:var(--lime) !important; border:1px solid var(--lime) !important;}
textarea{direction:rtl; text-align:right; font-family:'Noto Kufi Arabic' !important; background:var(--glass) !important;
  color:var(--text) !important; border:1px solid var(--line) !important; border-radius:16px !important;}
[data-baseweb="select"] > div{background:var(--glass) !important; border-color:var(--line) !important; color:var(--text) !important;}
.stTabs [data-baseweb="tab-list"]{gap:.5rem; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"] p{font-family:'Noto Kufi Arabic'; color:var(--muted);}
.stTabs [aria-selected="true"] p{color:var(--lime) !important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--lime) !important;}
[data-testid="stWidgetLabel"] p{color:var(--muted) !important; font-family:'Noto Kufi Arabic'; font-size:.82rem;}
[data-testid="stExpander"] details{background:var(--glass); border:1px solid var(--line) !important; border-radius:16px;}
@media (max-width: 900px){ .k-pipe{grid-template-columns:1fr 1fr;} .k-metrics{grid-template-columns:1fr 1fr;}
  .k-teal .grid{grid-template-columns:1fr;} .k-hero .copy{max-width:100%;} .k-shape{opacity:.35;} .k-hero .kayl{font-size:3.4rem;} }
"""


# ---------------------------------------------------------------- display helpers

AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def ar_num(text) -> str:
    """للعرض فقط: أرقام عربية مشرقية، وفاصلة عشرية ٫، وفاصل آلاف ٬، ونسبة ٪.
    لا نلمس الأرقام الملتصقة بحروف لاتينية (مثل gpt-oss-20b)."""
    def conv(m):
        s = m.group(0)
        s = re.sub(r"(?<=\d)\.(?=\d)", "٫", s)
        s = re.sub(r"(?<=\d),(?=\d{3})", "٬", s)
        return s.translate(AR_DIGITS)
    t = re.sub(r"(?<![A-Za-z\-\d])\d[\d.,]*(?![\dA-Za-z])", conv, str(text))
    return t.replace("%", "٪")


def esc(x) -> str:
    return html.escape(str(x))


def ltr(x) -> str:
    """مقطع لاتيني داخل نص عربي (اسم نموذج، TFLOPs) يُعزل حتى لا يختلط الاتجاه."""
    return f'<bdi class="ltr">{esc(x)}</bdi>'


def label(ar: str, en: str, cls: str = "") -> str:
    return f'<div class="k-l {cls}"><span class="ar">{esc(ar)}</span><span class="en">{esc(en)}</span></div>'


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


def example_text(task: dict) -> str:
    """نص المثال كما يظهر في صندوق الإدخال (مع الخيارات للمعرفة)."""
    text = task["question"]
    if task["type"] == "knowledge":
        text += "\n" + "\n".join(f"{'ABCDE'[i]}) {c}" for i, c in enumerate(task["choices"]))
    return text.strip()


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
        largest = {"compute": avg, "ar": "متوسط المعايرة لهذا النوع (لم يُسجَّل لهذه المهمة)",
                   "en": "calibration average for this task type (not recorded for this task)"}
    else:
        how = ("مسجّل", "recorded") if big.cached else ("استدعاء مباشر", "live call")
        largest = {"compute": big.compute_flops, "ar": f"نفس المهمة على gpt-oss-120b بتفكير متوسط ({how[0]})",
                   "en": f"same task, gpt-oss-120b · medium ({how[1]})"}
    missing = any(a["verify_reason"] and "not recorded" in a["verify_reason"] for a in out["attempts"])
    return {"out": out, "largest": largest, "missing": missing, "task": task}


# ---------------------------------------------------------------- texts

def conf_parts(cfg: dict, name: str) -> tuple[str, str | None]:
    c = {c["name"]: c for c in cfg["sweep"]}.get(name)
    if c:
        return c["model"], c["effort"]
    m = re.match(r"(allam|20b|120b)(?:-(low|medium|high))?", name or "")
    model = {"allam": "allam-2-7b", "20b": "openai/gpt-oss-20b", "120b": "openai/gpt-oss-120b"}.get(m.group(1)) if m else name
    return model, (m.group(2) if m else None)


def conf_ar(cfg: dict, name: str) -> str:
    model, effort = conf_parts(cfg, name)
    return ltr(PRETTY_MODEL.get(model, model)) + (f" بتفكير {EFFORT_AR[effort]}" if effort else " بدون تحكم بالتفكير")


def conf_en(cfg: dict, name: str) -> str:
    model, effort = conf_parts(cfg, name)
    return f"{PRETTY_MODEL.get(model, model)} · {effort + ' effort' if effort else 'no effort control'}"


def final_ar(task_type: str, final: str | None) -> str:
    """قيمة FINAL كما نعرضها: رقم بأرقام مشرقية، أو حرف، أو اسم النية بالعربية."""
    if not final:
        return "—"
    if task_type == "intent":
        code = normalize_label(final)
        return esc(INTENT_AR.get(code, "نية غير معروفة")) + " " + f'<bdi class="k-code" dir="ltr">{esc(code)}</bdi>'
    if task_type == "math":
        return esc(ar_num(final))
    return ltr(final)


def verify_ar(reason: str) -> str:
    if reason.startswith("passed"):
        return "نجح: شكل الجواب صحيح والجواب بالعربية"
    return next((ar for key, ar in VERIFY_AR if key in reason), "فشل التحقق")


def step_text(d: dict, out: dict, cfg: dict, attempt_no: int) -> tuple[str, str, str]:
    """(النص العربي HTML، النص الإنجليزي، الحالة) لخطوة واحدة."""
    step, why, t = d["step"], d["reason"], out["task_type"]
    if step == "estimate":
        return esc(TYPE_AR[t]) + " · تم تحميل ملف المعايرة", f"{TYPE_EN[t].capitalize()} task · calibration profile loaded", "done"
    if step == "choose":
        first = out["fallbacks"][0]["from"] if out["fallbacks"] else out["attempts"][0]["config"]
        return conf_ar(cfg, first), conf_en(cfg, first), "done"
    if step == "fallback":
        to = out["fallbacks"][0]["to"]
        sim = "SIMULATED" in why
        return ("انتقال تلقائي إلى " + conf_ar(cfg, to) + (" · انقطاع مُحاكى" if sim else ""),
                f"Fallback → {conf_en(cfg, to)}" + (" · SIMULATED outage" if sim else ""), "fail")
    if step == "solve":
        a = out["attempts"][attempt_no]
        prefix = f"المحاولة {ar_num(attempt_no + 1)}: " if attempt_no else ""
        return (prefix + ltr("FINAL:") + " " + final_ar(t, a["final"]) + "<br>" +
                esc(ar_num(f"{a['input_tokens']} مدخلات / {a['output_tokens']} مخرجات")),
                f"{'Attempt ' + str(attempt_no + 1) + ': ' if attempt_no else ''}"
                f"{a['input_tokens']} in / {a['output_tokens']} out tokens", "done")
    if step == "verify":
        ok = why.startswith("passed")
        return esc(verify_ar(why)), ("Passed: valid format, Arabic answer" if ok else why.removeprefix("failed: ").capitalize()), \
            ("done" if ok else "fail")
    if step == "escalate":
        m = re.search(r"raise effort (\w+) → (\w+) on (\S+)", why)
        if m:
            ar = f"رفع مستوى التفكير من {EFFORT_AR[m.group(1)]} إلى {EFFORT_AR[m.group(2)]}"
            en = f"Raise effort {m.group(1)} → {m.group(2)}"
        else:
            m = re.search(r"bigger model (\S+)", why)
            ar = "الانتقال إلى نموذج أكبر " + ltr(m.group(1) if m else "")
            en = f"Move to the bigger model {m.group(1) if m else ''}"
        return esc(ar) if "<bdi" not in ar else ar, en, "active"
    return esc(why), why, "done"


def reason_texts(out: dict, cfg: dict) -> tuple[str, str]:
    """جملة السبب بالعربية (HTML) وبالإنجليزية، من أرقام المعايرة."""
    t, prof = out["task_type"], profile()
    first = out["fallbacks"][0]["from"] if out["fallbacks"] else out["attempts"][0]["config"]
    s = prof.get(t, {}).get(first)
    choose = next(d["reason"] for d in out["decisions"] if d["step"] == "choose")
    if s and choose.startswith("chose"):
        floor, _ = accuracy_floor(cfg, prof, t)
        tf = f"{s['compute_per_success'] / 1e12:.2f}"
        ar = (f"{esc(TYPE_AR_FOR[t])} اختار كَيْل {conf_ar(cfg, first)}، لأنه أجاب صحيحًا عن "
              f"{esc(ar_num(s['correct']))} من {esc(ar_num(s['n']))} سؤال معايرة، وهذا يحقق حد الدقة "
              f"{esc(ar_num(f'{floor:.0%}'))}، وبأقل حوسبة تقديرية لكل إجابة صحيحة ({ltr(tf + ' TFLOPs')}).")
        en = (f"For {TYPE_EN[t]} tasks, KAYL chose {conf_en(cfg, first)} because on {s['n']} calibration questions it "
              f"scored {s['correct']}/{s['n']}, meeting the {floor:.0%} accuracy floor, with the lowest estimated "
              f"compute per correct answer ({tf} TFLOPs).")
    else:
        ar, en = esc(choose), choose[0].upper() + choose[1:] + "."
    for fb in out["fallbacks"]:
        sim = "SIMULATED" in fb["reason"]
        ar += (f" ثم كان {ltr('gpt-oss-20b')} غير متاح{' (انقطاع مُحاكى)' if sim else ''}، فانتقل تلقائيًا إلى "
               f"{conf_ar(cfg, fb['to'])}، وهو الإعداد التالي الأقل حوسبة الذي يحقق الحدود.")
        en += (f" Then gpt-oss-20b was unavailable{' (SIMULATED outage)' if sim else ''}, so it switched to "
               f"{conf_en(cfg, fb['to'])}, the next cheapest configuration that meets the floors.")
    for d in out["decisions"]:
        if d["step"] == "escalate":
            step, because = d["reason"].split(": ", 1)[1].split(", because ")
            ar_step, _, _ = step_text(d, out, cfg, 0)
            ar += f" فشل التحقق ({esc(verify_ar('failed: ' + because))})، فقرر: {ar_step}."
            en += f" Verification failed ({because}), so it chose to {step}."
    return ar, en


# ---------------------------------------------------------------- rendering

def pipeline_html(states: dict, results: dict) -> str:
    cells = []
    for i, (key, ar, en) in enumerate(PIPELINE, 1):
        res = "".join(f'<div>{a}</div><span class="en">{esc(e)}</span>' for a, e in results.get(key, [])) or "&nbsp;"
        cells.append(f'<div class="k-step {states.get(key, "")}"><div class="n">{ar_num(f"0{i}")}</div>'
                     f'<div class="h"><span class="ar">{esc(ar)}</span><span class="en">{esc(en)}</span></div>'
                     f'<div class="r">{res}</div></div>')
    return f'<div class="k-pipe">{"".join(cells)}</div>'


def animate_pipeline(placeholder, out: dict, cfg: dict, animate: bool):
    states, results, attempt_no = {}, {}, -1
    for d in (d for d in out["decisions"] if d["step"] in STEP_OF):
        key = STEP_OF[d["step"]]
        attempt_no += d["step"] == "solve"
        ar, en, state = step_text(d, out, cfg, attempt_no)
        results.setdefault(key, []).append((ar, en))
        if animate:
            states[key] = "active"
            placeholder.markdown(pipeline_html(states, results), unsafe_allow_html=True)
            time.sleep(0.55)
        states[key] = "done" if state == "active" else state
    if "escalate" not in results:
        results["escalate"] = [("غير مطلوب — نجح التحقق", "Not needed — verification passed")] if out["verified"] \
            else [("لا توجد خطوة متبقية", "No step left")]
        states["escalate"] = "skip"
    placeholder.markdown(pipeline_html(states, results), unsafe_allow_html=True)


def question_card(task: dict):
    opts = ""
    if task.get("choices") and task.get("type") == "knowledge":
        opts = '<div class="k-options">' + "<br>".join(
            f"{ltr('ABCDE'[i] + ')')} {esc(ar_num(c))}" for i, c in enumerate(task["choices"])) + "</div>"
    st.markdown(f'<div class="k-card k-ar" style="margin:.4rem 0 .3rem">{label("السؤال", "Task")}'
                f'<div class="k-question">{esc(ar_num(task["question"]))}</div>{opts}</div>'
                f'<div class="k-note k-ar">{esc(DIGITS_NOTE[0])} <bdi class="en" dir="ltr">{esc(DIGITS_NOTE[1])}</bdi></div>',
                unsafe_allow_html=True)


def banner(simulated: bool, live: bool = False):
    if live:
        ar, en = "تشغيل مباشر", "Live run"
    else:
        ar, en = RECORDED
        if simulated:
            ar += " · انقطاع مُحاكى (gpt-oss-20b غير متاح)"
            en += " · SIMULATED outage (gpt-oss-20b marked unavailable)"
    st.markdown(f'<div class="k-banner {"sim" if simulated else ""}"><span>{esc(ar)}</span><bdi class="en" dir="ltr">{esc(en)}</bdi></div>',
                unsafe_allow_html=True)


def results_view(r: dict, cfg: dict, animate: bool):
    out, largest = r["out"], r["largest"]
    st.markdown(label("مسار القرار", "Decision pipeline", "k-title"), unsafe_allow_html=True)
    animate_pipeline(st.empty(), out, cfg, animate)

    ar, en = reason_texts(out, cfg)
    st.markdown(label("لماذا هذا الاختيار", "Why this choice", "k-title") +
                f'<div class="k-reason"><div class="ar k-ar">{ar}</div><div class="en">{esc(en)}</div></div>',
                unsafe_allow_html=True)
    if r["missing"]:
        st.markdown('<div class="k-banner"><span>جزء من هذا التشغيل غير مسجّل</span>'
                    '<span class="en">Part of this run was not recorded</span></div>', unsafe_allow_html=True)

    tokens = sum(a["input_tokens"] + a["output_tokens"] for a in out["attempts"])
    latency = sum(a["latency_s"] or 0 for a in out["attempts"])
    compute = out["total_compute_flops"]
    n_calls = len(out["attempts"])
    badge = ('<span class="k-badge yes">نعم · YES</span>' if out["verified"] else '<span class="k-badge no">لا · NO</span>')
    st.markdown(f"""<div class="k-metrics">
      <div class="k-card k-metric">{label("التوكنز", "Tokens")}<div class="v">{tokens:,}</div><div class="u">مدخلات + مخرجات، كما أرجعها مزوّد الخدمة</div></div>
      <div class="k-card k-metric">{label("زمن الاستجابة", "Latency")}<div class="v">{latency:.2f} s</div><div class="u">{esc(ar_num(n_calls))} استدعاء</div></div>
      <div class="k-card k-metric">{label("الحوسبة التقديرية", "Est. compute")}<div class="v">{compute / 1e12:.2f} TF</div><div class="u">مجموع كل المحاولات</div></div>
      <div class="k-card k-metric">{label("تم التحقق", "Verified")}<div class="v">{badge}</div><div class="u">انتقال: {esc(ar_num(len(out['fallbacks'])))} · تصعيد: {esc(ar_num(out['escalations']))}</div></div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown(label("الحوسبة لهذه المهمة", "Compute for this task", "k-title"), unsafe_allow_html=True)
        big = largest["compute"]
        if big:
            top = max(big, compute)
            pct = round(100 * abs(1 - compute / big))
            more = compute > big
            save_ar = f"{ar_num(pct)}٪ حوسبة {'أكثر' if more else 'أقل'} من «كل شيء للأكبر»"
            save_en = f"{pct}% {'more' if more else 'less'} compute than all-to-largest"
            st.markdown(f"""<div class="k-card k-bars">
              <div class="row">{label("كل شيء للأكبر", "All to largest")}<div class="track"><div class="fill big" style="width:{100 * big / top:.1f}%"></div></div><div class="val">{big / 1e12:.2f} TF</div></div>
              <div class="row">{label("كَيْل", "KAYL")}<div class="track"><div class="fill kayl" style="width:{100 * compute / top:.1f}%"></div></div><div class="val">{compute / 1e12:.2f} TF</div></div>
              <div class="k-save"><div class="ar">{esc(save_ar)}</div><div class="en" dir="ltr" style="text-align:right">{esc(save_en)}</div></div>
              <div class="k-src">«كل شيء للأكبر»: {esc(largest['ar'])}<div dir="ltr" style="text-align:right">All to largest: {esc(largest['en'])}</div></div>
            </div>""", unsafe_allow_html=True)
    with c2:
        t = out["task_type"]
        if t == "intent":
            # لمهام النية: النية المتوقعة فقط، بدون رد النموذج الحر
            st.markdown(label("النية المتوقعة", "Predicted intent", "k-title") +
                        f'<div class="k-card k-ar"><div class="k-intent">{final_ar(t, out["final"])}</div></div>',
                        unsafe_allow_html=True)
        else:
            answer = esc(ar_num(out["answer_text"])) if out["answer_text"] else "—"
            st.markdown(label("الإجابة", "Answer", "k-title") +
                        f'<div class="k-card"><div class="k-answer">{answer}</div>'
                        f'<div class="k-ar" style="margin-top:.6rem">{label("النتيجة النهائية", "FINAL")}'
                        f'<div class="k-answer">{final_ar(t, out["final"])}</div></div></div>'
                        f'<div class="k-note k-ar">{esc(DIGITS_NOTE[0])} <bdi class="en" dir="ltr">{esc(DIGITS_NOTE[1])}</bdi></div>',
                        unsafe_allow_html=True)

    with st.expander("كل القرارات وأسبابها · Every decision, with its reason"):
        for d in out["decisions"]:
            st.markdown(f'<div dir="ltr" style="text-align:left;margin-bottom:.55rem"><span class="k-code">{esc(d["step"])}</span>'
                        f'<div style="font-family:Inter;font-size:.86rem">{esc(d["reason"])}</div></div>', unsafe_allow_html=True)


def th(ar: str, en: str) -> str:
    return f'<th><span class="ar">{esc(ar)}</span><span class="en">{esc(en)}</span></th>'


def results_tab():
    s = final_summary()
    if not s:
        st.markdown('<div class="k-card">لا توجد نتائج · No final test results found.</div>', unsafe_allow_html=True)
        return
    summ = s["summary"]
    kayl, big = summ["D KAYL"][-1], summ["A 120b-medium (largest)"][-1]
    saved = round(100 * (1 - kayl["compute_per_success"] / big["compute_per_success"]))
    st.markdown(f"""<div class="k-teal k-ar">{label("الاختبار المحجوب · ٩٠ سؤالًا · تشغيل واحد · temperature 0", "Held-out test · 90 questions · one run · temperature 0")}
      <div class="grid">
        <div class="t"><div class="big">{ar_num(saved)}٪</div>حوسبة أقل لكل إجابة صحيحة من «كل شيء للأكبر»<span class="en">less compute per correct answer than all-to-largest</span></div>
        <div class="t"><div class="big">{ar_num(kayl['correct'])}/{ar_num(kayl['n'])}</div>إجابات كَيْل الصحيحة («كل شيء للأكبر»: {ar_num(big['correct'])}/{ar_num(big['n'])})<span class="en">KAYL correct (all-to-largest: {big['correct']}/{big['n']})</span></div>
        <div class="t"><div class="big">{ar_num(1)} من {ar_num(3)}</div>أنواع حقق فيها كَيْل حد الدقة (فاته الحد في الرياضيات والنوايا بسؤال أو سؤالين)<span class="en">task types where KAYL met its own floor (missed math and intent by 1–2 questions)</span></div>
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
    st.markdown(f"""<div class="k-card"><table class="k-table"><tr>
      {th("التجربة", "Experiment")}{th("صحيح", "Correct")}{th("الحوسبة لكل نجاح", "Compute / success (TFLOPs, est.)")}
      {th("التكلفة لكل نجاح", "Cost / success")}{th("متوسط الزمن", "Avg latency (s)")}{th("حدود محققة", "Floors met")}
      {th("رياضيات · معرفة · نوايا", "Math · Knowledge · Intent (/30)")}</tr>{''.join(rows)}</table></div>""",
                unsafe_allow_html=True)
    st.markdown('<div class="k-card" style="margin-top:.8rem">' + label("ملاحظات", "Notes") +
                '<div class="k-ar" style="font-size:.86rem;line-height:1.9">خيارات كَيْل حُددت من بيانات المعايرة فقط ولم تتغير بعد الاختبار. '
                f'{ltr("120b-low")} حقق كل الحدود بحوسبة أقل بحوالي ٣١٪، وهذه ملاحظة على بيانات الاختبار وليست اختيارًا. '
                'التحقق يفحص شكل الجواب وليس صحته.</div>'
                '<div dir="ltr" style="color:var(--muted);font-size:.78rem;margin-top:.5rem;font-family:Inter">'
                "KAYL's choices were made on calibration data only and were not changed after the test. 120b-low met all "
                "floors with about 31% less compute; this is an observation on the test set, not a choice. The verifier "
                "checks answer format, not correctness. E is a simple difficulty proxy (question length), not RouteLLM. "
                "Final-test gpt-oss calls went through OpenRouter pinned to the Groq provider. Full details: RESULTS.md.</div></div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------- page

def hero():
    st.markdown("""<div class="k-hero">
      <div class="copy">
        <div class="eyebrow">Arabic AI · less compute · same answers</div>
        <div class="words"><div class="kayl k-grad">KAYL</div><div class="kayl-ar k-grad">كَيْل</div></div>
        <div class="tag">Intelligence Without Waste</div>
        <div class="tag-ar">الذكاء بلا هدر</div>
        <div class="sub">The right measure of AI for every Arabic task: the smallest model and the least thinking that still meets the accuracy floor.</div>
      </div>
      <div class="k-shape"><i></i><i></i><i></i><i></i></div>
    </div>""", unsafe_allow_html=True)


def set_example(key: str):
    _, tid = EXAMPLES[key]
    st.session_state["prompt"] = example_text(calibration_tasks()[tid])
    st.session_state["example"] = tid


def main():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    hero()
    qp = st.query_params
    keys = [r[0] for r in REPLAYS]

    demo, results = st.tabs(["العرض · Demo", "النتائج · Results"])
    with demo:
        replay = st.toggle("وضع الإعادة: تشغيلات مسجّلة بلا استدعاء نماذج · Replay mode", value=True)
        task, cfg, simulated, go = None, None, False, False
        if replay:
            names = {r[0]: f"{r[1]} · {r[2]}" for r in REPLAYS}
            default = keys.index(qp["case"]) if qp.get("case") in keys else 0
            key = st.radio("التشغيل المسجّل · Recorded run", keys, index=default, format_func=names.get, horizontal=True)
            _, _, _, source, tid, simulated, ov = next(r for r in REPLAYS if r[0] == key)
            try:
                task = (calibration_tasks() if source == "calibration" else drill_tasks())[tid]
            except Exception as e:  # بيانات التمرين تحتاج نسخة Hugging Face المحلية
                st.markdown(f'<div class="k-card">هذا التسجيل يحتاج نسخة البيانات المحلية · needs the local dataset cache ({esc(e)})</div>',
                            unsafe_allow_html=True)
            cfg = case_config(ov["route"], ov.get("unavailable"))
            if task:
                question_card(task)
            go = st.button("▶  تشغيل التسجيل · Play recording", disabled=task is None)
            if qp.get("autoplay") == "1" and not st.session_state.get("autoplayed"):
                st.session_state["autoplayed"] = go = True
        else:
            st.markdown('<div class="k-banner"><span>الوضع المباشر يستدعي النماذج</span>'
                        '<span class="en">Live mode calls the models: gpt-oss via OpenRouter pinned to Groq, ALLaM on Groq</span></div>',
                        unsafe_allow_html=True)
            b1, b2, b3, _ = st.columns([1, 1, 1.6, 1.6])
            b1.button(EXAMPLES["math"][0], on_click=set_example, args=("math",), type="secondary")
            b2.button(EXAMPLES["knowledge"][0], on_click=set_example, args=("knowledge",), type="secondary")
            b3.button(EXAMPLES["intent"][0], on_click=set_example, args=("intent",), type="secondary")
            text = st.text_area("المهمة بالعربية · Arabic task", key="prompt", height=120,
                                placeholder="اكتب سؤالًا بالعربية… (للاختيار من متعدد: سطر لكل خيار مثل A) …)")
            kind = st.selectbox("نوع المهمة · Task type", ["Auto", "Math", "Knowledge", "Intent"], index=0)
            go = st.button("تشغيل كَيْل · Run KAYL", disabled=not (text or "").strip())
            if go:
                example = calibration_tasks().get(st.session_state.get("example"))
                if example and example_text(example) == text.strip():
                    task = example
                else:
                    task, _ = build_task(text, kind)  # الوكيل نفسه يكتب كيف عرف نوع المهمة (estimate)
                cfg = case_config()

        if go and task:
            banner(simulated, live=not replay)
            with st.spinner("كَيْل يعمل… · Running KAYL…"):
                st.session_state["last"] = (run_case(task, cfg, replay), cfg, simulated, not replay)
            results_view(st.session_state["last"][0], cfg, animate=qp.get("autoplay") != "1")
        elif st.session_state.get("last"):
            r, cfg_last, sim_last, live_last = st.session_state["last"]
            banner(sim_last, live=live_last)
            results_view(r, cfg_last, animate=False)

    with results:
        results_tab()

    st.markdown(f'<div class="k-foot"><div class="k-ar">{esc(FOOTNOTE_AR)} <br><bdi class="en" dir="ltr">{esc(FOOTNOTE_EN)}</bdi></div>'
                f'<div class="en" dir="ltr">KAYL — research prototype</div></div>', unsafe_allow_html=True)


main()
