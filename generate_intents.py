# -*- coding: utf-8 -*-
"""
كَيْل — توليد بيانات تصنيف النية (بيانات اصطناعية)

ماذا يفعل:
  يطلب من نموذج لغوي كتابة رسائل قصيرة يكتبها الناس لروبوت محادثة حكومي،
  لكل نية (intent) ولكل لهجة (فصحى بسيطة / سعودية)، ويحفظها في
  data/synthetic/intents.csv بالأعمدة: text, intent, dialect, human_reviewed.
  ملف CSV حتى تستطيع مراجعته وتعديله يدويًا.

لماذا qwen/qwen3.8-27b؟ لأنه ليس من النماذج التي يختار بينها كَيْل،
فلا يكون نموذج مرشح هو من كتب البيانات التي سيُختبر عليها (تقليل التحيز).
(كانت الخطة llama-3.3-70b-versatile، لكنه غير متاح في حساب Groq.)

لماذا ملف منفصل؟ لأن هذا السكربت يستدعي نموذجًا (يكلف ويحتاج مفتاح API)،
أما prepare_data.py فيبقى بدون أي استدعاء نموذج. الناتج محفوظ في git.

تحذير: إعادة التشغيل تستبدل intents.csv وتمسح أي مراجعة يدوية.

التشغيل:
    python generate_intents.py
"""

import csv
import hashlib
import json
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

MODEL = "groq/qwen/qwen3.8-27b"
# max_tokens أقل من حد Groq المجاني (1000 output tokens في الدقيقة لهذا النموذج).
# reasoning_effort="none" يوقف وضع التفكير في qwen3 (لا نحتاجه لكتابة رسائل قصيرة).
PARAMS = {"temperature": 0, "max_tokens": 800, "reasoning_effort": "none"}
RATE_LIMIT_WAIT_S = 60
OUT = Path("data/synthetic")
CACHE = Path("cache/generation")
MIN_WORDS, MAX_WORDS = 4, 15

# 8 نوايا مختلفة بوضوح + other. الوصف دقيق حتى يطابق النصُّ اسمَ النية تمامًا
# (مثلًا: تجديد جواز موجود فقط، وليس إصدار جواز جديد أو بدل فاقد).
INTENTS = {
    "renew_national_id": "تجديد بطاقة هوية وطنية موجودة انتهت أو قربت تنتهي (فقط تجديد، ليس إصدار أول مرة ولا بدل فاقد)",
    "renew_passport": "تجديد جواز سفر موجود انتهى أو قرب ينتهي (فقط تجديد، ليس إصدار أول جواز ولا بدل فاقد)",
    "traffic_violation_inquiry": "الاستعلام عن المخالفات المرورية أو سدادها",
    "vehicle_ownership_transfer": "نقل ملكية سيارة من شخص لآخر (بيع أو شراء سيارة مستعملة)",
    "book_medical_appointment": "حجز موعد في مركز صحي أو مستشفى حكومي",
    "report_utility_outage": "الإبلاغ عن انقطاع الكهرباء أو الماء في البيت",
    "social_support_inquiry": "الاستفسار عن الضمان الاجتماعي أو حساب المواطن (الأهلية، موعد الصرف، حالة الطلب)",
    "file_complaint": "تقديم شكوى على جهة حكومية أو موظف بسبب سوء خدمة أو تأخير",
    "other": "رسائل لا علاقة لها بأي خدمة حكومية: سلام وتحية، الطقس، كرة القدم، وصفات أكل، نكت، أسئلة عامة عشوائية",
}

# عدد الرسائل لكل (نية، لهجة): 8 نوايا × 2 لهجة × 4 = 64، و other × 2 لهجة × 8 = 16. المجموع 80.
N_PER_CELL = {label: (8 if label == "other" else 4) for label in INTENTS}
N_SPARE = 2

DIALECTS = {
    "msa": ("فصحى بسيطة كما يكتبها الناس العاديون، ليست رسمية ولا بلغة المعاملات الحكومية. "
            "مثال على الأسلوب: «كيف أجدد جوازي؟» وليس «أرجو التكرم بإفادتي عن إجراءات التجديد»"),
    "saudi": ("لهجة سعودية حقيقية يومية كما يكتبها الناس في الواتساب. "
              "استخدم كلمات مثل: ابي، ابغى، ابغا، وش، كيف اسوي، ودي، ليش، الحين، وين، مره، يعطيك العافية"),
}

PROMPT_TEMPLATE = """اكتب {n} رسائل مختلفة يكتبها أشخاص حقيقيون في السعودية لروبوت محادثة (chatbot) تابع لبوابة خدمات حكومية.
موضوع كل الرسائل: {intent_desc}
اللغة: {dialect_desc}

القواعد:
- كل رسالة قصيرة: من 4 إلى 15 كلمة.
- عفوية كأن الشخص يكتب بسرعة من جواله: أحيانًا بدون علامات ترقيم، وأحيانًا فيها خطأ إملائي بسيط.
- كل رسالة تبدأ بشكل مختلف، ولا تكرر نفس الكلمات.
- كل رسالة يجب أن تطابق الموضوع بالضبط، ولا تخلطه بخدمة أخرى.
- لا تكتب أي شرح أو ترقيم أو أسماء.

أعد JSON فقط بهذا الشكل: {{"messages": ["...", "..."]}}"""


def build_prompt(intent: str, dialect: str) -> str:
    """يبني نص الطلب لنية ولهجة محددتين."""
    # نطلب رسالتين إضافيتين احتياطًا، لأن بعض الرسائل قد تُستبعد (تكرار أو طول خارج 4–15 كلمة).
    return PROMPT_TEMPLATE.format(
        n=N_PER_CELL[intent] + N_SPARE, intent_desc=INTENTS[intent], dialect_desc=DIALECTS[dialect]
    )


def call_model(prompt: str) -> tuple[str, dict]:
    """يستدعي النموذج مع كاش على القرص (نفس الطلب = نفس الجواب بدون استدعاء جديد)."""
    import litellm

    key = hashlib.sha256(json.dumps([MODEL, PARAMS, prompt], ensure_ascii=False).encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached["content"], cached["usage"]

    for attempt in range(5):
        try:
            resp = litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                **PARAMS,
            )
            break
        except litellm.RateLimitError:
            print(f"  حد الطلبات في الدقيقة، ننتظر {RATE_LIMIT_WAIT_S} ثانية (محاولة {attempt + 1})")
            time.sleep(RATE_LIMIT_WAIT_S)
    else:
        raise RuntimeError("rate limit: فشلت كل المحاولات")
    content, usage = resp.choices[0].message.content, resp.usage.model_dump()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"content": content, "usage": usage}, ensure_ascii=False),
                    encoding="utf-8")
    return content, usage


def main():
    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, seen, problems = [], set(), []
    tokens_in = tokens_out = 0

    for intent in INTENTS:
        for dialect in DIALECTS:
            content, usage = call_model(build_prompt(intent, dialect))
            tokens_in += usage["prompt_tokens"]
            tokens_out += usage["completion_tokens"]
            messages = [m.strip() for m in json.loads(content)["messages"]]
            kept = 0
            for text in messages:
                if kept == N_PER_CELL[intent]:
                    break  # طلبنا رسالة إضافية احتياطًا، ونأخذ أول N صالحة فقط
                n_words = len(text.split())
                if text in seen:
                    problems.append(f"skipped duplicate: {text}")
                    continue
                if not MIN_WORDS <= n_words <= MAX_WORDS:
                    problems.append(f"skipped {n_words} words ({intent}/{dialect}): {text}")
                    continue
                seen.add(text)
                rows.append({"text": text, "intent": intent, "dialect": dialect,
                             "human_reviewed": "false"})
                kept += 1
            print(f"{intent:28s} {dialect:6s} -> kept {kept}/{N_PER_CELL[intent]} (got {len(messages)})")
            if kept < N_PER_CELL[intent]:
                problems.append(f"SHORT: {intent}/{dialect} has only {kept}")

    # utf-8-sig حتى يفتح Excel الحروف العربية بشكل صحيح
    with (OUT / "intents.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "intent", "dialect", "human_reviewed"])
        writer.writeheader()
        writer.writerows(rows)

    meta = {"model": MODEL, "params": PARAMS, "n_per_cell": N_PER_CELL, "n_spare": N_SPARE,
            "generated_on": date.today().isoformat(), "prompt_template": PROMPT_TEMPLATE,
            "intents": INTENTS, "dialects": DIALECTS, "n_rows": len(rows),
            "tokens": {"input": tokens_in, "output": tokens_out}}
    (OUT / "intents_generation.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nحُفظ {len(rows)} رسالة في {OUT / 'intents.csv'} (tokens in={tokens_in} out={tokens_out})")
    for p in problems:
        print("PROBLEM:", p)


if __name__ == "__main__":
    main()
