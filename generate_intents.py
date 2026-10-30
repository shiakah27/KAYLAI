# -*- coding: utf-8 -*-
"""
كَيْل — توليد بيانات تصنيف النية (بيانات اصطناعية)

ماذا يفعل:
  يطلب من نموذج لغوي كتابة طلبات خدمات حكومية بالعربية، لكل نية (intent)
  ولكل لهجة (فصحى / سعودية)، ويحفظها في data/synthetic/intents_raw.jsonl.
  prepare_data.py يقرأ هذا الملف لاحقًا ويقسمه إلى معايرة واختبار.

لماذا ملف منفصل؟ لأن هذا السكربت يستدعي نموذجًا (يكلف ويحتاج مفتاح API)،
أما prepare_data.py فيبقى بدون أي استدعاء نموذج. الناتج محفوظ في git،
فلا يحتاج أحد لإعادة التوليد لكي يعيد التجارب.

التشغيل:
    python generate_intents.py
"""

import hashlib
import json
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

MODEL = "groq/openai/gpt-oss-120b"
PARAMS = {"temperature": 0, "reasoning_effort": "low", "max_tokens": 4096}
N_PER_CELL = 5  # عدد الطلبات لكل (نية، لهجة). نولّد أكثر من الحاجة (3) لنستبعد المكرر.
OUT = Path("data/synthetic")
CACHE = Path("cache/generation")

# النوايا العشر: label بالإنجليزية (هو الجواب الصحيح)، ووصف عربي يفهمه النموذج.
INTENTS = {
    "renew_national_id": "تجديد بطاقة الهوية الوطنية",
    "renew_passport": "تجديد جواز السفر أو إصداره",
    "renew_driving_license": "تجديد رخصة القيادة",
    "traffic_violation_inquiry": "الاستعلام عن المخالفات المرورية أو سدادها أو الاعتراض عليها",
    "exit_reentry_visa": "إصدار تأشيرة خروج وعودة لعامل أو مقيم",
    "vehicle_ownership_transfer": "نقل ملكية مركبة (سيارة) من شخص لآخر",
    "book_medical_appointment": "حجز موعد في مركز صحي أو مستشفى حكومي",
    "report_utility_outage": "الإبلاغ عن انقطاع الكهرباء أو الماء",
    "social_support_inquiry": "الاستفسار عن الضمان الاجتماعي أو الدعم الحكومي",
    "file_complaint": "تقديم شكوى على جهة حكومية أو موظف بسبب سوء خدمة",
}

DIALECTS = {
    "msa": "العربية الفصحى الحديثة",
    "saudi": "اللهجة السعودية العامية (كما يكتبها الناس في الواتساب وتويتر)",
}

PROMPT_TEMPLATE = """اكتب {n} طلبات مختلفة يكتبها مواطن أو مقيم في السعودية لخدمة حكومية إلكترونية.
الهدف من كل الطلبات: {intent_desc}
اكتبها بـ: {dialect_desc}

الشروط:
- كل طلب جملة أو جملتان فقط، كما يكتبها شخص حقيقي.
- نوّع الصياغة: بعضها مباشر، وبعضها يشرح المشكلة دون ذكر اسم الخدمة صراحة.
- لا تكرر نفس الكلمات في كل الطلبات.
- لا تضف أي شرح أو ترقيم.

أعد JSON فقط بهذا الشكل: {{"requests": ["...", "..."]}}"""


def build_prompt(intent: str, dialect: str) -> str:
    """يبني نص الطلب لنية ولهجة محددتين."""
    return PROMPT_TEMPLATE.format(
        n=N_PER_CELL, intent_desc=INTENTS[intent], dialect_desc=DIALECTS[dialect]
    )


def call_model(prompt: str) -> str:
    """يستدعي النموذج مع كاش على القرص (نفس الطلب = نفس الجواب بدون استدعاء جديد)."""
    import litellm

    key = hashlib.sha256(json.dumps([MODEL, PARAMS, prompt], ensure_ascii=False).encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["content"]

    resp = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        num_retries=3,
        **PARAMS,
    )
    content = resp.choices[0].message.content
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"content": content, "usage": resp.usage.model_dump()},
                               ensure_ascii=False), encoding="utf-8")
    return content


def main():
    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for intent in INTENTS:
        for dialect in DIALECTS:
            requests = json.loads(call_model(build_prompt(intent, dialect)))["requests"]
            for text in requests:
                rows.append({"intent": intent, "dialect": dialect, "text": text.strip()})
            print(f"{intent:28s} {dialect:6s} -> {len(requests)}")

    with (OUT / "intents_raw.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {"model": MODEL, "params": PARAMS, "n_per_cell": N_PER_CELL,
            "generated_on": date.today().isoformat(), "prompt_template": PROMPT_TEMPLATE,
            "intents": INTENTS, "dialects": DIALECTS, "n_rows": len(rows)}
    (OUT / "intents_generation.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nحُفظ {len(rows)} طلبًا في {OUT / 'intents_raw.jsonl'}")


if __name__ == "__main__":
    main()
