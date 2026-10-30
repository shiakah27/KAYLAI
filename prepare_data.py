# -*- coding: utf-8 -*-
"""
كَيْل — تجهيز البيانات

ماذا يفعل:
  1. يحمّل مسائل Arabic-GSM8K (مجموعة الاختبار) من Hugging Face.
  2. يستخرج الجواب النهائي (الرقم بعد ####).
  3. يختار عينة عشوائية ببذرة ثابتة، ويقسمها:
       - معايرة (calibration): يتعلم منها كَيْل.
       - اختبار (test): لا يراها كَيْل إلا في التقييم النهائي.
  4. يحفظهما في data/ بصيغة JSONL.

لماذا بذرة ثابتة؟ لأن أي شخص يشغّل السكربت يحصل على نفس الأسئلة بالضبط
(معيار 5: تشغيل قابل للتكرار).

التثبيت:
    pip install datasets

التشغيل:
    python prepare_data.py
"""

import json
import random
import re
from pathlib import Path

DATASET = "Omartificial-Intelligence-Space/Arabic-gsm8k"
SEED = 42
N_CALIBRATION = 30
N_TEST = 30
OUT = Path("data")

# تحويل الأرقام العربية-الهندية (٠١٢...) إلى أرقام لاتينية (012...)
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_number(text: str) -> str | None:
    """يحول نصًا إلى رقم موحد: يزيل الفواصل، ويحول الأرقام العربية، ويحذف .0 الزائدة."""
    if text is None:
        return None
    t = text.translate(ARABIC_DIGITS).replace(",", "").replace("٬", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    if not m:
        return None
    num = m.group(0)
    if "." in num:
        num = num.rstrip("0").rstrip(".")
    return num


def final_answer(answer_field: str) -> str | None:
    """في GSM8K الجواب النهائي يأتي بعد ####."""
    if "####" not in answer_field:
        return None
    return normalize_number(answer_field.split("####")[-1])


def load_test_split():
    """يحمّل مجموعة الاختبار. في هذه النسخة اسمها main_test (وليس test)."""
    from datasets import load_dataset
    try:
        return load_dataset(DATASET, split="main_test")
    except Exception:
        try:
            return load_dataset(DATASET, "main", split="test")
        except Exception:
            return load_dataset(DATASET, split="test")


def main():
    ds = load_test_split()
    print(f"تم تحميل {len(ds)} مسألة من مجموعة الاختبار.")

    items = []
    for i, row in enumerate(ds):
        ans = final_answer(row["answer"])
        if ans is None:
            continue  # نتجاهل أي مسألة بلا جواب واضح
        items.append({"id": f"gsm8k-ar-{i}", "type": "math",
                      "question": row["question"], "answer": ans})

    random.Random(SEED).shuffle(items)
    calibration = items[:N_CALIBRATION]
    test = items[N_CALIBRATION:N_CALIBRATION + N_TEST]

    OUT.mkdir(exist_ok=True)
    for name, subset in [("calibration", calibration), ("test", test)]:
        path = OUT / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"حُفظ {len(subset)} مسألة في {path}")

    # تأكد: لا تداخل بين المعايرة والاختبار
    assert not {x["id"] for x in calibration} & {x["id"] for x in test}

    print("\nمثال من المعايرة:")
    print("السؤال:", calibration[0]["question"][:200])
    print("الجواب:", calibration[0]["answer"])


if __name__ == "__main__":
    main()
