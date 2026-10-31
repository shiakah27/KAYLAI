# -*- coding: utf-8 -*-
"""
كَيْل — تجهيز البيانات

ماذا يفعل:
  يبني ثلاثة أنواع من المهام، ولكل نوع 30 سؤال معايرة و30 سؤال اختبار (بذرة 42):
    1. math       — مسائل Arabic-GSM8K، التصحيح بالرقم النهائي.
    2. knowledge  — أسئلة اختيار من متعدد من ArabicMMLU، التصحيح بحرف الخيار الصحيح.
    3. intent     — رسائل اصطناعية لروبوت خدمات حكومية (فصحى + سعودية)، 8 نوايا + other،
                    التصحيح باسم النية.
  ومجموعة متانة: 20 مسألة إنجليزية من GSM8K الأصلي (اختبار فقط).

  الملفات الناتجة:
    data/calibration.jsonl     — يتعلم منها كَيْل فقط (90 سؤالًا).
    data/test.jsonl            — لا يراها كَيْل إلا في التقييم النهائي (90 سؤالًا).
    data/robustness_en.jsonl   — اختبار متانة فقط (20 سؤالًا).

لماذا بذرة ثابتة؟ لأن أي شخص يشغّل السكربت يحصل على نفس الأسئلة بالضبط
(معيار 5: تشغيل قابل للتكرار).

قبل التشغيل: بيانات النية تأتي من data/synthetic/intents.csv
(أنشأها generate_intents.py، وهي محفوظة في git، ويمكن تعديلها يدويًا).

التشغيل:
    python prepare_data.py
"""

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from kayl.normalize import normalize_number  # نفس الدالة التي يستخدمها المصحح

GSM8K_AR = "Omartificial-Intelligence-Space/Arabic-gsm8k"
GSM8K_EN = "openai/gsm8k"
MMLU_AR = "MBZUAI/ArabicMMLU"
INTENTS_CSV = Path("data/synthetic/intents.csv")
INTENTS_REVIEW = Path("data/synthetic/intents_review.csv")  # المراجعة اليدوية (my_check لكل صف)

SEED = 42
N_CALIBRATION = 30
N_TEST = 30
N_ROBUSTNESS_EN = 20
OUT = Path("data")

LETTERS = "ABCDE"


def final_answer(answer_field: str) -> str | None:
    """في GSM8K الجواب النهائي يأتي بعد ####.
    في النسخة العربية بعض المسائل تستخدم ### (ثلاث فقط) بالخطأ، فنقبل الاثنين."""
    if "###" not in answer_field:
        return None
    return normalize_number(re.split(r"#{3,}", answer_field)[-1])


def shuffled(items: list, seed: int = SEED) -> list:
    """نسخة مخلوطة ببذرة ثابتة (لا تغيّر القائمة الأصلية)."""
    items = list(items)
    random.Random(seed).shuffle(items)
    return items


# ---------- 1. math: Arabic-GSM8K ----------

def build_math() -> tuple[list, list, set[int]]:
    """يرجع (معايرة، اختبار، أرقام صفوف المعايرة) — نحتاج الأخيرة لمجموعة الإنجليزية."""
    from datasets import load_dataset
    ds = load_dataset(GSM8K_AR, split="main_test")  # في هذه النسخة اسمها main_test وليس test

    items = []
    for i, row in enumerate(ds):
        ans = final_answer(row["answer"])
        if ans is None:
            continue  # نتجاهل أي مسألة بلا جواب واضح
        items.append({"id": f"gsm8k-ar-{i}", "type": "math",
                      "question": row["question"], "answer": ans})
    print(f"math: {len(ds)} مسألة، منها {len(items)} بجواب واضح.")

    items = shuffled(items)
    cal, test = items[:N_CALIBRATION], items[N_CALIBRATION:N_CALIBRATION + N_TEST]
    cal_rows = {int(x["id"].rsplit("-", 1)[1]) for x in cal}
    return cal, test, cal_rows


# ---------- 2. knowledge: ArabicMMLU ----------

def build_knowledge() -> tuple[list, list]:
    """عينة موزعة على المواد: نأخذ سؤالًا من كل مادة بالتناوب حتى نصل إلى 60،
    ثم نوزعها بالتبادل بين المعايرة والاختبار، فتظهر كل مادة تقريبًا في الاثنين."""
    from datasets import load_dataset
    ds = load_dataset(MMLU_AR, "All", split="test")

    by_subject = defaultdict(list)
    for row in ds:
        if row["is_few_shot"] == "1":
            continue
        options = [row[f"Option {k}"] for k in range(1, 6) if row[f"Option {k}"]]
        question = row["Question"]
        if row["Context"]:
            question = f"{row['Context']}\n\n{question}"
        by_subject[row["Subject"]].append({
            "id": f"arabicmmlu-{row['ID']}", "type": "knowledge",
            "question": question, "choices": options, "answer": row["Answer Key"],
            "subject": row["Subject"], "level": row["Level"],
        })
    print(f"knowledge: {sum(map(len, by_subject.values()))} سؤالًا في {len(by_subject)} مادة.")

    pools = {s: shuffled(v) for s, v in sorted(by_subject.items())}
    picked, round_ = [], 0
    while len(picked) < N_CALIBRATION + N_TEST:
        for subject in pools:
            if round_ < len(pools[subject]) and len(picked) < N_CALIBRATION + N_TEST:
                picked.append(pools[subject][round_])
        round_ += 1

    for x in picked:  # تأكد: حرف الجواب يشير إلى خيار موجود
        assert LETTERS.index(x["answer"]) < len(x["choices"]), x["id"]
    return picked[0::2], picked[1::2]


# ---------- 3. intent: synthetic citizen service messages ----------

def build_intent() -> tuple[list, list]:
    """لكل نية من الثماني: 3 رسائل في المعايرة و3 في الاختبار.
    لـ other: 6 في المعايرة (3 فصحى + 3 سعودي) و6 في الاختبار.
    المجموع 30 لكل ملف، واللهجة متوازنة: 15 فصحى و15 سعودية."""
    with INTENTS_CSV.open(encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    labels = sorted({r["intent"] for r in raw})
    assert len({r["text"] for r in raw}) == len(raw), "رسائل مكررة في intents.csv"

    # نتيجة المراجعة اليدوية لكل صف (ok / not natural / ambiguous)، من ملف المراجعة إن وُجد.
    # المراجعة لا تحذف أي صف: كل الصفوف تبقى، ومعها نتيجة المراجعة.
    review = {}
    if INTENTS_REVIEW.exists():
        with INTENTS_REVIEW.open(encoding="utf-8-sig", newline="") as f:
            review = {r["id"]: r["my_check"].strip() for r in csv.DictReader(f) if r["my_check"].strip()}

    cells = defaultdict(list)  # (intent, dialect) -> رسائل
    for i, r in enumerate(raw):
        rid = f"intent-{i}"
        cells[(r["intent"], r["dialect"])].append({
            "id": rid, "type": "intent", "question": r["text"],
            "choices": labels, "answer": r["intent"], "dialect": r["dialect"],
            "synthetic": True,
            "human_reviewed": rid in review,
            "review": review.get(rid),
        })

    cal, test = [], []
    for k, intent in enumerate(l for l in labels if l != "other"):
        msa, sa = shuffled(cells[(intent, "msa")]), shuffled(cells[(intent, "saudi")])
        a, b = (2, 1) if k % 2 == 0 else (1, 2)  # بالتناوب: (فصحى، سعودي) في المعايرة
        assert len(msa) >= a + b and len(sa) >= a + b, intent
        cal += msa[:a] + sa[:b]
        test += msa[a:a + b] + sa[b:b + a]
    msa, sa = shuffled(cells[("other", "msa")]), shuffled(cells[("other", "saudi")])
    assert len(msa) >= 6 and len(sa) >= 6, "other"
    cal += msa[:3] + sa[:3]
    test += msa[3:6] + sa[3:6]

    print(f"intent: {len(raw)} رسالة اصطناعية، {len(labels)} أصناف، راجعها إنسان: {len(review)}.")
    return shuffled(cal), shuffled(test)


# ---------- robustness: English GSM8K ----------

def build_robustness_en(math_cal_rows: set[int]) -> list:
    """20 مسألة إنجليزية من GSM8K الأصلي (اختبار فقط).
    الصف i في الإنجليزية هو نفس المسألة في الصف i بالعربية، لذلك نستبعد صفوف
    معايرة math حتى لا يُختبر كَيْل على مسألة تعلّم منها."""
    from datasets import load_dataset
    ds = load_dataset(GSM8K_EN, "main", split="test")
    items = [{"id": f"gsm8k-en-{i}", "type": "math", "lang": "en",
              "question": row["question"], "answer": final_answer(row["answer"])}
             for i, row in enumerate(ds) if i not in math_cal_rows]
    items = [x for x in items if x["answer"] is not None]
    return shuffled(items)[:N_ROBUSTNESS_EN]


def write_jsonl(path: Path, items: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"حُفظ {len(items)} سؤالًا في {path}")


def main():
    math_cal, math_test, math_cal_rows = build_math()
    know_cal, know_test = build_knowledge()
    int_cal, int_test = build_intent()
    robust_en = build_robustness_en(math_cal_rows)

    calibration = math_cal + know_cal + int_cal
    test = math_test + know_test + int_test

    # تأكد: لا تداخل بين المعايرة والاختبار، وكل عنصر له type
    assert not {x["id"] for x in calibration} & {x["id"] for x in test}
    assert all("type" in x for x in calibration + test + robust_en)

    OUT.mkdir(exist_ok=True)
    print()
    write_jsonl(OUT / "calibration.jsonl", calibration)
    write_jsonl(OUT / "test.jsonl", test)
    write_jsonl(OUT / "robustness_en.jsonl", robust_en)


if __name__ == "__main__":
    main()
