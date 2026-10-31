"""Build the KAYL data splits. Makes no model calls.

Three task types, each with 30 calibration and 30 test items (seed 42):
  math       Arabic-GSM8K, graded by the final number
  knowledge  ArabicMMLU multiple choice, graded by the option letter
  intent     synthetic citizen-service messages (MSA + Saudi), 8 intents + other, graded by the label
plus 20 English GSM8K problems for robustness (test only).

Outputs:
  data/calibration.jsonl     the only data KAYL learns from (90 items)
  data/test.jsonl            held out for the final evaluation (90 items)
  data/robustness_en.jsonl   robustness test only (20 items)

Intent data is read from data/synthetic/intents.csv (made by scripts/generate_intents.py,
committed) and joined with the human review in data/synthetic/intents_review.csv.

Run from the repository root:
    python -m scripts.prepare_data
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

from kayl.data import GSM8K_AR, GSM8K_EN, MMLU_AR, final_answer, shuffled

INTENTS_CSV = Path("data/synthetic/intents.csv")
INTENTS_REVIEW = Path("data/synthetic/intents_review.csv")

N_CALIBRATION = 30
N_TEST = 30
N_ROBUSTNESS_EN = 20
OUT = Path("data")

LETTERS = "ABCDE"


def build_math() -> tuple[list, list, set[int]]:
    """Returns (calibration, test, calibration row numbers); the row numbers are needed to keep
    the English set free of calibration problems."""
    from datasets import load_dataset
    ds = load_dataset(GSM8K_AR, split="main_test")  # this dataset names its test split main_test

    items = []
    for i, row in enumerate(ds):
        ans = final_answer(row["answer"])
        if ans is None:
            continue
        items.append({"id": f"gsm8k-ar-{i}", "type": "math",
                      "question": row["question"], "answer": ans})
    print(f"math: {len(ds)} problems, {len(items)} with a clear final answer.")

    items = shuffled(items)
    cal, test = items[:N_CALIBRATION], items[N_CALIBRATION:N_CALIBRATION + N_TEST]
    cal_rows = {int(x["id"].rsplit("-", 1)[1]) for x in cal}
    return cal, test, cal_rows


def build_knowledge() -> tuple[list, list]:
    """Round-robin over subjects until 60 items, then alternate them between calibration and
    test, so most subjects appear in both."""
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
    print(f"knowledge: {sum(map(len, by_subject.values()))} questions in {len(by_subject)} subjects.")

    pools = {s: shuffled(v) for s, v in sorted(by_subject.items())}
    picked, round_ = [], 0
    while len(picked) < N_CALIBRATION + N_TEST:
        for subject in pools:
            if round_ < len(pools[subject]) and len(picked) < N_CALIBRATION + N_TEST:
                picked.append(pools[subject][round_])
        round_ += 1

    for x in picked:
        assert LETTERS.index(x["answer"]) < len(x["choices"]), x["id"]
    return picked[0::2], picked[1::2]


def build_intent() -> tuple[list, list]:
    """Per intent, 3 messages in calibration and 3 in test; for other, 6 and 6.
    30 items per file, balanced 15 MSA / 15 Saudi."""
    with INTENTS_CSV.open(encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    labels = sorted({r["intent"] for r in raw})
    assert len({r["text"] for r in raw}) == len(raw), "duplicate messages in intents.csv"

    # The review never removes rows: every row is kept, with its review result attached.
    review = {}
    if INTENTS_REVIEW.exists():
        with INTENTS_REVIEW.open(encoding="utf-8-sig", newline="") as f:
            review = {r["id"]: r["my_check"].strip() for r in csv.DictReader(f) if r["my_check"].strip()}

    cells = defaultdict(list)  # (intent, dialect) -> messages
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
        a, b = (2, 1) if k % 2 == 0 else (1, 2)  # alternate the MSA / Saudi mix to balance dialects
        assert len(msa) >= a + b and len(sa) >= a + b, intent
        cal += msa[:a] + sa[:b]
        test += msa[a:a + b] + sa[b:b + a]
    msa, sa = shuffled(cells[("other", "msa")]), shuffled(cells[("other", "saudi")])
    assert len(msa) >= 6 and len(sa) >= 6, "other"
    cal += msa[:3] + sa[:3]
    test += msa[3:6] + sa[3:6]

    print(f"intent: {len(raw)} synthetic messages, {len(labels)} labels, human-reviewed: {len(review)}.")
    return shuffled(cal), shuffled(test)


def build_robustness_en(math_cal_rows: set[int]) -> list:
    """20 English GSM8K problems. Row i of English GSM8K is the same problem as row i of
    Arabic-GSM8K, so math calibration rows are excluded."""
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
    print(f"saved {len(items)} items to {path}")


def main():
    math_cal, math_test, math_cal_rows = build_math()
    know_cal, know_test = build_knowledge()
    int_cal, int_test = build_intent()
    robust_en = build_robustness_en(math_cal_rows)

    calibration = math_cal + know_cal + int_cal
    test = math_test + know_test + int_test

    assert not {x["id"] for x in calibration} & {x["id"] for x in test}
    assert all("type" in x for x in calibration + test + robust_en)

    OUT.mkdir(exist_ok=True)
    print()
    write_jsonl(OUT / "calibration.jsonl", calibration)
    write_jsonl(OUT / "test.jsonl", test)
    write_jsonl(OUT / "robustness_en.jsonl", robust_en)


if __name__ == "__main__":
    main()
