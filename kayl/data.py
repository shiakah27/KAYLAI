"""Dataset helpers shared by the data script, the run scripts and the UI."""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from kayl.normalize import normalize_number

GSM8K_AR = "Omartificial-Intelligence-Space/Arabic-gsm8k"
GSM8K_EN = "openai/gsm8k"
MMLU_AR = "MBZUAI/ArabicMMLU"
SEED = 42

SPLITS = {"calibration": "calibration.jsonl", "test": "test.jsonl", "robustness_en": "robustness_en.jsonl"}

# Robustness drill items: positions 60-64 of the same seed-42 math shuffle.
# Calibration uses positions 0-29 and test 30-59, so these items are in neither.
DRILL_SLICE = (60, 65)


def final_answer(answer_field: str) -> str | None:
    """The final number of a GSM8K answer, which follows '####'.
    Some Arabic-GSM8K rows use '###' by mistake, so three or more '#' are accepted."""
    if "###" not in answer_field:
        return None
    return normalize_number(re.split(r"#{3,}", answer_field)[-1])


def shuffled(items: list, seed: int = SEED) -> list:
    """A seeded shuffle that leaves the input list unchanged."""
    items = list(items)
    random.Random(seed).shuffle(items)
    return items


def load_tasks(cfg: dict, split: str, n_per_type: int | None = None) -> list[dict]:
    """Tasks of a split in file order, optionally only the first n of each type."""
    path = Path(cfg["paths"]["data"]) / SPLITS[split]
    by_type = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        task = json.loads(line)
        if n_per_type is None or len(by_type[task["type"]]) < n_per_type:
            by_type[task["type"]].append(task)
    return [t for tasks in by_type.values() for t in tasks]


def drill_tasks() -> list[dict]:
    """The five robustness drill items, built with the same steps as the math split."""
    from datasets import load_dataset
    ds = load_dataset(GSM8K_AR, split="main_test")
    items = [{"id": f"gsm8k-ar-{i}", "type": "math", "question": row["question"],
              "answer": final_answer(row["answer"])}
             for i, row in enumerate(ds) if final_answer(row["answer"]) is not None]
    return shuffled(items)[DRILL_SLICE[0]:DRILL_SLICE[1]]
