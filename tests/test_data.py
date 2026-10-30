# -*- coding: utf-8 -*-
"""اختبارات انحدار على البيانات المحفوظة: الأحجام، عدم التداخل، وأن كل جواب صحيح يمكن تصحيحه."""

import json
from collections import Counter
from pathlib import Path

import pytest

from kayl.grading import grade


def load(name):
    return [json.loads(line) for line in Path("data", name).open(encoding="utf-8")]


CAL, TEST, ROBUST = load("calibration.jsonl"), load("test.jsonl"), load("robustness_en.jsonl")


def test_sizes_per_type():
    for split in (CAL, TEST):
        assert Counter(x["type"] for x in split) == {"math": 30, "knowledge": 30, "intent": 30}
    assert len(ROBUST) == 20


def test_no_overlap_between_calibration_and_test():
    assert not {x["id"] for x in CAL} & {x["id"] for x in TEST}
    assert not {x["question"] for x in CAL} & {x["question"] for x in TEST}


@pytest.mark.parametrize("item", CAL + TEST + ROBUST, ids=lambda x: x["id"])
def test_gold_answer_grades_as_correct(item):
    # إذا أعطى النموذج الجواب الصحيح بالضبط، يجب أن يقبله المصحح
    assert grade(item, item["answer"]).correct
