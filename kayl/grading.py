# -*- coding: utf-8 -*-
"""المصححات: نقارن قيمة سطر FINAL بالجواب الصحيح، حسب نوع المهمة.

    math       -> الرقم (بعد التوحيد: أرقام عربية، فواصل، .0)
    knowledge  -> حرف الخيار (نقبل A أو (A) أو A. أو الحرف العربي أ)
    intent     -> اسم النية بالضبط (نتجاهل حالة الأحرف والمسافات وعلامات التنصيص)

صارم لكن عادل: نقبل فروق الشكل البسيطة، أما الجواب الغامض فيُحسب خطأ.
كل نتيجة معها سبب، حتى نفرق بين "جواب خاطئ" و"لم يلتزم بالشكل"."""

import re
from dataclasses import dataclass

from kayl.normalize import ARABIC_DIGITS, normalize_number

LETTERS = "ABCDE"
ARABIC_LETTERS = {"أ": "A", "ا": "A", "ب": "B", "ج": "C", "د": "D", "ه": "E", "هـ": "E"}

# الأسباب
OK = "ok"                # صحيح
WRONG = "wrong"          # جواب صالح لكنه خاطئ
NO_FINAL = "no_final"    # لا يوجد سطر FINAL
AMBIGUOUS = "ambiguous"  # أكثر من جواب محتمل (مثلًا رقمان)
INVALID = "invalid"      # FINAL موجود لكن ليس رقمًا / حرفًا / نية معروفة


@dataclass(frozen=True)
class Grade:
    correct: bool
    reason: str
    parsed: str | None = None  # الجواب بعد القراءة (رقم، حرف، أو نية)


def _numbers(text: str) -> list[str]:
    """كل الأرقام في النص بعد التوحيد."""
    t = text.translate(ARABIC_DIGITS).replace(",", "").replace("٬", "").replace("٫", ".")
    return [normalize_number(n) for n in re.findall(r"-?\d+(?:\.\d+)?", t)]


def grade_math(final: str, answer: str) -> Grade:
    """يجب أن يحتوي FINAL على رقم واحد فقط."""
    nums = set(_numbers(final))
    if not nums:
        return Grade(False, INVALID)
    if len(nums) > 1:
        return Grade(False, AMBIGUOUS)
    num = nums.pop()
    return Grade(num == answer, OK if num == answer else WRONG, num)


def parse_letter(final: str, n_choices: int) -> str | None:
    """يقرأ حرف الخيار: "B" أو "(B)" أو "B." أو "B) نص الخيار" أو "ب"."""
    t = final.strip().strip("*`'\" ")
    m = re.match(r"^[(\[]?([A-Ea-e])(?:[)\].:]|\s|$)", t + " ")
    if m:
        letter = m.group(1).upper()
    else:
        m = re.match(r"^[(\[]?(هـ|[أابجده])(?:[)\].:]|\s|$)", t + " ")
        if not m:
            return None
        letter = ARABIC_LETTERS[m.group(1)]
    return letter if LETTERS.index(letter) < n_choices else None


def grade_knowledge(final: str, answer: str, n_choices: int) -> Grade:
    letter = parse_letter(final, n_choices)
    if letter is None:
        return Grade(False, INVALID)
    return Grade(letter == answer, OK if letter == answer else WRONG, letter)


def normalize_label(text: str) -> str:
    """"Renew Passport" أو "`renew-passport`" -> "renew_passport"."""
    t = text.strip().strip("*`'\".").strip().lower()
    return re.sub(r"[\s\-]+", "_", t)


def grade_intent(final: str, answer: str, labels: list[str]) -> Grade:
    label = normalize_label(final)
    if label not in labels:
        return Grade(False, INVALID, label)
    return Grade(label == answer, OK if label == answer else WRONG, label)


def grade(task: dict, final: str | None) -> Grade:
    """يصحح جوابًا واحدًا حسب نوع المهمة."""
    if final is None or not final.strip():
        return Grade(False, NO_FINAL)
    if task["type"] == "math":
        return grade_math(final, task["answer"])
    if task["type"] == "knowledge":
        return grade_knowledge(final, task["answer"], len(task["choices"]))
    if task["type"] == "intent":
        return grade_intent(final, task["answer"], task["choices"])
    raise ValueError(f"unknown task type: {task['type']}")
