"""Graders: compare the FINAL value with the gold answer, by task type.

    math       -> the number (after normalizing Arabic digits, separators and .0)
    knowledge  -> the option letter (A, (A), A. or the Arabic letter أ are accepted)
    intent     -> the exact label (case, spaces, hyphens and quotes are ignored)

Strict but fair: small format differences are accepted, an ambiguous answer counts as wrong.
Every grade carries a reason, so a wrong answer can be told apart from a broken format."""

import re
from dataclasses import dataclass

from kayl.normalize import ARABIC_DIGITS, normalize_number

LETTERS = "ABCDE"
ARABIC_LETTERS = {"أ": "A", "ا": "A", "ب": "B", "ج": "C", "د": "D", "ه": "E", "هـ": "E"}

# Grade reasons
OK = "ok"
WRONG = "wrong"          # a valid answer, but not the gold one
NO_FINAL = "no_final"    # no FINAL line
AMBIGUOUS = "ambiguous"  # more than one candidate answer (e.g. two numbers)
INVALID = "invalid"      # FINAL is not a number / option letter / known label


@dataclass(frozen=True)
class Grade:
    correct: bool
    reason: str
    parsed: str | None = None  # the parsed number, letter or label


def _numbers(text: str) -> list[str]:
    """Every number in the text, normalized."""
    t = text.translate(ARABIC_DIGITS).replace(",", "").replace("٬", "").replace("٫", ".")
    return [normalize_number(n) for n in re.findall(r"-?\d+(?:\.\d+)?", t)]


def grade_math(final: str, answer: str) -> Grade:
    """FINAL must contain exactly one number."""
    nums = set(_numbers(final))
    if not nums:
        return Grade(False, INVALID)
    if len(nums) > 1:
        return Grade(False, AMBIGUOUS)
    num = nums.pop()
    return Grade(num == answer, OK if num == answer else WRONG, num)


def parse_letter(final: str, n_choices: int) -> str | None:
    """Read the option letter from "B", "(B)", "B.", "B) option text" or "ب"."""
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
    """"Renew Passport" or "`renew-passport`" -> "renew_passport"."""
    t = text.strip().strip("*`'\".").strip().lower()
    return re.sub(r"[\s\-]+", "_", t)


def grade_intent(final: str, answer: str, labels: list[str]) -> Grade:
    label = normalize_label(final)
    if label not in labels:
        return Grade(False, INVALID, label)
    return Grade(label == answer, OK if label == answer else WRONG, label)


def grade(task: dict, final: str | None) -> Grade:
    """Grade one answer for its task type."""
    if final is None or not final.strip():
        return Grade(False, NO_FINAL)
    if task["type"] == "math":
        return grade_math(final, task["answer"])
    if task["type"] == "knowledge":
        return grade_knowledge(final, task["answer"], len(task["choices"]))
    if task["type"] == "intent":
        return grade_intent(final, task["answer"], task["choices"])
    raise ValueError(f"unknown task type: {task['type']}")


def validate(task: dict, final: str | None) -> tuple[bool, str]:
    """Whether the answer is valid for the task type, without knowing the gold answer.
    Used by the agent's verify step. Returns (valid, reason)."""
    if final is None or not final.strip():
        return False, NO_FINAL
    if task["type"] == "math":
        nums = set(_numbers(final))
        if not nums:
            return False, INVALID
        return (True, OK) if len(nums) == 1 else (False, AMBIGUOUS)
    if task["type"] == "knowledge":
        return (True, OK) if parse_letter(final, len(task["choices"])) else (False, INVALID)
    if task["type"] == "intent":
        return (True, OK) if normalize_label(final) in task["choices"] else (False, INVALID)
    raise ValueError(f"unknown task type: {task['type']}")
