"""Grader tests: a number for math, a letter for knowledge, a label for intent."""

import pytest

from kayl.grading import AMBIGUOUS, INVALID, NO_FINAL, OK, WRONG, grade, parse_letter

MATH = {"id": "m", "type": "math", "answer": "75"}
KNOW = {"id": "k", "type": "knowledge", "answer": "B", "choices": ["w", "x", "y", "z"]}
LABELS = ["renew_passport", "renew_national_id", "other"]
INTENT = {"id": "i", "type": "intent", "answer": "renew_passport", "choices": LABELS}


# ---------- math ----------

@pytest.mark.parametrize("final", ["75", " 75 ", "75.0", "75.00", "٧٥", "$75", "75 دولار", "75 دولارًا"])
def test_math_correct_forms(final):
    g = grade(MATH, final)
    assert g.correct and g.reason == OK and g.parsed == "75"


def test_math_thousands_separator():
    assert grade({**MATH, "answer": "1234"}, "1,234").correct
    assert grade({**MATH, "answer": "1234"}, "١٬٢٣٤").correct


def test_math_wrong_number():
    g = grade(MATH, "76")
    assert not g.correct and g.reason == WRONG


def test_math_two_numbers_is_ambiguous():
    # a real allam reply: a sentence containing both 75 and 30
    g = grade(MATH, "سينفق تيري 75 دولارًا على الزبادي خلال 30 يومًا.")
    assert not g.correct and g.reason == AMBIGUOUS


def test_math_same_number_twice_is_fine():
    assert grade(MATH, "75 = 75").correct


def test_math_no_number():
    assert grade(MATH, "خمسة وسبعون").reason == INVALID


def test_math_negative_and_decimal():
    assert grade({**MATH, "answer": "-5"}, "-5").correct
    assert grade({**MATH, "answer": "2.5"}, "2.50").correct
    assert grade({**MATH, "answer": "2.5"}, "٢٫٥").correct


# ---------- knowledge ----------

@pytest.mark.parametrize("final", ["B", "b", "(B)", "B.", "B)", "B. x", "**B**", "ب"])
def test_knowledge_correct_forms(final):
    g = grade(KNOW, final)
    assert g.correct and g.parsed == "B"


def test_knowledge_wrong_letter():
    g = grade(KNOW, "C")
    assert not g.correct and g.reason == WRONG


@pytest.mark.parametrize("final", ["E", "the answer", "BC", "x"])
def test_knowledge_invalid(final):
    # E does not exist with only 4 options
    assert grade(KNOW, final).reason == INVALID


def test_parse_letter_arabic_letters():
    assert parse_letter("أ", 4) == "A"
    assert parse_letter("د", 4) == "D"
    assert parse_letter("هـ", 5) == "E"


# ---------- intent ----------

@pytest.mark.parametrize("final", ["renew_passport", "Renew_Passport", " renew passport ", "`renew-passport`", "renew_passport."])
def test_intent_correct_forms(final):
    assert grade(INTENT, final).correct


def test_intent_wrong_label():
    g = grade(INTENT, "renew_national_id")
    assert not g.correct and g.reason == WRONG


def test_intent_unknown_label():
    assert grade(INTENT, "renew_driving_license").reason == INVALID


# ---------- common ----------

@pytest.mark.parametrize("task", [MATH, KNOW, INTENT])
@pytest.mark.parametrize("final", [None, "", "   "])
def test_missing_final(task, final):
    g = grade(task, final)
    assert not g.correct and g.reason == NO_FINAL


def test_unknown_type():
    with pytest.raises(ValueError):
        grade({"type": "poetry", "answer": "x"}, "x")
