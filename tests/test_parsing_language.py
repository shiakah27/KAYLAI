# -*- coding: utf-8 -*-
"""اختبارات قراءة سطر FINAL الصارم، وكشف لغة التفكير."""

from kayl.language import arabic_ratio, detect_lang, strip_copied
from kayl.parsing import parse_answer, parse_final, parse_thinking
from kayl.prompts import system_prompt

ALLAM_STYLE = """THINKING: 2 boxes a day for 30 days is 60 boxes.
60 / 4 = 15 packs, 15 * 5 = 75.
ANSWER: سينفق 75 دولارًا.
FINAL: 75"""


def test_parse_all_sections():
    assert parse_final(ALLAM_STYLE) == "75"
    assert parse_answer(ALLAM_STYLE) == "سينفق 75 دولارًا."
    assert parse_thinking(ALLAM_STYLE).startswith("2 boxes a day")
    assert "ANSWER" not in parse_thinking(ALLAM_STYLE)


def test_final_takes_last_line():
    assert parse_final("FINAL: 3\nwait, let me fix it\nFINAL: 4") == "4"


def test_final_is_strict():
    # لا نقبل أشكالًا أخرى: هذه إشارة لعدم الالتزام بالشكل
    assert parse_final("**FINAL:** 75") is None
    assert parse_final("Final answer: 75") is None
    assert parse_final("final: 75") is None
    assert parse_final("") is None
    assert parse_final(None) is None


def test_final_allows_leading_spaces():
    assert parse_final("ANSWER: x\n   FINAL:   B  ") == "B"


def test_no_thinking_section():
    assert parse_thinking("ANSWER: x\nFINAL: 1") is None


def test_detect_lang():
    assert detect_lang("We need the cost: 60 * 1.25 = 75") == "en"
    assert detect_lang("نحتاج التكلفة: 60 × 1.25 = 75") == "ar"
    assert detect_lang("التكلفة cost هي") == "mixed"
    assert detect_lang("60 * 1.25 = 75") == "none"   # أرقام فقط لا تدل على لغة
    assert detect_lang(None) == "none"


def test_strip_copied_quoted_question():
    # حالة حقيقية من gpt-oss: تفكير إنجليزي يقتبس السؤال العربي
    question = "كم عدد التذاكر التي سيحصل عليها كل خريج لتوزيعها على أصدقائهم"
    reasoning = ('Remaining seats = 4750. It says "كم عدد التذاكر التي سيحصل عليها كل خريج" '
                 "so each graduate gets 4750/950 = 5 tickets.")
    assert detect_lang(reasoning) == "mixed"
    assert detect_lang(strip_copied(reasoning, question)) == "en"


def test_strip_copied_unquoted_ngram():
    question = "يمتلك جو خمسين دولارا لشراء زي جديد لرحلته"
    reasoning = "Joe has يمتلك جو خمسين دولارا لشراء and spends 42, so 8 is left."
    assert "يمتلك" not in strip_copied(reasoning, question)


def test_strip_copied_keeps_real_arabic_reasoning():
    # تفكير عربي حقيقي يبقى عربيًا حتى لو استخدم كلمات من السؤال
    question = "يمتلك جو خمسين دولارا لشراء زي جديد لرحلته"
    reasoning = "جو معه خمسون دولارا، والقميص والسروال بعد الخصم ٤٢، فيبقى معه ٨."
    assert detect_lang(strip_copied(reasoning, question)) == "ar"


def test_arabic_ratio_ignores_digits():
    assert arabic_ratio("abc 123 456 789") == 0.0
    assert arabic_ratio("ابت 123") == 1.0


def test_format_example_only_without_reasoning_channel():
    assert "Example of a correct reply" in system_prompt("math", "en", reasoning_channel=False, with_example=True)
    assert "Example" not in system_prompt("math", "en", reasoning_channel=True, with_example=True)
    assert "Example" not in system_prompt("math", "en", reasoning_channel=False, with_example=False)


def test_format_example_follows_think_lang_and_type():
    ar = system_prompt("knowledge", "ar", reasoning_channel=False, with_example=True)
    en = system_prompt("intent", "en", reasoning_channel=False, with_example=True)
    assert "عاصمة السعودية" in ar and "FINAL: C" in ar
    assert "weather" in en and "FINAL: other" in en
