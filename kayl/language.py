# -*- coding: utf-8 -*-
"""كشف لغة نص (عربي أم إنجليزي) لنتحقق: هل فكّر النموذج باللغة التي طلبناها؟

الطريقة بسيطة وقابلة للفحص: نعدّ الحروف العربية والحروف اللاتينية فقط
(الأرقام والرموز لا تُحسب، لأن المعادلات لا تدل على اللغة).

قبل العدّ نحذف ما نسخه النموذج من السؤال (strip_copied)، لأن النموذج قد يفكر
بالإنجليزية لكنه يقتبس جملة السؤال العربية، فيبدو تفكيره "mixed" وهو ليس كذلك."""

import re

AR_LETTER = re.compile(r"[ء-يٱ-ۓ]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
QUOTED = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”|«[^»\n]*»")
WORD = re.compile(r"\w+")

AR_THRESHOLD = 0.7   # نسبة الحروف العربية >= 0.7  -> "ar"
EN_THRESHOLD = 0.3   # نسبة الحروف العربية <= 0.3  -> "en"، وبينهما "mixed"
COPY_NGRAM = 4       # 4 كلمات متتالية موجودة في السؤال = نص منسوخ


def strip_copied(text: str | None, source: str | None) -> str | None:
    """يحذف من text: (1) أي نص بين علامات تنصيص، (2) أي 4 كلمات متتالية موجودة في source.
    ما يبقى هو كلام النموذج نفسه، وهو ما يدل على لغة تفكيره."""
    if not text or not source:
        return text
    text = QUOTED.sub(" ", text)
    src_words = WORD.findall(source)
    src_ngrams = {tuple(src_words[i:i + COPY_NGRAM]) for i in range(len(src_words) - COPY_NGRAM + 1)}
    words = WORD.findall(text)
    copied = [False] * len(words)
    for i in range(len(words) - COPY_NGRAM + 1):
        if tuple(words[i:i + COPY_NGRAM]) in src_ngrams:
            copied[i:i + COPY_NGRAM] = [True] * COPY_NGRAM
    return " ".join(w for w, c in zip(words, copied) if not c)


def arabic_ratio(text: str | None) -> float | None:
    """نسبة الحروف العربية من كل الحروف. None إذا لا توجد حروف."""
    if not text:
        return None
    ar, lat = len(AR_LETTER.findall(text)), len(LATIN_LETTER.findall(text))
    if ar + lat == 0:
        return None
    return ar / (ar + lat)


def detect_lang(text: str | None) -> str:
    """يرجع "ar" أو "en" أو "mixed"، أو "none" إذا لا يوجد نص."""
    r = arabic_ratio(text)
    if r is None:
        return "none"
    if r >= AR_THRESHOLD:
        return "ar"
    if r <= EN_THRESHOLD:
        return "en"
    return "mixed"
