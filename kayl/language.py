"""Detect whether a text is Arabic or English, to check the model reasoned in the requested language.

Simple and inspectable: count Arabic letters and Latin letters only. Digits and symbols are
ignored, because equations say nothing about the language.

Text copied from the question is removed first (strip_copied): a model reasoning in English
often quotes the Arabic question, which would otherwise make its reasoning look "mixed"."""

import re

AR_LETTER = re.compile(r"[ء-يٱ-ۓ]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
QUOTED = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”|«[^»\n]*»")
WORD = re.compile(r"\w+")

AR_THRESHOLD = 0.7   # Arabic letter share >= 0.7 -> "ar"
EN_THRESHOLD = 0.3   # Arabic letter share <= 0.3 -> "en"; in between -> "mixed"
COPY_NGRAM = 4       # 4 consecutive words found in the question count as copied


def strip_copied(text: str | None, source: str | None) -> str | None:
    """Remove from text (1) anything in quotes and (2) any 4 consecutive words found in source.
    What is left is the model's own words, which show the language it reasons in."""
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
    """Share of Arabic letters among all letters, or None when there are no letters."""
    if not text:
        return None
    ar, lat = len(AR_LETTER.findall(text)), len(LATIN_LETTER.findall(text))
    if ar + lat == 0:
        return None
    return ar / (ar + lat)


def detect_lang(text: str | None) -> str:
    """"ar", "en" or "mixed", or "none" when there is no text."""
    r = arabic_ratio(text)
    if r is None:
        return "none"
    if r >= AR_THRESHOLD:
        return "ar"
    if r <= EN_THRESHOLD:
        return "en"
    return "mixed"
