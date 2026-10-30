# -*- coding: utf-8 -*-
"""كشف لغة نص (عربي أم إنجليزي) لنتحقق: هل فكّر النموذج باللغة التي طلبناها؟

الطريقة بسيطة وقابلة للفحص: نعدّ الحروف العربية والحروف اللاتينية فقط
(الأرقام والرموز لا تُحسب، لأن المعادلات لا تدل على اللغة)."""

import re

AR_LETTER = re.compile(r"[ء-يٱ-ۓ]")
LATIN_LETTER = re.compile(r"[A-Za-z]")

AR_THRESHOLD = 0.7   # نسبة الحروف العربية >= 0.7  -> "ar"
EN_THRESHOLD = 0.3   # نسبة الحروف العربية <= 0.3  -> "en"، وبينهما "mixed"


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
