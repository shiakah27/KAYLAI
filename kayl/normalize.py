"""Number normalization, shared by data preparation and grading so both use the same rule."""

import re

# Arabic-Indic (٠١٢...) and Persian (۰۱۲...) digits to ASCII digits.
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_number(text: str | None) -> str | None:
    """The first number in the text, with separators removed, digits converted to ASCII and
    a trailing .0 dropped. None when the text has no number."""
    if text is None:
        return None
    t = text.translate(ARABIC_DIGITS).replace(",", "").replace("٬", "").replace("٫", ".").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    if not m:
        return None
    num = m.group(0)
    if "." in num:
        num = num.rstrip("0").rstrip(".")
    return num
