# -*- coding: utf-8 -*-
"""توحيد الأرقام: نستخدمه في تجهيز البيانات وفي تصحيح أجوبة النماذج، حتى يكون المعيار واحدًا."""

import re

# تحويل الأرقام العربية-الهندية (٠١٢...) والفارسية (۰۱۲...) إلى أرقام لاتينية (012...)
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_number(text: str | None) -> str | None:
    """يحول نصًا إلى رقم موحد: يزيل الفواصل، ويحول الأرقام العربية، ويحذف .0 الزائدة.
    يرجع أول رقم في النص، أو None إذا لم يوجد رقم."""
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
