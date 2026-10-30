# -*- coding: utf-8 -*-
"""قراءة جواب النموذج: سطر FINAL الصارم، وسطر ANSWER، وقسم THINKING."""

import re

# صارم: السطر يبدأ بـ "FINAL:" بالضبط (بأحرف كبيرة). لا نقبل "**FINAL:**" أو "Final answer:".
# إذا لم يلتزم النموذج بالشكل، نعتبر الجواب غير صالح (هذه إشارة مفيدة للتحقق والتصعيد).
FINAL_RE = re.compile(r"^\s*FINAL:\s*(.+?)\s*$", re.MULTILINE)
ANSWER_RE = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$", re.MULTILINE)
THINKING_RE = re.compile(r"THINKING:\s*(.*?)(?=^\s*ANSWER:|^\s*FINAL:|\Z)", re.DOTALL | re.MULTILINE)


def parse_final(text: str | None) -> str | None:
    """قيمة آخر سطر FINAL، أو None إذا لا يوجد."""
    if not text:
        return None
    matches = FINAL_RE.findall(text)
    return matches[-1] if matches else None


def parse_answer(text: str | None) -> str | None:
    """قيمة آخر سطر ANSWER (الجملة العربية للمستخدم)، أو None."""
    if not text:
        return None
    matches = ANSWER_RE.findall(text)
    return matches[-1] if matches else None


def parse_thinking(text: str | None) -> str | None:
    """نص قسم THINKING (للنماذج بدون قناة تفكير)، أو None."""
    if not text:
        return None
    m = THINKING_RE.search(text)
    return m.group(1).strip() if m and m.group(1).strip() else None
