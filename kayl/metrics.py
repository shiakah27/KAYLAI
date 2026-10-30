# -*- coding: utf-8 -*-
"""حساب الحوسبة التقديرية (المقياس الأساسي) والتكلفة بالدولار (مقياس ثانوي).
دائمًا من أعداد التوكنز التي يرجعها مزود الخدمة، وليس من تقدير."""


def call_compute_flops(info: dict, input_tokens: int, output_tokens: int) -> float:
    """حوسبة تقديرية لاستدعاء واحد: 2 × المعاملات النشطة × كل التوكنز (FLOPs، تقدير)."""
    return 2 * info["active_params_b"] * 1e9 * (input_tokens + output_tokens)


def call_cost_usd(info: dict, input_tokens: int, output_tokens: int) -> float | None:
    """التكلفة بالدولار، أو None إذا كان سعر النموذج غير مؤكد."""
    if not info.get("price_verified"):
        return None
    return (input_tokens * info["input_price"] + output_tokens * info["output_price"]) / 1_000_000


def per_success(values: list[float | None], n_correct: int) -> float | None:
    """المجموع (يشمل المحاولات الفاشلة والتصعيد) ÷ عدد الأجوبة الصحيحة.
    None إذا لا يوجد جواب صحيح، أو إذا كانت أي قيمة None (مثلًا سعر غير مؤكد)."""
    if n_correct == 0 or any(v is None for v in values):
        return None
    return sum(values) / n_correct
