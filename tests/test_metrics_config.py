# -*- coding: utf-8 -*-
"""اختبارات حساب الحوسبة والتكلفة، وقواعد النماذج من config.yaml (السياق ومستوى التفكير)."""

import pytest

from kayl.config import effort_options, fits_context, load_config, max_output_tokens
from kayl.metrics import call_compute_flops, call_cost_usd, per_success

CFG = load_config()
GPT20 = CFG["models"]["openai/gpt-oss-20b"]
ALLAM = CFG["models"]["allam-2-7b"]


def test_compute_formula():
    # 2 × 3.6B × (197 + 93) = 2.088e12
    assert call_compute_flops(GPT20, 197, 93) == pytest.approx(2 * 3.6e9 * 290)
    assert call_compute_flops(ALLAM, 161, 127) == pytest.approx(2 * 7e9 * 288)


def test_cost_formula_verified_price():
    # (197 × 0.075 + 93 × 0.30) / 1e6
    assert call_cost_usd(GPT20, 197, 93) == pytest.approx((197 * 0.075 + 93 * 0.30) / 1e6)


def test_cost_is_none_without_verified_price():
    assert call_cost_usd(ALLAM, 161, 127) is None


def test_per_success_includes_failed_attempts():
    # 3 استدعاءات (واحد فاشل) وجوابان صحيحان: المجموع ÷ 2
    assert per_success([1.0, 2.0, 3.0], 2) == pytest.approx(3.0)


def test_per_success_none_cases():
    assert per_success([1.0, 2.0], 0) is None          # لا جواب صحيح
    assert per_success([1.0, None], 1) is None         # سعر غير مؤكد في أحد الاستدعاءات


def test_effort_only_for_gpt_oss():
    assert effort_options(CFG, "allam-2-7b") == [None]
    assert effort_options(CFG, "openai/gpt-oss-20b") == ["low", "medium", "high"]
    assert effort_options(CFG, "openai/gpt-oss-120b") == ["low", "medium", "high"]


def test_max_output_override():
    assert max_output_tokens(CFG, "allam-2-7b") == 1024
    assert max_output_tokens(CFG, "openai/gpt-oss-20b") == CFG["generation"]["max_output_tokens"]


def test_fits_context_allam():
    # 4096 >= est × 1.2 + 1024  ->  est <= 2560
    assert fits_context(CFG, "allam-2-7b", 2560)
    assert not fits_context(CFG, "allam-2-7b", 2561)
    assert fits_context(CFG, "openai/gpt-oss-20b", 50_000)
