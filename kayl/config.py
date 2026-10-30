# -*- coding: utf-8 -*-
"""قراءة config.yaml، وأسئلة بسيطة عن كل نموذج: هل يدعم مستوى التفكير؟ هل تتسع المهمة لسياقه؟"""

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path("config.yaml")


@lru_cache(maxsize=None)
def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """يقرأ ملف الإعدادات مرة واحدة ويحفظه في الذاكرة."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_info(cfg: dict, model: str) -> dict:
    """معلومات نموذج من config.yaml (السعر، المعاملات النشطة، السياق...)."""
    if model not in cfg["models"]:
        raise KeyError(f"unknown model: {model}")
    return cfg["models"][model]


def max_output_tokens(cfg: dict, model: str, effort: str | None = None) -> int:
    """حد طول الجواب: قيمة مستوى التفكير لهذا النموذج إن وُجدت (مثل high في 120b)،
    ثم قيمة النموذج، وإلا القيمة العامة."""
    info = model_info(cfg, model)
    by_effort = info.get("max_output_tokens_by_effort") or {}
    if effort in by_effort:
        return by_effort[effort]
    return info.get("max_output_tokens", cfg["generation"]["max_output_tokens"])


def effort_options(cfg: dict, model: str) -> list[str | None]:
    """مستويات التفكير المسموحة لهذا النموذج. None = النموذج لا يدعم التحكم بالتفكير."""
    if model_info(cfg, model).get("supports_reasoning_effort"):
        return list(cfg["reasoning_efforts"])
    return [None]


def fits_context(cfg: dict, model: str, est_prompt_tokens: int, effort: str | None = None) -> bool:
    """هل تتسع المهمة لسياق النموذج؟
    التقدير × معامل أمان + حد الجواب <= نافذة السياق."""
    window = model_info(cfg, model)["context_window"]
    factor = cfg["context_check"]["safety_factor"]
    return est_prompt_tokens * factor + max_output_tokens(cfg, model, effort) <= window


def accuracy_floor(cfg: dict, profile: dict, task_type: str) -> tuple[float, str]:
    """حد الدقة لهذا النوع: دقة baseline A في المعايرة ناقص سؤال واحد (3.3 نقطة عند n=30).
    إذا لم تُقس baseline A لهذا النوع، نستخدم الحد الثابت في config. يرجع (الحد، شرح قصير)."""
    floors = cfg["floors"]
    base = profile.get(task_type, {}).get(floors.get("baseline_config", ""))
    if not base:
        return floors["accuracy"], f"accuracy ≥ {floors['accuracy']:.0%} (fixed; no baseline A data)"
    margin = floors["accuracy_margin_questions"]
    floor = (base["correct"] - margin) / base["n"]
    return floor, (f"accuracy ≥ {floor:.1%} (baseline A {floors['baseline_config']} "
                   f"{base['correct']}/{base['n']} minus {margin} question)")
