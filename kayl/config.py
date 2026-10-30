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


def max_output_tokens(cfg: dict, model: str) -> int:
    """حد طول الجواب: قيمة النموذج إن وُجدت، وإلا القيمة العامة."""
    return model_info(cfg, model).get("max_output_tokens", cfg["generation"]["max_output_tokens"])


def effort_options(cfg: dict, model: str) -> list[str | None]:
    """مستويات التفكير المسموحة لهذا النموذج. None = النموذج لا يدعم التحكم بالتفكير."""
    if model_info(cfg, model).get("supports_reasoning_effort"):
        return list(cfg["reasoning_efforts"])
    return [None]


def fits_context(cfg: dict, model: str, est_prompt_tokens: int) -> bool:
    """هل تتسع المهمة لسياق النموذج؟
    التقدير × معامل أمان + حد الجواب <= نافذة السياق."""
    window = model_info(cfg, model)["context_window"]
    factor = cfg["context_check"]["safety_factor"]
    return est_prompt_tokens * factor + max_output_tokens(cfg, model) <= window
