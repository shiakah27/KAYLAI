"""Read config.yaml and answer simple questions about each model."""

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path("config.yaml")


@lru_cache(maxsize=None)
def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Read the config once per process."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_info(cfg: dict, model: str) -> dict:
    """A model's entry in the config (price, active parameters, context window...)."""
    if model not in cfg["models"]:
        raise KeyError(f"unknown model: {model}")
    return cfg["models"][model]


def max_output_tokens(cfg: dict, model: str, effort: str | None = None) -> int:
    """Output cap: the model's per-effort value if set (e.g. high on 120b), then the model's
    own value, then the global default."""
    info = model_info(cfg, model)
    by_effort = info.get("max_output_tokens_by_effort") or {}
    if effort in by_effort:
        return by_effort[effort]
    return info.get("max_output_tokens", cfg["generation"]["max_output_tokens"])


def fits_context(cfg: dict, model: str, est_prompt_tokens: int, effort: str | None = None) -> bool:
    """Whether a task fits the model's context window:
    estimated prompt tokens × safety factor + output cap <= context window."""
    window = model_info(cfg, model)["context_window"]
    factor = cfg["context_check"]["safety_factor"]
    return est_prompt_tokens * factor + max_output_tokens(cfg, model, effort) <= window


def accuracy_floor(cfg: dict, profile: dict, task_type: str) -> tuple[float, str]:
    """Accuracy floor for a task type: baseline A's calibration accuracy minus one question
    (3.3 points at n = 30). Falls back to the fixed floor in the config when baseline A has
    no data for the type. Returns (floor, short explanation)."""
    floors = cfg["floors"]
    base = profile.get(task_type, {}).get(floors.get("baseline_config", ""))
    if not base:
        return floors["accuracy"], f"accuracy ≥ {floors['accuracy']:.0%} (fixed; no baseline A data)"
    margin = floors["accuracy_margin_questions"]
    floor = (base["correct"] - margin) / base["n"]
    return floor, (f"accuracy ≥ {floor:.1%} (baseline A {floors['baseline_config']} "
                   f"{base['correct']}/{base['n']} minus {margin} question)")
