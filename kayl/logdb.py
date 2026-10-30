# -*- coding: utf-8 -*-
"""سجل SQLite لكل استدعاء: الوقت، المهمة، الإعدادات، النموذج، التوكنز من الـ API،
زمن الاستجابة، التكلفة، الحوسبة، الصحة، والتصعيد."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = {
    "ts": "TEXT", "run_id": "TEXT", "config_name": "TEXT",
    "task_id": "TEXT", "task_type": "TEXT",
    "model": "TEXT", "effort": "TEXT", "think_lang": "TEXT",
    "input_tokens": "INTEGER", "output_tokens": "INTEGER", "reasoning_tokens": "INTEGER",
    "latency_s": "REAL", "cost_usd": "REAL", "compute_flops": "REAL",
    "final": "TEXT", "correct": "INTEGER", "escalated": "INTEGER",
    "reasoning_lang": "TEXT", "think_lang_followed": "INTEGER", "answer_in_arabic": "INTEGER",
    "finish_reason": "TEXT", "cached": "INTEGER", "error": "TEXT",
    "provider": "TEXT", "route": "TEXT",
}


def connect(path: str | Path) -> sqlite3.Connection:
    """يفتح قاعدة البيانات وينشئ جدول calls إذا لم يكن موجودًا."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    cols = ", ".join(f"{k} {v}" for k, v in COLUMNS.items())
    conn.execute(f"CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
    # قاعدة قديمة بدون أعمدة جديدة (مثل provider): نضيفها
    existing = {row[1] for row in conn.execute("PRAGMA table_info(calls)")}
    for k, v in COLUMNS.items():
        if k not in existing:
            conn.execute(f"ALTER TABLE calls ADD COLUMN {k} {v}")
    return conn


def log_call(conn: sqlite3.Connection, result, *, run_id: str, config_name: str,
             correct: bool | None, escalated: bool = False) -> None:
    """يسجل استدعاء واحدًا (result من نوع CallResult)."""
    row = {
        "ts": datetime.now(timezone.utc).isoformat(), "run_id": run_id, "config_name": config_name,
        "task_id": result.task_id, "task_type": result.task_type,
        "model": result.model, "effort": result.effort, "think_lang": result.think_lang,
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "latency_s": result.latency_s, "cost_usd": result.cost_usd, "compute_flops": result.compute_flops,
        "final": result.final, "correct": correct, "escalated": escalated,
        "reasoning_lang": result.reasoning_lang, "think_lang_followed": result.think_lang_followed,
        "answer_in_arabic": result.answer_in_arabic,
        "finish_reason": result.finish_reason, "cached": result.cached, "error": result.error,
        "provider": result.provider, "route": result.route,
    }
    keys = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO calls ({keys}) VALUES ({marks})", list(row.values()))
    conn.commit()
