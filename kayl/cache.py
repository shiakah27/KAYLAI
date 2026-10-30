# -*- coding: utf-8 -*-
"""كاش على القرص لكل رد خام من النموذج.

المفتاح = بصمة (sha256) للنموذج + الإعدادات + الرسائل. نفس الطلب = نفس الملف،
فيمكن إعادة كل النتائج بدون استدعاءات جديدة."""

import hashlib
import json
from pathlib import Path


def cache_key(model_id: str, params: dict, messages: list[dict]) -> str:
    """بصمة ثابتة للطلب (ترتيب المفاتيح ثابت حتى لا تتغير البصمة)."""
    blob = json.dumps([model_id, params, messages], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    """ملف JSON لكل طلب داخل مجلد الكاش."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, record: dict) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, default=str), encoding="utf-8")
