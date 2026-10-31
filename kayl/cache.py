"""Disk cache of every raw model response.

The key is a sha256 of the model id, the parameters and the messages, so the same request
always maps to the same file and every result can be reproduced without new API calls."""

import hashlib
import json
from pathlib import Path


def cache_key(model_id: str, params: dict, messages: list[dict]) -> str:
    """A stable fingerprint of the request (sorted keys, so dict order does not matter)."""
    blob = json.dumps([model_id, params, messages], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    """One JSON file per request inside the cache folder."""

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
