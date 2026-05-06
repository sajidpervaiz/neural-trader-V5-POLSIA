import time
import json
from typing import Optional, Dict, Any
from .idempotency import IdempotencyManager, IdempotencyRecord

class PersistentIdempotencyManager(IdempotencyManager):
    """
    Idempotency manager with file-based persistence (MVP).
    """
    def __init__(self, ttl: int = 3600, max_size: int = 10000, filepath: str = "idempotency_store.json", save_interval: float = 30.0):
        super().__init__(ttl=ttl, max_size=max_size)
        self.filepath = filepath
        self._save_interval = save_interval
        self._last_save: float = 0.0
        self._dirty = False
        self._load()

    def _load(self):
        try:
            with open(self.filepath) as f:
                data = json.load(f)
            now = time.time()
            for key, rec in data.items():
                if rec["expires_at"] > now:
                    self.records[key] = IdempotencyRecord(
                        idempotency_key=key,
                        result=rec["result"],
                        created_at=rec["created_at"],
                        expires_at=rec["expires_at"],
                        metadata=rec.get("metadata", {})
                    )
        except Exception:
            pass

    def _save(self):
        try:
            data = {k: {
                "result": v.result,
                "created_at": v.created_at,
                "expires_at": v.expires_at,
                "metadata": v.metadata
            } for k, v in self.records.items()}
            with open(self.filepath, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _maybe_save(self):
        self._dirty = True
        now = time.time()
        if now - self._last_save >= self._save_interval:
            self._save()
            self._last_save = now
            self._dirty = False

    def check_and_set(self, idempotency_key: str) -> bool:
        res = super().check_and_set(idempotency_key)
        self._maybe_save()
        return res

    def set_result(self, idempotency_key: str, result: Any, metadata: Optional[Dict] = None) -> None:
        super().set_result(idempotency_key, result, metadata)
        self._maybe_save()

    def delete(self, idempotency_key: str) -> bool:
        res = super().delete(idempotency_key)
        self._maybe_save()
        return res
