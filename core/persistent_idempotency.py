import os
import time
import json
from dataclasses import asdict, is_dataclass
from typing import Optional, Dict, Any

from loguru import logger

from .idempotency import IdempotencyManager, IdempotencyRecord


def _json_default(obj: Any) -> Any:
    """JSON fallback for non-serializable values stored as idempotency results.

    Dataclasses (e.g. OrderManager passes Order) → dict. Everything else
    becomes a string. Idempotency only needs the *key*; result is a hint.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return asdict(obj)
        except Exception:
            return repr(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return repr(obj)

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
        except FileNotFoundError:
            return  # first run is normal, not an error
        except json.JSONDecodeError as exc:
            logger.critical(
                "Idempotency store CORRUPT at {} ({}). Starting empty — "
                "duplicate-order risk until next clean save.",
                self.filepath, exc,
            )
            return
        except Exception as exc:
            logger.error("Idempotency load failed at {}: {}", self.filepath, exc)
            return
        try:
            now = time.time()
            for key, rec in data.items():
                if rec.get("expires_at", 0) > now:
                    self.records[key] = IdempotencyRecord(
                        idempotency_key=key,
                        result=rec.get("result"),
                        created_at=rec.get("created_at", now),
                        expires_at=rec["expires_at"],
                        metadata=rec.get("metadata", {}),
                    )
        except Exception as exc:
            logger.error("Idempotency record reconstruction failed: {}", exc)

    def _save(self):
        if not self.filepath:
            return
        try:
            data = {
                k: {
                    "result": v.result,
                    "created_at": v.created_at,
                    "expires_at": v.expires_at,
                    "metadata": v.metadata,
                }
                for k, v in self.records.items()
            }
            # Atomic write: tmp + replace. Avoids corrupt half-written file
            # if the process crashes mid-write. _json_default handles
            # non-serializable result objects (e.g. Order dataclass).
            tmp = f"{self.filepath}.tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, default=_json_default)
            os.replace(tmp, self.filepath)
        except Exception as exc:
            logger.error(
                "Idempotency persistence FAILED at {}: {} — duplicate orders "
                "POSSIBLE on next restart. Investigate disk / permissions.",
                self.filepath, exc,
            )

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
