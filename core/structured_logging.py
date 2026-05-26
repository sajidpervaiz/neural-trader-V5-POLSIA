"""Structured JSON logging with correlation ID support.

Provides:
- JSON formatter for loguru (stdout + file)
- Per-signal and per-order correlation IDs
- Context-local correlation tracking
"""
from __future__ import annotations

import contextvars
import json
import secrets
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from core.logging_utils import configure_sensitive_logging_redaction, redact_sensitive_data


# Context-local correlation ID
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def new_correlation_id(prefix: str = "") -> str:
    """Generate a new correlation ID and set it in context."""
    unique_part = secrets.token_hex(6)
    cid = f"{prefix}{unique_part}" if prefix else unique_part
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return _correlation_id.get()


def set_correlation_id(cid: str) -> None:
    """Manually set correlation ID (e.g., from incoming request)."""
    _correlation_id.set(cid)


def _json_formatter(record: dict[str, Any]) -> str:
    """Loguru custom formatter that outputs structured JSON."""
    ts = record["time"]
    level = record["level"]

    log_entry: dict[str, Any] = {
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "level": level.name,
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": redact_sensitive_data(record["message"]),
    }

    cid = _correlation_id.get()
    if cid:
        log_entry["correlation_id"] = cid

    # Include extra data if present
    extra = record.get("extra", {})
    if extra:
        log_entry["extra"] = {
            k: v for k, v in extra.items()
            if k not in ("_json",)
        }

    # Exception info
    exc = record.get("exception")
    if exc:
        log_entry["exception"] = {
            "type": str(exc.type.__name__) if exc.type else "",
            "value": str(exc.value) if exc.value else "",
        }

    return json.dumps(log_entry, default=str) + "\n"


def setup_structured_logging(
    log_level: str = "INFO",
    log_dir: str | Path = "logs",
    json_stdout: bool = False,
) -> None:
    """Configure loguru with structured JSON logging.

    Args:
        log_level: Minimum log level.
        log_dir: Directory for log files.
        json_stdout: If True, stdout gets JSON format too.  Otherwise human-readable.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    configure_sensitive_logging_redaction()

    # Stdout — human-readable by default, JSON if requested
    if json_stdout:
        logger.add(
            sys.stdout,
            level=log_level,
            format=_json_formatter,
            colorize=False,
        )
    else:
        logger.add(
            sys.stdout,
            level=log_level,
            colorize=True,
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{line}</cyan> — {message}",
        )

    # JSON file — always structured
    logger.add(
        log_path / "neural_trader_{time:YYYY-MM-DD}.json",
        level="DEBUG",
        format=_json_formatter,
        rotation="00:00",
        retention="30 days",
        compression="gz",
    )

    # Human-readable rotating file
    logger.add(
        log_path / "neural_trader_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        compression="gz",
    )

    logger.info("Structured logging initialized (level={}, json_stdout={})", log_level, json_stdout)
