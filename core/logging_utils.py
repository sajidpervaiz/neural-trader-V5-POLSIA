"""Utilities for redacting sensitive values from log output.

The trading bot can log exceptions and formatted operational messages from many
modules. Keep this module dependency-free so it can be imported by low-level
code and used by Loguru's record patcher.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

_REDACTION = "***"

_SENSITIVE_KEYS = (
    "api_key",
    "apikey",
    "api-secret",
    "api_secret",
    "apisecret",
    "passphrase",
    "private_key",
    "private-key",
    "password",
    "secret",
    "token",
    "session_token",
    "session-token",
    "access_token",
    "access-token",
    "refresh_token",
    "refresh-token",
    "client_order_id",
    "client-order-id",
    "order_id",
    "order-id",
    "idempotency_key",
    "idempotency-key",
    "oid",
    "orderid",
)

_KEY_PATTERN = "|".join(re.escape(key) for key in sorted(_SENSITIVE_KEYS, key=len, reverse=True))

_KEY_VALUE_RE = re.compile(
    rf"(?P<key_quote>[\"']?)"
    rf"(?P<key>\b(?:{_KEY_PATTERN})\b)"
    rf"(?P=key_quote)"
    rf"(?P<sep>\s*[:=]\s*)"
    rf"(?P<value_quote>[\"']?)"
    rf"(?P<value>[^\"'\s,&}}\]]+)"
    rf"(?P=value_quote)",
    re.IGNORECASE,
)

_AUTH_BEARER_RE = re.compile(
    r"\b(?P<header>Authorization\s*:\s*Bearer\s+)(?P<value>[^\s,;]+)",
    re.IGNORECASE,
)

_API_KEY_HEADER_RE = re.compile(
    r"\b(?P<header>X-API-Key\s*:\s*)(?P<value>[^\s,;]+)",
    re.IGNORECASE,
)

_ORDER_TOKEN_RE = re.compile(
    r"\b(?P<label>Order|order)\s+(?P<value>[A-Za-z0-9][A-Za-z0-9_.:-]{2,})(?=\s|:|$)"
)

_ORDER_TOKEN_ALLOWLIST = {
    "already",
    "cancelled",
    "canceled",
    "created",
    "expired",
    "filled",
    "found",
    "not",
    "submitted",
}


def _redact_key_value(match: re.Match[str]) -> str:
    value_quote = match.group("value_quote") or ""
    return (
        f"{match.group('key_quote')}{match.group('key')}{match.group('key_quote')}"
        f"{match.group('sep')}{value_quote}{_REDACTION}{value_quote}"
    )


def _redact_bearer(match: re.Match[str]) -> str:
    return f"{match.group('header')}{_REDACTION}"


def _redact_order_token(match: re.Match[str]) -> str:
    value = match.group("value")
    if value.lower() in _ORDER_TOKEN_ALLOWLIST:
        return match.group(0)
    return f"{match.group('label')} {_REDACTION}"


def redact_sensitive_data(message: Any) -> str:
    """Return *message* as a string with sensitive values redacted.

    Redacts common key/value, JSON/dict-style, URL query, header, token, and
    labeled order identifier patterns while preserving non-sensitive text.
    """
    text = str(message)
    text = _KEY_VALUE_RE.sub(_redact_key_value, text)
    text = _AUTH_BEARER_RE.sub(_redact_bearer, text)
    text = _API_KEY_HEADER_RE.sub(_redact_bearer, text)
    text = _ORDER_TOKEN_RE.sub(_redact_order_token, text)
    return text


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_sensitive_data(value)
    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and re.fullmatch(_KEY_PATTERN, key, re.IGNORECASE):
                redacted[key] = _REDACTION
            else:
                redacted[key] = _redact_value(item)
        return redacted
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return value
    return value


def redact_log_record(record: Any) -> None:
    """Loguru patcher that redacts formatted messages and bound extras."""
    record["message"] = redact_sensitive_data(record.get("message", ""))
    if "extra" in record:
        record["extra"] = _redact_value(record["extra"])


def configure_sensitive_logging_redaction() -> None:
    """Install global Loguru redaction for subsequently emitted records."""
    logger.configure(patcher=redact_log_record)
