"""Safe error handling helpers for logs and user-facing responses."""
from __future__ import annotations

import re
from typing import Any

from core.logging_utils import redact_sensitive_data

_GENERIC_ERROR = "Internal error"

_TRACEBACK_MARKERS = (
    "Traceback (most recent call last)",
    'File "',
    "File '",
    "line ",
)

_INTERNAL_PATH_RE = re.compile(r"(?:/[\w.\-]+)+/[\w.\-]+\.py")


def _looks_internal(message: str) -> bool:
    if not message.strip():
        return True
    if any(marker in message for marker in _TRACEBACK_MARKERS):
        return True
    if _INTERNAL_PATH_RE.search(message):
        return True
    return False


def sanitize_exception(exception: BaseException | Any) -> str:
    """Return a safe, redacted error message for an exception.

    Validation-style messages without internal paths or traceback fragments are
    preserved after redaction. Empty messages, traceback fragments, and internal
    filesystem paths collapse to a generic message.
    """
    raw = str(exception)
    redacted = redact_sensitive_data(raw)
    if _looks_internal(redacted):
        return _GENERIC_ERROR
    return redacted


def create_safe_error_response(error_code: str, message: Any) -> dict[str, str]:
    """Create a sanitized API error payload without leaking sensitive values."""
    safe_code = redact_sensitive_data(error_code)
    safe_message = redact_sensitive_data(message)
    if _looks_internal(safe_message):
        safe_message = _GENERIC_ERROR
    return {"error": safe_code, "message": safe_message}
