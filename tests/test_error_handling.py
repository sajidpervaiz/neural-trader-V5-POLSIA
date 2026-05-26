from __future__ import annotations

from core.error_handling import create_safe_error_response, sanitize_exception


class SecretBearingError(Exception):
    pass


def test_sanitize_exception_redacts_sensitive_values() -> None:
    exc = SecretBearingError(
        "exchange failed api_key=live-key api_secret='live-secret' "
        "Authorization: Bearer bearer-token orderId=exchange-order-1"
    )

    message = sanitize_exception(exc)

    assert "live-key" not in message
    assert "live-secret" not in message
    assert "bearer-token" not in message
    assert "exchange-order-1" not in message
    assert "api_key=***" in message
    assert "api_secret='***'" in message
    assert "Authorization: Bearer ***" in message
    assert "orderId=***" in message


def test_sanitize_exception_does_not_expose_traceback_or_internal_paths() -> None:
    exc = RuntimeError(
        "Traceback (most recent call last): File '/home/ubuntu/nueral-trader-V5/core/config.py', "
        "line 12, in load private_key=0xabc"
    )

    message = sanitize_exception(exc)

    assert "Traceback" not in message
    assert "/home/ubuntu" not in message
    assert "private_key=0xabc" not in message
    assert message == "Internal error"


def test_sanitize_exception_handles_empty_message() -> None:
    assert sanitize_exception(Exception()) == "Internal error"


def test_sanitize_exception_preserves_safe_validation_message() -> None:
    exc = ValueError("invalid symbol")

    assert sanitize_exception(exc) == "invalid symbol"


def test_create_safe_error_response_sanitizes_message() -> None:
    response = create_safe_error_response(
        "exchange_error",
        "failed with api_key=abc123 passphrase=secret-pass order_id=order-1",
    )

    assert response["error"] == "exchange_error"
    assert response["message"] == "failed with api_key=*** passphrase=*** order_id=***"
    assert "abc123" not in str(response)
    assert "secret-pass" not in str(response)
    assert "order-1" not in str(response)


def test_create_safe_error_response_sanitizes_error_code() -> None:
    response = create_safe_error_response("token=unsafe", "safe message")

    assert response == {"error": "token=***", "message": "safe message"}
