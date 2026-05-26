from __future__ import annotations

from loguru import logger

from core.logging_utils import configure_sensitive_logging_redaction, redact_sensitive_data


def test_redacts_key_value_sensitive_fields() -> None:
    message = (
        "api_key=AKIAEXAMPLE api_secret='super-secret' passphrase=topsecret "
        "private_key=0xabc123 password=hunter2 token=tok_123 session_token=sess_456"
    )

    redacted = redact_sensitive_data(message)

    assert "AKIAEXAMPLE" not in redacted
    assert "super-secret" not in redacted
    assert "topsecret" not in redacted
    assert "0xabc123" not in redacted
    assert "hunter2" not in redacted
    assert "tok_123" not in redacted
    assert "sess_456" not in redacted
    assert "api_key=***" in redacted
    assert "api_secret='***'" in redacted
    assert "passphrase=***" in redacted
    assert "private_key=***" in redacted
    assert "password=***" in redacted
    assert "token=***" in redacted
    assert "session_token=***" in redacted


def test_redacts_json_and_dict_style_sensitive_fields() -> None:
    message = (
        '{"api_key": "json-key", "secret": "json-secret", '
        "'passphrase': 'dict-pass', 'private_key': 'dict-private'}"
    )

    redacted = redact_sensitive_data(message)

    assert "json-key" not in redacted
    assert "json-secret" not in redacted
    assert "dict-pass" not in redacted
    assert "dict-private" not in redacted
    assert '"api_key": "***"' in redacted
    assert '"secret": "***"' in redacted
    assert "'passphrase': '***'" in redacted
    assert "'private_key': '***'" in redacted


def test_redacts_sensitive_query_parameters_and_headers() -> None:
    message = (
        "https://example.test/callback?api_key=query-key&token=query-token "
        "Authorization: Bearer bearer-token X-API-Key: header-key"
    )

    redacted = redact_sensitive_data(message)

    assert "query-key" not in redacted
    assert "query-token" not in redacted
    assert "bearer-token" not in redacted
    assert "header-key" not in redacted
    assert "api_key=***" in redacted
    assert "token=***" in redacted
    assert "Authorization: Bearer ***" in redacted
    assert "X-API-Key: ***" in redacted


def test_redacts_labeled_order_identifiers() -> None:
    message = (
        "Order order-123 submitted orderId=exchange-456 client_order_id=client-789 "
        "idempotency_key=idem-abc oid=999"
    )

    redacted = redact_sensitive_data(message)

    assert "order-123" not in redacted
    assert "exchange-456" not in redacted
    assert "client-789" not in redacted
    assert "idem-abc" not in redacted
    assert "oid=999" not in redacted
    assert "Order *** submitted" in redacted
    assert "orderId=***" in redacted
    assert "client_order_id=***" in redacted
    assert "idempotency_key=***" in redacted
    assert "oid=***" in redacted


def test_preserves_non_sensitive_data() -> None:
    message = "BTC/USDT buy quantity=0.25 price=65000 mode=paper"

    assert redact_sensitive_data(message) == message


def test_non_string_values_are_supported() -> None:
    assert redact_sensitive_data(None) == "None"
    assert redact_sensitive_data({"api_key": "dict-key", "safe": "value"}).find("dict-key") == -1


def test_loguru_patcher_redacts_formatted_message_and_extra() -> None:
    captured: list[str] = []
    handler_id = None
    try:
        configure_sensitive_logging_redaction()
        handler_id = logger.add(lambda msg: captured.append(str(msg)), format="{message} | {extra}")

        logger.bind(api_key="bound-key").info("login api_secret={} orderId={}", "formatted-secret", "ord-1")

        output = "".join(captured)
        assert "bound-key" not in output
        assert "formatted-secret" not in output
        assert "ord-1" not in output
        assert "api_secret=***" in output
        assert "orderId=***" in output
        assert "api_key': '***'" in output or 'api_key": "***"' in output
    finally:
        if handler_id is not None:
            logger.remove(handler_id)
        logger.configure(patcher=None)
