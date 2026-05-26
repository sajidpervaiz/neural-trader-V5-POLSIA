"""Regression tests for Config credential validation and runtime secret persistence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from core.config import Config


def _write_config(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _minimal_config(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "system": {
            "paper_mode": True,
            "log_level": "INFO",
        },
        "exchanges": {},
        "risk": {
            "min_balance_usd": 100,
            "default_leverage": 1.0,
        },
    }
    data.update(overrides)
    return data


def _contains_key(node: Any, forbidden: set[str]) -> bool:
    if isinstance(node, dict):
        return any(str(key) in forbidden or _contains_key(value, forbidden) for key, value in node.items())
    if isinstance(node, list):
        return any(_contains_key(value, forbidden) for value in node)
    return False


def test_live_enabled_exchange_missing_api_secret_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "settings.yaml"
    monkeypatch.setenv("NT_RUNTIME_CONFIG_PATH", str(tmp_path / "settings.runtime.yaml"))
    _write_config(
        config_path,
        _minimal_config(
            system={"paper_mode": False, "log_level": "INFO"},
            exchanges={
                "binance": {
                    "enabled": True,
                    "api_key": "present-key",
                    "api_secret": "",
                    "testnet": False,
                    "symbols": ["BTC/USDT:USDT"],
                }
            },
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        Config(config_path=config_path)

    assert "api_secret" in str(exc_info.value)
    assert "binance" in str(exc_info.value)


def test_runtime_overrides_do_not_persist_sensitive_fields_or_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "settings.yaml"
    runtime_path = tmp_path / "settings.runtime.yaml"
    monkeypatch.setenv("NT_RUNTIME_CONFIG_PATH", str(runtime_path))
    _write_config(
        config_path,
        _minimal_config(
            exchanges={
                "binance": {
                    "enabled": True,
                    "api_key": "literal-binance-key",
                    "api_secret": "literal-binance-secret",
                    "passphrase": "literal-binance-passphrase",
                    "testnet": True,
                    "demo": True,
                    "type": "futures",
                    "symbols": ["BTC/USDT:USDT"],
                }
            },
            dex={
                "enabled": True,
                "rpc_url": "https://example.invalid/rpc",
                "private_key": "literal-private-key",
            },
            ai_agent={
                "enabled": True,
                "provider": "claude",
                "model": "claude-sonnet-4-6",
                "api_key": "literal-ai-key",
            },
        ),
    )

    cfg = Config(config_path=config_path)
    written_path = cfg.persist_runtime_overrides()

    assert written_path == runtime_path
    payload = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    assert not _contains_key(payload, {"api_key", "api_secret", "passphrase", "private_key"})
    serialized = runtime_path.read_text(encoding="utf-8")
    assert "literal-binance-key" not in serialized
    assert "literal-binance-secret" not in serialized
    assert "literal-binance-passphrase" not in serialized
    assert "literal-private-key" not in serialized
    assert "literal-ai-key" not in serialized
