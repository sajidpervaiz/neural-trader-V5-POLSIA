#!/usr/bin/env python3
"""Preflight validation before enabling personal/live auto-trading.

This script intentionally checks dashboard exposure and secret hygiene even when
paper_mode is still true. Paper trading can still expose control endpoints and
credentials, so a locked-down personal paper config should pass while unsafe
paper/live configs fail loudly.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "config" / "settings.yaml"
EXCHANGE_ENV_REQUIREMENTS = {
    "binance": ("BINANCE_API_KEY", "BINANCE_API_SECRET"),
    "bybit": ("BYBIT_API_KEY", "BYBIT_API_SECRET"),
    "okx": ("OKX_API_KEY", "OKX_API_SECRET", "OKX_PASSPHRASE"),
    "kraken": ("KRAKEN_API_KEY", "KRAKEN_API_SECRET"),
}
LIVE_STORAGE_ENV_REQUIREMENTS = ("POSTGRES_PASSWORD",)

_SECRET_PATHS: tuple[tuple[str, ...], ...] = (
    ("storage", "postgres", "password"),
    ("exchanges", "*", "api_key"),
    ("exchanges", "*", "api_secret"),
    ("exchanges", "*", "passphrase"),
    ("ai_agent", "api_key"),
    ("notifications", "telegram", "bot_token"),
    ("monitoring", "dashboard_api", "auth", "api_key"),
    ("monitoring", "telegram", "token"),
    ("monitoring", "alerts", "telegram", "token"),
    ("monitoring", "alerts", "discord", "webhook_url"),
    ("monitoring", "alerts", "webhook", "url"),
    ("monitoring", "alerts", "webhook", "secret"),
    ("dex", "private_key"),
    ("dex", "networks", "*", "rpc_http"),
    ("dex", "networks", "*", "rpc_ws"),
    ("dex", "dydx", "mnemonic"),
    ("variational", "api_key"),
    ("variational", "api_secret"),
)

_ENV_REF_RE = re.compile(r"^\$\{[A-Z0-9_]+(?::-[^}]*)?\}$")
_ALLOWED_LOCAL_DASHBOARD_KEYS = {"local-dev-key", "local-dev-dashboard-key"}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged[key], value) if key in merged else value
        return merged
    return override


def _runtime_config_path(config_path: Path) -> Path:
    env_path = os.getenv("NT_RUNTIME_CONFIG_PATH", "").strip()
    if env_path:
        return Path(env_path).resolve()
    return config_path.with_name(f"{config_path.stem}.runtime.yaml")


def _fail(message: str) -> int:
    print(f"FAIL: {message}")
    return 1


def _required_env_for_enabled_exchanges(exchanges: dict[str, Any]) -> tuple[list[str], list[str]]:
    enabled_names: list[str] = []
    required: set[str] = set()

    for name, cfg in exchanges.items():
        if not isinstance(cfg, dict):
            continue
        if bool(cfg.get("enabled", False)):
            enabled_names.append(name)
            required.update(EXCHANGE_ENV_REQUIREMENTS.get(name, ()))

    return enabled_names, sorted(required)


def _walk_path(node: Any, pattern: tuple[str, ...]) -> list[Any]:
    if not pattern:
        return [node]
    head, *tail = pattern
    if head == "*":
        if not isinstance(node, dict):
            return []
        values: list[Any] = []
        for child in node.values():
            values.extend(_walk_path(child, tuple(tail)))
        return values
    if not isinstance(node, dict) or head not in node:
        return []
    return _walk_path(node[head], tuple(tail))


def _is_env_reference(value: str) -> bool:
    return bool(_ENV_REF_RE.match(value.strip()))


def _looks_like_plaintext_secret(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    if _is_env_reference(stripped):
        return False
    return True


def _find_plaintext_secret_paths(data: dict[str, Any]) -> list[str]:
    offenders: list[str] = []
    for path in _SECRET_PATHS:
        for value in _walk_path(data, path):
            if _looks_like_plaintext_secret(value):
                offenders.append(".".join(path))
                break
    return offenders


def _dashboard_auth_key_available(auth: dict[str, Any]) -> bool:
    api_key = str(auth.get("api_key", "") or "").strip()
    if not api_key:
        return False
    if not _is_env_reference(api_key):
        return True

    # Support ${VAR} and ${VAR:-default}; if default is non-empty it is usable.
    expr = api_key[2:-1]
    if ":-" in expr:
        env_name, default = expr.split(":-", 1)
        return bool(os.getenv(env_name) or default)
    return bool(os.getenv(expr))


def _dashboard_config(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    dashboard = ((data.get("monitoring") or {}).get("dashboard_api") or {})
    auth = dashboard.get("auth", {}) if isinstance(dashboard, dict) else {}
    return dashboard, auth


def _validate_dashboard_exposure(data: dict[str, Any]) -> str | None:
    dashboard, auth = _dashboard_config(data)
    if not isinstance(dashboard, dict):
        return "dashboard_api config must be a mapping"

    allow_origins = dashboard.get("allow_origins", [])
    if isinstance(allow_origins, str):
        allow_origins = [allow_origins]
    if any(str(origin).strip() == "*" for origin in allow_origins):
        return "dashboard CORS must not allow '*'"

    if not isinstance(auth, dict):
        return "dashboard API auth config must be a mapping"
    if not bool(auth.get("require_api_key", False)):
        return "dashboard API key auth must be enabled"
    return None


def _validate_dashboard_key(data: dict[str, Any]) -> str | None:
    _, auth = _dashboard_config(data)
    if not isinstance(auth, dict) or not _dashboard_auth_key_available(auth):
        return "dashboard API key must be provided via DASHBOARD_API_KEY or a non-empty configured value"
    return None


def _validate_with_application_schema(cfg_path: Path) -> str | None:
    """Run the same Config/Pydantic validation path used by main.py."""
    try:
        from core.config import Config

        Config._instance = None
        Config(path=cfg_path)
        Config._instance = None
        return None
    except SystemExit as exc:
        return f"application config schema validation failed: {exc}"
    except Exception as exc:
        return f"application config schema validation failed: {exc}"


def main() -> int:
    cfg_path = Path(os.getenv("NT_CONFIG_PATH", str(DEFAULT_CONFIG))).resolve()
    if not cfg_path.exists():
        return _fail(f"config not found: {cfg_path}")

    base_data = _load(cfg_path)
    runtime_path = _runtime_config_path(cfg_path)
    runtime_data: dict[str, Any] = {}
    if runtime_path.exists():
        runtime_data = _load(runtime_path)

    base_secret_paths = _find_plaintext_secret_paths(base_data)
    if base_secret_paths:
        return _fail("plain-text secret material in config: " + ", ".join(sorted(base_secret_paths)))
    runtime_secret_paths = _find_plaintext_secret_paths(runtime_data)
    if runtime_secret_paths:
        return _fail("plain-text secret material in runtime config: " + ", ".join(sorted(runtime_secret_paths)))

    data = _deep_merge(base_data, runtime_data)
    system = data.get("system", {})
    exchanges = data.get("exchanges", {})

    schema_error = _validate_with_application_schema(cfg_path)
    if schema_error:
        return _fail(schema_error)

    dashboard_error = _validate_dashboard_exposure(data)
    if dashboard_error:
        return _fail(dashboard_error)

    dashboard_key_error = _validate_dashboard_key(data)
    if dashboard_key_error:
        return _fail(dashboard_key_error)

    paper_mode = bool(system.get("paper_mode", True))
    enabled = [
        (name, cfg) for name, cfg in exchanges.items() if isinstance(cfg, dict) and bool(cfg.get("enabled", False))
    ]

    if paper_mode:
        print("PASS: personal production preflight checks succeeded")
        print(f"PASS: dashboard API/CORS and secret hygiene validated for {cfg_path}")
        return 0

    if not enabled:
        return _fail("no enabled exchanges configured")

    bad_testnet = [name for name, cfg in enabled if bool(cfg.get("testnet", True))]
    if bad_testnet:
        return _fail(f"enabled exchanges still using testnet: {', '.join(bad_testnet)}")
    bad_demo = [name for name, cfg in enabled if bool(cfg.get("demo", False))]
    if bad_demo:
        return _fail(f"enabled exchanges still using demo: {', '.join(bad_demo)}")

    enabled_names, required_env = _required_env_for_enabled_exchanges(exchanges)
    required_env = sorted(set(required_env).union(LIVE_STORAGE_ENV_REQUIREMENTS))
    missing_env = [key for key in required_env if not os.getenv(key)]
    if missing_env:
        return _fail(f"missing required env vars: {', '.join(missing_env)}")

    if os.getenv("LIVE_TRADING_CONFIRMED", "").lower() != "true":
        return _fail("LIVE_TRADING_CONFIRMED=true is required for real-money live preflight")

    print(f"PASS: live preflight checks succeeded for {cfg_path}")
    print(f"PASS: enabled exchanges validated: {', '.join(enabled_names)}")
    print("PASS: auto-trading mode can be started with NT_CONFIG_PATH set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
