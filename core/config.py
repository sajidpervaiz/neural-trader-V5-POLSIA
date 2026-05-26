from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from core.config_schema import validate_config, AppConfig


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")
_SENSITIVE_RUNTIME_KEYS = {"api_key", "api_secret", "passphrase", "private_key"}

_SECRET_ENV_BY_PATH: dict[tuple[str, ...], str] = {
    ("exchanges", "binance", "api_key"): "BINANCE_API_KEY",
    ("exchanges", "binance", "api_secret"): "BINANCE_API_SECRET",
    ("exchanges", "bybit", "api_key"): "BYBIT_API_KEY",
    ("exchanges", "bybit", "api_secret"): "BYBIT_API_SECRET",
    ("exchanges", "okx", "api_key"): "OKX_API_KEY",
    ("exchanges", "okx", "api_secret"): "OKX_API_SECRET",
    ("exchanges", "okx", "passphrase"): "OKX_PASSPHRASE",
    ("exchanges", "kraken", "api_key"): "KRAKEN_API_KEY",
    ("exchanges", "kraken", "api_secret"): "KRAKEN_API_SECRET",
    ("dex", "private_key"): "DEX_PRIVATE_KEY",
    ("ai_agent", "api_key"): "ANTHROPIC_API_KEY",
    ("notifications", "telegram", "bot_token"): "TELEGRAM_BOT_TOKEN",
    ("monitoring", "dashboard_api", "auth", "api_key"): "DASHBOARD_API_KEY",
    ("monitoring", "telegram", "token"): "TELEGRAM_BOT_TOKEN",
    ("monitoring", "alerts", "telegram", "token"): "TELEGRAM_BOT_TOKEN",
    ("monitoring", "alerts", "discord", "webhook_url"): "DISCORD_WEBHOOK_URL",
    ("monitoring", "alerts", "webhook", "url"): "ALERT_WEBHOOK_URL",
    ("monitoring", "alerts", "webhook", "secret"): "ALERT_WEBHOOK_SECRET",
}


def _secret_placeholder(*path: str) -> str:
    env_name = _SECRET_ENV_BY_PATH.get(tuple(path))
    if env_name is None:
        env_name = "_".join(part.upper() for part in path if part)
    return f"${{{env_name}:-}}"


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def replace(m: re.Match) -> str:
            expr = m.group(1)
            if ":-" in expr:
                var_name, default = expr.split(":-", 1)
                env_value = os.environ.get(var_name)
                return env_value if env_value not in (None, "") else default
            if "-" in expr:
                var_name, default = expr.split("-", 1)
                return os.environ.get(var_name, default)
            return os.environ.get(expr, "")
        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged[key], value) if key in merged else value
        return merged
    return override


class Config:
    _instance: Config | None = None

    def __init__(self, path: str | Path | None = None, config_path: str | Path | None = None) -> None:
        # Backward compatibility: prefer explicit config_path if provided.
        if config_path is not None:
            path = config_path
        if path is None:
            path = Path(__file__).parent.parent / "config" / "settings.yaml"

        self._path = Path(path)
        self._runtime_override_path = self._resolve_runtime_override_path()

        with open(self._path) as fh:
            raw = yaml.safe_load(fh) or {}

        if self._runtime_override_path.exists():
            try:
                with open(self._runtime_override_path) as fh:
                    override_raw = yaml.safe_load(fh) or {}
                raw = _deep_merge(raw, override_raw)
                logger.debug("Applied runtime overrides from {}", self._runtime_override_path)
            except Exception as exc:
                logger.warning("Failed to load runtime overrides from {}: {}", self._runtime_override_path, exc)

        self._data: dict[str, Any] = _interpolate(raw)

        # Pydantic validation — fail fast on bad config
        try:
            self._validated: AppConfig = validate_config(self._data)
        except Exception as exc:
            logger.error("Config validation FAILED: {}", exc)
            raise SystemExit(f"FATAL: Invalid configuration — {exc}") from exc

        try:
            self._validate_credentials_present()
        except Exception as exc:
            logger.error("Credential validation FAILED: {}", exc)
            raise SystemExit(f"FATAL: Invalid configuration — {exc}") from exc

        logger.debug("Configuration loaded and validated from {}", self._path)

    @classmethod
    def get(cls, path: str | Path | None = None) -> "Config":
        if cls._instance is None:
            cls._instance = cls(path)
        return cls._instance

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def section(self, *keys: str) -> Any:
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                raise KeyError(f"Key '{k}' not found in config path {keys!r}")
            node = node[k]
        return node

    def get_value(self, *keys: str, default: Any = None) -> Any:
        try:
            return self.section(*keys)
        except (KeyError, TypeError):
            return default

    def _resolve_runtime_override_path(self) -> Path:
        env_path = os.getenv("NT_RUNTIME_CONFIG_PATH", "").strip()
        if env_path:
            return Path(env_path)
        return self._path.with_name(f"{self._path.stem}.runtime.yaml")

    def _validate_credentials_present(self) -> None:
        if self.paper_mode:
            return

        exchanges = self._data.get("exchanges", {})
        if isinstance(exchanges, dict):
            for venue, venue_cfg in exchanges.items():
                if not isinstance(venue_cfg, dict) or not bool(venue_cfg.get("enabled", False)):
                    continue
                missing = [
                    key for key in ("api_key", "api_secret")
                    if not str(venue_cfg.get(key, "")).strip()
                ]
                if str(venue).lower() == "okx" and not str(venue_cfg.get("passphrase", "")).strip():
                    missing.append("passphrase")
                if missing:
                    raise ValueError(
                        f"Exchange '{venue}' is enabled in non-paper mode but missing credentials: "
                        f"{', '.join(missing)}"
                    )

        dex_cfg = self._data.get("dex", {})
        if isinstance(dex_cfg, dict) and bool(dex_cfg.get("enabled", False)):
            if "private_key" in dex_cfg and not str(dex_cfg.get("private_key", "")).strip():
                raise ValueError("DEX is enabled in non-paper mode but missing credentials: private_key")

    def persist_runtime_overrides(self) -> Path:
        system_block: dict[str, Any] = {
            "paper_mode": self.paper_mode,
        }
        trading_mode = (self._data.get("system") or {}).get("trading_mode")
        if trading_mode:
            system_block["trading_mode"] = trading_mode
        payload: dict[str, Any] = {"system": system_block}

        exchanges = self._data.get("exchanges", {})
        if isinstance(exchanges, dict):
            payload["exchanges"] = {}
            for venue, venue_cfg in exchanges.items():
                if not isinstance(venue_cfg, dict):
                    continue
                persisted: dict[str, Any] = {}
                for key in ("enabled", "api_key", "api_secret", "passphrase", "testnet", "demo", "type"):
                    if key in venue_cfg and key not in _SENSITIVE_RUNTIME_KEYS:
                        persisted[key] = venue_cfg.get(key)
                if "symbols" in venue_cfg:
                    persisted["symbols"] = deepcopy(venue_cfg.get("symbols"))
                if persisted:
                    payload["exchanges"][venue] = persisted

        dex_cfg = self._data.get("dex", {})
        if isinstance(dex_cfg, dict):
            dex_payload: dict[str, Any] = {}
            for key in ("enabled", "rpc_url", "private_key"):
                if key in dex_cfg and key not in _SENSITIVE_RUNTIME_KEYS:
                    dex_payload[key] = dex_cfg.get(key)
            for venue in ("uniswap", "sushiswap", "dydx"):
                venue_cfg = dex_cfg.get(venue)
                if isinstance(venue_cfg, dict) and "enabled" in venue_cfg:
                    dex_payload[venue] = {"enabled": bool(venue_cfg.get("enabled", False))}
            if dex_payload:
                payload["dex"] = dex_payload

        notifications = self._data.get("notifications", {})
        if isinstance(notifications, dict):
            telegram_cfg = notifications.get("telegram", {})
            if isinstance(telegram_cfg, dict):
                payload["notifications"] = {
                    "telegram": {
                        "bot_token": _secret_placeholder("notifications", "telegram", "bot_token"),
                        "chat_id": telegram_cfg.get("chat_id", ""),
                    }
                }

        ai_cfg = self._data.get("ai_agent", {})
        if isinstance(ai_cfg, dict):
            payload["ai_agent"] = {
                "enabled": bool(ai_cfg.get("enabled", True)),
                "mode": ai_cfg.get("mode", "advisory"),
                "provider": ai_cfg.get("provider", "local"),
                "model": ai_cfg.get("model", "claude-3-5-sonnet-latest"),
                "timeout_seconds": float(ai_cfg.get("timeout_seconds", 8.0) or 8.0),
                "remote_weight": float(ai_cfg.get("remote_weight", 0.35) or 0.35),
            }

        self._runtime_override_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._runtime_override_path, "w") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
        try:
            os.chmod(self._runtime_override_path, 0o600)
        except OSError:
            pass
        logger.info("Persisted runtime overrides to {}", self._runtime_override_path)
        return self._runtime_override_path

    @property
    def paper_mode(self) -> bool:
        return bool(self._data.get("system", {}).get("paper_mode", True))

    @paper_mode.setter
    def paper_mode(self, value: bool) -> None:
        if "system" not in self._data:
            self._data["system"] = {}
        self._data["system"]["paper_mode"] = bool(value)

    @property
    def log_level(self) -> str:
        return str(self._data.get("system", {}).get("log_level", "INFO"))
