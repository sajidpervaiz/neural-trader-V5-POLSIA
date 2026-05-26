from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from core.config import Config
from storage.db_handler import DBHandler

def test_db_handler_uses_real_encoded_dsn_and_separate_redacted_diagnostics():
    class _Cfg:
        def get_value(self, *keys, default=None):
            if keys == ("storage", "postgres"):
                return {
                    "user": "trade user",
                    "password": "p@ss:word/with#chars",
                    "host": "db.local",
                    "port": 5433,
                    "database": "nt live",
                    "pool_size": 2,
                }
            return default

    db = DBHandler(_Cfg())

    assert db._dsn == "postgresql://trade%20user:p%40ss%3Aword%2Fwith%23chars@db.local:5433/nt%20live"
    assert db._redacted_dsn == "postgresql://trade%20user:***@db.local:5433/nt%20live"
    assert "***" not in db._dsn
    assert "p%40ss%3Aword%2Fwith%23chars" not in db._redacted_dsn


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "config" / "settings.yaml"
PREFLIGHT = ROOT / "scripts" / "preflight_live_trading.py"


def _fresh_config(monkeypatch, tmp_path: Path) -> Config:
    runtime = tmp_path / "settings.runtime.yaml"
    monkeypatch.setenv("NT_RUNTIME_CONFIG_PATH", str(runtime))
    Config._instance = None
    cfg = Config(path=BASE_CONFIG)
    return cfg


def test_runtime_override_persistence_never_writes_plaintext_secrets(monkeypatch, tmp_path):
    cfg = _fresh_config(monkeypatch, tmp_path)
    cfg._data["exchanges"]["binance"]["api_key"] = "live_binance_key_should_not_be_written"
    cfg._data["exchanges"]["binance"]["api_secret"] = "live_binance_secret_should_not_be_written"
    cfg._data["exchanges"]["okx"]["passphrase"] = "okx_passphrase_should_not_be_written"
    cfg._data["ai_agent"]["api_key"] = "Bearer nvapi-secret-should-not-be-written"
    cfg._data["notifications"] = {"telegram": {"bot_token": "telegram_secret", "chat_id": "12345"}}

    out_path = cfg.persist_runtime_overrides()
    raw_text = out_path.read_text()

    assert "live_binance_key_should_not_be_written" not in raw_text
    assert "live_binance_secret_should_not_be_written" not in raw_text
    assert "okx_passphrase_should_not_be_written" not in raw_text
    assert "nvapi-secret-should-not-be-written" not in raw_text
    assert "telegram_secret" not in raw_text

    data = yaml.safe_load(raw_text)
    assert "api_key" not in data["exchanges"]["binance"]
    assert "api_secret" not in data["exchanges"]["binance"]
    assert "passphrase" not in data["exchanges"].get("okx", {})
    assert "api_key" not in data["ai_agent"]
    assert data["notifications"]["telegram"]["bot_token"] == "${TELEGRAM_BOT_TOKEN:-}"


def _write_config(path: Path, overrides: dict) -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text())
    # minimal deep merge for test overrides
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v
    merge(base, overrides)
    path.write_text(yaml.safe_dump(base, sort_keys=False))


def _run_preflight(config_path: Path, env: dict[str, str] | None = None):
    run_env = os.environ.copy()
    run_env.update(env or {})
    run_env["NT_CONFIG_PATH"] = str(config_path)
    return subprocess.run(
        [sys.executable, str(PREFLIGHT)],
        cwd=str(ROOT),
        env=run_env,
        text=True,
        capture_output=True,
        timeout=20,
    )


def test_preflight_rejects_dashboard_without_api_auth_even_in_paper(tmp_path):
    cfg_path = tmp_path / "settings.personal.yaml"
    _write_config(cfg_path, {
        "system": {"paper_mode": True},
        "monitoring": {"dashboard_api": {"auth": {"require_api_key": False, "api_key": ""}}},
    })

    result = _run_preflight(cfg_path)

    assert result.returncode == 1
    assert "dashboard API key auth must be enabled" in result.stdout


def test_preflight_rejects_wildcard_cors(tmp_path):
    cfg_path = tmp_path / "settings.personal.yaml"
    _write_config(cfg_path, {
        "system": {"paper_mode": True},
        "monitoring": {"dashboard_api": {
            "allow_origins": ["*"],
            "auth": {"require_api_key": True, "api_key": "${DASHBOARD_API_KEY:-}"},
        }},
    })

    result = _run_preflight(cfg_path)

    assert result.returncode == 1
    assert "dashboard CORS must not allow '*'" in result.stdout


def test_preflight_rejects_plaintext_secret_in_yaml(tmp_path):
    cfg_path = tmp_path / "settings.personal.yaml"
    _write_config(cfg_path, {
        "system": {"paper_mode": True},
        "exchanges": {"binance": {"api_key": "plain-text-secret-key", "api_secret": "${BINANCE_API_SECRET:-}"}},
        "monitoring": {"dashboard_api": {
            "allow_origins": ["http://127.0.0.1:8000"],
            "auth": {"require_api_key": True, "api_key": "${DASHBOARD_API_KEY:-}"},
        }},
    })

    result = _run_preflight(cfg_path)

    assert result.returncode == 1
    assert "plain-text secret material in config" in result.stdout


def test_preflight_accepts_locked_down_personal_paper_config(tmp_path):
    cfg_path = tmp_path / "settings.personal.yaml"
    _write_config(cfg_path, {
        "system": {"paper_mode": True},
        "exchanges": {"binance": {"enabled": True, "demo": True, "testnet": False, "api_key": "${BINANCE_API_KEY:-}", "api_secret": "${BINANCE_API_SECRET:-}"}},
        "dex": {"enabled": False},
        "ai_agent": {"enabled": False, "api_key": "${ANTHROPIC_API_KEY:-}"},
        "risk": {"max_daily_loss_pct": 0.01, "max_drawdown_pct": 0.03, "max_position_size_pct": 0.005, "risk_per_trade_pct": 0.0025, "max_open_positions": 2, "max_order_size_usd": 1000},
        "monitoring": {"dashboard_api": {
            "allow_origins": ["http://127.0.0.1:8000", "http://localhost:8000"],
            "auth": {"require_api_key": True, "api_key": "${DASHBOARD_API_KEY:-}"},
        }},
    })

    result = _run_preflight(cfg_path, env={"DASHBOARD_API_KEY": "local-dev-dashboard-key"})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: personal production preflight checks succeeded" in result.stdout


def test_preflight_rejects_live_mode_without_postgres_password(tmp_path):
    cfg_path = tmp_path / "settings.live.yaml"
    _write_config(cfg_path, {
        "system": {"paper_mode": False},
        "exchanges": {"binance": {
            "enabled": True,
            "demo": False,
            "testnet": False,
            "api_key": "${BINANCE_API_KEY:-}",
            "api_secret": "${BINANCE_API_SECRET:-}",
        }},
        "dex": {"enabled": False},
        "ai_agent": {"enabled": False, "api_key": "${ANTHROPIC_API_KEY:-}"},
        "storage": {"postgres": {"password": "${POSTGRES_PASSWORD}"}},
        "monitoring": {"dashboard_api": {
            "allow_origins": ["http://127.0.0.1:8000"],
            "auth": {"require_api_key": True, "api_key": "${DASHBOARD_API_KEY:-}"},
        }},
    })

    result = _run_preflight(cfg_path, env={
        "DASHBOARD_API_KEY": "local-dev-dashboard-key",
        "BINANCE_API_KEY": "dummy",
        "BINANCE_API_SECRET": "dummy",
        "POSTGRES_PASSWORD": "",
    })

    assert result.returncode == 1
    assert "missing required env vars: POSTGRES_PASSWORD" in result.stdout


def test_preflight_rejects_real_live_when_exchange_demo_enabled(tmp_path):
    cfg_path = tmp_path / "settings.live-demo.yaml"
    _write_config(cfg_path, {
        "system": {"paper_mode": False},
        "exchanges": {"binance": {
            "enabled": True,
            "demo": True,
            "testnet": False,
            "api_key": "${BINANCE_API_KEY:-}",
            "api_secret": "${BINANCE_API_SECRET:-}",
        }},
        "dex": {"enabled": False},
        "ai_agent": {"enabled": False, "api_key": "${ANTHROPIC_API_KEY:-}"},
        "storage": {"postgres": {"password": "${POSTGRES_PASSWORD}"}},
        "monitoring": {"dashboard_api": {
            "allow_origins": ["http://127.0.0.1:8000"],
            "auth": {"require_api_key": True, "api_key": "${DASHBOARD_API_KEY:-}"},
        }},
    })

    result = _run_preflight(cfg_path, env={
        "DASHBOARD_API_KEY": "local-dev-dashboard-key",
        "BINANCE_API_KEY": "dummy",
        "BINANCE_API_SECRET": "dummy",
        "POSTGRES_PASSWORD": "dummy",
    })

    assert result.returncode == 1
    assert "enabled exchanges still using demo" in result.stdout


def test_preflight_scans_runtime_override_for_plaintext_secrets(tmp_path):
    cfg_path = tmp_path / "settings.yaml"
    runtime_path = tmp_path / "settings.runtime.yaml"
    _write_config(cfg_path, {
        "system": {"paper_mode": True},
        "exchanges": {"binance": {"api_key": "${BINANCE_API_KEY:-}", "api_secret": "${BINANCE_API_SECRET:-}"}},
        "monitoring": {"dashboard_api": {
            "allow_origins": ["http://127.0.0.1:8000"],
            "auth": {"require_api_key": True, "api_key": "${DASHBOARD_API_KEY:-}"},
        }},
    })
    runtime_path.write_text(yaml.safe_dump({
        "exchanges": {"binance": {"api_key": "runtime-plain-secret", "api_secret": "${BINANCE_API_SECRET:-}"}},
    }))

    result = _run_preflight(cfg_path, env={"DASHBOARD_API_KEY": "local-dev-dashboard-key"})

    assert result.returncode == 1
    assert "plain-text secret material in runtime config" in result.stdout


def test_preflight_rejects_schema_invalid_live_config(tmp_path):
    cfg_path = tmp_path / "settings.schema-invalid-live.yaml"
    _write_config(cfg_path, {
        "system": {"paper_mode": False},
        "exchanges": {"binance": {
            "enabled": True,
            "demo": False,
            "testnet": False,
            "api_key": "${BINANCE_API_KEY:-}",
            "api_secret": "${BINANCE_API_SECRET:-}",
        }},
        "risk": {"max_drawdown_pct": 1.0},
        "dex": {"enabled": False},
        "ai_agent": {"enabled": False, "api_key": "${ANTHROPIC_API_KEY:-}"},
        "storage": {"postgres": {"password": "${POSTGRES_PASSWORD}"}},
        "monitoring": {"dashboard_api": {
            "allow_origins": ["http://127.0.0.1:8000"],
            "auth": {"require_api_key": True, "api_key": "${DASHBOARD_API_KEY:-}"},
        }},
    })

    result = _run_preflight(cfg_path, env={
        "DASHBOARD_API_KEY": "local-dev-dashboard-key",
        "BINANCE_API_KEY": "dummy",
        "BINANCE_API_SECRET": "dummy",
        "POSTGRES_PASSWORD": "dummy",
        "LIVE_TRADING_CONFIRMED": "true",
    })

    assert result.returncode == 1
    assert "application config schema validation failed" in result.stdout


def test_preflight_requires_live_trading_confirmed_for_real_money(tmp_path):
    cfg_path = tmp_path / "settings.live.yaml"
    _write_config(cfg_path, {
        "system": {"paper_mode": False},
        "exchanges": {"binance": {
            "enabled": True,
            "demo": False,
            "testnet": False,
            "api_key": "${BINANCE_API_KEY:-}",
            "api_secret": "${BINANCE_API_SECRET:-}",
        }},
        "dex": {"enabled": False},
        "ai_agent": {"enabled": False, "api_key": "${ANTHROPIC_API_KEY:-}"},
        "storage": {"postgres": {"password": "${POSTGRES_PASSWORD}"}},
        "monitoring": {"dashboard_api": {
            "allow_origins": ["http://127.0.0.1:8000"],
            "auth": {"require_api_key": True, "api_key": "${DASHBOARD_API_KEY:-}"},
        }},
    })

    result = _run_preflight(cfg_path, env={
        "DASHBOARD_API_KEY": "local-dev-dashboard-key",
        "BINANCE_API_KEY": "dummy",
        "BINANCE_API_SECRET": "dummy",
        "POSTGRES_PASSWORD": "dummy",
        "LIVE_TRADING_CONFIRMED": "",
    })

    assert result.returncode == 1
    assert "LIVE_TRADING_CONFIRMED=true" in result.stdout


def test_canary_live_example_loads_with_application_schema(monkeypatch):
    monkeypatch.setenv("DASHBOARD_API_KEY", "local-dev-dashboard-key")
    monkeypatch.setenv("BINANCE_API_KEY", "dummy")
    monkeypatch.setenv("BINANCE_API_SECRET", "dummy")
    monkeypatch.setenv("POSTGRES_PASSWORD", "dummy")
    monkeypatch.setenv("NT_RUNTIME_CONFIG_PATH", str(ROOT / ".nonexistent-canary-runtime.yaml"))
    Config._instance = None

    cfg = Config(path=ROOT / "config" / "settings.canary-live.example.yaml")

    assert cfg.paper_mode is False
    assert cfg.get_value("risk", "risk_per_trade_pct") == 0.001
    assert cfg.get_value("risk", "default_leverage") == 1
    assert cfg.get_value("signals", "min_score_threshold") == 0.80
    assert cfg.get_value("signals", "min_contributing_factors") == 3
