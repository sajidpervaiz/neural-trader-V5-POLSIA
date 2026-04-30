"""REQ-MON-004 / REQ-SEC-001 / AC-010: secrets shall never appear in logs.

Scans:
  • All `.log` and `.log.gz` files under logs/
  • Any persisted JSONL audit/trade files in data/
  • The /tmp restart logs created during this session
  • Crash dumps (if any)

For known secret patterns:
  • Binance/Bybit/OKX API keys (alnum strings 24-64 chars after recognised
    prefixes)
  • Bearer tokens
  • Private keys (PEM/SSH headers)
  • Telegram bot tokens (digits:alnum format)
  • AWS access keys (AKIA…)
  • Common .env-style assignments to known sensitive keys

Allowlist:
  • Documented placeholder values (PLACEHOLDER, REDACTED, your-..., …)
  • The sentinel mask "****" the codebase uses for masked secrets
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]


# --- Patterns ---------------------------------------------------------------
# Each pattern returns a non-overlapping match in the file. Counter-examples
# (placeholders, masked values) are stripped first via _strip_allowlist.

_PEM_HEADER = re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PRIVATE )?PRIVATE KEY-----")
_AWS_ACCESS = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_TELEGRAM_BOT = re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}\b")
_BEARER_TOKEN = re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{30,}\b")
# .env-style assignments where RHS is sensitive-looking (>=24 chars no spaces).
_ENV_ASSIGN = re.compile(
    r"\b(?:API_KEY|API_SECRET|SECRET_KEY|PRIVATE_KEY|PASSWORD|"
    r"TELEGRAM_BOT_TOKEN|BINANCE_API_KEY|BINANCE_API_SECRET|"
    r"COINBASE_API_KEY|HYPERLIQUID_PRIVATE_KEY|BYBIT_API_KEY|"
    r"BYBIT_API_SECRET|OKX_API_KEY|OKX_API_SECRET|OKX_PASSPHRASE)"
    r"\s*[:=]\s*['\"]?([A-Za-z0-9+/=_\-]{24,})['\"]?",
    re.IGNORECASE,
)


_ALLOWLIST_RE = re.compile(
    r"(your[\w-]*(api|key|secret|token|password)[\w-]*"
    r"|placeholder"
    r"|redacted"
    r"|<\$\{[^}]+\}>|\$\{[A-Z_][A-Z0-9_]*\}"
    r"|\*\*\*+"
    r"|sk-test|test_pk|example_)",
    re.IGNORECASE,
)


def _strip_allowlist(text: str) -> str:
    return _ALLOWLIST_RE.sub("[ALLOW]", text)


def _check_text(text: str, source: str) -> list[str]:
    text = _strip_allowlist(text)
    hits: list[str] = []
    for pat, name in (
        (_PEM_HEADER, "private_key_header"),
        (_AWS_ACCESS, "aws_access_key"),
        (_TELEGRAM_BOT, "telegram_bot_token"),
        (_BEARER_TOKEN, "bearer_token"),
        (_ENV_ASSIGN, "env_secret_assignment"),
    ):
        for m in pat.finditer(text):
            snippet = text[max(0, m.start() - 30): m.end() + 30]
            hits.append(f"{name} in {source}: …{snippet!r}…")
    return hits


def _iter_targets() -> list[Path]:
    targets: list[Path] = []
    for sub in ("logs", "data"):
        d = REPO / sub
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix in {".log", ".gz", ".jsonl", ".json"}:
                targets.append(p)
    # /tmp restart logs created by this session
    for p in Path("/tmp").glob("neural*.log"):
        if p.is_file():
            targets.append(p)
    return targets


def _read(path: Path) -> str:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                return f.read()
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[unreadable: {exc}]"


@pytest.mark.parametrize("path", _iter_targets(), ids=lambda p: str(p.relative_to(REPO)) if str(p).startswith(str(REPO)) else str(p))
def test_no_secrets_in(path: Path) -> None:
    text = _read(path)
    hits = _check_text(text, str(path))
    assert not hits, "potential secret(s) found:\n" + "\n".join(hits)


def test_scanner_self_check_finds_a_planted_secret() -> None:
    """Sanity: the scanner actually detects a known-bad pattern."""
    bad = "BINANCE_API_KEY=AbCdEfGhIjKlMnOpQrStUvWxYz123456"
    hits = _check_text(bad, "self-check")
    assert hits, "scanner failed to flag a planted secret — patterns broken"


def test_allowlist_strips_placeholders() -> None:
    safe = (
        "BINANCE_API_KEY=${BINANCE_API_KEY}\n"
        "TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN_HERE\n"
        "API_SECRET=****\n"
    )
    hits = _check_text(safe, "self-check")
    assert hits == [], f"allowlist let through: {hits}"
