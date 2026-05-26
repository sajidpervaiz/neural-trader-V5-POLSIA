# Live Start Runbook — NUERAL-TRADER-5

This runbook is intentionally conservative. Do not skip steps. A bot that can place real orders without audit DB, reconciliation, and kill-switch visibility is not live-ready.

## 1. Prepare a local live config

Copy the canary template to a gitignored local file:

```bash
cp config/settings.canary-live.example.yaml config/settings.canary-live.yaml
```

Keep these defaults for the first real-money canary:

- `risk.max_open_positions: 1`
- `risk.default_leverage: 1`
- `risk.risk_per_trade_pct: 0.001`
- `monitoring.dashboard_api.auth.allow_manual_live_trading: false`
- `auto_trading.enabled: false` until the operator explicitly enables it after startup checks

## 2. Export secrets from environment only

```bash
export NT_CONFIG_PATH=/home/ubuntu/nueral-trader-V5/config/settings.canary-live.yaml
export DASHBOARD_API_KEY='replace-with-strong-random-key'
export BINANCE_API_KEY='restricted-binance-key'
export BINANCE_API_SECRET='restricted-binance-secret'
export POSTGRES_PASSWORD='real-postgres-password'
export LIVE_TRADING_CONFIRMED=true
```

Exchange API-key requirements:

- withdrawals disabled
- IP allowlist enabled where possible
- only required trading permissions
- smallest practical canary balance/subaccount

## 3. Verify PostgreSQL before starting the bot

```bash
psql -h localhost -U trader -d neural_trader -c 'select now();'
```

If this fails, stop. Live mode must not run without durable audit persistence.

## 4. Run tests

```bash
cd /home/ubuntu/nueral-trader-V5
python3 -m pytest tests/unit tests/integration -q
git diff --check
```

Expected:

- all tests pass
- no whitespace errors

## 5. Run live preflight

```bash
NT_CONFIG_PATH=/home/ubuntu/nueral-trader-V5/config/settings.canary-live.yaml \
DASHBOARD_API_KEY="$DASHBOARD_API_KEY" \
BINANCE_API_KEY="$BINANCE_API_KEY" \
BINANCE_API_SECRET="$BINANCE_API_SECRET" \
POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
LIVE_TRADING_CONFIRMED="$LIVE_TRADING_CONFIRMED" \
python3 scripts/preflight_live_trading.py
```

Expected:

- live preflight checks succeeded
- enabled exchanges validated
- no dashboard/CORS/secret failures

## 6. Start the bot

```bash
NT_CONFIG_PATH=/home/ubuntu/nueral-trader-V5/config/settings.canary-live.yaml \
LIVE_TRADING_CONFIRMED=true \
DASHBOARD_API_KEY="$DASHBOARD_API_KEY" \
BINANCE_API_KEY="$BINANCE_API_KEY" \
BINANCE_API_SECRET="$BINANCE_API_SECRET" \
POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
python3 main.py
```

Startup must show:

- database connected
- audit trail initialized
- startup validation passed
- startup reconciliation complete
- no `Database connection FAILED`
- no `db_unavailable`
- no `Audit log offline`

## 7. Check dashboard readiness

Open:

```text
http://127.0.0.1:8000/#api_key=<DASHBOARD_API_KEY>
```

Then query:

```bash
curl -H "X-API-Key: $DASHBOARD_API_KEY" http://127.0.0.1:8000/api/live/readiness
```

Required before enabling auto trading:

- `ready_for_live: true`
- audit_db ok
- exchange ok
- dashboard_auth ok
- risk ok
- reconciliation ok: no mismatches, no safe mode, no positions without exchange-side SL coverage
- user_stream ok: connected before any live entries are allowed

## 8. Canary rules

For the first live session:

- one symbol only
- max one position
- 1x leverage
- tiny notional
- dashboard monitored
- logs monitored
- PostgreSQL monitored
- no manual live dashboard trades

Stop immediately if any of these occur:

- DB unavailable
- reconciliation mismatch
- unknown order
- stale data critical
- repeated exchange errors
- unexpected position
- kill switch cannot close/cancel

## 9. Emergency stop

Dashboard:

- use Kill Switch to block new entries and cancel open orders
- independently verify exchange positions in the exchange UI/API
- if any position remains open, flatten it using the exchange UI/API or a separately verified reduce-only close path
- do not assume the bot is flat until exchange truth confirms zero exposure

CLI/API:

```bash
curl -X POST -H "X-API-Key: $DASHBOARD_API_KEY" http://127.0.0.1:8000/v1/kill
curl -H "X-API-Key: $DASHBOARD_API_KEY" http://127.0.0.1:8000/api/live/readiness
```

After any emergency, do not resume until reconciliation is clean and the incident is reviewed.
