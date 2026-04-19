# ARMS Tab Redesign + Production Hardening

**Status:** Approved 2026-04-19
**Target files:**
- `interface/static/index.html` — `renderArms()` at line 1760, tab wiring at line 819
- `interface/dashboard_api.py` — new endpoint group `/api/arms/*`
- `execution/advanced_orders.py` — new module (TWAP / Iceberg / Shadow-SL)
- `execution/cex_executor.py` — wire slicing into `place_order()`
- `main.py` — wire `startup_validation.py` on boot
- `config/settings.yaml` — new `arms.stress_scenarios`, `execution.slicing`, `execution.shadow_sl` blocks

## Goal

1. Rebuild ARMS tab to match 3×3 panel reference image.
2. Back every panel with real backend data (no placeholders).
3. Add TWAP / Iceberg / Shadow-SL execution styles.
4. Add live-API-key entry flow that auto-switches paper → live and fetches real balances.
5. Wire existing `startup_validation.py` so boot fails fast on misconfiguration.

## Architecture

Five sub-projects, executed in dependency order **C → B → A → D → E**:

| # | Sub-project | Depends on |
|---|-------------|------------|
| C | Execution slicing module (`advanced_orders.py`) | — |
| B | New `/api/arms/*` endpoints | C (execution endpoint reads `scheduler_state()`) |
| A | ARMS 3×3 frontend grid | B |
| D | Live-key entry + auto-switch | — |
| E | Startup validation wire-up | — |

---

## Sub-project A — ARMS Tab 3×3 Grid (frontend)

**Target:** `interface/static/index.html` — replace `renderArms()` at line 1760; extend tab load at line 819.

### Layout

CSS grid `grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px`, stacks to 1 col below 1100px. New class prefix `.arms2-*` (keeps old `.arms-*` classes in place until removed at end of sub-project).

### 9 Panels (row-major)

| Row | Left | Middle | Right |
|-----|------|--------|-------|
| 1 | **Risk Monitor** | **Guardrails** | **Latency** |
| 2 | **Regime** | **Session** | **Quality Score** |
| 3 | **ARMS Detail** (tabs: Tier Risk / Exposure / Dyn. Lev) | **Stress Test** (4 scenarios) | **Execution** (TWAP / Iceberg / Shadow-SL) |

### Panel Structure

Each panel: `<div class="arms2-panel">` with:
- `.arms2-head` — icon + title (left) + status pill (right)
- `.arms2-body` — content

### Data Binding

Single `Promise.all` in extended `loadArms()`:

```js
const [snap, guard, lat, regime, session, quality, stress, exec, weights, prewarm] =
  await Promise.all([
    fetch('/api/risk/arms/snapshot').then(r=>r.json()),
    fetch('/api/guardrails').then(r=>r.json()),
    fetch('/api/latency').then(r=>r.json()),
    fetch('/api/regime').then(r=>r.json()),
    fetch('/api/session').then(r=>r.json()),
    fetch('/api/quality').then(r=>r.json()),
    fetch('/api/arms/stress-test').then(r=>r.json()),
    fetch('/api/arms/execution').then(r=>r.json()),
    fetch('/api/arms/weights').then(r=>r.json()),
    fetch('/api/arms/prewarm').then(r=>r.json()),
  ]);
```

### Colors

All use existing theme vars. Status pill colors:
- `NORMAL` / `SAFE` / `OK` → `--green`
- `WARN` / `ELEVATED` → `--orange`
- `TRIPPED` / `DANGER` → `--red`
- Neutral info → `--blue`

### Polling

Reuse existing 3s interval. One `loadArms()` call per tick hits 10 endpoints in parallel.

---

## Sub-project B — New Backend Endpoints

All added to `interface/dashboard_api.py` inside the existing route-registration function. No new files.

### `GET /api/arms/stress-test`

Deterministic simulation over current open positions.

**Response:**
```json
{
  "scenarios": [
    {
      "id": "btc_flash_crash_10",
      "label": "-10% BTC flash crash",
      "loss_pct": -4.2,
      "worst_symbol": "BTC/USDT:USDT",
      "worst_side": "long",
      "recovery_hours": 18,
      "status": "ok"
    },
    ...
  ],
  "scenarios_count": 4,
  "equity": 99849.9,
  "generated_at": "2026-04-19T19:12:00Z"
}
```

**Sim logic:**
1. Load open positions from `risk_manager.positions`
2. For each scenario in `config.arms.stress_scenarios`:
   - Apply `price_shock_pct` to positions matching `symbol_filter`
   - For `outage` type: compute slippage cost = `positions_notional * outage_slippage_pct`
   - For `funding_spike` type: `funding_pct * |notional| * duration_h / 8`
   - Sum equity delta; identify worst-hit single position
3. Recovery hours = `|loss_usd| / (rolling_30d_avg_daily_return_usd)`, floor to 1h, cap to 720h (30d). If no history, use 24h default.
4. Status: `ok` if `|loss_pct| < 5%`, `warn` if `< 15%`, `danger` otherwise.

**Scenario config** in `config/settings.yaml`:
```yaml
arms:
  stress_scenarios:
    - id: btc_flash_crash_10
      label: "-10% BTC flash crash"
      type: price_shock
      symbol_filter: ["BTC/USDT", "BTC/USDT:USDT"]
      price_shock_pct: -0.10
    - id: market_correction_25
      label: "-25% market correction"
      type: price_shock
      symbol_filter: "*"
      price_shock_pct: -0.25
    - id: funding_spike_05
      label: "Funding spike +0.5%"
      type: funding_spike
      funding_pct: 0.005
      duration_h: 8
    - id: exchange_outage_1h
      label: "Exchange outage 1h"
      type: outage
      duration_h: 1
      outage_slippage_pct: 0.02
```

### `GET /api/arms/execution`

Current state of advanced-order schedulers.

**Response:**
```json
{
  "twap": {
    "active": 0,
    "orders": [],
    "avg_latency_ms": 195,
    "status": "active"
  },
  "iceberg": {
    "active": 0,
    "orders": [],
    "reveal_pct": 0.20,
    "status": "armed"
  },
  "shadow_sl": {
    "watching": 0,
    "status": "monitoring"
  },
  "prewarm_on": true
}
```

Reads from module in sub-project C (`advanced_orders.scheduler_state()`).

### `GET /api/arms/weights`

Adaptive signal-weight breakdown.

**Response:**
```json
{
  "weights": {
    "technical": 0.35,
    "ml": 0.30,
    "sentiment": 0.10,
    "macro": 0.05,
    "news": 0.10,
    "orderbook": 0.10
  },
  "profile_name": "balanced_trend_up",
  "regime_hint": "STRONG_TREND_UP"
}
```

Reads `signal_generator._adaptive_weights` (internal dict) or falls back to config defaults. If `signal_generator` is None, returns zeros with `profile_name: "unavailable"`.

### `GET /api/arms/prewarm`

Consolidated latency panel data.

**Response:**
```json
{
  "ws_feed_lag_ms": 12,
  "order_exec_avg_ms": 195,
  "order_exec_p95_ms": 340,
  "last_order_ms": 182,
  "cache_hit_rate": 0.87,
  "cache_age_sec": 1.4,
  "prewarm_active": true,
  "prewarm_before_ms": 2150,
  "prewarm_after_ms": 195,
  "status": "normal"
}
```

Pulls from existing `/api/latency` source + new counters added to `cex_executor` (track p95, cache age).

Status: `normal` if `order_exec_p95_ms < 500`, `elevated` if `< 1000`, `degraded` otherwise.

---

## Sub-project C — Execution Slicing

**New file:** `execution/advanced_orders.py`

### Classes

#### `TWAPScheduler`
- `schedule(order, n_slices, window_sec)` — splits order into `n_slices`, posts at `window_sec / n_slices` intervals
- `active_orders` — dict of in-flight TWAP parents
- `state()` — returns list for dashboard
- Cancellation: parent cancel cascades to child slices

#### `IcebergSlicer`
- `submit(order, reveal_pct)` — posts only `reveal_pct` of quantity
- On fill notification: posts next slice until remaining qty = 0
- `active_orders` — dict keyed by parent order id
- `state()` — returns list for dashboard

#### `ShadowStopLoss`
- `attach(position, stop_price, side)` — registers watchdog
- Background task (asyncio) polls mark price every 1s
- On breach: invokes `executor.close_position(position)`
- `state()` — returns list of watched positions

#### `scheduler_state()` — module-level
Returns combined state dict consumed by `/api/arms/execution`.

### Integration

**In `execution/cex_executor.py: place_order()`:**
```python
style = getattr(order, 'execution_style', 'market')
if style == 'twap':
    return await self.twap_scheduler.schedule(order, ...)
elif style == 'iceberg':
    return await self.iceberg_slicer.submit(order, ...)
else:
    return await self._place_order_raw(order)  # existing logic
```

**On position creation (post-fill):**
```python
if self.config.execution.shadow_sl.enabled and order.stop_loss:
    self.shadow_sl.attach(position, order.stop_loss, position.direction)
```

### Config (`config/settings.yaml`)

```yaml
execution:
  slicing:
    enabled: false           # master flag — off by default
    twap:
      default_slices: 5
      default_window_sec: 300
    iceberg:
      default_reveal_pct: 0.20
  shadow_sl:
    enabled: true            # on by default — local watchdog is always safer
    poll_interval_sec: 1.0
```

### Order dataclass extension

In `execution/order_manager.py`, `Order` dataclass:
```python
execution_style: str = "market"   # "market" | "twap" | "iceberg" | "limit"
twap_slices: int | None = None
twap_window_sec: int | None = None
iceberg_reveal_pct: float | None = None
```

---

## Sub-project D — Live-Key Entry + Auto Mode-Switch

### Frontend

New Settings modal triggered from a gear-icon button in the header. If no gear exists, add `<button id="settingsBtn">⚙</button>` next to the mode toggle.

Modal fields:
- Exchange: dropdown (Binance / Bybit / OKX / Hyperliquid)
- API Key: text input
- API Secret: password input
- Testnet: checkbox
- Submit button

On submit: `POST /api/config/keys`. On 200 response: close modal, toast success, auto-poll balance. On 400: show inline error.

### Backend `POST /api/config/keys`

Request:
```json
{"exchange": "binance", "api_key": "...", "api_secret": "...", "testnet": false}
```

Flow:
1. Instantiate throwaway `ccxt.binance({apiKey, secret, ...})`
2. Call `fetch_balance()` — 5s timeout
3. **Success path:**
   - Encrypt secrets with `cryptography.Fernet` (master key from `.keyfile`, auto-generated on first boot with 0600 perms)
   - Write to `.env.local` (gitignored) — format: `BINANCE_API_KEY=<encrypted>`
   - Reload exchange config in `cex_executor`
   - Auto-invoke existing `/api/mode/toggle` to flip to live
   - Return `{success: true, balance: {...}}`
4. **Failure path:**
   - Return `{success: false, error: "Invalid API key"}`
   - Do not persist anything
   - Keep paper mode

### Encryption

```python
from cryptography.fernet import Fernet

def _load_master_key():
    path = Path(".keyfile")
    if not path.exists():
        key = Fernet.generate_key()
        path.write_bytes(key)
        path.chmod(0o600)
    return path.read_bytes()
```

### Security

- `.env.local` and `.keyfile` added to `.gitignore`
- Secrets never logged (log `api_key[:4] + "***"` only)
- POST endpoint requires existing `/api/clientkey` auth header
- Rate-limited to 5 attempts / minute per IP

---

## Sub-project E — Startup Validation Wire-Up

**Current state:** `startup_validation.py` exists but is not invoked (per memory).

### Changes

`main.py` — in the FastAPI `startup_event`:
```python
from startup_validation import run_startup_checks

@app.on_event("startup")
async def startup_event():
    result = run_startup_checks()
    app.state.startup_result = result
    if result.has_critical_failures:
        logger.error("Startup validation FAILED:", result.critical)
        sys.exit(1)
    for warn in result.warnings:
        logger.warning(warn)
    # ... existing startup logic ...
```

### Checks added to `startup_validation.py` (if missing)

- Config YAML schema valid
- All referenced env vars resolve (or skip if `enabled: false`)
- Models directory contains expected `.pkl`/`.lgb` files
- DB file writable
- Free disk space > 1GB
- NTP offset < 5s (warn only)

### Health exposure

Add to `/api/health/detailed`:
```json
{"startup": {"ok": true, "warnings": [...], "ran_at": "..."}}
```

---

## Data Flow (summary)

```
Browser (ARMS tab)
   │
   ├── loadArms() Promise.all ─── 10 endpoints
   │                                   │
   │                                   ├── /api/risk/arms/snapshot ──► risk_manager
   │                                   ├── /api/guardrails ──────────► risk_manager
   │                                   ├── /api/latency ─────────────► executor stats
   │                                   ├── /api/regime ──────────────► regime_detector
   │                                   ├── /api/session ─────────────► session_manager
   │                                   ├── /api/quality ─────────────► signal_generator
   │                                   ├── /api/arms/stress-test ────► NEW: scenario sim
   │                                   ├── /api/arms/execution ──────► NEW: advanced_orders
   │                                   ├── /api/arms/weights ────────► signal_generator
   │                                   └── /api/arms/prewarm ────────► NEW: latency consolidated
   │
   └── Settings modal → POST /api/config/keys → ccxt validate → encrypt → reload → auto mode-flip
```

## Error Handling

- All new endpoints return `{error: "..."}` with 500 on exception, never crash the server
- Frontend panels show `<div class="arms2-unavailable">—</div>` if their endpoint errors; other panels render normally
- Stress-test with no open positions: returns all scenarios with `loss_pct: 0, status: "ok", note: "no open positions"`
- Execution module failures do not block regular market orders (try/except with fallback to raw placement)

## Testing

- New unit tests `tests/test_stress_test.py`: scenario sim with fixed positions, assert expected losses
- New unit tests `tests/test_advanced_orders.py`: TWAP slicing math, iceberg refill logic, shadow-SL breach detection
- New integration test `tests/test_config_keys.py`: POST with bad key → no persistence; good key (mocked ccxt) → persistence + mode flip
- Manual UI test: hard-refresh browser, verify all 9 panels render, status pills correct colors

## Non-Goals

- Multi-exchange smart order routing
- Co-location / FIX protocol
- New ML models or new signal layers
- Database schema changes
- Rewriting Chart / Pipeline / Portfolio / News / Logs / other tabs
- Self-improvement advisor auto-applying changes (deferred — existing `/api/efficiency/*` stays advisory)
- Keyring / HSM secret storage (file + Fernet is sufficient for single-operator deployment)

## Order of Implementation

1. **C** — `execution/advanced_orders.py` (blockable by tests, self-contained)
2. **B** — `/api/arms/*` endpoints (depends on C for execution endpoint)
3. **A** — ARMS frontend grid (depends on B)
4. **D** — Live-key flow (independent)
5. **E** — Startup validation wiring (independent, last because highest blast radius if broken)
