#!/usr/bin/env bash
# Captures the actual failure mode of NeuralTrader V5.
# Run from the repo root: bash scripts/diagnose.sh
set -u
cd "$(dirname "$0")/.."
OUT="logs/diagnose_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee "$OUT") 2>&1

echo "=== 1. Python + Rust module sanity ==="
python3 --version
python3 -c "import neural_trader_rust as r; print('neural_trader_rust OK:', r.__file__)" \
    || echo "FAIL: neural_trader_rust import"

echo "=== 2. Top-level imports from main.py (one at a time) ==="
python3 - <<'PY'
import importlib, traceback
mods = [
    "core.config","core.event_bus","core.dispatcher",
    "data_ingestion.cex_websocket","data_ingestion.tick_processor",
    "engine.signal_generator","engine.geopolitical_scorer",
    "execution.risk_manager","execution.order_manager",
    "execution.exchange_factory","execution.smart_order_router",
    "execution.startup_validation","execution.reconciliation",
    "storage.db_handler","storage.cache","storage.sqlite_store",
    "monitoring.metrics","monitoring.health_checks","monitoring.alert_manager",
    "interface.dashboard_api",
]
fails = []
for m in mods:
    try:
        importlib.import_module(m); print("ok  ", m)
    except Exception:
        fails.append(m); print("FAIL", m); traceback.print_exc()
print("\nimport_failures:", fails)
PY

echo "=== 3. Git working-tree state ==="
git -C . log --oneline -1
git -C . status --short | head -40

echo "=== 4. Boot main.py for 25s with hard wall-clock timeout ==="
timeout --kill-after=5 25 python3 main.py
echo "(exit=$?  124=timeout-as-expected, anything else=real failure)"

echo "=== 5. Dashboard reachable? ==="
(timeout --kill-after=2 30 python3 main.py &)
for i in $(seq 1 25); do sleep 1; curl -sf -m 1 http://127.0.0.1:8000/health >/dev/null && break; done
curl -sS -m 3 http://127.0.0.1:8000/health || echo "FAIL: dashboard unreachable"
pkill -f "python3 main.py" 2>/dev/null

echo "=== Done. Log: $OUT ==="
