#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python3 scripts/prepare_live_config.py
export NT_CONFIG_PATH=config/settings.live.yaml
python3 scripts/preflight_live_trading.py
export LIVE_TRADING_CONFIRMED="true"

echo "Starting live auto-trading with config: $NT_CONFIG_PATH"
exec python3 main.py
