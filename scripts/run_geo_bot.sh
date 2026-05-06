#!/usr/bin/env bash
# Launch the geo-political paper bot, sourcing .env for the LLM key.
#   start :  bash scripts/run_geo_bot.sh start
#   stop  :  bash scripts/run_geo_bot.sh stop
#   tail  :  bash scripts/run_geo_bot.sh tail
#   trades:  bash scripts/run_geo_bot.sh trades
set -u
cd "$(dirname "$0")/.."

PIDFILE="/tmp/geo.pid"
LOGFILE="logs/geo.log"
mkdir -p logs data

cmd="${1:-start}"
case "$cmd" in
  start)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "geo bot already running (PID $(cat "$PIDFILE")). Run: bash $0 stop"
      exit 1
    fi
    if [[ ! -f .env ]]; then
      echo "ERROR: .env not found at repo root. Create it from the template first."
      exit 1
    fi
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    if [[ -z "${GEO_LLM_API_KEY:-}" ]]; then
      echo "WARNING: GEO_LLM_API_KEY is empty in .env — bot will run but skip every news item (no paper trades)."
      echo "         Edit .env to set the key, then re-run: bash $0 start"
    else
      echo "GEO_LLM_API_KEY set (provider: ${GEO_LLM_BASE_URL:-<default openai>}, model: ${GEO_LLM_MODEL:-<default>})"
    fi
    nohup python3 -c "
import asyncio, sys
from loguru import logger
logger.remove()
logger.add('${LOGFILE}', level='INFO', rotation='10 MB', retention=5)
logger.add(sys.stderr, level='INFO', format='{time:HH:mm:ss} | {level:<5} | {message}')
from strategies.geo_political_orchestrator import GeoPolStrategy
asyncio.run(GeoPolStrategy(config={'scan_interval_sec':120}).run())
" >> "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 1
    if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "started PID $(cat "$PIDFILE")  ·  log: $LOGFILE  ·  ledger: data/geo_paper_trades.jsonl"
    else
      echo "FAILED to start — check $LOGFILE"
      rm -f "$PIDFILE"
      exit 1
    fi
    ;;
  stop)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      kill "$(cat "$PIDFILE")"
      echo "stopped PID $(cat "$PIDFILE")"
      rm -f "$PIDFILE"
    else
      echo "no running geo bot."
    fi
    ;;
  status)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "running (PID $(cat "$PIDFILE"))"
    else
      echo "stopped"
    fi
    ;;
  tail)
    exec tail -F "$LOGFILE"
    ;;
  trades)
    if [[ -f data/geo_paper_trades.jsonl ]]; then
      echo "$(wc -l < data/geo_paper_trades.jsonl) ledger entries:"
      cat data/geo_paper_trades.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    t = json.loads(line)
    closed = t.get('exit_reason') is not None
    pnl = t.get('pnl_pct', 0) if closed else 0
    print(f\"  {t['opened_at'][:19]}  {t['symbol']:18}  {t['direction']:5}  conf={t.get('news_confidence',0):3}  {'CLOSED' if closed else 'OPEN'}  pnl={pnl:+.2f}%  {t['headline'][:60]}\")
"
    else
      echo "no trades yet (ledger file does not exist)."
    fi
    ;;
  *)
    echo "usage: $0 {start|stop|status|tail|trades}"
    exit 2
    ;;
esac
