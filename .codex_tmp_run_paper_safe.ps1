$ErrorActionPreference = "Stop"

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:BINANCE_API_KEY = ""
$env:BINANCE_API_SECRET = ""
$env:GEO_LLM_API_KEY = ""
$env:GEO_LLM_BASE_URL = ""
$env:GEO_LLM_MODEL = ""
$env:LIVE_TRADING_CONFIRMED = "false"
$env:NT_CONFIG_PATH = "config/settings.yaml"
$env:NT_RUNTIME_CONFIG_PATH = "config/codex.safe.runtime.yaml"

Set-Location -LiteralPath $PSScriptRoot
& ".\.venv-run\Scripts\python.exe" "main.py"
