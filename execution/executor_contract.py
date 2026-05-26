from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from engine.signal_generator import TradingSignal
from execution.cex_executor import OrderResult


RUNTIME_METHODS = ("run", "stop", "execute_signal")
ORDER_CONTROL_METHODS = ("cancel_order", "close_position")
MARKET_DATA_METHODS = ("get_orderbook_snapshot",)


@runtime_checkable
class TradingExecutorContract(Protocol):
    """Common runtime contract for exchange executors."""

    exchange_id: str

    async def run(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def execute_signal(self, signal: TradingSignal, size: float) -> OrderResult | None:
        ...

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        ...

    async def close_position(
        self,
        symbol: str,
        price: float,
        *,
        reason: str = "manual_close",
    ) -> OrderResult | None:
        ...

    async def get_orderbook_snapshot(self, symbol: str, depth: int = 10) -> Any:
        ...


def executor_contract_status(
    executor: Any,
    *,
    require_order_controls: bool = True,
    require_market_data: bool = True,
) -> dict[str, Any]:
    """Return a JSON-safe contract audit for one runtime executor.

    This is intentionally structural: exchange executors can remain lightweight
    Protocol implementers while readiness still fails closed when a safety
    method such as reduce-only close is missing.
    """
    exchange_id = str(getattr(executor, "exchange_id", type(executor).__name__) or type(executor).__name__)
    methods = {
        name: callable(getattr(executor, name, None))
        for name in (*RUNTIME_METHODS, *ORDER_CONTROL_METHODS, *MARKET_DATA_METHODS)
    }
    missing_runtime = [name for name in RUNTIME_METHODS if not methods.get(name, False)]
    missing_order_controls = [name for name in ORDER_CONTROL_METHODS if not methods.get(name, False)]
    missing_market_data = [name for name in MARKET_DATA_METHODS if not methods.get(name, False)]

    client = getattr(executor, "_client", None)
    markets = getattr(client, "markets", None) if client is not None else None
    initialized = bool(
        getattr(executor, "_running", False)
        or getattr(executor, "_hl_exchange", None) is not None
        or (client is not None and bool(markets))
    )
    is_paper = bool(getattr(executor, "is_paper", False)) or "simulated" in type(executor).__name__.lower()

    blockers = list(missing_runtime)
    if require_order_controls:
        blockers.extend(missing_order_controls)
    if require_market_data:
        blockers.extend(missing_market_data)

    return {
        "exchange": exchange_id,
        "class": type(executor).__name__,
        "paper": is_paper,
        "initialized": initialized,
        "client": client is not None,
        "methods": methods,
        "missing_runtime": missing_runtime,
        "missing_order_controls": missing_order_controls,
        "missing_market_data": missing_market_data,
        "contract_ok": not blockers,
        "blockers": blockers,
    }
