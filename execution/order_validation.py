"""Order input validation helpers.

These functions intentionally raise ValueError so callers can choose whether to
surface an exception or translate the validation failure into an API/order result.
"""
from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any

_ALLOWED_SYMBOL = re.compile(r"^[A-Za-z0-9_./:-]+$")
_ALLOWED_SIDES = {"buy", "sell"}
_ALLOWED_ORDER_TYPES = {"limit", "market", "post_only", "ioc"}


def _value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def validate_symbol(symbol: Any) -> None:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    if not _ALLOWED_SYMBOL.fullmatch(symbol.strip()):
        raise ValueError("symbol contains invalid characters")


def validate_quantity(quantity: Any) -> None:
    try:
        value = float(quantity)
    except (TypeError, ValueError) as exc:
        raise ValueError("quantity must be numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("quantity must be positive and finite")


def validate_order_side(side: Any) -> None:
    value = _value(side)
    if not isinstance(value, str) or value.strip().lower() not in _ALLOWED_SIDES:
        raise ValueError("side must be one of: buy, sell")


def validate_order_type(order_type: Any) -> None:
    value = _value(order_type)
    if not isinstance(value, str) or value.strip().lower() not in _ALLOWED_ORDER_TYPES:
        raise ValueError("order_type must be one of: limit, market, post_only, ioc")


def validate_price(price: Any, order_type: Any) -> None:
    validate_order_type(order_type)
    order_type_value = str(_value(order_type)).strip().lower()
    if order_type_value == "market" and price in (None, ""):
        return
    try:
        value = float(price)
    except (TypeError, ValueError) as exc:
        raise ValueError("price must be numeric") from exc
    if order_type_value == "market" and value == 0:
        return
    if not math.isfinite(value) or value <= 0:
        raise ValueError("price must be positive and finite")
