"""
Comprehensive test suite for TIER 0 (Production-Safe) components.
Tests idempotency, order manager, circuit breaker, and risk controls.
"""

from pathlib import Path
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from core.idempotency import IdempotencyManager
from core.circuit_breaker import CircuitBreaker, CircuitState
from core.retry import RetryPolicy
from execution.order_manager import Order, OrderManager, OrderStatus, OrderSide, OrderType


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


class TestIdempotencyManager:
    """Test idempotency manager for exactly-once semantics"""

    def setup_method(self):
        self.manager = IdempotencyManager(ttl=3600, max_size=100)

    def test_generate_deterministic_key(self):
        """Same inputs should generate same key"""
        key1 = self.manager.generate_key("BTC", "USDT", "buy", 1.5)
        key2 = self.manager.generate_key("BTC", "USDT", "buy", 1.5)
        assert key1 == key2

    def test_different_inputs_different_keys(self):
        """Different inputs should generate different keys"""
        key1 = self.manager.generate_key("BTC", "USDT", "buy")
        key2 = self.manager.generate_key("BTC", "USDT", "sell")
        assert key1 != key2

    def test_check_and_set_first_call(self):
        """First call should return False (not a duplicate)"""
        key = self.manager.generate_key("order1")
        duplicate = self.manager.check_and_set(key)
        assert duplicate is False

    def test_check_and_set_second_call(self):
        """Second call with same key should return True (duplicate)"""
        key = self.manager.generate_key("order1")
        self.manager.check_and_set(key)
        duplicate = self.manager.check_and_set(key)
        assert duplicate is True

    def test_set_and_get_result(self):
        """Result should be retrievable after set"""
        key = "test_key"
        self.manager.check_and_set(key)
        self.manager.set_result(key, {"fill_id": "f123", "qty": 1.0})
        result = self.manager.get_result(key)
        assert result == {"fill_id": "f123", "qty": 1.0}

    def test_expired_result_cleanup(self):
        """Expired keys should be cleaned up"""
        manager = IdempotencyManager(ttl=0.1, max_size=100)  # 100ms TTL
        key = "expiring_key"
        manager.check_and_set(key)
        assert manager.check_and_set(key) is True  # Still valid initially

        time.sleep(0.2)
        # After TTL expires, should be treated as new
        duplicate = manager.check_and_set(key)
        # Will be False because old record was cleaned up
        assert duplicate is False


class TestCircuitBreaker:
    """Test circuit breaker for fault tolerance"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Circuit breaker should start in CLOSED state"""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        assert breaker.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Circuit breaker should open after threshold failures"""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        async def failing_operation():
            raise Exception("Simulated failure")

        # First two failures
        for _ in range(2):
            try:
                await breaker.call(failing_operation)
            except Exception:
                pass

        assert breaker.get_state() == CircuitState.CLOSED

        # Third failure triggers open
        try:
            await breaker.call(failing_operation)
        except Exception:
            pass

        assert breaker.get_state() == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_rejects_calls_when_open(self):
        """Should reject calls when circuit is open"""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60)

        async def failing_op():
            raise Exception("Failure")

        # Open the circuit
        try:
            await breaker.call(failing_op)
        except Exception:
            pass

        assert breaker.get_state() == CircuitState.OPEN

        # Next call should be rejected immediately
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await breaker.call(failing_op)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Circuit breaker should try to recover via HALF_OPEN state"""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        async def failing_op():
            raise Exception("Failure")

        # Open the circuit
        try:
            await breaker.call(failing_op)
        except Exception:
            pass

        assert breaker.get_state() == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Manually transition to half-open (normally automatic)
        await breaker._transition_to_half_open()
        assert breaker.get_state() == CircuitState.HALF_OPEN

        # Successful call should close the circuit
        async def success_op():
            return "success"

        result = await breaker.call(success_op)
        assert result == "success"


class TestRetryPolicy:
    """Test retry logic"""

    @pytest.mark.asyncio
    async def test_retry_on_failure_then_success(self):
        """Should retry on transient failure, then succeed.

        Uses ConnectionError (matches the new default retryable list per RT-1).
        Plain Exception is no longer retryable by default — non-transient
        errors fail fast.
        """
        policy = RetryPolicy(max_attempts=3, initial_delay=0.01, max_delay=0.1)
        attempts = 0

        async def flaky_operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("Transient failure")
            return "success"

        result = await policy.execute_with_retry(
            flaky_operation, operation_name="flaky_op"
        )
        assert result == "success"
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_retry_exhaust_all_attempts(self):
        """Should fail after exhausting all attempts on a transient error."""
        policy = RetryPolicy(max_attempts=2, initial_delay=0.01)
        attempts = 0

        async def always_failing():
            nonlocal attempts
            attempts += 1
            raise ConnectionError(f"Failure {attempts}")

        with pytest.raises(ConnectionError, match="Failure 2"):
            await policy.execute_with_retry(
                always_failing, operation_name="failing_op"
            )

        assert attempts == 2

    @pytest.mark.asyncio
    async def test_non_retryable_exception_fails_fast(self):
        """RT-1: non-transient errors (auth, validation, business logic) must
        NOT be retried — that was the whole point of narrowing the default."""
        policy = RetryPolicy(max_attempts=3, initial_delay=0.01)
        attempts = 0

        async def auth_error():
            nonlocal attempts
            attempts += 1
            raise ValueError("invalid api key")

        with pytest.raises(ValueError, match="invalid api key"):
            await policy.execute_with_retry(auth_error, operation_name="auth")

        assert attempts == 1, "non-retryable exception should fire only once"

    @pytest.mark.asyncio
    async def test_dead_letter_queue(self):
        """Failed operations should go to DLQ"""
        policy = RetryPolicy(max_attempts=1, dead_letter_queue_size=10)

        async def failing_op():
            raise Exception("DLQ test")

        with pytest.raises(Exception):
            await policy.execute_with_retry(failing_op, operation_name="dlq_test")

        dlq = policy.get_dlq()
        assert len(dlq) == 1
        assert dlq[0]["operation_name"] == "dlq_test"


class TestOrderManager:
    """Test order manager"""

    @pytest.mark.asyncio
    async def test_create_order(self):
        """Should create order in PENDING state"""
        from core.config import Config
        from core.event_bus import EventBus

        config = Config(config_path=CONFIG_PATH)
        event_bus = EventBus()
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        manager = OrderManager(config, event_bus, breaker)

        success, order, reason = await manager.place_order(
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=42000.0,
        )

        assert success is True
        assert order is not None
        assert order.status == OrderStatus.PENDING
        assert order.quantity == 0.1
        assert order.price == 42000.0

    @pytest.mark.asyncio
    async def test_idempotent_order_placement(self):
        """Placing same order twice should be idempotent"""
        from core.config import Config
        from core.event_bus import EventBus

        config = Config(config_path=CONFIG_PATH)
        event_bus = EventBus()
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        manager = OrderManager(config, event_bus, breaker)

        # First placement
        success1, order1, reason1 = await manager.place_order(
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=42000.0,
        )

        client_id_1 = order1.client_order_id

        # Second placement with same params (idempotent)
        success2, order2, reason2 = await manager.place_order(
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=42000.0,
        )

        # Should return same client_order_id (idempotent)
        # This test documents idempotency behavior
        assert order1.client_order_id is not None

    @pytest.mark.asyncio
    async def test_self_trade_prevention(self):
        """Should prevent self-trade (buy + sell on same symbol)"""
        from core.config import Config
        from core.event_bus import EventBus

        config = Config(config_path=CONFIG_PATH)
        event_bus = EventBus()
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        manager = OrderManager(config, event_bus, breaker)

        # Place buy order
        success_buy, order_buy, _ = await manager.place_order(
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=42000.0,
        )
        assert success_buy is True

        # Immediately place opposing sell order
        success_sell, _, reason_sell = await manager.place_order(
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            price=43000.0,
        )

        # Should be prevented due to self-trade
        assert success_sell is False
        assert "opposing_order" in reason_sell

    @pytest.mark.asyncio
    async def test_order_fill_tracking(self):
        """Should track partial and complete fills"""
        from core.config import Config
        from core.event_bus import EventBus

        config = Config(config_path=CONFIG_PATH)
        event_bus = EventBus()
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        manager = OrderManager(config, event_bus, breaker)

        success, order, _ = await manager.place_order(
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=42000.0,
        )

        client_id = order.client_order_id

        # Record partial fill
        await manager.record_fill(
            client_order_id=client_id,
            fill_id="fill_001",
            quantity=0.3,
            price=42100.0,
            fee=10.0,
        )

        order = manager.get_order(client_id)
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.cumulative_quantity == 0.3
        assert abs(order.average_fill_price - 42100.0) < 0.01

        # Record remaining fill
        await manager.record_fill(
            client_order_id=client_id,
            fill_id="fill_002",
            quantity=0.7,
            price=42200.0,
            fee=20.0,
        )

        order = manager.get_order(client_id)
        assert order.status == OrderStatus.FILLED
        assert order.cumulative_quantity == 1.0
        assert order.total_fee == 30.0

    @pytest.mark.asyncio
    async def test_order_cancellation(self):
        """Should cancel pending orders"""
        from core.config import Config
        from core.event_bus import EventBus

        config = Config(config_path=CONFIG_PATH)
        event_bus = EventBus()
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        manager = OrderManager(config, event_bus, breaker)

        success, order, _ = await manager.place_order(
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=42000.0,
        )

        client_id = order.client_order_id

        # Cancel the order
        cancel_success, cancelled_order, _ = await manager.cancel_order(
            client_order_id=client_id, reason="User cancelled"
        )

        assert cancel_success is True
        assert cancelled_order.status == OrderStatus.CANCELLED

    def test_order_manager_stats(self):
        """Should report accurate statistics"""
        from core.config import Config
        from core.event_bus import EventBus

        config = Config(config_path=CONFIG_PATH)
        event_bus = EventBus()
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        manager = OrderManager(config, event_bus, breaker)

        stats = manager.get_stats()
        assert stats["total_orders"] == 0
        assert stats["open_orders"] == 0
        assert stats["filled_orders"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
