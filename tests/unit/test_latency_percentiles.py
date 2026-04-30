"""REQ-MON-002: latency percentile computations must be safe (no IndexError
on small samples) and must include p50/p95/p99."""
from __future__ import annotations

from monitoring.metrics import Metrics


def test_percentile_empty_returns_zero() -> None:
    assert Metrics._percentile_ms([], 50) == 0.0
    assert Metrics._percentile_ms([], 95) == 0.0
    assert Metrics._percentile_ms([], 99) == 0.0


def test_percentile_single_element() -> None:
    # Single sample — p50/p95/p99 all return that value (no IndexError).
    assert Metrics._percentile_ms([0.123], 50) == 123.0
    assert Metrics._percentile_ms([0.123], 95) == 123.0
    assert Metrics._percentile_ms([0.123], 99) == 123.0


def test_percentile_ordered_samples() -> None:
    # 100 samples 0.001s..0.100s → 1ms..100ms; nearest-rank percentiles
    samples = [(i + 1) / 1000.0 for i in range(100)]
    p50 = Metrics._percentile_ms(samples, 50)
    p95 = Metrics._percentile_ms(samples, 95)
    p99 = Metrics._percentile_ms(samples, 99)
    assert 49 <= p50 <= 51
    assert 94 <= p95 <= 96
    assert 98 <= p99 <= 100


def test_percentile_is_monotonic() -> None:
    samples = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0]
    p50 = Metrics._percentile_ms(samples, 50)
    p95 = Metrics._percentile_ms(samples, 95)
    p99 = Metrics._percentile_ms(samples, 99)
    assert p50 <= p95 <= p99
