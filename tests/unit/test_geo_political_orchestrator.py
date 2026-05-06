"""Unit tests for the geo-political trading strategy.

Covers:
- Phase 2 parser (clean JSON, fenced JSON, NL fallback, ambiguous input)
- Exit logic (FLAT-TIMEOUT, HARD-TIMEOUT, SL, TP, BE, trailing)
- V1 quality gates
- Keyword relevance scoring
- Rail clamping
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from strategies.llm_client import (
    parse_sentiment, parse_confirmation, parse_trade_plan,
)
from strategies.paper_ledger import (
    PaperTrade, PaperLedger, check_exits, update_protective_stops,
    MAX_HOLD_MIN,
)
from strategies.market_configs import (
    MARKET_OIL, MARKET_BTC, MARKET_GOLD, MIN_RELEVANCE, ALL_MARKETS,
)
from strategies.rss_fetcher import RSSItem


# ═══════════════════════════════════════════════════════════════════════════════
#  Parser tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase2Parser:
    def test_clean_json(self):
        raw = '{"verdict": "CONFIRM", "score": 85, "reasoning": "Price rising fast"}'
        result = parse_confirmation(raw)
        assert result is not None
        assert result.verdict == "CONFIRM"
        assert result.score == 85

    def test_fenced_json(self):
        raw = '```json\n{"verdict": "CONTRADICT", "score": 30, "reasoning": "Falling"}\n```'
        result = parse_confirmation(raw)
        assert result is not None
        assert result.verdict == "CONTRADICT"
        assert result.score == 30

    def test_thinking_then_json(self):
        raw = (
            "Let me analyze the data. The price is rising which aligns with LONG.\n"
            '{"verdict": "CONFIRM", "score": 72, "reasoning": "Aligned with signal"}'
        )
        result = parse_confirmation(raw)
        assert result is not None
        assert result.verdict == "CONFIRM"
        assert result.score == 72

    def test_natural_language_confirm(self):
        raw = (
            "Looking at the data, the price has risen 0.5% in the last hour "
            "with strong volume. The market CONFIRMS the signal."
        )
        result = parse_confirmation(raw)
        assert result is not None
        assert result.verdict == "CONFIRM"

    def test_natural_language_verdict_prefix(self):
        raw = "Based on analysis, verdict: CONTRADICT with low volume."
        result = parse_confirmation(raw)
        assert result is not None
        assert result.verdict == "CONTRADICT"

    def test_natural_language_therefore(self):
        raw = "Price is flat, volume is low, therefore WEAK signal."
        result = parse_confirmation(raw)
        assert result is not None
        assert result.verdict == "WEAK"

    def test_rule_restatement_filtered(self):
        raw = (
            "If direction is LONG and price is RISING then market CONFIRMS. "
            "If direction is LONG and price is FALLING then market CONTRADICTS. "
            "The current price is falling. Therefore the market CONTRADICTS."
        )
        result = parse_confirmation(raw)
        assert result is not None
        assert result.verdict == "CONTRADICT"

    def test_empty_input(self):
        assert parse_confirmation("") is None
        assert parse_confirmation("   ") is None

    def test_ambiguous_no_verdict(self):
        raw = "I'm not sure what to make of this data. It could go either way."
        result = parse_confirmation(raw)
        assert result is None

    def test_truncated_json(self):
        raw = '{"verdict": "CONFIRM", "score": 80, "reasoning": "Strong move'
        result = parse_confirmation(raw)
        assert result is None or result.verdict == "CONFIRM"


class TestPhase1Parser:
    def test_clean_json(self):
        raw = '{"direction": "LONG", "confidence": 78, "reasoning": "Supply threat"}'
        result = parse_sentiment(raw)
        assert result is not None
        assert result.direction == "LONG"
        assert result.confidence == 78

    def test_skip_direction(self):
        raw = '{"direction": "SKIP", "confidence": 20, "reasoning": "Irrelevant"}'
        result = parse_sentiment(raw)
        assert result is not None
        assert result.direction == "SKIP"

    def test_short_direction(self):
        raw = '{"direction": "SHORT", "confidence": 65, "reasoning": "De-escalation"}'
        result = parse_sentiment(raw)
        assert result is not None
        assert result.direction == "SHORT"


class TestPhase3Parser:
    def test_clean_plan(self):
        raw = '{"skip": false, "sl_pct": 2.2, "tp_pct": 4.8, "timeout_minutes": 40, "confidence": 82, "reasoning": "Strong event"}'
        plan = parse_trade_plan(raw)
        assert plan is not None
        assert not plan.skip
        assert plan.sl_pct == 2.2
        assert plan.tp_pct == 4.8
        assert plan.timeout_minutes == 40

    def test_skip_plan(self):
        raw = '{"skip": true, "sl_pct": 1.0, "tp_pct": 1.5, "timeout_minutes": 10, "confidence": 25, "reasoning": "Weak"}'
        plan = parse_trade_plan(raw)
        assert plan is not None
        assert plan.skip is True

    def test_rail_clamping_sl(self):
        raw = '{"skip": false, "sl_pct": 0.1, "tp_pct": 10.0, "timeout_minutes": 1, "confidence": 50, "reasoning": "X"}'
        plan = parse_trade_plan(raw)
        assert plan is not None
        assert plan.sl_pct == 0.5
        assert plan.tp_pct == 6.0
        assert plan.timeout_minutes == 5

    def test_rail_clamping_high(self):
        raw = '{"skip": false, "sl_pct": 5.0, "tp_pct": 0.3, "timeout_minutes": 120, "confidence": 50, "reasoning": "X"}'
        plan = parse_trade_plan(raw)
        assert plan is not None
        assert plan.sl_pct == 3.0
        assert plan.tp_pct == 0.8
        assert plan.timeout_minutes == 60


# ═══════════════════════════════════════════════════════════════════════════════
#  Exit logic tests
# ═══════════════════════════════════════════════════════════════════════════════

def _make_trade(
    direction: str = "LONG", entry: float = 100.0,
    sl_pct: float = 2.0, tp_pct: float = 4.0,
    timeout_minutes: int = 30, age_minutes: float = 0,
) -> PaperTrade:
    opened_at = datetime.fromtimestamp(
        time.time() - age_minutes * 60, tz=timezone.utc,
    ).isoformat()
    if direction == "LONG":
        sl = entry * (1 - sl_pct / 100)
        tp = entry * (1 + tp_pct / 100)
    else:
        sl = entry * (1 + sl_pct / 100)
        tp = entry * (1 - tp_pct / 100)
    return PaperTrade(
        trade_id="PT-test-001", signal_uid="abc123",
        opened_at=opened_at, symbol="BTC/USDT:USDT",
        direction=direction, entry_price=entry,
        sl_price=sl, tp_price=tp,
        sl_pct=sl_pct, tp_pct=tp_pct,
        timeout_minutes=timeout_minutes,
        margin_usd=5.0, leverage=5, notional_usd=25.0, qty=0.25,
        headline="Test", source="Test",
        news_confidence=80, phase2_verdict="CONFIRM",
        phase2_score=80, phase2_reasoning="Test",
        plan_confidence=75, plan_reasoning="Test",
    )


class TestExitLogic:
    def test_sl_hit_long(self):
        trade = _make_trade("LONG", entry=100, sl_pct=2.0, tp_pct=4.0)
        reason, price = check_exits(trade, 97.5)
        assert reason == "SL"

    def test_tp_hit_long(self):
        trade = _make_trade("LONG", entry=100, sl_pct=2.0, tp_pct=4.0)
        reason, price = check_exits(trade, 104.5)
        assert reason == "TP"

    def test_sl_hit_short(self):
        trade = _make_trade("SHORT", entry=100, sl_pct=2.0, tp_pct=4.0)
        reason, price = check_exits(trade, 102.5)
        assert reason == "SL"

    def test_tp_hit_short(self):
        trade = _make_trade("SHORT", entry=100, sl_pct=2.0, tp_pct=4.0)
        reason, price = check_exits(trade, 95.5)
        assert reason == "TP"

    def test_no_exit_in_range(self):
        trade = _make_trade("LONG", entry=100, sl_pct=2.0, tp_pct=4.0)
        reason, price = check_exits(trade, 101.0)
        assert reason is None

    def test_hard_timeout(self):
        trade = _make_trade("LONG", entry=100, age_minutes=MAX_HOLD_MIN + 1)
        reason, price = check_exits(trade, 100.1)
        assert reason == "HARD-TIMEOUT"

    def test_flat_timeout_closes_flat_trade(self):
        trade = _make_trade("LONG", entry=100, timeout_minutes=30, age_minutes=35)
        reason, price = check_exits(trade, 100.1)
        assert reason == "FLAT-TIMEOUT"

    def test_flat_timeout_does_not_close_moved_trade(self):
        trade = _make_trade("LONG", entry=100, timeout_minutes=30, age_minutes=35)
        reason, price = check_exits(trade, 102.0)
        assert reason is None

    def test_be_vs_sl(self):
        trade = _make_trade("LONG", entry=100, sl_pct=2.0, tp_pct=4.0)
        trade.be_activated = True
        trade.sl_price = 100.05
        reason, price = check_exits(trade, 99.5)
        assert reason == "BE"

    def test_closed_trade_no_exit(self):
        trade = _make_trade("LONG", entry=100)
        trade.status = "CLOSED"
        reason, price = check_exits(trade, 50.0)
        assert reason is None


class TestProtectiveStops:
    def test_breakeven_activation_long(self):
        trade = _make_trade("LONG", entry=100, sl_pct=2.0, tp_pct=4.0)
        assert not trade.be_activated
        update_protective_stops(trade, 102.5)
        assert trade.be_activated
        assert trade.sl_price > 100.0

    def test_breakeven_activation_short(self):
        trade = _make_trade("SHORT", entry=100, sl_pct=2.0, tp_pct=4.0)
        assert not trade.be_activated
        update_protective_stops(trade, 97.5)
        assert trade.be_activated
        assert trade.sl_price < 100.0

    def test_trailing_stop_long(self):
        trade = _make_trade("LONG", entry=100, sl_pct=2.0, tp_pct=4.0)
        update_protective_stops(trade, 102.5)
        assert trade.be_activated
        initial_sl = trade.sl_price
        update_protective_stops(trade, 103.5)
        assert trade.trailing_activated
        assert trade.sl_price > initial_sl

    def test_no_activation_below_threshold(self):
        trade = _make_trade("LONG", entry=100, sl_pct=2.0, tp_pct=4.0)
        update_protective_stops(trade, 101.0)
        assert not trade.be_activated


# ═══════════════════════════════════════════════════════════════════════════════
#  Keyword relevance tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestKeywordRelevance:
    def test_oil_hormuz_headline(self):
        score = MARKET_OIL.relevance_score(
            "Iran threatens to close Strait of Hormuz amid rising tensions"
        )
        assert score >= MIN_RELEVANCE

    def test_oil_irrelevant(self):
        score = MARKET_OIL.relevance_score("Apple releases new iPhone model")
        assert score < MIN_RELEVANCE

    def test_btc_etf_headline(self):
        score = MARKET_BTC.relevance_score(
            "SEC approves Bitcoin spot ETF for institutional investors"
        )
        assert score >= MIN_RELEVANCE

    def test_gold_inflation_headline(self):
        score = MARKET_GOLD.relevance_score(
            "CPI inflation surges to 5%, Federal Reserve considers rate cuts"
        )
        assert score >= MIN_RELEVANCE

    def test_best_market_assignment(self):
        text = "OPEC announces surprise oil production cuts, crude prices surge"
        scores = [(m, m.relevance_score(text)) for m in ALL_MARKETS]
        best = max(scores, key=lambda x: x[1])
        assert best[0].name == "WTI Crude Oil"

    def test_hard_action_verb_alone_passes(self):
        score = MARKET_OIL.relevance_score("Pipeline seized by armed group")
        assert score >= MIN_RELEVANCE


# ═══════════════════════════════════════════════════════════════════════════════
#  Ledger tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPaperLedger:
    def test_open_and_close(self, tmp_path):
        ledger = PaperLedger(path=str(tmp_path / "test.jsonl"))
        trade = _make_trade("LONG", entry=100)
        ledger.open_trade(trade)
        assert ledger.get_open_count() == 1

        closed = ledger.close_trade(trade.trade_id, 104.0, "TP")
        assert closed is not None
        assert closed.pnl_pct == pytest.approx(4.0, abs=0.1)
        assert ledger.get_open_count() == 0

    def test_duplicate_detection(self, tmp_path):
        ledger = PaperLedger(path=str(tmp_path / "test.jsonl"))
        trade = _make_trade("LONG", entry=100)
        ledger.open_trade(trade)
        assert ledger.has_open_for("BTC/USDT:USDT", "LONG")
        assert not ledger.has_open_for("BTC/USDT:USDT", "SHORT")

    def test_recently_traded(self, tmp_path):
        ledger = PaperLedger(path=str(tmp_path / "test.jsonl"))
        trade = _make_trade("LONG", entry=100)
        ledger.open_trade(trade)
        assert ledger.was_recently_traded("abc123")
        assert not ledger.was_recently_traded("xyz999")

    def test_stats_empty(self, tmp_path):
        ledger = PaperLedger(path=str(tmp_path / "test.jsonl"))
        stats = ledger.get_stats()
        assert stats["total_trades"] == 0

    def test_stats_after_trades(self, tmp_path):
        ledger = PaperLedger(path=str(tmp_path / "test.jsonl"))
        t1 = _make_trade("LONG", entry=100)
        t1.trade_id = "PT-t1-001"
        ledger.open_trade(t1)
        ledger.close_trade(t1.trade_id, 104.0, "TP")

        t2 = _make_trade("LONG", entry=100)
        t2.trade_id = "PT-t2-002"
        ledger.open_trade(t2)
        ledger.close_trade(t2.trade_id, 97.0, "SL")

        stats = ledger.get_stats()
        assert stats["total_trades"] == 2
        assert stats["win_rate"] == 50.0


# ═══════════════════════════════════════════════════════════════════════════════
#  RSS item dedup test
# ═══════════════════════════════════════════════════════════════════════════════

class TestRSSItem:
    def test_signal_uid_deterministic(self):
        a = RSSItem(title="Test", url="http://example.com", summary="", source="test", published_ts=0)
        b = RSSItem(title="Test", url="http://example.com", summary="", source="test", published_ts=0)
        assert a.signal_uid == b.signal_uid

    def test_signal_uid_differs(self):
        a = RSSItem(title="Test A", url="http://a.com", summary="", source="test", published_ts=0)
        b = RSSItem(title="Test B", url="http://b.com", summary="", source="test", published_ts=0)
        assert a.signal_uid != b.signal_uid


class TestMarketHours:
    def test_btc_always_open(self):
        assert MARKET_BTC.is_market_open()

    def test_gold_always_open(self):
        assert MARKET_GOLD.is_market_open()
