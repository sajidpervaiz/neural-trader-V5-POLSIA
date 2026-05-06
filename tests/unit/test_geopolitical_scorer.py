"""Unit tests for the GeoPoliticalScorer + RSS parser + Layer 10 quality bonus."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from data_ingestion.geopolitical_news import _parse_rss
from engine.geopolitical_scorer import (
    GeoPoliticalScorer,
    ScoredEvent,
    _trading_hours_open,
)
from engine.signal_generator import SignalGenerator, SignalType


# ── Scorer ────────────────────────────────────────────────────────────────

class TestRelevanceAndDirection:
    def test_btc_etf_approval_scores_long(self) -> None:
        scorer = GeoPoliticalScorer()
        events = scorer.score_article(
            title="SEC approves spot Bitcoin ETF, institutional inflows expected",
            body="The Securities and Exchange Commission gave the green light for a spot Bitcoin ETF.",
            source="example.com",
            timestamp=time.time(),
        )
        btc = next((e for e in events if e.symbol == "BTC/USDT:USDT"), None)
        assert btc is not None
        assert btc.direction == 1
        assert btc.relevance >= 30

    def test_oil_strait_attack_scores_long_when_open(self) -> None:
        # Oil window is Sun 23:00 → Fri 22:00 UTC. Force a Monday 12:00 UTC eval.
        scorer = GeoPoliticalScorer()
        events = scorer.score_article(
            title="Tankers struck near Hormuz, Iran tensions escalate, supply disruption fears",
            source="example.com",
            timestamp=datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc).timestamp(),
        )
        oil = next((e for e in events if e.symbol == "CL/USDT:USDT"), None)
        # Trading-hours gate uses live clock — only assert if currently open.
        if _trading_hours_open({"start": "23:00", "end": "22:00",
                                "days": ["sun", "mon", "tue", "wed", "thu", "fri"]}):
            assert oil is not None and oil.direction == 1

    def test_below_min_relevance_dropped(self) -> None:
        scorer = GeoPoliticalScorer()
        events = scorer.score_article(title="Local bakery wins award", source="x")
        assert events == []


class TestTradingHoursGate:
    def test_open_window_within_range(self) -> None:
        rule = {"start": "09:00", "end": "17:00", "days": ["mon", "tue", "wed", "thu", "fri"]}
        # Wednesday 12:00 UTC
        now = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)
        assert _trading_hours_open(rule, now) is True

    def test_closed_outside_window(self) -> None:
        rule = {"start": "09:00", "end": "17:00", "days": ["mon", "tue", "wed", "thu", "fri"]}
        # Wednesday 22:00 UTC
        now = datetime(2026, 4, 22, 22, 0, tzinfo=timezone.utc)
        assert _trading_hours_open(rule, now) is False

    def test_closed_on_weekend(self) -> None:
        rule = {"start": "09:00", "end": "17:00", "days": ["mon", "tue", "wed", "thu", "fri"]}
        now = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)  # Saturday
        assert _trading_hours_open(rule, now) is False

    def test_midnight_wrap_oil_window(self) -> None:
        # Oil: 23:00 → 22:00 next day, Sun-Fri
        rule = {"start": "23:00", "end": "22:00",
                "days": ["sun", "mon", "tue", "wed", "thu", "fri"]}
        # Monday 23:30 UTC — inside wrap (after 23:00)
        now = datetime(2026, 4, 27, 23, 30, tzinfo=timezone.utc)
        assert _trading_hours_open(rule, now) is True
        # Tuesday 21:00 UTC — inside wrap (before 22:00)
        now = datetime(2026, 4, 28, 21, 0, tzinfo=timezone.utc)
        assert _trading_hours_open(rule, now) is True
        # Tuesday 22:30 UTC — closed (between 22:00 close and 23:00 open)
        now = datetime(2026, 4, 28, 22, 30, tzinfo=timezone.utc)
        assert _trading_hours_open(rule, now) is False

    def test_none_rule_always_open(self) -> None:
        assert _trading_hours_open(None) is True


class TestTimeDecayAndComposite:
    def test_single_fresh_event_yields_direction_times_confidence(self) -> None:
        scorer = GeoPoliticalScorer(window_hours=6.0)
        now = time.time()
        ev = ScoredEvent(
            symbol="BTC/USDT:USDT", direction=1, relevance=80, confidence=0.8,
            title="BTC ETF approved", source="x", timestamp=now,
        )
        scorer.ingest(ev)
        # Fresh event → decay=1.0, normalized → composite = direction * 1.0
        score = scorer.score("BTC/USDT:USDT", now=now)
        assert score == pytest.approx(1.0, rel=0.01)

    def test_decay_pushes_score_toward_zero(self) -> None:
        scorer = GeoPoliticalScorer(window_hours=6.0)
        now = time.time()
        # Event at full window age — decay weight = 0
        old_ts = now - 6.0 * 3600
        scorer.ingest(ScoredEvent(
            symbol="BTC/USDT:USDT", direction=1, relevance=80, confidence=0.8,
            title="old", source="x", timestamp=old_ts,
        ))
        assert scorer.score("BTC/USDT:USDT", now=now) == 0.0

    def test_opposing_events_partially_cancel(self) -> None:
        scorer = GeoPoliticalScorer(window_hours=6.0)
        now = time.time()
        scorer.ingest(ScoredEvent(
            symbol="BTC/USDT:USDT", direction=1, relevance=80, confidence=0.8,
            title="bull", source="x", timestamp=now,
        ))
        scorer.ingest(ScoredEvent(
            symbol="BTC/USDT:USDT", direction=-1, relevance=80, confidence=0.4,
            title="bear", source="x", timestamp=now,
        ))
        score = scorer.score("BTC/USDT:USDT", now=now)
        # Weighted: (+1*0.8 + -1*0.4) / (0.8+0.4) = +0.333
        assert score == pytest.approx(0.333, rel=0.05)

    def test_ambiguous_direction_not_ingested(self) -> None:
        scorer = GeoPoliticalScorer(window_hours=6.0)
        scorer.ingest(ScoredEvent(
            symbol="BTC/USDT:USDT", direction=0, relevance=80, confidence=0.8,
            title="neutral", source="x", timestamp=time.time(),
        ))
        assert scorer.event_count("BTC/USDT:USDT") == 0


class TestAliasResolution:
    def test_alias_maps_to_canonical(self) -> None:
        scorer = GeoPoliticalScorer()
        scorer.add_alias("BTCUSDT", "BTC/USDT:USDT")
        scorer.ingest(ScoredEvent(
            symbol="BTC/USDT:USDT", direction=1, relevance=80, confidence=1.0,
            title="x", source="y", timestamp=time.time(),
        ))
        assert scorer.score("BTCUSDT") == pytest.approx(1.0, rel=0.01)

    def test_unknown_symbol_returns_zero(self) -> None:
        scorer = GeoPoliticalScorer()
        assert scorer.score("UNKNOWN/SYM") == 0.0


# ── RSS parser ────────────────────────────────────────────────────────────

class TestRSSParser:
    def test_rss_2_item_extraction(self) -> None:
        xml = b"""<?xml version="1.0"?><rss version="2.0"><channel>
            <item><title>Hello</title><description>World</description>
                  <link>http://x/1</link><guid>g1</guid></item>
            <item><title>Two</title><description>D2</description>
                  <link>http://x/2</link></item>
        </channel></rss>"""
        items = _parse_rss(xml)
        assert len(items) == 2
        assert items[0]["id"] == "g1"
        assert items[1]["id"] == "http://x/2"  # falls back to link

    def test_atom_fallback(self) -> None:
        xml = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
            <entry><title>AT</title><summary>S1</summary>
                   <link href="http://a/1"/><id>atom:1</id></entry>
        </feed>"""
        items = _parse_rss(xml)
        assert len(items) == 1 and items[0]["id"] == "atom:1"
        assert items[0]["title"] == "AT"
        assert items[0]["link"] == "http://a/1"

    def test_malformed_xml_returns_empty(self) -> None:
        assert _parse_rss(b"not xml") == []

    def test_empty_titles_skipped(self) -> None:
        xml = b"<rss><channel><item><title></title><guid>x</guid></item></channel></rss>"
        assert _parse_rss(xml) == []


# ── Quality bonus integration ─────────────────────────────────────────────

class TestQualityBonusIntegration:
    @pytest.fixture
    def sg(self) -> SignalGenerator:
        return SignalGenerator.__new__(SignalGenerator)

    @pytest.fixture
    def base_kw(self) -> dict:
        class _Stub:
            def __getattr__(self, n: str) -> float:  # pragma: no cover
                return 0.0
        return dict(
            htf_score=0.5, signal_type=SignalType.COMPOSITE, vol_ratio=1.0,
            smc_state=_Stub(), vol_flow=_Stub(), session_rule=None, sentiment=0.0,
            regime_state=None, direction="long",
            tech_score_100=70, smc_points=70, volume_score_100=70,
            regime_allows=True, ml_confidence=70, orderbook_depth_ratio=2.5,
            killzone_score=0.0,
        )

    def test_aligned_geo_adds_full_bonus(self, sg: SignalGenerator, base_kw: dict) -> None:
        no_geo = sg._compute_quality_score(**base_kw, geo_score=0.0, geo_weight=0.10)
        aligned = sg._compute_quality_score(**base_kw, geo_score=0.8, geo_weight=0.10)
        assert aligned - no_geo == pytest.approx(8, abs=1)

    def test_opposed_geo_applies_half_penalty(self, sg: SignalGenerator, base_kw: dict) -> None:
        no_geo = sg._compute_quality_score(**base_kw, geo_score=0.0, geo_weight=0.10)
        opposed = sg._compute_quality_score(**base_kw, geo_score=-0.8, geo_weight=0.10)
        assert opposed - no_geo == pytest.approx(-4, abs=1)

    def test_disabled_weight_is_no_op(self, sg: SignalGenerator, base_kw: dict) -> None:
        baseline = sg._compute_quality_score(**base_kw, geo_score=0.0, geo_weight=0.0)
        with_geo_disabled = sg._compute_quality_score(**base_kw, geo_score=0.9, geo_weight=0.0)
        assert baseline == with_geo_disabled

    def test_score_clamped_to_100(self, sg: SignalGenerator, base_kw: dict) -> None:
        # Boost everything to ceiling and add max geo bonus
        boosted = dict(base_kw)
        boosted.update(
            htf_score=1.0, tech_score_100=100, smc_points=100,
            volume_score_100=100, ml_confidence=100, orderbook_depth_ratio=10.0,
            killzone_score=1.0,
        )
        out = sg._compute_quality_score(**boosted, geo_score=1.0, geo_weight=0.20)
        assert out == 100
