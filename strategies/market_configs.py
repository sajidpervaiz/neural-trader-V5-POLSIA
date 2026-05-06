"""Market configurations: keywords, sentiment rules, trading hours per asset."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class MarketConfig:
    name: str
    symbol: str
    keywords: dict[str, int]
    sentiment_rules: str
    trades_24_7: bool = True
    nymex_hours: bool = False

    def relevance_score(self, text: str) -> int:
        lower = text.lower()
        return sum(w for kw, w in self.keywords.items() if kw in lower)

    def is_market_open(self) -> bool:
        if self.trades_24_7:
            return True
        now = datetime.now(tz=timezone.utc)
        wd = now.weekday()  # Mon=0, Sun=6
        h, m = now.hour, now.minute
        mins = h * 60 + m
        if wd == 6 and mins >= 23 * 60:
            return True
        if wd == 4 and mins >= 22 * 60:
            return False
        if 5 <= wd <= 6:
            return wd == 6 and mins >= 23 * 60
        return True


OIL_SENTIMENT_RULES = """
- War / conflict / escalation / supply disruption = LONG (supply fear)
- Peace / ceasefire / diplomacy = SHORT (supply easing)
- Iran / Hormuz / Saudi disruption = LONG strongly
- OPEC cuts = LONG. OPEC increases = SHORT.
- Demand destruction / recession = SHORT
""".strip()

BTC_SENTIMENT_RULES = """
- Crypto-friendly regulation / ETF approval / institutional adoption = LONG
- Regulatory crackdown / bans / SEC enforcement = SHORT
- Risk-on sentiment / Fed rate cuts / liquidity injection = LONG
- Risk-off / recession fear / rate hikes = SHORT
- Tariff escalation / trade war = SHORT (risk-off)
- Tariff relief / trade deals = LONG (risk-on)
- Major exchange hack / insolvency = SHORT
- Bitcoin halving / supply reduction = LONG
""".strip()

GOLD_SENTIMENT_RULES = """
- Geopolitical tension / war / conflict: direction is CONTEXT-DEPENDENT.
  Traditional safe-haven theory says LONG, but when the dollar is rallying
  as the crisis asset, gold often DUMPS on escalation and rallies on
  de-escalation. Read the current chart reaction.
- Inflation rising / CPI above expectations = LONG
- Fed rate cuts / dovish signals = LONG
- Risk-off event + WEAKENING dollar = LONG (classic safe haven)
- Risk-off event + STRENGTHENING dollar = SHORT (dollar wins the flight)
- Fed rate hikes / hawkish signals = SHORT
- Central bank gold buying = LONG
- Tariff escalation / trade war: direction CONTEXT-DEPENDENT
""".strip()


MARKET_OIL = MarketConfig(
    name="WTI Crude Oil",
    symbol="CL/USDT:USDT",
    keywords={
        "oil": 15, "crude": 15, "wti": 20, "brent": 20, "petroleum": 10,
        "opec": 15, "strait of hormuz": 25, "hormuz": 20, "iran": 15,
        "saudi": 10, "pipeline": 10, "refinery": 10, "energy": 5,
        "barrel": 10, "drilling": 8, "lng": 8, "natural gas": 8,
        "embargo": 15, "sanction": 12, "blockade": 30, "war": 10,
        "ceasefire": 15, "escalat": 12, "missile": 10, "strike": 8,
        "nuclear": 12, "geopolit": 10, "middle east": 12, "israel": 8,
        "lebanon": 8, "hezbollah": 10, "supply disruption": 20,
        "closed": 30, "closure": 30, "shut down": 30, "shut": 25,
        "seized": 30, "sank": 25, "attacked": 25, "struck": 20,
        "halted": 25, "blocked": 25, "fired on": 25, "hit": 15,
    },
    sentiment_rules=OIL_SENTIMENT_RULES,
    trades_24_7=False,
    nymex_hours=True,
)

MARKET_BTC = MarketConfig(
    name="Bitcoin",
    symbol="BTC/USDT:USDT",
    keywords={
        "bitcoin": 20, "btc": 20, "crypto": 15, "cryptocurrency": 15,
        "ethereum": 8, "blockchain": 8, "defi": 8,
        "sec crypto": 15, "spot etf": 20, "bitcoin etf": 20,
        "binance": 10, "coinbase": 10, "mining": 8, "halving": 15,
        "stablecoin": 8, "tether": 8, "digital asset": 10,
        "fed rate": 12, "interest rate": 10, "inflation": 8,
        "tariff": 10, "trade war": 12, "risk-on": 10, "risk-off": 10,
        "institutional": 10, "crypto ban": 20, "crypto regulation": 15,
    },
    sentiment_rules=BTC_SENTIMENT_RULES,
)

MARKET_GOLD = MarketConfig(
    name="Gold",
    symbol="XAU/USDT:USDT",
    keywords={
        "gold": 20, "bullion": 15, "precious metal": 15, "xau": 20,
        "fed": 10, "federal reserve": 12, "interest rate": 12,
        "inflation": 15, "cpi": 12, "deflation": 10,
        "dollar": 10, "usd": 8, "treasury": 10, "bond yield": 10,
        "safe haven": 20, "risk-off": 15, "uncertainty": 10,
        "geopolit": 12, "war": 12, "conflict": 10, "sanction": 10,
        "central bank": 12, "reserve": 8, "recession": 10,
        "tariff": 12, "trade war": 15,
    },
    sentiment_rules=GOLD_SENTIMENT_RULES,
)

ALL_MARKETS = [MARKET_OIL, MARKET_BTC, MARKET_GOLD]
MARKETS_BY_SYMBOL = {m.symbol: m for m in ALL_MARKETS}

MIN_RELEVANCE = 30
