"""Geo-Political News-Driven Trading Strategy — config constants.

Source of truth: docs/GEO_POLITICAL_TRADING_STRATEGY.md (mirrored on Desktop).
The actual scorer + RSS poller live in engine/geopolitical_scorer.py and
data_ingestion/geopolitical_news.py respectively. This module exports only
the per-market keyword/sentiment configs they consume.
"""
from __future__ import annotations


RSS_FEEDS: list[str] = [
    # General / geopolitical
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    # Reuters retired feeds.reuters.com in 2020; FoxNews world feed is a
    # reliable English-language replacement covering geopolitics + energy.
    "https://moxie.foxnews.com/google-publisher/world.xml",
    "https://www.theguardian.com/world/middleeast/rss",
    "https://feeds.npr.org/1004/rss.xml",
    # Markets / finance
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    # Oil / energy
    "https://oilprice.com/rss/main",
    # Crypto
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://bitcoinmagazine.com/.rss/full/",
    # Gold
    "https://www.kitco.com/rss/gold.xml",
]


# Per-symbol relevance keywords (token → relevance weight).
# Direction (long/short) is decided downstream from BULLISH_TOKENS / BEARISH_TOKENS.
MARKET_CONFIGS: dict[str, dict] = {
    "CL/USDT:USDT": {  # WTI Crude Oil
        "name": "WTI Crude Oil",
        "symbol": "CL/USDT:USDT",
        "keywords": {
            "oil": 15, "crude": 15, "wti": 20, "brent": 20, "petroleum": 10,
            "opec": 15, "strait of hormuz": 25, "hormuz": 20, "iran": 15,
            "saudi": 10, "pipeline": 10, "refinery": 10, "energy": 5, "barrel": 10,
            "drilling": 8, "lng": 8, "natural gas": 8,
            "embargo": 15, "sanction": 12, "blockade": 30, "war": 10,
            "ceasefire": 15, "escalat": 12, "missile": 10, "strike": 8, "nuclear": 12,
            "geopolit": 10, "middle east": 12, "israel": 8, "lebanon": 8, "hezbollah": 10,
            "supply disruption": 20,
            "closed": 30, "closure": 30, "shut down": 30, "shut": 25,
            "seized": 30, "sank": 25, "attacked": 25, "struck": 20,
            "halted": 25, "blocked": 25, "fired on": 25, "hit": 15,
        },
        "trading_hours": {
            "start": "23:00", "end": "22:00",
            "days": ["sun", "mon", "tue", "wed", "thu", "fri"],
        },
    },
    "BTC/USDT:USDT": {  # Bitcoin
        "name": "Bitcoin",
        "symbol": "BTC/USDT:USDT",
        "keywords": {
            "bitcoin": 20, "btc": 20, "crypto": 15, "cryptocurrency": 15,
            "ethereum": 8, "blockchain": 8, "defi": 8,
            "sec crypto": 15, "spot etf": 20, "bitcoin etf": 20,
            "binance": 10, "coinbase": 10, "mining": 8, "halving": 15,
            "stablecoin": 8, "tether": 8, "digital asset": 10,
            "fed rate": 12, "interest rate": 10, "inflation": 8,
            "tariff": 10, "trade war": 12, "risk-on": 10, "risk-off": 10,
            "institutional": 10, "crypto ban": 20, "crypto regulation": 15,
        },
        "trading_hours": None,  # 24/7
    },
    "XAU/USDT:USDT": {  # Gold
        "name": "Gold",
        "symbol": "XAU/USDT:USDT",
        "keywords": {
            "gold": 20, "bullion": 15, "precious metal": 15, "xau": 20,
            "fed": 10, "federal reserve": 12, "interest rate": 12,
            "inflation": 15, "cpi": 12, "deflation": 10,
            "dollar": 10, "usd": 8, "treasury": 10, "bond yield": 10,
            "safe haven": 20, "risk-off": 15, "uncertainty": 10,
            "geopolit": 12, "war": 12, "conflict": 10, "sanction": 10,
            "central bank": 12, "reserve": 8, "recession": 10,
            "tariff": 12, "trade war": 15,
        },
        "trading_hours": None,  # 24/7 on crypto exchanges
    },
}


# Direction inference — applied to each (article, symbol) pair after relevance passes MIN_RELEVANCE.
# Each token's sign is the direction it pushes for the symbol it's matched against.
# OIL: war/disruption/escalation = LONG (supply fear); peace/ceasefire = SHORT.
# BTC: ETF approval / risk-on / institutional = LONG; ban/hack/risk-off = SHORT.
# GOLD: inflation/Fed dovish/safe-haven/geopolitics = LONG; Fed hawkish/strong-dollar = SHORT.
DIRECTION_TOKENS: dict[str, dict[str, int]] = {
    "CL/USDT:USDT": {
        "war": 1, "escalat": 1, "missile": 1, "strike": 1, "attacked": 1, "struck": 1,
        "blockade": 1, "embargo": 1, "sanction": 1, "closed": 1, "closure": 1,
        "shut": 1, "seized": 1, "sank": 1, "halted": 1, "blocked": 1, "fired on": 1, "hit": 1,
        "supply disruption": 1, "opec cut": 1, "pipeline attack": 1, "refinery fire": 1,
        "hormuz": 1, "iran tension": 1,
        "ceasefire": -1, "peace": -1, "truce": -1, "deal": -1, "diplomacy": -1,
        "opec increase": -1, "supply boost": -1, "demand destruction": -1, "recession": -1,
        "oversupply": -1, "glut": -1,
    },
    "BTC/USDT:USDT": {
        "etf approv": 1, "spot etf": 1, "institutional": 1, "halving": 1,
        "rate cut": 1, "dovish": 1, "risk-on": 1, "adoption": 1, "bullish": 1,
        "rally": 1, "surge": 1, "ath": 1, "all-time high": 1, "all time high": 1,
        "tariff relief": 1, "trade deal": 1, "approval": 1,
        "ban": -1, "crackdown": -1, "lawsuit": -1, "sec sue": -1, "hack": -1, "exploit": -1,
        "fraud": -1, "insolvenc": -1, "bankrupt": -1, "rate hike": -1, "hawkish": -1,
        "risk-off": -1, "trade war": -1, "tariff": -1, "regulation": -1, "crypto ban": -1,
        "selloff": -1, "crash": -1, "plunge": -1, "dump": -1,
    },
    "XAU/USDT:USDT": {
        "inflation": 1, "cpi rising": 1, "rate cut": 1, "dovish": 1, "weak dollar": 1,
        "weakening dollar": 1, "safe haven": 1, "risk-off": 1, "war": 1, "conflict": 1,
        "central bank buy": 1, "geopolit": 1, "uncertainty": 1, "recession": 1,
        "rate hike": -1, "hawkish": -1, "strong dollar": -1, "strengthening dollar": -1,
        "deflation": -1, "rally in stocks": -1, "risk-on": -1,
    },
}


# Strategy parameters
MIN_RELEVANCE: int = 30
NEWS_CONFIDENCE_THRESHOLD: int = 60
WINDOW_HOURS: float = 6.0
MAX_CONCURRENT_TRADES: int = 3
