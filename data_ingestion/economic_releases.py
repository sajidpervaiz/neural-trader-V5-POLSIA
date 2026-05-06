"""
Economic Releases with BLS API for CPI/PPI/NFP, consensus deviation scoring.
"""

from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

try:
    import aiohttp
    _AIOHTTP = True
except ImportError:
    aiohttp = None
    _AIOHTTP = False


class IndicatorType(Enum):
    CPI = "CPI"
    PPI = "PPI"
    NFP = "NFP"
    GDP = "GDP"
    RETAIL_SALES = "RETAIL_SALES"
    PMI = "PMI"
    UNEMPLOYMENT = "UNEMPLOYMENT"


@dataclass
class EconomicRelease:
    release_id: str
    indicator_type: IndicatorType
    title: str
    release_date: datetime
    actual: Optional[float] = None
    consensus: Optional[float] = None
    previous: Optional[float] = None
    deviation: Optional[float] = None
    deviation_pct: Optional[float] = None
    importance: int = 3
    source: str = "BLS"


class EconomicReleases:
    """
    Economic releases manager with:
    - BLS API integration
    - Consensus deviation calculation
    - Historical tracking
    - Impact scoring
    """

    def __init__(self, enable_paper_mode: bool = True):
        self.paper_mode = enable_paper_mode
        self.releases: List[EconomicRelease] = []
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Initialize Economic Releases."""
        if _AIOHTTP:
            self._http_session = aiohttp.ClientSession()
        elif not self.paper_mode:
            logger.warning("aiohttp not installed — falling back to paper mode for EconomicReleases")
            self.paper_mode = True

        await self._load_historical_releases()
        await self._load_scheduled_releases()

        logger.info(f"Loaded {len(self.releases)} economic releases")

    async def _load_historical_releases(self) -> None:
        """Load historical economic data."""
        if self.paper_mode:
            now = datetime.now()

            self.releases.extend([
                EconomicRelease(
                    release_id=f"cpi_{i}",
                    indicator_type=IndicatorType.CPI,
                    title=f"Consumer Price Index {i}",
                    release_date=now - timedelta(days=30 * i),
                    actual=3.0 + (i % 3) * 0.1,
                    consensus=3.1,
                    previous=3.0 + ((i - 1) % 3) * 0.1,
                    importance=5,
                    source="BLS",
                )
                for i in range(1, 7)
            ])

            return

        try:
            url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
            series_ids = ["CUUR0000SA0", "WPSFD49207", "CES0000000001"]

            async with self._http_session.post(
                url,
                json={"seriesid": series_ids, "startyear": "2018", "endyear": "2024"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self._process_bls_data(data)

        except Exception as e:
            logger.error(f"Error loading historical releases: {e}")

    def _process_bls_data(self, data: dict) -> None:
        """Process BLS API response."""
        for series in data.get('Results', {}).get('series', []):
            series_id = series.get('seriesID')
            indicator = self._map_series_to_indicator(series_id)

            if not indicator:
                continue

            for item in series.get('data', []):
                try:
                    year = int(item.get('year'))
                    period = item.get('period', '')

                    if not period:
                        continue
                    if period.startswith('M'):
                        month = int(period[1:])
                    elif period.startswith('Q'):
                        quarter = int(period[1:])
                        month = quarter * 3
                    else:
                        continue

                    release_date = datetime(year, month, 1)
                    value = float(item.get('value', 0))

                    release = EconomicRelease(
                        release_id=f"{series_id}_{year}_{month}",
                        indicator_type=indicator,
                        title=f"{indicator.value} {year}-{month:02d}",
                        release_date=release_date,
                        actual=value,
                        importance=5,
                        source="BLS",
                    )

                    self.releases.append(release)

                except (ValueError, TypeError):
                    continue

    def _map_series_to_indicator(self, series_id: str) -> Optional[IndicatorType]:
        """Map BLS series ID to indicator type."""
        if "CUUR" in series_id:
            return IndicatorType.CPI
        elif "WPS" in series_id:
            return IndicatorType.PPI
        elif "CES" in series_id:
            return IndicatorType.NFP
        return None

    async def _load_scheduled_releases(self) -> None:
        """Load scheduled economic releases."""
        now = datetime.now()
        scheduled = [
            (IndicatorType.CPI, "Consumer Price Index", 5),
            (IndicatorType.PPI, "Producer Price Index", 4),
            (IndicatorType.NFP, "Non-Farm Payrolls", 5),
            (IndicatorType.PMI, "Manufacturing PMI", 4),
        ]

        existing_upcoming_types = {
            release.indicator_type
            for release in self.releases
            if release.release_date > now
        }

        # Keep the system operational even without a dedicated schedule API by
        # adding deterministic upcoming entries for missing indicators.
        for i, (indicator, title, importance) in enumerate(scheduled):
            if indicator in existing_upcoming_types:
                continue
            release = EconomicRelease(
                release_id=f"{indicator.value.lower()}_{i}",
                indicator_type=indicator,
                title=title,
                release_date=now + timedelta(days=7 + i * 5),
                importance=importance,
                source="BLS",
            )
            self.releases.append(release)

    async def calculate_consensus_deviation(
        self,
        release_id: str,
        consensus: float,
    ) -> Optional[float]:
        """Calculate deviation from consensus."""
        for release in self.releases:
            if release.release_id == release_id and release.actual:
                deviation = release.actual - consensus
                release.deviation = deviation
                release.deviation_pct = (deviation / consensus * 100) if consensus != 0 else None
                return deviation

        return None

    def get_upcoming_releases(
        self,
        days: int = 14,
        min_importance: int = 3,
    ) -> List[EconomicRelease]:
        """Get upcoming economic releases."""
        now = datetime.now()
        cutoff = now + timedelta(days=days)

        return [
            release
            for release in self.releases
            if now < release.release_date <= cutoff
            and release.importance >= min_importance
        ]

    def get_recent_releases(
        self,
        days: int = 30,
    ) -> List[EconomicRelease]:
        """Get recent economic releases."""
        now = datetime.now()
        cutoff = now - timedelta(days=days)

        return [
            release
            for release in self.releases
            if cutoff <= release.release_date <= now
        ]

    def get_high_deviation_releases(
        self,
        threshold_pct: float = 10.0,
        days: int = 90,
    ) -> List[EconomicRelease]:
        """Get releases with high deviation from consensus."""
        now = datetime.now()
        cutoff = now - timedelta(days=days)

        return [
            release
            for release in self.releases
            if release.release_date >= cutoff
            and release.deviation_pct
            and abs(release.deviation_pct) >= threshold_pct
        ]

    def get_indicator_impact_score(
        self,
        indicator_type: IndicatorType,
    ) -> float:
        """Get market impact score for indicator."""
        impact_scores = {
            IndicatorType.CPI: 0.95,
            IndicatorType.NFP: 0.93,
            IndicatorType.PPI: 0.85,
            IndicatorType.GDP: 0.90,
            IndicatorType.RETAIL_SALES: 0.80,
            IndicatorType.PMI: 0.75,
            IndicatorType.UNEMPLOYMENT: 0.88,
        }
        return impact_scores.get(indicator_type, 0.70)

    async def close(self) -> None:
        """Clean up resources."""
        if self._http_session:
            await self._http_session.close()
