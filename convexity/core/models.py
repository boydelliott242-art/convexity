"""Shared Pydantic v2 data models — the canonical contract for all of Convexity.

Every module (providers, analyzers, ranking, API, CLI) imports the types defined
here and codes against them exactly; nothing else may redefine these shapes.

Design notes
------------
* Financial fields are ``Optional[float]`` on purpose. Thin micro-caps frequently
  lack coverage for a metric. A ``None`` means "we genuinely do not have this
  datum"; it is *never* silently replaced by a fabricated number. Analyzers treat
  ``None`` as missing data, lower their confidence, and record a ``data_coverage``
  fraction so the honesty of each score is auditable.
* Every score is paired with a ``confidence`` and a ``data_coverage`` so a reader
  can always see *how much real evidence* stands behind a number.
* This module has **zero import side effects** and must remain importable on a
  machine with only pydantic installed.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ScoreCategory(str, Enum):
    """The twelve independent evidence categories aggregated into a composite.

    Each category is scored by exactly one analyzer. They are intended to be as
    *independent* as possible so that agreement across them is genuinely
    informative (correlated signals would overstate conviction).
    """

    VALUE = "value"
    GROWTH = "growth"
    QUALITY = "quality"
    FINANCIAL_HEALTH = "financial_health"
    TECHNICAL = "technical"
    MOMENTUM = "momentum"
    CATALYST = "catalyst"
    RISK = "risk"
    MANAGEMENT = "management"
    COMPETITIVE = "competitive"
    OWNERSHIP = "ownership"
    HISTORICAL_ANALOG = "historical_analog"


class CapTier(str, Enum):
    """Market-capitalisation bucket used for screening and peer construction."""

    NANO = "nano"      # < ~$50M
    MICRO = "micro"    # ~$50M – $300M
    SMALL = "small"    # ~$300M – $2B


# ---------------------------------------------------------------------------
# Raw market & fundamental data
# ---------------------------------------------------------------------------


class PriceBar(BaseModel):
    """A single OHLCV price bar (daily, unless a provider documents otherwise)."""

    model_config = ConfigDict(extra="ignore")

    date: _dt.date
    open: float
    high: float
    low: float
    close: float
    adj_close: Optional[float] = None
    volume: float


class FundamentalsPeriod(BaseModel):
    """One fiscal period of fundamentals (annual or quarterly).

    All financial line items are ``Optional[float]``. Derived ratios are stored
    rather than recomputed on the fly so that a provider that supplies them
    directly can do so, while analyzers may compute any that are missing.
    """

    model_config = ConfigDict(extra="ignore")

    period_end: _dt.date
    period_label: str  # e.g. "FY2025" or "Q1 2026"

    # Income statement
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    ebitda: Optional[float] = None
    eps_diluted: Optional[float] = None

    # Cash flow
    free_cash_flow: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    capex: Optional[float] = None

    # Balance sheet
    total_assets: Optional[float] = None
    total_debt: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    total_equity: Optional[float] = None
    shares_diluted: Optional[float] = None

    # Derived margins & returns (fractions, e.g. 0.35 == 35%)
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    fcf_margin: Optional[float] = None
    roic: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None

    # Liquidity & solvency
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    interest_coverage: Optional[float] = None


class ValuationSnapshot(BaseModel):
    """Point-in-time valuation multiples. Every field optional (often missing)."""

    model_config = ConfigDict(extra="ignore")

    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    pe: Optional[float] = None
    forward_pe: Optional[float] = None
    ev_ebitda: Optional[float] = None
    ev_sales: Optional[float] = None
    p_fcf: Optional[float] = None
    p_b: Optional[float] = None
    p_s: Optional[float] = None
    peg: Optional[float] = None


class NewsItem(BaseModel):
    """A single news headline with optional pre-computed sentiment in [-1, 1]."""

    model_config = ConfigDict(extra="ignore")

    published: _dt.datetime
    title: str
    source: str
    url: Optional[str] = None
    summary: Optional[str] = None
    sentiment: Optional[float] = None


class Filing(BaseModel):
    """A regulatory filing reference (e.g. an SEC 8-K, 10-Q, 10-K, Form 4)."""

    model_config = ConfigDict(extra="ignore")

    filed: _dt.date
    form_type: str
    title: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None


class InsiderTransaction(BaseModel):
    """An insider buy/sell as disclosed (e.g. via Form 4)."""

    model_config = ConfigDict(extra="ignore")

    date: _dt.date
    insider_name: str
    role: Optional[str] = None
    transaction_type: str  # e.g. "buy", "sell", "grant", "exercise"
    shares: Optional[float] = None
    value: Optional[float] = None


class InstitutionalHolding(BaseModel):
    """An institutional position (e.g. derived from 13F filings)."""

    model_config = ConfigDict(extra="ignore")

    holder: str
    shares: Optional[float] = None
    value: Optional[float] = None
    change_pct: Optional[float] = None  # period-over-period change in position
    as_of: Optional[_dt.date] = None


# ---------------------------------------------------------------------------
# Evidence — the atomic, auditable unit behind every score
# ---------------------------------------------------------------------------


Direction = Literal["bullish", "bearish", "neutral"]


class Evidence(BaseModel):
    """One auditable fact that contributed to a sub-score.

    Evidence is the heart of Convexity's honesty guarantee: every point of a
    score should trace back to one or more :class:`Evidence` items that name the
    datum, its value, its source and (where applicable) the as-of date and URL.
    """

    model_config = ConfigDict(extra="ignore")

    label: str
    value: str
    detail: Optional[str] = None
    source: str
    as_of: Optional[_dt.date] = None
    url: Optional[str] = None
    direction: Direction = "neutral"

    @classmethod
    def from_number(
        cls,
        label: str,
        value: Optional[float],
        *,
        source: str,
        direction: Direction = "neutral",
        unit: str = "",
        precision: int = 2,
        detail: Optional[str] = None,
        as_of: Optional[_dt.date] = None,
        url: Optional[str] = None,
    ) -> Evidence:
        """Build an :class:`Evidence` from a numeric value with consistent formatting.

        A ``None`` value is rendered as ``"n/a"`` and forced to ``neutral`` so a
        missing datum can never masquerade as a bullish or bearish signal.
        """
        if value is None:
            rendered = "n/a"
            direction = "neutral"
        else:
            rendered = f"{value:,.{precision}f}".rstrip("0").rstrip(".") if precision else f"{value:,.0f}"
            if unit:
                rendered = f"{rendered}{unit}"
        return cls(
            label=label,
            value=rendered,
            detail=detail,
            source=source,
            as_of=as_of,
            url=url,
            direction=direction,
        )


# ---------------------------------------------------------------------------
# Aggregated security data (the input to every analyzer)
# ---------------------------------------------------------------------------


class SecurityData(BaseModel):
    """All raw, provider-sourced data assembled for a single security.

    This is the sole input contract for analyzers. The aggregator merges output
    from one or more :class:`~convexity.core.contracts.DataProvider` instances
    into this object, recording each contributing source in ``data_sources`` and
    any gaps or anomalies in ``data_warnings`` (never fabricating missing values).
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None
    cap_tier: Optional[CapTier] = None
    currency: str = "USD"
    as_of: _dt.datetime

    valuation: ValuationSnapshot = Field(default_factory=ValuationSnapshot)
    fundamentals: List[FundamentalsPeriod] = Field(default_factory=list)  # newest first
    price_history: List[PriceBar] = Field(default_factory=list)           # oldest first
    news: List[NewsItem] = Field(default_factory=list)
    filings: List[Filing] = Field(default_factory=list)
    insider_transactions: List[InsiderTransaction] = Field(default_factory=list)
    institutional_holdings: List[InstitutionalHolding] = Field(default_factory=list)
    peers: List[str] = Field(default_factory=list)

    data_sources: List[str] = Field(default_factory=list)
    data_warnings: List[str] = Field(default_factory=list)

    @property
    def latest_fundamentals(self) -> Optional[FundamentalsPeriod]:
        """The most recent fundamentals period, or ``None`` if none are available."""
        return self.fundamentals[0] if self.fundamentals else None

    @property
    def market_cap(self) -> Optional[float]:
        """Market cap from the valuation snapshot, if known."""
        return self.valuation.market_cap


# ---------------------------------------------------------------------------
# Scoring outputs
# ---------------------------------------------------------------------------


class SubScore(BaseModel):
    """The output of a single analyzer for a single category.

    A sub-score is deliberately self-describing: alongside the 0–100 ``score`` it
    carries the ``confidence`` (how trustworthy the score is given data quality),
    the ``weight`` it should receive in the composite, a human ``rationale``, the
    list of :class:`Evidence` behind it, any ``flags`` (e.g. ``MISSING_DATA``),
    and a ``data_coverage`` fraction describing how much of the required input was
    actually present.
    """

    model_config = ConfigDict(extra="ignore")

    category: ScoreCategory
    score: float = Field(..., ge=0.0, le=100.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(..., ge=0.0)
    rationale: str
    evidence: List[Evidence] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)
    data_coverage: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("score")
    @classmethod
    def _check_score(cls, v: float) -> float:
        """Defensive clamp: a score must live in [0, 100] (validated, not silently)."""
        if not (0.0 <= v <= 100.0):
            raise ValueError(f"score must be within [0, 100]; got {v}")
        return v


class CompanyAnalysis(BaseModel):
    """The full, explainable analysis of one company after ranking.

    Combines every :class:`SubScore` into a ``composite_score`` plus narrative
    summaries. ``signal_agreement`` records how many independent categories point
    the same direction — high conviction is justified only when that agreement is
    high, not when a single category is extreme.
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str
    name: str
    industry: Optional[str] = None
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    cap_tier: Optional[CapTier] = None

    composite_score: float = Field(..., ge=0.0, le=100.0)
    conviction_confidence: float = Field(..., ge=0.0, le=1.0)
    rank: Optional[int] = None

    subscores: List[SubScore] = Field(default_factory=list)

    thesis: str = ""
    bull_case: List[str] = Field(default_factory=list)
    bear_case: List[str] = Field(default_factory=list)
    catalysts: List[str] = Field(default_factory=list)
    principal_risks: List[str] = Field(default_factory=list)

    valuation_summary: str = ""
    fundamental_summary: str = ""
    technical_summary: str = ""
    confidence_explanation: str = ""
    monitoring_checklist: List[str] = Field(default_factory=list)

    signal_agreement: float = Field(default=0.0, ge=0.0, le=1.0)

    def subscore_by_category(self, category: ScoreCategory) -> Optional[SubScore]:
        """Return the sub-score for ``category``, or ``None`` if it was not produced."""
        for sub in self.subscores:
            if sub.category == category:
                return sub
        return None


# ---------------------------------------------------------------------------
# Scan parameters & results
# ---------------------------------------------------------------------------


class ScanParams(BaseModel):
    """User-tunable parameters controlling a screen + analysis run."""

    model_config = ConfigDict(extra="ignore")

    min_market_cap: float = 50_000_000
    max_market_cap: float = 2_000_000_000
    min_avg_dollar_volume: float = 200_000
    exclude_sectors: List[str] = Field(default_factory=list)
    top_n: int = 5
    universe_limit: Optional[int] = None


class ScanResult(BaseModel):
    """The complete output of one scan: timings, counts, rankings and notes."""

    model_config = ConfigDict(extra="ignore")

    generated_at: _dt.datetime
    params: ScanParams
    universe_size: int
    screened_count: int
    analyzed_count: int
    error_count: int
    top: List[CompanyAnalysis] = Field(default_factory=list)
    all_ranked: List[CompanyAnalysis] = Field(default_factory=list)
    category_weights: Dict[str, float] = Field(default_factory=dict)
    elapsed_seconds: float = 0.0
    notes: List[str] = Field(default_factory=list)


__all__ = [
    "ScoreCategory",
    "CapTier",
    "PriceBar",
    "FundamentalsPeriod",
    "ValuationSnapshot",
    "NewsItem",
    "Filing",
    "InsiderTransaction",
    "InstitutionalHolding",
    "Direction",
    "Evidence",
    "SecurityData",
    "SubScore",
    "CompanyAnalysis",
    "ScanParams",
    "ScanResult",
]
