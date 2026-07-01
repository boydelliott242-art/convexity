"""Unit tests for :class:`convexity.analysis.risk.RiskAnalyzer`.

The RISK analyzer aggregates many independent fragility dimensions (negative
equity, leverage, cash runway, dilution, liquidity, volatility, going-concern /
litigation language) into a single :class:`SubScore`. By the platform-wide
convention a **higher RISK score means LOWER risk (a safer profile)**.

These tests build small, hand-crafted :class:`SecurityData` objects inline (no
conftest fixtures) and assert the shared honesty guarantees:

* A fortress profile (positive equity, low leverage, long runway, no dilution,
  deep liquidity, calm tape, no distress language) scores meaningfully
  **higher** (safer) than a fragile one.
* **All-missing** inputs fall back to a neutral (50), low-confidence,
  ``MISSING_DATA``-flagged sub-score.
* Every score lives in ``[0, 100]`` with a populated, auditable evidence list.
* The emitted :class:`SubScore` carries the RISK category.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pytest

from convexity.analysis.risk import RiskAnalyzer
from convexity.core.contracts import AnalysisContext
from convexity.core.models import (
    Filing,
    FundamentalsPeriod,
    NewsItem,
    PriceBar,
    ScoreCategory,
    SecurityData,
    SubScore,
    ValuationSnapshot,
)

_AS_OF = dt.datetime(2026, 1, 1, 0, 0, 0)
_START = dt.date(2025, 1, 1)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _period(
    year: int,
    *,
    total_equity: Optional[float] = None,
    total_assets: Optional[float] = None,
    debt_to_equity: Optional[float] = None,
    interest_coverage: Optional[float] = None,
    cash_and_equivalents: Optional[float] = None,
    free_cash_flow: Optional[float] = None,
    shares_diluted: Optional[float] = None,
) -> FundamentalsPeriod:
    """Build one annual fundamentals period labelled by fiscal ``year``."""
    return FundamentalsPeriod(
        period_end=dt.date(year, 12, 31),
        period_label=f"FY{year}",
        total_equity=total_equity,
        total_assets=total_assets,
        debt_to_equity=debt_to_equity,
        interest_coverage=interest_coverage,
        cash_and_equivalents=cash_and_equivalents,
        free_cash_flow=free_cash_flow,
        shares_diluted=shares_diluted,
    )


def _calm_prices(start: float = 20.0, n: int = 60, volume: float = 200_000.0) -> List[PriceBar]:
    """A long, calm, deeply-traded price series (low volatility, deep liquidity)."""
    bars: List[PriceBar] = []
    price = start
    for i in range(n):
        # Tiny alternating wiggle keeps realised volatility low but non-zero.
        price = start * (1.0 + (0.001 if i % 2 == 0 else -0.001))
        bars.append(
            PriceBar(
                date=_START + dt.timedelta(days=i),
                open=price,
                high=price * 1.002,
                low=price * 0.998,
                close=price,
                adj_close=price,
                volume=volume,
            )
        )
    return bars


def _wild_prices(start: float = 5.0, n: int = 60, volume: float = 1_000.0) -> List[PriceBar]:
    """A thin, violently-volatile price series (high vol, thin liquidity)."""
    bars: List[PriceBar] = []
    price = start
    for i in range(n):
        # Large alternating swings -> high realised volatility.
        price = start * (1.0 + (0.25 if i % 2 == 0 else -0.20))
        bars.append(
            PriceBar(
                date=_START + dt.timedelta(days=i),
                open=price,
                high=price * 1.20,
                low=price * 0.80,
                close=price,
                adj_close=price,
                volume=volume,
            )
        )
    return bars


def _security(
    fundamentals: Optional[List[FundamentalsPeriod]] = None,
    price_history: Optional[List[PriceBar]] = None,
    filings: Optional[List[Filing]] = None,
    news: Optional[List[NewsItem]] = None,
) -> SecurityData:
    """Wrap the supplied inputs into a minimal SecurityData (fundamentals newest-first)."""
    return SecurityData(
        ticker="TEST",
        name="Test Co",
        currency="USD",
        as_of=_AS_OF,
        valuation=ValuationSnapshot(market_cap=250_000_000.0),
        fundamentals=fundamentals or [],
        price_history=price_history or [],
        filings=filings or [],
        news=news or [],
    )


def _ctx() -> AnalysisContext:
    """A bare AnalysisContext (no peer/universe stats)."""
    return AnalysisContext(peer_stats=None, universe_stats=None, config=None)


def _safe_security() -> SecurityData:
    """A fortress: deep equity, low debt, long runway, buybacks, calm/liquid tape."""
    funds = [
        _period(
            2025,
            total_equity=500.0,
            total_assets=700.0,
            debt_to_equity=0.2,
            interest_coverage=15.0,
            cash_and_equivalents=300.0,
            free_cash_flow=80.0,  # cash-generative -> no runway risk.
            shares_diluted=95.0,  # share count fell vs prior -> buybacks.
        ),
        _period(2024, shares_diluted=100.0),
    ]
    return _security(
        fundamentals=funds,
        price_history=_calm_prices(),
        news=[NewsItem(published=_AS_OF, title="Company reports a solid, steady quarter", source="Reuters")],
    )


def _fragile_security() -> SecurityData:
    """A fragile micro-cap: negative equity, heavy debt, short runway, dilution,
    thin/violent tape, and a going-concern disclosure in an SEC filing."""
    funds = [
        _period(
            2025,
            total_equity=-50.0,  # shareholders' deficit (hard red flag).
            total_assets=200.0,
            debt_to_equity=4.0,
            interest_coverage=0.5,
            cash_and_equivalents=10.0,
            free_cash_flow=-60.0,  # burning cash -> short runway.
            shares_diluted=180.0,  # heavy dilution vs prior.
        ),
        _period(2024, shares_diluted=100.0),
    ]
    filings = [
        Filing(
            filed=dt.date(2025, 12, 1),
            form_type="10-K",
            title="Annual report",
            summary="There is substantial doubt about the company's ability to continue as a going concern.",
        ),
    ]
    return _security(fundamentals=funds, price_history=_wild_prices(), filings=filings)


# ---------------------------------------------------------------------------
# Core contract: safe high, fragile low, missing -> neutral
# ---------------------------------------------------------------------------


class TestRiskScoring:
    def test_safe_profile_scores_high(self) -> None:
        sub = RiskAnalyzer().analyze(_safe_security(), _ctx())
        assert isinstance(sub, SubScore)
        assert sub.category == ScoreCategory.RISK
        assert sub.score >= 65.0, f"a fortress should read SAFE (high), got {sub.score}"

    def test_fragile_profile_scores_low(self) -> None:
        sub = RiskAnalyzer().analyze(_fragile_security(), _ctx())
        assert sub.score <= 30.0, f"a fragile micro-cap should read RISKY (low), got {sub.score}"
        assert "MISSING_DATA" not in sub.flags

    def test_safe_strictly_above_fragile(self) -> None:
        safe = RiskAnalyzer().analyze(_safe_security(), _ctx()).score
        fragile = RiskAnalyzer().analyze(_fragile_security(), _ctx()).score
        assert safe > fragile + 30.0

    def test_all_missing_is_neutral_low_confidence(self) -> None:
        sub = RiskAnalyzer().analyze(_security(), _ctx())
        assert sub.score == pytest.approx(50.0)
        assert sub.confidence <= 0.2
        assert "MISSING_DATA" in sub.flags
        assert sub.data_coverage == pytest.approx(0.0)

    def test_fragile_raises_expected_flags(self) -> None:
        sub = RiskAnalyzer().analyze(_fragile_security(), _ctx())
        assert "NEGATIVE_EQUITY" in sub.flags
        assert "GOING_CONCERN_RISK" in sub.flags


# ---------------------------------------------------------------------------
# Evidence, range
# ---------------------------------------------------------------------------


class TestRiskEvidence:
    def test_evidence_is_populated(self) -> None:
        sub = RiskAnalyzer().analyze(_safe_security(), _ctx())
        assert sub.evidence, "a scored risk profile must emit evidence"
        for ev in sub.evidence:
            assert ev.value and isinstance(ev.value, str)
            assert ev.source
            assert ev.direction in {"bullish", "bearish", "neutral"}

    @pytest.mark.parametrize("builder", [_safe_security, _fragile_security])
    def test_score_always_within_range(self, builder) -> None:
        sub = RiskAnalyzer().analyze(builder(), _ctx())
        assert 0.0 <= sub.score <= 100.0
        assert 0.0 <= sub.confidence <= 1.0
        assert 0.0 <= sub.data_coverage <= 1.0

    def test_risk_carries_zero_additive_weight(self) -> None:
        # RISK is applied as a dampener, never summed into the additive composite.
        sub = RiskAnalyzer().analyze(_safe_security(), _ctx())
        assert sub.weight == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


class TestRiskPurity:
    def test_deterministic(self) -> None:
        sec = _fragile_security()
        a = RiskAnalyzer().analyze(sec, _ctx())
        b = RiskAnalyzer().analyze(sec, _ctx())
        assert a.score == b.score
        assert a.confidence == b.confidence
        assert a.flags == b.flags

    def test_does_not_mutate_input(self) -> None:
        sec = _safe_security()
        n_funds = len(sec.fundamentals)
        n_bars = len(sec.price_history)
        RiskAnalyzer().analyze(sec, _ctx())
        assert len(sec.fundamentals) == n_funds
        assert len(sec.price_history) == n_bars


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_analyzer_is_registered_for_risk() -> None:
    from convexity.core.registry import get_analyzer

    cls = get_analyzer(ScoreCategory.RISK)
    assert cls is RiskAnalyzer


def test_class_attrs() -> None:
    assert RiskAnalyzer.category == ScoreCategory.RISK
    assert RiskAnalyzer.default_weight == pytest.approx(0.0)
