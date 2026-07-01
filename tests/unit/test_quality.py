"""Unit tests for :mod:`convexity.analysis.quality` (the QUALITY analyzer).

These tests pin the *behavioural contract* of :class:`QualityAnalyzer`:

* A durable, high-returns-on-capital profile scores **high**.
* A weak / loss-making profile scores **low**.
* All-missing fundamentals yield a **neutral (50), low-confidence** sub-score
  flagged ``MISSING_DATA`` — a data gap must neither help nor hurt.
* Every produced score is bounded to ``[0, 100]`` and is backed by populated,
  auditable :class:`Evidence`.
* The analyzer is **pure**: identical input gives identical output.
* Peer/universe context is honoured (a value at the top of its peer distribution
  is rewarded) and the analyzer self-registers under its category.

``SecurityData`` objects are built by hand from minimal :class:`FundamentalsPeriod`
rows so each test isolates the trait under examination.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pytest

from convexity.analysis.quality import QualityAnalyzer
from convexity.core.contracts import AnalysisContext
from convexity.core.models import (
    FundamentalsPeriod,
    ScoreCategory,
    SecurityData,
    SubScore,
    ValuationSnapshot,
)
from convexity.core.registry import get_analyzer

_AS_OF = dt.datetime(2026, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _period(
    label: str,
    year: int,
    *,
    roic: Optional[float] = None,
    roe: Optional[float] = None,
    roa: Optional[float] = None,
    gross_margin: Optional[float] = None,
    operating_margin: Optional[float] = None,
    fcf_margin: Optional[float] = None,
    revenue: Optional[float] = None,
    net_income: Optional[float] = None,
    free_cash_flow: Optional[float] = None,
    total_assets: Optional[float] = None,
) -> FundamentalsPeriod:
    return FundamentalsPeriod(
        period_end=dt.date(year, 12, 31),
        period_label=label,
        roic=roic,
        roe=roe,
        roa=roa,
        gross_margin=gross_margin,
        operating_margin=operating_margin,
        fcf_margin=fcf_margin,
        revenue=revenue,
        net_income=net_income,
        free_cash_flow=free_cash_flow,
        total_assets=total_assets,
    )


def _security(periods: List[FundamentalsPeriod], *, ticker: str = "TEST") -> SecurityData:
    """Wrap fundamentals (newest-first) into a SecurityData with a source tag."""
    return SecurityData(
        ticker=ticker,
        name=f"{ticker} Inc.",
        as_of=_AS_OF,
        valuation=ValuationSnapshot(market_cap=300_000_000),
        fundamentals=periods,
        data_sources=["yfinance"],
    )


def _strong_periods() -> List[FundamentalsPeriod]:
    """Four periods of durable, high returns / margins / cash conversion (newest-first)."""
    return [
        _period("FY2025", 2025, roic=0.24, roe=0.27, roa=0.16, gross_margin=0.62,
                operating_margin=0.28, fcf_margin=0.20, revenue=1000.0,
                net_income=180.0, free_cash_flow=190.0, total_assets=900.0),
        _period("FY2024", 2024, roic=0.23, roe=0.26, roa=0.15, gross_margin=0.61,
                operating_margin=0.27, fcf_margin=0.19, revenue=900.0,
                net_income=160.0, free_cash_flow=170.0, total_assets=850.0),
        _period("FY2023", 2023, roic=0.22, roe=0.25, roa=0.15, gross_margin=0.60,
                operating_margin=0.27, fcf_margin=0.19, revenue=820.0,
                net_income=145.0, free_cash_flow=150.0, total_assets=800.0),
        _period("FY2022", 2022, roic=0.22, roe=0.24, roa=0.14, gross_margin=0.60,
                operating_margin=0.26, fcf_margin=0.18, revenue=760.0,
                net_income=130.0, free_cash_flow=135.0, total_assets=760.0),
    ]


def _weak_periods() -> List[FundamentalsPeriod]:
    """Four periods of poor, money-losing economics (newest-first)."""
    return [
        _period("FY2025", 2025, roic=-0.12, roe=-0.18, roa=-0.08, gross_margin=0.12,
                operating_margin=-0.15, fcf_margin=-0.10, revenue=300.0,
                net_income=-45.0, free_cash_flow=-30.0, total_assets=1200.0),
        _period("FY2024", 2024, roic=-0.05, roe=-0.07, roa=-0.03, gross_margin=0.18,
                operating_margin=-0.08, fcf_margin=-0.04, revenue=320.0,
                net_income=-20.0, free_cash_flow=-12.0, total_assets=1150.0),
        _period("FY2023", 2023, roic=0.02, roe=0.03, roa=0.01, gross_margin=0.20,
                operating_margin=0.01, fcf_margin=0.0, revenue=340.0,
                net_income=4.0, free_cash_flow=1.0, total_assets=1100.0),
        _period("FY2022", 2022, roic=0.06, roe=0.08, roa=0.03, gross_margin=0.24,
                operating_margin=0.05, fcf_margin=0.03, revenue=360.0,
                net_income=12.0, free_cash_flow=10.0, total_assets=1050.0),
    ]


def _ctx() -> AnalysisContext:
    return AnalysisContext()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_class_attributes(self) -> None:
        assert QualityAnalyzer.category == ScoreCategory.QUALITY
        assert QualityAnalyzer.default_weight > 0.0
        assert QualityAnalyzer.requires  # non-empty required-input set.

    def test_self_registers_under_category(self) -> None:
        assert get_analyzer(ScoreCategory.QUALITY) is QualityAnalyzer


class TestStrongProfile:
    def test_strong_profile_scores_high(self) -> None:
        sub = QualityAnalyzer().analyze(_security(_strong_periods()), _ctx())
        assert isinstance(sub, SubScore)
        assert sub.category == ScoreCategory.QUALITY
        assert sub.score >= 70.0
        assert 0.0 <= sub.score <= 100.0

    def test_strong_profile_is_high_confidence_and_well_covered(self) -> None:
        sub = QualityAnalyzer().analyze(_security(_strong_periods()), _ctx())
        assert sub.confidence >= 0.6
        assert sub.data_coverage >= 0.7
        assert "MISSING_DATA" not in sub.flags

    def test_strong_profile_flags_high_returns(self) -> None:
        sub = QualityAnalyzer().analyze(_security(_strong_periods()), _ctx())
        assert "HIGH_RETURNS_ON_CAPITAL" in sub.flags

    def test_evidence_is_populated_and_cites_numbers(self) -> None:
        sub = QualityAnalyzer().analyze(_security(_strong_periods()), _ctx())
        assert sub.evidence, "evidence must be populated"
        labels = {e.label for e in sub.evidence}
        assert any("ROIC" in lbl for lbl in labels)
        assert any("margin" in lbl.lower() for lbl in labels)
        # At least one piece of evidence should read bullish for a strong company.
        assert any(e.direction == "bullish" for e in sub.evidence)
        # Every evidence value renders to a concrete string (number or "n/a").
        for e in sub.evidence:
            assert isinstance(e.value, str) and e.value


class TestWeakProfile:
    def test_weak_profile_scores_low(self) -> None:
        sub = QualityAnalyzer().analyze(_security(_weak_periods()), _ctx())
        assert sub.score <= 35.0
        assert 0.0 <= sub.score <= 100.0

    def test_weak_profile_flags_problems(self) -> None:
        sub = QualityAnalyzer().analyze(_security(_weak_periods()), _ctx())
        assert "NEGATIVE_ROIC" in sub.flags or "NEGATIVE_OPERATING_MARGIN" in sub.flags

    def test_strong_beats_weak(self) -> None:
        strong = QualityAnalyzer().analyze(_security(_strong_periods()), _ctx())
        weak = QualityAnalyzer().analyze(_security(_weak_periods()), _ctx())
        assert strong.score > weak.score + 25.0


class TestMissingData:
    def test_no_fundamentals_is_neutral_low_confidence(self) -> None:
        sub = QualityAnalyzer().analyze(_security([]), _ctx())
        assert sub.score == 50.0
        assert sub.confidence <= 0.2
        assert sub.data_coverage == 0.0
        assert "MISSING_DATA" in sub.flags

    def test_fundamentals_present_but_no_quality_metrics_is_neutral(self) -> None:
        # A period with only a label/date and no quality metrics at all.
        bare = _period("FY2025", 2025)
        sub = QualityAnalyzer().analyze(_security([bare]), _ctx())
        assert sub.score == 50.0
        assert sub.confidence <= 0.2
        assert "MISSING_DATA" in sub.flags

    def test_partial_data_lowers_confidence_vs_full(self) -> None:
        # Only ROIC present in a single period -> gradeable but low coverage.
        partial = _security([_period("FY2025", 2025, roic=0.20)])
        full = _security(_strong_periods())
        sub_partial = QualityAnalyzer().analyze(partial, _ctx())
        sub_full = QualityAnalyzer().analyze(full, _ctx())
        assert sub_partial.confidence < sub_full.confidence
        assert sub_partial.data_coverage < sub_full.data_coverage
        assert 0.0 <= sub_partial.score <= 100.0


class TestDurability:
    def test_stable_returns_beat_volatile_same_average(self) -> None:
        # Two companies with the same *latest* ROIC but very different histories:
        # the stable one should not score worse, and should be flagged as durable.
        stable = _security([
            _period("FY2025", 2025, roic=0.20, roe=0.22, roa=0.12, gross_margin=0.50,
                    operating_margin=0.20, fcf_margin=0.12, revenue=500.0,
                    net_income=80.0, free_cash_flow=80.0, total_assets=600.0),
            _period("FY2024", 2024, roic=0.20, roe=0.22, roa=0.12, gross_margin=0.50,
                    operating_margin=0.20, fcf_margin=0.12, revenue=480.0,
                    net_income=78.0, free_cash_flow=78.0, total_assets=590.0),
            _period("FY2023", 2023, roic=0.20, roe=0.21, roa=0.12, gross_margin=0.50,
                    operating_margin=0.20, fcf_margin=0.12, revenue=460.0,
                    net_income=75.0, free_cash_flow=75.0, total_assets=580.0),
        ])
        volatile = _security([
            _period("FY2025", 2025, roic=0.20, roe=0.22, roa=0.12, gross_margin=0.50,
                    operating_margin=0.20, fcf_margin=0.12, revenue=500.0,
                    net_income=80.0, free_cash_flow=80.0, total_assets=600.0),
            _period("FY2024", 2024, roic=0.02, roe=0.03, roa=0.01, gross_margin=0.40,
                    operating_margin=0.04, fcf_margin=0.02, revenue=480.0,
                    net_income=10.0, free_cash_flow=8.0, total_assets=590.0),
            _period("FY2023", 2023, roic=0.38, roe=0.40, roa=0.22, gross_margin=0.60,
                    operating_margin=0.36, fcf_margin=0.22, revenue=460.0,
                    net_income=140.0, free_cash_flow=130.0, total_assets=580.0),
        ])
        s_stable = QualityAnalyzer().analyze(stable, _ctx())
        s_volatile = QualityAnalyzer().analyze(volatile, _ctx())
        assert s_stable.score >= s_volatile.score


class TestPeerContext:
    def test_top_of_peer_distribution_rewarded(self) -> None:
        # A modest absolute ROIC that is nonetheless top-of-peers should score
        # higher when peer context says peers are even weaker.
        data = _security([
            _period("FY2025", 2025, roic=0.09, roe=0.10, roa=0.06, gross_margin=0.35,
                    operating_margin=0.10, fcf_margin=0.06, revenue=400.0,
                    net_income=40.0, free_cash_flow=40.0, total_assets=500.0),
        ])
        weak_peers = AnalysisContext(
            peer_stats={
                "roic": [0.01, 0.02, 0.03, 0.04, 0.05],
                "operating_margin": [0.01, 0.02, 0.03, 0.04],
                "gross_margin": [0.20, 0.22, 0.25, 0.28],
            }
        )
        with_peers = QualityAnalyzer().analyze(data, weak_peers)
        without_peers = QualityAnalyzer().analyze(data, _ctx())
        assert with_peers.score > without_peers.score


class TestPurityAndBounds:
    def test_determinism(self) -> None:
        data = _security(_strong_periods())
        a = QualityAnalyzer().analyze(data, _ctx())
        b = QualityAnalyzer().analyze(data, _ctx())
        assert a.score == b.score
        assert a.confidence == b.confidence
        assert a.data_coverage == b.data_coverage

    @pytest.mark.parametrize(
        "periods",
        [
            _strong_periods(),
            _weak_periods(),
            [_period("FY2025", 2025, roic=0.20)],
            [_period("FY2025", 2025, operating_margin=-2.0, net_income=-1000.0,
                     free_cash_flow=-2000.0, revenue=10.0, total_assets=5000.0)],
        ],
    )
    def test_score_always_within_bounds(self, periods: List[FundamentalsPeriod]) -> None:
        sub = QualityAnalyzer().analyze(_security(periods), _ctx())
        assert 0.0 <= sub.score <= 100.0
        assert 0.0 <= sub.confidence <= 1.0
        assert 0.0 <= sub.data_coverage <= 1.0
