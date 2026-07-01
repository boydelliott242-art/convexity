"""Unit tests for :class:`convexity.analysis.growth.GrowthAnalyzer`.

These tests pin the behavioural contract of the GROWTH analyzer using small
:class:`SecurityData` objects built by hand (newest-fundamentals-first, per the
model contract). They assert the core honesty guarantees of any Convexity
analyzer:

* A *strong* growth profile (accelerating, margin-accretive revenue/earnings/FCF
  growth) scores **high** and reads bullish.
* A *weak* profile (shrinking, decelerating, margin-dilutive) scores **low**.
* **All-missing** fundamentals fall back to a neutral (50), low-confidence,
  ``MISSING_DATA``-flagged sub-score — a data gap neither helps nor hurts.
* Every score lives in ``[0, 100]`` with a populated, auditable evidence list.
* The analyzer is pure/deterministic and uses peer context when supplied while
  degrading gracefully when it is absent.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pytest

from convexity.analysis.growth import GrowthAnalyzer
from convexity.core.contracts import AnalysisContext
from convexity.core.models import (
    FundamentalsPeriod,
    ScoreCategory,
    SecurityData,
    SubScore,
    ValuationSnapshot,
)

_AS_OF = dt.datetime(2026, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _period(
    year: int,
    *,
    revenue: Optional[float] = None,
    net_income: Optional[float] = None,
    eps_diluted: Optional[float] = None,
    free_cash_flow: Optional[float] = None,
    operating_income: Optional[float] = None,
    operating_margin: Optional[float] = None,
) -> FundamentalsPeriod:
    """Build one annual fundamentals period labelled by fiscal ``year``."""
    return FundamentalsPeriod(
        period_end=dt.date(year, 12, 31),
        period_label=f"FY{year}",
        revenue=revenue,
        net_income=net_income,
        eps_diluted=eps_diluted,
        free_cash_flow=free_cash_flow,
        operating_income=operating_income,
        operating_margin=operating_margin,
    )


def _security(periods: List[FundamentalsPeriod]) -> SecurityData:
    """Wrap fundamentals (passed newest-first) into a minimal SecurityData."""
    return SecurityData(
        ticker="TEST",
        name="Test Co",
        sector="Technology",
        industry="Software",
        currency="USD",
        as_of=_AS_OF,
        valuation=ValuationSnapshot(market_cap=250_000_000.0),
        fundamentals=periods,
    )


def _ctx(peer_growth: Optional[List[float]] = None) -> AnalysisContext:
    """An AnalysisContext, optionally carrying a peer revenue-growth distribution."""
    peer_stats = {"revenue_growth": peer_growth} if peer_growth is not None else None
    return AnalysisContext(peer_stats=peer_stats, universe_stats=None, config=None)


def _strong_security() -> SecurityData:
    """A clearly strong, accelerating, margin-accretive grower (newest-first).

    Revenue: 100 -> 130 -> 175 -> 240 (YoY ~37%, accelerating, ~34% CAGR).
    Earnings and FCF rising; operating margin expanding 12% -> 22%.
    """
    return _security(
        [
            _period(2025, revenue=240.0, net_income=34.0, eps_diluted=3.4,
                    free_cash_flow=30.0, operating_income=52.8, operating_margin=0.22),
            _period(2024, revenue=175.0, net_income=22.0, eps_diluted=2.2,
                    free_cash_flow=20.0, operating_income=31.5, operating_margin=0.18),
            _period(2023, revenue=130.0, net_income=13.0, eps_diluted=1.3,
                    free_cash_flow=12.0, operating_income=19.5, operating_margin=0.15),
            _period(2022, revenue=100.0, net_income=8.0, eps_diluted=0.8,
                    free_cash_flow=7.0, operating_income=12.0, operating_margin=0.12),
        ]
    )


def _weak_security() -> SecurityData:
    """A shrinking, decelerating, margin-dilutive profile (newest-first).

    Revenue: 200 -> 170 -> 150 -> 130 (declining and decelerating downward).
    Earnings/FCF falling; operating margin contracting 15% -> 4%.
    """
    return _security(
        [
            _period(2025, revenue=130.0, net_income=2.0, eps_diluted=0.2,
                    free_cash_flow=1.0, operating_income=5.2, operating_margin=0.04),
            _period(2024, revenue=150.0, net_income=6.0, eps_diluted=0.6,
                    free_cash_flow=5.0, operating_income=12.0, operating_margin=0.08),
            _period(2023, revenue=170.0, net_income=14.0, eps_diluted=1.4,
                    free_cash_flow=14.0, operating_income=20.4, operating_margin=0.12),
            _period(2022, revenue=200.0, net_income=24.0, eps_diluted=2.4,
                    free_cash_flow=24.0, operating_income=30.0, operating_margin=0.15),
        ]
    )


# ---------------------------------------------------------------------------
# Core contract: strong high, weak low, missing -> neutral
# ---------------------------------------------------------------------------


class TestGrowthScoring:
    def test_strong_profile_scores_high(self) -> None:
        sub = GrowthAnalyzer().analyze(_strong_security(), _ctx())
        assert isinstance(sub, SubScore)
        assert sub.category == ScoreCategory.GROWTH
        assert sub.score >= 70.0, f"strong grower should score high, got {sub.score}"
        assert sub.confidence > 0.5
        assert sub.data_coverage > 0.8

    def test_weak_profile_scores_low(self) -> None:
        sub = GrowthAnalyzer().analyze(_weak_security(), _ctx())
        assert sub.score <= 35.0, f"shrinking grower should score low, got {sub.score}"
        assert "MISSING_DATA" not in sub.flags

    def test_strong_scores_strictly_above_weak(self) -> None:
        strong = GrowthAnalyzer().analyze(_strong_security(), _ctx()).score
        weak = GrowthAnalyzer().analyze(_weak_security(), _ctx()).score
        assert strong > weak + 30.0

    def test_all_missing_is_neutral_low_confidence(self) -> None:
        # No fundamentals at all -> neutral fallback.
        empty = _security([])
        sub = GrowthAnalyzer().analyze(empty, _ctx())
        assert sub.score == pytest.approx(50.0)
        assert sub.confidence <= 0.2
        assert "MISSING_DATA" in sub.flags
        assert sub.data_coverage == pytest.approx(0.0)

    def test_single_period_is_neutral(self) -> None:
        # Only one period: no growth is computable -> neutral fallback.
        one = _security([_period(2025, revenue=100.0, net_income=10.0)])
        sub = GrowthAnalyzer().analyze(one, _ctx())
        assert sub.score == pytest.approx(50.0)
        assert "MISSING_DATA" in sub.flags

    def test_revenue_present_but_only_revenue(self) -> None:
        # Two periods of revenue only (earnings/FCF/margins missing): scores, but
        # with partial coverage and the PARTIAL_GROWTH_DATA flag.
        sec = _security(
            [
                _period(2025, revenue=150.0),
                _period(2024, revenue=100.0),
            ]
        )
        sub = GrowthAnalyzer().analyze(sec, _ctx())
        assert 0.0 <= sub.score <= 100.0
        assert "MISSING_DATA" not in sub.flags
        assert "PARTIAL_GROWTH_DATA" in sub.flags
        assert sub.data_coverage < 1.0
        # 50% YoY revenue growth is strong -> clearly above neutral.
        assert sub.score > 60.0


# ---------------------------------------------------------------------------
# Evidence, range, flags
# ---------------------------------------------------------------------------


class TestGrowthEvidence:
    def test_evidence_is_populated_and_cites_numbers(self) -> None:
        sub = GrowthAnalyzer().analyze(_strong_security(), _ctx())
        assert sub.evidence, "a scored growth profile must emit evidence"
        labels = [e.label for e in sub.evidence]
        assert any("Revenue YoY growth" == lbl for lbl in labels)
        # Every evidence item carries a non-empty rendered value and a source.
        for ev in sub.evidence:
            assert ev.value and isinstance(ev.value, str)
            assert ev.source
            assert ev.direction in {"bullish", "bearish", "neutral"}

    def test_strong_revenue_evidence_is_bullish(self) -> None:
        sub = GrowthAnalyzer().analyze(_strong_security(), _ctx())
        rev = next(e for e in sub.evidence if e.label == "Revenue YoY growth")
        assert rev.direction == "bullish"

    def test_weak_revenue_evidence_is_bearish(self) -> None:
        sub = GrowthAnalyzer().analyze(_weak_security(), _ctx())
        rev = next(e for e in sub.evidence if e.label == "Revenue YoY growth")
        assert rev.direction == "bearish"

    @pytest.mark.parametrize("builder", [_strong_security, _weak_security])
    def test_score_always_within_range(self, builder) -> None:
        sub = GrowthAnalyzer().analyze(builder(), _ctx())
        assert 0.0 <= sub.score <= 100.0
        assert 0.0 <= sub.confidence <= 1.0
        assert 0.0 <= sub.data_coverage <= 1.0

    def test_acceleration_flag_on_accelerating_grower(self) -> None:
        sub = GrowthAnalyzer().analyze(_strong_security(), _ctx())
        assert "ACCELERATING_GROWTH" in sub.flags

    def test_margin_accretive_flag_on_strong_grower(self) -> None:
        sub = GrowthAnalyzer().analyze(_strong_security(), _ctx())
        assert "MARGIN_ACCRETIVE_GROWTH" in sub.flags

    def test_margin_dilutive_flag_when_revenue_up_margins_down(self) -> None:
        # Revenue rising while operating margin collapses -> margin-dilutive flag.
        sec = _security(
            [
                _period(2025, revenue=200.0, operating_margin=0.05),
                _period(2024, revenue=150.0, operating_margin=0.20),
                _period(2023, revenue=120.0, operating_margin=0.22),
            ]
        )
        sub = GrowthAnalyzer().analyze(sec, _ctx())
        assert "MARGIN_DILUTIVE_GROWTH" in sub.flags


# ---------------------------------------------------------------------------
# Acceleration vs deceleration drives the score
# ---------------------------------------------------------------------------


class TestAccelerationEffect:
    def test_accelerating_beats_decelerating_at_same_latest_rate(self) -> None:
        # Both reach the same latest revenue level/growth, but one accelerated
        # into it and the other decelerated. The accelerating one should score
        # at least as high.
        accelerating = _security(
            [
                _period(2025, revenue=150.0, operating_margin=0.15),  # +50% YoY
                _period(2024, revenue=100.0, operating_margin=0.15),  # +11% YoY
                _period(2023, revenue=90.0, operating_margin=0.15),
            ]
        )
        decelerating = _security(
            [
                _period(2025, revenue=150.0, operating_margin=0.15),  # +7% YoY
                _period(2024, revenue=140.0, operating_margin=0.15),  # +75% YoY
                _period(2023, revenue=80.0, operating_margin=0.15),
            ]
        )
        a = GrowthAnalyzer().analyze(accelerating, _ctx()).score
        d = GrowthAnalyzer().analyze(decelerating, _ctx()).score
        assert a >= d

    def test_decelerating_flag_set(self) -> None:
        decelerating = _security(
            [
                _period(2025, revenue=150.0),  # +7% YoY
                _period(2024, revenue=140.0),  # +75% YoY
                _period(2023, revenue=80.0),
            ]
        )
        sub = GrowthAnalyzer().analyze(decelerating, _ctx())
        assert "DECELERATING_GROWTH" in sub.flags


# ---------------------------------------------------------------------------
# Relative context & purity
# ---------------------------------------------------------------------------


class TestContextAndPurity:
    def test_peer_context_changes_score_for_relative_grower(self) -> None:
        # A modest 20% grower scored against a low-growth peer set (ranks high)
        # should not score lower than against a high-growth peer set (ranks low).
        sec = _security(
            [
                _period(2025, revenue=120.0, operating_margin=0.15),
                _period(2024, revenue=100.0, operating_margin=0.15),
                _period(2023, revenue=95.0, operating_margin=0.15),
            ]
        )
        high_rank = GrowthAnalyzer().analyze(
            sec, _ctx(peer_growth=[-0.10, -0.05, 0.0, 0.02, 0.05])
        ).score
        low_rank = GrowthAnalyzer().analyze(
            sec, _ctx(peer_growth=[0.30, 0.40, 0.55, 0.60, 0.80])
        ).score
        assert high_rank > low_rank

    def test_relative_evidence_added_when_peers_present(self) -> None:
        sec = _strong_security()
        sub = GrowthAnalyzer().analyze(sec, _ctx(peer_growth=[0.05, 0.10, 0.15, 0.20]))
        labels = [e.label for e in sub.evidence]
        assert any("percentile" in lbl.lower() for lbl in labels)

    def test_degrades_gracefully_without_context(self) -> None:
        # No peer/universe stats at all: still produces a valid, in-range score.
        sub = GrowthAnalyzer().analyze(_strong_security(), AnalysisContext())
        assert 0.0 <= sub.score <= 100.0
        assert sub.evidence

    def test_deterministic(self) -> None:
        sec = _strong_security()
        ctx = _ctx(peer_growth=[0.0, 0.1, 0.2, 0.3])
        a = GrowthAnalyzer().analyze(sec, ctx)
        b = GrowthAnalyzer().analyze(sec, ctx)
        assert a.score == b.score
        assert a.confidence == b.confidence
        assert a.data_coverage == b.data_coverage
        assert a.flags == b.flags

    def test_does_not_mutate_input(self) -> None:
        sec = _strong_security()
        n_periods_before = len(sec.fundamentals)
        rev_before = sec.fundamentals[0].revenue
        GrowthAnalyzer().analyze(sec, _ctx())
        assert len(sec.fundamentals) == n_periods_before
        assert sec.fundamentals[0].revenue == rev_before


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_analyzer_is_registered_for_growth() -> None:
    from convexity.core.registry import get_analyzer

    # Importing convexity.analysis.growth self-registers the analyzer.
    cls = get_analyzer(ScoreCategory.GROWTH)
    assert cls is GrowthAnalyzer


def test_class_attrs() -> None:
    assert GrowthAnalyzer.category == ScoreCategory.GROWTH
    assert GrowthAnalyzer.default_weight == pytest.approx(0.15)
    assert "fundamentals" in GrowthAnalyzer.requires
