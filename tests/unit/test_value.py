"""Unit tests for :class:`convexity.analysis.value.ValueAnalyzer`.

These tests pin the behavioural contract of the VALUE analyzer using small,
hand-crafted :class:`SecurityData` objects (fundamentals newest-first per the
model contract) and inline :class:`AnalysisContext` objects — no conftest
fixtures. They assert the honesty guarantees shared by every Convexity analyzer:

* A *cheap* security backed by improving economics scores meaningfully **higher**
  than an *expensive* one.
* **All-missing** valuation multiples fall back to a neutral (50),
  low-confidence, ``MISSING_DATA``-flagged sub-score — a data gap neither helps
  nor hurts.
* Every score lives in ``[0, 100]`` with a populated, auditable evidence list.
* The emitted :class:`SubScore` carries the VALUE category.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pytest

from convexity.analysis.value import ValueAnalyzer
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
    free_cash_flow: Optional[float] = None,
    operating_margin: Optional[float] = None,
) -> FundamentalsPeriod:
    """Build one annual fundamentals period labelled by fiscal ``year``."""
    return FundamentalsPeriod(
        period_end=dt.date(year, 12, 31),
        period_label=f"FY{year}",
        revenue=revenue,
        net_income=net_income,
        free_cash_flow=free_cash_flow,
        operating_margin=operating_margin,
    )


def _security(
    valuation: ValuationSnapshot,
    periods: Optional[List[FundamentalsPeriod]] = None,
) -> SecurityData:
    """Wrap a valuation snapshot (and optional fundamentals) into SecurityData."""
    return SecurityData(
        ticker="TEST",
        name="Test Co",
        sector="Technology",
        industry="Software",
        currency="USD",
        as_of=_AS_OF,
        valuation=valuation,
        fundamentals=periods or [],
    )


def _ctx() -> AnalysisContext:
    """A bare AnalysisContext (no peer/universe stats)."""
    return AnalysisContext(peer_stats=None, universe_stats=None, config=None)


def _strong_security() -> SecurityData:
    """A clearly cheap name whose cheapness is *earned* by improving economics.

    Low multiples across the board, positive and rising earnings/FCF, growing
    revenue and expanding margins (so no value-trap dampener applies).
    """
    valuation = ValuationSnapshot(
        market_cap=250_000_000.0,
        ev_ebitda=5.0,
        p_fcf=7.0,
        pe=7.0,
        ev_sales=0.8,
        p_b=0.9,
        peg=0.6,
    )
    periods = [
        _period(2025, revenue=160.0, net_income=30.0, free_cash_flow=28.0, operating_margin=0.22),
        _period(2024, revenue=130.0, net_income=20.0, free_cash_flow=18.0, operating_margin=0.18),
    ]
    return _security(valuation, periods)


def _weak_security() -> SecurityData:
    """A clearly expensive name (rich multiples) with healthy-but-pricey economics."""
    valuation = ValuationSnapshot(
        market_cap=250_000_000.0,
        ev_ebitda=22.0,
        p_fcf=40.0,
        pe=35.0,
        ev_sales=7.0,
        p_b=6.0,
        peg=3.0,
    )
    periods = [
        _period(2025, revenue=160.0, net_income=10.0, free_cash_flow=8.0, operating_margin=0.20),
        _period(2024, revenue=150.0, net_income=9.0, free_cash_flow=7.0, operating_margin=0.20),
    ]
    return _security(valuation, periods)


# ---------------------------------------------------------------------------
# Core contract: cheap > expensive, missing -> neutral
# ---------------------------------------------------------------------------


class TestValueScoring:
    def test_cheap_scores_high(self) -> None:
        sub = ValueAnalyzer().analyze(_strong_security(), _ctx())
        assert isinstance(sub, SubScore)
        assert sub.category == ScoreCategory.VALUE
        assert sub.score >= 65.0, f"a clearly cheap name should score high, got {sub.score}"

    def test_expensive_scores_low(self) -> None:
        sub = ValueAnalyzer().analyze(_weak_security(), _ctx())
        assert sub.score <= 35.0, f"a clearly expensive name should score low, got {sub.score}"

    def test_cheap_strictly_above_expensive(self) -> None:
        cheap = ValueAnalyzer().analyze(_strong_security(), _ctx()).score
        rich = ValueAnalyzer().analyze(_weak_security(), _ctx()).score
        assert cheap > rich + 30.0

    def test_all_missing_is_neutral_low_confidence(self) -> None:
        # No usable valuation multiples at all -> neutral fallback.
        empty = _security(ValuationSnapshot(market_cap=250_000_000.0))
        sub = ValueAnalyzer().analyze(empty, _ctx())
        assert sub.score == pytest.approx(50.0)
        assert sub.confidence <= 0.2
        assert "MISSING_DATA" in sub.flags
        assert sub.data_coverage == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Evidence, range
# ---------------------------------------------------------------------------


class TestValueEvidence:
    def test_evidence_is_populated(self) -> None:
        sub = ValueAnalyzer().analyze(_strong_security(), _ctx())
        assert sub.evidence, "a scored value profile must emit evidence"
        for ev in sub.evidence:
            assert ev.value and isinstance(ev.value, str)
            assert ev.source
            assert ev.direction in {"bullish", "bearish", "neutral"}

    @pytest.mark.parametrize("builder", [_strong_security, _weak_security])
    def test_score_always_within_range(self, builder) -> None:
        sub = ValueAnalyzer().analyze(builder(), _ctx())
        assert 0.0 <= sub.score <= 100.0
        assert 0.0 <= sub.confidence <= 1.0
        assert 0.0 <= sub.data_coverage <= 1.0

    def test_value_trap_dampens_a_cheap_but_deteriorating_name(self) -> None:
        # Cheap multiples but losses, negative FCF and a shrinking top line: the
        # value-trap dampener should pull the score down vs the same multiples on
        # healthy economics, and flag the trap.
        cheap_multiples = ValuationSnapshot(
            market_cap=250_000_000.0,
            ev_ebitda=5.0,
            p_fcf=7.0,
            pe=7.0,
            ev_sales=0.8,
        )
        trap = _security(
            cheap_multiples,
            [
                _period(2025, revenue=100.0, net_income=-12.0, free_cash_flow=-8.0, operating_margin=0.02),
                _period(2024, revenue=140.0, net_income=5.0, free_cash_flow=4.0, operating_margin=0.15),
            ],
        )
        healthy = _security(
            cheap_multiples,
            [
                _period(2025, revenue=160.0, net_income=30.0, free_cash_flow=28.0, operating_margin=0.22),
                _period(2024, revenue=130.0, net_income=20.0, free_cash_flow=18.0, operating_margin=0.18),
            ],
        )
        trap_sub = ValueAnalyzer().analyze(trap, _ctx())
        healthy_sub = ValueAnalyzer().analyze(healthy, _ctx())
        assert trap_sub.score < healthy_sub.score
        assert "VALUE_TRAP_RISK" in trap_sub.flags


# ---------------------------------------------------------------------------
# Relative context & purity
# ---------------------------------------------------------------------------


class TestValueContextAndPurity:
    def test_peer_context_changes_score(self) -> None:
        # The same EV/EBITDA scored against cheap-peer vs expensive-peer sets.
        val = ValuationSnapshot(market_cap=250_000_000.0, ev_ebitda=10.0)
        sec = _security(
            val,
            [_period(2025, revenue=100.0, net_income=10.0, free_cash_flow=9.0)],
        )
        looks_cheap = ValueAnalyzer().analyze(
            sec, AnalysisContext(peer_stats={"ev_ebitda": [12.0, 15.0, 18.0, 20.0, 25.0]})
        ).score
        looks_rich = ValueAnalyzer().analyze(
            sec, AnalysisContext(peer_stats={"ev_ebitda": [3.0, 4.0, 5.0, 6.0, 7.0]})
        ).score
        assert looks_cheap > looks_rich

    def test_deterministic(self) -> None:
        sec = _strong_security()
        a = ValueAnalyzer().analyze(sec, _ctx())
        b = ValueAnalyzer().analyze(sec, _ctx())
        assert a.score == b.score
        assert a.confidence == b.confidence
        assert a.flags == b.flags

    def test_does_not_mutate_input(self) -> None:
        sec = _strong_security()
        ev_ebitda_before = sec.valuation.ev_ebitda
        n_before = len(sec.fundamentals)
        ValueAnalyzer().analyze(sec, _ctx())
        assert sec.valuation.ev_ebitda == ev_ebitda_before
        assert len(sec.fundamentals) == n_before


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_analyzer_is_registered_for_value() -> None:
    from convexity.core.registry import get_analyzer

    cls = get_analyzer(ScoreCategory.VALUE)
    assert cls is ValueAnalyzer


def test_class_attrs() -> None:
    assert ValueAnalyzer.category == ScoreCategory.VALUE
    assert "valuation" in ValueAnalyzer.requires
