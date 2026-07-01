"""Unit tests for :class:`convexity.analysis.competitive.CompetitiveAnalyzer`.

The COMPETITIVE analyzer proxies a durable moat with four financially-grounded
sub-signals: margin level vs peers, gross-margin durability (level x stability),
revenue growth vs peers/universe, and the level-and-persistence of returns on
capital. These tests build small, hand-crafted :class:`SecurityData` objects
inline (no conftest fixtures) and assert the shared honesty guarantees:

* A wide-and-stable-margin, high-and-persistent-ROIC, share-taking business
  scores meaningfully **higher** than a thin-margin, erratic, money-losing one.
* **No fundamentals at all** falls back to a neutral (50), low-confidence,
  ``MISSING_DATA``-flagged sub-score.
* Every score lives in ``[0, 100]`` with a populated, auditable evidence list.
* The emitted :class:`SubScore` carries the COMPETITIVE category.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pytest

from convexity.analysis.competitive import CompetitiveAnalyzer
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
    gross_margin: Optional[float] = None,
    operating_margin: Optional[float] = None,
    roic: Optional[float] = None,
    roe: Optional[float] = None,
) -> FundamentalsPeriod:
    """Build one annual fundamentals period labelled by fiscal ``year``."""
    return FundamentalsPeriod(
        period_end=dt.date(year, 12, 31),
        period_label=f"FY{year}",
        revenue=revenue,
        gross_margin=gross_margin,
        operating_margin=operating_margin,
        roic=roic,
        roe=roe,
    )


def _security(periods: List[FundamentalsPeriod]) -> SecurityData:
    """Wrap fundamentals (newest-first) into a minimal SecurityData."""
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


def _ctx() -> AnalysisContext:
    """A bare AnalysisContext (no peer/universe stats -> absolute bands)."""
    return AnalysisContext(peer_stats=None, universe_stats=None, config=None)


def _strong_security() -> SecurityData:
    """Wide & stable gross margins, high & persistent ROIC, growing revenue."""
    return _security(
        [
            _period(2025, revenue=150.0, gross_margin=0.62, operating_margin=0.28, roic=0.22, roe=0.24),
            _period(2024, revenue=125.0, gross_margin=0.61, operating_margin=0.27, roic=0.21, roe=0.23),
            _period(2023, revenue=105.0, gross_margin=0.62, operating_margin=0.27, roic=0.22, roe=0.24),
            _period(2022, revenue=90.0, gross_margin=0.61, operating_margin=0.26, roic=0.21, roe=0.23),
        ]
    )


def _weak_security() -> SecurityData:
    """Thin, erratic gross margins, negative ROIC, shrinking revenue."""
    return _security(
        [
            _period(2025, revenue=80.0, gross_margin=0.10, operating_margin=-0.05, roic=-0.08, roe=-0.10),
            _period(2024, revenue=95.0, gross_margin=0.28, operating_margin=0.02, roic=0.01, roe=0.00),
            _period(2023, revenue=110.0, gross_margin=0.09, operating_margin=-0.04, roic=-0.05, roe=-0.07),
            _period(2022, revenue=120.0, gross_margin=0.30, operating_margin=0.03, roic=0.02, roe=0.01),
        ]
    )


# ---------------------------------------------------------------------------
# Core contract: strong high, weak low, missing -> neutral
# ---------------------------------------------------------------------------


class TestCompetitiveScoring:
    def test_strong_profile_scores_high(self) -> None:
        sub = CompetitiveAnalyzer().analyze(_strong_security(), _ctx())
        assert isinstance(sub, SubScore)
        assert sub.category == ScoreCategory.COMPETITIVE
        assert sub.score >= 60.0, f"a wide-moat profile should score high, got {sub.score}"

    def test_weak_profile_scores_low(self) -> None:
        sub = CompetitiveAnalyzer().analyze(_weak_security(), _ctx())
        assert sub.score <= 40.0, f"a contestable profile should score low, got {sub.score}"
        assert "MISSING_DATA" not in sub.flags

    def test_strong_strictly_above_weak(self) -> None:
        strong = CompetitiveAnalyzer().analyze(_strong_security(), _ctx()).score
        weak = CompetitiveAnalyzer().analyze(_weak_security(), _ctx()).score
        assert strong > weak + 20.0

    def test_no_fundamentals_is_neutral_low_confidence(self) -> None:
        sub = CompetitiveAnalyzer().analyze(_security([]), _ctx())
        assert sub.score == pytest.approx(50.0)
        assert sub.confidence <= 0.2
        assert "MISSING_DATA" in sub.flags
        assert sub.data_coverage == pytest.approx(0.0)

    def test_present_but_unusable_fields_is_neutral(self) -> None:
        # A period exists but carries none of the margin/growth/returns fields.
        sec = _security([_period(2025)])
        sub = CompetitiveAnalyzer().analyze(sec, _ctx())
        assert sub.score == pytest.approx(50.0)
        assert "MISSING_DATA" in sub.flags
        assert "NO_USABLE_FIELDS" in sub.flags


# ---------------------------------------------------------------------------
# Evidence, range, flags
# ---------------------------------------------------------------------------


class TestCompetitiveEvidence:
    def test_evidence_is_populated(self) -> None:
        sub = CompetitiveAnalyzer().analyze(_strong_security(), _ctx())
        assert sub.evidence, "a scored competitive profile must emit evidence"
        for ev in sub.evidence:
            assert ev.value and isinstance(ev.value, str)
            assert ev.source
            assert ev.direction in {"bullish", "bearish", "neutral"}

    @pytest.mark.parametrize("builder", [_strong_security, _weak_security])
    def test_score_always_within_range(self, builder) -> None:
        sub = CompetitiveAnalyzer().analyze(builder(), _ctx())
        assert 0.0 <= sub.score <= 100.0
        assert 0.0 <= sub.confidence <= 1.0
        assert 0.0 <= sub.data_coverage <= 1.0

    def test_no_peer_context_flag_when_no_peers(self) -> None:
        sub = CompetitiveAnalyzer().analyze(_strong_security(), _ctx())
        assert "NO_PEER_CONTEXT" in sub.flags


# ---------------------------------------------------------------------------
# Relative context & purity
# ---------------------------------------------------------------------------


class TestCompetitiveContextAndPurity:
    def test_peer_context_changes_margin_read(self) -> None:
        # The same 40% gross margin scored against low-margin vs high-margin peers.
        sec = _security(
            [
                _period(2025, revenue=120.0, gross_margin=0.40, operating_margin=0.15, roic=0.10),
                _period(2024, revenue=100.0, gross_margin=0.40, operating_margin=0.15, roic=0.10),
            ]
        )
        looks_strong = CompetitiveAnalyzer().analyze(
            sec, AnalysisContext(peer_stats={"gross_margin": [0.10, 0.15, 0.20, 0.25]})
        ).score
        looks_weak = CompetitiveAnalyzer().analyze(
            sec, AnalysisContext(peer_stats={"gross_margin": [0.55, 0.60, 0.65, 0.70]})
        ).score
        assert looks_strong > looks_weak

    def test_deterministic(self) -> None:
        sec = _strong_security()
        a = CompetitiveAnalyzer().analyze(sec, _ctx())
        b = CompetitiveAnalyzer().analyze(sec, _ctx())
        assert a.score == b.score
        assert a.confidence == b.confidence
        assert a.flags == b.flags

    def test_does_not_mutate_input(self) -> None:
        sec = _strong_security()
        n_before = len(sec.fundamentals)
        gm_before = sec.fundamentals[0].gross_margin
        CompetitiveAnalyzer().analyze(sec, _ctx())
        assert len(sec.fundamentals) == n_before
        assert sec.fundamentals[0].gross_margin == gm_before


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_analyzer_is_registered_for_competitive() -> None:
    from convexity.core.registry import get_analyzer

    cls = get_analyzer(ScoreCategory.COMPETITIVE)
    assert cls is CompetitiveAnalyzer


def test_class_attrs() -> None:
    assert CompetitiveAnalyzer.category == ScoreCategory.COMPETITIVE
    assert "fundamentals" in CompetitiveAnalyzer.requires
