"""Unit tests for :class:`convexity.ranking.engine.DefaultRankingEngine`.

These pin the ranking engine's two contract methods (``score_company`` and
``rank``) and the honesty principle they encode — *a high composite is not the same
as conviction*:

* **Ordering.** :meth:`rank` sorts best-first by composite score, with deterministic
  tie-breaks (conviction, then agreement, then ticker) and contiguous 1-based ranks.
* **Conviction rises with breadth of independent agreement** — many confident,
  bullish categories convict more than one lone extreme category.
* **Conviction rises with data coverage / depth** — the same agreeing signals
  convict more when more real data underpins them.
* **Conviction falls with contradiction** — widely-disagreeing categories cannot
  manufacture conviction however high one of them scores.

Sub-scores are built by hand (no provider) and fed through the real engine, so the
tests exercise the genuine composite + conviction maths via
:func:`convexity.core.scoring.combine_subscores`.
"""

from __future__ import annotations

import datetime as _dt
from typing import List, Optional

import pytest

from convexity.core.models import (
    CompanyAnalysis,
    ScanParams,
    ScoreCategory,
    SecurityData,
    SubScore,
    ValuationSnapshot,
)
from convexity.ranking.engine import DefaultRankingEngine

_ADDITIVE = [
    ScoreCategory.VALUE,
    ScoreCategory.GROWTH,
    ScoreCategory.QUALITY,
    ScoreCategory.FINANCIAL_HEALTH,
    ScoreCategory.CATALYST,
    ScoreCategory.COMPETITIVE,
]


def _sub(
    category: ScoreCategory,
    score: float,
    *,
    confidence: float = 0.8,
    weight: float = 0.1,
    coverage: float = 0.9,
) -> SubScore:
    """Build a minimal :class:`SubScore` for ``category``."""
    return SubScore(
        category=category,
        score=score,
        confidence=confidence,
        weight=weight,
        rationale="synthetic",
        evidence=[],
        flags=[],
        data_coverage=coverage,
    )


def _security(ticker: str) -> SecurityData:
    """A minimal :class:`SecurityData` carrying only identity for scoring."""
    return SecurityData(
        ticker=ticker,
        name=f"{ticker} Inc",
        sector="Technology",
        as_of=_dt.datetime(2026, 1, 1),
        valuation=ValuationSnapshot(market_cap=250_000_000.0),
    )


def _score(
    ticker: str,
    subs: List[SubScore],
    engine: Optional[DefaultRankingEngine] = None,
) -> CompanyAnalysis:
    """Score one company's sub-scores through the engine (default weights)."""
    eng = engine or DefaultRankingEngine()
    return eng.score_company(_security(ticker), subs, {})


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestOrdering:
    def test_rank_orders_by_composite_desc_with_contiguous_ranks(self) -> None:
        engine = DefaultRankingEngine()
        strong = _score("AAA", [_sub(cat, 85.0) for cat in _ADDITIVE], engine)
        middle = _score("BBB", [_sub(cat, 55.0) for cat in _ADDITIVE], engine)
        weak = _score("CCC", [_sub(cat, 25.0) for cat in _ADDITIVE], engine)

        # Feed in deliberately-scrambled order; rank must reorder best-first.
        ranked = engine.rank([weak, strong, middle], ScanParams())
        assert [c.ticker for c in ranked] == ["AAA", "BBB", "CCC"]
        assert [c.rank for c in ranked] == [1, 2, 3]
        scores = [c.composite_score for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ticker_tiebreak_is_deterministic(self) -> None:
        """Identical scores break ties by ticker ascending (stable, reproducible)."""
        engine = DefaultRankingEngine()
        subs = [_sub(cat, 60.0) for cat in _ADDITIVE]
        a = _score("ZZZ", subs, engine)
        b = _score("AAA", subs, engine)
        ranked = engine.rank([a, b], ScanParams())
        # Same composite + conviction + agreement -> alphabetical ticker wins.
        assert [c.ticker for c in ranked] == ["AAA", "ZZZ"]

    def test_rank_does_not_drop_or_duplicate(self) -> None:
        engine = DefaultRankingEngine()
        companies = [
            _score(t, [_sub(cat, 50.0 + i) for cat in _ADDITIVE], engine)
            for i, t in enumerate(["AAA", "BBB", "CCC", "DDD"])
        ]
        ranked = engine.rank(companies, ScanParams())
        assert sorted(c.ticker for c in ranked) == ["AAA", "BBB", "CCC", "DDD"]
        assert len(ranked) == 4

    def test_rank_is_deterministic(self) -> None:
        engine = DefaultRankingEngine()
        companies = [
            _score(t, [_sub(cat, 40.0 + 7 * i) for cat in _ADDITIVE], engine)
            for i, t in enumerate(["AAA", "BBB", "CCC"])
        ]
        first = [c.ticker for c in engine.rank(list(companies), ScanParams())]
        second = [c.ticker for c in engine.rank(list(companies), ScanParams())]
        assert first == second


# ---------------------------------------------------------------------------
# Conviction rises with breadth of independent agreement
# ---------------------------------------------------------------------------


class TestConvictionBreadth:
    def test_many_confirming_beats_one_lone_extreme(self) -> None:
        """Six confident bullish categories convict more than one lone extreme.

        The lone-extreme company has a single category pinned at 100 (and the rest
        neutral); the broad company has all six categories solidly bullish. Even if
        their composites are close, conviction must reward the breadth.
        """
        broad = _score("BROAD", [_sub(cat, 78.0) for cat in _ADDITIVE])

        lone = _score(
            "LONE",
            [_sub(_ADDITIVE[0], 100.0)]
            + [_sub(cat, 50.0, confidence=0.8) for cat in _ADDITIVE[1:]],
        )

        assert broad.conviction_confidence > lone.conviction_confidence

    def test_single_extreme_does_not_create_high_conviction(self) -> None:
        """One extreme category alone cannot push conviction high."""
        lone = _score(
            "LONE",
            [_sub(_ADDITIVE[0], 100.0)]
            + [_sub(cat, 50.0) for cat in _ADDITIVE[1:]],
        )
        # A high composite component, but conviction stays modest because only one
        # independent category confirms.
        assert lone.conviction_confidence < 0.5

    def test_conviction_increases_with_number_of_confirmations(self) -> None:
        """Adding more confident bullish categories monotonically lifts conviction."""

        def conviction_with(n_bull: int) -> float:
            subs = []
            for i, cat in enumerate(_ADDITIVE):
                subs.append(_sub(cat, 75.0 if i < n_bull else 50.0))
            return _score("X", subs).conviction_confidence

        c2 = conviction_with(2)
        c4 = conviction_with(4)
        c6 = conviction_with(6)
        assert c2 < c4 < c6


# ---------------------------------------------------------------------------
# Conviction rises with coverage; falls with contradiction
# ---------------------------------------------------------------------------


class TestConvictionCoverageAndDispersion:
    def test_conviction_rises_with_coverage(self) -> None:
        """Same agreeing signals, more data coverage -> more conviction."""
        thin = _score(
            "THIN",
            [_sub(cat, 78.0, confidence=0.4, coverage=0.2) for cat in _ADDITIVE],
        )
        rich = _score(
            "RICH",
            [_sub(cat, 78.0, confidence=0.9, coverage=0.95) for cat in _ADDITIVE],
        )
        assert rich.conviction_confidence > thin.conviction_confidence

    def test_contradiction_lowers_conviction(self) -> None:
        """Widely-disagreeing categories cannot manufacture conviction."""
        agreeing = _score("AGREE", [_sub(cat, 78.0) for cat in _ADDITIVE])
        split = _score(
            "SPLIT",
            [
                _sub(_ADDITIVE[i], 85.0 if i % 2 == 0 else 18.0)
                for i in range(len(_ADDITIVE))
            ],
        )
        assert split.conviction_confidence < agreeing.conviction_confidence

    def test_all_missing_has_near_zero_conviction(self) -> None:
        """A company whose every category is a zero-coverage gap has ~no conviction."""
        gaps = _score(
            "GAP",
            [
                _sub(cat, 50.0, confidence=0.1, coverage=0.0)
                for cat in _ADDITIVE
            ],
        )
        # No real evidence -> conviction left at (or below) the small floor.
        assert gaps.conviction_confidence <= 0.05

    def test_conviction_in_unit_interval(self) -> None:
        for score in (10.0, 50.0, 90.0):
            company = _score("X", [_sub(cat, score) for cat in _ADDITIVE])
            assert 0.0 <= company.conviction_confidence <= 1.0
            assert 0.0 <= company.composite_score <= 100.0
            assert 0.0 <= company.signal_agreement <= 1.0


# ---------------------------------------------------------------------------
# score_company plumbing
# ---------------------------------------------------------------------------


class TestScoreCompanyPlumbing:
    def test_identity_and_metadata_carried_through(self) -> None:
        company = _score("AAA", [_sub(cat, 60.0) for cat in _ADDITIVE])
        assert company.ticker == "AAA"
        assert company.name == "AAA Inc"
        assert company.sector == "Technology"
        assert company.market_cap == pytest.approx(250_000_000.0)
        # Narrative is left empty for the explainability engine.
        assert company.thesis == ""
        assert company.bull_case == []
        # rank is unset until rank() runs.
        assert company.rank is None

    def test_risk_dampener_lowers_a_companys_composite(self) -> None:
        """A risky company composes below an otherwise-identical safe one."""
        base = [_sub(cat, 70.0) for cat in _ADDITIVE]
        safe = _score("SAFE", base + [_sub(ScoreCategory.RISK, 90.0, weight=0.0)])
        risky = _score("RISK", base + [_sub(ScoreCategory.RISK, 5.0, weight=0.0)])
        assert risky.composite_score < safe.composite_score
