"""Unit tests for :class:`convexity.analysis.technical.TechnicalAnalyzer`.

The TECHNICAL analyzer reads a security's *historical* OHLCV price bars
(oldest-first per the model contract) and emits a single auditable
:class:`SubScore`. These tests build small, hand-crafted price series inline (no
conftest fixtures) and assert the shared honesty guarantees:

* A constructive uptrend (price above a rising average, near its highs, calm
  tape, rising volume) scores meaningfully **higher** than a broken-down
  downtrend.
* **Too little / no price history** falls back to a neutral (50), low-confidence,
  ``MISSING_DATA``-flagged sub-score.
* Every score lives in ``[0, 100]`` with a populated, auditable evidence list.
* The emitted :class:`SubScore` carries the TECHNICAL category.
"""

from __future__ import annotations

import datetime as dt
from typing import List

import pytest

from convexity.analysis.technical import TechnicalAnalyzer
from convexity.core.contracts import AnalysisContext
from convexity.core.models import (
    PriceBar,
    ScoreCategory,
    SecurityData,
    SubScore,
    ValuationSnapshot,
)

_AS_OF = dt.datetime(2026, 1, 1, 0, 0, 0)
_START = dt.date(2024, 1, 1)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _bar(day_index: int, close: float, *, volume: float = 1_000.0) -> PriceBar:
    """Build one daily bar ``day_index`` days after the start date.

    The intraday range is a tight +/-0.5% band around the close so the tape reads
    as orderly (low ATR), keeping the volatility facet constructive.
    """
    return PriceBar(
        date=_START + dt.timedelta(days=day_index),
        open=close,
        high=close * 1.005,
        low=close * 0.995,
        close=close,
        adj_close=close,
        volume=volume,
    )


def _security(bars: List[PriceBar]) -> SecurityData:
    """Wrap price bars (oldest-first) into a minimal SecurityData."""
    return SecurityData(
        ticker="TEST",
        name="Test Co",
        currency="USD",
        as_of=_AS_OF,
        valuation=ValuationSnapshot(market_cap=250_000_000.0),
        price_history=bars,
    )


def _ctx() -> AnalysisContext:
    """A bare AnalysisContext (no peer/universe stats)."""
    return AnalysisContext(peer_stats=None, universe_stats=None, config=None)


def _strong_security() -> SecurityData:
    """A long, steady uptrend with rising participation into the move.

    260 bars climbing ~0.4%/bar so price sits well above a rising SMA50/SMA200,
    near its 52-week high and far above its low, with recent volume above
    baseline.
    """
    bars: List[PriceBar] = []
    price = 10.0
    for i in range(260):
        price *= 1.004  # steady, orderly advance.
        # Volume rises modestly over the last stretch to confirm the move.
        vol = 1_000.0 + (i * 3.0)
        bars.append(_bar(i, price, volume=vol))
    return _security(bars)


def _weak_security() -> SecurityData:
    """A long, steady downtrend with fading participation.

    260 bars falling ~0.4%/bar so price sits below a falling SMA50/SMA200, near
    its 52-week low and far below its high, with recent volume below baseline.
    """
    bars: List[PriceBar] = []
    price = 60.0
    for i in range(260):
        price *= 0.996  # steady decline.
        vol = 4_000.0 - (i * 10.0)
        bars.append(_bar(i, price, volume=max(vol, 100.0)))
    return _security(bars)


# ---------------------------------------------------------------------------
# Core contract: uptrend high, downtrend low, missing -> neutral
# ---------------------------------------------------------------------------


class TestTechnicalScoring:
    def test_uptrend_scores_high(self) -> None:
        sub = TechnicalAnalyzer().analyze(_strong_security(), _ctx())
        assert isinstance(sub, SubScore)
        assert sub.category == ScoreCategory.TECHNICAL
        assert sub.score >= 65.0, f"a clean uptrend should score high, got {sub.score}"
        assert sub.confidence > 0.5
        assert sub.data_coverage > 0.8

    def test_downtrend_scores_low(self) -> None:
        sub = TechnicalAnalyzer().analyze(_weak_security(), _ctx())
        assert sub.score <= 35.0, f"a clean downtrend should score low, got {sub.score}"
        assert "MISSING_DATA" not in sub.flags

    def test_uptrend_strictly_above_downtrend(self) -> None:
        up = TechnicalAnalyzer().analyze(_strong_security(), _ctx()).score
        down = TechnicalAnalyzer().analyze(_weak_security(), _ctx()).score
        assert up > down + 30.0

    def test_no_history_is_neutral_low_confidence(self) -> None:
        sub = TechnicalAnalyzer().analyze(_security([]), _ctx())
        assert sub.score == pytest.approx(50.0)
        assert sub.confidence <= 0.2
        assert "MISSING_DATA" in sub.flags
        assert sub.data_coverage == pytest.approx(0.0)

    def test_short_history_is_neutral(self) -> None:
        # Fewer than the minimum bars -> neutral fallback with a SHORT_HISTORY flag.
        short = _security([_bar(i, 10.0 + i) for i in range(10)])
        sub = TechnicalAnalyzer().analyze(short, _ctx())
        assert sub.score == pytest.approx(50.0)
        assert "MISSING_DATA" in sub.flags
        assert "SHORT_HISTORY" in sub.flags


# ---------------------------------------------------------------------------
# Evidence, range, flags
# ---------------------------------------------------------------------------


class TestTechnicalEvidence:
    def test_evidence_is_populated(self) -> None:
        sub = TechnicalAnalyzer().analyze(_strong_security(), _ctx())
        assert sub.evidence, "a scored technical profile must emit evidence"
        for ev in sub.evidence:
            assert ev.value and isinstance(ev.value, str)
            assert ev.source
            assert ev.direction in {"bullish", "bearish", "neutral"}

    @pytest.mark.parametrize("builder", [_strong_security, _weak_security])
    def test_score_always_within_range(self, builder) -> None:
        sub = TechnicalAnalyzer().analyze(builder(), _ctx())
        assert 0.0 <= sub.score <= 100.0
        assert 0.0 <= sub.confidence <= 1.0
        assert 0.0 <= sub.data_coverage <= 1.0

    def test_golden_cross_flag_on_uptrend(self) -> None:
        sub = TechnicalAnalyzer().analyze(_strong_security(), _ctx())
        assert "SMA50_ABOVE_SMA200" in sub.flags


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


class TestTechnicalPurity:
    def test_deterministic(self) -> None:
        sec = _strong_security()
        a = TechnicalAnalyzer().analyze(sec, _ctx())
        b = TechnicalAnalyzer().analyze(sec, _ctx())
        assert a.score == b.score
        assert a.confidence == b.confidence
        assert a.flags == b.flags

    def test_does_not_mutate_input(self) -> None:
        sec = _strong_security()
        n_before = len(sec.price_history)
        last_close_before = sec.price_history[-1].close
        TechnicalAnalyzer().analyze(sec, _ctx())
        assert len(sec.price_history) == n_before
        assert sec.price_history[-1].close == last_close_before


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_analyzer_is_registered_for_technical() -> None:
    from convexity.core.registry import get_analyzer

    cls = get_analyzer(ScoreCategory.TECHNICAL)
    assert cls is TechnicalAnalyzer


def test_class_attrs() -> None:
    assert TechnicalAnalyzer.category == ScoreCategory.TECHNICAL
    assert "price_history" in TechnicalAnalyzer.requires
