"""Unit tests for :class:`convexity.analysis.momentum.MomentumAnalyzer`.

These tests pin the analyzer's *behavioural contract* using small, hand-built
:class:`~convexity.core.models.SecurityData` objects with synthetic price
histories — no network, no real data:

* A strong, persistent up-trend scores **high**.
* A sustained down-trend scores **low**.
* Missing / too-short price history yields a **neutral, low-confidence**
  sub-score flagged ``MISSING_DATA`` (a data gap must neither help nor hurt).
* Every scored case populates auditable :class:`Evidence` and stays within the
  validated ``[0, 100]`` band with a sane confidence/coverage.
* Blow-off extremes (a parabolic last-month spike, extreme overbought RSI) are
  flagged and discounted rather than rewarded.
* Relative strength from ``ctx.universe_stats`` lifts a universe-leading name and
  the analyzer degrades gracefully when that benchmark is absent.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Sequence
from typing import List, Optional

import pytest

from convexity.analysis.momentum import MomentumAnalyzer
from convexity.core.contracts import AnalysisContext
from convexity.core.models import PriceBar, ScoreCategory, SecurityData, ValuationSnapshot

_START = _dt.date(2024, 1, 1)


def _bars_from_closes(closes: Sequence[float]) -> List[PriceBar]:
    """Build a daily, oldest-first OHLCV series from a list of close prices."""
    bars: List[PriceBar] = []
    for i, c in enumerate(closes):
        c = float(c)
        bars.append(
            PriceBar(
                date=_START + _dt.timedelta(days=i),
                open=c,
                high=c * 1.01,
                low=c * 0.99,
                close=c,
                adj_close=c,
                volume=100_000.0,
            )
        )
    return bars


def _make_security(closes: Sequence[float], ticker: str = "TEST") -> SecurityData:
    """Wrap a close series in a minimal :class:`SecurityData`."""
    return SecurityData(
        ticker=ticker,
        name=f"{ticker} Inc.",
        as_of=_dt.datetime(2024, 12, 31, tzinfo=_dt.timezone.utc),
        valuation=ValuationSnapshot(),
        price_history=_bars_from_closes(closes),
    )


def _ctx(universe_stats: Optional[dict] = None) -> AnalysisContext:
    return AnalysisContext(peer_stats=None, universe_stats=universe_stats, config=None)


def _steady_trend(n: int, daily_rate: float, base: float = 100.0, noise: float = 0.0) -> List[float]:
    """A deterministic compounding price path of ``n`` bars at ``daily_rate``.

    ``noise`` adds a tiny deterministic ripple (a sine wave) so RSI and MACD have
    realistic up/down deltas to chew on without breaking determinism.
    """
    out: List[float] = []
    px = base
    for i in range(n):
        px *= 1.0 + daily_rate
        ripple = 1.0 + noise * math.sin(i / 5.0)
        out.append(px * ripple)
    return out


# 13 months of bars so the 12-month / 12–1 look-backs are fully available.
_N = 280


# ---------------------------------------------------------------------------
# Strong vs weak profiles
# ---------------------------------------------------------------------------


class TestStrongVsWeak:
    def test_strong_uptrend_scores_high(self) -> None:
        # ~0.2%/day compounding ≈ strong, persistent up-trend across all horizons.
        data = _make_security(_steady_trend(_N, 0.002, noise=0.01))
        sub = MomentumAnalyzer().analyze(data, _ctx())
        assert sub.category == ScoreCategory.MOMENTUM
        assert sub.score >= 65.0
        assert "PERSISTENT_TREND" in sub.flags
        assert sub.confidence > 0.4
        assert sub.data_coverage > 0.8

    def test_sustained_downtrend_scores_low(self) -> None:
        data = _make_security(_steady_trend(_N, -0.002, noise=0.01))
        sub = MomentumAnalyzer().analyze(data, _ctx())
        assert sub.score <= 35.0
        assert sub.data_coverage > 0.8

    def test_strong_beats_weak(self) -> None:
        up = MomentumAnalyzer().analyze(_make_security(_steady_trend(_N, 0.002, noise=0.01)), _ctx())
        down = MomentumAnalyzer().analyze(_make_security(_steady_trend(_N, -0.002, noise=0.01)), _ctx())
        assert up.score > down.score

    def test_flat_market_is_near_neutral(self) -> None:
        # No drift: momentum should land mid-scale, not at an extreme.
        data = _make_security(_steady_trend(_N, 0.0, noise=0.01))
        sub = MomentumAnalyzer().analyze(data, _ctx())
        assert 35.0 <= sub.score <= 65.0


# ---------------------------------------------------------------------------
# Missing / insufficient data -> neutral, low confidence
# ---------------------------------------------------------------------------


class TestMissingData:
    def test_no_price_history_is_neutral_low_confidence(self) -> None:
        data = _make_security([])
        sub = MomentumAnalyzer().analyze(data, _ctx())
        assert sub.score == 50.0
        assert sub.confidence <= 0.1
        assert "MISSING_DATA" in sub.flags
        assert sub.data_coverage == 0.0

    def test_too_short_history_is_neutral(self) -> None:
        # Below the 3-month minimum -> cannot measure momentum honestly.
        data = _make_security(_steady_trend(40, 0.002))
        sub = MomentumAnalyzer().analyze(data, _ctx())
        assert sub.score == 50.0
        assert sub.confidence <= 0.1
        assert "MISSING_DATA" in sub.flags

    def test_partial_history_lowers_coverage_and_confidence(self) -> None:
        # ~4 months: enough to score, but no 6m/12m horizon -> partial coverage.
        full = MomentumAnalyzer().analyze(_make_security(_steady_trend(_N, 0.001, noise=0.01)), _ctx())
        partial = MomentumAnalyzer().analyze(_make_security(_steady_trend(80, 0.001, noise=0.01)), _ctx())
        assert partial.data_coverage < full.data_coverage
        assert partial.confidence < full.confidence
        assert 0.0 <= partial.score <= 100.0


# ---------------------------------------------------------------------------
# Evidence, range and determinism
# ---------------------------------------------------------------------------


class TestEvidenceAndRange:
    def test_evidence_is_populated_and_cites_numbers(self) -> None:
        data = _make_security(_steady_trend(_N, 0.0015, noise=0.01))
        sub = MomentumAnalyzer().analyze(data, _ctx())
        assert sub.evidence, "evidence must be populated for a scored momentum read"
        labels = {e.label for e in sub.evidence}
        assert any("12-month return" in lbl for lbl in labels)
        assert any("RSI" in lbl for lbl in labels)
        # Every evidence item carries a concrete rendered value and a direction.
        for e in sub.evidence:
            assert e.value
            assert e.direction in {"bullish", "bearish", "neutral"}

    @pytest.mark.parametrize("rate", [-0.004, -0.001, 0.0, 0.001, 0.004])
    def test_score_within_bounds(self, rate: float) -> None:
        sub = MomentumAnalyzer().analyze(_make_security(_steady_trend(_N, rate, noise=0.02)), _ctx())
        assert 0.0 <= sub.score <= 100.0
        assert 0.0 <= sub.confidence <= 1.0
        assert 0.0 <= sub.data_coverage <= 1.0

    def test_rationale_is_non_empty_and_honest(self) -> None:
        sub = MomentumAnalyzer().analyze(_make_security(_steady_trend(_N, 0.002, noise=0.01)), _ctx())
        assert sub.rationale
        assert "not a forecast" in sub.rationale.lower()

    def test_determinism(self) -> None:
        closes = _steady_trend(_N, 0.0012, noise=0.015)
        a = MomentumAnalyzer().analyze(_make_security(closes), _ctx())
        b = MomentumAnalyzer().analyze(_make_security(closes), _ctx())
        assert a.score == b.score
        assert a.confidence == b.confidence


# ---------------------------------------------------------------------------
# Blow-off / overbought discounting
# ---------------------------------------------------------------------------


class TestBlowOff:
    def test_parabolic_spike_is_flagged_and_discounted(self) -> None:
        # Flat for ~11 months, then a violent +120% one-month vertical move.
        flat = _steady_trend(_N - _N // 13, 0.0, base=100.0)
        last_flat = flat[-1]
        spike = [last_flat * (1.0 + 1.20 * (j + 1) / 21.0) for j in range(21)]
        data = _make_security(flat + spike)
        sub = MomentumAnalyzer().analyze(data, _ctx())
        assert "PARABOLIC_SPIKE" in sub.flags or "OVERBOUGHT_BLOWOFF" in sub.flags
        # Despite a huge raw 1-month gain, the blow-off discount keeps it from
        # pinning the score at the very top.
        assert sub.score <= 90.0

    def test_overbought_extreme_flagged(self) -> None:
        # A near-monotonic steep climb drives RSI to an extreme overbought level.
        data = _make_security(_steady_trend(_N, 0.01))  # ~1%/day, almost no down days
        sub = MomentumAnalyzer().analyze(data, _ctx())
        assert "OVERBOUGHT_BLOWOFF" in sub.flags


# ---------------------------------------------------------------------------
# Relative strength vs the universe
# ---------------------------------------------------------------------------


class TestRelativeStrength:
    def test_universe_leader_is_rewarded(self) -> None:
        # Moderate absolute up-trend, but top of a weak-to-negative universe.
        closes = _steady_trend(_N, 0.0008, noise=0.01)
        weak_universe = {"mom_12_1": [-0.30, -0.20, -0.10, -0.05, 0.0, 0.02]}
        with_rel = MomentumAnalyzer().analyze(_make_security(closes), _ctx(weak_universe))
        without_rel = MomentumAnalyzer().analyze(_make_security(closes), _ctx())
        assert with_rel.score >= without_rel.score
        assert "NO_RELATIVE_BENCHMARK" not in with_rel.flags
        assert any("Relative strength" in e.label for e in with_rel.evidence)
        # Knowing the relative standing should not lower confidence.
        assert with_rel.confidence >= without_rel.confidence

    def test_universe_laggard_is_penalised(self) -> None:
        closes = _steady_trend(_N, 0.0008, noise=0.01)
        strong_universe = {"mom_12_1": [0.40, 0.50, 0.60, 0.75, 0.90, 1.10]}
        laggard = MomentumAnalyzer().analyze(_make_security(closes), _ctx(strong_universe))
        leader = MomentumAnalyzer().analyze(
            _make_security(closes), _ctx({"mom_12_1": [-0.3, -0.2, -0.1, 0.0]})
        )
        assert laggard.score < leader.score

    def test_degrades_gracefully_without_universe(self) -> None:
        sub = MomentumAnalyzer().analyze(_make_security(_steady_trend(_N, 0.001, noise=0.01)), _ctx())
        assert "NO_RELATIVE_BENCHMARK" in sub.flags
        assert 0.0 <= sub.score <= 100.0

    def test_dict_shaped_universe_stats_accepted(self) -> None:
        # The nested {"values": [...]} shape must also be parsed.
        closes = _steady_trend(_N, 0.001, noise=0.01)
        nested = {"mom_12_1": {"values": [-0.2, -0.1, 0.0, 0.05]}}
        sub = MomentumAnalyzer().analyze(_make_security(closes), _ctx(nested))
        assert "NO_RELATIVE_BENCHMARK" not in sub.flags


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_analyzer_is_registered_for_momentum() -> None:
    from convexity.core.registry import get_analyzer

    cls = get_analyzer(ScoreCategory.MOMENTUM)
    assert cls is MomentumAnalyzer
