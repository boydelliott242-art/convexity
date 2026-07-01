"""Unit tests for :class:`convexity.analysis.historical_analog.HistoricalAnalogAnalyzer`.

The HISTORICAL_ANALOG analyzer measures how strongly a company's current,
observable data *resembles* one of a handful of named small-cap re-rating
archetypes (a *similarity* score, never a forecast). These tests build small,
hand-crafted :class:`SecurityData` objects inline (no conftest fixtures) and
assert the shared honesty guarantees:

* A company that cleanly fits an encoded archetype (e.g. an under-followed
  compounder / capacity-led operating-leverage setup) scores meaningfully
  **higher** than one that resembles no archetype.
* **Too little observable substance** falls back to a neutral (50),
  low-confidence, ``MISSING_DATA``-flagged sub-score.
* Every score lives in ``[0, 100]`` with a populated, auditable evidence list.
* The emitted :class:`SubScore` carries the HISTORICAL_ANALOG category.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pytest

from convexity.analysis.historical_analog import HistoricalAnalogAnalyzer, extract_features
from convexity.core.contracts import AnalysisContext
from convexity.core.models import (
    FundamentalsPeriod,
    InstitutionalHolding,
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
    revenue: Optional[float] = None,
    operating_income: Optional[float] = None,
    net_income: Optional[float] = None,
    free_cash_flow: Optional[float] = None,
    operating_margin: Optional[float] = None,
    gross_margin: Optional[float] = None,
    roic: Optional[float] = None,
    roe: Optional[float] = None,
) -> FundamentalsPeriod:
    """Build one annual fundamentals period labelled by fiscal ``year``."""
    return FundamentalsPeriod(
        period_end=dt.date(year, 12, 31),
        period_label=f"FY{year}",
        revenue=revenue,
        operating_income=operating_income,
        net_income=net_income,
        free_cash_flow=free_cash_flow,
        operating_margin=operating_margin,
        gross_margin=gross_margin,
        roic=roic,
        roe=roe,
    )


def _security(
    periods: List[FundamentalsPeriod],
    *,
    price_history: Optional[List[PriceBar]] = None,
    institutional_holdings: Optional[List[InstitutionalHolding]] = None,
    news_titles: Optional[List[str]] = None,
    valuation: Optional[ValuationSnapshot] = None,
) -> SecurityData:
    """Wrap fundamentals (newest-first) into a minimal SecurityData."""
    from convexity.core.models import NewsItem

    news = [NewsItem(published=_AS_OF, title=t, source="Reuters") for t in (news_titles or [])]
    return SecurityData(
        ticker="TEST",
        name="Test Co",
        currency="USD",
        as_of=_AS_OF,
        valuation=valuation or ValuationSnapshot(market_cap=250_000_000.0),
        fundamentals=periods,
        price_history=price_history or [],
        institutional_holdings=institutional_holdings or [],
        news=news,
    )


def _ctx() -> AnalysisContext:
    """A bare AnalysisContext (no peer/universe stats)."""
    return AnalysisContext(peer_stats=None, universe_stats=None, config=None)


def _strong_security() -> SecurityData:
    """An under-followed compounder with capacity-led operating leverage.

    Strong revenue growth, expanding operating margin, positive free cash flow,
    high ROIC/ROE, and almost no institutional following — a clean fit for the
    encoded compounder / operating-leverage archetypes.
    """
    periods = [
        _period(2025, revenue=150.0, operating_income=33.0, net_income=24.0, free_cash_flow=22.0,
                operating_margin=0.22, gross_margin=0.60, roic=0.18, roe=0.20),
        _period(2024, revenue=110.0, operating_income=16.5, net_income=11.0, free_cash_flow=10.0,
                operating_margin=0.15, gross_margin=0.59, roic=0.12, roe=0.14),
    ]
    # One institutional holder only -> "under-followed".
    holdings = [InstitutionalHolding(holder="Lone Fund LP", shares=10_000.0)]
    return _security(periods, institutional_holdings=holdings)


def _weak_security() -> SecurityData:
    """A profile that resembles no encoded archetype.

    Shrinking revenue, contracting margins, negative free cash flow, negative
    returns on capital, and heavy institutional ownership — no turn, no
    compounding, no neglect.
    """
    periods = [
        _period(2025, revenue=80.0, operating_income=-10.0, net_income=-12.0, free_cash_flow=-8.0,
                operating_margin=-0.12, gross_margin=0.20, roic=-0.10, roe=-0.12),
        _period(2024, revenue=120.0, operating_income=6.0, net_income=4.0, free_cash_flow=5.0,
                operating_margin=0.05, gross_margin=0.30, roic=0.04, roe=0.05),
    ]
    holdings = [InstitutionalHolding(holder=f"Big Fund {i}", shares=1_000_000.0) for i in range(40)]
    return _security(periods, institutional_holdings=holdings)


# ---------------------------------------------------------------------------
# Core contract: strong high, weak low, missing -> neutral
# ---------------------------------------------------------------------------


class TestHistoricalAnalogScoring:
    def test_strong_archetype_fit_scores_high(self) -> None:
        sub = HistoricalAnalogAnalyzer().analyze(_strong_security(), _ctx())
        assert isinstance(sub, SubScore)
        assert sub.category == ScoreCategory.HISTORICAL_ANALOG
        assert sub.score >= 55.0, f"a clean archetype fit should score high, got {sub.score}"

    def test_strong_strictly_above_weak(self) -> None:
        strong = HistoricalAnalogAnalyzer().analyze(_strong_security(), _ctx()).score
        weak = HistoricalAnalogAnalyzer().analyze(_weak_security(), _ctx()).score
        assert strong > weak + 20.0

    def test_no_substance_is_neutral_low_confidence(self) -> None:
        # No fundamentals, prices, insider or institutional data at all -> too
        # little to compare against any archetype.
        empty = _security([])
        sub = HistoricalAnalogAnalyzer().analyze(empty, _ctx())
        assert sub.score == pytest.approx(50.0)
        assert sub.confidence <= 0.2
        assert "MISSING_DATA" in sub.flags
        assert "NO_ARCHETYPE_BASIS" in sub.flags

    def test_single_substantive_feature_is_neutral(self) -> None:
        # Only one period (no period/period deltas, no growth) yields fewer than
        # the two substantive features the analyzer requires to assert an analogy.
        one = _security([_period(2025, revenue=100.0)])
        sub = HistoricalAnalogAnalyzer().analyze(one, _ctx())
        assert sub.score == pytest.approx(50.0)
        assert "MISSING_DATA" in sub.flags


# ---------------------------------------------------------------------------
# Evidence, range
# ---------------------------------------------------------------------------


class TestHistoricalAnalogEvidence:
    def test_evidence_is_populated_and_names_an_archetype(self) -> None:
        sub = HistoricalAnalogAnalyzer().analyze(_strong_security(), _ctx())
        assert sub.evidence, "a scored analog profile must emit evidence"
        labels = [e.label for e in sub.evidence]
        assert any("archetype" in lbl.lower() for lbl in labels)
        for ev in sub.evidence:
            assert ev.value and isinstance(ev.value, str)
            assert ev.source
            assert ev.direction in {"bullish", "bearish", "neutral"}

    @pytest.mark.parametrize("builder", [_strong_security, _weak_security])
    def test_score_always_within_range(self, builder) -> None:
        sub = HistoricalAnalogAnalyzer().analyze(builder(), _ctx())
        assert 0.0 <= sub.score <= 100.0
        assert 0.0 <= sub.confidence <= 1.0
        assert 0.0 <= sub.data_coverage <= 1.0


# ---------------------------------------------------------------------------
# Relative cheapness with the pipeline's summary-dict stats shape
# ---------------------------------------------------------------------------


def _pipeline_stat(values: List[float]) -> dict:
    """Build a metric entry in the pipeline's ``_summarise_group`` dict shape.

    In a real scan each ``ctx.peer_stats`` / ``ctx.universe_stats`` metric entry is
    a summary mapping ``{"values","count","min","max","mean","median"}`` — NOT a
    bare list. This mirrors that shape so the tests exercise the real code path.
    """
    ordered = sorted(values)
    n = len(ordered)
    return {
        "values": ordered,
        "count": n,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / n,
        "median": ordered[n // 2],
    }


def _cheap_security() -> SecurityData:
    """A compounder that also screens cheap on ``ev_ebitda`` / ``p_s`` / ``ev_sales``."""
    periods = [
        _period(2025, revenue=150.0, operating_income=33.0, net_income=24.0, free_cash_flow=22.0,
                operating_margin=0.22, gross_margin=0.60, roic=0.18, roe=0.20),
        _period(2024, revenue=110.0, operating_income=16.5, net_income=11.0, free_cash_flow=10.0,
                operating_margin=0.15, gross_margin=0.59, roic=0.12, roe=0.14),
    ]
    holdings = [InstitutionalHolding(holder="Lone Fund LP", shares=10_000.0)]
    valuation = ValuationSnapshot(
        market_cap=250_000_000.0, ev_ebitda=5.0, p_s=1.0, ev_sales=1.2
    )
    return _security(periods, institutional_holdings=holdings, valuation=valuation)


class TestHistoricalAnalogRelativeCheapness:
    """Peer/universe cheapness must work with the pipeline's summary-dict stats.

    ``ScanPipeline._summarise_group`` supplies each metric entry as a mapping
    (``{"values": [...], "count": ..., ...}``), not a bare list. Passing that dict
    straight to ``percentile_rank`` used to raise ``ValueError: could not convert
    string to float: 'values'`` (it iterated the dict's *keys*), which the pipeline
    caught and degraded HISTORICAL_ANALOG to a neutral subscore — silently
    disabling the relative-cheapness path. These tests pin that the summary-dict
    shape is coerced and the relative cheapness is actually computed.
    """

    def test_pipeline_shaped_peer_stats_do_not_raise_and_score(self) -> None:
        # The exact shape the pipeline builds for peer_stats.
        peer_stats = {
            "ev_ebitda": _pipeline_stat([10.0, 12.0, 14.0, 16.0, 18.0]),
            "p_s": _pipeline_stat([2.0, 3.0, 4.0, 5.0]),
            "ev_sales": _pipeline_stat([2.5, 3.5, 4.5]),
        }
        ctx = AnalysisContext(peer_stats=peer_stats, universe_stats=None, config=None)
        sub = HistoricalAnalogAnalyzer().analyze(_cheap_security(), ctx)  # must not raise
        assert isinstance(sub, SubScore)
        assert sub.category == ScoreCategory.HISTORICAL_ANALOG
        assert 0.0 <= sub.score <= 100.0
        # A genuine analogy was scored, not the MISSING_DATA neutral fallback the
        # pipeline would install if the analyzer had raised.
        assert "MISSING_DATA" not in sub.flags

    def test_relative_cheapness_engages_with_dict_stats(self) -> None:
        # With the company cheaper than every peer, the relative cheapness feature
        # should be computed (near 1.0) rather than falling back to None.
        peer_stats = {
            "ev_ebitda": _pipeline_stat([10.0, 12.0, 14.0, 16.0, 18.0]),
            "p_s": _pipeline_stat([2.0, 3.0, 4.0, 5.0]),
            "ev_sales": _pipeline_stat([2.5, 3.5, 4.5]),
        }
        ctx = AnalysisContext(peer_stats=peer_stats, universe_stats=None, config=None)
        feats = extract_features(_cheap_security(), ctx)
        cheapness = feats["valuation_cheapness"].value
        assert cheapness is not None, "relative cheapness must engage with dict-shaped stats"
        assert cheapness == pytest.approx(1.0)

        # Without any stats there is no distribution to rank against, so the feature
        # is honestly absent — confirming the value above came from the relative path.
        bare_ctx = AnalysisContext(peer_stats=None, universe_stats=None, config=None)
        assert extract_features(_cheap_security(), bare_ctx)["valuation_cheapness"].value is None

    def test_dict_shape_matches_bare_list_shape(self) -> None:
        # The dict shape and a plain list must yield the same cheapness (backward
        # compatibility with the bare-sequence form).
        values = [10.0, 12.0, 14.0, 16.0, 18.0]
        dict_ctx = AnalysisContext(
            peer_stats={"ev_ebitda": _pipeline_stat(values)}, universe_stats=None, config=None
        )
        list_ctx = AnalysisContext(
            peer_stats={"ev_ebitda": values}, universe_stats=None, config=None
        )
        dict_cheap = extract_features(_cheap_security(), dict_ctx)["valuation_cheapness"].value
        list_cheap = extract_features(_cheap_security(), list_ctx)["valuation_cheapness"].value
        assert dict_cheap == pytest.approx(list_cheap)

    def test_universe_stats_dict_shape_also_works(self) -> None:
        # Falls through to universe_stats when peers give nothing, coercing its
        # dict-shaped entries too.
        universe_stats = {"ev_ebitda": _pipeline_stat([11.0, 13.0, 15.0])}
        ctx = AnalysisContext(peer_stats=None, universe_stats=universe_stats, config=None)
        feats = extract_features(_cheap_security(), ctx)
        assert feats["valuation_cheapness"].value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


class TestHistoricalAnalogPurity:
    def test_deterministic(self) -> None:
        sec = _strong_security()
        a = HistoricalAnalogAnalyzer().analyze(sec, _ctx())
        b = HistoricalAnalogAnalyzer().analyze(sec, _ctx())
        assert a.score == b.score
        assert a.confidence == b.confidence
        assert a.flags == b.flags

    def test_does_not_mutate_input(self) -> None:
        sec = _strong_security()
        n_before = len(sec.fundamentals)
        rev_before = sec.fundamentals[0].revenue
        HistoricalAnalogAnalyzer().analyze(sec, _ctx())
        assert len(sec.fundamentals) == n_before
        assert sec.fundamentals[0].revenue == rev_before


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_analyzer_is_registered_for_historical_analog() -> None:
    from convexity.core.registry import get_analyzer

    cls = get_analyzer(ScoreCategory.HISTORICAL_ANALOG)
    assert cls is HistoricalAnalogAnalyzer


def test_class_attrs() -> None:
    assert HistoricalAnalogAnalyzer.category == ScoreCategory.HISTORICAL_ANALOG
    assert "fundamentals" in HistoricalAnalogAnalyzer.requires
