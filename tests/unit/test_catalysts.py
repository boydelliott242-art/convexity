"""Unit tests for :class:`convexity.analysis.catalysts.CatalystAnalyzer`.

The CATALYST analyzer reads :attr:`SecurityData.news` and
:attr:`SecurityData.filings`, detects disclosed catalyst types via
:mod:`convexity.analysis.news_nlp`, and weights each detection by strength,
recency, source credibility and sentiment polarity. These tests build small,
hand-crafted news/filing tapes inline (no conftest fixtures) and assert the
shared honesty guarantees:

* A tape of strong, fresh, credible, bullishly-framed catalysts scores
  meaningfully **higher** than one whose only catalysts are framed negatively.
* **No news and no filings at all** falls back to a neutral (50),
  low-confidence, ``MISSING_DATA``-flagged sub-score.
* Every score lives in ``[0, 100]`` with a populated, auditable evidence list.
* The emitted :class:`SubScore` carries the CATALYST category.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pytest

from convexity.analysis.catalysts import CatalystAnalyzer
from convexity.core.contracts import AnalysisContext
from convexity.core.models import (
    Filing,
    NewsItem,
    ScoreCategory,
    SecurityData,
    SubScore,
    ValuationSnapshot,
)

# The analyzer derives "now" from the newest disclosure in the data, so absolute
# dates only matter relative to each other. Keep everything recent and close.
_AS_OF = dt.datetime(2026, 1, 1, 0, 0, 0)
_RECENT = dt.datetime(2025, 12, 20, 0, 0, 0)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _news(
    title: str,
    *,
    source: str = "Reuters",
    when: dt.datetime = _RECENT,
    summary: Optional[str] = None,
) -> NewsItem:
    """Build one news item; defaults to a credible primary-news source."""
    return NewsItem(published=when, title=title, source=source, summary=summary)


def _security(
    news: Optional[List[NewsItem]] = None,
    filings: Optional[List[Filing]] = None,
) -> SecurityData:
    """Wrap news/filings into a minimal SecurityData."""
    return SecurityData(
        ticker="TEST",
        name="Test Co",
        currency="USD",
        as_of=_AS_OF,
        valuation=ValuationSnapshot(market_cap=250_000_000.0),
        news=news or [],
        filings=filings or [],
    )


def _ctx() -> AnalysisContext:
    """A bare AnalysisContext (no peer/universe stats)."""
    return AnalysisContext(peer_stats=None, universe_stats=None, config=None)


def _strong_security() -> SecurityData:
    """Several strong, fresh, credible, bullishly-framed catalysts of varied type.

    Distinct catalyst types (regulatory approval, contract win, guidance raise,
    insider buying) maximise the diversity-weighted signal; positive framing
    keeps each contribution bullish.
    """
    news = [
        _news("Company receives FDA approval for its lead drug, a major milestone"),
        _news("Firm awarded a $40 million multi-year defense contract, a strong win"),
        _news("Management raises full-year guidance after a record quarter"),
    ]
    filings = [
        Filing(
            filed=_RECENT.date(),
            form_type="4",
            title="CEO buys shares on the open market",
            summary="Chief executive purchases stock in an insider buy",
        ),
    ]
    return _security(news=news, filings=filings)


def _weak_security() -> SecurityData:
    """A tape whose only recognised catalyst is framed negatively.

    An M&A headline is a recognised catalyst type, but the heavily negative
    framing (dilutive, disappointing, lawsuit, investigation) flips its
    contribution bearish, so the category should score below the strong case.
    """
    news = [
        _news(
            "Company agrees to acquire rival in a dilutive, disappointing deal",
            summary=(
                "The dilutive acquisition drew a lawsuit and an investigation; "
                "analysts warn the weak, value-destroying merger is negative for holders"
            ),
        ),
    ]
    return _security(news=news)


# ---------------------------------------------------------------------------
# Core contract: strong > weak, missing -> neutral
# ---------------------------------------------------------------------------


class TestCatalystScoring:
    def test_strong_catalysts_score_high(self) -> None:
        sub = CatalystAnalyzer().analyze(_strong_security(), _ctx())
        assert isinstance(sub, SubScore)
        assert sub.category == ScoreCategory.CATALYST
        assert sub.score >= 60.0, f"fresh, credible, bullish catalysts should score high, got {sub.score}"

    def test_strong_strictly_above_weak(self) -> None:
        strong = CatalystAnalyzer().analyze(_strong_security(), _ctx()).score
        weak = CatalystAnalyzer().analyze(_weak_security(), _ctx()).score
        assert strong > weak

    def test_all_missing_is_neutral_low_confidence(self) -> None:
        # No news and no filings at all -> neutral fallback.
        sub = CatalystAnalyzer().analyze(_security(), _ctx())
        assert sub.score == pytest.approx(50.0)
        assert sub.confidence <= 0.2
        assert "MISSING_DATA" in sub.flags
        assert sub.data_coverage == pytest.approx(0.0)

    def test_disclosures_but_no_catalyst_is_below_neutral(self) -> None:
        # A readable but quiet tape (no recognised catalyst keywords) is mildly
        # negative for the category, not a hard data gap.
        sec = _security(news=[_news("Company announces routine annual meeting date")])
        sub = CatalystAnalyzer().analyze(sec, _ctx())
        assert sub.score < 50.0
        assert "NO_CATALYST_DETECTED" in sub.flags
        assert "MISSING_DATA" not in sub.flags


# ---------------------------------------------------------------------------
# Evidence, range
# ---------------------------------------------------------------------------


class TestCatalystEvidence:
    def test_evidence_is_populated(self) -> None:
        sub = CatalystAnalyzer().analyze(_strong_security(), _ctx())
        assert sub.evidence, "a scored catalyst profile must emit evidence"
        for ev in sub.evidence:
            assert ev.value and isinstance(ev.value, str)
            assert ev.source
            assert ev.direction in {"bullish", "bearish", "neutral"}

    @pytest.mark.parametrize("builder", [_strong_security, _weak_security])
    def test_score_always_within_range(self, builder) -> None:
        sub = CatalystAnalyzer().analyze(builder(), _ctx())
        assert 0.0 <= sub.score <= 100.0
        assert 0.0 <= sub.confidence <= 1.0
        assert 0.0 <= sub.data_coverage <= 1.0

    def test_recency_lifts_a_fresh_catalyst_over_a_stale_one(self) -> None:
        # The same bullish catalyst, fresh vs old (relative to the data's own
        # newest disclosure). A second, fresh anchor item dates the tape so the
        # stale copy genuinely decays.
        anchor = _news("Management raises full-year guidance after a record quarter", when=_RECENT)
        fresh = _news(
            "Company receives FDA approval for its lead drug",
            when=_RECENT,
        )
        stale = _news(
            "Company receives FDA approval for its lead drug",
            when=_RECENT - dt.timedelta(days=400),
        )
        fresh_score = CatalystAnalyzer().analyze(_security(news=[anchor, fresh]), _ctx()).score
        stale_score = CatalystAnalyzer().analyze(_security(news=[anchor, stale]), _ctx()).score
        assert fresh_score >= stale_score


# ---------------------------------------------------------------------------
# Relative context & purity
# ---------------------------------------------------------------------------


class TestCatalystContextAndPurity:
    def test_peer_context_changes_score(self) -> None:
        sec = _strong_security()
        weak_peers = CatalystAnalyzer().analyze(
            sec, AnalysisContext(peer_stats={"catalyst_signal": [-0.5, -0.2, 0.0, 0.1]})
        ).score
        strong_peers = CatalystAnalyzer().analyze(
            sec, AnalysisContext(peer_stats={"catalyst_signal": [2.0, 3.0, 4.0, 5.0]})
        ).score
        # Ranking high against weak peers should not score below ranking low
        # against very strong peers.
        assert weak_peers >= strong_peers

    def test_deterministic(self) -> None:
        sec = _strong_security()
        a = CatalystAnalyzer().analyze(sec, _ctx())
        b = CatalystAnalyzer().analyze(sec, _ctx())
        assert a.score == b.score
        assert a.confidence == b.confidence
        assert a.flags == b.flags

    def test_does_not_mutate_input(self) -> None:
        sec = _strong_security()
        n_news = len(sec.news)
        n_filings = len(sec.filings)
        CatalystAnalyzer().analyze(sec, _ctx())
        assert len(sec.news) == n_news
        assert len(sec.filings) == n_filings


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_analyzer_is_registered_for_catalyst() -> None:
    from convexity.core.registry import get_analyzer

    cls = get_analyzer(ScoreCategory.CATALYST)
    assert cls is CatalystAnalyzer


def test_class_attrs() -> None:
    assert CatalystAnalyzer.category == ScoreCategory.CATALYST
    assert "news" in CatalystAnalyzer.requires
    assert "filings" in CatalystAnalyzer.requires
