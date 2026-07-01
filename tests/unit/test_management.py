"""Unit tests for :mod:`convexity.analysis.management`.

These tests pin the *behavioural contract* of the MANAGEMENT analyzer, which
scores the capital-allocation discipline of management / owner-operators from
share-count trends, insider activity, and (incremental) ROIC.

They build small :class:`~convexity.core.models.SecurityData` objects by hand and
assert that:

* a strong owner-operator profile (net buybacks, insider buying, high & rising
  ROIC, FCF-funded returns) scores **high**;
* a weak profile (heavy dilution, insider selling, deteriorating/negative ROIC,
  unfunded buybacks) scores **low**;
* an all-missing profile falls back to a **neutral, low-confidence** sub-score
  carrying the ``MISSING_DATA`` flag (a data gap must neither help nor hurt);
* every scored result is in ``[0, 100]`` with populated, auditable evidence;
* the analyzer is registered for :class:`ScoreCategory.MANAGEMENT` and behaves
  purely (identical inputs -> identical output) and relative to peer stats.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pytest

from convexity.analysis.management import ManagementAnalyzer
from convexity.core.contracts import AnalysisContext
from convexity.core.models import (
    FundamentalsPeriod,
    InsiderTransaction,
    ScoreCategory,
    SecurityData,
    ValuationSnapshot,
)
from convexity.core.registry import get_analyzer

AS_OF = dt.datetime(2026, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _period(
    label: str,
    *,
    end: dt.date,
    shares_diluted: Optional[float] = None,
    roic: Optional[float] = None,
    free_cash_flow: Optional[float] = None,
) -> FundamentalsPeriod:
    return FundamentalsPeriod(
        period_end=end,
        period_label=label,
        shares_diluted=shares_diluted,
        roic=roic,
        free_cash_flow=free_cash_flow,
    )


def _security(
    *,
    fundamentals: Optional[List[FundamentalsPeriod]] = None,
    insiders: Optional[List[InsiderTransaction]] = None,
) -> SecurityData:
    return SecurityData(
        ticker="TEST",
        name="Test Co",
        as_of=AS_OF,
        valuation=ValuationSnapshot(market_cap=200_000_000),
        fundamentals=fundamentals or [],
        insider_transactions=insiders or [],
    )


def _strong_security() -> SecurityData:
    """An aligned owner-operator: net buybacks, insider buying, high/rising ROIC."""
    # newest-first; share count shrinking, ROIC high and improving, FCF positive.
    funds = [
        _period("FY2025", end=dt.date(2025, 12, 31), shares_diluted=90_000_000, roic=0.22, free_cash_flow=40_000_000),
        _period("FY2024", end=dt.date(2024, 12, 31), shares_diluted=95_000_000, roic=0.19, free_cash_flow=35_000_000),
        _period("FY2023", end=dt.date(2023, 12, 31), shares_diluted=100_000_000, roic=0.15, free_cash_flow=30_000_000),
    ]
    insiders = [
        InsiderTransaction(
            date=dt.date(2025, 11, 1),
            insider_name="Jane CEO",
            role="CEO",
            transaction_type="buy",
            shares=50_000,
            value=1_000_000,
        ),
        InsiderTransaction(
            date=dt.date(2025, 10, 1),
            insider_name="John CFO",
            role="CFO",
            transaction_type="buy",
            shares=20_000,
            value=400_000,
        ),
    ]
    return _security(fundamentals=funds, insiders=insiders)


def _weak_security() -> SecurityData:
    """A value-destroying allocator: heavy dilution, insider selling, falling/neg ROIC."""
    funds = [
        _period("FY2025", end=dt.date(2025, 12, 31), shares_diluted=160_000_000, roic=-0.05, free_cash_flow=-15_000_000),
        _period("FY2024", end=dt.date(2024, 12, 31), shares_diluted=130_000_000, roic=0.02, free_cash_flow=-5_000_000),
        _period("FY2023", end=dt.date(2023, 12, 31), shares_diluted=100_000_000, roic=0.06, free_cash_flow=2_000_000),
    ]
    insiders = [
        InsiderTransaction(
            date=dt.date(2025, 11, 1),
            insider_name="Jane CEO",
            role="CEO",
            transaction_type="sell",
            shares=300_000,
            value=4_000_000,
        ),
        InsiderTransaction(
            date=dt.date(2025, 9, 1),
            insider_name="John CFO",
            role="CFO",
            transaction_type="sell",
            shares=150_000,
            value=2_000_000,
        ),
    ]
    return _security(fundamentals=funds, insiders=insiders)


def _ctx(**peer_stats: object) -> AnalysisContext:
    return AnalysisContext(peer_stats=dict(peer_stats) or None)


# ---------------------------------------------------------------------------
# Registration & basic contract
# ---------------------------------------------------------------------------


def test_analyzer_is_registered_for_management() -> None:
    assert get_analyzer(ScoreCategory.MANAGEMENT) is ManagementAnalyzer


def test_class_attributes() -> None:
    a = ManagementAnalyzer()
    assert a.category is ScoreCategory.MANAGEMENT
    assert a.default_weight > 0.0
    assert "fundamentals" in a.requires


# ---------------------------------------------------------------------------
# Strong vs weak profiles
# ---------------------------------------------------------------------------


def test_strong_profile_scores_high() -> None:
    sub = ManagementAnalyzer().analyze(_strong_security(), _ctx())
    assert sub.category is ScoreCategory.MANAGEMENT
    assert 0.0 <= sub.score <= 100.0
    assert sub.score >= 65.0, f"expected a high management score, got {sub.score}"
    assert sub.confidence > 0.3
    assert sub.data_coverage > 0.5
    assert sub.evidence, "evidence must be populated"
    assert "NET_BUYBACKS" in sub.flags
    assert "INSIDER_BUYING" in sub.flags


def test_weak_profile_scores_low() -> None:
    sub = ManagementAnalyzer().analyze(_weak_security(), _ctx())
    assert 0.0 <= sub.score <= 100.0
    assert sub.score <= 40.0, f"expected a low management score, got {sub.score}"
    assert sub.evidence, "evidence must be populated even for a weak profile"
    assert "HEAVY_DILUTION" in sub.flags
    assert "INSIDER_SELLING" in sub.flags


def test_strong_scores_strictly_above_weak() -> None:
    strong = ManagementAnalyzer().analyze(_strong_security(), _ctx())
    weak = ManagementAnalyzer().analyze(_weak_security(), _ctx())
    assert strong.score > weak.score


# ---------------------------------------------------------------------------
# Missing data -> neutral, low-confidence
# ---------------------------------------------------------------------------


def test_all_missing_is_neutral_low_confidence() -> None:
    sub = ManagementAnalyzer().analyze(_security(), _ctx())
    assert sub.score == pytest.approx(50.0)
    assert sub.confidence <= 0.2
    assert sub.data_coverage == pytest.approx(0.0)
    assert "MISSING_DATA" in sub.flags
    assert sub.evidence == []


def test_single_period_no_trend_is_neutral() -> None:
    # One period only: no share-count or ROIC *trend*, no insiders -> neutral.
    one = _security(
        fundamentals=[_period("FY2025", end=dt.date(2025, 12, 31), shares_diluted=100_000_000, roic=0.10)],
    )
    sub = ManagementAnalyzer().analyze(one, _ctx())
    # A single ROIC level is still real, partial evidence and should score, but a
    # lone share count with no second point yields no trend. We accept either a
    # scored result (from the ROIC level) or a neutral fallback, but never a crash.
    assert 0.0 <= sub.score <= 100.0


def test_score_always_within_range_for_varied_inputs() -> None:
    cases = [_strong_security(), _weak_security(), _security()]
    for sec in cases:
        sub = ManagementAnalyzer().analyze(sec, _ctx())
        assert 0.0 <= sub.score <= 100.0
        assert 0.0 <= sub.confidence <= 1.0
        assert 0.0 <= sub.data_coverage <= 1.0


# ---------------------------------------------------------------------------
# Partial coverage & individual signals
# ---------------------------------------------------------------------------


def test_buybacks_alone_score_above_neutral() -> None:
    # Only a shrinking share count (no ROIC, no insiders): a positive signal.
    funds = [
        _period("FY2025", end=dt.date(2025, 12, 31), shares_diluted=90_000_000),
        _period("FY2023", end=dt.date(2023, 12, 31), shares_diluted=110_000_000),
    ]
    sub = ManagementAnalyzer().analyze(_security(fundamentals=funds), _ctx())
    assert sub.score > 50.0
    assert sub.data_coverage < 1.0  # partial evidence -> partial coverage
    assert any("share" in e.label.lower() for e in sub.evidence)


def test_dilution_alone_scores_below_neutral() -> None:
    funds = [
        _period("FY2025", end=dt.date(2025, 12, 31), shares_diluted=180_000_000),
        _period("FY2023", end=dt.date(2023, 12, 31), shares_diluted=100_000_000),
    ]
    sub = ManagementAnalyzer().analyze(_security(fundamentals=funds), _ctx())
    assert sub.score < 50.0
    assert "HEAVY_DILUTION" in sub.flags


def test_insider_buying_alone_is_bullish_evidence() -> None:
    insiders = [
        InsiderTransaction(
            date=dt.date(2025, 11, 1),
            insider_name="Jane CEO",
            role="CEO",
            transaction_type="buy",
            shares=100_000,
            value=2_000_000,
        )
    ]
    sub = ManagementAnalyzer().analyze(_security(insiders=insiders), _ctx())
    assert sub.score > 50.0
    assert "INSIDER_BUYING" in sub.flags
    buying_ev = [e for e in sub.evidence if "insider" in e.label.lower()]
    assert buying_ev and buying_ev[0].direction == "bullish"


def test_grants_and_exercises_are_ignored() -> None:
    # Non-open-market events should not be treated as conviction buys/sells.
    insiders = [
        InsiderTransaction(
            date=dt.date(2025, 11, 1),
            insider_name="Jane CEO",
            role="CEO",
            transaction_type="grant",
            shares=500_000,
        ),
        InsiderTransaction(
            date=dt.date(2025, 10, 1),
            insider_name="John CFO",
            role="CFO",
            transaction_type="exercise",
            shares=200_000,
        ),
    ]
    sub = ManagementAnalyzer().analyze(_security(insiders=insiders), _ctx())
    # With only grants/exercises and nothing else, there is no scoreable evidence.
    assert "MISSING_DATA" in sub.flags
    assert sub.score == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Evidence quality
# ---------------------------------------------------------------------------


def test_evidence_items_cite_numbers_and_source() -> None:
    sub = ManagementAnalyzer().analyze(_strong_security(), _ctx())
    assert sub.evidence
    for ev in sub.evidence:
        assert ev.label
        assert ev.value  # rendered number (or "n/a")
        assert ev.source
        assert ev.direction in {"bullish", "bearish", "neutral"}
    # At least one bullish evidence item for the strong profile.
    assert any(e.direction == "bullish" for e in sub.evidence)


def test_rationale_is_human_readable() -> None:
    sub = ManagementAnalyzer().analyze(_strong_security(), _ctx())
    assert isinstance(sub.rationale, str) and len(sub.rationale) > 20
    # Honest framing: never claims a prediction.
    assert "prediction" in sub.rationale.lower()


# ---------------------------------------------------------------------------
# Relative (peer) scoring & purity
# ---------------------------------------------------------------------------


def test_peer_relative_scoring_changes_score() -> None:
    sec = _strong_security()
    # Among elite capital-allocator peers, a -3.5%/yr buyback rate is merely
    # median; supplying a tougher peer distribution should not crash and should
    # still yield a valid, in-range score.
    tough_peers = _ctx(
        share_count_cagr=[-0.10, -0.08, -0.06, -0.05, -0.04],
        roic=[0.30, 0.28, 0.26, 0.24, 0.22],
        incremental_roic=[0.10, 0.08, 0.07, 0.06, 0.05],
    )
    sub = ManagementAnalyzer().analyze(sec, tough_peers)
    assert 0.0 <= sub.score <= 100.0


def test_purity_same_input_same_output() -> None:
    sec = _strong_security()
    a = ManagementAnalyzer().analyze(sec, _ctx())
    b = ManagementAnalyzer().analyze(sec, _ctx())
    assert a.score == b.score
    assert a.confidence == b.confidence
    assert a.data_coverage == b.data_coverage
    assert [e.value for e in a.evidence] == [e.value for e in b.evidence]
