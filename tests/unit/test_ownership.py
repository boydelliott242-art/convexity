"""Unit tests for :class:`convexity.analysis.ownership.OwnershipAnalyzer`.

These tests pin the *behavioural contract* of the OWNERSHIP analyzer using small
:class:`~convexity.core.models.SecurityData` objects built by hand:

* A **strong** ownership profile (cluster insider buying + broad institutional
  accumulation) scores high, with auditable, populated evidence.
* A **weak** profile (insider distribution + institutional selling) scores low.
* **All-missing** ownership data returns a neutral (50), low-confidence sub-score
  flagged ``MISSING_DATA`` — a gap is never treated as a negative.
* The score is always within ``[0, 100]`` and is **pure** (identical inputs ->
  identical output, no I/O / clock / randomness).
* Comparative context (``peer_stats`` / ``universe_stats``) is used when present
  and the analyzer degrades gracefully when it is ``None``.
"""

from __future__ import annotations

import datetime as _dt
from typing import List, Optional

import pytest

from convexity.analysis.ownership import OwnershipAnalyzer
from convexity.core.contracts import AnalysisContext
from convexity.core.models import (
    InsiderTransaction,
    InstitutionalHolding,
    ScoreCategory,
    SecurityData,
)
from convexity.core.registry import get_analyzer

_AS_OF = _dt.datetime(2026, 1, 2, 12, 0, 0)
_D = _dt.date(2025, 12, 1)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _security(
    *,
    insiders: Optional[List[InsiderTransaction]] = None,
    institutions: Optional[List[InstitutionalHolding]] = None,
) -> SecurityData:
    """Build a minimal SecurityData carrying only ownership records."""
    return SecurityData(
        ticker="TST",
        name="Test Micro Co",
        as_of=_AS_OF,
        insider_transactions=insiders or [],
        institutional_holdings=institutions or [],
    )


def _buy(name: str, *, value: float = 250_000.0) -> InsiderTransaction:
    return InsiderTransaction(
        date=_D,
        insider_name=name,
        role="Director",
        transaction_type="buy",
        shares=value / 10.0,
        value=value,
    )


def _sell(name: str, *, value: float = 250_000.0) -> InsiderTransaction:
    return InsiderTransaction(
        date=_D,
        insider_name=name,
        role="Officer",
        transaction_type="sell",
        shares=value / 10.0,
        value=value,
    )


def _holding(holder: str, *, change_pct: Optional[float]) -> InstitutionalHolding:
    return InstitutionalHolding(
        holder=holder,
        shares=100_000.0,
        value=1_000_000.0,
        change_pct=change_pct,
        as_of=_D,
    )


def _ctx(**kwargs) -> AnalysisContext:
    return AnalysisContext(**kwargs)


# ---------------------------------------------------------------------------
# Registration & class contract
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registered_for_ownership_category(self) -> None:
        assert get_analyzer(ScoreCategory.OWNERSHIP) is OwnershipAnalyzer

    def test_class_attrs(self) -> None:
        a = OwnershipAnalyzer()
        assert a.category is ScoreCategory.OWNERSHIP
        assert a.default_weight > 0.0
        assert {"insider_transactions", "institutional_holdings"} <= a.requires


# ---------------------------------------------------------------------------
# Strong vs weak profiles
# ---------------------------------------------------------------------------


class TestStrongVsWeak:
    def test_strong_profile_scores_high(self) -> None:
        # Three distinct insiders buying (cluster) + broad institutional accumulation.
        data = _security(
            insiders=[_buy("Alice"), _buy("Bob"), _buy("Carol")],
            institutions=[
                _holding("Vanguard", change_pct=15.0),
                _holding("BlackRock", change_pct=20.0),
                _holding("Fidelity", change_pct=10.0),
                _holding("State Street", change_pct=8.0),
            ],
        )
        sub = OwnershipAnalyzer().analyze(data, _ctx())
        assert sub.category is ScoreCategory.OWNERSHIP
        assert sub.score >= 70.0
        assert 0.0 <= sub.score <= 100.0
        assert sub.evidence, "strong profile must carry auditable evidence"
        assert sub.confidence > 0.4
        assert sub.data_coverage == 1.0
        assert "INSIDER_CLUSTER_BUY" in sub.flags
        # At least one bullish evidence item.
        assert any(e.direction == "bullish" for e in sub.evidence)

    def test_weak_profile_scores_low(self) -> None:
        # Cluster insider selling + institutions broadly distributing.
        data = _security(
            insiders=[_sell("Alice"), _sell("Bob"), _sell("Carol")],
            institutions=[
                _holding("Vanguard", change_pct=-20.0),
                _holding("BlackRock", change_pct=-15.0),
                _holding("Fidelity", change_pct=-12.0),
            ],
        )
        sub = OwnershipAnalyzer().analyze(data, _ctx())
        assert sub.score <= 35.0
        assert 0.0 <= sub.score <= 100.0
        assert sub.evidence
        assert any(e.direction == "bearish" for e in sub.evidence)
        assert "INSIDER_CLUSTER_SELL" in sub.flags
        assert "INSTITUTIONAL_DISTRIBUTION" in sub.flags

    def test_strong_beats_weak(self) -> None:
        strong = OwnershipAnalyzer().analyze(
            _security(
                insiders=[_buy("A"), _buy("B")],
                institutions=[_holding("X", change_pct=12.0), _holding("Y", change_pct=9.0)],
            ),
            _ctx(),
        )
        weak = OwnershipAnalyzer().analyze(
            _security(
                insiders=[_sell("A"), _sell("B")],
                institutions=[_holding("X", change_pct=-12.0), _holding("Y", change_pct=-9.0)],
            ),
            _ctx(),
        )
        assert strong.score > weak.score


# ---------------------------------------------------------------------------
# Missing / partial data honesty
# ---------------------------------------------------------------------------


class TestMissingData:
    def test_all_missing_is_neutral_low_confidence(self) -> None:
        sub = OwnershipAnalyzer().analyze(_security(), _ctx())
        assert sub.score == pytest.approx(50.0)
        assert sub.confidence <= 0.15
        assert sub.data_coverage == pytest.approx(0.0)
        assert "MISSING_DATA" in sub.flags
        assert sub.evidence == []

    def test_only_grants_have_no_directional_signal(self) -> None:
        # Grant/exercise records carry no open-market conviction; with no
        # institutional data this collapses to the missing-data fallback.
        grant = InsiderTransaction(
            date=_D, insider_name="Alice", transaction_type="grant", shares=1000.0, value=None
        )
        sub = OwnershipAnalyzer().analyze(_security(insiders=[grant]), _ctx())
        assert sub.score == pytest.approx(50.0)
        assert "MISSING_DATA" in sub.flags

    def test_partial_data_flagged_and_coverage_half(self) -> None:
        # Insider data only -> coverage 0.5, capped confidence, partial flag.
        sub = OwnershipAnalyzer().analyze(
            _security(insiders=[_buy("Alice"), _buy("Bob")]), _ctx()
        )
        assert sub.data_coverage == pytest.approx(0.5)
        assert sub.confidence <= 0.5
        assert "PARTIAL_OWNERSHIP_DATA" in sub.flags
        assert sub.score > 50.0  # net buying still reads bullish.

    def test_institutions_without_change_data(self) -> None:
        # Holders present but no change_pct -> breadth-only, gap flagged.
        sub = OwnershipAnalyzer().analyze(
            _security(
                institutions=[_holding("X", change_pct=None), _holding("Y", change_pct=None)]
            ),
            _ctx(),
        )
        assert 0.0 <= sub.score <= 100.0
        assert "NO_INSTITUTIONAL_CHANGE_DATA" in sub.flags
        assert "PARTIAL_OWNERSHIP_DATA" in sub.flags


# ---------------------------------------------------------------------------
# Comparative context (peer / universe stats)
# ---------------------------------------------------------------------------


class TestComparativeContext:
    def test_peer_relative_breadth_rewards_high_holder_count(self) -> None:
        data = _security(institutions=[_holding(f"H{i}", change_pct=None) for i in range(20)])
        # 20 holders sits at the top of this peer distribution.
        ctx = _ctx(peer_stats={"institutional_holder_count": [1, 2, 3, 5, 8, 20]})
        sub = OwnershipAnalyzer().analyze(data, ctx)
        # Top-of-distribution breadth should read clearly attractive.
        assert sub.score >= 60.0

    def test_peer_relative_breadth_penalizes_low_holder_count(self) -> None:
        data = _security(institutions=[_holding("Solo", change_pct=None)])
        ctx = _ctx(peer_stats={"institutional_holder_count": [1, 10, 20, 30, 40, 50]})
        sub = OwnershipAnalyzer().analyze(data, ctx)
        assert sub.score <= 55.0

    def test_degrades_gracefully_without_context(self) -> None:
        data = _security(
            insiders=[_buy("Alice")],
            institutions=[_holding("X", change_pct=5.0)],
        )
        # No peer/universe stats supplied at all.
        sub = OwnershipAnalyzer().analyze(data, _ctx())
        assert 0.0 <= sub.score <= 100.0
        assert sub.evidence

    def test_universe_stats_used_when_no_peer_stats(self) -> None:
        data = _security(institutions=[_holding(f"H{i}", change_pct=None) for i in range(15)])
        ctx = _ctx(universe_stats={"institutional_holder_count": [1, 2, 4, 6, 15]})
        sub = OwnershipAnalyzer().analyze(data, ctx)
        assert sub.score >= 60.0


# ---------------------------------------------------------------------------
# Evidence quality, range & purity
# ---------------------------------------------------------------------------


class TestEvidenceAndPurity:
    def test_evidence_items_cite_numbers_and_sources(self) -> None:
        data = _security(
            insiders=[_buy("Alice"), _buy("Bob")],
            institutions=[_holding("X", change_pct=10.0), _holding("Y", change_pct=5.0)],
        )
        sub = OwnershipAnalyzer().analyze(data, _ctx())
        assert sub.evidence
        for ev in sub.evidence:
            assert ev.label and ev.value
            assert ev.source
            assert ev.direction in {"bullish", "bearish", "neutral"}
        labels = {e.label for e in sub.evidence}
        assert "Net insider buy ratio" in labels
        assert "Institutional holders" in labels

    @pytest.mark.parametrize(
        "insiders,institutions",
        [
            ([_buy("A"), _buy("B"), _buy("C")], [_holding("X", change_pct=50.0)]),
            ([_sell("A"), _sell("B"), _sell("C")], [_holding("X", change_pct=-50.0)]),
            ([_buy("A"), _sell("B")], [_holding("X", change_pct=0.0)]),
            ([], [_holding("X", change_pct=200.0)]),
        ],
    )
    def test_score_always_within_range(
        self,
        insiders: List[InsiderTransaction],
        institutions: List[InstitutionalHolding],
    ) -> None:
        sub = OwnershipAnalyzer().analyze(
            _security(insiders=insiders, institutions=institutions), _ctx()
        )
        assert 0.0 <= sub.score <= 100.0
        assert 0.0 <= sub.confidence <= 1.0
        assert 0.0 <= sub.data_coverage <= 1.0

    def test_determinism(self) -> None:
        data = _security(
            insiders=[_buy("Alice"), _sell("Bob")],
            institutions=[_holding("X", change_pct=7.5), _holding("Y", change_pct=-3.0)],
        )
        a = OwnershipAnalyzer().analyze(data, _ctx())
        b = OwnershipAnalyzer().analyze(data, _ctx())
        assert a.score == b.score
        assert a.confidence == b.confidence
        assert a.data_coverage == b.data_coverage
        assert [e.value for e in a.evidence] == [e.value for e in b.evidence]

    def test_extreme_change_pct_does_not_break_score(self) -> None:
        # A single absurd +10000% change must be clamped, not dominate.
        data = _security(
            institutions=[
                _holding("X", change_pct=10_000.0),
                _holding("Y", change_pct=2.0),
            ]
        )
        sub = OwnershipAnalyzer().analyze(data, _ctx())
        assert 0.0 <= sub.score <= 100.0
