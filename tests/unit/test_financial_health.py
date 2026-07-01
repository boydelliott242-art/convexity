"""Unit tests for :class:`convexity.analysis.financial_health.FinancialHealthAnalyzer`.

These tests pin the *behavioural contract* of the FINANCIAL_HEALTH analyzer:

* A strong balance sheet (low leverage, ample liquidity, deep interest coverage,
  positive free cash flow, safe Altman-Z) scores **high** with solid confidence.
* A weak balance sheet (high leverage, sub-1x liquidity, uncovered interest,
  cash-burning with short runway, distress-zone Z) scores **low**.
* All-missing fundamentals -> a neutral (50) low-confidence sub-score carrying the
  ``MISSING_DATA`` flag (a data gap must neither help nor hurt).
* Every produced score is in ``[0, 100]``, carries populated, auditable
  :class:`Evidence`, and the analyzer is pure (same input -> same output).
* Peer/universe context is honoured: identical raw leverage scores differently
  depending on where it sits in the peer distribution.
* Hazards (negative equity, short runway, dilution) raise the documented flags.

The tests build small :class:`SecurityData` objects by hand so the analyzer is
exercised against the real Pydantic models, not stand-ins.
"""

from __future__ import annotations

import datetime as dt
from typing import List

import pytest

from convexity.analysis.financial_health import FinancialHealthAnalyzer
from convexity.core.contracts import AnalysisContext
from convexity.core.models import (
    FundamentalsPeriod,
    ScoreCategory,
    SecurityData,
    SubScore,
    ValuationSnapshot,
)

_AS_OF = dt.datetime(2026, 1, 1, 12, 0, 0)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _security(fundamentals: List[FundamentalsPeriod]) -> SecurityData:
    """Wrap fundamentals (newest-first) into a minimal :class:`SecurityData`."""
    return SecurityData(
        ticker="TEST",
        name="Test Co",
        as_of=_AS_OF,
        valuation=ValuationSnapshot(market_cap=150_000_000),
        fundamentals=fundamentals,
    )


def _strong_period() -> FundamentalsPeriod:
    """A robust balance sheet: net cash, deep coverage, self-funding, safe Z."""
    return FundamentalsPeriod(
        period_end=dt.date(2025, 12, 31),
        period_label="FY2025",
        revenue=500_000_000,
        operating_income=90_000_000,
        ebitda=110_000_000,
        net_income=70_000_000,
        free_cash_flow=80_000_000,          # strongly positive -> self-funding.
        operating_cash_flow=100_000_000,
        total_assets=600_000_000,
        total_debt=40_000_000,
        cash_and_equivalents=150_000_000,    # net cash.
        total_equity=420_000_000,
        shares_diluted=50_000_000,
        current_ratio=3.2,
        quick_ratio=2.6,
        debt_to_equity=0.10,
        interest_coverage=25.0,
    )


def _weak_period() -> FundamentalsPeriod:
    """A fragile balance sheet: high leverage, illiquid, uncovered, burning cash."""
    return FundamentalsPeriod(
        period_end=dt.date(2025, 12, 31),
        period_label="FY2025",
        revenue=40_000_000,
        operating_income=-12_000_000,
        ebitda=-8_000_000,                   # negative EBITDA.
        net_income=-20_000_000,
        free_cash_flow=-30_000_000,          # heavy burn.
        operating_cash_flow=-25_000_000,
        total_assets=120_000_000,
        total_debt=140_000_000,              # debt > assets.
        cash_and_equivalents=12_000_000,     # ~4.8 months runway at the burn.
        total_equity=-25_000_000,            # negative equity.
        shares_diluted=44_000_000,           # up 10% on prior -> dilution flag.
        current_ratio=0.6,
        quick_ratio=0.4,
        debt_to_equity=8.0,
        interest_coverage=0.4,               # interest not covered.
    )


def _weak_prior_period() -> FundamentalsPeriod:
    """Prior period for the weak company (lower share count -> dilution YoY)."""
    return FundamentalsPeriod(
        period_end=dt.date(2024, 12, 31),
        period_label="FY2024",
        shares_diluted=40_000_000,
    )


def _ctx(**kwargs: object) -> AnalysisContext:
    return AnalysisContext(**kwargs)  # type: ignore[arg-type]


def _evidence_labels(sub: SubScore) -> List[str]:
    return [e.label for e in sub.evidence]


# ---------------------------------------------------------------------------
# Registration & class contract
# ---------------------------------------------------------------------------


def test_class_contract() -> None:
    a = FinancialHealthAnalyzer()
    assert a.category is ScoreCategory.FINANCIAL_HEALTH
    assert a.default_weight > 0.0
    assert "fundamentals" in a.requires


def test_self_registers() -> None:
    # Importing convexity.analysis must register this analyzer by its category.
    import convexity.analysis  # noqa: F401  (triggers self-registration)
    from convexity.core.registry import get_analyzer

    cls = get_analyzer(ScoreCategory.FINANCIAL_HEALTH)
    assert cls is FinancialHealthAnalyzer


# ---------------------------------------------------------------------------
# Strong / weak / missing
# ---------------------------------------------------------------------------


class TestScoring:
    def test_strong_profile_scores_high(self) -> None:
        sub = FinancialHealthAnalyzer().analyze(_security([_strong_period()]), _ctx())
        assert sub.category is ScoreCategory.FINANCIAL_HEALTH
        assert 0.0 <= sub.score <= 100.0
        assert sub.score >= 70.0, f"expected a high score, got {sub.score}"
        assert sub.confidence > 0.6
        assert sub.data_coverage > 0.6
        assert sub.evidence, "strong profile must cite evidence"
        assert "MISSING_DATA" not in sub.flags

    def test_weak_profile_scores_low(self) -> None:
        sub = FinancialHealthAnalyzer().analyze(
            _security([_weak_period(), _weak_prior_period()]), _ctx()
        )
        assert 0.0 <= sub.score <= 100.0
        assert sub.score <= 35.0, f"expected a low score, got {sub.score}"
        assert sub.evidence, "weak profile must cite evidence"
        # The fragility hazards must be surfaced.
        assert "NEGATIVE_EQUITY" in sub.flags
        assert "INTEREST_UNCOVERED" in sub.flags
        assert "CASH_RUNWAY_UNDER_12M" in sub.flags
        assert "SHARE_DILUTION" in sub.flags

    def test_strong_scores_strictly_higher_than_weak(self) -> None:
        strong = FinancialHealthAnalyzer().analyze(_security([_strong_period()]), _ctx())
        weak = FinancialHealthAnalyzer().analyze(
            _security([_weak_period(), _weak_prior_period()]), _ctx()
        )
        assert strong.score > weak.score + 30.0

    def test_all_missing_is_neutral_low_confidence(self) -> None:
        # Fundamentals present but every scorable line item is None.
        empty = FundamentalsPeriod(
            period_end=dt.date(2025, 12, 31), period_label="FY2025"
        )
        sub = FinancialHealthAnalyzer().analyze(_security([empty]), _ctx())
        assert sub.score == pytest.approx(50.0)
        assert sub.confidence <= 0.2
        assert sub.data_coverage == pytest.approx(0.0)
        assert "MISSING_DATA" in sub.flags

    def test_no_fundamentals_is_neutral_low_confidence(self) -> None:
        sub = FinancialHealthAnalyzer().analyze(_security([]), _ctx())
        assert sub.score == pytest.approx(50.0)
        assert sub.confidence <= 0.2
        assert "MISSING_DATA" in sub.flags


# ---------------------------------------------------------------------------
# Evidence quality & honesty
# ---------------------------------------------------------------------------


class TestEvidence:
    def test_evidence_directions_are_honest_for_strong(self) -> None:
        sub = FinancialHealthAnalyzer().analyze(_security([_strong_period()]), _ctx())
        # A strong company should produce at least one bullish evidence item and
        # no bearish ones from its (uniformly healthy) inputs.
        directions = {e.direction for e in sub.evidence}
        assert "bullish" in directions
        assert "bearish" not in directions

    def test_evidence_covers_each_lens_for_strong(self) -> None:
        sub = FinancialHealthAnalyzer().analyze(_security([_strong_period()]), _ctx())
        labels = " ".join(_evidence_labels(sub)).lower()
        # Leverage, liquidity, coverage, runway/FCF and distress all represented.
        assert "debt" in labels
        assert "ratio" in labels  # current/quick ratio.
        assert "interest coverage" in labels
        assert "cash flow" in labels or "cash runway" in labels
        assert "altman" in labels

    def test_every_evidence_value_is_a_string(self) -> None:
        sub = FinancialHealthAnalyzer().analyze(_security([_strong_period()]), _ctx())
        for e in sub.evidence:
            assert isinstance(e.value, str) and e.value
            assert e.source == "fundamentals"


# ---------------------------------------------------------------------------
# Peer/universe relativity & graceful degradation
# ---------------------------------------------------------------------------


class TestRelativeScoring:
    def _moderate_company(self) -> SecurityData:
        # A company with middling leverage (debt/equity = 1.0) and decent liquidity.
        fp = FundamentalsPeriod(
            period_end=dt.date(2025, 12, 31),
            period_label="FY2025",
            ebitda=20_000_000,
            operating_income=15_000_000,
            total_assets=200_000_000,
            total_debt=80_000_000,
            cash_and_equivalents=20_000_000,
            total_equity=80_000_000,
            free_cash_flow=10_000_000,
            current_ratio=1.5,
            quick_ratio=1.1,
            debt_to_equity=1.0,
            interest_coverage=5.0,
        )
        return _security([fp])

    def test_leverage_relative_to_lenient_vs_strict_peers(self) -> None:
        analyzer = FinancialHealthAnalyzer()
        data = self._moderate_company()

        # Against highly-levered peers, this company's 1.0x debt/equity is *low*
        # (safe) -> should rank well on leverage. Against pristine peers it is
        # *high* (risky) -> should rank worse. The overall scores must differ in
        # the expected direction.
        levered_peers = _ctx(
            peer_stats={"debt_to_equity": {"values": [3.0, 4.0, 5.0, 6.0, 8.0]}}
        )
        pristine_peers = _ctx(
            peer_stats={"debt_to_equity": {"values": [0.0, 0.1, 0.2, 0.3, 0.4]}}
        )
        s_lenient = analyzer.analyze(data, levered_peers)
        s_strict = analyzer.analyze(data, pristine_peers)
        assert s_lenient.score > s_strict.score

    def test_accepts_bare_sequence_distribution(self) -> None:
        analyzer = FinancialHealthAnalyzer()
        data = self._moderate_company()
        # peer_stats values may be a bare list, not a {"values": [...]} mapping.
        ctx = _ctx(peer_stats={"current_ratio": [0.5, 0.8, 1.0, 1.2, 1.4]})
        sub = analyzer.analyze(data, ctx)
        assert 0.0 <= sub.score <= 100.0
        # 1.5 current ratio sits at/above the top of that peer distribution.
        cr_ev = next(e for e in sub.evidence if "current ratio" in e.label.lower())
        assert "peer" in (cr_ev.detail or "").lower()

    def test_degrades_gracefully_without_context(self) -> None:
        analyzer = FinancialHealthAnalyzer()
        data = self._moderate_company()
        sub = analyzer.analyze(data, _ctx())  # no peer/universe stats.
        assert 0.0 <= sub.score <= 100.0
        assert sub.evidence
        # With no distribution, the detail must say absolute bands were used.
        de_ev = next(e for e in sub.evidence if e.label == "Debt / equity")
        assert "absolute" in (de_ev.detail or "").lower()


# ---------------------------------------------------------------------------
# Partial data, purity, determinism
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_partial_data_lowers_coverage_but_still_scores(self) -> None:
        # Only liquidity is available.
        fp = FundamentalsPeriod(
            period_end=dt.date(2025, 12, 31),
            period_label="FY2025",
            current_ratio=2.0,
            quick_ratio=1.5,
        )
        sub = FinancialHealthAnalyzer().analyze(_security([fp]), _ctx())
        assert 0.0 <= sub.score <= 100.0
        assert 0.0 < sub.data_coverage < 0.6  # only part of the picture.
        assert sub.confidence <= 0.55         # thin breadth caps confidence.
        assert sub.evidence

    def test_net_cash_company_is_not_penalised_on_leverage(self) -> None:
        fp = FundamentalsPeriod(
            period_end=dt.date(2025, 12, 31),
            period_label="FY2025",
            ebitda=30_000_000,
            total_debt=10_000_000,
            cash_and_equivalents=60_000_000,   # net cash.
            total_equity=200_000_000,
            total_assets=300_000_000,
            operating_income=25_000_000,
            debt_to_equity=0.05,
        )
        sub = FinancialHealthAnalyzer().analyze(_security([fp]), _ctx())
        nd_ev = next(
            e for e in sub.evidence if "net debt / ebitda" in e.label.lower()
        )
        assert nd_ev.direction == "bullish"

    def test_negative_equity_floors_leverage_and_flags(self) -> None:
        fp = FundamentalsPeriod(
            period_end=dt.date(2025, 12, 31),
            period_label="FY2025",
            total_debt=100_000_000,
            total_equity=-10_000_000,
            total_assets=90_000_000,
            debt_to_equity=-10.0,
        )
        sub = FinancialHealthAnalyzer().analyze(_security([fp]), _ctx())
        assert "NEGATIVE_EQUITY" in sub.flags

    def test_score_always_within_range_across_profiles(self) -> None:
        analyzer = FinancialHealthAnalyzer()
        for data in (
            _security([_strong_period()]),
            _security([_weak_period(), _weak_prior_period()]),
            _security([FundamentalsPeriod(period_end=dt.date(2025, 12, 31), period_label="FY2025")]),
            _security([]),
        ):
            sub = analyzer.analyze(data, _ctx())
            assert 0.0 <= sub.score <= 100.0
            assert 0.0 <= sub.confidence <= 1.0
            assert 0.0 <= sub.data_coverage <= 1.0

    def test_determinism(self) -> None:
        analyzer = FinancialHealthAnalyzer()
        data = _security([_strong_period()])
        a = analyzer.analyze(data, _ctx())
        b = analyzer.analyze(data, _ctx())
        assert a.score == b.score
        assert a.confidence == b.confidence
        assert a.data_coverage == b.data_coverage
        assert sorted(a.flags) == sorted(b.flags)

    def test_does_not_mutate_input(self) -> None:
        analyzer = FinancialHealthAnalyzer()
        data = _security([_strong_period()])
        before = data.model_dump()
        analyzer.analyze(data, _ctx())
        assert data.model_dump() == before


# ---------------------------------------------------------------------------
# Altman-Z behaviour
# ---------------------------------------------------------------------------


class TestAltmanZ:
    def test_distress_company_flags_altman_z(self) -> None:
        # Weak company sits firmly in the distress zone.
        sub = FinancialHealthAnalyzer().analyze(
            _security([_weak_period(), _weak_prior_period()]), _ctx()
        )
        assert "ALTMAN_Z_DISTRESS" in sub.flags
        z_ev = next(e for e in sub.evidence if "altman" in e.label.lower())
        assert z_ev.direction == "bearish"

    def test_safe_company_reports_safe_zone(self) -> None:
        sub = FinancialHealthAnalyzer().analyze(_security([_strong_period()]), _ctx())
        z_ev = next(e for e in sub.evidence if "altman" in e.label.lower())
        assert "ALTMAN_Z_DISTRESS" not in sub.flags
        assert z_ev.direction == "bullish"
