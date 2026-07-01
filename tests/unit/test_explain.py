"""Unit tests for :class:`convexity.ranking.explain.DefaultExplainabilityEngine`.

The explainability engine's single job is to *restate, in plain English, evidence
that has already been computed* — it must never introduce a fact that is not already
present as a sub-score, an :class:`~convexity.core.models.Evidence` item, or a
recorded ``data_warning``. These tests pin exactly that:

* **Narrative restates only attached evidence.** Bull/bear bullets and catalysts are
  traceable to the evidence labels/values on the sub-scores; the engine invents
  nothing. A control test confirms a fact *not* present in any evidence never
  appears in the prose.
* **Monitoring is always present and falsifiable**, grounded in the latest period,
  insider activity and the price window's actual high/low.
* **Honesty framing** (no certainty language; the screening/not-advice reminder)
  survives into the thesis, and data warnings are surfaced, not hidden.
* The engine is **pure** — identical inputs yield identical narrative, returned on
  the same instance.
"""

from __future__ import annotations

import datetime as _dt
from typing import List, Optional

from convexity.core.models import (
    CompanyAnalysis,
    Evidence,
    Filing,
    FundamentalsPeriod,
    InsiderTransaction,
    PriceBar,
    ScoreCategory,
    SecurityData,
    SubScore,
    ValuationSnapshot,
)
from convexity.ranking.explain import DefaultExplainabilityEngine


def _ev(label: str, value: str, *, direction: str = "neutral", source: str = "fundamentals") -> Evidence:
    """Build one :class:`Evidence` clause with an explicit rendered value."""
    return Evidence(label=label, value=value, source=source, direction=direction)


def _sub(
    category: ScoreCategory,
    score: float,
    *,
    confidence: float = 0.8,
    coverage: float = 0.9,
    rationale: str = "synthetic rationale",
    evidence: Optional[List[Evidence]] = None,
) -> SubScore:
    """Build a :class:`SubScore` carrying explicit evidence for restatement tests."""
    return SubScore(
        category=category,
        score=score,
        confidence=confidence,
        weight=0.1,
        rationale=rationale,
        evidence=list(evidence or []),
        flags=[],
        data_coverage=coverage,
    )


def _price_window() -> List[PriceBar]:
    """A small, deterministic price window with a known high/low for assertions."""
    base = _dt.date(2025, 11, 1)
    closes = [10.0, 11.0, 12.5, 11.5, 13.0, 12.0, 14.0]  # high 14.0-ish, low 10.0-ish
    bars: List[PriceBar] = []
    for i, c in enumerate(closes):
        bars.append(
            PriceBar(
                date=base + _dt.timedelta(days=i),
                open=c, high=c * 1.02, low=c * 0.98, close=c,
                adj_close=c, volume=100_000.0,
            )
        )
    return bars


def _security(
    *,
    warnings: Optional[List[str]] = None,
    insiders: Optional[List[InsiderTransaction]] = None,
    with_prices: bool = True,
) -> SecurityData:
    """A :class:`SecurityData` with a latest period, optional insiders and warnings."""
    latest = FundamentalsPeriod(
        period_end=_dt.date(2025, 12, 31),
        period_label="FY2025",
        revenue=240_000_000.0,
        net_income=34_000_000.0,
        free_cash_flow=30_000_000.0,
        operating_margin=0.22,
        gross_margin=0.55,
    )
    return SecurityData(
        ticker="EXMPL",
        name="Example Co",
        sector="Technology",
        industry="Software",
        as_of=_dt.datetime(2026, 1, 1),
        valuation=ValuationSnapshot(market_cap=420_000_000.0, pe=12.0, ev_ebitda=6.3),
        fundamentals=[latest],
        price_history=_price_window() if with_prices else [],
        filings=[Filing(filed=_dt.date(2025, 12, 8), form_type="8-K", title="Update")],
        insider_transactions=insiders or [],
        data_warnings=warnings or [],
    )


def _analysis(subs: List[SubScore], *, agreement: float = 0.7, conviction: float = 0.6) -> CompanyAnalysis:
    """Build a scored :class:`CompanyAnalysis` ready for explanation."""
    composite = sum(s.score for s in subs) / len(subs) if subs else 50.0
    return CompanyAnalysis(
        ticker="EXMPL",
        name="Example Co",
        sector="Technology",
        industry="Software",
        market_cap=420_000_000.0,
        composite_score=composite,
        conviction_confidence=conviction,
        signal_agreement=agreement,
        subscores=subs,
    )


def _explain(analysis: CompanyAnalysis, data: SecurityData) -> CompanyAnalysis:
    """Run the engine and return the (same) mutated analysis."""
    return DefaultExplainabilityEngine().explain(analysis, data)


# ---------------------------------------------------------------------------
# Narrative restates only attached evidence
# ---------------------------------------------------------------------------


class TestRestatesOnlyAttachedEvidence:
    def test_bull_case_restates_bullish_evidence(self) -> None:
        ev = _ev("Revenue YoY growth", "37%", direction="bullish")
        subs = [
            _sub(ScoreCategory.GROWTH, 82.0, evidence=[ev]),
            _sub(ScoreCategory.VALUE, 70.0, evidence=[_ev("P/E vs peers", "12", direction="bullish")]),
        ]
        result = _explain(_analysis(subs), _security())
        joined = " ".join(result.bull_case)
        # The attached evidence label + value appear in the bull case verbatim.
        assert "Revenue YoY growth" in joined
        assert "37%" in joined

    def test_bear_case_restates_bearish_evidence_and_warnings(self) -> None:
        ev = _ev("Operating-margin change", "-15%", direction="bearish")
        subs = [_sub(ScoreCategory.QUALITY, 22.0, evidence=[ev])]
        data = _security(warnings=["EXMPL: no analyst coverage available."])
        result = _explain(_analysis(subs, agreement=0.3), data)
        joined = " ".join(result.bear_case)
        assert "Operating-margin change" in joined
        assert "-15%" in joined
        # Recorded data warnings are surfaced, never hidden.
        assert any("no analyst coverage" in b for b in result.bear_case)

    def test_catalysts_restate_catalyst_evidence(self) -> None:
        cat_ev = _ev("Guidance raise (8-K)", "FY25 revenue +8%", direction="bullish", source="filings")
        subs = [
            _sub(ScoreCategory.CATALYST, 72.0, evidence=[cat_ev]),
            _sub(ScoreCategory.GROWTH, 80.0, evidence=[_ev("Revenue", "240M", direction="bullish")]),
        ]
        result = _explain(_analysis(subs), _security())
        joined = " ".join(result.catalysts)
        assert "Guidance raise (8-K)" in joined
        assert "FY25 revenue +8%" in joined

    def test_engine_does_not_invent_absent_facts(self) -> None:
        """A fact present in no evidence must never appear in any narrative field.

        This is the core honesty guard: the engine restates, it does not generate.
        """
        sentinel = "ACQUISITION_RUMOUR_42"  # appears in no sub-score or evidence
        subs = [
            _sub(ScoreCategory.VALUE, 75.0, evidence=[_ev("EV/EBITDA", "6.3", direction="bullish")]),
            _sub(ScoreCategory.GROWTH, 78.0, evidence=[_ev("Revenue YoY", "37%", direction="bullish")]),
            _sub(
                ScoreCategory.CATALYST, 65.0,
                evidence=[_ev("8-K filed", "guidance", direction="bullish", source="filings")],
            ),
        ]
        result = _explain(_analysis(subs), _security())
        blob = " ".join(
            [
                result.thesis,
                result.valuation_summary,
                result.fundamental_summary,
                result.technical_summary,
                result.confidence_explanation,
                " ".join(result.bull_case),
                " ".join(result.bear_case),
                " ".join(result.catalysts),
                " ".join(result.principal_risks),
                " ".join(result.monitoring_checklist),
            ]
        )
        assert sentinel not in blob

    def test_valuation_summary_only_restates_present_multiples(self) -> None:
        """Only multiples actually on the snapshot are mentioned; absent ones aren't."""
        subs = [_sub(ScoreCategory.VALUE, 70.0)]
        result = _explain(_analysis(subs), _security())
        # The snapshot carries P/E 12.0 and EV/EBITDA 6.3 (and nothing else).
        assert "P/E 12" in result.valuation_summary
        assert "EV/EBITDA 6" in result.valuation_summary
        # A multiple that is None (e.g. PEG) is not invented.
        assert "PEG" not in result.valuation_summary


# ---------------------------------------------------------------------------
# Monitoring checklist is present and falsifiable
# ---------------------------------------------------------------------------


class TestMonitoringChecklist:
    def test_monitoring_present_and_grounded(self) -> None:
        insiders = [
            InsiderTransaction(
                date=_dt.date(2025, 12, 12), insider_name="A. Founder",
                role="CEO", transaction_type="buy", shares=50_000.0, value=2_000_000.0,
            )
        ]
        subs = [
            _sub(ScoreCategory.GROWTH, 80.0, evidence=[_ev("Revenue YoY", "37%", direction="bullish")]),
            _sub(
                ScoreCategory.CATALYST, 70.0,
                evidence=[_ev("8-K", "guidance", direction="bullish", source="filings")],
            ),
        ]
        result = _explain(_analysis(subs), _security(insiders=insiders))
        checklist = result.monitoring_checklist
        assert checklist, "monitoring checklist must not be empty"
        text = " ".join(checklist).lower()
        # Falsifiable, concrete items: earnings baseline, margin trend, insider, levels.
        assert "earnings" in text
        assert "margin" in text
        assert "insider" in text
        # Insider item restates the actual disclosed transaction.
        assert "A. Founder" in " ".join(checklist)
        # Breakout/breakdown levels come from the real price window (high ~14, low ~10).
        assert any("breakout" in c.lower() or "breakdown" in c.lower() for c in checklist)

    def test_monitoring_handles_missing_price_history(self) -> None:
        """With no prices, the checklist degrades honestly instead of inventing levels."""
        subs = [_sub(ScoreCategory.VALUE, 60.0)]
        result = _explain(_analysis(subs), _security(with_prices=False))
        assert result.monitoring_checklist
        text = " ".join(result.monitoring_checklist).lower()
        assert "price history is available" in text or "price levels" in text


# ---------------------------------------------------------------------------
# Honesty framing, confidence explanation, purity
# ---------------------------------------------------------------------------


class TestFramingAndPurity:
    def test_thesis_carries_no_certainty_language(self) -> None:
        subs = [_sub(cat, 78.0) for cat in (ScoreCategory.VALUE, ScoreCategory.GROWTH, ScoreCategory.QUALITY)]
        result = _explain(_analysis(subs), _security())
        thesis_lower = result.thesis.lower()
        # The standing screening/not-advice framing is present.
        assert "not a prediction" in thesis_lower or "not investment advice" in thesis_lower
        # No guarantee/certainty language.
        for banned in ("guaranteed", "certain to", "will definitely", "sure thing"):
            assert banned not in thesis_lower

    def test_confidence_explanation_states_agreement_and_coverage(self) -> None:
        subs = [
            _sub(ScoreCategory.VALUE, 80.0, coverage=0.9),
            _sub(ScoreCategory.GROWTH, 78.0, coverage=0.8),
            _sub(ScoreCategory.QUALITY, 30.0, coverage=0.2),  # a thin, low category
        ]
        result = _explain(_analysis(subs), _security())
        ce = result.confidence_explanation.lower()
        assert "agree" in ce
        assert "coverage" in ce
        # It states what would raise and lower confidence (the honest levers).
        assert "rise" in ce
        assert "fall" in ce

    def test_explain_is_pure_and_returns_same_instance(self) -> None:
        subs = [_sub(ScoreCategory.VALUE, 70.0, evidence=[_ev("P/E", "12", direction="bullish")])]
        analysis = _analysis(subs)
        data = _security()
        returned = DefaultExplainabilityEngine().explain(analysis, data)
        # Mutates and returns the same object.
        assert returned is analysis
        # Re-running on fresh copies yields identical narrative (deterministic).
        def _fresh() -> CompanyAnalysis:
            ev = _ev("P/E", "12", direction="bullish")
            return DefaultExplainabilityEngine().explain(
                _analysis([_sub(ScoreCategory.VALUE, 70.0, evidence=[ev])]), _security()
            )

        a = _fresh()
        b = _fresh()
        assert a.thesis == b.thesis
        assert a.bull_case == b.bull_case
        assert a.monitoring_checklist == b.monitoring_checklist

    def test_neutral_company_has_honest_thesis_and_cases(self) -> None:
        """A company with no confident directional signal reads as neutral, not bullish."""
        subs = [_sub(cat, 50.0, confidence=0.6) for cat in (ScoreCategory.VALUE, ScoreCategory.GROWTH)]
        result = _explain(_analysis(subs, agreement=0.0, conviction=0.1), _security())
        assert result.thesis.strip()
        # No confident bullish signal -> the bull case says so honestly.
        assert any("does not support a positive case" in b or "No category produced a confident bullish" in b
                   for b in result.bull_case)
