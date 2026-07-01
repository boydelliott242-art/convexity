"""DETERMINISTIC explainability — turn a scored CompanyAnalysis into honest narrative.

Part of Convexity, an evidence-driven equity **research and screening** tool. This
module is **not** a predictor and **not** investment advice. Its single job is to
*restate, in plain English, evidence that has already been computed* by the
analyzers and attached to a :class:`~convexity.core.models.CompanyAnalysis`. It
**never** introduces a fact that is not already present as a
:class:`~convexity.core.models.SubScore`, an :class:`~convexity.core.models.Evidence`
item, or a recorded ``data_warning`` on the supplied
:class:`~convexity.core.models.SecurityData`.

Why a deterministic, templated engine?
--------------------------------------
A research tool's explanations must be reproducible and auditable: the same scored
analysis must always yield the same words, and every clause must be traceable to a
specific sub-score or evidence item the reader can inspect. No language model, no
randomness, no wall-clock — :meth:`DefaultExplainabilityEngine.explain` is a pure
function of its two inputs.

Honesty rules honoured here
---------------------------
* **Only restate attached evidence.** Bull/bear cases, catalysts and summaries are
  drawn from the sub-scores and their evidence; nothing is invented.
* **Conviction tracks agreement, not extremity.** The thesis and the confidence
  explanation foreground *how many independent categories agree* and the *data
  coverage* behind the read, because a single extreme category never justifies
  conviction on its own.
* **Missing data lowers confidence, visibly.** Data warnings and low-coverage
  categories are surfaced in the bear case / risks and the confidence explanation,
  never hidden.
* **No certainty language.** The narrative speaks of what the evidence *suggests*
  and *would raise or lower* confidence — never of guaranteed or certain outcomes.

The :class:`DefaultExplainabilityEngine` satisfies the
:class:`~convexity.core.contracts.ExplainabilityEngine` Protocol (one method,
``explain(analysis, data) -> CompanyAnalysis``) and returns the *same*
``CompanyAnalysis`` instance with its narrative fields populated in place.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from convexity.core.models import (
    CompanyAnalysis,
    Evidence,
    FundamentalsPeriod,
    PriceBar,
    ScoreCategory,
    SecurityData,
    SubScore,
)

# ---------------------------------------------------------------------------
# Tunable, transparent thresholds (named once so the narrative stays auditable)
# ---------------------------------------------------------------------------
#
# These are presentation thresholds only — they decide which already-computed
# signals are loud enough to mention, never what the underlying scores are.

# A sub-score at or above this is treated as a genuine positive (bullish) signal.
_STRONG_SCORE = 60.0
# A sub-score at or below this is treated as a genuine negative (bearish) signal.
_WEAK_SCORE = 40.0
# Only count a category toward "agreement" when its confidence clears this floor,
# so a near-blind guess does not inflate apparent conviction.
_AGREE_CONF_FLOOR = 0.25
# Coverage below this is called out as a thin-data category in the risks list.
_THIN_COVERAGE = 0.34
# How many evidence bullets (at most) to surface per case, to keep cases readable.
_MAX_BULLETS = 4

# Human-readable category labels for prose.
_CATEGORY_LABEL: Dict[ScoreCategory, str] = {
    ScoreCategory.VALUE: "valuation",
    ScoreCategory.GROWTH: "growth",
    ScoreCategory.QUALITY: "business quality",
    ScoreCategory.FINANCIAL_HEALTH: "financial health",
    ScoreCategory.TECHNICAL: "price technicals",
    ScoreCategory.MOMENTUM: "momentum",
    ScoreCategory.CATALYST: "catalysts",
    ScoreCategory.RISK: "risk",
    ScoreCategory.MANAGEMENT: "management",
    ScoreCategory.COMPETITIVE: "competitive position",
    ScoreCategory.OWNERSHIP: "ownership / insider activity",
    ScoreCategory.HISTORICAL_ANALOG: "historical analogs",
}


def _label(category: ScoreCategory) -> str:
    """Return a human prose label for ``category`` (falls back to its value)."""
    return _CATEGORY_LABEL.get(category, category.value.replace("_", " "))


def _evidence_phrase(ev: Evidence) -> str:
    """Render one :class:`Evidence` item as a compact, self-contained clause.

    Combines the label, value and (when present) the ``detail`` into a single
    human-readable phrase. Purely a restatement of the attached evidence — no new
    facts are introduced.
    """
    base = f"{ev.label}: {ev.value}"
    if ev.detail:
        return f"{base} ({ev.detail})"
    return base


def _directional_evidence(
    sub: SubScore,
    direction: str,
    *,
    limit: int = _MAX_BULLETS,
) -> List[str]:
    """Return up to ``limit`` evidence phrases from ``sub`` matching ``direction``.

    ``direction`` is one of ``"bullish"`` / ``"bearish"`` / ``"neutral"``. Only
    evidence already attached to the sub-score is used; ordering is preserved so the
    analyzer's own priority is respected.
    """
    out: List[str] = []
    for ev in sub.evidence:
        if ev.direction == direction:
            out.append(_evidence_phrase(ev))
            if len(out) >= limit:
                break
    return out


def _agreeing_categories(
    subscores: List[SubScore],
) -> Tuple[List[SubScore], List[SubScore]]:
    """Split confident, non-RISK sub-scores into bullish and bearish groups.

    A category "agrees" on the bullish side when its score clears
    :data:`_STRONG_SCORE` and on the bearish side when it falls to
    :data:`_WEAK_SCORE` or below — in both cases only if its confidence clears
    :data:`_AGREE_CONF_FLOOR`, so low-confidence guesses do not inflate apparent
    agreement. RISK is excluded here because, by contract, it is a dampener whose
    polarity (higher == safer) is handled separately, not a directional thesis
    signal. Each returned list is sorted by descending and ascending score
    respectively so the strongest signals lead.
    """
    bullish: List[SubScore] = []
    bearish: List[SubScore] = []
    for sub in subscores:
        if sub.category == ScoreCategory.RISK:
            continue
        if sub.confidence < _AGREE_CONF_FLOOR:
            continue
        if sub.score >= _STRONG_SCORE:
            bullish.append(sub)
        elif sub.score <= _WEAK_SCORE:
            bearish.append(sub)
    bullish.sort(key=lambda s: s.score, reverse=True)
    bearish.sort(key=lambda s: s.score)
    return bullish, bearish


def _mean_data_coverage(subscores: List[SubScore]) -> float:
    """Return the mean ``data_coverage`` across ``subscores`` (0.0 when empty)."""
    if not subscores:
        return 0.0
    return sum(s.data_coverage for s in subscores) / len(subscores)


def _format_money(value: Optional[float]) -> Optional[str]:
    """Format a dollar figure compactly (``$1.2B`` / ``$340.0M`` / ``$0.9M``).

    Returns ``None`` when ``value`` is ``None`` so callers can omit the clause
    rather than print a fabricated figure.
    """
    if value is None:
        return None
    abs_v = abs(value)
    if abs_v >= 1e9:
        return f"${value / 1e9:.1f}B"
    if abs_v >= 1e6:
        return f"${value / 1e6:.1f}M"
    return f"${value:,.0f}"


def _dedupe_preserving_order(items: List[str]) -> List[str]:
    """Return ``items`` with duplicates removed, preserving first-seen order."""
    seen: Set[str] = set()
    out: List[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


class DefaultExplainabilityEngine:
    """Deterministic, template-driven implementation of the explainability contract.

    Satisfies the :class:`~convexity.core.contracts.ExplainabilityEngine` Protocol.
    Given a :class:`CompanyAnalysis` whose ``subscores`` (with their attached
    :class:`Evidence`) and ``composite_score`` / ``signal_agreement`` /
    ``conviction_confidence`` are already populated, plus the originating
    :class:`SecurityData`, it fills the narrative fields — ``thesis``,
    ``bull_case``, ``bear_case``, ``catalysts``, ``principal_risks``,
    ``valuation_summary``, ``fundamental_summary``, ``technical_summary``,
    ``confidence_explanation`` and ``monitoring_checklist`` — by *restating only
    evidence already present*. It mutates and returns the same instance.

    The engine is pure: identical inputs always yield identical narrative, with no
    I/O, randomness or wall-clock access.
    """

    def explain(self, analysis: CompanyAnalysis, data: SecurityData) -> CompanyAnalysis:
        """Populate ``analysis``'s narrative from its sub-scores and ``data``.

        Args:
            analysis: A scored :class:`CompanyAnalysis` with sub-scores + evidence
                already attached (composite, agreement and confidence set by the
                ranking engine).
            data: The :class:`SecurityData` the sub-scores were computed from, used
                only to restate already-present figures (valuation multiples, the
                latest fundamentals, recent price levels) and ``data_warnings``.

        Returns:
            The same :class:`CompanyAnalysis` instance with every narrative field
            populated. No new facts are introduced; everything restates attached
            evidence, sub-scores or recorded data warnings.
        """
        subscores = list(analysis.subscores)
        bullish, bearish = _agreeing_categories(subscores)
        risk_sub = analysis.subscore_by_category(ScoreCategory.RISK)

        analysis.thesis = self._build_thesis(analysis, bullish, bearish)
        analysis.bull_case = self._build_bull_case(bullish)
        analysis.bear_case = self._build_bear_case(bearish, risk_sub, data)
        analysis.catalysts = self._build_catalysts(analysis)
        analysis.principal_risks = self._build_principal_risks(
            bearish, risk_sub, subscores, data
        )
        analysis.valuation_summary = self._build_valuation_summary(analysis, data)
        analysis.fundamental_summary = self._build_fundamental_summary(analysis, data)
        analysis.technical_summary = self._build_technical_summary(analysis, data)
        analysis.confidence_explanation = self._build_confidence_explanation(
            analysis, bullish, bearish, subscores
        )
        analysis.monitoring_checklist = self._build_monitoring_checklist(
            analysis, data
        )
        return analysis

    # ------------------------------------------------------------------ #
    # Thesis                                                              #
    # ------------------------------------------------------------------ #

    def _build_thesis(
        self,
        analysis: CompanyAnalysis,
        bullish: List[SubScore],
        bearish: List[SubScore],
    ) -> str:
        """Synthesise a 2–4 sentence thesis from the *agreeing* signals.

        Leads with the count of independent categories that agree (because
        conviction is justified by agreement, not by any single extreme), names the
        strongest agreeing categories, balances them against the strongest
        dissenting categories, and closes with the honest-framing reminder that this
        is screening evidence, not a recommendation.
        """
        name = analysis.name or analysis.ticker
        comp = analysis.composite_score
        n_bull = len(bullish)
        n_bear = len(bearish)

        if n_bull >= n_bear and n_bull > 0:
            lead_cats = ", ".join(_label(s.category) for s in bullish[:3])
            stance = (
                f"{name} ({analysis.ticker}) screens constructively, with "
                f"{n_bull} independent categor{'y' if n_bull == 1 else 'ies'} "
                f"pointing favourably — led by {lead_cats}."
            )
        elif n_bear > 0:
            lead_cats = ", ".join(_label(s.category) for s in bearish[:3])
            stance = (
                f"{name} ({analysis.ticker}) screens cautiously, with "
                f"{n_bear} independent categor{'y' if n_bear == 1 else 'ies'} "
                f"flagging concerns — led by {lead_cats}."
            )
        else:
            stance = (
                f"{name} ({analysis.ticker}) screens around neutral: no category "
                "produced a confident directional signal."
            )

        composite_sentence = (
            f"The weighted composite score is {comp:.0f}/100, and "
            f"{n_bull} of the scored categories agree on the bullish side versus "
            f"{n_bear} on the bearish side — conviction here rests on that breadth "
            "of agreement, not on any single category."
        )

        if bearish and (n_bull >= n_bear):
            counter = (
                f" The main offsetting concern is {_label(bearish[0].category)} "
                f"(score {bearish[0].score:.0f}/100)."
            )
        elif bullish and (n_bear > n_bull):
            counter = (
                f" The main offsetting positive is {_label(bullish[0].category)} "
                f"(score {bullish[0].score:.0f}/100)."
            )
        else:
            counter = ""

        framing = (
            " This is a research/screening read of independent evidence, not a "
            "prediction or investment advice."
        )
        return stance + " " + composite_sentence + counter + framing

    # ------------------------------------------------------------------ #
    # Bull case                                                          #
    # ------------------------------------------------------------------ #

    def _build_bull_case(self, bullish: List[SubScore]) -> List[str]:
        """Build the bull case from the highest sub-scores and their evidence.

        For each agreeing-bullish category (strongest first) it emits a header
        bullet naming the category, its score and confidence, followed by the
        category's own bullish evidence phrases. Only attached evidence is restated.
        """
        bullets: List[str] = []
        for sub in bullish:
            header = (
                f"{_label(sub.category).capitalize()} scores "
                f"{sub.score:.0f}/100 (confidence {sub.confidence:.0%})"
            )
            ev_phrases = _directional_evidence(sub, "bullish")
            if ev_phrases:
                header += ": " + "; ".join(ev_phrases)
            elif sub.rationale:
                header += f": {sub.rationale}"
            bullets.append(header)
        if not bullets:
            bullets.append(
                "No category produced a confident bullish signal; the evidence "
                "does not support a positive case at this time."
            )
        return _dedupe_preserving_order(bullets)

    # ------------------------------------------------------------------ #
    # Bear case                                                          #
    # ------------------------------------------------------------------ #

    def _build_bear_case(
        self,
        bearish: List[SubScore],
        risk_sub: Optional[SubScore],
        data: SecurityData,
    ) -> List[str]:
        """Build the bear case from the lowest sub-scores, RISK and data warnings.

        Restates the weakest agreeing categories with their bearish evidence, adds a
        RISK-category line when RISK screens unsafe (low score == higher risk by
        contract), and surfaces recorded ``data_warnings`` so data gaps are visible
        rather than hidden.
        """
        bullets: List[str] = []
        for sub in bearish:
            header = (
                f"{_label(sub.category).capitalize()} scores "
                f"{sub.score:.0f}/100 (confidence {sub.confidence:.0%})"
            )
            ev_phrases = _directional_evidence(sub, "bearish")
            if ev_phrases:
                header += ": " + "; ".join(ev_phrases)
            elif sub.rationale:
                header += f": {sub.rationale}"
            bullets.append(header)

        # RISK: by contract a *higher* score is safer, so a low RISK score is a
        # bear-case item. Surface its bearish evidence too.
        if risk_sub is not None and risk_sub.score <= _WEAK_SCORE:
            risk_phrases = _directional_evidence(risk_sub, "bearish")
            risk_line = (
                f"Elevated risk: RISK screens {risk_sub.score:.0f}/100 "
                "(higher is safer)"
            )
            if risk_phrases:
                risk_line += ": " + "; ".join(risk_phrases)
            elif risk_sub.rationale:
                risk_line += f": {risk_sub.rationale}"
            bullets.append(risk_line)

        for warning in data.data_warnings[:_MAX_BULLETS]:
            bullets.append(f"Data gap: {warning}")

        if not bullets:
            bullets.append(
                "No category produced a confident bearish signal and no data "
                "warnings were recorded; no material bear case is evident from the "
                "attached evidence."
            )
        return _dedupe_preserving_order(bullets)

    # ------------------------------------------------------------------ #
    # Catalysts                                                          #
    # ------------------------------------------------------------------ #

    def _build_catalysts(self, analysis: CompanyAnalysis) -> List[str]:
        """Build the catalyst list from the CATALYST sub-score's evidence.

        Restates every piece of evidence attached to the CATALYST category (any
        direction — a catalyst can cut either way) so the reader sees the concrete,
        already-identified events. Falls back to the category rationale, then to an
        honest "none identified" note.
        """
        catalyst_sub = analysis.subscore_by_category(ScoreCategory.CATALYST)
        bullets: List[str] = []
        if catalyst_sub is not None:
            for ev in catalyst_sub.evidence:
                bullets.append(_evidence_phrase(ev))
            if not bullets and catalyst_sub.rationale:
                bullets.append(catalyst_sub.rationale)
        if not bullets:
            bullets.append(
                "No specific catalysts were identified in the available evidence."
            )
        return _dedupe_preserving_order(bullets)[: _MAX_BULLETS * 2]

    # ------------------------------------------------------------------ #
    # Principal risks                                                    #
    # ------------------------------------------------------------------ #

    def _build_principal_risks(
        self,
        bearish: List[SubScore],
        risk_sub: Optional[SubScore],
        subscores: List[SubScore],
        data: SecurityData,
    ) -> List[str]:
        """Assemble principal risks from RISK evidence, weak categories, thin data.

        Draws on (1) the RISK category's evidence, (2) the lowest-scoring agreeing
        categories, (3) categories scored on thin data coverage (a credibility
        risk), and (4) recorded ``data_warnings``. Everything restates attached
        evidence or sub-score metadata; nothing is invented.
        """
        bullets: List[str] = []

        if risk_sub is not None:
            for ev in risk_sub.evidence:
                if ev.direction == "bearish":
                    bullets.append(_evidence_phrase(ev))
            if not bullets and risk_sub.score <= _WEAK_SCORE and risk_sub.rationale:
                bullets.append(risk_sub.rationale)

        for sub in bearish[:_MAX_BULLETS]:
            bullets.append(
                f"Weak {_label(sub.category)} (score {sub.score:.0f}/100)"
            )

        thin = [
            s
            for s in subscores
            if s.data_coverage < _THIN_COVERAGE and s.category != ScoreCategory.RISK
        ]
        if thin:
            names = ", ".join(_label(s.category) for s in thin[:_MAX_BULLETS])
            bullets.append(
                f"Thin data coverage in {names} — these scores carry less weight "
                "and could shift materially as more data arrives."
            )

        for warning in data.data_warnings[:_MAX_BULLETS]:
            bullets.append(f"Data limitation: {warning}")

        if not bullets:
            bullets.append(
                "No principal risks were flagged by the risk analyzer or surfaced "
                "as data warnings; absence of flagged risk is not absence of risk."
            )
        return _dedupe_preserving_order(bullets)

    # ------------------------------------------------------------------ #
    # Valuation summary                                                  #
    # ------------------------------------------------------------------ #

    def _build_valuation_summary(
        self, analysis: CompanyAnalysis, data: SecurityData
    ) -> str:
        """Summarise valuation from the VALUE sub-score and the valuation snapshot.

        States the VALUE score and confidence, then restates whichever valuation
        multiples are actually present on the snapshot (omitting any that are
        ``None`` so no figure is fabricated).
        """
        value_sub = analysis.subscore_by_category(ScoreCategory.VALUE)
        v = data.valuation

        multiple_bits: List[str] = []
        for attr, lbl in (
            ("pe", "P/E"),
            ("ev_ebitda", "EV/EBITDA"),
            ("p_fcf", "P/FCF"),
            ("ev_sales", "EV/Sales"),
            ("p_b", "P/B"),
            ("peg", "PEG"),
        ):
            raw = getattr(v, attr, None)
            if raw is not None:
                multiple_bits.append(f"{lbl} {raw:,.1f}")

        mc = _format_money(data.market_cap)
        cap_phrase = f"Market cap {mc}. " if mc else ""

        if value_sub is None:
            base = f"{cap_phrase}Valuation was not scored for this security."
        else:
            base = (
                f"{cap_phrase}VALUE screens {value_sub.score:.0f}/100 "
                f"(confidence {value_sub.confidence:.0%}). {value_sub.rationale}"
            )
        if multiple_bits:
            base += " Reported multiples: " + ", ".join(multiple_bits) + "."
        else:
            base += " No valuation multiples were available from the data sources."
        return base.strip()

    # ------------------------------------------------------------------ #
    # Fundamental summary                                                #
    # ------------------------------------------------------------------ #

    def _build_fundamental_summary(
        self, analysis: CompanyAnalysis, data: SecurityData
    ) -> str:
        """Summarise fundamentals from GROWTH/QUALITY/HEALTH scores + latest period.

        Restates the headline figures from the most recent
        :class:`FundamentalsPeriod` that are actually present (revenue, net income,
        free cash flow, margins) and the scores of the fundamental-facing
        categories. Omits any line item that is ``None``.
        """
        latest: Optional[FundamentalsPeriod] = data.latest_fundamentals
        parts: List[str] = []

        cat_bits: List[str] = []
        for cat in (
            ScoreCategory.GROWTH,
            ScoreCategory.QUALITY,
            ScoreCategory.FINANCIAL_HEALTH,
        ):
            sub = analysis.subscore_by_category(cat)
            if sub is not None:
                cat_bits.append(f"{_label(cat)} {sub.score:.0f}/100")
        if cat_bits:
            parts.append("Category scores — " + ", ".join(cat_bits) + ".")

        if latest is not None:
            line_bits: List[str] = []
            rev = _format_money(latest.revenue)
            if rev:
                line_bits.append(f"revenue {rev}")
            ni = _format_money(latest.net_income)
            if ni:
                line_bits.append(f"net income {ni}")
            fcf = _format_money(latest.free_cash_flow)
            if fcf:
                line_bits.append(f"free cash flow {fcf}")
            if latest.operating_margin is not None:
                line_bits.append(f"operating margin {latest.operating_margin * 100:.1f}%")
            if latest.gross_margin is not None:
                line_bits.append(f"gross margin {latest.gross_margin * 100:.1f}%")
            if line_bits:
                parts.append(
                    f"Most recent period ({latest.period_label}): "
                    + ", ".join(line_bits)
                    + "."
                )
            else:
                parts.append(
                    f"Most recent period ({latest.period_label}) carried no "
                    "populated headline line items."
                )
        else:
            parts.append("No fundamentals periods were available from the data sources.")

        return " ".join(parts).strip()

    # ------------------------------------------------------------------ #
    # Technical summary                                                  #
    # ------------------------------------------------------------------ #

    def _build_technical_summary(
        self, analysis: CompanyAnalysis, data: SecurityData
    ) -> str:
        """Summarise technicals from the TECHNICAL/MOMENTUM scores + price history.

        Restates the TECHNICAL and MOMENTUM scores and, when price history is
        present, the latest close and the high/low range of the available window —
        all already-present figures, never fabricated levels.
        """
        parts: List[str] = []
        cat_bits: List[str] = []
        for cat in (ScoreCategory.TECHNICAL, ScoreCategory.MOMENTUM):
            sub = analysis.subscore_by_category(cat)
            if sub is not None:
                cat_bits.append(
                    f"{_label(cat)} {sub.score:.0f}/100 "
                    f"(confidence {sub.confidence:.0%})"
                )
        if cat_bits:
            parts.append("Category scores — " + ", ".join(cat_bits) + ".")

        history: List[PriceBar] = data.price_history
        if history:
            last = history[-1]
            highs = [b.high for b in history]
            lows = [b.low for b in history]
            window_hi = max(highs)
            window_lo = min(lows)
            parts.append(
                f"Latest close {last.close:,.2f} on {last.date.isoformat()}; "
                f"the available {len(history)}-bar window ranged "
                f"{window_lo:,.2f}–{window_hi:,.2f}."
            )
            # Position within the window is a plain restatement of the levels above.
            span = window_hi - window_lo
            if span > 0:
                pos = (last.close - window_lo) / span
                parts.append(
                    f"The last close sits at the {pos * 100:.0f}% point of that range "
                    "(0% = window low, 100% = window high)."
                )
        else:
            parts.append("No price history was available to assess technicals.")
        return " ".join(parts).strip()

    # ------------------------------------------------------------------ #
    # Confidence explanation                                             #
    # ------------------------------------------------------------------ #

    def _build_confidence_explanation(
        self,
        analysis: CompanyAnalysis,
        bullish: List[SubScore],
        bearish: List[SubScore],
        subscores: List[SubScore],
    ) -> str:
        """Explain conviction: agreement count, coverage %, and what would move it.

        Per the assignment this MUST state how many independent categories agreed,
        the data-coverage percentage, and what would raise or lower confidence. All
        figures restate already-computed metadata (sub-score confidences/coverage
        and the analysis's ``signal_agreement`` / ``conviction_confidence``).
        """
        n_bull = len(bullish)
        n_bear = len(bearish)
        scored = [s for s in subscores if s.category != ScoreCategory.RISK]
        n_scored = len(scored)
        coverage_pct = _mean_data_coverage(subscores) * 100.0

        majority = bullish if n_bull >= n_bear else bearish
        majority_dir = "bullish" if n_bull >= n_bear else "bearish"
        n_agree = len(majority)

        agreement_sentence = (
            f"Of {n_scored} scored non-risk categories, {n_agree} agree on the "
            f"{majority_dir} side ({n_bull} bullish, {n_bear} bearish); "
            f"signal-agreement is {analysis.signal_agreement:.0%} and overall "
            f"conviction confidence is {analysis.conviction_confidence:.0%}. "
            "Conviction is intentionally driven by how many INDEPENDENT categories "
            "agree, not by any single extreme score."
        )

        coverage_sentence = (
            f"Mean data coverage across categories is {coverage_pct:.0f}% — the "
            "fraction of required inputs that were actually present; missing data "
            "lowers, never inflates, the score."
        )

        # Concrete, evidence-grounded levers.
        thin = [
            s
            for s in subscores
            if s.data_coverage < _THIN_COVERAGE and s.category != ScoreCategory.RISK
        ]
        low_conf = [
            s
            for s in scored
            if s.confidence < _AGREE_CONF_FLOOR
        ]
        raise_bits: List[str] = []
        if thin:
            raise_bits.append(
                "filling the thin-coverage categories ("
                + ", ".join(_label(s.category) for s in thin[:3])
                + ")"
            )
        if low_conf:
            raise_bits.append(
                "raising confidence in "
                + ", ".join(_label(s.category) for s in low_conf[:3])
            )
        raise_bits.append("more independent categories moving into agreement")
        raise_sentence = (
            "Confidence would RISE if: " + "; ".join(raise_bits) + "."
        )

        lower_sentence = (
            "Confidence would FALL if: categories began to disagree, fresh data "
            "warnings appeared, or already-strong categories weakened toward "
            "neutral."
        )

        return " ".join(
            [agreement_sentence, coverage_sentence, raise_sentence, lower_sentence]
        )

    # ------------------------------------------------------------------ #
    # Monitoring checklist                                               #
    # ------------------------------------------------------------------ #

    def _build_monitoring_checklist(
        self, analysis: CompanyAnalysis, data: SecurityData
    ) -> List[str]:
        """Build concrete, FALSIFIABLE monitoring items the reader can track.

        Each item names a specific, checkable observation — the next earnings print
        vs the latest period, the margin trend, insider activity, and key price
        breakout/breakdown levels drawn from the available window. Every level or
        figure restates already-present data; nothing is invented.
        """
        items: List[str] = []
        latest: Optional[FundamentalsPeriod] = data.latest_fundamentals

        # 1. Next earnings — falsifiable against the latest reported period.
        if latest is not None:
            rev = _format_money(latest.revenue)
            ni = _format_money(latest.net_income)
            ref_bits: List[str] = []
            if rev:
                ref_bits.append(f"revenue {rev}")
            if ni:
                ref_bits.append(f"net income {ni}")
            ref = (
                f" (vs {latest.period_label}: " + ", ".join(ref_bits) + ")"
                if ref_bits
                else f" (vs {latest.period_label})"
            )
            items.append(
                f"Next earnings report: confirm revenue and earnings do not "
                f"deteriorate{ref}."
            )
        else:
            items.append(
                "Next earnings report: establish a baseline — no prior fundamentals "
                "period was available."
            )

        # 2. Margin trend — falsifiable direction.
        if latest is not None and latest.operating_margin is not None:
            items.append(
                f"Operating-margin trend: watch whether it holds above the latest "
                f"{latest.operating_margin * 100:.1f}%; sustained compression would "
                "weaken the quality and value reads."
            )
        else:
            items.append(
                "Operating-margin trend: begin tracking once margin data is "
                "reported (not currently available)."
            )

        # 3. Insider activity — falsifiable event.
        if data.insider_transactions:
            recent = data.insider_transactions[0]
            items.append(
                f"Insider activity: latest disclosed transaction was a "
                f"{recent.transaction_type} by {recent.insider_name} on "
                f"{recent.date.isoformat()}; watch for further open-market buys "
                "(supportive) or cluster selling (a warning)."
            )
        else:
            items.append(
                "Insider activity: watch upcoming Form 4 filings for open-market "
                "buying or selling (none on record in the available data)."
            )

        # 4 & 5. Key breakout / breakdown price levels from the available window.
        history: List[PriceBar] = data.price_history
        if history:
            window_hi = max(b.high for b in history)
            window_lo = min(b.low for b in history)
            last_close = history[-1].close
            items.append(
                f"Breakout level: a sustained close above {window_hi:,.2f} (the "
                f"{len(history)}-bar window high) would confirm upside technicals; "
                f"last close was {last_close:,.2f}."
            )
            items.append(
                f"Breakdown level: a sustained close below {window_lo:,.2f} (the "
                f"{len(history)}-bar window low) would invalidate the technical "
                "read and warrant reassessment."
            )
        else:
            items.append(
                "Key price levels: establish breakout/breakdown levels once price "
                "history is available (none currently)."
            )

        # 6. Catalyst follow-through, when catalysts were identified.
        catalyst_sub = analysis.subscore_by_category(ScoreCategory.CATALYST)
        if catalyst_sub is not None and catalyst_sub.evidence:
            first = catalyst_sub.evidence[0]
            items.append(
                f"Catalyst follow-through: track resolution of '{first.label}: "
                f"{first.value}' and whether it plays out as the evidence suggests."
            )

        return _dedupe_preserving_order(items)


__all__ = ["DefaultExplainabilityEngine"]
