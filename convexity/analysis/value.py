"""VALUE analyzer — how cheap is the security, and is the cheapness *earned*?

Part of Convexity, an evidence-driven equity **research and screening** tool. This
module is **not** a predictor and **not** investment advice. It produces one
independent, fully auditable piece of evidence — a 0–100 VALUE sub-score — that the
ranking layer aggregates alongside many *other* independent signals. A cheap
valuation on its own proves nothing; conviction is justified only when this signal
agrees with the growth, quality, financial-health and other categories.

What "value" means here
-----------------------
Cheapness is measured across up to six valuation multiples — ``ev_ebitda``,
``ev_sales``, ``p_fcf``, ``p_b``, ``pe`` and ``peg`` — each scored *relative to the
security's sector peers and/or the screened universe* when a comparison
distribution is supplied (:attr:`AnalysisContext.peer_stats` /
``universe_stats``). Relative scoring matters for micro-caps: a P/E of 12 is dear
for a no-growth utility and cheap for a compounder, and there is no universal
threshold. When no comparison distribution is available the analyzer degrades
gracefully to transparent absolute bands (clearly labelled as such) rather than
guessing.

Honesty rules honoured here
---------------------------
* **Never fabricate.** A missing multiple contributes nothing — it is not imputed.
  Its absence lowers ``data_coverage`` and ``confidence`` instead.
* **A cheap multiple only counts when it is backed by real economics.** Negative or
  meaningless multiples (e.g. a P/E on a loss, an EV/EBITDA on negative EBITDA, a
  PEG with no growth) are *not* read as "infinitely cheap"; they are excluded from
  the cheapness blend and flagged, because a low headline ratio there is an
  artefact, not value.
* **Value-trap penalty.** A statistically cheap stock whose fundamentals are
  *deteriorating* (falling revenue, shrinking margins, negative free cash flow,
  losses) has its score pulled back toward neutral and is flagged
  ``VALUE_TRAP_RISK`` — cheapness without improving economics is often a trap, and
  saying so is more honest than rewarding the low multiple blindly.
* Every point of the score traces to an :class:`Evidence` item naming the metric,
  its value, its peer/universe percentile (where used) and an honest ``direction``.

The :meth:`ValueAnalyzer.analyze` method is pure: it reads only the passed
:class:`SecurityData` and :class:`AnalysisContext` and performs no I/O, clock or
random access, so the sub-score is reproducible and auditable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Set, Tuple

from convexity.core.contracts import AnalysisContext, Analyzer
from convexity.core.models import (
    Evidence,
    FundamentalsPeriod,
    ScoreCategory,
    SecurityData,
    SubScore,
)
from convexity.core.registry import register_analyzer
from convexity.core.scoring import clamp, percentile_rank, scale_to_score

# ---------------------------------------------------------------------------
# Multiple specifications
# ---------------------------------------------------------------------------
#
# Each valuation multiple the analyzer understands is described once here so the
# scoring loop stays declarative and auditable. For every multiple we record:
#
# * ``attr``        — the field on ``ValuationSnapshot`` holding the raw value.
# * ``label``       — a human label used in rationale + evidence.
# * ``weight``      — how much this multiple counts in the cheapness blend. Cash-flow
#                     and earnings multiples (EV/EBITDA, P/FCF, P/E) are weighted
#                     above sales/book multiples because they rest on profit the
#                     business actually earns rather than on the top line or
#                     accounting equity.
# * ``abs_lo``/``abs_hi`` — the absolute band used as a *fallback* when no peer or
#                     universe distribution is available. ``abs_lo`` is the cheap end
#                     (maps toward 100) and ``abs_hi`` the expensive end (toward 0).
#                     These are deliberately wide, generic bands, clearly flagged as
#                     absolute when used so a reader knows no relative context backed
#                     the score.
#
# For *all* of these multiples, lower is cheaper, i.e. more attractive — so when we
# convert a percentile rank (fraction of peers at-or-below the value) into a score
# we invert it: being *below* most peers (a low percentile) is the attractive case.

_MultipleSpec = Dict[str, Any]

_MULTIPLES: List[_MultipleSpec] = [
    {"attr": "ev_ebitda", "label": "EV/EBITDA", "weight": 1.3, "abs_lo": 4.0, "abs_hi": 18.0},
    {"attr": "p_fcf", "label": "P/FCF", "weight": 1.3, "abs_lo": 6.0, "abs_hi": 30.0},
    {"attr": "pe", "label": "P/E", "weight": 1.1, "abs_lo": 6.0, "abs_hi": 28.0},
    {"attr": "ev_sales", "label": "EV/Sales", "weight": 0.9, "abs_lo": 0.5, "abs_hi": 6.0},
    {"attr": "p_b", "label": "P/B", "weight": 0.8, "abs_lo": 0.6, "abs_hi": 5.0},
    {"attr": "peg", "label": "PEG", "weight": 1.0, "abs_lo": 0.5, "abs_hi": 2.5},
]

# A multiple is only treated as meaningful (i.e. a low value really means "cheap")
# when it is strictly positive. A negative or zero earnings/EBITDA/FCF multiple is
# an artefact of negative economics, not a bargain, so it is excluded and flagged.
_MIN_MEANINGFUL = 1e-9


def _peer_or_universe_distribution(
    attr: str,
    ctx: AnalysisContext,
) -> Tuple[Optional[List[float]], Optional[str]]:
    """Return ``(distribution, scope_label)`` for ``attr`` from the context.

    Prefers peer statistics (the tightest, most comparable cohort) and falls back
    to the wider screened-universe statistics. The distribution may be supplied as
    a plain sequence of numbers (``{"ev_ebitda": [6.1, 8.0, 12.4]}``) or as a dict
    carrying a ``"values"``/``"distribution"`` sequence; both shapes are tolerated
    so this analyzer composes with whatever the pipeline assembles. Returns
    ``(None, None)`` when no usable distribution exists.
    """
    for stats, scope in ((ctx.peer_stats, "peers"), (ctx.universe_stats, "universe")):
        if not stats:
            continue
        raw = stats.get(attr)
        values = _coerce_distribution(raw)
        if values:
            return values, scope
    return None, None


def _coerce_distribution(raw: Any) -> Optional[List[float]]:
    """Coerce a peer/universe stat entry into a clean list of positive floats.

    Accepts either a bare sequence of numbers or a mapping exposing a ``values`` or
    ``distribution`` sequence. Non-numeric and non-positive entries are dropped
    (a non-positive multiple is meaningless for cheapness ranking). Returns ``None``
    when nothing usable remains.
    """
    seq: Optional[Sequence[Any]] = None
    if isinstance(raw, dict):
        for key in ("values", "distribution", "samples"):
            candidate = raw.get(key)
            if isinstance(candidate, (list, tuple)):
                seq = candidate
                break
    elif isinstance(raw, (list, tuple)):
        seq = raw

    if seq is None:
        return None

    clean: List[float] = []
    for v in seq:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > _MIN_MEANINGFUL:
            clean.append(f)
    return clean or None


def _score_one_multiple(
    value: Optional[float],
    spec: _MultipleSpec,
    ctx: AnalysisContext,
) -> Optional[Tuple[float, Evidence, bool]]:
    """Score a single multiple into a 0–100 cheapness score plus its evidence.

    Returns ``None`` when the multiple is missing or not meaningful (non-positive),
    so the caller can simply skip it. Otherwise returns a 3-tuple
    ``(score, evidence, used_relative)`` where ``score`` is in ``[0, 100]`` (higher
    == cheaper == more attractive), ``evidence`` is the auditable
    :class:`Evidence` behind it, and ``used_relative`` records whether a
    peer/universe percentile (``True``) or the absolute fallback band (``False``)
    produced the score.
    """
    if value is None or value <= _MIN_MEANINGFUL:
        return None

    attr = str(spec["attr"])
    label = str(spec["label"])

    distribution, scope = _peer_or_universe_distribution(attr, ctx)
    if distribution:
        pct = percentile_rank(value, distribution)
        if pct is not None:
            # ``pct`` is the fraction of comparables at-or-below this value. Lower
            # multiple == cheaper == more attractive, so invert: a value below most
            # peers (small pct) earns a high score.
            score = clamp((1.0 - pct) * 100.0, 0.0, 100.0)
            detail = (
                f"cheaper than {(1.0 - pct) * 100.0:.0f}% of {scope} "
                f"(n={len(distribution)})"
            )
            direction = "bullish" if score >= 60.0 else ("bearish" if score <= 40.0 else "neutral")
            evidence = Evidence.from_number(
                f"{label} vs {scope}",
                value,
                source="valuation",
                direction=direction,
                detail=detail,
            )
            return score, evidence, True

    # Fallback: transparent absolute band, clearly labelled so the reader knows no
    # relative context backed this component.
    abs_score = scale_to_score(
        value,
        lo=float(spec["abs_lo"]),
        hi=float(spec["abs_hi"]),
        higher_is_better=False,  # a lower multiple is more attractive
    )
    if abs_score is None:  # pragma: no cover - guarded by the None check above
        return None
    abs_score = clamp(abs_score, 0.0, 100.0)
    direction = "bullish" if abs_score >= 60.0 else ("bearish" if abs_score <= 40.0 else "neutral")
    evidence = Evidence.from_number(
        f"{label} (absolute)",
        value,
        source="valuation",
        direction=direction,
        detail=f"absolute band {spec['abs_lo']:g}–{spec['abs_hi']:g}; no peer/universe context",
    )
    return abs_score, evidence, False


def _pct_change(newer: Optional[float], older: Optional[float]) -> Optional[float]:
    """Return the fractional change ``(newer - older) / |older|`` or ``None``.

    ``None`` is returned when either input is missing or the older base is zero, so
    a missing trend never masquerades as growth or decline.
    """
    if newer is None or older is None:
        return None
    if older == 0:
        return None
    return (newer - older) / abs(older)


def _assess_fundamental_quality(
    data: SecurityData,
) -> Tuple[float, List[Evidence], List[str]]:
    """Judge whether a cheap multiple is *earned* by real, improving economics.

    Returns ``(quality_factor, evidence, flags)`` where ``quality_factor`` is in
    ``[0, 1]``: 1.0 means the fundamentals fully support reading the cheapness as
    genuine value, while values toward 0 mean the cheapness looks like a value trap
    (deteriorating revenue/margins, negative free cash flow, losses). The factor is
    used to nudge the blended cheapness score toward neutral when economics are
    weak, never to invent attractiveness where the multiples are rich.

    The assessment is conservative and evidence-backed: each contributing signal
    emits its own :class:`Evidence` so the adjustment is auditable, and signals
    with no data are simply skipped (they cannot lower the factor).
    """
    evidence: List[Evidence] = []
    flags: List[str] = []

    latest: Optional[FundamentalsPeriod] = data.latest_fundamentals
    if latest is None:
        # No fundamentals to corroborate the multiples: we cannot confirm the
        # cheapness is earned, so apply a mild discount and flag it.
        flags.append("UNCONFIRMED_EARNINGS")
        return 0.85, evidence, flags

    # Collect signed "health" signals in [-1, +1]; we average them into a factor.
    signals: List[float] = []

    # 1. Real, positive earnings / FCF behind the price multiples.
    ni = latest.net_income
    fcf = latest.free_cash_flow
    if ni is not None:
        if ni > 0:
            signals.append(1.0)
            evidence.append(
                Evidence.from_number(
                    "Net income",
                    ni,
                    source="fundamentals",
                    direction="bullish",
                    detail="positive earnings back the valuation",
                )
            )
        else:
            signals.append(-1.0)
            flags.append("NEGATIVE_EARNINGS")
            evidence.append(
                Evidence.from_number(
                    "Net income",
                    ni,
                    source="fundamentals",
                    direction="bearish",
                    detail="loss-making; a low P/E here would be an artefact",
                )
            )
    if fcf is not None:
        if fcf > 0:
            signals.append(1.0)
            evidence.append(
                Evidence.from_number(
                    "Free cash flow",
                    fcf,
                    source="fundamentals",
                    direction="bullish",
                    detail="positive FCF supports the cheapness",
                )
            )
        else:
            signals.append(-1.0)
            flags.append("NEGATIVE_FCF")
            evidence.append(
                Evidence.from_number(
                    "Free cash flow",
                    fcf,
                    source="fundamentals",
                    direction="bearish",
                    detail="cash-burning; cheapness may be a trap",
                )
            )

    # 2. Revenue trend (latest vs the prior period available; fundamentals are
    #    newest-first so index 1 is the comparison period).
    if len(data.fundamentals) >= 2:
        prior = data.fundamentals[1]
        rev_chg = _pct_change(latest.revenue, prior.revenue)
        if rev_chg is not None:
            if rev_chg <= -0.05:
                signals.append(-1.0)
                flags.append("REVENUE_DECLINE")
                evidence.append(
                    Evidence.from_number(
                        "Revenue change vs prior period",
                        rev_chg * 100.0,
                        source="fundamentals",
                        direction="bearish",
                        unit="%",
                        detail="shrinking top line is a value-trap warning",
                    )
                )
            elif rev_chg >= 0.05:
                signals.append(1.0)
                evidence.append(
                    Evidence.from_number(
                        "Revenue change vs prior period",
                        rev_chg * 100.0,
                        source="fundamentals",
                        direction="bullish",
                        unit="%",
                        detail="growing top line supports genuine value",
                    )
                )
            else:
                signals.append(0.0)

        # 3. Operating-margin trend (uses the stored derived margin when present).
        margin_chg = _pct_change(latest.operating_margin, prior.operating_margin)
        if margin_chg is not None:
            if margin_chg <= -0.10:
                signals.append(-1.0)
                flags.append("MARGIN_COMPRESSION")
                evidence.append(
                    Evidence.from_number(
                        "Operating-margin change vs prior period",
                        margin_chg * 100.0,
                        source="fundamentals",
                        direction="bearish",
                        unit="%",
                        detail="eroding profitability undercuts the cheapness",
                    )
                )
            elif margin_chg >= 0.10:
                signals.append(1.0)
                evidence.append(
                    Evidence.from_number(
                        "Operating-margin change vs prior period",
                        margin_chg * 100.0,
                        source="fundamentals",
                        direction="bullish",
                        unit="%",
                        detail="expanding margins reinforce genuine value",
                    )
                )
            else:
                signals.append(0.0)

    if not signals:
        # Some fundamentals existed but none of our quality signals had data.
        flags.append("UNCONFIRMED_EARNINGS")
        return 0.9, evidence, flags

    # Map the mean signal in [-1, 1] onto a factor in [0.6, 1.0]: fully healthy
    # economics leave the cheapness untouched (1.0); maximally deteriorating
    # economics damp it to 0.6 (and the VALUE_TRAP flag is added by the caller).
    mean_signal = sum(signals) / len(signals)
    quality_factor = 0.8 + 0.2 * mean_signal  # mean -1 -> 0.6, 0 -> 0.8, +1 -> 1.0
    quality_factor = max(0.6, min(1.0, quality_factor))
    return quality_factor, evidence, flags


@register_analyzer
class ValueAnalyzer(Analyzer):
    """Scores the VALUE category: how cheap the security is, and whether that
    cheapness is backed by real, non-deteriorating economics.

    The score blends up to six valuation multiples — each ranked against sector
    peers and/or the screened universe when those distributions are supplied, and
    otherwise against transparent absolute bands — then applies a value-trap
    adjustment so a statistically cheap but fundamentally deteriorating company is
    pulled back toward neutral rather than rewarded. Higher score == cheaper *and*
    better-supported value. This is one independent evidence category; it is never,
    on its own, a recommendation.
    """

    category = ScoreCategory.VALUE
    default_weight = 0.16
    requires: Set[str] = {"valuation"}

    def analyze(self, data: SecurityData, ctx: AnalysisContext) -> SubScore:
        """Return the VALUE :class:`SubScore` for ``data`` (pure; no I/O).

        Steps:

        1. Score each available, *meaningful* multiple (positive only) into a 0–100
           cheapness score — relative to peers/universe when a distribution exists,
           else against an absolute band.
        2. Blend the per-multiple scores with the configured weights to a base
           cheapness score; ``data_coverage`` reflects how many of the six expected
           multiples were actually usable.
        3. Assess whether the cheapness is *earned* by real economics and apply a
           value-trap dampener (with a ``VALUE_TRAP_RISK`` flag) when fundamentals
           are deteriorating.
        4. Set ``confidence`` from coverage and whether relative context was
           available, and assemble the auditable rationale + evidence.

        When no usable valuation multiple exists at all, returns
        :meth:`neutral_subscore` (score 50, low confidence, ``MISSING_DATA``).
        """
        valuation = data.valuation
        evidence: List[Evidence] = []
        flags: List[str] = []

        component_scores: List[float] = []
        component_weights: List[float] = []
        used_any_relative = False
        usable_count = 0

        for spec in _MULTIPLES:
            raw_value: Optional[float] = getattr(valuation, str(spec["attr"]), None)
            scored = _score_one_multiple(raw_value, spec, ctx)
            if scored is None:
                # Distinguish "present but meaningless" (negative/zero) from absent,
                # so a negative earnings multiple is recorded honestly as neutral
                # evidence rather than silently dropped.
                if raw_value is not None and raw_value <= _MIN_MEANINGFUL:
                    flags.append(f"NONMEANINGFUL_{str(spec['attr']).upper()}")
                    evidence.append(
                        Evidence.from_number(
                            f"{spec['label']} (excluded)",
                            raw_value,
                            source="valuation",
                            direction="neutral",
                            detail="non-positive multiple is an artefact, not cheapness",
                        )
                    )
                continue

            score, ev, used_relative = scored
            component_scores.append(score)
            component_weights.append(float(spec["weight"]))
            evidence.append(ev)
            used_any_relative = used_any_relative or used_relative
            usable_count += 1

        # ------------------------------------------------------------------ #
        # No usable multiple at all -> honest neutral, low-confidence score.   #
        # ------------------------------------------------------------------ #
        if usable_count == 0:
            return self.neutral_subscore(
                rationale=(
                    "No usable valuation multiple was available (all missing or "
                    "non-meaningful), so cheapness cannot be assessed; this data gap "
                    "neither helps nor hurts the company."
                ),
                coverage=0.0,
                extra_flags=flags or None,
            )

        # ------------------------------------------------------------------ #
        # Blend the per-multiple cheapness scores.                            #
        # ------------------------------------------------------------------ #
        weight_sum = sum(component_weights)
        base_cheapness = (
            sum(s * w for s, w in zip(component_scores, component_weights)) / weight_sum
            if weight_sum > 0
            else sum(component_scores) / len(component_scores)
        )

        # ``data_coverage`` = fraction of the six expected multiples that were usable.
        coverage = usable_count / float(len(_MULTIPLES))

        # ------------------------------------------------------------------ #
        # Value-trap adjustment: is the cheapness earned by real economics?   #
        # ------------------------------------------------------------------ #
        quality_factor, quality_evidence, quality_flags = _assess_fundamental_quality(data)
        evidence.extend(quality_evidence)
        flags.extend(quality_flags)

        # Apply the dampener only when the stock screens cheap (so a richly-valued
        # name is not "rewarded" by a weak-fundamentals discount) and economics are
        # below the neutral bar. The dampener pulls a cheap score toward neutral 50.
        adjusted = base_cheapness
        if base_cheapness > 55.0 and quality_factor < 1.0:
            adjusted = 50.0 + (base_cheapness - 50.0) * quality_factor
            if quality_factor <= 0.75:
                flags.append("VALUE_TRAP_RISK")

        final_score = clamp(adjusted, 0.0, 100.0)

        # ------------------------------------------------------------------ #
        # Confidence: grows with coverage; relative (peer/universe) context   #
        # makes the read more trustworthy than absolute-band fallbacks.       #
        # ------------------------------------------------------------------ #
        confidence = 0.25 + 0.55 * coverage
        if used_any_relative:
            confidence += 0.15
        else:
            flags.append("ABSOLUTE_BANDS_ONLY")
        # Unconfirmed economics make the read less trustworthy.
        if "UNCONFIRMED_EARNINGS" in flags:
            confidence -= 0.10
        confidence = max(0.05, min(1.0, confidence))

        # ------------------------------------------------------------------ #
        # Rationale.                                                          #
        # ------------------------------------------------------------------ #
        scope_phrase = (
            "ranked against sector peers / the screened universe"
            if used_any_relative
            else "scored against transparent absolute valuation bands (no peer/universe context available)"
        )
        if final_score >= 65.0:
            verdict = "screens cheap"
        elif final_score <= 35.0:
            verdict = "screens expensive"
        else:
            verdict = "screens around fair value"
        trap_note = ""
        if "VALUE_TRAP_RISK" in flags:
            trap_note = (
                " The headline cheapness is discounted because fundamentals are "
                "deteriorating (possible value trap)."
            )
        rationale = (
            f"{data.ticker} {verdict}: {usable_count} of {len(_MULTIPLES)} valuation "
            f"multiples were usable and {scope_phrase}, for a blended VALUE score of "
            f"{final_score:.0f}/100.{trap_note}"
        )

        return SubScore(
            category=self.category,
            score=final_score,
            confidence=confidence,
            weight=self.default_weight,
            rationale=rationale,
            evidence=evidence,
            flags=_dedupe_preserving_order(flags),
            data_coverage=coverage,
        )


def _dedupe_preserving_order(items: List[str]) -> List[str]:
    """Return ``items`` with duplicates removed, preserving first-seen order."""
    seen: Set[str] = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


__all__ = ["ValueAnalyzer"]
