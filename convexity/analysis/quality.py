"""QUALITY analyzer — durable returns on capital, margins, and cash conversion.

Part of Convexity, an evidence-driven equity **research and screening** tool. It
is **not** a predictor and **not** investment advice. This analyzer turns a
company's fundamentals into one transparent, auditable :class:`SubScore` for the
:class:`~convexity.core.models.ScoreCategory.QUALITY` category. It contributes a
single, *independent* piece of evidence that the ranking layer aggregates with
many others — a high quality score never constitutes a thesis on its own.

What "quality" means here
-------------------------
A high-quality business earns *durable, high returns on the capital it employs*,
converts accounting profit into real cash, and sustains healthy margins without
eroding them. This analyzer scores exactly those traits, each from concrete
reported numbers, and rewards **durability** explicitly: a company that has held
strong returns and margins across several periods scores above one with the same
latest figure but a volatile or deteriorating history.

The components (each scored 0–100, higher = better) are:

* **Returns on capital** — ROIC, ROE and ROA *levels*. ROIC is weighted most
  heavily because it is the cleanest measure of value creation per dollar of
  capital; ROE/ROA round out the picture.
* **Return stability** — how *consistent* those returns have been across the
  available history (low dispersion and no decline = durable).
* **Margins** — gross and operating margin *levels* (pricing power and operating
  leverage).
* **Margin stability** — consistency of those margins across history.
* **Cash conversion** — free cash flow as a fraction of net income (and FCF
  margin), i.e. how much reported profit becomes spendable cash.
* **Capital efficiency** — asset turnover (revenue / assets), a structural read
  on how productively the balance sheet is used.

Honesty rules (non-negotiable)
------------------------------
* Pure & deterministic: operates only on the passed
  :class:`~convexity.core.models.SecurityData`; no network, no clock, no random.
* Never fabricate: a missing metric is computed from its components only when
  *those* are present, otherwise it stays absent and lowers ``data_coverage`` and
  ``confidence``. When too little exists to judge quality at all, the analyzer
  returns :meth:`~convexity.core.contracts.Analyzer.neutral_subscore` rather than
  guessing.
* Relative when possible: when ``ctx.peer_stats`` / ``ctx.universe_stats`` carry
  a distribution for a metric, the level is graded by its
  :func:`~convexity.core.scoring.percentile_rank` against comparable companies
  (what counts as a "high" ROIC is sector-relative). Absent that context the
  analyzer degrades gracefully to transparent absolute bands.
* Auditable: every component emits :class:`~convexity.core.models.Evidence`
  citing the concrete number and an honest ``direction``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import List, Optional, Set, Tuple

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

# How many of the most-recent fundamentals periods to consider when judging
# durability. Enough to see a trend without reaching into stale, possibly
# structurally-different history.
_STABILITY_LOOKBACK = 5

# Minimum periods required before a "stability" component is meaningful. With a
# single period we can grade levels but have nothing to say about durability.
_MIN_PERIODS_FOR_STABILITY = 2


# ---------------------------------------------------------------------------
# Small, pure numeric helpers (local to quality; deterministic)
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> Optional[float]:
    """Arithmetic mean of ``values``, or ``None`` if empty."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _coefficient_of_variation(values: Sequence[float]) -> Optional[float]:
    """Return |stdev / mean| for ``values`` (a unitless dispersion measure).

    Lower means more consistent. Returns ``None`` when there are fewer than two
    points or the mean is ~0 (the ratio would be undefined / explode).
    """
    clean = [float(v) for v in values if v is not None]
    if len(clean) < _MIN_PERIODS_FOR_STABILITY:
        return None
    mean = sum(clean) / len(clean)
    if abs(mean) < 1e-9:
        return None
    var = sum((v - mean) ** 2 for v in clean) / len(clean)
    stdev = math.sqrt(var)
    return abs(stdev / mean)


def _trend_penalty(newest_first: Sequence[float]) -> float:
    """Penalty in ``[0, 1]`` for a *declining* series (0 = improving/flat).

    Given values ordered newest-first, compares the most recent value to the
    oldest in the window. A material decline returns a penalty up to 1.0; an
    improving or flat series returns 0.0. Used to dock durability when returns or
    margins are eroding even if their average level looks fine.
    """
    clean = [float(v) for v in newest_first if v is not None]
    if len(clean) < _MIN_PERIODS_FOR_STABILITY:
        return 0.0
    newest, oldest = clean[0], clean[-1]
    if oldest <= 0:
        # Can't form a stable ratio; only penalise an outright drop below the
        # starting point, scaled by the raw gap (kept modest).
        return 0.0 if newest >= oldest else min(1.0, abs(newest - oldest))
    rel_change = (newest - oldest) / abs(oldest)
    if rel_change >= 0:
        return 0.0
    # A 50%+ erosion from the window's start saturates the penalty.
    return min(1.0, -rel_change / 0.5)


def _stability_score(newest_first: Sequence[float]) -> Optional[float]:
    """Grade the durability of a newest-first series on a 0–100 scale.

    Combines two transparent ideas: low period-to-period dispersion (coefficient
    of variation) is good, and an eroding trend is bad. Returns ``None`` when
    there are too few periods to judge durability at all.
    """
    cov = _coefficient_of_variation(newest_first)
    if cov is None:
        return None
    # CoV of 0 -> 100; CoV of 0.6 (very erratic) -> 0. Linear, transparent band.
    base = scale_to_score(cov, lo=0.0, hi=0.6, higher_is_better=False)
    if base is None:  # pragma: no cover - cov is non-None here, so base is too.
        return None
    penalty = _trend_penalty(newest_first)
    return clamp(base * (1.0 - 0.5 * penalty), 0.0, 100.0)


def _series(periods: Sequence[FundamentalsPeriod], attr: str) -> List[float]:
    """Collect the non-``None`` values of ``attr`` across ``periods`` (in order)."""
    out: List[float] = []
    for p in periods:
        v = getattr(p, attr, None)
        if v is not None:
            out.append(float(v))
    return out


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Safe ratio; ``None`` if either input is missing or the denominator is ~0."""
    if numerator is None or denominator is None:
        return None
    if abs(denominator) < 1e-9:
        return None
    return numerator / denominator


def _distribution(ctx: AnalysisContext, metric: str) -> Optional[List[float]]:
    """Pull a peer-or-universe distribution for ``metric`` from the context.

    Prefers ``peer_stats`` (most comparable) then ``universe_stats``. Accepts any
    sequence of numbers; returns ``None`` when neither is present or usable so the
    caller falls back to absolute bands.
    """
    for stats in (ctx.peer_stats, ctx.universe_stats):
        if not stats:
            continue
        dist = stats.get(metric)
        if dist is None:
            continue
        try:
            clean = [float(v) for v in dist if v is not None]
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if len(clean) >= 3:  # need a few points for a percentile to mean anything.
            return clean
    return None


def _level_score(
    value: Optional[float],
    ctx: AnalysisContext,
    metric: str,
    *,
    lo: float,
    hi: float,
) -> Optional[float]:
    """Grade a level either by peer percentile (preferred) or an absolute band.

    When a peer/universe distribution exists for ``metric`` the value is graded by
    its percentile rank against comparable companies (sector-relative). Otherwise
    it falls back to a transparent absolute band ``[lo, hi]`` mapped onto 0–100.
    Returns ``None`` for a missing value.
    """
    if value is None:
        return None
    dist = _distribution(ctx, metric)
    if dist is not None:
        pr = percentile_rank(value, dist)
        if pr is not None:
            return clamp(pr * 100.0, 0.0, 100.0)
    return scale_to_score(value, lo=lo, hi=hi, higher_is_better=True)


@register_analyzer
class QualityAnalyzer(Analyzer):
    """Score business quality: durable high returns on capital, margins, cash.

    The sub-score is a weighted blend of six components (returns level, return
    stability, margin level, margin stability, cash conversion, capital
    efficiency). Each component is graded relative to peers/universe when a
    distribution is available and by a transparent absolute band otherwise. Only
    the components for which real data exists contribute; ``data_coverage`` and
    ``confidence`` scale with how many of the analyzer's required inputs were
    actually present, so a thin-data company cannot earn false conviction.
    """

    category = ScoreCategory.QUALITY
    default_weight = 0.14
    # Capability/field names this analyzer needs for a fully-confident score.
    # These are the building blocks of the six quality components.
    requires: Set[str] = {
        "roic",
        "roe",
        "roa",
        "gross_margin",
        "operating_margin",
        "free_cash_flow",
        "net_income",
    }

    # Component weights (relative; normalised over the components that have data).
    # Returns-on-capital and their durability carry the most weight because they
    # are the cleanest, hardest-to-fake evidence of a quality franchise.
    _COMPONENT_WEIGHTS = {
        "returns_level": 0.28,
        "returns_stability": 0.18,
        "margin_level": 0.18,
        "margin_stability": 0.12,
        "cash_conversion": 0.16,
        "capital_efficiency": 0.08,
    }

    def analyze(self, data: SecurityData, ctx: AnalysisContext) -> SubScore:
        """Return the QUALITY :class:`SubScore` for ``data`` (pure, no I/O)."""
        periods = data.fundamentals  # newest-first per the SecurityData contract.
        latest = data.latest_fundamentals
        if not periods or latest is None:
            return self.neutral_subscore(
                "No fundamentals available; cannot assess business quality.",
                coverage=0.0,
            )

        window = periods[:_STABILITY_LOOKBACK]
        evidence: List[Evidence] = []
        flags: List[str] = []
        # (component_key, score_0_100) for every component we could actually grade.
        components: List[Tuple[str, float]] = []

        source = ", ".join(data.data_sources) if data.data_sources else "fundamentals"
        as_of = latest.period_end

        # -- 1. Returns on capital (level): ROIC weighted most, then ROE, ROA. ---
        roic = latest.roic
        roe = latest.roe
        roa = latest.roa
        roic_s = _level_score(roic, ctx, "roic", lo=0.0, hi=0.20)
        roe_s = _level_score(roe, ctx, "roe", lo=0.0, hi=0.20)
        roa_s = _level_score(roa, ctx, "roa", lo=0.0, hi=0.12)
        returns_parts: List[Tuple[float, float]] = []  # (score, sub-weight)
        if roic_s is not None:
            returns_parts.append((roic_s, 0.5))
        if roe_s is not None:
            returns_parts.append((roe_s, 0.3))
        if roa_s is not None:
            returns_parts.append((roa_s, 0.2))
        if returns_parts:
            num = sum(s * w for s, w in returns_parts)
            den = sum(w for _, w in returns_parts)
            components.append(("returns_level", clamp(num / den, 0.0, 100.0)))
        evidence.append(
            Evidence.from_number(
                "ROIC (latest)", None if roic is None else roic * 100.0,
                source=source, unit="%",
                direction=_dir(roic, 0.10, 0.05), as_of=as_of,
                detail="Return on invested capital — value created per dollar of capital.",
            )
        )
        evidence.append(
            Evidence.from_number(
                "ROE (latest)", None if roe is None else roe * 100.0,
                source=source, unit="%", direction=_dir(roe, 0.12, 0.05), as_of=as_of,
            )
        )
        evidence.append(
            Evidence.from_number(
                "ROA (latest)", None if roa is None else roa * 100.0,
                source=source, unit="%", direction=_dir(roa, 0.06, 0.02), as_of=as_of,
            )
        )
        if roic is not None and roic >= 0.15:
            flags.append("HIGH_RETURNS_ON_CAPITAL")
        if roic is not None and roic < 0.0:
            flags.append("NEGATIVE_ROIC")

        # -- 2. Return stability (durability across history). -------------------
        roic_series = _series(window, "roic")
        roe_series = _series(window, "roe")
        return_durability_series = roic_series if len(roic_series) >= _MIN_PERIODS_FOR_STABILITY else roe_series
        ret_stab = _stability_score(return_durability_series)
        if ret_stab is not None:
            components.append(("returns_stability", ret_stab))
            cov = _coefficient_of_variation(return_durability_series)
            evidence.append(
                Evidence.from_number(
                    "Return-on-capital variability (CoV)", cov, source=source,
                    direction="bullish" if (cov is not None and cov < 0.25) else "bearish" if (cov is not None and cov > 0.5) else "neutral",
                    detail=f"Dispersion of returns across {len(return_durability_series)} periods; lower = more durable.",
                )
            )
            if _trend_penalty(return_durability_series) > 0.3:
                flags.append("DECLINING_RETURNS")
        else:
            flags.append("RETURN_HISTORY_TOO_SHORT")

        # -- 3. Margin level: gross & operating margin. -------------------------
        gm = latest.gross_margin
        om = latest.operating_margin
        gm_s = _level_score(gm, ctx, "gross_margin", lo=0.10, hi=0.60)
        om_s = _level_score(om, ctx, "operating_margin", lo=0.0, hi=0.25)
        margin_parts: List[Tuple[float, float]] = []
        if gm_s is not None:
            margin_parts.append((gm_s, 0.45))
        if om_s is not None:
            margin_parts.append((om_s, 0.55))
        if margin_parts:
            num = sum(s * w for s, w in margin_parts)
            den = sum(w for _, w in margin_parts)
            components.append(("margin_level", clamp(num / den, 0.0, 100.0)))
        evidence.append(
            Evidence.from_number(
                "Gross margin (latest)", None if gm is None else gm * 100.0,
                source=source, unit="%", direction=_dir(gm, 0.40, 0.20), as_of=as_of,
            )
        )
        evidence.append(
            Evidence.from_number(
                "Operating margin (latest)", None if om is None else om * 100.0,
                source=source, unit="%", direction=_dir(om, 0.15, 0.0), as_of=as_of,
            )
        )
        if om is not None and om < 0.0:
            flags.append("NEGATIVE_OPERATING_MARGIN")

        # -- 4. Margin stability. ----------------------------------------------
        om_series = _series(window, "operating_margin")
        gm_series = _series(window, "gross_margin")
        margin_durability_series = om_series if len(om_series) >= _MIN_PERIODS_FOR_STABILITY else gm_series
        mar_stab = _stability_score(margin_durability_series)
        if mar_stab is not None:
            components.append(("margin_stability", mar_stab))
            mcov = _coefficient_of_variation(margin_durability_series)
            evidence.append(
                Evidence.from_number(
                    "Margin variability (CoV)", mcov, source=source,
                    direction="bullish" if (mcov is not None and mcov < 0.20) else "bearish" if (mcov is not None and mcov > 0.45) else "neutral",
                    detail=f"Dispersion of margins across {len(margin_durability_series)} periods; lower = more durable.",
                )
            )
            if _trend_penalty(margin_durability_series) > 0.3:
                flags.append("ERODING_MARGINS")

        # -- 5. Cash conversion: FCF / net income, plus FCF margin. -------------
        fcf = latest.free_cash_flow
        ni = latest.net_income
        fcf_margin = latest.fcf_margin
        if fcf_margin is None:
            fcf_margin = _ratio(fcf, latest.revenue)
        conversion = _ratio(fcf, ni) if (ni is not None and ni > 0) else None
        conv_parts: List[Tuple[float, float]] = []
        if conversion is not None:
            # 1.0x conversion (FCF == earnings) is excellent; ~0.4x is mediocre.
            conv_score = scale_to_score(conversion, lo=0.2, hi=1.0, higher_is_better=True)
            if conv_score is not None:
                conv_parts.append((conv_score, 0.6))
        if fcf_margin is not None:
            fcfm_score = _level_score(fcf_margin, ctx, "fcf_margin", lo=0.0, hi=0.15)
            if fcfm_score is not None:
                conv_parts.append((fcfm_score, 0.4))
        if conv_parts:
            num = sum(s * w for s, w in conv_parts)
            den = sum(w for _, w in conv_parts)
            components.append(("cash_conversion", clamp(num / den, 0.0, 100.0)))
        evidence.append(
            Evidence.from_number(
                "FCF / net income (cash conversion)", conversion, source=source,
                direction="bullish" if (conversion is not None and conversion >= 0.8) else "bearish" if (conversion is not None and conversion < 0.4) else "neutral",
                as_of=as_of, precision=2,
                detail="How much reported profit becomes free cash flow.",
            )
        )
        evidence.append(
            Evidence.from_number(
                "FCF margin (latest)", None if fcf_margin is None else fcf_margin * 100.0,
                source=source, unit="%", direction=_dir(fcf_margin, 0.08, 0.0), as_of=as_of,
            )
        )
        if fcf is not None and fcf < 0.0:
            flags.append("NEGATIVE_FREE_CASH_FLOW")

        # -- 6. Capital efficiency: asset turnover (revenue / assets). ----------
        asset_turnover = _ratio(latest.revenue, latest.total_assets)
        if asset_turnover is not None:
            cap_eff = _level_score(asset_turnover, ctx, "asset_turnover", lo=0.2, hi=1.5)
            if cap_eff is not None:
                components.append(("capital_efficiency", cap_eff))
            evidence.append(
                Evidence.from_number(
                    "Asset turnover (revenue / assets)", asset_turnover, source=source,
                    direction="bullish" if asset_turnover >= 0.8 else "bearish" if asset_turnover < 0.3 else "neutral",
                    as_of=as_of, precision=2,
                    detail="How productively the balance sheet generates revenue.",
                )
            )

        # -- Blend the components that actually have data. ----------------------
        if not components:
            return self.neutral_subscore(
                "Fundamentals present but none of the quality metrics "
                "(returns, margins, cash conversion, asset turnover) could be computed.",
                coverage=0.0,
                extra_flags=["NO_QUALITY_METRICS"],
            )

        total_w = sum(self._COMPONENT_WEIGHTS[k] for k, _ in components)
        score = sum(s * self._COMPONENT_WEIGHTS[k] for k, s in components) / total_w
        score = clamp(score, 0.0, 100.0)

        # -- Coverage & confidence reflect how much real input existed. ---------
        # Coverage = fraction of the six quality components we could grade,
        # blended with how many of the `requires` raw fields were present.
        component_coverage = len(components) / len(self._COMPONENT_WEIGHTS)
        present_fields = sum(
            1 for f in self.requires if getattr(latest, f, None) is not None
        )
        field_coverage = present_fields / len(self.requires)
        data_coverage = clamp((component_coverage + field_coverage) / 2.0, 0.0, 1.0)

        # Confidence grows with coverage and with how much history backed the
        # durability components (a one-period snapshot is inherently weaker).
        n_periods = min(len(window), _STABILITY_LOOKBACK)
        history_factor = clamp(n_periods / float(_STABILITY_LOOKBACK), 0.0, 1.0)
        confidence = clamp(0.15 + 0.65 * data_coverage + 0.20 * history_factor, 0.0, 1.0)

        if data_coverage < 0.5:
            flags.append("PARTIAL_DATA")

        rationale = self._build_rationale(score, components, n_periods, data_coverage)

        return SubScore(
            category=self.category,
            score=score,
            confidence=confidence,
            weight=self.default_weight,
            rationale=rationale,
            evidence=evidence,
            flags=flags,
            data_coverage=data_coverage,
        )

    @staticmethod
    def _build_rationale(
        score: float,
        components: Sequence[Tuple[str, float]],
        n_periods: int,
        coverage: float,
    ) -> str:
        """Compose a short, honest human explanation of the quality score."""
        if score >= 70:
            band = "high-quality"
        elif score >= 50:
            band = "average-quality"
        elif score >= 30:
            band = "below-average-quality"
        else:
            band = "low-quality"
        scored = {k: s for k, s in components}
        bits: List[str] = []
        if "returns_level" in scored:
            bits.append(f"returns-on-capital {scored['returns_level']:.0f}/100")
        if "margin_level" in scored:
            bits.append(f"margins {scored['margin_level']:.0f}/100")
        if "cash_conversion" in scored:
            bits.append(f"cash conversion {scored['cash_conversion']:.0f}/100")
        detail = "; ".join(bits) if bits else "limited metrics"
        durability = (
            f" across {n_periods} periods of history"
            if n_periods >= _MIN_PERIODS_FOR_STABILITY
            else " (single-period snapshot; durability unverified)"
        )
        coverage_note = "" if coverage >= 0.5 else " Data coverage is thin, so confidence is reduced."
        return (
            f"Assessed as {band} (score {score:.0f}/100){durability}: {detail}. "
            f"Durable, high returns on capital with stable margins and strong cash "
            f"conversion score best; this reflects only the evidence present and is "
            f"one independent input, not a recommendation.{coverage_note}"
        )


def _dir(value: Optional[float], bullish_at: float, bearish_below: float) -> str:
    """Map a metric value to an honest evidence direction.

    ``bullish`` at/above ``bullish_at``, ``bearish`` below ``bearish_below``,
    otherwise ``neutral``. A missing value is always ``neutral`` (never allowed to
    masquerade as a directional signal).
    """
    if value is None:
        return "neutral"
    if value >= bullish_at:
        return "bullish"
    if value < bearish_below:
        return "bearish"
    return "neutral"


__all__ = ["QualityAnalyzer"]
