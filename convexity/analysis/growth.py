"""GROWTH analyzer — scores a company's growth *trajectory* and its quality.

Part of Convexity, an evidence-driven equity **research and screening** tool. It
is **not** a predictor and **not** investment advice. This analyzer does not
forecast future growth; it transparently summarises the growth a company has
*already reported* across its fundamentals history into one auditable sub-score,
to be aggregated alongside many other independent pieces of evidence. Growth
agreeing with value, quality, financial health and the rest is what builds
conviction — a single fast top line never should on its own.

What "growth" means here
------------------------
Three orthogonal questions about the historical record, scored from real data:

1. **Trajectory** — how fast did the top and bottom lines grow? We blend the
   most-recent year-over-year (YoY) growth with a longer multi-year compound
   annual growth rate (CAGR) for revenue, plus YoY for earnings (net income /
   EPS) and free cash flow. A company growing revenue ~25%+ a year scores near
   the top of the band; a shrinking one scores near the bottom.

2. **Acceleration** — is growth speeding up or slowing down? We compare the
   latest YoY revenue growth against the prior YoY (and against the multi-year
   CAGR). Acceleration is a bullish modifier; deceleration is bearish. A company
   whose growth is decelerating sharply is penalised even if its absolute rate is
   still positive.

3. **Quality of growth** — is it *margin-accretive and organic-looking*? Growth
   that comes with rising (or at least stable) operating margins is far more
   valuable than growth bought with collapsing profitability. We reward an
   expanding operating margin alongside positive revenue growth and lightly
   penalise the opposite (top line up while margins crater — a red flag).

Each component is mapped to a 0–100 sub-score with the pure helpers in
:mod:`convexity.core.scoring` (no absolute magic numbers buried in code paths),
combined by a transparent weighted mean, and then — when peer/universe context
is supplied via :class:`~convexity.core.contracts.AnalysisContext` — *nudged*
toward where the company's revenue growth sits in the cross-sectional
distribution (``percentile_rank``). For micro-caps "fast" is sector-relative, so
relative context matters, but we degrade gracefully to absolute bands when no
context is present.

Honesty rules honoured here
---------------------------
* **No fabrication.** A missing line item is ``None`` and simply does not
  contribute; it never becomes a guessed number. ``data_coverage`` reports the
  fraction of the growth signals we could actually compute, and ``confidence``
  scales with both coverage and how many fiscal periods of history existed.
* **Missing-data fallback.** With no fundamentals history (or no usable
  revenue series) we return :meth:`Analyzer.neutral_subscore` — a 50, low
  confidence, ``MISSING_DATA``-flagged score — so a data gap neither helps nor
  hurts the company.
* **Pure & deterministic.** ``analyze`` performs no I/O, reads no clock, draws no
  randomness; it operates solely on the passed :class:`SecurityData`.
* **Auditable.** Every contributing number is emitted as an :class:`Evidence`
  item (via :meth:`Evidence.from_number`) with an honest direction, so a reader
  can reconstruct exactly how the score was reached.
"""

from __future__ import annotations

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
from convexity.core.scoring import (
    clamp,
    percentile_rank,
    scale_to_score,
    weighted_mean,
)

# Source label used on every Evidence item this analyzer emits. The underlying
# numbers all come from the aggregated fundamentals history.
_SOURCE = "fundamentals history"

# Growth-rate scoring bands (expressed as fractions; 0.25 == 25%). A rate at or
# below ``_LO`` maps to 0, at or above ``_HI`` maps to 100, linear between. The
# bands are deliberately generous on the downside (we let modest negatives score
# low rather than zero) and saturate at a strong-but-attainable small-cap rate.
_REV_GROWTH_LO = -0.15   # a 15% revenue decline -> floor of the band
_REV_GROWTH_HI = 0.40    # 40%+ revenue growth -> top of the band
_EARN_GROWTH_LO = -0.30  # earnings are noisier; widen the band
_EARN_GROWTH_HI = 0.60
_FCF_GROWTH_LO = -0.40   # FCF is the noisiest series of the three
_FCF_GROWTH_HI = 0.75

# Acceleration band: the change in YoY revenue growth versus the prior YoY
# (delta of fractions). A +10pp acceleration tops the band; a -10pp deceleration
# bottoms it. 50 is "growth holding steady".
_ACCEL_LO = -0.10
_ACCEL_HI = 0.10

# Margin-trend band: change in operating margin (fraction) latest-vs-prior. A
# +5pp expansion tops the band; a -5pp contraction bottoms it.
_MARGIN_TREND_LO = -0.05
_MARGIN_TREND_HI = 0.05

# Flag-emission thresholds are intentionally *looser* than the scoring bands
# above: the bands decide *how much* a move helps the score, while these decide
# *whether* the qualitative flag ("accelerating", "margin-accretive") is worth
# surfacing to a human reader. A material single-year move of ~2pp earns the tag
# even if it does not saturate the scoring band.
_ACCEL_FLAG_HI = 0.02
_ACCEL_FLAG_LO = -0.02
_MARGIN_FLAG_HI = 0.02
_MARGIN_FLAG_LO = -0.02

# Internal weights blending the component scores into the raw growth score.
# Revenue (the most durable, least-manipulable growth signal) dominates;
# acceleration and margin-accretion shape it; earnings/FCF corroborate.
_W_REVENUE = 0.34
_W_EARNINGS = 0.16
_W_FCF = 0.12
_W_ACCEL = 0.20
_W_MARGIN = 0.18

# How heavily a peer/universe percentile blends into the absolute score when
# relative context is available (0..1). Kept below 0.5 so absolute reality always
# carries more weight than where the crowd happens to sit.
_RELATIVE_BLEND = 0.35

# Keys we look for in ctx.peer_stats / ctx.universe_stats. The value may be a raw
# distribution (sequence of revenue growth fractions) which we percentile-rank
# against. We try several conventional spellings and degrade gracefully if absent.
_PEER_REVENUE_GROWTH_KEYS = (
    "revenue_growth",
    "rev_growth",
    "revenue_yoy",
    "revenue_growth_yoy",
)


def _pct(value: Optional[float], precision: int = 1) -> Optional[float]:
    """Render a growth *fraction* as a percentage number for evidence display."""
    return None if value is None else value * 100.0


def _growth_rate(newer: Optional[float], older: Optional[float]) -> Optional[float]:
    """Period-over-period growth of ``newer`` vs ``older`` as a fraction.

    Returns ``None`` when either value is missing or the base is not strictly
    positive (growth off a zero/negative base is undefined and would otherwise
    fabricate a meaningless, often explosive, number). This keeps the signal
    honest: we only report a growth rate when there is a sound base to grow from.
    """
    if newer is None or older is None:
        return None
    if older <= 0.0:
        return None
    return (newer - older) / older


def _cagr(newest: Optional[float], oldest: Optional[float], years: int) -> Optional[float]:
    """Compound annual growth rate over ``years`` periods, as a fraction.

    Returns ``None`` if endpoints are missing, the base is not strictly positive,
    the latest value is non-positive (a sign change makes a real CAGR undefined),
    or ``years`` is not positive.
    """
    if newest is None or oldest is None or years <= 0:
        return None
    if oldest <= 0.0 or newest <= 0.0:
        return None
    return (newest / oldest) ** (1.0 / years) - 1.0


def _series(periods: List[FundamentalsPeriod], attr: str) -> List[Tuple[int, float]]:
    """Return ``(index, value)`` pairs for a fundamentals attribute, newest-first.

    Only periods where the attribute is present (not ``None``) are included, so
    callers can reason about the actual available series without fabricating gaps.
    The index is the position in the original newest-first ``periods`` list.
    """
    out: List[Tuple[int, float]] = []
    for i, p in enumerate(periods):
        v = getattr(p, attr, None)
        if v is not None:
            out.append((i, float(v)))
    return out


def _latest_two(periods: List[FundamentalsPeriod], attr: str) -> Tuple[Optional[float], Optional[float]]:
    """The two most-recent non-missing values of ``attr`` (newest, prior)."""
    s = _series(periods, attr)
    newest = s[0][1] if len(s) >= 1 else None
    prior = s[1][1] if len(s) >= 2 else None
    return newest, prior


def _yoy(periods: List[FundamentalsPeriod], attr: str) -> Optional[float]:
    """Latest year-over-year growth fraction for ``attr`` (newest vs prior)."""
    newest, prior = _latest_two(periods, attr)
    return _growth_rate(newest, prior)


def _prior_yoy(periods: List[FundamentalsPeriod], attr: str) -> Optional[float]:
    """The *previous* YoY growth (period-2 vs period-3), for acceleration."""
    s = _series(periods, attr)
    if len(s) < 3:
        return None
    return _growth_rate(s[1][1], s[2][1])


def _multiyear_cagr(periods: List[FundamentalsPeriod], attr: str, max_years: int = 4) -> Tuple[Optional[float], int]:
    """CAGR of ``attr`` from the oldest usable period to the newest.

    Uses up to ``max_years`` spans of available history. Returns the CAGR (or
    ``None``) and the number of years it spans (0 if not computable).
    """
    s = _series(periods, attr)
    if len(s) < 2:
        return None, 0
    newest = s[0][1]
    # Walk back at most ``max_years`` periods to find an endpoint.
    end_idx = min(len(s) - 1, max_years)
    oldest = s[end_idx][1]
    years = end_idx  # number of compounding intervals between the two endpoints
    return _cagr(newest, oldest, years), years


def _peer_distribution(ctx: AnalysisContext) -> Optional[List[float]]:
    """Pull a revenue-growth distribution from peer_stats, else universe_stats."""
    for stats in (ctx.peer_stats, ctx.universe_stats):
        if not stats:
            continue
        for key in _PEER_REVENUE_GROWTH_KEYS:
            if key in stats:
                dist = stats[key]
                try:
                    clean = [float(v) for v in dist if v is not None]
                except (TypeError, ValueError):
                    continue
                if clean:
                    return clean
    return None


@register_analyzer
class GrowthAnalyzer(Analyzer):
    """Scores GROWTH: the trajectory, acceleration and quality of reported growth.

    The score answers "how strong, accelerating and margin-accretive is the
    growth this company has *already shown*?" — purely descriptive of the
    historical record, never a forecast. Higher means a stronger, accelerating,
    profitably-financed growth profile; lower means flat, decelerating, or
    margin-dilutive growth.
    """

    category = ScoreCategory.GROWTH
    default_weight = 0.15
    requires: Set[str] = {"fundamentals", "revenue"}

    # The growth components we attempt to score, in evidence/declaration order.
    # (attribute on FundamentalsPeriod, human label, low band, high band)
    _RATE_COMPONENTS = (
        ("revenue", "Revenue YoY growth", _REV_GROWTH_LO, _REV_GROWTH_HI),
        ("net_income", "Net income YoY growth", _EARN_GROWTH_LO, _EARN_GROWTH_HI),
        ("free_cash_flow", "Free cash flow YoY growth", _FCF_GROWTH_LO, _FCF_GROWTH_HI),
    )

    def analyze(self, data: SecurityData, ctx: AnalysisContext) -> SubScore:  # noqa: C901 - explicit & auditable
        periods = data.fundamentals  # newest-first per the contract

        # --- Missing-data guard ------------------------------------------------
        # We need at least two periods of revenue to compute any growth at all.
        rev_series = _series(periods, "revenue")
        if len(periods) < 2 or len(rev_series) < 2:
            return self.neutral_subscore(
                rationale=(
                    "Insufficient fundamentals history to assess growth: at least two "
                    "periods of revenue are required and were not available."
                ),
                coverage=0.0,
            )

        evidence: List[Evidence] = []
        flags: List[str] = []

        # ------------------------------------------------------------------ #
        # 1. Trajectory — YoY + multi-year CAGR for revenue; YoY for earnings/FCF
        # ------------------------------------------------------------------ #
        rev_yoy = _yoy(periods, "revenue")
        rev_cagr, cagr_years = _multiyear_cagr(periods, "revenue")
        ni_yoy = _yoy(periods, "net_income")
        eps_yoy = _yoy(periods, "eps_diluted")
        fcf_yoy = _yoy(periods, "free_cash_flow")

        # Prefer diluted-EPS growth when net income is absent (covers buybacks too).
        earnings_yoy = ni_yoy if ni_yoy is not None else eps_yoy
        earnings_label = "Net income YoY growth" if ni_yoy is not None else "Diluted EPS YoY growth"

        # Revenue component blends YoY and multi-year CAGR (CAGR damps a noisy YoY).
        rev_yoy_score = scale_to_score(rev_yoy, _REV_GROWTH_LO, _REV_GROWTH_HI, higher_is_better=True)
        rev_cagr_score = scale_to_score(rev_cagr, _REV_GROWTH_LO, _REV_GROWTH_HI, higher_is_better=True)
        revenue_score = weighted_mean([rev_yoy_score, rev_cagr_score], [0.6, 0.4])

        earnings_score = scale_to_score(earnings_yoy, _EARN_GROWTH_LO, _EARN_GROWTH_HI, higher_is_better=True)
        fcf_score = scale_to_score(fcf_yoy, _FCF_GROWTH_LO, _FCF_GROWTH_HI, higher_is_better=True)

        evidence.append(
            Evidence.from_number(
                "Revenue YoY growth",
                _pct(rev_yoy),
                source=_SOURCE,
                direction=_dir(rev_yoy, 0.0),
                unit="%",
                precision=1,
                detail="Newest fiscal period vs the prior period.",
            )
        )
        if rev_cagr is not None and cagr_years >= 2:
            evidence.append(
                Evidence.from_number(
                    f"Revenue {cagr_years}y CAGR",
                    _pct(rev_cagr),
                    source=_SOURCE,
                    direction=_dir(rev_cagr, 0.0),
                    unit="%",
                    precision=1,
                    detail=f"Compound annual revenue growth over {cagr_years} years.",
                )
            )
        evidence.append(
            Evidence.from_number(
                earnings_label,
                _pct(earnings_yoy),
                source=_SOURCE,
                direction=_dir(earnings_yoy, 0.0),
                unit="%",
                precision=1,
            )
        )
        evidence.append(
            Evidence.from_number(
                "Free cash flow YoY growth",
                _pct(fcf_yoy),
                source=_SOURCE,
                direction=_dir(fcf_yoy, 0.0),
                unit="%",
                precision=1,
            )
        )

        # ------------------------------------------------------------------ #
        # 2. Acceleration — latest YoY revenue growth vs the prior YoY
        # ------------------------------------------------------------------ #
        prior_rev_yoy = _prior_yoy(periods, "revenue")
        accel: Optional[float] = None
        accel_score: Optional[float] = None
        if rev_yoy is not None and prior_rev_yoy is not None:
            accel = rev_yoy - prior_rev_yoy
            accel_score = scale_to_score(accel, _ACCEL_LO, _ACCEL_HI, higher_is_better=True)
            evidence.append(
                Evidence.from_number(
                    "Revenue growth acceleration",
                    _pct(accel),
                    source=_SOURCE,
                    direction=_dir(accel, 0.0),
                    unit="pp",
                    precision=1,
                    detail=(
                        f"Latest YoY {_pct(rev_yoy):.1f}% vs prior YoY "
                        f"{_pct(prior_rev_yoy):.1f}% (percentage-point change)."
                    ),
                )
            )
            if accel <= _ACCEL_FLAG_LO:
                flags.append("DECELERATING_GROWTH")
            elif accel >= _ACCEL_FLAG_HI:
                flags.append("ACCELERATING_GROWTH")
        # Fall back to YoY-vs-CAGR if we cannot compute a clean prior YoY.
        elif rev_yoy is not None and rev_cagr is not None:
            accel = rev_yoy - rev_cagr
            accel_score = scale_to_score(accel, _ACCEL_LO, _ACCEL_HI, higher_is_better=True)
            evidence.append(
                Evidence.from_number(
                    "Latest growth vs multi-year trend",
                    _pct(accel),
                    source=_SOURCE,
                    direction=_dir(accel, 0.0),
                    unit="pp",
                    precision=1,
                    detail="Latest YoY revenue growth minus the multi-year CAGR.",
                )
            )

        # ------------------------------------------------------------------ #
        # 3. Quality of growth — operating-margin trend alongside revenue growth
        # ------------------------------------------------------------------ #
        margin_new, margin_prior = _latest_two(periods, "operating_margin")
        # Fall back to computing operating_margin from operating_income/revenue.
        if margin_new is None:
            margin_new = _derived_op_margin(periods, 0)
        if margin_prior is None:
            margin_prior = _derived_op_margin(periods, 1)

        margin_trend: Optional[float] = None
        margin_score: Optional[float] = None
        if margin_new is not None and margin_prior is not None:
            margin_trend = margin_new - margin_prior
            margin_score = scale_to_score(
                margin_trend, _MARGIN_TREND_LO, _MARGIN_TREND_HI, higher_is_better=True
            )
            evidence.append(
                Evidence.from_number(
                    "Operating margin trend",
                    _pct(margin_trend),
                    source=_SOURCE,
                    direction=_dir(margin_trend, 0.0),
                    unit="pp",
                    precision=1,
                    detail=(
                        f"Operating margin {_pct(margin_new):.1f}% latest vs "
                        f"{_pct(margin_prior):.1f}% prior."
                    ),
                )
            )
            # Red flag: revenue up but margins contracting meaningfully — growth
            # that is not (yet) margin-accretive.
            if rev_yoy is not None and rev_yoy > 0.0 and margin_trend <= _MARGIN_FLAG_LO:
                flags.append("MARGIN_DILUTIVE_GROWTH")
            elif rev_yoy is not None and rev_yoy > 0.0 and margin_trend >= _MARGIN_FLAG_HI:
                flags.append("MARGIN_ACCRETIVE_GROWTH")

        # ------------------------------------------------------------------ #
        # Blend the components into a raw absolute growth score
        # ------------------------------------------------------------------ #
        component_scores = [revenue_score, earnings_score, fcf_score, accel_score, margin_score]
        component_weights = [_W_REVENUE, _W_EARNINGS, _W_FCF, _W_ACCEL, _W_MARGIN]
        absolute_score = weighted_mean(component_scores, component_weights)
        if absolute_score is None:
            # Revenue is guaranteed present (guard above) so this is unreachable in
            # practice, but stay defensive rather than crash a scan.
            absolute_score = 50.0

        # ------------------------------------------------------------------ #
        # Relative context — nudge toward the cross-sectional revenue-growth rank
        # ------------------------------------------------------------------ #
        score = absolute_score
        peer_dist = _peer_distribution(ctx)
        used_relative = False
        if peer_dist is not None and rev_yoy is not None:
            pr = percentile_rank(rev_yoy, peer_dist)
            if pr is not None:
                relative_score = pr * 100.0
                score = (1.0 - _RELATIVE_BLEND) * absolute_score + _RELATIVE_BLEND * relative_score
                used_relative = True
                evidence.append(
                    Evidence.from_number(
                        "Revenue-growth percentile vs peers",
                        pr * 100.0,
                        source="peer/universe distribution",
                        direction=_dir(pr, 0.5),
                        unit="pct-ile",
                        precision=0,
                        detail=(
                            "Where the company's latest revenue growth ranks within the "
                            f"comparison set of {len(peer_dist)} companies."
                        ),
                    )
                )

        score = clamp(score, 0.0, 100.0)

        # ------------------------------------------------------------------ #
        # Coverage & confidence — honest about how much real signal existed
        # ------------------------------------------------------------------ #
        # Coverage = fraction of the five growth signals we could actually compute.
        computed = sum(1 for s in component_scores if s is not None)
        data_coverage = computed / float(len(component_scores))

        # Confidence scales with coverage and with depth of history (more periods =
        # a more trustworthy trajectory), with a small uplift when relative context
        # corroborated the read. Capped at a sober 0.95 (no certainty is claimed).
        history_factor = min(len(rev_series) / 4.0, 1.0)  # 4+ revenue periods -> full
        confidence = clamp(
            0.25 + 0.5 * data_coverage + 0.2 * history_factor + (0.05 if used_relative else 0.0),
            0.0,
            0.95,
        )

        if len(rev_series) < 3:
            flags.append("SHORT_HISTORY")
        if computed < len(component_scores):
            flags.append("PARTIAL_GROWTH_DATA")

        rationale = _build_rationale(
            rev_yoy=rev_yoy,
            rev_cagr=rev_cagr,
            cagr_years=cagr_years,
            earnings_yoy=earnings_yoy,
            accel=accel,
            margin_trend=margin_trend,
            score=score,
            used_relative=used_relative,
        )

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


# ---------------------------------------------------------------------------
# Module-level helpers (kept pure; no SecurityData mutation, no I/O)
# ---------------------------------------------------------------------------


def _dir(value: Optional[float], midpoint: float) -> str:
    """Honest direction for a number relative to a neutral ``midpoint``.

    ``None`` is always ``neutral`` (a missing datum must never look directional).
    """
    if value is None:
        return "neutral"
    if value > midpoint:
        return "bullish"
    if value < midpoint:
        return "bearish"
    return "neutral"


def _derived_op_margin(periods: List[FundamentalsPeriod], idx: int) -> Optional[float]:
    """Operating margin at position ``idx`` derived from operating_income/revenue.

    Returns ``None`` when either input is missing or revenue is non-positive
    (never fabricates a ratio off a meaningless base).
    """
    if idx < 0 or idx >= len(periods):
        return None
    p = periods[idx]
    oi = getattr(p, "operating_income", None)
    rev = getattr(p, "revenue", None)
    if oi is None or rev is None or rev <= 0.0:
        return None
    return oi / rev


def _build_rationale(
    *,
    rev_yoy: Optional[float],
    rev_cagr: Optional[float],
    cagr_years: int,
    earnings_yoy: Optional[float],
    accel: Optional[float],
    margin_trend: Optional[float],
    score: float,
    used_relative: bool,
) -> str:
    """Compose a short, honest, human rationale from the computed signals."""
    parts: List[str] = []

    if rev_yoy is not None:
        verb = "grew" if rev_yoy >= 0 else "declined"
        parts.append(f"Revenue {verb} {abs(rev_yoy) * 100:.1f}% year over year")
        if rev_cagr is not None and cagr_years >= 2:
            parts[-1] += f" ({rev_cagr * 100:.1f}% {cagr_years}y CAGR)"
    if earnings_yoy is not None:
        verb = "rose" if earnings_yoy >= 0 else "fell"
        parts.append(f"earnings {verb} {abs(earnings_yoy) * 100:.1f}% YoY")
    if accel is not None:
        if accel > 0:
            parts.append("growth is accelerating")
        elif accel < 0:
            parts.append("growth is decelerating")
        else:
            parts.append("growth is holding steady")
    if margin_trend is not None:
        if margin_trend > 0:
            parts.append("with operating margins expanding (margin-accretive)")
        elif margin_trend < 0:
            parts.append("but operating margins are contracting (margin-dilutive)")

    body = "; ".join(parts) if parts else "Limited growth signals were computable from the history"
    suffix = " (scored relative to peers)" if used_relative else ""
    return (
        f"Growth score {score:.0f}/100{suffix}: {body}. "
        "Descriptive of reported history only — not a forecast."
    )


__all__ = ["GrowthAnalyzer"]
