"""Competitive-positioning analyzer for Convexity.

Convexity is an evidence-driven equity **research and screening** tool. It is
**not** a predictor and **not** investment advice. This module contributes one
*independent* piece of evidence — a competitive-position sub-score — to a
transparent aggregation of many such signals. A company earns conviction only
when many independent categories agree; the competitive read here is never, on
its own, a recommendation.

What "competitive position" means here
--------------------------------------
A durable competitive advantage (a "moat") is not directly observable, so we
proxy it with four auditable, financially-grounded sub-signals, each scored on
the 0–100 attractiveness scale and then blended:

1. **Margin level vs peers.** A business that earns a *higher* gross/operating
   margin than its peer set is, all else equal, better positioned — it can
   charge more or produce more cheaply. Scored relative to ``ctx.peer_stats``
   when present, against fixed reference bands otherwise.
2. **Gross-margin durability (moat proxy).** A wide *and stable* gross margin
   across the available fundamental history is the cleanest public proxy for
   pricing power. We reward both the level and the *low volatility* of the
   gross margin over time — an erratic margin signals a contestable position.
3. **Growth vs peers.** Persistently taking share shows up as revenue growth
   *above* the peer/universe distribution. Out-growing peers is competitive
   evidence; matching them is neutral.
4. **Returns-on-capital persistence.** A company that *consistently* earns a
   return on invested capital above its cost of capital is, by definition,
   compounding an advantage. We reward both the level of ROIC/ROE and its
   persistence across periods.

Honesty rules honoured throughout
---------------------------------
* **Pure.** ``analyze`` performs no I/O, reads no clock, uses no randomness;
  given the same :class:`SecurityData` it always returns the same sub-score.
* **Never fabricated.** Every missing input lowers ``data_coverage`` and
  ``confidence`` rather than being guessed. With *no* usable inputs the
  analyzer returns :meth:`neutral_subscore` (score 50, low confidence,
  ``MISSING_DATA``) so a data gap neither helps nor hurts the company.
* **Relative when it can be.** Peer- and universe-relative comparisons are
  used when ``ctx.peer_stats`` / ``ctx.universe_stats`` are supplied, and the
  analyzer degrades to absolute reference bands when they are not — flagging
  the degradation so a reader knows the score is less context-aware.
* **Auditable.** Every contributing number is emitted as an :class:`Evidence`
  item citing the concrete value, its source and an honest direction.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import List, Optional, Set

from convexity.core.contracts import AnalysisContext, Analyzer
from convexity.core.models import (
    Evidence,
    FundamentalsPeriod,
    ScoreCategory,
    SecurityData,
    SubScore,
)
from convexity.core.registry import register_analyzer
from convexity.core.scoring import clamp, percentile_rank, scale_to_score, weighted_mean

# Source label used on every Evidence item this analyzer emits. The underlying
# numbers all originate from the company's reported fundamentals, surfaced by
# the aggregator; the label keeps provenance explicit and auditable.
_SOURCE = "Convexity competitive analyzer (reported fundamentals)"

# How many of the most-recent fundamental periods we look back over when
# measuring durability/persistence. Newest-first per the SecurityData contract.
_HISTORY_WINDOW = 8

# Absolute reference bands used only when peer/universe context is absent. They
# are deliberately broad, industry-agnostic anchors — a fallback, not a precise
# benchmark — and the fallback is always flagged so a reader can discount it.
_GROSS_MARGIN_BAND = (0.10, 0.65)      # 10% -> 0, 65% -> 100
_OPERATING_MARGIN_BAND = (-0.05, 0.30)  # -5% -> 0, 30% -> 100
_ROIC_BAND = (0.0, 0.25)                # 0% -> 0, 25% -> 100
_REVENUE_GROWTH_BAND = (-0.10, 0.30)    # -10% -> 0, 30% -> 100

# Internal weights blending the four sub-signals into the category score. They
# are normalised over whichever signals are actually available, so a missing
# signal redistributes weight rather than dragging the score toward zero.
_W_MARGIN_LEVEL = 0.30
_W_MARGIN_DURABILITY = 0.30
_W_GROWTH_VS_PEERS = 0.20
_W_RETURNS_PERSISTENCE = 0.20


def _peer_distribution(ctx: AnalysisContext, *keys: str) -> Optional[List[float]]:
    """Return a numeric peer distribution for the first matching key, else ``None``.

    ``ctx.peer_stats`` is analyzer-defined in shape; this helper accepts any
    mapping whose value for a key is an iterable of numbers (e.g.
    ``{"gross_margin": [0.41, 0.38, 0.55]}``) and tolerates ``None`` entries by
    dropping them. Returns ``None`` when no key resolves to a non-empty numeric
    sequence so callers can fall back to absolute bands.
    """
    stats = ctx.peer_stats
    if not stats:
        return None
    for key in keys:
        raw = stats.get(key) if hasattr(stats, "get") else None
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            return [float(raw)]
        try:
            clean = [float(v) for v in raw if v is not None]
        except (TypeError, ValueError):
            continue
        if clean:
            return clean
    return None


def _universe_distribution(ctx: AnalysisContext, *keys: str) -> Optional[List[float]]:
    """Like :func:`_peer_distribution` but reading ``ctx.universe_stats``."""
    stats = ctx.universe_stats
    if not stats:
        return None
    for key in keys:
        raw = stats.get(key) if hasattr(stats, "get") else None
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            return [float(raw)]
        try:
            clean = [float(v) for v in raw if v is not None]
        except (TypeError, ValueError):
            continue
        if clean:
            return clean
    return None


def _series(
    periods: Sequence[FundamentalsPeriod],
    attr: str,
    *,
    limit: int = _HISTORY_WINDOW,
) -> List[float]:
    """Collect up to ``limit`` non-``None`` values of ``attr`` across periods.

    Periods are newest-first (per the :class:`SecurityData` contract); the
    returned list preserves that ordering and silently skips missing values.
    """
    out: List[float] = []
    for period in periods[:limit]:
        val = getattr(period, attr, None)
        if val is not None:
            out.append(float(val))
    return out


def _stability_score(values: Sequence[float]) -> Optional[float]:
    """Score the *stability* of a margin/return series on 0–100 (higher = steadier).

    Uses the coefficient of variation (population stdev / |mean|) so the measure
    is scale-free, then maps it through a band where a CV of 0 (perfectly flat)
    scores 100 and a CV at/above 0.6 (very erratic) scores 0. Needs at least two
    observations; returns ``None`` otherwise (insufficient data).
    """
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 2:
        return None
    mean = sum(clean) / len(clean)
    if mean == 0.0:
        return None
    stdev = statistics.pstdev(clean)
    cv = abs(stdev / mean)
    # CV 0 -> 100 (perfectly stable); CV 0.6+ -> 0 (highly volatile).
    return scale_to_score(cv, 0.0, 0.6, higher_is_better=False)


def _persistence_fraction(values: Sequence[float], threshold: float = 0.0) -> Optional[float]:
    """Fraction of ``values`` strictly above ``threshold`` (a persistence proxy).

    Returns a value in ``[0, 1]`` — e.g. ROIC positive in 7 of 8 periods -> 0.875
    — or ``None`` when there are no observations.
    """
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    above = sum(1 for v in clean if v > threshold)
    return above / len(clean)


@register_analyzer
class CompetitiveAnalyzer(Analyzer):
    """Scores :class:`ScoreCategory.COMPETITIVE` — durability of competitive position.

    Blends four independent, financially-grounded proxies for competitive
    advantage: margin level vs peers, gross-margin durability (a moat proxy),
    revenue growth vs peers/universe, and the level-and-persistence of returns on
    capital. Peer/universe context is used when available and the analyzer
    degrades gracefully (with an explicit flag) to absolute reference bands when
    it is not. The result is one auditable sub-score among the twelve independent
    categories Convexity aggregates — never a standalone verdict.
    """

    category: ScoreCategory = ScoreCategory.COMPETITIVE
    default_weight: float = 0.07
    requires: Set[str] = {"fundamentals"}

    def analyze(self, data: SecurityData, ctx: AnalysisContext) -> SubScore:
        """Return the competitive-position :class:`SubScore` for ``data``.

        Pure: depends only on ``data`` and ``ctx``. Emits one
        :meth:`neutral_subscore` when no usable fundamentals exist; otherwise
        blends whichever of the four sub-signals could be computed, scales
        ``data_coverage``/``confidence`` to how much real input was present, and
        attaches an :class:`Evidence` item for every contributing number.
        """
        periods = list(data.fundamentals)
        latest: Optional[FundamentalsPeriod] = data.latest_fundamentals

        if not periods or latest is None:
            return self.neutral_subscore(
                rationale=(
                    "No fundamentals available, so competitive positioning "
                    "(margins, durability, returns on capital) cannot be assessed; "
                    "scored neutral with minimal confidence."
                ),
                coverage=0.0,
            )

        evidence: List[Evidence] = []
        flags: List[str] = []
        component_scores: List[Optional[float]] = []
        component_weights: List[float] = []

        # We track how many of the four sub-signals were genuinely computable so
        # data_coverage and confidence reflect real input, not assumptions.
        signals_possible = 4
        signals_present = 0

        has_peers = bool(ctx.peer_stats)
        if not has_peers:
            flags.append("NO_PEER_CONTEXT")

        # ------------------------------------------------------------------
        # 1. Margin level vs peers (gross margin preferred, operating as backup)
        # ------------------------------------------------------------------
        margin_level_score = self._score_margin_level(latest, ctx, evidence, flags)
        if margin_level_score is not None:
            component_scores.append(margin_level_score)
            component_weights.append(_W_MARGIN_LEVEL)
            signals_present += 1

        # ------------------------------------------------------------------
        # 2. Gross-margin durability (level x stability) — the moat proxy
        # ------------------------------------------------------------------
        durability_score = self._score_margin_durability(periods, ctx, evidence, flags)
        if durability_score is not None:
            component_scores.append(durability_score)
            component_weights.append(_W_MARGIN_DURABILITY)
            signals_present += 1

        # ------------------------------------------------------------------
        # 3. Growth vs peers / universe
        # ------------------------------------------------------------------
        growth_score = self._score_growth_vs_peers(periods, ctx, evidence, flags)
        if growth_score is not None:
            component_scores.append(growth_score)
            component_weights.append(_W_GROWTH_VS_PEERS)
            signals_present += 1

        # ------------------------------------------------------------------
        # 4. Returns-on-capital level & persistence
        # ------------------------------------------------------------------
        returns_score = self._score_returns_persistence(periods, evidence, flags)
        if returns_score is not None:
            component_scores.append(returns_score)
            component_weights.append(_W_RETURNS_PERSISTENCE)
            signals_present += 1

        # If not a single sub-signal could be computed, fall back to neutral.
        if signals_present == 0:
            return self.neutral_subscore(
                rationale=(
                    "Fundamentals were present but lacked the margin, growth and "
                    "returns-on-capital fields needed to judge competitive position; "
                    "scored neutral with minimal confidence."
                ),
                coverage=0.0,
                extra_flags=["NO_USABLE_FIELDS"],
            )

        # Blend available sub-signals (weights renormalise over what exists).
        blended = weighted_mean(component_scores, component_weights)
        score = clamp(blended if blended is not None else 50.0, 0.0, 100.0)

        # ------------------------------------------------------------------
        # Coverage & confidence — both anchored to how much REAL input existed.
        # ------------------------------------------------------------------
        coverage = signals_present / float(signals_possible)

        # Confidence starts from coverage, is lifted modestly when we have peer
        # context (a relative read is more trustworthy than an absolute band) and
        # when we have multiple history points for durability/persistence, and is
        # capped below 1.0 because four public proxies can never fully capture a
        # moat. This keeps the honesty contract: thin data -> low confidence.
        history_depth = min(len(periods), _HISTORY_WINDOW)
        history_factor = min(history_depth / 4.0, 1.0)  # saturates at 4 periods
        peer_factor = 1.0 if has_peers else 0.7
        confidence = clamp(
            0.25 + 0.45 * coverage * peer_factor + 0.2 * history_factor,
            0.0,
            0.9,
        )

        rationale = self._build_rationale(
            score=score,
            signals_present=signals_present,
            has_peers=has_peers,
        )

        return SubScore(
            category=self.category,
            score=score,
            confidence=confidence,
            weight=self.default_weight,
            rationale=rationale,
            evidence=evidence,
            flags=flags,
            data_coverage=clamp(coverage, 0.0, 1.0),
        )

    # ----------------------------------------------------------------------
    # Sub-signal scorers (each returns an Optional[float] 0..100, or None when
    # the inputs are missing, and appends its own Evidence/flags as a side
    # effect so the caller can blend only what was genuinely measured).
    # ----------------------------------------------------------------------

    def _score_margin_level(
        self,
        latest: FundamentalsPeriod,
        ctx: AnalysisContext,
        evidence: List[Evidence],
        flags: List[str],
    ) -> Optional[float]:
        """Score the latest margin level, relative to peers when available."""
        gm = latest.gross_margin
        om = latest.operating_margin

        # Prefer gross margin (cleanest pricing-power signal); fall back to
        # operating margin if gross is missing.
        if gm is not None:
            metric_name = "gross margin"
            value = gm
            peer_dist = _peer_distribution(ctx, "gross_margin")
            if peer_dist is None:
                peer_dist = _universe_distribution(ctx, "gross_margin")
            band = _GROSS_MARGIN_BAND
        elif om is not None:
            metric_name = "operating margin"
            value = om
            peer_dist = _peer_distribution(ctx, "operating_margin")
            if peer_dist is None:
                peer_dist = _universe_distribution(ctx, "operating_margin")
            band = _OPERATING_MARGIN_BAND
        else:
            return None

        if peer_dist:
            pr = percentile_rank(value, peer_dist)
            score = (pr if pr is not None else 0.5) * 100.0
            direction = "bullish" if score >= 60.0 else "bearish" if score <= 40.0 else "neutral"
            evidence.append(
                Evidence.from_number(
                    f"{metric_name.title()} vs peers (percentile)",
                    score,
                    source=_SOURCE,
                    direction=direction,
                    unit="th pctile",
                    precision=0,
                    detail=(
                        f"{metric_name.title()} of {value * 100:.1f}% ranks in the "
                        f"{score:.0f}th percentile of {len(peer_dist)} peer value(s)."
                    ),
                )
            )
        else:
            scaled = scale_to_score(value, band[0], band[1], higher_is_better=True)
            score = scaled if scaled is not None else 50.0
            direction = "bullish" if score >= 60.0 else "bearish" if score <= 40.0 else "neutral"
            flags.append("MARGIN_LEVEL_ABSOLUTE")
            evidence.append(
                Evidence.from_number(
                    f"{metric_name.title()} (absolute, no peer set)",
                    value * 100.0,
                    source=_SOURCE,
                    direction=direction,
                    unit="%",
                    precision=1,
                    detail=(
                        f"Scored against an absolute {band[0] * 100:.0f}%–"
                        f"{band[1] * 100:.0f}% reference band; no peer distribution "
                        "was available for a relative read."
                    ),
                )
            )
        return clamp(score, 0.0, 100.0)

    def _score_margin_durability(
        self,
        periods: Sequence[FundamentalsPeriod],
        ctx: AnalysisContext,
        evidence: List[Evidence],
        flags: List[str],
    ) -> Optional[float]:
        """Score gross-margin durability: wide *and* stable margins = a moat proxy."""
        gm_series = _series(periods, "gross_margin")
        if not gm_series:
            return None

        latest_gm = gm_series[0]
        # Level component: how wide is the current gross margin?
        peer_dist = _peer_distribution(ctx, "gross_margin")
        if peer_dist is None:
            peer_dist = _universe_distribution(ctx, "gross_margin")
        if peer_dist:
            pr = percentile_rank(latest_gm, peer_dist)
            level_score = (pr if pr is not None else 0.5) * 100.0
        else:
            scaled = scale_to_score(latest_gm, *_GROSS_MARGIN_BAND, higher_is_better=True)
            level_score = scaled if scaled is not None else 50.0

        # Stability component: how steady has the gross margin been over time?
        stability = _stability_score(gm_series)

        if stability is None:
            # Only one period of gross margin — we can speak to level only, and we
            # discount the durability read (we cannot yet see persistence).
            flags.append("MARGIN_HISTORY_THIN")
            durability = level_score
            evidence.append(
                Evidence.from_number(
                    "Gross-margin level (single period; durability unconfirmed)",
                    latest_gm * 100.0,
                    source=_SOURCE,
                    direction=(
                        "bullish" if level_score >= 60.0
                        else "bearish" if level_score <= 40.0
                        else "neutral"
                    ),
                    unit="%",
                    precision=1,
                    detail=(
                        "Only one period of gross margin is available, so stability "
                        "over time could not be confirmed; durability inferred from "
                        "level alone."
                    ),
                )
            )
        else:
            # Durability rewards both a wide margin and a steady one.
            durability = 0.5 * level_score + 0.5 * stability
            mean_gm = sum(gm_series) / len(gm_series)
            evidence.append(
                Evidence.from_number(
                    "Gross-margin durability (level x stability)",
                    durability,
                    source=_SOURCE,
                    direction=(
                        "bullish" if durability >= 60.0
                        else "bearish" if durability <= 40.0
                        else "neutral"
                    ),
                    unit="/100",
                    precision=0,
                    detail=(
                        f"Gross margin averaged {mean_gm * 100:.1f}% across "
                        f"{len(gm_series)} periods with a stability score of "
                        f"{stability:.0f}/100 (steadier margins indicate stronger "
                        "pricing power)."
                    ),
                )
            )
        return clamp(durability, 0.0, 100.0)

    def _score_growth_vs_peers(
        self,
        periods: Sequence[FundamentalsPeriod],
        ctx: AnalysisContext,
        evidence: List[Evidence],
        flags: List[str],
    ) -> Optional[float]:
        """Score revenue growth relative to peers/universe (share-taking proxy)."""
        rev_series = _series(periods, "revenue")
        if len(rev_series) < 2:
            return None
        latest_rev, prior_rev = rev_series[0], rev_series[1]
        if prior_rev is None or prior_rev <= 0.0:
            # Cannot compute a meaningful growth rate off a non-positive base.
            return None
        growth = (latest_rev - prior_rev) / prior_rev

        peer_dist = _peer_distribution(ctx, "revenue_growth", "revenue_growth_yoy")
        if peer_dist is None:
            peer_dist = _universe_distribution(ctx, "revenue_growth", "revenue_growth_yoy")

        if peer_dist:
            pr = percentile_rank(growth, peer_dist)
            score = (pr if pr is not None else 0.5) * 100.0
            direction = "bullish" if score >= 60.0 else "bearish" if score <= 40.0 else "neutral"
            evidence.append(
                Evidence.from_number(
                    "Revenue growth vs peers (percentile)",
                    score,
                    source=_SOURCE,
                    direction=direction,
                    unit="th pctile",
                    precision=0,
                    detail=(
                        f"Latest revenue growth of {growth * 100:.1f}% ranks in the "
                        f"{score:.0f}th percentile of {len(peer_dist)} peer value(s); "
                        "out-growing peers is evidence of share gains."
                    ),
                )
            )
        else:
            scaled = scale_to_score(growth, *_REVENUE_GROWTH_BAND, higher_is_better=True)
            score = scaled if scaled is not None else 50.0
            direction = "bullish" if score >= 60.0 else "bearish" if score <= 40.0 else "neutral"
            flags.append("GROWTH_ABSOLUTE")
            evidence.append(
                Evidence.from_number(
                    "Revenue growth (absolute, no peer set)",
                    growth * 100.0,
                    source=_SOURCE,
                    direction=direction,
                    unit="%",
                    precision=1,
                    detail=(
                        "Scored against an absolute reference band; no peer growth "
                        "distribution was available for a relative read."
                    ),
                )
            )
        return clamp(score, 0.0, 100.0)

    def _score_returns_persistence(
        self,
        periods: Sequence[FundamentalsPeriod],
        evidence: List[Evidence],
        flags: List[str],
    ) -> Optional[float]:
        """Score the level and persistence of returns on capital (ROIC, else ROE)."""
        roic_series = _series(periods, "roic")
        metric_name = "ROIC"
        band = _ROIC_BAND
        if not roic_series:
            roic_series = _series(periods, "roe")
            metric_name = "ROE"
            # ROE band is comparable enough to reuse the ROIC band as an anchor.
        if not roic_series:
            return None

        latest_ret = roic_series[0]
        # Level: how high is the current return on capital?
        level_score = scale_to_score(latest_ret, *band, higher_is_better=True)
        level_score = level_score if level_score is not None else 50.0

        # Persistence: how often was the return positive across the window?
        persistence = _persistence_fraction(roic_series, threshold=0.0)
        if persistence is None:
            persistence = 1.0 if latest_ret > 0.0 else 0.0

        if len(roic_series) < 2:
            flags.append("RETURNS_HISTORY_THIN")

        # Blend level (how much value created) with persistence (how reliably).
        score = clamp(0.6 * level_score + 0.4 * persistence * 100.0, 0.0, 100.0)
        direction = "bullish" if score >= 60.0 else "bearish" if score <= 40.0 else "neutral"
        evidence.append(
            Evidence.from_number(
                f"{metric_name} level & persistence",
                score,
                source=_SOURCE,
                direction=direction,
                unit="/100",
                precision=0,
                detail=(
                    f"Latest {metric_name} of {latest_ret * 100:.1f}% with positive "
                    f"returns in {persistence * 100:.0f}% of {len(roic_series)} "
                    "available period(s); persistent high returns on capital "
                    "indicate a compounding advantage."
                ),
            )
        )
        return score

    # ----------------------------------------------------------------------
    # Rationale builder
    # ----------------------------------------------------------------------

    @staticmethod
    def _build_rationale(*, score: float, signals_present: int, has_peers: bool) -> str:
        """Compose a short, honest, human-readable rationale string."""
        if score >= 70.0:
            stance = "a strong, defensible competitive position"
        elif score >= 55.0:
            stance = "an above-average competitive position"
        elif score >= 45.0:
            stance = "an average / unremarkable competitive position"
        elif score >= 30.0:
            stance = "a below-average competitive position"
        else:
            stance = "a weak, contestable competitive position"

        basis = (
            "benchmarked against peers"
            if has_peers
            else "scored on absolute reference bands (no peer set supplied)"
        )
        return (
            f"Competitive read indicates {stance} (score {score:.0f}/100), "
            f"derived from {signals_present} of 4 positioning sub-signals "
            f"(margin level, gross-margin durability, growth vs peers, "
            f"returns-on-capital persistence), {basis}. One independent input "
            "among many — not a standalone verdict."
        )


__all__ = ["CompetitiveAnalyzer"]
