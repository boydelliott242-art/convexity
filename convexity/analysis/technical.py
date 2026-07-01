"""Technical analyzer — price-structure evidence for one security.

This module is part of Convexity, an evidence-driven equity **research and
screening** tool. It is **not** a predictor and **not** investment advice. The
:class:`TechnicalAnalyzer` reads a security's *historical* price/volume bars and
turns them into a single, auditable :class:`~convexity.core.models.SubScore` for
:data:`~convexity.core.models.ScoreCategory.TECHNICAL`. Technical structure is
only *one* of many independent evidence categories; on its own it asserts
nothing about future returns. Convexity earns conviction only when this signal
agrees with many other, independent signals.

What "technical evidence" means here
------------------------------------
Higher TECHNICAL scores describe a **constructive price structure** — the kind a
patient researcher would flag as "worth a closer fundamental look", never a buy
signal:

* **Trend** — price above a rising 50- and 200-day simple moving average, with a
  positive slope, is constructive; price below falling averages is not.
* **Structure** — proximity to the 52-week high (and distance above the 52-week
  low) plus a recent pattern of higher highs / higher lows signals an
  established uptrend rather than a broken-down chart.
* **Volatility** — a *moderate* Average True Range (ATR) relative to price is
  healthier than a chaotic, high-volatility tape; extreme volatility lowers both
  the score and our confidence in it.
* **Volume behaviour** — rising participation (recent volume above its longer
  average) confirms a move; fading volume is a caution.

Each facet is scored 0–100 with the shared, pure helpers in
:mod:`convexity.core.scoring`, blended into a composite, and every contributing
number is emitted as a piece of :class:`~convexity.core.models.Evidence` so the
score is fully auditable.

Honesty rules honoured
----------------------
* **Pure & deterministic.** ``analyze`` performs no I/O, reads no wall-clock and
  uses no randomness; it operates only on the passed
  :class:`~convexity.core.models.SecurityData`. Identical input ⇒ identical
  output.
* **Missing data is never fabricated.** With too little price history to compute
  anything meaningful, the analyzer returns
  :meth:`~convexity.core.contracts.Analyzer.neutral_subscore` (score 50, low
  confidence, ``MISSING_DATA`` flag). Where only *some* facets can be computed,
  ``data_coverage`` and ``confidence`` fall accordingly and the gap is flagged.
* **Relative when possible.** When ``ctx.peer_stats`` / ``ctx.universe_stats``
  carry a distribution of a comparable metric (e.g. ``"distance_from_52w_high"``
  or ``"atr_pct"``), the analyzer scores the security's value by its percentile
  within that distribution; otherwise it degrades gracefully to sensible
  absolute bands.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from convexity.core.contracts import AnalysisContext, Analyzer
from convexity.core.models import Evidence, ScoreCategory, SubScore
from convexity.core.registry import register_analyzer
from convexity.core.scoring import clamp, logistic_score, percentile_rank, scale_to_score

if TYPE_CHECKING:  # pragma: no cover - typing only.
    from convexity.core.models import PriceBar, SecurityData


# ---------------------------------------------------------------------------
# Tunable constants (documented so every threshold is auditable)
# ---------------------------------------------------------------------------

# Minimum bars required to attempt *any* technical read at all. Below this we
# have no honest structure to describe and fall back to a neutral sub-score.
_MIN_BARS_FOR_ANY_SIGNAL = 20

# Lookbacks (in trading days). The 252-day window approximates one year.
_SMA_SHORT = 50
_SMA_LONG = 200
_ATR_WINDOW = 14
_VOLUME_SHORT = 20
_VOLUME_LONG = 60
_YEAR_BARS = 252
_SLOPE_WINDOW = 20          # window for the short-MA slope (recent trend).
_SWING_WINDOW = 60          # window over which to look for higher highs/lows.

# A "full coverage" technical read can compute all of these facets. Coverage is
# the fraction of these that we could actually evaluate from the data on hand.
_FACETS: Tuple[str, ...] = ("trend", "structure", "volatility", "volume")


# ---------------------------------------------------------------------------
# Reusable, pure indicator math (numpy-based)
# ---------------------------------------------------------------------------
#
# These functions are deliberately small, side-effect-free and independently
# testable. They take plain float sequences (oldest-first, matching
# ``SecurityData.price_history``) and return plain floats / ``None`` so callers
# never have to reason about numpy edge cases. ``None`` always means "not enough
# data to compute this honestly" — it is never silently coerced to a number.


def simple_moving_average(values: Sequence[float], window: int) -> Optional[float]:
    """Return the simple moving average of the last ``window`` values.

    Args:
        values: Oldest-first numeric series (e.g. closing prices).
        window: Number of trailing observations to average; must be positive.

    Returns:
        The mean of the final ``window`` values, or ``None`` if there are fewer
        than ``window`` finite values available.
    """
    if window <= 0:
        return None
    arr = _finite(values)
    if arr.size < window:
        return None
    return float(np.mean(arr[-window:]))


def sma_slope_pct(values: Sequence[float], window: int, slope_window: int) -> Optional[float]:
    """Return the recent slope of a moving average as a per-bar percentage.

    Computes the ``window``-bar SMA at each point over the final
    ``slope_window`` bars and fits a least-squares line to it, expressing the
    slope as a fraction of the latest SMA level (so it is scale-free across
    securities). A positive value means the trend line is rising.

    Args:
        values: Oldest-first numeric series.
        window: SMA window.
        slope_window: How many of the most recent SMA points to regress.

    Returns:
        Per-bar slope as a fraction of the latest SMA (e.g. ``0.001`` == +0.1%
        per bar), or ``None`` if there is insufficient data or the SMA level is
        non-positive.
    """
    arr = _finite(values)
    needed = window + slope_window - 1
    if arr.size < needed or slope_window < 2:
        return None
    smas: List[float] = []
    for end in range(arr.size - slope_window + 1, arr.size + 1):
        smas.append(float(np.mean(arr[end - window:end])))
    sma_arr = np.asarray(smas, dtype=float)
    level = sma_arr[-1]
    if level <= 0:
        return None
    x = np.arange(sma_arr.size, dtype=float)
    # Least-squares slope (deterministic, closed-form via numpy.polyfit deg=1).
    slope = float(np.polyfit(x, sma_arr, 1)[0])
    return slope / level


def average_true_range(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    window: int = _ATR_WINDOW,
) -> Optional[float]:
    """Return the Wilder-style Average True Range over ``window`` bars.

    True range for a bar is ``max(high-low, |high-prev_close|,
    |low-prev_close|)``. The ATR is the simple mean of the final ``window`` true
    ranges (a transparent, non-recursive variant so the result is fully
    reproducible).

    Returns:
        The ATR in price units, or ``None`` if fewer than ``window + 1`` aligned
        bars are available.
    """
    h = np.asarray(highs, dtype=float)
    low = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    n = min(h.size, low.size, c.size)
    if n < window + 1:
        return None
    h, low, c = h[-n:], low[-n:], c[-n:]
    prev_close = c[:-1]
    cur_high = h[1:]
    cur_low = low[1:]
    tr = np.maximum.reduce(
        [
            cur_high - cur_low,
            np.abs(cur_high - prev_close),
            np.abs(cur_low - prev_close),
        ]
    )
    if tr.size < window:
        return None
    atr = float(np.mean(tr[-window:]))
    return atr if np.isfinite(atr) else None


def higher_highs_and_lows(highs: Sequence[float], lows: Sequence[float], window: int) -> Optional[float]:
    """Return a 0–1 "uptrend structure" score from swing highs/lows.

    Splits the final ``window`` bars into two halves and checks whether the more
    recent half made both a higher high and a higher low than the earlier half —
    the textbook definition of an uptrend. Returns a graded score: 1.0 for both
    higher high *and* higher low, 0.5 for one of the two, 0.0 for neither.

    Returns:
        A float in ``[0, 1]`` or ``None`` if there is not enough data.
    """
    h = _finite(highs)
    low = _finite(lows)
    n = min(h.size, low.size)
    if n < max(window, 4):
        return None
    h = h[-window:]
    low = low[-window:]
    half = h.size // 2
    if half < 1:
        return None
    earlier_high = float(np.max(h[:half]))
    recent_high = float(np.max(h[half:]))
    earlier_low = float(np.min(low[:half]))
    recent_low = float(np.min(low[half:]))
    score = 0.0
    if recent_high > earlier_high:
        score += 0.5
    if recent_low > earlier_low:
        score += 0.5
    return score


def distance_from_extreme(
    closes: Sequence[float],
    window: int = _YEAR_BARS,
) -> Optional[Tuple[float, float]]:
    """Return ``(pct_below_high, pct_above_low)`` over the trailing ``window``.

    Both are non-negative fractions of the latest close:

    * ``pct_below_high`` — how far the latest close sits *below* the window high
      (0.0 == at a new high).
    * ``pct_above_low`` — how far the latest close sits *above* the window low
      (0.0 == at a new low).

    Returns ``None`` if there is no usable price data or the latest close is
    non-positive.
    """
    arr = _finite(closes)
    if arr.size == 0:
        return None
    window_arr = arr[-window:] if arr.size > window else arr
    last = float(arr[-1])
    if last <= 0:
        return None
    hi = float(np.max(window_arr))
    low = float(np.min(window_arr))
    pct_below_high = (hi - last) / last if hi > 0 else 0.0
    pct_above_low = (last - low) / last if last > 0 else 0.0
    return max(pct_below_high, 0.0), max(pct_above_low, 0.0)


def volume_trend_ratio(volumes: Sequence[float], short: int, long: int) -> Optional[float]:
    """Return recent-vs-baseline volume participation as a ratio.

    Ratio of the mean of the last ``short`` bars' volume to the mean of the last
    ``long`` bars' volume. A value > 1 means participation is rising into the
    current move; < 1 means it is fading. Returns ``None`` with insufficient
    data or a non-positive baseline.
    """
    arr = _finite(volumes)
    if arr.size < long or short <= 0 or long <= 0:
        return None
    recent = float(np.mean(arr[-short:]))
    baseline = float(np.mean(arr[-long:]))
    if baseline <= 0:
        return None
    return recent / baseline


def _finite(values: Sequence[float]) -> np.ndarray:
    """Return a 1-D float array with non-finite entries (NaN/inf) removed."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


# ---------------------------------------------------------------------------
# The analyzer
# ---------------------------------------------------------------------------


@register_analyzer
class TechnicalAnalyzer(Analyzer):
    """Score the TECHNICAL category from a security's price/volume history.

    The analyzer blends four sub-facets — trend, structure, volatility and
    volume — each scored 0–100, into a single composite. Constructive uptrends
    and tight, orderly bases score high; broken-down, high-volatility charts
    score low. Confidence and ``data_coverage`` track how many facets could be
    computed and how much history backed them, so a thin chart yields an honest,
    low-confidence read rather than a falsely precise one.

    Class attributes:
        category: :data:`~convexity.core.models.ScoreCategory.TECHNICAL`.
        default_weight: Matches ``DEFAULT_CATEGORY_WEIGHTS`` (technical evidence
            is supporting, not primary, so its weight is small).
        requires: The single input it needs — ``price_history``.
    """

    category = ScoreCategory.TECHNICAL
    default_weight = 0.04
    requires: Set[str] = {"price_history"}

    # Source label recorded on every Evidence item produced here.
    _SOURCE = "price_history"

    def analyze(self, data: SecurityData, ctx: AnalysisContext) -> SubScore:
        """Return the TECHNICAL :class:`SubScore` for ``data``.

        Args:
            data: The assembled security data. Only ``price_history`` (OHLCV bars,
                oldest-first) is read.
            ctx: Comparative context; ``peer_stats``/``universe_stats`` are used
                for relative scoring of structure/volatility when present.

        Returns:
            A validated :class:`SubScore` (score clamped to 0–100) with evidence,
            flags, a human rationale, a confidence and a ``data_coverage``
            fraction. Falls back to :meth:`neutral_subscore` when history is too
            short to read.
        """
        bars: List[PriceBar] = list(data.price_history)
        n_bars = len(bars)

        if n_bars < _MIN_BARS_FOR_ANY_SIGNAL:
            return self.neutral_subscore(
                rationale=(
                    f"Only {n_bars} price bar(s) available "
                    f"(need >= {_MIN_BARS_FOR_ANY_SIGNAL}); insufficient history "
                    "to read technical structure."
                ),
                coverage=0.0,
                extra_flags=["SHORT_HISTORY"],
            )

        # price_history is OLDEST-FIRST per the contract; extract aligned series.
        closes = [float(b.close) for b in bars]
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        volumes = [float(b.volume) for b in bars]
        last_close = closes[-1]

        evidence: List[Evidence] = []
        flags: List[str] = []
        facet_scores: Dict[str, float] = {}
        facet_weights: Dict[str, float] = {}

        # --- 1. Trend -------------------------------------------------------
        trend_score = self._score_trend(
            closes=closes, last_close=last_close, n_bars=n_bars, evidence=evidence, flags=flags
        )
        if trend_score is not None:
            facet_scores["trend"] = trend_score
            facet_weights["trend"] = 0.40  # the dominant facet.

        # --- 2. Structure ---------------------------------------------------
        structure_score = self._score_structure(
            closes=closes,
            highs=highs,
            lows=lows,
            ctx=ctx,
            evidence=evidence,
            flags=flags,
        )
        if structure_score is not None:
            facet_scores["structure"] = structure_score
            facet_weights["structure"] = 0.30

        # --- 3. Volatility --------------------------------------------------
        volatility_score = self._score_volatility(
            highs=highs,
            lows=lows,
            closes=closes,
            last_close=last_close,
            ctx=ctx,
            evidence=evidence,
            flags=flags,
        )
        if volatility_score is not None:
            facet_scores["volatility"] = volatility_score
            facet_weights["volatility"] = 0.15

        # --- 4. Volume ------------------------------------------------------
        volume_score = self._score_volume(volumes=volumes, evidence=evidence, flags=flags)
        if volume_score is not None:
            facet_scores["volume"] = volume_score
            facet_weights["volume"] = 0.15

        # --- Blend ----------------------------------------------------------
        if not facet_scores:
            # We had >= _MIN_BARS_FOR_ANY_SIGNAL bars but somehow could not score
            # a single facet (e.g. all-equal degenerate data). Stay honest.
            return self.neutral_subscore(
                rationale=(
                    "Price history was present but too degenerate to compute any "
                    "technical facet (trend, structure, volatility, volume)."
                ),
                coverage=0.0,
                extra_flags=["DEGENERATE_PRICES"],
            )

        total_w = sum(facet_weights[name] for name in facet_scores)
        composite = sum(facet_scores[name] * facet_weights[name] for name in facet_scores) / total_w
        composite = clamp(composite, 0.0, 100.0)

        # --- Coverage & confidence -----------------------------------------
        # Coverage = fraction of the four facets we could compute.
        data_coverage = len(facet_scores) / float(len(_FACETS))
        confidence = self._confidence(n_bars=n_bars, facet_count=len(facet_scores), flags=flags)

        rationale = self._build_rationale(
            facet_scores=facet_scores, composite=composite, n_bars=n_bars
        )

        return SubScore(
            category=self.category,
            score=composite,
            confidence=clamp(confidence, 0.0, 1.0),
            weight=self.default_weight,
            rationale=rationale,
            evidence=evidence,
            flags=flags,
            data_coverage=clamp(data_coverage, 0.0, 1.0),
        )

    # ------------------------------------------------------------------
    # Facet scorers (each returns 0..100 or None, and appends evidence)
    # ------------------------------------------------------------------

    def _score_trend(
        self,
        *,
        closes: List[float],
        last_close: float,
        n_bars: int,
        evidence: List[Evidence],
        flags: List[str],
    ) -> Optional[float]:
        """Score trend from price vs SMA50/SMA200 and the short-MA slope."""
        sma_short = simple_moving_average(closes, _SMA_SHORT)
        sma_long = simple_moving_average(closes, _SMA_LONG)
        slope = sma_slope_pct(closes, _SMA_SHORT, _SLOPE_WINDOW)

        if sma_short is None and slope is None:
            # Not even a 50-day average is computable: no honest trend read.
            return None

        components: List[float] = []

        # (a) Price vs short MA: how far above/below, mapped through a band of
        # +/-15%. Above the average is constructive.
        if sma_short is not None and sma_short > 0:
            pct_vs_short = (last_close - sma_short) / sma_short
            s = scale_to_score(pct_vs_short, lo=-0.15, hi=0.15, higher_is_better=True)
            if s is not None:
                components.append(s)
                direction = (
                    "bullish" if pct_vs_short > 0.01 else "bearish" if pct_vs_short < -0.01 else "neutral"
                )
                evidence.append(
                    Evidence.from_number(
                        "Price vs 50-day SMA",
                        pct_vs_short * 100.0,
                        source=self._SOURCE,
                        direction=direction,
                        unit="%",
                        precision=1,
                        detail=f"close {last_close:,.2f} vs SMA50 {sma_short:,.2f}",
                    )
                )

        # (b) Price vs long MA (only when 200 bars exist): a primary regime gauge.
        if sma_long is not None and sma_long > 0:
            pct_vs_long = (last_close - sma_long) / sma_long
            s = scale_to_score(pct_vs_long, lo=-0.25, hi=0.25, higher_is_better=True)
            if s is not None:
                components.append(s)
                direction = (
                    "bullish" if pct_vs_long > 0.01 else "bearish" if pct_vs_long < -0.01 else "neutral"
                )
                evidence.append(
                    Evidence.from_number(
                        "Price vs 200-day SMA",
                        pct_vs_long * 100.0,
                        source=self._SOURCE,
                        direction=direction,
                        unit="%",
                        precision=1,
                        detail=f"close {last_close:,.2f} vs SMA200 {sma_long:,.2f}",
                    )
                )
            # Golden/death-cross context (50 above/below 200) as a discrete flag.
            if sma_short is not None:
                if sma_short > sma_long:
                    flags.append("SMA50_ABOVE_SMA200")
                else:
                    flags.append("SMA50_BELOW_SMA200")
        else:
            flags.append("NO_200D_HISTORY")

        # (c) Slope of the short MA: a rising average confirms the trend's
        # health. Map per-bar slope through a logistic centred at zero.
        if slope is not None:
            # +-0.2% per bar is a meaningful daily drift; steepness scales it.
            s = logistic_score(slope, midpoint=0.0, steepness=700.0)
            if s is not None:
                components.append(s)
                direction = (
                    "bullish" if slope > 0.0002 else "bearish" if slope < -0.0002 else "neutral"
                )
                evidence.append(
                    Evidence.from_number(
                        "50-day SMA slope (per bar)",
                        slope * 100.0,
                        source=self._SOURCE,
                        direction=direction,
                        unit="%",
                        precision=3,
                        detail="least-squares slope of SMA50 over the last 20 bars",
                    )
                )

        if not components:
            return None
        return clamp(sum(components) / len(components), 0.0, 100.0)

    def _score_structure(
        self,
        *,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        ctx: AnalysisContext,
        evidence: List[Evidence],
        flags: List[str],
    ) -> Optional[float]:
        """Score structure: 52w-high/low position and higher-high/-low pattern."""
        dist = distance_from_extreme(closes, _YEAR_BARS)
        swing = higher_highs_and_lows(highs, lows, _SWING_WINDOW)

        if dist is None and swing is None:
            return None

        components: List[float] = []

        if dist is not None:
            pct_below_high, pct_above_low = dist

            # (a) Distance below the 52-week high. Closer is more constructive.
            # Prefer a peer/universe-relative read when a distribution exists.
            rel = self._relative_score(
                value=pct_below_high,
                ctx=ctx,
                key="distance_from_52w_high",
                higher_is_better=False,
            )
            if rel is not None:
                s_high = rel
            else:
                # Absolute band: at-high (0%) -> 100, 50%+ below -> 0.
                s_high = scale_to_score(pct_below_high, lo=0.0, hi=0.50, higher_is_better=False)
            if s_high is not None:
                components.append(s_high)
                direction = (
                    "bullish" if pct_below_high < 0.10 else "bearish" if pct_below_high > 0.30 else "neutral"
                )
                evidence.append(
                    Evidence.from_number(
                        "Distance below 52-week high",
                        pct_below_high * 100.0,
                        source=self._SOURCE,
                        direction=direction,
                        unit="%",
                        precision=1,
                        detail="0% = at a new 52-week high",
                    )
                )
                if pct_below_high < 0.05:
                    flags.append("NEAR_52W_HIGH")

            # (b) Distance above the 52-week low. Far above a low is healthier
            # than sitting on it. A moderate band: at-low -> 0, 100%+ above -> 100.
            s_low = scale_to_score(pct_above_low, lo=0.0, hi=1.00, higher_is_better=True)
            if s_low is not None:
                components.append(s_low)
                direction = (
                    "bearish" if pct_above_low < 0.05 else "bullish" if pct_above_low > 0.30 else "neutral"
                )
                evidence.append(
                    Evidence.from_number(
                        "Distance above 52-week low",
                        pct_above_low * 100.0,
                        source=self._SOURCE,
                        direction=direction,
                        unit="%",
                        precision=1,
                        detail="0% = at a new 52-week low",
                    )
                )
                if pct_above_low < 0.05:
                    flags.append("NEAR_52W_LOW")

        if swing is not None:
            # swing in {0.0, 0.5, 1.0}; map directly onto 0..100.
            s_swing = clamp(swing * 100.0, 0.0, 100.0)
            components.append(s_swing)
            direction = "bullish" if swing >= 1.0 else "bearish" if swing <= 0.0 else "neutral"
            evidence.append(
                Evidence.from_number(
                    "Higher-high / higher-low structure",
                    swing,
                    source=self._SOURCE,
                    direction=direction,
                    precision=1,
                    detail="1.0 = both a higher high and a higher low over the recent window",
                )
            )

        if not components:
            return None
        return clamp(sum(components) / len(components), 0.0, 100.0)

    def _score_volatility(
        self,
        *,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        last_close: float,
        ctx: AnalysisContext,
        evidence: List[Evidence],
        flags: List[str],
    ) -> Optional[float]:
        """Score volatility: a moderate ATR% is healthiest; extremes penalised."""
        atr = average_true_range(highs, lows, closes, _ATR_WINDOW)
        if atr is None or last_close <= 0:
            return None
        atr_pct = atr / last_close  # ATR as a fraction of price.

        # Prefer a relative read: lower ATR% than peers is calmer (better).
        rel = self._relative_score(
            value=atr_pct,
            ctx=ctx,
            key="atr_pct",
            higher_is_better=False,
        )
        if rel is not None:
            score = rel
        else:
            # Absolute band: ~1%/day ATR is calm (-> ~100), ~8%/day is chaotic
            # (-> ~0). Lower is better.
            score = scale_to_score(atr_pct, lo=0.01, hi=0.08, higher_is_better=False)
        if score is None:
            return None

        direction = "bullish" if atr_pct < 0.03 else "bearish" if atr_pct > 0.06 else "neutral"
        evidence.append(
            Evidence.from_number(
                "ATR(14) as % of price",
                atr_pct * 100.0,
                source=self._SOURCE,
                direction=direction,
                unit="%",
                precision=2,
                detail="lower = calmer, more orderly tape",
            )
        )
        if atr_pct > 0.08:
            flags.append("HIGH_VOLATILITY")
        return clamp(score, 0.0, 100.0)

    def _score_volume(
        self,
        *,
        volumes: List[float],
        evidence: List[Evidence],
        flags: List[str],
    ) -> Optional[float]:
        """Score volume behaviour: rising participation confirms a move."""
        ratio = volume_trend_ratio(volumes, _VOLUME_SHORT, _VOLUME_LONG)
        if ratio is None:
            return None
        # ratio of 1.0 (recent == baseline) is neutral; >1 confirms, <1 fades.
        # Logistic centred at 1.0 so a 1.5x pickup scores well above midpoint.
        score = logistic_score(ratio, midpoint=1.0, steepness=2.5)
        if score is None:
            return None
        direction = "bullish" if ratio > 1.1 else "bearish" if ratio < 0.9 else "neutral"
        evidence.append(
            Evidence.from_number(
                "Recent vs baseline volume",
                ratio,
                source=self._SOURCE,
                direction=direction,
                unit="x",
                precision=2,
                detail=f"mean {_VOLUME_SHORT}-bar volume / mean {_VOLUME_LONG}-bar volume",
            )
        )
        if ratio > 1.5:
            flags.append("VOLUME_EXPANSION")
        elif ratio < 0.6:
            flags.append("VOLUME_CONTRACTION")
        return clamp(score, 0.0, 100.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _relative_score(
        *,
        value: Optional[float],
        ctx: AnalysisContext,
        key: str,
        higher_is_better: bool,
    ) -> Optional[float]:
        """Score ``value`` by its percentile within a peer/universe distribution.

        Looks for ``key`` first in ``ctx.peer_stats`` then ``ctx.universe_stats``;
        the value there must be an iterable distribution of comparable numbers.
        Returns a 0–100 score (inverted when ``higher_is_better`` is False), or
        ``None`` when no usable distribution is present so the caller can fall
        back to an absolute band.
        """
        if value is None:
            return None
        dist = _lookup_distribution(ctx, key)
        if not dist:
            return None
        pr = percentile_rank(value, dist)
        if pr is None:
            return None
        if not higher_is_better:
            pr = 1.0 - pr
        return clamp(pr * 100.0, 0.0, 100.0)

    @staticmethod
    def _confidence(*, n_bars: int, facet_count: int, flags: List[str]) -> float:
        """Confidence from history length, facets computed, and any cautions.

        More history and more independent facets ⇒ higher confidence. Short
        history (no 200-day average) and explicit caution flags trim it.
        """
        # Base on how much of a full year of history we have (caps at 1.0).
        history_term = clamp(n_bars / float(_YEAR_BARS), 0.0, 1.0)
        # Reward having computed more of the four facets.
        facet_term = facet_count / float(len(_FACETS))
        confidence = 0.25 + 0.45 * history_term + 0.30 * facet_term
        if "NO_200D_HISTORY" in flags:
            confidence *= 0.85
        if "HIGH_VOLATILITY" in flags:
            confidence *= 0.90
        return clamp(confidence, 0.0, 1.0)

    @staticmethod
    def _build_rationale(*, facet_scores: Dict[str, float], composite: float, n_bars: int) -> str:
        """Compose a short, human, evidence-anchored rationale string."""
        parts: List[str] = []
        if "trend" in facet_scores:
            parts.append(f"trend {facet_scores['trend']:.0f}/100")
        if "structure" in facet_scores:
            parts.append(f"structure {facet_scores['structure']:.0f}/100")
        if "volatility" in facet_scores:
            parts.append(f"volatility {facet_scores['volatility']:.0f}/100")
        if "volume" in facet_scores:
            parts.append(f"volume {facet_scores['volume']:.0f}/100")
        facet_str = ", ".join(parts) if parts else "no facets"
        tone = (
            "constructive" if composite >= 60 else "weak/broken" if composite < 40 else "mixed/neutral"
        )
        return (
            f"Technical structure scores {composite:.0f}/100 ({tone}) over {n_bars} bars: "
            f"{facet_str}. Price-only evidence — one input among many, not a forecast."
        )


def _lookup_distribution(ctx: AnalysisContext, key: str) -> List[float]:
    """Return a clean float distribution for ``key`` from peer/universe stats.

    Searches ``ctx.peer_stats`` first (closer comparables), then
    ``ctx.universe_stats``. Accepts either a raw iterable of numbers or a mapping
    exposing a ``"values"`` / ``"distribution"`` iterable. Non-finite and
    non-numeric entries are dropped. Returns an empty list when nothing usable is
    found (the caller then degrades to absolute scoring).
    """
    for stats in (getattr(ctx, "peer_stats", None), getattr(ctx, "universe_stats", None)):
        if not stats or key not in stats:
            continue
        raw = stats[key]
        if isinstance(raw, dict):
            raw = raw.get("values") or raw.get("distribution")
        if raw is None:
            continue
        out: List[float] = []
        try:
            for v in raw:
                if v is None:
                    continue
                fv = float(v)
                if fv == fv and fv not in (float("inf"), float("-inf")):  # drop NaN/inf
                    out.append(fv)
        except (TypeError, ValueError):
            continue
        if out:
            return out
    return []


__all__ = [
    "TechnicalAnalyzer",
    "simple_moving_average",
    "sma_slope_pct",
    "average_true_range",
    "higher_highs_and_lows",
    "distance_from_extreme",
    "volume_trend_ratio",
]
