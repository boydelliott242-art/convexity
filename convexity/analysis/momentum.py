"""Momentum analyzer — absolute and relative trend strength as auditable evidence.

This module is part of Convexity, an evidence-driven equity **research and
screening** tool. It is **not** a predictor and **not** investment advice. The
:class:`MomentumAnalyzer` turns a security's price history into one transparent,
0–100 sub-score for the :data:`~convexity.core.models.ScoreCategory.MOMENTUM`
category. Momentum is only *one* of many independent pieces of evidence; high
conviction is justified only when many such signals agree, never on trend alone.

What "momentum" means here
--------------------------
Momentum is the empirically-documented tendency for recent relative strength to
persist over intermediate horizons. We measure it the way the academic
literature does — and report every input so a reader can audit it:

* **Trailing total returns** over 1-, 3-, 6- and 12-month windows.
* **12–1 momentum** — the classic Jegadeesh–Titman measure: the 12-month return
  that *excludes the most recent month*, which strips out short-term reversal
  noise and is the horizon momentum is strongest at.
* **RSI(14)** — Wilder's Relative Strength Index, a bounded 0–100 oscillator
  describing how one-sided recent up/down moves have been.
* **MACD posture** — the sign and slope of the 12/26 EMA difference versus its
  9-period signal line: a simple, transparent trend-direction read.
* **Relative strength** — where this name's 12–1 momentum sits within the
  *screened universe's* distribution (via ``ctx.universe_stats``), so a name is
  scored against its actual opportunity set, not an absolute threshold.

Honesty constraints (non-negotiable)
------------------------------------
* **Pure & deterministic.** No I/O, no wall-clock, no randomness. The analyzer
  operates only on the passed :class:`~convexity.core.models.SecurityData`.
* **Never fabricate.** With too little price history we return
  :meth:`~convexity.core.contracts.Analyzer.neutral_subscore` (score 50, low
  confidence, ``MISSING_DATA`` flag) so a data gap neither helps nor hurts.
  Partial history lowers ``data_coverage`` and ``confidence`` honestly.
* **Reward persistence, punish blow-offs.** Steady, broad-based positive
  momentum scores well; a parabolic one-month spike or an extreme overbought
  reading is flagged and *discounted*, because blow-off extremes are fragile,
  not durable — this is a quality control on the signal, not a price forecast.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional

from convexity.core.contracts import AnalysisContext, Analyzer
from convexity.core.models import Evidence, PriceBar, ScoreCategory, SecurityData, SubScore
from convexity.core.registry import register_analyzer
from convexity.core.scoring import clamp, percentile_rank, scale_to_score

# Trading-day approximations for each lookback window (≈21 sessions per month).
_DAYS_1M = 21
_DAYS_3M = 63
_DAYS_6M = 126
_DAYS_12M = 252

# Wilder RSI period.
_RSI_PERIOD = 14

# MACD EMA spans.
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9

# Minimum bars to compute anything meaningful (need a 3-month look-back at least).
_MIN_BARS = _DAYS_3M + 1

# Horizon weights for blending absolute trailing returns into a sub-score. The
# 12–1 (intermediate) horizon carries the most weight because that is where the
# momentum effect is strongest and least contaminated by short-term reversal;
# the 1-month horizon carries the least (it is the noisiest and most prone to
# mean-reversion). Weights are renormalised over whichever horizons are present.
_HORIZON_WEIGHTS: Dict[str, float] = {
    "mom_12_1": 0.40,
    "ret_6m": 0.25,
    "ret_3m": 0.20,
    "ret_1m": 0.15,
}

# Calibration bands for mapping a trailing return onto a 0–100 score. A return at
# or below ``lo`` maps to 0, at or above ``hi`` maps to 100; the band widens with
# the horizon because larger moves are expected over longer windows. These are
# deliberately wide so ordinary names land mid-scale and only genuine outliers
# saturate the ends.
_RETURN_BANDS: Dict[str, tuple] = {
    "ret_1m": (-0.20, 0.20),
    "ret_3m": (-0.30, 0.35),
    "ret_6m": (-0.40, 0.50),
    "mom_12_1": (-0.50, 0.75),
}


def _series_closes(bars: Sequence[PriceBar]) -> List[float]:
    """Return the (oldest-first) adjusted-close series, falling back to ``close``.

    ``adj_close`` is preferred because it folds in splits/dividends, which is
    essential for honest multi-month returns; when a provider omits it we use the
    raw ``close``. Non-positive prices are dropped (they cannot yield a valid
    return and would only be data corruption).
    """
    out: List[float] = []
    for bar in bars:
        px = bar.adj_close if bar.adj_close is not None else bar.close
        if px is not None and px > 0:
            out.append(float(px))
    return out


def _trailing_return(closes: Sequence[float], lookback: int) -> Optional[float]:
    """Total return over the last ``lookback`` sessions, or ``None`` if too short.

    Computed as ``last / closes[-1-lookback] - 1`` so a 21-session look-back is a
    genuine one-month total return. Returns ``None`` when the series is too short
    or the reference price is non-positive (never fabricated).
    """
    if lookback <= 0 or len(closes) < lookback + 1:
        return None
    start = closes[-1 - lookback]
    end = closes[-1]
    if start <= 0:
        return None
    return end / start - 1.0


def _mom_12_1(closes: Sequence[float]) -> Optional[float]:
    """The 12–1 momentum: 12-month return excluding the most recent month.

    Defined as ``P_{t-1m} / P_{t-12m} - 1`` — i.e. the return from twelve months
    ago up to *one month ago*. Skipping the last month removes short-term
    reversal, which is why this is the canonical academic momentum horizon.
    Returns ``None`` without ~12 months of history.
    """
    if len(closes) < _DAYS_12M + 1:
        return None
    p_12m_ago = closes[-1 - _DAYS_12M]
    p_1m_ago = closes[-1 - _DAYS_1M]
    if p_12m_ago <= 0:
        return None
    return p_1m_ago / p_12m_ago - 1.0


def _rsi(closes: Sequence[float], period: int = _RSI_PERIOD) -> Optional[float]:
    """Wilder's RSI over ``period`` sessions, in ``[0, 100]`` (or ``None``).

    Uses Wilder's smoothing of average gains/losses. An all-up window returns
    100, an all-down window returns 0. Returns ``None`` when there are fewer than
    ``period + 1`` prices (the first delta needs two prices).
    """
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    # Seed with the simple average of the first ``period`` deltas, then smooth.
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _ema_series(values: Sequence[float], span: int) -> List[float]:
    """Exponential moving average of ``values`` with the standard ``2/(span+1)`` alpha."""
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    ema = [float(values[0])]
    for v in values[1:]:
        ema.append(alpha * float(v) + (1.0 - alpha) * ema[-1])
    return ema


def _macd_posture(closes: Sequence[float]) -> Optional[Dict[str, float]]:
    """Return MACD line, signal line and histogram, or ``None`` if too short.

    MACD = EMA(fast) − EMA(slow); signal = EMA(MACD, 9); histogram = MACD − signal.
    A positive MACD line means the faster average leads (up-trend); a positive,
    *rising* histogram means that up-trend is strengthening.
    """
    if len(closes) < _MACD_SLOW + _MACD_SIGNAL:
        return None
    ema_fast = _ema_series(closes, _MACD_FAST)
    ema_slow = _ema_series(closes, _MACD_SLOW)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = _ema_series(macd_line, _MACD_SIGNAL)
    macd = macd_line[-1]
    signal = signal_line[-1]
    hist = macd - signal
    hist_prev = (macd_line[-2] - signal_line[-2]) if len(macd_line) >= 2 else hist
    return {"macd": macd, "signal": signal, "hist": hist, "hist_slope": hist - hist_prev}


def _universe_distribution(ctx: AnalysisContext, key: str) -> Optional[List[float]]:
    """Pull a numeric distribution for ``key`` from ``ctx.universe_stats``.

    Tolerant of the analyzer-defined shape documented on
    :class:`AnalysisContext`: accepts a bare sequence of numbers, or a mapping
    that nests the sample under ``"values"``/``"distribution"``/``"samples"``.
    Returns ``None`` when no usable numeric sample is present (degrade gracefully).
    """
    stats: Optional[Dict[str, Any]] = ctx.universe_stats
    if not stats or key not in stats:
        return None
    raw = stats[key]
    if isinstance(raw, dict):
        for nested in ("values", "distribution", "samples"):
            if nested in raw:
                raw = raw[nested]
                break
    if not isinstance(raw, (list, tuple)):
        return None
    nums = [float(v) for v in raw if isinstance(v, (int, float))]
    return nums or None


@register_analyzer
class MomentumAnalyzer(Analyzer):
    """Score absolute and relative price momentum into one auditable SubScore.

    The score blends 1/3/6/12-month trailing returns (with the academic 12–1
    horizon weighted most), confirms direction with RSI(14) and MACD posture, and
    — when ``ctx.universe_stats`` supplies a peer distribution of 12–1 momentum —
    lifts or lowers the score toward this name's relative-strength percentile so
    leaders are rewarded and laggards penalised. Persistent, broad-based positive
    momentum scores high; blow-off extremes (a parabolic last-month spike, an
    extreme overbought RSI) are flagged and discounted because such moves are
    fragile rather than durable.

    Requires daily ``price_history`` (oldest-first). With fewer than ~3 months of
    bars it returns a neutral, low-confidence sub-score rather than guessing.
    """

    category = ScoreCategory.MOMENTUM
    default_weight = 0.05
    requires = {"prices"}

    #: Number of distinct momentum components we *attempt* to compute. Used to
    #: derive an honest ``data_coverage`` (fraction actually available).
    _N_COMPONENTS = 6  # ret_1m, ret_3m, ret_6m, mom_12_1, RSI, MACD

    def analyze(self, data: SecurityData, ctx: AnalysisContext) -> SubScore:
        """Return the MOMENTUM :class:`SubScore` for ``data`` (pure; no I/O)."""
        closes = _series_closes(data.price_history)

        # --- Hard data gate: too little history to say anything honest. --------
        if len(closes) < _MIN_BARS:
            return self.neutral_subscore(
                rationale=(
                    f"Only {len(closes)} usable daily price bars available "
                    f"(need at least {_MIN_BARS} for a 3-month look-back); momentum "
                    "cannot be measured, so this category is scored neutral."
                ),
                coverage=0.0,
                extra_flags=["INSUFFICIENT_PRICE_HISTORY"],
            )

        # --- Component computation (each may be None on shorter histories). ----
        ret_1m = _trailing_return(closes, _DAYS_1M)
        ret_3m = _trailing_return(closes, _DAYS_3M)
        ret_6m = _trailing_return(closes, _DAYS_6M)
        ret_12m = _trailing_return(closes, _DAYS_12M)
        mom_12_1 = _mom_12_1(closes)
        rsi = _rsi(closes)
        macd = _macd_posture(closes)

        available = sum(
            1 for c in (ret_1m, ret_3m, ret_6m, mom_12_1, rsi, macd) if c is not None
        )
        coverage = available / float(self._N_COMPONENTS)

        evidence: List[Evidence] = []
        flags: List[str] = []

        # --- 1. Absolute trailing-return blend --------------------------------
        horizon_values: Dict[str, Optional[float]] = {
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_6m": ret_6m,
            "mom_12_1": mom_12_1,
        }
        component_scores: List[float] = []
        component_weights: List[float] = []
        for key, weight in _HORIZON_WEIGHTS.items():
            value = horizon_values[key]
            if value is None:
                continue
            lo, hi = _RETURN_BANDS[key]
            score = scale_to_score(value, lo, hi, higher_is_better=True)
            if score is None:
                continue
            component_scores.append(score)
            component_weights.append(weight)

        if component_scores:
            total_w = sum(component_weights)
            abs_score = sum(s * w for s, w in zip(component_scores, component_weights)) / total_w
        else:
            abs_score = 50.0

        # Evidence for each available trailing return (percentages, signed).
        for label, key, value in (
            ("1-month return", "ret_1m", ret_1m),
            ("3-month return", "ret_3m", ret_3m),
            ("6-month return", "ret_6m", ret_6m),
            ("12-month return", "ret_12m", ret_12m),
            ("12–1 momentum (12m excl. last month)", "mom_12_1", mom_12_1),
        ):
            direction = "neutral"
            if value is not None:
                direction = "bullish" if value > 0 else "bearish" if value < 0 else "neutral"
            evidence.append(
                Evidence.from_number(
                    label,
                    None if value is None else value * 100.0,
                    source="price_history",
                    direction=direction,
                    unit="%",
                    precision=1,
                    detail="Total return incl. dividends (adj. close)." if value is not None else None,
                )
            )

        # --- 2. Persistence / consistency across horizons ---------------------
        # Reward momentum that points the SAME way across multiple windows; a
        # single isolated move is far weaker evidence than a broad-based trend.
        decided = [v for v in (ret_1m, ret_3m, ret_6m, mom_12_1) if v is not None]
        persistence_adj = 0.0
        if len(decided) >= 2:
            positives = sum(1 for v in decided if v > 0)
            negatives = sum(1 for v in decided if v < 0)
            agreement = max(positives, negatives) / len(decided)  # 0.5..1.0
            sign = 1.0 if positives >= negatives else -1.0
            # Scale agreement above the 0.5 "no-consensus" floor into ±10 points.
            persistence_adj = sign * (agreement - 0.5) * 20.0
            if agreement >= 0.99 and len(decided) >= 3:
                flags.append("PERSISTENT_TREND")
            evidence.append(
                Evidence(
                    label="Cross-horizon trend agreement",
                    value=f"{int(round(agreement * 100))}%",
                    detail=(
                        f"{positives} of {len(decided)} horizons positive"
                        if sign > 0
                        else f"{negatives} of {len(decided)} horizons negative"
                    ),
                    source="derived:price_history",
                    direction="bullish" if sign > 0 and agreement > 0.5 else
                    "bearish" if sign < 0 and agreement > 0.5 else "neutral",
                )
            )

        # --- 3. RSI confirmation + blow-off / overbought discount -------------
        rsi_adj = 0.0
        if rsi is not None:
            # Mild confirmation: RSI above 50 is up-pressure, below 50 down.
            rsi_adj = (rsi - 50.0) * 0.15  # at RSI 70 -> +3, at RSI 30 -> -3
            rsi_dir = "bullish" if rsi > 55 else "bearish" if rsi < 45 else "neutral"
            if rsi >= 80.0:
                # Extreme overbought: a fragile, blow-off condition — discount it.
                rsi_adj -= 8.0
                flags.append("OVERBOUGHT_BLOWOFF")
                rsi_dir = "bearish"
            elif rsi <= 20.0:
                flags.append("DEEPLY_OVERSOLD")
            evidence.append(
                Evidence.from_number(
                    "RSI(14)",
                    rsi,
                    source="derived:price_history",
                    direction=rsi_dir,
                    precision=1,
                    detail="Wilder RSI; >70 overbought, <30 oversold, ≥80 flagged blow-off.",
                )
            )

        # Parabolic last-month spike relative to the 6-month trend is a classic
        # blow-off pattern: a big 1-month pop that dwarfs the longer trend is
        # fragile, so we discount rather than reward it.
        if ret_1m is not None and ret_1m > 0.50 and (ret_6m is None or ret_1m > abs(ret_6m)):
            flags.append("PARABOLIC_SPIKE")
            rsi_adj -= 6.0
            evidence.append(
                Evidence.from_number(
                    "1-month parabolic spike",
                    ret_1m * 100.0,
                    source="derived:price_history",
                    direction="bearish",
                    unit="%",
                    precision=1,
                    detail="One-month gain >50% and exceeding the 6-month move — discounted as fragile.",
                )
            )

        # --- 4. MACD posture --------------------------------------------------
        macd_adj = 0.0
        if macd is not None:
            up_trend = macd["macd"] > macd["signal"]
            strengthening = macd["hist_slope"] > 0
            if up_trend and strengthening:
                macd_adj = 5.0
            elif up_trend:
                macd_adj = 2.5
            elif not up_trend and macd["hist_slope"] < 0:
                macd_adj = -5.0
            else:
                macd_adj = -2.5
            macd_dir = "bullish" if macd_adj > 0 else "bearish" if macd_adj < 0 else "neutral"
            evidence.append(
                Evidence(
                    label="MACD posture (12/26/9)",
                    value=("above signal" if up_trend else "below signal")
                    + (", strengthening" if strengthening else ", weakening"),
                    detail=f"MACD={macd['macd']:.4f}, signal={macd['signal']:.4f}, hist={macd['hist']:.4f}.",
                    source="derived:price_history",
                    direction=macd_dir,
                )
            )

        # --- 5. Relative strength vs the screened universe (12–1) -------------
        # Blend the absolute read with where this name's intermediate momentum
        # ranks across the universe, so leaders are rewarded vs their opportunity
        # set rather than vs a fixed threshold. Degrade gracefully when absent.
        rel_score: Optional[float] = None
        dist = _universe_distribution(ctx, "mom_12_1")
        if dist is None:
            dist = _universe_distribution(ctx, "ret_6m") if mom_12_1 is None else None
        rel_value = mom_12_1 if mom_12_1 is not None else ret_6m
        if dist is not None and rel_value is not None:
            pct = percentile_rank(rel_value, dist)
            if pct is not None:
                rel_score = pct * 100.0
                evidence.append(
                    Evidence.from_number(
                        "Relative strength percentile (12–1 vs universe)",
                        rel_score,
                        source="universe_stats",
                        direction="bullish" if pct >= 0.6 else "bearish" if pct <= 0.4 else "neutral",
                        unit="th pct",
                        precision=0,
                        detail=f"Ranked against {len(dist)} screened peers.",
                    )
                )
        else:
            flags.append("NO_RELATIVE_BENCHMARK")

        # --- Combine ----------------------------------------------------------
        base = abs_score + persistence_adj + rsi_adj + macd_adj
        if rel_score is not None:
            # 60% absolute trend / 40% relative standing.
            score = 0.6 * clamp(base) + 0.4 * rel_score
        else:
            score = base
        score = clamp(score, 0.0, 100.0)

        # --- Confidence: scale with coverage and with how much history we have.
        history_factor = clamp(len(closes) / float(_DAYS_12M + 1), 0.0, 1.0)
        confidence = clamp(0.25 + 0.55 * coverage * history_factor, 0.0, 1.0)
        if rel_score is not None:
            confidence = clamp(confidence + 0.08, 0.0, 1.0)

        rationale = self._build_rationale(
            score=score,
            ret_1m=ret_1m,
            ret_6m=ret_6m,
            mom_12_1=mom_12_1,
            rsi=rsi,
            rel_score=rel_score,
            flags=flags,
        )

        return SubScore(
            category=self.category,
            score=score,
            confidence=confidence,
            weight=self.default_weight,
            rationale=rationale,
            evidence=evidence,
            flags=flags,
            data_coverage=coverage,
        )

    @staticmethod
    def _build_rationale(
        *,
        score: float,
        ret_1m: Optional[float],
        ret_6m: Optional[float],
        mom_12_1: Optional[float],
        rsi: Optional[float],
        rel_score: Optional[float],
        flags: List[str],
    ) -> str:
        """Compose a short, honest, human rationale citing the driving numbers."""
        if score >= 65:
            tone = "Strong, broad-based positive momentum"
        elif score >= 55:
            tone = "Modestly positive momentum"
        elif score <= 35:
            tone = "Negative momentum / downtrend"
        elif score <= 45:
            tone = "Soft, fading momentum"
        else:
            tone = "Mixed, range-bound momentum"

        parts: List[str] = [tone]
        if mom_12_1 is not None:
            parts.append(f"12–1 momentum {mom_12_1 * 100:+.1f}%")
        elif ret_6m is not None:
            parts.append(f"6-month return {ret_6m * 100:+.1f}%")
        if rsi is not None:
            parts.append(f"RSI {rsi:.0f}")
        if rel_score is not None:
            parts.append(f"{rel_score:.0f}th pct vs universe")

        sentence = "; ".join(parts) + "."
        caveats: List[str] = []
        if "PARABOLIC_SPIKE" in flags or "OVERBOUGHT_BLOWOFF" in flags:
            caveats.append("a blow-off/overbought extreme was detected and discounted")
        if "NO_RELATIVE_BENCHMARK" in flags:
            caveats.append("no universe benchmark was available, so this is absolute-only")
        if caveats:
            sentence += " Note: " + "; ".join(caveats) + "."
        sentence += " Momentum is one signal among many and is not a forecast."
        return sentence


__all__ = ["MomentumAnalyzer"]
