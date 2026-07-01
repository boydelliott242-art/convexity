"""Pure numeric helpers shared by analyzers and the ranking engine.

Everything here is deterministic and side-effect free: no wall-clock, no random
seeds, no I/O. Given identical inputs these functions always return identical
outputs, which is what makes Convexity's scores reproducible and auditable.

Scores live on a 0–100 scale where higher means "more attractive evidence" for
the category in question. ``RISK`` is the one category scored so that a *higher*
sub-score means *lower* risk (a safer profile), allowing it to be combined with
the others uniformly and to act as a dampener when risk is elevated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import List, Optional, Tuple

from convexity.core.models import ScoreCategory, SubScore


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp ``value`` into the inclusive range ``[lo, hi]``."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def winsorize(values: Sequence[float], lower: float = 0.05, upper: float = 0.95) -> List[float]:
    """Clip extreme values to the given percentiles to tame outliers.

    Micro-cap fundamentals are noisy and a single absurd ratio can dominate a
    distribution. Winsorising replaces values beyond the ``lower``/``upper``
    percentile with the percentile boundary itself (it does not drop them).

    Args:
        values: The sample to winsorize. Empty input returns an empty list.
        lower: Lower tail fraction in ``[0, 1)``.
        upper: Upper tail fraction in ``(0, 1]``.

    Returns:
        A new list of the same length with the tails clipped.
    """
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return []
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("require 0 <= lower < upper <= 1")
    ordered = sorted(clean)
    n = len(ordered)
    lo_idx = min(n - 1, max(0, int(math.floor(lower * (n - 1)))))
    hi_idx = min(n - 1, max(0, int(math.ceil(upper * (n - 1)))))
    lo_val, hi_val = ordered[lo_idx], ordered[hi_idx]
    return [min(max(v, lo_val), hi_val) for v in clean]


def percentile_rank(value: Optional[float], distribution: Sequence[float]) -> Optional[float]:
    """Return the fraction of ``distribution`` that ``value`` meets or exceeds.

    Result is in ``[0, 1]``. Uses the standard "<=" definition so a value at the
    maximum scores 1.0 and a value at the minimum scores ``1/n``. Returns ``None``
    when ``value`` is ``None`` or the distribution is empty (insufficient data).
    """
    if value is None:
        return None
    clean = [float(v) for v in distribution if v is not None]
    if not clean:
        return None
    count_le = sum(1 for v in clean if v <= value)
    return count_le / len(clean)


def scale_to_score(
    value: Optional[float],
    lo: float,
    hi: float,
    higher_is_better: bool = True,
) -> Optional[float]:
    """Linearly map ``value`` from the band ``[lo, hi]`` onto a 0–100 score.

    Values at or beyond the favourable end map to 100, values at or beyond the
    unfavourable end map to 0, and anything between interpolates linearly.

    Args:
        value: The raw metric. ``None`` yields ``None`` (missing data).
        lo: The metric value mapped to 0 (when ``higher_is_better``) or 100.
        hi: The metric value mapped to 100 (when ``higher_is_better``) or 0.
        higher_is_better: If ``False`` the mapping is inverted (e.g. a P/E where
            lower is more attractive).

    Returns:
        A score in ``[0, 100]`` or ``None`` if ``value`` is missing.
    """
    if value is None:
        return None
    if lo == hi:
        return 50.0
    frac = (value - lo) / (hi - lo)
    frac = min(max(frac, 0.0), 1.0)
    if not higher_is_better:
        frac = 1.0 - frac
    return frac * 100.0


def logistic_score(value: Optional[float], midpoint: float, steepness: float = 1.0) -> Optional[float]:
    """Map ``value`` to 0–100 through a logistic curve centred at ``midpoint``.

    Useful when the relationship saturates: scores rise quickly near the midpoint
    and flatten at the extremes. ``steepness`` controls how sharp the transition
    is (larger == steeper). Returns ``None`` for missing input.
    """
    if value is None:
        return None
    try:
        z = steepness * (value - midpoint)
        # Guard against overflow for very large |z|.
        if z >= 0:
            sig = 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            sig = ez / (1.0 + ez)
    except OverflowError:  # pragma: no cover - defensive
        sig = 0.0 if value < midpoint else 1.0
    return sig * 100.0


def weighted_mean(values: Sequence[Optional[float]], weights: Sequence[float]) -> Optional[float]:
    """Weighted mean that ignores ``None`` entries (and their weights).

    Returns ``None`` if every value is ``None`` or all surviving weights are zero.
    """
    if len(values) != len(weights):
        raise ValueError("values and weights must be the same length")
    num = 0.0
    den = 0.0
    for v, w in zip(values, weights):
        if v is None or w is None:
            continue
        num += float(v) * float(w)
        den += float(w)
    if den == 0.0:
        return None
    return num / den


def combine_subscores(
    subscores: Sequence[SubScore],
    weights: Optional[dict] = None,
) -> Tuple[float, float, float]:
    """Combine per-category sub-scores into a composite with honesty metrics.

    Args:
        subscores: The category sub-scores for one company.
        weights: Optional mapping of :class:`ScoreCategory` (or its string value)
            to a weight. When omitted, each sub-score's own ``weight`` is used.

    Returns:
        A 3-tuple ``(composite, signal_agreement, overall_confidence)``:

        * **composite** — confidence-and-weight weighted mean of the sub-scores in
          ``[0, 100]``. Each sub-score is weighted by ``weight * confidence`` so a
          low-confidence (thin-data) category pulls the composite less. The
          ``RISK`` category is treated as a *dampener*: rather than being averaged
          in, an elevated risk score reduces the composite proportionally. A risk
          sub-score of 50 is neutral; below 50 (riskier) it shaves points off,
          above 50 (safer) it leaves the composite essentially unchanged.
        * **signal_agreement** — in ``[0, 1]``. Measures how strongly the
          *non-risk* categories agree on a bullish read. It is the product of two
          terms: (a) the directional consensus — ``max(bullish, bearish) /
          decisive`` where bullish = count scoring > 60, bearish = count scoring
          < 40, decisive = bullish + bearish; and (b) a low-dispersion term
          ``1 - normalised_stdev`` so that tightly clustered scores are rewarded.
          Conviction should be asserted only when this number is high — i.e. when
          many *independent* signals point the same way.
        * **overall_confidence** — coverage-weighted mean of the sub-score
          confidences in ``[0, 1]``, reflecting how much real data underpins the
          whole analysis.

    The function is pure and deterministic.
    """
    scored = [s for s in subscores if s is not None]
    if not scored:
        return 0.0, 0.0, 0.0

    def _weight_for(sub: SubScore) -> float:
        if weights is None:
            return float(sub.weight)
        # Accept either enum keys or string keys.
        if sub.category in weights:
            return float(weights[sub.category])
        if sub.category.value in weights:
            return float(weights[sub.category.value])
        return float(sub.weight)

    # --- Composite (excluding RISK from the average; RISK is a dampener) ------
    non_risk = [s for s in scored if s.category != ScoreCategory.RISK]
    risk_sub = next((s for s in scored if s.category == ScoreCategory.RISK), None)

    if non_risk:
        vals = [s.score for s in non_risk]
        wts = [max(_weight_for(s) * s.confidence, 0.0) for s in non_risk]
        base = weighted_mean(vals, wts)
        if base is None:
            base = sum(vals) / len(vals)
    else:
        base = 50.0

    # RISK dampener: map risk score (0..100, higher == safer) into a multiplier.
    # 50 -> 1.0 (neutral). A maximally risky 0 caps the dampener at 0.6 so risk
    # tempers but never zeroes a thesis. Safer-than-neutral risk gives a tiny
    # uplift toward 1.0 only (never inflating the thesis above its evidence).
    if risk_sub is not None:
        risk_norm = risk_sub.score / 100.0  # 0..1
        # Below 0.5: dampen down to 0.6; at/above 0.5: scale gently up to 1.0.
        if risk_norm < 0.5:
            multiplier = 0.6 + 0.8 * risk_norm  # risk_norm 0 -> 0.6, 0.5 -> 1.0
        else:
            multiplier = 1.0
        composite = base * multiplier
    else:
        composite = base
    composite = clamp(composite, 0.0, 100.0)

    # --- Signal agreement (independent directional consensus) ----------------
    bullish = sum(1 for s in non_risk if s.score > 60.0)
    bearish = sum(1 for s in non_risk if s.score < 40.0)
    decisive = bullish + bearish
    if decisive == 0:
        consensus = 0.0
    else:
        consensus = max(bullish, bearish) / decisive

    if len(non_risk) >= 2:
        mean_s = sum(s.score for s in non_risk) / len(non_risk)
        var = sum((s.score - mean_s) ** 2 for s in non_risk) / len(non_risk)
        stdev = math.sqrt(var)
        # 50 is the maximal stdev for scores in [0,100]; normalise against it.
        dispersion_term = 1.0 - min(stdev / 50.0, 1.0)
    else:
        dispersion_term = 0.0

    signal_agreement = clamp(consensus * dispersion_term, 0.0, 1.0)

    # --- Overall confidence (coverage-weighted mean of confidences) ----------
    conf_vals = [s.confidence for s in scored]
    cov_wts = [max(s.data_coverage, 0.0) for s in scored]
    if sum(cov_wts) == 0.0:
        overall_confidence = sum(conf_vals) / len(conf_vals)
    else:
        oc = weighted_mean(conf_vals, cov_wts)
        overall_confidence = oc if oc is not None else sum(conf_vals) / len(conf_vals)
    overall_confidence = clamp(overall_confidence, 0.0, 1.0)

    return composite, signal_agreement, overall_confidence


__all__ = [
    "clamp",
    "winsorize",
    "percentile_rank",
    "scale_to_score",
    "logistic_score",
    "weighted_mean",
    "combine_subscores",
]
