"""Ranking engine — folds a company's sub-scores into one scored, ranked row.

Part of Convexity, an evidence-driven equity **research and screening** tool. This
module is **not** a predictor and **not** investment advice. It does not decide what
to buy; it aggregates many *independent* pieces of evidence (the per-category
:class:`~convexity.core.models.SubScore` objects produced by the analyzers) into a
single composite score, an honest conviction figure, and a rank within the screened
universe.

The honesty principle encoded here
----------------------------------
A high composite score on its own is **not** conviction. A company can post an
extreme score in one category (say, VALUE) while every other category is silent or
neutral; rewarding that with high conviction would be exactly the kind of
single-signal overreach this tool exists to avoid. Conviction is therefore computed
separately from the composite and is engineered to:

* **rise** with the number of *independent* categories that agree bullishly — many
  distinct lines of evidence pointing the same way is the only thing that justifies
  conviction;
* **rise** with how much real data underpins the analysis (mean ``data_coverage``
  and the analyzers' own ``confidence``);
* **fall** with dispersion — when the non-risk categories disagree (a wide spread of
  scores), the signals are contradicting each other and conviction must drop;
* **fall** with missing data — thin coverage and ``MISSING_DATA`` flags pull
  conviction down, never up.

All composite/agreement/confidence arithmetic is delegated to
:func:`convexity.core.scoring.combine_subscores` (which also applies ``RISK`` as the
dampener per the scoring convention) — this engine never re-derives that maths. It
adds only the conviction synthesis and the universe sort, then leaves every
narrative field empty for the explainability engine to populate.

The engine is pure and deterministic: given the same inputs it always returns the
same ordering and the same numbers, so a ranking is reproducible and auditable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Dict, List, Optional

from convexity.core.config import DEFAULT_CATEGORY_WEIGHTS
from convexity.core.models import (
    CompanyAnalysis,
    ScanParams,
    ScoreCategory,
    SecurityData,
    SubScore,
)
from convexity.core.scoring import clamp, combine_subscores

# ---------------------------------------------------------------------------
# Conviction-tuning constants
# ---------------------------------------------------------------------------
#
# These govern how the *separate* conviction figure is built from the sub-scores.
# They are deliberately conservative: conviction starts low and is *earned* by the
# breadth of independent agreement, the depth of real data, and tight clustering of
# the signals — never by a single extreme category.

# A non-risk category is read as "bullishly confirming" only when its score clears
# this bar. Matches the bullish threshold used in
# :func:`convexity.core.scoring.combine_subscores` so the two agree on what counts.
_BULLISH_THRESHOLD: float = 60.0

# A category only *counts* toward the breadth-of-agreement tally when it carries at
# least this much of its own confidence — a bullish reading drawn from almost no
# data is not independent evidence and must not inflate conviction.
_MIN_CONFIRMING_CONFIDENCE: float = 0.35

# How many independent confirming categories saturate the breadth term. Reaching
# this many genuinely-confirming, independent signals is what fully unlocks the
# breadth component of conviction; fewer scales it down proportionally.
_BREADTH_SATURATION: int = 6

# Floor on conviction once *some* evidence exists, so a fully-analysed company is
# never reported at literally zero conviction (which would read as "certainly not"
# rather than the honest "the evidence does not converge").
_CONVICTION_FLOOR: float = 0.02

# How the three conviction ingredients are mixed. Breadth of *independent* agreement
# dominates by design (it is the core honesty requirement), with data depth and
# low dispersion as supporting, confirming factors.
_W_BREADTH: float = 0.55
_W_DEPTH: float = 0.25
_W_AGREEMENT: float = 0.20


def _count_independent_confirmations(subscores: Sequence[SubScore]) -> int:
    """Count non-risk categories that *independently* confirm a bullish read.

    A category confirms only when (a) it is not the ``RISK`` dampener category,
    (b) its score clears :data:`_BULLISH_THRESHOLD`, and (c) it carries at least
    :data:`_MIN_CONFIRMING_CONFIDENCE` confidence — so a bullish score resting on
    almost no data is excluded rather than allowed to manufacture conviction.

    Because each :class:`ScoreCategory` is scored by exactly one analyzer from a
    distinct evidence base, every counted category is an *independent* signal; the
    tally is therefore a direct measure of how many separate lines of evidence
    agree, which is the only thing that justifies conviction.

    Args:
        subscores: The company's per-category sub-scores.

    Returns:
        The number of independent, sufficiently-confident bullish confirmations.
    """
    count = 0
    for sub in subscores:
        if sub is None or sub.category == ScoreCategory.RISK:
            continue
        if sub.score >= _BULLISH_THRESHOLD and sub.confidence >= _MIN_CONFIRMING_CONFIDENCE:
            count += 1
    return count


def _mean_data_coverage(subscores: Sequence[SubScore]) -> float:
    """Return the mean ``data_coverage`` across the non-risk categories in ``[0, 1]``.

    This measures how much real, present data underpins the analysis as a whole.
    The ``RISK`` category is excluded so the figure reflects the additive evidence
    that drives the composite. Returns ``0.0`` when there are no non-risk
    sub-scores (no evidence at all).
    """
    non_risk = [s for s in subscores if s is not None and s.category != ScoreCategory.RISK]
    if not non_risk:
        return 0.0
    return clamp(sum(s.data_coverage for s in non_risk) / len(non_risk), 0.0, 1.0)


def _has_real_evidence(subscores: Sequence[SubScore]) -> bool:
    """Return whether any non-risk category carries genuine (non-trivial) data.

    Used to decide whether the conviction floor applies: a company every one of
    whose categories is a zero-coverage ``MISSING_DATA`` placeholder has no evidence
    to convict on, and its conviction is left at zero rather than floored upward.
    """
    for sub in subscores:
        if sub is None or sub.category == ScoreCategory.RISK:
            continue
        if sub.data_coverage > 0.0 or sub.confidence > 0.1:
            return True
    return False


def _compute_conviction(
    subscores: Sequence[SubScore],
    signal_agreement: float,
    overall_confidence: float,
) -> float:
    """Synthesise the conviction figure in ``[0, 1]`` from the sub-scores.

    Conviction is engineered to encode the core Convexity principle that a high
    composite is *not* conviction: it RISES with the number of independent
    confirming categories and with data depth, and FALLS with dispersion and
    missing data. It is built from three ingredients:

    * **breadth** — the count of independent, confident bullish confirmations
      (:func:`_count_independent_confirmations`) scaled toward
      :data:`_BREADTH_SATURATION`. This is the dominant term: many distinct
      signals agreeing is what justifies conviction. With zero or one confirming
      signal this term is near zero, so a lone extreme category cannot create
      conviction however high it scores.
    * **depth** — the data underpinning the analysis: the geometric blend of mean
      ``data_coverage`` (:func:`_mean_data_coverage`) and the analyzers' own
      coverage-weighted ``overall_confidence``. Thin data drags this toward zero.
    * **agreement** — ``signal_agreement`` from
      :func:`~convexity.core.scoring.combine_subscores`, which already folds in
      directional consensus *and* a low-dispersion reward, so contradictory or
      widely-spread signals lower it.

    The three are combined with breadth weighted most heavily, then the breadth and
    agreement terms additionally *gate* the result multiplicatively: even strong
    data depth cannot lift conviction when scarcely any independent signals agree or
    when the signals disagree. The result is floored just above zero only when some
    real evidence exists, so a fully-analysed but non-converging company reads as
    "evidence does not converge" rather than a hard zero.

    Args:
        subscores: The company's per-category sub-scores.
        signal_agreement: The directional-consensus-and-low-dispersion term in
            ``[0, 1]`` from :func:`combine_subscores`.
        overall_confidence: The coverage-weighted mean confidence in ``[0, 1]``
            from :func:`combine_subscores`.

    Returns:
        The conviction confidence in ``[0, 1]``.
    """
    confirmations = _count_independent_confirmations(subscores)
    breadth = clamp(confirmations / float(_BREADTH_SATURATION), 0.0, 1.0)

    coverage = _mean_data_coverage(subscores)
    conf = clamp(overall_confidence, 0.0, 1.0)
    # Geometric blend: a near-zero in *either* coverage or analyzer-confidence
    # collapses depth, so missing data cannot be papered over by high stated
    # confidence (and vice versa).
    depth = math.sqrt(max(coverage, 0.0) * max(conf, 0.0))

    agreement = clamp(signal_agreement, 0.0, 1.0)

    # Weighted blend of the three earned ingredients.
    blended = _W_BREADTH * breadth + _W_DEPTH * depth + _W_AGREEMENT * agreement

    # Multiplicative gate: breadth of independent agreement and directional
    # consensus must BOTH be present for conviction to survive. With one lone
    # confirming signal (breadth ~ 1/6) or contradictory signals (agreement ~ 0),
    # this gate crushes conviction toward zero regardless of the blend — encoding
    # that a high composite alone never earns conviction. The 0.15 additive term
    # keeps the gate from being a hard all-or-nothing switch while still requiring
    # genuine breadth + consensus to reach high conviction.
    gate = clamp(0.15 + 0.85 * (breadth * agreement), 0.0, 1.0)

    conviction = clamp(blended * gate, 0.0, 1.0)

    if conviction < _CONVICTION_FLOOR and _has_real_evidence(subscores):
        conviction = _CONVICTION_FLOOR
    return clamp(conviction, 0.0, 1.0)


class DefaultRankingEngine:
    """Default :class:`~convexity.core.contracts.RankingEngine` implementation.

    Conforms to the ``RankingEngine`` Protocol exactly: it exposes
    :meth:`score_company` (fold one security's sub-scores into a scored
    :class:`CompanyAnalysis`) and :meth:`rank` (order a list of analyses best-first
    and stamp 1-based ranks).

    Scoring delegates entirely to
    :func:`convexity.core.scoring.combine_subscores` for the composite,
    ``signal_agreement`` and overall confidence — including the ``RISK`` dampener —
    and uses :data:`~convexity.core.config.DEFAULT_CATEGORY_WEIGHTS` when the caller
    does not supply weights. On top of that it computes a *separate* conviction
    figure that is high only when many independent categories confirm with real
    data behind them. Narrative fields are intentionally left empty for the
    explainability engine to fill.

    The engine holds no mutable state and is safe to share across scans.
    """

    def __init__(
        self,
        weights: Optional[Dict[ScoreCategory, float]] = None,
    ) -> None:
        """Create a ranking engine.

        Args:
            weights: Optional category-weighting map used when
                :meth:`score_company` is called without explicit weights. Defaults
                to a copy of :data:`~convexity.core.config.DEFAULT_CATEGORY_WEIGHTS`.
        """
        self._default_weights: Dict[ScoreCategory, float] = (
            dict(weights) if weights is not None else dict(DEFAULT_CATEGORY_WEIGHTS)
        )

    # ------------------------------------------------------------------ #
    # RankingEngine protocol: score_company                               #
    # ------------------------------------------------------------------ #
    def score_company(
        self,
        data: SecurityData,
        subscores: List[SubScore],
        weights: Dict[ScoreCategory, float],
    ) -> CompanyAnalysis:
        """Fold a security's sub-scores into a single scored :class:`CompanyAnalysis`.

        Computes the composite score, ``signal_agreement`` and overall confidence
        via :func:`combine_subscores` (which applies the ``RISK`` dampener), then
        derives the separate conviction figure. All narrative fields
        (``thesis``, ``bull_case``, summaries, ``monitoring_checklist`` …) are left
        empty for the explainability engine; ``rank`` is left ``None`` until
        :meth:`rank` orders the universe.

        Args:
            data: The aggregated security data (used for identity/metadata only;
                the numeric scoring rests entirely on ``subscores``).
            subscores: The per-category sub-scores for this security.
            weights: The category-weighting map to apply. When falsy (``None`` or
                empty) the engine's configured default weights are used.

        Returns:
            A scored :class:`CompanyAnalysis` with ``composite_score``,
            ``conviction_confidence``, ``signal_agreement`` and ``subscores`` filled
            and every narrative field empty.
        """
        effective_weights: Dict[ScoreCategory, float] = (
            dict(weights) if weights else dict(self._default_weights)
        )

        clean_subscores: List[SubScore] = [s for s in subscores if s is not None]

        composite, signal_agreement, overall_confidence = combine_subscores(
            clean_subscores, effective_weights
        )
        conviction = _compute_conviction(
            clean_subscores, signal_agreement, overall_confidence
        )

        return CompanyAnalysis(
            ticker=data.ticker,
            name=data.name,
            industry=data.industry,
            sector=data.sector,
            market_cap=data.market_cap,
            cap_tier=data.cap_tier,
            composite_score=clamp(composite, 0.0, 100.0),
            conviction_confidence=clamp(conviction, 0.0, 1.0),
            rank=None,
            subscores=clean_subscores,
            signal_agreement=clamp(signal_agreement, 0.0, 1.0),
        )

    # ------------------------------------------------------------------ #
    # RankingEngine protocol: rank                                        #
    # ------------------------------------------------------------------ #
    def rank(
        self,
        analyses: List[CompanyAnalysis],
        params: ScanParams,
    ) -> List[CompanyAnalysis]:
        """Order ``analyses`` best-first and stamp 1-based integer ranks.

        Companies are sorted by ``composite_score`` descending, breaking ties by
        ``conviction_confidence`` descending (a more-confirmed company outranks an
        equally-scored but thinner one), then by ``signal_agreement`` descending,
        and finally by ``ticker`` ascending so the ordering is fully deterministic.
        Each returned analysis has its ``rank`` set to its 1-based position.

        The input list is not mutated in place beyond setting ``rank``; a new
        ordered list is returned. ``params`` is accepted to satisfy the
        :class:`~convexity.core.contracts.RankingEngine` Protocol and is available
        for future rank-time policy (it does not alter the ordering here, which is
        a pure function of the already-computed scores).

        Args:
            analyses: The scored analyses to rank (typically the output of
                :meth:`score_company` for each screened security).
            params: The active scan parameters (Protocol-required; unused for the
                current ordering policy).

        Returns:
            A new list ordered best-first, each element's ``rank`` assigned.
        """
        ordered = sorted(
            [a for a in analyses if a is not None],
            key=lambda a: (
                -float(a.composite_score),
                -float(a.conviction_confidence),
                -float(a.signal_agreement),
                a.ticker,
            ),
        )
        for position, analysis in enumerate(ordered, start=1):
            analysis.rank = position
        return ordered


__all__ = ["DefaultRankingEngine"]
