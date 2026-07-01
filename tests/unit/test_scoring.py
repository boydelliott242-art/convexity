"""Unit tests for :func:`convexity.core.scoring.combine_subscores` and helpers.

These pin the honesty-critical behaviour of the scoring core with hand-built
sub-scores (no provider, no pipeline):

* **All-missing** -> a neutral composite with low overall confidence (a data gap
  neither helps nor hurts, and the read is openly low-confidence).
* **Unanimous bullish** -> a high ``signal_agreement`` (many independent categories
  pointing the same way is exactly what the metric should reward).
* **Contradiction** -> a low ``signal_agreement`` (signals fighting each other must
  not read as conviction).
* The **RISK dampener** lowers the composite when risk is elevated (low RISK score),
  and is neutral when risk is benign — RISK is never averaged into the composite.

The supporting pure helpers (:func:`clamp`, :func:`scale_to_score`,
:func:`percentile_rank`, :func:`weighted_mean`) get focused coverage too.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from convexity.core.models import ScoreCategory, SubScore
from convexity.core.scoring import (
    clamp,
    combine_subscores,
    logistic_score,
    percentile_rank,
    scale_to_score,
    weighted_mean,
)

# Additive (non-risk) categories used to build synthetic sub-score sets.
_ADDITIVE = [
    ScoreCategory.VALUE,
    ScoreCategory.GROWTH,
    ScoreCategory.QUALITY,
    ScoreCategory.FINANCIAL_HEALTH,
    ScoreCategory.CATALYST,
    ScoreCategory.COMPETITIVE,
]


def _sub(
    category: ScoreCategory,
    score: float,
    *,
    confidence: float = 0.8,
    weight: float = 0.1,
    coverage: float = 0.9,
    flags: Optional[List[str]] = None,
) -> SubScore:
    """Build a minimal :class:`SubScore` for ``category`` at ``score``."""
    return SubScore(
        category=category,
        score=score,
        confidence=confidence,
        weight=weight,
        rationale="synthetic",
        evidence=[],
        flags=list(flags or []),
        data_coverage=coverage,
    )


def _missing(category: ScoreCategory) -> SubScore:
    """A neutral, zero-coverage, low-confidence MISSING_DATA sub-score."""
    return _sub(
        category, 50.0, confidence=0.1, coverage=0.0, flags=["MISSING_DATA"]
    )


# ---------------------------------------------------------------------------
# combine_subscores: all-missing -> neutral / low confidence
# ---------------------------------------------------------------------------


class TestAllMissing:
    def test_empty_subscores_is_zeroed(self) -> None:
        """No sub-scores at all -> the documented all-zero tuple."""
        composite, agreement, confidence = combine_subscores([])
        assert composite == 0.0
        assert agreement == 0.0
        assert confidence == 0.0

    def test_all_missing_is_neutral_low_confidence(self) -> None:
        """Every category missing -> composite ~50, agreement 0, low confidence."""
        subs = [_missing(cat) for cat in _ADDITIVE]
        composite, agreement, confidence = combine_subscores(subs)
        # All sit at the neutral midpoint, so the composite is ~50.
        assert composite == pytest.approx(50.0, abs=1e-6)
        # Nothing is decisively bullish or bearish -> no agreement.
        assert agreement == pytest.approx(0.0)
        # Confidence reflects the low per-category confidence (data gap is honest).
        assert confidence <= 0.2

    def test_missing_data_does_not_create_agreement(self) -> None:
        """A wall of neutral 50s must not manufacture signal agreement."""
        subs = [_missing(cat) for cat in _ADDITIVE]
        _, agreement, _ = combine_subscores(subs)
        assert agreement == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# combine_subscores: unanimous bullish -> high agreement
# ---------------------------------------------------------------------------


class TestSignalAgreement:
    def test_unanimous_bullish_has_high_agreement(self) -> None:
        """Tightly-clustered, all-bullish categories yield high agreement."""
        subs = [_sub(cat, 80.0) for cat in _ADDITIVE]
        composite, agreement, _ = combine_subscores(subs)
        assert agreement >= 0.85, f"expected strong agreement, got {agreement}"
        assert composite >= 75.0

    def test_unanimous_bearish_also_has_high_agreement(self) -> None:
        """Agreement measures *consensus*, so unanimous bearish is high too."""
        subs = [_sub(cat, 20.0) for cat in _ADDITIVE]
        composite, agreement, _ = combine_subscores(subs)
        assert agreement >= 0.85
        assert composite <= 25.0

    def test_contradiction_lowers_agreement(self) -> None:
        """A 50/50 bull/bear split is the contradiction case -> low agreement."""
        subs = [
            _sub(ScoreCategory.VALUE, 85.0),
            _sub(ScoreCategory.GROWTH, 82.0),
            _sub(ScoreCategory.QUALITY, 80.0),
            _sub(ScoreCategory.FINANCIAL_HEALTH, 18.0),
            _sub(ScoreCategory.CATALYST, 20.0),
            _sub(ScoreCategory.COMPETITIVE, 22.0),
        ]
        _, agreement, _ = combine_subscores(subs)
        # Three bullish vs three bearish, widely dispersed -> agreement is crushed.
        assert agreement <= 0.55

    def test_unanimous_beats_contradiction(self) -> None:
        """Unanimity always out-agrees an even split, all else equal."""
        unanimous = [_sub(cat, 80.0) for cat in _ADDITIVE]
        split = [
            _sub(_ADDITIVE[i], 80.0 if i % 2 == 0 else 20.0)
            for i in range(len(_ADDITIVE))
        ]
        _, agree_u, _ = combine_subscores(unanimous)
        _, agree_s, _ = combine_subscores(split)
        assert agree_u > agree_s


# ---------------------------------------------------------------------------
# combine_subscores: RISK dampener
# ---------------------------------------------------------------------------


class TestRiskDampener:
    def test_elevated_risk_lowers_composite(self) -> None:
        """A low RISK score (riskier) shaves points off the composite."""
        base = [_sub(cat, 70.0) for cat in _ADDITIVE]
        composite_no_risk, _, _ = combine_subscores(base)

        # RISK = 0 (maximally risky) should apply the strongest dampener.
        risky = base + [_sub(ScoreCategory.RISK, 0.0, weight=0.0)]
        composite_risky, _, _ = combine_subscores(risky)

        assert composite_risky < composite_no_risk
        # Per the documented dampener, 0 risk multiplies the base by 0.6.
        assert composite_risky == pytest.approx(composite_no_risk * 0.6, rel=1e-3)

    def test_neutral_risk_does_not_change_composite(self) -> None:
        """A RISK score of 50 is neutral -> the composite is unchanged."""
        base = [_sub(cat, 70.0) for cat in _ADDITIVE]
        composite_no_risk, _, _ = combine_subscores(base)

        neutral = base + [_sub(ScoreCategory.RISK, 50.0, weight=0.0)]
        composite_neutral, _, _ = combine_subscores(neutral)
        assert composite_neutral == pytest.approx(composite_no_risk, rel=1e-6)

    def test_safe_risk_does_not_inflate_above_evidence(self) -> None:
        """A high (safe) RISK score never lifts the composite above its base.

        Safer-than-neutral risk only nudges the multiplier toward 1.0 — it must not
        inflate a thesis beyond the additive evidence that earned it.
        """
        base = [_sub(cat, 70.0) for cat in _ADDITIVE]
        composite_no_risk, _, _ = combine_subscores(base)

        safe = base + [_sub(ScoreCategory.RISK, 100.0, weight=0.0)]
        composite_safe, _, _ = combine_subscores(safe)
        assert composite_safe <= composite_no_risk + 1e-6
        assert composite_safe == pytest.approx(composite_no_risk, rel=1e-6)

    def test_riskier_strictly_below_safer(self) -> None:
        """A riskier profile composes strictly below an otherwise-identical safer one."""
        base = [_sub(cat, 65.0) for cat in _ADDITIVE]
        risky, _, _ = combine_subscores(base + [_sub(ScoreCategory.RISK, 10.0)])
        safe, _, _ = combine_subscores(base + [_sub(ScoreCategory.RISK, 90.0)])
        assert risky < safe

    def test_risk_excluded_from_directional_agreement(self) -> None:
        """RISK is a dampener, not a directional vote: it doesn't change agreement."""
        base = [_sub(cat, 80.0) for cat in _ADDITIVE]
        _, agree_base, _ = combine_subscores(base)
        # Add a very low RISK score (which would be "bearish" if it counted).
        _, agree_with_risk, _ = combine_subscores(
            base + [_sub(ScoreCategory.RISK, 5.0)]
        )
        assert agree_with_risk == pytest.approx(agree_base)


# ---------------------------------------------------------------------------
# combine_subscores: confidence & weighting behaviour
# ---------------------------------------------------------------------------


class TestConfidenceWeighting:
    def test_low_confidence_category_pulls_composite_less(self) -> None:
        """A category is weighted by weight*confidence, so thin data counts less."""
        # A bullish-but-blind category vs the same category at full confidence,
        # against a bearish backdrop. The confident bullish one should move the
        # composite more.
        backdrop = [_sub(cat, 30.0) for cat in _ADDITIVE[1:]]
        blind = backdrop + [_sub(ScoreCategory.VALUE, 95.0, confidence=0.05)]
        confident = backdrop + [_sub(ScoreCategory.VALUE, 95.0, confidence=0.95)]
        comp_blind, _, _ = combine_subscores(blind)
        comp_conf, _, _ = combine_subscores(confident)
        assert comp_conf > comp_blind

    def test_explicit_weights_override_subscore_weight(self) -> None:
        """A passed weights map (enum or string keys) overrides per-sub weights."""
        subs = [
            _sub(ScoreCategory.VALUE, 90.0, weight=0.01),
            _sub(ScoreCategory.GROWTH, 10.0, weight=0.01),
        ]
        # Weight VALUE far above GROWTH -> composite leans toward 90.
        by_enum = {ScoreCategory.VALUE: 0.9, ScoreCategory.GROWTH: 0.1}
        comp_enum, _, _ = combine_subscores(subs, by_enum)
        # Same via string keys (the function accepts both).
        by_str = {"value": 0.9, "growth": 0.1}
        comp_str, _, _ = combine_subscores(subs, by_str)
        assert comp_enum == pytest.approx(comp_str)
        assert comp_enum > 50.0

    def test_overall_confidence_in_unit_interval(self) -> None:
        """Overall confidence always lands in [0, 1]."""
        subs = [_sub(cat, 60.0, confidence=0.7, coverage=0.6) for cat in _ADDITIVE]
        _, _, confidence = combine_subscores(subs)
        assert 0.0 <= confidence <= 1.0
        assert confidence == pytest.approx(0.7, abs=1e-6)

    def test_determinism(self) -> None:
        """Identical inputs yield identical outputs (pure function)."""
        subs = [_sub(cat, 55.0 + i) for i, cat in enumerate(_ADDITIVE)]
        a = combine_subscores(subs)
        b = combine_subscores(subs)
        assert a == b


# ---------------------------------------------------------------------------
# Pure helper coverage
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_clamp(self) -> None:
        assert clamp(150.0) == 100.0
        assert clamp(-10.0) == 0.0
        assert clamp(42.0) == 42.0
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_scale_to_score_direction(self) -> None:
        # higher_is_better: at hi -> 100, at lo -> 0.
        assert scale_to_score(10.0, 0.0, 10.0) == pytest.approx(100.0)
        assert scale_to_score(0.0, 0.0, 10.0) == pytest.approx(0.0)
        # inverted: a low value is more attractive.
        assert scale_to_score(0.0, 0.0, 10.0, higher_is_better=False) == pytest.approx(100.0)
        # missing input -> None (never fabricated).
        assert scale_to_score(None, 0.0, 10.0) is None

    def test_percentile_rank(self) -> None:
        dist = [1.0, 2.0, 3.0, 4.0]
        assert percentile_rank(4.0, dist) == pytest.approx(1.0)  # at the max
        assert percentile_rank(1.0, dist) == pytest.approx(0.25)  # at the min
        assert percentile_rank(None, dist) is None
        assert percentile_rank(2.0, []) is None  # empty distribution

    def test_weighted_mean_ignores_none(self) -> None:
        # None values (and their weights) drop out of the mean.
        assert weighted_mean([10.0, None, 20.0], [1.0, 5.0, 1.0]) == pytest.approx(15.0)
        # All-None / zero-weight -> None.
        assert weighted_mean([None, None], [1.0, 1.0]) is None
        assert weighted_mean([1.0, 2.0], [0.0, 0.0]) is None

    def test_logistic_score_monotonic_and_bounded(self) -> None:
        lo = logistic_score(-10.0, midpoint=0.0, steepness=1.0)
        mid = logistic_score(0.0, midpoint=0.0, steepness=1.0)
        hi = logistic_score(10.0, midpoint=0.0, steepness=1.0)
        assert lo is not None and mid is not None and hi is not None
        assert 0.0 <= lo < mid < hi <= 100.0
        assert mid == pytest.approx(50.0)
        assert logistic_score(None, midpoint=0.0) is None
