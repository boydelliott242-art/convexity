"""Ownership analyzer — transparent "smart-money" signals for one security.

Part of Convexity, an evidence-driven equity **research and screening** tool. It
is **not** a predictor and **not** investment advice. This analyzer turns the raw
ownership records attached to a :class:`~convexity.core.models.SecurityData`
(insider Form-4-style transactions and institutional 13F-style holdings) into a
single auditable :class:`~convexity.core.models.SubScore` for the
:class:`~convexity.core.models.ScoreCategory.OWNERSHIP` category.

What "ownership" evidence means here
------------------------------------
Two independent, well-studied behavioural signals are aggregated:

* **Insider activity.** Corporate insiders (officers, directors) buy on the open
  market for essentially one reason — they expect the shares to be worth more —
  whereas they *sell* for many unrelated reasons (diversification, taxes, option
  exercises). Net **open-market buying**, and especially a *cluster* of several
  distinct insiders buying around the same time, is therefore treated as a
  bullish ownership signal; heavy net selling is treated as a mild bearish one.
* **Institutional ownership.** The breadth of institutional holders and whether
  they are, in aggregate, **accumulating** (increasing positions) or
  **distributing** (cutting positions) period-over-period. Many independent
  professional holders adding to a thinly-covered micro-cap is corroborating
  evidence; broad distribution is a caution.

Honesty constraints (non-negotiable)
------------------------------------
* **Pure.** No I/O, no wall-clock, no randomness — operates only on the passed
  ``SecurityData`` so the score is reproducible and auditable.
* **No fabrication.** When the required ownership records are absent the analyzer
  returns :meth:`Analyzer.neutral_subscore` (score 50, low confidence,
  ``MISSING_DATA``) rather than guessing. Partial data lowers ``confidence`` and
  ``data_coverage`` so the honesty of the score is always visible.
* **Not a forecast.** Each :class:`~convexity.core.models.Evidence` item reports a
  concrete observed number (net insider buying, distinct buyer count, holder
  count, average institutional change) and an honest direction. None of it claims
  a price will move; it is one of *many independent* evidence categories whose
  agreement — not any single signal — is where conviction comes from.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Set, Tuple

from convexity.core.contracts import AnalysisContext, Analyzer
from convexity.core.models import (
    Evidence,
    InsiderTransaction,
    InstitutionalHolding,
    ScoreCategory,
    SecurityData,
    SubScore,
)
from convexity.core.registry import register_analyzer
from convexity.core.scoring import (
    clamp,
    logistic_score,
    percentile_rank,
    scale_to_score,
)

# Source label recorded on every Evidence item this analyzer emits.
_SOURCE = "Convexity ownership analyzer"

# Transaction-type tokens that count as open-market acquisitions vs disposals.
# Grants, awards and option *exercises* are excluded from the "open-market"
# conviction signal because they are compensation mechanics, not a deliberate
# cash purchase that reveals a view. They are still surfaced as context.
_BUY_TOKENS: Tuple[str, ...] = ("buy", "purchase", "acquire", "acquisition")
_SELL_TOKENS: Tuple[str, ...] = ("sell", "sale", "dispose", "disposition")
_NEUTRAL_TOKENS: Tuple[str, ...] = ("grant", "award", "exercise", "gift", "convert")


def _classify_transaction(tx: InsiderTransaction) -> str:
    """Classify an insider transaction as ``"buy"``, ``"sell"`` or ``"other"``.

    Classification is driven by the free-text ``transaction_type`` token (matched
    case-insensitively). Grants/awards/exercises are deliberately ``"other"`` so
    they neither help nor hurt the open-market conviction signal.
    """
    kind = (tx.transaction_type or "").strip().lower()
    if not kind:
        return "other"
    # Neutral compensation mechanics take precedence (an "option exercise" may
    # contain the token "buy"-adjacent wording but is not an open-market buy).
    if any(tok in kind for tok in _NEUTRAL_TOKENS):
        return "other"
    if any(tok in kind for tok in _BUY_TOKENS):
        return "buy"
    if any(tok in kind for tok in _SELL_TOKENS):
        return "sell"
    return "other"


def _transaction_magnitude(tx: InsiderTransaction) -> Optional[float]:
    """Best available non-negative magnitude (dollar ``value``, else ``shares``).

    Returns ``None`` when neither is known, so a record with no size contributes
    to the *direction* tally only (as a discrete buy/sell) and not to the netted
    magnitude — never fabricating a size.
    """
    if tx.value is not None:
        return abs(float(tx.value))
    if tx.shares is not None:
        return abs(float(tx.shares))
    return None


def _summarize_insiders(
    transactions: Sequence[InsiderTransaction],
) -> Dict[str, Any]:
    """Aggregate insider transactions into auditable directional statistics.

    Returns a dict with: counts of buy/sell/other records, the set of distinct
    *buying* and *selling* insiders (for cluster detection), summed buy and sell
    magnitudes (dollars where available else shares), and the net-buy ratio in
    ``[-1, 1]`` (``(buy - sell) / (buy + sell)`` by magnitude, or by record count
    when no magnitudes are present).
    """
    buy_count = sell_count = other_count = 0
    buy_magnitude = sell_magnitude = 0.0
    magnitude_known = False
    buyers: Set[str] = set()
    sellers: Set[str] = set()

    for tx in transactions:
        klass = _classify_transaction(tx)
        name = (tx.insider_name or "").strip().lower()
        mag = _transaction_magnitude(tx)
        if klass == "buy":
            buy_count += 1
            if name:
                buyers.add(name)
            if mag is not None:
                buy_magnitude += mag
                magnitude_known = True
        elif klass == "sell":
            sell_count += 1
            if name:
                sellers.add(name)
            if mag is not None:
                sell_magnitude += mag
                magnitude_known = True
        else:
            other_count += 1

    # Net-buy ratio in [-1, 1]: prefer magnitude (dollars/shares) when any size is
    # known, else fall back to a count-based ratio so direction is still captured.
    if magnitude_known and (buy_magnitude + sell_magnitude) > 0.0:
        net_ratio: Optional[float] = (buy_magnitude - sell_magnitude) / (
            buy_magnitude + sell_magnitude
        )
    elif (buy_count + sell_count) > 0:
        net_ratio = (buy_count - sell_count) / float(buy_count + sell_count)
    else:
        net_ratio = None  # only neutral/grant records -> no directional signal.

    return {
        "buy_count": buy_count,
        "sell_count": sell_count,
        "other_count": other_count,
        "distinct_buyers": len(buyers),
        "distinct_sellers": len(sellers),
        "buy_magnitude": buy_magnitude if magnitude_known else None,
        "sell_magnitude": sell_magnitude if magnitude_known else None,
        "magnitude_known": magnitude_known,
        "net_ratio": net_ratio,
        "decisive_count": buy_count + sell_count,
    }


def _summarize_institutions(
    holdings: Sequence[InstitutionalHolding],
) -> Dict[str, Any]:
    """Aggregate institutional holdings into breadth and accumulation statistics.

    Returns the holder count, the count of holders with a known
    ``change_pct``, how many of those increased vs decreased their position, and
    the average ``change_pct`` across holders that report one (``None`` when none
    do). The average is winsor-free here but each individual change is clamped to a
    sane band so one mis-reported figure cannot dominate the mean.
    """
    holder_count = len(holdings)
    changes: List[float] = []
    increasing = decreasing = 0
    for h in holdings:
        if h.change_pct is None:
            continue
        # Clamp an individual reported change to +/-200% so a single absurd value
        # (e.g. a brand-new position reported as +10000%) cannot swamp the mean.
        chg = clamp(float(h.change_pct), -200.0, 200.0)
        changes.append(chg)
        if chg > 0.0:
            increasing += 1
        elif chg < 0.0:
            decreasing += 1

    avg_change: Optional[float] = sum(changes) / len(changes) if changes else None
    return {
        "holder_count": holder_count,
        "changes_known": len(changes),
        "increasing": increasing,
        "decreasing": decreasing,
        "avg_change_pct": avg_change,
    }


def _distribution_for(ctx: AnalysisContext, key: str) -> Optional[Sequence[float]]:
    """Pull a numeric peer/universe distribution for ``key`` from the context.

    Prefers peer stats (the tightest comparison) and falls back to the wider
    universe. Returns ``None`` when neither is available or the value is not a
    non-empty numeric sequence, so callers degrade to absolute bands gracefully.
    """
    for stats in (ctx.peer_stats, ctx.universe_stats):
        if not stats:
            continue
        dist = stats.get(key)
        if isinstance(dist, (list, tuple)) and dist:
            numeric = [float(v) for v in dist if isinstance(v, (int, float))]
            if numeric:
                return numeric
    return None


@register_analyzer
class OwnershipAnalyzer(Analyzer):
    """Score OWNERSHIP from insider transactions and institutional holdings.

    The score blends two independent sub-signals on a 0–100 scale (higher = more
    attractive ownership backdrop):

    * **Insider signal** — net open-market buying direction, with a bonus when a
      *cluster* of distinct insiders buys (a broadly-watched bullish pattern) and
      a penalty for cluster selling.
    * **Institutional signal** — breadth (number of holders, peer-relative when a
      distribution is supplied) and net accumulation vs distribution from
      period-over-period position changes.

    When only one sub-signal has data, the score rests on that signal alone with a
    correspondingly lower ``confidence``/``data_coverage``. When neither is present
    the analyzer returns a neutral, low-confidence sub-score (never a guess).
    """

    category: ScoreCategory = ScoreCategory.OWNERSHIP
    default_weight: float = 0.06
    requires: Set[str] = {"insider_transactions", "institutional_holdings"}

    # ----- tunable, documented constants (kept explicit for auditability) -----
    # A "cluster buy" needs at least this many *distinct* insiders buying.
    _CLUSTER_BUYERS = 2
    # Holder counts mapped onto 0..100 breadth when no peer distribution exists.
    _BREADTH_LO = 0.0
    _BREADTH_HI = 25.0
    # Logistic steepness for the net-buy ratio (centred at 0 == balanced).
    _INSIDER_STEEPNESS = 4.0
    # Average institutional change (%) mapped onto 0..100 when no distribution.
    _ACCUM_LO = -25.0
    _ACCUM_HI = 25.0

    def analyze(self, data: SecurityData, ctx: AnalysisContext) -> SubScore:
        """Return the OWNERSHIP :class:`SubScore` for ``data``.

        Pure: reads only ``data.insider_transactions`` /
        ``data.institutional_holdings`` and the comparative distributions in
        ``ctx``. Builds component scores for the insider and institutional signals,
        blends whatever is present, and attaches one Evidence item per concrete
        observation with an honest direction.
        """
        insiders = _summarize_insiders(data.insider_transactions)
        institutions = _summarize_institutions(data.institutional_holdings)

        evidence: List[Evidence] = []
        flags: List[str] = []
        component_scores: List[float] = []
        component_weights: List[float] = []
        # Track how many of the two required input families actually had usable
        # data, to drive data_coverage honestly.
        signals_present = 0

        # ------------------------------------------------------------------
        # 1. Insider signal
        # ------------------------------------------------------------------
        insider_score, insider_present = self._score_insiders(insiders, evidence, flags)
        if insider_present:
            component_scores.append(insider_score)
            # Insider open-market activity is a high-salience signal; weight it
            # slightly above the institutional breadth signal.
            component_weights.append(1.2)
            signals_present += 1

        # ------------------------------------------------------------------
        # 2. Institutional signal
        # ------------------------------------------------------------------
        inst_score, inst_present = self._score_institutions(institutions, ctx, evidence, flags)
        if inst_present:
            component_scores.append(inst_score)
            component_weights.append(1.0)
            signals_present += 1

        # ------------------------------------------------------------------
        # 3. No usable ownership data anywhere -> honest neutral fallback.
        # ------------------------------------------------------------------
        if signals_present == 0:
            return self.neutral_subscore(
                rationale=(
                    "No insider transactions or institutional holdings were available, "
                    "so the ownership ('smart-money') signal could not be assessed. "
                    "This is recorded as missing data, not a negative."
                ),
                coverage=0.0,
            )

        # ------------------------------------------------------------------
        # 4. Blend present components into the final 0..100 score.
        # ------------------------------------------------------------------
        total_w = sum(component_weights)
        blended = sum(s * w for s, w in zip(component_scores, component_weights)) / total_w
        score = clamp(blended, 0.0, 100.0)

        # data_coverage: fraction of the two required input families present.
        data_coverage = signals_present / 2.0

        # Confidence reflects both how many signals were available and how much
        # raw evidence underpinned them (more transactions / holders -> firmer).
        confidence = self._confidence(insiders, institutions, signals_present)

        rationale = self._rationale(insiders, institutions, score, signals_present)

        if signals_present == 1:
            flags.append("PARTIAL_OWNERSHIP_DATA")

        return SubScore(
            category=self.category,
            score=score,
            confidence=clamp(confidence, 0.0, 1.0),
            weight=self.default_weight,
            rationale=rationale,
            evidence=evidence,
            flags=flags,
            data_coverage=clamp(data_coverage, 0.0, 1.0),
        )

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    def _score_insiders(
        self,
        ins: Dict[str, Any],
        evidence: List[Evidence],
        flags: List[str],
    ) -> Tuple[float, bool]:
        """Score the insider sub-signal in 0..100 and append its evidence.

        Returns ``(score, present)`` where ``present`` is ``False`` when there were
        no buy/sell records to read a direction from (only grants/neutral records,
        or none at all).
        """
        decisive = int(ins["decisive_count"])
        net_ratio = ins["net_ratio"]

        if decisive == 0 or net_ratio is None:
            # Either no transactions, or only neutral/grant records: nothing to
            # read directionally. Surface a neutral context note if there were any.
            if int(ins["other_count"]) > 0:
                evidence.append(
                    Evidence.from_number(
                        "Insider open-market transactions",
                        0.0,
                        source=_SOURCE,
                        direction="neutral",
                        precision=0,
                        detail=(
                            f"{ins['other_count']} non-open-market insider record(s) "
                            "(grants/exercises) carry no directional signal."
                        ),
                    )
                )
            return 50.0, False

        # Base score from the net-buy ratio via a logistic centred at balanced (0).
        base = logistic_score(net_ratio, midpoint=0.0, steepness=self._INSIDER_STEEPNESS)
        if base is None:  # pragma: no cover - net_ratio is not None here.
            base = 50.0

        # Cluster adjustment: multiple distinct insiders buying is a stronger
        # bullish signal than a lone buyer; cluster selling is a mild caution.
        distinct_buyers = int(ins["distinct_buyers"])
        distinct_sellers = int(ins["distinct_sellers"])
        cluster_bonus = 0.0
        if distinct_buyers >= self._CLUSTER_BUYERS and net_ratio > 0.0:
            # +6 points per distinct buyer beyond the first, capped at +18.
            cluster_bonus = min(18.0, 6.0 * (distinct_buyers - 1))
            flags.append("INSIDER_CLUSTER_BUY")
        elif distinct_sellers >= self._CLUSTER_BUYERS and net_ratio < 0.0:
            cluster_bonus = -min(12.0, 4.0 * (distinct_sellers - 1))
            flags.append("INSIDER_CLUSTER_SELL")

        score = clamp(base + cluster_bonus, 0.0, 100.0)

        direction = "bullish" if net_ratio > 0.05 else "bearish" if net_ratio < -0.05 else "neutral"

        # Evidence: the net-buy ratio (signed) with concrete supporting counts.
        size_note = ""
        if ins["magnitude_known"]:
            bm = ins["buy_magnitude"] or 0.0
            sm = ins["sell_magnitude"] or 0.0
            size_note = f"; gross buys {bm:,.0f} vs sells {sm:,.0f} (value/shares)"
        evidence.append(
            Evidence.from_number(
                "Net insider buy ratio",
                net_ratio,
                source=_SOURCE,
                direction=direction,
                precision=2,
                detail=(
                    f"{ins['buy_count']} buy vs {ins['sell_count']} sell record(s) "
                    f"from {distinct_buyers} distinct buyer(s) / "
                    f"{distinct_sellers} seller(s){size_note}. "
                    "Ratio in [-1,1]; >0 = net open-market buying."
                ),
            )
        )
        if distinct_buyers >= self._CLUSTER_BUYERS and net_ratio > 0.0:
            evidence.append(
                Evidence.from_number(
                    "Distinct insider buyers (cluster)",
                    float(distinct_buyers),
                    source=_SOURCE,
                    direction="bullish",
                    precision=0,
                    detail="Multiple independent insiders buying on the open market.",
                )
            )

        return score, True

    def _score_institutions(
        self,
        inst: Dict[str, Any],
        ctx: AnalysisContext,
        evidence: List[Evidence],
        flags: List[str],
    ) -> Tuple[float, bool]:
        """Score the institutional sub-signal in 0..100 and append its evidence.

        Returns ``(score, present)``; ``present`` is ``False`` when there are no
        institutional holders at all (nothing to score).
        """
        holder_count = int(inst["holder_count"])
        if holder_count == 0:
            return 50.0, False

        # --- Breadth: peer-relative when a distribution is supplied, else band. --
        breadth_dist = _distribution_for(ctx, "institutional_holder_count")
        if breadth_dist is not None:
            pr = percentile_rank(float(holder_count), breadth_dist)
            breadth_score = (pr * 100.0) if pr is not None else 50.0
            breadth_basis = "peer-relative"
        else:
            breadth_score = scale_to_score(
                float(holder_count), self._BREADTH_LO, self._BREADTH_HI, higher_is_better=True
            )
            if breadth_score is None:  # pragma: no cover - holder_count is a number.
                breadth_score = 50.0
            breadth_basis = "absolute band"

        evidence.append(
            Evidence.from_number(
                "Institutional holders",
                float(holder_count),
                source=_SOURCE,
                direction="bullish" if breadth_score >= 60.0 else "neutral",
                precision=0,
                detail=f"Breadth of professional ownership ({breadth_basis}).",
            )
        )

        # --- Accumulation vs distribution from period-over-period changes. -------
        avg_change = inst["avg_change_pct"]
        changes_known = int(inst["changes_known"])
        if avg_change is not None and changes_known > 0:
            accum_dist = _distribution_for(ctx, "institutional_avg_change_pct")
            if accum_dist is not None:
                pr = percentile_rank(float(avg_change), accum_dist)
                accum_score = (pr * 100.0) if pr is not None else 50.0
            else:
                accum_score = scale_to_score(
                    float(avg_change), self._ACCUM_LO, self._ACCUM_HI, higher_is_better=True
                )
                if accum_score is None:  # pragma: no cover
                    accum_score = 50.0

            direction = (
                "bullish" if avg_change > 1.0 else "bearish" if avg_change < -1.0 else "neutral"
            )
            evidence.append(
                Evidence.from_number(
                    "Avg institutional position change",
                    avg_change,
                    source=_SOURCE,
                    direction=direction,
                    unit="%",
                    precision=1,
                    detail=(
                        f"{inst['increasing']} holder(s) adding vs {inst['decreasing']} "
                        f"trimming across {changes_known} with reported changes. "
                        ">0 = net accumulation."
                    ),
                )
            )
            if avg_change <= -1.0 and inst["decreasing"] > inst["increasing"]:
                flags.append("INSTITUTIONAL_DISTRIBUTION")
            # Blend breadth and accumulation (accumulation weighted slightly more
            # as it is a more direct smart-money read than raw breadth).
            score = (0.45 * breadth_score) + (0.55 * accum_score)
        else:
            # No change data: breadth is all we can read; note the gap.
            flags.append("NO_INSTITUTIONAL_CHANGE_DATA")
            score = breadth_score

        return clamp(score, 0.0, 100.0), True

    # ------------------------------------------------------------------
    # Confidence & rationale
    # ------------------------------------------------------------------

    def _confidence(
        self,
        ins: Dict[str, Any],
        inst: Dict[str, Any],
        signals_present: int,
    ) -> float:
        """Confidence in ``[0, 1]`` from breadth of evidence and signals present.

        Starts from how many of the two signal families were present (each worth up
        to 0.5) and scales each by how much raw evidence stood behind it (more
        decisive insider records and more institutional holders -> firmer).
        """
        conf = 0.0
        # Insider contribution (up to ~0.5), saturating around 6 decisive records.
        decisive = int(ins["decisive_count"])
        if decisive > 0:
            conf += 0.15 + 0.35 * min(decisive, 6) / 6.0
        # Institutional contribution (up to ~0.5), saturating around 15 holders.
        holders = int(inst["holder_count"])
        if holders > 0:
            holder_term = 0.15 + 0.25 * min(holders, 15) / 15.0
            # A reward for actually having change data (a richer read).
            if int(inst["changes_known"]) > 0:
                holder_term += 0.10
            conf += holder_term
        # If only one family was present, cap confidence so a single-signal read is
        # never over-trusted.
        if signals_present == 1:
            conf = min(conf, 0.5)
        return clamp(conf, 0.0, 1.0)

    def _rationale(
        self,
        ins: Dict[str, Any],
        inst: Dict[str, Any],
        score: float,
        signals_present: int,
    ) -> str:
        """One- to two-sentence human explanation tied to the observed numbers."""
        parts: List[str] = []
        decisive = int(ins["decisive_count"])
        if decisive > 0 and ins["net_ratio"] is not None:
            nr = ins["net_ratio"]
            lean = "net buying" if nr > 0.05 else "net selling" if nr < -0.05 else "balanced"
            parts.append(
                f"insiders show {lean} ({ins['buy_count']} buys / {ins['sell_count']} sells, "
                f"{ins['distinct_buyers']} distinct buyer(s))"
            )
        holders = int(inst["holder_count"])
        if holders > 0:
            if inst["avg_change_pct"] is not None:
                tilt = (
                    "accumulating"
                    if inst["avg_change_pct"] > 1.0
                    else "distributing"
                    if inst["avg_change_pct"] < -1.0
                    else "roughly flat"
                )
                parts.append(
                    f"{holders} institutional holder(s) are {tilt} "
                    f"(avg change {inst['avg_change_pct']:+.1f}%)"
                )
            else:
                parts.append(f"{holders} institutional holder(s) on record (no change data)")

        body = "; ".join(parts) if parts else "limited ownership records"
        coverage_note = (
            " Only one ownership signal was available, so confidence is capped."
            if signals_present == 1
            else ""
        )
        return (
            f"Ownership score {score:.0f}/100: {body}. This aggregates independent "
            f"smart-money signals as research evidence, not a recommendation or forecast."
            f"{coverage_note}"
        )


__all__ = ["OwnershipAnalyzer"]
