"""MANAGEMENT analyzer — capital-allocation discipline of owner-operators.

This module is part of Convexity, an evidence-driven equity **research and
screening** tool. It is **not** a predictor and **not** investment advice. The
score produced here is a transparent aggregation of several *independent*
pieces of evidence about how a management team allocates capital — it never
implies certainty, a forecast, or a guaranteed outcome. As with every category
in Convexity, a single management signal should never carry a thesis on its own;
conviction is justified only when this evidence agrees with many *other*
independent categories.

What "good capital allocation" means here
-----------------------------------------
The classic owner-operator question is: *for every dollar that flows through
this business, does management create value or destroy it?* We approximate an
answer from four independent, auditable strands of evidence found in the
provider-sourced :class:`~convexity.core.models.SecurityData`:

1. **Share-count trend (dilution vs buybacks).** Persistent growth in
   ``shares_diluted`` across fiscal periods silently transfers value away from
   existing owners (dilution); a shrinking share count returns it (buybacks).
   We compute the compound annualised change in diluted shares over the
   available fundamentals history. Steady shrinkage scores well, steady
   dilution scores poorly.

2. **Insider ownership & recent open-market buying.** Aligned owner-operators
   put their own money at risk. We read disclosed
   :class:`~convexity.core.models.InsiderTransaction` records and measure recent
   *net* insider buying (buys minus sells, by dollar value where available, else
   by transaction count). Net buying is a costly, honest signal of alignment;
   heavy net selling is a (weaker, noisier) caution.

3. **Incremental returns on invested capital (reinvestment quality).** A great
   allocator reinvests at high and ideally *rising* returns. We compare the
   trailing ``roic`` of the most recent period with an older period to estimate
   whether reinvestment is compounding or eroding returns, blended with the
   absolute level of ROIC so a high, stable allocator is rewarded even without a
   rising trend.

4. **Shareholder-return discipline (buyback / dividend affordability).** Returning
   capital is only disciplined when it is *funded by* free cash flow rather than
   debt. We sanity-check that buybacks/dividends (inferred from a falling share
   count alongside positive free cash flow) are affordable, and we lightly
   reward a self-funding profile.

Each strand yields a 0–100 component score (higher = more attractive). The
sub-score is their data-coverage-weighted blend, the ``confidence`` and
``data_coverage`` reflect how much of this evidence was actually present, and
every component is backed by an :class:`~convexity.core.models.Evidence` item
citing the concrete numbers. When the required inputs are absent we return a
neutral, low-confidence sub-score (via :meth:`Analyzer.neutral_subscore`) rather
than guessing — a data gap must neither help nor hurt a company.

Relative scoring
----------------
Where ``ctx.peer_stats`` / ``ctx.universe_stats`` supply distributions for the
metrics we compute (``share_count_cagr``, ``roic``, ``incremental_roic``), the
analyzer scores the security by its percentile *within that comparable set*
rather than against absolute thresholds alone — what counts as disciplined
share management is sector- and size-relative for micro-caps. When no comparison
set is supplied the analyzer degrades gracefully to sensible absolute bands.

Purity
------
:meth:`ManagementAnalyzer.analyze` is pure: it performs no I/O, reads no
wall-clock, uses no randomness, and operates only on the passed
:class:`~convexity.core.models.SecurityData` and
:class:`~convexity.core.contracts.AnalysisContext`. Given identical inputs it
returns an identical sub-score.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Optional, Set, Tuple

from convexity.core.contracts import AnalysisContext, Analyzer
from convexity.core.models import (
    Evidence,
    FundamentalsPeriod,
    InsiderTransaction,
    ScoreCategory,
    SecurityData,
    SubScore,
)
from convexity.core.registry import register_analyzer
from convexity.core.scoring import clamp, percentile_rank, scale_to_score

# Source label recorded on every Evidence item this analyzer emits. The
# underlying numbers come from provider-supplied fundamentals and insider
# disclosures aggregated into SecurityData; we attribute to that lineage.
_SOURCE = "fundamentals/insider disclosures"

# Minimum number of fundamentals periods needed to estimate a share-count or
# ROIC *trend*. With a single period there is no trend to measure.
_MIN_TREND_PERIODS = 2

# Absolute fallback bands (used only when no peer/universe distribution exists).
# share_count_cagr: annualised growth in diluted shares. Negative == buybacks
# (good), positive == dilution (bad). +20%/yr dilution -> 0, -5%/yr buyback ->100.
_SC_CAGR_DILUTION_CAP = 0.20   # +20%/yr maps to score 0
_SC_CAGR_BUYBACK_FLOOR = -0.05  # -5%/yr maps to score 100

# Absolute ROIC level band: 0% -> 0, 25% -> 100 (a strong reinvestment engine).
_ROIC_LO = 0.0
_ROIC_HI = 0.25

# Incremental ROIC (change in ROIC across the window) band: -10pp -> 0, +10pp ->100.
_DROIC_LO = -0.10
_DROIC_HI = 0.10

# Component blend weights (within the management score, before coverage scaling).
# Share-count discipline and insider alignment are the most direct, hardest-to-
# fake signals of owner-operator behaviour, so they carry the most weight.
_W_SHARE_COUNT = 0.35
_W_INSIDER = 0.30
_W_INCREMENTAL_ROIC = 0.25
_W_RETURN_DISCIPLINE = 0.10


def _safe_positive(value: Optional[float]) -> Optional[float]:
    """Return ``value`` only if it is a finite, strictly positive number."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if v != v or v <= 0.0:  # NaN or non-positive
        return None
    return v


def _share_count_cagr(periods: Sequence[FundamentalsPeriod]) -> Optional[Tuple[float, float, float, int]]:
    """Compound annualised change in diluted share count over the history.

    ``periods`` is newest-first (as stored on :class:`SecurityData`). We pair the
    most recent period that reports a positive ``shares_diluted`` with the oldest
    such period and compute the per-period geometric rate of change.

    Returns ``(cagr, newest_shares, oldest_shares, n_steps)`` where ``cagr`` is the
    per-period compound rate (negative == net buybacks, positive == net dilution)
    and ``n_steps`` is the number of compounding intervals between the two
    endpoints. Returns ``None`` when fewer than two periods carry a usable share
    count.
    """
    usable: List[float] = []
    for p in periods:
        s = _safe_positive(p.shares_diluted)
        if s is not None:
            usable.append(s)
    if len(usable) < _MIN_TREND_PERIODS:
        return None
    newest = usable[0]
    oldest = usable[-1]
    n_steps = len(usable) - 1
    # Geometric per-period rate: (newest / oldest) ** (1/n) - 1.
    ratio = newest / oldest
    cagr = ratio ** (1.0 / n_steps) - 1.0
    return cagr, newest, oldest, n_steps


def _net_insider_buying(transactions: Sequence[InsiderTransaction]) -> Optional[Tuple[float, int, int, bool]]:
    """Summarise net insider buying from disclosed transactions.

    Buys (and exercises that are open-market purchases) count positively; sells
    count negatively. We prefer dollar ``value`` when present (more meaningful
    than raw share counts across price levels) and fall back to share counts, and
    finally to a simple +1/-1 per transaction when neither magnitude is given.

    Returns ``(net_magnitude, buy_count, sell_count, magnitude_is_dollars)`` or
    ``None`` when there are no buy/sell transactions to summarise.
    """
    buy_mag = 0.0
    sell_mag = 0.0
    buy_count = 0
    sell_count = 0
    used_dollars = False
    used_any_magnitude = False
    saw_relevant = False

    for t in transactions:
        kind = (t.transaction_type or "").strip().lower()
        if kind.startswith("buy") or kind in {"purchase", "p", "acquire", "acquisition"}:
            sign = 1.0
            buy_count += 1
        elif kind.startswith("sell") or kind in {"sale", "s", "dispose", "disposition"}:
            sign = -1.0
            sell_count += 1
        else:
            # grants, option exercises, gifts, etc. are not open-market conviction
            # signals — skip them so they neither help nor hurt.
            continue
        saw_relevant = True

        val = t.value
        if val is not None and float(val) > 0.0:
            magnitude = abs(float(val))
            used_dollars = True
            used_any_magnitude = True
        elif t.shares is not None and float(t.shares) > 0.0:
            magnitude = abs(float(t.shares))
            used_any_magnitude = True
        else:
            magnitude = 1.0  # unit weight when no magnitude is disclosed

        if sign > 0.0:
            buy_mag += magnitude
        else:
            sell_mag += magnitude

    if not saw_relevant:
        return None

    net = buy_mag - sell_mag
    # If *no* transaction carried a real magnitude, the dollar flag is meaningless.
    magnitude_is_dollars = used_dollars and used_any_magnitude
    return net, buy_count, sell_count, magnitude_is_dollars


def _incremental_roic(periods: Sequence[FundamentalsPeriod]) -> Optional[Tuple[float, float, float]]:
    """Estimate the change in ROIC across the available history.

    ``periods`` is newest-first. We compare the most recent period reporting a
    ``roic`` with the oldest such period. Returns ``(delta_roic, latest_roic,
    oldest_roic)`` (deltas as a fraction, e.g. +0.05 == +5pp) or ``None`` when
    fewer than two periods report ROIC.
    """
    roics: List[float] = []
    for p in periods:
        if p.roic is not None:
            try:
                roics.append(float(p.roic))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
    if len(roics) < _MIN_TREND_PERIODS:
        return None
    latest = roics[0]
    oldest = roics[-1]
    return latest - oldest, latest, oldest


def _score_from_percentile_or_band(
    value: Optional[float],
    *,
    distribution: Optional[Sequence[float]],
    band_lo: float,
    band_hi: float,
    higher_is_better: bool,
) -> Optional[float]:
    """Percentile-rank ``value`` within ``distribution`` when present, else band-scale.

    Returns a 0–100 score or ``None`` when ``value`` is missing. When a peer /
    universe ``distribution`` is supplied the score is relative (percentile);
    otherwise it falls back to an absolute linear band via
    :func:`~convexity.core.scoring.scale_to_score`.
    """
    if value is None:
        return None
    if distribution:
        pr = percentile_rank(value, distribution)
        if pr is not None:
            if not higher_is_better:
                pr = 1.0 - pr
            return clamp(pr * 100.0, 0.0, 100.0)
    return scale_to_score(value, band_lo, band_hi, higher_is_better=higher_is_better)


def _distribution(ctx: AnalysisContext, key: str) -> Optional[List[float]]:
    """Pull a numeric distribution for ``key`` from peer stats, else universe stats.

    Peer comparisons are preferred (tighter, more relevant than the whole
    universe). Returns a cleaned list of floats or ``None`` when neither context
    supplies a usable, non-empty distribution for ``key``.
    """
    for source in (ctx.peer_stats, ctx.universe_stats):
        if not source:
            continue
        raw = source.get(key)
        if raw is None:
            continue
        try:
            cleaned = [float(v) for v in raw if v is not None]
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if cleaned:
            return cleaned
    return None


@register_analyzer
class ManagementAnalyzer(Analyzer):
    """Score capital-allocation discipline of management / owner-operators.

    Combines four independent, auditable signals — share-count trend (dilution
    vs buybacks), insider ownership / recent net buying, incremental ROIC, and
    shareholder-return affordability — into a single 0–100
    :class:`~convexity.core.models.SubScore` for
    :attr:`~convexity.core.models.ScoreCategory.MANAGEMENT`. Higher = a more
    aligned, value-creating allocator. The score is *evidence*, not a prediction:
    it summarises what the disclosed numbers say about past behaviour, nothing
    more.
    """

    category: ScoreCategory = ScoreCategory.MANAGEMENT
    default_weight: float = 0.06
    requires: Set[str] = {"fundamentals"}

    def analyze(self, data: SecurityData, ctx: AnalysisContext) -> SubScore:
        """Produce the MANAGEMENT sub-score for ``data`` (pure; no I/O).

        Returns a neutral, low-confidence sub-score when neither a multi-period
        fundamentals history nor any insider transactions are available — there
        is then no real capital-allocation evidence to score, and a data gap must
        not move the composite.
        """
        periods = data.fundamentals  # newest-first
        insiders = data.insider_transactions

        has_trend_history = len([p for p in periods if p.shares_diluted is not None]) >= _MIN_TREND_PERIODS
        has_roic_history = len([p for p in periods if p.roic is not None]) >= _MIN_TREND_PERIODS
        has_insiders = bool(insiders)

        if not (has_trend_history or has_roic_history or has_insiders):
            return self.neutral_subscore(
                rationale=(
                    "No multi-period share-count / ROIC history and no insider "
                    "transactions were available, so management's capital "
                    "allocation cannot be assessed from the data on hand."
                ),
                coverage=0.0,
            )

        evidence: List[Evidence] = []
        flags: List[str] = []
        # Each entry: (component_score 0..100, component_weight). Only components
        # with real data are appended, so coverage reflects what truly existed.
        components: List[Tuple[float, float]] = []

        # --- 1. Share-count trend (dilution vs buybacks) ----------------------
        sc = _share_count_cagr(periods)
        if sc is not None:
            cagr, newest, oldest, n_steps = sc
            dist = _distribution(ctx, "share_count_cagr")
            # Lower share-count growth (more buybacks) is better. The band runs
            # from the favourable buyback floor (-5%/yr) up to the unfavourable
            # dilution cap (+20%/yr); ``higher_is_better=False`` then maps heavy
            # dilution to 0 and net buybacks to 100.
            sc_score = _score_from_percentile_or_band(
                cagr,
                distribution=dist,
                band_lo=_SC_CAGR_BUYBACK_FLOOR,
                band_hi=_SC_CAGR_DILUTION_CAP,
                higher_is_better=False,
            )
            if sc_score is not None:
                components.append((sc_score, _W_SHARE_COUNT))
                if cagr > 0.0:
                    direction = "bearish"
                elif cagr < 0.0:
                    direction = "bullish"
                else:
                    direction = "neutral"
                evidence.append(
                    Evidence.from_number(
                        "Diluted share count CAGR",
                        cagr * 100.0,
                        source=_SOURCE,
                        direction=direction,
                        unit="%/period",
                        precision=1,
                        detail=(
                            f"{oldest:,.0f} -> {newest:,.0f} diluted shares over "
                            f"{n_steps} period(s); "
                            + ("net buybacks" if cagr < 0 else "net dilution" if cagr > 0 else "flat")
                        ),
                    )
                )
                if cagr >= 0.10:
                    flags.append("HEAVY_DILUTION")
                elif cagr <= -0.02:
                    flags.append("NET_BUYBACKS")
        else:
            flags.append("NO_SHARE_COUNT_TREND")

        # --- 2. Insider ownership / recent net buying -------------------------
        ins = _net_insider_buying(insiders)
        if ins is not None:
            net, buy_count, sell_count, is_dollars = ins
            total = buy_count + sell_count
            # Net-buy *ratio* of transactions is a robust, scale-free alignment
            # proxy; the dollar/share magnitude only sets direction & emphasis.
            buy_ratio = (buy_count - sell_count) / total if total else 0.0
            # Map buy_ratio in [-1, 1] onto [0, 100]; net buying -> >50.
            insider_score = clamp((buy_ratio + 1.0) * 50.0, 0.0, 100.0)
            # Nudge for the magnitude of net buying when its sign is positive and
            # we have a real magnitude — costly, large open-market buys are the
            # strongest alignment signal of all.
            if net > 0.0:
                insider_score = clamp(insider_score + 5.0, 0.0, 100.0)
            components.append((insider_score, _W_INSIDER))

            net_direction = "bullish" if net > 0.0 else "bearish" if net < 0.0 else "neutral"
            magnitude_unit = "$" if is_dollars else ""
            evidence.append(
                Evidence.from_number(
                    "Net insider buying",
                    net if is_dollars else net,
                    source=_SOURCE,
                    direction=net_direction,
                    unit=magnitude_unit,
                    precision=0,
                    detail=(
                        f"{buy_count} buy / {sell_count} sell insider transaction(s); "
                        + ("net open-market buying" if net > 0 else "net selling" if net < 0 else "balanced")
                        + (" (by $ value)" if is_dollars else " (by count/shares)")
                    ),
                )
            )
            if net > 0.0 and buy_count > 0:
                flags.append("INSIDER_BUYING")
            elif net < 0.0 and sell_count > buy_count:
                flags.append("INSIDER_SELLING")
        else:
            flags.append("NO_INSIDER_ACTIVITY")

        # --- 3. Incremental ROIC (reinvestment quality) -----------------------
        droic = _incremental_roic(periods)
        latest_roic_for_level: Optional[float] = None
        if droic is not None:
            delta, latest_roic, oldest_roic = droic
            latest_roic_for_level = latest_roic
            # Trend component (is reinvestment compounding returns?).
            dist_d = _distribution(ctx, "incremental_roic")
            trend_score = _score_from_percentile_or_band(
                delta,
                distribution=dist_d,
                band_lo=_DROIC_LO,
                band_hi=_DROIC_HI,
                higher_is_better=True,
            )
            # Level component (is the absolute return high?).
            dist_l = _distribution(ctx, "roic")
            level_score = _score_from_percentile_or_band(
                latest_roic,
                distribution=dist_l,
                band_lo=_ROIC_LO,
                band_hi=_ROIC_HI,
                higher_is_better=True,
            )
            blended = [s for s in (trend_score, level_score) if s is not None]
            if blended:
                # Weight level and trend equally: a high, stable allocator and a
                # rapidly improving one both deserve credit.
                roic_score = sum(blended) / len(blended)
                components.append((roic_score, _W_INCREMENTAL_ROIC))
                delta_dir = "bullish" if delta > 0.005 else "bearish" if delta < -0.005 else "neutral"
                evidence.append(
                    Evidence.from_number(
                        "Incremental ROIC",
                        delta * 100.0,
                        source=_SOURCE,
                        direction=delta_dir,
                        unit="pp",
                        precision=1,
                        detail=(
                            f"ROIC {oldest_roic * 100:,.1f}% -> {latest_roic * 100:,.1f}% "
                            "across the available history"
                        ),
                    )
                )
                evidence.append(
                    Evidence.from_number(
                        "Latest ROIC",
                        latest_roic * 100.0,
                        source=_SOURCE,
                        direction="bullish" if latest_roic > 0.10 else "bearish" if latest_roic < 0.0 else "neutral",
                        unit="%",
                        precision=1,
                    )
                )
                if latest_roic < 0.0:
                    flags.append("NEGATIVE_ROIC")
        else:
            # Fall back to a single-period ROIC *level* if a trend is unavailable
            # but the latest period still reports one — partial but real evidence.
            latest = data.latest_fundamentals
            if latest is not None and latest.roic is not None:
                latest_roic_for_level = float(latest.roic)
                dist_l = _distribution(ctx, "roic")
                level_score = _score_from_percentile_or_band(
                    latest_roic_for_level,
                    distribution=dist_l,
                    band_lo=_ROIC_LO,
                    band_hi=_ROIC_HI,
                    higher_is_better=True,
                )
                if level_score is not None:
                    components.append((level_score, _W_INCREMENTAL_ROIC))
                    evidence.append(
                        Evidence.from_number(
                            "Latest ROIC",
                            latest_roic_for_level * 100.0,
                            source=_SOURCE,
                            direction=(
                                "bullish" if latest_roic_for_level > 0.10
                                else "bearish" if latest_roic_for_level < 0.0
                                else "neutral"
                            ),
                            unit="%",
                            precision=1,
                            detail="single-period ROIC level (no multi-period trend available)",
                        )
                    )
                    flags.append("ROIC_LEVEL_ONLY")
                    if latest_roic_for_level < 0.0:
                        flags.append("NEGATIVE_ROIC")
            else:
                flags.append("NO_ROIC_HISTORY")

        # --- 4. Shareholder-return discipline (affordability) -----------------
        # Returning capital is disciplined only when funded by free cash flow.
        # We infer return-of-capital from a shrinking share count, and check it
        # against the latest free cash flow. This is a light confirmatory signal.
        latest = data.latest_fundamentals
        if sc is not None and latest is not None and latest.free_cash_flow is not None:
            cagr = sc[0]
            fcf = float(latest.free_cash_flow)
            returning_capital = cagr < 0.0  # net buybacks
            if returning_capital and fcf > 0.0:
                discipline_score = 80.0  # self-funded buybacks: disciplined
                disc_dir = "bullish"
                disc_detail = "net buybacks funded by positive free cash flow"
            elif returning_capital and fcf <= 0.0:
                discipline_score = 30.0  # buying back stock without FCF to fund it
                disc_dir = "bearish"
                disc_detail = "net buybacks despite non-positive free cash flow"
                flags.append("UNFUNDED_BUYBACKS")
            elif (not returning_capital) and fcf > 0.0:
                discipline_score = 55.0  # FCF-positive but diluting/holding — neutral+
                disc_dir = "neutral"
                disc_detail = "positive free cash flow; share count not reduced"
            else:
                discipline_score = 45.0  # diluting and FCF-negative — slight caution
                disc_dir = "bearish"
                disc_detail = "share count rising with non-positive free cash flow"
            components.append((discipline_score, _W_RETURN_DISCIPLINE))
            evidence.append(
                Evidence.from_number(
                    "Free cash flow (return-of-capital funding)",
                    fcf,
                    source=_SOURCE,
                    direction=disc_dir,
                    unit="",
                    precision=0,
                    detail=disc_detail,
                )
            )

        # --- Blend components into the final sub-score ------------------------
        if not components:
            # Inputs existed but none yielded a scoreable component (e.g. all
            # share counts were zero/None and ROIC was missing). Stay neutral.
            return self.neutral_subscore(
                rationale=(
                    "Capital-allocation inputs were present but too sparse to "
                    "score (no usable share-count trend, ROIC, or insider "
                    "activity)."
                ),
                coverage=0.0,
                extra_flags=flags or None,
            )

        total_weight = sum(w for _, w in components)
        score = sum(s * w for s, w in components) / total_weight if total_weight else 50.0
        score = clamp(score, 0.0, 100.0)

        # data_coverage: how much of the *possible* evidence weight we actually
        # filled. The four strands carry the constant weights defined above; we
        # measure realised weight against the full available menu.
        max_weight = _W_SHARE_COUNT + _W_INSIDER + _W_INCREMENTAL_ROIC + _W_RETURN_DISCIPLINE
        data_coverage = clamp(total_weight / max_weight, 0.0, 1.0)

        # Confidence rises with coverage and with the number of independent
        # strands that agreed enough to be measured; capped below 1.0 because
        # disclosed history is always an imperfect proxy for future allocation.
        n_strands = len(components)
        confidence = clamp(0.20 + 0.55 * data_coverage + 0.06 * (n_strands - 1), 0.0, 0.9)

        rationale = self._build_rationale(score, components, latest_roic_for_level, flags)

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
        components: List[Tuple[float, float]],
        latest_roic: Optional[float],
        flags: List[str],
    ) -> str:
        """Compose a short, honest, human-readable rationale string.

        Describes the overall read in plain language and names the standout
        signals that drove it, without overstating certainty.
        """
        if score >= 70.0:
            head = "Capital allocation looks disciplined and owner-aligned"
        elif score >= 55.0:
            head = "Capital allocation looks reasonable, with some positive signals"
        elif score >= 45.0:
            head = "Capital-allocation evidence is mixed or neutral"
        else:
            head = "Capital allocation shows concerning patterns"

        notes: List[str] = []
        if "NET_BUYBACKS" in flags:
            notes.append("a shrinking share count (net buybacks)")
        if "HEAVY_DILUTION" in flags:
            notes.append("persistent share dilution")
        if "INSIDER_BUYING" in flags:
            notes.append("net insider open-market buying")
        if "INSIDER_SELLING" in flags:
            notes.append("net insider selling")
        if latest_roic is not None and latest_roic >= 0.10:
            notes.append(f"a solid ~{latest_roic * 100:.0f}% ROIC")
        if "NEGATIVE_ROIC" in flags:
            notes.append("negative returns on invested capital")
        if "UNFUNDED_BUYBACKS" in flags:
            notes.append("buybacks not covered by free cash flow")

        if notes:
            body = "; key signals: " + ", ".join(notes) + "."
        else:
            body = " based on the available capital-allocation evidence."

        coverage_caveat = (
            " This is research evidence about past behaviour, not a prediction; "
            "weigh it alongside the other independent categories."
        )
        return head + body + coverage_caveat


__all__ = ["ManagementAnalyzer"]
