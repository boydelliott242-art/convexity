"""RISK analyzer — the composite-dampening "how likely is capital impairment?" score.

This module is part of Convexity, an evidence-driven equity **research and
screening** tool. It is **not** a predictor and **not** investment advice. The
``RiskAnalyzer`` does not forecast returns; it transparently aggregates *many
independent* warning signs that a small/micro-cap could impair shareholder
capital, so the ranking layer can temper a thesis when the evidence of fragility
is strong. No single signal drives the score — high conviction (here, a strong
*safety* read) is justified only when many independent risk checks agree, and
missing data lowers confidence rather than inventing a clean bill of health.

Score convention (documented and load-bearing)
----------------------------------------------
By the platform-wide convention, **a higher RISK sub-score means LOWER risk
(a safer profile)**, exactly like every other category where higher == more
attractive. A maximally fragile company scores near ``0``; a fortress balance
sheet with deep liquidity scores near ``100``. The ranking layer
(:func:`convexity.core.scoring.combine_subscores`) then applies RISK as a
*dampener*: a low (risky) score shaves points off the composite, a high (safe)
score leaves it essentially untouched. RISK therefore carries additive weight
``0.0`` in :data:`~convexity.core.config.DEFAULT_CATEGORY_WEIGHTS` — it penalises,
it does not contribute to the additive mean.

Risk dimensions assessed (each independent, each auditable)
-----------------------------------------------------------
* **Negative equity** — a shareholders' deficit is a hard red flag.
* **Leverage** — debt/equity and interest coverage (can the company service debt?).
* **Cash burn / runway** — months of runway = cash ÷ monthly burn when free cash
  flow is negative; a short runway implies imminent dilution or distress.
* **Dilution** — growth in diluted share count across reported periods.
* **Liquidity** — average daily dollar volume; a thin micro-float is hard to exit.
* **Volatility** — annualised realised volatility of daily returns.
* **Going-concern / litigation language** — going-concern, default, delisting,
  investigation and lawsuit wording detected in filings and news via the
  dependency-light :mod:`convexity.analysis.news_nlp` lexicon.

Each dimension yields a 0–100 safety score (higher == safer) for the inputs that
are actually present. They are combined with a *conservative* aggregation: the
blended mean is pulled toward the single worst dimension, because risk is about
the tail — one fatal flaw (e.g. a going-concern warning) should not be averaged
away by several benign metrics. Dimensions with no data are skipped and reduce
``data_coverage`` and ``confidence`` rather than being scored as "safe".

Purity
------
:meth:`RiskAnalyzer.analyze` is pure: it reads only the passed
:class:`~convexity.core.models.SecurityData` and
:class:`~convexity.core.contracts.AnalysisContext`. It performs no I/O, consults
no wall-clock and uses no randomness, so the score is reproducible and auditable.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import List, Optional, Tuple

from convexity.analysis.news_nlp import source_credibility
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

# ---------------------------------------------------------------------------
# Distress / litigation language lexicon
# ---------------------------------------------------------------------------
#
# Phrases (and a few single words) that, in a filing or news headline, evidence a
# *going-concern, solvency or litigation* risk specifically — a narrower, more
# severe set than the general negative sentiment lexicon. Each maps to a severity
# weight: the higher the weight, the more directly the phrase signals possible
# capital impairment. Detection is a transparent, case-insensitive regex match;
# we report exactly which phrase matched so the evidence is auditable. Matching a
# phrase is an *observation that the wording appears*, never a claim of outcome.

_DISTRESS_PATTERNS: List[Tuple[str, float, str]] = [
    # (regex, severity_weight, human_label)
    (r"going\s+concern", 1.0, "going-concern language"),
    (r"substantial\s+doubt", 1.0, "substantial-doubt language"),
    (r"\bbankrupt(?:cy)?\b", 1.0, "bankruptcy reference"),
    (r"chapter\s+(?:7|11)\b", 1.0, "Chapter 7/11 reference"),
    (r"\bdefault(?:ed)?\b", 0.8, "debt-default language"),
    (r"covenant\s+(?:breach|violation|default|waiver)", 0.8, "covenant breach/waiver"),
    (r"material\s+weakness", 0.7, "material-weakness disclosure"),
    (r"\bdelist(?:ed|ing)?\b", 0.7, "delisting reference"),
    (r"(?:below|non[-\s]?)compliance", 0.6, "listing non-compliance"),
    (r"reverse\s+(?:stock\s+)?split", 0.5, "reverse-split reference"),
    (r"restat(?:e|ed|ement)", 0.6, "financial restatement"),
    (r"\bimpairment\b", 0.4, "impairment charge"),
    (r"going\s+private|liquidat(?:e|ion|ing)", 0.7, "liquidation/going-private"),
    # Litigation / regulatory.
    (r"\blawsuit\b|\blitigation\b|\bsued?\b", 0.5, "litigation reference"),
    (r"\b(?:sec|doj|ftc)\s+investigation\b|\binvestigation\b|\bprobe\b|\bsubpoena\b", 0.6, "investigation/subpoena"),
    (r"\bfraud\b", 0.8, "fraud allegation"),
    (r"short[-\s]?seller|short\s+report", 0.4, "short-seller report"),
    (r"\brecall\b", 0.4, "product recall"),
]

_COMPILED_DISTRESS: List[Tuple[re.Pattern[str], float, str]] = [
    (re.compile(pat, re.IGNORECASE), weight, label) for pat, weight, label in _DISTRESS_PATTERNS
]


def _scan_distress_language(
    texts: Sequence[Tuple[Optional[str], Optional[str]]],
) -> List[Tuple[str, float, str]]:
    """Find distress/litigation phrases across (text, source) pairs.

    Args:
        texts: An iterable of ``(text, source)`` tuples (e.g. a filing's
            ``title``/``summary`` joined, paired with its source label). ``None``
            or empty text is skipped.

    Returns:
        A list of ``(human_label, severity, matched_text)`` detections, one per
        distinct distress *type* per text (first match of a type wins, so the
        same phrasing is not double-counted within one document). The list is in
        input/text order, then distress-pattern declaration order.
    """
    detections: List[Tuple[str, float, str]] = []
    for text, _source in texts:
        if not text:
            continue
        seen_labels = set()
        for pattern, weight, label in _COMPILED_DISTRESS:
            if label in seen_labels:
                continue
            m = pattern.search(text)
            if m:
                seen_labels.add(label)
                detections.append((label, weight, m.group(0).strip()))
    return detections


def _daily_log_returns(closes: Sequence[float]) -> List[float]:
    """Compute daily log returns from an oldest-first close series.

    Non-positive prices are skipped defensively (a log return is undefined). A
    series shorter than two valid prices yields an empty list.
    """
    rets: List[float] = []
    prev: Optional[float] = None
    for c in closes:
        if c is None or c <= 0:
            prev = None  # break the chain across a bad print.
            continue
        if prev is not None:
            rets.append(math.log(c / prev))
        prev = c
    return rets


def _annualised_volatility(closes: Sequence[float]) -> Optional[float]:
    """Annualised realised volatility (fraction) from daily closes, or ``None``.

    Uses the population standard deviation of daily log returns scaled by
    ``sqrt(252)``. Requires at least ``5`` valid returns to be meaningful;
    otherwise returns ``None`` (insufficient data).
    """
    rets = _daily_log_returns(closes)
    if len(rets) < 5:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    daily_sd = math.sqrt(var)
    return daily_sd * math.sqrt(252.0)


def _avg_dollar_volume(data: SecurityData, lookback: int = 30) -> Optional[float]:
    """Average daily dollar volume over the last ``lookback`` bars, or ``None``.

    Dollar volume per bar is ``close * volume``. Bars with a missing/non-positive
    close or volume are ignored. Returns ``None`` if no usable bars exist.
    """
    bars = data.price_history[-lookback:] if data.price_history else []
    dollar_vols: List[float] = []
    for bar in bars:
        if bar.close is not None and bar.close > 0 and bar.volume is not None and bar.volume > 0:
            dollar_vols.append(bar.close * bar.volume)
    if not dollar_vols:
        return None
    return sum(dollar_vols) / len(dollar_vols)


@register_analyzer
class RiskAnalyzer(Analyzer):
    """Score the RISK category — higher score == LOWER risk (safer profile).

    The analyzer inspects up to seven *independent* fragility dimensions
    (negative equity, leverage, cash-burn runway, dilution, liquidity,
    volatility and going-concern/litigation language), scores each one it has
    data for on a 0–100 *safety* scale, and conservatively blends them so a
    single fatal flaw is not averaged away. The result is the dampener the
    ranking layer applies to the composite.

    Class attributes:
        category: :attr:`ScoreCategory.RISK`.
        default_weight: ``0.0`` — RISK is applied as a dampener, never summed
            into the additive composite (see the module docstring and
            :data:`~convexity.core.config.DEFAULT_CATEGORY_WEIGHTS`).
        requires: The data inputs that, when present, let the analyzer produce a
            confident score (any one of which is enough to begin scoring).
    """

    category = ScoreCategory.RISK
    default_weight = 0.0
    requires = {"fundamentals", "price_history", "filings", "valuation"}

    # Number of distinct risk dimensions this analyzer can evaluate; used to
    # normalise data_coverage honestly against what *could* have been scored.
    _MAX_DIMENSIONS = 7

    def analyze(self, data: SecurityData, ctx: AnalysisContext) -> SubScore:
        """Return the RISK :class:`SubScore` (higher == safer) for ``data``.

        Pure: reads only ``data`` and ``ctx``. Each available dimension yields a
        0–100 safety score; the dimensions are conservatively combined (mean
        pulled toward the worst). When *no* dimension can be scored, returns
        :meth:`neutral_subscore` so a data gap neither helps nor hurts.
        """
        evidence: List[Evidence] = []
        flags: List[str] = []

        # Each entry: (safety_score_0_100, weight, dimension_name).
        dimension_scores: List[Tuple[float, float, str]] = []

        latest = data.latest_fundamentals

        # --- 1. Negative equity (hard red flag) ------------------------------
        if latest is not None and latest.total_equity is not None:
            equity = latest.total_equity
            if equity < 0:
                # A shareholders' deficit: pin this dimension to maximally risky.
                neg_equity_score = 5.0
                flags.append("NEGATIVE_EQUITY")
                direction = "bearish"
            else:
                # Equity cushion relative to assets, where available.
                if latest.total_assets and latest.total_assets > 0:
                    equity_ratio = equity / latest.total_assets
                    # 0 -> risky (0), >=0.5 equity/assets -> safe (100).
                    neg_equity_score = scale_to_score(equity_ratio, 0.0, 0.5, higher_is_better=True) or 50.0
                else:
                    neg_equity_score = 70.0  # positive equity, no asset base to compare.
                direction = "bullish" if neg_equity_score >= 60 else "neutral"
            dimension_scores.append((clamp(neg_equity_score), 1.4, "equity"))
            evidence.append(
                Evidence.from_number(
                    "Total equity",
                    equity,
                    source="fundamentals",
                    direction=direction,
                    unit="",
                    precision=0,
                    detail="Negative equity is a shareholders' deficit (high risk)."
                    if equity < 0
                    else "Positive shareholders' equity.",
                    as_of=latest.period_end,
                )
            )

        # --- 2. Leverage: debt/equity & interest coverage --------------------
        if latest is not None and latest.debt_to_equity is not None:
            d_e = latest.debt_to_equity
            # Lower D/E is safer. 0 -> 100 (safe), >= 3.0 -> 0 (very levered).
            # A negative D/E (from negative equity) is treated as maximally risky.
            if d_e < 0:
                de_score = 5.0
            else:
                de_score = scale_to_score(d_e, 0.0, 3.0, higher_is_better=False) or 50.0
            dimension_scores.append((clamp(de_score), 1.0, "leverage_de"))
            if d_e >= 2.0 or d_e < 0:
                flags.append("HIGH_LEVERAGE")
            evidence.append(
                Evidence.from_number(
                    "Debt / equity",
                    d_e,
                    source="fundamentals",
                    direction="bearish" if (d_e >= 2.0 or d_e < 0) else ("bullish" if d_e <= 0.5 else "neutral"),
                    precision=2,
                    detail="Higher leverage raises solvency risk.",
                    as_of=latest.period_end,
                )
            )

        if latest is not None and latest.interest_coverage is not None:
            cov = latest.interest_coverage
            # EBIT/interest: < 1 cannot cover interest (risky); >= 8 is comfortable.
            cov_score = scale_to_score(cov, 1.0, 8.0, higher_is_better=True) or 50.0
            dimension_scores.append((clamp(cov_score), 0.9, "interest_coverage"))
            if cov < 1.5:
                flags.append("THIN_INTEREST_COVERAGE")
            evidence.append(
                Evidence.from_number(
                    "Interest coverage",
                    cov,
                    source="fundamentals",
                    direction="bearish" if cov < 1.5 else ("bullish" if cov >= 5 else "neutral"),
                    unit="x",
                    precision=2,
                    detail="EBIT relative to interest expense; below ~1x cannot service debt.",
                    as_of=latest.period_end,
                )
            )

        # --- 3. Cash burn / runway ------------------------------------------
        runway_months = self._cash_runway_months(latest)
        if runway_months is not None:
            # > 36 months runway -> safe (100); <= 6 months -> very risky (0).
            runway_score = scale_to_score(runway_months, 6.0, 36.0, higher_is_better=True) or 50.0
            dimension_scores.append((clamp(runway_score), 1.3, "runway"))
            if runway_months <= 12.0:
                flags.append("SHORT_CASH_RUNWAY")
            evidence.append(
                Evidence.from_number(
                    "Cash runway",
                    runway_months,
                    source="fundamentals",
                    direction="bearish" if runway_months <= 12 else ("bullish" if runway_months >= 30 else "neutral"),
                    unit=" months",
                    precision=1,
                    detail="Cash & equivalents divided by monthly free-cash-flow burn.",
                    as_of=latest.period_end if latest else None,
                )
            )

        # --- 4. Dilution: growth in diluted share count ----------------------
        dilution = self._annual_dilution_rate(data.fundamentals)
        if dilution is not None:
            # Annualised share-count growth. 0% (or buybacks) -> safe (100);
            # >= 25%/yr dilution -> very risky (0).
            dil_score = scale_to_score(dilution, 0.0, 0.25, higher_is_better=False) or 50.0
            dimension_scores.append((clamp(dil_score), 1.0, "dilution"))
            if dilution >= 0.10:
                flags.append("HEAVY_DILUTION")
            evidence.append(
                Evidence.from_number(
                    "Annualised share dilution",
                    dilution * 100.0,
                    source="fundamentals",
                    direction="bearish" if dilution >= 0.10 else ("bullish" if dilution <= 0.0 else "neutral"),
                    unit="%/yr",
                    precision=1,
                    detail="Growth in diluted share count across reported periods (negative = buybacks).",
                )
            )

        # --- 5. Liquidity: average daily dollar volume -----------------------
        adv = _avg_dollar_volume(data)
        if adv is not None:
            liq_score = self._liquidity_score(adv, ctx)
            dimension_scores.append((clamp(liq_score), 0.9, "liquidity"))
            if adv < 200_000:
                flags.append("THIN_LIQUIDITY")
            evidence.append(
                Evidence.from_number(
                    "Avg daily $ volume",
                    adv,
                    source="price_history",
                    direction="bearish" if adv < 200_000 else ("bullish" if adv >= 2_000_000 else "neutral"),
                    precision=0,
                    detail="A thin micro-float is hard to exit without moving the price.",
                )
            )

        # --- 6. Volatility ---------------------------------------------------
        vol = _annualised_volatility([b.close for b in data.price_history])
        if vol is not None:
            # 30%/yr annualised vol -> safe (100); >= 120%/yr -> very risky (0).
            vol_score = scale_to_score(vol, 0.30, 1.20, higher_is_better=False) or 50.0
            dimension_scores.append((clamp(vol_score), 0.7, "volatility"))
            if vol >= 0.80:
                flags.append("EXTREME_VOLATILITY")
            evidence.append(
                Evidence.from_number(
                    "Annualised volatility",
                    vol * 100.0,
                    source="price_history",
                    direction="bearish" if vol >= 0.80 else ("bullish" if vol <= 0.40 else "neutral"),
                    unit="%",
                    precision=0,
                    detail="Realised volatility of daily returns (higher = more fragile price).",
                )
            )

        # --- 7. Going-concern / litigation language --------------------------
        distress_score, distress_evidence, distress_flags = self._distress_language_dimension(data)
        if distress_score is not None:
            dimension_scores.append((clamp(distress_score), 1.5, "distress_language"))
            evidence.extend(distress_evidence)
            flags.extend(distress_flags)

        # --- No data at all: honest neutral fallback -------------------------
        if not dimension_scores:
            return self.neutral_subscore(
                rationale=(
                    "No fundamentals, price history or filings were available to assess "
                    "risk; defaulting to a neutral, low-confidence score (risk neither "
                    "helps nor penalises this company)."
                ),
                coverage=0.0,
            )

        # --- Conservative aggregation ---------------------------------------
        # Weighted mean of the safety dimensions, then pulled toward the single
        # worst dimension so one fatal flaw cannot be averaged away by benign
        # metrics (risk is about the tail). Blend: 65% weighted mean, 35% worst.
        scores = [s for s, _w, _n in dimension_scores]
        weights = [w for _s, w, _n in dimension_scores]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        mean_score = weighted_sum / total_weight if total_weight > 0 else (sum(scores) / len(scores))
        worst = min(scores)
        composite = 0.65 * mean_score + 0.35 * worst
        score = clamp(composite, 0.0, 100.0)

        # --- Coverage & confidence ------------------------------------------
        n_dims = len(dimension_scores)
        data_coverage = clamp(n_dims / float(self._MAX_DIMENSIONS), 0.0, 1.0)
        # Confidence rises with the number of independent dimensions actually
        # scored (never claims certainty from a single signal).
        confidence = clamp(0.25 + 0.5 * (n_dims / float(self._MAX_DIMENSIONS)), 0.0, 1.0)

        rationale = self._build_rationale(score, dimension_scores, flags)

        # De-duplicate flags while preserving order (defensive: a flag may be
        # appended by more than one branch in edge cases).
        seen_flags: List[str] = []
        for f in flags:
            if f not in seen_flags:
                seen_flags.append(f)

        return SubScore(
            category=self.category,
            score=score,
            confidence=confidence,
            weight=self.default_weight,
            rationale=rationale,
            evidence=evidence,
            flags=seen_flags,
            data_coverage=data_coverage,
        )

    # ------------------------------------------------------------------ #
    # Helpers (all pure)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cash_runway_months(latest: Optional[FundamentalsPeriod]) -> Optional[float]:
        """Months of cash runway from the latest period, or ``None`` if N/A.

        Runway only applies when the company is *burning* cash (free cash flow
        negative). Computed as ``cash_and_equivalents / monthly_burn`` where the
        monthly burn is ``|annual FCF| / 12``. A cash-generative company has no
        runway risk: we return a large sentinel (120 months) so the dimension
        scores as safe without fabricating a precise figure. Returns ``None``
        when cash or free cash flow is missing.
        """
        if latest is None:
            return None
        cash = latest.cash_and_equivalents
        fcf = latest.free_cash_flow
        if cash is None or fcf is None:
            return None
        if fcf >= 0:
            # Self-funding: no burn-down risk. Cap the runway proxy generously.
            return 120.0
        monthly_burn = abs(fcf) / 12.0
        if monthly_burn <= 0:
            return 120.0
        if cash <= 0:
            return 0.0
        return cash / monthly_burn

    @staticmethod
    def _annual_dilution_rate(fundamentals: Sequence[FundamentalsPeriod]) -> Optional[float]:
        """Annualised growth in diluted share count, or ``None`` if undeterminable.

        Compares the newest period's ``shares_diluted`` against the oldest
        available period's, annualising by the number of periods between them
        (each period treated as one step). A negative result means net buybacks
        (share count shrank), which the caller scores as *safe*. Returns ``None``
        when fewer than two periods carry a positive share count.
        """
        counts: List[float] = []
        for fp in fundamentals:
            if fp.shares_diluted is not None and fp.shares_diluted > 0:
                counts.append(fp.shares_diluted)
        if len(counts) < 2:
            return None
        # fundamentals are newest-first: counts[0] newest, counts[-1] oldest.
        newest = counts[0]
        oldest = counts[-1]
        steps = len(counts) - 1
        if oldest <= 0 or steps <= 0:
            return None
        total_growth = newest / oldest
        if total_growth <= 0:
            return None
        # Annualise per reported step (CAGR-style over the number of steps).
        return total_growth ** (1.0 / steps) - 1.0

    @staticmethod
    def _liquidity_score(adv: float, ctx: AnalysisContext) -> float:
        """Map average daily dollar volume to a 0–100 safety score.

        When the context supplies a universe distribution of average dollar
        volumes (``ctx.universe_stats["avg_dollar_volume"]``), the security is
        scored by its *percentile* within that distribution (relative liquidity,
        the right frame for micro-caps). Otherwise it degrades gracefully to an
        absolute log-scaled band: ``$50k/day`` -> risky, ``$5M/day`` -> safe.
        """
        dist = None
        if ctx is not None and ctx.universe_stats:
            dist = ctx.universe_stats.get("avg_dollar_volume")
        if dist:
            pct = percentile_rank(adv, dist)
            if pct is not None:
                return clamp(pct * 100.0, 0.0, 100.0)
        # Absolute fallback on a log10 scale ($50k -> 0, $5M -> 100).
        log_adv = math.log10(adv) if adv > 0 else 0.0
        score = scale_to_score(log_adv, math.log10(50_000.0), math.log10(5_000_000.0), higher_is_better=True)
        return clamp(score if score is not None else 50.0, 0.0, 100.0)

    def _distress_language_dimension(
        self, data: SecurityData
    ) -> Tuple[Optional[float], List[Evidence], List[str]]:
        """Score going-concern/litigation language risk from filings and news.

        Scans the combined title/summary of every filing and news item for the
        distress lexicon in :data:`_DISTRESS_PATTERNS`. The severity of each
        detection is weighted by the credibility of its source (a going-concern
        clause in an SEC filing weighs far more than the same word in a blog),
        using :func:`convexity.analysis.news_nlp.source_credibility`. The
        accumulated, source-weighted severity is mapped to a 0–100 *safety*
        score (more/severer distress language -> lower score).

        Returns ``(score, evidence, flags)``. ``score`` is ``None`` only when
        there are no filings *and* no news to scan (the dimension is then
        skipped, lowering coverage rather than implying safety).
        """
        evidence: List[Evidence] = []
        flags: List[str] = []

        # Build (text, source) pairs from filings and news. Filings are primary
        # records and carry their form type as the source for credibility.
        pairs: List[Tuple[Optional[str], Optional[str]]] = []
        sources: List[str] = []
        for f in data.filings:
            text = "  ".join(p for p in (f.title, f.summary, f.form_type) if p)
            src = f.form_type or "SEC filing"
            pairs.append((text, src))
            sources.append(src)
        for n in data.news:
            text = "  ".join(p for p in (n.title, n.summary) if p)
            pairs.append((text, n.source))
            sources.append(n.source)

        if not pairs:
            return None, evidence, flags

        detections = _scan_distress_language(pairs)

        if not detections:
            # We had documents to read and found no distress language: this is a
            # mildly *positive* (safe) signal, not a neutral one.
            evidence.append(
                Evidence(
                    label="Distress / litigation language",
                    value="none detected",
                    detail=(
                        f"Scanned {len(pairs)} filing/news item(s); no going-concern, "
                        "default, delisting, investigation or litigation wording found."
                    ),
                    source="filings+news",
                    direction="bullish",
                )
            )
            return 85.0, evidence, flags

        # Accumulate source-weighted severity. We weight each detection's
        # severity by the credibility of its source so primary filings dominate.
        # Use the maximum severity across detections to set the floor (one
        # going-concern clause is enough), plus a smaller additive term for
        # breadth (many warnings across many documents).
        max_weighted = 0.0
        breadth = 0.0
        worst_label = ""
        worst_match = ""
        worst_source = ""
        # Pair detections back to a source by re-scanning per text (kept simple
        # and deterministic): detections preserve text order, so iterate pairs.
        di = 0
        per_text_counts: List[int] = []
        for text, src in pairs:
            local = _scan_distress_language([(text, src)])
            per_text_counts.append(len(local))
            for label, severity, matched in local:
                cred = source_credibility(src)
                weighted = severity * cred
                breadth += weighted
                if weighted > max_weighted:
                    max_weighted = weighted
                    worst_label = label
                    worst_match = matched
                    worst_source = src or "unknown"
            di += len(local)

        # Map accumulated severity to a safety score. The single worst weighted
        # detection sets most of the penalty; breadth adds a little more.
        # max_weighted in ~[0, 1]; breadth can exceed 1 with many hits.
        penalty = clamp((max_weighted * 70.0) + (min(breadth, 3.0) / 3.0) * 30.0, 0.0, 100.0)
        score = clamp(100.0 - penalty, 0.0, 100.0)

        if max_weighted >= 0.6:
            flags.append("DISTRESS_LANGUAGE")
        # A high-credibility going-concern / bankruptcy / fraud hit gets its own flag.
        if max_weighted >= 0.85:
            flags.append("GOING_CONCERN_RISK")

        n_detections = sum(per_text_counts)
        evidence.append(
            Evidence(
                label="Distress / litigation language",
                value=f"{n_detections} detection(s)",
                detail=(
                    f"Most severe: {worst_label} (matched '{worst_match}') in {worst_source}. "
                    "Source-credibility-weighted; primary filings weigh most."
                ),
                source="filings+news",
                direction="bearish",
            )
        )
        return score, evidence, flags

    @staticmethod
    def _build_rationale(
        score: float,
        dimension_scores: Sequence[Tuple[float, float, str]],
        flags: Sequence[str],
    ) -> str:
        """Compose a short, honest, human rationale string.

        States the overall safety read (higher == safer), how many independent
        dimensions backed it, and names the most acute concern (lowest-scoring
        dimension) so the reader sees what is driving the number.
        """
        n = len(dimension_scores)
        if score >= 70:
            band = "low risk (safe profile)"
        elif score >= 55:
            band = "moderate-to-low risk"
        elif score >= 45:
            band = "moderate risk"
        elif score >= 30:
            band = "elevated risk"
        else:
            band = "high risk (fragile profile)"

        # Identify the most acute (lowest) dimension for transparency.
        worst_name = ""
        worst_val = 101.0
        for s, _w, name in dimension_scores:
            if s < worst_val:
                worst_val = s
                worst_name = name
        readable = {
            "equity": "shareholders' equity",
            "leverage_de": "leverage (debt/equity)",
            "interest_coverage": "interest coverage",
            "runway": "cash runway",
            "dilution": "share dilution",
            "liquidity": "trading liquidity",
            "volatility": "price volatility",
            "distress_language": "going-concern/litigation language",
        }.get(worst_name, worst_name or "n/a")

        flag_note = ""
        notable = [f for f in flags if f != "MISSING_DATA"]
        if notable:
            flag_note = f" Flags raised: {', '.join(notable)}."

        return (
            f"RISK scored {score:.0f}/100 (higher = safer): {band}, from {n} independent "
            f"risk check(s). The most acute concern is {readable}."
            f"{flag_note} Research signal only — not a prediction or advice; this score "
            "dampens the composite when fragility is evident."
        )


__all__ = ["RiskAnalyzer"]
