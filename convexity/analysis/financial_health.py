"""Financial-health analyzer — leverage, liquidity, coverage, runway, distress.

Part of Convexity, an evidence-driven equity **research and screening** tool. It
is **not** a predictor and **not** investment advice. This analyzer scores a
single category, :class:`~convexity.core.models.ScoreCategory.FINANCIAL_HEALTH`,
and contributes one transparent, auditable :class:`SubScore` to the composite
alongside many *other* independent categories. A high financial-health score is
never a recommendation; it is only the statement "the balance-sheet evidence we
could verify points to a solvent, well-funded company." Conviction still requires
*many independent signals agreeing*, which is the job of the ranking layer.

What "financial health" means here
----------------------------------
Solvency and funding durability — the question of whether a company can survive
and self-fund without being forced into dilutive financing, a covenant breach, or
distress. For thin micro-caps this is frequently the *binding* risk, so it is
scored from five deliberately independent balance-sheet/cash-flow lenses:

1. **Leverage** — debt/equity and net-debt/EBITDA. Less debt relative to equity
   and earnings is safer.
2. **Liquidity** — current and quick ratios. More short-term assets relative to
   short-term obligations is safer.
3. **Interest coverage** — EBIT (proxied) over interest expense, i.e. how many
   times operating earnings cover the interest bill.
4. **Cash runway** — months of cash at the current free-cash-flow burn rate;
   only a *negative* free cash flow consumes runway, a positive FCF is
   self-funding.
5. **Distress proxy** — an Altman-Z-style composite of working-capital,
   retained-earnings (proxied by equity), operating-profitability and
   leverage signals, mapped to a 0–100 distress-safety score.

Each lens is computed only from data that is actually present; a missing input
lowers ``data_coverage`` and ``confidence`` rather than being guessed. Where
``ctx.peer_stats`` / ``ctx.universe_stats`` are supplied, leverage and liquidity
are also read *relative* to comparable companies (a 1.5x current ratio means
different things in different industries), and the analyzer degrades gracefully
to absolute bands when no comparison set is available.

Honesty rules honoured
-----------------------
* Pure & deterministic: no I/O, no wall-clock, no randomness. Operates solely on
  the passed :class:`SecurityData`.
* Never fabricates: absent metrics stay absent; the score reflects only verified
  evidence and the ``confidence`` / ``data_coverage`` advertise how much that was.
* Every contributing number is recorded as an :class:`Evidence` item with an
  honest ``direction`` (a missing value is always neutral).
* Notable hazards (negative equity, going-concern-scale runway, heavy share-count
  dilution, an outright distress reading) are surfaced as ``flags`` so a reader
  can see *why* a score is low, not just *that* it is low.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Set, Tuple

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
    logistic_score,
    percentile_rank,
    scale_to_score,
)

# Provenance label recorded on every Evidence item this analyzer emits.
_SOURCE = "fundamentals"

# A free-cash-flow burn so small it is treated as effectively break-even; avoids
# reporting an absurd runway (or dividing by ~0) when FCF hovers around zero.
_NEGLIGIBLE_BURN = 1_000.0

# Months of runway at/under which we treat funding as a going-concern-scale risk.
_RUNWAY_GOING_CONCERN_MONTHS = 12.0

# Year-over-year diluted-share growth above which we flag meaningful dilution.
_DILUTION_FLAG_THRESHOLD = 0.10  # +10% share count YoY.

# Altman-Z (private-firm Z'') interpretive bands, used only to *label* the
# distress reading in evidence/flags — the numeric score itself is mapped
# continuously via a logistic so there is no cliff at a threshold.
_Z_DISTRESS = 1.1   # below ~1.1 is the classic "distress" zone.
_Z_SAFE = 2.6       # above ~2.6 is the classic "safe" zone.


@register_analyzer
class FinancialHealthAnalyzer(Analyzer):
    """Scores balance-sheet solvency & funding durability into a 0–100 SubScore.

    Higher means a *healthier* (safer, better-funded) balance sheet. The score is
    a confidence-weighted blend of up to five independent sub-signals (leverage,
    liquidity, interest coverage, cash runway, an Altman-Z-style distress proxy),
    each scored relative to peers/universe when such context is available and to
    sensible absolute bands otherwise. Sub-signals whose inputs are missing are
    simply dropped from the blend (and lower coverage/confidence) rather than
    guessed; if nothing scorable remains, a neutral low-confidence sub-score is
    returned via :meth:`neutral_subscore`.
    """

    category: ScoreCategory = ScoreCategory.FINANCIAL_HEALTH
    default_weight: float = 0.13
    # Capability/data-field names this analyzer needs for full coverage. The
    # aggregator/pipeline can use these for routing; coverage is computed from how
    # many of the underlying metrics were actually present (see ``analyze``).
    requires: Set[str] = {"fundamentals"}

    # Relative blend weights of the five lenses (normalised over whichever lenses
    # actually produced a score). They sum to 1.0 when all five are present.
    _LENS_WEIGHTS: Dict[str, float] = {
        "leverage": 0.26,
        "liquidity": 0.22,
        "coverage": 0.20,
        "runway": 0.14,
        "distress": 0.18,
    }

    # ------------------------------------------------------------------ public
    def analyze(self, data: SecurityData, ctx: AnalysisContext) -> SubScore:
        """Return the FINANCIAL_HEALTH :class:`SubScore` for ``data``.

        Pure: derives everything from the most recent
        :class:`~convexity.core.models.FundamentalsPeriod` (and, for runway and
        dilution, the prior period), plus any peer/universe distributions in
        ``ctx``. Performs no I/O.
        """
        fp = data.latest_fundamentals
        if fp is None:
            return self.neutral_subscore(
                "No fundamentals reported; balance-sheet health cannot be assessed.",
                coverage=0.0,
            )

        prior = data.fundamentals[1] if len(data.fundamentals) > 1 else None
        peer_stats = ctx.peer_stats or {}
        universe_stats = ctx.universe_stats or {}

        evidence: List[Evidence] = []
        flags: List[str] = []

        # Each lens returns (score_or_None, coverage_fraction_for_lens). The lens
        # also appends its own Evidence/flags. Coverage tracks how much of the
        # lens's required input was present (used to weight + report honesty).
        lens_scores: Dict[str, Optional[float]] = {}
        lens_coverage: Dict[str, float] = {}

        lens_scores["leverage"], lens_coverage["leverage"] = self._score_leverage(
            fp, peer_stats, universe_stats, evidence, flags
        )
        lens_scores["liquidity"], lens_coverage["liquidity"] = self._score_liquidity(
            fp, peer_stats, universe_stats, evidence, flags
        )
        lens_scores["coverage"], lens_coverage["coverage"] = self._score_coverage(
            fp, evidence, flags
        )
        lens_scores["runway"], lens_coverage["runway"] = self._score_runway(
            fp, evidence, flags
        )
        lens_scores["distress"], lens_coverage["distress"] = self._score_distress(
            fp, evidence, flags
        )

        # Surface dilution as a standalone evidence/flag signal (it does not score
        # a lens directly, but it is a key funding-durability tell for micro-caps).
        self._note_dilution(fp, prior, evidence, flags)

        # --- Blend the lenses that produced a score -------------------------
        present = [(name, s) for name, s in lens_scores.items() if s is not None]
        if not present:
            return self.neutral_subscore(
                "Fundamentals were reported but contained no usable leverage, "
                "liquidity, coverage, runway, or distress inputs.",
                coverage=0.0,
                extra_flags=flags or None,
            )

        weighted_num = 0.0
        weighted_den = 0.0
        for name, s in present:
            w = self._LENS_WEIGHTS[name]
            weighted_num += s * w
            weighted_den += w
        composite = clamp(weighted_num / weighted_den, 0.0, 100.0)

        # --- Coverage & confidence ------------------------------------------
        # Coverage: average of the five lenses' coverage (missing lenses count 0),
        # so a company scored from only one lens advertises low coverage honestly.
        data_coverage = clamp(
            sum(lens_coverage.values()) / float(len(self._LENS_WEIGHTS)),
            0.0,
            1.0,
        )

        # Confidence blends data_coverage with breadth (how many of the five
        # independent lenses agreed enough to be computed). One lens is thin
        # evidence; four or five mutually corroborating lenses is strong.
        breadth = len(present) / float(len(self._LENS_WEIGHTS))
        confidence = clamp(0.30 + 0.45 * data_coverage + 0.25 * breadth, 0.0, 1.0)
        # A hard distress reading or negative equity should not masquerade as a
        # confident *high* score, but it is itself high-signal; keep confidence
        # honest by not letting a single thin lens claim certainty.
        if breadth < 0.4:
            confidence = min(confidence, 0.5)

        rationale = self._build_rationale(composite, present, flags)

        return SubScore(
            category=self.category,
            score=composite,
            confidence=confidence,
            weight=self.default_weight,
            rationale=rationale,
            evidence=evidence,
            flags=sorted(set(flags)),
            data_coverage=data_coverage,
        )

    # --------------------------------------------------------------- leverage
    def _score_leverage(
        self,
        fp: FundamentalsPeriod,
        peer_stats: Dict[str, Any],
        universe_stats: Dict[str, Any],
        evidence: List[Evidence],
        flags: List[str],
    ) -> Tuple[Optional[float], float]:
        """Score balance-sheet leverage from debt/equity and net-debt/EBITDA.

        Returns ``(score, coverage)``. Lower leverage is safer. Two independent
        gauges are averaged when both are available; either alone still scores.
        Coverage is the fraction of the two gauges that were computable.
        """
        sub_scores: List[float] = []
        present = 0
        total = 2

        # --- Debt / equity --------------------------------------------------
        d_to_e = self._debt_to_equity(fp)
        if d_to_e is not None:
            present += 1
            # Negative equity makes the ratio meaningless and is itself a red
            # flag: book value is gone. Score it at the floor and flag it.
            if fp.total_equity is not None and fp.total_equity <= 0:
                flags.append("NEGATIVE_EQUITY")
                evidence.append(
                    Evidence(
                        label="Shareholders' equity",
                        value=_fmt(fp.total_equity),
                        detail="Negative book equity — liabilities exceed assets.",
                        source=_SOURCE,
                        direction="bearish",
                    )
                )
                lev_score = 0.0
            else:
                # Relative to peers if available, else absolute band.
                rel = _relative_score(
                    d_to_e,
                    peer_stats.get("debt_to_equity"),
                    universe_stats.get("debt_to_equity"),
                    higher_is_better=False,
                )
                if rel is None:
                    # Absolute band: 0x debt -> 100, ~3x -> 0 (lower is safer).
                    rel = scale_to_score(d_to_e, 0.0, 3.0, higher_is_better=False)
                lev_score = rel if rel is not None else 50.0
                direction = _dir(lev_score)
                evidence.append(
                    Evidence.from_number(
                        "Debt / equity",
                        d_to_e,
                        source=_SOURCE,
                        direction=direction,
                        unit="x",
                        detail=_relative_detail(
                            "debt_to_equity", peer_stats, universe_stats
                        ),
                    )
                )
            sub_scores.append(lev_score)

        # --- Net debt / EBITDA ----------------------------------------------
        nd_ebitda = self._net_debt_to_ebitda(fp)
        if nd_ebitda is not None:
            present += 1
            # Net cash (negative net debt) is maximally safe on this gauge.
            if nd_ebitda <= 0:
                nd_score = 100.0
                evidence.append(
                    Evidence.from_number(
                        "Net debt / EBITDA",
                        nd_ebitda,
                        source=_SOURCE,
                        direction="bullish",
                        unit="x",
                        detail="Net cash position (cash exceeds total debt).",
                    )
                )
            else:
                rel = _relative_score(
                    nd_ebitda,
                    peer_stats.get("net_debt_to_ebitda"),
                    universe_stats.get("net_debt_to_ebitda"),
                    higher_is_better=False,
                )
                if rel is None:
                    # Absolute band: 0x -> 100, 5x -> 0 (a common covenant zone).
                    rel = scale_to_score(nd_ebitda, 0.0, 5.0, higher_is_better=False)
                nd_score = rel if rel is not None else 50.0
                if nd_ebitda >= 4.0:
                    flags.append("HIGH_LEVERAGE")
                evidence.append(
                    Evidence.from_number(
                        "Net debt / EBITDA",
                        nd_ebitda,
                        source=_SOURCE,
                        direction=_dir(nd_score),
                        unit="x",
                        detail=_relative_detail(
                            "net_debt_to_ebitda", peer_stats, universe_stats
                        ),
                    )
                )
            sub_scores.append(nd_score)

        if not sub_scores:
            return None, 0.0
        return sum(sub_scores) / len(sub_scores), present / float(total)

    # -------------------------------------------------------------- liquidity
    def _score_liquidity(
        self,
        fp: FundamentalsPeriod,
        peer_stats: Dict[str, Any],
        universe_stats: Dict[str, Any],
        evidence: List[Evidence],
        flags: List[str],
    ) -> Tuple[Optional[float], float]:
        """Score short-term liquidity from current and quick ratios.

        Returns ``(score, coverage)``. Higher ratios are safer (more current
        assets per dollar of current liabilities). The quick ratio is the
        stricter gauge (it excludes inventory) and is weighted slightly more.
        """
        parts: List[Tuple[float, float]] = []  # (score, weight)
        present = 0
        total = 2

        if fp.current_ratio is not None:
            present += 1
            rel = _relative_score(
                fp.current_ratio,
                peer_stats.get("current_ratio"),
                universe_stats.get("current_ratio"),
                higher_is_better=True,
            )
            if rel is None:
                # Absolute band: <=0.5x -> 0, >=2.5x -> 100 (1.0x roughly neutral).
                rel = scale_to_score(fp.current_ratio, 0.5, 2.5, higher_is_better=True)
            score = rel if rel is not None else 50.0
            if fp.current_ratio < 1.0:
                flags.append("LIQUIDITY_BELOW_1X")
            evidence.append(
                Evidence.from_number(
                    "Current ratio",
                    fp.current_ratio,
                    source=_SOURCE,
                    direction=_dir(score),
                    unit="x",
                    detail=_relative_detail("current_ratio", peer_stats, universe_stats),
                )
            )
            parts.append((score, 0.45))

        if fp.quick_ratio is not None:
            present += 1
            rel = _relative_score(
                fp.quick_ratio,
                peer_stats.get("quick_ratio"),
                universe_stats.get("quick_ratio"),
                higher_is_better=True,
            )
            if rel is None:
                rel = scale_to_score(fp.quick_ratio, 0.3, 2.0, higher_is_better=True)
            score = rel if rel is not None else 50.0
            evidence.append(
                Evidence.from_number(
                    "Quick ratio",
                    fp.quick_ratio,
                    source=_SOURCE,
                    direction=_dir(score),
                    unit="x",
                    detail=_relative_detail("quick_ratio", peer_stats, universe_stats),
                )
            )
            parts.append((score, 0.55))

        if not parts:
            return None, 0.0
        num = sum(s * w for s, w in parts)
        den = sum(w for _, w in parts)
        return num / den, present / float(total)

    # --------------------------------------------------------------- coverage
    def _score_coverage(
        self,
        fp: FundamentalsPeriod,
        evidence: List[Evidence],
        flags: List[str],
    ) -> Tuple[Optional[float], float]:
        """Score interest coverage — operating earnings over interest expense.

        Returns ``(score, coverage)``. Uses the reported ``interest_coverage``
        when present. A logistic centred at ~3x (a common minimum-comfort level)
        rewards rising coverage while saturating, so a 50x coverer is not scored
        wildly higher than a comfortable 12x one.
        """
        ic = fp.interest_coverage
        if ic is None:
            return None, 0.0

        # If a company has debt-service obligations it cannot cover, that is a
        # primary solvency hazard; flag sub-1x coverage explicitly.
        if ic < 1.0:
            flags.append("INTEREST_UNCOVERED")

        # Logistic centred at 3x; steepness chosen so 1x~=12, 3x=50, 6x~=82.
        score = logistic_score(ic, midpoint=3.0, steepness=0.6)
        score = score if score is not None else 50.0
        evidence.append(
            Evidence.from_number(
                "Interest coverage (EBIT / interest)",
                ic,
                source=_SOURCE,
                direction=_dir(score),
                unit="x",
                detail="Times operating earnings cover the interest bill.",
            )
        )
        return score, 1.0

    # ----------------------------------------------------------------- runway
    def _score_runway(
        self,
        fp: FundamentalsPeriod,
        evidence: List[Evidence],
        flags: List[str],
    ) -> Tuple[Optional[float], float]:
        """Score cash runway — months of cash at the current FCF burn rate.

        Returns ``(score, coverage)``. A *positive* free cash flow means the
        company is self-funding and scores at the top of this lens. A *negative*
        FCF consumes cash; runway = cash / monthly-burn, scored so that <=12
        months is a going-concern-scale hazard and >=36 months is comfortable.
        """
        cash = fp.cash_and_equivalents
        fcf = fp.free_cash_flow
        if cash is None or fcf is None:
            return None, 0.0

        if fcf >= -_NEGLIGIBLE_BURN:
            # Self-funding (or essentially break-even): runway is not a constraint.
            evidence.append(
                Evidence.from_number(
                    "Free cash flow",
                    fcf,
                    source=_SOURCE,
                    direction="bullish" if fcf > 0 else "neutral",
                    detail="Positive / break-even FCF — operations self-fund; cash is not being burned.",
                )
            )
            return 100.0, 1.0

        monthly_burn = abs(fcf) / 12.0
        runway_months = cash / monthly_burn if monthly_burn > 0 else float("inf")

        if runway_months <= _RUNWAY_GOING_CONCERN_MONTHS:
            flags.append("CASH_RUNWAY_UNDER_12M")
        # Band: 6 months -> 0, 36 months -> 100 (linear between).
        score = scale_to_score(runway_months, 6.0, 36.0, higher_is_better=True)
        score = score if score is not None else 50.0
        evidence.append(
            Evidence.from_number(
                "Cash runway",
                runway_months,
                source=_SOURCE,
                direction=_dir(score),
                unit=" months",
                precision=1,
                detail=(
                    f"Cash {_fmt(cash)} at a {_fmt(abs(fcf))}/yr FCF burn; "
                    "months until cash is exhausted absent new financing."
                ),
            )
        )
        return score, 1.0

    # --------------------------------------------------------------- distress
    def _score_distress(
        self,
        fp: FundamentalsPeriod,
        evidence: List[Evidence],
        flags: List[str],
    ) -> Tuple[Optional[float], float]:
        """Altman-Z-style distress proxy mapped to a 0–100 *safety* score.

        Returns ``(score, coverage)``. Uses the Altman Z''-score form intended for
        non-manufacturers / private firms, which avoids needing a market value or
        sales-driven term that thin micro-caps often lack:

            Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

        where (all relative to total assets unless noted):

        * ``X1`` = working capital / total assets (proxied via current ratio when
          working-capital line items are absent),
        * ``X2`` = retained earnings / total assets (proxied by total equity /
          total assets — book cushion built up over time),
        * ``X3`` = operating income / total assets (operating profitability),
        * ``X4`` = total equity / total debt (book leverage cushion).

        Coverage scales with how many of the four terms were computable. The raw
        Z'' is mapped to 0–100 with a logistic so distress (~<1.1) lands low and
        the safe zone (~>2.6) lands high, with no hard cliff.
        """
        ta = fp.total_assets
        if ta is None or ta == 0:
            return None, 0.0

        terms_present = 0
        terms_total = 4
        z = 0.0

        # X1 — working-capital intensity. Prefer a true working-capital proxy;
        # fall back to (current_ratio - 1) clipped, a monotone stand-in.
        x1: Optional[float] = None
        if fp.current_ratio is not None:
            # (current_ratio - 1) is positive when current assets exceed current
            # liabilities; clip to a sane band so a wild ratio cannot dominate.
            x1 = max(-1.0, min(1.0, fp.current_ratio - 1.0))
        if x1 is not None:
            z += 6.56 * x1
            terms_present += 1

        # X2 — retained-earnings cushion, proxied by equity / assets.
        if fp.total_equity is not None:
            x2 = fp.total_equity / ta
            z += 3.26 * x2
            terms_present += 1

        # X3 — operating profitability on assets.
        if fp.operating_income is not None:
            x3 = fp.operating_income / ta
            z += 6.72 * x3
            terms_present += 1

        # X4 — equity / debt leverage cushion.
        if fp.total_equity is not None and fp.total_debt is not None:
            if fp.total_debt > 0:
                x4 = fp.total_equity / fp.total_debt
            else:
                x4 = 5.0  # no debt: cap the cushion term generously.
            x4 = max(-5.0, min(5.0, x4))
            z += 1.05 * x4
            terms_present += 1

        if terms_present == 0:
            return None, 0.0

        # Map Z'' to a 0–100 safety score. Centre the logistic between the
        # classic distress (1.1) and safe (2.6) zones at ~1.85.
        score = logistic_score(z, midpoint=1.85, steepness=1.1)
        score = score if score is not None else 50.0

        if z < _Z_DISTRESS:
            flags.append("ALTMAN_Z_DISTRESS")
            zone = "distress zone"
            direction = "bearish"
        elif z > _Z_SAFE:
            zone = "safe zone"
            direction = "bullish"
        else:
            zone = "grey zone"
            direction = _dir(score)

        evidence.append(
            Evidence.from_number(
                "Altman Z''-score (distress proxy)",
                z,
                source=_SOURCE,
                direction=direction,
                precision=2,
                detail=(
                    f"{zone}; computed from {terms_present}/{terms_total} terms "
                    "(working capital, equity cushion, operating profitability, leverage)."
                ),
            )
        )
        return score, terms_present / float(terms_total)

    # --------------------------------------------------------------- dilution
    def _note_dilution(
        self,
        fp: FundamentalsPeriod,
        prior: Optional[FundamentalsPeriod],
        evidence: List[Evidence],
        flags: List[str],
    ) -> None:
        """Record share-count dilution as evidence and flag heavy dilution.

        Compares diluted shares this period vs the prior period. Persistent equity
        issuance is the classic micro-cap funding tell: it keeps the lights on but
        erodes per-share value, so material dilution is surfaced explicitly. This
        does not score a lens; it informs the reader and the flags.
        """
        cur = fp.shares_diluted
        prev = prior.shares_diluted if prior is not None else None
        if cur is None or prev is None or prev <= 0:
            return
        growth = (cur - prev) / prev
        if growth >= _DILUTION_FLAG_THRESHOLD:
            flags.append("SHARE_DILUTION")
            direction = "bearish"
        elif growth <= -0.02:
            direction = "bullish"  # buyback / share-count reduction.
        else:
            direction = "neutral"
        evidence.append(
            Evidence.from_number(
                "Diluted shares — YoY change",
                growth * 100.0,
                source=_SOURCE,
                direction=direction,
                unit="%",
                precision=1,
                detail=(
                    f"Diluted share count {_fmt(prev)} -> {_fmt(cur)}; "
                    "rising counts dilute per-share value and often signal external funding needs."
                ),
            )
        )

    # ----------------------------------------------------------------- ratios
    @staticmethod
    def _debt_to_equity(fp: FundamentalsPeriod) -> Optional[float]:
        """Return debt/equity, preferring the reported field then computing it."""
        if fp.debt_to_equity is not None:
            return fp.debt_to_equity
        if (
            fp.total_debt is not None
            and fp.total_equity is not None
            and fp.total_equity != 0
        ):
            return fp.total_debt / fp.total_equity
        return None

    @staticmethod
    def _net_debt_to_ebitda(fp: FundamentalsPeriod) -> Optional[float]:
        """Return (total debt - cash) / EBITDA, or ``None`` if uncomputable.

        Requires total debt and a positive EBITDA (a non-positive EBITDA makes the
        multiple meaningless; that condition is captured by other lenses instead).
        Cash is treated as 0 only if explicitly absent is *not* assumed — if cash
        is ``None`` we cannot net it, so we return ``None`` to avoid overstating
        leverage.
        """
        if fp.total_debt is None or fp.ebitda is None:
            return None
        if fp.ebitda <= 0:
            return None
        if fp.cash_and_equivalents is None:
            net_debt = fp.total_debt
        else:
            net_debt = fp.total_debt - fp.cash_and_equivalents
        return net_debt / fp.ebitda

    # -------------------------------------------------------------- narrative
    @staticmethod
    def _build_rationale(
        composite: float,
        present: Sequence[Tuple[str, float]],
        flags: Sequence[str],
    ) -> str:
        """Compose a short, honest, human rationale string for the sub-score."""
        if composite >= 70:
            verdict = "a solid, well-funded balance sheet"
        elif composite >= 50:
            verdict = "an adequate but unremarkable balance sheet"
        elif composite >= 30:
            verdict = "a stretched balance sheet with funding pressure"
        else:
            verdict = "a fragile balance sheet with material solvency risk"

        lenses = ", ".join(name for name, _ in present)
        base = (
            f"Financial-health score {composite:.0f}/100 indicates {verdict}, "
            f"blended from the {len(present)} computable lens(es): {lenses}."
        )
        hazard_flags = [
            f
            for f in flags
            if f
            in {
                "NEGATIVE_EQUITY",
                "CASH_RUNWAY_UNDER_12M",
                "INTEREST_UNCOVERED",
                "ALTMAN_Z_DISTRESS",
                "HIGH_LEVERAGE",
            }
        ]
        if hazard_flags:
            base += " Hazards flagged: " + ", ".join(sorted(set(hazard_flags))) + "."
        return base


# ---------------------------------------------------------------------------
# Module-level pure helpers (no I/O, no state)
# ---------------------------------------------------------------------------


def _dir(score: Optional[float]) -> str:
    """Map a 0–100 sub-score to an honest evidence direction."""
    if score is None:
        return "neutral"
    if score >= 60.0:
        return "bullish"
    if score <= 40.0:
        return "bearish"
    return "neutral"


def _fmt(value: Optional[float]) -> str:
    """Compactly format a (possibly large) monetary/scalar value for detail text."""
    if value is None:
        return "n/a"
    a = abs(value)
    if a >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f}B"
    if a >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if a >= 1_000:
        return f"{value / 1_000:,.1f}K"
    return f"{value:,.2f}"


def _distribution(stat: Any) -> Optional[List[float]]:
    """Coerce a peer/universe stat entry into a list of floats, or ``None``.

    Accepts either a bare sequence of numbers (``[6.1, 8.0, 12.4]``) or a mapping
    that carries a distribution under a ``"values"``/``"distribution"`` key. Any
    other shape (e.g. a pre-summarised ``{"median": ...}`` with no raw values)
    yields ``None`` so the caller falls back to absolute bands.
    """
    if stat is None:
        return None
    if isinstance(stat, dict):
        for key in ("values", "distribution", "samples"):
            if key in stat:
                stat = stat[key]
                break
        else:
            return None
    if isinstance(stat, (list, tuple)):
        nums = [float(v) for v in stat if isinstance(v, (int, float))]
        return nums or None
    return None


def _relative_score(
    value: Optional[float],
    peer_stat: Any,
    universe_stat: Any,
    *,
    higher_is_better: bool,
) -> Optional[float]:
    """Percentile-rank ``value`` against peers (preferred) then the universe.

    Returns a 0–100 score, or ``None`` when no usable distribution is available
    (the caller then degrades to absolute bands). When ``higher_is_better`` is
    ``False`` the percentile is inverted so that being *low* in the distribution
    (e.g. low leverage) scores *high* (safer).
    """
    if value is None:
        return None
    dist = _distribution(peer_stat)
    if dist is None:
        dist = _distribution(universe_stat)
    if dist is None:
        return None
    pct = percentile_rank(value, dist)
    if pct is None:
        return None
    if not higher_is_better:
        pct = 1.0 - pct
    return clamp(pct * 100.0, 0.0, 100.0)


def _relative_detail(
    metric: str,
    peer_stats: Dict[str, Any],
    universe_stats: Dict[str, Any],
) -> str:
    """Return a short note stating which comparison set (if any) was used."""
    if _distribution(peer_stats.get(metric)) is not None:
        return "Scored relative to peer distribution."
    if _distribution(universe_stats.get(metric)) is not None:
        return "Scored relative to the screened-universe distribution."
    return "Scored against absolute bands (no peer/universe distribution available)."


__all__ = ["FinancialHealthAnalyzer"]
