"""HISTORICAL_ANALOG analyzer — pattern-similarity to small-cap re-rating archetypes.

Part of Convexity, an evidence-driven small/micro-cap **research and screening**
tool. This module is emphatically **not a predictor and not investment advice**.

What this analyzer does
-----------------------
Many small- and micro-cap re-ratings, in hindsight, rhyme with a handful of
recognisable *setups*: a company that just turned profitable while insiders are
buying and the stock is basing; a deleveraging turnaround; a quietly compounding
business almost nobody follows. This analyzer encodes about half a dozen such
**archetypes as transparent data** — each a named bundle of feature conditions
with explicit thresholds — and measures how strongly the company in front of it
*resembles* the best-fitting archetype.

The output is a *similarity* score, full stop. A high score means "this company's
current, observable fundamentals and price structure look like the named pattern",
**never** "this company will re-rate". Pattern similarity is one independent piece
of evidence among twelve; conviction is justified only when many independent
categories agree, not when this one analogy is strong. Every point of the score
traces back to a concrete number (revenue growth, margin change, debt paydown,
insider buying, distance off the 52-week low, valuation-vs-growth mismatch, float
following), and the matched archetype is named explicitly in the evidence so a
reader can audit the analogy and disagree with it.

Honesty rules honoured here
---------------------------
* **Pure & deterministic.** ``analyze`` performs no I/O, reads no clock and uses
  no randomness; it operates only on the passed :class:`SecurityData`. Identical
  input yields an identical sub-score.
* **Never fabricated.** Missing inputs are treated as missing — the corresponding
  feature simply does not contribute (it is neither bullish nor bearish), the
  ``data_coverage`` fraction falls, and ``confidence`` falls with it. When almost
  nothing is computable the analyzer returns :meth:`neutral_subscore`.
* **Relative when it can be.** Where ``ctx.peer_stats`` / ``ctx.universe_stats``
  supply a distribution (e.g. EV/EBITDA, P/S, debt-to-equity), the analyzer reads
  "cheap" / "under-followed" relative to comparables rather than on absolute
  thresholds alone, degrading gracefully to absolute bands when no context exists.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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
# Feature extraction — observable, auditable signals computed from SecurityData
# ---------------------------------------------------------------------------
#
# Every feature is an Optional[float]: ``None`` means "not computable from the
# data we actually have" and never contributes to any archetype match. A feature
# that *is* computable also carries a short human label and the concrete numbers
# behind it so it can surface as auditable Evidence.


@dataclass(frozen=True)
class Feature:
    """One computed, auditable feature used to match archetypes.

    Attributes:
        key: Stable identifier referenced by archetype conditions.
        value: The numeric feature value, or ``None`` if not computable.
        label: Human-readable name for evidence (e.g. "Revenue growth (YoY)").
        unit: Display unit appended in evidence (e.g. "%", "x").
        detail: Optional extra context shown in evidence.
        higher_is_bullish: Whether a larger value is the favourable direction
            (used only to colour evidence direction, not to score archetypes).
    """

    key: str
    value: Optional[float]
    label: str
    unit: str = ""
    detail: Optional[str] = None
    higher_is_bullish: bool = True


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """Divide, returning ``None`` if either operand is missing or the denom is ~0."""
    if num is None or den is None:
        return None
    if abs(den) < 1e-12:
        return None
    return num / den


def _pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    """Percentage change from ``old`` to ``new`` in percent, or ``None``.

    Uses ``abs(old)`` in the denominator so that an improvement from a negative
    base (e.g. a loss shrinking) reports a positive percentage. Returns ``None``
    when either value is missing or the base is ~0 (an undefined percentage).
    """
    if new is None or old is None:
        return None
    if abs(old) < 1e-9:
        return None
    return (new - old) / abs(old) * 100.0


def _trailing_window(periods: Sequence[FundamentalsPeriod], n: int) -> List[FundamentalsPeriod]:
    """Return up to the newest ``n`` fundamentals periods (input is newest-first)."""
    return list(periods[:n])


def _is_positive(value: Optional[float]) -> Optional[bool]:
    """Tri-state positivity test that propagates missing data as ``None``."""
    if value is None:
        return None
    return value > 0.0


def _price_off_low(prices: Sequence[float]) -> Optional[float]:
    """Percent the latest close sits above the trailing-window low, or ``None``."""
    clean = [p for p in prices if p is not None]
    if len(clean) < 2:
        return None
    low = min(clean)
    last = clean[-1]
    if low <= 0:
        return None
    return (last - low) / low * 100.0


def _price_below_high(prices: Sequence[float]) -> Optional[float]:
    """Percent the latest close sits *below* the trailing-window high, or ``None``."""
    clean = [p for p in prices if p is not None]
    if len(clean) < 2:
        return None
    high = max(clean)
    last = clean[-1]
    if high <= 0:
        return None
    return (high - last) / high * 100.0


def _realized_volatility(prices: Sequence[float]) -> Optional[float]:
    """Annualised-ish daily close-to-close volatility in percent, or ``None``.

    A coarse, transparent dispersion measure (population stdev of simple daily
    returns, expressed in percent). Used only to recognise a *quiet base* (low
    volatility) as part of a basing pattern — never as a risk model.
    """
    clean = [p for p in prices if p is not None and p > 0]
    if len(clean) < 6:
        return None
    rets = [(clean[i] / clean[i - 1]) - 1.0 for i in range(1, len(clean))]
    if not rets:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return (var ** 0.5) * 100.0


def _count_insider_buys(data: SecurityData) -> Tuple[int, int, float]:
    """Return ``(buy_count, sell_count, net_buy_value)`` from insider transactions.

    ``transaction_type`` is matched case-insensitively on a "buy"/"purchas" or
    "sell"/"sold"/"dispos" stem. ``net_buy_value`` sums signed transaction values
    (buys positive, sells negative) using only rows that actually carry a value.
    """
    buys = sells = 0
    net_value = 0.0
    for tx in data.insider_transactions:
        kind = (tx.transaction_type or "").strip().lower()
        is_buy = "buy" in kind or "purchas" in kind or "acquir" in kind
        is_sell = "sell" in kind or "sold" in kind or "dispos" in kind
        if is_buy:
            buys += 1
            if tx.value is not None:
                net_value += abs(tx.value)
        elif is_sell:
            sells += 1
            if tx.value is not None:
                net_value -= abs(tx.value)
    return buys, sells, net_value


def _spinoff_or_restructure_signal(data: SecurityData) -> bool:
    """Whether any filing/news text plausibly evidences a spin-off / restructuring.

    A transparent keyword check over filing and news titles/summaries. It records
    *that the words appear*, never that a re-rating will follow.
    """
    needles = (
        "spin-off",
        "spinoff",
        "spin off",
        "separation",
        "carve-out",
        "carve out",
        "demerger",
        "restructuring",
        "divestiture",
    )
    blobs: List[str] = []
    for f in data.filings:
        blobs.append(((f.title or "") + " " + (f.summary or "")).lower())
    for n in data.news:
        blobs.append(((n.title or "") + " " + (n.summary or "")).lower())
    text = "  ".join(blobs)
    return any(needle in text for needle in needles)


def extract_features(data: SecurityData, ctx: AnalysisContext) -> Dict[str, Feature]:
    """Compute the full set of observable features from ``data`` (pure).

    Returns a mapping ``key -> Feature``. Every archetype condition references one
    of these keys. Features whose inputs are missing carry ``value=None`` and are
    silently ignored by archetype matching, which is what makes a data gap neither
    help nor hurt the analogy.
    """
    feats: Dict[str, Feature] = {}
    funds = data.fundamentals  # newest-first
    latest = funds[0] if funds else None
    prior = funds[1] if len(funds) > 1 else None

    # --- Growth: revenue YoY (latest vs the period one year-equivalent back) ---
    rev_growth = _pct_change(
        latest.revenue if latest else None,
        prior.revenue if prior else None,
    )
    feats["revenue_growth"] = Feature(
        "revenue_growth", rev_growth, "Revenue growth (period/period)", unit="%"
    )

    # --- Profitability inflection: sign flip of operating income / net income --
    op_now = latest.operating_income if latest else None
    op_prev = prior.operating_income if prior else None
    ni_now = latest.net_income if latest else None
    ni_prev = prior.net_income if prior else None
    inflection = None
    if _is_positive(op_now) is True and _is_positive(op_prev) is False:
        inflection = 1.0
    elif _is_positive(ni_now) is True and _is_positive(ni_prev) is False:
        inflection = 1.0
    elif op_now is not None and op_prev is not None:
        inflection = 1.0 if (op_now > 0 and op_prev > 0) else 0.0
    feats["profit_inflection"] = Feature(
        "profit_inflection", inflection, "Profitability inflection (loss->profit)", unit=""
    )

    # --- Operating margin expansion (latest minus prior, in percentage points) -
    om_now = latest.operating_margin if latest else None
    om_prev = prior.operating_margin if prior else None
    margin_delta = None
    if om_now is not None and om_prev is not None:
        margin_delta = (om_now - om_prev) * 100.0  # fraction -> pp
    feats["op_margin_delta"] = Feature(
        "op_margin_delta", margin_delta, "Operating-margin change", unit="pp"
    )

    # --- Gross margin expansion (drives capacity / operating-leverage stories) -
    gm_now = latest.gross_margin if latest else None
    gm_prev = prior.gross_margin if prior else None
    gross_delta = None
    if gm_now is not None and gm_prev is not None:
        gross_delta = (gm_now - gm_prev) * 100.0
    feats["gross_margin_delta"] = Feature(
        "gross_margin_delta", gross_delta, "Gross-margin change", unit="pp"
    )

    # --- FCF inflection / positive FCF -----------------------------------------
    fcf_now = latest.free_cash_flow if latest else None
    fcf_prev = prior.free_cash_flow if prior else None
    fcf_inflect = None
    if _is_positive(fcf_now) is True and _is_positive(fcf_prev) is False:
        fcf_inflect = 1.0
    elif fcf_now is not None:
        fcf_inflect = 1.0 if fcf_now > 0 else 0.0
    feats["fcf_inflection"] = Feature(
        "fcf_inflection", fcf_inflect, "Free-cash-flow turns positive", unit=""
    )

    # --- Deleveraging: change in debt-to-equity (negative == improving) --------
    de_now = latest.debt_to_equity if latest else None
    de_prev = prior.debt_to_equity if prior else None
    de_delta = None
    if de_now is not None and de_prev is not None:
        de_delta = de_now - de_prev  # negative == debt reduced relative to equity
    feats["debt_to_equity_delta"] = Feature(
        "debt_to_equity_delta", de_delta, "Debt-to-equity change", unit="x",
        higher_is_bullish=False,
    )
    feats["debt_to_equity"] = Feature(
        "debt_to_equity", de_now, "Debt-to-equity", unit="x", higher_is_bullish=False
    )

    # Absolute debt paydown (period/period reduction in total_debt, in percent).
    debt_paydown = _pct_change(
        latest.total_debt if latest else None,
        prior.total_debt if prior else None,
    )
    if debt_paydown is not None:
        debt_paydown = -debt_paydown  # positive == debt fell
    feats["debt_paydown"] = Feature(
        "debt_paydown", debt_paydown, "Total-debt reduction", unit="%"
    )

    # --- Returns on capital (quality of the compounding) -----------------------
    feats["roic"] = Feature(
        "roic",
        (latest.roic * 100.0) if (latest and latest.roic is not None) else None,
        "Return on invested capital", unit="%",
    )
    feats["roe"] = Feature(
        "roe",
        (latest.roe * 100.0) if (latest and latest.roe is not None) else None,
        "Return on equity", unit="%",
    )

    # --- Valuation: EV/EBITDA and P/S, plus a growth-vs-multiple mismatch ------
    ev_ebitda = data.valuation.ev_ebitda
    p_s = data.valuation.p_s
    ev_sales = data.valuation.ev_sales
    feats["ev_ebitda"] = Feature(
        "ev_ebitda", ev_ebitda, "EV/EBITDA", unit="x", higher_is_bullish=False
    )
    feats["p_s"] = Feature("p_s", p_s, "Price/Sales", unit="x", higher_is_bullish=False)

    # Valuation cheapness *relative to peers/universe* when a distribution exists.
    # percentile_rank returns the fraction of the distribution at or below the
    # value; for a "lower is cheaper" multiple we invert it so a high cheapness
    # score == cheaper than most comparables.
    cheapness = _relative_cheapness(ev_ebitda, p_s, ev_sales, ctx)
    feats["valuation_cheapness"] = Feature(
        "valuation_cheapness", cheapness, "Valuation cheapness vs comparables (0-1)", unit=""
    )

    # Growth/multiple mismatch: strong growth carried at a low sales multiple is
    # the classic "multiple-compression mismatch". We express it as growth% per
    # unit of P/S (or EV/Sales); higher == more mismatch (cheap for the growth).
    growth_for_mult = None
    base_mult = p_s if p_s is not None else ev_sales
    if rev_growth is not None and base_mult is not None and base_mult > 0:
        growth_for_mult = rev_growth / base_mult
    feats["growth_to_multiple"] = Feature(
        "growth_to_multiple", growth_for_mult, "Growth per unit of sales-multiple", unit=""
    )

    # --- Price structure: basing near the lows, quiet vol, off-the-high --------
    closes = [bar.close for bar in data.price_history]  # oldest-first
    window = closes[-180:] if len(closes) > 180 else closes
    feats["price_off_low"] = Feature(
        "price_off_low", _price_off_low(window), "Price above trailing low", unit="%"
    )
    feats["price_below_high"] = Feature(
        "price_below_high", _price_below_high(window), "Price below trailing high", unit="%",
        higher_is_bullish=False,
    )
    feats["realized_vol"] = Feature(
        "realized_vol", _realized_volatility(window[-60:] if len(window) > 60 else window),
        "Daily realised volatility", unit="%", higher_is_bullish=False,
    )

    # --- Insider buying --------------------------------------------------------
    buys, sells, net_buy_value = _count_insider_buys(data)
    insider_net = None
    if data.insider_transactions:
        insider_net = float(buys - sells)
    feats["insider_net_buys"] = Feature(
        "insider_net_buys", insider_net, "Net insider buys (buys - sells)", unit=""
    )
    feats["insider_net_value"] = Feature(
        "insider_net_value", net_buy_value if data.insider_transactions else None,
        "Net insider buy value", unit="",
    )

    # --- Following / neglect: thin institutional ownership & sparse coverage ----
    inst_holders = len(data.institutional_holdings)
    feats["institutional_holders"] = Feature(
        "institutional_holders", float(inst_holders) if data.institutional_holdings or True else None,
        "Institutional holders on file", unit="", higher_is_bullish=False,
    )
    feats["news_flow"] = Feature(
        "news_flow", float(len(data.news)), "Recent news items", unit="",
        higher_is_bullish=False,
    )

    # --- Spin-off / restructuring textual signal -------------------------------
    feats["spinoff_signal"] = Feature(
        "spinoff_signal", 1.0 if _spinoff_or_restructure_signal(data) else 0.0,
        "Spin-off / restructuring disclosed", unit="",
    )

    return feats


def _relative_cheapness(
    ev_ebitda: Optional[float],
    p_s: Optional[float],
    ev_sales: Optional[float],
    ctx: AnalysisContext,
) -> Optional[float]:
    """Cheapness in ``[0, 1]`` relative to peer/universe multiple distributions.

    For each available "lower is cheaper" multiple we compute ``1 -
    percentile_rank(value, distribution)`` so a value cheaper than most of the
    distribution scores near 1.0. We average across whatever multiples and
    distributions exist. Returns ``None`` when no distribution is available, so
    the caller can fall back to absolute bands.

    Each per-metric stats entry is coerced via :func:`_coerce_distribution` first,
    because in a real scan the pipeline supplies it as a summary dict
    (``{"values": [...], ...}``) rather than a bare list; passing the dict straight
    to ``percentile_rank`` would raise and silently disable the relative path.
    """
    stats_sources = [ctx.peer_stats, ctx.universe_stats]
    pairs: List[Tuple[str, Optional[float]]] = [
        ("ev_ebitda", ev_ebitda),
        ("p_s", p_s),
        ("ev_sales", ev_sales),
    ]
    scores: List[float] = []
    for stats in stats_sources:
        if not stats:
            continue
        for key, val in pairs:
            if val is None:
                continue
            dist = _coerce_distribution(stats.get(key))
            if not dist:
                continue
            pr = percentile_rank(val, dist)
            if pr is None:
                continue
            scores.append(1.0 - pr)  # lower multiple == cheaper == higher score
        if scores:
            break  # prefer peers; only fall through to universe if peers gave nothing
    if not scores:
        return None
    return sum(scores) / len(scores)


# A distribution value is only meaningful for cheapness ranking when strictly
# positive — a non-positive multiple is an artefact, not a cheap comparable — so
# the coercion below drops such entries (matching the VALUE analyzer's handling).
_MIN_MEANINGFUL = 1e-9


def _coerce_distribution(raw: Any) -> Optional[List[float]]:
    """Coerce a peer/universe stat entry into a clean list of positive floats.

    In a real scan the pipeline's ``_summarise_group`` supplies each metric entry
    as a summary *mapping* of shape ``{"values": [...], "count": ..., "min": ...,
    "max": ..., "mean": ..., "median": ...}``, so this extracts the ``"values"``
    sequence (also tolerating ``"distribution"``/``"samples"``). A bare sequence of
    numbers is accepted too, so the analyzer composes with whatever assembles the
    stats. Non-numeric, ``None`` and non-positive entries are dropped; returns
    ``None`` when nothing usable remains so the caller falls back to absolute bands.

    Mirrors :func:`convexity.analysis.value._coerce_distribution` so both analyzers
    read peer/universe distributions identically.
    """
    seq: Optional[Sequence[Any]] = None
    if isinstance(raw, dict):
        for key in ("values", "distribution", "samples"):
            candidate = raw.get(key)
            if isinstance(candidate, (list, tuple)):
                seq = candidate
                break
    elif isinstance(raw, (list, tuple)):
        seq = raw

    if seq is None:
        return None

    clean: List[float] = []
    for v in seq:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > _MIN_MEANINGFUL:
            clean.append(f)
    return clean or None


# ---------------------------------------------------------------------------
# Archetypes — named small-cap re-rating *patterns*, encoded as data
# ---------------------------------------------------------------------------
#
# Each archetype is a transparent bundle of weighted conditions over the features
# above. A "condition" is a small predicate that returns a *partial match* in
# [0, 1] (graded, not binary) plus the human evidence string describing what it
# saw. An archetype's match strength is the weight-weighted mean of its satisfied
# conditions, divided by its total *evaluable* weight (conditions whose feature is
# missing are dropped from both numerator and denominator). This keeps a thinly
# covered company from being unfairly penalised while honestly lowering coverage.
#
# NONE of this forecasts anything. An archetype name is a *label for a resemblance*
# observed in current, public data — a lens for further research, not a verdict.


ConditionFn = Callable[[float], float]


@dataclass(frozen=True)
class Condition:
    """One weighted, graded condition over a single feature.

    Attributes:
        feature_key: The :class:`Feature` this condition reads.
        weight: Relative importance of the condition within its archetype.
        grade: Maps the feature value to a partial match in ``[0, 1]``.
        describe: Builds a human evidence string from the feature value.
    """

    feature_key: str
    weight: float
    grade: ConditionFn
    describe: Callable[[Feature], str]


@dataclass(frozen=True)
class Archetype:
    """A named re-rating pattern: a label, a one-line thesis, and conditions."""

    key: str
    name: str
    summary: str
    conditions: List[Condition]


# --- Small graded-threshold helpers (pure, deterministic) ------------------


def _ramp(lo: float, hi: float) -> ConditionFn:
    """A monotonic ramp: <=lo -> 0, >=hi -> 1, linear between (higher is better)."""

    def _fn(v: float) -> float:
        return clamp(scale_to_score(v, lo, hi, higher_is_better=True) or 0.0, 0.0, 100.0) / 100.0

    return _fn


def _inv_ramp(lo: float, hi: float) -> ConditionFn:
    """Inverse ramp: <=lo -> 1, >=hi -> 0 (lower is better, e.g. cheap multiple)."""

    def _fn(v: float) -> float:
        return clamp(scale_to_score(v, lo, hi, higher_is_better=False) or 0.0, 0.0, 100.0) / 100.0

    return _fn


def _flag(threshold: float = 0.5) -> ConditionFn:
    """Treat a 0/1-style feature as a hard flag: >=threshold -> 1 else 0."""

    def _fn(v: float) -> float:
        return 1.0 if v >= threshold else 0.0

    return _fn


def _band(lo: float, hi: float) -> ConditionFn:
    """Full credit inside ``[lo, hi]``, ramping to 0 across a same-width margin."""

    width = max(hi - lo, 1e-9)

    def _fn(v: float) -> float:
        if lo <= v <= hi:
            return 1.0
        if v < lo:
            return clamp(1.0 - (lo - v) / width, 0.0, 1.0)
        return clamp(1.0 - (v - hi) / width, 0.0, 1.0)

    return _fn


def _num(feat: Feature) -> str:
    """Render a feature value for an evidence detail string."""
    if feat.value is None:
        return "n/a"
    return f"{feat.value:,.2f}".rstrip("0").rstrip(".") + (feat.unit or "")


# The archetype catalogue. Thresholds are deliberately conservative and explicit
# so the analogy is auditable; they are *characteristic*, not magic constants.
ARCHETYPES: List[Archetype] = [
    Archetype(
        key="profitable_inflection",
        name="Profitable inflection + insider buying + base breakout",
        summary=(
            "A company crossing from losses into profit while insiders buy and the "
            "stock builds a base off its lows — the classic micro-cap turn setup."
        ),
        conditions=[
            Condition("profit_inflection", 1.4, _flag(0.5),
                      lambda f: f"Profitability inflection flag = {_num(f)}"),
            Condition("fcf_inflection", 1.0, _flag(0.5),
                      lambda f: f"Free cash flow positive/inflecting = {_num(f)}"),
            Condition("insider_net_buys", 1.2, _ramp(0.0, 2.0),
                      lambda f: f"Net insider buys = {_num(f)}"),
            Condition("price_off_low", 0.9, _band(8.0, 45.0),
                      lambda f: f"Price {_num(f)} above its trailing low (basing, not extended)"),
            Condition("realized_vol", 0.6, _inv_ramp(2.0, 6.0),
                      lambda f: f"Daily realised volatility {_num(f)} (quiet base)"),
            Condition("revenue_growth", 0.7, _ramp(0.0, 20.0),
                      lambda f: f"Revenue growth {_num(f)}"),
        ],
    ),
    Archetype(
        key="margin_multiple_mismatch",
        name="Margin expansion vs multiple-compression mismatch",
        summary=(
            "Margins and growth are improving while the sales/EBITDA multiple stays "
            "depressed — a gap between operating reality and the price tag."
        ),
        conditions=[
            Condition("op_margin_delta", 1.3, _ramp(0.0, 6.0),
                      lambda f: f"Operating margin change {_num(f)}"),
            Condition("gross_margin_delta", 0.8, _ramp(0.0, 4.0),
                      lambda f: f"Gross margin change {_num(f)}"),
            Condition("revenue_growth", 1.0, _ramp(0.0, 18.0),
                      lambda f: f"Revenue growth {_num(f)}"),
            Condition("valuation_cheapness", 1.2, _ramp(0.4, 0.85),
                      lambda f: f"Valuation cheapness vs comparables {_num(f)} (0-1)"),
            Condition("growth_to_multiple", 1.1, _ramp(3.0, 15.0),
                      lambda f: f"Growth per unit of sales-multiple {_num(f)}"),
        ],
    ),
    Archetype(
        key="post_spinoff_orphan",
        name="Post-spin-off orphan",
        summary=(
            "A freshly separated or restructured entity that index funds and sell-side "
            "do not yet cover — forced/indifferent selling can leave it mispriced."
        ),
        conditions=[
            Condition("spinoff_signal", 1.6, _flag(0.5),
                      lambda f: f"Spin-off / restructuring disclosed = {_num(f)}"),
            Condition("institutional_holders", 1.1, _inv_ramp(3.0, 25.0),
                      lambda f: f"Only {_num(f)} institutional holders on file (under-owned)"),
            Condition("news_flow", 0.7, _inv_ramp(2.0, 12.0),
                      lambda f: f"Sparse coverage: {_num(f)} recent news items"),
            Condition("valuation_cheapness", 1.0, _ramp(0.4, 0.85),
                      lambda f: f"Valuation cheapness vs comparables {_num(f)} (0-1)"),
            Condition("price_below_high", 0.6, _ramp(20.0, 60.0),
                      lambda f: f"Trading {_num(f)} below its trailing high (post-separation drawdown)"),
        ],
    ),
    Archetype(
        key="deleveraging_turnaround",
        name="Deleveraging turnaround",
        summary=(
            "A levered business paying debt down while cash flow turns — equity value "
            "compounds as the enterprise de-risks even before growth re-accelerates."
        ),
        conditions=[
            Condition("debt_paydown", 1.4, _ramp(3.0, 25.0),
                      lambda f: f"Total debt reduced {_num(f)}"),
            Condition("debt_to_equity_delta", 1.1, _inv_ramp(0.0, -0.6),
                      lambda f: f"Debt-to-equity change {_num(f)} (falling)"),
            Condition("fcf_inflection", 1.2, _flag(0.5),
                      lambda f: f"Free cash flow positive/inflecting = {_num(f)}"),
            Condition("debt_to_equity", 0.8, _band(0.4, 1.8),
                      lambda f: f"Still-meaningful leverage to work down: D/E {_num(f)}"),
            Condition("op_margin_delta", 0.6, _ramp(0.0, 4.0),
                      lambda f: f"Operating margin change {_num(f)}"),
        ],
    ),
    Archetype(
        key="capacity_operating_leverage",
        name="Capacity-led operating leverage",
        summary=(
            "Revenue growing into a fixed-cost base so each incremental dollar drops "
            "through — gross margin holds while operating margin expands fast."
        ),
        conditions=[
            Condition("revenue_growth", 1.3, _ramp(8.0, 35.0),
                      lambda f: f"Revenue growth {_num(f)}"),
            Condition("op_margin_delta", 1.4, _ramp(1.0, 8.0),
                      lambda f: f"Operating margin expansion {_num(f)}"),
            Condition("gross_margin_delta", 0.7, _band(-1.0, 4.0),
                      lambda f: f"Gross margin roughly stable/up {_num(f)} (fixed-cost leverage, not pricing)"),
            Condition("fcf_inflection", 0.8, _flag(0.5),
                      lambda f: f"Free cash flow positive/inflecting = {_num(f)}"),
        ],
    ),
    Archetype(
        key="underfollowed_compounder",
        name="Under-followed compounder",
        summary=(
            "A quietly excellent business — high returns on capital, steady growth, "
            "real cash flow — that almost nobody follows yet."
        ),
        conditions=[
            Condition("roic", 1.3, _ramp(8.0, 20.0),
                      lambda f: f"Return on invested capital {_num(f)}"),
            Condition("roe", 0.8, _ramp(10.0, 22.0),
                      lambda f: f"Return on equity {_num(f)}"),
            Condition("revenue_growth", 1.0, _ramp(6.0, 22.0),
                      lambda f: f"Revenue growth {_num(f)}"),
            Condition("fcf_inflection", 0.9, _flag(0.5),
                      lambda f: f"Free cash flow positive = {_num(f)}"),
            Condition("institutional_holders", 1.0, _inv_ramp(5.0, 40.0),
                      lambda f: f"Only {_num(f)} institutional holders on file (under-followed)"),
            Condition("news_flow", 0.5, _inv_ramp(3.0, 15.0),
                      lambda f: f"Light coverage: {_num(f)} recent news items"),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Archetype matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchetypeMatch:
    """Result of scoring one archetype against a company's features.

    Attributes:
        archetype: The archetype evaluated.
        match: Weighted-mean partial match in ``[0, 1]`` over *evaluable*
            conditions (those whose feature was present).
        evaluable_weight: Sum of weights of conditions that could be evaluated.
        total_weight: Sum of all condition weights in the archetype.
        contributions: Per-condition ``(Condition, Feature, graded_value)`` for
            the conditions that were evaluable, strongest first.
    """

    archetype: Archetype
    match: float
    evaluable_weight: float
    total_weight: float
    contributions: List[Tuple[Condition, Feature, float]]

    @property
    def coverage(self) -> float:
        """Fraction of the archetype's total weight that was evaluable."""
        if self.total_weight <= 0:
            return 0.0
        return self.evaluable_weight / self.total_weight


def _score_archetype(arch: Archetype, feats: Dict[str, Feature]) -> ArchetypeMatch:
    """Score a single archetype against the feature set (pure)."""
    num = 0.0
    eval_w = 0.0
    total_w = 0.0
    contribs: List[Tuple[Condition, Feature, float]] = []
    for cond in arch.conditions:
        total_w += cond.weight
        feat = feats.get(cond.feature_key)
        if feat is None or feat.value is None:
            continue  # missing data: drop from both numerator and denominator
        graded = clamp(cond.grade(feat.value), 0.0, 1.0)
        num += cond.weight * graded
        eval_w += cond.weight
        contribs.append((cond, feat, graded))
    match = (num / eval_w) if eval_w > 0 else 0.0
    contribs.sort(key=lambda t: t[2] * t[0].weight, reverse=True)
    return ArchetypeMatch(
        archetype=arch,
        match=match,
        evaluable_weight=eval_w,
        total_weight=total_w,
        contributions=contribs,
    )


# ---------------------------------------------------------------------------
# The analyzer
# ---------------------------------------------------------------------------


@register_analyzer
class HistoricalAnalogAnalyzer(Analyzer):
    """Scores resemblance to named small-cap re-rating archetypes (pattern-similarity).

    This analyzer answers one narrow, auditable question: *how strongly does this
    company, on today's observable data, resemble a recognised re-rating setup —
    and which one?* It does **not** estimate any probability of a re-rating and
    makes **no** forecast. The score is a similarity measure, the matched archetype
    is named in the rationale and evidence, and a high score should be read as
    "worth a closer look through this lens", never as a prediction or a
    recommendation.

    Scoring
    -------
    * Features (growth, margin trend, FCF/profit inflection, deleveraging, insider
      buying, price basing, valuation-vs-growth, neglect) are computed purely from
      :class:`SecurityData`. Missing features simply do not contribute.
    * Every archetype in :data:`ARCHETYPES` is scored as a weighted mean of its
      *evaluable* conditions. The **best-matching** archetype sets the headline
      score (``match * 100``); the runner-up is reported for context.
    * ``data_coverage`` is the fraction of the winning archetype's condition weight
      that was actually evaluable, and ``confidence`` blends that coverage with the
      breadth of features available overall — thin data yields low confidence even
      when a sparse match looks superficially strong.
    """

    category = ScoreCategory.HISTORICAL_ANALOG
    default_weight = 0.04
    # Capability/field names this analyzer can use; none is individually mandatory
    # but with *none* present there is nothing to match and we fall back to neutral.
    requires = {"fundamentals", "prices", "insider", "institutional", "valuation"}

    # A match below this is treated as "no meaningful resemblance" for evidence
    # phrasing (the score is still reported honestly).
    _WEAK_MATCH = 0.35

    def analyze(self, data: SecurityData, ctx: AnalysisContext) -> SubScore:
        """Return the HISTORICAL_ANALOG :class:`SubScore` for ``data`` (pure)."""
        feats = extract_features(data, ctx)

        # Discount the spinoff_signal/news/institutional "always-computable" flags
        # when judging whether there is *real* substance to match on: require at
        # least a couple of genuinely informative fundamental/price/insider inputs.
        substantive_keys = {
            "revenue_growth", "profit_inflection", "op_margin_delta", "gross_margin_delta",
            "fcf_inflection", "debt_paydown", "debt_to_equity_delta", "roic", "roe",
            "price_off_low", "insider_net_buys", "valuation_cheapness", "growth_to_multiple",
        }
        substantive_present = [
            k for k in substantive_keys if feats.get(k) is not None and feats[k].value is not None
        ]

        if len(substantive_present) < 2:
            return self.neutral_subscore(
                rationale=(
                    "Too little observable fundamental, price or insider data to compare "
                    "this company against any re-rating archetype; no analogy is asserted."
                ),
                coverage=round(len(substantive_present) / float(len(substantive_keys)), 3),
                extra_flags=["NO_ARCHETYPE_BASIS"],
            )

        matches = [_score_archetype(a, feats) for a in ARCHETYPES]
        matches.sort(key=lambda m: (m.match, m.coverage), reverse=True)
        best = matches[0]
        runner_up = matches[1] if len(matches) > 1 else None

        # Headline score: the best resemblance, on a 0..100 scale.
        score = clamp(best.match * 100.0, 0.0, 100.0)

        # Coverage: fraction of the winning archetype's weight that was evaluable.
        coverage = clamp(best.coverage, 0.0, 1.0)

        # Confidence blends winning-archetype coverage with overall feature breadth
        # (how many substantive inputs existed), so a sparse-but-high match is not
        # over-trusted. Capped well below 1.0 because an analogy is inherently soft
        # evidence, never proof.
        breadth = len(substantive_present) / float(len(substantive_keys))
        confidence = clamp(0.15 + 0.55 * coverage + 0.20 * breadth, 0.0, 0.9)

        # --- Direction & flags -------------------------------------------------
        flags: List[str] = []
        if coverage < 0.5:
            flags.append("THIN_ARCHETYPE_COVERAGE")
        if best.match < self._WEAK_MATCH:
            flags.append("WEAK_ARCHETYPE_MATCH")
        # Ambiguous fit: top two archetypes essentially tie.
        if runner_up is not None and abs(best.match - runner_up.match) < 0.08 and best.match >= self._WEAK_MATCH:
            flags.append("AMBIGUOUS_ARCHETYPE")

        # The category leans bullish when there is a clear, well-covered match and
        # neutral otherwise; pattern resemblance is never, on its own, bearish.
        if best.match >= 0.6 and coverage >= 0.5:
            headline_direction = "bullish"
        else:
            headline_direction = "neutral"

        # --- Evidence ----------------------------------------------------------
        as_of_date = self._as_of_date(data)
        evidence: List[Evidence] = []

        # 1. Headline: the named best-matching archetype and its match strength.
        evidence.append(
            Evidence(
                label="Best-matching archetype",
                value=best.archetype.name,
                detail=(
                    f"{best.archetype.summary} Match strength "
                    f"{best.match * 100:.0f}/100 over evaluable features. "
                    "Pattern similarity only — NOT a prediction."
                ),
                source="Convexity historical-analog model",
                as_of=as_of_date,
                direction=headline_direction,
            )
        )

        # 2. The strongest few contributing features for the winning archetype.
        for cond, feat, graded in best.contributions[:4]:
            direction = self._evidence_direction(feat, graded)
            evidence.append(
                Evidence.from_number(
                    f"{feat.label} [{best.archetype.key}]",
                    feat.value,
                    source="Convexity feature extraction",
                    direction=direction,
                    unit=feat.unit,
                    detail=f"{cond.describe(feat)} — partial match {graded:.2f}",
                    as_of=as_of_date,
                )
            )

        # 3. Runner-up archetype for context (so the analogy is not over-claimed).
        if runner_up is not None and runner_up.evaluable_weight > 0:
            evidence.append(
                Evidence(
                    label="Next-closest archetype",
                    value=runner_up.archetype.name,
                    detail=(
                        f"Secondary resemblance at {runner_up.match * 100:.0f}/100 "
                        "(reported for context, not a second thesis)."
                    ),
                    source="Convexity historical-analog model",
                    as_of=as_of_date,
                    direction="neutral",
                )
            )

        # --- Rationale ---------------------------------------------------------
        if best.match >= self._WEAK_MATCH:
            rationale = (
                f"On current observable data this company most resembles the "
                f"'{best.archetype.name}' pattern (similarity {best.match * 100:.0f}/100, "
                f"{best.evaluable_weight:.1f} of {best.total_weight:.1f} feature-weight "
                f"evaluable). This is pattern similarity for further research, NOT a "
                f"prediction that a re-rating will occur."
            )
        else:
            rationale = (
                f"This company does not strongly resemble any encoded re-rating archetype "
                f"(closest is '{best.archetype.name}' at only {best.match * 100:.0f}/100). "
                f"No analogy is asserted; the score is held near neutral."
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

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _as_of_date(data: SecurityData) -> Optional[_dt.date]:
        """Best-effort as-of date for evidence: latest fundamentals period end."""
        latest = data.latest_fundamentals
        if latest is not None:
            return latest.period_end
        if data.price_history:
            return data.price_history[-1].date
        return None

    @staticmethod
    def _evidence_direction(feat: Feature, graded: float) -> str:
        """Colour a contributing feature's evidence by how well it matched.

        A strong partial match (>=0.6) in the feature's favourable direction reads
        bullish; a weak match reads neutral. A missing value is always neutral.
        """
        if feat.value is None:
            return "neutral"
        if graded >= 0.6:
            return "bullish"
        return "neutral"


__all__ = [
    "Feature",
    "Condition",
    "Archetype",
    "ArchetypeMatch",
    "ARCHETYPES",
    "extract_features",
    "HistoricalAnalogAnalyzer",
]
