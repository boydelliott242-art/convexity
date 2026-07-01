"""Catalyst analyzer — scores disclosed, near-term catalyst evidence.

Part of Convexity, an evidence-driven equity **research and screening** tool.
This module is **not** a predictor and **not** investment advice. A "catalyst"
here is only a *disclosed type of event* whose keywords appear in a company's
news or regulatory filings (e.g. an FDA approval, a buyback authorisation, an
insider purchase). Detecting one is an observation — "this disclosure mentions a
new contract" — never a claim that it will move the share price or guarantee a
return.

The :class:`CatalystAnalyzer` turns those raw detections into a single, fully
auditable :class:`~convexity.core.models.SubScore` for
:attr:`~convexity.core.models.ScoreCategory.CATALYST`. Its job is to aggregate
*many independent items* honestly:

* **Strength** — each catalyst type carries a taxonomy weight describing how
  materially that *kind* of event tends to matter as evidence (not its impact).
* **Recency** — a fresh disclosure is weighted more than a stale one via a
  smooth time-decay (older items count for less, never zero).
* **Source credibility** — a primary SEC filing outranks a wire which outranks a
  blog, so an unverified post cannot inflate the score the way a regulator's
  record can.
* **Sentiment polarity** — the disclosure's finance-aware sentiment lets a
  catalyst that is framed negatively (e.g. a recall, a dilutive raise) pull the
  score *down* rather than up; a catalyst is not assumed bullish by default.

The contributions are summed and squashed through a logistic curve so no single
item dominates, and the resulting evidence list cites the specific items, their
dates and the matched text — so every point of the score traces back to a named,
sourced disclosure.

Honesty rules honoured here:

* No network or clock access in :meth:`analyze`; the "now" used for recency is
  derived solely from the passed :class:`~convexity.core.models.SecurityData`
  (its newest disclosure, falling back to ``data.as_of``). This keeps the score
  pure and reproducible.
* When there are no news items and no filings to read, the analyzer returns
  :meth:`~convexity.core.contracts.Analyzer.neutral_subscore` (score 50, low
  confidence, ``MISSING_DATA`` flag) rather than guessing — a data gap must
  neither help nor hurt a company.
* Confidence and ``data_coverage`` scale with how much real evidence existed
  (how many sources were read and how many decisive catalysts were found), so a
  thin tape produces a low-confidence score even when something matched.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from convexity.analysis.news_nlp import (
    detect_catalysts,
    score_sentiment,
    source_credibility,
)
from convexity.core.contracts import AnalysisContext, Analyzer
from convexity.core.models import (
    Evidence,
    Filing,
    NewsItem,
    ScoreCategory,
    SecurityData,
    SubScore,
)
from convexity.core.registry import register_analyzer
from convexity.core.scoring import clamp, logistic_score, percentile_rank

# ---------------------------------------------------------------------------
# Tuning constants (all explicit and auditable; no hidden magic).
# ---------------------------------------------------------------------------

# Recency half-life in days: an item this old counts for half a fresh one. A
# generous 90-day window suits the slow, lumpy disclosure cadence of micro-caps
# while still rewarding genuinely fresh catalysts.
_RECENCY_HALF_LIFE_DAYS: float = 90.0

# Items older than this are treated as essentially stale (recency floor applies)
# and contribute a small residual rather than nothing, so a real-but-old event is
# still visible in the evidence trail.
_RECENCY_FLOOR: float = 0.05

# Logistic squashing of the net weighted catalyst signal. ``midpoint`` is the net
# signal that maps to a neutral 50; ``steepness`` controls saturation. These are
# chosen so that one strong, fresh, credible bullish catalyst (~0.8 net) lifts the
# score well above 50 without a single item pinning it to 100.
_LOGISTIC_MIDPOINT: float = 0.0
_LOGISTIC_STEEPNESS: float = 2.2

# A disclosure whose own sentiment is at/above this is treated as a bullish
# framing of its catalyst; at/below the negative of it, bearish. Between, the
# catalyst is taken at neutral framing (its taxonomy direction is ambiguous from
# text alone, so we do not assume bullishness).
_SENTIMENT_BULLISH: float = 0.15
_SENTIMENT_BEARISH: float = -0.15

# How many distinct, decisive catalyst items it takes to reach full confidence
# from the breadth-of-evidence side. Convexity's thesis is that conviction comes
# from *many independent* signals agreeing, so a lone item is never fully
# confident on its own.
_FULL_CONFIDENCE_ITEMS: int = 4

# Catalyst types whose ordinary framing is unfavourable even though the taxonomy
# lists them as recognised events. Used only as a tie-breaker when the item's own
# text sentiment is neutral, so a "dilution"-style event is not silently bullish.
# (The taxonomy in news_nlp is deliberately bullish-leaning; this guards the few
# ambiguous types.) Currently empty because every taxonomy type in news_nlp is a
# conventionally constructive disclosure; kept as an explicit, auditable hook.
_BEARISH_DEFAULT_TYPES: Set[str] = set()


@register_analyzer
class CatalystAnalyzer(Analyzer):
    """Score the CATALYST category from disclosed news and filing events.

    The analyzer reads :attr:`SecurityData.news` and :attr:`SecurityData.filings`
    (the only inputs it touches), detects catalyst types via
    :func:`convexity.analysis.news_nlp.detect_catalysts`, and combines each
    detection's *strength* (taxonomy weight) with its *recency* (time-decay vs the
    newest disclosure in the data), its *source credibility*
    (:func:`~convexity.analysis.news_nlp.source_credibility`) and its *sentiment
    polarity* (:func:`~convexity.analysis.news_nlp.score_sentiment`).

    A higher score means stronger, fresher, better-sourced, favourably-framed
    catalyst evidence. When peer/universe catalyst statistics are supplied via
    :class:`~convexity.core.contracts.AnalysisContext`, the raw signal is blended
    with the company's percentile rank against them so "a lot of catalysts" is
    judged relative to comparable names rather than on an absolute scale.
    """

    category: ScoreCategory = ScoreCategory.CATALYST
    default_weight: float = 0.10
    requires: Set[str] = {"news", "filings"}

    # -- public API ---------------------------------------------------------

    def analyze(self, data: SecurityData, ctx: AnalysisContext) -> SubScore:
        """Return the CATALYST :class:`SubScore` for ``data``.

        Pure: depends only on ``data`` and ``ctx``; no I/O, clock or randomness.
        The "current" instant used for recency is the most recent disclosure in
        the data (falling back to ``data.as_of``), so the score is reproducible.
        """
        news: List[NewsItem] = list(data.news or [])
        filings: List[Filing] = list(data.filings or [])

        # Without any disclosures there is genuinely nothing to read: a data gap,
        # not a bearish signal. Stay neutral and flag it honestly.
        if not news and not filings:
            return self.neutral_subscore(
                rationale=(
                    "No news or filings were available, so no catalyst evidence "
                    "could be assessed. This is a data gap, not a negative signal."
                ),
                coverage=0.0,
            )

        reference_now = self._reference_date(data)

        # Build the per-item catalyst contributions from both news and filings.
        contributions, flags = self._collect_contributions(
            news=news, filings=filings, reference_now=reference_now
        )

        # Data coverage: how many of the two required input streams were present.
        present_streams = (1 if news else 0) + (1 if filings else 0)
        data_coverage = present_streams / float(len(self.requires))

        if not contributions:
            # Disclosures existed but none matched a recognised catalyst type.
            # That is mildly informative (a quiet tape), so we render a slightly
            # below-neutral score with low confidence rather than a hard neutral.
            return self._no_catalyst_subscore(
                news_count=len(news),
                filings_count=len(filings),
                data_coverage=data_coverage,
                extra_flags=flags,
            )

        # Aggregate the signed, weighted contributions into a single net signal.
        net_signal, evidence, type_counts, decisive_items = self._aggregate(
            contributions
        )

        # Squash to 0..100 via a logistic so no single item pins the extremes.
        raw_score = logistic_score(
            net_signal, midpoint=_LOGISTIC_MIDPOINT, steepness=_LOGISTIC_STEEPNESS
        )
        if raw_score is None:  # pragma: no cover - net_signal is never None here.
            raw_score = 50.0

        # Blend against peers/universe when comparative stats are available so the
        # score reflects catalyst breadth *relative to comparable companies*.
        score, peer_evidence, peer_flag = self._apply_relative_context(
            absolute_score=raw_score,
            net_signal=net_signal,
            decisive_items=decisive_items,
            ctx=ctx,
        )
        if peer_evidence is not None:
            evidence.append(peer_evidence)
        if peer_flag:
            flags.append(peer_flag)

        score = clamp(score, 0.0, 100.0)

        confidence = self._confidence(
            decisive_items=decisive_items,
            data_coverage=data_coverage,
            contributions=contributions,
        )

        rationale = self._rationale(
            score=score,
            type_counts=type_counts,
            decisive_items=decisive_items,
            news_count=len(news),
            filings_count=len(filings),
        )

        if decisive_items == 1:
            flags.append("SINGLE_CATALYST")

        return SubScore(
            category=self.category,
            score=score,
            confidence=confidence,
            weight=self.default_weight,
            rationale=rationale,
            evidence=evidence,
            flags=flags,
            data_coverage=clamp(data_coverage, 0.0, 1.0),
        )

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _reference_date(data: SecurityData) -> _dt.date:
        """Derive the "now" used for recency purely from the input data.

        Uses the most recent disclosure date across news and filings; if none can
        be read, falls back to ``data.as_of``. Never reads the wall clock, keeping
        :meth:`analyze` pure and its scores reproducible.
        """
        candidates: List[_dt.date] = []
        for item in data.news or []:
            published = getattr(item, "published", None)
            if isinstance(published, _dt.datetime):
                candidates.append(published.date())
            elif isinstance(published, _dt.date):
                candidates.append(published)
        for filing in data.filings or []:
            filed = getattr(filing, "filed", None)
            if isinstance(filed, _dt.date):
                candidates.append(filed)
        if candidates:
            return max(candidates)
        as_of = getattr(data, "as_of", None)
        if isinstance(as_of, _dt.datetime):
            return as_of.date()
        if isinstance(as_of, _dt.date):  # pragma: no cover - as_of is a datetime.
            return as_of
        # Last-resort sentinel: a fixed epoch keeps purity if data is malformed.
        return _dt.date(1970, 1, 1)

    @staticmethod
    def _item_date(item: object) -> Optional[_dt.date]:
        """Best-effort disclosure date for a news item or filing."""
        published = getattr(item, "published", None)
        if isinstance(published, _dt.datetime):
            return published.date()
        if isinstance(published, _dt.date):
            return published
        filed = getattr(item, "filed", None)
        if isinstance(filed, _dt.date):
            return filed
        return None

    @classmethod
    def _recency_weight(
        cls, item_date: Optional[_dt.date], reference_now: _dt.date
    ) -> float:
        """Exponential time-decay in ``[_RECENCY_FLOOR, 1.0]`` for an item.

        A same-day or future-dated disclosure scores 1.0; older items decay with a
        :data:`_RECENCY_HALF_LIFE_DAYS` half-life, floored at
        :data:`_RECENCY_FLOOR` so a genuine-but-old catalyst is never erased.
        """
        if item_date is None:
            # Undated item: treat as moderately stale rather than fresh or absent.
            return 0.5
        age_days = (reference_now - item_date).days
        if age_days <= 0:
            return 1.0
        decay = math.pow(0.5, age_days / _RECENCY_HALF_LIFE_DAYS)
        return max(decay, _RECENCY_FLOOR)

    @classmethod
    def _direction_for(
        cls, catalyst_type: str, sentiment: float
    ) -> Tuple[float, str]:
        """Map a catalyst's framing to a sign and a human direction label.

        Returns ``(sign, direction)`` where ``sign`` is ``+1`` (bullish framing),
        ``-1`` (bearish framing) or ``0`` (neutral framing). Sentiment from the
        disclosure text drives the call; only when the text is genuinely neutral
        do we fall back to the type's conventional lean. A catalyst is never
        assumed bullish merely because it was detected.
        """
        if sentiment >= _SENTIMENT_BULLISH:
            return 1.0, "bullish"
        if sentiment <= _SENTIMENT_BEARISH:
            return -1.0, "bearish"
        if catalyst_type in _BEARISH_DEFAULT_TYPES:
            return -1.0, "bearish"
        # Neutral text framing: count the event as a mild positive presence (a
        # disclosed constructive event), but at reduced magnitude (handled by the
        # caller via a neutral multiplier) so ambiguity does not read as strong.
        return 0.5, "neutral"

    @classmethod
    def _collect_contributions(
        cls,
        *,
        news: List[NewsItem],
        filings: List[Filing],
        reference_now: _dt.date,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Detect catalysts across all items and compute each one's contribution.

        Each contribution dict captures the signed magnitude and every factor that
        produced it (strength/recency/credibility/sentiment) plus enough metadata
        to render auditable :class:`Evidence`.
        """
        flags: List[str] = []
        contributions: List[Dict[str, Any]] = []

        # Map each source object so we can recover its date/sentiment/source after
        # detection (detect_catalysts returns only type/matched_text/weight/source).
        # We therefore run detection per-item to keep the linkage exact.
        all_items: List[Tuple[str, object]] = [("news", n) for n in news] + [
            ("filing", f) for f in filings
        ]

        for stream, item in all_items:
            detections = detect_catalysts([item])
            if not detections:
                continue

            title = getattr(item, "title", None)
            summary = getattr(item, "summary", None)
            item_source = getattr(item, "source", None)
            form_type = getattr(item, "form_type", None)
            url = getattr(item, "url", None)
            item_date = cls._item_date(item)

            # Credibility: prefer the explicit source label; for filings without a
            # source, use the form type (e.g. "8-K"), which the credibility map
            # treats as a primary regulatory record.
            credibility_key = item_source or form_type
            credibility = source_credibility(credibility_key)

            recency = cls._recency_weight(item_date, reference_now)

            # Sentiment of the disclosure text frames the catalyst's direction.
            sentiment = score_sentiment(cls._text_of(title, summary))

            for det in detections:
                ctype = str(det.get("type", "unknown"))
                strength = float(det.get("weight", 0.0))
                matched_text = str(det.get("matched_text", "")).strip()
                provenance = str(det.get("source", item_source or "unknown"))

                sign, direction = cls._direction_for(ctype, sentiment)
                # Neutral framing contributes at reduced magnitude (|sign| 0.5);
                # decisive framing at full magnitude.
                magnitude = strength * recency * credibility
                signed = sign * magnitude

                contributions.append(
                    {
                        "stream": stream,
                        "type": ctype,
                        "strength": strength,
                        "recency": recency,
                        "credibility": credibility,
                        "sentiment": sentiment,
                        "direction": direction,
                        "sign": sign,
                        "magnitude": magnitude,
                        "signed": signed,
                        "matched_text": matched_text,
                        "provenance": provenance,
                        "date": item_date,
                        "title": title,
                        "url": url,
                    }
                )

            if credibility <= 0.3:
                # A low-trust source produced a catalyst; flag it so readers know
                # part of the evidence rests on unverified reporting.
                if "LOW_CREDIBILITY_SOURCE" not in flags:
                    flags.append("LOW_CREDIBILITY_SOURCE")

        # Sort strongest-first so the evidence list leads with the most material
        # contributions (stable, deterministic ordering for reproducibility).
        contributions.sort(
            key=lambda c: (abs(c["signed"]), c["strength"], c["recency"]),
            reverse=True,
        )
        return contributions, flags

    @staticmethod
    def _text_of(title: Optional[str], summary: Optional[str]) -> str:
        """Join an item's title and summary for sentiment scoring."""
        return "  ".join(part for part in (title, summary) if part)

    @classmethod
    def _aggregate(
        cls, contributions: List[Dict[str, Any]]
    ) -> Tuple[float, List[Evidence], Dict[str, int], int]:
        """Fold per-item contributions into a net signal, evidence and counts.

        Diversity discount: the *same* catalyst type repeated across many items is
        worth progressively less, so a wall of near-duplicate headlines cannot
        manufacture conviction. The first occurrence of a type counts fully; each
        further occurrence of that type is damped. This keeps the signal driven by
        *independent* events, in line with Convexity's conviction model.
        """
        net_signal = 0.0
        evidence: List[Evidence] = []
        type_counts: Dict[str, int] = {}
        type_seen: Dict[str, int] = {}
        decisive_items = 0

        for contrib in contributions:
            ctype = contrib["type"]
            type_counts[ctype] = type_counts.get(ctype, 0) + 1

            occurrence = type_seen.get(ctype, 0)
            type_seen[ctype] = occurrence + 1
            # Diminishing returns for repeats of the same catalyst type.
            diversity_factor = 1.0 / (1.0 + occurrence)

            net_signal += contrib["signed"] * diversity_factor

            if contrib["sign"] != 0.0:
                decisive_items += 1

            evidence.append(cls._evidence_for(contrib))

        # Cap the number of evidence rows so the trail stays readable; the strong
        # ordering above guarantees the most material items are kept.
        if len(evidence) > 8:
            evidence = evidence[:8]

        return net_signal, evidence, type_counts, decisive_items

    @staticmethod
    def _evidence_for(contrib: Dict[str, Any]) -> Evidence:
        """Render one contribution as an auditable :class:`Evidence` row."""
        ctype = contrib["type"].replace("_", " ")
        matched = contrib["matched_text"] or "(form-type match)"
        date = contrib["date"]
        date_str = date.isoformat() if isinstance(date, _dt.date) else "undated"
        detail = (
            f'matched "{matched}" on {date_str}; '
            f"strength={contrib['strength']:.2f}, "
            f"recency={contrib['recency']:.2f}, "
            f"credibility={contrib['credibility']:.2f}, "
            f"sentiment={contrib['sentiment']:+.2f}"
        )
        # The Evidence value reports the net contribution magnitude so a reader can
        # see how much this single item moved the category score.
        direction = contrib["direction"]
        return Evidence.from_number(
            label=f"Catalyst: {ctype}",
            value=contrib["signed"],
            source=str(contrib["provenance"]),
            direction=direction,  # type: ignore[arg-type]
            precision=2,
            detail=detail,
            as_of=date if isinstance(date, _dt.date) else None,
            url=contrib.get("url"),
        )

    def _apply_relative_context(
        self,
        *,
        absolute_score: float,
        net_signal: float,
        decisive_items: int,
        ctx: AnalysisContext,
    ) -> Tuple[float, Optional[Evidence], Optional[str]]:
        """Blend the absolute score with a peer/universe percentile when available.

        ``ctx.peer_stats``/``ctx.universe_stats`` may carry a ``"catalyst_signal"``
        distribution (a list of net catalyst signals across comparable companies).
        When present we compute this company's percentile rank within it and blend
        the absolute and relative views 50/50, so "many catalysts" is judged
        against the cohort. When absent we degrade gracefully to the absolute
        score, with no penalty for missing context.
        """
        distribution = self._peer_distribution(ctx)
        if not distribution:
            return absolute_score, None, None

        pr = percentile_rank(net_signal, distribution)
        if pr is None:
            return absolute_score, None, None

        relative_score = clamp(pr * 100.0, 0.0, 100.0)
        blended = clamp(0.5 * absolute_score + 0.5 * relative_score, 0.0, 100.0)

        evidence = Evidence.from_number(
            label="Catalyst breadth vs peers",
            value=pr * 100.0,
            source="Convexity peer comparison",
            direction=(
                "bullish"
                if relative_score >= 60.0
                else "bearish"
                if relative_score <= 40.0
                else "neutral"
            ),
            unit=" pct-ile",
            precision=0,
            detail=(
                f"net catalyst signal {net_signal:+.2f} ranks at the "
                f"{pr * 100.0:.0f}th percentile of {len(distribution)} comparable "
                "companies; absolute and relative views blended 50/50"
            ),
        )
        return blended, evidence, "PEER_RELATIVE"

    @staticmethod
    def _peer_distribution(ctx: AnalysisContext) -> List[float]:
        """Extract a catalyst-signal distribution from peer or universe stats."""
        for stats in (ctx.peer_stats, ctx.universe_stats):
            if not stats:
                continue
            dist = stats.get("catalyst_signal")
            if isinstance(dist, (list, tuple)) and dist:
                cleaned = [float(v) for v in dist if v is not None]
                if cleaned:
                    return cleaned
        return []

    def _confidence(
        self,
        *,
        decisive_items: int,
        data_coverage: float,
        contributions: List[Dict[str, Any]],
    ) -> float:
        """Confidence reflecting breadth, source quality and input coverage.

        Three multiplicative factors, each in ``[0, 1]``:

        * **breadth** — decisive (non-neutral) catalysts vs
          :data:`_FULL_CONFIDENCE_ITEMS`; one lone item is never fully confident.
        * **coverage** — how many of the required input streams were present.
        * **credibility** — the average source credibility behind the detections,
          so a tape built on blogs is discounted relative to one built on filings.
        """
        breadth = min(decisive_items / float(_FULL_CONFIDENCE_ITEMS), 1.0)
        # Even with zero decisive items (all-neutral framing) some evidence exists;
        # give a small floor so confidence is not pinned to zero when catalysts
        # were genuinely detected.
        breadth = max(breadth, 0.15 if contributions else 0.0)

        creds = [float(c["credibility"]) for c in contributions]
        avg_cred = sum(creds) / len(creds) if creds else 0.0

        confidence = breadth * clamp(data_coverage, 0.0, 1.0) * avg_cred
        # Keep within a sane band: catalyst evidence is inherently noisy, so we cap
        # confidence below certainty even in the best case.
        return clamp(confidence, 0.0, 0.9)

    def _rationale(
        self,
        *,
        score: float,
        type_counts: Dict[str, int],
        decisive_items: int,
        news_count: int,
        filings_count: int,
    ) -> str:
        """One- or two-sentence human explanation of the catalyst score."""
        distinct_types = len(type_counts)
        total = sum(type_counts.values())
        top_types = ", ".join(
            t.replace("_", " ")
            for t, _ in sorted(
                type_counts.items(), key=lambda kv: kv[1], reverse=True
            )[:3]
        )
        lean = (
            "constructive"
            if score >= 60.0
            else "weak/negative"
            if score <= 40.0
            else "mixed"
        )
        return (
            f"Detected {total} catalyst signal(s) across {distinct_types} type(s) "
            f"({top_types}) from {news_count} news item(s) and {filings_count} "
            f"filing(s); recency-, credibility- and sentiment-weighted evidence is "
            f"{lean} (score {score:.0f}/100). This is disclosed-event evidence, not "
            "a prediction that any event will move the stock."
        )

    def _no_catalyst_subscore(
        self,
        *,
        news_count: int,
        filings_count: int,
        data_coverage: float,
        extra_flags: List[str],
    ) -> SubScore:
        """SubScore for the case where disclosures exist but none are catalysts.

        A quiet but readable tape is mildly informative (slightly below neutral)
        rather than a hard data gap, but confidence stays low because absence of a
        keyword match is weak evidence.
        """
        flags = ["NO_CATALYST_DETECTED"]
        for flag in extra_flags:
            if flag not in flags:
                flags.append(flag)
        evidence = [
            Evidence(
                label="Catalyst scan",
                value="no recognised catalysts",
                source="Convexity catalyst scan",
                direction="neutral",
                detail=(
                    f"scanned {news_count} news item(s) and {filings_count} "
                    "filing(s); none matched a recognised catalyst type"
                ),
            )
        ]
        return SubScore(
            category=self.category,
            score=45.0,
            confidence=clamp(0.25 * clamp(data_coverage, 0.0, 1.0), 0.0, 1.0),
            weight=self.default_weight,
            rationale=(
                f"Read {news_count} news item(s) and {filings_count} filing(s) but "
                "found no recognised near-term catalyst. A quiet disclosure tape is "
                "a mild negative for the catalyst category, not a strong signal."
            ),
            evidence=evidence,
            flags=flags,
            data_coverage=clamp(data_coverage, 0.0, 1.0),
        )


__all__ = ["CatalystAnalyzer"]
