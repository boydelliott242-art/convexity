"""Dependency-light financial NLP helpers for news and filing evidence.

This module is part of Convexity, an evidence-driven equity **research and
screening** tool. It is **not** a predictor and **not** investment advice. The
helpers here turn unstructured text (news headlines, filing summaries) into
small, auditable signals — a finance-aware sentiment polarity, a set of
catalyst tags, and a source-credibility weight — so that downstream analyzers
can fold them into transparent sub-scores alongside many *other* independent
pieces of evidence. No single text signal should ever drive a thesis on its
own; value comes only from many independent signals agreeing.

Design constraints
------------------
* **No heavy ML.** Everything here is a deterministic, transparent,
  lexicon/regex approach (Loughran-McDonald-style word lists and keyword
  patterns). A reader can audit exactly which words or phrases produced a
  signal — there is no opaque model.
* **Pure & deterministic.** Given identical input the functions always return
  identical output: no I/O, no wall-clock, no randomness. This keeps every
  derived score reproducible.
* **Honest about uncertainty.** Sentiment is a coarse polarity in ``[-1, 1]``,
  not a probability of anything. Catalyst detection reports *what keywords
  matched*, never a claim that an event will move a price. Credibility weights
  encode only how directly a source observes facts (a regulatory filing is a
  primary record; an anonymous blog is not), never an endorsement.
* **Dependency-light.** Only the Python standard library is imported at
  runtime; the Pydantic model types are referenced for type hints only (under
  ``TYPE_CHECKING``) so this module stays importable with nothing installed and
  operates on any duck-typed object exposing ``.title`` / ``.summary`` /
  ``.form_type`` attributes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from re import Pattern
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a hard pydantic dependency.
    pass


# ---------------------------------------------------------------------------
# Finance sentiment lexicon (Loughran-McDonald-style)
# ---------------------------------------------------------------------------
#
# General-purpose sentiment lexicons mislabel finance text: words like
# "liability", "tax" or "cost" are neutral accounting terms, not negatives.
# The Loughran-McDonald (LM) financial sentiment dictionary was built precisely
# to fix this. The sets below are a compact, hand-curated subset of LM-style
# positive and negative terms relevant to small/micro-cap news and filings.
# They are intentionally finance-tuned: e.g. "going concern", "dilution" and
# "delist" are strong negatives, while "record", "beat" and "upgraded" are
# strong positives. Words are stored lower-case and matched on word boundaries.

POSITIVE_WORDS: Set[str] = {
    "accelerate",
    "accelerated",
    "accelerating",
    "achieve",
    "achieved",
    "advantage",
    "approval",
    "approved",
    "award",
    "awarded",
    "beat",
    "beats",
    "benefit",
    "benefited",
    "best",
    "better",
    "boost",
    "boosted",
    "breakthrough",
    "collaboration",
    "complete",
    "completed",
    "exceed",
    "exceeded",
    "exceeds",
    "expand",
    "expanded",
    "expansion",
    "favorable",
    "gain",
    "gained",
    "gains",
    "good",
    "grow",
    "growing",
    "growth",
    "high",
    "higher",
    "improve",
    "improved",
    "improvement",
    "improving",
    "innovative",
    "leading",
    "milestone",
    "momentum",
    "outperform",
    "outperformed",
    "partnership",
    "positive",
    "profit",
    "profitable",
    "progress",
    "raise",
    "raised",
    "rebound",
    "record",
    "recovery",
    "robust",
    "strength",
    "strong",
    "stronger",
    "strongest",
    "success",
    "successful",
    "surge",
    "surged",
    "upgrade",
    "upgraded",
    "upside",
    "win",
    "winning",
    "wins",
}

NEGATIVE_WORDS: Set[str] = {
    "adverse",
    "bankruptcy",
    "bankrupt",
    "breach",
    "concern",
    "concerns",
    "decline",
    "declined",
    "declines",
    "default",
    "defaulted",
    "deficiency",
    "deficit",
    "delay",
    "delayed",
    "delays",
    "delist",
    "delisted",
    "delisting",
    "deteriorate",
    "deteriorating",
    "dilution",
    "dilutive",
    "disappointing",
    "downgrade",
    "downgraded",
    "downturn",
    "drop",
    "dropped",
    "fail",
    "failed",
    "failure",
    "fall",
    "falling",
    "fell",
    "fraud",
    "halt",
    "halted",
    "impairment",
    "investigation",
    "lawsuit",
    "litigation",
    "loss",
    "losses",
    "miss",
    "missed",
    "misses",
    "negative",
    "plunge",
    "plunged",
    "probe",
    "recall",
    "recession",
    "restate",
    "restated",
    "restatement",
    "shortfall",
    "slump",
    "subpoena",
    "sue",
    "sued",
    "suspend",
    "suspended",
    "underperform",
    "underperformed",
    "warn",
    "warned",
    "warning",
    "weak",
    "weaker",
    "weakness",
    "worse",
    "worst",
    "writedown",
    "write-down",
}

# Multi-word phrases carry stronger, less ambiguous polarity than single tokens.
# Each maps to a signed weight added once per occurrence to the raw polarity sum.
POSITIVE_PHRASES: Dict[str, float] = {
    "record revenue": 2.0,
    "record quarter": 2.0,
    "beat expectations": 2.0,
    "raised guidance": 2.0,
    "raises guidance": 2.0,
    "better than expected": 2.0,
    "ahead of schedule": 1.5,
    "fda approval": 2.0,
    "fda approved": 2.0,
}

NEGATIVE_PHRASES: Dict[str, float] = {
    "going concern": 2.5,
    "material weakness": 2.0,
    "below expectations": 2.0,
    "missed expectations": 2.0,
    "lowered guidance": 2.0,
    "cut guidance": 2.0,
    "worse than expected": 2.0,
    "short seller": 1.5,
    "short-seller": 1.5,
    "reverse split": 1.5,
    "reverse stock split": 1.5,
    "below compliance": 1.5,
}

# Negators that flip the polarity of a sentiment word appearing shortly after.
_NEGATORS: Set[str] = {"no", "not", "never", "without", "lack", "lacks", "lacking", "fails", "failing"}

# How many tokens after a negator remain "negated".
_NEGATION_WINDOW = 3

# Pre-compiled tokenizer: words and hyphenated/possessive words, lower-cased.
_TOKEN_RE: Pattern[str] = re.compile(r"[A-Za-z][A-Za-z\-']*")


def _tokenize(text: str) -> List[str]:
    """Split ``text`` into lower-cased word tokens (deterministic, regex-based)."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def score_sentiment(text: Optional[str]) -> float:
    """Return a finance-aware sentiment polarity for ``text`` in ``[-1, 1]``.

    The score is a transparent, lexicon-based polarity — **not** a probability
    and **not** a forecast. A positive value means the wording leans favourable,
    a negative value means it leans unfavourable, and ``0.0`` means neutral,
    empty, or balanced. The magnitude reflects how lopsided the matched
    sentiment terms are, squashed into ``[-1, 1]`` so a single inflammatory word
    cannot dominate.

    Method (fully auditable):

    1. Score multi-word phrases first (they are less ambiguous than single
       tokens) using the signed weights in :data:`POSITIVE_PHRASES` /
       :data:`NEGATIVE_PHRASES`.
    2. Tokenize and add ``+1`` for each :data:`POSITIVE_WORDS` hit and ``-1``
       for each :data:`NEGATIVE_WORDS` hit, **flipping** the contribution of any
       sentiment word that falls within :data:`_NEGATION_WINDOW` tokens after a
       negator (so "not strong" reads bearish, "no loss" reads bullish).
    3. Normalise the net polarity by the number of sentiment-bearing matches so
       the result is intensity-aware but bounded, then clamp to ``[-1, 1]``.

    Args:
        text: The headline or summary to score. ``None`` or whitespace-only
            input returns ``0.0`` (no signal — never fabricated).

    Returns:
        A float in ``[-1.0, 1.0]``.
    """
    if not text:
        return 0.0
    lowered = text.lower()

    polarity = 0.0
    matches = 0

    # 1. Multi-word phrases (counted by non-overlapping occurrences).
    for phrase, weight in POSITIVE_PHRASES.items():
        n = lowered.count(phrase)
        if n:
            polarity += weight * n
            matches += n
    for phrase, weight in NEGATIVE_PHRASES.items():
        n = lowered.count(phrase)
        if n:
            polarity -= weight * n
            matches += n

    # 2. Single tokens with a small negation window.
    tokens = _tokenize(text)
    negate_until = -1
    for i, tok in enumerate(tokens):
        if tok in _NEGATORS:
            negate_until = i + _NEGATION_WINDOW
            continue
        contribution = 0.0
        if tok in POSITIVE_WORDS:
            contribution = 1.0
        elif tok in NEGATIVE_WORDS:
            contribution = -1.0
        if contribution != 0.0:
            if i <= negate_until:
                contribution = -contribution
            polarity += contribution
            matches += 1

    if matches == 0:
        return 0.0

    # 3. Normalise by match count (intensity-aware) and clamp.
    raw = polarity / float(matches)
    if raw > 1.0:
        return 1.0
    if raw < -1.0:
        return -1.0
    return raw


# ---------------------------------------------------------------------------
# Catalyst taxonomy
# ---------------------------------------------------------------------------
#
# A "catalyst" here is simply a recognised *type of disclosed event* whose
# keywords appear in a headline or filing. Detecting one is an observation
# ("this text mentions a buyback"), never a claim that it will move the stock.
# Each catalyst type carries a small ``weight`` describing how materially that
# *kind* of event tends to matter as evidence — not its probability or impact.
# Patterns are case-insensitive and pre-compiled. Keeping them as explicit regex
# makes every detection auditable: you can see exactly which phrase matched.

CATALYST_TAXONOMY: Dict[str, Dict[str, object]] = {
    "guidance_raise": {
        "weight": 0.9,
        "patterns": [
            r"rais(?:e|es|ed|ing)\s+(?:its\s+|the\s+|full[-\s]?year\s+|fy\s*\d*\s+)?(?:guidance|outlook|forecast)",
            r"(?:guidance|outlook|forecast)\s+rais(?:e|ed)",
            r"(?:increas|boost|lift|hik)(?:e|es|ed|ing)\s+(?:its\s+)?(?:guidance|outlook|forecast)",
            r"upbeat\s+(?:guidance|outlook|forecast)",
            r"(?:raised|higher)\s+(?:full[-\s]?year|annual)\s+(?:guidance|outlook|forecast)",
        ],
    },
    "earnings_beat": {
        "weight": 0.7,
        "patterns": [
            r"beat(?:s|en)?\s+(?:analyst|wall\s*street|consensus|street|earnings|revenue|estimates|expectations)",
            r"top(?:s|ped|ping)?\s+(?:analyst|consensus|street|estimates|expectations|forecasts)",
            r"(?:earnings|revenue|results|profit|eps)\s+beat",
            r"better[-\s]than[-\s]expected\s+(?:earnings|revenue|results|profit)",
            r"exceed(?:s|ed)?\s+(?:estimates|expectations|forecasts|consensus)",
        ],
    },
    "new_contract": {
        "weight": 0.7,
        "patterns": [
            # Verb ... (optional amount/adjectives, up to ~4 tokens) ... target noun.
            r"(?:award(?:ed)?|win(?:s|ning)?|won|secur(?:e|es|ed))\s+(?:[\$\w][\$\w.,-]*\s+){0,4}(?:contract|order|deal|award)\b",
            r"new\s+(?:contract|order|customer|client)\s+(?:win|award|with)",
            r"(?:multi[-\s]?year|government|defense|defence)\s+contract",
            r"purchase\s+order\s+(?:worth|valued|for)",
            r"backlog\s+(?:grow|increas|expand)",
        ],
    },
    "product_launch": {
        "weight": 0.6,
        "patterns": [
            # Verb ... (optional adjectives, up to ~4 tokens) ... product noun.
            r"(?:launch(?:es|ed|ing)?|unveil(?:s|ed|ing)?|introduc(?:e|es|ed|ing)|debut(?:s|ed|ing)?)\s+(?:[\$\w][\$\w.,-]*\s+){0,4}(?:product|platform|device|service|app|solution|model|software|chip|drug)\b",
            r"(?:product|platform|service)\s+launch",
            r"commercial\s+(?:launch|availability|rollout)",
            r"general\s+availability",
            r"begins?\s+shipping",
        ],
    },
    "regulatory_approval": {
        "weight": 1.0,
        "patterns": [
            r"\bfda\b.*\bapprov",
            r"approv(?:al|ed|es).*\bfda\b",
            r"(?:received|grant(?:ed|s)?|obtain(?:ed|s)?)\s+(?:fda\s+|regulatory\s+|ce\s+mark\s+|510\(k\)\s+|marketing\s+)?(?:approval|clearance|authorization|authorisation|designation)",
            r"(?:regulatory|marketing)\s+(?:approval|clearance|authorization|authorisation)",
            r"(?:breakthrough|orphan\s+drug|fast[-\s]track)\s+designation",
            r"(?:phase\s+(?:2|ii|3|iii))\s+(?:trial\s+)?(?:success|met|positive|results)",
        ],
    },
    "insider_buying": {
        "weight": 0.8,
        "patterns": [
            r"insider(?:s)?\s+(?:buy(?:ing)?|bought|purchas(?:e|es|ed|ing)|acquir(?:e|es|ed))",
            r"(?:ceo|cfo|chief\s+executive|chief\s+financial|director|officer|chairman|founder)\s+(?:buy(?:s|ing)?|bought|purchas(?:es|ed)|acquir(?:es|ed))\s+(?:shares|stock)",
            r"(?:open[-\s]market|insider)\s+(?:purchase|buying)",
            r"form\s*4\s+(?:purchase|buy)",
        ],
    },
    "buyback": {
        "weight": 0.6,
        "patterns": [
            r"(?:share|stock)\s+(?:buy[-\s]?back|repurchase)",
            r"(?:buy[-\s]?back|repurchase)\s+(?:program|programme|plan|authoriz|authoris)",
            r"(?:authoriz|authoris)(?:e|es|ed|ing)\s+(?:a\s+)?(?:\$[\d.,]+\s*(?:million|billion|m|bn)?\s+)?(?:buy[-\s]?back|repurchase)",
            r"repurchas(?:e|es|ed|ing)\s+(?:its\s+|common\s+)?(?:shares|stock)",
        ],
    },
    "m_and_a": {
        "weight": 0.9,
        "patterns": [
            r"\b(?:merger|acquisition)\b",
            r"(?:to\s+)?acquir(?:e|es|ed|ing)\b",
            r"(?:agree(?:s|d)?\s+to\s+(?:acquire|buy|merge)|definitive\s+(?:merger|agreement))",
            r"(?:takeover|buyout)\s+(?:offer|bid|proposal|agreement)",
            r"(?:to\s+be\s+|being\s+)acquired\s+by",
            r"all[-\s]cash\s+(?:deal|transaction|acquisition)",
        ],
    },
    "index_inclusion": {
        "weight": 0.6,
        "patterns": [
            r"(?:add(?:ed|s|ition)?|join(?:s|ed|ing)?|inclu(?:de|ded|sion))\s+to\s+(?:the\s+)?(?:s&p|russell|nasdaq|dow|msci|ftse)\b",
            r"(?:s&p|russell|nasdaq|msci|ftse)\s+(?:\d+\s+)?(?:index\s+)?(?:inclusion|addition)",
            r"will\s+(?:be\s+)?(?:add|join|includ)(?:ed)?\s+to\s+(?:the\s+)?(?:s&p|russell|nasdaq|dow|msci|ftse)",
            r"index\s+(?:inclusion|reconstitution|rebalanc)",
        ],
    },
    "debt_reduction": {
        "weight": 0.7,
        "patterns": [
            r"(?:reduc|lower|cut|pay(?:ing)?\s+down|repay(?:ing)?|retir(?:e|es|ed|ing)|eliminat(?:e|es|ed|ing))\s+(?:its\s+|the\s+|net\s+|total\s+|long[-\s]?term\s+)?debt",
            r"debt\s+(?:reduction|repayment|paydown|refinanc)",
            r"(?:deleverag|de[-\s]leverag)(?:e|es|ed|ing)",
            r"refinanc(?:e|es|ed|ing)\s+(?:its\s+|the\s+)?(?:debt|notes|credit\s+facility|loan)",
            r"(?:strengthen(?:s|ed)?|improv(?:e|es|ed))\s+(?:its\s+)?balance\s+sheet",
        ],
    },
}

# The SEC form types that, on their own, evidence a given catalyst type. A
# Form 4 is the canonical insider-transaction filing; an 8-K with no parseable
# body is at least a material-event disclosure. These let a filing contribute a
# catalyst even when its title/summary text is sparse.
_FORM_CATALYSTS: Dict[str, str] = {
    "4": "insider_buying",
    "form 4": "insider_buying",
    "form4": "insider_buying",
    "sc 13d": "m_and_a",
    "sc 13d/a": "m_and_a",
    "425": "m_and_a",
    "defm14a": "m_and_a",
}

# Pre-compile every catalyst pattern once at import (deterministic, no runtime
# recompilation). Stored as {catalyst_type: (weight, [compiled_pattern, ...])}.
_COMPILED_CATALYSTS: Dict[str, Tuple[float, List[Pattern[str]]]] = {
    name: (
        float(spec["weight"]),  # type: ignore[arg-type]
        [re.compile(p, re.IGNORECASE) for p in spec["patterns"]],  # type: ignore[union-attr]
    )
    for name, spec in CATALYST_TAXONOMY.items()
}


def _coerce_text(*parts: Optional[str]) -> str:
    """Join the non-empty string parts with a separator for pattern matching."""
    return "  ".join(p for p in parts if p)


def detect_catalysts(items: Iterable[object]) -> List[Dict[str, object]]:
    """Detect disclosed catalyst types in an iterable of news/filing-like items.

    Each input item is duck-typed: any object exposing some of ``.title``,
    ``.summary`` and ``.form_type`` (e.g. a :class:`~convexity.core.models.NewsItem`
    or :class:`~convexity.core.models.Filing`) is accepted. The function is pure
    and deterministic — it reports *which catalyst keywords matched*, nothing
    more. It never asserts that a catalyst will affect price or returns.

    For every item:

    * The ``form_type`` (if any) is checked against :data:`_FORM_CATALYSTS` so a
      Form 4 evidences ``insider_buying`` even with no descriptive text.
    * The combined ``title`` + ``summary`` text is matched against every pattern
      in :data:`CATALYST_TAXONOMY`. A type is reported at most once per item (the
      first matching pattern wins) to avoid double-counting the same event.

    Args:
        items: Iterable of news/filing-like objects.

    Returns:
        A list of dicts, one per (item, catalyst-type) detection, each with:

        * ``type`` (str) — the catalyst key (e.g. ``"buyback"``).
        * ``matched_text`` (str) — the exact substring/snippet that matched,
          so the detection is auditable.
        * ``weight`` (float) — the taxonomy weight for that catalyst type.
        * ``source`` (str) — the item's reported source/form, for provenance.

        The list is in input order; within one item, detections follow
        taxonomy declaration order. An empty iterable yields an empty list.
    """
    results: List[Dict[str, object]] = []

    for item in items:
        title = getattr(item, "title", None)
        summary = getattr(item, "summary", None)
        form_type = getattr(item, "form_type", None)
        source = getattr(item, "source", None)

        # Provenance label: prefer an explicit news source, else the form type.
        provenance = source or (f"SEC {form_type}" if form_type else "unknown")

        # Track which catalyst types we have already recorded for this item so we
        # never emit the same type twice for one disclosure.
        seen: Set[str] = set()

        # 1. Form-type-driven catalysts (no text needed).
        if form_type:
            norm = str(form_type).strip().lower()
            mapped = _FORM_CATALYSTS.get(norm)
            if mapped and mapped not in seen:
                seen.add(mapped)
                results.append(
                    {
                        "type": mapped,
                        "matched_text": f"form {form_type}",
                        "weight": _COMPILED_CATALYSTS.get(
                            mapped, (CATALYST_TAXONOMY[mapped]["weight"], [])  # type: ignore[index]
                        )[0],
                        "source": provenance,
                    }
                )

        # 2. Text-pattern-driven catalysts.
        text = _coerce_text(title, summary)
        if text:
            for ctype, (weight, patterns) in _COMPILED_CATALYSTS.items():
                if ctype in seen:
                    continue
                for pat in patterns:
                    m = pat.search(text)
                    if m:
                        seen.add(ctype)
                        results.append(
                            {
                                "type": ctype,
                                "matched_text": m.group(0).strip(),
                                "weight": weight,
                                "source": provenance,
                            }
                        )
                        break  # one match per catalyst type per item.

    return results


# ---------------------------------------------------------------------------
# Source credibility
# ---------------------------------------------------------------------------
#
# Not all "evidence" is equally trustworthy. A regulatory filing is a primary,
# legally-accountable record; a wire from a major newswire is a vetted secondary
# source; an anonymous blog or forum post is unverified. ``source_credibility``
# returns a multiplicative weight in (0, 1] so downstream analyzers can discount
# a signal by how directly its source observes the underlying facts. This is a
# transparency/quality control — never an endorsement of any view a source holds.

# Highest-trust: primary regulatory / company-of-record sources (~1.0).
_REGULATORY_SOURCES: Set[str] = {
    "sec",
    "sec edgar",
    "edgar",
    "sec.gov",
    "8-k",
    "10-k",
    "10-q",
    "form 4",
    "s-1",
    "company filing",
    "regulatory filing",
    "fda",
    "fda.gov",
}

# Company-issued primary communications (~0.85): the issuer's own words. Trusted
# as to fact-of-statement, but promotional, hence just below a regulator.
_PRIMARY_COMPANY_SOURCES: Set[str] = {
    "press release",
    "company press release",
    "businesswire",
    "business wire",
    "globenewswire",
    "globe newswire",
    "pr newswire",
    "prnewswire",
    "accesswire",
    "newsfile",
    "investor relations",
    "company",
}

# Reputable primary/secondary news organisations (~0.7).
_PRIMARY_NEWS_SOURCES: Set[str] = {
    "reuters",
    "bloomberg",
    "associated press",
    "the wall street journal",
    "wall street journal",
    "wsj",
    "financial times",
    "ft",
    "the new york times",
    "new york times",
    "cnbc",
    "barron's",
    "barrons",
    "marketwatch",
    "dow jones",
    "forbes",
    "the economist",
}

# Aggregators / mainstream secondary finance media (~0.5).
_AGGREGATOR_SOURCES: Set[str] = {
    "yahoo finance",
    "yahoo",
    "google finance",
    "google news",
    "benzinga",
    "thestreet",
    "investing.com",
    "zacks",
    "morningstar",
    "marketbeat",
    "simply wall st",
    "tipranks",
}

# Low-trust: opinion/blog/UGC platforms (~0.3). Useful as a weak signal only.
_BLOG_SOURCES: Set[str] = {
    "seeking alpha",
    "seekingalpha",
    "motley fool",
    "the motley fool",
    "fool.com",
    "blog",
    "substack",
    "medium",
    "reddit",
    "r/wallstreetbets",
    "wallstreetbets",
    "stocktwits",
    "twitter",
    "x.com",
    "youtube",
    "discord",
    "message board",
    "forum",
}

# Default for an unrecognised source: cautious, below an aggregator (~0.4).
_DEFAULT_CREDIBILITY = 0.4

# Substring hints (checked when the exact source string is not in a set above).
# Order matters: most-trusted first so a "sec ..." string out-ranks a generic hit.
_CREDIBILITY_HINTS: List[Tuple[Tuple[str, ...], float]] = [
    (("sec", "edgar", "10-k", "10-q", "8-k", "form 4", "s-1", ".gov", "fda"), 1.0),
    (("press release", "wire", "globenewswire", "newswire", "investor relations"), 0.85),
    (
        (
            "reuters",
            "bloomberg",
            "associated press",
            "wall street journal",
            "wsj",
            "financial times",
            "cnbc",
            "barron",
            "marketwatch",
            "dow jones",
            "forbes",
        ),
        0.7,
    ),
    (("yahoo", "google", "benzinga", "thestreet", "investing.com", "zacks", "morningstar", "marketbeat"), 0.5),
    (
        (
            "seeking alpha",
            "seekingalpha",
            "motley fool",
            "fool",
            "blog",
            "substack",
            "medium",
            "reddit",
            "stocktwits",
            "twitter",
            "youtube",
            "forum",
            "discord",
        ),
        0.3,
    ),
]


def source_credibility(source: Optional[str]) -> float:
    """Return a credibility weight in ``(0, 1]`` for a named ``source``.

    Higher means the source observes the underlying facts more directly:

    * ``~1.0`` — primary regulatory / company-of-record filings (SEC, FDA).
    * ``~0.85`` — company-issued primary communications (press releases, wires).
    * ``~0.7`` — reputable primary/secondary news organisations.
    * ``~0.5`` — mainstream aggregators / secondary finance media.
    * ``~0.4`` — unrecognised sources (cautious default).
    * ``~0.3`` — opinion / blog / user-generated platforms.

    Matching is case-insensitive and tolerant: it first tries an exact
    (normalised) match against the curated sets, then falls back to substring
    hints (most-trusted first). The weight encodes *only* observational
    directness and accountability — never agreement with the source's stance.

    Args:
        source: The source label (e.g. ``"Reuters"``, ``"SEC EDGAR"``,
            ``"Seeking Alpha"``). ``None``/empty returns the cautious default.

    Returns:
        A float in ``(0.0, 1.0]``.
    """
    if not source:
        return _DEFAULT_CREDIBILITY
    norm = source.strip().lower()
    if not norm:
        return _DEFAULT_CREDIBILITY

    # Exact (normalised) membership — fastest and most precise.
    if norm in _REGULATORY_SOURCES:
        return 1.0
    if norm in _PRIMARY_COMPANY_SOURCES:
        return 0.85
    if norm in _PRIMARY_NEWS_SOURCES:
        return 0.7
    if norm in _AGGREGATOR_SOURCES:
        return 0.5
    if norm in _BLOG_SOURCES:
        return 0.3

    # Substring hints (most-trusted tier first).
    for needles, weight in _CREDIBILITY_HINTS:
        for needle in needles:
            if needle in norm:
                return weight

    return _DEFAULT_CREDIBILITY


__all__ = [
    "POSITIVE_WORDS",
    "NEGATIVE_WORDS",
    "POSITIVE_PHRASES",
    "NEGATIVE_PHRASES",
    "score_sentiment",
    "CATALYST_TAXONOMY",
    "detect_catalysts",
    "source_credibility",
]
