"""Unit tests for :mod:`convexity.analysis.news_nlp`.

These tests pin the *behavioural contract* of the dependency-light NLP helpers:

* ``score_sentiment`` returns a finance-aware polarity in ``[-1, 1]`` with the
  correct sign, handles negation, and treats missing/neutral text as ``0.0``.
* ``detect_catalysts`` recognises each catalyst type from headline/summary text
  and from SEC form types, reporting auditable ``matched_text`` and the correct
  ``weight`` / ``source`` provenance, on duck-typed news/filing-like objects.
* ``source_credibility`` orders sources by observational directness
  (regulatory > primary news > blog) and stays within ``(0, 1]``.

The module under test depends only on the standard library, so these tests use
lightweight stand-in objects with ``.title`` / ``.summary`` / ``.form_type`` /
``.source`` attributes rather than the Pydantic models — exactly the duck-typed
contract ``detect_catalysts`` is documented to accept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pytest

from convexity.analysis.news_nlp import (
    CATALYST_TAXONOMY,
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
    detect_catalysts,
    score_sentiment,
    source_credibility,
)

# ---------------------------------------------------------------------------
# Lightweight duck-typed stand-ins for NewsItem / Filing
# ---------------------------------------------------------------------------


@dataclass
class FakeNews:
    """A news-like object exposing the attributes ``detect_catalysts`` reads."""

    title: Optional[str] = None
    summary: Optional[str] = None
    source: Optional[str] = None
    form_type: Optional[str] = None  # news items have no form type by default.


@dataclass
class FakeFiling:
    """A filing-like object exposing ``.form_type`` (and optional text)."""

    form_type: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    source: Optional[str] = None


def _types(detections: List[dict]) -> List[str]:
    return [d["type"] for d in detections]


# ---------------------------------------------------------------------------
# score_sentiment — polarity, range, negation, neutrality
# ---------------------------------------------------------------------------


class TestSentimentPolarity:
    def test_positive_text_is_positive(self) -> None:
        s = score_sentiment("Company reports record revenue and strong growth, beats estimates")
        assert s > 0.0

    def test_negative_text_is_negative(self) -> None:
        s = score_sentiment("Company warns of weak demand; reports a loss and a going concern")
        assert s < 0.0

    def test_neutral_text_is_zero(self) -> None:
        assert score_sentiment("The company held its annual shareholder meeting on Tuesday") == 0.0

    @pytest.mark.parametrize("text", [None, "", "   ", "\n\t"])
    def test_empty_or_missing_text_is_zero(self, text: Optional[str]) -> None:
        assert score_sentiment(text) == 0.0

    @pytest.mark.parametrize(
        "text",
        [
            "record revenue strong growth profit surge upgraded beat win success",
            "loss fraud bankruptcy lawsuit halt delist plunge fail weak going concern",
            "the quick brown fox",
            "mixed: strong growth but a disappointing loss and weak guidance",
        ],
    )
    def test_score_always_within_unit_range(self, text: str) -> None:
        s = score_sentiment(text)
        assert -1.0 <= s <= 1.0

    def test_negation_flips_positive(self) -> None:
        # "not strong" / "no growth" should read at most neutral, never bullish.
        assert score_sentiment("results were not strong this quarter") <= 0.0

    def test_negation_flips_negative_to_non_bearish(self) -> None:
        # "no loss" should not be counted as bearish.
        assert score_sentiment("the firm reported no loss for the period") >= 0.0

    def test_positive_phrase_outweighs_single_words(self) -> None:
        # A strong multi-word positive phrase yields a clearly bullish read.
        assert score_sentiment("management raised guidance for the full year") > 0.3

    def test_negative_phrase_is_strongly_bearish(self) -> None:
        assert score_sentiment("auditor flags going concern and material weakness") < -0.3

    def test_more_positive_text_scores_higher_than_mildly_positive(self) -> None:
        strong = score_sentiment("record profit, strong growth, beats estimates, upgraded")
        mild = score_sentiment("results were good")
        assert strong >= mild > 0.0

    def test_determinism(self) -> None:
        text = "Company beats estimates but warns of weak guidance"
        assert score_sentiment(text) == score_sentiment(text)


def test_lexicons_are_disjoint_and_nonempty() -> None:
    """A word must not be both positive and negative (auditable, unambiguous)."""
    assert POSITIVE_WORDS, "positive lexicon must not be empty"
    assert NEGATIVE_WORDS, "negative lexicon must not be empty"
    assert POSITIVE_WORDS.isdisjoint(NEGATIVE_WORDS)


# ---------------------------------------------------------------------------
# detect_catalysts — per-type detection, form types, provenance, weights
# ---------------------------------------------------------------------------


class TestCatalystDetection:
    @pytest.mark.parametrize(
        "text,expected_type",
        [
            ("Acme raises full-year guidance after strong quarter", "guidance_raise"),
            ("Acme beats analyst estimates for Q2 earnings", "earnings_beat"),
            ("Acme awarded a $50 million contract by the Navy", "new_contract"),
            ("Acme launches new flagship platform for enterprise", "product_launch"),
            ("Acme receives FDA approval for its lead drug", "regulatory_approval"),
            ("CEO buys 100,000 shares in open-market purchase", "insider_buying"),
            ("Board authorizes a $20 million share buyback program", "buyback"),
            ("Acme agrees to acquire rival in all-cash deal", "m_and_a"),
            ("Acme to be added to the S&P 600 index", "index_inclusion"),
            ("Acme repays its long-term debt and strengthens balance sheet", "debt_reduction"),
        ],
    )
    def test_each_catalyst_type_detected(self, text: str, expected_type: str) -> None:
        detections = detect_catalysts([FakeNews(title=text, source="Reuters")])
        assert expected_type in _types(detections)

    def test_detection_record_shape(self) -> None:
        detections = detect_catalysts(
            [FakeNews(title="Board authorizes a share repurchase program", source="Business Wire")]
        )
        assert detections, "expected at least one detection"
        rec = next(d for d in detections if d["type"] == "buyback")
        assert set(rec.keys()) == {"type", "matched_text", "weight", "source"}
        assert isinstance(rec["matched_text"], str) and rec["matched_text"]
        assert rec["weight"] == CATALYST_TAXONOMY["buyback"]["weight"]
        assert rec["source"] == "Business Wire"

    def test_form4_filing_detects_insider_buying_without_text(self) -> None:
        detections = detect_catalysts([FakeFiling(form_type="4")])
        assert "insider_buying" in _types(detections)
        rec = next(d for d in detections if d["type"] == "insider_buying")
        assert "form 4" in rec["matched_text"].lower()
        assert rec["source"].startswith("SEC")

    def test_no_false_positive_on_neutral_text(self) -> None:
        detections = detect_catalysts(
            [FakeNews(title="Company to present at an investor conference next week")]
        )
        assert detections == []

    def test_summary_is_searched_when_title_is_thin(self) -> None:
        item = FakeNews(
            title="Corporate update",
            summary="The company completed an acquisition of a competitor.",
            source="GlobeNewswire",
        )
        assert "m_and_a" in _types(detect_catalysts([item]))

    def test_each_type_reported_at_most_once_per_item(self) -> None:
        # Text that could match a catalyst via several phrasings should yield one.
        item = FakeNews(
            title="Acme launches new product; Acme unveils new product line today",
        )
        detections = detect_catalysts([item])
        assert _types(detections).count("product_launch") == 1

    def test_empty_iterable_returns_empty_list(self) -> None:
        assert detect_catalysts([]) == []

    def test_multiple_items_preserve_input_order(self) -> None:
        items = [
            FakeNews(title="Acme raises guidance", source="Reuters"),
            FakeNews(title="Acme announces share buyback", source="PR Newswire"),
        ]
        detections = detect_catalysts(items)
        assert _types(detections) == ["guidance_raise", "buyback"]

    def test_weights_are_taxonomy_weights(self) -> None:
        detections = detect_catalysts(
            [FakeNews(title="Acme receives FDA approval for new therapy")]
        )
        rec = next(d for d in detections if d["type"] == "regulatory_approval")
        assert rec["weight"] == CATALYST_TAXONOMY["regulatory_approval"]["weight"]

    def test_taxonomy_has_all_required_types(self) -> None:
        required = {
            "guidance_raise",
            "new_contract",
            "product_launch",
            "regulatory_approval",
            "insider_buying",
            "buyback",
            "m_and_a",
            "index_inclusion",
            "debt_reduction",
            "earnings_beat",
        }
        assert required.issubset(set(CATALYST_TAXONOMY.keys()))


# ---------------------------------------------------------------------------
# source_credibility — ordering and range
# ---------------------------------------------------------------------------


class TestSourceCredibility:
    def test_credibility_ordering(self) -> None:
        sec = source_credibility("SEC EDGAR")
        wire = source_credibility("Business Wire")
        news = source_credibility("Reuters")
        aggregator = source_credibility("Yahoo Finance")
        blog = source_credibility("Seeking Alpha")
        # Strictly decreasing tiers: filing > company wire > news > aggregator > blog.
        assert sec > wire > news > aggregator > blog

    def test_sec_filing_is_highest(self) -> None:
        assert source_credibility("SEC") == pytest.approx(1.0)

    def test_blog_is_low(self) -> None:
        assert source_credibility("Reddit") <= 0.35

    @pytest.mark.parametrize(
        "source",
        ["SEC EDGAR", "Reuters", "Bloomberg", "Seeking Alpha", "Yahoo Finance", "an unheard-of site", None, ""],
    )
    def test_credibility_within_unit_range(self, source: Optional[str]) -> None:
        c = source_credibility(source)
        assert 0.0 < c <= 1.0

    def test_unknown_source_uses_cautious_default(self) -> None:
        c = source_credibility("Some Random Newsletter XYZ")
        # Below an aggregator, above a blog — a cautious middle default.
        assert source_credibility("Reddit") < c < source_credibility("Yahoo Finance")

    def test_case_insensitive(self) -> None:
        assert source_credibility("reuters") == source_credibility("REUTERS")

    def test_substring_hint_matches_regulatory(self) -> None:
        # An unlisted but clearly-regulatory string still scores top via hints.
        assert source_credibility("Filed with SEC via EDGAR system") == pytest.approx(1.0)

    def test_none_and_empty_return_default(self) -> None:
        assert source_credibility(None) == source_credibility("")
