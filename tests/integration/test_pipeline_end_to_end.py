"""End-to-end scan integration test over the synthetic FakeProvider universe.

Runs the real :class:`~convexity.pipeline.ScanPipeline` — universe -> screen ->
fetch -> analyze -> rank -> explain — against the six invented small-caps from
``conftest.py``, entirely offline. The assertions pin the *shape and honesty* of a
scan rather than exact numbers:

* the :class:`~convexity.core.models.ScanResult` is well-formed and self-consistent
  (counts, weights, notes, timing);
* exactly ``top_n`` companies appear in ``.top`` and each is fully explained
  (non-empty thesis, bull case, bear case and monitoring checklist);
* every score across every ranked company lives in ``[0, 100]`` and every
  confidence in ``[0, 1]``;
* ``all_ranked`` is sorted by composite score descending with contiguous 1-based
  ranks; and
* the scan is **deterministic** — two runs produce identical orderings and scores
  (only the wall-clock stamps differ).
"""

from __future__ import annotations

import datetime as _dt
from typing import List

from convexity.core.models import (
    CompanyAnalysis,
    ScanParams,
    ScanResult,
    SecurityData,
    ValuationSnapshot,
)
from convexity.pipeline import ScanPipeline


def test_scan_result_is_well_formed(pipeline, scan_params: ScanParams) -> None:
    """A full scan returns a self-consistent, fully-populated ScanResult."""
    result = pipeline.scan(scan_params)

    assert isinstance(result, ScanResult)
    # Counts: the six synthetic companies all fetch and analyse cleanly.
    assert result.universe_size == 6
    assert result.screened_count == 6
    assert result.analyzed_count == 6
    assert result.error_count == 0
    # Every ranked company is present; top is the explained slice.
    assert len(result.all_ranked) == 6
    assert result.elapsed_seconds >= 0.0
    assert result.generated_at is not None
    # The eleven additive category weights plus RISK travel with the result.
    assert result.category_weights
    assert result.category_weights["value"] > 0.0
    assert result.category_weights["risk"] == 0.0
    # Notes record the run honestly (at least the completion summary).
    assert any("Scan complete" in note for note in result.notes)


def test_top_has_exactly_top_n(pipeline) -> None:
    """``.top`` contains exactly ``params.top_n`` companies (and no more)."""
    for n in (1, 2, 4):
        result = pipeline.scan(ScanParams(top_n=n))
        assert len(result.top) == n
        # The top slice is the best-first prefix of the full ranking.
        assert [c.ticker for c in result.top] == [
            c.ticker for c in result.all_ranked[:n]
        ]


def test_all_scores_within_bounds(pipeline, scan_params: ScanParams) -> None:
    """Every composite, sub-score and confidence is in its valid range."""
    result = pipeline.scan(scan_params)
    for company in result.all_ranked:
        assert 0.0 <= company.composite_score <= 100.0
        assert 0.0 <= company.conviction_confidence <= 1.0
        assert 0.0 <= company.signal_agreement <= 1.0
        assert company.subscores, f"{company.ticker} produced no sub-scores"
        for sub in company.subscores:
            assert 0.0 <= sub.score <= 100.0, f"{company.ticker}/{sub.category}"
            assert 0.0 <= sub.confidence <= 1.0
            assert 0.0 <= sub.data_coverage <= 1.0


def test_every_top_company_is_fully_explained(pipeline, scan_params: ScanParams) -> None:
    """Each explained top company carries a non-empty narrative."""
    result = pipeline.scan(scan_params)
    assert result.top, "expected at least one explained company"
    for company in result.top:
        assert company.thesis.strip(), f"{company.ticker} has an empty thesis"
        assert company.bull_case, f"{company.ticker} has an empty bull case"
        assert company.bear_case, f"{company.ticker} has an empty bear case"
        assert company.monitoring_checklist, f"{company.ticker} has no monitoring items"
        # The honest framing must survive into the thesis prose.
        assert "not a prediction" in company.thesis.lower() or \
            "not investment advice" in company.thesis.lower()


def test_all_ranked_sorted_desc_with_contiguous_ranks(
    pipeline, scan_params: ScanParams
) -> None:
    """``all_ranked`` is composite-descending and 1-based contiguously ranked."""
    result = pipeline.scan(scan_params)
    ranked: List[CompanyAnalysis] = result.all_ranked

    scores = [c.composite_score for c in ranked]
    assert scores == sorted(scores, reverse=True), "composite scores not descending"

    ranks = [c.rank for c in ranked]
    assert ranks == list(range(1, len(ranked) + 1)), f"ranks not contiguous: {ranks}"


def test_strong_outranks_weak(pipeline, scan_params: ScanParams) -> None:
    """The across-the-board-strong company outranks the across-the-board-weak one.

    This is the core sanity of the whole funnel: a company where many independent
    categories agree bullishly (STRONGCO) should sit well above one where they agree
    bearishly (WEAKCO). It is a directional, not a numeric, assertion.
    """
    result = pipeline.scan(scan_params)
    rank_of = {c.ticker: c.rank for c in result.all_ranked}
    assert rank_of["STRONGCO"] < rank_of["WEAKCO"]
    # And the strong name should be the top-ranked of the cohort.
    assert result.all_ranked[0].ticker == "STRONGCO"


def test_scan_is_deterministic_across_two_runs(pipeline, scan_params: ScanParams) -> None:
    """Two scans of the same universe yield identical orderings and scores.

    Only the wall-clock fields (``generated_at`` / ``elapsed_seconds``) may differ;
    every score, rank, conviction and ticker ordering must be byte-identical because
    the whole funnel is a pure function of the (fixed) synthetic data.
    """
    first = pipeline.scan(scan_params)
    second = pipeline.scan(scan_params)

    assert [c.ticker for c in first.all_ranked] == [
        c.ticker for c in second.all_ranked
    ]
    for a, b in zip(first.all_ranked, second.all_ranked):
        assert a.ticker == b.ticker
        assert a.rank == b.rank
        assert a.composite_score == b.composite_score
        assert a.conviction_confidence == b.conviction_confidence
        assert a.signal_agreement == b.signal_agreement


def test_thin_company_has_lower_conviction_than_strong(
    pipeline, scan_params: ScanParams
) -> None:
    """The data-starved micro-cap carries lower conviction than the strong name.

    Missing data must lower conviction, never inflate it — THINCO (one sparse
    period, no news/insiders/institutions) should be far less convicted than the
    richly-covered STRONGCO, regardless of where their composite scores land.
    """
    result = pipeline.scan(scan_params)
    by_ticker = {c.ticker: c for c in result.all_ranked}
    assert by_ticker["THINCO"].conviction_confidence < \
        by_ticker["STRONGCO"].conviction_confidence


def test_cap_band_enforced_on_fetched_data(pipeline) -> None:
    """Names whose *fetched* market cap sits outside the band are dropped.

    The fetched data is authoritative: with ``max_market_cap=400M`` the synthetic
    STRONGCO ($420M) and MIXEDTWO ($1.1B) must be excluded post-fetch even though
    the (patched) universe stage surfaced all six tickers, and the exclusion must
    be recorded honestly in counts and notes.
    """
    result = pipeline.scan(ScanParams(top_n=2, max_market_cap=400_000_000))

    ranked_tickers = {c.ticker for c in result.all_ranked}
    assert "STRONGCO" not in ranked_tickers
    assert "MIXEDTWO" not in ranked_tickers
    assert ranked_tickers == {"WEAKCO", "MIXEDONE", "STEADYCO", "THINCO"}

    # Counts stay truthful: 6 candidates surfaced, 4 survived the band.
    assert result.universe_size == 6
    assert result.screened_count == 4
    assert result.analyzed_count == 4
    assert result.error_count == 0

    # The exclusion is recorded as an explicit, auditable note.
    assert any(
        "excluded post-fetch: outside cap band" in note for note in result.notes
    ), result.notes


def test_min_cap_floor_enforced_on_fetched_data(pipeline) -> None:
    """The lower bound of the band is enforced too (drops the sub-$100M names)."""
    result = pipeline.scan(ScanParams(top_n=2, min_market_cap=100_000_000))

    ranked_tickers = {c.ticker for c in result.all_ranked}
    # MIXEDONE ($95M) and THINCO ($55M) fall below the floor.
    assert ranked_tickers == {"STRONGCO", "WEAKCO", "MIXEDTWO", "STEADYCO"}
    assert result.screened_count == 4
    assert any(
        "excluded post-fetch: outside cap band" in note for note in result.notes
    )


def test_unknown_market_cap_is_kept_with_warning() -> None:
    """A ``None`` market cap is never silently dropped — kept, with a warning.

    Honesty over tidiness: absent data must not exclude a name; it is kept in the
    cohort with an appended ``data_warning`` so conviction can reflect the gap.
    """
    unknown = SecurityData(
        ticker="NOCAP",
        name="NoCap Corp",
        as_of=_dt.datetime(2026, 6, 30),
        valuation=ValuationSnapshot(market_cap=None),
    )
    out_of_band = SecurityData(
        ticker="BIGCO",
        name="BigCo Inc",
        as_of=_dt.datetime(2026, 6, 30),
        valuation=ValuationSnapshot(market_cap=27_000_000_000.0),
    )
    in_band = SecurityData(
        ticker="OKCO",
        name="OkCo Ltd",
        as_of=_dt.datetime(2026, 6, 30),
        valuation=ValuationSnapshot(market_cap=300_000_000.0),
    )

    params = ScanParams(min_market_cap=50_000_000, max_market_cap=2_000_000_000)
    kept, excluded, unknown_count = ScanPipeline._apply_cap_band(
        [unknown, out_of_band, in_band], params
    )

    assert [d.ticker for d in kept] == ["NOCAP", "OKCO"]
    assert excluded == 1
    assert unknown_count == 1
    assert any("Market cap unknown after fetch" in w for w in unknown.data_warnings)
    # Re-applying must not duplicate the warning.
    ScanPipeline._apply_cap_band([unknown], params)
    assert (
        sum("Market cap unknown after fetch" in w for w in unknown.data_warnings) == 1
    )


def test_universe_limit_shrinks_the_scan(pipeline) -> None:
    """``universe_limit`` deterministically caps how many tickers are scanned."""
    result = pipeline.scan(ScanParams(top_n=2, universe_limit=3))
    assert result.universe_size == 3
    assert result.analyzed_count == 3
    assert len(result.all_ranked) == 3
