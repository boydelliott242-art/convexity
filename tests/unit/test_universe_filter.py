"""Name-layer universe filter: investment-vehicle exclusion (SPACs & CEFs).

Regression suite for the real-scan defect where listed investment vehicles
reached the top 25: closed-end funds (abrdn Healthcare Investors HQH, Putnam
Managed Municipal Income Trust PMM, Eaton Vance Tax-Advantaged Global Dividend
ETO, ASA Gold & Precious Metals ASA) and a SPAC (Breeze Acquisition Corp. II
BREZ) all passed ``_looks_like_common_stock``. The platform researches
OPERATING companies — funds and blank-check shells have no revenue, margins or
insiders, so their scores lean on a thin subset of categories and mislead.

These tests pin, fully offline:

* each of the five real escapees is now EXCLUDED, via the SPAC name patterns,
  the closed-end-fund name patterns, or the curated CEF fund-family prefixes;
* operating companies survive: LPs ("Star Group, L.P."), REITs (including
  names carrying "Trust"), operating asset managers ("Artisan Partners Asset
  Management" — no bare "asset management" pattern exists), and plain
  industrials — the filter is a NAME layer only, so pre-revenue biotechs and
  other thin-data operating companies are untouched;
* word-boundary matching: "spac" does not fire inside "Spacelabs";
* every new exclusion path is COUNTED (``excluded_spac``,
  ``excluded_closed_end_fund``, ``excluded_fund_family``) through the parsers
  and :func:`fetch_listed_symbols` into the ``stats`` accounting, so scan notes
  stay truthful — conservative exclusions are reported, never silent.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from convexity.data import universe as universe_mod
from convexity.data.universe import (
    _common_stock_exclusion_reason,
    _investment_vehicle_reason,
    _looks_like_common_stock,
    _parse_nasdaq_listed,
    _parse_other_listed,
    fetch_listed_symbols,
)

# ---------------------------------------------------------------------------
# The five real escapees from the full scan — all must now be excluded
# ---------------------------------------------------------------------------


class TestInvestmentVehiclesExcluded:
    """CEFs and SPACs that previously slipped through are dropped by name."""

    @pytest.mark.parametrize(
        ("ticker", "name", "expected_reason"),
        [
            # HQH: no fund vocabulary in the name at all — only the sponsor
            # prefix gives it away.
            ("HQH", "abrdn Healthcare Investors", "fund_family"),
            # PMM: previously RESCUED by the "income trust" REIT rescue; the
            # sponsor prefix must win before the rescue is even consulted.
            ("PMM", "Putnam Managed Municipal Income Trust", "fund_family"),
            # ETO: sponsor prefix and CEF vocabulary both present.
            (
                "ETO",
                "Eaton Vance Tax-Advantaged Global Dividend Opportunities Fund",
                "fund_family",
            ),
            # ASA: the "asa gold" prefix marks the fund.
            ("ASA", "ASA Gold & Precious Metals Limited", "fund_family"),
            # BREZ: blank-check shell — "Acquisition Corp." with punctuation
            # after "Corp" must still match (word-boundary, not whole-word-list).
            ("BREZ", "Breeze Acquisition Corp. II", "spac"),
            # GAIN / GLAD: BDCs (registered investment companies) whose names
            # carry NO fund vocabulary at all — the real escapee from the
            # 2026-07-01 full scan. Only the decisive whole-name identity
            # catches them — in the full spelling, the abbreviated "Corp."
            # the CQS/other-listed directory uses, and the incident's casing.
            ("GAIN", "Gladstone Investment Corporation", "fund_family"),
            ("GLAD", "Gladstone Capital Corporation", "fund_family"),
            ("GAIN", "Gladstone Investment Corp.", "fund_family"),
            ("GLAD", "Gladstone Capital Corp.", "fund_family"),
            ("GAIN", "GLADSTONE INVESTMENT CORPORATION", "fund_family"),
        ],
    )
    def test_real_escapees_are_excluded(
        self, ticker: str, name: str, expected_reason: str
    ) -> None:
        assert not _looks_like_common_stock(ticker, name)
        assert _common_stock_exclusion_reason(ticker, name) == expected_reason

    @pytest.mark.parametrize(
        "name",
        [
            "Frontier Acquisition Corporation",
            "Global Partner Acquisition Co II",
            "Example Acquisition Company Holdings",
            "Anywhere Blank Check Corp",
        ],
    )
    def test_spac_name_pattern_variants(self, name: str) -> None:
        assert _investment_vehicle_reason(name) == "spac"
        assert not _looks_like_common_stock("XXXX", name)

    @pytest.mark.parametrize(
        "name",
        [
            "Anywhere Municipal Income Trust",
            "Somebody Floating Rate Income Fund",
            "Somebody Senior Loan Trust",
            "Somebody Total Return Fund",
            "Somebody Multi-Sector Income Trust",
            "Somebody Premium Income Trust",
            "Somebody Closed-End Opportunities Trust",
        ],
    )
    def test_cef_strategy_vocabulary_is_excluded(self, name: str) -> None:
        # Sponsor-less names carrying CEF strategy vocabulary — several of
        # these would otherwise be rescued by the REIT "income trust" rescue.
        assert _investment_vehicle_reason(name) == "closed_end_fund"
        assert not _looks_like_common_stock("XXX", name)

    @pytest.mark.parametrize(
        "name",
        [
            "Nuveen Quality Municipal Income Fund",
            "BlackRock Health Sciences Trust",
            "Gabelli Equity Trust Inc.",
            "The Gabelli Dividend & Income Trust",
            "Liberty All-Star Equity Fund",
            "Tri-Continental Corporation",
            "General American Investors Company, Inc.",
            # Sponsor prefix + fund vocabulary / unrescued "Trust" — the fund
            # side of sponsors that are ALSO listed operating companies.
            "Invesco Bond Fund",
            "Franklin Universal Trust",
            "Virtus Convertible & Income Fund",
            "Cohen & Steers Infrastructure Fund",
            "BlackRock Capital Allocation Term Trust",
            "Royce Micro-Cap Trust",
            "John Hancock Investors Trust",
            # Whole-name fund identities where the prefix alone is decisive.
            "Tortoise Energy Infrastructure Corporation",
            "Adams Diversified Equity Fund, Inc.",
        ],
    )
    def test_fund_family_prefix_is_a_prefix_match(self, name: str) -> None:
        reason = _common_stock_exclusion_reason("XXX", name)
        assert reason in ("fund_family", "closed_end_fund", "non_common_name")
        assert not _looks_like_common_stock("XXX", name)


# ---------------------------------------------------------------------------
# Operating companies must survive — the filter stays conservative
# ---------------------------------------------------------------------------


class TestOperatingCompaniesKept:
    """LPs, REITs, operating managers and plain industrials are untouched."""

    @pytest.mark.parametrize(
        ("ticker", "name"),
        [
            # Operating LP (propane distributor) — "L.P." is not a vehicle marker.
            ("SGU", "Star Group, L.P."),
            # Operating REITs, including one carrying "Trust" in the name (the
            # trust rescue must keep working alongside the CEF patterns).
            ("ARE", "Alexandria Real Estate Equities, Inc."),
            ("IIPR", "Innovative Industrial Properties, Inc."),
            ("CPT", "Camden Property Trust"),
            ("UHT", "Universal Health Realty Income Trust"),
            # Operating asset manager — there is deliberately no bare
            # "asset management" pattern.
            ("APAM", "Artisan Partners Asset Management Inc."),
            # Plain operating company.
            ("DAKT", "Daktronics Inc"),
            # CEF sponsors that are THEMSELVES listed operating companies: a
            # bare sponsor prefix must not exclude the sponsor's own stock.
            ("BLK", "BlackRock, Inc."),
            ("IVZ", "Invesco Ltd"),
            ("BEN", "Franklin Resources, Inc."),
            ("VRTS", "Virtus Investment Partners, Inc."),
            ("CNS", "Cohen & Steers, Inc."),
            # Operating companies that merely share a sponsor's surname —
            # small/micro-caps squarely in the platform's scope.
            ("FELE", "Franklin Electric Co., Inc."),
            ("FC", "Franklin Covey Company"),
            ("FSP", "Franklin Street Properties Corp."),
            ("FRAF", "Franklin Financial Services Corporation"),
            # Sponsor-prefixed operating REIT: "Realty Trust" is rescued, so
            # the sponsor-prefix "Trust" heuristic must not fire either.
            ("FBRT", "Franklin BSP Realty Trust, Inc."),
            # Sponsor-prefixed BDC/mREIT with no fund vocabulary: kept
            # (conservative — when in doubt, keep).
            ("TCPC", "BlackRock TCP Capital Corp."),
            ("IVR", "Invesco Mortgage Capital Inc."),
            # Gladstone OPERATING companies: the BDC exclusions above key on
            # the "Capital"/"Investment" token (whole-name identities), so the
            # family's farmland REIT and net-lease REIT stay in the universe.
            ("LAND", "Gladstone Land Corporation"),
            ("GOOD", "Gladstone Commercial Corporation"),
            ("LAND", "Gladstone Land Corp."),
        ],
    )
    def test_operating_companies_survive(self, ticker: str, name: str) -> None:
        assert _common_stock_exclusion_reason(ticker, name) is None
        assert _looks_like_common_stock(ticker, name)

    def test_spac_pattern_is_word_boundary_aware(self) -> None:
        # "spac" must not fire inside longer words ("Spacelabs", "Aerospace").
        assert _investment_vehicle_reason("Spacelabs Healthcare Inc") is None
        assert _investment_vehicle_reason("Ducommun Aerospace Inc") is None
        assert _looks_like_common_stock("SLAB", "Spacelabs Healthcare Inc")

    def test_family_prefix_never_matches_mid_name(self) -> None:
        # Prefix match only: a sponsor name *inside* an operating company's
        # name must not exclude it.
        assert _investment_vehicle_reason("First Franklin Bancshares Inc") is None
        assert _looks_like_common_stock("FFB", "First Franklin Bancshares Inc")

    def test_name_layer_only_no_fundamentals_consulted(self) -> None:
        # A pre-revenue biotech is indistinguishable from any operating company
        # at this layer — nothing about fundamentals is (or can be) consulted.
        assert _looks_like_common_stock("XBIO", "Example Therapeutics, Inc.")


# ---------------------------------------------------------------------------
# Conservative-exclusion accounting: every new path is counted
# ---------------------------------------------------------------------------

_NASDAQ_LISTED_FIXTURE = "\n".join(
    [
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
        "Round Lot Size|ETF|NextShares",
        "BREZ|Breeze Acquisition Corp. II|G|N|N|100|N|N",
        "DAKT|Daktronics Inc|Q|N|N|100|N|N",
        "PMM|Putnam Managed Municipal Income Trust|G|N|N|100|N|N",
        "File Creation Time: 0630202617:03|||||||",
    ]
)

_OTHER_LISTED_FIXTURE = "\n".join(
    [
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|"
        "Test Issue|NASDAQ Symbol",
        "HQH|abrdn Healthcare Investors|N|HQH|N|100|N|HQH",
        "ETO|Eaton Vance Tax-Advantaged Global Dividend Opportunities Fund"
        "|N|ETO|N|100|N|ETO",
        "ASA|ASA Gold & Precious Metals Limited|N|ASA|N|100|N|ASA",
        "XYZ|Anywhere Municipal Income Trust|N|XYZ|N|100|N|XYZ",
        "SGU|Star Group, L.P.|N|SGU|N|100|N|SGU",
        "CPT|Camden Property Trust|N|CPT|N|100|N|CPT",
        "File Creation Time: 0630202617:03|||||||",
    ]
)


class TestExclusionAccounting:
    """The new exclusion reasons flow into the honest ``stats`` counters."""

    def test_nasdaq_parser_counts_vehicle_exclusions(self) -> None:
        counters: Dict[str, int] = {}
        kept = _parse_nasdaq_listed(_NASDAQ_LISTED_FIXTURE, counters=counters)
        assert [t for t, _ in kept] == ["DAKT"]
        assert counters == {"excluded_spac": 1, "excluded_fund_family": 1}

    def test_other_parser_counts_vehicle_exclusions(self) -> None:
        counters: Dict[str, int] = {}
        kept = _parse_other_listed(_OTHER_LISTED_FIXTURE, counters=counters)
        assert [t for t, _ in kept] == ["SGU", "CPT"]
        # HQH + ETO + ASA carry sponsor prefixes; the sponsor-less
        # "Municipal Income" name is counted as a CEF pattern hit.
        assert counters == {
            "excluded_fund_family": 3,
            "excluded_closed_end_fund": 1,
        }

    def test_fetch_listed_symbols_fills_stats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        responses = {
            universe_mod._NASDAQ_LISTED_URL: _NASDAQ_LISTED_FIXTURE,
            universe_mod._OTHER_LISTED_URL: _OTHER_LISTED_FIXTURE,
        }
        monkeypatch.setattr(
            universe_mod,
            "_http_get_text",
            lambda url, *, user_agent, timeout: responses.get(url),
        )
        stats: Dict[str, int] = {}
        tickers = fetch_listed_symbols(stats=stats)
        assert tickers == ["CPT", "DAKT", "SGU"]
        assert stats["excluded_spac"] == 1
        assert stats["excluded_fund_family"] == 4
        assert stats["excluded_closed_end_fund"] == 1

    def test_stats_counters_present_even_when_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An all-operating-company listing still reports the counters (at 0),
        # so downstream notes can distinguish "none excluded" from "not tracked".
        clean = "\n".join(
            [
                "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
                "Round Lot Size|ETF|NextShares",
                "DAKT|Daktronics Inc|Q|N|N|100|N|N",
            ]
        )
        responses = {universe_mod._NASDAQ_LISTED_URL: clean}
        monkeypatch.setattr(
            universe_mod,
            "_http_get_text",
            lambda url, *, user_agent, timeout: responses.get(url),
        )
        stats: Dict[str, int] = {}
        tickers = fetch_listed_symbols(stats=stats)
        assert tickers == ["DAKT"]
        assert stats["excluded_spac"] == 0
        assert stats["excluded_closed_end_fund"] == 0
        assert stats["excluded_fund_family"] == 0

    def test_pipeline_turns_vehicle_counters_into_a_note(self) -> None:
        from convexity.pipeline import ScanPipeline

        notes: List[str] = []
        ScanPipeline._note_universe_screen_stats(
            {
                "excluded_spac": 1,
                "excluded_closed_end_fund": 2,
                "excluded_fund_family": 4,
                "used_seed_fallback": 0,
            },
            notes,
        )
        note = next(n for n in notes if "investment" in n)
        assert "7 listed investment vehicle(s)" in note
        assert "SPAC/blank-check: 1" in note
        assert "closed-end fund pattern: 2" in note
        assert "fund-family prefix: 4" in note

    def test_pipeline_adds_no_vehicle_note_when_none_excluded(self) -> None:
        from convexity.pipeline import ScanPipeline

        notes: List[str] = []
        ScanPipeline._note_universe_screen_stats(
            {
                "excluded_spac": 0,
                "excluded_closed_end_fund": 0,
                "excluded_fund_family": 0,
                "used_seed_fallback": 0,
            },
            notes,
        )
        assert not any("investment vehicle" in n for n in notes)
