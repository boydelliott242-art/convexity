"""Regression tests for post-fetch cap-band enforcement in the pipeline (BUG B).

The original defect: the pipeline never re-checked ``min_market_cap`` /
``max_market_cap`` after fetching, so a scan with ``max_market_cap=2e9`` ranked
$27B and $9.3B names. The fix added ``ScanPipeline._apply_cap_band`` (stage 3),
which drops out-of-band names, keeps unknown-cap names with a recorded data
warning, and surfaces both in the :class:`ScanResult` notes.

These tests run a real end-to-end scan (real analyzers, ranking and
explainability) against a five-name synthetic cohort, fully offline and
deterministic:

* ``BIGCAP`` ($27B, the MLI shape) and ``TINYCAP`` ($10M) must be excluded from
  ``all_ranked``;
* ``NOCAP`` (market cap ``None``) must be *retained* with a data warning — the
  pipeline never silently drops on absent data;
* the ``ScanResult`` notes must mention both the out-of-band exclusions and the
  unknown-cap retention.
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Optional

import pytest

from convexity.core.models import (
    CapTier,
    ScanParams,
    ScanResult,
    SecurityData,
    ValuationSnapshot,
)
from convexity.pipeline import ScanPipeline
from tests.conftest import FakeProvider, build_security, make_price_history

# Scan band used throughout: the small/micro-cap default shape.
_MIN_CAP = 50_000_000.0
_MAX_CAP = 2_000_000_000.0


def _make_security(
    ticker: str,
    market_cap: Optional[float],
    *,
    sector: str = "Technology",
) -> SecurityData:
    """Build a minimal-but-valid synthetic security with the given market cap."""
    valuation = (
        ValuationSnapshot(market_cap=market_cap)
        if market_cap is not None
        else ValuationSnapshot()  # market_cap stays None — the unknown-cap case
    )
    bars = make_price_history(
        start=_dt.date(2025, 6, 1),
        days=90,
        start_price=10.0,
        end_price=12.0,
        volume=100_000.0,
    )
    return build_security(
        ticker=ticker,
        name=f"{ticker.title()} Synthetic Co",
        sector=sector,
        industry="Testing",
        cap_tier=CapTier.SMALL,
        valuation=valuation,
        price_history=bars,
    )


def _registry() -> Dict[str, SecurityData]:
    """Five synthetics: one above band, one below, one unknown, two in-band."""
    companies = [
        _make_security("BIGCAP", 27_000_000_000.0),  # $27B — the real-scan MLI case
        _make_security("TINYCAP", 10_000_000.0),     # $10M — below the floor
        _make_security("NOCAP", None),               # unknown cap — kept + warned
        _make_security("INBAND1", 420_000_000.0),
        _make_security("INBAND2", 275_000_000.0),
    ]
    return {c.ticker: c for c in companies}


@pytest.fixture()
def registry() -> Dict[str, SecurityData]:
    return _registry()


@pytest.fixture()
def scan_result(
    registry: Dict[str, SecurityData], monkeypatch: pytest.MonkeyPatch
) -> ScanResult:
    """Run one offline scan over the five synthetics and share the result."""
    from convexity.data import universe as universe_mod

    provider = FakeProvider(registry)
    tickers: List[str] = list(registry.keys())

    def _fake_universe(params: ScanParams, price_provider: object = None, **_kw: object) -> List[str]:
        limit = params.universe_limit
        if limit is not None and limit >= 0:
            return tickers[:limit]
        return list(tickers)

    # The pipeline resolves ``universe_mod.build_universe_or_seed`` at call time,
    # so patching the module attribute keeps the whole scan offline.
    monkeypatch.setattr(universe_mod, "build_universe_or_seed", _fake_universe)

    pipeline = ScanPipeline(provider=provider)
    params = ScanParams(
        min_market_cap=_MIN_CAP,
        max_market_cap=_MAX_CAP,
        min_avg_dollar_volume=0.0,
        top_n=3,
    )
    return pipeline.scan(params)


class TestScanCapBandEnforcement:
    def test_out_of_band_names_excluded_from_all_ranked(self, scan_result: ScanResult) -> None:
        ranked_tickers = {a.ticker for a in scan_result.all_ranked}
        assert "BIGCAP" not in ranked_tickers    # $27B > max_market_cap
        assert "TINYCAP" not in ranked_tickers   # $10M < min_market_cap
        # Nothing out-of-band leaks into the explained top slice either.
        assert all(a.ticker not in {"BIGCAP", "TINYCAP"} for a in scan_result.top)

    def test_in_band_and_unknown_cap_names_are_ranked(self, scan_result: ScanResult) -> None:
        ranked_tickers = {a.ticker for a in scan_result.all_ranked}
        assert ranked_tickers == {"INBAND1", "INBAND2", "NOCAP"}
        assert scan_result.screened_count == 3
        assert scan_result.analyzed_count == 3

    def test_notes_mention_cap_band_exclusions(self, scan_result: ScanResult) -> None:
        band_notes = [n for n in scan_result.notes if "outside cap band" in n]
        assert len(band_notes) == 1
        # Exactly the two out-of-band names were counted.
        assert band_notes[0].startswith("2 name(s) excluded post-fetch")
        # The note states the band actually enforced.
        assert f"${_MIN_CAP:,.0f}" in band_notes[0]
        assert f"${_MAX_CAP:,.0f}" in band_notes[0]

    def test_notes_mention_unknown_cap_retention(self, scan_result: ScanResult) -> None:
        unknown_notes = [n for n in scan_result.notes if "unknown market cap" in n]
        assert len(unknown_notes) == 1
        assert unknown_notes[0].startswith("1 name(s) kept with unknown market cap")


class TestApplyCapBandUnit:
    """Direct unit coverage of ``ScanPipeline._apply_cap_band`` semantics."""

    def _params(self) -> ScanParams:
        return ScanParams(min_market_cap=_MIN_CAP, max_market_cap=_MAX_CAP)

    def test_partition_and_counts(self, registry: Dict[str, SecurityData]) -> None:
        fetched = list(registry.values())
        kept, excluded, unknown = ScanPipeline._apply_cap_band(fetched, self._params())

        assert [d.ticker for d in kept] == ["NOCAP", "INBAND1", "INBAND2"]  # order preserved
        assert excluded == 2   # BIGCAP + TINYCAP
        assert unknown == 1    # NOCAP

    def test_boundary_caps_are_inclusive(self) -> None:
        at_min = _make_security("ATMIN", _MIN_CAP)
        at_max = _make_security("ATMAX", _MAX_CAP)
        kept, excluded, unknown = ScanPipeline._apply_cap_band(
            [at_min, at_max], self._params()
        )
        assert [d.ticker for d in kept] == ["ATMIN", "ATMAX"]
        assert excluded == 0 and unknown == 0

    def test_unknown_cap_gets_data_warning_without_duplication(self) -> None:
        nocap = _make_security("NOCAP", None)
        assert not any("cap-band" in w for w in nocap.data_warnings)

        kept, _excluded, unknown = ScanPipeline._apply_cap_band([nocap], self._params())
        assert unknown == 1
        assert kept == [nocap]
        warnings = [w for w in nocap.data_warnings if "cap-band eligibility" in w]
        assert len(warnings) == 1
        assert "Market cap unknown after fetch" in warnings[0]

        # Re-applying the band must not stack a duplicate warning.
        ScanPipeline._apply_cap_band([nocap], self._params())
        warnings = [w for w in nocap.data_warnings if "cap-band eligibility" in w]
        assert len(warnings) == 1
