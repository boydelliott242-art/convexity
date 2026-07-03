"""Derived-market-cap fallback in the composite aggregation (rate-limit incident).

Regression suite for the real full-universe scan where Yahoo throttled the
``ticker.info`` endpoint: 2,261 of 2,438 names came back with an UNKNOWN market
cap even though the price history (chart endpoint, not throttled) and
``shares_diluted`` (SEC EDGAR companyfacts) were both on hand. The fix is a
post-merge normalisation step in ``CompositeProvider.get_security_data``: when
the merged record has no provider-supplied ``valuation.market_cap`` but a price
history AND a positive ``shares_diluted`` are present, the cap is derived as
``last close × diluted shares`` and an explicit provenance warning is appended
(real arithmetic on real data — labelled, never fabricated).

These tests pin, fully offline:

* the derivation fires with both inputs present — exact value AND the exact
  provenance warning text;
* no derivation when either input is missing (empty price history, missing
  fundamentals, ``None``/non-positive ``shares_diluted``) — the cap honestly
  stays ``None`` and no derived-cap warning appears;
* a provider-supplied market cap is NEVER overwritten — including when it
  arrives from a lower-priority provider merged after the derivation inputs;
* ``cap_tier`` is back-filled from the derived cap using the existing
  ``CapTier`` thresholds.
"""

from __future__ import annotations

import datetime as _dt
from typing import List, Optional, Set

from convexity.core.contracts import DataProvider
from convexity.core.models import (
    CapTier,
    FundamentalsPeriod,
    PriceBar,
    SecurityData,
    ValuationSnapshot,
)
from convexity.data.aggregator import _DERIVED_MARKET_CAP_WARNING, CompositeProvider

_AS_OF = _dt.datetime(2026, 6, 30, 12, 0, 0)


# ---------------------------------------------------------------------------
# Offline builders (no network, no I/O)
# ---------------------------------------------------------------------------


def _bar(day: int, close: float) -> PriceBar:
    return PriceBar(
        date=_dt.date(2026, 6, day),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=10_000.0,
    )


def _fundamentals(shares_diluted: Optional[float]) -> FundamentalsPeriod:
    return FundamentalsPeriod(
        period_end=_dt.date(2026, 3, 31),
        period_label="Q1 2026",
        revenue=12_000_000.0,
        shares_diluted=shares_diluted,
    )


def _security(
    *,
    market_cap: Optional[float] = None,
    price_history: Optional[List[PriceBar]] = None,
    fundamentals: Optional[List[FundamentalsPeriod]] = None,
    cap_tier: Optional[CapTier] = None,
    source: str = "yfinance",
) -> SecurityData:
    return SecurityData(
        ticker="TEST",
        name="Test Operating Co",
        cap_tier=cap_tier,
        as_of=_AS_OF,
        valuation=ValuationSnapshot(market_cap=market_cap),
        price_history=list(price_history or []),
        fundamentals=list(fundamentals or []),
        data_sources=[source],
    )


class _FakeProvider(DataProvider):
    """Fake member provider serving one canned ``SecurityData``."""

    def __init__(self, name: str, data: SecurityData) -> None:
        self._name = name
        self._data = data

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> Set[str]:
        return {"prices", "fundamentals", "valuation"}

    def get_security_data(self, ticker: str) -> SecurityData:
        return self._data


def _fetch(*providers: DataProvider) -> SecurityData:
    composite = CompositeProvider(providers=list(providers))
    return composite.get_security_data("TEST")


# ---------------------------------------------------------------------------
# (a) Derivation happens when BOTH inputs are present — value and warning
# ---------------------------------------------------------------------------


class TestDerivationWithBothInputs:
    def test_cap_derived_from_last_close_times_diluted_shares(self) -> None:
        data = _security(
            price_history=[_bar(27, 3.5), _bar(30, 4.0)],  # oldest-first; last close 4.0
            fundamentals=[_fundamentals(shares_diluted=10_000_000.0)],
        )
        merged = _fetch(_FakeProvider("yfinance", data))

        assert merged.valuation.market_cap == 4.0 * 10_000_000.0

    def test_provenance_warning_has_exact_text(self) -> None:
        data = _security(
            price_history=[_bar(30, 4.0)],
            fundamentals=[_fundamentals(shares_diluted=10_000_000.0)],
        )
        merged = _fetch(_FakeProvider("yfinance", data))

        assert (
            "market cap derived from last close x diluted shares "
            "(approximation: period-average diluted shares from the latest "
            "reported fundamentals, not current shares outstanding; "
            "info endpoint unavailable)" in merged.data_warnings
        )
        assert _DERIVED_MARKET_CAP_WARNING in merged.data_warnings

    def test_inputs_from_different_providers_combine(self) -> None:
        # The incident shape: yfinance supplies the (un-throttled) price
        # history, SEC EDGAR supplies shares_diluted; neither supplies a cap.
        prices_only = _security(price_history=[_bar(30, 2.5)], source="yfinance")
        shares_only = _security(
            fundamentals=[_fundamentals(shares_diluted=40_000_000.0)],
            source="sec_edgar",
        )
        merged = _fetch(
            _FakeProvider("sec_edgar", shares_only),
            _FakeProvider("yfinance", prices_only),
        )

        assert merged.valuation.market_cap == 2.5 * 40_000_000.0
        assert _DERIVED_MARKET_CAP_WARNING in merged.data_warnings


# ---------------------------------------------------------------------------
# (b) No derivation when either input is missing — never fabricate
# ---------------------------------------------------------------------------


class TestNoDerivationWhenInputMissing:
    def test_no_price_history_means_no_derivation(self) -> None:
        data = _security(fundamentals=[_fundamentals(shares_diluted=10_000_000.0)])
        merged = _fetch(_FakeProvider("yfinance", data))

        assert merged.valuation.market_cap is None
        assert _DERIVED_MARKET_CAP_WARNING not in merged.data_warnings

    def test_no_fundamentals_means_no_derivation(self) -> None:
        data = _security(price_history=[_bar(30, 4.0)])
        merged = _fetch(_FakeProvider("yfinance", data))

        assert merged.valuation.market_cap is None
        assert _DERIVED_MARKET_CAP_WARNING not in merged.data_warnings

    def test_none_shares_diluted_means_no_derivation(self) -> None:
        data = _security(
            price_history=[_bar(30, 4.0)],
            fundamentals=[_fundamentals(shares_diluted=None)],
        )
        merged = _fetch(_FakeProvider("yfinance", data))

        assert merged.valuation.market_cap is None
        assert _DERIVED_MARKET_CAP_WARNING not in merged.data_warnings

    def test_non_positive_shares_diluted_means_no_derivation(self) -> None:
        data = _security(
            price_history=[_bar(30, 4.0)],
            fundamentals=[_fundamentals(shares_diluted=0.0)],
        )
        merged = _fetch(_FakeProvider("yfinance", data))

        assert merged.valuation.market_cap is None
        assert _DERIVED_MARKET_CAP_WARNING not in merged.data_warnings


# ---------------------------------------------------------------------------
# (c) A provider-supplied market cap is NEVER overwritten
# ---------------------------------------------------------------------------


class TestProviderSuppliedCapNeverOverwritten:
    def test_supplied_cap_wins_over_derivable_inputs(self) -> None:
        # Both derivation inputs are present and would yield 4.0 * 10M = 40M,
        # but the provider supplied 123M — the supplied figure must stand.
        data = _security(
            market_cap=123_000_000.0,
            price_history=[_bar(30, 4.0)],
            fundamentals=[_fundamentals(shares_diluted=10_000_000.0)],
        )
        merged = _fetch(_FakeProvider("yfinance", data))

        assert merged.valuation.market_cap == 123_000_000.0
        assert _DERIVED_MARKET_CAP_WARNING not in merged.data_warnings

    def test_lower_priority_providers_supplied_cap_beats_derivation(self) -> None:
        # sec_edgar (highest priority) has the derivation inputs but no cap;
        # yfinance (lower priority, merged later) supplies a real cap. The
        # fallback runs only AFTER every provider has had its say, so the real
        # cap fills the blank and no derivation happens.
        derivable = _security(
            price_history=[_bar(30, 4.0)],
            fundamentals=[_fundamentals(shares_diluted=10_000_000.0)],
            source="sec_edgar",
        )
        with_cap = _security(market_cap=88_000_000.0, source="yfinance")
        merged = _fetch(
            _FakeProvider("sec_edgar", derivable),
            _FakeProvider("yfinance", with_cap),
        )

        assert merged.valuation.market_cap == 88_000_000.0
        assert _DERIVED_MARKET_CAP_WARNING not in merged.data_warnings

    def test_providers_cached_object_is_not_mutated(self) -> None:
        # The provider may cache and re-serve its SecurityData; the fallback
        # must replace (not mutate) the shared valuation snapshot.
        data = _security(
            price_history=[_bar(30, 4.0)],
            fundamentals=[_fundamentals(shares_diluted=10_000_000.0)],
        )
        _fetch(_FakeProvider("yfinance", data))

        assert data.valuation.market_cap is None
        assert data.cap_tier is None
        assert _DERIVED_MARKET_CAP_WARNING not in data.data_warnings


# ---------------------------------------------------------------------------
# (d) cap_tier is assigned from the derived cap (existing thresholds)
# ---------------------------------------------------------------------------


class TestCapTierFromDerivedCap:
    def test_derived_nano_cap_yields_nano_tier(self) -> None:
        data = _security(
            price_history=[_bar(30, 4.0)],
            fundamentals=[_fundamentals(shares_diluted=10_000_000.0)],  # 40M
        )
        merged = _fetch(_FakeProvider("yfinance", data))

        assert merged.valuation.market_cap == 40_000_000.0
        assert merged.cap_tier is CapTier.NANO

    def test_derived_micro_cap_yields_micro_tier(self) -> None:
        data = _security(
            price_history=[_bar(30, 3.0)],
            fundamentals=[_fundamentals(shares_diluted=50_000_000.0)],  # 150M
        )
        merged = _fetch(_FakeProvider("yfinance", data))

        assert merged.cap_tier is CapTier.MICRO

    def test_derived_small_cap_yields_small_tier(self) -> None:
        data = _security(
            price_history=[_bar(30, 10.0)],
            fundamentals=[_fundamentals(shares_diluted=50_000_000.0)],  # 500M
        )
        merged = _fetch(_FakeProvider("yfinance", data))

        assert merged.cap_tier is CapTier.SMALL

    def test_provider_assigned_tier_is_never_overwritten(self) -> None:
        # A provider already classified the tier; the fallback must not touch it
        # even though the derived cap would land in a different bucket.
        data = _security(
            cap_tier=CapTier.SMALL,
            price_history=[_bar(30, 4.0)],
            fundamentals=[_fundamentals(shares_diluted=10_000_000.0)],  # 40M = nano
        )
        merged = _fetch(_FakeProvider("yfinance", data))

        assert merged.cap_tier is CapTier.SMALL

    def test_no_derivation_leaves_tier_none(self) -> None:
        data = _security(price_history=[_bar(30, 4.0)])
        merged = _fetch(_FakeProvider("yfinance", data))

        assert merged.valuation.market_cap is None
        assert merged.cap_tier is None
