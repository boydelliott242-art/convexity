"""Unit tests for the batched screening-quote path (BUG A fix).

Covers, fully offline:

* :meth:`convexity.data.providers.yfinance_provider.YFinanceProvider.get_quotes`
  against a fake ``yfinance`` module — bulk price/volume parsing, the
  dollar-volume prefilter gating the per-symbol market-cap lookup, symbol-form
  translation (``BRK.B`` ↔ ``BRK-B``), missing-data omission and chunk-level
  failure containment.
* :meth:`convexity.data.aggregator.CompositeProvider.get_quotes` delegation —
  first member with a callable ``get_quotes`` wins, members that lack the method
  or raise are skipped, and an empty result falls through to the next member.
* End-to-end: ``build_universe`` screening through a composite provider actually
  enforces the cap/liquidity band (the original bug was that no provider exposed
  a batched-quote method at all, so screening silently fell back to the seed list).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd
import pytest

from convexity.core.contracts import DataProvider
from convexity.core.exceptions import DataUnavailable
from convexity.core.models import ScanParams, SecurityData
from convexity.data.aggregator import CompositeProvider
from convexity.data.providers.yfinance_provider import YFinanceProvider
from convexity.data.universe import build_universe

# ---------------------------------------------------------------------------
# Fake yfinance module
# ---------------------------------------------------------------------------


class _FakeFastInfo:
    """Mapping-style ``fast_info`` stand-in; unknown keys raise like the real one."""

    def __init__(self, data: Dict[str, float]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> float:
        if key in self._data:
            return self._data[key]
        raise KeyError(key)


class _FakeYFTicker:
    def __init__(self, fast_info: _FakeFastInfo) -> None:
        self.fast_info = fast_info


class _FakeYF:
    """Minimal fake of the ``yfinance`` module surface ``get_quotes`` touches."""

    def __init__(
        self,
        frame: Any,
        fast_infos: Optional[Dict[str, Dict[str, float]]] = None,
        *,
        download_exc: Optional[Exception] = None,
    ) -> None:
        self._frame = frame
        self._fast_infos = fast_infos or {}
        self._download_exc = download_exc
        self.download_calls: List[Dict[str, Any]] = []
        self.ticker_calls: List[str] = []

    def download(self, tickers: Any, **kwargs: Any) -> Any:
        self.download_calls.append({"tickers": list(tickers), **kwargs})
        if self._download_exc is not None:
            raise self._download_exc
        return self._frame

    def Ticker(self, symbol: str) -> _FakeYFTicker:  # noqa: N802 (mirrors yfinance API)
        self.ticker_calls.append(symbol)
        return _FakeYFTicker(_FakeFastInfo(self._fast_infos.get(symbol, {})))


def _bulk_frame(per_symbol: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
    """Build a ``yf.download(group_by='ticker')``-shaped MultiIndex frame."""
    n_rows = max(len(cols["Close"]) for cols in per_symbol.values())
    index = pd.date_range("2026-06-01", periods=n_rows, freq="D")
    data: Dict[Any, List[float]] = {}
    for symbol, cols in per_symbol.items():
        for field, values in cols.items():
            data[(symbol, field)] = values
    return pd.DataFrame(data, index=index)


def _patch_yf(monkeypatch: pytest.MonkeyPatch, fake: _FakeYF) -> None:
    monkeypatch.setattr(YFinanceProvider, "_import_yfinance", staticmethod(lambda: fake))


# ---------------------------------------------------------------------------
# YFinanceProvider.get_quotes
# ---------------------------------------------------------------------------


class TestYFinanceGetQuotes:
    def test_batched_quotes_with_prefiltered_market_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame = _bulk_frame(
            {
                # Liquid: adv = mean(close*volume) = mean(10*100k, 12*100k) = 1.1M
                "LIQ": {"Close": [10.0, 12.0], "Volume": [100_000.0, 100_000.0]},
                # Thin: adv = 2*1000 = 2k -> below the prefilter -> no cap lookup
                "THIN": {"Close": [2.0, 2.0], "Volume": [1_000.0, 1_000.0]},
                # Liquid but fast_info reports only shares -> cap = price*shares
                "SHR": {"Close": [5.0, 4.0], "Volume": [50_000.0, 50_000.0]},
                # Dead column (yf.download emits NaN for failed symbols)
                "DEAD": {"Close": [math.nan, math.nan], "Volume": [math.nan, math.nan]},
            }
        )
        fake = _FakeYF(
            frame,
            fast_infos={
                "LIQ": {"market_cap": 500_000_000.0},
                "SHR": {"shares": 10_000_000.0},
            },
        )
        _patch_yf(monkeypatch, fake)

        quotes = YFinanceProvider().get_quotes(["LIQ", "THIN", "SHR", "DEAD", "GONE"])

        # LIQ: price, adv and a direct fast_info market cap.
        assert quotes["LIQ"]["price"] == pytest.approx(12.0)
        assert quotes["LIQ"]["avg_dollar_volume"] == pytest.approx(1_100_000.0)
        assert quotes["LIQ"]["market_cap"] == pytest.approx(500_000_000.0)

        # THIN: liquidity figures present, but market_cap OMITTED (prefilter) —
        # never fabricated — and no per-symbol lookup was spent on it.
        assert quotes["THIN"]["avg_dollar_volume"] == pytest.approx(2_000.0)
        assert "market_cap" not in quotes["THIN"]
        assert "THIN" not in fake.ticker_calls

        # SHR: cap reconstructed from last price x shares outstanding.
        assert quotes["SHR"]["market_cap"] == pytest.approx(4.0 * 10_000_000.0)

        # DEAD (all-NaN) and GONE (absent from payload) are omitted entirely.
        assert "DEAD" not in quotes
        assert "GONE" not in quotes

        # The bulk call used the batched, quiet, threaded download path.
        call = fake.download_calls[0]
        assert call["group_by"] == "ticker"
        assert call["threads"] is True
        assert call["progress"] is False

    def test_symbol_form_translation_round_trips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Listing directories say BRK.B; Yahoo quotes BRK-B. Keys must come back
        # in the *requested* form so the universe screen can match them.
        frame = _bulk_frame(
            {
                "BRK-B": {"Close": [400.0], "Volume": [10_000.0]},
                "AAA": {"Close": [3.0], "Volume": [200_000.0]},
            }
        )
        fake = _FakeYF(frame, fast_infos={"BRK-B": {"market_cap": 9e11}, "AAA": {"market_cap": 6e7}})
        _patch_yf(monkeypatch, fake)

        quotes = YFinanceProvider().get_quotes(["BRK.B", "aaa"])
        assert quotes["BRK.B"]["market_cap"] == pytest.approx(9e11)
        assert "BRK-B" in fake.download_calls[0]["tickers"]
        assert quotes["AAA"]["price"] == pytest.approx(3.0)

    def test_download_failure_returns_empty_never_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeYF(frame=None, download_exc=RuntimeError("rate limited"))
        _patch_yf(monkeypatch, fake)
        assert YFinanceProvider().get_quotes(["AAA", "BBB"]) == {}

    def test_empty_and_blank_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeYF(frame=None)
        _patch_yf(monkeypatch, fake)
        provider = YFinanceProvider()
        assert provider.get_quotes([]) == {}
        assert provider.get_quotes(["", "  "]) == {}
        assert fake.download_calls == []  # nothing worth a network call

    def test_failed_fast_info_omits_cap_but_keeps_liquidity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame = _bulk_frame({"AAA": {"Close": [10.0], "Volume": [100_000.0]}})
        fake = _FakeYF(frame, fast_infos={})  # fast_info has no usable keys
        _patch_yf(monkeypatch, fake)

        quotes = YFinanceProvider().get_quotes(["AAA"])
        assert quotes["AAA"]["avg_dollar_volume"] == pytest.approx(1_000_000.0)
        assert "market_cap" not in quotes["AAA"]


# ---------------------------------------------------------------------------
# CompositeProvider.get_quotes delegation
# ---------------------------------------------------------------------------


class _QuoteMember(DataProvider):
    """Fake member provider with a configurable ``get_quotes``."""

    def __init__(
        self,
        name: str,
        quotes: Optional[Dict[str, Dict[str, float]]] = None,
        exc: Optional[Exception] = None,
    ) -> None:
        self._name = name
        self._quotes = quotes if quotes is not None else {}
        self._exc = exc
        self.calls: List[List[str]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> Set[str]:
        return {"prices", "universe-screen"}

    def get_security_data(self, ticker: str) -> SecurityData:
        raise DataUnavailable("fake member serves quotes only", ticker=ticker)

    def get_quotes(self, tickers: Sequence[str]) -> Dict[str, Dict[str, float]]:
        self.calls.append([str(t) for t in tickers])
        if self._exc is not None:
            raise self._exc
        return dict(self._quotes)


class _NoQuotesMember(DataProvider):
    """Fake member provider with no ``get_quotes`` at all."""

    @property
    def name(self) -> str:
        return "no-quotes"

    @property
    def capabilities(self) -> Set[str]:
        return {"fundamentals"}

    def get_security_data(self, ticker: str) -> SecurityData:
        raise DataUnavailable("fake member has no data", ticker=ticker)


class TestCompositeGetQuotes:
    def test_delegates_to_first_member_with_get_quotes(self) -> None:
        first = _QuoteMember("first", {"aaa": {"market_cap": 1e8, "avg_dollar_volume": 5e5}})
        second = _QuoteMember("second", {"AAA": {"market_cap": 9e9}})
        composite = CompositeProvider(providers=[_NoQuotesMember(), first, second])

        quotes = composite.get_quotes(["AAA"])
        assert quotes == {"AAA": {"market_cap": 1e8, "avg_dollar_volume": 5e5}}  # keys uppercased
        assert first.calls == [["AAA"]]
        assert second.calls == []  # first capable member won; second untouched

    def test_raising_member_is_skipped(self) -> None:
        broken = _QuoteMember("broken", exc=RuntimeError("boom"))
        healthy = _QuoteMember("healthy", {"BBB": {"market_cap": 2e8, "avg_dollar_volume": 3e5}})
        composite = CompositeProvider(providers=[broken, healthy])

        quotes = composite.get_quotes(["BBB"])
        assert quotes["BBB"]["market_cap"] == pytest.approx(2e8)
        assert broken.calls and healthy.calls  # both were tried, in order

    def test_empty_result_falls_through_to_next_member(self) -> None:
        empty = _QuoteMember("empty", {})
        healthy = _QuoteMember("healthy", {"CCC": {"market_cap": 1e8}})
        composite = CompositeProvider(providers=[empty, healthy])
        assert composite.get_quotes(["CCC"]) == {"CCC": {"market_cap": 1e8}}

    def test_no_capable_member_returns_empty(self) -> None:
        composite = CompositeProvider(providers=[_NoQuotesMember()])
        assert composite.get_quotes(["AAA"]) == {}

    def test_build_universe_screens_through_composite(self) -> None:
        """End-to-end: the cap/liquidity band is actually enforced via the composite."""
        member = _QuoteMember(
            "screener",
            {
                "INBAND": {"market_cap": 5e8, "avg_dollar_volume": 1e6},
                "TOOBIG": {"market_cap": 27e9, "avg_dollar_volume": 5e7},  # the MLI case
                "TOOSMALL": {"market_cap": 1e7, "avg_dollar_volume": 1e6},
                "ILLIQUID": {"market_cap": 5e8, "avg_dollar_volume": 1e4},
                "NOCAP": {"avg_dollar_volume": 1e6},  # unknown cap -> excluded, not assumed
            },
        )
        composite = CompositeProvider(providers=[_NoQuotesMember(), member])

        params = ScanParams()  # defaults: cap in [50M, 2B], adv >= 200k
        eligible = build_universe(
            params,
            composite,
            candidates=["INBAND", "TOOBIG", "TOOSMALL", "ILLIQUID", "NOCAP", "NOQUOTE"],
        )
        assert eligible == ["INBAND"]
