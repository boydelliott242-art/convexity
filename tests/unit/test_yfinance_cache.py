"""Offline tests for the yfinance provider's freshness-bounded caching.

Guards the two halves of the rate-limit incident fix (2026-07):

* :meth:`YFinanceProvider.get_security_data` memoises COMPLETE fetches on
  :mod:`convexity.data.cache` — a second call within the TTL is served with no
  network at all, so a healthy morning run's data survives an afternoon 429.
* HONESTY RULE: a *partial* fetch (Yahoo's ``info`` endpoint rate-limited, so
  the market cap is unknown) is returned to the caller once — warnings intact —
  but is **never** written to the cache; the next call refetches live instead
  of freezing the gap for the whole TTL.
* Screening quotes (:meth:`YFinanceProvider.get_quotes`) are cached per chunk
  under a short (~4h) TTL keyed by a hash of the sorted chunk symbols, so a
  re-run within the window skips re-screening and the coverage a partially
  rate-limited screen did achieve persists chunk-by-chunk.
* The same HONESTY RULE applies per chunk: a chunk whose liquid
  (prefilter-clearing) symbols are missing their market caps — the bulk chart
  download succeeded but the per-symbol ``fast_info`` cap endpoint was
  throttled, the incident's exact signature — is returned once but never
  cached, so a later healthy run re-screens it instead of serving capless
  quotes (which the universe screen would exclude as ``cap_unknown``) for 4h.
* Every cache failure (read, write, corrupt payload) degrades to a live fetch;
  the cache can slow the provider down, never break it.

Everything here monkeypatches the network layer (``_import_yfinance``) — no
test touches the real Yahoo endpoints or the real on-disk cache location.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pytest

from convexity.core.config import Settings
from convexity.core.models import SecurityData
from convexity.data.cache import Cache, make_key
from convexity.data.providers.yfinance_provider import (
    _QUOTES_CACHE_TTL_SECONDS,
    YFinanceProvider,
)

# ---------------------------------------------------------------------------
# Fakes: the minimal yfinance surface get_security_data / get_quotes touch
# ---------------------------------------------------------------------------


def _price_frame(days: int = 5) -> pd.DataFrame:
    """A small, well-formed daily OHLCV frame for ``Ticker.history``."""
    index = pd.date_range("2026-06-01", periods=days, freq="D")
    closes = [10.0 + i for i in range(days)]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 0.5 for c in closes],
            "Low": [c - 0.5 for c in closes],
            "Close": closes,
            "Adj Close": closes,
            "Volume": [100_000.0] * days,
        },
        index=index,
    )


class _FakeTicker:
    """``yf.Ticker`` stand-in: an ``info`` payload (or failure) plus history."""

    def __init__(self, info: Any, hist: pd.DataFrame) -> None:
        self._info = info
        self._hist = hist
        self.news: List[Any] = []

    @property
    def info(self) -> Dict[str, Any]:
        if isinstance(self._info, Exception):
            raise self._info
        return self._info

    def history(self, **_kwargs: Any) -> pd.DataFrame:
        return self._hist


class _FakeYF:
    """Fake yfinance module for the ``get_security_data`` path, counting calls."""

    def __init__(self, info: Any, hist: Optional[pd.DataFrame] = None) -> None:
        self._info = info
        self._hist = hist if hist is not None else _price_frame()
        self.ticker_calls: List[str] = []

    def Ticker(self, symbol: str) -> _FakeTicker:  # noqa: N802 (mirrors yfinance API)
        self.ticker_calls.append(symbol)
        return _FakeTicker(self._info, self._hist)


class _FakeFastInfo:
    def __init__(self, data: Dict[str, float]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> float:
        if key in self._data:
            return self._data[key]
        raise KeyError(key)


class _FakeQuoteTicker:
    def __init__(self, fast_info: _FakeFastInfo) -> None:
        self.fast_info = fast_info


class _FakeQuotesYF:
    """Fake yfinance module for the ``get_quotes`` path.

    ``download`` serves one pre-built frame per requested chunk (keyed by the
    tuple of requested Yahoo symbols) and can be told to fail for specific
    symbols, which lets a test model a partially rate-limited screen.
    """

    def __init__(
        self,
        frames: Dict[Tuple[str, ...], pd.DataFrame],
        fast_infos: Optional[Dict[str, Dict[str, float]]] = None,
        fail_symbols: Optional[Set[str]] = None,
    ) -> None:
        self._frames = frames
        self._fast_infos = fast_infos or {}
        self._fail_symbols = fail_symbols or set()
        self.download_calls: List[List[str]] = []

    def download(self, tickers: Any, **_kwargs: Any) -> pd.DataFrame:
        requested = list(tickers)
        self.download_calls.append(requested)
        if any(sym in self._fail_symbols for sym in requested):
            raise RuntimeError("429 rate limited")
        return self._frames[tuple(requested)]

    def Ticker(self, symbol: str) -> _FakeQuoteTicker:  # noqa: N802
        return _FakeQuoteTicker(_FakeFastInfo(self._fast_infos.get(symbol, {})))


def _bulk_frame(per_symbol: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
    """Build a ``yf.download(group_by='ticker')``-shaped MultiIndex frame."""
    n_rows = max(len(cols["Close"]) for cols in per_symbol.values())
    index = pd.date_range("2026-06-01", periods=n_rows, freq="D")
    data: Dict[Any, List[float]] = {}
    for symbol, cols in per_symbol.items():
        for field, values in cols.items():
            data[(symbol, field)] = values
    return pd.DataFrame(data, index=index)


def _flat_frame(closes: List[float], volumes: List[float]) -> pd.DataFrame:
    """A single-symbol (flat-column) ``yf.download`` frame."""
    index = pd.date_range("2026-06-01", periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes, "Volume": volumes}, index=index)


def _aaa_bbb_frame() -> pd.DataFrame:
    """A liquid AAA + BBB bulk frame reused across the chunk-cache tests."""
    return _bulk_frame(
        {
            "AAA": {"Close": [10.0], "Volume": [100_000.0]},
            "BBB": {"Close": [2.0], "Volume": [50_000.0]},
        }
    )


def _patch_yf(monkeypatch: pytest.MonkeyPatch, fake: Any) -> None:
    monkeypatch.setattr(YFinanceProvider, "_import_yfinance", staticmethod(lambda: fake))


def _tmp_cache(tmp_path: Any) -> Cache:
    """An isolated Cache rooted in the test's temporary directory."""
    return Cache(Settings(data_dir=str(tmp_path / "cache")))


class _RecordingCache(Cache):
    """A real Cache that also records every ``set`` call's key and TTL."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.set_calls: List[Tuple[str, Optional[int]]] = []

    def set(self, key: str, value: Any, *, ttl: Optional[int] = None) -> None:
        self.set_calls.append((key, ttl))
        super().set(key, value, ttl=ttl)


class _ExplodingCache:
    """A cache whose every read/write raises — the degraded-disk worst case."""

    def __init__(self) -> None:
        self.get_calls = 0
        self.set_calls = 0

    def get_data(self, *_args: Any, **_kwargs: Any) -> Any:
        self.get_calls += 1
        raise RuntimeError("disk exploded")

    def set_data(self, *_args: Any, **_kwargs: Any) -> None:
        self.set_calls += 1
        raise RuntimeError("disk exploded")


_COMPLETE_INFO: Dict[str, Any] = {
    "longName": "Acme Micro Corp",
    "marketCap": 90_000_000,
    "sector": "Technology",
    "industry": "Software",
    "fullExchangeName": "NasdaqCM",
    "currency": "USD",
}


# ---------------------------------------------------------------------------
# (a) get_security_data: complete fetches are cached and served with no network
# ---------------------------------------------------------------------------


class TestSecurityDataCaching:
    def test_second_call_served_from_cache_without_network(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        fake = _FakeYF(info=dict(_COMPLETE_INFO))
        _patch_yf(monkeypatch, fake)
        provider = YFinanceProvider(cache=_tmp_cache(tmp_path))

        first = provider.get_security_data("ACME")
        assert fake.ticker_calls == ["ACME"]
        assert first.valuation.market_cap == pytest.approx(90_000_000.0)
        # Genuinely-absent data (no fundamentals/news for this thin fake) does
        # not block caching — only rate-limit-shaped partiality does.
        assert any("no fundamentals" in w for w in first.data_warnings)

        second = provider.get_security_data("ACME")
        assert fake.ticker_calls == ["ACME"]  # no second network fetch
        # Lossless round-trip: the cached copy is the same data, warnings included.
        assert second.model_dump(mode="json") == first.model_dump(mode="json")
        assert isinstance(second, SecurityData)

    def test_security_data_cached_under_default_ttl(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        fake = _FakeYF(info=dict(_COMPLETE_INFO))
        _patch_yf(monkeypatch, fake)
        cache = _RecordingCache(Settings(data_dir=str(tmp_path / "cache")))
        YFinanceProvider(cache=cache).get_security_data("ACME")

        expected_key = make_key("yfinance", "ACME", "security_data")
        assert (expected_key, None) in cache.set_calls  # None -> default TTL

    # -- (b) partial (rate-limited) fetches are returned but never cached -----

    def test_rate_limited_info_is_returned_but_not_cached(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        fake = _FakeYF(info=RuntimeError("429 Too Many Requests"))
        _patch_yf(monkeypatch, fake)
        cache = _tmp_cache(tmp_path)
        provider = YFinanceProvider(cache=cache)

        first = provider.get_security_data("ACME")
        # Partial data still flows to the caller, honestly labeled.
        assert first.ticker == "ACME"
        assert first.valuation.market_cap is None
        assert any("failed to fetch company info" in w for w in first.data_warnings)
        assert len(first.price_history) == 5  # the throttle hit info, not prices

        # ...but nothing was written to the cache...
        assert cache.get_data("yfinance", "ACME", "security_data") is None

        # ...so a later (healthy) run refetches instead of serving the gap.
        second = provider.get_security_data("ACME")
        assert fake.ticker_calls == ["ACME", "ACME"]
        assert second.valuation.market_cap is None  # fake is still throttled

    def test_missing_market_cap_is_not_cached(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        info = dict(_COMPLETE_INFO)
        del info["marketCap"]  # throttled endpoints often return a gutted payload
        fake = _FakeYF(info=info)
        _patch_yf(monkeypatch, fake)
        cache = _tmp_cache(tmp_path)
        provider = YFinanceProvider(cache=cache)

        provider.get_security_data("ACME")
        assert cache.get_data("yfinance", "ACME", "security_data") is None
        provider.get_security_data("ACME")
        assert len(fake.ticker_calls) == 2  # refetched, not served stale

    def test_empty_info_payload_is_not_cached(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        fake = _FakeYF(info={})
        _patch_yf(monkeypatch, fake)
        cache = _tmp_cache(tmp_path)
        provider = YFinanceProvider(cache=cache)

        first = provider.get_security_data("ACME")
        assert any("returned no company info" in w for w in first.data_warnings)
        assert cache.get_data("yfinance", "ACME", "security_data") is None

    # -- (c) cache failures degrade to a live fetch ---------------------------

    def test_cache_read_and_write_errors_fall_through_to_live_fetch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeYF(info=dict(_COMPLETE_INFO))
        _patch_yf(monkeypatch, fake)
        exploding = _ExplodingCache()
        provider = YFinanceProvider(cache=exploding)  # type: ignore[arg-type]

        sec = provider.get_security_data("ACME")
        assert sec.valuation.market_cap == pytest.approx(90_000_000.0)
        assert exploding.get_calls >= 1  # the read was attempted...
        assert exploding.set_calls >= 1  # ...and so was the write; both failed

        # With the cache permanently broken, every call is a live fetch.
        provider.get_security_data("ACME")
        assert fake.ticker_calls == ["ACME", "ACME"]

    def test_corrupt_cached_payload_forces_live_refetch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        fake = _FakeYF(info=dict(_COMPLETE_INFO))
        _patch_yf(monkeypatch, fake)
        cache = _tmp_cache(tmp_path)
        # Poison the exact slot with a payload that cannot validate.
        cache.set_data("yfinance", "ACME", "security_data", {"ticker": "ACME"})

        sec = YFinanceProvider(cache=cache).get_security_data("ACME")
        assert fake.ticker_calls == ["ACME"]  # corrupt hit -> live fetch
        assert sec.valuation.market_cap == pytest.approx(90_000_000.0)


# ---------------------------------------------------------------------------
# get_quotes: per-chunk short-TTL caching of screening batches
# ---------------------------------------------------------------------------


class TestQuotesChunkCaching:
    def test_second_screen_within_window_skips_the_network(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        frame = _bulk_frame(
            {
                "AAA": {"Close": [10.0, 12.0], "Volume": [100_000.0, 100_000.0]},
                "BBB": {"Close": [3.0, 3.0], "Volume": [200_000.0, 200_000.0]},
            }
        )
        fake = _FakeQuotesYF(
            frames={("AAA", "BBB"): frame},
            fast_infos={"AAA": {"market_cap": 5e8}, "BBB": {"market_cap": 6e7}},
        )
        _patch_yf(monkeypatch, fake)
        cache = _tmp_cache(tmp_path)
        provider = YFinanceProvider(cache=cache)

        first = provider.get_quotes(["AAA", "BBB"])
        assert len(fake.download_calls) == 1
        assert first["AAA"]["market_cap"] == pytest.approx(5e8)

        second = provider.get_quotes(["AAA", "BBB"])
        assert len(fake.download_calls) == 1  # served from cache, no re-screen
        assert second == first

    def test_quote_chunks_are_cached_under_the_short_ttl(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        fake = _FakeQuotesYF(
            frames={("AAA", "BBB"): _aaa_bbb_frame()},
            fast_infos={"AAA": {"market_cap": 1e8}, "BBB": {"market_cap": 9e7}},
        )
        _patch_yf(monkeypatch, fake)
        cache = _RecordingCache(Settings(data_dir=str(tmp_path / "cache")))

        YFinanceProvider(cache=cache).get_quotes(["AAA", "BBB"])

        quote_sets = [(key, ttl) for key, ttl in cache.set_calls if key.endswith("|quotes")]
        assert quote_sets, "the screening chunk was not cached at all"
        assert all(ttl == _QUOTES_CACHE_TTL_SECONDS for _, ttl in quote_sets)
        # A short window, not the (12h) default: ~4 hours.
        assert _QUOTES_CACHE_TTL_SECONDS == 4 * 60 * 60

    def test_partial_coverage_persists_per_chunk(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Chunk 1 succeeds, chunk 2 is rate-limited; a re-run only refetches chunk 2."""
        monkeypatch.setattr(YFinanceProvider, "_QUOTE_CHUNK_SIZE", 1)
        monkeypatch.setattr(YFinanceProvider, "_QUOTE_CHUNK_SLEEP_SECONDS", 0.0)
        cache = _tmp_cache(tmp_path)

        aaa_frame = _flat_frame([10.0, 10.0], [100_000.0, 100_000.0])
        bbb_frame = _flat_frame([4.0, 4.0], [150_000.0, 150_000.0])
        fast_infos = {"AAA": {"market_cap": 2e8}, "BBB": {"market_cap": 3e8}}

        # Run 1: AAA's chunk succeeds, BBB's chunk is throttled and yields nothing.
        run1 = _FakeQuotesYF(
            frames={("AAA",): aaa_frame, ("BBB",): bbb_frame},
            fast_infos=fast_infos,
            fail_symbols={"BBB"},
        )
        _patch_yf(monkeypatch, run1)
        quotes1 = YFinanceProvider(cache=cache).get_quotes(["AAA", "BBB"])
        assert "AAA" in quotes1 and "BBB" not in quotes1
        assert run1.download_calls == [["AAA"], ["BBB"]]

        # Run 2 (healthy, same cache): AAA is served from the cache — only the
        # chunk that failed is refetched. Partial coverage persisted.
        run2 = _FakeQuotesYF(
            frames={("AAA",): aaa_frame, ("BBB",): bbb_frame},
            fast_infos=fast_infos,
        )
        _patch_yf(monkeypatch, run2)
        quotes2 = YFinanceProvider(cache=cache).get_quotes(["AAA", "BBB"])
        assert run2.download_calls == [["BBB"]]
        assert quotes2["AAA"]["market_cap"] == pytest.approx(2e8)
        assert quotes2["BBB"]["market_cap"] == pytest.approx(3e8)

    def test_failed_chunk_is_not_cached_and_is_retried(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(YFinanceProvider, "_QUOTE_CHUNK_SLEEP_SECONDS", 0.0)
        cache = _tmp_cache(tmp_path)
        fake = _FakeQuotesYF(frames={}, fail_symbols={"AAA"})
        _patch_yf(monkeypatch, fake)
        provider = YFinanceProvider(cache=cache)

        assert provider.get_quotes(["AAA"]) == {}
        assert provider.get_quotes(["AAA"]) == {}
        # An empty (failed) chunk must never be served from cache: both calls hit
        # the network so a later healthy run can recover the coverage.
        assert len(fake.download_calls) == 2

    def test_rate_limited_market_caps_chunk_is_returned_but_not_cached(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """The incident signature: bulk download healthy, ``fast_info`` throttled.

        The chunk's quotes come back liquid but capless. They must flow to the
        caller (honest partial data), but the chunk must NOT be cached — a
        healthy run within the 4h window has to re-screen it live, or every
        one of those names would be excluded as ``cap_unknown`` until expiry.
        """
        monkeypatch.setattr(YFinanceProvider, "_QUOTE_CHUNK_SLEEP_SECONDS", 0.0)
        cache = _tmp_cache(tmp_path)
        frame = _aaa_bbb_frame()  # both symbols clear the 25k ADV prefilter

        # Run 1: fast_info answers for neither symbol (throttled) -> no caps.
        throttled = _FakeQuotesYF(frames={("AAA", "BBB"): frame}, fast_infos={})
        _patch_yf(monkeypatch, throttled)
        quotes1 = YFinanceProvider(cache=cache).get_quotes(["AAA", "BBB"])
        assert quotes1["AAA"]["price"] == pytest.approx(10.0)
        assert "market_cap" not in quotes1["AAA"]  # partial data, honestly capless

        # Run 2 (healthy, same cache, within the TTL): the chunk is re-screened
        # live — the capless run 1 result was never frozen into the cache.
        healthy = _FakeQuotesYF(
            frames={("AAA", "BBB"): frame},
            fast_infos={"AAA": {"market_cap": 1e8}, "BBB": {"market_cap": 6e7}},
        )
        _patch_yf(monkeypatch, healthy)
        quotes2 = YFinanceProvider(cache=cache).get_quotes(["AAA", "BBB"])
        assert healthy.download_calls, "capless chunk was served from cache (poisoned)"
        assert quotes2["AAA"]["market_cap"] == pytest.approx(1e8)
        assert quotes2["BBB"]["market_cap"] == pytest.approx(6e7)

        # Run 3: the healthy (complete) chunk WAS cached — no third screen.
        _patch_yf(monkeypatch, healthy)
        quotes3 = YFinanceProvider(cache=cache).get_quotes(["AAA", "BBB"])
        assert len(healthy.download_calls) == 1
        assert quotes3 == quotes2

    def test_illiquid_capless_quotes_do_not_block_chunk_caching(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Symbols below the ADV prefilter carry no cap BY DESIGN — such a chunk
        is complete and must still be cached (no cap lookup was ever attempted,
        so a missing cap there is policy, not rate limiting)."""
        monkeypatch.setattr(YFinanceProvider, "_QUOTE_CHUNK_SLEEP_SECONDS", 0.0)
        cache = _tmp_cache(tmp_path)
        frame = _bulk_frame(
            {
                # AAA: ADV 1,000,000 (clears the 25k prefilter, cap fetched).
                "AAA": {"Close": [10.0], "Volume": [100_000.0]},
                # CCC: ADV 1,000 (below the prefilter, capless by design).
                "CCC": {"Close": [1.0], "Volume": [1_000.0]},
            }
        )
        fake = _FakeQuotesYF(
            frames={("AAA", "CCC"): frame}, fast_infos={"AAA": {"market_cap": 1e8}}
        )
        _patch_yf(monkeypatch, fake)
        provider = YFinanceProvider(cache=cache)

        first = provider.get_quotes(["AAA", "CCC"])
        assert "market_cap" not in first["CCC"]
        second = provider.get_quotes(["AAA", "CCC"])
        assert len(fake.download_calls) == 1  # served from cache: chunk complete
        assert second == first

    def test_quote_cache_errors_fall_through_to_live_fetch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeQuotesYF(
            frames={("AAA", "BBB"): _aaa_bbb_frame()},
            fast_infos={"AAA": {"market_cap": 1e8}, "BBB": {"market_cap": 6e7}},
        )
        _patch_yf(monkeypatch, fake)
        exploding = _ExplodingCache()

        quotes = YFinanceProvider(cache=exploding).get_quotes(["AAA", "BBB"])  # type: ignore[arg-type]
        assert quotes["AAA"]["market_cap"] == pytest.approx(1e8)
        assert exploding.get_calls >= 1 and exploding.set_calls >= 1

    def test_corrupt_cached_chunk_forces_live_refetch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        fake = _FakeQuotesYF(
            frames={("AAA", "BBB"): _aaa_bbb_frame()}, fast_infos={"AAA": {"market_cap": 1e8}}
        )
        _patch_yf(monkeypatch, fake)
        cache = _tmp_cache(tmp_path)
        # Poison the chunk slot with a non-numeric figure.
        chunk_hash = YFinanceProvider._quotes_chunk_hash(["AAA", "BBB"])
        cache.set_data("yfinance", chunk_hash, "quotes", {"AAA": {"market_cap": "lots"}})

        quotes = YFinanceProvider(cache=cache).get_quotes(["AAA", "BBB"])
        assert len(fake.download_calls) == 1  # corrupt hit -> live re-screen
        assert quotes["AAA"]["market_cap"] == pytest.approx(1e8)
