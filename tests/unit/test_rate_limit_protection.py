"""Rate-limit protection on the production universe-screening path.

Regression suite for the silent-universe-collapse defect: the universe screen
batched quotes at 100 symbols while the yfinance provider chunked at 200, so the
provider's inter-chunk sleep never engaged; the per-symbol ``fast_info`` market
cap lookups were fired back-to-back; and once Yahoo answered 429, every affected
name lost its cap, was counted ``cap_unknown`` and was silently excluded with
only an INFO log — no :class:`ScanResult` note.

These tests pin the three-part fix, fully offline:

* :func:`convexity.data.universe.build_universe` pauses between its own quote
  batches (provider-agnostic pacing);
* :class:`~convexity.data.providers.yfinance_provider.YFinanceProvider` paces
  its per-symbol ``fast_info`` lookups;
* the screen's exclusion counters travel out via ``stats`` and the pipeline
  surfaces unverified exclusions (and the seed fallback) as scan notes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd
import pytest

from convexity.core.contracts import DataProvider
from convexity.core.exceptions import DataUnavailable
from convexity.core.models import ScanParams, SecurityData
from convexity.data import universe as universe_mod
from convexity.data.providers.yfinance_provider import YFinanceProvider
from convexity.pipeline import ScanPipeline
from tests.conftest import FakeProvider

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _BatchRecordingProvider(DataProvider):
    """Quote provider that records every batch it is asked to serve."""

    def __init__(self, quotes: Dict[str, Dict[str, float]]) -> None:
        self._quotes = dict(quotes)
        self.batches: List[List[str]] = []

    @property
    def name(self) -> str:
        return "batch-recorder"

    @property
    def capabilities(self) -> Set[str]:
        return {"prices", "universe-screen"}

    def get_security_data(self, ticker: str) -> SecurityData:
        raise DataUnavailable("quotes-only fake", ticker=ticker)

    def get_quotes(self, tickers: Sequence[str]) -> Dict[str, Dict[str, float]]:
        batch = [str(t).upper() for t in tickers]
        self.batches.append(batch)
        return {t: self._quotes[t] for t in batch if t in self._quotes}


class _SleepRecorder:
    """Stand-in for the ``time`` module recording every requested pause."""

    def __init__(self) -> None:
        self.calls: List[float] = []

    def sleep(self, seconds: float) -> None:
        self.calls.append(seconds)


class _FakeClock:
    """Deterministic ``time`` module: ``sleep`` advances ``monotonic``."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start
        self.sleeps: List[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _FakeFastInfo:
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
    """Minimal fake of the ``yfinance`` surface ``get_quotes`` touches."""

    def __init__(
        self,
        frame: Any,
        fast_infos: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self._frame = frame
        self._fast_infos = fast_infos or {}
        self.ticker_calls: List[str] = []

    def download(self, tickers: Any, **kwargs: Any) -> Any:
        return self._frame

    def Ticker(self, symbol: str) -> _FakeYFTicker:  # noqa: N802 (mirrors yfinance)
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


# ---------------------------------------------------------------------------
# (a) build_universe paces its own quote batches
# ---------------------------------------------------------------------------


class TestUniverseBatchPacing:
    def test_pause_between_batches_but_not_before_the_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _SleepRecorder()
        monkeypatch.setattr(universe_mod, "time", recorder)

        quotes = {
            f"T{i:03d}": {"market_cap": 1.0e8, "avg_dollar_volume": 5.0e5}
            for i in range(25)
        }
        provider = _BatchRecordingProvider(quotes)

        eligible = universe_mod.build_universe(
            ScanParams(), provider, candidates=sorted(quotes), batch_size=10
        )

        assert len(provider.batches) == 3  # 10 + 10 + 5
        # One pause per batch boundary — the throttle now provably engages on
        # the exact multi-batch shape a full-listing screen produces.
        assert recorder.calls == [universe_mod._QUOTE_BATCH_PAUSE_S] * 2
        assert len(eligible) == 25

    def test_single_batch_does_not_pause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _SleepRecorder()
        monkeypatch.setattr(universe_mod, "time", recorder)
        provider = _BatchRecordingProvider(
            {"AAA": {"market_cap": 1.0e8, "avg_dollar_volume": 5.0e5}}
        )
        universe_mod.build_universe(ScanParams(), provider, candidates=["AAA"])
        assert recorder.calls == []


# ---------------------------------------------------------------------------
# (b) the screen reports its exclusion counters via ``stats``
# ---------------------------------------------------------------------------


class TestScreenStats:
    def test_build_universe_fills_stats(self) -> None:
        provider = _BatchRecordingProvider(
            {
                "INBAND": {"market_cap": 5.0e8, "avg_dollar_volume": 1.0e6},
                "NOCAP": {"avg_dollar_volume": 1.0e6},  # cap unverifiable
                "NOLIQ": {"market_cap": 5.0e8},  # liquidity unverifiable
                "TOOBIG": {"market_cap": 3.0e10, "avg_dollar_volume": 1.0e6},
                "THIN": {"market_cap": 5.0e8, "avg_dollar_volume": 1.0e3},
            }
        )
        stats: Dict[str, int] = {}
        eligible = universe_mod.build_universe(
            ScanParams(),
            provider,
            candidates=["INBAND", "NOCAP", "NOLIQ", "TOOBIG", "THIN", "GONE"],
            stats=stats,
        )

        assert eligible == ["INBAND"]
        assert stats["candidates"] == 6
        assert stats["eligible"] == 1
        assert stats["quoted"] == 5
        assert stats["no_quote"] == 1  # GONE
        assert stats["cap_unknown"] == 1  # NOCAP
        assert stats["liquidity_unknown"] == 1  # NOLIQ
        assert stats["cap_out_of_band"] == 1  # TOOBIG
        assert stats["illiquid"] == 1  # THIN

    def test_build_universe_or_seed_marks_seed_fallback(self) -> None:
        stats: Dict[str, int] = {}
        tickers = universe_mod.build_universe_or_seed(
            ScanParams(universe_limit=5), price_provider=None, stats=stats
        )
        assert stats["used_seed_fallback"] == 1
        assert len(tickers) == 5

    def test_build_universe_or_seed_marks_live_screen(self) -> None:
        provider = _BatchRecordingProvider(
            {"AAA": {"market_cap": 1.0e8, "avg_dollar_volume": 5.0e5}}
        )
        stats: Dict[str, int] = {}

        # Route the enumeration to a known candidate list without the network.
        tickers = universe_mod.build_universe(
            ScanParams(), provider, candidates=["AAA"], stats=stats
        )
        assert tickers == ["AAA"]
        # The wrapper marks the fallback flag; exercise it via the wrapper too.
        stats2: Dict[str, int] = {}
        original = universe_mod.fetch_listed_symbols
        try:
            universe_mod.fetch_listed_symbols = (  # type: ignore[assignment]
                lambda **_kw: ["AAA"]
            )
            wrapped = universe_mod.build_universe_or_seed(
                ScanParams(), provider, stats=stats2
            )
        finally:
            universe_mod.fetch_listed_symbols = original  # type: ignore[assignment]
        assert wrapped == ["AAA"]
        assert stats2["used_seed_fallback"] == 0


# ---------------------------------------------------------------------------
# (c) YFinanceProvider paces per-symbol fast_info lookups
# ---------------------------------------------------------------------------


class TestFastInfoThrottle:
    def test_consecutive_fast_info_lookups_are_paced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame = _bulk_frame(
            {
                "AAA": {"Close": [10.0, 10.0], "Volume": [100_000.0, 100_000.0]},
                "BBB": {"Close": [20.0, 20.0], "Volume": [100_000.0, 100_000.0]},
            }
        )
        fake = _FakeYF(
            frame,
            fast_infos={
                "AAA": {"market_cap": 1.0e8},
                "BBB": {"market_cap": 2.0e8},
            },
        )
        monkeypatch.setattr(
            YFinanceProvider, "_import_yfinance", staticmethod(lambda: fake)
        )
        clock = _FakeClock()
        monkeypatch.setattr(
            "convexity.data.providers.yfinance_provider.time", clock
        )

        quotes = YFinanceProvider().get_quotes(["AAA", "BBB"])

        assert quotes["AAA"]["market_cap"] == pytest.approx(1.0e8)
        assert quotes["BBB"]["market_cap"] == pytest.approx(2.0e8)
        assert len(fake.ticker_calls) == 2
        # The first lookup runs immediately; the second waits the min interval.
        assert clock.sleeps == [
            pytest.approx(YFinanceProvider._FAST_INFO_MIN_INTERVAL_S)
        ]

    def test_prefiltered_symbols_cost_no_throttle_wait(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A symbol below the dollar-volume prefilter never triggers a fast_info
        # lookup, so it also never pays (or causes) a pacing sleep.
        frame = _bulk_frame(
            {"THIN": {"Close": [2.0, 2.0], "Volume": [1_000.0, 1_000.0]}}
        )
        fake = _FakeYF(frame, fast_infos={})
        monkeypatch.setattr(
            YFinanceProvider, "_import_yfinance", staticmethod(lambda: fake)
        )
        clock = _FakeClock()
        monkeypatch.setattr(
            "convexity.data.providers.yfinance_provider.time", clock
        )

        quotes = YFinanceProvider().get_quotes(["THIN"])
        assert "market_cap" not in quotes["THIN"]
        assert fake.ticker_calls == []
        assert clock.sleeps == []


# ---------------------------------------------------------------------------
# (d) the pipeline surfaces unverified exclusions / seed fallback as notes
# ---------------------------------------------------------------------------


class TestScanNotesSurfaceScreenShrinkage:
    def _scan_with_stats(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stats_payload: Dict[str, int],
        tickers: List[str],
    ) -> List[str]:
        fake = FakeProvider()

        def fake_build(
            params: ScanParams,
            provider: object = None,
            *,
            user_agent: Optional[str] = None,
            timeout: Optional[float] = None,
            stats: Optional[Dict[str, int]] = None,
            **_kw: object,
        ) -> List[str]:
            if stats is not None:
                stats.update(stats_payload)
            return list(tickers)

        monkeypatch.setattr(universe_mod, "build_universe_or_seed", fake_build)
        result = ScanPipeline(provider=fake).scan(ScanParams(top_n=1))
        return result.notes

    def test_unverified_exclusions_become_a_scan_note(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        notes = self._scan_with_stats(
            monkeypatch,
            {
                "candidates": 5_000,
                "eligible": 2,
                "quoted": 2_600,
                "no_quote": 2_400,
                "cap_unknown": 2_500,
                "liquidity_unknown": 98,
                "cap_out_of_band": 0,
                "illiquid": 0,
                "used_seed_fallback": 0,
            },
            ["STRONGCO", "THINCO"],
        )
        note = next(n for n in notes if "could not be verified" in n)
        assert "4998 candidate(s)" in note
        assert "no quote: 2400" in note
        assert "unknown market cap: 2500" in note
        assert "unknown liquidity: 98" in note
        assert "rate limiting" in note  # names the likely cause honestly

    def test_seed_fallback_becomes_a_scan_note(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        notes = self._scan_with_stats(
            monkeypatch, {"used_seed_fallback": 1}, ["STRONGCO"]
        )
        assert any("seed list" in n for n in notes)

    def test_fully_verified_screen_adds_no_shrinkage_note(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        notes = self._scan_with_stats(
            monkeypatch,
            {
                "candidates": 2,
                "eligible": 2,
                "quoted": 2,
                "no_quote": 0,
                "cap_unknown": 0,
                "liquidity_unknown": 0,
                "cap_out_of_band": 0,
                "illiquid": 0,
                "used_seed_fallback": 0,
            },
            ["STRONGCO", "THINCO"],
        )
        assert not any("could not be verified" in n for n in notes)
        assert not any("seed list" in n for n in notes)
