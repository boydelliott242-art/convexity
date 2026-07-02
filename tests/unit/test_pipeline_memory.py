"""Bounded-memory cohort handling in the scan pipeline.

Regression suite for the unbounded-cohort-memory defect: ``_fetch_all`` used to
retain every full :class:`SecurityData` (two years of price bars, news,
filings — ~0.6 MB each) from the fetch stage until ``scan()`` returned, so a
broad scan held the whole cohort resident simultaneously (~GBs at a few
thousand names).

The fix spills each full payload to a per-scan temporary directory as soon as
it is fetched and keeps only a *slim* copy (heavy list fields stripped) in
memory; the full object is re-hydrated one company at a time for the analyze
and explain stages. These tests pin that contract, fully offline:

* the spill/re-hydrate round trip is lossless (honesty: no field is dropped,
  imputed or altered) and post-fetch ``data_warnings`` travel with it;
* the store degrades to in-memory retention when the disk is unavailable
  (correctness over the memory bound);
* during a real scan the retained cohort holds slim copies only, while every
  analyzer still receives the full payload.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from convexity.core.models import ScanParams
from convexity.data import universe as universe_mod
from convexity.pipeline import _HEAVY_COHORT_FIELDS, ScanPipeline, _CohortStore
from tests.conftest import FakeProvider

# ---------------------------------------------------------------------------
# _CohortStore unit behaviour
# ---------------------------------------------------------------------------


class TestCohortStore:
    def test_put_returns_slim_copy_with_heavy_fields_stripped(self) -> None:
        data = FakeProvider().get_security_data("STRONGCO")
        store = _CohortStore()
        try:
            slim = store.put(data)

            for field in _HEAVY_COHORT_FIELDS:
                assert getattr(slim, field) == [], field
            # Everything the screening/context stages read survives on the slim.
            assert slim.ticker == data.ticker
            assert slim.sector == data.sector
            assert slim.market_cap == data.market_cap
            assert slim.valuation.pe == data.valuation.pe
            assert [p.period_end for p in slim.fundamentals] == [
                p.period_end for p in data.fundamentals
            ]
        finally:
            store.close()

    def test_load_round_trip_is_lossless_and_carries_new_warnings(self) -> None:
        data = FakeProvider().get_security_data("STRONGCO")
        original = data.model_dump()  # snapshot before any store interaction
        store = _CohortStore()
        try:
            slim = store.put(data)
            # A post-fetch stage (cap-band enforcement) appends a warning to the
            # slim copy; it must travel onto the re-hydrated full object.
            slim.data_warnings.append("Market cap unknown after fetch (test)")

            full = store.load(slim)

            expected = dict(original)
            expected["data_warnings"] = original["data_warnings"] + [
                "Market cap unknown after fetch (test)"
            ]
            # Byte-for-byte honest round trip: nothing dropped, imputed or altered.
            assert full.model_dump() == expected
            assert len(full.price_history) == len(original["price_history"])
        finally:
            store.close()

    def test_store_degrades_to_memory_when_spill_dir_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _NoDisk:
            @staticmethod
            def mkdtemp(prefix: str) -> str:
                raise OSError("read-only filesystem")

        monkeypatch.setattr("convexity.pipeline.tempfile", _NoDisk)
        store = _CohortStore()
        try:
            data = FakeProvider().get_security_data("WEAKCO")
            slim = store.put(data)
            assert slim.price_history == []

            slim.data_warnings.append("appended after fetch")
            full = store.load(slim)

            assert full is data  # in-memory fallback serves the original object
            assert len(full.price_history) == 400
            assert "appended after fetch" in full.data_warnings
        finally:
            store.close()

    def test_close_is_idempotent(self) -> None:
        store = _CohortStore()
        store.put(FakeProvider().get_security_data("THINCO"))
        store.close()
        store.close()  # second close must not raise


# ---------------------------------------------------------------------------
# End-to-end: slim cohort in memory, full payloads for the analyzers
# ---------------------------------------------------------------------------


class TestScanMemoryContract:
    def test_cohort_is_slim_while_analyzers_get_full_payloads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeProvider()
        monkeypatch.setattr(
            universe_mod,
            "build_universe_or_seed",
            lambda params, provider=None, **_kw: fake.tickers,
        )
        pipe = ScanPipeline(provider=fake)

        # Spy on the cohort handed to the context builder (what the scan retains
        # in memory) and on the data each analyzer run actually receives.
        captured_cohort: Dict[str, List] = {}
        orig_ctx = pipe._build_context

        def spy_ctx(cohort):  # noqa: ANN001, ANN202 - test spy
            captured_cohort["cohort"] = list(cohort)
            return orig_ctx(cohort)

        bars_seen: Dict[str, int] = {}
        orig_run = pipe._run_analyzers

        def spy_run(data, analyzers, ctx):  # noqa: ANN001, ANN202 - test spy
            bars_seen[data.ticker] = len(data.price_history)
            return orig_run(data, analyzers, ctx)

        monkeypatch.setattr(pipe, "_build_context", spy_ctx)
        monkeypatch.setattr(pipe, "_run_analyzers", spy_run)

        result = pipe.scan(ScanParams(top_n=2))

        # The retained cohort carries slim copies only (heavy fields stripped)…
        cohort = captured_cohort["cohort"]
        assert len(cohort) == 6
        for slim in cohort:
            for field in _HEAVY_COHORT_FIELDS:
                assert getattr(slim, field) == [], f"{slim.ticker}.{field}"

        # …while every analyzer run received the full re-hydrated payload.
        assert bars_seen["STRONGCO"] == 400  # the full 2y-equivalent history
        assert bars_seen["THINCO"] == 40  # thin name keeps its real 40 bars
        assert all(count > 0 for count in bars_seen.values())

        # And the scan output is unaffected in shape.
        assert result.analyzed_count == 6
        assert len(result.top) == 2
        assert all(c.thesis.strip() for c in result.top)

    def test_scan_results_identical_with_and_without_disk_spill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The spill is an implementation detail: results must not change."""
        fake = FakeProvider()
        monkeypatch.setattr(
            universe_mod,
            "build_universe_or_seed",
            lambda params, provider=None, **_kw: fake.tickers,
        )
        params = ScanParams(top_n=3)

        spilled = ScanPipeline(provider=fake).scan(params)

        class _NoDisk:
            @staticmethod
            def mkdtemp(prefix: str) -> str:
                raise OSError("read-only filesystem")

        monkeypatch.setattr("convexity.pipeline.tempfile", _NoDisk)
        in_memory = ScanPipeline(provider=fake).scan(params)

        assert [c.ticker for c in spilled.all_ranked] == [
            c.ticker for c in in_memory.all_ranked
        ]
        for a, b in zip(spilled.all_ranked, in_memory.all_ranked):
            assert a.composite_score == b.composite_score
            assert a.conviction_confidence == b.conviction_confidence
            assert a.signal_agreement == b.signal_agreement
            assert a.rank == b.rank
