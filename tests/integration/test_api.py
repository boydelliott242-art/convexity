"""HTTP API integration tests, driven entirely by the synthetic FakeProvider.

These exercise the real FastAPI app (:func:`convexity.api.app.create_app`) through
a :class:`fastapi.testclient.TestClient`, with the pipeline's data source swapped
for the offline :class:`~tests.conftest.FakeProvider` and the universe stage
overridden to the fake tickers. No network is touched.

Coverage:

* ``GET /health`` reports liveness and carries the standing research disclaimer.
* ``POST /api/scans`` starts a background scan; polling ``GET /api/scans/{id}``
  reaches ``completed`` and ``GET /api/scans/{id}/result`` returns a well-formed,
  fully-explained result.
* ``GET /api/companies/{ticker}`` analyses a single synthetic company and returns
  its explained analysis; an unknown ticker honestly 404s.

The fixtures monkeypatch the pipeline that the routes resolve (and the universe
builder) and point the scan-job store at a temporary directory so persisted results
never leak between tests.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from convexity.api.schemas import RESEARCH_DISCLAIMER


@pytest.fixture()
def client(
    fake_provider, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> Iterator[TestClient]:
    """A TestClient whose API pipeline is wired to the FakeProvider, offline.

    The route modules import ``get_pipeline`` into their own namespace, so the
    pipeline is overridden by patching that name on each route module to return a
    single fake-backed :class:`~convexity.pipeline.ScanPipeline`. The universe
    builder is patched to the fake tickers, and the scan-job store is rebuilt
    against an isolated ``tmp_path`` so persistence is sandboxed.
    """
    from convexity.api import store as store_mod
    from convexity.api.routes import companies as companies_route
    from convexity.api.routes import scans as scans_route
    from convexity.core.config import Settings
    from convexity.data import universe as universe_mod
    from convexity.pipeline import ScanPipeline

    # 1) Universe -> the fake tickers (honouring any universe_limit).
    def _fake_universe(params, price_provider=None, **_kwargs):
        tickers = fake_provider.tickers
        limit = params.universe_limit
        if limit is not None and limit >= 0:
            tickers = tickers[:limit]
        return list(tickers)

    monkeypatch.setattr(universe_mod, "build_universe_or_seed", _fake_universe)

    # 2) One fake-backed pipeline, returned wherever a route asks for one.
    fake_pipeline = ScanPipeline(provider=fake_provider)
    monkeypatch.setattr(scans_route, "get_pipeline", lambda: fake_pipeline)
    monkeypatch.setattr(companies_route, "get_pipeline", lambda: fake_pipeline)

    # 3) An isolated, temp-backed store so persisted scans never leak across tests.
    isolated_store = store_mod.ScanJobStore(
        settings=Settings(data_dir=str(tmp_path))
    )
    monkeypatch.setattr(store_mod, "_DEFAULT_STORE", isolated_store)

    from convexity.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_probe(client: TestClient) -> None:
    """``/health`` is live and echoes the research disclaimer."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["disclaimer"] == RESEARCH_DISCLAIMER
    assert "active_scans" in body
    assert "persisted_scans" in body


def _poll_until_done(client: TestClient, job_id: str, timeout_s: float = 10.0) -> dict:
    """Poll a scan job until it reaches a terminal state, returning its summary.

    With the synchronous TestClient the background task typically completes before
    the first poll, but this loop tolerates either ordering and fails loudly on a
    timeout so a hang never masquerades as a pass.
    """
    deadline = time.monotonic() + timeout_s
    summary: dict = {}
    while time.monotonic() < deadline:
        resp = client.get(f"/api/scans/{job_id}")
        assert resp.status_code == 200
        summary = resp.json()
        if summary["status"] in ("completed", "failed"):
            return summary
        time.sleep(0.05)
    raise AssertionError(f"scan {job_id} did not finish within {timeout_s}s: {summary}")


def test_start_scan_poll_to_done_and_fetch_result(client: TestClient) -> None:
    """Start a scan, poll it to completion, then fetch its full result."""
    start = client.post("/api/scans", json={"params": {"top_n": 3}})
    assert start.status_code == 202
    created = start.json()
    job_id = created["id"]
    assert created["status"] in ("pending", "running", "completed")

    summary = _poll_until_done(client, job_id)
    assert summary["status"] == "completed", summary
    assert summary["has_result"] is True

    result_resp = client.get(f"/api/scans/{job_id}/result")
    assert result_resp.status_code == 200
    payload = result_resp.json()
    assert payload["job_id"] == job_id
    assert payload["disclaimer"] == RESEARCH_DISCLAIMER

    result = payload["result"]
    assert result["universe_size"] == 6
    assert result["analyzed_count"] == 6
    assert result["error_count"] == 0
    assert len(result["top"]) == 3
    assert len(result["all_ranked"]) == 6

    # The top company is fully explained over the wire.
    top0 = result["top"][0]
    assert top0["thesis"].strip()
    assert top0["bull_case"]
    assert top0["bear_case"]
    assert top0["monitoring_checklist"]
    # Strong name leads the ranking.
    assert result["all_ranked"][0]["ticker"] == "STRONGCO"


def test_latest_scan_reflects_completed_run(client: TestClient) -> None:
    """After a completed scan, ``/api/scans/latest`` serves that result."""
    # Before any scan, latest is a clean 404.
    assert client.get("/api/scans/latest").status_code == 404

    start = client.post("/api/scans", json={"params": {"top_n": 2}})
    job_id = start.json()["id"]
    _poll_until_done(client, job_id)

    latest = client.get("/api/scans/latest")
    assert latest.status_code == 200
    assert latest.json()["result"]["analyzed_count"] == 6


def test_get_company(client: TestClient) -> None:
    """``GET /api/companies/{ticker}`` analyses and explains a single name."""
    resp = client.get("/api/companies/strongco")  # case-insensitive
    assert resp.status_code == 200
    body = resp.json()
    assert body["disclaimer"] == RESEARCH_DISCLAIMER

    analysis = body["analysis"]
    assert analysis["ticker"] == "STRONGCO"
    assert 0.0 <= analysis["composite_score"] <= 100.0
    assert 0.0 <= analysis["conviction_confidence"] <= 1.0
    assert analysis["subscores"], "single-company analysis should carry sub-scores"
    assert analysis["thesis"].strip()
    assert analysis["monitoring_checklist"]


def test_get_unknown_company_is_404(client: TestClient) -> None:
    """An uncovered ticker is an honest 404 (data gap, not a server error)."""
    resp = client.get("/api/companies/NOSUCHTICKER")
    assert resp.status_code == 404
    detail = resp.json().get("detail", "")
    assert "NOSUCHTICKER" in detail
