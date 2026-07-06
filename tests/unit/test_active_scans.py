"""Regression tests: in-flight scan discovery (the reattach path).

A dashboard that reloads (or a browser that stopped watching) must be able to
rediscover a running scan without knowing its job id. Locks in:

* ``ScanJobStore.active_jobs()`` — pending/running only, newest first;
* ``GET /api/scans/active`` — returns a LIST (proving the route is matched
  before the ``/{job_id}`` catch-all rather than "active" being parsed as an id).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from convexity.api.app import create_app
from convexity.api.store import ScanJobStore
from convexity.core.models import ScanParams, ScanResult


def _params() -> ScanParams:
    return ScanParams(universe_limit=5)


def test_active_jobs_lists_pending_and_running_newest_first(tmp_path) -> None:
    from convexity.core.config import Settings

    store = ScanJobStore(Settings(data_dir=str(tmp_path)))
    first = store.create(_params())
    second = store.create(_params())
    store.mark_running(second.id)
    third = store.create(_params())
    store.mark_running(third.id)
    store.mark_completed(
        third.id,
        ScanResult(
            generated_at=third.created_at,
            params=_params(),
            universe_size=0, screened_count=0, analyzed_count=0, error_count=0,
            top=[], all_ranked=[], category_weights={}, elapsed_seconds=0.0, notes=[],
        ),
    )
    active = store.active_jobs()
    ids = [j.id for j in active]
    assert third.id not in ids, "completed jobs are not active"
    assert ids == [second.id, first.id], "newest first, pending and running both included"


def test_active_route_returns_list_not_job_lookup() -> None:
    client = TestClient(create_app())
    resp = client.get("/api/scans/active")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list), "'active' must match the list route, not /{job_id}"
