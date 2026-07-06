"""Scan endpoints: start a background scan, poll it, stream progress, fetch latest.

Routes
------
* ``POST /api/scans`` — register a scan and run it as a FastAPI ``BackgroundTask``;
  returns a :class:`ScanJobSummary` immediately with the new job id.
* ``GET  /api/scans/latest`` — the most recent completed :class:`ScanResult`
  (survives a restart via on-disk persistence loaded at startup).
* ``GET  /api/scans/{job_id}`` — the lightweight job summary (status + progress).
* ``GET  /api/scans/{job_id}/result`` — the full result once the job has completed.
* ``GET  /api/scans/{job_id}/events`` — a Server-Sent-Events stream of progress
  ticks that ends when the job reaches a terminal state.

The scan body runs on the background-task threadpool. Its progress callback writes
into the shared :class:`~convexity.api.store.ScanJobStore` so both the polling and
the SSE endpoints observe live updates. The scan is wrapped so that any failure is
recorded as a ``FAILED`` job rather than crashing the worker — honest about the
gap, never fabricating a result.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from convexity.api.deps import get_pipeline
from convexity.api.schemas import (
    CreateScanRequest,
    ScanJobSummary,
    ScanResultResponse,
    ScanStatus,
)
from convexity.api.store import get_store
from convexity.core.logging import get_logger
from convexity.core.models import ScanParams

_log = get_logger(__name__)

router = APIRouter(prefix="/scans", tags=["scans"])

# How often (seconds) the SSE endpoint samples job state to emit a progress event.
_SSE_POLL_INTERVAL = 0.5
# A heartbeat comment is emitted at least this often so proxies keep the stream
# open even when no progress tick has changed.
_SSE_HEARTBEAT_EVERY = 15.0


def _run_scan(job_id: str, params: ScanParams) -> None:
    """Execute one scan to completion, updating the shared store throughout.

    Runs on the background-task threadpool. It marks the job running, wires the
    pipeline's progress callback to the store, and records either the completed
    result or the failure. It never raises: a failure is captured on the job.

    Args:
        job_id: The store key of the job to drive.
        params: The validated scan parameters to run.
    """
    store = get_store()
    store.mark_running(job_id)

    def _progress(stage: str, done: int, total: int, message: str) -> None:
        store.update_progress(job_id, stage, done, total, message)

    try:
        pipeline = get_pipeline()
        result = pipeline.scan(params, progress=_progress)
        store.mark_completed(job_id, result)
        _log.info(
            "scan %s complete: %d analysed, %d error(s)",
            job_id,
            result.analyzed_count,
            result.error_count,
        )
    except Exception as exc:  # defensive: a scan failure must not crash the worker
        _log.error("scan %s failed: %s", job_id, exc)
        store.mark_failed(job_id, str(exc))


@router.post(
    "",
    response_model=ScanJobSummary,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a research scan as a background job",
)
def create_scan(
    request: CreateScanRequest,
    background_tasks: BackgroundTasks,
) -> ScanJobSummary:
    """Register and start a scan; return its job summary immediately.

    The scan runs asynchronously as a background task. Poll ``GET
    /api/scans/{id}`` (or stream ``/events``) for progress, then fetch the result
    from ``GET /api/scans/{id}/result`` once the status is ``completed``.
    """
    store = get_store()
    job = store.create(request.params)
    background_tasks.add_task(_run_scan, job.id, request.params)
    return ScanJobSummary.from_job(job)


@router.get(
    "/latest",
    response_model=ScanResultResponse,
    summary="Fetch the most recent completed scan result",
)
def latest_scan() -> ScanResultResponse:
    """Return the most recent completed scan result.

    Backed by the store's persisted-result memory, so it works after a restart
    once the startup hook has loaded the last result from disk.

    Raises:
        HTTPException: ``404`` when no scan has ever completed.
    """
    store = get_store()
    result = store.latest_result()
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed scan is available yet.",
        )
    return ScanResultResponse(job_id="latest", result=result)


@router.get(
    "/active",
    response_model=List[ScanJobSummary],
    summary="List scans that are currently pending or running",
)
def active_scans() -> List[ScanJobSummary]:
    """Return summaries of every in-flight scan, newest first.

    Lets a dashboard that lost track of a scan (page reload, browser sleep, a
    client that stopped watching) rediscover and reattach to it without knowing
    the job id. Declared before ``/{job_id}`` so "active" is never captured as
    an id.
    """
    store = get_store()
    return [ScanJobSummary.from_job(job) for job in store.active_jobs()]


@router.get(
    "/{job_id}",
    response_model=ScanJobSummary,
    summary="Poll a scan job's status and progress",
)
def get_scan(job_id: str) -> ScanJobSummary:
    """Return the lightweight summary (status + progress) of a scan job.

    Raises:
        HTTPException: ``404`` when ``job_id`` is unknown (or was evicted).
    """
    store = get_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown scan job '{job_id}'.",
        )
    return ScanJobSummary.from_job(job)


@router.get(
    "/{job_id}/result",
    response_model=ScanResultResponse,
    summary="Fetch a completed scan job's full result",
)
def get_scan_result(job_id: str) -> ScanResultResponse:
    """Return the full :class:`ScanResult` for a completed job.

    Raises:
        HTTPException: ``404`` if the job is unknown; ``409`` if it has not yet
            completed (still pending/running) or failed without a result.
    """
    store = get_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown scan job '{job_id}'.",
        )
    if job.status == ScanStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Scan '{job_id}' failed: {job.error or 'unknown error'}.",
        )
    if job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Scan '{job_id}' is '{job.status.value}'; no result yet.",
        )
    return ScanResultResponse(job_id=job.id, result=job.result)


def _sse_event(event: str, data: dict) -> str:
    """Format one Server-Sent-Events frame (``event:`` + JSON ``data:``)."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _scan_event_stream(job_id: str, request: Request) -> AsyncIterator[str]:
    """Yield SSE frames tracking a job until it reaches a terminal state.

    Emits an initial ``progress`` frame, then further ``progress`` frames whenever
    the snapshot changes, periodic ``:heartbeat`` comments to keep the connection
    alive, and a final ``done`` (completed) or ``error`` (failed) frame. The stream
    also ends early if the client disconnects.
    """
    store = get_store()
    job = store.get(job_id)
    if job is None:
        yield _sse_event("error", {"job_id": job_id, "error": "unknown job"})
        return

    last_signature: tuple = ()
    since_heartbeat = 0.0

    while True:
        if await request.is_disconnected():
            return

        job = store.get(job_id)
        if job is None:
            yield _sse_event("error", {"job_id": job_id, "error": "job evicted"})
            return

        prog = job.progress
        signature = (job.status.value, prog.stage, prog.done, prog.total, prog.message)
        if signature != last_signature:
            last_signature = signature
            since_heartbeat = 0.0
            yield _sse_event(
                "progress",
                {
                    "job_id": job.id,
                    "status": job.status.value,
                    "stage": prog.stage,
                    "done": prog.done,
                    "total": prog.total,
                    "fraction": prog.fraction,
                    "message": prog.message,
                },
            )

        if job.status == ScanStatus.COMPLETED:
            yield _sse_event(
                "done",
                {"job_id": job.id, "status": job.status.value, "message": "Scan complete."},
            )
            return
        if job.status == ScanStatus.FAILED:
            yield _sse_event(
                "error",
                {"job_id": job.id, "status": job.status.value, "error": job.error or "scan failed"},
            )
            return

        await asyncio.sleep(_SSE_POLL_INTERVAL)
        since_heartbeat += _SSE_POLL_INTERVAL
        if since_heartbeat >= _SSE_HEARTBEAT_EVERY:
            since_heartbeat = 0.0
            yield ": heartbeat\n\n"


@router.get(
    "/{job_id}/events",
    summary="Stream a scan job's progress as Server-Sent Events",
)
async def scan_events(job_id: str, request: Request) -> StreamingResponse:
    """Open a Server-Sent-Events stream of a scan job's progress.

    The response has ``Content-Type: text/event-stream`` and yields ``progress``
    events as the scan advances, terminating with a ``done`` or ``error`` event.
    Unknown job ids still open a (short) stream that emits a single ``error`` frame
    so the client gets a uniform channel rather than an HTTP error mid-poll.
    """
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _scan_event_stream(job_id, request),
        media_type="text/event-stream",
        headers=headers,
    )


__all__ = ["router"]
