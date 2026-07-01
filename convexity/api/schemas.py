"""Pydantic request/response schemas for the Convexity HTTP API.

These models are the wire contract of the FastAPI service. Wherever possible they
**reuse the canonical core models** (:class:`~convexity.core.models.ScanParams`,
:class:`~convexity.core.models.ScanResult`,
:class:`~convexity.core.models.CompanyAnalysis`) rather than redefining their
shapes — the API never invents a second source of truth for a scan, a company or a
score. Only the small amount of state that is *purely* an HTTP concern (a scan
job's id/status/progress, a few light envelopes) is defined here.

Honesty framing
---------------
Convexity is an evidence-driven research and **screening** tool, not a predictor
and not investment advice. The response envelopes carry that framing forward
verbatim so a consumer of the JSON sees it too: conviction is only warranted when
many *independent* signals agree, missing data lowers confidence, and nothing here
implies certainty or guaranteed returns.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from convexity.core.models import (
    CompanyAnalysis,
    ScanParams,
    ScanResult,
)

# A short, always-on disclaimer echoed by the API so the honest framing travels
# with the data rather than living only in the UI.
RESEARCH_DISCLAIMER: str = (
    "Convexity is an evidence-driven research and screening tool, not a predictor "
    "and not investment advice. Scores reflect agreement across independent "
    "signals; missing data lowers confidence. Nothing here implies certainty or "
    "guaranteed returns. Always do your own research."
)


# ---------------------------------------------------------------------------
# Scan job lifecycle (the only genuinely API-local state)
# ---------------------------------------------------------------------------


class ScanStatus(str, Enum):
    """Lifecycle states of a background scan job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ScanProgress(BaseModel):
    """A point-in-time progress snapshot for a running scan.

    Mirrors the pipeline's ``progress(stage, done, total, message)`` callback so a
    client can render a live progress bar. ``fraction`` is a convenience 0..1 view
    of ``done/total`` (``0.0`` when ``total`` is unknown/zero).
    """

    model_config = ConfigDict(extra="ignore")

    stage: str = ""
    done: int = 0
    total: int = 0
    fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = ""


class CreateScanRequest(BaseModel):
    """Request body for ``POST /api/scans``.

    The scan parameters reuse the canonical :class:`ScanParams` model directly so
    there is exactly one definition of what a screen is. The field is optional;
    when omitted the pipeline's documented defaults apply.
    """

    model_config = ConfigDict(extra="ignore")

    params: ScanParams = Field(default_factory=ScanParams)


class ScanJob(BaseModel):
    """The full server-side record of one background scan.

    Combines the API-local lifecycle fields (``id``, ``status``, ``progress``,
    timing, any ``error``) with the eventual :class:`ScanResult` once the scan
    completes. The ``result`` is ``None`` until the job finishes successfully.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    status: ScanStatus = ScanStatus.PENDING
    progress: ScanProgress = Field(default_factory=ScanProgress)
    created_at: _dt.datetime
    started_at: Optional[_dt.datetime] = None
    finished_at: Optional[_dt.datetime] = None
    error: Optional[str] = None
    result: Optional[ScanResult] = None


class ScanJobSummary(BaseModel):
    """A lightweight view of a :class:`ScanJob` without the (large) result body.

    Returned by ``POST /api/scans`` and ``GET /api/scans/{id}`` so a poller can
    cheaply track status and progress; fetch the full result separately once the
    status is ``completed``.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    status: ScanStatus
    progress: ScanProgress
    created_at: _dt.datetime
    started_at: Optional[_dt.datetime] = None
    finished_at: Optional[_dt.datetime] = None
    error: Optional[str] = None
    has_result: bool = False

    @classmethod
    def from_job(cls, job: ScanJob) -> ScanJobSummary:
        """Project a full :class:`ScanJob` down to its summary view."""
        return cls(
            id=job.id,
            status=job.status,
            progress=job.progress,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error=job.error,
            has_result=job.result is not None,
        )


# ---------------------------------------------------------------------------
# Response envelopes
# ---------------------------------------------------------------------------


class ScanResultResponse(BaseModel):
    """Envelope returned for a completed scan's result.

    Wraps the canonical :class:`ScanResult` alongside the job ``id`` it belongs to
    and the standing research ``disclaimer``.
    """

    model_config = ConfigDict(extra="ignore")

    job_id: str
    disclaimer: str = RESEARCH_DISCLAIMER
    result: ScanResult


class CompanyResponse(BaseModel):
    """Envelope returned by ``GET /api/companies/{ticker}``.

    Wraps a single canonical :class:`CompanyAnalysis` with the research
    ``disclaimer`` so the honest framing accompanies the analysis.
    """

    model_config = ConfigDict(extra="ignore")

    disclaimer: str = RESEARCH_DISCLAIMER
    analysis: CompanyAnalysis


class UniverseCandidate(BaseModel):
    """One screened candidate in a universe-preview response."""

    model_config = ConfigDict(extra="ignore")

    ticker: str


class UniversePreviewResponse(BaseModel):
    """Envelope returned by ``GET /api/universe/preview``.

    Reports the screened candidate tickers (bounded by ``limit``) together with the
    effective ``ScanParams`` used to screen and any human-readable ``notes`` about
    gaps (e.g. a fallback to the bundled seed list). The preview is a screen
    *only* — no per-ticker data is fetched or analysed.
    """

    model_config = ConfigDict(extra="ignore")

    disclaimer: str = RESEARCH_DISCLAIMER
    requested_limit: Optional[int] = None
    count: int = 0
    candidates: List[UniverseCandidate] = Field(default_factory=list)
    params: ScanParams = Field(default_factory=ScanParams)
    notes: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Body of the ``/health`` probe."""

    model_config = ConfigDict(extra="ignore")

    status: str = "ok"
    service: str = "convexity-api"
    disclaimer: str = RESEARCH_DISCLAIMER
    active_scans: int = 0
    persisted_scans: int = 0


class ErrorResponse(BaseModel):
    """Standard JSON error body used by the API exception handlers."""

    model_config = ConfigDict(extra="ignore")

    error: str
    detail: Optional[str] = None
    status_code: int = 500


__all__ = [
    "RESEARCH_DISCLAIMER",
    "ScanStatus",
    "ScanProgress",
    "CreateScanRequest",
    "ScanJob",
    "ScanJobSummary",
    "ScanResultResponse",
    "CompanyResponse",
    "UniverseCandidate",
    "UniversePreviewResponse",
    "HealthResponse",
    "ErrorResponse",
]
