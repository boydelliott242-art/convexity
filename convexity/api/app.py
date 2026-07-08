"""FastAPI application factory for the Convexity research API.

:func:`create_app` assembles the HTTP service: permissive CORS for local frontends,
a ``/health`` probe, JSON exception handlers that keep the honest framing intact,
the ``/api`` routers (scans, companies, universe), and — *only if it exists* — the
bundled single-page frontend mounted at ``/``. A module-level :data:`app` instance
is exported so ``uvicorn convexity.api.app:app`` just works.

Honesty & safety framing
-------------------------
Convexity is an evidence-driven research and **screening** tool — not a predictor,
not investment advice. The API never fabricates data: an upstream gap becomes a
``404``/``502`` with an honest message, and the standing research disclaimer is
echoed on the health probe and every result envelope.

Import-safety
-------------
Importing this module (and calling :func:`create_app`) starts no scan, touches no
network and requires no provider credentials. The pipeline and scan-job store are
built lazily on first use; the only startup-time I/O is a best-effort read of the
most recent persisted scan result so ``/api/scans/latest`` is populated after a
restart — a missing directory or no prior scan is handled gracefully.
"""

from __future__ import annotations

import os
import pathlib

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

from convexity.api.routes import api_router
from convexity.api.schemas import ErrorResponse, HealthResponse
from convexity.api.store import get_store
from convexity.core.exceptions import ConvexityError, DataUnavailable
from convexity.core.logging import get_logger

_log = get_logger(__name__)

# Optional bundled frontend, mounted at "/" only if present so the API is fully
# usable headless (and import-safe) when no frontend is shipped. The directory is
# derived from the installed package location (repo root when editable-installed)
# — NEVER hardcoded to a machine path: a previous hardcoded ~/Desktop path broke
# after the repo moved, and macOS TCC additionally denies Desktop reads to
# launchd-run services, silently degrading the server to headless. An explicit
# CONVEXITY_FRONTEND_DIR environment variable overrides the derived default.
_FRONTEND_DIR = os.environ.get(
    "CONVEXITY_FRONTEND_DIR",
    str(pathlib.Path(__file__).resolve().parent.parent.parent / "frontend"),
)

_API_TITLE = "Convexity"
_API_DESCRIPTION = (
    "Evidence-driven small/micro-cap equity research and screening API. "
    "A research tool, not a predictor and not investment advice: conviction is "
    "warranted only when many independent signals agree, missing data lowers "
    "confidence, and nothing here implies certainty or guaranteed returns."
)
_API_VERSION = "1.0.0"


def _install_cors(app: FastAPI) -> None:
    """Attach permissive CORS so a local/static frontend can call the API.

    The allowlist is intentionally open (any origin) because this service exposes
    only read-oriented research endpoints and carries no credentials or secrets in
    its responses; a deployment behind auth can tighten this.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _install_exception_handlers(app: FastAPI) -> None:
    """Register handlers that render every error as the standard JSON envelope.

    Keeps responses uniform (and honest): a Convexity data gap is a clean ``404``,
    a provider failure a ``502``, validation a ``422``, and anything unexpected a
    ``500`` — never a raw stack trace leaked to the client.
    """

    @app.exception_handler(DataUnavailable)
    async def _handle_data_unavailable(
        _request: Request, exc: DataUnavailable
    ) -> JSONResponse:
        """An expected data gap becomes a clean ``404`` (not a server error)."""
        body = ErrorResponse(
            error="data_unavailable",
            detail=(
                f"{exc} This reflects data availability, not a market view."
            ),
            status_code=status.HTTP_404_NOT_FOUND,
        )
        return JSONResponse(status_code=body.status_code, content=body.model_dump())

    @app.exception_handler(ConvexityError)
    async def _handle_convexity_error(
        _request: Request, exc: ConvexityError
    ) -> JSONResponse:
        """Any other Convexity-family error becomes a ``502`` upstream failure."""
        _log.warning("convexity error: %s", exc)
        body = ErrorResponse(
            error="upstream_error",
            detail=str(exc),
            status_code=status.HTTP_502_BAD_GATEWAY,
        )
        return JSONResponse(status_code=body.status_code, content=body.model_dump())

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Render explicit ``HTTPException``s through the standard envelope."""
        body = ErrorResponse(
            error="http_error",
            detail=str(exc.detail),
            status_code=exc.status_code,
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Surface request-validation problems as a ``422`` with the details."""
        body = ErrorResponse(
            error="validation_error",
            detail=str(exc.errors()),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
        return JSONResponse(status_code=body.status_code, content=body.model_dump())

    @app.exception_handler(Exception)
    async def _handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        """Last-resort handler: log and return a generic ``500`` (no stack leak)."""
        _log.error("unhandled error: %s", exc)
        body = ErrorResponse(
            error="internal_error",
            detail="An unexpected error occurred.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        return JSONResponse(status_code=body.status_code, content=body.model_dump())


def _register_routes(app: FastAPI) -> None:
    """Mount the ``/health`` probe and the aggregated ``/api`` router."""

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["meta"],
        summary="Liveness probe and standing research disclaimer",
    )
    def health() -> HealthResponse:
        """Report liveness plus how many scans are active/persisted.

        Reading the store here triggers its lazy creation but performs no scan and
        no network I/O, so the probe is cheap and import-safe.
        """
        store = get_store()
        return HealthResponse(
            active_scans=store.active_count(),
            persisted_scans=store.persisted_count(),
        )

    app.include_router(api_router)


def _mount_frontend(app: FastAPI) -> None:
    """Mount the bundled single-page frontend at ``/`` *iff* the directory exists.

    Guarded against a missing directory so the service runs headless as a pure API
    when no frontend is shipped. ``html=True`` serves ``index.html`` for ``/`` and
    enables SPA-style fallback. Mounted last so it never shadows ``/api`` or
    ``/health``.
    """
    if os.path.isdir(_FRONTEND_DIR):
        app.mount(
            "/",
            StaticFiles(directory=_FRONTEND_DIR, html=True),
            name="frontend",
        )
        _log.info("mounted frontend from %s", _FRONTEND_DIR)
    else:
        _log.info(
            "no frontend directory at %s; running as a headless API.", _FRONTEND_DIR
        )


def _register_startup(app: FastAPI) -> None:
    """Register the startup hook that loads the most recent persisted scan.

    Best-effort: a missing directory, an empty directory or a corrupt file is
    handled inside the store and simply leaves ``/api/scans/latest`` returning
    ``404`` until the first scan completes. The app starts fine with no prior scan.
    """

    @app.on_event("startup")
    def _load_latest_scan() -> None:
        """Populate the store's latest-result memory from disk, if any exists."""
        try:
            result = get_store().load_persisted()
            if result is not None:
                _log.info(
                    "startup: latest persisted scan has %d ranked company(ies).",
                    len(result.all_ranked),
                )
        except Exception as exc:  # pragma: no cover - defensive; startup must not fail
            _log.warning("startup: could not load persisted scans: %s", exc)


def create_app() -> FastAPI:
    """Build and return the configured Convexity FastAPI application.

    Wires CORS, exception handlers, the ``/health`` probe, the ``/api`` routers, the
    startup persisted-scan loader and (only if present) the static frontend at
    ``/``. Calling this performs no scan and no network I/O — the pipeline and store
    are built lazily on first request.

    Returns:
        A ready-to-serve :class:`fastapi.FastAPI` instance.
    """
    app = FastAPI(
        title=_API_TITLE,
        description=_API_DESCRIPTION,
        version=_API_VERSION,
    )

    _install_cors(app)
    _install_exception_handlers(app)
    _register_routes(app)
    _register_startup(app)
    # Frontend is mounted last so its catch-all ``/`` mount cannot shadow the API.
    _mount_frontend(app)

    return app


# Module-level application instance for ``uvicorn convexity.api.app:app``.
app = create_app()


__all__ = ["create_app", "app"]
