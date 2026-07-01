"""HTTP route routers for the Convexity API.

Each submodule defines a :class:`fastapi.APIRouter` for one resource family. They
are aggregated here into :data:`api_router`, a single router the application mounts
under the ``/api`` prefix. Importing this package only imports the routers; it
starts no scan and performs no I/O, so it stays import-safe.
"""

from __future__ import annotations

from fastapi import APIRouter

from convexity.api.routes.companies import router as companies_router
from convexity.api.routes.scans import router as scans_router
from convexity.api.routes.universe import router as universe_router

# Aggregate router mounted at "/api" by the application factory.
api_router = APIRouter(prefix="/api")
api_router.include_router(scans_router)
api_router.include_router(companies_router)
api_router.include_router(universe_router)

__all__ = ["api_router"]
