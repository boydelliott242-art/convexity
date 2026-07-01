"""Universe-preview endpoint: screen candidate tickers without analysing them.

``GET /api/universe/preview?limit=`` runs only the *screen* stage of the funnel —
:func:`convexity.data.universe.build_universe_or_seed` — and returns the eligible
candidate tickers (bounded by ``limit``). No per-ticker data is fetched and nothing
is analysed, so the preview is fast and cheap: it answers "what would a scan look at
right now?" while honestly noting when it falls back to the bundled seed list
because the live screen was unavailable.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from convexity.api.deps import get_pipeline
from convexity.api.schemas import (
    UniverseCandidate,
    UniversePreviewResponse,
)
from convexity.core.config import get_settings
from convexity.core.logging import get_logger
from convexity.core.models import ScanParams

_log = get_logger(__name__)

router = APIRouter(prefix="/universe", tags=["universe"])

# Hard ceiling on how many candidates a single preview returns, to keep the
# response bounded regardless of the requested limit.
_MAX_PREVIEW = 500
# Default number of candidates returned when no ``limit`` query is supplied.
_DEFAULT_PREVIEW = 50


@router.get(
    "/preview",
    response_model=UniversePreviewResponse,
    summary="Preview the screened candidate universe (no analysis)",
)
def universe_preview(
    limit: Optional[int] = Query(
        default=_DEFAULT_PREVIEW,
        ge=1,
        le=_MAX_PREVIEW,
        description="Maximum number of candidate tickers to return.",
    ),
    min_market_cap: Optional[float] = Query(
        default=None, ge=0.0, description="Override the minimum market-cap floor."
    ),
    max_market_cap: Optional[float] = Query(
        default=None, ge=0.0, description="Override the maximum market-cap ceiling."
    ),
    min_avg_dollar_volume: Optional[float] = Query(
        default=None, ge=0.0, description="Override the minimum average dollar volume."
    ),
) -> UniversePreviewResponse:
    """Screen and return candidate tickers without fetching or analysing them.

    Builds a :class:`ScanParams` from the (optional) overrides plus the requested
    ``limit`` as ``universe_limit``, then runs the screen-or-seed step. Whether the
    result came from the live screen or the bundled seed fallback is reflected in
    the ``notes``.

    Args:
        limit: Maximum candidates to return (also caps ``universe_limit``).
        min_market_cap: Optional override of the cap floor.
        max_market_cap: Optional override of the cap ceiling.
        min_avg_dollar_volume: Optional override of the liquidity floor.

    Returns:
        A :class:`UniversePreviewResponse` with the candidates, the effective
        params and any screening notes.

    Raises:
        HTTPException: ``400`` if the cap band is inverted (max below min); ``502``
            if the screen step fails unexpectedly.
    """
    settings = get_settings()
    base = ScanParams()

    effective = ScanParams(
        min_market_cap=min_market_cap if min_market_cap is not None else base.min_market_cap,
        max_market_cap=max_market_cap if max_market_cap is not None else base.max_market_cap,
        min_avg_dollar_volume=(
            min_avg_dollar_volume
            if min_avg_dollar_volume is not None
            else base.min_avg_dollar_volume
        ),
        exclude_sectors=base.exclude_sectors,
        top_n=base.top_n,
        universe_limit=limit,
    )

    if effective.max_market_cap < effective.min_market_cap:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="max_market_cap must be greater than or equal to min_market_cap.",
        )

    notes: List[str] = []
    pipeline = get_pipeline()
    # Reuse the pipeline's configured provider so the preview screens the same
    # universe a real scan would. Access via the documented attribute name.
    provider = getattr(pipeline, "_provider", None)

    from convexity.data import universe as universe_mod

    try:
        tickers = universe_mod.build_universe_or_seed(
            effective,
            provider,
            user_agent=settings.sec_user_agent,
            timeout=settings.request_timeout,
        )
    except Exception as exc:  # defensive: never leak a raw stack to the client
        _log.error("universe preview screen failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Universe screen failed: {exc}.",
        )

    # De-duplicate while preserving order, normalise casing, and bound the count.
    seen: set = set()
    ordered: List[str] = []
    for raw in tickers:
        sym = (raw or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        ordered.append(sym)
        if len(ordered) >= (limit or _DEFAULT_PREVIEW):
            break

    if not ordered:
        notes.append(
            "The screen produced no eligible candidates; this reflects data "
            "availability, not a market view."
        )
    else:
        notes.append(
            f"Screened {len(ordered)} candidate ticker(s). A preview screens only — "
            "no per-ticker data is fetched and nothing is analysed."
        )

    candidates = [UniverseCandidate(ticker=t) for t in ordered]
    return UniversePreviewResponse(
        requested_limit=limit,
        count=len(candidates),
        candidates=candidates,
        params=effective,
        notes=notes,
    )


__all__ = ["router"]
