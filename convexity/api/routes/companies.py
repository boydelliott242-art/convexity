"""Single-company endpoint: analyse one ticker on demand.

``GET /api/companies/{ticker}`` runs the pipeline's single-ticker path
(:meth:`~convexity.pipeline.ScanPipeline.analyze_one`) — fetch, analyse against an
empty (peerless) context, score and explain — and returns the explained
:class:`~convexity.core.models.CompanyAnalysis` wrapped with the standing research
disclaimer.

This is a synchronous fetch + analyse (no background job): a single ticker is cheap
enough to serve inline. Expected data gaps (an uncovered micro-cap) surface as a
``404``; any unexpected provider failure surfaces as a ``502`` so the caller can
tell "no data" apart from "something broke upstream".
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from convexity.api.deps import get_pipeline
from convexity.api.schemas import CompanyResponse
from convexity.core.exceptions import ConvexityError, DataUnavailable
from convexity.core.logging import get_logger

_log = get_logger(__name__)

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get(
    "/{ticker}",
    response_model=CompanyResponse,
    summary="Fetch, analyse and explain a single company",
)
def get_company(ticker: str) -> CompanyResponse:
    """Analyse one ``ticker`` and return its explained analysis.

    Args:
        ticker: The security symbol to analyse (case-insensitive; trimmed).

    Returns:
        A :class:`CompanyResponse` wrapping the explained
        :class:`CompanyAnalysis` and the research disclaimer.

    Raises:
        HTTPException: ``400`` for an empty symbol; ``404`` when no provider can
            supply data for the ticker (an expected gap); ``502`` for an unexpected
            upstream provider failure.
    """
    symbol = (ticker or "").strip()
    if not symbol:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A ticker symbol is required.",
        )

    pipeline = get_pipeline()
    try:
        analysis = pipeline.analyze_one(symbol)
    except DataUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No data is available for '{symbol.upper()}': {exc}. "
                "This reflects data availability, not a market view."
            ),
        ) from exc
    except ConvexityError as exc:
        _log.warning("provider error analysing %s: %s", symbol, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream data provider failed for '{symbol.upper()}': {exc}.",
        ) from exc

    return CompanyResponse(analysis=analysis)


__all__ = ["router"]
