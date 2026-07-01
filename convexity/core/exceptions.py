"""Exception hierarchy for Convexity.

All Convexity-specific errors derive from :class:`ConvexityError` so that callers
(the pipeline, the aggregator, the CLI) can catch the whole family with a single
``except ConvexityError`` and degrade gracefully — a single bad ticker or a single
unsupported provider capability must never crash a scan.
"""

from __future__ import annotations

from typing import Optional


class ConvexityError(Exception):
    """Base class for every error raised inside Convexity."""


class DataUnavailable(ConvexityError):
    """Requested data could not be obtained for a ticker or field.

    This is an *expected* condition (a thin micro-cap may simply have no analyst
    coverage, no institutional holdings, etc.). Callers should mark the data as
    missing and lower confidence rather than treating it as a hard failure.
    """

    def __init__(self, message: str, *, ticker: Optional[str] = None, field: Optional[str] = None) -> None:
        super().__init__(message)
        self.ticker = ticker
        self.field = field


class NotSupported(ConvexityError):
    """A provider was asked for a capability it does not implement.

    For example, calling :meth:`DataProvider.get_universe` on a provider that can
    only fetch per-ticker data. Registries and the aggregator use this to skip a
    provider cleanly and try the next one.
    """


class ProviderError(ConvexityError):
    """A data provider failed (HTTP error, malformed payload, parse failure)."""

    def __init__(self, message: str, *, provider: Optional[str] = None, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class RateLimited(ProviderError):
    """A provider signalled that we are exceeding its rate limit.

    ``retry_after`` is the suggested wait in seconds when the provider supplies it.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message, provider=provider, status_code=429)
        self.retry_after = retry_after


__all__ = [
    "ConvexityError",
    "DataUnavailable",
    "NotSupported",
    "ProviderError",
    "RateLimited",
]
