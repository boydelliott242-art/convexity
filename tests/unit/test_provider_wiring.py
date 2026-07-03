"""Regression tests: provider construction wiring and SEC share-count fallback.

Two production incidents these lock in:

1. ``CompositeProvider._instantiate`` used to call ``cls(settings)`` positionally.
   For ``YFinanceProvider(cache=None)`` the Settings object landed in the *cache*
   slot, every cache call failed (``'Settings' object has no attribute
   'get_data'``) and the provider silently never cached anything. Instantiation
   is now signature-aware (settings forwarded only to a parameter literally
   named ``settings``) and the yfinance ``cache`` parameter is keyword-only.

2. Smaller SEC filers (e.g. VTVT) tag no weighted-average share series in
   companyfacts — only the point-in-time ``CommonStockSharesOutstanding``. The
   provider now falls back to instant share counts so the derived-market-cap
   path has real shares to work with.
"""

from __future__ import annotations

from convexity.core.config import Settings
from convexity.data.aggregator import CompositeProvider
from convexity.data.cache import Cache
from convexity.data.providers.sec_edgar import SecEdgarProvider
from convexity.data.providers.yfinance_provider import YFinanceProvider


def test_yfinance_cache_param_is_keyword_only() -> None:
    """Positional construction must fail — that is what shielded the original bug."""
    try:
        YFinanceProvider(object())  # type: ignore[misc]
    except TypeError:
        pass
    else:  # pragma: no cover - the regression itself
        raise AssertionError("YFinanceProvider accepted a positional arg into the cache slot")


def test_instantiate_never_puts_settings_in_wrong_slot() -> None:
    """Signature-aware instantiation: yfinance gets NO settings, sec_edgar gets them by name."""
    settings = Settings()
    yf = CompositeProvider._instantiate(YFinanceProvider, settings)
    assert yf is not None
    # The resolved cache must be a real Cache (or None), never the Settings object.
    resolved = yf._resolve_cache()
    assert resolved is None or isinstance(resolved, Cache)
    assert not isinstance(yf._cache, Settings)

    sec = CompositeProvider._instantiate(SecEdgarProvider, settings)
    assert sec is not None


def test_instantiate_handles_settings_free_constructors() -> None:
    class NoArgs:
        def __init__(self) -> None:
            self.built = True

    inst = CompositeProvider._instantiate(NoArgs, Settings())
    assert inst is not None and inst.built


def _facts(gaap: dict) -> dict:
    return {"facts": {"us-gaap": gaap}}


def _instant_fact(end: str, val: float) -> dict:
    return {"end": end, "val": val, "form": "10-K", "fp": "FY", "fy": 2025, "filed": "2026-02-01"}


def _flow_fact(start: str, end: str, val: float) -> dict:
    return {"start": start, "end": end, "val": val, "form": "10-K", "fp": "FY", "fy": 2025, "filed": "2026-02-01"}


def test_instant_shares_outstanding_fallback() -> None:
    """An issuer with only CommonStockSharesOutstanding still yields shares_diluted."""
    provider = SecEdgarProvider()
    gaap = {
        "Revenues": {"units": {"USD": [_flow_fact("2025-01-01", "2025-12-31", 1_000_000.0)]}},
        "CommonStockSharesOutstanding": {"units": {"shares": [_instant_fact("2025-12-31", 14_634_420.0)]}},
    }
    periods = provider._extract_fundamentals(_facts(gaap))
    assert periods, "expected at least one period"
    assert periods[0].shares_diluted == 14_634_420.0


def test_weighted_average_preferred_over_instant() -> None:
    """When both series exist, the weighted-average (flow) figure wins."""
    provider = SecEdgarProvider()
    gaap = {
        "Revenues": {"units": {"USD": [_flow_fact("2025-01-01", "2025-12-31", 1_000_000.0)]}},
        "WeightedAverageNumberOfDilutedSharesOutstanding": {
            "units": {"shares": [_flow_fact("2025-01-01", "2025-12-31", 14_000_000.0)]}
        },
        "CommonStockSharesOutstanding": {"units": {"shares": [_instant_fact("2025-12-31", 15_000_000.0)]}},
    }
    periods = provider._extract_fundamentals(_facts(gaap))
    assert periods and periods[0].shares_diluted == 14_000_000.0
