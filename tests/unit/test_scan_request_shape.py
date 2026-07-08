"""Regression: POST /api/scans must honor UI-chosen parameters in either shape.

The dashboard once posted screen fields flat while the API expected them nested
under "params"; with extra="ignore" every UI choice (cap band, top_n,
universe_limit) was silently discarded and scans ran on pure defaults.
"""

from __future__ import annotations

from convexity.api.schemas import CreateScanRequest


def test_flat_body_is_lifted_into_params() -> None:
    req = CreateScanRequest.model_validate(
        {"min_market_cap": 1e7, "max_market_cap": 2e9, "top_n": 25, "universe_limit": 20}
    )
    assert req.params.top_n == 25
    assert req.params.universe_limit == 20
    assert req.params.min_market_cap == 1e7


def test_canonical_nested_body_still_works() -> None:
    req = CreateScanRequest.model_validate({"params": {"top_n": 9, "universe_limit": None}})
    assert req.params.top_n == 9
    assert req.params.universe_limit is None


def test_empty_body_uses_defaults() -> None:
    req = CreateScanRequest.model_validate({})
    assert req.params.top_n == 5


def test_unknown_fields_still_ignored() -> None:
    req = CreateScanRequest.model_validate({"bogus": 1, "top_n": 7})
    assert req.params.top_n == 7
