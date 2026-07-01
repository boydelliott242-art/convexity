"""Integration tests for Convexity's end-to-end scan and HTTP API.

These wire the real :class:`~convexity.pipeline.ScanPipeline` (and the real FastAPI
app) to the synthetic :class:`~tests.conftest.FakeProvider`, so a full
universe -> screen -> fetch -> analyze -> rank -> explain run — and the API that
drives it — are exercised **offline and deterministically**. They assert the scan
produces a well-formed, reproducible :class:`~convexity.core.models.ScanResult` and
that the API can start a scan, be polled to completion, and serve a single company.
"""

from __future__ import annotations
