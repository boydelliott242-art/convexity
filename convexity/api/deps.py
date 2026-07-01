"""Shared, lazily-constructed dependencies for the HTTP API.

The FastAPI routes need a single :class:`~convexity.pipeline.ScanPipeline` instance
(building one wires up the default composite data provider, the ranking engine and
the explainability engine). Constructing it is comparatively expensive and touches
provider discovery, so it must **not** happen at import time — importing the API
package has to stay side-effect-free and runnable with no network. This module
therefore builds the pipeline lazily on first use and caches it.
"""

from __future__ import annotations

import threading
from typing import Optional

from convexity.pipeline import ScanPipeline

_PIPELINE: Optional[ScanPipeline] = None
_PIPELINE_LOCK = threading.Lock()


def get_pipeline() -> ScanPipeline:
    """Return the process-wide :class:`ScanPipeline`, building it on first use.

    The pipeline is constructed lazily and memoised so every request reuses one
    instance (the pipeline holds no per-scan mutable state, so reuse is safe and
    avoids re-discovering providers on each call). Import of this module triggers
    no construction, keeping the API import-safe.
    """
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                _PIPELINE = ScanPipeline()
    return _PIPELINE


__all__ = ["get_pipeline"]
