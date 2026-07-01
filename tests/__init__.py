"""Test suite for Convexity.

Every test in this package is **network-free and deterministic**: data comes from
the synthetic :class:`~tests.conftest.FakeProvider` (see ``conftest.py``), never
from a live provider, so the whole suite runs offline and reproducibly. This mirrors
Convexity's own honesty contract — the tests assert that the pipeline aggregates
*independent* evidence, that missing data lowers (never inflates) confidence, and
that narratives only restate evidence that was actually computed.
"""

from __future__ import annotations
