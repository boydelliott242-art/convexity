"""Unit tests for Convexity's pure, deterministic core.

These exercise the scoring math (:mod:`convexity.core.scoring`), the ranking engine
(:mod:`convexity.ranking.engine`) and the explainability engine
(:mod:`convexity.ranking.explain`) in isolation, with hand-built sub-scores and
securities. They run with no network and assert the honesty guarantees directly:
all-missing data yields a neutral, low-confidence read; conviction rises only with
breadth of *independent* agreement and data coverage; and narrative restates only
attached evidence.
"""

from __future__ import annotations
