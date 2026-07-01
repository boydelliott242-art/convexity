"""Abstract base classes and protocols every downstream module implements.

These are the seams of Convexity. Data providers, analyzers, the ranking engine
and the explainability engine all conform to the contracts defined here, which is
what lets the pipeline compose them without knowing their concrete types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from typing_extensions import Protocol, runtime_checkable

from convexity.core.exceptions import NotSupported
from convexity.core.models import (
    CompanyAnalysis,
    ScanParams,
    ScanResult,
    ScoreCategory,
    SecurityData,
    SubScore,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from convexity.core.config import Settings


# ---------------------------------------------------------------------------
# Analysis context shared with every analyzer
# ---------------------------------------------------------------------------


@dataclass
class AnalysisContext:
    """Comparative context handed to every analyzer's ``analyze`` call.

    Analyzers use this to score a security *relative to its peers and the wider
    screened universe* rather than on absolute thresholds alone — essential for
    micro-caps where what counts as "cheap" or "fast-growing" is sector-relative.

    Attributes:
        peer_stats: Optional mapping of metric-name to a distribution / summary of
            that metric across the security's peers (shape is analyzer-defined,
            e.g. ``{"ev_ebitda": [6.1, 8.0, 12.4]}``).
        universe_stats: Optional mapping of the same kind across the whole
            screened universe.
        config: The active :class:`~convexity.core.config.Settings`.
    """

    peer_stats: Optional[Dict[str, Any]] = None
    universe_stats: Optional[Dict[str, Any]] = None
    config: Optional[Settings] = None
    extras: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data providers
# ---------------------------------------------------------------------------


class DataProvider(ABC):
    """Source of raw security data (prices, fundamentals, filings, news, …).

    A provider advertises a set of capability strings (e.g. ``"universe"``,
    ``"fundamentals"``, ``"prices"``, ``"news"``, ``"filings"``, ``"insider"``,
    ``"institutional"``). The aggregator inspects :attr:`capabilities` to decide
    which providers to ask for which data, and tolerates a provider raising
    :class:`~convexity.core.exceptions.NotSupported` for anything it cannot do.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, stable identifier recorded in ``SecurityData.data_sources``."""

    @property
    @abstractmethod
    def capabilities(self) -> Set[str]:
        """The set of capability strings this provider supports."""

    def supports(self, capability: str) -> bool:
        """Return whether this provider advertises ``capability``."""
        return capability in self.capabilities

    def get_universe(self, params: ScanParams) -> List[str]:
        """Return candidate tickers for the screen.

        Providers that cannot enumerate a universe must leave this default in
        place, which raises :class:`~convexity.core.exceptions.NotSupported`.
        """
        raise NotSupported(f"{self.name} does not support universe enumeration")

    @abstractmethod
    def get_security_data(self, ticker: str) -> SecurityData:
        """Fetch and assemble whatever this provider can supply for ``ticker``.

        Implementations must never fabricate values: unknown fields stay ``None``
        and gaps are appended to ``SecurityData.data_warnings``. On a hard failure
        raise :class:`~convexity.core.exceptions.ProviderError` (or
        :class:`~convexity.core.exceptions.DataUnavailable` for an expected gap).
        """


# ---------------------------------------------------------------------------
# Analyzers
# ---------------------------------------------------------------------------


class Analyzer(ABC):
    """Scores one :class:`ScoreCategory` for a security, producing a SubScore.

    Concrete analyzers set the three class attributes below and implement
    :meth:`analyze`. They should be pure with respect to their inputs (no I/O, no
    wall-clock) so scores are reproducible.

    Class attributes:
        category: The category this analyzer is responsible for.
        default_weight: Suggested weight if config supplies none.
        requires: The set of capability/data-field names the analyzer needs to
            produce a confident score (used to compute ``data_coverage`` and to
            decide when to fall back to :meth:`neutral_subscore`).
    """

    category: ScoreCategory
    default_weight: float = 0.0
    requires: Set[str] = set()

    @abstractmethod
    def analyze(self, data: SecurityData, ctx: AnalysisContext) -> SubScore:
        """Return the :class:`SubScore` for :attr:`category` given ``data``."""

    def neutral_subscore(
        self,
        rationale: str = "Insufficient data to score this category.",
        *,
        weight: Optional[float] = None,
        coverage: float = 0.0,
        extra_flags: Optional[List[str]] = None,
    ) -> SubScore:
        """Build a low-confidence, neutral (50) sub-score for missing data.

        Used when the inputs in :attr:`requires` are absent. The score sits at the
        neutral midpoint so a data gap neither helps nor hurts the company, while
        the low confidence and ``MISSING_DATA`` flag make the gap auditable.
        """
        flags = ["MISSING_DATA"]
        if extra_flags:
            flags.extend(extra_flags)
        return SubScore(
            category=self.category,
            score=50.0,
            confidence=0.1,
            weight=self.default_weight if weight is None else weight,
            rationale=rationale,
            evidence=[],
            flags=flags,
            data_coverage=coverage,
        )


# ---------------------------------------------------------------------------
# Ranking & explainability protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class RankingEngine(Protocol):
    """Combines per-company sub-scores into ranked :class:`CompanyAnalysis` rows."""

    def rank(
        self,
        analyses: List[CompanyAnalysis],
        params: ScanParams,
    ) -> List[CompanyAnalysis]:
        """Assign composite scores and integer ranks; return sorted best-first."""
        ...

    def score_company(
        self,
        data: SecurityData,
        subscores: List[SubScore],
        weights: Dict[ScoreCategory, float],
    ) -> CompanyAnalysis:
        """Fold a security's sub-scores into a single scored CompanyAnalysis."""
        ...


@runtime_checkable
class ExplainabilityEngine(Protocol):
    """Turns a scored :class:`CompanyAnalysis` into human-readable narrative."""

    def explain(self, analysis: CompanyAnalysis, data: SecurityData) -> CompanyAnalysis:
        """Populate thesis, bull/bear cases, summaries and the monitoring list."""
        ...


@runtime_checkable
class PipelineProtocol(Protocol):
    """The end-to-end scan: universe -> screen -> fetch -> analyze -> rank -> explain."""

    def scan(self, params: ScanParams) -> ScanResult:
        """Execute a full scan and return the assembled :class:`ScanResult`."""
        ...


__all__ = [
    "AnalysisContext",
    "DataProvider",
    "Analyzer",
    "RankingEngine",
    "ExplainabilityEngine",
    "PipelineProtocol",
]
