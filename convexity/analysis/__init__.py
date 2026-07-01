"""Analyzer modules: each produces an independent, auditable SubScore for a category.

Importing this package self-registers each analyzer with
:mod:`convexity.core.registry` (one analyzer per :class:`ScoreCategory`), so the
pipeline can discover implementations simply by importing ``convexity.analysis``.

Each analyzer is intentionally narrow and independent — the platform's conviction
signal only emerges when *many* of these independent views agree. No single module
should ever be read as a recommendation on its own.
"""

from __future__ import annotations

# Importing each module triggers its ``@register_analyzer`` decorator.
from convexity.analysis import (
    catalysts,  # noqa: F401  (CatalystAnalyzer)
    competitive,  # noqa: F401  (CompetitiveAnalyzer)
    financial_health,  # noqa: F401  (FinancialHealthAnalyzer)
    growth,  # noqa: F401  (GrowthAnalyzer)
    historical_analog,  # noqa: F401  (HistoricalAnalogAnalyzer)
    management,  # noqa: F401  (ManagementAnalyzer)
    momentum,  # noqa: F401  (MomentumAnalyzer)
    ownership,  # noqa: F401  (OwnershipAnalyzer)
    quality,  # noqa: F401  (QualityAnalyzer)
    risk,  # noqa: F401  (RiskAnalyzer)
    technical,  # noqa: F401  (TechnicalAnalyzer)
    value,  # noqa: F401  (ValueAnalyzer)
)

__all__ = [
    "catalysts",
    "competitive",
    "financial_health",
    "growth",
    "historical_analog",
    "management",
    "momentum",
    "ownership",
    "quality",
    "risk",
    "technical",
    "value",
]
