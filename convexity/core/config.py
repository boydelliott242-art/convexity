"""Application configuration and the default category-weighting scheme.

Settings are loaded once (cached) from environment variables and an optional
``.env`` file via pydantic-settings. Secrets (API keys) are optional so the tool
runs in a degraded-but-honest mode when a provider key is absent: the relevant
data is simply marked missing and the affected sub-scores lower their confidence.

Weighting scheme
----------------
``DEFAULT_CATEGORY_WEIGHTS`` maps the eleven *additive* categories to weights that
sum to 1.0. The fundamentals-driven categories (VALUE, GROWTH, QUALITY,
FINANCIAL_HEALTH, CATALYST) carry the most weight because they rest on the most
durable, independently verifiable evidence. The remaining additive categories
(TECHNICAL, MOMENTUM, MANAGEMENT, COMPETITIVE, OWNERSHIP, HISTORICAL_ANALOG) get
smaller weights.

RISK is deliberately given weight ``0.0`` in this additive map: it is **not**
averaged into the composite. Instead the ranking layer applies RISK as a
penalty/dampener (see :func:`convexity.core.scoring.combine_subscores`). Keeping
it out of the additive weights — while still listing it — documents the scheme
explicitly and keeps the additive weights summing to exactly 1.0.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from convexity.core.models import ScoreCategory

# Additive weights for the eleven categories that contribute to the composite
# mean. RISK is listed with 0.0 because it is applied as a dampener, not a term
# in the mean. The eleven non-risk weights sum to 1.0.
DEFAULT_CATEGORY_WEIGHTS: Dict[ScoreCategory, float] = {
    # Highest-conviction, durable fundamental evidence.
    ScoreCategory.VALUE: 0.16,
    ScoreCategory.GROWTH: 0.15,
    ScoreCategory.QUALITY: 0.14,
    ScoreCategory.FINANCIAL_HEALTH: 0.13,
    ScoreCategory.CATALYST: 0.10,
    # Secondary, supporting evidence.
    ScoreCategory.COMPETITIVE: 0.07,
    ScoreCategory.MANAGEMENT: 0.06,
    ScoreCategory.OWNERSHIP: 0.06,
    ScoreCategory.MOMENTUM: 0.05,
    ScoreCategory.TECHNICAL: 0.04,
    ScoreCategory.HISTORICAL_ANALOG: 0.04,
    # Applied as a dampener/penalty by the ranking layer, not averaged in.
    ScoreCategory.RISK: 0.0,
}


class Settings(BaseSettings):
    """Runtime configuration sourced from the environment and ``.env``.

    Environment variables are matched case-insensitively. Provider API keys are
    optional; absent keys degrade gracefully rather than raising.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # Where on-disk caches live (price history, fundamentals, HTTP responses).
    data_dir: str = "./.convexity_data"

    # A descriptive User-Agent is mandatory for the SEC; keep it identifiable.
    sec_user_agent: str = "Convexity research tool (contact: set SEC_USER_AGENT)"

    # Optional provider credentials.
    fmp_api_key: Optional[str] = None
    alphavantage_api_key: Optional[str] = None

    # HTTP behaviour.
    request_timeout: float = 20.0

    # Cache freshness in seconds (default 12 hours).
    cache_ttl_seconds: int = 43_200

    def category_weights(self) -> Dict[ScoreCategory, float]:
        """Return a copy of the default additive category-weighting map."""
        return dict(DEFAULT_CATEGORY_WEIGHTS)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance."""
    return Settings()


__all__ = ["Settings", "get_settings", "DEFAULT_CATEGORY_WEIGHTS"]
