"""Concrete data providers package.

Importing this package imports every concrete provider module, and each module's
``@register_provider`` decorator self-registers its class in
:mod:`convexity.core.registry`. This is the single import the aggregator performs
to discover all available data sources, so adding a new provider only requires
adding its module import here (no central manifest to maintain).

Each provider is honest about what it can supply: it advertises only the
capabilities it genuinely fills, marks missing data rather than fabricating it,
and (where applicable) reports itself unavailable when a required credential is
absent so the composite skips it cleanly.
"""

from __future__ import annotations

# Importing each module triggers its @register_provider side effect. The imports
# are wrapped individually so that a broken or dependency-missing provider module
# cannot prevent the others (and the registry) from loading.
from convexity.core.logging import get_logger

_log = get_logger(__name__)

try:
    from convexity.data.providers import yfinance_provider  # noqa: F401
except Exception as exc:  # pragma: no cover - defensive import isolation
    _log.warning("could not import yfinance provider: %s", exc)

try:
    from convexity.data.providers import sec_edgar  # noqa: F401
except Exception as exc:  # pragma: no cover - defensive import isolation
    _log.warning("could not import sec_edgar provider: %s", exc)

try:
    from convexity.data.providers import fmp  # noqa: F401
except Exception as exc:  # pragma: no cover - defensive import isolation
    _log.warning("could not import fmp provider: %s", exc)

__all__ = ["yfinance_provider", "sec_edgar", "fmp"]
