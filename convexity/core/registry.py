"""Self-registration registries for providers and analyzers.

Concrete provider and analyzer classes decorate themselves with
:func:`register_provider` / :func:`register_analyzer`. Simply importing the module
that defines a class causes it to register, so the pipeline can discover every
implementation without an explicit manifest. Analyzers are keyed by their
:class:`~convexity.core.models.ScoreCategory` (one analyzer per category);
providers are keyed by their ``name``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from convexity.core.contracts import Analyzer, DataProvider
from convexity.core.exceptions import ConvexityError
from convexity.core.logging import get_logger
from convexity.core.models import ScoreCategory

_log = get_logger(__name__)

_PROVIDERS: Dict[str, Type[DataProvider]] = {}
_ANALYZERS: Dict[ScoreCategory, Type[Analyzer]] = {}


def register_provider(cls: Type[DataProvider]) -> Type[DataProvider]:
    """Class decorator that registers a :class:`DataProvider` by its ``name``.

    The provider is instantiated once (no args) to read its ``name`` property.
    Re-registering the same name overwrites the prior entry and logs a warning.
    """
    try:
        name = cls().name  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - defensive
        raise ConvexityError(f"cannot register provider {cls!r}: {exc}") from exc
    if name in _PROVIDERS and _PROVIDERS[name] is not cls:
        _log.warning("provider name %r re-registered; overwriting", name)
    _PROVIDERS[name] = cls
    _log.debug("registered provider %r -> %s", name, cls.__name__)
    return cls


def register_analyzer(cls: Type[Analyzer]) -> Type[Analyzer]:
    """Class decorator that registers an :class:`Analyzer` by its ``category``."""
    category = getattr(cls, "category", None)
    if not isinstance(category, ScoreCategory):
        raise ConvexityError(
            f"cannot register analyzer {cls.__name__}: missing/invalid 'category' class attr"
        )
    if category in _ANALYZERS and _ANALYZERS[category] is not cls:
        _log.warning("analyzer category %s re-registered; overwriting", category.value)
    _ANALYZERS[category] = cls
    _log.debug("registered analyzer %s -> %s", category.value, cls.__name__)
    return cls


def get_providers() -> List[Type[DataProvider]]:
    """Return all registered provider classes (insertion order)."""
    return list(_PROVIDERS.values())


def get_provider(name: str) -> Optional[Type[DataProvider]]:
    """Return the provider class registered under ``name``, or ``None``."""
    return _PROVIDERS.get(name)


def get_analyzers() -> List[Type[Analyzer]]:
    """Return all registered analyzer classes (insertion order)."""
    return list(_ANALYZERS.values())


def get_analyzer(category: ScoreCategory) -> Optional[Type[Analyzer]]:
    """Return the analyzer class for ``category``, or ``None`` if unregistered."""
    return _ANALYZERS.get(category)


def clear() -> None:
    """Empty both registries (primarily for test isolation)."""
    _PROVIDERS.clear()
    _ANALYZERS.clear()


__all__ = [
    "register_provider",
    "register_analyzer",
    "get_providers",
    "get_provider",
    "get_analyzers",
    "get_analyzer",
    "clear",
]
