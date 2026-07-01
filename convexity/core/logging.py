"""Centralised, structured logging for Convexity.

Every module obtains its logger via :func:`get_logger` so that formatting and the
log level are configured in exactly one place. The level is read once from the
``CONVEXITY_LOG_LEVEL`` environment variable (default ``INFO``).

The formatter emits a compact, single-line, key-aligned record that is easy to
grep while a scan streams across hundreds of tickers::

    2026-06-29 12:00:00 | INFO     | convexity.pipeline | scanning universe size=1834
"""

from __future__ import annotations

import logging
import os
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Guard so that repeated imports / get_logger calls do not attach duplicate handlers.
_CONFIGURED = False
_ROOT_NAME = "convexity"


def _coerce_level(value: Optional[str]) -> int:
    """Translate an env string (name or number) into a logging level int."""
    if not value:
        return logging.INFO
    value = value.strip()
    if value.isdigit():
        return int(value)
    return getattr(logging, value.upper(), logging.INFO)


def _configure() -> None:
    """Attach a single stream handler to the ``convexity`` root logger once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = _coerce_level(os.environ.get("CONVEXITY_LOG_LEVEL"))
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)

    # Do not propagate to the python root logger to avoid duplicate lines when an
    # embedding application has also configured logging.
    root.propagate = False

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
        handler.setLevel(level)
        root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger namespaced under ``convexity``.

    Args:
        name: Usually ``__name__`` of the calling module. A bare name (e.g.
            ``"pipeline"``) is automatically namespaced under ``convexity`` so all
            project loggers share one configured parent.

    Returns:
        A :class:`logging.Logger` that inherits the project handler and level.
    """
    _configure()
    if name == _ROOT_NAME or name.startswith(_ROOT_NAME + "."):
        full_name = name
    elif name == "__main__":
        full_name = _ROOT_NAME
    else:
        full_name = f"{_ROOT_NAME}.{name}"
    return logging.getLogger(full_name)


def set_level(level: str) -> None:
    """Override the log level at runtime (used by the CLI ``--verbose`` flag)."""
    _configure()
    lvl = _coerce_level(level)
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(lvl)
    for handler in root.handlers:
        handler.setLevel(lvl)


__all__ = ["get_logger", "set_level"]
