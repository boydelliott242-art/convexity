"""A thin, honest disk cache for provider responses and JSON blobs.

Convexity fetches the same micro-cap data (prices, fundamentals, filings, news)
repeatedly across a scan and across re-runs. This module memoises those fetches on
disk so a scan is fast and so we are polite to the upstream providers (especially
the SEC, which expects modest request rates). It is *only* a freshness-bounded
cache: it never invents data, and an expired or missing entry simply triggers a
real re-fetch by the caller.

Honesty note
------------
Caching changes *when* we fetch, never *what* we report. Every cached value was a
real provider response at write time; the ``ttl`` (default
``Settings.cache_ttl_seconds``) bounds how stale a reused value may be. Callers
that care about staleness can inspect the stored timestamp via
:meth:`Cache.get_entry`. Nothing here fabricates or extrapolates a missing datum.

Graceful degradation
---------------------
The cache prefers :mod:`diskcache` for a robust, process-safe on-disk store. If
``diskcache`` cannot be imported (or its directory cannot be opened), the cache
falls back to a process-local in-memory dictionary so the rest of the system keeps
working — just without cross-process or cross-run persistence. The fallback is
logged once at WARNING so the degraded mode is auditable, never silent.

Keying
------
Entries are keyed by the triple ``(provider, ticker, kind)`` — for example
``("yfinance", "ABEO", "prices")`` — normalised to a stable string. The same
triple always maps to the same slot, so a provider can overwrite its own prior
value for a ticker/kind without colliding with another provider.
"""

from __future__ import annotations

import json
import os
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

from convexity.core.config import Settings, get_settings
from convexity.core.logging import get_logger

_log = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Sentinel distinguishing "key absent / expired" from a genuinely cached ``None``.
_MISSING = object()


# ---------------------------------------------------------------------------
# Optional diskcache backend (degrades gracefully if unavailable)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import availability is environment-dependent
    import diskcache as _diskcache  # type: ignore

    _DISKCACHE_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - exercised only without diskcache
    _diskcache = None  # type: ignore[assignment]
    _DISKCACHE_AVAILABLE = False
    _log.warning(
        "diskcache unavailable (%s); falling back to an in-memory cache "
        "(no cross-process or cross-run persistence)",
        _exc,
    )


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------


def make_key(provider: str, ticker: str, kind: str) -> str:
    """Build the stable cache key for a ``(provider, ticker, kind)`` triple.

    The pieces are lower-cased and stripped so that e.g. ``"YFinance"`` / ``" AAPL "``
    and ``"yfinance"`` / ``"aapl"`` resolve to the same slot. The ``|`` separator and
    the per-piece sanitisation keep keys unambiguous and filesystem-safe.

    Args:
        provider: The data provider's stable id (``DataProvider.name``).
        ticker: The security symbol.
        kind: The data kind, e.g. ``"prices"``, ``"fundamentals"``, ``"news"``.

    Returns:
        A normalised key string such as ``"yfinance|abeo|prices"``.
    """

    def _norm(piece: str) -> str:
        cleaned = str(piece).strip().lower()
        # Collapse the reserved separator so it can never appear inside a piece.
        return cleaned.replace("|", "_") or "_"

    return f"{_norm(provider)}|{_norm(ticker)}|{_norm(kind)}"


# ---------------------------------------------------------------------------
# Cache entry envelope
# ---------------------------------------------------------------------------


class _Entry:
    """An in-memory cache record carrying its value and write time.

    ``stored_at`` is wall-clock seconds at write time; expiry is computed against
    it at read time. Scores never depend on this timestamp — it is bookkeeping for
    freshness only — so cache use does not break the analyzers' determinism.
    """

    __slots__ = ("value", "stored_at", "ttl")

    def __init__(self, value: Any, stored_at: float, ttl: int | float | None) -> None:
        self.value = value
        self.stored_at = stored_at
        self.ttl = ttl

    def is_expired(self, now: float) -> bool:
        """Return whether this entry has outlived its TTL as of ``now``."""
        if self.ttl is None or self.ttl <= 0:
            return False
        return (now - self.stored_at) >= self.ttl


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


class Cache:
    """A freshness-bounded key/value cache over diskcache, with a memory fallback.

    The cache is keyed by opaque strings (build them with :func:`make_key`) and
    stores arbitrary JSON-serialisable values. Each write records a timestamp so a
    read can honour the configured TTL. If :mod:`diskcache` is unavailable the
    cache transparently uses a thread-safe in-memory dictionary instead; behaviour
    is identical apart from the loss of persistence.

    Args:
        settings: Active settings; defaults to :func:`get_settings`. Provides
            ``data_dir`` (the cache lives under ``data_dir/cache``) and
            ``cache_ttl_seconds`` (the default TTL).
        namespace: Subdirectory under ``data_dir/cache`` isolating this cache's
            entries (useful for tests and for separating blob stores).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        namespace: str = "providers",
    ) -> None:
        self._settings = settings or get_settings()
        self._namespace = namespace
        self._default_ttl = int(self._settings.cache_ttl_seconds)
        self._lock = threading.Lock()
        self._mem: dict[str, _Entry] = {}
        self._disk: Any | None = None
        self._directory = self._resolve_directory()
        self._open_backend()

    # -- backend lifecycle --------------------------------------------------

    def _resolve_directory(self) -> Path:
        """Return ``<data_dir>/cache/<namespace>`` as an absolute path."""
        base = Path(self._settings.data_dir).expanduser()
        return (base / "cache" / self._namespace).resolve()

    def _open_backend(self) -> None:
        """Open the diskcache backend, falling back to memory on any failure."""
        if not _DISKCACHE_AVAILABLE:
            self._disk = None
            return
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            # ``diskcache.Cache`` is process- and thread-safe and stores entries in
            # an SQLite-backed directory under ``self._directory``.
            self._disk = _diskcache.Cache(str(self._directory))
            _log.debug("disk cache open at %s", self._directory)
        except Exception as exc:  # pragma: no cover - filesystem-dependent
            self._disk = None
            _log.warning(
                "could not open disk cache at %s (%s); using in-memory fallback",
                self._directory,
                exc,
            )

    @property
    def backend(self) -> str:
        """Return ``"disk"`` if persistent storage is active, else ``"memory"``."""
        return "disk" if self._disk is not None else "memory"

    @property
    def directory(self) -> Path:
        """The on-disk directory backing this cache (created lazily)."""
        return self._directory

    # -- core get / set -----------------------------------------------------

    def _effective_ttl(self, ttl: int | None) -> int | None:
        """Resolve the TTL to use: explicit ``ttl`` else the configured default."""
        return self._default_ttl if ttl is None else ttl

    def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        """Store ``value`` under ``key`` with an optional ``ttl`` override.

        Args:
            key: The cache key (see :func:`make_key`).
            value: A JSON-serialisable value (the same constraint diskcache and the
                blob helpers share). Non-serialisable values are rejected up front
                so a cache write can never silently corrupt the store.
            ttl: Time-to-live in seconds; ``None`` uses ``cache_ttl_seconds``. A
                value ``<= 0`` means "never expire".
        """
        # Validate serialisability eagerly: it keeps the disk and memory backends
        # behaviourally identical and surfaces programmer error at the call site.
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"cache value for {key!r} is not JSON-serialisable: {exc}") from exc

        eff_ttl = self._effective_ttl(ttl)
        now = time.time()
        with self._lock:
            if self._disk is not None:
                try:
                    # diskcache treats expire<=0/None as "no expiry"; normalise ours.
                    expire = None if (eff_ttl is None or eff_ttl <= 0) else float(eff_ttl)
                    self._disk.set(key, {"v": value, "t": now}, expire=expire)
                    return
                except Exception as exc:  # pragma: no cover - defensive
                    _log.warning("disk cache set failed for %r (%s); using memory", key, exc)
                    self._disk = None
            self._mem[key] = _Entry(value, now, eff_ttl)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the cached value for ``key`` if present and fresh, else ``default``.

        An expired entry is treated as a miss (and removed) so the caller re-fetches
        live data; this is what bounds how stale a reused value can be.
        """
        entry = self.get_entry(key)
        if entry is _MISSING:
            return default
        return entry["value"]  # type: ignore[index]

    def get_entry(self, key: str) -> Any:
        """Return ``{"value", "stored_at", "age"}`` for a fresh hit, else ``_MISSING``.

        Exposes the write timestamp so freshness-sensitive callers can decide
        whether a value is recent enough for their purpose, independent of the TTL.
        """
        now = time.time()
        with self._lock:
            if self._disk is not None:
                try:
                    raw = self._disk.get(key, default=_MISSING)
                except Exception as exc:  # pragma: no cover - defensive
                    _log.warning("disk cache get failed for %r (%s)", key, exc)
                    raw = _MISSING
                if raw is _MISSING:
                    return _MISSING
                # diskcache enforces expiry itself; a returned value is fresh.
                if isinstance(raw, dict) and "v" in raw:
                    stored_at = float(raw.get("t", now))
                    return {"value": raw["v"], "stored_at": stored_at, "age": max(0.0, now - stored_at)}
                # Legacy / unexpected shape: surface the raw value rather than crash.
                return {"value": raw, "stored_at": now, "age": 0.0}

            entry = self._mem.get(key)
            if entry is None:
                return _MISSING
            if entry.is_expired(now):
                # Evict lazily so the memory store does not grow without bound.
                self._mem.pop(key, None)
                return _MISSING
            return {
                "value": entry.value,
                "stored_at": entry.stored_at,
                "age": max(0.0, now - entry.stored_at),
            }

    # -- triple-keyed convenience ------------------------------------------

    def get_data(self, provider: str, ticker: str, kind: str, default: Any = None) -> Any:
        """Convenience ``get`` keyed by the ``(provider, ticker, kind)`` triple."""
        return self.get(make_key(provider, ticker, kind), default)

    def set_data(
        self,
        provider: str,
        ticker: str,
        kind: str,
        value: Any,
        *,
        ttl: int | None = None,
    ) -> None:
        """Convenience ``set`` keyed by the ``(provider, ticker, kind)`` triple."""
        self.set(make_key(provider, ticker, kind), value, ttl=ttl)

    # -- maintenance --------------------------------------------------------

    def delete(self, key: str) -> bool:
        """Remove ``key`` from the cache. Returns whether anything was removed."""
        with self._lock:
            if self._disk is not None:
                try:
                    return bool(self._disk.delete(key))
                except Exception as exc:  # pragma: no cover - defensive
                    _log.warning("disk cache delete failed for %r (%s)", key, exc)
                    return False
            return self._mem.pop(key, _MISSING) is not _MISSING

    def clear(self) -> None:
        """Empty the cache (primarily for test isolation and forced refreshes)."""
        with self._lock:
            self._mem.clear()
            if self._disk is not None:
                try:
                    self._disk.clear()
                except Exception as exc:  # pragma: no cover - defensive
                    _log.warning("disk cache clear failed (%s)", exc)

    def close(self) -> None:
        """Release the disk backend's resources, if any. Safe to call repeatedly."""
        with self._lock:
            if self._disk is not None:
                try:
                    self._disk.close()
                except Exception as exc:  # pragma: no cover - defensive
                    _log.debug("disk cache close raised (%s)", exc)

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Process-wide default cache
# ---------------------------------------------------------------------------

_DEFAULT_CACHE: Cache | None = None
_DEFAULT_CACHE_LOCK = threading.Lock()


def get_cache() -> Cache:
    """Return the lazily-constructed, process-wide default :class:`Cache`.

    Most callers use this shared instance; tests that need isolation can construct
    their own :class:`Cache` with a temporary ``data_dir`` instead.
    """
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        with _DEFAULT_CACHE_LOCK:
            if _DEFAULT_CACHE is None:
                _DEFAULT_CACHE = Cache()
    return _DEFAULT_CACHE


def reset_cache() -> None:
    """Drop the process-wide default cache (closing it). Mainly for tests."""
    global _DEFAULT_CACHE
    with _DEFAULT_CACHE_LOCK:
        if _DEFAULT_CACHE is not None:
            _DEFAULT_CACHE.close()
        _DEFAULT_CACHE = None


# ---------------------------------------------------------------------------
# JSON blob helpers (read/write JSON files under data_dir/cache/blobs)
# ---------------------------------------------------------------------------


def _blob_dir(settings: Settings | None = None) -> Path:
    """Return ``<data_dir>/cache/blobs``, creating it if necessary."""
    cfg = settings or get_settings()
    path = (Path(cfg.data_dir).expanduser() / "cache" / "blobs").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _blob_path(name: str, settings: Settings | None = None) -> Path:
    """Resolve a safe ``.json`` path for blob ``name`` under the blob directory.

    ``name`` is sanitised to a flat filename so a blob can never escape the blob
    directory (no path traversal via ``..`` or absolute paths).
    """
    base = _blob_dir(settings)
    safe = str(name).strip().replace("/", "_").replace("\\", "_").replace("..", "_")
    safe = safe or "blob"
    if not safe.endswith(".json"):
        safe = f"{safe}.json"
    return base / safe


def write_json_blob(name: str, payload: Any, *, settings: Settings | None = None) -> Path:
    """Write ``payload`` as pretty JSON to ``data_dir/cache/blobs/<name>.json``.

    The write is atomic (temp file + ``os.replace``) so a crash mid-write can never
    leave a half-written, unparseable blob behind.

    Args:
        name: Logical blob name; sanitised into a flat filename.
        payload: Any JSON-serialisable object.
        settings: Optional settings override (defaults to :func:`get_settings`).

    Returns:
        The :class:`~pathlib.Path` the blob was written to.
    """
    path = _blob_path(name, settings)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    _log.debug("wrote JSON blob %s (%d bytes)", path, len(text))
    return path


def read_json_blob(
    name: str,
    default: Any = None,
    *,
    settings: Settings | None = None,
) -> Any:
    """Read and parse ``data_dir/cache/blobs/<name>.json``.

    Returns ``default`` if the blob is absent or unparseable (a corrupt blob is a
    cache miss, never a crash). Corruption is logged so it is auditable.
    """
    path = _blob_path(name, settings)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        _log.warning("could not read JSON blob %s (%s); treating as miss", path, exc)
        return default


# ---------------------------------------------------------------------------
# The @cached decorator
# ---------------------------------------------------------------------------


def _normalise_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Render call arguments into a stable, JSON-based key fragment.

    Falls back to ``repr`` for anything not JSON-serialisable so the decorator
    never raises merely because an argument is exotic; the only requirement is that
    equal calls produce equal fragments.
    """
    try:
        return json.dumps({"a": args, "k": kwargs}, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return repr((args, sorted(kwargs.items())))


def cached(
    kind: str,
    *,
    provider: str | None = None,
    ttl: int | None = None,
    ticker_arg: int = 0,
    cache: Cache | None = None,
) -> Callable[[F], F]:
    """Memoise a per-ticker fetch function on the freshness-bounded cache.

    Intended for the I/O methods of data providers — for example a
    ``get_security_data(self, ticker)`` — so repeated calls within a scan (and
    across runs, when diskcache is available) reuse a recent response instead of
    hitting the network again.

    The cache key is the ``(provider, ticker, kind)`` triple plus a fragment of any
    *extra* arguments, so two calls that differ only in their ticker (or in extra
    args) never collide. When ``provider`` is not given it is inferred from the
    wrapped instance's ``name`` attribute (a :class:`DataProvider`), else from the
    function's qualified name.

    Args:
        kind: The data kind label for the key (e.g. ``"security_data"``).
        provider: Explicit provider id; if ``None`` it is inferred at call time.
        ttl: TTL override in seconds; ``None`` uses ``cache_ttl_seconds``.
        ticker_arg: Positional index (after ``self``-style binding is accounted for
            by the wrapped signature) of the ticker argument used in the key.
        cache: Cache instance to use; defaults to the process-wide
            :func:`get_cache`.

    Returns:
        A decorator that wraps the target callable, transparently caching its
        return value. The wrapped function exposes ``__wrapped__`` (via
        :func:`functools.wraps`) so the un-cached original remains reachable.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            store = cache or get_cache()

            # Infer the provider id: explicit > instance.name > function qualname.
            prov = provider
            self_obj = args[0] if args else None
            if prov is None:
                prov = getattr(self_obj, "name", None) or getattr(func, "__qualname__", func.__name__)

            # Locate the ticker. ``ticker_arg`` indexes the *positional* arguments
            # as the wrapped function sees them (including any leading ``self``).
            ticker = kwargs.get("ticker")
            if ticker is None and len(args) > ticker_arg:
                ticker = args[ticker_arg]
            ticker_str = str(ticker) if ticker is not None else "_"

            # Extra args beyond the ticker contribute to the key so distinct calls
            # are cached distinctly (e.g. a date range or limit).
            extra_args = tuple(a for i, a in enumerate(args) if i != ticker_arg and a is not self_obj)
            extra_kwargs = {k: v for k, v in kwargs.items() if k != "ticker"}
            suffix = _normalise_args(extra_args, extra_kwargs)
            full_kind = kind if suffix in ("{}", '{"a": [], "k": {}}') else f"{kind}:{suffix}"

            key = make_key(str(prov), ticker_str, full_kind)

            hit = store.get_entry(key)
            if hit is not _MISSING:
                _log.debug("cache hit %s", key)
                return hit["value"]  # type: ignore[index]

            result = func(*args, **kwargs)

            # Only cache JSON-serialisable results; otherwise skip caching rather
            # than crash the call (the fetch still returns its value).
            try:
                store.set(key, result, ttl=ttl)
                _log.debug("cache store %s", key)
            except TypeError:
                _log.debug(
                    "result of %s is not JSON-serialisable; not cached", getattr(func, "__name__", "fn")
                )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


__all__ = [
    "Cache",
    "cached",
    "make_key",
    "get_cache",
    "reset_cache",
    "read_json_blob",
    "write_json_blob",
]
