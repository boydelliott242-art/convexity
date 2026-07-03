"""Provider aggregation — merge every available data source into one canonical view.

Convexity treats *evidence aggregation* as its core honesty mechanism: a single
provider rarely covers a thin micro-cap completely, so :class:`CompositeProvider`
fans a request out to every registered, *available* provider and folds their
partial :class:`~convexity.core.models.SecurityData` objects into one canonical
record.

The merge is deliberately conservative and transparent:

* **Never fabricate.** A field is only filled from a provider that actually
  supplied it. When no provider supplies a field it stays ``None`` and the gap is
  recorded in ``data_warnings`` — exactly as the single-provider contract requires.
* **Prefer richer / more authoritative sources per field.** Fundamentals and
  filings from a structured regulator feed (SEC EDGAR) outrank screen-scraped
  equivalents; identity/price/news fields prefer the source that actually has a
  non-empty value, breaking ties by a small, documented source-priority order.
* **Union the list-valued evidence** (news, filings, insider transactions,
  institutional holdings, peers, price bars) across providers and de-duplicate so
  the same headline or filing from two feeds is not double-counted.
* **Accumulate provenance.** Every contributing provider is recorded in
  ``data_sources`` and every provider's warnings (plus any provider-level failure)
  are accumulated into ``data_warnings`` so the result is fully auditable.

This is a research/screening aggregation step, not a predictor: it makes the
underlying evidence explainable and traceable, it does not assert that any merged
figure is correct or that the security will move in any direction.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from convexity.core.contracts import DataProvider
from convexity.core.exceptions import (
    ConvexityError,
    DataUnavailable,
    NotSupported,
    ProviderError,
)
from convexity.core.logging import get_logger
from convexity.core.models import (
    Filing,
    FundamentalsPeriod,
    InsiderTransaction,
    InstitutionalHolding,
    NewsItem,
    PriceBar,
    ScanParams,
    SecurityData,
    ValuationSnapshot,
)
from convexity.core.registry import get_providers

# Reuse the exact cap-tier thresholds the providers already apply when they
# classify a provider-supplied market cap (nano < ~$50M, micro < ~$300M,
# small < ~$2B, >= ~$2B -> None). Importing the provider's classifier — rather
# than writing a third copy — keeps the derived-cap fallback below in lockstep
# with how ``cap_tier`` is normally assigned.
from convexity.data.providers.yfinance_provider import _classify_cap_tier

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Source-priority ordering
# ---------------------------------------------------------------------------
#
# When two providers each supply a value for the *same scalar field*, the value
# from the higher-priority (lower index) source wins. The ordering encodes which
# feed is the most authoritative for the kind of data it provides:
#
#   * ``sec_edgar`` — the regulator's own structured filings: most authoritative
#     for fundamentals and filings.
#   * ``fmp`` — a curated commercial financial-data API: strong, key-gated
#     fundamentals / valuation / ownership coverage.
#   * ``yfinance`` — a free best-effort feed: broadest identity/price/news
#     coverage, lowest authority for precise fundamentals.
#
# Any provider not listed here is treated as lowest priority (it can still fill a
# field that nobody else supplied — it simply never overrides a listed source).
_SOURCE_PRIORITY: Tuple[str, ...] = ("sec_edgar", "fmp", "yfinance")


def _priority_rank(source: str) -> int:
    """Return a sort key for ``source`` (lower == more authoritative)."""
    try:
        return _SOURCE_PRIORITY.index(source)
    except ValueError:
        # Unknown sources rank after all known ones but keep a stable order.
        return len(_SOURCE_PRIORITY)


def _is_available(provider: DataProvider) -> bool:
    """Whether ``provider`` reports itself usable.

    ``is_available`` is an *optional* part of the provider surface (only the
    key-gated providers implement it). A provider that does not define it is
    assumed available. Any exception from the check is treated as "unavailable"
    so a misbehaving provider can never crash discovery.
    """
    checker = getattr(provider, "is_available", None)
    if checker is None:
        return True
    try:
        return bool(checker())
    except Exception:  # pragma: no cover - defensive; availability must not raise
        return False


# ---------------------------------------------------------------------------
# Field-level merge helpers
# ---------------------------------------------------------------------------


def _coalesce_scalar(current: Optional[Any], candidate: Optional[Any]) -> Optional[Any]:
    """Return ``candidate`` only if ``current`` is unset (``None`` or blank).

    Used for scalar identity fields where the merge order already encodes
    priority: providers are merged best-first, so the first non-empty value wins
    and later providers never overwrite it.
    """
    if current is not None and not (isinstance(current, str) and not current.strip()):
        return current
    if candidate is None:
        return current
    if isinstance(candidate, str) and not candidate.strip():
        return current
    return candidate


def _merge_valuation(
    base: ValuationSnapshot, incoming: ValuationSnapshot
) -> ValuationSnapshot:
    """Field-by-field union of two valuation snapshots (existing values win).

    Every multiple is optional and frequently missing for micro-caps; we fill a
    blank field from ``incoming`` but never overwrite a value already established
    by a higher-priority source.
    """
    # Shallow copy is sufficient (and purity-preserving): every field on a
    # valuation snapshot is an immutable scalar, so mutating the copy via
    # ``setattr`` can never write through to ``base``.
    merged = base.model_copy()
    for field_name in ValuationSnapshot.model_fields:
        if getattr(merged, field_name) is None:
            value = getattr(incoming, field_name)
            if value is not None:
                setattr(merged, field_name, value)
    return merged


def _fundamentals_richness(period: FundamentalsPeriod) -> int:
    """Count the populated (non-``None``) line items in a fundamentals period.

    Used to decide which provider's version of an overlapping fiscal period to
    keep as the base when two sources both report the same period — the richer
    (more fully populated) record is preferred, then gaps in it are back-filled
    from the other.
    """
    return sum(
        1
        for field_name in FundamentalsPeriod.model_fields
        if field_name not in {"period_end", "period_label"}
        and getattr(period, field_name) is not None
    )


def _merge_fundamentals_period(
    base: FundamentalsPeriod, incoming: FundamentalsPeriod
) -> FundamentalsPeriod:
    """Merge two fundamentals records for the same period (base values win)."""
    # Shallow copy: a fundamentals period holds only immutable scalars/dates, so
    # per-field ``setattr`` on the copy cannot affect ``base``.
    merged = base.model_copy()
    for field_name in FundamentalsPeriod.model_fields:
        if field_name in {"period_end", "period_label"}:
            continue
        if getattr(merged, field_name) is None:
            value = getattr(incoming, field_name)
            if value is not None:
                setattr(merged, field_name, value)
    return merged


def _merge_fundamentals(
    base: List[FundamentalsPeriod], incoming: List[FundamentalsPeriod]
) -> List[FundamentalsPeriod]:
    """Union two newest-first fundamentals lists, keyed by ``period_end``.

    For a period present in both lists the richer record is used as the base and
    the other back-fills its gaps. The result is re-sorted newest-first so the
    ``SecurityData.latest_fundamentals`` contract continues to hold.
    """
    by_period: Dict[_dt.date, FundamentalsPeriod] = {}
    order: List[_dt.date] = []

    for period in list(base) + list(incoming):
        key = period.period_end
        if key not in by_period:
            by_period[key] = period
            order.append(key)
            continue
        existing = by_period[key]
        # Keep the richer record as the base, back-fill from the leaner one.
        if _fundamentals_richness(period) > _fundamentals_richness(existing):
            by_period[key] = _merge_fundamentals_period(period, existing)
        else:
            by_period[key] = _merge_fundamentals_period(existing, period)

    merged = [by_period[key] for key in order]
    merged.sort(key=lambda p: p.period_end, reverse=True)
    return merged


def _merge_price_history(
    base: List[PriceBar], incoming: List[PriceBar]
) -> List[PriceBar]:
    """Union two oldest-first price-bar lists, de-duplicated by date.

    The first provider to supply a bar for a given date wins (price feeds are
    treated as interchangeable; we simply avoid double-listing a date). The
    result is re-sorted oldest-first per the ``SecurityData`` contract.
    """
    by_date: Dict[_dt.date, PriceBar] = {}
    for bar in list(base) + list(incoming):
        by_date.setdefault(bar.date, bar)
    merged = list(by_date.values())
    merged.sort(key=lambda b: b.date)
    return merged


def _news_key(item: NewsItem) -> Tuple[str, str]:
    """De-duplication key for a news item (normalised title + day)."""
    title = " ".join(item.title.lower().split())
    return title, item.published.date().isoformat()


def _merge_news(base: List[NewsItem], incoming: List[NewsItem]) -> List[NewsItem]:
    """Union two news lists, de-duplicating by normalised title + publish day."""
    seen: Set[Tuple[str, str]] = set()
    merged: List[NewsItem] = []
    for item in list(base) + list(incoming):
        key = _news_key(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    merged.sort(key=lambda n: n.published, reverse=True)
    return merged


def _filing_key(filing: Filing) -> Tuple[str, str, str]:
    """De-duplication key for a filing (form type + filed date + url-or-title)."""
    discriminator = (filing.url or filing.title or "").strip().lower()
    return filing.form_type.strip().lower(), filing.filed.isoformat(), discriminator


def _merge_filings(base: List[Filing], incoming: List[Filing]) -> List[Filing]:
    """Union two filing lists, de-duplicating by form type + date + url/title."""
    seen: Set[Tuple[str, str, str]] = set()
    merged: List[Filing] = []
    for filing in list(base) + list(incoming):
        key = _filing_key(filing)
        if key in seen:
            continue
        seen.add(key)
        merged.append(filing)
    merged.sort(key=lambda f: f.filed, reverse=True)
    return merged


def _insider_key(txn: InsiderTransaction) -> Tuple[str, str, str, str]:
    """De-duplication key for an insider transaction."""
    name = " ".join(txn.insider_name.lower().split())
    shares = "" if txn.shares is None else f"{txn.shares:.4f}"
    return txn.date.isoformat(), name, txn.transaction_type.strip().lower(), shares


def _merge_insider(
    base: List[InsiderTransaction], incoming: List[InsiderTransaction]
) -> List[InsiderTransaction]:
    """Union two insider-transaction lists, de-duplicating by date/name/type/size."""
    seen: Set[Tuple[str, str, str, str]] = set()
    merged: List[InsiderTransaction] = []
    for txn in list(base) + list(incoming):
        key = _insider_key(txn)
        if key in seen:
            continue
        seen.add(key)
        merged.append(txn)
    merged.sort(key=lambda t: t.date, reverse=True)
    return merged


def _institutional_key(holding: InstitutionalHolding) -> Tuple[str, str]:
    """De-duplication key for an institutional holding (holder + as-of)."""
    holder = " ".join(holding.holder.lower().split())
    as_of = holding.as_of.isoformat() if holding.as_of else ""
    return holder, as_of


def _merge_institutional(
    base: List[InstitutionalHolding], incoming: List[InstitutionalHolding]
) -> List[InstitutionalHolding]:
    """Union two institutional-holding lists, de-duplicating by holder + as-of."""
    seen: Set[Tuple[str, str]] = set()
    merged: List[InstitutionalHolding] = []
    for holding in list(base) + list(incoming):
        key = _institutional_key(holding)
        if key in seen:
            continue
        seen.add(key)
        merged.append(holding)
    return merged


def _merge_str_list(base: List[str], incoming: List[str]) -> List[str]:
    """Union two string lists preserving order and de-duplicating case-insensitively."""
    seen: Set[str] = set()
    merged: List[str] = []
    for value in list(base) + list(incoming):
        if value is None:
            continue
        norm = value.strip()
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(norm)
    return merged


def merge_security_data(base: SecurityData, incoming: SecurityData) -> SecurityData:
    """Merge ``incoming`` into ``base`` and return a new canonical ``SecurityData``.

    ``base`` is treated as the higher-priority source: its scalar identity fields
    win and ``incoming`` only fills the gaps, while every list-valued field is
    unioned and de-duplicated. Provenance (``data_sources``) and notes
    (``data_warnings``) are accumulated. The inputs are not mutated.

    This helper is pure and deterministic given its inputs, which keeps the
    aggregation auditable and reproducible.
    """
    # A shallow copy keeps the merge pure without deep-copying ~0.6 MB of price
    # bars/news per provider merge: every mutable field on the copy (valuation,
    # fundamentals, price_history, news, filings, insiders, holdings, peers,
    # data_sources, data_warnings) is *reassigned* below to a freshly-built
    # object, and the element models themselves are never mutated — so ``base``
    # and ``incoming`` remain untouched.
    merged = base.model_copy()

    # Scalar identity — base wins; fill blanks from incoming (merge order encodes
    # priority, so we never overwrite a higher-priority non-empty value).
    merged.name = _coalesce_scalar(merged.name, incoming.name) or merged.name
    merged.sector = _coalesce_scalar(merged.sector, incoming.sector)
    merged.industry = _coalesce_scalar(merged.industry, incoming.industry)
    merged.exchange = _coalesce_scalar(merged.exchange, incoming.exchange)
    merged.cap_tier = merged.cap_tier if merged.cap_tier is not None else incoming.cap_tier
    # Currency has a non-optional default of "USD"; only override a default-USD
    # base if the incoming source names a different, explicit currency.
    if merged.currency == "USD" and incoming.currency and incoming.currency != "USD":
        merged.currency = incoming.currency

    # as_of — keep the most recent observation timestamp across sources.
    if incoming.as_of and (merged.as_of is None or incoming.as_of > merged.as_of):
        merged.as_of = incoming.as_of

    # Structured value-typed fields.
    merged.valuation = _merge_valuation(merged.valuation, incoming.valuation)
    merged.fundamentals = _merge_fundamentals(merged.fundamentals, incoming.fundamentals)
    merged.price_history = _merge_price_history(
        merged.price_history, incoming.price_history
    )

    # List-valued evidence — union + de-duplicate.
    merged.news = _merge_news(merged.news, incoming.news)
    merged.filings = _merge_filings(merged.filings, incoming.filings)
    merged.insider_transactions = _merge_insider(
        merged.insider_transactions, incoming.insider_transactions
    )
    merged.institutional_holdings = _merge_institutional(
        merged.institutional_holdings, incoming.institutional_holdings
    )
    merged.peers = _merge_str_list(merged.peers, incoming.peers)

    # Provenance + notes.
    merged.data_sources = _merge_str_list(merged.data_sources, incoming.data_sources)
    merged.data_warnings = _merge_str_list(merged.data_warnings, incoming.data_warnings)

    return merged


# ---------------------------------------------------------------------------
# Post-merge normalisation: derived market cap (labelled, never fabricated)
# ---------------------------------------------------------------------------

#: Exact provenance note appended whenever the fallback below derives a cap.
#: It names both approximations: the diluted share count is the latest reported
#: period's WEIGHTED AVERAGE (what both yfinance's "Diluted Average Shares" and
#: SEC EDGAR's WeightedAverageNumberOfDilutedSharesOutstanding report), not
#: today's shares outstanding, and that period may lag the price by a quarter+.
_DERIVED_MARKET_CAP_WARNING = (
    "market cap derived from last close x diluted shares "
    "(approximation: period-average diluted shares from the latest reported "
    "fundamentals, not current shares outstanding; info endpoint unavailable)"
)


def _apply_market_cap_fallback(merged: SecurityData) -> None:
    """Back-fill a missing market cap (and cap tier) from real, on-hand inputs.

    A rate-limited quote/info endpoint frequently leaves ``valuation.market_cap``
    ``None`` even though the price history (a separate, un-throttled endpoint)
    and ``shares_diluted`` (SEC companyfacts) both arrived. When — and only
    when — **both** of those real inputs are present, the cap is *derived* as
    ``last close × diluted shares`` and an explicit provenance note is appended
    to ``data_warnings`` so the reader knows the figure is arithmetic on real
    data rather than a provider-supplied quote — and that the share count is the
    latest period's *weighted average* diluted shares (both share sources report
    period averages), so the figure approximates a true point-in-time cap.
    Nothing is ever guessed:

    * a provider-supplied market cap is **never** overwritten;
    * if the price history is empty, or the latest fundamentals period is
      missing / carries no positive ``shares_diluted``, no cap is derived and
      the field honestly stays ``None``.

    ``cap_tier`` is likewise back-filled from the (possibly derived) market cap
    when no provider assigned one, using the same thresholds the providers use,
    so a derived cap keeps the record screenable by tier.

    Mutates ``merged`` in place (it is the composite's own post-merge copy);
    the valuation snapshot is *replaced*, not mutated, because it may still be
    shared with a provider-cached object via the shallow merge copies.
    """
    if merged.valuation.market_cap is None and merged.price_history:
        last_close = merged.price_history[-1].close  # price_history is oldest-first
        latest = merged.latest_fundamentals
        shares = latest.shares_diluted if latest is not None else None
        if last_close > 0 and shares is not None and shares > 0:
            merged.valuation = merged.valuation.model_copy(
                update={"market_cap": last_close * shares}
            )
            merged.data_warnings.append(_DERIVED_MARKET_CAP_WARNING)
    if merged.cap_tier is None:
        merged.cap_tier = _classify_cap_tier(merged.valuation.market_cap)


# ---------------------------------------------------------------------------
# The composite provider
# ---------------------------------------------------------------------------


class CompositeProvider(DataProvider):
    """A :class:`DataProvider` that fans out to every available registered provider.

    On construction it imports :mod:`convexity.data.providers` (triggering each
    concrete provider's self-registration), instantiates every registered
    provider class, and keeps those that report themselves available
    (``is_available()`` is treated as ``True`` when not implemented). The composite
    skips *itself* to avoid infinite recursion.

    :meth:`get_security_data` queries each member provider, merging their partial
    :class:`SecurityData` objects into one canonical record (see
    :func:`merge_security_data`). A provider raising
    :class:`~convexity.core.exceptions.DataUnavailable` is an expected gap (the
    ticker simply is not covered there) and is recorded as a warning; any other
    error from one provider is caught, logged and recorded so it can never crash
    the scan of a ticker. Only when *no* provider yields any data is
    :class:`~convexity.core.exceptions.DataUnavailable` raised.

    This is transparent evidence aggregation, not prediction: the composite makes
    the union of available facts auditable, it does not assert correctness of any
    figure nor forecast price movement.
    """

    #: Stable identifier recorded in ``SecurityData.data_sources``.
    _NAME = "composite"

    def __init__(
        self,
        settings: Optional[Any] = None,
        *,
        providers: Optional[List[DataProvider]] = None,
    ) -> None:
        """Build the composite over all available providers.

        Args:
            settings: Optional :class:`~convexity.core.config.Settings` forwarded to
                providers whose constructor accepts one. ``None`` lets each provider
                fall back to the process-wide cached settings.
            providers: Optional explicit list of provider instances to compose,
                bypassing registry discovery (used primarily by tests). When given,
                availability is still respected and the composite itself is skipped.
        """
        self._settings = settings
        if providers is not None:
            self._providers: List[DataProvider] = [
                p for p in providers if not isinstance(p, CompositeProvider)
            ]
        else:
            self._providers = self._discover_providers(settings)

        available = [p for p in self._providers if _is_available(p)]
        skipped = [p for p in self._providers if p not in available]
        self._providers = available
        if skipped:
            _log.debug(
                "composite skipping unavailable providers: %s",
                ", ".join(self._safe_name(p) for p in skipped),
            )
        _log.info(
            "composite provider initialised with %d available source(s): %s",
            len(self._providers),
            ", ".join(self._safe_name(p) for p in self._providers) or "(none)",
        )

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def _discover_providers(settings: Optional[Any]) -> List[DataProvider]:
        """Import the providers package and instantiate every registered provider.

        Importing :mod:`convexity.data.providers` triggers the ``@register_provider``
        decorators so the registry is populated. Each provider class is then
        instantiated defensively: a constructor that does not accept ``settings`` is
        retried with no arguments, and any provider that cannot be built at all is
        skipped with a warning rather than aborting discovery.
        """
        try:
            import convexity.data.providers  # noqa: F401  (import for registration side effect)
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning("could not import providers package for registration: %s", exc)

        instances: List[DataProvider] = []
        for cls in get_providers():
            if cls is CompositeProvider:
                # Never compose the composite into itself.
                continue
            instance = CompositeProvider._instantiate(cls, settings)
            if instance is not None:
                instances.append(instance)
        return instances

    @staticmethod
    def _instantiate(cls: Any, settings: Optional[Any]) -> Optional[DataProvider]:
        """Instantiate a provider class, tolerating differing constructor signatures."""
        # Prefer passing settings through when the provider accepts it; fall back
        # to a no-arg construction for providers (e.g. yfinance) that take none.
        attempts = ()
        if settings is not None:
            attempts = ((settings,), {}), ((), {})
        else:
            attempts = ((), {}),
        last_exc: Optional[Exception] = None
        for args, kwargs in attempts:  # type: ignore[assignment]
            try:
                return cls(*args, **kwargs)
            except TypeError as exc:
                last_exc = exc
                continue
            except Exception as exc:  # pragma: no cover - defensive per provider
                last_exc = exc
                break
        _log.warning(
            "could not instantiate provider %s: %s",
            getattr(cls, "__name__", repr(cls)),
            last_exc,
        )
        return None

    @staticmethod
    def _safe_name(provider: DataProvider) -> str:
        """Best-effort provider name for logging (never raises)."""
        try:
            return provider.name
        except Exception:  # pragma: no cover - defensive
            return type(provider).__name__

    # -- DataProvider contract --------------------------------------------

    @property
    def name(self) -> str:
        """Stable identifier for the composite provider."""
        return self._NAME

    @property
    def capabilities(self) -> Set[str]:
        """Union of the capabilities of every member provider."""
        caps: Set[str] = set()
        for provider in self._providers:
            try:
                caps |= set(provider.capabilities)
            except Exception:  # pragma: no cover - defensive
                continue
        return caps

    @property
    def providers(self) -> List[DataProvider]:
        """The available member providers backing this composite (read-only copy)."""
        return list(self._providers)

    def get_universe(self, params: ScanParams) -> List[str]:
        """Delegate universe enumeration to the first capable member provider.

        Tries each member that does not raise :class:`NotSupported`; the first to
        return a non-empty list wins. Raises :class:`NotSupported` only when no
        member can enumerate a universe.
        """
        last_error: Optional[Exception] = None
        for provider in self._providers:
            try:
                universe = provider.get_universe(params)
            except NotSupported:
                continue
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                _log.warning(
                    "provider %s failed during universe enumeration: %s",
                    self._safe_name(provider),
                    exc,
                )
                continue
            if universe:
                return list(universe)
        if last_error is not None:
            raise NotSupported(
                "no registered provider could enumerate a universe "
                f"(last error: {last_error})"
            )
        raise NotSupported("no registered provider supports universe enumeration")

    def get_quotes(self, tickers: Sequence[str]) -> Dict[str, Dict[str, float]]:
        """Delegate batched screening quotes to the first capable member provider.

        The universe screen (:func:`convexity.data.universe.build_universe`) asks
        its price provider for cheap batched quotes carrying ``market_cap`` /
        ``avg_dollar_volume``. Members are tried in the composite's availability
        order (unavailable providers were already dropped at construction); a
        member without a callable ``get_quotes`` is skipped, and a member whose
        call raises or returns nothing is logged and skipped so the next source
        can try. Returns ``{}`` — never raises — when no member can quote the
        batch, which the screen treats as "could not verify, exclude".
        """
        for provider in self._providers:
            method = getattr(provider, "get_quotes", None)
            if not callable(method):
                continue
            pname = self._safe_name(provider)
            try:
                result = method(tickers)
            except Exception as exc:  # defensive: one bad member never aborts a screen
                _log.warning("composite: %s.get_quotes failed: %s", pname, exc)
                continue
            if result:
                return {str(k).upper(): v for k, v in dict(result).items()}
            _log.debug("composite: %s.get_quotes returned no quotes for this batch", pname)
        return {}

    def get_security_data(self, ticker: str) -> SecurityData:
        """Fetch ``ticker`` from every member provider and merge into one record.

        Each provider is called in source-priority order (most authoritative
        first) so that scalar identity fields resolve to the best source. A
        provider raising :class:`DataUnavailable` is an expected miss and is noted
        as a warning; any other provider error is caught, logged and recorded so a
        single bad source never aborts the ticker. Only when *no* provider returns
        any data does this raise :class:`DataUnavailable`.
        """
        symbol = (ticker or "").strip().upper()
        if not symbol:
            raise DataUnavailable("empty ticker symbol", ticker=ticker)

        ordered = sorted(self._providers, key=lambda p: _priority_rank(self._safe_name(p)))

        merged: Optional[SecurityData] = None
        contributed: List[str] = []
        collected_warnings: List[str] = []

        for provider in ordered:
            pname = self._safe_name(provider)
            try:
                partial = provider.get_security_data(symbol)
            except DataUnavailable as exc:
                # Expected gap: this source simply does not cover the ticker.
                collected_warnings.append(f"{pname}: no data ({exc})")
                _log.debug("composite: %s has no data for %s (%s)", pname, symbol, exc)
                continue
            except (ProviderError, NotSupported, ConvexityError) as exc:
                # A handled provider failure must not crash the scan.
                collected_warnings.append(f"{pname}: provider error ({exc})")
                _log.warning("composite: %s failed for %s: %s", pname, symbol, exc)
                continue
            except Exception as exc:  # pragma: no cover - defensive catch-all
                collected_warnings.append(f"{pname}: unexpected error ({exc})")
                _log.warning(
                    "composite: %s raised an unexpected error for %s: %s",
                    pname,
                    symbol,
                    exc,
                )
                continue

            if partial is None:  # pragma: no cover - providers return data or raise
                collected_warnings.append(f"{pname}: returned no data object")
                continue

            contributed.append(pname)
            if merged is None:
                # Shallow copy with fresh list containers for the two fields the
                # pipeline appends to downstream (warnings/provenance), so a
                # provider that caches and re-serves its object can never be
                # mutated through the merged record. A deep copy of the whole
                # payload (~0.6 MB of bars/news per ticker, per fetch thread) is
                # unnecessary: element models are never mutated downstream.
                merged = partial.model_copy(
                    update={
                        "data_warnings": list(partial.data_warnings),
                        "data_sources": list(partial.data_sources),
                    }
                )
            else:
                merged = merge_security_data(merged, partial)

        if merged is None:
            # No provider yielded anything usable for this ticker.
            detail = "; ".join(collected_warnings) if collected_warnings else "no providers available"
            raise DataUnavailable(
                f"no provider returned data for {symbol} ({detail})", ticker=symbol
            )

        # Fold in warnings from providers that produced nothing (their own warnings
        # never reached ``merged`` because they raised before returning a record).
        if collected_warnings:
            merged.data_warnings = _merge_str_list(merged.data_warnings, collected_warnings)

        # Record the canonical, de-duplicated provenance.
        merged.data_sources = _merge_str_list(merged.data_sources, contributed)

        # Post-merge normalisation: only after every provider has had its say
        # (so a real, provider-supplied market cap always wins) derive a missing
        # cap from last close × diluted shares, with an explicit provenance note.
        _apply_market_cap_fallback(merged)

        _log.info(
            "composite assembled %s from %d source(s): %s",
            symbol,
            len(contributed),
            ", ".join(contributed) or "(none)",
        )
        return merged


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_default_provider(settings: Optional[Any] = None) -> CompositeProvider:
    """Return the default Convexity data provider: a configured :class:`CompositeProvider`.

    This is the single entry point the pipeline/CLI should use to obtain a data
    source. It imports and registers every concrete provider, composes the
    available ones, and forwards ``settings`` to those that accept it. When
    ``settings`` is ``None`` each provider falls back to the process-wide cached
    settings.

    Args:
        settings: Optional :class:`~convexity.core.config.Settings` instance.

    Returns:
        A ready-to-use :class:`CompositeProvider` spanning every available source.
    """
    return CompositeProvider(settings)


__all__ = [
    "CompositeProvider",
    "get_default_provider",
    "merge_security_data",
]
