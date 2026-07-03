"""yfinance-backed :class:`DataProvider` for Convexity.

This provider assembles a :class:`~convexity.core.models.SecurityData` for a single
ticker out of the *free* `yfinance <https://pypi.org/project/yfinance/>`_ library,
which in turn scrapes Yahoo Finance's public, unauthenticated endpoints. Those
endpoints are best-effort and undocumented: fields appear and disappear, coverage
for thin micro-caps is patchy, and occasional rate-limiting or empty payloads are
normal. Accordingly every sub-fetch here is wrapped defensively — a failure in one
section (say, news) appends a human-readable note to ``SecurityData.data_warnings``
and leaves the rest of the object intact rather than raising.

Honesty notes
-------------
* This is a research/screening input, **not** a predictor and **not** advice. The
  provider only transcribes what Yahoo reports; it never fabricates a missing
  datum. Anything Yahoo does not supply stays ``None`` and is recorded as a
  warning so downstream analyzers can lower their confidence.
* Derived figures (margins, ROIC/ROE/ROA, FCF, valuation multiples) are computed
  *only* from the raw statement line items actually present. If an input is
  missing the derived value is left ``None`` rather than guessed.

The provider advertises the capabilities ``{"prices", "fundamentals",
"valuation", "news", "universe-screen"}``. It does **not** enumerate a universe;
``get_universe`` is intentionally left raising
:class:`~convexity.core.exceptions.NotSupported` (the screening universe is built
by ``convexity/data/universe.py``). It *does* serve the screening side of that
universe build through :meth:`YFinanceProvider.get_quotes` — cheap, batched
market-cap / average-dollar-volume quotes consumed by
:func:`convexity.data.universe.build_universe`.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set

from convexity.core.contracts import DataProvider
from convexity.core.exceptions import DataUnavailable, ProviderError
from convexity.core.logging import get_logger
from convexity.core.models import (
    CapTier,
    FundamentalsPeriod,
    NewsItem,
    PriceBar,
    SecurityData,
    ValuationSnapshot,
)
from convexity.core.registry import register_provider
from convexity.data import cache as data_cache

_log = get_logger(__name__)

#: TTL (seconds) for cached screening-quote chunks. Deliberately short (~4 hours):
#: screening quotes are point-in-time liquidity/cap figures, so a re-run within
#: the window can skip re-screening the same chunk, while anything older is
#: refetched live. Keyed per chunk so partial coverage from an interrupted or
#: rate-limited screen still persists chunk-by-chunk.
_QUOTES_CACHE_TTL_SECONDS = 4 * 60 * 60


# ---------------------------------------------------------------------------
# Small numeric helpers (pure, defensive)
# ---------------------------------------------------------------------------


def _is_finite_number(value: Any) -> bool:
    """Return True only for a real, finite int/float (rejects None, NaN, inf, bool)."""
    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _to_float(value: Any) -> Optional[float]:
    """Coerce ``value`` to a finite ``float``; return ``None`` for anything else.

    Pandas frequently yields ``NaN`` (a float that is not a number) for missing
    cells; treating those as ``None`` is essential so a gap never masquerades as a
    real zero.
    """
    if not _is_finite_number(value):
        return None
    return float(value)


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Divide two optionals, returning ``None`` if either is missing or the denom is ~0."""
    if numerator is None or denominator is None:
        return None
    if abs(denominator) < 1e-12:
        return None
    result = numerator / denominator
    return result if math.isfinite(result) else None


def _coerce_date(value: Any) -> Optional[_dt.date]:
    """Best-effort coercion of a pandas/py datetime-ish value to a ``datetime.date``."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    # pandas.Timestamp exposes .to_pydatetime(); numpy datetime64 exposes .astype.
    to_py = getattr(value, "to_pydatetime", None)
    if callable(to_py):
        try:
            return to_py().date()
        except Exception:  # pragma: no cover - defensive
            return None
    date_attr = getattr(value, "date", None)
    if callable(date_attr):
        try:
            return date_attr()
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _coerce_datetime(value: Any) -> Optional[_dt.datetime]:
    """Best-effort coercion of a value to a timezone-naive ``datetime``."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day)
    if _is_finite_number(value):
        # Yahoo news timestamps are unix epoch seconds.
        try:
            return _dt.datetime.utcfromtimestamp(float(value))
        except (OverflowError, OSError, ValueError):  # pragma: no cover - defensive
            return None
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        for parser in (
            lambda s: _dt.datetime.fromisoformat(s),
            lambda s: _dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S"),
            lambda s: _dt.datetime.strptime(s, "%Y-%m-%d"),
        ):
            try:
                parsed = parser(text)
                return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
            except (ValueError, TypeError):
                continue
    to_py = getattr(value, "to_pydatetime", None)
    if callable(to_py):
        try:
            parsed = to_py()
            return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _classify_cap_tier(market_cap: Optional[float]) -> Optional[CapTier]:
    """Bucket a market cap into a :class:`CapTier` using the documented thresholds.

    Thresholds mirror ``CapTier`` in ``convexity.core.models``: nano ``< ~$50M``,
    micro ``~$50M–$300M``, small ``~$300M–$2B``. A cap at or above ~$2B is outside
    the small-cap remit and returns ``None`` so the universe layer can exclude it.
    """
    if market_cap is None or market_cap <= 0:
        return None
    if market_cap < 50_000_000:
        return CapTier.NANO
    if market_cap < 300_000_000:
        return CapTier.MICRO
    if market_cap < 2_000_000_000:
        return CapTier.SMALL
    return None


# Mapping of FundamentalsPeriod line-item -> the candidate row labels yfinance may
# use in its income / cash-flow / balance-sheet frames. Yahoo's row names drift
# across versions, so several aliases are tried in order.
_INCOME_ROW_ALIASES: Dict[str, List[str]] = {
    "revenue": ["Total Revenue", "TotalRevenue", "Operating Revenue", "OperatingRevenue"],
    "gross_profit": ["Gross Profit", "GrossProfit"],
    "operating_income": ["Operating Income", "OperatingIncome", "Total Operating Income As Reported"],
    "net_income": ["Net Income", "NetIncome", "Net Income Common Stockholders", "NetIncomeCommonStockholders"],
    "ebitda": ["EBITDA", "Normalized EBITDA", "NormalizedEBITDA"],
    "eps_diluted": ["Diluted EPS", "DilutedEPS"],
    "interest_expense": ["Interest Expense", "InterestExpense", "Interest Expense Non Operating"],
    "shares_diluted": ["Diluted Average Shares", "DilutedAverageShares", "Diluted Shares"],
}

_CASHFLOW_ROW_ALIASES: Dict[str, List[str]] = {
    "operating_cash_flow": [
        "Operating Cash Flow",
        "OperatingCashFlow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
    ],
    "capex": ["Capital Expenditure", "CapitalExpenditure", "Capital Expenditures"],
    "free_cash_flow": ["Free Cash Flow", "FreeCashFlow"],
}

_BALANCE_ROW_ALIASES: Dict[str, List[str]] = {
    "total_assets": ["Total Assets", "TotalAssets"],
    "total_debt": ["Total Debt", "TotalDebt"],
    "long_term_debt": ["Long Term Debt", "LongTermDebt"],
    "current_debt": ["Current Debt", "CurrentDebt", "Current Debt And Capital Lease Obligation"],
    "cash_and_equivalents": [
        "Cash And Cash Equivalents",
        "CashAndCashEquivalents",
        "Cash Cash Equivalents And Short Term Investments",
    ],
    "total_equity": [
        "Stockholders Equity",
        "StockholdersEquity",
        "Total Equity Gross Minority Interest",
        "Total Stockholder Equity",
    ],
    "current_assets": ["Current Assets", "CurrentAssets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "CurrentLiabilities", "Total Current Liabilities"],
    "inventory": ["Inventory", "Inventories"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued", "OrdinarySharesNumber"],
}


@register_provider
class YFinanceProvider(DataProvider):
    """Build a :class:`SecurityData` for one ticker from the free yfinance/Yahoo feed.

    The provider performs several independent sub-fetches (company info, annual and
    quarterly statements, valuation multiples, ~2 years of daily bars, recent
    news). Each is isolated in its own ``try``/``except`` so a partial Yahoo
    outage degrades coverage gracefully instead of failing the whole ticker. Only
    a total inability to identify the security (no info *and* no price history)
    raises :class:`~convexity.core.exceptions.DataUnavailable`.
    """

    # How much history to request. ``2y`` of daily bars is enough for the
    # technical/momentum analyzers without bloating the payload.
    _PRICE_PERIOD = "2y"
    _PRICE_INTERVAL = "1d"
    # Cap on how many annual / quarterly periods and news items we keep.
    _MAX_ANNUAL_PERIODS = 5
    _MAX_QUARTERLY_PERIODS = 1
    _MAX_NEWS = 25
    #: Cache ``kind`` labels (see :func:`convexity.data.cache.make_key`).
    _SECURITY_DATA_CACHE_KIND = "security_data"
    _QUOTES_CACHE_KIND = "quotes"

    def __init__(self, *, cache: Optional[data_cache.Cache] = None) -> None:
        """Initialise the ``fast_info`` throttle state and wire up the cache.

        The screening path issues one lightweight ``fast_info`` HTTP request per
        prefilter-clearing symbol; across a full listing that is thousands of
        requests, so they are paced (see ``_fast_info_throttle``). The lock makes
        the pacing correct even if a caller shares one provider across threads.

        Args:
            cache: Optional :class:`~convexity.data.cache.Cache` to memoise
                fetches on (mainly for tests). ``None`` resolves lazily to the
                process-wide default cache, so constructing the provider (as the
                registry does at import time) never touches the disk.
        """
        self._fast_info_lock = threading.Lock()
        self._last_fast_info_ts = 0.0
        self._cache = cache

    # -- caching plumbing ----------------------------------------------------
    #
    # Honesty rule: caching changes *when* we fetch, never *what* we report.
    # Only a COMPLETE fetch (see ``_is_complete_fetch``) is written to the
    # cache; a partial one — e.g. Yahoo's ``info`` endpoint rate-limited so the
    # market cap is unknown — is still returned to the caller with its
    # ``data_warnings`` intact, but is NOT cached, so a later healthy run
    # refetches instead of serving the gap for the whole TTL. Every cache
    # failure (open, read, parse, write) degrades to a live fetch; it is never
    # allowed to break the provider.

    def _resolve_cache(self) -> Optional[data_cache.Cache]:
        """Return the cache to use, or ``None`` if none can be obtained."""
        if self._cache is not None:
            return self._cache
        try:
            return data_cache.get_cache()
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning("provider cache unavailable (%s); fetching live", exc)
            return None

    def _cache_load_security_data(self, symbol: str) -> Optional[SecurityData]:
        """Return a fresh cached :class:`SecurityData` for ``symbol``, else ``None``.

        Any cache read error or validation failure is logged and treated as a
        miss (forcing a live fetch) — corrupt cache content is never served.
        """
        store = self._resolve_cache()
        if store is None:
            return None
        try:
            payload = store.get_data(self.name, symbol, self._SECURITY_DATA_CACHE_KIND)
        except Exception as exc:
            _log.warning(
                "cache read failed for %s security_data (%s); fetching live", symbol, exc
            )
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return SecurityData.model_validate(payload)
        except Exception as exc:
            _log.warning(
                "cached security_data for %s failed validation (%s); refetching", symbol, exc
            )
            return None

    def _cache_store_security_data(self, symbol: str, sec: SecurityData) -> None:
        """Serialise ``sec`` to the cache (default TTL); failures degrade silently."""
        store = self._resolve_cache()
        if store is None:
            return
        try:
            store.set_data(
                self.name,
                symbol,
                self._SECURITY_DATA_CACHE_KIND,
                sec.model_dump(mode="json"),
            )
        except Exception as exc:
            _log.warning("could not cache security_data for %s (%s)", symbol, exc)

    @staticmethod
    def _is_complete_fetch(
        info: Dict[str, Any],
        market_cap: Optional[float],
        price_history: List[PriceBar],
        warnings: List[str],
    ) -> bool:
        """Decide whether a just-assembled fetch is complete enough to cache.

        A fetch is *complete* when the ``info`` endpoint answered with a usable
        market cap, price bars came back, and no sub-fetch raised (every raised
        sub-fetch appends a ``"failed to fetch ..."`` warning — the signature of
        rate limiting or a transient outage). Genuinely absent data that Yahoo
        answered *about* (e.g. ``"no fundamentals ... available"`` or
        ``"no recent news"`` for a thin micro-cap) does not block caching: the
        TTL bounds its staleness and the warnings travel with the cached object.

        Requiring the market cap is deliberate (the incident this guards
        against): a throttled ``info`` endpoint yields an UNKNOWN cap, and
        caching that would freeze the gap for the whole TTL. The cost of the
        strictness is only an extra refetch, never a wrong value.
        """
        if not info or market_cap is None or not price_history:
            return False
        return not any("failed to fetch" in w for w in warnings)

    @property
    def name(self) -> str:
        """Stable identifier recorded in ``SecurityData.data_sources``."""
        return "yfinance"

    @property
    def capabilities(self) -> Set[str]:
        """Capabilities advertised to the aggregator."""
        return {"prices", "fundamentals", "valuation", "news", "universe-screen"}

    # -- public contract ---------------------------------------------------

    def get_security_data(self, ticker: str) -> SecurityData:
        """Assemble everything yfinance can supply for ``ticker``.

        Never raises on a single missing field; partial failures are recorded in
        ``SecurityData.data_warnings``. Raises
        :class:`~convexity.core.exceptions.DataUnavailable` only when the ticker
        cannot be identified at all, and
        :class:`~convexity.core.exceptions.ProviderError` if the yfinance library
        itself is unavailable.

        Fetches are memoised on the freshness-bounded disk cache
        (:mod:`convexity.data.cache`, default ``cache_ttl_seconds``): a fresh
        cached fetch is served without any network, and only *complete* fetches
        are cached (see ``_is_complete_fetch``) so a rate-limited partial result
        is returned once — warnings intact — but never frozen into the cache.
        """
        symbol = (ticker or "").strip().upper()
        if not symbol:
            raise DataUnavailable("empty ticker symbol", ticker=ticker)

        cached = self._cache_load_security_data(symbol)
        if cached is not None:
            _log.debug("yfinance served %s from cache", symbol)
            return cached

        yf = self._import_yfinance()
        warnings: List[str] = []

        try:
            yf_ticker = yf.Ticker(symbol)
        except Exception as exc:  # pragma: no cover - constructor rarely fails
            raise ProviderError(
                f"yfinance could not construct a Ticker for {symbol!r}: {exc}",
                provider=self.name,
            ) from exc

        info = self._fetch_info(yf_ticker, symbol, warnings)
        price_history = self._fetch_prices(yf_ticker, symbol, warnings)

        # Hard guard: if we have neither identifying info nor any prices, Yahoo has
        # nothing for this symbol — treat it as an expected gap, not a crash.
        if not info and not price_history:
            raise DataUnavailable(
                f"yfinance returned no info and no price history for {symbol!r}",
                ticker=symbol,
            )

        name = self._extract_name(info, symbol)
        market_cap = _to_float(info.get("marketCap")) if info else None

        fundamentals = self._fetch_fundamentals(yf_ticker, symbol, warnings)
        valuation = self._build_valuation(info, fundamentals, market_cap)
        news = self._fetch_news(yf_ticker, symbol, warnings)

        sec = SecurityData(
            ticker=symbol,
            name=name,
            sector=self._clean_str(info.get("sector")) if info else None,
            industry=self._clean_str(info.get("industry")) if info else None,
            exchange=self._extract_exchange(info),
            cap_tier=_classify_cap_tier(market_cap),
            currency=self._extract_currency(info),
            as_of=_dt.datetime.utcnow(),
            valuation=valuation,
            fundamentals=fundamentals,
            price_history=price_history,
            news=news,
            data_sources=[self.name],
            data_warnings=warnings,
        )
        _log.debug(
            "yfinance assembled %s: %d fundamentals, %d bars, %d news, %d warnings",
            symbol,
            len(fundamentals),
            len(price_history),
            len(news),
            len(warnings),
        )
        # HONESTY: cache only complete fetches. A partial result (rate-limited
        # info endpoint, missing market cap, failed sub-fetch) is returned to
        # the caller with its warnings, but a later healthy run must refetch it.
        if self._is_complete_fetch(info, market_cap, price_history, warnings):
            self._cache_store_security_data(symbol, sec)
        else:
            _log.info(
                "yfinance: not caching %s (incomplete fetch, %d warning(s)); "
                "a later run will refetch",
                symbol,
                len(warnings),
            )
        return sec

    # -- batched screening quotes (universe construction) --------------------
    #
    # Strategy (documented for the universe-screen consumer):
    #
    # 1. **Prices + volume in bulk.** ``yf.download(chunk, period="1mo",
    #    group_by="ticker", threads=True, progress=False)`` fetches ~1 month of
    #    daily bars for up to ``_QUOTE_CHUNK_SIZE`` (~200) tickers in a single
    #    HTTP round-trip. From those bars we compute the *average dollar volume*
    #    as ``mean(close × volume)`` over the rows actually returned, plus the
    #    last close as ``price``. This is the cheap part: ~25 requests cover a
    #    5,000-name listing.
    # 2. **Market cap via a cheap, prefiltered path.** There is no batched
    #    market-cap endpoint in yfinance, and calling ``fast_info`` for every one
    #    of ~5,000 names is far too slow. So a per-symbol ``fast_info`` lookup
    #    (Yahoo's lightweight quote endpoint) is made **only** for names that
    #    first clear a conservative dollar-volume prefilter
    #    (``_MCAP_PREFILTER_MIN_DOLLAR_VOLUME``, set well below the default
    #    ``ScanParams.min_avg_dollar_volume`` screen floor of $200k/day so the
    #    prefilter never excludes a name the real screen would have kept). For
    #    each surviving name we take ``fast_info`` market cap directly, or
    #    reconstruct it as ``last price × shares outstanding`` when only shares
    #    are reported. Names failing the prefilter simply carry **no**
    #    ``market_cap`` key — the universe screen then conservatively excludes
    #    them (missing data lowers coverage; it is never fabricated).
    # 3. **Defensive + rate-limit friendly.** One bad chunk (network error,
    #    schema drift, empty payload) is logged and skipped — it can never raise
    #    out of :meth:`get_quotes` — a small sleep separates chunks, and every
    #    per-symbol ``fast_info`` lookup is paced through ``_fast_info_throttle``
    #    (~8 req/s) so the thousands of market-cap lookups a full-listing screen
    #    performs do not trip Yahoo's rate limiting. Getting 429'd here is not
    #    cosmetic: a rate-limited ``fast_info`` yields no market cap, the screen
    #    then conservatively excludes the name, and the universe silently
    #    shrinks — pacing prevents that failure mode at the source.
    #
    # The returned per-ticker dicts use exactly the keys
    # ``convexity.data.universe`` recognises: ``market_cap``,
    # ``avg_dollar_volume`` and ``price``. A value that could not be determined
    # is *omitted* rather than defaulted.

    #: Max tickers per bulk ``yf.download`` call.
    _QUOTE_CHUNK_SIZE = 200
    #: History window used to average daily dollar volume.
    _QUOTE_HISTORY_PERIOD = "1mo"
    #: Pause between bulk download chunks (rate-limit friendliness).
    _QUOTE_CHUNK_SLEEP_SECONDS = 0.75
    #: Only names whose avg dollar volume clears this floor get the (slower)
    #: per-symbol market-cap lookup. Kept far below the default screen floor
    #: ($200k/day) so the prefilter is strictly cheaper, never stricter.
    _MCAP_PREFILTER_MIN_DOLLAR_VOLUME = 25_000.0
    #: Minimum spacing between per-symbol ``fast_info`` lookups (~8 req/s).
    #: Without pacing, a 5,000-name screen fires thousands of back-to-back
    #: requests, Yahoo answers 429, ``_fast_market_cap`` degrades to ``None``
    #: and the universe screen silently drops those names as cap-unknown.
    _FAST_INFO_MIN_INTERVAL_S = 0.12

    def get_quotes(self, tickers: Sequence[str]) -> Dict[str, Dict[str, float]]:
        """Return cheap screening quotes ``{ticker: {market_cap, avg_dollar_volume, price}}``.

        Designed for :func:`convexity.data.universe.build_universe`: batched,
        chunked (~200 symbols per ``yf.download`` call), with market cap resolved
        through a per-symbol ``fast_info`` lookup only for names that pass a
        cheap dollar-volume prefilter (see the strategy note above). Keys are the
        requested tickers upper-cased; any figure that could not be positively
        determined is omitted from that ticker's dict (never fabricated), and a
        ticker with no usable data at all is absent from the result.

        Never raises: a failed chunk or symbol is logged and skipped so one bad
        batch cannot abort a full-universe screen.

        Each chunk's result is cached for a short window
        (``_QUOTES_CACHE_TTL_SECONDS``, ~4h) keyed by a hash of the sorted chunk
        symbols, so a re-run within the window serves the chunk from the cache
        instead of re-screening — and because the key is per chunk, the coverage
        an interrupted or partially rate-limited screen *did* achieve persists.
        HONESTY: only *complete* chunks are cached (see
        ``_quotes_chunk_is_complete``): a chunk that yielded nothing (download
        failure, dead symbols) or whose per-symbol market-cap lookups were
        rate-limited — liquid quotes missing their caps, the incident signature —
        is always retried live rather than frozen capless for the whole window.
        """
        symbols: List[str] = []
        seen: Set[str] = set()
        for raw in tickers:
            sym = str(raw or "").strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                symbols.append(sym)
        if not symbols:
            return {}

        try:
            yf = self._import_yfinance()
        except ProviderError as exc:
            _log.warning("get_quotes unavailable (yfinance missing): %s", exc)
            return {}

        out: Dict[str, Dict[str, float]] = {}
        chunks = [
            symbols[i : i + self._QUOTE_CHUNK_SIZE]
            for i in range(0, len(symbols), self._QUOTE_CHUNK_SIZE)
        ]
        fetched_any = False  # pace only between *network* chunks, not cache hits
        for idx, chunk in enumerate(chunks):
            cached = self._cache_load_quotes_chunk(chunk)
            if cached is not None:
                _log.debug("quote chunk %d/%d served from cache", idx + 1, len(chunks))
                out.update(cached)
                continue
            if fetched_any and self._QUOTE_CHUNK_SLEEP_SECONDS > 0:
                time.sleep(self._QUOTE_CHUNK_SLEEP_SECONDS)
            fetched_any = True
            try:
                chunk_quotes = self._quote_chunk(yf, chunk)
            except Exception as exc:  # defensive: one bad chunk never aborts the screen
                _log.warning(
                    "yfinance quote chunk %d/%d failed (%s: %s); skipping %d symbols",
                    idx + 1,
                    len(chunks),
                    type(exc).__name__,
                    exc,
                    len(chunk),
                )
                continue
            out.update(chunk_quotes)
            # HONESTY: cache only complete chunks. A chunk whose liquid symbols
            # are missing their market caps was hit by fast_info rate limiting
            # (the bulk chart download succeeded but the per-symbol cap endpoint
            # was throttled — the exact incident signature); caching it would
            # serve capless quotes for the whole TTL and silently shrink every
            # re-screen's universe. It is returned to the caller as-is, but a
            # later run must refetch it live.
            if self._quotes_chunk_is_complete(chunk_quotes):
                self._cache_store_quotes_chunk(chunk, chunk_quotes)
            elif chunk_quotes:
                _log.info(
                    "yfinance: not caching a %d-symbol quote chunk "
                    "(liquid symbols missing market caps — rate-limited?); "
                    "a later run will re-screen it",
                    len(chunk),
                )
        return out

    # -- screening-quote chunk cache -----------------------------------------

    @classmethod
    def _quotes_chunk_is_complete(cls, quotes: Dict[str, Dict[str, float]]) -> bool:
        """Decide whether a just-fetched quote chunk is complete enough to cache.

        A chunk is *complete* when it is non-empty and every symbol that cleared
        the market-cap prefilter (``avg_dollar_volume >=
        _MCAP_PREFILTER_MIN_DOLLAR_VOLUME`` — exactly the condition under which
        ``_quote_chunk`` attempts a ``fast_info`` cap lookup) actually carries a
        ``market_cap``. A liquid quote *without* a cap means the per-symbol cap
        lookup failed — under load that is Yahoo rate-limiting ``fast_info``
        while the bulk chart download still succeeds (the 2026-07 incident
        signature) — and caching it would freeze capless quotes for the whole
        TTL, silently excluding those names from every re-screen in the window.

        Symbols *below* the prefilter carry no cap by design (the screen excludes
        them conservatively rather than us guessing), so their capless quotes do
        not block caching. The cost of the strictness is only an extra
        re-screen of the chunk, never a wrong or frozen-partial value.
        """
        if not quotes:
            return False
        for quote in quotes.values():
            adv = quote.get("avg_dollar_volume")
            if (
                adv is not None
                and adv >= cls._MCAP_PREFILTER_MIN_DOLLAR_VOLUME
                and "market_cap" not in quote
            ):
                return False
        return True

    @staticmethod
    def _quotes_chunk_hash(chunk: Sequence[str]) -> str:
        """Return a stable hex digest of the sorted chunk symbols (the cache key's
        ticker slot); the same set of symbols always maps to the same slot."""
        joined = "|".join(sorted(str(s).strip().upper() for s in chunk))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _cache_load_quotes_chunk(
        self, chunk: Sequence[str]
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Return a fresh cached quote dict for ``chunk``, else ``None``.

        Read errors and malformed payloads are logged misses — corrupt cache
        content forces a live re-screen, it is never served.
        """
        store = self._resolve_cache()
        if store is None:
            return None
        try:
            payload = store.get_data(self.name, self._quotes_chunk_hash(chunk), self._QUOTES_CACHE_KIND)
        except Exception as exc:
            _log.warning("cache read failed for a quote chunk (%s); fetching live", exc)
            return None
        return self._restore_cached_quotes(payload)

    def _cache_store_quotes_chunk(
        self, chunk: Sequence[str], quotes: Dict[str, Dict[str, float]]
    ) -> None:
        """Cache one chunk's quotes under the short screening TTL; never raises."""
        store = self._resolve_cache()
        if store is None:
            return
        try:
            store.set_data(
                self.name,
                self._quotes_chunk_hash(chunk),
                self._QUOTES_CACHE_KIND,
                quotes,
                ttl=_QUOTES_CACHE_TTL_SECONDS,
            )
        except Exception as exc:
            _log.warning("could not cache a %d-symbol quote chunk (%s)", len(chunk), exc)

    @staticmethod
    def _restore_cached_quotes(payload: Any) -> Optional[Dict[str, Dict[str, float]]]:
        """Rebuild a ``{ticker: {figure: float}}`` quote dict from cached JSON.

        Returns ``None`` (a miss, forcing a live fetch) if any entry is not the
        expected shape or any figure is not a finite number — a corrupt cached
        chunk must never leak fabricated-looking values into a screen.
        """
        if not isinstance(payload, dict) or not payload:
            return None
        out: Dict[str, Dict[str, float]] = {}
        for sym, quote in payload.items():
            if not isinstance(quote, dict):
                return None
            clean: Dict[str, float] = {}
            for key, value in quote.items():
                num = _to_float(value)
                if num is None:
                    return None
                clean[str(key)] = num
            out[str(sym).upper()] = clean
        return out

    @staticmethod
    def _to_yahoo_symbol(ticker: str) -> str:
        """Convert a Nasdaq/CQS class-share separator ('.', '/', '$') to Yahoo's '-'.

        The listing directories write e.g. ``BRK.B``; Yahoo Finance quotes it as
        ``BRK-B``. Result keys are still the *original* tickers so the universe
        screen can match them back.
        """
        return ticker.strip().upper().replace(".", "-").replace("/", "-").replace("$", "-")

    def _quote_chunk(self, yf: Any, chunk: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch one chunk of screening quotes (best-effort, no raise).

        Returns the chunk's own ``{ticker: figures}`` dict — empty on a failed
        or empty download — so the caller can both merge it into the overall
        result and decide whether the chunk is worth caching.
        """
        out: Dict[str, Dict[str, float]] = {}
        yahoo_by_symbol = {sym: self._to_yahoo_symbol(sym) for sym in chunk}
        try:
            data = yf.download(
                list(yahoo_by_symbol.values()),
                period=self._QUOTE_HISTORY_PERIOD,
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:
            _log.warning(
                "yf.download failed for a %d-symbol chunk (%s: %s)",
                len(chunk),
                type(exc).__name__,
                exc,
            )
            return out
        if data is None or getattr(data, "empty", True):
            _log.warning("yf.download returned no data for a %d-symbol chunk", len(chunk))
            return out

        single = len(yahoo_by_symbol) == 1
        for sym, ysym in yahoo_by_symbol.items():
            try:
                frame = self._extract_symbol_frame(data, ysym, single=single)
                quote = self._quote_from_frame(frame)
            except Exception:  # pragma: no cover - defensive per symbol
                quote = {}
            if not quote:
                continue

            # Market cap only for names that clear the cheap liquidity prefilter;
            # everything else keeps its liquidity figures but no cap (the screen
            # then excludes it conservatively rather than us guessing a cap).
            adv = quote.get("avg_dollar_volume")
            if adv is not None and adv >= self._MCAP_PREFILTER_MIN_DOLLAR_VOLUME:
                market_cap = self._fast_market_cap(yf, ysym, quote.get("price"))
                if market_cap is not None:
                    quote["market_cap"] = market_cap
            out[sym] = quote
        return out

    @staticmethod
    def _extract_symbol_frame(data: Any, yahoo_symbol: str, *, single: bool) -> Any:
        """Pull one ticker's OHLCV sub-frame out of a bulk ``yf.download`` result.

        With ``group_by="ticker"`` a multi-symbol download has two-level columns
        keyed by ticker; a single-symbol download comes back flat. Returns
        ``None`` when the symbol is absent from the payload.
        """
        columns = getattr(data, "columns", None)
        if columns is not None and getattr(columns, "nlevels", 1) > 1:
            try:
                if yahoo_symbol in set(columns.get_level_values(0)):
                    return data[yahoo_symbol]
            except Exception:  # pragma: no cover - defensive
                return None
            return None
        return data if single else None

    @staticmethod
    def _quote_from_frame(frame: Any) -> Dict[str, float]:
        """Compute ``price`` / ``avg_dollar_volume`` from one ticker's daily bars.

        ``price`` is the last finite close; ``avg_dollar_volume`` is
        ``mean(close × volume)`` over the rows where both are present. Missing
        figures are omitted, never defaulted — an empty dict means the symbol had
        no usable rows (e.g. delisted between the listing snapshot and now).
        """
        quote: Dict[str, float] = {}
        if frame is None or getattr(frame, "empty", True):
            return quote

        columns = {str(c).lower(): c for c in getattr(frame, "columns", [])}
        col_close = columns.get("close")
        col_vol = columns.get("volume")
        if col_close is None:
            return quote

        last_price: Optional[float] = None
        dollar_volumes: List[float] = []
        for _, row in frame.iterrows():
            close = _to_float(row[col_close])
            if close is None or close <= 0:
                continue
            last_price = close
            volume = _to_float(row[col_vol]) if col_vol is not None else None
            if volume is not None and volume >= 0:
                dollar_volumes.append(close * volume)

        if last_price is not None:
            quote["price"] = last_price
        if dollar_volumes:
            quote["avg_dollar_volume"] = sum(dollar_volumes) / len(dollar_volumes)
        return quote

    def _fast_info_throttle(self) -> None:
        """Sleep just enough to keep ``fast_info`` lookups under ~8 requests/s.

        The bulk ``yf.download`` path is already batched and paced, but market
        caps require one HTTP request per surviving symbol; on a full-listing
        screen that is the request volume that actually trips Yahoo's rate
        limiting. Thread-safe so a shared provider instance still paces globally.
        """
        with self._fast_info_lock:
            elapsed = time.monotonic() - self._last_fast_info_ts
            wait = self._FAST_INFO_MIN_INTERVAL_S - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_fast_info_ts = time.monotonic()

    def _fast_market_cap(self, yf: Any, yahoo_symbol: str, price: Optional[float]) -> Optional[float]:
        """Resolve one symbol's market cap via ``fast_info`` (the cheap quote endpoint).

        Prefers the endpoint's own market cap; falls back to ``last price ×
        shares outstanding`` when only shares are reported (using the bulk-download
        close when ``fast_info`` lacks a price). Returns ``None`` — never a guess —
        when neither path yields a positive figure. Each lookup is paced through
        :meth:`_fast_info_throttle` so a screening pass stays under Yahoo's rate
        limits instead of silently losing caps to 429 responses.
        """
        self._fast_info_throttle()
        try:
            fast_info = getattr(yf.Ticker(yahoo_symbol), "fast_info", None)
        except Exception as exc:
            _log.debug("fast_info unavailable for %s: %s", yahoo_symbol, exc)
            return None
        if fast_info is None:
            return None

        market_cap = self._fast_info_value(fast_info, ("market_cap", "marketCap"))
        if market_cap is not None and market_cap > 0:
            return market_cap

        shares = self._fast_info_value(fast_info, ("shares", "sharesOutstanding", "shares_outstanding"))
        if price is None or price <= 0:
            price = self._fast_info_value(
                fast_info, ("last_price", "lastPrice", "regular_market_previous_close")
            )
        if shares is not None and shares > 0 and price is not None and price > 0:
            return shares * price
        return None

    @staticmethod
    def _fast_info_value(fast_info: Any, keys: Sequence[str]) -> Optional[float]:
        """Read the first finite numeric among ``keys`` from a ``fast_info`` object.

        ``fast_info`` behaves like a lazy mapping in recent yfinance versions and
        like an attribute bag in older ones; individual key accesses can raise
        (each triggers a lazy fetch), so every access is isolated.
        """
        for key in keys:
            value: Any = None
            try:
                value = fast_info[key]
            except Exception:
                try:
                    value = getattr(fast_info, key)
                except Exception:
                    value = None
            num = _to_float(value)
            if num is not None:
                return num
        return None

    # -- yfinance import ---------------------------------------------------

    @staticmethod
    def _import_yfinance() -> Any:
        """Import yfinance lazily so the package imports even when it is absent.

        Importing at call-time (rather than module top-level) keeps the provider
        registrable on a machine without yfinance installed; the error only
        surfaces when someone actually requests data.
        """
        try:
            import yfinance as yf  # type: ignore import-not-found
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ProviderError(
                "the 'yfinance' library is not installed; cannot fetch data",
                provider="yfinance",
            ) from exc
        return yf

    # -- info --------------------------------------------------------------

    def _fetch_info(self, yf_ticker: Any, symbol: str, warnings: List[str]) -> Dict[str, Any]:
        """Return Yahoo's ``info`` dict, or an empty dict on failure."""
        try:
            info = getattr(yf_ticker, "info", None)
            if isinstance(info, dict) and info:
                return info
            # Newer yfinance prefers get_info(); fall back to it.
            getter = getattr(yf_ticker, "get_info", None)
            if callable(getter):
                fetched = getter()
                if isinstance(fetched, dict) and fetched:
                    return fetched
            warnings.append(f"{symbol}: yfinance returned no company info.")
            return {}
        except Exception as exc:
            warnings.append(f"{symbol}: failed to fetch company info ({type(exc).__name__}: {exc}).")
            return {}

    @staticmethod
    def _clean_str(value: Any) -> Optional[str]:
        """Return a stripped non-empty string, or ``None``."""
        if not isinstance(value, str):
            return None
        text = value.strip()
        return text or None

    def _extract_name(self, info: Dict[str, Any], symbol: str) -> str:
        """Pick the best available company name, falling back to the symbol itself."""
        for key in ("longName", "shortName", "displayName"):
            name = self._clean_str(info.get(key)) if info else None
            if name:
                return name
        return symbol

    def _extract_exchange(self, info: Dict[str, Any]) -> Optional[str]:
        """Prefer a human-readable exchange name, else the exchange code."""
        if not info:
            return None
        for key in ("fullExchangeName", "exchange", "exchangeName"):
            value = self._clean_str(info.get(key))
            if value:
                return value
        return None

    def _extract_currency(self, info: Dict[str, Any]) -> str:
        """Return the reporting currency, defaulting to USD when unspecified."""
        if info:
            cur = self._clean_str(info.get("currency") or info.get("financialCurrency"))
            if cur:
                return cur.upper()
        return "USD"

    # -- prices ------------------------------------------------------------

    def _fetch_prices(self, yf_ticker: Any, symbol: str, warnings: List[str]) -> List[PriceBar]:
        """Fetch ~2 years of daily OHLCV bars, oldest-first; ``[]`` on failure."""
        try:
            hist = yf_ticker.history(
                period=self._PRICE_PERIOD,
                interval=self._PRICE_INTERVAL,
                auto_adjust=False,
                actions=False,
            )
        except Exception as exc:
            warnings.append(f"{symbol}: failed to fetch price history ({type(exc).__name__}: {exc}).")
            return []

        if hist is None or getattr(hist, "empty", True):
            warnings.append(f"{symbol}: yfinance returned no price history.")
            return []

        bars: List[PriceBar] = []
        skipped = 0
        try:
            columns = {str(c).lower(): c for c in hist.columns}
            col_open = columns.get("open")
            col_high = columns.get("high")
            col_low = columns.get("low")
            col_close = columns.get("close")
            col_adj = columns.get("adj close") or columns.get("adjclose")
            col_vol = columns.get("volume")

            for idx, row in hist.iterrows():
                bar_date = _coerce_date(idx)
                close = _to_float(row[col_close]) if col_close is not None else None
                if bar_date is None or close is None:
                    skipped += 1
                    continue
                open_ = _to_float(row[col_open]) if col_open is not None else None
                high = _to_float(row[col_high]) if col_high is not None else None
                low = _to_float(row[col_low]) if col_low is not None else None
                volume = _to_float(row[col_vol]) if col_vol is not None else None
                adj_close = _to_float(row[col_adj]) if col_adj is not None else None
                bars.append(
                    PriceBar(
                        date=bar_date,
                        open=open_ if open_ is not None else close,
                        high=high if high is not None else close,
                        low=low if low is not None else close,
                        close=close,
                        adj_close=adj_close,
                        volume=volume if volume is not None else 0.0,
                    )
                )
        except Exception as exc:
            warnings.append(f"{symbol}: error parsing price history ({type(exc).__name__}: {exc}).")

        bars.sort(key=lambda b: b.date)  # oldest first, per SecurityData contract
        if skipped:
            warnings.append(f"{symbol}: skipped {skipped} price rows with missing date/close.")
        if not bars:
            warnings.append(f"{symbol}: no usable price bars after parsing.")
        return bars

    # -- fundamentals ------------------------------------------------------

    def _fetch_fundamentals(
        self, yf_ticker: Any, symbol: str, warnings: List[str]
    ) -> List[FundamentalsPeriod]:
        """Build annual (+ latest quarter) :class:`FundamentalsPeriod` rows, newest-first."""
        annual = self._fetch_statement_set(
            yf_ticker,
            symbol,
            warnings,
            income_attr="financials",
            cashflow_attr="cashflow",
            balance_attr="balance_sheet",
            label_prefix="FY",
            quarterly=False,
            limit=self._MAX_ANNUAL_PERIODS,
        )
        quarterly = self._fetch_statement_set(
            yf_ticker,
            symbol,
            warnings,
            income_attr="quarterly_financials",
            cashflow_attr="quarterly_cashflow",
            balance_attr="quarterly_balance_sheet",
            label_prefix="Q",
            quarterly=True,
            limit=self._MAX_QUARTERLY_PERIODS,
        )

        # Merge: keep the latest quarter ahead of the annuals (newest-first), but
        # avoid duplicating a period that the annual frame already covers.
        periods: List[FundamentalsPeriod] = []
        seen: Set[_dt.date] = set()
        for period in quarterly + annual:
            if period.period_end in seen:
                continue
            seen.add(period.period_end)
            periods.append(period)
        periods.sort(key=lambda p: p.period_end, reverse=True)  # newest first

        if not periods:
            warnings.append(f"{symbol}: no fundamentals (income/cash-flow/balance-sheet) available.")
        return periods

    def _fetch_statement_set(
        self,
        yf_ticker: Any,
        symbol: str,
        warnings: List[str],
        *,
        income_attr: str,
        cashflow_attr: str,
        balance_attr: str,
        label_prefix: str,
        quarterly: bool,
        limit: int,
    ) -> List[FundamentalsPeriod]:
        """Read one trio of statement frames and map each period column to a model."""
        income = self._read_statement_frame(yf_ticker, income_attr, symbol, warnings)
        cashflow = self._read_statement_frame(yf_ticker, cashflow_attr, symbol, warnings)
        balance = self._read_statement_frame(yf_ticker, balance_attr, symbol, warnings)

        # Period columns come from whichever frame is present; income is primary.
        period_dates = self._collect_period_dates(income, cashflow, balance)
        if not period_dates:
            return []

        periods: List[FundamentalsPeriod] = []
        for col, period_end in period_dates[:limit]:
            try:
                period = self._build_period(
                    period_end=period_end,
                    label=self._period_label(period_end, label_prefix, quarterly),
                    income_col=self._column_values(income, col),
                    cashflow_col=self._column_values(cashflow, col),
                    balance_col=self._column_values(balance, col),
                )
                periods.append(period)
            except Exception as exc:  # defensive: one bad column never kills the rest
                warnings.append(
                    f"{symbol}: failed to map {'quarter' if quarterly else 'year'} "
                    f"ending {period_end} ({type(exc).__name__}: {exc})."
                )
        return periods

    @staticmethod
    def _read_statement_frame(yf_ticker: Any, attr: str, symbol: str, warnings: List[str]) -> Any:
        """Return a yfinance statement DataFrame (or ``None``) without raising."""
        try:
            frame = getattr(yf_ticker, attr, None)
            if frame is None or getattr(frame, "empty", True):
                return None
            return frame
        except Exception as exc:
            warnings.append(f"{symbol}: failed to fetch {attr} ({type(exc).__name__}: {exc}).")
            return None

    @staticmethod
    def _collect_period_dates(*frames: Any) -> List[Any]:
        """Union the column labels across frames, returning ``[(col, date)]`` newest-first."""
        seen: Dict[_dt.date, Any] = {}
        for frame in frames:
            if frame is None:
                continue
            for col in list(getattr(frame, "columns", [])):
                col_date = _coerce_date(col)
                if col_date is not None and col_date not in seen:
                    seen[col_date] = col
        ordered = sorted(seen.items(), key=lambda kv: kv[0], reverse=True)
        return [(col, col_date) for col_date, col in ordered]

    @staticmethod
    def _column_values(frame: Any, col: Any) -> Dict[str, Any]:
        """Extract one period column from a frame as a ``{row_label: value}`` dict."""
        if frame is None or col is None:
            return {}
        try:
            if col not in frame.columns:
                return {}
            series = frame[col]
            return {str(idx): val for idx, val in series.items()}
        except Exception:  # pragma: no cover - defensive
            return {}

    @staticmethod
    def _period_label(period_end: _dt.date, prefix: str, quarterly: bool) -> str:
        """Render a human period label, e.g. ``FY2025`` or ``Q1 2026``."""
        if quarterly:
            quarter = (period_end.month - 1) // 3 + 1
            return f"Q{quarter} {period_end.year}"
        return f"{prefix}{period_end.year}"

    @staticmethod
    def _pick(values: Dict[str, Any], aliases: List[str]) -> Optional[float]:
        """Return the first finite value among ``aliases`` (case-insensitive)."""
        if not values:
            return None
        # Direct hits first (cheap), then a case-insensitive scan.
        for alias in aliases:
            if alias in values:
                num = _to_float(values[alias])
                if num is not None:
                    return num
        lowered = {str(k).lower(): v for k, v in values.items()}
        for alias in aliases:
            key = alias.lower()
            if key in lowered:
                num = _to_float(lowered[key])
                if num is not None:
                    return num
        return None

    def _build_period(
        self,
        *,
        period_end: _dt.date,
        label: str,
        income_col: Dict[str, Any],
        cashflow_col: Dict[str, Any],
        balance_col: Dict[str, Any],
    ) -> FundamentalsPeriod:
        """Map one period's raw statement rows into a :class:`FundamentalsPeriod`.

        Derived margins, returns and FCF are computed only from line items that
        are actually present; any missing input leaves the derived field ``None``.
        """
        # --- raw line items ---
        revenue = self._pick(income_col, _INCOME_ROW_ALIASES["revenue"])
        gross_profit = self._pick(income_col, _INCOME_ROW_ALIASES["gross_profit"])
        operating_income = self._pick(income_col, _INCOME_ROW_ALIASES["operating_income"])
        net_income = self._pick(income_col, _INCOME_ROW_ALIASES["net_income"])
        ebitda = self._pick(income_col, _INCOME_ROW_ALIASES["ebitda"])
        eps_diluted = self._pick(income_col, _INCOME_ROW_ALIASES["eps_diluted"])
        interest_expense = self._pick(income_col, _INCOME_ROW_ALIASES["interest_expense"])
        shares_from_income = self._pick(income_col, _INCOME_ROW_ALIASES["shares_diluted"])

        operating_cash_flow = self._pick(cashflow_col, _CASHFLOW_ROW_ALIASES["operating_cash_flow"])
        capex = self._pick(cashflow_col, _CASHFLOW_ROW_ALIASES["capex"])
        free_cash_flow = self._pick(cashflow_col, _CASHFLOW_ROW_ALIASES["free_cash_flow"])

        total_assets = self._pick(balance_col, _BALANCE_ROW_ALIASES["total_assets"])
        total_debt = self._pick(balance_col, _BALANCE_ROW_ALIASES["total_debt"])
        long_term_debt = self._pick(balance_col, _BALANCE_ROW_ALIASES["long_term_debt"])
        current_debt = self._pick(balance_col, _BALANCE_ROW_ALIASES["current_debt"])
        cash = self._pick(balance_col, _BALANCE_ROW_ALIASES["cash_and_equivalents"])
        total_equity = self._pick(balance_col, _BALANCE_ROW_ALIASES["total_equity"])
        current_assets = self._pick(balance_col, _BALANCE_ROW_ALIASES["current_assets"])
        current_liabilities = self._pick(balance_col, _BALANCE_ROW_ALIASES["current_liabilities"])
        inventory = self._pick(balance_col, _BALANCE_ROW_ALIASES["inventory"])
        shares_from_balance = self._pick(balance_col, _BALANCE_ROW_ALIASES["shares_outstanding"])

        shares_diluted = shares_from_income if shares_from_income is not None else shares_from_balance

        # If total debt isn't reported directly, reconstruct it from the parts.
        if total_debt is None:
            debt_parts = [d for d in (long_term_debt, current_debt) if d is not None]
            if debt_parts:
                total_debt = sum(debt_parts)

        # Free cash flow = operating cash flow + capex (capex is reported negative
        # by Yahoo, so addition is correct).
        if free_cash_flow is None and operating_cash_flow is not None and capex is not None:
            free_cash_flow = operating_cash_flow + capex

        # --- derived margins (fractions) ---
        gross_margin = _safe_div(gross_profit, revenue)
        operating_margin = _safe_div(operating_income, revenue)
        fcf_margin = _safe_div(free_cash_flow, revenue)

        # --- returns ---
        roe = _safe_div(net_income, total_equity)
        roa = _safe_div(net_income, total_assets)
        roic = self._compute_roic(
            operating_income=operating_income,
            net_income=net_income,
            total_debt=total_debt,
            total_equity=total_equity,
            cash=cash,
        )

        # --- liquidity & solvency ---
        current_ratio = _safe_div(current_assets, current_liabilities)
        quick_assets = None
        if current_assets is not None:
            quick_assets = current_assets - (inventory if inventory is not None else 0.0)
        quick_ratio = _safe_div(quick_assets, current_liabilities)
        debt_to_equity = _safe_div(total_debt, total_equity)
        interest_coverage = None
        if interest_expense is not None and abs(interest_expense) > 1e-9 and operating_income is not None:
            # Interest expense is usually reported as a positive magnitude or a
            # negative number; use its magnitude for the coverage ratio.
            interest_coverage = _safe_div(operating_income, abs(interest_expense))

        return FundamentalsPeriod(
            period_end=period_end,
            period_label=label,
            revenue=revenue,
            gross_profit=gross_profit,
            operating_income=operating_income,
            net_income=net_income,
            ebitda=ebitda,
            eps_diluted=eps_diluted,
            free_cash_flow=free_cash_flow,
            operating_cash_flow=operating_cash_flow,
            capex=capex,
            total_assets=total_assets,
            total_debt=total_debt,
            cash_and_equivalents=cash,
            total_equity=total_equity,
            shares_diluted=shares_diluted,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            fcf_margin=fcf_margin,
            roic=roic,
            roe=roe,
            roa=roa,
            current_ratio=current_ratio,
            quick_ratio=quick_ratio,
            debt_to_equity=debt_to_equity,
            interest_coverage=interest_coverage,
        )

    @staticmethod
    def _compute_roic(
        *,
        operating_income: Optional[float],
        net_income: Optional[float],
        total_debt: Optional[float],
        total_equity: Optional[float],
        cash: Optional[float],
    ) -> Optional[float]:
        """Return on invested capital ≈ NOPAT / (debt + equity − cash).

        Uses a flat 21% notional tax shield on operating income to approximate
        NOPAT when operating income is available, otherwise falls back to net
        income. Invested capital nets out cash. Returns ``None`` if the inputs
        needed for a non-degenerate denominator are missing.
        """
        if operating_income is not None:
            nopat: Optional[float] = operating_income * (1.0 - 0.21)
        elif net_income is not None:
            nopat = net_income
        else:
            return None

        debt = total_debt if total_debt is not None else 0.0
        equity = total_equity
        if equity is None:
            return None
        invested = debt + equity - (cash if cash is not None else 0.0)
        return _safe_div(nopat, invested)

    # -- valuation ---------------------------------------------------------

    @staticmethod
    def _latest_with(
        fundamentals: List[FundamentalsPeriod], attr: str
    ) -> Optional[FundamentalsPeriod]:
        """Return the newest period whose ``attr`` is populated, or ``None``.

        Quarterly statements (the very latest period) frequently omit balance-sheet
        lines; using the newest period that *actually has* the needed datum avoids
        both fabricating a zero and discarding a usable annual figure.
        """
        for period in fundamentals:  # already newest-first
            if getattr(period, attr, None) is not None:
                return period
        return None

    def _build_valuation(
        self,
        info: Dict[str, Any],
        fundamentals: List[FundamentalsPeriod],
        market_cap: Optional[float],
    ) -> ValuationSnapshot:
        """Assemble a :class:`ValuationSnapshot` from Yahoo's info dict + statements.

        Multiples reported directly by Yahoo are preferred. Where Yahoo omits one
        we recompute it from ``market_cap`` and the newest fundamentals period that
        actually carries the required input — never by treating a missing input as
        zero. Any multiple whose inputs are absent is left ``None``.
        """
        info = info or {}
        enterprise_value = _to_float(info.get("enterpriseValue"))
        pe = _to_float(info.get("trailingPE"))
        forward_pe = _to_float(info.get("forwardPE"))
        ev_ebitda = _to_float(info.get("enterpriseToEbitda"))
        ev_sales = _to_float(info.get("enterpriseToRevenue"))
        p_b = _to_float(info.get("priceToBook"))
        p_s = _to_float(info.get("priceToSalesTrailing12Months"))
        peg = _to_float(info.get("pegRatio") or info.get("trailingPegRatio"))

        # Reconstruct enterprise value (EV = market cap + total debt − cash) from
        # the newest period that reports debt; cash from that same period if any.
        if enterprise_value is None and market_cap is not None:
            debt_period = self._latest_with(fundamentals, "total_debt")
            if debt_period is not None:
                debt = debt_period.total_debt or 0.0
                cash = debt_period.cash_and_equivalents or 0.0
                enterprise_value = market_cap + debt - cash

        rev_period = self._latest_with(fundamentals, "revenue")
        ebitda_period = self._latest_with(fundamentals, "ebitda")
        equity_period = self._latest_with(fundamentals, "total_equity")
        fcf_period = self._latest_with(fundamentals, "free_cash_flow")

        # Recompute multiples only when Yahoo omitted them and the inputs exist.
        if ev_ebitda is None and ebitda_period is not None:
            ev_ebitda = _safe_div(enterprise_value, ebitda_period.ebitda)
        if ev_sales is None and rev_period is not None:
            ev_sales = _safe_div(enterprise_value, rev_period.revenue)
        if p_s is None and rev_period is not None:
            p_s = _safe_div(market_cap, rev_period.revenue)
        if p_b is None and equity_period is not None:
            p_b = _safe_div(market_cap, equity_period.total_equity)

        # Price-to-free-cash-flow has no direct Yahoo field; derive it when possible.
        p_fcf = None
        if fcf_period is not None:
            p_fcf = _safe_div(market_cap, fcf_period.free_cash_flow)

        return ValuationSnapshot(
            market_cap=market_cap,
            enterprise_value=enterprise_value,
            pe=pe,
            forward_pe=forward_pe,
            ev_ebitda=ev_ebitda,
            ev_sales=ev_sales,
            p_fcf=p_fcf,
            p_b=p_b,
            p_s=p_s,
            peg=peg,
        )

    # -- news --------------------------------------------------------------

    def _fetch_news(self, yf_ticker: Any, symbol: str, warnings: List[str]) -> List[NewsItem]:
        """Fetch recent Yahoo news headlines, newest-first; ``[]`` on failure."""
        try:
            raw = getattr(yf_ticker, "news", None)
            if not raw:
                getter = getattr(yf_ticker, "get_news", None)
                if callable(getter):
                    raw = getter()
        except Exception as exc:
            warnings.append(f"{symbol}: failed to fetch news ({type(exc).__name__}: {exc}).")
            return []

        if not raw or not isinstance(raw, list):
            warnings.append(f"{symbol}: no recent news returned by yfinance.")
            return []

        items: List[NewsItem] = []
        skipped = 0
        for entry in raw[: self._MAX_NEWS]:
            try:
                parsed = self._parse_news_entry(entry)
            except Exception:  # pragma: no cover - defensive
                parsed = None
            if parsed is None:
                skipped += 1
                continue
            items.append(parsed)

        items.sort(key=lambda n: n.published, reverse=True)  # newest first
        if skipped:
            warnings.append(f"{symbol}: skipped {skipped} news items with no title/timestamp.")
        return items

    def _parse_news_entry(self, entry: Any) -> Optional[NewsItem]:
        """Parse one yfinance news dict across the older and newer Yahoo schemas."""
        if not isinstance(entry, dict):
            return None
        # Newer yfinance nests the payload under "content".
        content = entry.get("content") if isinstance(entry.get("content"), dict) else entry

        title = self._clean_str(content.get("title")) or self._clean_str(entry.get("title"))
        if not title:
            return None

        published = self._extract_news_datetime(entry, content)
        if published is None:
            return None

        source = (
            self._extract_news_source(content)
            or self._clean_str(entry.get("publisher"))
            or "Yahoo Finance"
        )
        url = self._extract_news_url(entry, content)
        summary = self._clean_str(content.get("summary")) or self._clean_str(content.get("description"))

        return NewsItem(
            published=published,
            title=title,
            source=source,
            url=url,
            summary=summary,
            sentiment=None,  # honesty: we do not fabricate a sentiment score here.
        )

    @staticmethod
    def _extract_news_datetime(entry: Dict[str, Any], content: Dict[str, Any]) -> Optional[_dt.datetime]:
        """Resolve a publish datetime across both schemas (epoch or ISO string)."""
        # Older schema: providerPublishTime is unix seconds.
        epoch = entry.get("providerPublishTime")
        dt = _coerce_datetime(epoch)
        if dt is not None:
            return dt
        # Newer schema: ISO 8601 string under pubDate / displayTime.
        for key in ("pubDate", "displayTime", "providerPublishTime"):
            dt = _coerce_datetime(content.get(key))
            if dt is not None:
                return dt
        return None

    @staticmethod
    def _extract_news_source(content: Dict[str, Any]) -> Optional[str]:
        """Pull the publisher name from the (possibly nested) provider field."""
        provider = content.get("provider")
        if isinstance(provider, dict):
            name = provider.get("displayName") or provider.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        if isinstance(provider, str) and provider.strip():
            return provider.strip()
        return None

    @staticmethod
    def _extract_news_url(entry: Dict[str, Any], content: Dict[str, Any]) -> Optional[str]:
        """Resolve the canonical article URL across both schemas."""
        # Newer schema: content.canonicalUrl.url or content.clickThroughUrl.url
        for key in ("canonicalUrl", "clickThroughUrl"):
            link = content.get(key)
            if isinstance(link, dict):
                url = link.get("url")
                if isinstance(url, str) and url.strip():
                    return url.strip()
        # Older schema: top-level "link".
        link = entry.get("link")
        if isinstance(link, str) and link.strip():
            return link.strip()
        return None


__all__ = ["YFinanceProvider"]
