"""SEC EDGAR data provider — fundamentals and recent filings from official APIs.

This provider sources **only primary, public regulatory data** from the U.S.
Securities and Exchange Commission's EDGAR system:

* ``https://www.sec.gov/files/company_tickers.json`` — the official ticker → CIK
  map, used to resolve a ticker to its Central Index Key (CIK).
* ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`` — XBRL
  "company facts", from which we read US-GAAP line items (Revenues,
  NetIncomeLoss, Assets, …) and fold them into :class:`FundamentalsPeriod`.
* ``https://data.sec.gov/submissions/CIK##########.json`` — the recent-filings
  index, from which we build the :class:`Filing` list (form type, filed date,
  document URL).
* ``https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}`` — the raw
  Form 4 ``ownershipDocument`` XML for each recent insider filing, from which we
  build the :class:`InsiderTransaction` list (reporting owner, role, transaction
  date/code/shares/price).

Honesty notes (these are load-bearing, not decoration):

* This is a **research/screening** input, not advice and not a predictor. EDGAR
  reports what issuers filed; it does not tell you what a stock will do.
* We never fabricate. A US-GAAP concept that an issuer did not tag is left
  ``None`` on the resulting :class:`FundamentalsPeriod`, and a human-readable note
  is appended to ``SecurityData.data_warnings`` so the gap is auditable.
* EDGAR asks API consumers to identify themselves and to stay under ~10 requests
  per second. We send a descriptive ``User-Agent`` from
  :attr:`Settings.sec_user_agent` on every request and apply a small inter-request
  delay so a many-ticker scan stays well-mannered.

Capabilities advertised: ``{"fundamentals", "filings", "insider"}``. EDGAR does
not provide prices, news, valuation multiples, or a screening universe, so this
provider does not advertise those.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import time
import xml.etree.ElementTree as _ET
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from convexity.core.config import Settings, get_settings
from convexity.core.exceptions import DataUnavailable, ProviderError, RateLimited
from convexity.core.logging import get_logger
from convexity.core.models import (
    Filing,
    FundamentalsPeriod,
    InsiderTransaction,
    SecurityData,
)
from convexity.core.registry import register_provider

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Endpoint + behavioural constants
# ---------------------------------------------------------------------------

_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_FILING_INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_doc}"
)
_FILING_FOLDER_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/"

# SEC asks consumers to stay under ~10 requests/second. We deliberately throttle a
# little harder than that to be a good citizen across a multi-ticker scan.
_MIN_REQUEST_INTERVAL_S = 0.15

# The CIK map is large (~700k+ tickers across forms); cache it on disk for a day.
_CIK_MAP_TTL_S = 86_400

# How many of the most-recent filings to surface by default.
_DEFAULT_FILINGS_LIMIT = 40

#: Maximum number of Form 4 filings fetched and parsed per ticker. Each Form 4 is
#: a separate Archives request, so this cap bounds both latency and SEC load for a
#: multi-ticker scan while still capturing recent insider activity. Ten filings
#: comfortably covers a year of insider trading at a typical small/micro-cap.
_FORM4_MAX_FILINGS = 10

#: Only Form 4 filings filed within this many days are considered "recent" insider
#: activity (12 months). Older activity is stale for the ownership/management
#: analyzers and is deliberately excluded rather than silently mixed in.
_FORM4_LOOKBACK_DAYS = 365

#: SEC Form 4 transaction codes → Convexity ``InsiderTransaction.transaction_type``.
#: Only ``P`` (open-market/private purchase) and ``S`` (open-market/private sale)
#: represent voluntary, at-risk trades; the ownership/management analyzers weight
#: those most heavily. Everything else is mapped honestly to a non-market label
#: (``award``, ``exercise``) or preserved verbatim as ``other:<code>`` so no
#: compensation grant can masquerade as an open-market buy.
_FORM4_TRANSACTION_CODE_MAP: Dict[str, str] = {
    "P": "buy",       # open-market or private purchase
    "S": "sell",      # open-market or private sale
    "A": "award",     # grant, award or other acquisition (compensation, not a buy)
    "M": "exercise",  # exercise/conversion of a derivative security
}

# US-GAAP concept → FundamentalsPeriod attribute. The first concept present for a
# given period wins (concepts are ordered by how directly they map to the field).
# Every concept here is a standard us-gaap XBRL tag filed in 10-K/10-Q reports.
_REVENUE_CONCEPTS: Tuple[str, ...] = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
)
_NET_INCOME_CONCEPTS: Tuple[str, ...] = (
    "NetIncomeLoss",
    "ProfitLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
)
_GROSS_PROFIT_CONCEPTS: Tuple[str, ...] = ("GrossProfit",)
_OPERATING_INCOME_CONCEPTS: Tuple[str, ...] = (
    "OperatingIncomeLoss",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
)
_OCF_CONCEPTS: Tuple[str, ...] = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
_CAPEX_CONCEPTS: Tuple[str, ...] = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
)
_TOTAL_ASSETS_CONCEPTS: Tuple[str, ...] = ("Assets",)
_TOTAL_EQUITY_CONCEPTS: Tuple[str, ...] = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
_CASH_CONCEPTS: Tuple[str, ...] = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
)
_LONG_TERM_DEBT_CONCEPTS: Tuple[str, ...] = (
    "LongTermDebtNoncurrent",
    "LongTermDebt",
)
_SHORT_TERM_DEBT_CONCEPTS: Tuple[str, ...] = (
    "LongTermDebtCurrent",
    "DebtCurrent",
)
_EPS_DILUTED_CONCEPTS: Tuple[str, ...] = (
    "EarningsPerShareDiluted",
    "IncomeLossFromContinuingOperationsPerDilutedShare",
)
_SHARES_DILUTED_CONCEPTS: Tuple[str, ...] = (
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    "WeightedAverageNumberOfSharesOutstandingBasic",
)
# Point-in-time share counts, used as a FALLBACK for ``shares_diluted`` when the
# issuer tags no weighted-average series at all (common among smaller filers —
# e.g. VTVT tags only CommonStockSharesOutstanding). Shares *outstanding* is a
# point-in-time count rather than a period average; for the derived-market-cap
# fallback (last close x shares) that is, if anything, the better input. The
# flow ingest runs first and "first writer wins", so this never overrides a
# weighted-average figure.
_SHARES_OUTSTANDING_INSTANT_CONCEPTS: Tuple[str, ...] = (
    "CommonStockSharesOutstanding",
    "CommonStockSharesIssued",
)

# Flow (duration) vs. instant (point-in-time) concepts. Flow facts have a
# ``start``+``end`` window; instant facts have only ``end``. Knowing which is which
# lets us group them into coherent fiscal periods.
_FLOW_CONCEPTS: Set[str] = set(
    _REVENUE_CONCEPTS
    + _NET_INCOME_CONCEPTS
    + _GROSS_PROFIT_CONCEPTS
    + _OPERATING_INCOME_CONCEPTS
    + _OCF_CONCEPTS
    + _CAPEX_CONCEPTS
    + _EPS_DILUTED_CONCEPTS
    + _SHARES_DILUTED_CONCEPTS
)
_INSTANT_CONCEPTS: Set[str] = set(
    _TOTAL_ASSETS_CONCEPTS
    + _TOTAL_EQUITY_CONCEPTS
    + _CASH_CONCEPTS
    + _LONG_TERM_DEBT_CONCEPTS
    + _SHORT_TERM_DEBT_CONCEPTS
    + _SHARES_OUTSTANDING_INSTANT_CONCEPTS
)


# ---------------------------------------------------------------------------
# Tiny, dependency-light on-disk JSON cache
# ---------------------------------------------------------------------------


class _JsonFileCache:
    """A minimal, thread-safe, TTL'd JSON cache backed by files on disk.

    The provider only needs to cache a handful of JSON blobs (the CIK map, plus
    per-company facts/submissions). Rather than couple to a sibling cache module
    that may not exist yet, this keeps the provider self-contained and import-safe.
    A failed cache read or write is *never* fatal: it degrades to a live fetch.
    """

    def __init__(self, base_dir: str) -> None:
        self._dir = os.path.join(base_dir, "sec_edgar")
        self._lock = threading.Lock()
        try:
            os.makedirs(self._dir, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            _log.warning("sec_edgar cache dir unavailable (%s); caching disabled", exc)
            self._dir = ""

    def _path(self, key: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return os.path.join(self._dir, f"{safe}.json")

    def get(self, key: str, ttl_seconds: int) -> Optional[Any]:
        """Return the cached value for ``key`` if present and fresh, else ``None``."""
        if not self._dir:
            return None
        path = self._path(key)
        try:
            with self._lock:
                if not os.path.exists(path):
                    return None
                age = time.time() - os.path.getmtime(path)
                if age > ttl_seconds:
                    return None
                with open(path, encoding="utf-8") as fh:
                    return json.load(fh)
        except (OSError, ValueError) as exc:  # pragma: no cover - defensive
            _log.debug("sec_edgar cache read failed for %r: %s", key, exc)
            return None

    def set(self, key: str, value: Any) -> None:
        """Persist ``value`` for ``key`` (best-effort; failures are swallowed)."""
        if not self._dir:
            return
        path = self._path(key)
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with self._lock:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(value, fh)
                os.replace(tmp, path)
        except (OSError, TypeError) as exc:  # pragma: no cover - defensive
            _log.debug("sec_edgar cache write failed for %r: %s", key, exc)
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(value: Optional[str]) -> Optional[_dt.date]:
    """Parse an ISO ``YYYY-MM-DD`` date string, returning ``None`` on any failure."""
    if not value:
        return None
    try:
        return _dt.date.fromisoformat(value[:10])
    except (ValueError, TypeError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    """Coerce a JSON value to ``float``; return ``None`` if it is not numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_concept(units: Dict[str, Any], concepts: Tuple[str, ...]) -> Optional[str]:
    """Return the first concept name in ``concepts`` that is present in ``units``."""
    for concept in concepts:
        if concept in units:
            return concept
    return None


def _period_label(period_end: _dt.date, form: str, fp: Optional[str], fy: Optional[int]) -> str:
    """Build a human-readable period label like ``"FY2025"`` or ``"Q1 2026"``.

    Falls back to the period-end year when the filing's fiscal-period metadata is
    absent, so a label is always present and never fabricated beyond what EDGAR
    reported.
    """
    is_annual = (form or "").upper().startswith("10-K") or (fp or "").upper() == "FY"
    year = fy if isinstance(fy, int) else period_end.year
    if is_annual:
        return f"FY{year}"
    quarter = (fp or "").upper()
    if quarter in {"Q1", "Q2", "Q3", "Q4"}:
        return f"{quarter} {year}"
    # Derive a quarter from the calendar month as a transparent best-effort label.
    q = (period_end.month - 1) // 3 + 1
    return f"Q{q} {year}"


@register_provider
class SecEdgarProvider:
    """``DataProvider`` for SEC EDGAR fundamentals and filings.

    The provider resolves a ticker to its CIK via the official company-tickers
    map (cached on disk), then assembles:

    * ``SecurityData.fundamentals`` — a newest-first list of
      :class:`FundamentalsPeriod` built from US-GAAP companyfacts (annual periods
      preferred, with the most recent quarters retained when present).
    * ``SecurityData.filings`` — a newest-first list of recent :class:`Filing`
      references (form type, filed date, document URL) from the submissions index.

    It is registered under the stable name ``"sec_edgar"`` and advertises the
    capabilities ``{"fundamentals", "filings"}``.
    """

    #: Maximum fiscal periods retained per fundamentals series (keeps payloads sane).
    MAX_PERIODS = 12

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Create the provider, wiring up settings, an HTTP session and the cache.

        Args:
            settings: Optional :class:`Settings`; falls back to the cached
                process-wide settings (which is what the registry uses when it
                instantiates the provider with no arguments).
        """
        self._settings = settings or get_settings()
        self._cache = _JsonFileCache(self._settings.data_dir)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self._settings.sec_user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json",
                # data.sec.gov is content-negotiated; Host is set automatically by
                # requests, but being explicit about Accept keeps responses JSON.
            }
        )
        self._last_request_ts = 0.0
        self._throttle_lock = threading.Lock()
        # In-process memo of the resolved CIK map so repeated lookups in one scan
        # do not even touch the disk cache.
        self._cik_map: Optional[Dict[str, int]] = None

    # -- DataProvider contract ------------------------------------------------

    @property
    def name(self) -> str:
        """Stable identifier recorded in ``SecurityData.data_sources``."""
        return "sec_edgar"

    @property
    def capabilities(self) -> Set[str]:
        """Capabilities EDGAR truly fills: fundamentals, filings and Form 4 insiders."""
        return {"fundamentals", "filings", "insider"}

    def supports(self, capability: str) -> bool:
        """Return whether this provider advertises ``capability``."""
        return capability in self.capabilities

    def get_security_data(self, ticker: str) -> SecurityData:
        """Assemble fundamentals + recent filings for ``ticker`` from EDGAR.

        Unknown fields stay ``None`` and every gap (no CIK, missing companyfacts,
        un-tagged concepts) is recorded in ``data_warnings`` — never fabricated.

        Raises:
            DataUnavailable: when the ticker cannot be resolved to a CIK.
            ProviderError / RateLimited: on an HTTP or parse failure.
        """
        ticker_norm = (ticker or "").strip().upper()
        if not ticker_norm:
            raise DataUnavailable("empty ticker", ticker=ticker)

        cik, company_name = self._resolve_cik(ticker_norm)

        data = SecurityData(
            ticker=ticker_norm,
            name=company_name or ticker_norm,
            as_of=_dt.datetime.now(_dt.timezone.utc),
            data_sources=[self.name],
        )

        # --- Filings (from submissions) --------------------------------------
        submissions_payload: Optional[Dict[str, Any]] = None
        try:
            submissions = self._fetch_submissions(cik)
            submissions_payload = submissions
            if not company_name and isinstance(submissions, dict):
                data.name = submissions.get("name") or data.name
            self._populate_company_meta(data, submissions)
            data.filings = self._extract_filings(cik, submissions, _DEFAULT_FILINGS_LIMIT)
            if not data.filings:
                data.data_warnings.append("sec_edgar: no recent filings found in submissions index")
        except DataUnavailable as exc:
            data.data_warnings.append(f"sec_edgar: filings unavailable ({exc})")
        except ProviderError as exc:
            data.data_warnings.append(f"sec_edgar: filings fetch failed ({exc})")

        # --- Insider transactions (from Form 4 filings) -----------------------
        if submissions_payload is not None:
            try:
                data.insider_transactions = self.get_insider_transactions(
                    ticker_norm, cik=cik, submissions=submissions_payload
                )
                if not data.insider_transactions:
                    data.data_warnings.append(
                        "sec_edgar: no Form 4 insider transactions in the last 12 months"
                    )
            except (DataUnavailable, ProviderError) as exc:
                data.data_warnings.append(
                    f"sec_edgar: insider transactions unavailable ({exc})"
                )
        else:
            data.data_warnings.append(
                "sec_edgar: insider transactions unavailable (submissions index missing)"
            )

        # --- Fundamentals (from companyfacts) --------------------------------
        try:
            facts = self._fetch_company_facts(cik)
            periods = self._extract_fundamentals(facts)
            data.fundamentals = periods
            if not periods:
                data.data_warnings.append(
                    "sec_edgar: companyfacts present but no recognised US-GAAP "
                    "fundamentals could be extracted"
                )
        except DataUnavailable as exc:
            data.data_warnings.append(f"sec_edgar: fundamentals unavailable ({exc})")
        except ProviderError as exc:
            data.data_warnings.append(f"sec_edgar: fundamentals fetch failed ({exc})")

        return data

    # -- Public convenience: filings only ------------------------------------

    def get_filings(self, ticker: str, limit: int = _DEFAULT_FILINGS_LIMIT) -> List[Filing]:
        """Fetch only the recent :class:`Filing` list for a known ``ticker``.

        Provided so catalyst/news/ownership analyzers can reuse EDGAR filings
        without paying for a full :meth:`get_security_data` assembly. Returns an
        empty list (never raises) when the ticker has no resolvable CIK or no
        filings, so a single bad ticker cannot crash a scan.

        Args:
            ticker: The equity ticker symbol (case-insensitive).
            limit: Maximum number of most-recent filings to return.

        Returns:
            Newest-first list of :class:`Filing`. Empty if none are available.
        """
        ticker_norm = (ticker or "").strip().upper()
        if not ticker_norm:
            return []
        try:
            cik, _ = self._resolve_cik(ticker_norm)
            submissions = self._fetch_submissions(cik)
            return self._extract_filings(cik, submissions, limit)
        except (DataUnavailable, ProviderError) as exc:
            _log.info("sec_edgar: filings for %s unavailable: %s", ticker_norm, exc)
            return []

    def resolve_cik(self, ticker: str) -> Optional[int]:
        """Return the integer CIK for ``ticker`` (or ``None`` if unknown).

        A convenience wrapper around the cached company-tickers map for callers
        (e.g. an ownership/insider analyzer) that need the CIK to build their own
        EDGAR URLs. Never raises.
        """
        try:
            cik, _ = self._resolve_cik((ticker or "").strip().upper())
            return cik
        except (DataUnavailable, ProviderError):
            return None

    # -- Public convenience: insider transactions (Form 4) --------------------

    def get_insider_transactions(
        self,
        ticker: str,
        *,
        cik: Optional[int] = None,
        submissions: Optional[Dict[str, Any]] = None,
    ) -> List[InsiderTransaction]:
        """Return recent Form 4 insider transactions for ``ticker``, newest first.

        Selects Form ``4`` filings from the submissions index filed within the
        last :data:`_FORM4_LOOKBACK_DAYS` days (most recent first, capped at
        :data:`_FORM4_MAX_FILINGS`), fetches each filing's ``ownershipDocument``
        XML from the EDGAR Archives, and parses the non-derivative transactions
        into :class:`InsiderTransaction` models.

        Robustness contract:

        * Every HTTP request goes through the shared throttle.
        * A single *deterministically* bad Form 4 (missing XML, malformed
          document) is skipped (logged), never fatal for the ticker.
        * Parsed results are cached on disk under ``form4_<TICKER>`` (inside the
          provider's ``sec_edgar`` cache namespace) so repeat scans are cheap —
          but **only** when every selected filing was either parsed or failed for
          a deterministic (parse/404) reason. A transient failure (network error,
          5xx) skips the cache write so a later, healthy scan refetches instead
          of serving a stale partial/empty list for ``cache_ttl_seconds``.
        * A rate-limit signal (:class:`RateLimited`, i.e. HTTP 429/403) aborts
          the Form 4 loop and propagates: continuing would just hammer the SEC,
          and caching the partial result would later masquerade as "no insider
          activity" — a factually false claim this provider must never make.

        Args:
            ticker: The equity ticker (case-insensitive); used for the cache key
                and resolved to a CIK when ``cik`` is not supplied.
            cik: Optional pre-resolved CIK (skips a company-tickers lookup).
            submissions: Optional pre-fetched submissions payload (skips a fetch).

        Returns:
            Newest-first list of :class:`InsiderTransaction`. Empty when the
            issuer has no recent Form 4 filings.

        Raises:
            DataUnavailable: when the ticker cannot be resolved to a CIK or the
                submissions index is missing.
            RateLimited: when the SEC throttles the submissions fetch *or* any
                Form 4 fetch (aborts, so insider evidence is "unavailable", never
                falsely "absent").
            ProviderError: when the submissions index cannot be fetched, or when
                every selected Form 4 filing failed for a transient reason (the
                insider picture is unknown, not empty).
        """
        ticker_norm = (ticker or "").strip().upper()
        if not ticker_norm:
            return []

        cache_key = f"form4_{ticker_norm}"
        cached = self._cache.get(cache_key, self._settings.cache_ttl_seconds)
        if isinstance(cached, list):
            restored = self._restore_cached_transactions(cached)
            if restored is not None:
                return restored

        if cik is None:
            cik, _ = self._resolve_cik(ticker_norm)
        if submissions is None:
            submissions = self._fetch_submissions(cik)

        selected = self._select_form4_filings(submissions)
        transactions: List[InsiderTransaction] = []
        transient_failures = 0
        for accession, primary_doc in selected:
            try:
                xml_text = self._fetch_form4_xml(cik, accession, primary_doc)
                transactions.extend(self._parse_form4_xml(xml_text))
            except RateLimited:
                # A throttle (429/403) means every further Archives fetch would be
                # throttled too. Abort and propagate rather than silently returning
                # (and caching) a partial/empty list that a later scan would read
                # as "no insider activity" — insider evidence is *unavailable*
                # right now, not absent.
                raise
            except (DataUnavailable, ValueError) as exc:
                # Deterministic per-filing gap (no ownership XML, malformed XML):
                # skip it — one bad filing never kills the ticker, and the gap is
                # stable so the result stays cacheable.
                _log.info(
                    "sec_edgar: skipping Form 4 %s for %s: %s",
                    accession,
                    ticker_norm,
                    exc,
                )
                continue
            except ProviderError as exc:
                # Transient fetch failure (network error, 5xx): skip the filing
                # but remember the result is incomplete for a *transient* reason
                # so it must not be cached as if it were the whole truth.
                transient_failures += 1
                _log.warning(
                    "sec_edgar: transient failure fetching Form 4 %s for %s: %s",
                    accession,
                    ticker_norm,
                    exc,
                )
                continue

        transactions.sort(key=lambda t: t.date, reverse=True)
        if transient_failures and not transactions:
            # Every selected filing failed transiently: the insider picture is
            # unknown, not empty. Raising keeps the caller from recording a
            # factually false "no Form 4 insider transactions" warning.
            raise ProviderError(
                f"could not fetch any of the {len(selected)} recent Form 4 "
                f"filing(s) for {ticker_norm} ({transient_failures} transient "
                "failure(s)); insider activity is unknown, not absent",
                provider=self.name,
            )
        if transient_failures:
            _log.info(
                "sec_edgar: not caching Form 4 results for %s (%d transient "
                "failure(s) left the list incomplete)",
                ticker_norm,
                transient_failures,
            )
        else:
            self._cache.set(
                cache_key, [t.model_dump(mode="json") for t in transactions]
            )
        return transactions

    @staticmethod
    def _restore_cached_transactions(
        cached: List[Any],
    ) -> Optional[List[InsiderTransaction]]:
        """Rebuild :class:`InsiderTransaction` models from a cached JSON list.

        Returns ``None`` when any cached row fails validation, which forces a
        fresh fetch rather than serving corrupt cache content.
        """
        out: List[InsiderTransaction] = []
        for row in cached:
            if not isinstance(row, dict):
                return None
            try:
                out.append(InsiderTransaction.model_validate(row))
            except Exception:  # pydantic ValidationError, without importing it here
                return None
        return out

    # -- HTTP plumbing --------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep just enough to honour the SEC's polite-use rate guidance."""
        with self._throttle_lock:
            elapsed = time.monotonic() - self._last_request_ts
            wait = _MIN_REQUEST_INTERVAL_S - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_request_ts = time.monotonic()

    def _get_json(self, url: str, *, what: str) -> Any:
        """GET ``url`` and return parsed JSON, mapping failures to typed errors.

        Raises:
            DataUnavailable: on a 404 (an expected gap, e.g. no companyfacts).
            RateLimited: on a 429 (or 403 throttle), with the SEC's retry hint.
            ProviderError: on any other HTTP, network or JSON-decode failure.
        """
        self._throttle()
        try:
            resp = self._session.get(url, timeout=self._settings.request_timeout)
        except requests.RequestException as exc:
            raise ProviderError(
                f"network error fetching {what}: {exc}", provider=self.name
            ) from exc

        status = resp.status_code
        if status == 404:
            raise DataUnavailable(f"{what} not found (404)", field=what)
        if status == 429:
            retry_after = resp.headers.get("Retry-After")
            raise RateLimited(
                f"SEC rate limit hit fetching {what}",
                provider=self.name,
                retry_after=_coerce_float(retry_after),
            )
        if status == 403:
            # The SEC returns 403 when the User-Agent is missing/blocked or when a
            # client is throttled. Surface it clearly so the operator fixes UA/rate.
            raise RateLimited(
                f"SEC returned 403 fetching {what} (check SEC_USER_AGENT / rate limit)",
                provider=self.name,
            )
        if status >= 400:
            raise ProviderError(
                f"HTTP {status} fetching {what}", provider=self.name, status_code=status
            )

        try:
            return resp.json()
        except ValueError as exc:
            raise ProviderError(
                f"invalid JSON for {what}: {exc}", provider=self.name, status_code=status
            ) from exc

    def _get_text(self, url: str, *, what: str) -> str:
        """GET ``url`` and return the response body as text (for Archives XML).

        Shares the throttle and error mapping of :meth:`_get_json` but does not
        attempt JSON decoding; used for the raw Form 4 ``ownershipDocument`` XML.

        Raises:
            DataUnavailable: on a 404.
            RateLimited: on a 429/403 throttle.
            ProviderError: on any other HTTP or network failure.
        """
        self._throttle()
        try:
            resp = self._session.get(
                url,
                timeout=self._settings.request_timeout,
                headers={"Accept": "application/xml, text/xml, */*"},
            )
        except requests.RequestException as exc:
            raise ProviderError(
                f"network error fetching {what}: {exc}", provider=self.name
            ) from exc

        status = resp.status_code
        if status == 404:
            raise DataUnavailable(f"{what} not found (404)", field=what)
        if status == 429:
            raise RateLimited(
                f"SEC rate limit hit fetching {what}",
                provider=self.name,
                retry_after=_coerce_float(resp.headers.get("Retry-After")),
            )
        if status == 403:
            raise RateLimited(
                f"SEC returned 403 fetching {what} (check SEC_USER_AGENT / rate limit)",
                provider=self.name,
            )
        if status >= 400:
            raise ProviderError(
                f"HTTP {status} fetching {what}", provider=self.name, status_code=status
            )
        return resp.text

    # -- CIK resolution -------------------------------------------------------

    def _load_cik_map(self) -> Dict[str, int]:
        """Return the ``{TICKER: cik_int}`` map, using the disk cache when fresh."""
        if self._cik_map is not None:
            return self._cik_map

        cached = self._cache.get("company_tickers", _CIK_MAP_TTL_S)
        raw: Optional[Any] = cached
        if raw is None:
            raw = self._get_json(_COMPANY_TICKERS_URL, what="company_tickers map")
            self._cache.set("company_tickers", raw)

        mapping: Dict[str, int] = {}
        # The file is a JSON object keyed by row index: {"0": {"cik_str":..,
        # "ticker":"AAPL","title":"Apple Inc."}, ...}. Be tolerant of list form too.
        rows: List[Any]
        if isinstance(raw, dict):
            rows = list(raw.values())
        elif isinstance(raw, list):
            rows = raw
        else:
            rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            tkr = row.get("ticker")
            cik = row.get("cik_str", row.get("cik"))
            if not tkr or cik is None:
                continue
            try:
                mapping[str(tkr).strip().upper()] = int(cik)
            except (TypeError, ValueError):
                continue

        if not mapping:
            raise ProviderError(
                "SEC company_tickers map was empty or unparseable", provider=self.name
            )
        self._cik_map = mapping
        _log.debug("sec_edgar: loaded %d ticker->CIK entries", len(mapping))
        return mapping

    def _resolve_cik(self, ticker: str) -> Tuple[int, Optional[str]]:
        """Resolve ``ticker`` to ``(cik_int, company_title_or_None)``.

        Raises:
            DataUnavailable: when the ticker is not present in the official map.
        """
        mapping = self._load_cik_map()
        cik = mapping.get(ticker)
        if cik is None:
            # EDGAR uses '-' where some vendors use '.' for share classes
            # (e.g. BRK.B -> BRK-B). Try the common normalisation transparently.
            for alt in (ticker.replace(".", "-"), ticker.replace("-", ".")):
                if alt in mapping:
                    cik = mapping[alt]
                    break
        if cik is None:
            raise DataUnavailable(
                f"ticker {ticker!r} not found in SEC company_tickers map", ticker=ticker
            )
        return cik, None

    # -- Submissions / filings ------------------------------------------------

    def _fetch_submissions(self, cik: int) -> Dict[str, Any]:
        """Fetch (cache-first) the submissions index JSON for ``cik``."""
        key = f"submissions_{cik:010d}"
        cached = self._cache.get(key, self._settings.cache_ttl_seconds)
        if cached is not None and isinstance(cached, dict):
            return cached
        payload = self._get_json(_SUBMISSIONS_URL.format(cik=cik), what=f"submissions CIK{cik}")
        if not isinstance(payload, dict):
            raise ProviderError("submissions payload was not an object", provider=self.name)
        self._cache.set(key, payload)
        return payload

    def _populate_company_meta(self, data: SecurityData, submissions: Dict[str, Any]) -> None:
        """Fill exchange / sic-description fields from submissions, when present."""
        sic_desc = submissions.get("sicDescription")
        if sic_desc and not data.industry:
            data.industry = str(sic_desc)
        exchanges = submissions.get("exchanges")
        if isinstance(exchanges, list) and exchanges and not data.exchange:
            data.exchange = str(exchanges[0])

    def _extract_filings(
        self, cik: int, submissions: Dict[str, Any], limit: int
    ) -> List[Filing]:
        """Build a newest-first :class:`Filing` list from the submissions index.

        The submissions index stores recent filings as parallel arrays under
        ``filings.recent`` (``form``, ``filingDate``, ``accessionNumber``,
        ``primaryDocument``, ``primaryDocDescription``). We zip them into
        :class:`Filing` objects, skipping any row missing a form or filing date.
        """
        recent = ((submissions.get("filings") or {}).get("recent")) or {}
        forms = recent.get("form") or []
        dates = recent.get("filingDate") or []
        accessions = recent.get("accessionNumber") or []
        primary_docs = recent.get("primaryDocument") or []
        descriptions = recent.get("primaryDocDescription") or []

        n = min(len(forms), len(dates))
        out: List[Filing] = []
        for i in range(n):
            filed = _parse_date(dates[i])
            form = forms[i]
            if filed is None or not form:
                continue
            accession = accessions[i] if i < len(accessions) else ""
            primary = primary_docs[i] if i < len(primary_docs) else ""
            desc = descriptions[i] if i < len(descriptions) else None
            url = self._build_filing_url(cik, accession, primary)
            out.append(
                Filing(
                    filed=filed,
                    form_type=str(form),
                    title=str(desc) if desc else None,
                    url=url,
                    summary=None,
                )
            )

        # Submissions arrays are already newest-first, but sort defensively so the
        # contract ("newest first") holds even if EDGAR changes ordering.
        out.sort(key=lambda f: f.filed, reverse=True)
        return out[: max(0, limit)]

    @staticmethod
    def _build_filing_url(cik: int, accession: str, primary_doc: str) -> Optional[str]:
        """Construct the public Archives URL for a filing's primary document.

        Falls back to the filing's folder URL when the primary document name is
        absent, so the link always points at real EDGAR content (never fabricated).
        """
        if not accession:
            return None
        nodash = accession.replace("-", "")
        if primary_doc:
            return _FILING_INDEX_URL.format(
                cik=cik, accession_nodash=nodash, primary_doc=primary_doc
            )
        return _FILING_FOLDER_URL.format(cik=cik, accession_nodash=nodash)

    # -- Form 4 insider transactions -------------------------------------------

    @staticmethod
    def _select_form4_filings(
        submissions: Dict[str, Any],
        *,
        today: Optional[_dt.date] = None,
    ) -> List[Tuple[str, str]]:
        """Select recent Form 4 filings from the submissions index.

        Filters the parallel ``filings.recent`` arrays down to form ``"4"`` rows
        filed within the last :data:`_FORM4_LOOKBACK_DAYS` days, sorts them most
        recent first, and caps the result at :data:`_FORM4_MAX_FILINGS`.

        Args:
            submissions: The submissions index JSON payload.
            today: Injectable "now" for deterministic tests; defaults to the
                real current date.

        Returns:
            List of ``(accession_number, primary_document)`` tuples, newest first.
        """
        recent = ((submissions.get("filings") or {}).get("recent")) or {}
        forms = recent.get("form") or []
        dates = recent.get("filingDate") or []
        accessions = recent.get("accessionNumber") or []
        primary_docs = recent.get("primaryDocument") or []

        cutoff = (today or _dt.date.today()) - _dt.timedelta(days=_FORM4_LOOKBACK_DAYS)
        rows: List[Tuple[_dt.date, str, str]] = []
        n = min(len(forms), len(dates), len(accessions))
        for i in range(n):
            if str(forms[i]).strip() != "4":
                continue
            filed = _parse_date(dates[i])
            if filed is None or filed < cutoff:
                continue
            accession = str(accessions[i] or "").strip()
            if not accession:
                continue
            primary = str(primary_docs[i]).strip() if i < len(primary_docs) else ""
            rows.append((filed, accession, primary))

        rows.sort(key=lambda r: r[0], reverse=True)
        return [(accession, primary) for _, accession, primary in rows[:_FORM4_MAX_FILINGS]]

    def _fetch_form4_xml(self, cik: int, accession: str, primary_doc: str) -> str:
        """Fetch the ``ownershipDocument`` XML text for one Form 4 filing.

        Tries the filing's ``primaryDocument`` first when it names an ``.xml``
        file (stripping any ``xsl.../`` rendering prefix, which points at an HTML
        rendition rather than the raw XML). When the primary document is not the
        XML — or its fetch 404s — falls back to scanning the filing's Archives
        ``index.json`` for the first ownership ``.xml`` member.

        Raises:
            DataUnavailable / ProviderError: when no XML document can be located
                or fetched for this filing.
        """
        nodash = accession.replace("-", "")
        candidates: List[str] = []

        primary = (primary_doc or "").strip()
        if primary.lower().endswith(".xml"):
            # e.g. "xslF345X05/wk-form4_1.xml" renders HTML; the raw XML lives at
            # the bare document name in the same folder.
            candidates.append(primary.rsplit("/", 1)[-1])

        for doc in candidates:
            url = _FILING_INDEX_URL.format(cik=cik, accession_nodash=nodash, primary_doc=doc)
            try:
                return self._get_text(url, what=f"Form 4 XML {accession}")
            except DataUnavailable:
                continue  # fall through to the index scan

        # Fallback: scan the filing folder index for an .xml member.
        index_url = _FILING_FOLDER_URL.format(cik=cik, accession_nodash=nodash) + "index.json"
        index = self._get_json(index_url, what=f"filing index {accession}")
        xml_name = self._find_xml_in_index(index)
        if not xml_name:
            raise DataUnavailable(
                f"no ownership XML found in filing index for {accession}",
                field="form4_xml",
            )
        url = _FILING_INDEX_URL.format(cik=cik, accession_nodash=nodash, primary_doc=xml_name)
        return self._get_text(url, what=f"Form 4 XML {accession}")

    @staticmethod
    def _find_xml_in_index(index: Any) -> Optional[str]:
        """Pick the ownership ``.xml`` document name from an Archives ``index.json``.

        Prefers a name containing ``form4``, otherwise takes the first ``.xml``
        entry that is not an ``xsl`` rendering artefact. Returns ``None`` when the
        index carries no XML member.
        """
        if not isinstance(index, dict):
            return None
        items = ((index.get("directory") or {}).get("item")) or []
        xml_names = [
            str(item.get("name"))
            for item in items
            if isinstance(item, dict)
            and str(item.get("name") or "").lower().endswith(".xml")
            and not str(item.get("name") or "").lower().startswith("xsl")
        ]
        for name in xml_names:
            if "form4" in name.lower():
                return name
        return xml_names[0] if xml_names else None

    def _parse_form4_xml(self, xml_text: str) -> List[InsiderTransaction]:
        """Parse a Form 4 ``ownershipDocument`` XML into insider transactions.

        Reads the reporting owner's name and role (officer title, director or
        10%-owner flag) and every ``nonDerivativeTransaction``:

        * ``transactionDate/value`` → :attr:`InsiderTransaction.date`
        * ``transactionCoding/transactionCode`` → ``transaction_type`` via
          :data:`_FORM4_TRANSACTION_CODE_MAP` (unknown codes become
          ``"other:<code>"`` — honestly labelled, never guessed).
        * ``transactionShares/value`` → ``shares``
        * ``shares * transactionPricePerShare/value`` → ``value`` (only when both
          are present; otherwise ``None``, never fabricated).

        Transactions missing a parseable date are skipped (a dateless transaction
        cannot be placed on the insider timeline).

        Raises:
            ValueError: when ``xml_text`` is not well-formed XML.
        """
        try:
            root = _ET.fromstring(xml_text)
        except _ET.ParseError as exc:
            raise ValueError(f"malformed Form 4 XML: {exc}") from exc

        insider_name = self._xml_text(root, "reportingOwner/reportingOwnerId/rptOwnerName")
        role = self._extract_owner_role(root)
        if not insider_name:
            insider_name = "Unknown insider"

        out: List[InsiderTransaction] = []
        for txn in root.iter("nonDerivativeTransaction"):
            date = _parse_date(self._xml_text(txn, "transactionDate/value"))
            if date is None:
                continue
            code = (self._xml_text(txn, "transactionCoding/transactionCode") or "").strip().upper()
            transaction_type = _FORM4_TRANSACTION_CODE_MAP.get(
                code, f"other:{code}" if code else "other:?"
            )
            shares = _coerce_float(
                self._xml_text(txn, "transactionAmounts/transactionShares/value")
            )
            price = _coerce_float(
                self._xml_text(txn, "transactionAmounts/transactionPricePerShare/value")
            )
            value = shares * price if (shares is not None and price is not None) else None
            out.append(
                InsiderTransaction(
                    date=date,
                    insider_name=insider_name,
                    role=role,
                    transaction_type=transaction_type,
                    shares=shares,
                    value=value,
                )
            )
        return out

    def _extract_owner_role(self, root: _ET.Element) -> Optional[str]:
        """Derive the reporting owner's role from ``reportingOwnerRelationship``.

        Preference order: an explicit ``officerTitle``, then ``Director``, then
        ``10% owner``. Returns ``None`` when the filing declares none of these —
        the role is left missing rather than invented.
        """
        rel_path = "reportingOwner/reportingOwnerRelationship"
        title = self._xml_text(root, f"{rel_path}/officerTitle")
        if title:
            return title
        if self._xml_flag(self._xml_text(root, f"{rel_path}/isDirector")):
            return "Director"
        if self._xml_flag(self._xml_text(root, f"{rel_path}/isTenPercentOwner")):
            return "10% owner"
        return None

    @staticmethod
    def _xml_text(elem: _ET.Element, path: str) -> Optional[str]:
        """Return the stripped text at ``path`` under ``elem``, or ``None``."""
        node = elem.find(path)
        if node is None or node.text is None:
            return None
        text = node.text.strip()
        return text or None

    @staticmethod
    def _xml_flag(value: Optional[str]) -> bool:
        """Interpret a Form 4 boolean field, which may be ``"1"`` or ``"true"``."""
        return (value or "").strip().lower() in {"1", "true"}

    # -- Company facts / fundamentals -----------------------------------------

    def _fetch_company_facts(self, cik: int) -> Dict[str, Any]:
        """Fetch (cache-first) the XBRL companyfacts JSON for ``cik``."""
        key = f"companyfacts_{cik:010d}"
        cached = self._cache.get(key, self._settings.cache_ttl_seconds)
        if cached is not None and isinstance(cached, dict):
            return cached
        payload = self._get_json(
            _COMPANY_FACTS_URL.format(cik=cik), what=f"companyfacts CIK{cik}"
        )
        if not isinstance(payload, dict):
            raise ProviderError("companyfacts payload was not an object", provider=self.name)
        self._cache.set(key, payload)
        return payload

    def _extract_fundamentals(self, facts: Dict[str, Any]) -> List[FundamentalsPeriod]:
        """Fold US-GAAP companyfacts into a newest-first FundamentalsPeriod list.

        Strategy:

        1. Read the ``us-gaap`` namespace from ``facts.facts``.
        2. For each tracked concept, pick its USD (or shares / per-share) unit
           series and index facts by ``(period_key)`` where the key is the
           ``end`` date for instant (balance-sheet) concepts and the
           ``(start, end)`` window — collapsed to its ``end`` — for flow
           (income/cash-flow) concepts that cover a full annual or quarterly span.
        3. Group everything by period-end date, prefer annual ("FY") frames, and
           emit one :class:`FundamentalsPeriod` per period, leaving any concept the
           issuer did not tag as ``None``.

        Derived margins (gross/operating/FCF) and FCF itself are computed only when
        their inputs are present; otherwise they remain ``None``.
        """
        gaap = ((facts.get("facts") or {}).get("us-gaap")) or {}
        if not isinstance(gaap, dict) or not gaap:
            return []

        # period_end (date) -> {field_name: value, plus meta form/fp/fy}
        periods: Dict[_dt.date, Dict[str, Any]] = {}

        def ingest(concepts: Tuple[str, ...], field: str, *, instant: bool) -> None:
            concept = _first_concept(gaap, concepts)
            if concept is None:
                return
            facts_by_end = self._select_facts(gaap[concept], instant=instant)
            for end_date, fact in facts_by_end.items():
                bucket = periods.setdefault(end_date, {})
                # First writer wins per (period, field): concepts are pre-ordered by
                # preference, and we iterate concept-by-concept, so this preserves
                # the most-direct mapping without overwriting it later.
                if field not in bucket or bucket[field] is None:
                    bucket[field] = _coerce_float(fact.get("val"))
                # Capture period metadata from a flow fact (it carries form/fp/fy).
                if not instant:
                    bucket.setdefault("_form", fact.get("form"))
                    bucket.setdefault("_fp", fact.get("fp"))
                    bucket.setdefault("_fy", fact.get("fy"))
                else:
                    bucket.setdefault("_form", fact.get("form"))
                    bucket.setdefault("_fp", fact.get("fp"))
                    bucket.setdefault("_fy", fact.get("fy"))

        ingest(_REVENUE_CONCEPTS, "revenue", instant=False)
        ingest(_GROSS_PROFIT_CONCEPTS, "gross_profit", instant=False)
        ingest(_OPERATING_INCOME_CONCEPTS, "operating_income", instant=False)
        ingest(_NET_INCOME_CONCEPTS, "net_income", instant=False)
        ingest(_OCF_CONCEPTS, "operating_cash_flow", instant=False)
        ingest(_CAPEX_CONCEPTS, "capex", instant=False)
        ingest(_EPS_DILUTED_CONCEPTS, "eps_diluted", instant=False)
        ingest(_SHARES_DILUTED_CONCEPTS, "shares_diluted", instant=False)
        # Fallback share count for issuers with no weighted-average series (first
        # writer wins per period, so this never overrides the flow figure above).
        ingest(_SHARES_OUTSTANDING_INSTANT_CONCEPTS, "shares_diluted", instant=True)
        ingest(_TOTAL_ASSETS_CONCEPTS, "total_assets", instant=True)
        ingest(_TOTAL_EQUITY_CONCEPTS, "total_equity", instant=True)
        ingest(_CASH_CONCEPTS, "cash_and_equivalents", instant=True)
        ingest(_LONG_TERM_DEBT_CONCEPTS, "_long_term_debt", instant=True)
        ingest(_SHORT_TERM_DEBT_CONCEPTS, "_short_term_debt", instant=True)

        out: List[FundamentalsPeriod] = []
        for end_date in sorted(periods.keys(), reverse=True):
            bucket = periods[end_date]
            fp = self._build_period(end_date, bucket)
            if fp is not None:
                out.append(fp)

        return out[: self.MAX_PERIODS]

    def _select_facts(self, concept_block: Dict[str, Any], *, instant: bool) -> Dict[_dt.date, Dict[str, Any]]:
        """Pick one fact per period-end from a concept's unit series.

        For *flow* concepts we keep only facts whose ``start``/``end`` window spans
        a full fiscal period (≈ a quarter or a year), preferring the longest span
        ending on a given date (i.e. the annual figure over a partial-year YTD).
        For *instant* concepts we simply take the fact at each ``end`` date.

        We prefer 10-K/10-Q form facts and the latest ``filed`` value when EDGAR
        carries multiple (e.g. an original and a restated figure share an end date).
        """
        units = concept_block.get("units")
        if not isinstance(units, dict) or not units:
            return {}

        # Choose the most appropriate unit: monetary (USD), per-share (USD/shares),
        # or share counts. company facts key units like "USD", "USD/shares", "shares".
        unit_key = self._pick_unit(units)
        series = units.get(unit_key) or []
        if not isinstance(series, list):
            return {}

        chosen: Dict[_dt.date, Dict[str, Any]] = {}
        chosen_span: Dict[_dt.date, int] = {}
        for fact in series:
            if not isinstance(fact, dict):
                continue
            end_date = _parse_date(fact.get("end"))
            if end_date is None:
                continue

            if instant:
                self._consider(chosen, end_date, fact)
                continue

            start_date = _parse_date(fact.get("start"))
            if start_date is None:
                continue
            span_days = (end_date - start_date).days
            # Keep only quarterly (~80-100d) or annual (~330-380d) spans; drop
            # 6- and 9-month year-to-date cumulatives that would double-count.
            is_quarter = 80 <= span_days <= 100
            is_annual = 330 <= span_days <= 380
            if not (is_quarter or is_annual):
                continue
            prev_span = chosen_span.get(end_date, -1)
            # Prefer the annual span on a shared end date; otherwise prefer the
            # latest-filed fact for the same span.
            if span_days > prev_span:
                if self._consider(chosen, end_date, fact, force=True):
                    chosen_span[end_date] = span_days
            elif span_days == prev_span:
                self._consider(chosen, end_date, fact)

        return chosen

    @staticmethod
    def _consider(
        chosen: Dict[_dt.date, Dict[str, Any]],
        end_date: _dt.date,
        fact: Dict[str, Any],
        *,
        force: bool = False,
    ) -> bool:
        """Insert ``fact`` for ``end_date`` if it is newer-filed (or ``force``).

        Returns ``True`` if the fact was stored. Among facts sharing an end date we
        keep the one with the latest ``filed`` date so the most recently reported
        (and typically corrected) value wins, which is the deterministic choice.
        """
        existing = chosen.get(end_date)
        if existing is None or force:
            chosen[end_date] = fact
            return True
        new_filed = _parse_date(fact.get("filed"))
        old_filed = _parse_date(existing.get("filed"))
        if new_filed and (old_filed is None or new_filed > old_filed):
            chosen[end_date] = fact
            return True
        return False

    @staticmethod
    def _pick_unit(units: Dict[str, Any]) -> str:
        """Choose the most relevant unit key from a companyfacts ``units`` block.

        Prefers plain ``USD`` (monetary), then a per-share unit, then ``shares``,
        otherwise the first key — so monetary line items resolve to dollars while
        EPS and share-count concepts still find their natural unit.
        """
        if "USD" in units:
            return "USD"
        for key in units:
            if key.startswith("USD/"):
                return key
        if "shares" in units:
            return "shares"
        return next(iter(units))

    def _build_period(
        self, end_date: _dt.date, bucket: Dict[str, Any]
    ) -> Optional[FundamentalsPeriod]:
        """Assemble one :class:`FundamentalsPeriod` from a per-period value bucket.

        Computes derived margins / FCF / total debt only when the necessary inputs
        are present; everything else is left ``None`` (the honesty contract).
        Returns ``None`` if the bucket carries no usable financial line item at all.
        """
        revenue = bucket.get("revenue")
        gross_profit = bucket.get("gross_profit")
        operating_income = bucket.get("operating_income")
        net_income = bucket.get("net_income")
        ocf = bucket.get("operating_cash_flow")
        capex = bucket.get("capex")
        total_assets = bucket.get("total_assets")
        total_equity = bucket.get("total_equity")
        cash = bucket.get("cash_and_equivalents")
        ltd = bucket.get("_long_term_debt")
        std = bucket.get("_short_term_debt")
        eps_diluted = bucket.get("eps_diluted")
        shares_diluted = bucket.get("shares_diluted")

        # Total debt = long-term + current portion, when at least one is present.
        total_debt: Optional[float] = None
        if ltd is not None or std is not None:
            total_debt = (ltd or 0.0) + (std or 0.0)

        # Free cash flow = operating cash flow - capex (capex is reported positive).
        free_cash_flow: Optional[float] = None
        if ocf is not None and capex is not None:
            free_cash_flow = ocf - abs(capex)

        # Derived margins, only when revenue is a meaningful positive denominator.
        gross_margin = self._safe_ratio(gross_profit, revenue)
        operating_margin = self._safe_ratio(operating_income, revenue)
        fcf_margin = self._safe_ratio(free_cash_flow, revenue)

        # Returns, only when the denominator is meaningful.
        roe = self._safe_ratio(net_income, total_equity)
        roa = self._safe_ratio(net_income, total_assets)

        debt_to_equity = self._safe_ratio(total_debt, total_equity)

        # Drop a period that carries nothing useful at all.
        line_items = [
            revenue,
            gross_profit,
            operating_income,
            net_income,
            ocf,
            total_assets,
            total_equity,
            cash,
            total_debt,
            eps_diluted,
        ]
        if all(v is None for v in line_items):
            return None

        form = bucket.get("_form") or ""
        fp = bucket.get("_fp")
        fy = bucket.get("_fy")
        label = _period_label(end_date, str(form), fp if isinstance(fp, str) else None,
                              fy if isinstance(fy, int) else None)

        return FundamentalsPeriod(
            period_end=end_date,
            period_label=label,
            revenue=revenue,
            gross_profit=gross_profit,
            operating_income=operating_income,
            net_income=net_income,
            ebitda=None,  # not tagged uniformly in us-gaap; left missing, not faked
            eps_diluted=eps_diluted,
            free_cash_flow=free_cash_flow,
            operating_cash_flow=ocf,
            capex=(-abs(capex) if capex is not None else None),
            total_assets=total_assets,
            total_debt=total_debt,
            cash_and_equivalents=cash,
            total_equity=total_equity,
            shares_diluted=shares_diluted,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            fcf_margin=fcf_margin,
            roic=None,  # requires NOPAT + invested capital not directly tagged
            roe=roe,
            roa=roa,
            current_ratio=None,  # current assets/liabilities not in tracked set
            quick_ratio=None,
            debt_to_equity=debt_to_equity,
            interest_coverage=None,
        )

    @staticmethod
    def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        """Return ``numerator / denominator`` or ``None`` if it cannot be computed.

        Guards against a missing operand and a zero / non-positive denominator so a
        ratio is only produced when it is genuinely meaningful.
        """
        if numerator is None or denominator is None:
            return None
        if denominator == 0:
            return None
        return numerator / denominator


__all__ = ["SecEdgarProvider"]
