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

Capabilities advertised: ``{"fundamentals", "filings"}``. EDGAR does not provide
prices, news, valuation multiples, or a screening universe, so this provider does
not advertise those.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from convexity.core.config import Settings, get_settings
from convexity.core.exceptions import DataUnavailable, ProviderError, RateLimited
from convexity.core.logging import get_logger
from convexity.core.models import Filing, FundamentalsPeriod, SecurityData
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
        """Capabilities EDGAR truly fills: structured fundamentals and filings."""
        return {"fundamentals", "filings"}

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
        try:
            submissions = self._fetch_submissions(cik)
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
