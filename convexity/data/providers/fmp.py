"""Financial Modeling Prep (FMP) data provider.

This module maps the Financial Modeling Prep REST API onto Convexity's
:class:`~convexity.core.models.SecurityData` contract. It is an *optional*
provider: it only does anything when ``settings.fmp_api_key`` is set. When the key
is absent the provider marks itself unavailable via :meth:`FMPProvider.is_available`
so the aggregator skips it cleanly instead of issuing keyless requests.

Honesty rules honoured here
---------------------------
* Nothing is fabricated. Every field on :class:`SecurityData` is populated only
  from a real FMP response; any datum FMP does not return stays ``None`` and an
  explanatory note is appended to ``SecurityData.data_warnings``.
* The provider is *not* a predictor. It is a transparent adapter that reshapes a
  vendor's disclosures (profile, financial statements, key metrics, ratios,
  insider trades, institutional ownership) into auditable, typed records that a
  human reviewer can trace back to the source endpoint.
* Network and parsing problems never crash a scan: per-endpoint failures are
  logged and recorded as warnings, and only a total inability to identify the
  security raises :class:`~convexity.core.exceptions.DataUnavailable`.

The provider advertises the capabilities it genuinely fills so the aggregator can
route requests to it: ``fundamentals``, ``valuation``, ``profile``, ``insider``
and ``institutional``.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Set

from convexity.core.config import Settings, get_settings
from convexity.core.contracts import DataProvider
from convexity.core.exceptions import (
    DataUnavailable,
    ProviderError,
    RateLimited,
)
from convexity.core.logging import get_logger
from convexity.core.models import (
    CapTier,
    FundamentalsPeriod,
    InsiderTransaction,
    InstitutionalHolding,
    SecurityData,
    ValuationSnapshot,
)
from convexity.core.registry import register_provider

_log = get_logger(__name__)

# Base URL for the FMP "stable" / v3 REST API. All endpoints are GET requests that
# return JSON and accept the API key as the ``apikey`` query parameter.
_FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# How many historical statement periods to request per statement type. Enough to
# let growth/quality analyzers compute multi-year trends without over-fetching.
_STATEMENT_LIMIT = 8

# How many insider transactions / institutional holders to pull. Micro-caps have
# sparse disclosure, so a modest cap keeps payloads small while remaining useful.
_INSIDER_LIMIT = 50
_INSTITUTIONAL_LIMIT = 50


def _to_float(value: Any) -> Optional[float]:
    """Coerce a JSON value to ``float``; return ``None`` for missing/garbage.

    FMP frequently returns ``None``, empty strings, or the string ``"None"`` for a
    line item a company did not disclose. Each of those must become a genuine
    ``None`` so it is treated as missing rather than fabricated.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # guard: avoid True -> 1.0 surprises
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        # NaN/inf are not real data.
        if result != result or result in (float("inf"), float("-inf")):
            return None
        return result
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text == "" or text.lower() in ("none", "null", "nan", "-"):
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _to_date(value: Any) -> Optional[_dt.date]:
    """Parse an FMP date string (``YYYY-MM-DD`` or ISO datetime) into a ``date``."""
    if value is None:
        return None
    if isinstance(value, _dt.date) and not isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.datetime):
        return value.date()
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    # Common shapes: "2025-12-31", "2025-12-31 00:00:00", "2025-12-31T00:00:00".
    text = text.replace("T", " ")
    head = text.split(" ", 1)[0]
    try:
        return _dt.date.fromisoformat(head)
    except ValueError:
        return None


def _nonempty_str(value: Any) -> Optional[str]:
    """Return a stripped non-empty string, else ``None``."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _classify_cap_tier(market_cap: Optional[float]) -> Optional[CapTier]:
    """Bucket a market cap into Convexity's :class:`CapTier` (nano/micro/small).

    Returns ``None`` for caps above the small-cap ceiling (~$2B) or when unknown,
    so the aggregator/screen can decide how to treat out-of-universe names rather
    than this provider silently mislabelling them.
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


@register_provider
class FMPProvider(DataProvider):
    """Provider backed by the Financial Modeling Prep API (key-gated).

    Construct with no arguments to use the process-wide cached
    :class:`~convexity.core.config.Settings`, or pass an explicit ``settings`` (and
    optionally a pre-built ``requests``-like session) for testing. When no API key
    is configured the provider is *unavailable*: :meth:`is_available` returns
    ``False`` and the aggregator must skip it.
    """

    #: Stable identifier recorded in ``SecurityData.data_sources``.
    _NAME = "fmp"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        session: Optional[Any] = None,
    ) -> None:
        # ``register_provider`` instantiates each provider once with no args purely
        # to read ``.name``; that must never raise, so resolving settings is wrapped
        # defensively and the provider simply ends up unavailable on any failure.
        try:
            self._settings = settings if settings is not None else get_settings()
        except Exception:  # pragma: no cover - defensive; config should not raise
            self._settings = None  # type: ignore[assignment]
        self._api_key = getattr(self._settings, "fmp_api_key", None) if self._settings else None
        self._timeout = float(getattr(self._settings, "request_timeout", 20.0)) if self._settings else 20.0
        self._session = session

    # ------------------------------------------------------------------
    # Identity / capabilities
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short, stable identifier for this provider."""
        return self._NAME

    @property
    def capabilities(self) -> Set[str]:
        """Capabilities this provider fills (empty when no API key is configured).

        Advertising nothing when unavailable means an aggregator that routes purely
        by capability will also naturally skip a keyless instance.
        """
        if not self.is_available():
            return set()
        return {"profile", "valuation", "fundamentals", "insider", "institutional"}

    def is_available(self) -> bool:
        """Whether this provider can actually be used (i.e. an API key is set).

        The aggregator should call this and skip the provider when it returns
        ``False`` so that no keyless requests are ever issued.
        """
        return bool(self._api_key)

    @classmethod
    def available_for(cls, settings: Optional[Settings] = None) -> bool:
        """Class-level availability check used before instantiation.

        Returns ``True`` only when an FMP API key is present in ``settings`` (or the
        process settings when ``settings`` is ``None``).
        """
        try:
            cfg = settings if settings is not None else get_settings()
        except Exception:  # pragma: no cover - defensive
            return False
        return bool(getattr(cfg, "fmp_api_key", None))

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _get_session(self) -> Any:
        """Lazily build (and cache) a ``requests.Session``.

        ``requests`` is imported lazily so that merely importing this module never
        requires the dependency to be installed (e.g. on a machine that only has
        pydantic). The session is reused across endpoints for connection pooling.
        """
        if self._session is not None:
            return self._session
        try:
            import requests  # local import keeps module import-time dependency-free
        except Exception as exc:  # pragma: no cover - environment-specific
            raise ProviderError(
                "the 'requests' package is required for the FMP provider",
                provider=self._NAME,
            ) from exc
        self._session = requests.Session()
        return self._session

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET ``path`` under the FMP base URL and return parsed JSON.

        On any HTTP/transport/parse problem this raises a Convexity exception
        (:class:`RateLimited`, :class:`DataUnavailable` or :class:`ProviderError`)
        — it never raises a bare exception that could crash a scan. Callers wrap
        these per-endpoint so one failed endpoint degrades gracefully.
        """
        if not self.is_available():
            raise DataUnavailable(
                "FMP provider has no API key configured", field=path
            )

        session = self._get_session()
        query: Dict[str, Any] = {"apikey": self._api_key}
        if params:
            query.update(params)
        url = f"{_FMP_BASE_URL}/{path.lstrip('/')}"

        try:
            resp = session.get(url, params=query, timeout=self._timeout)
        except Exception as exc:  # network/transport error
            raise ProviderError(
                f"FMP request to {path} failed: {exc}", provider=self._NAME
            ) from exc

        status = getattr(resp, "status_code", None)
        if status == 429:
            retry_after = self._parse_retry_after(resp)
            raise RateLimited(
                "FMP rate limit hit", provider=self._NAME, retry_after=retry_after
            )
        if status == 401 or status == 403:
            raise ProviderError(
                f"FMP rejected the request to {path} (status {status}); check the API key",
                provider=self._NAME,
                status_code=status,
            )
        if status == 404:
            # Treat an explicit not-found as an expected gap for this ticker.
            raise DataUnavailable(
                f"FMP has no data at {path}", field=path
            )
        if status is None or status >= 400:
            raise ProviderError(
                f"FMP returned status {status} for {path}",
                provider=self._NAME,
                status_code=status,
            )

        try:
            payload = resp.json()
        except Exception as exc:  # malformed JSON
            raise ProviderError(
                f"FMP returned non-JSON for {path}: {exc}", provider=self._NAME
            ) from exc

        # FMP signals quota/auth problems with a JSON object carrying an
        # "Error Message" key (HTTP 200). Surface those explicitly.
        if isinstance(payload, dict):
            message = payload.get("Error Message") or payload.get("error")
            if message:
                lowered = str(message).lower()
                if "limit" in lowered or "quota" in lowered:
                    raise RateLimited(
                        f"FMP: {message}", provider=self._NAME
                    )
                raise ProviderError(
                    f"FMP error for {path}: {message}", provider=self._NAME
                )
        return payload

    @staticmethod
    def _parse_retry_after(resp: Any) -> Optional[float]:
        """Extract a ``Retry-After`` header (seconds) from a response, if present."""
        headers = getattr(resp, "headers", None)
        if not headers:
            return None
        raw = None
        try:
            raw = headers.get("Retry-After")
        except Exception:  # pragma: no cover - defensive
            return None
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _safe_list(
        self,
        path: str,
        warnings: List[str],
        *,
        params: Optional[Dict[str, Any]] = None,
        what: str,
    ) -> List[Dict[str, Any]]:
        """Fetch an endpoint expected to return a JSON list; degrade on failure.

        Returns an empty list (and appends a human-readable note to ``warnings``)
        for any expected gap or recoverable error so a single missing endpoint never
        aborts assembly of the rest of ``SecurityData``.
        """
        try:
            payload = self._request(path, params=params)
        except DataUnavailable:
            warnings.append(f"FMP: no {what} available")
            return []
        except RateLimited as exc:
            _log.warning("FMP rate limited fetching %s: %s", what, exc)
            warnings.append(f"FMP: {what} skipped (rate limited)")
            return []
        except ProviderError as exc:
            _log.warning("FMP error fetching %s: %s", what, exc)
            warnings.append(f"FMP: {what} unavailable ({exc})")
            return []
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            # Some endpoints wrap the array; otherwise treat a single object as one row.
            return [payload]
        warnings.append(f"FMP: unexpected {what} payload shape")
        return []

    # ------------------------------------------------------------------
    # DataProvider interface
    # ------------------------------------------------------------------

    def get_security_data(self, ticker: str) -> SecurityData:
        """Assemble all FMP-sourced data for ``ticker`` into a :class:`SecurityData`.

        Each endpoint is fetched defensively; a failure in one section is recorded
        as a warning and leaves the corresponding fields empty/``None`` rather than
        failing the whole fetch. The method raises
        :class:`~convexity.core.exceptions.DataUnavailable` only when the security
        cannot be identified at all (no profile *and* no statements), since there is
        then nothing trustworthy to return.
        """
        if not self.is_available():
            raise DataUnavailable(
                "FMP provider is unavailable (no API key configured)", ticker=ticker
            )

        symbol = (ticker or "").strip().upper()
        if not symbol:
            raise DataUnavailable("empty ticker symbol", ticker=ticker)

        warnings: List[str] = []

        # --- Profile (identity + a market-cap anchored valuation snapshot) -----
        profile = self._fetch_profile(symbol, warnings)

        # --- Financial statements + per-period ratios/metrics ------------------
        fundamentals = self._fetch_fundamentals(symbol, warnings)

        # --- Standalone valuation multiples (key-metrics + ratios, TTM) --------
        valuation = self._fetch_valuation(symbol, profile, warnings)

        # --- Insider transactions ---------------------------------------------
        insider = self._fetch_insider(symbol, warnings)

        # --- Institutional ownership ------------------------------------------
        institutional = self._fetch_institutional(symbol, warnings)

        if profile is None and not fundamentals:
            # Could not identify the security at all — an expected gap for many
            # micro-caps not covered by FMP. Let the aggregator move on.
            raise DataUnavailable(
                f"FMP returned no profile or financials for {symbol}", ticker=symbol
            )

        name = (profile.get("companyName") if profile else None) or symbol
        sector = _nonempty_str(profile.get("sector")) if profile else None
        industry = _nonempty_str(profile.get("industry")) if profile else None
        exchange = (
            _nonempty_str(profile.get("exchangeShortName"))
            or _nonempty_str(profile.get("exchange"))
            if profile
            else None
        )
        currency = (_nonempty_str(profile.get("currency")) if profile else None) or "USD"
        cap_tier = _classify_cap_tier(valuation.market_cap)

        data = SecurityData(
            ticker=symbol,
            name=name,
            sector=sector,
            industry=industry,
            exchange=exchange,
            cap_tier=cap_tier,
            currency=currency,
            as_of=_dt.datetime.now(_dt.timezone.utc),
            valuation=valuation,
            fundamentals=fundamentals,
            insider_transactions=insider,
            institutional_holdings=institutional,
            data_sources=[self._NAME],
            data_warnings=warnings,
        )
        return data

    # ------------------------------------------------------------------
    # Section fetchers / mappers
    # ------------------------------------------------------------------

    def _fetch_profile(
        self, symbol: str, warnings: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Fetch the company profile, returning the raw dict or ``None``."""
        rows = self._safe_list(f"profile/{symbol}", warnings, what="company profile")
        if not rows:
            warnings.append("FMP: no company profile (identity/sector unknown)")
            return None
        return rows[0]

    def _fetch_fundamentals(
        self, symbol: str, warnings: List[str]
    ) -> List[FundamentalsPeriod]:
        """Build a newest-first list of :class:`FundamentalsPeriod`.

        Merges the income statement, balance sheet, cash-flow statement, per-period
        ``key-metrics`` and ``ratios`` endpoints keyed on the reporting date so each
        :class:`FundamentalsPeriod` carries the richest available picture of a
        single fiscal period. Missing line items remain ``None``.
        """
        params = {"limit": _STATEMENT_LIMIT, "period": "annual"}
        income = self._safe_list(
            f"income-statement/{symbol}", warnings, params=params, what="income statement"
        )
        balance = self._safe_list(
            f"balance-sheet-statement/{symbol}", warnings, params=params, what="balance sheet"
        )
        cashflow = self._safe_list(
            f"cash-flow-statement/{symbol}", warnings, params=params, what="cash-flow statement"
        )
        metrics = self._safe_list(
            f"key-metrics/{symbol}", warnings, params=params, what="key metrics"
        )
        ratios = self._safe_list(
            f"ratios/{symbol}", warnings, params=params, what="financial ratios"
        )

        if not (income or balance or cashflow):
            return []

        by_key: Dict[str, FundamentalsPeriod] = {}
        order: List[str] = []

        def _key(row: Dict[str, Any]) -> Optional[str]:
            # Prefer the precise reporting date; fall back to calendar year.
            date = _nonempty_str(row.get("date")) or _nonempty_str(row.get("fillingDate"))
            if date:
                return date.split(" ", 1)[0]
            year = row.get("calendarYear")
            return str(year) if year is not None else None

        # Seed periods primarily from the income statement (broadest coverage),
        # then ensure any period present only in another statement also appears.
        for source in (income, balance, cashflow, metrics, ratios):
            for row in source:
                key = _key(row)
                if key is None:
                    continue
                if key not in by_key:
                    period_end = _to_date(row.get("date")) or _to_date(row.get("fillingDate"))
                    if period_end is None:
                        # Synthesize Dec-31 of the calendar year when only a year is known.
                        year = row.get("calendarYear")
                        try:
                            period_end = _dt.date(int(year), 12, 31)
                        except (TypeError, ValueError):
                            continue
                    label = self._period_label(row, period_end)
                    by_key[key] = FundamentalsPeriod(
                        period_end=period_end, period_label=label
                    )
                    order.append(key)

        self._apply_income(by_key, income)
        self._apply_balance(by_key, balance)
        self._apply_cashflow(by_key, cashflow)
        self._apply_key_metrics(by_key, metrics)
        self._apply_ratios(by_key, ratios)
        self._fill_derived_margins(by_key)

        # Newest first (SecurityData.fundamentals contract).
        periods = [by_key[k] for k in order]
        periods.sort(key=lambda p: p.period_end, reverse=True)
        return periods

    @staticmethod
    def _period_label(row: Dict[str, Any], period_end: _dt.date) -> str:
        """Produce a human label like ``FY2025`` or ``Q1 2026`` for a period."""
        period = _nonempty_str(row.get("period"))
        year = row.get("calendarYear")
        year_str = str(year) if year is not None else str(period_end.year)
        if period and period.upper().startswith("Q"):
            return f"{period.upper()} {year_str}"
        if period and period.upper() in ("FY", "ANNUAL"):
            return f"FY{year_str}"
        return f"FY{year_str}"

    def _matched(
        self, by_key: Dict[str, FundamentalsPeriod], row: Dict[str, Any]
    ) -> Optional[FundamentalsPeriod]:
        """Return the period a statement ``row`` belongs to, if it was seeded."""
        date = _nonempty_str(row.get("date")) or _nonempty_str(row.get("fillingDate"))
        key = date.split(" ", 1)[0] if date else None
        if key is None:
            year = row.get("calendarYear")
            key = str(year) if year is not None else None
        if key is None:
            return None
        return by_key.get(key)

    def _apply_income(
        self, by_key: Dict[str, FundamentalsPeriod], rows: List[Dict[str, Any]]
    ) -> None:
        """Map income-statement line items onto the matching periods."""
        for row in rows:
            period = self._matched(by_key, row)
            if period is None:
                continue
            self._set_if_none(period, "revenue", _to_float(row.get("revenue")))
            self._set_if_none(period, "gross_profit", _to_float(row.get("grossProfit")))
            self._set_if_none(
                period, "operating_income", _to_float(row.get("operatingIncome"))
            )
            self._set_if_none(period, "net_income", _to_float(row.get("netIncome")))
            self._set_if_none(period, "ebitda", _to_float(row.get("ebitda")))
            self._set_if_none(period, "eps_diluted", _to_float(row.get("epsdiluted")))
            shares = _to_float(row.get("weightedAverageShsOutDil")) or _to_float(
                row.get("weightedAverageShsOut")
            )
            self._set_if_none(period, "shares_diluted", shares)

    def _apply_balance(
        self, by_key: Dict[str, FundamentalsPeriod], rows: List[Dict[str, Any]]
    ) -> None:
        """Map balance-sheet line items onto the matching periods."""
        for row in rows:
            period = self._matched(by_key, row)
            if period is None:
                continue
            self._set_if_none(period, "total_assets", _to_float(row.get("totalAssets")))
            self._set_if_none(
                period, "total_equity",
                _to_float(row.get("totalStockholdersEquity"))
                or _to_float(row.get("totalEquity")),
            )
            self._set_if_none(
                period, "cash_and_equivalents",
                _to_float(row.get("cashAndCashEquivalents"))
                or _to_float(row.get("cashAndShortTermInvestments")),
            )
            total_debt = _to_float(row.get("totalDebt"))
            if total_debt is None:
                short_debt = _to_float(row.get("shortTermDebt")) or 0.0
                long_debt = _to_float(row.get("longTermDebt"))
                if long_debt is not None:
                    total_debt = short_debt + long_debt
            self._set_if_none(period, "total_debt", total_debt)

    def _apply_cashflow(
        self, by_key: Dict[str, FundamentalsPeriod], rows: List[Dict[str, Any]]
    ) -> None:
        """Map cash-flow-statement line items onto the matching periods."""
        for row in rows:
            period = self._matched(by_key, row)
            if period is None:
                continue
            ocf = _to_float(row.get("operatingCashFlow")) or _to_float(
                row.get("netCashProvidedByOperatingActivities")
            )
            self._set_if_none(period, "operating_cash_flow", ocf)
            capex = _to_float(row.get("capitalExpenditure"))
            self._set_if_none(period, "capex", capex)
            fcf = _to_float(row.get("freeCashFlow"))
            if fcf is None and ocf is not None and capex is not None:
                # capex is reported as a negative number by FMP; OCF + capex == FCF.
                fcf = ocf + capex
            self._set_if_none(period, "free_cash_flow", fcf)

    def _apply_key_metrics(
        self, by_key: Dict[str, FundamentalsPeriod], rows: List[Dict[str, Any]]
    ) -> None:
        """Map ``key-metrics`` returns/efficiency figures onto the periods."""
        for row in rows:
            period = self._matched(by_key, row)
            if period is None:
                continue
            self._set_if_none(period, "roic", _to_float(row.get("roic")))
            self._set_if_none(
                period, "roe",
                _to_float(row.get("roe")) or _to_float(row.get("returnOnEquity")),
            )
            self._set_if_none(
                period, "interest_coverage", _to_float(row.get("interestCoverage"))
            )
            self._set_if_none(
                period, "current_ratio", _to_float(row.get("currentRatio"))
            )

    def _apply_ratios(
        self, by_key: Dict[str, FundamentalsPeriod], rows: List[Dict[str, Any]]
    ) -> None:
        """Map the per-period ``ratios`` endpoint onto the periods."""
        for row in rows:
            period = self._matched(by_key, row)
            if period is None:
                continue
            self._set_if_none(
                period, "gross_margin", _to_float(row.get("grossProfitMargin"))
            )
            self._set_if_none(
                period, "operating_margin", _to_float(row.get("operatingProfitMargin"))
            )
            self._set_if_none(
                period, "roe", _to_float(row.get("returnOnEquity"))
            )
            self._set_if_none(
                period, "roa",
                _to_float(row.get("returnOnAssets")) or _to_float(row.get("returnOnTotalAssets")),
            )
            self._set_if_none(
                period, "current_ratio", _to_float(row.get("currentRatio"))
            )
            self._set_if_none(
                period, "quick_ratio", _to_float(row.get("quickRatio"))
            )
            self._set_if_none(
                period, "debt_to_equity",
                _to_float(row.get("debtEquityRatio")) or _to_float(row.get("debtToEquity")),
            )
            self._set_if_none(
                period, "interest_coverage", _to_float(row.get("interestCoverage"))
            )

    @staticmethod
    def _fill_derived_margins(by_key: Dict[str, FundamentalsPeriod]) -> None:
        """Compute margins from line items only where the vendor left them blank.

        These are deterministic identities (gross_profit/revenue etc.), not
        fabricated data: they are derived strictly from values FMP already supplied,
        and are skipped when any required input is missing or revenue is zero.
        """
        for period in by_key.values():
            rev = period.revenue
            if rev and rev != 0:
                if period.gross_margin is None and period.gross_profit is not None:
                    period.gross_margin = period.gross_profit / rev
                if period.operating_margin is None and period.operating_income is not None:
                    period.operating_margin = period.operating_income / rev
                if period.fcf_margin is None and period.free_cash_flow is not None:
                    period.fcf_margin = period.free_cash_flow / rev
            equity = period.total_equity
            if (
                period.debt_to_equity is None
                and period.total_debt is not None
                and equity
                and equity != 0
            ):
                period.debt_to_equity = period.total_debt / equity

    @staticmethod
    def _set_if_none(period: FundamentalsPeriod, attr: str, value: Optional[float]) -> None:
        """Set ``period.attr`` to ``value`` only if currently ``None`` and value is real.

        Preserves the first (most authoritative) non-null source for a field and
        never overwrites a real value with ``None``.
        """
        if value is None:
            return
        if getattr(period, attr) is None:
            setattr(period, attr, value)

    def _fetch_valuation(
        self,
        symbol: str,
        profile: Optional[Dict[str, Any]],
        warnings: List[str],
    ) -> ValuationSnapshot:
        """Build a :class:`ValuationSnapshot` from key-metrics-TTM + ratios-TTM.

        Anchors ``market_cap`` from the profile (or key-metrics), then layers TTM
        multiples. Any multiple FMP does not provide stays ``None``.
        """
        snapshot = ValuationSnapshot()

        if profile is not None:
            snapshot.market_cap = _to_float(profile.get("mktCap")) or _to_float(
                profile.get("marketCap")
            )

        metrics_ttm = self._safe_list(
            f"key-metrics-ttm/{symbol}", warnings, what="key metrics (TTM)"
        )
        if metrics_ttm:
            m = metrics_ttm[0]
            if snapshot.market_cap is None:
                snapshot.market_cap = _to_float(m.get("marketCapTTM")) or _to_float(
                    m.get("marketCap")
                )
            snapshot.enterprise_value = _to_float(m.get("enterpriseValueTTM")) or _to_float(
                m.get("enterpriseValue")
            )
            snapshot.pe = _to_float(m.get("peRatioTTM")) or _to_float(m.get("peRatio"))
            snapshot.ev_ebitda = _to_float(
                m.get("enterpriseValueOverEBITDATTM")
            ) or _to_float(m.get("evToEBITDATTM"))
            snapshot.ev_sales = _to_float(
                m.get("evToSalesTTM")
            ) or _to_float(m.get("enterpriseValueOverRevenueTTM"))
            snapshot.p_fcf = _to_float(m.get("pfcfRatioTTM")) or _to_float(
                m.get("priceToFreeCashFlowsRatioTTM")
            )
            snapshot.p_b = _to_float(m.get("pbRatioTTM")) or _to_float(m.get("ptbRatioTTM"))
            snapshot.p_s = _to_float(m.get("priceToSalesRatioTTM"))

        ratios_ttm = self._safe_list(
            f"ratios-ttm/{symbol}", warnings, what="financial ratios (TTM)"
        )
        if ratios_ttm:
            r = ratios_ttm[0]
            if snapshot.pe is None:
                snapshot.pe = _to_float(r.get("priceEarningsRatioTTM"))
            if snapshot.ev_ebitda is None:
                snapshot.ev_ebitda = _to_float(r.get("enterpriseValueMultipleTTM"))
            if snapshot.p_fcf is None:
                snapshot.p_fcf = _to_float(r.get("priceToFreeCashFlowsRatioTTM"))
            if snapshot.p_b is None:
                snapshot.p_b = _to_float(r.get("priceToBookRatioTTM"))
            if snapshot.p_s is None:
                snapshot.p_s = _to_float(r.get("priceToSalesRatioTTM"))
            if snapshot.peg is None:
                snapshot.peg = _to_float(r.get("priceEarningsToGrowthRatioTTM"))
            if snapshot.forward_pe is None:
                snapshot.forward_pe = _to_float(r.get("forwardPERatioTTM"))

        return snapshot

    def _fetch_insider(
        self, symbol: str, warnings: List[str]
    ) -> List[InsiderTransaction]:
        """Map FMP insider trading rows into :class:`InsiderTransaction` records."""
        rows = self._safe_list(
            "insider-trading",
            warnings,
            params={"symbol": symbol, "limit": _INSIDER_LIMIT},
            what="insider transactions",
        )
        out: List[InsiderTransaction] = []
        for row in rows:
            date = _to_date(row.get("transactionDate")) or _to_date(row.get("filingDate"))
            if date is None:
                continue
            name = (
                _nonempty_str(row.get("reportingName"))
                or _nonempty_str(row.get("name"))
                or "Unknown insider"
            )
            shares = _to_float(row.get("securitiesTransacted")) or _to_float(
                row.get("shares")
            )
            price = _to_float(row.get("price"))
            value = _to_float(row.get("value"))
            if value is None and shares is not None and price is not None:
                value = shares * price
            out.append(
                InsiderTransaction(
                    date=date,
                    insider_name=name,
                    role=_nonempty_str(row.get("typeOfOwner"))
                    or _nonempty_str(row.get("relationship")),
                    transaction_type=self._normalize_insider_type(row),
                    shares=shares,
                    value=value,
                )
            )
        if not out:
            warnings.append("FMP: no insider transactions reported")
        return out

    @staticmethod
    def _normalize_insider_type(row: Dict[str, Any]) -> str:
        """Normalise an FMP transaction code/type into buy/sell/grant/exercise/other.

        FMP supplies a single-letter SEC ``transactionType`` code (e.g. ``P-Purchase``,
        ``S-Sale``) and/or an ``acquistionOrDisposition`` flag (``A``/``D``). The raw
        code is preserved as ``other`` when it cannot be confidently classified so
        nothing is mislabelled.
        """
        raw = (
            _nonempty_str(row.get("transactionType"))
            or _nonempty_str(row.get("transactionCode"))
            or ""
        )
        code = raw.upper()
        if code.startswith("P") or "PURCHASE" in code:
            return "buy"
        if code.startswith("S") or "SALE" in code or "SELL" in code:
            return "sell"
        if code.startswith("A") and "ACQ" in code:
            return "buy"
        if code.startswith("G") or "GRANT" in code or code.startswith("A"):
            return "grant"
        if code.startswith("M") or "EXERCISE" in code or code.startswith("X"):
            return "exercise"
        ad = _nonempty_str(row.get("acquistionOrDisposition")) or _nonempty_str(
            row.get("acquisitionOrDisposition")
        )
        if ad:
            au = ad.upper()
            if au.startswith("A"):
                return "buy"
            if au.startswith("D"):
                return "sell"
        return "other"

    def _fetch_institutional(
        self, symbol: str, warnings: List[str]
    ) -> List[InstitutionalHolding]:
        """Map FMP institutional ownership rows into :class:`InstitutionalHolding`."""
        rows = self._safe_list(
            "institutional-holder",
            warnings,
            params={"symbol": symbol, "limit": _INSTITUTIONAL_LIMIT},
            what="institutional holdings",
        )
        # Newer FMP variants expose ``institutional-ownership/symbol-ownership``;
        # fall back to it when the simple holder list is empty.
        if not rows:
            rows = self._safe_list(
                "institutional-ownership/symbol-ownership",
                warnings,
                params={"symbol": symbol, "includeCurrentQuarter": "true"},
                what="institutional ownership",
            )

        out: List[InstitutionalHolding] = []
        for row in rows:
            holder = (
                _nonempty_str(row.get("holder"))
                or _nonempty_str(row.get("investorName"))
                or _nonempty_str(row.get("name"))
            )
            if holder is None:
                continue
            change_pct = _to_float(row.get("changeInSharesNumberPercentage"))
            if change_pct is None:
                change = _to_float(row.get("change"))
                prior = _to_float(row.get("sharesNumber"))
                # Only derive a percentage when both the change and a sensible base
                # (the prior position) are available; never invent one otherwise.
                if change is not None and prior is not None and (prior - change) not in (None, 0):
                    base = prior - change
                    if base:
                        change_pct = change / base * 100.0
            out.append(
                InstitutionalHolding(
                    holder=holder,
                    shares=_to_float(row.get("shares")) or _to_float(row.get("sharesNumber")),
                    value=_to_float(row.get("marketValue")) or _to_float(row.get("value")),
                    change_pct=change_pct,
                    as_of=_to_date(row.get("dateReported")) or _to_date(row.get("date")),
                )
            )
        if not out:
            warnings.append("FMP: no institutional holdings reported")
        return out


__all__ = ["FMPProvider"]
