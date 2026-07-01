"""Construction of the eligible US small-/micro-cap research universe.

What "the entire eligible universe" means here
-----------------------------------------------
Convexity screens the *whole* set of US-listed common stocks rather than a
hand-picked watchlist, so that the evidence-aggregation downstream is applied
without selection bias. Building that universe is a two-stage funnel:

1. **Listing enumeration** — :func:`fetch_listed_symbols` downloads the official
   Nasdaq Trader symbol directories (``nasdaqlisted.txt`` and ``otherlisted.txt``,
   which together cover every Nasdaq-, NYSE-, NYSE American- and Cboe-listed
   security) and keeps only ordinary common shares. ETFs, closed-end and
   open-end funds, warrants, units, rights, preferreds, notes and exchange *test*
   issues are filtered out using the directories' own flag columns plus a small
   set of suffix heuristics. If Nasdaq Trader is unreachable the function falls
   back to the SEC's ``company_tickers.json`` (every SEC-registered filer), and if
   *that* is unreachable too it falls back to the bundled curated seed list
   (:func:`load_seed_universe`) so the tool still runs offline.

2. **Eligibility screen** — :func:`build_universe` takes that raw list of common
   stocks and keeps only those whose **market cap** sits inside
   ``[params.min_market_cap, params.max_market_cap]`` and whose **average dollar
   volume** clears the liquidity floor ``params.min_avg_dollar_volume``. Market
   cap and dollar volume are obtained through quick *batched quotes* from the
   supplied price provider, never fabricated: a ticker whose cap or liquidity we
   cannot determine is conservatively excluded (and counted), so the eligible set
   only ever contains names we could positively verify as in-band.

Speed / coverage tradeoff
-------------------------
Enumerating ~6–8k common stocks and then quoting each one is the *coverage*
extreme: complete, unbiased, but network-heavy. For fast iterations the caller
sets ``params.universe_limit`` to cap how many symbols are quoted (the listing
list is deterministically ordered, so a limit yields a stable, reproducible
sub-universe rather than a random one). ``build_universe`` quotes in batches and
tolerates per-batch and per-ticker failures without aborting the scan. The
honest cost of a limited run is reduced coverage, not reduced correctness: every
ticker that survives the screen genuinely met the cap and liquidity bands.

This module is **not** a :class:`~convexity.core.contracts.DataProvider`; it is a
pure-ish helper used by the pipeline's screening stage. It performs network I/O
in :func:`fetch_listed_symbols`/:func:`build_universe` but degrades to the
bundled seed list whenever the network is unavailable, and never raises out of a
scan for a single bad symbol or batch.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Set, Tuple

from convexity.core.logging import get_logger
from convexity.core.models import ScanParams

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Source endpoints & on-disk locations
# ---------------------------------------------------------------------------

# Nasdaq Trader publishes pipe-delimited directories of every listed security.
# ``nasdaqlisted.txt`` covers Nasdaq; ``otherlisted.txt`` covers NYSE, NYSE
# American (AMEX), NYSE Arca, Cboe BZX, etc. Together they are the authoritative
# enumeration of US-exchange-listed securities.
_NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
_OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# SEC fallback: a JSON map of every SEC-registered ticker/CIK pair.
_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Bundled curated fallback shipped alongside this module.
_SEED_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_universe.csv")

# A generic, identifiable User-Agent for the SEC (which mandates one) and Nasdaq.
_DEFAULT_USER_AGENT = "Convexity research tool (contact: set SEC_USER_AGENT)"
_DEFAULT_TIMEOUT = 20.0

# How many symbols to request per batched-quote call when screening.
_QUOTE_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Symbol-classification heuristics (exclude non-common-stock instruments)
# ---------------------------------------------------------------------------

# Substrings appearing in a security's *name* that mark it as not-common-stock.
# These complement the structured flag columns in the Nasdaq directories and are
# the only signal available in the leaner SEC fallback.
_NON_COMMON_NAME_TOKENS: Tuple[str, ...] = (
    " etf",
    " etn",
    "exchange traded fund",
    "exchange-traded fund",
    " fund",
    " trust ",  # note: many REITs say "Realty Trust"/"Property Trust"; handled below
    " warrant",
    " warrants",
    " right",
    " rights",
    " unit",
    " units",
    " depositary",
    " preferred",
    " pfd",
    "% notes",
    " notes due",
    " debenture",
    " when issued",
    " when-issued",
    "index linked",
    " spdr",
    " ishares",
    " proshares",
    " invesco qqq",
)

# Name tokens that *rescue* a security the broad " trust " token would wrongly
# drop — operating REITs and a few corporates legitimately use "Trust" in their
# name and are common-stock equities we want to keep.
_TRUST_RESCUE_TOKENS: Tuple[str, ...] = (
    "realty trust",
    "property trust",
    "properties trust",
    "industrial trust",
    "residential trust",
    "office trust",
    "retail trust",
    "mortgage trust",
    "capital trust",
    "growth trust",
    "income trust",  # operating REITs; CEFs are caught by the structured ETF flag
    "water trust",
    "storage trust",
    "hotel trust",
    "infrastructure trust",
)

# Ticker suffixes (after a separator) that denote warrants / units / rights /
# preferred / when-issued lines rather than the common share.
_NON_COMMON_TICKER_SUFFIXES: Tuple[str, ...] = (
    "W",   # warrant (e.g. ABCDW)
    "WS",  # warrant
    "U",   # unit (e.g. ABCDU)
    "R",   # right
    "RT",  # right
    "P",   # preferred series marker on some feeds
    "WI",  # when-issued
)


def _looks_like_common_stock(ticker: str, name: str) -> bool:
    """Heuristically decide whether ``(ticker, name)`` is an ordinary common share.

    This is the name/suffix layer of the filter; the structured ETF/test flags in
    the Nasdaq directories are applied separately in :func:`_parse_nasdaq_listed`
    and :func:`_parse_other_listed`. The goal is to drop ETFs/funds, warrants,
    units, rights, preferreds and notes while retaining operating companies
    (including REITs). It is intentionally conservative: when in doubt it keeps
    the symbol, since the later cap/liquidity screen will still gate inclusion.
    """
    if not ticker or not name:
        return False

    upper_ticker = ticker.strip().upper()
    lower_name = f" {name.strip().lower()} "

    # Symbols carrying a class/share-class marker via '$', '.', '/' may be common
    # (e.g. BRK.B). We only reject suffixes that clearly denote derivatives.
    # Normalise the separator forms Nasdaq/CQS use.
    sep_ticker = upper_ticker.replace("$", ".").replace("/", ".")
    if "." in sep_ticker:
        base, _, suffix = sep_ticker.rpartition(".")
        if suffix in _NON_COMMON_TICKER_SUFFIXES and base:
            return False

    # Bare-suffix warrants/units/rights with no separator (e.g. "ABCDW", "ABCDU").
    # Only treat a trailing W/U/R/RT as a derivative when the ticker is long
    # enough that the suffix is plausibly an appended marker, to avoid nuking
    # legitimate 1–4 char tickers that merely end in those letters.
    if len(upper_ticker) >= 5:
        for suffix in ("WS", "RT", "WI"):
            if upper_ticker.endswith(suffix) and len(upper_ticker) > len(suffix):
                return False
        if upper_ticker.endswith(("W", "U", "R")):
            # e.g. a 5+ char symbol ending in W is almost always a warrant.
            return False

    # Name-based exclusion, with a rescue for operating REITs that use "Trust".
    is_rescued_trust = any(tok in lower_name for tok in _TRUST_RESCUE_TOKENS)
    for token in _NON_COMMON_NAME_TOKENS:
        if token in lower_name:
            if token == " trust " and is_rescued_trust:
                continue
            return False

    return True


# ---------------------------------------------------------------------------
# HTTP helper (httpx preferred, requests fallback, both optional)
# ---------------------------------------------------------------------------


def _http_get_text(url: str, *, user_agent: str, timeout: float) -> Optional[str]:
    """GET ``url`` and return its body as text, or ``None`` on any failure.

    Tries ``httpx`` then ``requests``; if neither is importable or the request
    fails for any reason, returns ``None`` so callers can fall back gracefully
    (this function never raises). A descriptive ``User-Agent`` is always sent,
    which the SEC requires.
    """
    headers = {"User-Agent": user_agent, "Accept": "text/plain,application/json,*/*"}

    try:
        import httpx  # type: ignore

        try:
            resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
            if resp.status_code == 200 and resp.text:
                return resp.text
            _log.warning("GET %s returned status %s (httpx)", url, resp.status_code)
        except Exception as exc:  # pragma: no cover - network dependent
            _log.warning("httpx GET %s failed: %s", url, exc)
    except ImportError:  # pragma: no cover - environment dependent
        pass

    try:
        import requests  # type: ignore

        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200 and resp.text:
                return resp.text
            _log.warning("GET %s returned status %s (requests)", url, resp.status_code)
        except Exception as exc:  # pragma: no cover - network dependent
            _log.warning("requests GET %s failed: %s", url, exc)
    except ImportError:  # pragma: no cover - environment dependent
        _log.warning("neither httpx nor requests is available to fetch %s", url)

    return None


# ---------------------------------------------------------------------------
# Parsing of the Nasdaq Trader directory files
# ---------------------------------------------------------------------------


def _parse_pipe_table(text: str) -> Tuple[List[str], List[List[str]]]:
    """Parse a Nasdaq Trader pipe-delimited file into ``(header, rows)``.

    The files end with a ``File Creation Time`` trailer line (no pipes in the
    leading field the way data rows have); that trailer and any blank lines are
    discarded. Returns the header tokens and the data rows split on ``|``.
    """
    header: List[str] = []
    rows: List[List[str]] = []
    for i, raw in enumerate(text.splitlines()):
        line = raw.rstrip("\r\n")
        if not line:
            continue
        # The trailer line begins with "File Creation Time".
        if line.startswith("File Creation Time"):
            continue
        parts = line.split("|")
        if i == 0 or (not header and parts):
            header = [p.strip() for p in parts]
            continue
        rows.append(parts)
    return header, rows


def _column_index(header: Sequence[str], name: str) -> Optional[int]:
    """Return the index of column ``name`` in ``header`` (case-insensitive)."""
    target = name.strip().lower()
    for idx, col in enumerate(header):
        if col.strip().lower() == target:
            return idx
    return None


def _parse_nasdaq_listed(text: str) -> List[Tuple[str, str]]:
    """Parse ``nasdaqlisted.txt`` -> list of ``(ticker, name)`` common stocks.

    Columns: ``Symbol|Security Name|Market Category|Test Issue|Financial Status|
    Round Lot Size|ETF|NextShares``. We drop test issues (``Test Issue == 'Y'``)
    and ETFs (``ETF == 'Y'``), then apply the name/suffix common-stock heuristic.
    """
    header, rows = _parse_pipe_table(text)
    i_sym = _column_index(header, "Symbol")
    i_name = _column_index(header, "Security Name")
    i_test = _column_index(header, "Test Issue")
    i_etf = _column_index(header, "ETF")
    if i_sym is None or i_name is None:
        _log.warning("nasdaqlisted.txt header not recognised: %s", header)
        return []

    out: List[Tuple[str, str]] = []
    for parts in rows:
        if len(parts) <= max(i_sym, i_name):
            continue
        ticker = parts[i_sym].strip()
        name = parts[i_name].strip()
        if i_test is not None and len(parts) > i_test and parts[i_test].strip().upper() == "Y":
            continue
        if i_etf is not None and len(parts) > i_etf and parts[i_etf].strip().upper() == "Y":
            continue
        if _looks_like_common_stock(ticker, name):
            out.append((ticker, name))
    return out


def _parse_other_listed(text: str) -> List[Tuple[str, str]]:
    """Parse ``otherlisted.txt`` -> list of ``(ticker, name)`` common stocks.

    Columns: ``ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|
    Test Issue|NASDAQ Symbol``. We prefer the ``ACT Symbol`` (the canonical CQS
    ticker), drop test issues and ETFs, then apply the common-stock heuristic.
    """
    header, rows = _parse_pipe_table(text)
    i_sym = _column_index(header, "ACT Symbol")
    if i_sym is None:
        i_sym = _column_index(header, "CQS Symbol")
    i_name = _column_index(header, "Security Name")
    i_test = _column_index(header, "Test Issue")
    i_etf = _column_index(header, "ETF")
    if i_sym is None or i_name is None:
        _log.warning("otherlisted.txt header not recognised: %s", header)
        return []

    out: List[Tuple[str, str]] = []
    for parts in rows:
        if len(parts) <= max(i_sym, i_name):
            continue
        ticker = parts[i_sym].strip()
        name = parts[i_name].strip()
        if i_test is not None and len(parts) > i_test and parts[i_test].strip().upper() == "Y":
            continue
        if i_etf is not None and len(parts) > i_etf and parts[i_etf].strip().upper() == "Y":
            continue
        if _looks_like_common_stock(ticker, name):
            out.append((ticker, name))
    return out


def _parse_sec_company_tickers(text: str) -> List[Tuple[str, str]]:
    """Parse SEC ``company_tickers.json`` -> ``(ticker, name)`` common stocks.

    The SEC file has no ETF/derivative flags, so only the name/suffix heuristic
    can be applied. It is a coverage fallback, not the primary source.
    """
    import json

    out: List[Tuple[str, str]] = []
    try:
        payload = json.loads(text)
    except Exception as exc:  # pragma: no cover - malformed payload
        _log.warning("could not parse SEC company_tickers.json: %s", exc)
        return out

    # The file is a dict keyed by stringified ints -> {"cik_str","ticker","title"}.
    entries = payload.values() if isinstance(payload, dict) else payload
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ticker = str(entry.get("ticker", "")).strip()
        name = str(entry.get("title", "")).strip()
        if ticker and name and _looks_like_common_stock(ticker, name):
            out.append((ticker, name))
    return out


def _dedupe_preserve_order(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """De-duplicate ``(ticker, name)`` pairs by upper-cased ticker, keep first seen."""
    seen: Set[str] = set()
    out: List[Tuple[str, str]] = []
    for ticker, name in pairs:
        key = ticker.strip().upper()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((key, name))
    return out


# ---------------------------------------------------------------------------
# Public: full listed-symbol enumeration
# ---------------------------------------------------------------------------


def fetch_listed_symbols(
    *,
    user_agent: Optional[str] = None,
    timeout: float = _DEFAULT_TIMEOUT,
    include_names: bool = False,
) -> List[Any]:
    """Return every US-listed *common stock* symbol, filtering out non-equities.

    Primary source is the pair of Nasdaq Trader directory files (``nasdaqlisted``
    + ``otherlisted``), which between them enumerate all Nasdaq-, NYSE-, NYSE
    American-, NYSE Arca- and Cboe-listed securities. ETFs, funds, warrants,
    units, rights, preferreds, notes and exchange test issues are removed using
    the files' structured flags plus name/suffix heuristics. If Nasdaq Trader is
    unreachable, falls back to the SEC ``company_tickers.json`` file; if that too
    is unreachable, falls back to the bundled curated seed list so the tool keeps
    working offline.

    Args:
        user_agent: Identifiable UA string (required by the SEC). Defaults to a
            generic Convexity UA; override with a contact address in production.
        timeout: Per-request timeout in seconds.
        include_names: When ``True`` return ``(ticker, name)`` tuples; otherwise
            return a flat, sorted list of ticker strings.

    Returns:
        A de-duplicated list of common-stock tickers (or ``(ticker, name)`` pairs
        if ``include_names``), deterministically ordered (ascending by ticker) so
        downstream limiting is reproducible. Never raises for a network failure.
    """
    ua = user_agent or _DEFAULT_USER_AGENT
    pairs: List[Tuple[str, str]] = []

    # --- Primary: Nasdaq Trader directories ----------------------------------
    nasdaq_text = _http_get_text(_NASDAQ_LISTED_URL, user_agent=ua, timeout=timeout)
    other_text = _http_get_text(_OTHER_LISTED_URL, user_agent=ua, timeout=timeout)
    if nasdaq_text:
        pairs.extend(_parse_nasdaq_listed(nasdaq_text))
    if other_text:
        pairs.extend(_parse_other_listed(other_text))

    if pairs:
        _log.info("fetched %d common stocks from Nasdaq Trader directories", len(pairs))
    else:
        # --- Secondary: SEC company_tickers.json -----------------------------
        _log.warning("Nasdaq Trader unavailable; falling back to SEC company_tickers.json")
        sec_text = _http_get_text(_SEC_TICKERS_URL, user_agent=ua, timeout=timeout)
        if sec_text:
            pairs.extend(_parse_sec_company_tickers(sec_text))
            _log.info("fetched %d common stocks from SEC company_tickers.json", len(pairs))

    if not pairs:
        # --- Tertiary: bundled curated seed list -----------------------------
        _log.warning("network symbol sources unavailable; using bundled seed universe")
        pairs = load_seed_universe(include_sector=False)  # type: ignore[assignment]

    deduped = _dedupe_preserve_order(pairs)
    deduped.sort(key=lambda p: p[0])  # deterministic order for reproducible limiting

    if include_names:
        return deduped
    return [ticker for ticker, _name in deduped]


# ---------------------------------------------------------------------------
# Bundled curated fallback universe
# ---------------------------------------------------------------------------


def load_seed_universe(*, include_sector: bool = False) -> List[Any]:
    """Load the bundled curated small-/micro-cap seed universe from CSV.

    Ships ~150 real US small- and micro-cap common stocks spread across all
    eleven GICS sectors so the tool produces a meaningful, diversified scan even
    with **no network access**. This is a *curated convenience list*, not a claim
    of completeness: the live :func:`fetch_listed_symbols` path is the real
    "entire eligible universe". Selection here is by liquidity and listing
    longevity, not by any forward-looking view — inclusion is not a recommendation.

    Args:
        include_sector: When ``True`` return ``(ticker, name, sector)`` triples;
            otherwise return ``(ticker, name)`` pairs.

    Returns:
        A list of tuples read from ``seed_universe.csv``. Returns an empty list if
        the bundled file is somehow missing (it never raises into a scan).
    """
    rows: List[Any] = []
    if not os.path.exists(_SEED_CSV_PATH):  # pragma: no cover - packaging guard
        _log.error("bundled seed universe CSV missing at %s", _SEED_CSV_PATH)
        return rows

    try:
        with open(_SEED_CSV_PATH, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ticker = (row.get("ticker") or "").strip().upper()
                name = (row.get("name") or "").strip()
                sector = (row.get("sector") or "").strip()
                if not ticker:
                    continue
                if include_sector:
                    rows.append((ticker, name, sector))
                else:
                    rows.append((ticker, name))
    except Exception as exc:  # pragma: no cover - defensive
        _log.error("failed reading seed universe CSV: %s", exc)
        return []
    return rows


def load_seed_tickers() -> List[str]:
    """Return just the tickers from the bundled curated seed universe."""
    return [t for t, _name in load_seed_universe(include_sector=False)]


# ---------------------------------------------------------------------------
# Batched-quote screening
# ---------------------------------------------------------------------------

# A "quote" for screening purposes is any mapping that may carry a market cap and
# enough information to estimate average dollar volume. We read it duck-typed so
# any price provider works as long as it exposes one of the batched-quote methods
# below; we look up these keys (first present wins).
_MARKET_CAP_KEYS: Tuple[str, ...] = ("market_cap", "marketCap", "mktcap", "market_capitalization")
_AVG_DOLLAR_VOL_KEYS: Tuple[str, ...] = (
    "avg_dollar_volume",
    "average_dollar_volume",
    "dollar_volume",
    "avgDollarVolume",
)
_PRICE_KEYS: Tuple[str, ...] = ("price", "last", "close", "regularMarketPrice", "last_price")
_AVG_VOLUME_KEYS: Tuple[str, ...] = (
    "avg_volume",
    "average_volume",
    "avgVolume",
    "averageDailyVolume3Month",
    "volume",
)


def _coerce_float(value: Any) -> Optional[float]:
    """Best-effort convert ``value`` to a positive-or-zero float, else ``None``."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf guard
        return None
    return f


def _first_present(mapping: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    """Return the first numeric value among ``keys`` found in ``mapping``."""
    for key in keys:
        if key in mapping:
            val = _coerce_float(mapping[key])
            if val is not None:
                return val
    return None


def _extract_cap_and_liquidity(quote: Any) -> Tuple[Optional[float], Optional[float]]:
    """Pull ``(market_cap, avg_dollar_volume)`` out of a single duck-typed quote.

    Accepts dicts or objects with attributes. Market cap is read directly. Average
    dollar volume is read directly if present, else estimated as ``price ×
    average_volume`` when both are available. Anything that cannot be derived from
    real fields stays ``None`` (never fabricated).
    """
    # Normalise objects to a dict-like view.
    if isinstance(quote, dict):
        mapping: Dict[str, Any] = quote
    else:
        mapping = {
            k: getattr(quote, k)
            for k in (
                _MARKET_CAP_KEYS
                + _AVG_DOLLAR_VOL_KEYS
                + _PRICE_KEYS
                + _AVG_VOLUME_KEYS
            )
            if hasattr(quote, k)
        }

    market_cap = _first_present(mapping, _MARKET_CAP_KEYS)

    dollar_volume = _first_present(mapping, _AVG_DOLLAR_VOL_KEYS)
    if dollar_volume is None:
        price = _first_present(mapping, _PRICE_KEYS)
        avg_volume = _first_present(mapping, _AVG_VOLUME_KEYS)
        if price is not None and avg_volume is not None:
            dollar_volume = price * avg_volume

    return market_cap, dollar_volume


def _call_batched_quotes(price_provider: Any, batch: Sequence[str]) -> Dict[str, Any]:
    """Ask ``price_provider`` for quotes on ``batch`` and normalise to a dict.

    The price provider is duck-typed. We try, in order, the common batched
    interfaces and finally a per-ticker method, so a wide range of provider shapes
    work without this module importing any of them:

    * ``get_quotes(tickers) -> Mapping[ticker, quote] | Sequence[quote]``
    * ``batch_quotes(tickers) -> ...`` / ``quotes(tickers) -> ...``
    * ``get_quote(ticker) -> quote`` (called per ticker as a last resort)

    Returns a ``{ticker: quote}`` mapping for whatever could be fetched. Never
    raises: a failed batch logs a warning and yields an empty mapping so the
    overall screen continues.
    """
    tickers = list(batch)

    def _normalise(result: Any) -> Dict[str, Any]:
        if result is None:
            return {}
        if isinstance(result, dict):
            return {str(k).upper(): v for k, v in result.items()}
        # A sequence aligned positionally with the requested tickers, or a
        # sequence of quote objects each carrying their own symbol.
        out: Dict[str, Any] = {}
        try:
            seq = list(result)
        except TypeError:
            return {}
        for idx, item in enumerate(seq):
            sym = None
            if isinstance(item, dict):
                for key in ("symbol", "ticker", "Symbol", "Ticker"):
                    if key in item:
                        sym = str(item[key]).upper()
                        break
            else:
                for key in ("symbol", "ticker"):
                    if hasattr(item, key):
                        sym = str(getattr(item, key)).upper()
                        break
            if sym is None and idx < len(tickers):
                sym = tickers[idx].upper()
            if sym is not None:
                out[sym] = item
        return out

    for method_name in ("get_quotes", "batch_quotes", "quotes"):
        method = getattr(price_provider, method_name, None)
        if callable(method):
            try:
                return _normalise(method(tickers))
            except Exception as exc:  # pragma: no cover - provider dependent
                _log.warning("price_provider.%s failed for a batch: %s", method_name, exc)
                return {}

    # Last resort: a per-ticker quote method.
    single = getattr(price_provider, "get_quote", None)
    if callable(single):
        out: Dict[str, Any] = {}
        for ticker in tickers:
            try:
                quote = single(ticker)
            except Exception as exc:  # pragma: no cover - provider dependent
                _log.debug("price_provider.get_quote(%s) failed: %s", ticker, exc)
                continue
            if quote is not None:
                out[ticker.upper()] = quote
        return out

    _log.error(
        "price_provider %r exposes no recognised quote method "
        "(get_quotes/batch_quotes/quotes/get_quote); cannot screen",
        type(price_provider).__name__,
    )
    return {}


def _chunk(seq: Sequence[str], size: int) -> List[List[str]]:
    """Split ``seq`` into consecutive chunks of at most ``size`` elements."""
    if size <= 0:
        return [list(seq)]
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def build_universe(
    params: ScanParams,
    price_provider: Any,
    *,
    candidates: Optional[Sequence[str]] = None,
    user_agent: Optional[str] = None,
    timeout: float = _DEFAULT_TIMEOUT,
    batch_size: int = _QUOTE_BATCH_SIZE,
) -> List[str]:
    """Build the eligible small-/micro-cap universe by cap + liquidity screen.

    Starting from the full list of US common stocks (``candidates`` if supplied,
    otherwise :func:`fetch_listed_symbols`), this keeps only tickers whose market
    cap lies in ``[params.min_market_cap, params.max_market_cap]`` **and** whose
    average dollar volume meets ``params.min_avg_dollar_volume``. Cap and dollar
    volume come from quick *batched quotes* off ``price_provider``; both must be
    positively verified — a ticker we cannot quote, or whose cap/liquidity is
    unknown, is conservatively **excluded** (never assumed in-band).

    Speed vs coverage: set ``params.universe_limit`` to cap how many symbols are
    quoted. Because the candidate list is deterministically ordered, a limit
    yields a stable, reproducible sub-universe (the first *N* tickers), trading
    coverage for speed without sacrificing correctness — every surviving ticker
    genuinely met the bands.

    Args:
        params: Screen parameters (cap band, liquidity floor, ``universe_limit``).
        price_provider: Any object exposing a batched quote method
            (``get_quotes`` / ``batch_quotes`` / ``quotes``) or a per-ticker
            ``get_quote``; read duck-typed so no provider import is needed.
        candidates: Optional pre-fetched candidate tickers (skips network
            enumeration); handy for tests and for reusing a cached listing.
        user_agent: UA passed to :func:`fetch_listed_symbols` when enumerating.
        timeout: Per-request timeout used during enumeration.
        batch_size: How many symbols to request per batched-quote call.

    Returns:
        A de-duplicated, order-preserving list of eligible tickers. Never raises
        for a single bad ticker or a failed quote batch.
    """
    # 1) Candidate symbols (full enumeration unless caller supplied a list).
    if candidates is None:
        candidate_list = fetch_listed_symbols(user_agent=user_agent, timeout=timeout)
    else:
        candidate_list = _dedupe_preserve_order([(c, "") for c in candidates])
        candidate_list = [t for t, _ in candidate_list]  # type: ignore[misc]

    if not candidate_list:
        _log.warning("no candidate symbols to screen; returning empty universe")
        return []

    # 2) Honour the fast-run cap on how many symbols we quote.
    limit = params.universe_limit
    if limit is not None and limit >= 0:
        if len(candidate_list) > limit:
            _log.info(
                "universe_limit=%d applied: quoting first %d of %d candidates",
                limit,
                limit,
                len(candidate_list),
            )
        candidate_list = candidate_list[:limit]

    min_cap = float(params.min_market_cap)
    max_cap = float(params.max_market_cap)
    min_dollar_vol = float(params.min_avg_dollar_volume)

    eligible: List[str] = []
    n_quoted = 0
    n_no_quote = 0
    n_cap_unknown = 0
    n_cap_out = 0
    n_illiquid = 0
    n_liq_unknown = 0

    # 3) Quote in batches; tolerate per-batch failures.
    for batch in _chunk(candidate_list, batch_size):
        quotes = _call_batched_quotes(price_provider, batch)
        for ticker in batch:
            quote = quotes.get(ticker.upper())
            if quote is None:
                n_no_quote += 1
                continue
            n_quoted += 1

            market_cap, dollar_volume = _extract_cap_and_liquidity(quote)

            # Market-cap band (must be positively known and in-band).
            if market_cap is None or market_cap <= 0:
                n_cap_unknown += 1
                continue
            if market_cap < min_cap or market_cap > max_cap:
                n_cap_out += 1
                continue

            # Liquidity floor (must be positively known and clear the floor).
            if min_dollar_vol > 0:
                if dollar_volume is None:
                    n_liq_unknown += 1
                    continue
                if dollar_volume < min_dollar_vol:
                    n_illiquid += 1
                    continue

            eligible.append(ticker.upper())

    eligible = [t for t, _ in _dedupe_preserve_order([(t, "") for t in eligible])]

    _log.info(
        "universe screen: %d candidates -> %d eligible "
        "(quoted=%d, no_quote=%d, cap_unknown=%d, cap_out_of_band=%d, "
        "liquidity_unknown=%d, illiquid=%d)",
        len(candidate_list),
        len(eligible),
        n_quoted,
        n_no_quote,
        n_cap_unknown,
        n_cap_out,
        n_liq_unknown,
        n_illiquid,
    )
    return eligible


def build_universe_or_seed(
    params: ScanParams,
    price_provider: Optional[Any] = None,
    *,
    user_agent: Optional[str] = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> List[str]:
    """Convenience wrapper: screen the live universe, else fall back to the seed.

    Returns the result of :func:`build_universe` when a usable ``price_provider``
    is supplied and the screen yields any eligible tickers; otherwise returns the
    bundled curated seed tickers (optionally truncated to ``params.universe_limit``).
    This keeps the pipeline runnable with zero network/credentials while still
    preferring the real, screened universe whenever it is available.
    """
    if price_provider is not None:
        try:
            screened = build_universe(
                params, price_provider, user_agent=user_agent, timeout=timeout
            )
        except Exception as exc:  # pragma: no cover - defensive top-level guard
            _log.error("build_universe failed unexpectedly; using seed list: %s", exc)
            screened = []
        if screened:
            return screened
        _log.warning("live universe screen produced no eligible tickers; using seed list")

    seed = load_seed_tickers()
    limit = params.universe_limit
    if limit is not None and limit >= 0:
        seed = seed[:limit]
    return seed


__all__ = [
    "fetch_listed_symbols",
    "build_universe",
    "build_universe_or_seed",
    "load_seed_universe",
    "load_seed_tickers",
]
