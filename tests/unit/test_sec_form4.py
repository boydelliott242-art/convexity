"""Offline unit tests for SEC EDGAR Form 4 insider-transaction ingestion.

Everything here is network-free: HTTP is monkeypatched at the provider's
``_get_text`` / ``_get_json`` seams and the submissions payloads are synthetic.
The canned XML below mirrors the real ``ownershipDocument`` schema (reporting
owner + non-derivative transactions) closely enough to exercise the honest
code-mapping contract: only ``P``/``S`` become ``buy``/``sell``; awards,
exercises and unknown codes are labelled as the non-market events they are.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Tuple

import pytest

from convexity.core.config import Settings
from convexity.core.exceptions import ProviderError, RateLimited
from convexity.data.providers.sec_edgar import (
    _FORM4_LOOKBACK_DAYS,
    _FORM4_MAX_FILINGS,
    SecEdgarProvider,
)

# ---------------------------------------------------------------------------
# Canned Form 4 ownershipDocument XML (realistic shape, fictional content)
# ---------------------------------------------------------------------------

VALID_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <schemaVersion>X0508</schemaVersion>
    <documentType>4</documentType>
    <issuer>
        <issuerCik>0000999999</issuerCik>
        <issuerName>Testco Industries Inc</issuerName>
        <issuerTradingSymbol>TSTC</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001234567</rptOwnerCik>
            <rptOwnerName>Doe Jane</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>1</isDirector>
            <isOfficer>1</isOfficer>
            <officerTitle>Chief Executive Officer</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-05-04</value></transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>P</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>10000</value></transactionShares>
                <transactionPricePerShare><value>2.50</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-05-06</value></transactionDate>
            <transactionCoding>
                <transactionCode>S</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>4000</value></transactionShares>
                <transactionPricePerShare><value>3.10</value></transactionPricePerShare>
            </transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-04-15</value></transactionDate>
            <transactionCoding>
                <transactionCode>A</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>5000</value></transactionShares>
            </transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-04-20</value></transactionDate>
            <transactionCoding>
                <transactionCode>M</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>2000</value></transactionShares>
                <transactionPricePerShare><value>1.00</value></transactionPricePerShare>
            </transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-04-22</value></transactionDate>
            <transactionCoding>
                <transactionCode>G</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>300</value></transactionShares>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
"""

DIRECTOR_ONLY_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerName>Smith Robert</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>true</isDirector>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionDate><value>2026-06-01</value></transactionDate>
            <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>1500</value></transactionShares>
                <transactionPricePerShare><value>4.00</value></transactionPricePerShare>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
"""

MALFORMED_XML = "<ownershipDocument><reportingOwner></ownershipDocument"


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


@pytest.fixture()
def provider(tmp_path) -> SecEdgarProvider:
    """A provider with an isolated on-disk cache and no network access needed."""
    settings = Settings(data_dir=str(tmp_path))
    return SecEdgarProvider(settings=settings)


def build_submissions(
    rows: List[Tuple[str, _dt.date, str, str]],
) -> Dict[str, Any]:
    """Build a synthetic submissions payload from (form, filed, accession, doc)."""
    return {
        "name": "Testco Industries Inc",
        "filings": {
            "recent": {
                "form": [r[0] for r in rows],
                "filingDate": [r[1].isoformat() for r in rows],
                "accessionNumber": [r[2] for r in rows],
                "primaryDocument": [r[3] for r in rows],
                "primaryDocDescription": ["" for _ in rows],
            }
        },
    }


class FetchRecorder:
    """Monkeypatch target for ``_get_text`` that serves canned XML by accession."""

    def __init__(self, xml_by_accession: Dict[str, str], default: Optional[str] = None):
        self.xml_by_accession = xml_by_accession
        self.default = default
        self.urls: List[str] = []

    def __call__(self, url: str, *, what: str) -> str:
        self.urls.append(url)
        for accession, xml in self.xml_by_accession.items():
            if accession.replace("-", "") in url:
                return xml
        if self.default is not None:
            return self.default
        raise AssertionError(f"unexpected Form 4 fetch: {url}")


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def test_parse_valid_form4_xml(provider: SecEdgarProvider) -> None:
    txns = provider._parse_form4_xml(VALID_FORM4_XML)
    assert len(txns) == 5

    by_code = {t.transaction_type: t for t in txns}

    buy = by_code["buy"]
    assert buy.insider_name == "Doe Jane"
    assert buy.role == "Chief Executive Officer"  # officerTitle wins over isDirector
    assert buy.date == _dt.date(2026, 5, 4)
    assert buy.shares == 10_000.0
    assert buy.value == pytest.approx(25_000.0)  # 10000 * 2.50

    sell = by_code["sell"]
    assert sell.date == _dt.date(2026, 5, 6)
    assert sell.shares == 4_000.0
    assert sell.value == pytest.approx(12_400.0)  # 4000 * 3.10

    award = by_code["award"]
    assert award.shares == 5_000.0
    assert award.value is None  # no price disclosed -> value never fabricated

    exercise = by_code["exercise"]
    assert exercise.shares == 2_000.0
    assert exercise.value == pytest.approx(2_000.0)

    other = by_code["other:G"]  # gift: honestly labelled, not a buy or sell
    assert other.shares == 300.0
    assert other.value is None


def test_parse_director_role_fallback(provider: SecEdgarProvider) -> None:
    txns = provider._parse_form4_xml(DIRECTOR_ONLY_FORM4_XML)
    assert len(txns) == 1
    assert txns[0].insider_name == "Smith Robert"
    assert txns[0].role == "Director"
    assert txns[0].transaction_type == "buy"
    assert txns[0].value == pytest.approx(6_000.0)


def test_parse_malformed_xml_raises_value_error(provider: SecEdgarProvider) -> None:
    with pytest.raises(ValueError):
        provider._parse_form4_xml(MALFORMED_XML)


# ---------------------------------------------------------------------------
# 12-month / 10-filing selection bounds
# ---------------------------------------------------------------------------


def test_select_form4_filings_applies_bounds() -> None:
    today = _dt.date(2026, 7, 1)
    rows: List[Tuple[str, _dt.date, str, str]] = []
    # 15 in-window Form 4s (every 10 days, oldest first so sorting is exercised).
    for i in range(15):
        filed = today - _dt.timedelta(days=10 * (15 - i))
        rows.append(("4", filed, f"0001-26-{i:06d}", "form4.xml"))
    # 3 stale Form 4s beyond the 12-month lookback.
    for i in range(3):
        filed = today - _dt.timedelta(days=_FORM4_LOOKBACK_DAYS + 30 + i)
        rows.append(("4", filed, f"0001-25-{i:06d}", "form4.xml"))
    # Other forms in-window must be ignored.
    rows.append(("10-K", today - _dt.timedelta(days=5), "0001-26-900001", "tenk.htm"))
    rows.append(("8-K", today - _dt.timedelta(days=3), "0001-26-900002", "eightk.htm"))

    submissions = build_submissions(rows)
    selected = SecEdgarProvider._select_form4_filings(submissions, today=today)

    assert len(selected) == _FORM4_MAX_FILINGS == 10
    accessions = [a for a, _ in selected]
    # Only Form 4 accessions, none stale, none from other forms.
    assert all(a.startswith("0001-26-") and not a.startswith("0001-26-9") for a in accessions)
    # Most-recent first: the newest in-window Form 4 is index 14, then 13, ...
    expected = [f"0001-26-{i:06d}" for i in range(14, 4, -1)]
    assert accessions == expected


# ---------------------------------------------------------------------------
# End-to-end (offline) ingestion: fetch cap, malformed-XML isolation, caching
# ---------------------------------------------------------------------------


def test_get_insider_transactions_skips_malformed_xml(
    provider: SecEdgarProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    today = _dt.date.today()
    rows = [
        ("4", today - _dt.timedelta(days=10), "0001-26-000001", "form4.xml"),
        ("4", today - _dt.timedelta(days=20), "0001-26-000002", "form4.xml"),
    ]
    submissions = build_submissions(rows)
    recorder = FetchRecorder(
        {
            "0001-26-000001": MALFORMED_XML,  # bad filing must be skipped, not fatal
            "0001-26-000002": VALID_FORM4_XML,
        }
    )
    monkeypatch.setattr(provider, "_get_text", recorder)

    txns = provider.get_insider_transactions("TSTC", cik=999_999, submissions=submissions)

    assert len(recorder.urls) == 2  # both filings were attempted
    assert len(txns) == 5  # only the valid filing's transactions survive
    assert {t.insider_name for t in txns} == {"Doe Jane"}
    # Newest-first ordering of the surviving transactions.
    dates = [t.date for t in txns]
    assert dates == sorted(dates, reverse=True)


def test_get_insider_transactions_caps_fetches_at_ten(
    provider: SecEdgarProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    today = _dt.date.today()
    rows = [
        ("4", today - _dt.timedelta(days=5 * (i + 1)), f"0001-26-{i:06d}", "form4.xml")
        for i in range(15)
    ]
    submissions = build_submissions(rows)
    recorder = FetchRecorder({}, default=DIRECTOR_ONLY_FORM4_XML)
    monkeypatch.setattr(provider, "_get_text", recorder)

    txns = provider.get_insider_transactions("TSTC", cik=999_999, submissions=submissions)

    assert len(recorder.urls) == _FORM4_MAX_FILINGS == 10
    assert len(txns) == 10  # one transaction per canned filing


def test_get_insider_transactions_uses_cache_on_repeat(
    provider: SecEdgarProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    today = _dt.date.today()
    rows = [("4", today - _dt.timedelta(days=7), "0001-26-000001", "form4.xml")]
    submissions = build_submissions(rows)
    recorder = FetchRecorder({"0001-26-000001": VALID_FORM4_XML})
    monkeypatch.setattr(provider, "_get_text", recorder)

    first = provider.get_insider_transactions("TSTC", cik=999_999, submissions=submissions)
    second = provider.get_insider_transactions("TSTC", cik=999_999, submissions=submissions)

    assert len(recorder.urls) == 1  # the second call was served from the disk cache
    assert [t.model_dump() for t in second] == [t.model_dump() for t in first]
    assert first and first[0].transaction_type == "sell"  # 2026-05-06 is newest
    assert any(t.transaction_type == "buy" for t in first)


def test_index_fallback_when_primary_document_is_not_xml(
    provider: SecEdgarProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    today = _dt.date.today()
    rows = [("4", today - _dt.timedelta(days=7), "0001-26-000001", "form4.html")]
    submissions = build_submissions(rows)

    index_payload = {
        "directory": {
            "item": [
                {"name": "form4.html"},
                {"name": "xslF345X05_form4.xml"},  # rendering artefact: skipped
                {"name": "wk-form4_123.xml"},
            ]
        }
    }
    json_urls: List[str] = []

    def fake_get_json(url: str, *, what: str) -> Any:
        json_urls.append(url)
        assert url.endswith("index.json")
        return index_payload

    recorder = FetchRecorder({}, default=VALID_FORM4_XML)
    monkeypatch.setattr(provider, "_get_json", fake_get_json)
    monkeypatch.setattr(provider, "_get_text", recorder)

    txns = provider.get_insider_transactions("TSTC", cik=999_999, submissions=submissions)

    assert len(json_urls) == 1
    assert recorder.urls and recorder.urls[0].endswith("wk-form4_123.xml")
    assert len(txns) == 5


# ---------------------------------------------------------------------------
# Transient-failure honesty: no cache poisoning, no false "no insider activity"
# ---------------------------------------------------------------------------


def test_rate_limited_form4_fetch_aborts_and_does_not_poison_cache(
    provider: SecEdgarProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 429/403 during the Form 4 loop propagates and caches nothing.

    Regression: RateLimited (a ProviderError subclass) used to be swallowed by
    the per-filing isolation, the empty list was cached for ``cache_ttl_seconds``
    and a later *healthy* scan was served the poisoned cache — suppressing real
    insider evidence and asserting a false absence for 12 hours.
    """
    today = _dt.date.today()
    rows = [("4", today - _dt.timedelta(days=7), "0001-26-000001", "form4.xml")]
    submissions = build_submissions(rows)

    def throttled(url: str, *, what: str) -> str:
        raise RateLimited("SEC rate limit hit fetching Form 4", provider="sec_edgar")

    monkeypatch.setattr(provider, "_get_text", throttled)
    with pytest.raises(RateLimited):
        provider.get_insider_transactions("TSTC", cik=999_999, submissions=submissions)

    # A later call with a healthy SEC must actually refetch (nothing cached).
    recorder = FetchRecorder({"0001-26-000001": VALID_FORM4_XML})
    monkeypatch.setattr(provider, "_get_text", recorder)
    txns = provider.get_insider_transactions("TSTC", cik=999_999, submissions=submissions)
    assert len(recorder.urls) == 1  # the filing was fetched fresh, not cache-served
    assert len(txns) == 5


def test_transient_fetch_failure_returns_partial_but_skips_cache_write(
    provider: SecEdgarProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A network-style ProviderError skips the filing AND the cache write."""
    today = _dt.date.today()
    rows = [
        ("4", today - _dt.timedelta(days=10), "0001-26-000001", "form4.xml"),
        ("4", today - _dt.timedelta(days=20), "0001-26-000002", "form4.xml"),
    ]
    submissions = build_submissions(rows)

    flaky_urls: List[str] = []

    def flaky(url: str, *, what: str) -> str:
        flaky_urls.append(url)
        if "000126000001" in url:  # first accession, no-dash form in the URL
            raise ProviderError("network error fetching Form 4", provider="sec_edgar")
        return VALID_FORM4_XML

    monkeypatch.setattr(provider, "_get_text", flaky)
    partial = provider.get_insider_transactions("TSTC", cik=999_999, submissions=submissions)
    assert len(flaky_urls) == 2  # both filings were attempted
    assert len(partial) == 5  # only the healthy filing's transactions

    # The partial result must NOT have been cached: a healthy retry refetches
    # both filings and recovers the full picture.
    recorder = FetchRecorder({}, default=VALID_FORM4_XML)
    monkeypatch.setattr(provider, "_get_text", recorder)
    full = provider.get_insider_transactions("TSTC", cik=999_999, submissions=submissions)
    assert len(recorder.urls) == 2  # cache miss -> both filings fetched fresh
    assert len(full) == 10


def test_all_transient_failures_raise_instead_of_claiming_absence(
    provider: SecEdgarProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If every selected filing failed transiently, raise — never return []."""
    today = _dt.date.today()
    rows = [
        ("4", today - _dt.timedelta(days=10), "0001-26-000001", "form4.xml"),
        ("4", today - _dt.timedelta(days=20), "0001-26-000002", "form4.xml"),
    ]
    submissions = build_submissions(rows)

    def down(url: str, *, what: str) -> str:
        raise ProviderError("HTTP 502 fetching Form 4", provider="sec_edgar")

    monkeypatch.setattr(provider, "_get_text", down)
    with pytest.raises(ProviderError, match="unknown, not absent"):
        provider.get_insider_transactions("TSTC", cik=999_999, submissions=submissions)

    # And nothing was cached: the next healthy call fetches both filings.
    recorder = FetchRecorder({}, default=VALID_FORM4_XML)
    monkeypatch.setattr(provider, "_get_text", recorder)
    txns = provider.get_insider_transactions("TSTC", cik=999_999, submissions=submissions)
    assert len(recorder.urls) == 2
    assert len(txns) == 10


def test_deterministic_parse_failures_still_cache(
    provider: SecEdgarProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed XML is a stable gap — the (partial) result stays cacheable."""
    today = _dt.date.today()
    rows = [
        ("4", today - _dt.timedelta(days=10), "0001-26-000001", "form4.xml"),
        ("4", today - _dt.timedelta(days=20), "0001-26-000002", "form4.xml"),
    ]
    submissions = build_submissions(rows)
    recorder = FetchRecorder(
        {
            "0001-26-000001": MALFORMED_XML,  # deterministic parse failure
            "0001-26-000002": VALID_FORM4_XML,
        }
    )
    monkeypatch.setattr(provider, "_get_text", recorder)

    first = provider.get_insider_transactions("TSTC", cik=999_999, submissions=submissions)
    second = provider.get_insider_transactions("TSTC", cik=999_999, submissions=submissions)

    assert len(recorder.urls) == 2  # second call was served from the disk cache
    assert [t.model_dump() for t in second] == [t.model_dump() for t in first]
    assert len(first) == 5


def test_get_security_data_never_claims_absence_when_rate_limited(
    provider: SecEdgarProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under a throttle, the warning says *unavailable* — never a false absence."""
    today = _dt.date.today()
    rows = [("4", today - _dt.timedelta(days=7), "0001-26-000001", "form4.xml")]
    submissions = build_submissions(rows)

    monkeypatch.setattr(provider, "_resolve_cik", lambda t: (999_999, "Testco Industries Inc"))
    monkeypatch.setattr(provider, "_fetch_submissions", lambda cik: submissions)
    monkeypatch.setattr(
        provider, "_fetch_company_facts", lambda cik: {"facts": {"us-gaap": {}}}
    )

    def throttled(url: str, *, what: str) -> str:
        raise RateLimited("SEC rate limit hit fetching Form 4", provider="sec_edgar")

    monkeypatch.setattr(provider, "_get_text", throttled)
    data = provider.get_security_data("TSTC")

    assert data.insider_transactions == []
    assert any("insider transactions unavailable" in w for w in data.data_warnings)
    # The factually false claim must not appear anywhere.
    assert not any(
        "no Form 4 insider transactions" in w for w in data.data_warnings
    )


# ---------------------------------------------------------------------------
# Provider surface: capabilities + get_security_data population
# ---------------------------------------------------------------------------


def test_provider_advertises_insider_capability(provider: SecEdgarProvider) -> None:
    assert "insider" in provider.capabilities
    assert provider.supports("insider")


def test_get_security_data_populates_insider_transactions(
    provider: SecEdgarProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    today = _dt.date.today()
    rows = [("4", today - _dt.timedelta(days=7), "0001-26-000001", "form4.xml")]
    submissions = build_submissions(rows)

    monkeypatch.setattr(provider, "_resolve_cik", lambda t: (999_999, "Testco Industries Inc"))
    monkeypatch.setattr(provider, "_fetch_submissions", lambda cik: submissions)
    monkeypatch.setattr(
        provider, "_fetch_company_facts", lambda cik: {"facts": {"us-gaap": {}}}
    )
    recorder = FetchRecorder({"0001-26-000001": VALID_FORM4_XML})
    monkeypatch.setattr(provider, "_get_text", recorder)

    data = provider.get_security_data("TSTC")

    assert len(data.insider_transactions) == 5
    types = {t.transaction_type for t in data.insider_transactions}
    assert {"buy", "sell", "award", "exercise", "other:G"} == types
    assert all(t.insider_name == "Doe Jane" for t in data.insider_transactions)
