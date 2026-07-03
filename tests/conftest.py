"""Network-free, deterministic test harness for Convexity.

This module is the single source of *synthetic* data for the whole test suite. It
provides:

* :class:`FakeProvider` — a concrete
  :class:`~convexity.core.contracts.DataProvider` that returns rich, fully-invented
  :class:`~convexity.core.models.SecurityData` for six fictional small-/micro-caps.
  The companies span the spectrum the analyzers care about — clearly *strong*,
  clearly *weak*, and several *mixed* — each with enough daily price history for the
  technical/momentum analyzers, multiple years of fundamentals, and some news,
  filings, insider transactions and institutional holdings so the catalyst,
  ownership and risk analyzers have real inputs. **No ticker is real**; every value
  is fabricated for testing only.
* :func:`build_security` — a small builder that assembles a
  :class:`SecurityData` from keyword arguments while honouring the model's ordering
  contracts (fundamentals newest-first, price history oldest-first).
* pytest fixtures (:func:`fake_provider`, :func:`fake_tickers`,
  :func:`securities`, :func:`pipeline`, :func:`scan_params`) exposing those objects
  and a :class:`~convexity.pipeline.ScanPipeline` wired to ``FakeProvider`` with its
  universe overridden to exactly the fake tickers — so an end-to-end scan runs
  entirely offline and deterministically.

Determinism
-----------
Every price series here is generated from a fixed closed-form formula (a linear
trend plus a deterministic sine wiggle), never from a random source, so a scan over
this data yields byte-identical scores on every run. The only non-determinism in a
real scan is the wall-clock ``generated_at`` / ``elapsed_seconds`` stamps, which do
not affect ordering or scores.

Honesty framing
---------------
The synthetic companies exist to test that the pipeline behaves honestly — that a
genuinely strong, multi-signal-agreeing company outranks a weak one, that a thin
company's missing data lowers its confidence, and that nothing is fabricated by the
analyzers themselves. The data is invented; the *behaviour under test* is the real
contract.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Sequence
from typing import Dict, List, Optional, Set

import pytest

from convexity.core.contracts import DataProvider
from convexity.core.exceptions import DataUnavailable
from convexity.core.models import (
    CapTier,
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

# A fixed "as of" instant so nothing in the synthetic data depends on the wall clock.
AS_OF: _dt.datetime = _dt.datetime(2026, 1, 2, 0, 0, 0)


# ---------------------------------------------------------------------------
# Deterministic price-history generator
# ---------------------------------------------------------------------------


def make_price_history(
    *,
    start: _dt.date,
    days: int,
    start_price: float,
    end_price: float,
    volume: float,
    wiggle: float = 0.02,
) -> List[PriceBar]:
    """Build ``days`` of deterministic daily OHLCV bars, oldest-first.

    The close follows a straight line from ``start_price`` to ``end_price`` with a
    small, *fully deterministic* sine wiggle of relative amplitude ``wiggle`` layered
    on top — enough intrabar movement for the technical/momentum analyzers to read a
    trend and a range without any randomness. Bars are emitted on consecutive
    calendar days (weekends included; the analyzers treat the series as an ordered
    sequence, not a trading calendar).

    Args:
        start: Date of the first (oldest) bar.
        days: Number of bars to generate (must be >= 1).
        start_price: Close of the first bar.
        end_price: Close of the final bar (the trend's destination).
        volume: Baseline share volume applied to every bar (modulated deterministically).
        wiggle: Relative amplitude of the deterministic sine wiggle on the close.

    Returns:
        A list of :class:`PriceBar`, oldest-first, per the SecurityData contract.
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    bars: List[PriceBar] = []
    span = max(days - 1, 1)
    for i in range(days):
        frac = i / span
        trend = start_price + (end_price - start_price) * frac
        # Deterministic oscillation: sine over the window, no RNG involved.
        osc = math.sin(i * 0.30) * wiggle * trend
        close = max(0.01, trend + osc)
        # A plausible OHLC envelope around the close (also deterministic).
        high = close * (1.0 + 0.5 * wiggle)
        low = close * (1.0 - 0.5 * wiggle)
        open_ = close * (1.0 - 0.25 * wiggle * math.cos(i * 0.30))
        bar_volume = volume * (1.0 + 0.25 * math.sin(i * 0.17))
        bars.append(
            PriceBar(
                date=start + _dt.timedelta(days=i),
                open=round(max(0.01, open_), 4),
                high=round(max(close, high), 4),
                low=round(max(0.01, min(close, low)), 4),
                close=round(close, 4),
                adj_close=round(close, 4),
                volume=round(max(0.0, bar_volume), 2),
            )
        )
    return bars


# ---------------------------------------------------------------------------
# SecurityData builder helper
# ---------------------------------------------------------------------------


def build_security(
    *,
    ticker: str,
    name: str,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    cap_tier: Optional[CapTier] = None,
    valuation: Optional[ValuationSnapshot] = None,
    fundamentals: Optional[Sequence[FundamentalsPeriod]] = None,
    price_history: Optional[Sequence[PriceBar]] = None,
    news: Optional[Sequence[NewsItem]] = None,
    filings: Optional[Sequence[Filing]] = None,
    insider_transactions: Optional[Sequence[InsiderTransaction]] = None,
    institutional_holdings: Optional[Sequence[InstitutionalHolding]] = None,
    peers: Optional[Sequence[str]] = None,
    data_warnings: Optional[Sequence[str]] = None,
    as_of: _dt.datetime = AS_OF,
) -> SecurityData:
    """Assemble a :class:`SecurityData` from parts, honouring the ordering contracts.

    The caller passes ``fundamentals`` newest-first and ``price_history``
    oldest-first (the model contract); this helper sorts each defensively so a test
    that supplies them out of order still produces a contract-correct object. The
    ``data_sources`` list is stamped to ``["fake"]`` so a scan's provenance is
    explicit, and ``market_cap`` is taken from the supplied valuation snapshot.

    Args:
        ticker: Symbol (upper-cased).
        name: Company display name.
        sector / industry / cap_tier: Optional classification metadata.
        valuation: Valuation snapshot (defaults to an empty one).
        fundamentals: Periods, newest-first (sorted newest-first defensively).
        price_history: Daily bars, oldest-first (sorted oldest-first defensively).
        news / filings / insider_transactions / institutional_holdings / peers:
            Optional supporting evidence collections.
        data_warnings: Optional recorded data-gap notes (kept verbatim, never hidden).
        as_of: The synthetic "as of" instant.

    Returns:
        A fully-formed :class:`SecurityData` ready to feed the analyzers/pipeline.
    """
    funds = list(fundamentals or [])
    funds.sort(key=lambda p: p.period_end, reverse=True)  # newest-first contract
    bars = list(price_history or [])
    bars.sort(key=lambda b: b.date)  # oldest-first contract

    return SecurityData(
        ticker=ticker.strip().upper(),
        name=name,
        sector=sector,
        industry=industry,
        cap_tier=cap_tier,
        currency="USD",
        as_of=as_of,
        valuation=valuation if valuation is not None else ValuationSnapshot(),
        fundamentals=funds,
        price_history=bars,
        news=list(news or []),
        filings=list(filings or []),
        insider_transactions=list(insider_transactions or []),
        institutional_holdings=list(institutional_holdings or []),
        peers=list(peers or []),
        data_sources=["fake"],
        data_warnings=list(data_warnings or []),
    )


def _annual(
    year: int,
    *,
    revenue: Optional[float] = None,
    gross_profit: Optional[float] = None,
    operating_income: Optional[float] = None,
    net_income: Optional[float] = None,
    ebitda: Optional[float] = None,
    eps_diluted: Optional[float] = None,
    free_cash_flow: Optional[float] = None,
    operating_cash_flow: Optional[float] = None,
    capex: Optional[float] = None,
    total_assets: Optional[float] = None,
    total_debt: Optional[float] = None,
    cash_and_equivalents: Optional[float] = None,
    total_equity: Optional[float] = None,
    shares_diluted: Optional[float] = None,
    gross_margin: Optional[float] = None,
    operating_margin: Optional[float] = None,
    fcf_margin: Optional[float] = None,
    roic: Optional[float] = None,
    roe: Optional[float] = None,
    roa: Optional[float] = None,
    current_ratio: Optional[float] = None,
    quick_ratio: Optional[float] = None,
    debt_to_equity: Optional[float] = None,
    interest_coverage: Optional[float] = None,
) -> FundamentalsPeriod:
    """Build one annual :class:`FundamentalsPeriod` labelled ``FY<year>``."""
    return FundamentalsPeriod(
        period_end=_dt.date(year, 12, 31),
        period_label=f"FY{year}",
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
        cash_and_equivalents=cash_and_equivalents,
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


# ---------------------------------------------------------------------------
# The six synthetic companies
# ---------------------------------------------------------------------------
#
# Each builder returns a SecurityData. The set is deliberately spread:
#   STRONGCO  — strong across the board (cheap, growing, high-quality, healthy,
#               uptrending, insider buying, institutions accumulating).
#   WEAKCO    — weak across the board (rich, shrinking, loss-making, levered,
#               downtrending, insider selling, institutions trimming).
#   MIXEDONE  — cheap + healthy but shrinking (value-trap-ish): signals disagree.
#   MIXEDTWO  — fast-growing but expensive and cash-burning: signals disagree.
#   STEADYCO  — solid, unspectacular compounder: mildly positive, broad agreement.
#   THINCO    — a genuinely thin micro-cap: most data missing, to exercise the
#               missing-data / low-confidence honesty paths.

_PRICE_START = _dt.date(2024, 1, 1)
_PRICE_DAYS = 400  # ~13 months of daily bars: ample for technical/momentum windows


def _strongco() -> SecurityData:
    """A clearly strong micro-cap: cheap, growing, profitable, healthy, uptrending."""
    funds = [
        _annual(
            2025, revenue=240.0, gross_profit=132.0, operating_income=52.8,
            net_income=34.0, ebitda=60.0, eps_diluted=3.4, free_cash_flow=30.0,
            operating_cash_flow=40.0, capex=-10.0, total_assets=300.0,
            total_debt=20.0, cash_and_equivalents=60.0, total_equity=180.0,
            shares_diluted=10.0, gross_margin=0.55, operating_margin=0.22,
            fcf_margin=0.125, roic=0.24, roe=0.189, roa=0.113,
            current_ratio=2.8, quick_ratio=2.1, debt_to_equity=0.11,
            interest_coverage=26.0,
        ),
        _annual(
            2024, revenue=175.0, gross_profit=92.75, operating_income=31.5,
            net_income=22.0, ebitda=40.0, eps_diluted=2.2, free_cash_flow=20.0,
            operating_cash_flow=28.0, capex=-8.0, total_assets=240.0,
            total_debt=22.0, cash_and_equivalents=45.0, total_equity=150.0,
            shares_diluted=10.0, gross_margin=0.53, operating_margin=0.18,
            fcf_margin=0.114, roic=0.18, roe=0.147, roa=0.092,
            current_ratio=2.5, quick_ratio=1.9, debt_to_equity=0.147,
            interest_coverage=18.0,
        ),
        _annual(
            2023, revenue=130.0, gross_profit=65.0, operating_income=19.5,
            net_income=13.0, ebitda=26.0, eps_diluted=1.3, free_cash_flow=12.0,
            operating_cash_flow=18.0, capex=-6.0, total_assets=190.0,
            total_debt=24.0, cash_and_equivalents=30.0, total_equity=120.0,
            shares_diluted=10.0, gross_margin=0.50, operating_margin=0.15,
            fcf_margin=0.092, roic=0.14, roe=0.108, roa=0.068,
            current_ratio=2.2, quick_ratio=1.7, debt_to_equity=0.20,
            interest_coverage=12.0,
        ),
        _annual(
            2022, revenue=100.0, gross_profit=48.0, operating_income=12.0,
            net_income=8.0, ebitda=17.0, eps_diluted=0.8, free_cash_flow=7.0,
            operating_cash_flow=12.0, capex=-5.0, total_assets=150.0,
            total_debt=26.0, cash_and_equivalents=22.0, total_equity=95.0,
            shares_diluted=10.0, gross_margin=0.48, operating_margin=0.12,
            fcf_margin=0.07, roic=0.11, roe=0.084, roa=0.053,
            current_ratio=2.0, quick_ratio=1.5, debt_to_equity=0.27,
            interest_coverage=9.0,
        ),
    ]
    valuation = ValuationSnapshot(
        market_cap=420_000_000.0,
        enterprise_value=380_000_000.0,
        pe=12.0,
        forward_pe=10.0,
        ev_ebitda=6.3,
        ev_sales=1.6,
        p_fcf=14.0,
        p_b=2.3,
        p_s=1.75,
        peg=0.6,
    )
    bars = make_price_history(
        start=_PRICE_START, days=_PRICE_DAYS,
        start_price=22.0, end_price=42.0, volume=180_000.0, wiggle=0.018,
    )
    news = [
        NewsItem(
            published=_dt.datetime(2025, 12, 10, 14, 0),
            title="StrongCo raises full-year guidance after record quarter",
            source="Fake Newswire", url="https://example.test/strongco-guidance",
            summary="Management lifted revenue and margin guidance.", sentiment=0.7,
        ),
        NewsItem(
            published=_dt.datetime(2025, 11, 2, 9, 30),
            title="StrongCo announces new contract win",
            source="Fake Newswire", url="https://example.test/strongco-contract",
            summary="A multi-year supply agreement was signed.", sentiment=0.5,
        ),
    ]
    filings = [
        Filing(filed=_dt.date(2025, 12, 8), form_type="8-K",
               title="Updated guidance", url="https://example.test/strongco-8k",
               summary="Guidance raised."),
        Filing(filed=_dt.date(2025, 3, 1), form_type="10-K",
               title="Annual report FY2024", url="https://example.test/strongco-10k"),
    ]
    insiders = [
        InsiderTransaction(date=_dt.date(2025, 12, 12), insider_name="A. Founder",
                           role="CEO", transaction_type="buy", shares=50_000.0,
                           value=2_000_000.0),
        InsiderTransaction(date=_dt.date(2025, 11, 20), insider_name="B. Director",
                           role="Director", transaction_type="buy", shares=10_000.0,
                           value=400_000.0),
    ]
    holdings = [
        InstitutionalHolding(holder="Fake Capital", shares=900_000.0,
                             value=37_800_000.0, change_pct=0.15,
                             as_of=_dt.date(2025, 9, 30)),
        InstitutionalHolding(holder="Test Asset Mgmt", shares=600_000.0,
                             value=25_200_000.0, change_pct=0.08,
                             as_of=_dt.date(2025, 9, 30)),
    ]
    return build_security(
        ticker="STRONGCO", name="StrongCo Industries", sector="Technology",
        industry="Software", cap_tier=CapTier.SMALL, valuation=valuation,
        fundamentals=funds, price_history=bars, news=news, filings=filings,
        insider_transactions=insiders, institutional_holdings=holdings,
        peers=["STEADYCO", "MIXEDONE"],
    )


def _weakco() -> SecurityData:
    """A clearly weak micro-cap: rich, shrinking, loss-making, levered, downtrending."""
    funds = [
        _annual(
            2025, revenue=130.0, gross_profit=32.5, operating_income=-8.0,
            net_income=-14.0, ebitda=2.0, eps_diluted=-1.4, free_cash_flow=-12.0,
            operating_cash_flow=-4.0, capex=-8.0, total_assets=220.0,
            total_debt=160.0, cash_and_equivalents=8.0, total_equity=30.0,
            shares_diluted=10.0, gross_margin=0.25, operating_margin=-0.062,
            fcf_margin=-0.092, roic=-0.06, roe=-0.467, roa=-0.064,
            current_ratio=0.8, quick_ratio=0.5, debt_to_equity=5.33,
            interest_coverage=-0.6,
        ),
        _annual(
            2024, revenue=150.0, gross_profit=42.0, operating_income=6.0,
            net_income=2.0, ebitda=16.0, eps_diluted=0.2, free_cash_flow=1.0,
            operating_cash_flow=9.0, capex=-8.0, total_assets=240.0,
            total_debt=150.0, cash_and_equivalents=14.0, total_equity=48.0,
            shares_diluted=10.0, gross_margin=0.28, operating_margin=0.04,
            fcf_margin=0.0067, roic=0.02, roe=0.042, roa=0.0083,
            current_ratio=1.0, quick_ratio=0.7, debt_to_equity=3.125,
            interest_coverage=0.5,
        ),
        _annual(
            2023, revenue=170.0, gross_profit=51.0, operating_income=20.4,
            net_income=14.0, ebitda=30.0, eps_diluted=1.4, free_cash_flow=14.0,
            operating_cash_flow=22.0, capex=-8.0, total_assets=250.0,
            total_debt=140.0, cash_and_equivalents=22.0, total_equity=70.0,
            shares_diluted=10.0, gross_margin=0.30, operating_margin=0.12,
            fcf_margin=0.082, roic=0.10, roe=0.20, roa=0.056,
            current_ratio=1.3, quick_ratio=0.9, debt_to_equity=2.0,
            interest_coverage=2.5,
        ),
        _annual(
            2022, revenue=200.0, gross_profit=64.0, operating_income=30.0,
            net_income=24.0, ebitda=42.0, eps_diluted=2.4, free_cash_flow=24.0,
            operating_cash_flow=32.0, capex=-8.0, total_assets=270.0,
            total_debt=130.0, cash_and_equivalents=30.0, total_equity=95.0,
            shares_diluted=10.0, gross_margin=0.32, operating_margin=0.15,
            fcf_margin=0.12, roic=0.16, roe=0.253, roa=0.089,
            current_ratio=1.6, quick_ratio=1.1, debt_to_equity=1.37,
            interest_coverage=4.0,
        ),
    ]
    valuation = ValuationSnapshot(
        market_cap=300_000_000.0,
        enterprise_value=452_000_000.0,
        pe=None,           # loss-making: no meaningful P/E
        forward_pe=None,
        ev_ebitda=226.0,   # absurdly rich on collapsing EBITDA
        ev_sales=3.5,
        p_fcf=None,        # negative FCF: no meaningful P/FCF
        p_b=10.0,
        p_s=2.3,
        peg=None,
    )
    bars = make_price_history(
        start=_PRICE_START, days=_PRICE_DAYS,
        start_price=40.0, end_price=18.0, volume=120_000.0, wiggle=0.03,
    )
    news = [
        NewsItem(
            published=_dt.datetime(2025, 12, 5, 16, 0),
            title="WeakCo cuts guidance amid falling demand",
            source="Fake Newswire", url="https://example.test/weakco-cut",
            summary="Management lowered its outlook.", sentiment=-0.6,
        ),
        NewsItem(
            published=_dt.datetime(2025, 10, 18, 11, 0),
            title="WeakCo explores debt restructuring options",
            source="Fake Newswire", url="https://example.test/weakco-debt",
            summary="The company is in talks with lenders.", sentiment=-0.5,
        ),
    ]
    filings = [
        Filing(filed=_dt.date(2025, 12, 4), form_type="8-K",
               title="Guidance reduction", url="https://example.test/weakco-8k",
               summary="Outlook cut."),
    ]
    insiders = [
        InsiderTransaction(date=_dt.date(2025, 12, 1), insider_name="C. Officer",
                           role="CFO", transaction_type="sell", shares=80_000.0,
                           value=1_600_000.0),
        InsiderTransaction(date=_dt.date(2025, 11, 15), insider_name="D. Officer",
                           role="COO", transaction_type="sell", shares=40_000.0,
                           value=800_000.0),
    ]
    holdings = [
        InstitutionalHolding(holder="Fake Capital", shares=200_000.0,
                             value=4_000_000.0, change_pct=-0.30,
                             as_of=_dt.date(2025, 9, 30)),
    ]
    return build_security(
        ticker="WEAKCO", name="WeakCo Holdings", sector="Industrials",
        industry="Machinery", cap_tier=CapTier.SMALL, valuation=valuation,
        fundamentals=funds, price_history=bars, news=news, filings=filings,
        insider_transactions=insiders, institutional_holdings=holdings,
        peers=["MIXEDTWO"],
        data_warnings=["WEAKCO: P/E and P/FCF undefined on negative earnings/FCF."],
    )


def _mixedone() -> SecurityData:
    """Cheap + financially healthy but shrinking — a value-trap-ish mixed signal."""
    funds = [
        _annual(
            2025, revenue=88.0, gross_profit=33.4, operating_income=9.7,
            net_income=7.0, ebitda=14.0, eps_diluted=0.7, free_cash_flow=8.0,
            operating_cash_flow=12.0, capex=-4.0, total_assets=180.0,
            total_debt=15.0, cash_and_equivalents=50.0, total_equity=140.0,
            shares_diluted=10.0, gross_margin=0.38, operating_margin=0.11,
            fcf_margin=0.091, roic=0.06, roe=0.05, roa=0.039,
            current_ratio=3.5, quick_ratio=2.8, debt_to_equity=0.107,
            interest_coverage=20.0,
        ),
        _annual(
            2024, revenue=100.0, gross_profit=40.0, operating_income=13.0,
            net_income=10.0, ebitda=18.0, eps_diluted=1.0, free_cash_flow=11.0,
            operating_cash_flow=15.0, capex=-4.0, total_assets=185.0,
            total_debt=16.0, cash_and_equivalents=48.0, total_equity=138.0,
            shares_diluted=10.0, gross_margin=0.40, operating_margin=0.13,
            fcf_margin=0.11, roic=0.08, roe=0.072, roa=0.054,
            current_ratio=3.3, quick_ratio=2.6, debt_to_equity=0.116,
            interest_coverage=22.0,
        ),
        _annual(
            2023, revenue=112.0, gross_profit=47.0, operating_income=16.8,
            net_income=12.5, ebitda=22.0, eps_diluted=1.25, free_cash_flow=13.0,
            operating_cash_flow=17.0, capex=-4.0, total_assets=190.0,
            total_debt=18.0, cash_and_equivalents=46.0, total_equity=135.0,
            shares_diluted=10.0, gross_margin=0.42, operating_margin=0.15,
            fcf_margin=0.116, roic=0.10, roe=0.093, roa=0.066,
            current_ratio=3.0, quick_ratio=2.4, debt_to_equity=0.133,
            interest_coverage=24.0,
        ),
    ]
    valuation = ValuationSnapshot(
        market_cap=95_000_000.0,
        enterprise_value=60_000_000.0,
        pe=13.6,
        ev_ebitda=4.3,
        ev_sales=0.68,
        p_fcf=11.9,
        p_b=0.68,
        p_s=1.08,
        peg=None,  # shrinking: no positive growth for PEG
    )
    bars = make_price_history(
        start=_PRICE_START, days=_PRICE_DAYS,
        start_price=10.0, end_price=9.2, volume=90_000.0, wiggle=0.025,
    )
    news = [
        NewsItem(
            published=_dt.datetime(2025, 11, 10, 13, 0),
            title="MixedOne declares special dividend from cash hoard",
            source="Fake Newswire", url="https://example.test/mixedone-div",
            summary="A one-time return of capital.", sentiment=0.2,
        ),
    ]
    filings = [
        Filing(filed=_dt.date(2025, 11, 9), form_type="8-K",
               title="Special dividend", url="https://example.test/mixedone-8k"),
    ]
    insiders = [
        InsiderTransaction(date=_dt.date(2025, 10, 30), insider_name="E. Insider",
                           role="Director", transaction_type="buy", shares=5_000.0,
                           value=46_000.0),
    ]
    holdings = [
        InstitutionalHolding(holder="Value Partners (fake)", shares=300_000.0,
                             value=2_760_000.0, change_pct=0.0,
                             as_of=_dt.date(2025, 9, 30)),
    ]
    return build_security(
        ticker="MIXEDONE", name="MixedOne Capital", sector="Financials",
        industry="Asset Management", cap_tier=CapTier.MICRO, valuation=valuation,
        fundamentals=funds, price_history=bars, news=news, filings=filings,
        insider_transactions=insiders, institutional_holdings=holdings,
        peers=["STRONGCO"],
    )


def _mixedtwo() -> SecurityData:
    """Fast-growing but expensive and cash-burning — the opposite mixed signal."""
    funds = [
        _annual(
            2025, revenue=210.0, gross_profit=147.0, operating_income=-6.0,
            net_income=-9.0, ebitda=4.0, eps_diluted=-0.9, free_cash_flow=-18.0,
            operating_cash_flow=-6.0, capex=-12.0, total_assets=260.0,
            total_debt=30.0, cash_and_equivalents=80.0, total_equity=170.0,
            shares_diluted=10.0, gross_margin=0.70, operating_margin=-0.029,
            fcf_margin=-0.086, roic=-0.03, roe=-0.053, roa=-0.035,
            current_ratio=2.6, quick_ratio=2.2, debt_to_equity=0.176,
            interest_coverage=-3.0,
        ),
        _annual(
            2024, revenue=140.0, gross_profit=96.0, operating_income=-4.0,
            net_income=-6.0, ebitda=2.0, eps_diluted=-0.6, free_cash_flow=-14.0,
            operating_cash_flow=-4.0, capex=-10.0, total_assets=210.0,
            total_debt=28.0, cash_and_equivalents=70.0, total_equity=150.0,
            shares_diluted=10.0, gross_margin=0.686, operating_margin=-0.029,
            fcf_margin=-0.10, roic=-0.02, roe=-0.04, roa=-0.029,
            current_ratio=2.4, quick_ratio=2.0, debt_to_equity=0.187,
            interest_coverage=-2.0,
        ),
        _annual(
            2023, revenue=90.0, gross_profit=60.0, operating_income=-3.0,
            net_income=-5.0, ebitda=1.0, eps_diluted=-0.5, free_cash_flow=-10.0,
            operating_cash_flow=-2.0, capex=-8.0, total_assets=160.0,
            total_debt=25.0, cash_and_equivalents=55.0, total_equity=120.0,
            shares_diluted=10.0, gross_margin=0.667, operating_margin=-0.033,
            fcf_margin=-0.111, roic=-0.02, roe=-0.042, roa=-0.031,
            current_ratio=2.2, quick_ratio=1.9, debt_to_equity=0.208,
            interest_coverage=-1.5,
        ),
    ]
    valuation = ValuationSnapshot(
        market_cap=1_100_000_000.0,
        enterprise_value=1_050_000_000.0,
        pe=None,
        forward_pe=120.0,
        ev_ebitda=262.0,
        ev_sales=5.0,
        p_fcf=None,
        p_b=6.5,
        p_s=5.24,
        peg=None,
    )
    bars = make_price_history(
        start=_PRICE_START, days=_PRICE_DAYS,
        start_price=70.0, end_price=110.0, volume=260_000.0, wiggle=0.04,
    )
    news = [
        NewsItem(
            published=_dt.datetime(2025, 12, 1, 12, 0),
            title="MixedTwo posts 50% revenue growth, still unprofitable",
            source="Fake Newswire", url="https://example.test/mixedtwo-growth",
            summary="Top line surged but losses widened.", sentiment=0.1,
        ),
        NewsItem(
            published=_dt.datetime(2025, 9, 14, 8, 0),
            title="MixedTwo raises capital to fund expansion",
            source="Fake Newswire", url="https://example.test/mixedtwo-raise",
            summary="A secondary offering was completed.", sentiment=-0.1,
        ),
    ]
    filings = [
        Filing(filed=_dt.date(2025, 9, 13), form_type="S-1",
               title="Securities offering", url="https://example.test/mixedtwo-s1"),
    ]
    insiders = [
        InsiderTransaction(date=_dt.date(2025, 10, 5), insider_name="F. Founder",
                           role="CEO", transaction_type="buy", shares=20_000.0,
                           value=2_000_000.0),
    ]
    holdings = [
        InstitutionalHolding(holder="Growth Fund (fake)", shares=1_200_000.0,
                             value=132_000_000.0, change_pct=0.25,
                             as_of=_dt.date(2025, 9, 30)),
    ]
    return build_security(
        ticker="MIXEDTWO", name="MixedTwo Bioscience", sector="Healthcare",
        industry="Biotechnology", cap_tier=CapTier.SMALL, valuation=valuation,
        fundamentals=funds, price_history=bars, news=news, filings=filings,
        insider_transactions=insiders, institutional_holdings=holdings,
        peers=["WEAKCO"],
    )


def _steadyco() -> SecurityData:
    """A solid, unspectacular compounder: mildly positive, broad gentle agreement."""
    funds = [
        _annual(
            2025, revenue=160.0, gross_profit=64.0, operating_income=24.0,
            net_income=17.0, ebitda=30.0, eps_diluted=1.7, free_cash_flow=18.0,
            operating_cash_flow=24.0, capex=-6.0, total_assets=210.0,
            total_debt=40.0, cash_and_equivalents=35.0, total_equity=130.0,
            shares_diluted=10.0, gross_margin=0.40, operating_margin=0.15,
            fcf_margin=0.1125, roic=0.13, roe=0.131, roa=0.081,
            current_ratio=2.1, quick_ratio=1.6, debt_to_equity=0.31,
            interest_coverage=11.0,
        ),
        _annual(
            2024, revenue=150.0, gross_profit=59.0, operating_income=21.0,
            net_income=15.0, ebitda=27.0, eps_diluted=1.5, free_cash_flow=16.0,
            operating_cash_flow=22.0, capex=-6.0, total_assets=200.0,
            total_debt=42.0, cash_and_equivalents=30.0, total_equity=120.0,
            shares_diluted=10.0, gross_margin=0.393, operating_margin=0.14,
            fcf_margin=0.107, roic=0.12, roe=0.125, roa=0.075,
            current_ratio=2.0, quick_ratio=1.5, debt_to_equity=0.35,
            interest_coverage=10.0,
        ),
        _annual(
            2023, revenue=142.0, gross_profit=54.0, operating_income=18.5,
            net_income=13.0, ebitda=24.0, eps_diluted=1.3, free_cash_flow=14.0,
            operating_cash_flow=20.0, capex=-6.0, total_assets=195.0,
            total_debt=45.0, cash_and_equivalents=26.0, total_equity=110.0,
            shares_diluted=10.0, gross_margin=0.38, operating_margin=0.13,
            fcf_margin=0.099, roic=0.11, roe=0.118, roa=0.067,
            current_ratio=1.9, quick_ratio=1.4, debt_to_equity=0.41,
            interest_coverage=9.0,
        ),
    ]
    valuation = ValuationSnapshot(
        market_cap=270_000_000.0,
        enterprise_value=275_000_000.0,
        pe=15.9,
        forward_pe=14.5,
        ev_ebitda=9.2,
        ev_sales=1.72,
        p_fcf=15.0,
        p_b=2.08,
        p_s=1.69,
        peg=1.4,
    )
    bars = make_price_history(
        start=_PRICE_START, days=_PRICE_DAYS,
        start_price=24.0, end_price=27.0, volume=140_000.0, wiggle=0.015,
    )
    news = [
        NewsItem(
            published=_dt.datetime(2025, 11, 25, 10, 0),
            title="SteadyCo reports another quarter of steady growth",
            source="Fake Newswire", url="https://example.test/steadyco-q",
            summary="Results were in line with expectations.", sentiment=0.2,
        ),
    ]
    filings = [
        Filing(filed=_dt.date(2025, 2, 20), form_type="10-K",
               title="Annual report FY2024", url="https://example.test/steadyco-10k"),
    ]
    insiders = [
        InsiderTransaction(date=_dt.date(2025, 9, 5), insider_name="G. Director",
                           role="Director", transaction_type="buy", shares=3_000.0,
                           value=78_000.0),
    ]
    holdings = [
        InstitutionalHolding(holder="Index Fund (fake)", shares=500_000.0,
                             value=13_000_000.0, change_pct=0.02,
                             as_of=_dt.date(2025, 9, 30)),
    ]
    return build_security(
        ticker="STEADYCO", name="SteadyCo Manufacturing", sector="Consumer Staples",
        industry="Packaged Foods", cap_tier=CapTier.SMALL, valuation=valuation,
        fundamentals=funds, price_history=bars, news=news, filings=filings,
        insider_transactions=insiders, institutional_holdings=holdings,
        peers=["STRONGCO", "MIXEDONE"],
    )


def _thinco() -> SecurityData:
    """A genuinely thin micro-cap: most data missing, to exercise honesty paths.

    Only a single sparse fundamentals period, a short price history, no news,
    filings, insiders or institutions, and several recorded data warnings. The
    analyzers should lower their confidence and flag missing data rather than
    guessing — and the pipeline must still rank it without crashing.
    """
    funds = [
        _annual(2025, revenue=30.0, net_income=1.0),  # almost everything else None
    ]
    valuation = ValuationSnapshot(
        market_cap=55_000_000.0,
        # Almost no multiples available — the thin-data case.
        p_s=1.8,
    )
    bars = make_price_history(
        start=_dt.date(2025, 10, 1), days=40,  # short window only
        start_price=5.0, end_price=5.3, volume=15_000.0, wiggle=0.05,
    )
    return build_security(
        ticker="THINCO", name="ThinCo Micro", sector="Energy",
        industry="Oil & Gas Exploration", cap_tier=CapTier.MICRO,
        valuation=valuation, fundamentals=funds, price_history=bars,
        peers=[],
        data_warnings=[
            "THINCO: no analyst coverage; most valuation multiples unavailable.",
            "THINCO: only one fundamentals period available.",
            "THINCO: no institutional holdings or insider transactions on record.",
        ],
    )


# Ordered registry of the synthetic companies, keyed by ticker.
def _build_registry() -> Dict[str, SecurityData]:
    """Construct the full ticker -> SecurityData map of synthetic companies."""
    companies = [
        _strongco(),
        _weakco(),
        _mixedone(),
        _mixedtwo(),
        _steadyco(),
        _thinco(),
    ]
    return {c.ticker: c for c in companies}


# ---------------------------------------------------------------------------
# FakeProvider
# ---------------------------------------------------------------------------


class FakeProvider(DataProvider):
    """A deterministic, offline :class:`DataProvider` over six synthetic companies.

    The provider serves pre-built :class:`SecurityData` for a fixed set of invented
    tickers and raises :class:`~convexity.core.exceptions.DataUnavailable` for any
    unknown symbol (mirroring how a real provider reports a thin/uncovered name).
    It also exposes :meth:`get_quotes` and :meth:`get_universe` so it can stand in
    for the universe-screening price provider, returning in-band caps/liquidity for
    its own tickers — though the test pipeline normally overrides the universe stage
    to the fake tickers directly for full determinism.

    The provider performs **no I/O** and holds immutable per-ticker data, so it is
    safe to share across threads (the pipeline fetches on a thread pool).
    """

    def __init__(self, registry: Optional[Dict[str, SecurityData]] = None) -> None:
        """Build the provider over ``registry`` (defaults to the six synthetics)."""
        self._registry: Dict[str, SecurityData] = (
            dict(registry) if registry is not None else _build_registry()
        )

    @property
    def name(self) -> str:
        """Stable identifier recorded in ``SecurityData.data_sources``."""
        return "fake"

    @property
    def capabilities(self) -> Set[str]:
        """Everything the analyzers might ask for — this provider supplies it all."""
        return {
            "universe",
            "prices",
            "fundamentals",
            "valuation",
            "news",
            "filings",
            "insider",
            "institutional",
        }

    @property
    def tickers(self) -> List[str]:
        """The synthetic tickers this provider can serve, in a stable order."""
        return list(self._registry.keys())

    def get_universe(self, params: ScanParams) -> List[str]:
        """Return the synthetic tickers as the screening universe (honours limit).

        The cap/liquidity bands in ``params`` are not re-applied here — the
        synthetic caps already sit inside the default band — but ``universe_limit``
        is honoured so a test can shrink the universe deterministically.
        """
        tickers = self.tickers
        limit = params.universe_limit
        if limit is not None and limit >= 0:
            tickers = tickers[:limit]
        return tickers

    def get_quotes(self, tickers: Sequence[str]) -> Dict[str, Dict[str, float]]:
        """Return ``{ticker: {market_cap, avg_dollar_volume}}`` for screening.

        Lets :func:`convexity.data.universe.build_universe` screen the fake universe
        without any network. Only known tickers are returned; an unknown symbol is
        simply omitted (the screen treats a missing quote as "exclude", honestly).
        """
        out: Dict[str, Dict[str, float]] = {}
        for raw in tickers:
            sym = str(raw).strip().upper()
            data = self._registry.get(sym)
            if data is None:
                continue
            cap = data.market_cap or 0.0
            last_close = data.price_history[-1].close if data.price_history else 0.0
            avg_vol = (
                sum(b.volume for b in data.price_history) / len(data.price_history)
                if data.price_history
                else 0.0
            )
            out[sym] = {
                "market_cap": float(cap),
                "avg_dollar_volume": float(last_close * avg_vol),
            }
        return out

    def get_security_data(self, ticker: str) -> SecurityData:
        """Return the synthetic :class:`SecurityData` for ``ticker``.

        Raises:
            DataUnavailable: For an unknown symbol — the honest "we have nothing for
                this name" signal a real provider would give for an uncovered stock.
        """
        symbol = (ticker or "").strip().upper()
        data = self._registry.get(symbol)
        if data is None:
            raise DataUnavailable(
                f"FakeProvider has no synthetic data for {symbol!r}", ticker=symbol
            )
        # Return a deep copy so a downstream mutation can never leak between tests.
        return data.model_copy(deep=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_provider_cache(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
):
    """Point the process-wide provider cache at a per-test temporary directory.

    Providers memoise fetches through :func:`convexity.data.cache.get_cache`,
    which by default persists under ``Settings.data_dir`` (``./.convexity_data``).
    Left unpatched, one test's cached quotes/fetches would leak into another —
    and into *real* runs of the tool from this working tree. This fixture lazily
    substitutes a fresh, throwaway :class:`~convexity.data.cache.Cache` per test
    (created only if the test actually touches the cache) so every test starts
    cold and leaves nothing behind.
    """
    from convexity.core.config import Settings
    from convexity.data import cache as cache_mod

    created: List[cache_mod.Cache] = []

    def _test_cache() -> cache_mod.Cache:
        if not created:
            base = tmp_path_factory.mktemp("convexity-cache")
            created.append(cache_mod.Cache(Settings(data_dir=str(base))))
        return created[0]

    monkeypatch.setattr(cache_mod, "get_cache", _test_cache)
    yield
    for cache in created:
        cache.close()


@pytest.fixture()
def fake_provider() -> FakeProvider:
    """A fresh :class:`FakeProvider` over the six synthetic companies."""
    return FakeProvider()


@pytest.fixture()
def fake_tickers(fake_provider: FakeProvider) -> List[str]:
    """The list of synthetic tickers served by :func:`fake_provider`."""
    return fake_provider.tickers


@pytest.fixture()
def securities(fake_provider: FakeProvider) -> Dict[str, SecurityData]:
    """A ``{ticker: SecurityData}`` map of all six synthetic companies."""
    return {t: fake_provider.get_security_data(t) for t in fake_provider.tickers}


@pytest.fixture()
def scan_params() -> ScanParams:
    """Default scan parameters with ``top_n=3`` so several companies make the cut."""
    return ScanParams(top_n=3)


@pytest.fixture()
def pipeline(fake_provider: FakeProvider, monkeypatch: pytest.MonkeyPatch):
    """A :class:`ScanPipeline` wired to :class:`FakeProvider`, universe overridden.

    The provider supplies all per-ticker data, while the universe stage is
    monkeypatched to return exactly the fake tickers — so the end-to-end scan runs
    with no network and is fully deterministic. The real ranking and explainability
    engines are used (only the data source is faked), so the scan exercises the true
    aggregation, ranking and narrative code paths.
    """
    from convexity import pipeline as pipeline_mod
    from convexity.data import universe as universe_mod

    def _fake_universe(params, price_provider=None, **_kwargs):
        """Stand-in for ``build_universe_or_seed`` returning the fake tickers."""
        tickers = fake_provider.tickers
        limit = params.universe_limit
        if limit is not None and limit >= 0:
            tickers = tickers[:limit]
        return list(tickers)

    # The pipeline imports the universe module lazily inside ``_build_universe`` via
    # ``from convexity.data import universe as universe_mod`` and then calls
    # ``universe_mod.build_universe_or_seed(...)`` — so patching the function on the
    # module object is what the pipeline will actually call.
    monkeypatch.setattr(universe_mod, "build_universe_or_seed", _fake_universe)

    return pipeline_mod.ScanPipeline(provider=fake_provider)
