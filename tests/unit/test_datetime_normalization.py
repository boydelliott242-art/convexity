"""Regression tests: datetimes from different providers must merge safely.

yfinance emits tz-naive UTC (``datetime.utcnow()``) while SEC EDGAR emits tz-aware
UTC (``datetime.now(timezone.utc)``). Before normalization, merging a security that
carried both sources raised ``TypeError: can't compare offset-naive and offset-aware
datetimes`` in the aggregator — silently dropping every multi-source ticker from a
scan. The model now coerces ``as_of`` and ``NewsItem.published`` to naive UTC on
construction so all downstream comparisons and sorts are consistent.
"""

from __future__ import annotations

import datetime as dt

from convexity.core.models import NewsItem, SecurityData
from convexity.data.aggregator import merge_security_data


def _sd(ticker: str, as_of: dt.datetime, news: list | None = None) -> SecurityData:
    return SecurityData(ticker=ticker, name=ticker, as_of=as_of, news=news or [])


def test_as_of_aware_is_normalized_to_naive() -> None:
    aware = dt.datetime(2026, 6, 30, 18, 0, tzinfo=dt.timezone.utc)
    sd = _sd("AAA", aware)
    assert sd.as_of.tzinfo is None
    # Same instant, just tz-stripped.
    assert sd.as_of == dt.datetime(2026, 6, 30, 18, 0)


def test_as_of_naive_is_left_unchanged() -> None:
    naive = dt.datetime(2026, 6, 30, 12, 0)
    assert _sd("BBB", naive).as_of == naive


def test_news_published_aware_is_normalized() -> None:
    aware = dt.datetime(2026, 6, 30, 15, 30, tzinfo=dt.timezone.utc)
    item = NewsItem(published=aware, title="t", source="s")
    assert item.published.tzinfo is None


def test_merge_across_aware_and_naive_sources_does_not_raise() -> None:
    # This is the exact shape that used to crash: one source aware, one naive.
    yfinance_like = _sd("CCC", dt.datetime(2026, 6, 29, 20, 0))  # naive (utcnow-style)
    sec_like = _sd("CCC", dt.datetime(2026, 6, 30, 20, 0, tzinfo=dt.timezone.utc))  # aware
    merged = merge_security_data(yfinance_like, sec_like)
    # The later observation (SEC, 6/30) wins, and everything stays naive.
    assert merged.as_of.tzinfo is None
    assert merged.as_of == dt.datetime(2026, 6, 30, 20, 0)


def test_merge_news_sort_across_mixed_tz_does_not_raise() -> None:
    naive_news = NewsItem(published=dt.datetime(2026, 6, 28, 9, 0), title="old", source="a")
    aware_news = NewsItem(
        published=dt.datetime(2026, 6, 30, 9, 0, tzinfo=dt.timezone.utc), title="new", source="b"
    )
    base = _sd("DDD", dt.datetime(2026, 6, 30, 0, 0), news=[naive_news])
    incoming = _sd("DDD", dt.datetime(2026, 6, 30, 0, 0), news=[aware_news])
    merged = merge_security_data(base, incoming)
    # Sorted newest-first without a TypeError; the aware item is the newest.
    assert merged.news[0].title == "new"
    assert all(n.published.tzinfo is None for n in merged.news)
