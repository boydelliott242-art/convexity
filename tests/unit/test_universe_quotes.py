"""Regression tests for the universe batched-quote screening path (BUG A).

The original defect: ``convexity.data.universe`` asked its price provider for
``get_quotes``/``batch_quotes``/``quotes``/``get_quote`` but no provider
implemented any of them, so the screen logged "cannot screen" and every scan
silently fell back to the bundled seed list. The fix gave
:class:`~convexity.data.aggregator.CompositeProvider` a real ``get_quotes``
that delegates to the first capable member. These tests pin that contract,
fully offline:

* the composite delegates to a member exposing ``get_quotes`` (and skips
  members that lack it);
* ``universe._call_batched_quotes`` discovers and uses a provider's
  ``get_quotes`` and ``universe._extract_cap_and_liquidity`` parses the
  returned dict keys into ``(market_cap, avg_dollar_volume)``;
* a provider with no recognised quote method yields ``{}`` without raising —
  the screen degrades, it never crashes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import pytest

from convexity.core.contracts import DataProvider
from convexity.core.exceptions import DataUnavailable
from convexity.core.models import SecurityData
from convexity.data.aggregator import CompositeProvider
from convexity.data.universe import (
    _call_batched_quotes,
    _extract_cap_and_liquidity,
)

# ---------------------------------------------------------------------------
# Tiny fake providers (no network, no I/O)
# ---------------------------------------------------------------------------


class _QuotingMember(DataProvider):
    """Fake member provider exposing a canned batched ``get_quotes``."""

    def __init__(self, name: str, quotes: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        self._name = name
        self._quotes = dict(quotes or {})
        self.calls: List[List[str]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> Set[str]:
        return {"prices"}

    def get_security_data(self, ticker: str) -> SecurityData:
        raise DataUnavailable("quotes-only fake member", ticker=ticker)

    def get_quotes(self, tickers: Sequence[str]) -> Dict[str, Dict[str, float]]:
        self.calls.append([str(t) for t in tickers])
        return {k: v for k, v in self._quotes.items() if k in {str(t).upper() for t in tickers}}


class _QuotelessMember(DataProvider):
    """Fake member provider with NO quote method of any recognised name."""

    @property
    def name(self) -> str:
        return "quoteless"

    @property
    def capabilities(self) -> Set[str]:
        return {"fundamentals"}

    def get_security_data(self, ticker: str) -> SecurityData:
        raise DataUnavailable("no data in this fake", ticker=ticker)


# ---------------------------------------------------------------------------
# (a) CompositeProvider.get_quotes delegates to a capable member
# ---------------------------------------------------------------------------


class TestCompositeDelegation:
    def test_composite_delegates_to_member_with_get_quotes(self) -> None:
        member = _QuotingMember(
            "screener",
            {"AAA": {"market_cap": 4.2e8, "avg_dollar_volume": 1.5e6}},
        )
        composite = CompositeProvider(providers=[_QuotelessMember(), member])

        quotes = composite.get_quotes(["AAA"])

        assert member.calls == [["AAA"]]  # the quoting member actually served the batch
        assert quotes == {"AAA": {"market_cap": 4.2e8, "avg_dollar_volume": 1.5e6}}

    def test_composite_uppercases_result_keys(self) -> None:
        class _LowercaseMember(_QuotingMember):
            def get_quotes(self, tickers: Sequence[str]) -> Dict[str, Dict[str, float]]:
                self.calls.append([str(t) for t in tickers])
                return {"bbb": {"market_cap": 9.0e7}}

        composite = CompositeProvider(providers=[_LowercaseMember("lower")])
        assert composite.get_quotes(["BBB"]) == {"BBB": {"market_cap": 9.0e7}}

    def test_composite_with_no_capable_member_returns_empty(self) -> None:
        composite = CompositeProvider(providers=[_QuotelessMember()])
        assert composite.get_quotes(["AAA"]) == {}


# ---------------------------------------------------------------------------
# (b) universe._call_batched_quotes + _extract_cap_and_liquidity
# ---------------------------------------------------------------------------


class TestBatchedQuoteDiscovery:
    def test_finds_and_uses_provider_get_quotes(self) -> None:
        provider = _QuotingMember(
            "direct",
            {
                "AAA": {"market_cap": 3.0e8, "avg_dollar_volume": 8.0e5},
                "BBB": {"market_cap": 6.0e7, "price": 4.0, "avg_volume": 50_000.0},
            },
        )

        quotes = _call_batched_quotes(provider, ["AAA", "BBB", "MISSING"])

        assert provider.calls == [["AAA", "BBB", "MISSING"]]
        assert set(quotes) == {"AAA", "BBB"}  # missing symbols simply absent

        # _extract parses the direct keys into (market_cap, avg_dollar_volume)…
        cap, adv = _extract_cap_and_liquidity(quotes["AAA"])
        assert cap == pytest.approx(3.0e8)
        assert adv == pytest.approx(8.0e5)

        # …and derives dollar volume from price x avg_volume when not direct.
        cap, adv = _extract_cap_and_liquidity(quotes["BBB"])
        assert cap == pytest.approx(6.0e7)
        assert adv == pytest.approx(4.0 * 50_000.0)

    def test_extract_returns_none_for_absent_fields(self) -> None:
        cap, adv = _extract_cap_and_liquidity({"price": 10.0})  # no cap, no volume
        assert cap is None
        assert adv is None  # never fabricated from price alone

    def test_composite_provider_is_usable_by_the_screen_helper(self) -> None:
        # Regression on the exact BUG A shape: the screen helper duck-types the
        # composite and must find its get_quotes (previously nothing matched).
        member = _QuotingMember("m", {"CCC": {"market_cap": 1.0e8, "avg_dollar_volume": 5.0e5}})
        composite = CompositeProvider(providers=[member])

        quotes = _call_batched_quotes(composite, ["CCC"])

        assert set(quotes) == {"CCC"}
        cap, adv = _extract_cap_and_liquidity(quotes["CCC"])
        assert cap == pytest.approx(1.0e8)
        assert adv == pytest.approx(5.0e5)


# ---------------------------------------------------------------------------
# (c) a provider with no quote method degrades to {} without raising
# ---------------------------------------------------------------------------


class TestNoQuoteMethodDegradation:
    def test_provider_without_quote_methods_returns_empty(self) -> None:
        result = _call_batched_quotes(_QuotelessMember(), ["AAA", "BBB"])
        assert result == {}  # logged, empty, and crucially: did not raise

    def test_object_with_nothing_at_all_returns_empty(self) -> None:
        result = _call_batched_quotes(object(), ["AAA"])
        assert result == {}
