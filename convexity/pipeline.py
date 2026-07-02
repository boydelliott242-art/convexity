"""End-to-end scan pipeline — universe -> screen -> fetch -> analyze -> rank -> explain.

This module wires together every other seam of Convexity into a single, runnable
research scan. It is the concrete implementation of
:class:`~convexity.core.contracts.PipelineProtocol`.

Part of Convexity, an evidence-driven equity **research and screening** tool. The
pipeline is **not** a predictor and **not** investment advice. It assembles many
*independent* pieces of evidence per company, scores each one, and surfaces the
names where many of those independent signals agree — never asserting certainty or
guaranteed returns. Missing data is recorded and lowers confidence; it is never
fabricated.

The scan stages
---------------
1. **Universe** — :func:`convexity.data.universe.build_universe_or_seed` enumerates
   the eligible US small-/micro-cap common stocks, honouring ``universe_limit``
   for fast iterations and screening by market cap + average dollar volume, with a
   bundled seed-list fallback so a scan always has something to analyse.
2. **Post-fetch screen enforcement** — the scan band is re-enforced on the
   *fetched* data, which is authoritative and fresher than any pre-screen:
   companies whose fetched ``market_cap`` falls outside
   ``[params.min_market_cap, params.max_market_cap]`` are dropped and counted in
   a note; companies whose market cap is *unknown* are kept with a recorded
   ``data_warning`` (honesty over tidiness — we never silently drop on absent
   data, and conviction already reflects the gap). Any ``params.exclude_sectors``
   are likewise dropped here (sector is a fetched field), recorded as a note.
3. **Fetch** — each surviving ticker's :class:`SecurityData` is assembled via the
   composite provider on a *bounded* :class:`~concurrent.futures.ThreadPoolExecutor`.
   Every per-ticker fetch is wrapped in a ``try/except`` that catches
   :class:`~convexity.core.exceptions.ConvexityError` (and any stray exception) so a
   single bad ticker can **never** abort the scan: the error is counted and noted.
4. **Context** — an :class:`~convexity.core.contracts.AnalysisContext` is built once
   from the successfully-fetched cohort: ``peer_stats`` grouped by sector and
   ``universe_stats`` across the whole cohort, so analyzers can score relative to
   peers rather than on absolute thresholds.
5. **Analyze** — importing :mod:`convexity.analysis` self-registers all analyzers;
   every registered analyzer is run for every company. A failing analyzer degrades
   to a neutral, low-confidence sub-score rather than dropping the company.
6. **Rank** — the ranking engine folds each company's sub-scores into a composite,
   a separate conviction figure and a deterministic universe rank.
7. **Explain** — the explainability engine populates the narrative for the top
   ``params.top_n`` ranked companies (the full ranking is still returned).

Determinism & safety
---------------------
The scan is deterministic given the same fetched data and analyzers: ordering,
context statistics and scoring are pure functions of their inputs. The only
non-determinism is the wall-clock ``generated_at`` / ``elapsed_seconds`` stamps and
network availability, neither of which changes the *ordering* of results. Importing
this module has no side effects beyond importing its dependencies, so it is
import-safe.
"""

from __future__ import annotations

import datetime as _dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from convexity.core.config import DEFAULT_CATEGORY_WEIGHTS, Settings, get_settings
from convexity.core.contracts import (
    AnalysisContext,
    Analyzer,
    DataProvider,
    ExplainabilityEngine,
    RankingEngine,
)
from convexity.core.exceptions import ConvexityError, DataUnavailable
from convexity.core.logging import get_logger
from convexity.core.models import (
    CompanyAnalysis,
    ScanParams,
    ScanResult,
    ScoreCategory,
    SecurityData,
    SubScore,
)
from convexity.core.registry import get_analyzers

_log = get_logger(__name__)

# How many concurrent per-ticker fetches to run. Bounded deliberately: the upstream
# providers (SEC EDGAR especially) are rate-sensitive, and an unbounded pool would
# both hammer them and make failures harder to reason about. Small and predictable
# keeps the scan polite and its behaviour reproducible.
_MAX_FETCH_WORKERS: int = 8

# Progress stage labels (stable strings a UI/CLI can switch on).
_STAGE_UNIVERSE = "universe"
_STAGE_FETCH = "fetch"
_STAGE_ANALYZE = "analyze"
_STAGE_RANK = "rank"
_STAGE_EXPLAIN = "explain"


# A progress callback receives ``(stage, done, total, message)``.
ProgressFn = Callable[[str, int, int, str], None]


class ScanPipeline:
    """Concrete :class:`~convexity.core.contracts.PipelineProtocol` implementation.

    Composes a data provider, a ranking engine and an explainability engine into a
    single :meth:`scan` that runs the full research funnel, plus a convenience
    :meth:`analyze_one` for a single ticker. All collaborators are injectable so the
    pipeline can be unit-tested with fakes; when omitted, the production defaults are
    used:

    * provider — :func:`convexity.data.aggregator.get_default_provider` (a
      :class:`~convexity.data.aggregator.CompositeProvider` over every available
      source);
    * ranking — :class:`~convexity.ranking.engine.DefaultRankingEngine`;
    * explainability — :class:`~convexity.ranking.explain.DefaultExplainabilityEngine`.

    The pipeline holds no mutable per-scan state on the instance, so a single
    instance may be reused across scans.
    """

    def __init__(
        self,
        provider: Optional[DataProvider] = None,
        ranking_engine: Optional[RankingEngine] = None,
        explain_engine: Optional[ExplainabilityEngine] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Build a pipeline, defaulting any collaborator that is not supplied.

        Args:
            provider: Data source used to fetch per-ticker :class:`SecurityData` and
                to screen the universe. Defaults to the composite default provider.
            ranking_engine: Engine that folds sub-scores into a ranked
                :class:`CompanyAnalysis`. Defaults to
                :class:`~convexity.ranking.engine.DefaultRankingEngine`.
            explain_engine: Engine that populates narrative for the top results.
                Defaults to
                :class:`~convexity.ranking.explain.DefaultExplainabilityEngine`.
            settings: Active :class:`~convexity.core.config.Settings`. Defaults to
                the process-wide cached settings via
                :func:`~convexity.core.config.get_settings`.
        """
        self._settings: Settings = settings if settings is not None else get_settings()

        if provider is not None:
            self._provider: DataProvider = provider
        else:
            # Imported lazily so importing the pipeline module never triggers
            # provider discovery / network setup as a side effect.
            from convexity.data.aggregator import get_default_provider

            self._provider = get_default_provider(self._settings)

        if ranking_engine is not None:
            self._ranking: RankingEngine = ranking_engine
        else:
            from convexity.ranking.engine import DefaultRankingEngine

            self._ranking = DefaultRankingEngine()

        if explain_engine is not None:
            self._explain: ExplainabilityEngine = explain_engine
        else:
            from convexity.ranking.explain import DefaultExplainabilityEngine

            self._explain = DefaultExplainabilityEngine()

    # ------------------------------------------------------------------ #
    # Public API: full scan                                              #
    # ------------------------------------------------------------------ #
    def scan(
        self,
        params: ScanParams,
        progress: Optional[ProgressFn] = None,
    ) -> ScanResult:
        """Run a full research scan and return the assembled :class:`ScanResult`.

        Executes the seven-stage funnel described in the module docstring. Every
        per-ticker fetch is isolated so one bad ticker only increments
        ``error_count`` and appends a note; the scan as a whole survives. When
        ``progress`` is supplied it is invoked as ``progress(stage, done, total,
        message)`` at each stage so a caller can render progress.

        Args:
            params: The screen + analysis parameters (cap band, liquidity floor,
                sector exclusions, ``top_n``, ``universe_limit``).
            progress: Optional callback ``(stage, done, total, message)`` invoked as
                the scan advances. Any exception it raises is swallowed so a faulty
                progress sink can never abort the scan.

        Returns:
            A fully-populated :class:`ScanResult` with counts, timings, the ranked
            companies, the explained ``top`` slice, the category weights used and a
            list of human-readable ``notes`` recording any gaps or exclusions.
        """
        start = time.monotonic()
        notes: List[str] = []
        weights = dict(DEFAULT_CATEGORY_WEIGHTS)

        # --- Stage 1: build & screen the universe ---------------------------
        self._emit(progress, _STAGE_UNIVERSE, 0, 1, "Building eligible universe…")
        universe = self._build_universe(params, notes)
        universe_size = len(universe)
        self._emit(
            progress,
            _STAGE_UNIVERSE,
            1,
            1,
            f"Universe screened to {universe_size} candidate ticker(s).",
        )

        # --- Stage 2: fetch SecurityData per ticker (bounded, isolated) -----
        fetched, error_count, fetch_notes = self._fetch_all(universe, progress)
        notes.extend(fetch_notes)

        # --- Stage 3: post-fetch screen enforcement (cap band + sector) -------
        # The fetched data is authoritative (fresher than any pre-fetch screen),
        # so the scan band is re-enforced here on what was actually fetched.
        in_band, band_excluded, unknown_cap = self._apply_cap_band(fetched, params)
        if band_excluded:
            notes.append(
                f"{band_excluded} name(s) excluded post-fetch: outside cap band "
                f"[${params.min_market_cap:,.0f}, ${params.max_market_cap:,.0f}]."
            )
        if unknown_cap:
            notes.append(
                f"{unknown_cap} name(s) kept with unknown market cap; cap-band "
                "eligibility could not be verified (recorded as a data warning; "
                "confidence reflects the gap rather than silently dropping)."
            )

        kept, excluded_count = self._apply_sector_exclusions(in_band, params)
        if excluded_count:
            notes.append(
                f"Excluded {excluded_count} company(ies) by sector filter "
                f"({', '.join(sorted(params.exclude_sectors))})."
            )
        screened_count = len(kept)

        if not kept:
            # Nothing survived to analyse — return an honest, empty-but-valid result.
            notes.append(
                "No companies survived fetching and screening; the scan produced no "
                "rankings. This reflects data availability, not a market view."
            )
            elapsed = time.monotonic() - start
            return ScanResult(
                generated_at=_dt.datetime.now(_dt.timezone.utc),
                params=params,
                universe_size=universe_size,
                screened_count=0,
                analyzed_count=0,
                error_count=error_count,
                top=[],
                all_ranked=[],
                category_weights={c.value: w for c, w in weights.items()},
                elapsed_seconds=elapsed,
                notes=notes,
            )

        # --- Stage 4: build comparative context once over the cohort --------
        ctx = self._build_context(kept)

        # --- Stage 5: analyze every kept company with every analyzer --------
        analyzers = self._load_analyzers(notes)
        analyses = self._analyze_all(kept, analyzers, weights, ctx, progress)
        analyzed_count = len(analyses)

        # --- Stage 6: rank the universe -------------------------------------
        self._emit(progress, _STAGE_RANK, 0, 1, "Ranking analysed companies…")
        ranked = self._ranking.rank(analyses, params)
        self._emit(
            progress, _STAGE_RANK, 1, 1, f"Ranked {len(ranked)} company(ies)."
        )

        # --- Stage 7: explain the top_n -------------------------------------
        top = self._explain_top(ranked, kept, params, progress)

        elapsed = time.monotonic() - start
        notes.append(
            f"Scan complete: {universe_size} screened candidate(s), "
            f"{screened_count} fetched & in-scope, {analyzed_count} analysed, "
            f"{error_count} fetch error(s), in {elapsed:.1f}s."
        )

        return ScanResult(
            generated_at=_dt.datetime.now(_dt.timezone.utc),
            params=params,
            universe_size=universe_size,
            screened_count=screened_count,
            analyzed_count=analyzed_count,
            error_count=error_count,
            top=top,
            all_ranked=ranked,
            category_weights={c.value: w for c, w in weights.items()},
            elapsed_seconds=elapsed,
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    # Public API: single-ticker analysis                                 #
    # ------------------------------------------------------------------ #
    def analyze_one(self, ticker: str) -> CompanyAnalysis:
        """Fetch, analyze, rank-of-one and explain a single ``ticker``.

        A convenience path mirroring :meth:`scan` for one security: it fetches the
        :class:`SecurityData`, runs every registered analyzer against an empty
        (peerless) :class:`AnalysisContext`, scores and ranks the lone company, and
        populates its narrative. Useful for drilling into one name without running a
        full universe scan.

        Args:
            ticker: The security symbol to analyse (case-insensitive).

        Returns:
            A fully-explained :class:`CompanyAnalysis` for ``ticker`` (its ``rank``
            is ``1`` since it is the only company considered).

        Raises:
            DataUnavailable: If no provider can supply any data for ``ticker``; the
                single-ticker path surfaces the gap rather than silently returning an
                empty analysis (callers wanting a survivable batch should use
                :meth:`scan`).
        """
        symbol = (ticker or "").strip().upper()
        if not symbol:
            raise DataUnavailable("empty ticker symbol", ticker=ticker)

        data = self._provider.get_security_data(symbol)

        ctx = AnalysisContext(
            peer_stats=None,
            universe_stats=None,
            config=self._settings,
            extras={},
        )
        weights = dict(DEFAULT_CATEGORY_WEIGHTS)
        analyzers = self._load_analyzers(notes=None)

        subscores = self._run_analyzers(data, analyzers, ctx)
        analysis = self._ranking.score_company(data, subscores, weights)
        ranked = self._ranking.rank([analysis], ScanParams())
        result = ranked[0] if ranked else analysis
        return self._explain.explain(result, data)

    # ------------------------------------------------------------------ #
    # Stage 1 helpers: universe                                          #
    # ------------------------------------------------------------------ #
    def _build_universe(self, params: ScanParams, notes: List[str]) -> List[str]:
        """Build & screen the eligible universe, honouring ``universe_limit``.

        Delegates to :func:`convexity.data.universe.build_universe_or_seed`, which
        screens the live listed universe by cap + liquidity off the configured
        provider and falls back to the bundled seed list when the network or quotes
        are unavailable. Any unexpected failure degrades to an empty universe with a
        recorded note rather than aborting the scan.
        """
        from convexity.data import universe as universe_mod

        try:
            tickers = universe_mod.build_universe_or_seed(
                params,
                self._provider,
                user_agent=self._settings.sec_user_agent,
                timeout=self._settings.request_timeout,
            )
        except Exception as exc:  # pragma: no cover - defensive top-level guard
            _log.error("universe construction failed; scanning empty universe: %s", exc)
            notes.append(f"Universe construction failed ({exc}); no candidates screened.")
            return []

        # De-duplicate defensively while preserving the deterministic order the
        # universe builder already established (so limiting stays reproducible).
        seen: set = set()
        ordered: List[str] = []
        for raw in tickers:
            sym = (raw or "").strip().upper()
            if not sym or sym in seen:
                continue
            seen.add(sym)
            ordered.append(sym)

        if not ordered:
            notes.append("Universe screen produced no eligible tickers.")
        return ordered

    # ------------------------------------------------------------------ #
    # Stage 2 helpers: fetch                                             #
    # ------------------------------------------------------------------ #
    def _fetch_all(
        self,
        tickers: List[str],
        progress: Optional[ProgressFn],
    ) -> Tuple[List[SecurityData], int, List[str]]:
        """Fetch :class:`SecurityData` for every ticker on a bounded thread pool.

        Each fetch is isolated: a :class:`ConvexityError` (the expected family for a
        thin or uncovered micro-cap) — or any other exception — is caught, counted,
        and turned into a note, so one failing ticker can never abort the scan.

        Args:
            tickers: The screened candidate symbols to fetch.
            progress: Optional progress callback for the ``fetch`` stage.

        Returns:
            A 3-tuple ``(fetched, error_count, notes)`` where ``fetched`` preserves
            the input ordering of the tickers that succeeded.
        """
        total = len(tickers)
        results: Dict[str, SecurityData] = {}
        errors: List[str] = []
        done = 0

        self._emit(progress, _STAGE_FETCH, 0, total, f"Fetching {total} ticker(s)…")
        if total == 0:
            return [], 0, []

        workers = max(1, min(_MAX_FETCH_WORKERS, total))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_ticker = {
                executor.submit(self._fetch_one_safe, ticker): ticker
                for ticker in tickers
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                done += 1
                # ``_fetch_one_safe`` never raises; it returns (data, error_message).
                data, error_message = future.result()
                if data is not None:
                    results[ticker] = data
                    self._emit(
                        progress, _STAGE_FETCH, done, total, f"Fetched {ticker}."
                    )
                else:
                    errors.append(f"{ticker}: {error_message}")
                    self._emit(
                        progress,
                        _STAGE_FETCH,
                        done,
                        total,
                        f"Skipped {ticker} ({error_message}).",
                    )

        # Preserve the deterministic input ordering for downstream stages.
        fetched = [results[t] for t in tickers if t in results]

        notes: List[str] = []
        if errors:
            notes.append(
                f"{len(errors)} ticker(s) could not be fetched and were skipped "
                "(data availability, not a market view)."
            )
            # Surface a bounded sample so the result stays readable but auditable.
            for detail in errors[:10]:
                notes.append(f"Fetch skipped — {detail}")
            if len(errors) > 10:
                notes.append(f"…and {len(errors) - 10} more fetch failure(s).")

        return fetched, len(errors), notes

    def _fetch_one_safe(self, ticker: str) -> Tuple[Optional[SecurityData], str]:
        """Fetch one ticker, converting every failure into a return value.

        This is the unit of work submitted to the thread pool. It NEVER raises: a
        :class:`ConvexityError` (expected gap or handled provider failure) or any
        stray exception is caught and reported as the second element of the tuple,
        guaranteeing the scan survives a single bad ticker.

        Args:
            ticker: The symbol to fetch.

        Returns:
            ``(data, "")`` on success, or ``(None, error_message)`` on any failure.
        """
        try:
            data = self._provider.get_security_data(ticker)
            if data is None:  # pragma: no cover - providers return data or raise
                return None, "provider returned no data object"
            return data, ""
        except ConvexityError as exc:
            _log.info("fetch skipped for %s: %s", ticker, exc)
            return None, str(exc)
        except Exception as exc:  # pragma: no cover - defensive catch-all
            _log.warning("unexpected error fetching %s: %s", ticker, exc)
            return None, f"unexpected error: {exc}"

    # ------------------------------------------------------------------ #
    # Stage 3 helpers: post-fetch screen enforcement                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_cap_band(
        fetched: List[SecurityData],
        params: ScanParams,
    ) -> Tuple[List[SecurityData], int, int]:
        """Enforce the scan's market-cap band on the *fetched* (authoritative) data.

        The pre-fetch universe screen works off quote-level estimates that can be
        stale or absent; the fetched :class:`SecurityData` is the freshest figure we
        have, so the band ``[params.min_market_cap, params.max_market_cap]`` is
        re-enforced here:

        * a company whose fetched ``market_cap`` lies outside the band is dropped
          and counted;
        * a company whose ``market_cap`` is ``None`` is **kept** — we never exclude
          on absent data — but a ``data_warning`` is appended to its
          :class:`SecurityData` so the gap is auditable and conviction (which
          already accounts for data coverage) reflects it.

        Args:
            fetched: The successfully-fetched securities, in deterministic order.
            params: The scan parameters carrying the cap band.

        Returns:
            A 3-tuple ``(kept, excluded_count, unknown_cap_count)`` where ``kept``
            preserves the input ordering.
        """
        kept: List[SecurityData] = []
        excluded = 0
        unknown = 0
        for data in fetched:
            cap = data.market_cap
            if cap is None:
                unknown += 1
                warning = (
                    "Market cap unknown after fetch; cap-band eligibility "
                    f"[${params.min_market_cap:,.0f}, ${params.max_market_cap:,.0f}] "
                    "could not be verified. Kept in the cohort; confidence reflects "
                    "this gap."
                )
                if warning not in data.data_warnings:
                    data.data_warnings.append(warning)
                kept.append(data)
                continue
            if cap < params.min_market_cap or cap > params.max_market_cap:
                excluded += 1
                continue
            kept.append(data)
        return kept, excluded, unknown

    @staticmethod
    def _apply_sector_exclusions(
        fetched: List[SecurityData],
        params: ScanParams,
    ) -> Tuple[List[SecurityData], int]:
        """Drop companies whose sector is in ``params.exclude_sectors``.

        Sector is a *fetched* field, so this filter runs after fetching. Matching is
        case-insensitive and trims whitespace; a company whose sector is unknown is
        conservatively *kept* (we never exclude on absent data). Returns the kept
        list (order preserved) and the count excluded.
        """
        if not params.exclude_sectors:
            return list(fetched), 0

        excluded_norm = {s.strip().lower() for s in params.exclude_sectors if s and s.strip()}
        if not excluded_norm:
            return list(fetched), 0

        kept: List[SecurityData] = []
        excluded = 0
        for data in fetched:
            sector = (data.sector or "").strip().lower()
            if sector and sector in excluded_norm:
                excluded += 1
                continue
            kept.append(data)
        return kept, excluded

    # ------------------------------------------------------------------ #
    # Stage 4 helpers: comparative context                              #
    # ------------------------------------------------------------------ #
    def _build_context(self, cohort: List[SecurityData]) -> AnalysisContext:
        """Build the comparative :class:`AnalysisContext` for the fetched cohort.

        Computes ``universe_stats`` across the whole cohort and ``peer_stats``
        grouped by sector, so analyzers can score a security relative to its peers
        and the wider screen rather than on absolute thresholds — important for
        micro-caps where "cheap"/"fast-growing" is sector-relative.

        The statistics are intentionally lightweight and metric-keyed (each metric
        maps to the sorted list of observed values plus simple summaries), matching
        the analyzer-defined shape documented on :class:`AnalysisContext`. Only
        positively-present values contribute; missing data simply does not appear.

        Args:
            cohort: The successfully-fetched securities to summarise.

        Returns:
            A populated :class:`AnalysisContext` carrying ``peer_stats`` (by sector),
            ``universe_stats`` and the active ``config``.
        """
        universe_stats = self._summarise_group(cohort)

        by_sector: Dict[str, List[SecurityData]] = {}
        for data in cohort:
            key = (data.sector or "Unknown").strip() or "Unknown"
            by_sector.setdefault(key, []).append(data)

        peer_stats: Dict[str, Any] = {
            "by_sector": {
                sector: self._summarise_group(members)
                for sector, members in by_sector.items()
            },
            # Convenience: each company's own sector cohort, keyed by ticker, so an
            # analyzer can fetch its peer distribution without re-grouping.
            "sector_of": {
                data.ticker: ((data.sector or "Unknown").strip() or "Unknown")
                for data in cohort
            },
        }

        return AnalysisContext(
            peer_stats=peer_stats,
            universe_stats=universe_stats,
            config=self._settings,
            extras={"cohort_size": len(cohort)},
        )

    @staticmethod
    def _summarise_group(group: List[SecurityData]) -> Dict[str, Any]:
        """Summarise a group of securities into metric-keyed distributions.

        For each tracked metric this collects every positively-present value across
        the group and records the sorted distribution alongside count / min / max /
        mean / median summaries. Only real, non-``None`` values are included — gaps
        are never imputed, so a sparse micro-cap cohort yields honestly sparse stats.

        Args:
            group: The securities to summarise (may be empty).

        Returns:
            A mapping ``{metric_name: {"values","count","min","max","mean","median"}}``
            plus a top-level ``"count"`` of group members. Metrics with no present
            values are omitted entirely.
        """
        # (metric_name, accessor) — accessors read already-present data only.
        metric_accessors: List[Tuple[str, Callable[[SecurityData], Optional[float]]]] = [
            ("market_cap", lambda d: d.market_cap),
            ("pe", lambda d: d.valuation.pe),
            ("ev_ebitda", lambda d: d.valuation.ev_ebitda),
            ("ev_sales", lambda d: d.valuation.ev_sales),
            ("p_fcf", lambda d: d.valuation.p_fcf),
            ("p_b", lambda d: d.valuation.p_b),
            ("peg", lambda d: d.valuation.peg),
            ("revenue", lambda d: _latest_field(d, "revenue")),
            ("net_income", lambda d: _latest_field(d, "net_income")),
            ("free_cash_flow", lambda d: _latest_field(d, "free_cash_flow")),
            ("operating_margin", lambda d: _latest_field(d, "operating_margin")),
            ("gross_margin", lambda d: _latest_field(d, "gross_margin")),
        ]

        summary: Dict[str, Any] = {"count": len(group)}
        for name, accessor in metric_accessors:
            values: List[float] = []
            for data in group:
                try:
                    raw = accessor(data)
                except Exception:  # pragma: no cover - defensive accessor guard
                    raw = None
                num = _coerce_finite(raw)
                if num is not None:
                    values.append(num)
            if not values:
                continue
            values.sort()
            n = len(values)
            summary[name] = {
                "values": values,
                "count": n,
                "min": values[0],
                "max": values[-1],
                "mean": sum(values) / n,
                "median": _median_sorted(values),
            }
        return summary

    # ------------------------------------------------------------------ #
    # Stage 5 helpers: analysis                                          #
    # ------------------------------------------------------------------ #
    def _load_analyzers(self, notes: Optional[List[str]]) -> List[Analyzer]:
        """Import the analysis package and instantiate every registered analyzer.

        Importing :mod:`convexity.analysis` triggers each analyzer's
        ``@register_analyzer`` decorator so the registry is populated. Every
        registered class is instantiated once (analyzers take no constructor args by
        contract); any that cannot be built is skipped with a recorded note rather
        than aborting the scan.

        Args:
            notes: Optional scan-level note list to append construction issues to
                (``None`` for the single-ticker path, which has no note channel).

        Returns:
            The list of instantiated analyzers (possibly empty if registration or
            construction failed for all of them).
        """
        try:
            import convexity.analysis  # noqa: F401  (import for registration side effect)
        except Exception as exc:  # pragma: no cover - defensive
            _log.error("could not import analysis package: %s", exc)
            if notes is not None:
                notes.append(f"Analyzer registration failed ({exc}); no categories scored.")
            return []

        analyzers: List[Analyzer] = []
        for cls in get_analyzers():
            try:
                analyzers.append(cls())  # type: ignore[call-arg]
            except Exception as exc:  # pragma: no cover - defensive per analyzer
                _log.warning(
                    "could not instantiate analyzer %s: %s",
                    getattr(cls, "__name__", repr(cls)),
                    exc,
                )
                if notes is not None:
                    notes.append(
                        f"Analyzer {getattr(cls, '__name__', cls)} unavailable ({exc})."
                    )
        if not analyzers and notes is not None:
            notes.append("No analyzers were available; companies were not scored.")
        return analyzers

    def _analyze_all(
        self,
        cohort: List[SecurityData],
        analyzers: List[Analyzer],
        weights: Dict[ScoreCategory, float],
        ctx: AnalysisContext,
        progress: Optional[ProgressFn],
    ) -> List[CompanyAnalysis]:
        """Run every analyzer over every company and score each one.

        Analyzing is CPU-light and pure (analyzers do no I/O by contract), so it runs
        serially in deterministic cohort order. Each company is scored via the
        ranking engine's :meth:`score_company` once its sub-scores are gathered.

        Args:
            cohort: The fetched, in-scope securities to analyse.
            analyzers: The instantiated analyzers to apply to each security.
            weights: The category-weighting map handed to ``score_company``.
            ctx: The shared comparative context.
            progress: Optional progress callback for the ``analyze`` stage.

        Returns:
            One :class:`CompanyAnalysis` per company (narrative fields still empty —
            the ranking engine leaves them for the explainability stage).
        """
        total = len(cohort)
        analyses: List[CompanyAnalysis] = []
        self._emit(progress, _STAGE_ANALYZE, 0, total, f"Analysing {total} company(ies)…")

        for index, data in enumerate(cohort, start=1):
            subscores = self._run_analyzers(data, analyzers, ctx)
            analysis = self._ranking.score_company(data, subscores, weights)
            analyses.append(analysis)
            self._emit(
                progress,
                _STAGE_ANALYZE,
                index,
                total,
                f"Analysed {data.ticker} ({index}/{total}).",
            )
        return analyses

    def _run_analyzers(
        self,
        data: SecurityData,
        analyzers: List[Analyzer],
        ctx: AnalysisContext,
    ) -> List[SubScore]:
        """Run every analyzer against one security, isolating per-analyzer failures.

        A misbehaving analyzer must not drop the whole company: any analyzer that
        raises is degraded to its own :meth:`Analyzer.neutral_subscore` (a neutral,
        low-confidence, ``MISSING_DATA``-flagged sub-score) so the category is still
        represented and the failure is auditable rather than silently absent.

        Args:
            data: The security to score.
            analyzers: The instantiated analyzers to apply.
            ctx: The shared comparative context.

        Returns:
            The list of sub-scores (one per analyzer that produced or fell back to
            one), in analyzer-registration order.
        """
        subscores: List[SubScore] = []
        for analyzer in analyzers:
            try:
                sub = analyzer.analyze(data, ctx)
            except Exception as exc:  # pragma: no cover - defensive per analyzer
                _log.warning(
                    "analyzer %s failed on %s: %s",
                    type(analyzer).__name__,
                    data.ticker,
                    exc,
                )
                sub = self._neutral_fallback(analyzer, exc)
            if sub is not None:
                subscores.append(sub)
        return subscores

    @staticmethod
    def _neutral_fallback(analyzer: Analyzer, exc: Exception) -> Optional[SubScore]:
        """Produce a neutral fallback sub-score for an analyzer that raised.

        Uses the analyzer's own :meth:`Analyzer.neutral_subscore` so the fallback
        carries the correct category and weight. If even that fails, returns
        ``None`` so the category is simply omitted (the scoring layer tolerates a
        missing category).
        """
        try:
            return analyzer.neutral_subscore(
                rationale=(
                    f"{type(analyzer).__name__} could not score this security "
                    f"({exc}); treated as neutral, low-confidence."
                ),
                extra_flags=["ANALYZER_ERROR"],
            )
        except Exception:  # pragma: no cover - defensive
            return None

    # ------------------------------------------------------------------ #
    # Stage 7 helpers: explanation                                       #
    # ------------------------------------------------------------------ #
    def _explain_top(
        self,
        ranked: List[CompanyAnalysis],
        cohort: List[SecurityData],
        params: ScanParams,
        progress: Optional[ProgressFn],
    ) -> List[CompanyAnalysis]:
        """Populate narrative for the top ``params.top_n`` ranked companies.

        The explainability engine restates already-attached evidence into prose. It
        is applied only to the top slice for efficiency (the full ranking is still
        returned by :meth:`scan`); each explained company is matched back to its
        originating :class:`SecurityData` by ticker. A failure to explain one company
        leaves its narrative empty rather than aborting the scan.

        Args:
            ranked: The full best-first ranking.
            cohort: The fetched securities (to recover each company's source data).
            params: The scan parameters (``top_n`` controls how many are explained).
            progress: Optional progress callback for the ``explain`` stage.

        Returns:
            The explained top slice (the same :class:`CompanyAnalysis` instances that
            also appear in ``ranked``, now with narrative populated).
        """
        top_n = max(0, int(params.top_n))
        top_slice = ranked[:top_n] if top_n else []
        total = len(top_slice)
        self._emit(progress, _STAGE_EXPLAIN, 0, total, f"Explaining top {total}…")

        data_by_ticker: Dict[str, SecurityData] = {d.ticker: d for d in cohort}
        for index, analysis in enumerate(top_slice, start=1):
            data = data_by_ticker.get(analysis.ticker)
            if data is None:  # pragma: no cover - top always comes from the cohort
                continue
            try:
                self._explain.explain(analysis, data)
            except Exception as exc:  # pragma: no cover - defensive per company
                _log.warning(
                    "explainability failed for %s: %s", analysis.ticker, exc
                )
            self._emit(
                progress,
                _STAGE_EXPLAIN,
                index,
                total,
                f"Explained {analysis.ticker} ({index}/{total}).",
            )
        return list(top_slice)

    # ------------------------------------------------------------------ #
    # Shared helpers                                                     #
    # ------------------------------------------------------------------ #
    def _load_analyzers_count(self) -> int:  # pragma: no cover - introspection aid
        """Return how many analyzers are currently registered (diagnostic helper)."""
        return len(self._load_analyzers(notes=None))

    @staticmethod
    def _emit(
        progress: Optional[ProgressFn],
        stage: str,
        done: int,
        total: int,
        message: str,
    ) -> None:
        """Invoke ``progress`` defensively (a faulty sink never aborts the scan)."""
        if progress is None:
            return
        try:
            progress(stage, done, total, message)
        except Exception as exc:  # pragma: no cover - defensive; progress is best-effort
            _log.debug("progress callback raised (ignored): %s", exc)


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------


def _coerce_finite(value: Any) -> Optional[float]:
    """Best-effort convert ``value`` to a finite float, else ``None``.

    Guards against ``None``, non-numeric values and NaN/inf so the cohort
    statistics never carry a non-finite figure into an analyzer's relative scoring.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _latest_field(data: SecurityData, field_name: str) -> Optional[float]:
    """Read a numeric field from a security's most recent fundamentals period.

    Returns ``None`` when there are no fundamentals or the field is absent on the
    latest period — never fabricating a value for a thin micro-cap.
    """
    latest = data.latest_fundamentals
    if latest is None:
        return None
    return getattr(latest, field_name, None)


def _median_sorted(values: List[float]) -> float:
    """Return the median of an already-sorted, non-empty list of floats."""
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


__all__ = ["ScanPipeline"]
