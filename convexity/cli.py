"""Command-line interface for Convexity — the evidence-driven research terminal.

Part of Convexity, an evidence-driven small-/micro-cap equity **research and
screening** tool. Nothing this CLI prints is a prediction, a recommendation or
investment advice. A scan aggregates many *independent* pieces of evidence per
company and surfaces the names where those independent signals agree; conviction
is asserted only when agreement is high, and missing data always lowers
confidence rather than being invented.

Commands
--------
``scan``       Run the full universe screen + analysis funnel with a live
               progress display, print a ranked table, optionally dump the
               :class:`~convexity.core.models.ScanResult` to JSON.
``analyze``    Deep-dive a single ticker: every category sub-score (score,
               confidence, rationale) plus the assembled narrative.
``universe``   Preview the eligible universe (count + a small sample).
``serve``      Launch the FastAPI app under uvicorn.
``version``    Print the installed package version.

The CLI is a thin presentation layer over :class:`convexity.pipeline.ScanPipeline`
and the shared contracts in :mod:`convexity.core`; it adds no scoring logic of its
own. It uses the central project logger, degrades gracefully when the optional
``rich`` rendering library is unavailable, handles ``Ctrl-C`` cleanly, and exits
non-zero on a hard error so it composes in shell pipelines and CI.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

import typer

from convexity.core.exceptions import ConvexityError
from convexity.core.logging import get_logger, set_level
from convexity.core.models import (
    CompanyAnalysis,
    ScanParams,
    ScanResult,
    SubScore,
)

_log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Optional rich rendering — degrade to plain text when rich is unavailable.    #
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - presentation only
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH = True
    _console = Console()
    _err_console = Console(stderr=True)
except Exception:  # pragma: no cover - rich is an optional nicety
    _RICH = False
    _console = None  # type: ignore[assignment]
    _err_console = None  # type: ignore[assignment]


# A standing reminder printed on user-facing surfaces so the framing is never lost.
_DISCLAIMER = (
    "Research & screening tool — NOT a predictor and NOT investment advice. "
    "Signals are evidence to investigate, never a recommendation to act."
)


app = typer.Typer(
    name="convexity",
    help=(
        "Convexity — evidence-driven small-/micro-cap equity research & screening. "
        "A research tool, not a predictor and not investment advice."
    ),
    add_completion=False,
    no_args_is_help=True,
)


# --------------------------------------------------------------------------- #
# Small output helpers (rich when available, plain print otherwise).           #
# --------------------------------------------------------------------------- #
def _out(message: str = "") -> None:
    """Write a line to stdout (via rich console when available)."""
    if _RICH and _console is not None:
        _console.print(message)
    else:
        print(message)


def _err(message: str) -> None:
    """Write a line to stderr (via rich console when available)."""
    if _RICH and _err_console is not None:
        _err_console.print(message)
    else:
        print(message, file=sys.stderr)


def _rule(title: str) -> None:
    """Print a section rule/header."""
    if _RICH and _console is not None:
        _console.rule(title)
    else:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)


def _fmt_money(value: Optional[float]) -> str:
    """Render a market cap as a compact human-readable string (or ``n/a``)."""
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:  # NaN
        return "n/a"
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= threshold:
            return f"${v / threshold:,.2f}{suffix}"
    return f"${v:,.0f}"


def _fmt_pct(value: Optional[float]) -> str:
    """Render a 0..1 fraction as a percentage string (or ``n/a``)."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_score(value: Optional[float]) -> str:
    """Render a 0..100 score to one decimal place (or ``n/a``)."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"


def _one_line(text: str, limit: int = 96) -> str:
    """Collapse whitespace and truncate ``text`` to a single readable line."""
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: max(0, limit - 1)].rstrip() + "…"


# --------------------------------------------------------------------------- #
# Pipeline construction (lazy: importing this module must stay side-effect free)#
# --------------------------------------------------------------------------- #
def _build_pipeline() -> Any:
    """Construct the default :class:`ScanPipeline`, surfacing a clean hard error.

    Importing and wiring the pipeline can fail if optional data dependencies are
    missing; we convert that into a :class:`typer.Exit` with a non-zero code and a
    readable message rather than dumping a raw traceback on the user.
    """
    try:
        from convexity.pipeline import ScanPipeline

        return ScanPipeline()
    except Exception as exc:  # pragma: no cover - environment/dependency dependent
        _err(f"[error] Could not initialise the analysis pipeline: {exc}")
        raise typer.Exit(code=1) from exc


# --------------------------------------------------------------------------- #
# Live progress display                                                        #
# --------------------------------------------------------------------------- #
class _ProgressDisplay:
    """A live, stage-aware progress sink passed to :meth:`ScanPipeline.scan`.

    Renders the current stage and a ``done/total`` counter. When ``rich`` is
    available it updates a single live panel in place; otherwise it prints terse,
    de-duplicated stage transitions so a piped/non-TTY run stays readable. The
    callback never raises — the pipeline already swallows progress errors, and we
    keep this side defensive too.
    """

    _STAGE_TITLES = {
        "universe": "Building universe",
        "fetch": "Fetching data",
        "analyze": "Analysing",
        "rank": "Ranking",
        "explain": "Explaining top results",
    }

    def __init__(self) -> None:
        self._live: Any = None
        self._last_plain_key: Optional[str] = None

    def __enter__(self) -> _ProgressDisplay:
        if _RICH and _console is not None:
            self._live = Live(
                self._render("universe", 0, 1, "Starting scan…"),
                console=_console,
                refresh_per_second=12,
                transient=True,
            )
            self._live.__enter__()
        else:
            _out("Scanning… (research tool — not advice)")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._live is not None:
            try:
                self._live.__exit__(*exc)
            except Exception:  # pragma: no cover - defensive teardown
                pass
            self._live = None

    def _render(self, stage: str, done: int, total: int, message: str) -> Any:
        title = self._STAGE_TITLES.get(stage, stage.title())
        counter = f"{done}/{total}" if total else f"{done}"
        body = f"[bold]{title}[/bold]  [cyan]{counter}[/cyan]\n{message}"
        return Panel(body, title="Convexity scan", border_style="cyan")

    def __call__(self, stage: str, done: int, total: int, message: str) -> None:
        try:
            if self._live is not None:
                self._live.update(self._render(stage, done, total, message))
                return
            # Plain mode: only emit on a stage change or completion to avoid noise.
            key = f"{stage}:{done}:{total}"
            stage_key = f"{stage}:{total}"
            if total and done >= total:
                _out(f"  - {self._STAGE_TITLES.get(stage, stage)}: {message}")
            elif self._last_plain_key != stage_key:
                _out(f"  - {self._STAGE_TITLES.get(stage, stage)}…")
            self._last_plain_key = stage_key
            _ = key
        except Exception:  # pragma: no cover - progress must never break a scan
            pass


# --------------------------------------------------------------------------- #
# Rendering: ranked table & single-company detail                             #
# --------------------------------------------------------------------------- #
def _print_ranked_table(result: ScanResult) -> None:
    """Print the best-first ranked table for a completed scan."""
    rows = result.top if result.top else result.all_ranked
    if not rows:
        _out("No companies were ranked. This reflects data availability, not a market view.")
        return

    if _RICH and _console is not None:
        table = Table(
            title=f"Top {len(rows)} of {result.analyzed_count} analysed "
            f"(universe {result.universe_size}, {result.error_count} fetch error(s))",
            header_style="bold",
            expand=False,
        )
        table.add_column("#", justify="right", style="dim")
        table.add_column("Ticker", style="bold cyan")
        table.add_column("Name", overflow="fold", max_width=28)
        table.add_column("Mkt Cap", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Convict.", justify="right")
        table.add_column("One-line thesis", overflow="fold", max_width=46)
        for idx, c in enumerate(rows, start=1):
            table.add_row(
                str(c.rank if c.rank is not None else idx),
                c.ticker,
                c.name or "",
                _fmt_money(c.market_cap),
                _fmt_score(c.composite_score),
                _fmt_pct(c.conviction_confidence),
                _one_line(c.thesis, 60) if c.thesis else "—",
            )
        _console.print(table)
    else:
        header = (
            f"{'#':>3}  {'TICKER':<8} {'NAME':<26} {'MKT CAP':>10} "
            f"{'SCORE':>6} {'CONV':>5}  THESIS"
        )
        _out(header)
        _out("-" * len(header))
        for idx, c in enumerate(rows, start=1):
            _out(
                f"{(c.rank if c.rank is not None else idx):>3}  "
                f"{c.ticker:<8} {(c.name or '')[:26]:<26} "
                f"{_fmt_money(c.market_cap):>10} {_fmt_score(c.composite_score):>6} "
                f"{_fmt_pct(c.conviction_confidence):>5}  "
                f"{_one_line(c.thesis, 60) if c.thesis else '—'}"
            )

    for note in result.notes:
        _log.info("scan note: %s", note)


def _print_subscore(sub: SubScore) -> None:
    """Print one category sub-score: score, confidence, coverage, rationale."""
    label = sub.category.value.replace("_", " ").title()
    flags = f"  [{', '.join(sub.flags)}]" if sub.flags else ""
    if _RICH and _console is not None:
        head = Text()
        head.append(f"{label:<18}", style="bold")
        head.append(f"score {_fmt_score(sub.score):>5}  ", style="green")
        head.append(f"conf {_fmt_pct(sub.confidence):>4}  ", style="yellow")
        head.append(f"cover {_fmt_pct(sub.data_coverage):>4}", style="magenta")
        if flags:
            head.append(flags, style="red")
        _console.print(head)
        _console.print(f"    {_one_line(sub.rationale, 110)}", style="dim")
    else:
        _out(
            f"{label:<18} score {_fmt_score(sub.score):>5}  "
            f"conf {_fmt_pct(sub.confidence):>4}  cover {_fmt_pct(sub.data_coverage):>4}{flags}"
        )
        _out(f"    {_one_line(sub.rationale, 110)}")


def _print_bullets(title: str, items: List[str]) -> None:
    """Print a titled bullet list (skipped silently when empty)."""
    if not items:
        return
    _out("")
    if _RICH and _console is not None:
        _console.print(f"[bold]{title}[/bold]")
    else:
        _out(title)
    for item in items:
        _out(f"  • {_one_line(item, 110)}")


def _print_company_detail(analysis: CompanyAnalysis) -> None:
    """Print the full single-company analysis: sub-scores + narrative."""
    cap = _fmt_money(analysis.market_cap)
    sector = analysis.sector or "—"
    industry = analysis.industry or "—"
    _rule(f"{analysis.ticker} — {analysis.name}")
    _out(
        f"Composite {_fmt_score(analysis.composite_score)} / 100   "
        f"Conviction {_fmt_pct(analysis.conviction_confidence)}   "
        f"Signal agreement {_fmt_pct(analysis.signal_agreement)}"
    )
    _out(f"Market cap {cap}   Sector {sector}   Industry {industry}")
    _out("")

    if analysis.thesis:
        _print_bullets("Thesis", [analysis.thesis])

    _out("")
    if _RICH and _console is not None:
        _console.print("[bold]Category sub-scores[/bold]")
    else:
        _out("Category sub-scores")
    if not analysis.subscores:
        _out("  (no category was scored — insufficient data)")
    for sub in analysis.subscores:
        _print_subscore(sub)

    _print_bullets("Bull case", analysis.bull_case)
    _print_bullets("Bear case", analysis.bear_case)
    _print_bullets("Catalysts to watch", analysis.catalysts)
    _print_bullets("Principal risks", analysis.principal_risks)
    _print_bullets("Monitoring checklist", analysis.monitoring_checklist)

    if analysis.confidence_explanation:
        _print_bullets("Why this confidence", [analysis.confidence_explanation])

    _out("")
    _out(_DISCLAIMER)


# --------------------------------------------------------------------------- #
# Commands                                                                     #
# --------------------------------------------------------------------------- #
@app.callback()
def _main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable DEBUG-level logging."
    ),
) -> None:
    """Convexity research terminal — global options applied before any command."""
    if verbose:
        set_level("DEBUG")


@app.command()
def scan(
    min_cap: float = typer.Option(
        50_000_000.0, "--min-cap", help="Minimum market cap (USD) for the screen."
    ),
    max_cap: float = typer.Option(
        2_000_000_000.0, "--max-cap", help="Maximum market cap (USD) for the screen."
    ),
    min_dollar_volume: float = typer.Option(
        200_000.0,
        "--min-dollar-volume",
        help="Minimum average daily dollar volume (liquidity floor).",
    ),
    top_n: int = typer.Option(
        5, "--top-n", help="How many top-ranked names to explain and display."
    ),
    universe_limit: Optional[int] = typer.Option(
        None,
        "--universe-limit",
        help="Cap how many symbols are quoted/screened (faster, less coverage).",
    ),
    exclude_sector: List[str] = typer.Option(
        [],
        "--exclude-sector",
        help="Sector to exclude (repeatable, case-insensitive).",
    ),
    json_out: Optional[str] = typer.Option(
        None,
        "--json",
        metavar="OUTPATH",
        help="Write the full ScanResult as JSON to this path.",
    ),
) -> None:
    """Run a full research scan and print the ranked results.

    Builds the eligible universe, fetches data per ticker, runs every analyzer,
    ranks the cohort and explains the top ``--top-n`` names — surfacing where many
    *independent* signals agree. This is screening evidence to investigate, never a
    recommendation. Use ``--json`` to persist the complete machine-readable result.
    """
    if min_cap > max_cap:
        _err("[error] --min-cap must be <= --max-cap.")
        raise typer.Exit(code=2)
    if top_n < 0:
        _err("[error] --top-n must be >= 0.")
        raise typer.Exit(code=2)

    params = ScanParams(
        min_market_cap=min_cap,
        max_market_cap=max_cap,
        min_avg_dollar_volume=min_dollar_volume,
        exclude_sectors=list(exclude_sector),
        top_n=top_n,
        universe_limit=universe_limit,
    )

    pipeline = _build_pipeline()

    try:
        with _ProgressDisplay() as progress:
            result = pipeline.scan(params, progress=progress)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _err("\n[aborted] Scan interrupted by user.")
        raise typer.Exit(code=130)
    except ConvexityError as exc:
        _err(f"[error] Scan failed: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        _err(f"[error] Unexpected error during scan: {exc}")
        raise typer.Exit(code=1) from exc

    _out("")
    _print_ranked_table(result)

    if json_out:
        try:
            _write_json(json_out, result)
        except Exception as exc:
            _err(f"[error] Could not write JSON to {json_out}: {exc}")
            raise typer.Exit(code=1) from exc
        _out(f"\nWrote ScanResult JSON to {json_out}")

    _out("")
    _out(_DISCLAIMER)


@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="The ticker symbol to analyse."),
    json_out: Optional[str] = typer.Option(
        None,
        "--json",
        metavar="OUTPATH",
        help="Write the full CompanyAnalysis as JSON to this path.",
    ),
) -> None:
    """Deep-dive a single ``TICKER``: every sub-score plus the full narrative.

    Fetches the security, runs every analyzer against a peerless context, scores
    and explains it, then prints each category's score / confidence / rationale and
    the assembled thesis, bull/bear cases, catalysts, risks and monitoring list.
    One name in isolation carries less conviction than a name confirmed across a
    full universe scan — this view is for investigation, not a verdict.
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        _err("[error] A non-empty ticker symbol is required.")
        raise typer.Exit(code=2)

    pipeline = _build_pipeline()

    try:
        analysis = pipeline.analyze_one(symbol)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _err("\n[aborted] Analysis interrupted by user.")
        raise typer.Exit(code=130)
    except ConvexityError as exc:
        _err(f"[error] Could not analyse {symbol}: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        _err(f"[error] Unexpected error analysing {symbol}: {exc}")
        raise typer.Exit(code=1) from exc

    _print_company_detail(analysis)

    if json_out:
        try:
            _write_json(json_out, analysis)
        except Exception as exc:
            _err(f"[error] Could not write JSON to {json_out}: {exc}")
            raise typer.Exit(code=1) from exc
        _out(f"\nWrote CompanyAnalysis JSON to {json_out}")


@app.command()
def universe(
    min_cap: float = typer.Option(
        50_000_000.0, "--min-cap", help="Minimum market cap (USD) for the screen."
    ),
    max_cap: float = typer.Option(
        2_000_000_000.0, "--max-cap", help="Maximum market cap (USD) for the screen."
    ),
    min_dollar_volume: float = typer.Option(
        200_000.0, "--min-dollar-volume", help="Minimum average daily dollar volume."
    ),
    universe_limit: Optional[int] = typer.Option(
        None, "--universe-limit", help="Cap how many symbols are quoted/screened."
    ),
    sample: int = typer.Option(
        20, "--sample", help="How many example tickers to preview."
    ),
) -> None:
    """Preview the eligible universe: how many names survive the screen, and a sample.

    Builds (and, where possible, screens) the universe without running any analysis
    so you can sanity-check the cap/liquidity bands before committing to a full
    scan. Falls back to the bundled curated seed list when the network is
    unavailable — a convenience list, not a claim of completeness.
    """
    params = ScanParams(
        min_market_cap=min_cap,
        max_market_cap=max_cap,
        min_avg_dollar_volume=min_dollar_volume,
        universe_limit=universe_limit,
    )

    try:
        from convexity.core.config import get_settings
        from convexity.data import universe as universe_mod

        settings = get_settings()
        try:
            from convexity.data.aggregator import get_default_provider

            provider: Any = get_default_provider(settings)
        except Exception as exc:  # pragma: no cover - provider optional
            _log.warning("no data provider available for universe screen: %s", exc)
            provider = None

        tickers = universe_mod.build_universe_or_seed(
            params,
            provider,
            user_agent=settings.sec_user_agent,
            timeout=settings.request_timeout,
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _err("\n[aborted] Universe build interrupted by user.")
        raise typer.Exit(code=130)
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        _err(f"[error] Could not build the universe: {exc}")
        raise typer.Exit(code=1) from exc

    _rule("Eligible universe preview")
    _out(f"Eligible tickers: {len(tickers)}")
    if tickers:
        n = max(0, min(sample, len(tickers)))
        preview = ", ".join(tickers[:n])
        _out(f"Sample ({n}): {preview}")
    else:
        _out("No eligible tickers were produced (data availability, not a market view).")
    _out("")
    _out(_DISCLAIMER)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host for the API."),
    port: int = typer.Option(8000, "--port", help="Bind port for the API."),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload (development only)."
    ),
) -> None:
    """Launch the FastAPI app (``convexity.api.app:app``) under uvicorn.

    Serves the research results over HTTP. Requires the optional ``uvicorn`` server
    and the API app module to be importable; a missing dependency exits non-zero
    with a readable message rather than a traceback.
    """
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - optional server dependency
        _err(
            "[error] uvicorn is not installed. Install the server extras "
            f"(pip install 'uvicorn[standard]') to use 'serve'. ({exc})"
        )
        raise typer.Exit(code=1) from exc

    _out(f"Serving Convexity API on http://{host}:{port}  (research tool — not advice)")
    try:
        uvicorn.run("convexity.api.app:app", host=host, port=port, reload=reload)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _err("\n[stopped] Server interrupted by user.")
        raise typer.Exit(code=130)
    except Exception as exc:  # pragma: no cover - server/runtime dependent
        _err(f"[error] Could not start the API server: {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def version() -> None:
    """Print the installed Convexity package version."""
    _out(f"Convexity {_package_version()}")


# --------------------------------------------------------------------------- #
# Shared command helpers                                                       #
# --------------------------------------------------------------------------- #
def _package_version() -> str:
    """Resolve the installed package version, falling back gracefully."""
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _v

        try:
            return _v("convexity")
        except PackageNotFoundError:
            return "0.1.0 (uninstalled)"
    except Exception:  # pragma: no cover - very old importlib edge
        return "unknown"


def _model_to_jsonable(model: Any) -> Dict[str, Any]:
    """Convert a pydantic v2 model to a JSON-serialisable dict.

    Prefers pydantic's ``model_dump(mode="json")`` so dates/datetimes/enums are
    rendered as JSON-native primitives; tolerates older method names defensively.
    """
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "dict"):  # pragma: no cover - pydantic v1 fallback
        return model.dict()
    raise TypeError(f"object of type {type(model).__name__} is not serialisable")


def _write_json(path: str, model: Any) -> None:
    """Serialise a pydantic model to ``path`` as pretty, UTF-8 JSON."""
    payload = _model_to_jsonable(model)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")


def main() -> None:
    """Console-script entry point wrapper with a clean ``Ctrl-C`` exit."""
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _err("\n[aborted] Interrupted by user.")
        raise SystemExit(130)


__all__ = ["app", "main"]


if __name__ == "__main__":  # pragma: no cover - module executed directly
    main()
