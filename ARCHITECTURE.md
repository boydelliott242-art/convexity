# Architecture

Convexity is a research and screening platform — **not a predictor and not investment
advice** (see [DISCLAIMER.md](DISCLAIMER.md)). Its architecture exists to make every score
*auditable*: each number traces to independent, sourced evidence, and conviction is only high
when many independent signals agree.

## Package map

```
convexity/
  core/                     Shared contracts imported by everything else.
    models.py               Pydantic v2 data models (the canonical shapes).
    contracts.py            DataProvider / Analyzer ABCs, AnalysisContext, ranking protocols.
    registry.py             @register_provider / @register_analyzer self-registration.
    config.py               Settings (pydantic-settings) + DEFAULT_CATEGORY_WEIGHTS.
    scoring.py              Pure numeric helpers (clamp, scale_to_score, combine_subscores, …).
    logging.py              get_logger(): one configured structured logger.
    exceptions.py           ConvexityError hierarchy.
  data/
    universe.py             Build the candidate ticker universe.
    providers/              Concrete DataProvider implementations (self-register).
    cache.py                On-disk caching (diskcache).
    aggregator.py           Merge providers -> one SecurityData per ticker.
  analysis/                 One Analyzer per ScoreCategory (value, growth, quality, …).
  ranking/
    engine.py               Combine sub-scores -> CompanyAnalysis, assign ranks.
    explain.py              Generate thesis / bull / bear / summaries / monitoring list.
  pipeline.py               Orchestrates the full scan.
  cli.py                    Typer entry point (console script `convexity`).
  api/                      FastAPI app (later phase).
tests/                      conftest.py, unit/, integration/.
```

## Data flow

```
                ScanParams
                    |
                    v
   (1) UNIVERSE   data/universe.py + providers.get_universe()
                    |  candidate tickers
                    v
   (2) SCREEN     pipeline applies ScanParams: market-cap band,
                  min average dollar volume, sector exclusions, universe_limit
                    |  screened tickers
                    v
   (3) FETCH      data/aggregator.py asks each DataProvider (by capability)
                  and merges into one SecurityData per ticker.
                  Missing data -> data_warnings (never fabricated).
                    |  SecurityData[]
                    v
   (4) ANALYZE    each registered Analyzer.analyze(data, AnalysisContext)
                  -> SubScore (score 0..100, confidence, weight, evidence,
                     flags, data_coverage). Peer/universe stats let analyzers
                     score relative to comparable companies.
                    |  SubScore[] per company
                    v
   (5) RANK       ranking/engine.py: combine_subscores() folds sub-scores into
                  a composite (confidence- & weight-weighted; RISK applied as a
                  dampener), computes signal_agreement and conviction_confidence,
                  sorts best-first and assigns integer ranks.
                    |  CompanyAnalysis[]
                    v
   (6) EXPLAIN    ranking/explain.py: thesis, bull_case, bear_case, catalysts,
                  principal_risks, valuation/fundamental/technical summaries,
                  confidence_explanation, monitoring_checklist.
                    |
                    v
                 ScanResult  ->  CLI / FastAPI / static frontend
```

## Design principles

- **Independence of evidence.** The twelve `ScoreCategory` values are chosen to be as
  uncorrelated as practical, so that agreement across them is genuinely informative rather
  than the same signal counted twice.
- **Honesty over completeness.** Every model field that may be unavailable is `Optional`.
  Missing inputs reduce `confidence` and `data_coverage` and are surfaced in `data_warnings`;
  they are never imputed with invented values.
- **Determinism.** `core/scoring.py` and the analyzers are pure: no wall-clock, no randomness.
  Identical inputs yield identical scores, which is what makes results reproducible.
- **Composability via contracts.** Providers, analyzers, ranking and explainability conform to
  the ABCs/protocols in `core/contracts.py`; the registry discovers implementations by import,
  so new sources or signals plug in without touching the pipeline.
- **RISK as a dampener.** RISK is not averaged into the composite mean; it tempers an otherwise
  attractive thesis when risk is elevated (see `combine_subscores`).
