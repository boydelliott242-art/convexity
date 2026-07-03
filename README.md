<div align="center">

# ◟ Convexity

**An evidence-driven research & screening platform for U.S. small- & micro-cap equities.**

*Rare, asymmetric ideas — surfaced only when many independent signals agree.*

![CI](https://img.shields.io/badge/CI-GitHub%20Actions%20ready-6e9bff)
![Python](https://img.shields.io/badge/python-3.9%2B-6e9bff)
![License](https://img.shields.io/badge/license-MIT-46e0c0)
![Tests](https://img.shields.io/badge/tests-480%20passing-3ed9a4)
![Ruff](https://img.shields.io/badge/ruff-clean-46e0c0)

**Live demo:** https://boydelliott242-art.github.io/convexity/ · **Repo:** https://github.com/boydelliott242-art/convexity

</div>

---

> [!IMPORTANT]
> **Convexity is a research and screening tool — not investment advice, and not a predictor.**
> No system can reliably forecast stock prices. Convexity's value is the *transparent
> aggregation of many independent pieces of evidence* into explainable 0–100 scores you can
> audit end to end. High conviction here means **many independent signals agree** — never a
> guarantee. Missing data lowers confidence rather than being fabricated. Do your own due
> diligence. See [DISCLAIMER.md](DISCLAIMER.md).

## What it does

Convexity searches the eligible universe of U.S. small- and micro-cap companies, screens it by
market cap and liquidity, and ranks the highest-conviction opportunities using **twelve
independent analyzers**. Every score traces back to the concrete data that produced it, and a
recommendation only rises to the top when *numerous distinct forms of evidence reinforce one
another*.

The output for each idea includes: ticker, company, industry, market cap, an investment
thesis, bull case, bear case, key catalysts, principal risks, valuation / fundamental /
technical summaries, a **confidence explanation** (that literally counts how many independent
categories agreed), and a falsifiable **monitoring checklist** of what would confirm or
invalidate the thesis.

## Highlights

- **12 independent analyzers**, each an auditable 0–100 sub-score with itemized evidence:
  Value · Growth · Quality · Financial Health · Technical · Momentum · Catalysts · Risk ·
  Management · Competitive · Ownership · Historical-Analog.
- **Conviction, not just score.** The ranking engine rewards *independent-signal agreement* and
  data coverage, and dampens on elevated risk — a high score alone is never enough.
- **Explainable by construction.** No opaque black box: every sub-score carries a rationale and
  a list of `Evidence` items citing the exact numbers and their bullish/bearish direction.
- **Runs out of the box.** Zero-key defaults via free data sources (Yahoo Finance + SEC EDGAR);
  optional premium providers (Financial Modeling Prep) unlock insider/institutional depth.
- **Zero-build dashboard.** A self-contained dark, responsive analyst dashboard (hand-built SVG
  charts, drill-down evidence, filtering/sorting, CSV/JSON export) served by the API — no Node.
- **Production shape.** FastAPI service, Typer CLI, 480 offline/deterministic tests, ruff, type
  hints throughout, Docker + docker-compose, GitHub Actions CI.

## Quickstart

```bash
git clone https://github.com/boydelliott242-art/convexity.git
cd convexity
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # or: make install
```

### Run a scan from the CLI

```bash
# Rank the top 5 ideas from a fast 60-name slice of the universe
convexity scan --universe-limit 60 --top-n 5

# Deep single-name analysis with full sub-score + evidence breakdown
convexity analyze AAPL

# Preview how large the eligible small/micro-cap universe is
convexity universe
```

### Launch the dashboard

```bash
convexity serve            # then open http://localhost:8000
# or: make serve
```

The dashboard also runs in **demo mode** straight from disk (no backend) against the bundled
`examples/sample_scan.json` — just open `frontend/index.html` through any static server.

## How the scoring works

```
 universe ─▶ screen (cap + liquidity) ─▶ fetch SecurityData ─▶ 12 analyzers
                                                                    │
                          ┌─────────────────────────────────────────┘
                          ▼
        each analyzer → SubScore { score 0–100, confidence, data_coverage,
                                   rationale, evidence[], flags[] }
                          │
                          ▼
     ranking engine → composite (weight × confidence blend, RISK as a dampener)
                    → signal_agreement (how many independent categories concur)
                    → conviction_confidence (agreement × coverage, penalised by dispersion)
                          │
                          ▼
     explainability engine → thesis · bull/bear · catalysts · risks · summaries
                           · confidence explanation · monitoring checklist
```

- **Composite** is a weight- and confidence-weighted blend of the eleven "attractiveness"
  categories; **Risk** is applied as a *dampener* rather than averaged in.
- **Conviction confidence** is the headline honesty metric: it rises only when many
  *independent* categories agree and real data underpins them, and falls under dispersion or
  thin data. This is the mechanism that keeps single-indicator noise from surfacing.
- Every number is reproducible — analyzers are pure functions (no I/O, clock, or randomness).

See [ARCHITECTURE.md](ARCHITECTURE.md) and [CONTRACTS.md](CONTRACTS.md) for the full design.

## API

Start with `convexity serve` (or Docker), then:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET`  | `/health` | Liveness probe |
| `POST` | `/api/scans` | Start a scan (background); returns a job id |
| `GET`  | `/api/scans/{id}` | Poll job status + progress |
| `GET`  | `/api/scans/{id}/events` | Server-Sent-Events progress stream |
| `GET`  | `/api/scans/{id}/result` | Full `ScanResult` once complete |
| `GET`  | `/api/scans/latest` | Most recent completed scan |
| `GET`  | `/api/companies/{ticker}` | Single-name analysis |
| `GET`  | `/api/universe/preview` | Eligible-universe preview |

Interactive OpenAPI docs are served at `/docs`.

## Data sources

| Provider | Needs key | Supplies |
| --- | --- | --- |
| **Yahoo Finance** (`yfinance`) | no | prices, fundamentals, valuation, news, quick screening |
| **SEC EDGAR** | no (just a User-Agent) | company facts, filings, **Form 4 insider transactions**, share counts |
| **Financial Modeling Prep** | `FMP_API_KEY` | statements, insider trades, institutional ownership |

All fetches are throttle-resilient: complete responses are disk-cached (partial or rate-limited
responses are deliberately **never** cached), and when the valuation endpoint is unavailable the
market cap is derived from last close × reported shares — real arithmetic on real data, always
labeled with its provenance in `data_warnings`.

The universe builder pulls the full list of U.S.-listed common stocks (Nasdaq Trader symbol
files, with an SEC fallback) and filters ETFs/funds/warrants/units before screening by cap and
liquidity. A bundled `seed_universe.csv` (~120 real names) provides an offline fallback.

## Configuration

Copy `.env.example` to `.env`. Everything has a sensible default:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONVEXITY_DATA_DIR` | `./data` | Disk cache + saved scans |
| `SEC_USER_AGENT` | (generic) | Sent to SEC EDGAR (set a real contact) |
| `FMP_API_KEY` | – | Enables the FMP provider |
| `ALPHAVANTAGE_API_KEY` | – | Optional additional provider |
| `CONVEXITY_LOG_LEVEL` | `INFO` | Logging verbosity |

## Docker

```bash
docker compose up --build          # API + dashboard on http://localhost:8000
# one-off CLI scan:
docker compose run --rm convexity convexity scan --top-n 5 --universe-limit 60
```

## Testing & quality

```bash
make test     # 480 tests, fully offline & deterministic (synthetic FakeProvider)
make lint     # ruff
make cov      # coverage report
```

The suite covers the scoring math, ranking/conviction logic, explainability (narratives only
restate attached evidence), every analyzer, the end-to-end pipeline, and the API.

**Continuous integration.** The full CI pipeline — ruff, an all-12-analyzers-register check, and
`pytest` across Python 3.9–3.12, followed by a Docker build + container health smoke test — ships
at [`.github/ci.yml`](.github/ci.yml). To activate it, move the file to `.github/workflows/ci.yml`
and push with a token that has the `workflow` scope (or paste it into GitHub's web editor, which
does not require that scope).

## Project layout

```
convexity/
├── core/        models · contracts · registry · scoring · config · logging
├── data/        providers (yfinance · sec_edgar · fmp) · universe · cache · aggregator
├── analysis/    12 analyzers + finance NLP
├── ranking/     composite/conviction engine · explainability engine
├── pipeline.py  universe → screen → fetch → analyze → rank → explain
├── api/         FastAPI app · routes · schemas · job store
└── cli.py       Typer CLI (scan · analyze · universe · serve)
frontend/        zero-build dashboard (index.html · app.js · charts.js · styles.css)
tests/           unit + integration (offline, deterministic)
```

## Contributing

Analyzers are independent and easy to add: implement `Analyzer` from
[`convexity/core/contracts.py`](convexity/core/contracts.py), decorate with
`@register_analyzer`, and the pipeline/ranking pick it up automatically. Keep `analyze()` pure
and back every claim with an `Evidence` item. Run `make lint test` before opening a PR.

## License

[MIT](LICENSE). Convexity is provided for research and educational purposes only and is not
investment advice.
