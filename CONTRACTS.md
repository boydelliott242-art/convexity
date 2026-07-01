# Contracts

This document is the **binding interface** every downstream module must implement. All shapes
referenced here live in `convexity/core/models.py` and the ABCs/protocols in
`convexity/core/contracts.py`. Import them; do not redefine them.

Reminder: Convexity is a research/screening tool, not advice and not a predictor. Implementers
must honour the honesty rules below — missing data is marked, never fabricated.

---

## 1. `DataProvider` (ABC) — `convexity.core.contracts.DataProvider`

```python
class MyProvider(DataProvider):
    @property
    def name(self) -> str: ...                 # stable id recorded in SecurityData.data_sources
    @property
    def capabilities(self) -> Set[str]: ...    # e.g. {"prices", "fundamentals", "news"}

    def get_universe(self, params: ScanParams) -> List[str]:
        # Optional. If unsupported, leave the base method (raises NotSupported).
        ...

    def get_security_data(self, ticker: str) -> SecurityData:   # REQUIRED
        ...
```

Rules:
- Advertise only capabilities you truly fill. The aggregator routes by `capabilities`.
- **Never fabricate.** Unknown fields stay `None`; append a human note to
  `SecurityData.data_warnings`.
- Raise `DataUnavailable` for an *expected* gap, `ProviderError`/`RateLimited` for a failure,
  `NotSupported` for an unsupported capability. Never raise a bare `Exception` that could crash
  a scan.
- Register the class with `@register_provider`.

---

## 2. `Analyzer` (ABC) — `convexity.core.contracts.Analyzer`

```python
@register_analyzer
class ValueAnalyzer(Analyzer):
    category = ScoreCategory.VALUE          # REQUIRED class attr (one analyzer per category)
    default_weight = 0.16                    # suggested weight if config supplies none
    requires = {"valuation", "fundamentals"} # data this analyzer needs for full coverage

    def analyze(self, data: SecurityData, ctx: AnalysisContext) -> SubScore:
        ...
```

Rules:
- `analyze` must return a `SubScore` for `self.category` and must be **pure** (no I/O, no
  wall-clock, no randomness).
- Use `ctx.peer_stats` / `ctx.universe_stats` to score *relative to comparable companies*, not
  on absolute thresholds alone.
- When required inputs are absent, return `self.neutral_subscore(...)` (score 50, low
  confidence, `MISSING_DATA` flag, low `data_coverage`). A data gap must neither help nor hurt.
- Populate `evidence` with `Evidence` items (use `Evidence.from_number` for metrics) so every
  point of the score is auditable. Set `direction` honestly; a missing value is always
  `neutral`.

### `SubScore` contract (`convexity.core.models.SubScore`)

| Field           | Type                 | Meaning / constraint                                   |
|-----------------|----------------------|--------------------------------------------------------|
| `category`      | `ScoreCategory`      | The category this score is for.                        |
| `score`         | `float`              | **0–100** (validated). Higher = more attractive. For RISK, higher = safer. |
| `confidence`    | `float`              | **0–1**. How trustworthy the score is given data quality. |
| `weight`        | `float`              | `>= 0`. Weight in the composite.                       |
| `rationale`     | `str`                | One- or two-sentence human explanation.                |
| `evidence`      | `List[Evidence]`     | The auditable facts behind the score.                  |
| `flags`         | `List[str]`          | e.g. `MISSING_DATA`, `STALE`, `ANOMALY`.               |
| `data_coverage` | `float`              | **0–1**. Fraction of required inputs actually present. |

---

## 3. `AnalysisContext` (dataclass)

```python
AnalysisContext(
    peer_stats: Optional[Dict[str, Any]],     # metric -> distribution across peers
    universe_stats: Optional[Dict[str, Any]], # metric -> distribution across screened universe
    config: Optional[Settings],
    extras: Dict[str, Any],                    # free-form, analyzer-defined
)
```

---

## 4. `RankingEngine` (Protocol) — `convexity.core.contracts.RankingEngine`

```python
def rank(self, analyses: List[CompanyAnalysis], params: ScanParams) -> List[CompanyAnalysis]: ...
def score_company(self, data: SecurityData, subscores: List[SubScore],
                  weights: Dict[ScoreCategory, float]) -> CompanyAnalysis: ...
```

- `score_company` must use `convexity.core.scoring.combine_subscores` to derive
  `composite_score`, `signal_agreement` and `conviction_confidence`. RISK is applied as a
  dampener, not averaged in.
- `rank` sorts best-first and sets each `CompanyAnalysis.rank` (1-based).

---

## 5. `ExplainabilityEngine` (Protocol) — `convexity.core.contracts.ExplainabilityEngine`

```python
def explain(self, analysis: CompanyAnalysis, data: SecurityData) -> CompanyAnalysis: ...
```

- Must populate `thesis`, `bull_case`, `bear_case`, `catalysts`, `principal_risks`,
  `valuation_summary`, `fundamental_summary`, `technical_summary`, `confidence_explanation`,
  and `monitoring_checklist` strictly from the sub-scores' `evidence` — no claims beyond the
  evidence, and `confidence_explanation` must state plainly when conviction is low or data is
  thin.

---

## 6. Registry — `convexity.core.registry`

```python
@register_provider     # decorate a DataProvider subclass; keyed by .name
@register_analyzer     # decorate an Analyzer subclass; keyed by .category
get_providers(); get_provider(name)
get_analyzers(); get_analyzer(category)
clear()                # test isolation
```

Importing a provider/analyzer module self-registers it; the pipeline discovers implementations
by importing their packages.

---

## 7. Default weights — `convexity.core.config.DEFAULT_CATEGORY_WEIGHTS`

A `Dict[ScoreCategory, float]`. The eleven additive categories sum to **1.0**; RISK has weight
`0.0` because it is applied as a penalty/dampener by the ranking layer, not averaged into the
composite. VALUE / GROWTH / QUALITY / FINANCIAL_HEALTH / CATALYST carry the highest weights.
