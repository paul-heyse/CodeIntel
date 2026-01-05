# Hamilton Best-in-Class CPG Pipeline Plan

## Goal

Deliver a Hamilton-native, Arrow-first CPG build that is cacheable, observable, and
deterministic, with outputs optimized for DuckDB ingestion and FastMCP delivery.

## Scope

- Resolve current build failures (cache normalization, diagnostics robustness, schema precision).
- Harden caching policy and Arrow materialization, avoiding opaque object caching.
- Improve DAG structure and lineage using Hamilton modifiers (pipe, with_columns, tags, schema).
- Replace external telemetry reliance with Hamilton-native tracking and diagnostics outputs.
- Ensure all artifacts land under `build/` with clear manifests and metadata.

## Non-Goals

- No changes to ingestion semantics beyond deterministic schema and lineage improvements.
- No introduction of new storage backends; output remains Arrow/Parquet and JSONL.
- No partial retention of deprecated telemetry systems (OTel removed for Hamilton DAGs).

## Design Principles

1. Arrow-first materialization: cache and export tabular outputs as Arrow/Parquet.
2. Determinism: avoid caching non-deterministic or environment-bound nodes.
3. Explicit lineage: surface intermediate transforms as DAG nodes with tags and schema.
4. Observability: diagnostics and telemetry are Hamilton-native and stored under `build/`.
5. Separation of concerns: IO is DAG materialization, not execution inputs.

## Current Failure Summary (Baseline)

- Cache adapter tries to normalize dataclasses with `init=False` fields, causing `TypeError`.
- Diagnostics cache event emission fails when cache logs are missing (KeyError).
- Schema mismatch for `function_goid_h128` in `graph.cfg_blocks` and `graph.cfg_edges`
  (precision 28 vs 38).
- OTel exporter attempts to send to `localhost:4317` with noisy failures.

## Phase 1: Stabilize Cache and Diagnostics

### Detailed Checklist

- [x] Inventory nodes that are non-cacheable or environment-bound (env/config/logging).
- [x] Identify dataclasses that pass through cache normalization and note `init=False` fields.
- [x] Update cache normalization to skip `init=False` fields in
  `src/codeintel/build/hamilton/cache_adapter.py`.
- [x] Ensure dataclass normalization handles nested dataclasses safely.
- [x] Make cache event emission resilient in `src/codeintel/build/hamilton/diagnostics.py`
  (treat missing `run_id` logs as empty).
- [x] Add initial node-level cache behavior tags for config/option nodes
  (`@cache(behavior="ignore")` for env/config/logging nodes).
- [x] Extend cache behavior tags to all non-cacheable nodes identified in inventory.
- [x] Add a short validation script or targeted run to confirm cache adapter stability.
- [ ] Confirm diagnostics output still includes `run_summary.json` and `node_telemetry.jsonl`.
- [x] Document cache ignore policy for plan context and telemetry nodes
  (`docs/architecture/hamilton_cache_policy.md`).

### Representative External Library Usage

Dataclass-safe normalization in cache adapter (Python stdlib):

```python
from dataclasses import fields, is_dataclass, replace

def _normalize_dataclass(value: object) -> object:
    if not is_dataclass(value):
        return value
    updates: dict[str, object] = {}
    for field in fields(value):
        if not field.init:
            continue
        updates[field.name] = _normalize_dataclass(getattr(value, field.name))
    return replace(value, **updates)
```

Hamilton cache ignore for non-deterministic nodes:

```python
from hamilton.function_modifiers import cache

@cache(behavior="ignore")
def plan_context(env: BuildEnv, catalog: DagCatalog) -> PlanContext:
    ...
```

Diagnostics: tolerate missing cache logs:

```python
try:
    logs_by_node = cache_adapter.logs(run_id=run_id, level="info")
except KeyError:
    logs_by_node = {}
```

### Acceptance Criteria

- `uv run codeintel build run --all --verbose=1` no longer fails in cache adapter.
- Diagnostics emit `run_summary.json` and do not crash when cache logs are missing.

## Phase 2: Schema Precision Alignment

Decision: use Decimal(38, 0) for `function_goid_h128` to align with DuckDB inputs.

### Checklist

- [x] Choose authoritative precision for `function_goid_h128` (selected: Decimal(38, 0)).
- [ ] Align producing nodes to the chosen precision via explicit cast.
- [ ] Update schema contract to match chosen precision.
- [ ] Add `@check_output` or Pandera validation to localize mismatches.

### Representative External Library Usage

Arrow cast to authoritative Decimal precision:

```python
import pyarrow as pa
import pyarrow.compute as pc

target_type = pa.decimal128(38, 0)
column = pc.cast(table["function_goid_h128"], target_type)
table = table.set_column(
    table.schema.get_field_index("function_goid_h128"),
    "function_goid_h128",
    column,
)
```

Hamilton output checks for schema enforcement:

```python
from hamilton.function_modifiers import check_output

@check_output(importance="fail")
def cfg_edges_table(...) -> pa.Table:
    ...
```

### Acceptance Criteria

- `graph.cfg_blocks` and `graph.cfg_edges` pass validation in full build.
- Schema mismatch failures are eliminated or downgraded to controlled warnings.

## Phase 3: Cache Policy and Arrow Materialization

### Checklist

- [ ] Switch to opt-in caching: `default_behavior="disable"` at builder, list
  deterministic nodes explicitly.
- [ ] Use `@cache(format="parquet")` for Arrow tabular nodes.
- [ ] Mark external inputs and environment/config nodes as `recompute` or `ignore`.
- [ ] Add cache policy inventory doc for node classes (ingestion, graph, analytics).

### Representative External Library Usage

Hamilton opt-in caching and Parquet cache format:

```python
from hamilton import driver
from hamilton.function_modifiers import cache

@cache(format="parquet")
def cpg_edges_table(...) -> pa.Table:
    ...

dr = (
    driver.Builder()
    .with_cache(default_behavior="disable", default=["cpg_edges_table"])
    .build()
)
```

### Acceptance Criteria

- Cache behavior is deterministic across runs with minimal invalidation surprises.
- Cached artifacts are Arrow/Parquet, readable by DuckDB without conversion.

## Phase 4: DAG Structure and Lineage Improvements

### Checklist

- [ ] Apply `@pipe_input` / `@pipe_output` to multi-step transforms so each step
  is a distinct node.
- [ ] Use `@with_columns` for column-level transforms in tabular nodes.
- [ ] Add `@schema` metadata to key table outputs (core, graph, analytics).
- [ ] Add `@tag`/`@tag_output` for `kind` and `schema_ref` to eliminate anchor warnings.
- [ ] Introduce `@parameterize`/`@inject` for registry-driven node generation.

### Representative External Library Usage

Hamilton pipe family and tags for explicit lineage:

```python
from hamilton.function_modifiers import pipe_input, tag

@pipe_input(step_a, step_b, on_input="raw_table", namespace="normalize")
@tag(kind="graph", schema_ref="graph.cfg_edges")
def cfg_edges_table(raw_table: pa.Table) -> pa.Table:
    return raw_table
```

Hamilton `with_columns` for column-level subDAGs:

```python
import polars as pl
from hamilton.plugins.h_polars import with_columns

@with_columns(select=["degree_in", "degree_out"], namespace="graph")
def graph_metrics_table(table: pl.DataFrame) -> pl.DataFrame:
    return table
```

Hamilton parameterization for registry-driven nodes:

```python
from hamilton.function_modifiers import parameterize, source, value

@parameterize(
    cfg_edges= {"table_key": value("graph.cfg_edges"), "table": source("cfg_edges")},
    cfg_nodes= {"table_key": value("graph.cfg_nodes"), "table": source("cfg_nodes")},
)
def export_table(table_key: str, table: pa.Table) -> pa.Table:
    return table
```

### Acceptance Criteria

- Guardrails warnings for missing tags are reduced or eliminated.
- DAG lineage shows intermediate steps with clear tags and schema metadata.

## Phase 5: Hamilton-Native Telemetry and Diagnostics

### Checklist

- [ ] Use Hamilton Tracker for run telemetry (local UI or self-hosted).
- [ ] Set capture controls via environment variables or config:
  - `HAMILTON_CAPTURE_DATA_STATISTICS=0` for production-like runs.
  - `HAMILTON_MAX_LIST_LENGTH_CAPTURE` and `HAMILTON_MAX_DICT_LENGTH_CAPTURE`.
- [ ] Ensure diagnostics output goes under `build/diagnostics`.
- [ ] Remove or disable OTel export for Hamilton DAG execution.

### Representative External Library Usage

Hamilton UI/Tracker integration:

```python
from hamilton import driver
from hamilton_sdk import adapters

tracker = adapters.HamiltonTracker(
    project_id="codeintel",
    username="build@codeintel",
    dag_name="codeintel_cpg",
    tags={"env": "dev", "repo": "codeintel"},
)

dr = driver.Builder().with_modules(...).with_adapters(tracker).build()
```

Telemetry capture controls:

```bash
export HAMILTON_CAPTURE_DATA_STATISTICS=0
export HAMILTON_MAX_LIST_LENGTH_CAPTURE=20
export HAMILTON_MAX_DICT_LENGTH_CAPTURE=50
```

### Acceptance Criteria

- Telemetry is available via Hamilton UI/tracker without OTel noise.
- Diagnostics artifacts are consistently written under `build/diagnostics`.

## Phase 6: Export and Materialization Strategy

### Checklist

- [ ] Ensure exports are materialization outputs, not DAG inputs.
- [ ] Validate export manifest creation for JSONL and Parquet under `build/exports`.
- [ ] Emit dataset manifests and audit logs under `build/`.
- [ ] Verify DuckDB can ingest datasets directly from `build/datasets`.

### Representative External Library Usage

Hamilton materializers for exports:

```python
from hamilton.io import materialization as mat

materials = [
    mat.to.parquet(id="export_parquet", dependencies=["cpg_edges_table"], path="build/exports"),
    mat.to.json(id="export_jsonl", dependencies=["cpg_edges_table"], path="build/exports"),
]
```

DuckDB ingest of Arrow/Parquet outputs:

```python
import duckdb

conn = duckdb.connect("build/db/codeintel.duckdb")
conn.execute(
    \"\"\"CREATE OR REPLACE VIEW cpg_edges AS
       SELECT * FROM 'build/datasets/graph/cpg_edges/*/*.parquet'\"\"\"
)
```

### Acceptance Criteria

- Export targets run without storage dependency.
- Artifacts are discoverable by DuckDB and documented in manifests.

## Phase 7: End-to-End Validation

### Checklist

- [ ] Run quality report:
  `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- [ ] Run full build:
  `uv run codeintel build run --all --verbose=1`
- [ ] Review diagnostics:
  - `build/diagnostics/run_summary.json`
  - `build/diagnostics/node_telemetry.jsonl`
  - `build/diagnostics/cache_events.jsonl` (if cache logs exist)
- [ ] Confirm datasets and manifests under `build/datasets`.

### Representative External Library Usage

Build + diagnostics verification:

```bash
uv run codeintel build run --all --verbose=1
cat build/diagnostics/run_summary.json
```

### Acceptance Criteria

- Full build succeeds and diagnostics are complete.
- No schema validation failures for core graph datasets.

## Deliverables

- Cache adapter and diagnostics fixes.
- Explicit cache policy and Arrow materialization configuration.
- Tagged, schema-annotated DAG nodes with improved lineage.
- Hamilton-native telemetry and diagnostics under `build/`.
- Updated manifests and export artifacts aligned with DuckDB ingestion.

## Risks and Mitigations

- Cache invalidation surprises due to helper function changes.
  Mitigation: use cache behaviors (`recompute`/`ignore`) and explicit cache formats.
- Telemetry capture overhead or data leakage.
  Mitigation: disable data stats capture and enforce size caps.
- Schema drift across datasets.
  Mitigation: use `@check_output` and Pandera checks early in the DAG.

## Open Decisions

- Final precision for `function_goid_h128`.
- Which node families are `default` cacheable under opt-in policy.
- Production policy for Hamilton telemetry capture settings.
