# BUILD MODULE — ORCHESTRATION LAYERING & DOMAIN DECOUPLING (IMPLEMENTATION PLAN)

## Context
This plan operationalizes a **build-centric, DAG-first** target state:

- `codeintel.build` is the orchestration/integration layer (targets, contracts, validation, persistence).
- Domain packages (`codeintel.analytics`, `codeintel.graphs`, `codeintel.ingestion`, `codeintel.core`) become
  **build-agnostic** and contain pure compute + domain protocols.
- All dataset writes become **DAG-visible** and flow through **Warehouse / Hamilton savers** (no ad-hoc writes).

This plan is complementary to (and should be executed alongside) the existing build DAG consolidation work:
`docs/build_refinement/BUILD_MODULE_DAG_CENTRIC_CONSOLIDATION_OPPORTUNITIES_IMPLEMENTATION_PLAN.md`.

---

## Success Criteria (Definition of Done)

### Architecture invariants
- No imports of `codeintel.build` from:
  - `src/codeintel/analytics/**`
  - `src/codeintel/graphs/**`
  - `src/codeintel/ingestion/**`
  - `src/codeintel/core/**`
  - (including within `TYPE_CHECKING`)
- Domain modules do not perform DuckDB writes directly:
  - No `gateway.policy.bulk_insert*`, `gateway.policy.delete_for_snapshot`, or similar write-path APIs outside
    `src/codeintel/build/**` and `src/codeintel/storage/**`.
- Build nodes compute; writes happen via **Warehouse** and/or **Hamilton saver** boundaries only.

### Behavioral outcomes
- All previously “side-effecting” analytics/graphs computations invoked by build targets are converted to
  “compute returns rows/expressions” + “build materializes”.
- The Hamilton DAG is the authoritative view of dependencies and I/O for those targets (caching/skip/records).
- Quality gates pass:
  - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
  - `uv run pytest -q`

---

## Current State (Problems to Fix)

### Domain → build import coupling (layering violation)
Examples of domain code importing build for contracts/validation:
- `src/codeintel/analytics/utilities/datasets.py:24` imports `validate_df` and `get_contract_for_table_key`.
- `src/codeintel/analytics/functions/metrics.py:49` imports `validate_df`.
- `src/codeintel/analytics/graphs/config_graph_metrics.py:18` imports `get_contract_for_table_key`.
- `src/codeintel/analytics/graphs/subsystem_graph_metrics.py:18` imports `get_contract_for_table_key`.
- `src/codeintel/analytics/graphs/symbol_orchestrator.py:24` imports `get_contract_for_table_key`.

### Domain modules performing persistence (DAG-invisible I/O)
Several `analytics/graphs` functions compute *and* write (often called from build targets):
- `src/codeintel/analytics/graphs/graph_metrics.py:190` (`compute_graph_metrics`) writes tables.
- `src/codeintel/analytics/graphs/graph_metrics_ext.py:231` writes via orchestrator.
- `src/codeintel/analytics/graphs/module_graph_metrics_ext.py:221` writes via orchestrator.
- `src/codeintel/analytics/graphs/graph_stats.py:47` writes tables.
- `src/codeintel/analytics/graphs/subsystem_graph_metrics.py:97` writes tables.
- `src/codeintel/analytics/graphs/config_graph_metrics.py:315` writes tables (appears unused).

Build currently calls many of these side-effecting functions inside a tool node:
- `src/codeintel/build/hamilton/native/graphs/graph_targets.py:1422` (`t__graph_metrics__compute`).

### Ingestion bridge lives in ingestion (and references build types)
- `src/codeintel/ingestion/adapters/build_tool_adapter.py:1` bridges build tool providers to ingestion ports.
  This is an integration concern and should live in build (or a dedicated boundary layer), not in ingestion.

---

## Target State (Concrete)

### Layering
- **Allowed dependencies**
  - `build -> {analytics, graphs, ingestion, core, storage, config}`
  - `{analytics, graphs, ingestion} -> {core, storage, config, stdlib, third-party}`
  - `storage -> {core, config, stdlib, third-party}` (no `storage -> build`)
- **Forbidden dependencies**
  - `{analytics, graphs, ingestion, core} -> build`

### Canonical write boundary
- All dataset writes happen via:
  - Hamilton savers (e.g., `DuckDBRowsSaver`, `DuckDBIbisTableSaver`), or
  - Build-owned persistence helpers that delegate to `Warehouse` with consistent `MaterializeOptions`.

### Canonical “row output shape”
Adopt a consistent convention to minimize conversion logic:
- For row-oriented outputs: `tuple[tuple[object, ...], ...]` + explicit `*_COLS` in build target modules.
- For table-expression outputs: `ibis.ir.Table` materialized by `DuckDBIbisTableSaver`.

Validation is done at the write boundary (savers), not inside domain compute.

---

## Proposed Code Organization (New/Adjusted Modules)

### New build persistence helpers
Add a small build-owned persistence package for the non-Hamilton and “glue” cases:

```
src/codeintel/build/persistence/
  __init__.py
  options.py        # MaterializeOptions constructors (owner_target/input_hash/snapshot defaults)
  rows.py           # mapping->tuple helpers; scalar normalization (numpy -> python)
  materialize.py    # Warehouse-based helpers for snapshot-scoped writes
```

Design notes:
- Keep these helpers thin and “boring”; prefer using Hamilton savers where possible.
- Do not embed schema registry logic here; schema/contract resolution stays in build schemas or gateway datasets.

### Move build↔ingestion bridge into build
Relocate the adapter:
- From: `src/codeintel/ingestion/adapters/build_tool_adapter.py`
- To: `src/codeintel/build/ingestion/build_tool_adapter.py` (or `src/codeintel/build/bridges/ingestion_tools.py`)

---

## Implementation Plan (Phased, 4–7 PRs)

### Phase 0 — Inventory + guardrail scaffolding (no behavior change)
Goal: make violations easy to see and prevent drift during migration.

1. Add a fast “architecture scan” test suite (initially report-only or introduced near the end if preferred):
   - Assert no `codeintel.build` imports under `src/codeintel/{analytics,graphs,ingestion,core}`.
   - Assert no persistence API usage under domain packages (e.g., `.policy.bulk_insert*`, `.policy.delete_for_snapshot`).
2. Add a developer command snippet to run the scan locally:
   - `uv run pytest -q tests/architecture/test_build_layering.py`

Acceptance gates:
- Tests exist and can be enabled (or enforced) once migration completes.

---

### Phase 1 — Introduce build persistence helpers (foundation)
Goal: provide a canonical, build-owned API for the remaining imperative write flows.

1. Add `src/codeintel/build/persistence/options.py`:
   - `materialize_options_for_target(env, target_name, *, mode=..., input_hash=...) -> MaterializeOptions`
2. Add `src/codeintel/build/persistence/rows.py`:
   - `normalize_scalar(value: object) -> object` (numpy scalars → python)
   - `mappings_to_row_tuples(rows, *, columns) -> tuple[tuple[object, ...], ...]`
3. Add `src/codeintel/build/persistence/materialize.py`:
   - `materialize_snapshot_rows(env, *, table_key, columns, rows, owner_target, input_hash) -> MaterializationResult`
   - `materialize_snapshot_table(env, *, table_key, expr, owner_target, input_hash) -> MaterializationResult`
4. Add unit tests for these helpers.

Acceptance gates:
- Helpers are used by at least one build-native module (to prove they are real and correct).
- No new build→storage “ad-hoc writes” are introduced.

---

### Phase 2 — Remove build imports from analytics compute paths (validation at boundary)
Goal: domain compute stops importing build for `validate_df` and schema/contract utilities.

1. Refactor `src/codeintel/analytics/functions/metrics.py`:
   - Remove `codeintel.build.hamilton.contracts.schemas.validation.validate_df` import and all compute-time validation.
   - Ensure compute functions return rows/results only; rely on saver validation (`DuckDBRowsSaver`) in:
     `src/codeintel/build/hamilton/native/analytics/function_metrics.py`.
2. Refactor `src/codeintel/analytics/graphs/config_graph_metrics.py`:
   - Remove `codeintel.build.schemas.get_contract_for_table_key` usage.
   - Remove `validate_tuple_rows` dependency on build-time validation.
   - Compute row tuples directly in the correct column order using existing `*_COLS` constants.
   - Delete the unused side-effecting `compute_config_graph_metrics(...)` if it is truly unreferenced.
3. Refactor `src/codeintel/analytics/graphs/subsystem_graph_metrics.py`:
   - Split into:
     - `compute_subsystem_graph_metrics_result(...) -> tuple[tuple[object, ...], ...]`
     - (optional) remove/replace `compute_subsystem_graph_metrics(...)` (side-effecting)
   - Remove contract lookups in analytics; let build materialize.
4. Refactor `src/codeintel/analytics/graphs/symbol_orchestrator.py` similarly:
   - Produce rows (not writes), no build schema import.

Build-side changes in this phase (minimal and mechanical):
- Update build native targets that call these functions to:
  - call the new `*_result` compute helpers,
  - materialize with savers or build persistence helpers.

Acceptance gates:
- `rg -n "\\bcodeintel\\.build\\b" src/codeintel/analytics` returns 0 matches.
- Build targets still produce identical tables (row counts and schema) for a known repo/commit.

---

### Phase 3 — Eliminate analytics persistence utilities (or relocate into build)
Goal: remove the “analytics.utilities.datasets” persistence layer that currently wraps build contracts.

1. Identify all call sites of:
   - `insert_analytics_rows`, `validate_contract_rows`, `validate_tuple_rows`,
     `get_analytics_dataset_contract`, and `get_delete_sql_by_table`.
2. Replace each call site with one of:
   - Hamilton saver-based materialization (preferred), or
   - Build persistence helpers (Phase 1) delegating to `Warehouse`.
3. Remove or relocate:
   - `src/codeintel/analytics/utilities/datasets.py`
   - `src/codeintel/analytics/utilities/persistence.py`
   - Update `src/codeintel/analytics/utilities/__init__.py` exports accordingly.

Acceptance gates:
- No remaining imports of `codeintel.analytics.utilities.datasets` from build-native modules.
- No remaining build imports anywhere in analytics.

---

### Phase 4 — Convert build “graph_metrics” tool-node writes into DAG-visible dataset materialization
Goal: remove side effects from `t__graph_metrics__compute` and represent graph metrics writes explicitly in DAG.

1. Refactor analytics graph metrics modules to expose pure compute primitives:
   - Convert `compute_graph_metrics(...)` (currently writes) into:
     - `compute_graph_metrics_result(...) -> GraphMetricsResult` (rows/expressions only)
   - Convert `compute_graph_stats(...)` similarly.
   - Convert `compute_graph_metrics_functions_ext` / `compute_graph_metrics_modules_ext` to return rows only
     (by refactoring `src/codeintel/analytics/graphs/orchestrator.py` to produce validated rows, not persist).
2. Update `src/codeintel/build/hamilton/native/graphs/graph_targets.py`:
   - Replace the “tool compute does writes” pattern with canonical dataset patterns:
     - Compute nodes: return `ir.Table` or row tuples per table.
     - Materialize nodes: use `DuckDBIbisTableSaver` or `DuckDBRowsSaver`.
     - Record nodes: aggregate `TargetRunRecord` via `record_from_duckdb_materializations`.
3. Ensure skip/manifest behavior still works:
   - Input hash computed once per target.
   - Materializers receive `owner_target` and `input_hash` consistently.

Acceptance gates:
- The graph metrics target’s I/O is visible in DAG nodes (materialize nodes exist per dataset).
- No `.policy.bulk_insert*` writes remain in analytics graph metrics modules.

---

### Phase 5 — Move ingestion build bridge into build (remove latent build typing deps)
Goal: ingestion remains a pure domain package with no conceptual dependency on build.

1. Move `src/codeintel/ingestion/adapters/build_tool_adapter.py` into build:
   - New location: `src/codeintel/build/ingestion/build_tool_adapter.py` (name can be finalized).
2. Update build-native code to import from the new location.
3. If external callers exist, add a temporary compatibility shim:
   - Prefer a re-export in build, not ingestion.

Acceptance gates:
- `rg -n \"\\bcodeintel\\.build\\b\" src/codeintel/ingestion` returns 0 matches.

---

### Phase 6 — Enforce guardrails + clean up docs/examples
Goal: make the target state self-maintaining.

1. Turn on architecture tests from Phase 0 as hard failures.
2. Add targeted invariants:
   - “No build imports in domain packages”
   - “No direct policy writes in domain packages”
3. Update any documentation examples that suggest domain packages should import build directly
   (e.g., code examples in docstrings).

Acceptance gates:
- Guardrail tests fail on intentional violations.
- All quality gates pass.

---

## Detailed Migration Map (Per File)

### Analytics (remove build imports + remove writes)
- `src/codeintel/analytics/functions/metrics.py`
  - Remove compute-time `validate_df` usage; rely on saver validation.
- `src/codeintel/analytics/graphs/config_graph_metrics.py`
  - Remove contract lookup; return ordered tuples using existing `*_COLS`; delete unused writer function.
- `src/codeintel/analytics/graphs/subsystem_graph_metrics.py`
  - Split compute (rows) from persistence; build owns persistence.
- `src/codeintel/analytics/graphs/symbol_orchestrator.py`
  - Split compute (rows) from persistence; build owns persistence.
- `src/codeintel/analytics/graphs/graph_metrics.py`
  - Split compute (rows/expr) from persistence; build owns persistence.
- `src/codeintel/analytics/graphs/orchestrator.py`
  - Convert to “compute returns rows”; remove persistence helper dependencies.
- `src/codeintel/analytics/graphs/graph_metrics_ext.py`
- `src/codeintel/analytics/graphs/module_graph_metrics_ext.py`
- `src/codeintel/analytics/graphs/graph_stats.py`

### Build (stop calling domain functions that write; materialize in DAG)
- `src/codeintel/build/hamilton/native/graphs/graph_targets.py`
  - Replace “compute does writes” with dataset materialize nodes.
- `src/codeintel/build/hamilton/native/analytics/metrics_targets.py`
  - Update subsystem/symbol metrics to call result producers + materialize.
- `src/codeintel/build/hamilton/native/analytics/config_graph_targets.py`
  - Ensure compute path doesn’t require build schemas in analytics.
- `src/codeintel/build/hamilton/native/analytics/metadata_targets.py`
  - Remove reliance on `analytics.utilities.datasets` persistence helpers (use savers/build persistence).

### Ingestion (move build bridge)
- `src/codeintel/ingestion/adapters/build_tool_adapter.py` → `src/codeintel/build/ingestion/...`

---

## Testing & Validation Strategy

### Per PR
- Run quality suite:
  - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Run unit tests:
  - `uv run pytest -q`

### Targeted regression checks
- Table schema/column order invariants:
  - Add tests that validate `*_COLS` constants match the schema provider for key tables.
- Snapshot semantics:
  - Verify “replace” mode deletes snapshot rows before append.
- Manifest/skip:
  - Verify unchanged inputs skip materialization and row counts are read from manifests when skipped.

---

## Risks & Mitigations
- **Hidden external callers rely on side-effecting analytics functions**:
  - Mitigation: repo-wide `rg` for call sites; if needed, add new build entrypoints and deprecate old APIs with
    explicit migration docs (avoid adding domain→build imports).
- **Row tuple ordering drift vs schema provider**:
  - Mitigation: add tests asserting `*_COLS` match schema provider and enforce in CI.
- **Circular imports during migration**:
  - Mitigation: introduce build persistence helpers first; migrate build-native modules off analytics persistence
    utilities before deleting them.

---

## Open Decisions (Choose Early)
1. Final location naming for persistence helpers:
   - `codeintel.build.persistence` (recommended) vs `codeintel.build.io` vs `codeintel.build.materialize`.
2. Row output convention:
   - Prefer `tuple[tuple[...], ...]` + `*_COLS` for saver-based materialization (recommended).
3. Guardrail enforcement timing:
   - Enable failures only after Phase 2/3 so the migration is not blocked mid-flight.

