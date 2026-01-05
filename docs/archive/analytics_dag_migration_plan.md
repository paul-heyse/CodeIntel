---
title: "Analytics DAG-Centric Migration Plan"
status: "design"
scope: "analytics pilots + replication checklist"
pilots:
  - "function_metrics"
  - "coverage_targets"
---

# Analytics DAG-Centric Migration Plan

This document defines the pilot migrations and replication checklist for moving analytics targets to the shared DAG-centric patterns described in `docs/full_dag_basis_implementation_plan.md`.

## 0) Decisions and constraints

- Start with analytics to validate the shared library patterns under high transparency and minimal external-tool coupling.
- Migrate aggressively (no compatibility or deprecation layers).
- Validate both rows-based and Ibis-based materialization flows before scaling across analytics.

## 1) Shared library baseline (assumed available)

The following shared modules are the foundation for the pilot migrations:

- `src/codeintel/build/hamilton/native/patterns/savers.py` (`save_rows`, `save_ibis_table`)
- `src/codeintel/build/hamilton/native/patterns/materialization_collectors.py`
- `src/codeintel/build/hamilton/native/patterns/paths.py`
- `src/codeintel/build/hamilton/native/patterns/tool_target.py`
- `src/codeintel/build/hamilton/native/materialization_records.py` (`record_from_materializations`)

## 2) Pilot A (simple): function_metrics

Target module: `src/codeintel/build/hamilton/native/analytics/function_metrics.py`

### 2.1 Current state summary

- Rows-only outputs with `SaveToObjectMetadataDecorator([DuckDBRowsSaver])`.
- Manual skip checks inside `t__function_metrics__compute`.
- Anchor uses `record_from_duckdb_materializations(...)` with a manual mapping.

### 2.2 Target end state

- All table outputs declared via `save_rows(...)` helper.
- Materialization metadata collected via a shared collector function.
- Anchor uses `record_from_materializations(...)` (tables only).
- Skip logic normalized: compute uses a shared hash/skip node or relies on saver skip for light compute.

### 2.3 Step-by-step implementation

1) **Add hash options node**
- `function_metrics__hash_options(env: BuildEnv) -> InputHashOptions`
- Use `options_hash_for_target(env, "function_metrics")` + `env.manifest_index`.
- This node becomes the single hash source for any skip gating and saver nodes.

2) **Normalize skip gating**
- Option A (preferred): add `function_metrics__skip(env, graph, function_metrics__hash_options) -> bool` using `NativeTargetExecutor.for_target(...).should_skip()`.
- `t__function_metrics__compute` returns `None` when skipped.
- Remove `should_skip_native_target` and `compute_input_hash` from the compute node.

3) **Replace SaveToObjectMetadataDecorator with save_rows**
- Replace each of the three `@SaveToObjectMetadataDecorator([DuckDBRowsSaver], ...)` blocks with:
  - `@save_rows(domain="analytics", target="function_metrics", table_key=..., hash_options_node="function_metrics__hash_options")`
- Preserve `deferred_columns_for_table_key(...)` usage via helper default.

4) **Add materialization collector**
- Use `make_table_materializations_collector(...)` for the three tables.
- Name it `function_metrics__table_materializations`.

5) **Update anchor to use record_from_materializations**
- `t__function_metrics` should call:
  - `record_from_materializations(env=..., graph=..., target_name="function_metrics", artifact_materializations=None, table_materializations=function_metrics__table_materializations)`
- Remove manual dict building in the anchor.

6) **Tighten __all__ and node naming**
- Ensure `__all__` reflects the new collector node and any new hash/skip nodes if public.

### 2.4 Acceptance criteria

- DAG-derived output inventory matches tables and schemas for function_metrics.
- `build validate` with `enforce_compute_io_purity=True` passes for function_metrics.
- No `SaveToObjectMetadataDecorator` remains in this module.
- Target record is built solely from saver metadata.

## 3) Pilot B (complex): coverage_targets

Target module: `src/codeintel/build/hamilton/native/analytics/coverage_targets.py`

### 3.1 Current state summary

- Multi-target module with both Ibis (`DuckDBIbisTableSaver`) and rows (`DuckDBRowsSaver`) materialization.
- Uses `SaveToObjectMetadataDecorator` and `record_from_duckdb_materialization(...)`.

### 3.2 Target end state

- All saver declarations use `save_rows` or `save_ibis_table`.
- Each target uses a collector + `record_from_materializations(...)`.
- Shared hash options nodes per target to keep skip semantics consistent.

### 3.3 Step-by-step implementation

For each target in the module:

#### A) coverage_functions (Ibis)
1) Add `coverage_functions__hash_options(env) -> InputHashOptions`.
2) Replace saver decorator with `save_ibis_table(domain="analytics", target="coverage_functions", table_key="analytics.coverage_functions", hash_options_node="coverage_functions__hash_options")`.
3) Add `coverage_functions__table_materializations` via `make_table_materializations_collector`.
4) Update `t__coverage_functions` anchor to use `record_from_materializations(...)`.

#### B) coverage_test_edges (rows)
1) Add `coverage_test_edges__hash_options(env) -> InputHashOptions`.
2) Replace saver decorator with `save_rows(...)`.
3) Add `coverage_test_edges__table_materializations` via `make_table_materializations_collector`.
4) Update `t__coverage_test_edges` anchor to use `record_from_materializations(...)`.

#### C) behavioral_coverage (rows)
1) Add `behavioral_coverage__hash_options(env) -> InputHashOptions`.
2) Replace saver decorator with `save_rows(...)`.
3) Add `behavioral_coverage__table_materializations` via `make_table_materializations_collector`.
4) Update `t__behavioral_coverage` anchor to use `record_from_materializations(...)`.

### 3.4 Acceptance criteria

- All three targets build records from saver metadata (no direct materialization mappings in anchors).
- Output inventory matches DAG-derived table keys for the module.
- No direct `SaveToObjectMetadataDecorator` usage remains.
- `build validate` with `enforce_compute_io_purity=True` passes for coverage targets.

## 4) Replication checklist for remaining analytics modules

Use this checklist once both pilots pass validation. Apply per module:

1) **Classify the module**
- Rows-only (DuckDBRowsSaver).
- Ibis-only (DuckDBIbisTableSaver).
- Mixed (both kinds).

2) **Normalize hash inputs**
- Add `<target>__hash_options(env)` for each target using `options_hash_for_target(...)` and `env.manifest_index`.
- Optionally add `<target>__skip(...)` if compute is expensive.

3) **Replace SaveToObjectMetadataDecorator**
- Use `save_rows(...)` or `save_ibis_table(...)` with static `table_key` and `hash_options_node`.

4) **Add collectors**
- Use `make_table_materializations_collector(...)` for all table outputs.

5) **Update anchors**
- Replace `record_from_duckdb_materializations(...)` with `record_from_materializations(...)`.
- Pass `artifact_materializations=None` for table-only targets.

6) **Remove bespoke skip helpers**
- Eliminate `should_skip_native_target` calls inside compute nodes.

7) **Validate**
- Run `build validate` with `enforce_compute_io_purity=True`.
- Compare row counts or sample rows against pre-migration baselines.

## 5) Suggested execution order after pilots

- Phase 1: rows-only, small surface modules (`subsystem_targets.py`, `subsystem_cache_targets.py`, `hotspots.py`).
- Phase 2: rows-only, larger multi-target modules (`function_detail_targets.py`, `classification_targets.py`).
- Phase 3: Ibis-heavy modules (`risk_factors.py`, `config_graph_targets.py`).
- Phase 4: large multi-target aggregation modules (`metadata_targets.py`, `metrics_targets.py`).

## 6) Validation gates

- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- Run `build validate` with `enforce_compute_io_purity=True` after each batch.
- Run targeted analytics tests for the modified targets (keep this focused, then widen).
