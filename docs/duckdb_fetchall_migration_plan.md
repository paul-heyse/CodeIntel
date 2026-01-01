# DuckDB fetchall Migration Plan (Streaming + Zero-Copy)

## Context

The guardrail `streaming_fetchall` is failing because we still call
`fetchall()` across build, storage, CLI, and ingestion paths. This violates the
architecture goal of zero-copy, streaming access (DuckDB -> Arrow -> Polars)
and risks unbounded memory use.

This plan replaces `fetchall()` with best-in-class DuckDB APIs:

- `fetch_record_batch(batch_size)` / `fetch_arrow_reader()` for streaming
  results.
- `fetchone()` / `fetchmany(n)` for bounded or scalar queries.
- `relation.to_parquet(...)` or `relation.create(...)` for large results that
  are only used as persisted artifacts.

References:
- `docs/python_library_reference/duckdb_advanced.md`
- `docs/python_library_reference/DuckDB_advanced_connection_and_relational_api.md`
- `docs/python_library_reference/polars_and_pyarrow_integration_with_duckdb.md`
- `.codeintel/duckdb.db` (for row-count and size verification)

## Goals

- Eliminate all `fetchall()` calls in code paths (and in examples).
- Make streaming the default for large result sets.
- Preserve correctness with minimal, low-risk code changes.
- Keep DuckDB authoritative and Arrow/Polars zero-copy at the boundaries.

## Non-goals

- Rewriting queries into entirely new SQL or relational plans.
- Introducing ad hoc row materialization in Python.
- Changing public API contracts unless strictly required.

## Canonical replacement patterns

Pattern A: Stream rows (Arrow batches)
- Use when the result size is unbounded or unknown.
- Primary call sequence:
  `reader = relation.fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)`
  then iterate batches and process per batch.
- Minimal conversion options:
  - Arrow native: iterate `batch.columns` or `batch.to_pylist()`.
  - Polars per batch: `pl.from_arrow(batch)` for columnar transforms.
  - Existing helpers: `iter_records_from_arrow_reader(...)` for dict rows.

Pattern B: Scalar or bounded results
- Use when query returns a scalar, a small metadata list, or is explicitly
  bounded via `LIMIT`.
- Primary call sequence:
  `row = con.execute(...).fetchone()` or `con.execute(...).fetchmany(n)`.
- Add or enforce `LIMIT` to make boundedness explicit.

Pattern C: Materialize to Parquet or DuckDB table
- Use when the only consumer is a persisted artifact (exports, caches, or
  intermediate build outputs).
- Primary call sequence:
  `relation.to_parquet("path")` or `relation.create("table_name")`.
- Avoids any Python row materialization.

## Shared helper upgrades (enables minimal code changes)

1) Add streaming row helpers in `src/codeintel/storage/query_results.py`:
   - `iter_records_from_relation(relation, columns=None)` -> iterator of dicts.
   - `iter_tuples_from_relation(relation, columns=None)` -> iterator of tuples.
   - Keep `records_from_relation(...)` as a convenience wrapper for small cases.

2) Add a repository-level helper in `src/codeintel/storage/repositories/base.py`:
   - `stream_rows(expr, params=None, batch_size=DEFAULT_ARROW_BATCH_SIZE)` that
     yields `RowDict` per batch.
   - Optional `fetch_rows_bounded(expr, params, limit)` that enforces `LIMIT`
     and uses `fetchmany`.

3) Centralize batch size:
   - Use `DEFAULT_ARROW_BATCH_SIZE` everywhere; allow override via settings if
     needed for CLI or build.

4) Update examples/docs:
   - Replace any `fetchall()` examples with streaming equivalents.

## Phased execution plan

### Phase 0: Validate boundedness and size (blocked)

- For each query that currently uses `fetchall()`, validate cardinality with
  `.codeintel/duckdb.db`:
  - Use `SELECT COUNT(*)` or `EXPLAIN` where feasible.
  - Mark each call site as Pattern A, B, or C.
- Add `LIMIT` to any query intended to be bounded (Pattern B).
- Current status: blocked because `.codeintel/duckdb.db` is empty (no tables).
  Re-run once the database is populated.

### Phase 1: Introduce streaming helpers (complete)

- Implement the helper additions in `query_results.py`.
- Implement the repository helper in `repositories/base.py`.
- Add a small helper for tuple iteration to avoid changing downstream logic.

### Phase 2: Convert build analytics + graphs (largest volume, complete)

- Replace all `fetchall()` in build analytics and graph modules with streaming
  (Pattern A).
- Where an aggregation is needed, reduce per batch to keep memory bounded.

### Phase 3: Convert storage and tracking (complete)

- Switch storage metadata/tracking to Pattern A or B based on bounds.
- Replace all raw `cursor.fetchall()` usage with streaming helpers.

### Phase 4: Convert CLI and ingestion (complete)

- For CLI list outputs, stream and print per batch (Pattern A).
- For ingestion detection, stream and compute in batches (Pattern A).

### Phase 5: Guardrail + validation (remaining)

- Re-run `tools.guardrails` and `tools.quality_report`.
- Add one targeted test that ensures a large table uses streaming (no
  `fetchall()`).
- Optionally extend the guardrail to flag `relation.arrow()`/`relation.pl()`
  on unbounded queries if you want stricter enforcement.

## Current completion summary

- `fetchall()` usage eliminated in `src/` (all call sites converted).
- Streaming helpers added in `codeintel.storage.query_results` plus repository
  support in `codeintel.storage.repositories.base`.
- Build analytics, graphs, storage/tracking, CLI, and ingestion conversions
  completed using `fetch_record_batch` + streaming helpers.
- Phase 0 data validation blocked until `.codeintel/duckdb.db` has tables.
- Phase 5 validation and type-safety cleanup still required (pyright errors on
  object-typed tuples).

## Remaining work (type safety + validation)

### A) Resolve pyright/pyrefly type errors caused by `tuple[object, ...]`

Most converted loops now iterate `iter_tuples_from_arrow_reader(...)`, which
returns `tuple[object, ...]`. Pyright flags direct `int(...)`/`float(...)`
coercions and literal assignments. Resolve by adding explicit coercion helpers
or typed adapters.

Recommended approach (best-in-class, minimal churn):

1) Extend `codeintel.storage.query_results` with small coercion helpers:
   - `coerce_str(value, *, ctx)` -> `str`
   - `coerce_optional_str(value, *, ctx)` -> `str | None`
   - `coerce_datetime(value, *, ctx)` -> `datetime`
   - `coerce_optional_datetime(value, *, ctx)` -> `datetime | None`
   - `coerce_literal(value, *, ctx, allowed)` -> validated `Literal`
2) For repetitive row shapes, add localized row-parsing helpers in each module
   (e.g., `_parse_row(...)`) that apply `coerce_*` calls once and return typed
   tuples or dataclasses.
3) Avoid unchecked `cast` unless the DB schema guarantees the type and the
   value is already normalized (use `normalize_decimal_id`, `coerce_int`,
   `coerce_float` where possible).

Hotspots to address (non-exhaustive):

- `src/codeintel/build/analytics/cfg_dfg/helpers.py`
  - Use `normalize_decimal_id` for GOID, and cast `block_id` to `str | int | None`
    before calling `parse_block_idx`.
- `src/codeintel/build/analytics/cfg_dfg/cfg_core.py`
- `src/codeintel/build/analytics/cfg_dfg/dfg_core.py`
  - Replace direct `int(...)`/`float(...)` calls on `object` with `coerce_int`
    or `coerce_float`.
- `src/codeintel/build/analytics/compute/data_models/usage.py`
  - Coerce numeric fields; validate `_parse_param_types` input as `str | dict`.
- `src/codeintel/build/analytics/entrypoints/core.py`
  - Use `coerce_int` for counts and `coerce_float` for rates; avoid bare `int(...)`.
- `src/codeintel/build/analytics/profiles/graph_features.py`
- `src/codeintel/build/analytics/testing/profiles/builder.py`
  - Replace numeric casts with `coerce_*` helpers.
- `src/codeintel/build/graphs/engine/views.py`
  - Convert row values via `coerce_*` before `module_attrs_from_row(...)` and
    avoid `int(...)` on `object`.
- `src/codeintel/build/graphs/validation/checks/anomaly.py`
- `src/codeintel/build/graphs/validation/checks/database.py`
  - Coerce numeric metrics explicitly before comparisons.
- `src/codeintel/storage/duckdb_policy_backend.py`
  - Ensure `columns` is a `list[str]` (e.g., `[str(row[1]) for row in ...]`) and
    filter `None` columns before passing to `_build_insert`.
- `src/codeintel/storage/tracking/asset_tracking.py`
- `src/codeintel/storage/tracking/run_tracking.py`
  - Coerce timestamps using `coerce_optional_datetime`.
  - Validate `PipelineStatus`, `StepStatus`, and `ModuleKind` via a small
    literal parser (e.g., `parse_pipeline_status(value)`).

### B) Finish validation gates

- Re-run `uv run python -m tools.guardrails`.
- Re-run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- If Phase 0 is still blocked, capture a note in the report and proceed with
  test-only validation; otherwise complete the boundedness review.

### C) Optional hardening (recommended)

- Add a small test that verifies a known large query path uses
  `fetch_record_batch` (no `fetchall()`), using a generated DuckDB table.
- Consider tightening the guardrail to flag `relation.arrow()` or `relation.pl()`
  on unbounded queries if you want to make streaming unavoidable.

## File-by-file mapping (recommended pattern + minimal change)

Legend:
- Pattern A = stream rows (Arrow batches)
- Pattern B = scalar/bounded (fetchone/fetchmany + LIMIT)
- Pattern C = materialize to Parquet/table

### Build analytics (Pattern A unless noted)

- `src/codeintel/build/analytics/cfg_dfg/cfg_core.py`
  - Pattern A.
  - Replace `relation.fetchall()` with `fetch_record_batch` and iterate.
- `src/codeintel/build/analytics/cfg_dfg/dfg_core.py`
  - Pattern A.
  - Replace `fetchall()` with streaming reader and batch reduction.
- `src/codeintel/build/analytics/cfg_dfg/helpers.py`
  - Pattern A.
  - Replace `fetchall()` with streaming reader.
- `src/codeintel/build/analytics/compute/data_models/usage.py`
  - Pattern A.
  - Use `fetch_record_batch` and update loops to handle batches.
- `src/codeintel/build/analytics/dependencies/core.py`
  - Pattern A.
  - Convert to streaming reader; reduce per batch.
- `src/codeintel/build/analytics/entrypoints/core.py`
  - Pattern A.
  - Convert `fetchall()` to streaming reader; keep logic per batch.
- `src/codeintel/build/analytics/functions/function_effects.py`
  - Pattern A.
  - Stream rows; avoid list materialization.
- `src/codeintel/build/analytics/graphs/config_data_flow.py`
  - Pattern A.
  - Stream rows and build maps incrementally.
- `src/codeintel/build/analytics/graphs/subsystem_agreement.py`
  - Pattern A.
  - Stream rows; update counters per batch.
- `src/codeintel/build/analytics/profiles/graph_features.py`
  - Pattern A.
  - Replace loop over `fetchall()` with streaming reader.
- `src/codeintel/build/analytics/semantic_roles/core.py`
  - Pattern A.
  - Replace all `fetchall()` uses with streaming.
- `src/codeintel/build/analytics/subsystems/affinity.py`
  - Pattern A.
  - Stream rows for `core.modules` and affinity edges.
- `src/codeintel/build/analytics/subsystems/cache.py`
  - Pattern A.
  - Stream rows; update cache per batch.
- `src/codeintel/build/analytics/subsystems/risk.py`
  - Pattern A.
  - Stream rows; update aggregate per batch.
- `src/codeintel/build/analytics/testing/behavioral/tags.py`
  - Pattern A.
  - Stream rows; compute tags per batch.
- `src/codeintel/build/analytics/testing/compute.py`
  - Pattern A.
  - Stream rows; update compute outputs per batch.
- `src/codeintel/build/analytics/testing/coverage/edges.py`
  - Pattern A.
  - Stream tests/goids rows; prefer DuckDB join where possible.
- `src/codeintel/build/analytics/testing/profiles/builder.py`
  - Pattern A (candidate for Pattern C if output is persisted).
  - Stream rows; if only used for artifacts, switch to `to_parquet`.

### Build graphs (Pattern A unless noted)

- `src/codeintel/build/graphs/engine/views.py`
  - Pattern A.
  - Replace `fetchall()` with streaming; build graphs per batch.
- `src/codeintel/build/graphs/validation/checks/anomaly.py`
  - Pattern A (B if query is limited).
  - Replace `fetchall()` with reader; short-circuit on first anomaly.
- `src/codeintel/build/graphs/validation/checks/database.py`
  - Pattern A (B if query is limited).
  - Replace `fetchall()` with reader; stream stats.

### Storage and metadata

- `src/codeintel/storage/datasets/registry.py`
  - Pattern A.
  - Stream rows and build registry incrementally.
- `src/codeintel/storage/duckdb_policy_backend.py`
  - Pattern A (B for bounded lookups).
  - Replace each `fetchall()` with streaming reader.
- `src/codeintel/storage/gateway/__init__.py`
  - Pattern B (doc example only).
  - Update example to `fetch_record_batch` or `fetchone`.
- `src/codeintel/storage/helpers/module_index.py`
  - Pattern A.
  - Stream rows into the module map; avoid full list.
- `src/codeintel/storage/metadata/meta_catalog.py`
  - Pattern B.
  - Replace `fetchall()` on `PRAGMA database_list` with `fetchmany(n)`.
- `src/codeintel/storage/metadata/sync.py`
  - Pattern A.
  - Stream lineage rows and build dict incrementally.
- `src/codeintel/storage/repositories/dataflow.py`
  - Pattern A (B if datasets are small and bounded).
  - Introduce repository streaming helper; avoid `cursor.fetchall()`.
- `src/codeintel/storage/schema/ddl.py`
  - Pattern B.
  - Replace `fetchall()` with `fetchmany` and explicit `LIMIT` if needed.
- `src/codeintel/storage/tracking/asset_tracking.py`
  - Pattern A (B where `LIMIT` is already used).
  - Replace `fetchall()` with streaming/bounded helpers.
- `src/codeintel/storage/tracking/build_tracking.py`
  - Mixed: Pattern A for unbounded queries, Pattern B for `LIMIT` queries.
  - Replace `fetchall()` with reader or `fetchmany(limit)`.
- `src/codeintel/storage/tracking/run_tracking.py`
  - Pattern B (appears to be bounded lookups).
  - Replace `fetchall()` with `fetchmany` or `fetchone`.
- `src/codeintel/storage/tracking/schema_catalog.py`
  - Mixed: Pattern A for inventory scans, Pattern B for `limit` lookups.
  - Replace `fetchall()` with reader and reduce per batch.

### CLI

- `src/codeintel/cli/handlers/storage.py`
  - Pattern A (safe for large metadata tables).
  - Stream rows into sets/lists; avoid full list.
- `src/codeintel/cli/handlers/build.py`
  - Pattern A (B if you add `LIMIT` to asset list).
  - Stream asset rows; build `assets_to_show` incrementally.

### Ingestion

- `src/codeintel/ingestion/adapters/duckdb_storage.py`
  - Pattern A.
  - Replace `fetchall()` with `fetch_record_batch`.
- `src/codeintel/ingestion/adapters/hash_change_detection.py`
  - Pattern A.
  - Stream rows and compute hashes per batch.

## Validation checklist

- Guardrails: `uv run python -m tools.guardrails`.
- Quality report: `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- Focused tests:
  - Any build analytics tests that exercise the migrated modules.
  - Storage tracking tests (asset/build/run/schema).
  - CLI handlers (smoke-level).

## Acceptance criteria

- Zero `fetchall()` usage in source code (including examples).
- No change in output semantics for analytics or tracking paths.
- Streaming batch processing verified on `.codeintel/duckdb.db`.
- Guardrails and quality report pass cleanly.
