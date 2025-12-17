# Build Module — Further Consolidation & Standardization Plan

This plan is a follow-up refinement scope for `src/codeintel/build` after the Hamilton-first
consolidation work. It focuses on **eliminating remaining duplication**, **standardizing
approaches**, and **hardening correctness/security** while keeping the system shippable in small
slices.

## Scope (workstreams)

This plan covers the following consolidation opportunities:

1. **Unify row-count/introspection helpers** across build + storage query utilities.
2. **Harden schema inference SQL** to avoid unsafe string construction.
3. **Unify exporter architecture** (JSONL + Parquet) behind a single export runner.
4. **Centralize snapshot filtering utilities** for Ibis expressions across build code.
5. **Complete TargetSystem convergence** by retiring overlapping catalog/registry entrypoints.
6. **Standardize compute-result dataclasses** used by executor-style templates.
7. **Consolidate materializer metadata schema** (savers → metadata dict → TargetRunRecord).
8. **Add tag-consistency guardrails** for Hamilton node observability invariants.

## Principles

- Prefer **one canonical module** per concept; keep wrappers only as short-lived migration shims.
- Keep public APIs stable during migration; delete deprecated code as soon as callsites are migrated.
- Make correctness/contract drift failures **loud** (raise with actionable errors).
- Keep every slice passing the project acceptance gates.

## Acceptance gates (run after each slice)

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

## Recommended execution order

This sequence minimizes cross-cutting churn:

1. Workstream 1 (row-count + introspection) → foundational for exporters/assets/analytics.
2. Workstream 4 (snapshot filter helpers) → removes repeated logic and clarifies semantics.
3. Workstream 2 (schema inference hardening) → security/correctness with contained blast radius.
4. Workstream 3 (export runner) → major consolidation, now unblocked by (1).
5. Workstream 6 (compute-result standardization) → low risk, enables (8).
6. Workstream 8 (tag guardrails) → tightens observability invariants.
7. Workstream 7 (materializer metadata schema) → reduces drift across savers/templates/records.
8. Workstream 5 (TargetSystem convergence) → biggest API surface change; do last.

---

# Workstream 1 — Unify Row-Count & Introspection Helpers

## Problem

Row counting and “does this table have rows for this snapshot?” exist in multiple places with
slightly different semantics and return-shape handling:

- `src/codeintel/build/storage_queries.py`
- `src/codeintel/build/exports/common.py` (`get_row_count`)
- `src/codeintel/build/assets/emitter.py` (row count coercion)
- `src/codeintel/storage/queries/safe.py` (`count_rows_for_snapshot`, `table_has_rows_for_snapshot`)

## Goal

Provide **one canonical API** for:

- `count_rows(...)` (unfiltered)
- `count_rows_for_snapshot(...)` (repo/commit filtered)
- `has_rows_for_snapshot(...)` (fast existence check)
- `coerce_count_scalar(...)` (DataFrame/Series/scalar handling)

## Proposed end-state

Create a single canonical module in storage (preferred):

- `codeintel.storage.queries.safe` owns:
  - `count_rows_for_snapshot(gateway, table_key, snapshot) -> int`
  - `table_has_rows_for_snapshot(gateway, table_key, snapshot) -> bool`
  - `coerce_scalar_int(raw) -> int`

Then:

- `codeintel.build.storage_queries` becomes a thin re-export shim (temporary), then removed.
- `codeintel.build.exports.common.get_row_count` routes through the canonical helper.
- `codeintel.build.assets.emitter` routes through the canonical helper.

## Implementation slices

### Slice 1.1 — Introduce canonical gateway-based helpers

- Add `count_rows_for_snapshot(gateway, *, table_key, snapshot)` and `has_rows_for_snapshot(...)`
  to `src/codeintel/storage/queries/safe.py`.
- Move/duplicate a single scalar-coercion helper (DataFrame/Series/scalar) into storage queries.
- Add focused unit tests for return-shape coercion.

### Slice 1.2 — Migrate build callsites

- Update:
  - `src/codeintel/build/assets/emitter.py`
  - `src/codeintel/build/exports/common.py`
  - `src/codeintel/build/storage_queries.py` (convert to shim or delete later)
  - any native targets that still do custom row counts (should be none post-refactor)
- Keep behavior consistent:
  - return `0` when empty / missing
  - raise `ValueError` when snapshot columns are missing (unless explicitly “tolerant” API).

### Slice 1.3 — Decommission duplicates

- Remove `src/codeintel/build/storage_queries.py` if no longer imported.
- Ensure all imports route to the canonical location.

## Decommission checklist

- [ ] No remaining imports of `codeintel.build.storage_queries`.
- [ ] No custom “DataFrame/Series/scalar → int” coercion in build modules.

---

# Workstream 2 — Harden Schema Inference SQL

## Problem

`src/codeintel/build/schemas/infer_duckdb.py` uses f-strings to build `DESCRIBE` statements:

- `con.execute(f"DESCRIBE {stripped_sql}")`
- `con.execute(f"DESCRIBE {schema_name}.{view_name}")`

This is brittle and can be unsafe if inputs are not tightly controlled.

## Goal

- Eliminate ad hoc string interpolation for SQL execution.
- Ensure schema inference only runs on **trusted inputs**:
  - Ibis-generated SQL, and/or
  - validated identifiers (schema/table/view).

## Proposed end-state

Option A (preferred): use DuckDB connection APIs that avoid SQL string construction:

- For views/tables: `con.table(view_key).limit(0)` then infer schema via a safe mechanism.
- For SQL: ensure SQL originates from `ibis.to_sql(...)` and disallow arbitrary user-provided SQL.

Option B: if `DESCRIBE <sql>` is required:

- Only accept SQL generated by Ibis (pass the expression instead of SQL text wherever possible).
- Refuse strings not originating from an expression compilation path.

## Implementation slices

### Slice 2.1 — Restrict SQL entrypoints

- Update `infer_table_schema_from_sql` to:
  - accept an `ir.Table` or a sealed “CompiledSql” value object rather than raw `str`.
  - refuse non-Ibis SQL sources.

### Slice 2.2 — Harden view schema inference

- Replace `DESCRIBE {schema}.{view}` with:
  - strict identifier quoting + validation, or
  - a safe DuckDB catalog query that returns column metadata.

### Slice 2.3 — Tests

- Add tests for:
  - rejecting unsafe identifiers
  - stable inference for normal views and Ibis expressions

---

# Workstream 3 — Exporter Architecture Unification

## Problem

Parquet and JSONL exporters share a large amount of logic:

- dataset selection / registry validation
- incremental markers
- per-dataset manifest writing
- audit logging
- validation profiles

But format-specific code paths drift and duplicate behavior.

## Goal

Create one export “runner” that:

- resolves targets once
- drives incremental/manifest/audit/validation consistently
- delegates only the write-format specifics

## Proposed end-state

Introduce:

- `codeintel.build.exports.runner` (new)
  - `run_exports(gateway, *, format, options) -> ExportManifestData`
  - `export_target(target, writer, ...) -> Path | None`
- Format writers:
  - `ParquetWriter`
  - `JsonlWriter`
  - (future) `ArrowIpcWriter`, `CsvWriter`, etc.

## Implementation slices

### Slice 3.1 — Introduce writer interface

- Define a protocol for format writers:
  - `write_table(gateway, table_key, output_path, *, options) -> WriteStats`
  - where `WriteStats` includes `rows`, `duration_s`, and any hashes.

### Slice 3.2 — Move common control flow into runner

- Extract shared logic from:
  - `src/codeintel/build/exports/parquet.py`
  - `src/codeintel/build/exports/jsonl.py`
- Runner handles:
  - incremental marker read/write
  - manifest writing
  - validation hooks
  - audit logging

### Slice 3.3 — Migrate existing entrypoints

- Keep `export_*` functions as wrappers that call the runner with the proper writer.

### Slice 3.4 — Delete duplication

- Remove duplicated “_export_dataset_*” logic when wrappers are thin.

## Decommission checklist

- [ ] No duplicate incremental/manifest logic across exporters.
- [ ] One audit pathway (one schema, one logger).

---

# Workstream 4 — Centralize Snapshot Filtering Helpers (Ibis)

## Problem

Many places build “repo/commit” predicates independently, risking subtle drift:

- different NULL behavior
- missing columns handling
- inconsistent `and_predicates` usage

## Goal

Define a single helper with strict semantics:

- `filter_relation_for_snapshot(relation, snapshot) -> relation`

## Proposed end-state

Add (or extend) a canonical helper module:

- `codeintel.storage.queries.safe` or `codeintel.storage.ibis_helpers` (pick one)
  - `snapshot_predicate(table, snapshot) -> ibis predicate`
  - `filter_for_snapshot(table, snapshot) -> table`

Then migrate callsites in:

- exporters
- asset emitter
- row count helpers
- schema inference that needs snapshot views (if any)

---

# Workstream 5 — TargetSystem Convergence (Retire Overlapping APIs)

## Problem

Multiple entrypoints compete for “the way” to get:

- target metadata
- dependency graph
- runtime mappings

Now that `TargetSystem` exists, keeping multiple APIs increases drift risk.

## Goal

Make `TargetSystem` the single entrypoint and decommission overlaps:

- `target_catalog` indexing responsibilities
- `target_registry` wrapper responsibilities
- `registry.get_target_graph` as a primary API (keep as a thin wrapper temporarily)

## Proposed end-state

- `codeintel.build.target_system.load_target_system()` is canonical.
- Backwards-compat wrappers exist only briefly, then are removed.

## Implementation slices

### Slice 5.1 — Migrate internal callsites

- Update all internal imports to prefer `load_target_system().graph` and catalog lookups.

### Slice 5.2 — Deprecation shim window

- Keep `codeintel.build.registry.get_target_graph()` as a wrapper (temporary).
- Add clear docstrings pointing to `TargetSystem`.

### Slice 5.3 — Delete old modules

- Remove redundant modules once no longer imported:
  - `codeintel.build.target_catalog` (or reduce to pure data types if still needed)
  - `codeintel.build.target_registry`
  - `codeintel.build.registry` helpers that duplicate `TargetSystem`

---

# Workstream 6 — Standardize Compute-Result Dataclasses

## Problem

The “executor materialize” pattern is widespread:

- compute node returns `{success, table_counts, error}`
- materialize converts to `TargetRunRecord`

But each module redefines a slightly different result dataclass.

## Goal

Define a canonical result type to reduce boilerplate and standardize semantics.

## Proposed end-state

Introduce:

- `codeintel.build.hamilton.types.ExecutionResult` (name TBD)
  - `success: bool`
  - `table_counts: dict[str, int]`
  - `error: str | None`
  - constructors like `ok(counts)` / `fail(error)` to standardize message shaping.

Then update `materialize_template.executor_materialize` to accept that type (or keep `Any`
for Hamilton matching, but use the canonical type in all native code).

---

# Workstream 7 — Consolidate Materializer Metadata Schema

## Problem

Savers emit metadata dicts; record builders parse dicts; templates and native modules rely on
implicit keys. Drift here creates hard-to-debug failures.

## Goal

Define a single typed metadata contract and ensure:

- all savers emit the same schema (key names + types)
- parsing lives in one place
- failures are explicit (missing keys / wrong types)

## Proposed end-state

- `codeintel.build.hamilton.materializers.types` becomes the single schema owner:
  - `TableMaterializationMetadata`
  - `ArtifactMaterializationMetadata`
  - `MultiMaterializationMetadata`
- `native/materialization_records.py` parses only those typed structures.
- Savers construct those types then convert to dict only at the Hamilton boundary if required.

---

# Workstream 8 — Tag Consistency Guardrails

## Problem

Observability relies on Hamilton node tags (`domain`, `target`, `node_type`). Missing tags lead to
telemetry/contract enforcement gaps that are silent.

## Goal

Ensure every node that participates in build execution:

- has consistent tags
- uses canonical keys (no drift)
- fails fast (or logs loudly) when tags are missing on nodes that should have them

## Proposed end-state

- Add a lightweight guardrail check invoked by quality tooling:
  - enumerate all driver nodes from `build_driver()`
  - assert tag presence/shape for nodes matching naming conventions (`t__*`, `q__*`, materializers)

## Implementation slices

### Slice 8.1 — Define invariants

- Document tag requirements:
  - required keys per node category
  - allowed values for `node_type`

### Slice 8.2 — Implement checker

- Add a checker module, e.g. `codeintel.build.hamilton.observability.tag_checks`.
- Integrate into `tools.guardrails` (or an adjacent check runner).

### Slice 8.3 — Fix any violations and delete old patterns

- Ensure native modules and templates comply.
- Remove any ad hoc tag keys in code.

---

# Deliverables checklist

- [ ] One canonical row-count/introspection API; duplicates removed.
- [ ] Schema inference no longer uses unsafe interpolated SQL.
- [ ] One export runner with pluggable writers; duplication removed.
- [ ] One canonical snapshot filtering helper for Ibis.
- [ ] TargetSystem is the canonical entrypoint; overlapping APIs retired.
- [ ] Canonical compute result type for executor materialization.
- [ ] Single typed materialization metadata schema across savers + parsers.
- [ ] Tag consistency guardrail integrated into quality checks.

