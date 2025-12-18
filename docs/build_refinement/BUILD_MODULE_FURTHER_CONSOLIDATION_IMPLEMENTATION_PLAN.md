# Build Module — Further Consolidation & Standardization Plan

This plan is a follow-up refinement scope for `src/codeintel/build` after the Hamilton-first
consolidation work. It focuses on **eliminating remaining duplication**, **standardizing
approaches**, and **hardening correctness/security** while keeping the system shippable in small
slices.

## Scope (workstreams)

This plan covers the following consolidation opportunities (updated to include follow-on findings
from `docs/data_operations_refinement/holistic_data_operations_enhancement_plan.md` and a deeper
review of `src/codeintel/build` and adjacent storage/serving surfaces):

1. **Unify row-count/introspection helpers** across build + storage query utilities.
2. **Centralize snapshot filtering utilities** for Ibis expressions across build + storage code.
3. **Harden schema inference SQL** to avoid unsafe string construction and raw SQL acceptance.
4. **Unify exporter architecture** behind Arrow-first streaming writers (JSONL + Parquet).
5. **Eliminate memory-buffering export artifacts** in Hamilton native export targets.
6. **Standardize table-key parsing/validation** via one canonical helper/value object.
7. **Add build-engine/plugin versioning to cache keys** (manifest/input-hash correctness).
8. **Introduce QueryTemplate + SQLGlot AST fingerprinting** (typed params + deterministic SQL).
9. **Unify export audit/tracking surfaces** under one canonical schema and query API.
10. **Transactional snapshot replace semantics** across snapshot-scoped write paths.
11. **Complete TargetSystem convergence** by retiring overlapping catalog/registry/metadata surfaces.
12. **Standardize compute-result dataclasses** used by executor-style templates and native targets.
13. **Consolidate materializer metadata schema** (savers/IO adapters → typed schema → TargetRunRecord).
14. **Add tag-consistency guardrails** for Hamilton node observability + introspection invariants.
15. **Decommission the legacy plugin execution stack** once Hamilton is canonical.

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

1. Workstreams 1–2 (row-count + snapshot filtering) → foundational primitives used everywhere.
2. Workstream 6 (table-key parsing) → reduces drift across all subsequent refactors.
3. Workstream 3 (schema inference hardening) → contained security/correctness.
4. Workstreams 4–5 (Arrow-first exporters + native export buffering removal) → biggest perf win.
5. Workstream 7 (engine/plugin versioning) → correctness for caching/manifest reuse.
6. Workstream 8 (QueryTemplate + AST fingerprinting) → unlocks further data-ops consolidation.
7. Workstream 9 (audit/tracking unification) → observability consolidation once exports stabilize.
8. Workstream 10 (transactional snapshot replace) → requires canonical write surface.
9. Workstream 11 (TargetSystem convergence) → broad API surface, do late.
10. Workstreams 12–14 (compute results, metadata schema, tag guardrails) → tighten contracts.
11. Workstream 15 (legacy execution stack decommission) → last; only after callsites migrate.

---

# Workstream 1 — Unify Row-Count & Introspection Helpers

## Problem

Row counting and “does this table have rows for this snapshot?” exist in multiple places with
slightly different semantics and return-shape handling:

- `src/codeintel/build/storage_queries.py`
- `src/codeintel/build/exports/common.py` (`get_row_count`)
- `src/codeintel/build/assets/emitter.py` (row count coercion)
- `src/codeintel/storage/queries/safe.py` (`safe_count`, `safe_count_with_scope`,
  `count_rows_for_snapshot`, `table_has_rows_for_snapshot`)

## Goal

Provide **one canonical API** for:

- `count_rows(...)` (unfiltered; tolerant + strict variants)
- `count_rows_for_snapshot(...)` (repo/commit filtered; tolerant + strict variants)
- `has_rows_for_snapshot(...)` (fast existence check)
- `coerce_count_scalar(...)` (DataFrame/Series/scalar handling)
- one canonical “missing snapshot columns” behavior (explicit error vs tolerant `None`)

## Proposed end-state

Create a single canonical module in storage (preferred):

- `codeintel.storage.queries.safe` owns all row-count primitives:
  - `count_rows(gateway, table_key) -> int` (strict; raises on failure)
  - `try_count_rows(gateway, table_key) -> int | None` (tolerant; never raises)
  - `count_rows_for_snapshot(gateway, table_key, snapshot) -> int` (strict)
  - `try_count_rows_for_snapshot(gateway, table_key, snapshot) -> int | None` (tolerant)
  - `has_rows_for_snapshot(gateway, table_key, snapshot) -> bool`
  - `coerce_scalar_int(raw) -> int` (single robust coercion helper)

Then:

- `codeintel.build.storage_queries` becomes a thin re-export shim (temporary), then removed.
- `codeintel.build.exports.common.get_row_count` routes through the canonical helper.
- `codeintel.build.assets.emitter` routes through the canonical helper.

## Implementation slices

### Slice 1.1 — Introduce canonical gateway-based helpers

- Implement `coerce_scalar_int(...)` in `src/codeintel/storage/queries/safe.py` and refactor:
  - `safe_count(...)` and `safe_count_with_scope(...)` to use it (stop assuming scalar-only).
  - `count_rows_for_snapshot(...)` to offer gateway-based and connection-based variants, with a
    clear preference for gateway-based APIs at the build layer.
- Split strict vs tolerant APIs explicitly (no ambiguous “returns 0 on error” behavior).
- Add focused unit tests for coercion (scalar, 1x1 DataFrame, Series, empty shapes).

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
- [ ] Exporters and asset emitter use the same canonical tolerant/strict semantics.

---

# Workstream 2 — Centralize Snapshot Filtering Helpers (Ibis)

## Problem

Many places build “repo/commit” predicates independently, risking subtle drift:

- different NULL behavior
- missing columns handling
- inconsistent `and_predicates` usage
- parallel helper implementations in build vs storage

## Goal

Define one canonical helper set with strict semantics:

- `snapshot_predicate(table, snapshot) -> ibis predicate`
- `filter_for_snapshot(table, snapshot) -> table`

## Proposed end-state

One canonical module owns snapshot filtering:

- Preferred: `codeintel.storage.queries.safe` (or an adjacent `codeintel.storage.ibis_helpers`)
  - `snapshot_predicate(table, snapshot) -> it.BooleanValue`
  - `filter_for_snapshot(table, snapshot) -> it.Table`

And:

- `src/codeintel/build/hamilton/native/ibis_helpers.py` becomes a thin shim (or is removed and
  callsites updated to storage-level helper).
- `src/codeintel/storage/queries/safe.py:safe_count_with_scope(...)` uses `filter_for_snapshot(...)`
  rather than re-implementing the predicate.

## Implementation slices

### Slice 2.1 — Choose canonical home and API surface

- Decide the canonical module (`codeintel.storage.queries.safe` vs a dedicated helper module).
- Keep `filter_for_snapshot(...)` signature stable across build + storage.

### Slice 2.2 — Migrate callsites and delete duplicates

- Migrate callsites in:
  - build exporters and assets
  - `src/codeintel/storage/queries/safe.py` scoped helpers
  - native targets that still inline snapshot filters (e.g., export targets)
- Delete/retire duplicates once imports are migrated.

### Slice 2.3 — Guardrails and tests

- Add tests for:
  - missing snapshot columns (strict vs tolerant behavior)
  - correct predicate composition

## Decommission checklist

- [ ] Only one implementation of snapshot predicate/filtering remains.
- [ ] No inline `(tbl.repo == snapshot.repo) & (tbl.commit == snapshot.commit)` outside the helper
      except in tests.

---

# Workstream 3 — Harden Schema Inference SQL (No Raw SQL Acceptance)

## Problem

`src/codeintel/build/schemas/infer_duckdb.py` uses f-strings to build `DESCRIBE` statements:

- `con.execute(f"DESCRIBE {stripped_sql}")`
- `con.execute(f"DESCRIBE {schema_name}.{view_name}")`

This is brittle and can be unsafe if inputs are not tightly controlled.

## Goal

- Eliminate ad hoc string interpolation for SQL execution.
- Ensure schema inference only runs on **trusted inputs**:
  - Ibis expressions, and/or
  - validated identifiers (schema/table/view).
- Remove any public API that accepts arbitrary `str` SQL.

## Proposed end-state

Option A (preferred): infer schema without `DESCRIBE <sql>`:

- For views/tables: use `con.table(view_key)` and retrieve column names/types from DuckDB relation
  metadata or catalog queries with validated identifiers.
- For expressions: accept `ibis.expr.types.Table` and use a sealed compilation path.

Option B (fallback): if DuckDB requires `DESCRIBE <sql>` for expression inference:

- Only accept SQL produced by our own compilation path (Ibis → SQLGlot AST → SQL).
- Parse and validate the AST:
  - exactly one statement
  - SELECT-only (no DDL/DML)
  - no trailing statements (no `;`)

## Implementation slices

### Slice 3.1 — Restrict entrypoints to expressions + identifiers

- Replace `infer_table_schema_from_sql(sql: str, ...)` with:
  - `infer_table_schema_from_ibis(expr: ir.Table, ...)` as the canonical entrypoint.
  - (optional) `infer_table_schema_from_compiled(compiled: CompiledQuery, ...)` where
    `CompiledQuery` is a sealed value object built only by the gateway/compiler.

### Slice 3.2 — Harden view schema inference (validated identifiers only)

- Replace `DESCRIBE {schema}.{view}` with catalog-backed metadata retrieval:
  - `information_schema.columns` scoped by schema/table
  - or `PRAGMA table_info` with strict identifier validation and quoting

### Slice 3.3 — Tests

- Add tests for:
  - rejecting unsafe identifiers
  - rejecting multi-statement SQL in any remaining fallback
  - stable inference for normal views and Ibis expressions

## Decommission checklist

- [ ] No public API accepts raw SQL strings for inference.
- [ ] No `con.execute(f"...")` remains in build schema inference.

---

# Workstream 4 — Exporter Architecture Unification (Arrow-First Streaming)

## Problem

Parquet and JSONL exporters share a large amount of logic:

- dataset selection / registry validation
- incremental markers
- per-dataset manifest writing
- audit logging
- validation profiles

But format-specific code paths drift and duplicate behavior.

Additionally, several export pathways still buffer via pandas or large in-memory payloads, which
conflicts with the Arrow-first streaming direction in
`docs/data_operations_refinement/holistic_data_operations_enhancement_plan.md`.

## Goal

Create one export “runner” that:

- resolves targets once
- drives incremental/manifest/audit/validation consistently
- delegates only the write-format specifics
- uses Arrow record-batch streaming end-to-end (bounded memory)

## Proposed end-state

Refactor the existing runner:

- `src/codeintel/build/exports/runner.py` becomes the canonical export orchestrator:
  - keep `run_validated_exports(...)` as the stable entrypoint
  - add a lower-level `run_exports(...) -> list[Path]` and `export_target(...) -> Path | None`
  - centralize dataset selection, incremental markers, manifest writing, validation, and auditing
- introduce (or reuse) a small Arrow streaming helper module that can also be shared with serving:
  - e.g., `codeintel.storage.arrow_streaming` or `codeintel.build.exports.streaming`
  - owns `iter_record_batches(relation, batch_size)` and any “rows written” accounting
- Format writers:
  - `ParquetWriter` (Arrow batches → Parquet file)
  - `JsonlWriter` (Arrow batches → JSONL file)
  - (future) `ArrowIpcWriter`, `CsvWriter`, etc.

## Implementation slices

### Slice 4.1 — Introduce writer interface (Arrow batches)

- Define a protocol for format writers:
  - `write_table(gateway, table_key, output_path, *, options) -> WriteStats`
  - where `WriteStats` includes `rows`, `duration_s`, and any hashes.
- Writers must stream from `DuckDBRelation.fetch_record_batch(batch_size)` or equivalent.

### Slice 4.2 — Move common control flow into runner

- Extract shared logic from:
  - `src/codeintel/build/exports/parquet.py`
  - `src/codeintel/build/exports/jsonl.py`
- Runner handles:
  - incremental marker read/write
  - manifest writing
  - validation hooks
  - audit logging

### Slice 4.3 — Migrate existing entrypoints

- Keep `export_*` functions as wrappers that call the runner with the proper writer.

### Slice 4.4 — Delete duplication

- Remove duplicated “_export_dataset_*” logic when wrappers are thin.

## Decommission checklist

- [ ] No duplicate incremental/manifest logic across exporters.
- [ ] One audit pathway (one schema, one logger).
- [ ] No pandas buffering in export pathways (`rel.df()`/`to_dict(...)`) for large tables.

---

# Workstream 5 — Eliminate Memory-Buffering Export Artifacts (Native Targets)

## Problem

Native Hamilton export targets still buffer entire datasets in memory:

- `src/codeintel/build/hamilton/native/export/export_targets.py`
  - JSONL export loads tables into pandas, then `to_dict(orient="records")`, then builds one giant
    JSONL string.
  - Parquet export loads into pandas then writes bytes into a `BytesIO` and returns the full
    payload as `bytes`.

## Goal

Make native export targets:

- Arrow-first and streaming (bounded memory).
- “DAG-visible I/O” without returning giant payloads from compute nodes.
- Unified with the canonical export runner so logic doesn’t drift.

## Proposed end-state

Option A (preferred): native export targets become thin wrappers around the unified export runner:

- Replace `export_jsonl__content(...) -> str` with a target that triggers runner output and records
  file artifacts written by the runner.
- Replace `export_parquet__bytes(...) -> bytes` similarly.

Option B: extend `FileArtifactSaver` to support streaming writes:

- Introduce a saver mode that accepts an iterator/reader and writes directly to disk.
- Keep compute nodes producing lightweight objects (paths, stats, manifests), not payload bytes.

## Implementation slices

### Slice 5.1 — Pick integration strategy and update contracts

- Decide whether export artifacts should be:
  - “per-dataset files” (preferred; aligns with `src/codeintel/build/exports/*`), or
  - a single combined artifact (legacy behavior; less scalable).
- Update `TARGET_SPECS` in native export modules accordingly.

### Slice 5.2 — Implement Arrow streaming and remove pandas buffering

- Ensure both JSONL and Parquet exports stream from record batches.
- Remove `pandas` from native export targets where it exists only for export serialization.

### Slice 5.3 — Align with incremental markers/manifests

- Reuse the unified export runner’s incremental + manifest logic.
- Ensure build manifest records include written artifact paths and stats.

## Decommission checklist

- [ ] No export target returns giant `str|bytes` payloads for large tables.
- [ ] No `to_dict(orient="records")` in export paths outside tests.

---

# Workstream 6 — Standardize Table-Key Parsing/Validation

## Problem

Table-key parsing exists in multiple forms (`table_key.split(".", 1)` vs helper functions) and can
silently diverge (missing validation, inconsistent error messages, inconsistent handling of edge
cases).

## Goal

Ensure *all* parsing/validation of `schema.table` keys goes through one canonical API:

- `split_table_key(table_key) -> (schema, table)`
- optional typed value object (e.g., `TableKey`) for internal use

## Proposed end-state

- Canonical parsing/validation lives in `codeintel.storage.helpers.table_key` (or a new
  `codeintel.storage.helpers.table_key_types` if a value object is introduced).
- Build/Hamilton code does not parse table keys manually.

## Implementation slices

### Slice 6.1 — Inventory and replace manual parsing

- Replace direct `.split(".", 1)` parsing in build modules with `split_table_key(...)`:
  - `src/codeintel/build/context_base.py`
  - `src/codeintel/build/hamilton/io/dataset_ref.py`
  - `src/codeintel/build/hamilton/nodes/support_factory.py`
  - `src/codeintel/build/hamilton/native/target_spec_helpers.py`
  - any other callsites found by search

### Slice 6.2 — Optional value object

- If beneficial, introduce `TableKey(schema: str, table: str)` with:
  - `from_str(...)`
  - `__str__` roundtrip
  - validation rules in one place

## Decommission checklist

- [ ] No manual `table_key.split(".", 1)` remains outside storage helper modules/tests.

---

# Workstream 7 — Add Build-Engine/Plugin Versioning To Cache Keys

## Problem

Build caching currently keys input hashes primarily on the analyzed repo/commit plus dependency
manifests and options:

- `src/codeintel/build/hashing.py:compute_input_hash(...)`

This can produce incorrect reuse when the CodeIntel build engine changes (schema logic, query
logic, export normalization, etc.) without changing the analyzed snapshot. Separately, several
surfaces still emit placeholder plugin versions (e.g., `"0.0.0"`) which reduces the usefulness of
constraint/lineage metadata.

## Goal

- Ensure manifests cannot be reused across incompatible build-engine versions.
- Ensure “plugin identity” is consistent and versioned across:
  - run records
  - constraints/lineage metadata
  - audit/tracking surfaces

## Proposed end-state

- Add a stable build-engine version string (one canonical source) into:
  - input hash computation
  - run records/manifests
- Standardize plugin version propagation so `plugin_constraints` can report meaningful versions.

## Implementation slices

### Slice 7.1 — Define build-engine version source

- Pick one canonical source (examples):
  - package version (recommended)
  - explicit constant in `codeintel.build.version`
- Ensure it is available without expensive imports.

### Slice 7.2 — Integrate into hashing and manifests

- Update `compute_input_hash*` to include the build-engine version in the hashed material.
- Persist the version alongside manifests/run records for debugging.

### Slice 7.3 — Standardize plugin identity/version across Hamilton-native and template targets

- Ensure native targets (`native:*`) and template targets (`template:*`) use a consistent identity
  model, and supply meaningful versions where possible.

## Decommission checklist

- [ ] Input hashes differ when build-engine version differs.
- [ ] No `"0.0.0"` placeholder versions remain where a real version is available.

---

# Workstream 8 — Introduce QueryTemplate + SQLGlot AST Fingerprinting

## Problem

Many data operations compile Ibis expressions directly to SQL strings. This loses opportunities for:

- typed parameterization (`ibis.param(...)`) and binding
- query fingerprinting/caching
- lineage extraction
- deterministic “semantic diff” across query revisions

## Goal

Define a first-class query abstraction that:

- standardizes typed parameters
- surfaces a SQLGlot AST for analysis
- provides deterministic fingerprinting for caching/tracking

## Proposed end-state

- Introduce a `QueryTemplate` concept in storage (preferred):
  - holds an Ibis expression factory + parameter schema
  - compiles to SQLGlot via `IbisGateway.to_sqlglot(...)`
  - provides `fingerprint()` based on canonicalized AST + bound param types

## Implementation slices

### Slice 8.1 — Define core types

- Define:
  - `QueryTemplate` (template + param definitions)
  - `BoundQuery` (template + param values)
  - `QueryFingerprint` (string wrapper/value object)

### Slice 8.2 — Standardize compilation and fingerprinting

- Build fingerprint from:
  - SQLGlot AST canonicalization
  - parameter types/values (as appropriate for cache semantics)

### Slice 8.3 — Adopt in one concrete path first

- Start with exports or a high-frequency warehouse query path.
- Expand to other areas once stable.

## Decommission checklist

- [ ] New dynamic queries use `ibis.param(...)` rather than ad hoc `ibis.literal(...)`.
- [ ] Query fingerprinting is available for caching/observability.

---

# Workstream 9 — Unify Export Audit/Tracking Surfaces

## Problem

Export auditing currently has its own optional pathways (log file + `metadata.export_audit`) which
can drift from the canonical run tracking surfaces.

## Goal

- Ensure “what was exported” is queryable alongside build run tracking and asset tracking.
- Avoid export-specific side tables that drift in schema/semantics.

## Proposed end-state

- Export operations write a single canonical “export event” record through the storage tracking
  layer, linked to:
  - run_id (when applicable)
  - repo/commit
  - dataset/table key
  - output path + hash + duration + row count

## Implementation slices

### Slice 9.1 — Define canonical event schema

- Decide whether to:
  - extend an existing tracking table, or
  - introduce a dedicated export tracking table in the tracking subsystem

### Slice 9.2 — Migrate `exports/common.py:write_audit_entry(...)`

- Route audit writes through the tracking layer.
- Keep env-var audit log as a temporary shim if needed, then delete.

## Decommission checklist

- [ ] One canonical export audit schema exists.
- [ ] No parallel “export audit” table is written outside the tracking layer.

---

# Workstream 10 — Transactional Snapshot Replace Semantics

## Problem

Snapshot-scoped replace operations can be implemented as “delete then write” without an explicit
transaction boundary, risking partial snapshot states when failures occur.

## Goal

Make snapshot replacement atomic across all write paths that support it.

## Proposed end-state

- Provide a storage-level API such as:
  - `replace_for_snapshot(table_key, snapshot, expr|rows|df, ...)` that guarantees atomicity
  - internal transaction boundaries (BEGIN/COMMIT/ROLLBACK)
- Ensure build materializers and any snapshot-scoped writers use this API.

## Implementation slices

### Slice 10.1 — Identify snapshot replace callsites

- Inventory snapshot-scoped deletes/writes in:
  - storage warehouse layer
  - build materializers
  - any custom native targets that do replace logic

### Slice 10.2 — Implement canonical transactional replace

- Implement and test an atomic replace primitive in storage.

### Slice 10.3 — Migrate build write paths

- Update build materializers and IO adapters to use the primitive.

## Decommission checklist

- [ ] No “delete_for_snapshot + write” without an explicit transaction remains.

---

# Workstream 11 — TargetSystem Convergence (Retire Overlapping APIs)

## Problem

Multiple entrypoints compete for “the way” to get:

- target metadata
- dependency graph
- runtime mappings

Now that `TargetSystem` exists, keeping multiple APIs increases drift risk.

Additionally, several “metadata surfaces” exist in parallel:

- `src/codeintel/build/registry.py`
- `src/codeintel/build/target_registry.py`
- `src/codeintel/build/target_catalog.py`
- `src/codeintel/build/hamilton/metadata_bridge.py`
- target-derived plugin metadata in `src/codeintel/build/hamilton/contracts/schemas/plugin_constraints.py`

## Goal

Make `TargetSystem` the single entrypoint and decommission overlaps:

- `target_catalog` indexing responsibilities
- `target_registry` wrapper responsibilities
- `registry.get_target_graph` as a primary API (keep as a thin wrapper temporarily)
- converge all “plugin metadata” needs to TargetSystem’s canonical metadata model

## Proposed end-state

- `codeintel.build.target_system.load_target_system()` is canonical.
- Backwards-compat wrappers exist only briefly, then are removed.
- `CanonicalPluginMeta` (or successor) is constructed by TargetSystem, not ad hoc shims.

## Implementation slices

### Slice 11.1 — Migrate internal callsites

- Update all internal imports to prefer `load_target_system().graph` and catalog lookups.

### Slice 11.2 — Deprecation shim window

- Keep `codeintel.build.registry.get_target_graph()` as a wrapper (temporary).
- Add clear docstrings pointing to `TargetSystem`.

### Slice 11.3 — Converge metadata surfaces

- Decide a single canonical metadata representation (likely TargetSystem-owned).
- Update:
  - `metadata_bridge.from_plugin_or_target(...)`
  - `plugin_constraints._get_all_plugins_metadata(...)`
  to use TargetSystem’s metadata directly rather than re-deriving.

### Slice 11.4 — Delete old modules

- Remove redundant modules once no longer imported:
  - `codeintel.build.target_catalog` (or reduce to pure data types if still needed)
  - `codeintel.build.target_registry`
  - `codeintel.build.registry` helpers that duplicate `TargetSystem`

---

# Workstream 12 — Standardize Compute-Result Dataclasses

## Problem

The “executor materialize” pattern is widespread:

- compute node returns `{success, table_counts, error}`
- materialize converts to `TargetRunRecord`

But each module redefines a slightly different result dataclass.

Also, some compute results currently embed large payloads (notably export targets), which is both a
performance and API design smell.

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

For export targets specifically:

- Prefer eliminating bespoke compute-result dataclasses by delegating to the export runner and
  returning only lightweight stats/paths.

---

# Workstream 13 — Consolidate Materializer Metadata Schema

## Problem

Savers emit metadata dicts; record builders parse dicts; templates and native modules rely on
implicit keys. Drift here creates hard-to-debug failures.

Additionally, IO adapters such as `src/codeintel/build/hamilton/io/ibis_adapter.py` emit their own
metadata dict shapes which are not governed by a single typed schema.

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
- IO adapters either:
  - emit the same typed metadata, or
  - are made internal helpers whose metadata is converted to the canonical type before it reaches
    record builders.

---

# Workstream 14 — Tag Consistency Guardrails (Observability + Introspection)

## Problem

Observability relies on Hamilton node tags (`domain`, `target`, `node_type`). Missing tags lead to
telemetry/contract enforcement gaps that are silent.

Graph introspection also relies on tags (e.g., dataset nodes must include `table_key`, artifact
nodes must include `artifact`), and missing tags cause runtime failures or silent drift in derived
dependency/output graphs.

## Goal

Ensure every node that participates in build execution:

- has consistent tags
- uses canonical keys (no drift)
- fails fast (or logs loudly) when tags are missing on nodes that should have them

## Proposed end-state

- Add a lightweight guardrail check invoked by quality tooling:
  - enumerate all driver nodes from `build_driver()`
  - assert tag presence/shape for nodes matching naming conventions (`t__*`, `q__*`, materializers)
  - assert tag invariants required by `src/codeintel/build/hamilton/introspect.py`

## Implementation slices

### Slice 14.1 — Define invariants

- Document tag requirements:
  - required keys per node category
  - allowed values for `node_type`
  - required keys for dataset/artifact nodes used by introspection (`table_key`, `artifact`)

### Slice 14.2 — Implement checker

- Add a checker module, e.g. `codeintel.build.hamilton.observability.tag_checks`.
- Integrate into `tools.guardrails` (or an adjacent check runner).

### Slice 14.3 — Fix any violations and delete old patterns

- Ensure native modules and templates comply.
- Remove any ad hoc tag keys in code.

---

# Workstream 15 — Decommission The Legacy Plugin Execution Stack

## Problem

If Hamilton is the canonical execution path, the legacy plugin execution stack represents a large
parallel abstraction surface that can drift:

- `src/codeintel/build/context.py`
- `src/codeintel/build/context_base.py`
- `src/codeintel/build/result.py`
- `src/codeintel/build/protocols.py`

## Goal

- Remove parallel execution abstractions once they are no longer required by any active runtime.
- Keep only the minimal public API surface needed for external consumers (if any).

## Proposed end-state

- Hamilton-first execution is the only supported build execution mode.
- Legacy plugin-context APIs are removed (or retained only as trivial compatibility shims if truly
  required, with a short removal window).

## Implementation slices

### Slice 15.1 — Identify active callsites

- Inventory runtime callsites importing legacy context/protocol/result types.
- Confirm whether any external consumer requires them (CLI, serving, tests).

### Slice 15.2 — Migrate to Hamilton equivalents

- Migrate remaining usage to:
  - `BuildEnv`
  - TargetSystem
  - Hamilton IO adapters + materializers

### Slice 15.3 — Delete deprecated modules

- Delete the legacy modules once unused (no dead-code retention in design phase).

## Decommission checklist

- [ ] No production code imports legacy plugin execution modules.
- [ ] Tests use Hamilton fakes/helpers rather than legacy contexts.

---

# Deliverables checklist

- [ ] One canonical row-count/introspection API; duplicates removed.
- [ ] One canonical snapshot filtering helper for Ibis; duplicates removed.
- [ ] Schema inference no longer uses unsafe interpolated SQL.
- [ ] One export runner with Arrow-first streaming writers; duplication removed.
- [ ] Native export targets no longer buffer large payloads.
- [ ] One canonical table-key parsing/validation surface.
- [ ] Input hashes include build-engine version; plugin identity/version standardized.
- [ ] QueryTemplate + SQLGlot AST fingerprinting available for dynamic queries.
- [ ] Canonical export audit/tracking schema in the tracking subsystem.
- [ ] Snapshot replacement semantics are atomic/transactional.
- [ ] TargetSystem is the canonical entrypoint; overlapping APIs retired.
- [ ] Canonical compute result type for executor materialization.
- [ ] Single typed materialization metadata schema across savers + parsers.
- [ ] Tag consistency guardrail integrated into quality checks.
- [ ] Legacy execution stack removed (or reduced to temporary shims with removal date).
