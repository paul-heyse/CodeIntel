# Best-in-Class Columnar Execution Plan (Arrow + Polars + DuckDB)

## Intent

Adopt advanced PyArrow, Polars, and DuckDB capabilities to harden the CodeIntel
data platform, improve extensibility, and align all build/serving paths with the
Hamilton DAG as the authoritative contract source.

## Goals

- Make Arrow schema metadata the primary contract plane derived from Hamilton DAG outputs.
- Enforce streaming-first execution paths for build, serving, and validation.
- Use Polars optimizer control surfaces and plan introspection to make query behavior explicit.
- Centralize DuckDB relational/Expression API usage for safe, composable query building.
- Introduce advanced types (STRUCT/LIST/MAP/UNION) where they reduce ambiguity or drift.
- Increase observability with batch-level metadata and Parquet stats for drift and planning.

## Non-goals

- Replacing Hamilton as the schema or planning authority.
- Rewriting all dataflow logic in a single PR.
- Introducing new external engines or distributed runtimes beyond current choices.

## Architectural Principles

- Hamilton DAG remains the source of truth for schema and target definitions.
- Arrow contracts are derived artifacts, not independent sources of truth.
- Streaming IO and batch-wise execution are the default; materialization is explicit.
- Query generation stays flexible; execution mode is a late-bound decision.

## Compatibility Notes (Hamilton + Storage/Serving Plans)

- SQLGlot AST remains the single semantic source of truth for queries; any Expression helpers
  must be derived from the AST, not replace it.
- Arrow contracts must be strictly DAG-derived (schema_output + tags) and never act as an
  independent schema source.
- Hamilton data-quality modifiers remain the primary validation surface; Arrow/Polars checks
  complement but do not replace those validations.
- Advanced type adoption (STRUCT/LIST/MAP/UNION) must align with schema_output conventions so
  tag coverage remains complete and consistent.
- No reintroduction of raw SQL templates or view SQL maps; all query paths remain AST-driven.

## Phase 0: Decisions and Guardrails

- Decide canonical Arrow contract metadata keys and versioning strategy.
- Decide allowed advanced types (STRUCT/LIST/MAP/UNION) and any disallowed types.
- Decide strictness policy for schema alignment (default vs ingest overrides).
- Update guardrails to enforce streaming-only paths and to block eager materialization.
- File targets: `tools/guardrails.py`, `src/codeintel/storage/schema/arrow_schema.py`.
- Acceptance: contract metadata keys are documented and guardrails block eager paths in core code.

## Phase 1: Arrow Contract Authority (DAG-derived)

- Ensure every DAG output table produces Arrow schema metadata as the canonical contract.
- Persist Arrow schema IPC payloads in the registry and prefer them at runtime.
- Add contract provenance metadata (Hamilton target, module, version).
- File targets: `src/codeintel/core/schemas/arrow_gen.py`, `src/codeintel/storage/schema/arrow_schema.py`,
  `src/codeintel/storage/tracking/schema_catalog.py`, `src/codeintel/storage/tracking/schema_catalog_compile.py`.
- Acceptance: registry contains Arrow schema payloads for all contract outputs and serving reads them first.

## Phase 2: Streaming-First IO and Validation

- Replace eager `to_table`/`read_all` usage with `RecordBatchReader` and batch iterators.
- Use `dataset.Scanner.to_reader()` and `ParquetFile.iter_batches()` for streaming reads.
- Use Polars `collect_batches` and `sink_batches` for streaming transforms and sinks.
- File targets: `src/codeintel/storage/datasets/arrow_store.py`,
  `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`,
  `src/codeintel/build/exports/writers.py`, `src/codeintel/serving/semantic/kernel.py`,
  `src/codeintel/storage/validation/columnar.py`.
- Acceptance: no eager materialization in build/serving paths; streaming APIs used end-to-end.

## Phase 3: Polars Optimizer Control + Plan Introspection

- Add `QueryOptFlags` support to serving/build query settings for explicit optimizer behavior.
- Use `LazyFrame.explain`, `show_graph`, and `profile` in debug/validation paths.
- Enforce selector-based column handling (`polars.selectors`) where schema-driven selection is expected.
- File targets: `src/codeintel/serving/semantic/engines/polars_engine.py`,
  `src/codeintel/build/hamilton/validate.py`, `src/codeintel/build/hamilton/native/*`.
- Acceptance: optimizer flags are configurable and plan introspection is available for all semantic queries.

## Phase 4: DuckDB Relational API + Safe Expression Building

- Prefer relational API over string SQL for programmatic queries.
- Use parameterized queries for all dynamic SQL.
- Introduce a small Expression builder helper for reusable filters and projections.
- File targets: `src/codeintel/storage/warehouse.py`, `src/codeintel/storage/datasets/arrow_store.py`,
  `src/codeintel/serving/semantic/registry_compiler.py`.
- Acceptance: dynamic query generation is centralized and uses parameter binding by default.

## Phase 5: Advanced Types for Contract Hardness

- Adopt STRUCT/LIST/MAP/UNION in contract definitions where structure matters.
- Map Arrow types to DuckDB advanced types consistently in schema translation.
- Update validators to recognize nested types and avoid JSON string fallbacks.
- File targets: `src/codeintel/core/schemas/primitives.py`, `src/codeintel/storage/schema/arrow_schema.py`,
  `src/codeintel/storage/validation/columnar.py`.
- Acceptance: nested structures are represented as typed columns, not opaque JSON blobs.

## Phase 6: Schema Observations + Metadata Enrichment

- Capture row-group stats and fragment metadata during ingest and validation.
- Persist dataset stats and schema drift signals in the registry.
- Use Arrow schema metadata to carry contract version and producer provenance.
- File targets: `src/codeintel/build/schemas/observations.py`,
  `src/codeintel/storage/tracking/schema_catalog.py`, `src/codeintel/storage/datasets/maintenance.py`.
- Acceptance: schema observations include batch-level stats and contract provenance.

## Phase 7: Serving IPC and Streaming Contracts

- Ensure IPC responses always use canonical Arrow schema metadata.
- Add configurable `IpcWriteOptions` and enforce streaming writer usage.
- Provide batch-wise JSON fallback via Arrow writer when IPC is not supported.
- File targets: `src/codeintel/serving/http/streaming.py`, `src/codeintel/serving/semantic/kernel.py`.
- Acceptance: serving outputs are streaming IPC by default with contract metadata preserved.

## Phase 8: Testing, Benchmarks, and Regression Gates

- Add round-trip tests for TableSchema -> Arrow schema -> TableSchema.
- Add streaming regression tests to ensure no eager materialization in core paths.
- Add Polars plan tests validating optimizer flags and streaming fallbacks.
- Add performance benchmarks for dataset scan + batch validation.
- File targets: `tests/_helpers/columnar_streams.py`, `tests/build/schemas/*`,
  `tests/serving/*`, `tests/storage/*`.
- Acceptance: tests enforce streaming-only behavior and contract round-trips.

## Phase 9: Rollout and Migration

- Stage changes behind feature flags where needed (serving/ingest boundaries).
- Backfill Arrow schema payloads in the registry for existing tables.
- Provide migration notes and a rollback path for each phase.
- File targets: `src/codeintel/storage/tracking/schema_catalog.py`,
  `src/codeintel/cli/handlers/meta.py`, `docs/storage_serving_best_in_class_plan.md`.
- Acceptance: migration is reversible and registry backfills are repeatable.

## Decision Log (to fill in during implementation)

- Canonical Arrow contract metadata keys and versioning scheme.
- Default schema alignment strictness and allowed widenings.
- Approved advanced types and any prohibited Arrow types (e.g., list_view).
- Streaming exception allowlist for tests/tools.

## Success Metrics

- No eager materialization in build/serving paths for large datasets.
- Contract metadata present in all IPC streams and registry entries.
- Reduced drift errors and faster schema alignment decisions.
- Lower memory footprints for large exports and validations.
