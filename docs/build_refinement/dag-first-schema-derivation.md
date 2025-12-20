---
name: dag-first-schema-derivation
description: DAG-first schema derivation plan for Hamilton-based build system
---

# Plan

Implement a DAG-first schema derivation system that makes the Hamilton graph the primary
source of truth for output schemas, while preserving global-graph determinism and keeping
source-table schemas as the only declared inputs. The plan removes import-time schema work,
introduces a schema index tied to the global DAG, and updates contracts/materializers to
resolve schema lazily at execution time.

## Requirements
- Derive schemas for produced tables from the Hamilton DAG by default (Ibis, row tuples, Pandera).
- Keep schema manifests tied to the global target graph, even in tests.
- Avoid import-time DAG construction or contract resolution that can deadlock pytest collection.
- Preserve existing public APIs via compatibility shims where needed.
- Provide deterministic ordering, hashing, and stable manifests.

## Scope
- In:
  - SchemaIndex and SchemaDerivation protocols built from the global DAG.
  - DAG-first SchemaProvider wired into SchemaService.
  - Deferred column resolution in materializers/savers.
  - ContractService refactor to avoid singleton re-entrancy and use DAG-first schema.
  - Manifest compilation that uses the global DAG and SchemaIndex.
  - Tests and guardrails for import-time safety and schema parity.
- Out:
  - Updates to tools/audit_plugin_schemas.py and the guardrail message (defer to later migration).
  - Non-schema changes to target execution semantics or export formats.

## Files and entry points
- src/codeintel/build/target_metadata.py
- src/codeintel/build/schemas/provider_unified.py
- src/codeintel/build/schemas/inference_service.py
- src/codeintel/build/schemas/contract_service.py
- src/codeintel/build/schemas/compile.py
- src/codeintel/build/schemas/service.py
- src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py
- src/codeintel/build/hamilton/save_to.py
- src/codeintel/core/schemas/row_models.py
- src/codeintel/core/schemas/declared.py
- src/codeintel/build/hamilton/native/** (targets with column_order_for_table_key)
- tests/build/** and tests/hamilton/**

## Data model / API changes
- Introduce SchemaIndex and SchemaDerivation types (IbisDerivation, RowDerivation, ExplicitOverride).
- Add a DAG-first SchemaProvider that resolves via SchemaIndex, falling back to declared sources.
- Add a deferred-columns sentinel for saver metadata, resolved at save time.
- Refactor ContractService initialization to avoid global singleton recursion and accept DAG-first
  providers while keeping existing function-level APIs stable.
- Optional: extend SchemaManifest with provenance and schema hash metadata (backward compatible).

## Action items
[x] Define SchemaDerivation protocol and SchemaIndex builder from the global TargetGraph/Runtime.
[x] Attach SchemaIndex to TargetMetadataService (global graph) and expose read-only access.
[x] Implement DagSchemaProvider and wire SchemaService to use it by default.
[x] Replace column_order_for_table_key import-time calls with deferred column resolution:
    - update saver decorator usage to pass a sentinel/empty tuple
    - resolve columns inside DuckDBRowsSaver using SchemaService/SchemaIndex
[x] Refactor ContractService to avoid singleton lock re-entrancy and to use DAG-first schemas.
[x] Update schema compilation to read from SchemaIndex and honor global-graph determinism:
    - keep manifest selection tied to TargetMetadataService
    - ensure infer_native defaults align with DAG-first behavior
[x] Migrate declared schemas to source-only and add explicit override hooks for non-inferable outputs.
[x] Add tests that:
    - [x] forbid import-time DAG/schema resolution
    - [x] verify manifest determinism against the global DAG (source-only registry)
    - [x] validate override behavior for non-inferable outputs
[x] Add performance and stability safeguards:
    - [x] cache SchemaIndex in memory
    - [x] guard singleton re-entrancy for DAG/bootstrap paths
    - [x] ensure inference error reporting is deterministic and traceable
[x] Update documentation and plan references to reflect DAG-first schema authority.

## Open implementation detail

### Declared schemas become source-only
- Implement `source_declared_schema_provider` in `src/codeintel/core/schemas/declared.py`.
- Build-facing provider filters out DAG outputs using `TargetSystem.all_table_keys`.
- Target spec helpers now use full declared registry only for explicit overrides.

### Explicit override hooks for non-inferable outputs
- `TargetSpecOptions` now accepts `override_tables`.
- `build_schema_index` treats placeholder schemas as missing overrides and raises for
  non-inferable outputs without explicit overrides.

### Schema provider behavior after source-only migration
- Unified provider now falls back to source-only declared schemas; DAG outputs are resolved
  via SchemaIndex/overrides.

### Tests to add/extend
- Deterministic manifest tests now use the DAG-first provider.
- Explicit override tests cover missing override errors and override resolution.
- Saver tests confirm deferred column resolution at execution time with contract enforcement.

### Optional manifest provenance
If adopted, extend `SchemaManifest` with additive, opt-in provenance for tables, views, and
artifacts (manifest version remains v2). The best-in-class design captures schema provenance for
tables/views and lineage back to source tables for artifacts.

- Tables and views: include `schema_hash`, `derivation_kind`, `derivation_source`.
- Artifacts: include `source_table_keys` plus `source_schema_hashes` (aligned to table keys).
- Keep provenance opt-in via `SchemaManifestRequest.include_provenance` to avoid default churn.
- Update manifest JSON snapshots and diff tooling to tolerate the additive fields.

## Testing and validation
- uv run python -m tools.quality_report --output build/quality-results/quality_report.json
- uv run pytest -q
- Targeted tests for schema inference, manifest compilation, and import-time safety.

## Risks and edge cases
- Schema inference may need env/gateway access; ensure isolated inference contexts are deterministic.
- Removing declared schemas can break source tables; keep explicit source allowlists and overrides.
- Deferred column resolution could mask mismatch errors; add validation at materialization time.
- Circular imports can reappear; prefer lazy loading and narrow provider dependencies.

## Open questions
- None.
