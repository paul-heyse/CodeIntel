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
[ ] Define SchemaDerivation protocol and SchemaIndex builder from the global TargetGraph/Runtime.
[ ] Attach SchemaIndex to TargetMetadataService (global graph) and expose read-only access.
[ ] Implement DagSchemaProvider and wire SchemaService to use it by default.
[ ] Replace column_order_for_table_key import-time calls with deferred column resolution:
    - update saver decorator usage to pass a sentinel/empty tuple
    - resolve columns inside DuckDBRowsSaver using SchemaService/SchemaIndex
[ ] Refactor ContractService to avoid singleton lock re-entrancy and to use DAG-first schemas.
[ ] Update schema compilation to read from SchemaIndex and honor global-graph determinism:
    - keep manifest selection tied to TargetMetadataService
    - ensure infer_native defaults align with DAG-first behavior
[ ] Migrate declared schemas to source-only and add explicit override hooks for non-inferable outputs.
[ ] Add tests that:
    - forbid import-time DAG/schema resolution
    - verify manifest determinism against the global DAG
    - validate fallback behavior for sources and inferable outputs
[ ] Add performance and stability safeguards:
    - cache SchemaIndex in memory
    - ensure error reporting is deterministic and traceable
[ ] Update documentation and plan references to reflect DAG-first schema authority.

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
