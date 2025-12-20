# Storage Refinement - Error and Consolidation Implementation Plan

**Status**: Implementation plan  
**Last updated**: 2025-12-19  
**Primary scope**: `src/codeintel/storage/**`  
**Secondary scope (required call-site migrations)**: `src/codeintel/core/**`, `src/codeintel/build/**`,
`src/codeintel/cli/**`, `src/codeintel/serving/**`, and `tests/**`.

## Context
This plan consolidates storage functionality around a small set of canonical surfaces for
schema access, registry usage, views, query safety, DDL generation, gateway composition,
and error taxonomy. It builds on:
- `docs/storage_refinement/STORAGE_BEST_IN_CLASS_CONSOLIDATION_IMPLEMENTATION_PLAN.md`
- `docs/storage_refinement/STORAGE_LEGACY_COMPAT_CLEANUP_IMPLEMENTATION_PLAN.md`

## Goals
1) Maintain functional parity for storage reads/writes, view materialization, and contract validation.
2) Keep table_key-first access as the only supported schema/registry lookup pattern.
3) Ensure a single, canonical surface per concern (views, DDL, query safety, errors, gateway composition).
4) Preserve or improve Ruff/pyright/pyrefly cleanliness in storage files.

## Non-goals
- Non-storage refactors except minimal call-site updates required by storage API changes.
- Build/serving export behavior changes beyond what is required to remove dataset-name schema lookups.
- View definition changes unrelated to orchestration and materialization.

## Design decisions (locked)
1) **Canonical error surface**: use `codeintel.core.errors.storage` for storage errors.
2) **Export validation**: resolve JSON schemas by table key only (dataset-name lookup removed).
3) **View orchestration**: view compilation and materialization flow only through
   `storage/views/materialization.py`.

## Scope
**In**
- JSON schema access and dataset registry access patterns
- View discovery/compilation/materialization surfaces
- SQL ingress safety and name qualification
- Scalar/JSON coercion and result shaping
- Gateway composition and bootstrapping entry points
- DDL builders and metadata bootstrap
- Safe query helper surfaces
- Error taxonomy and import paths

**Out**
- Build/serving refactors not required by storage API changes
- Schema contract redesigns outside table_key lookup consolidation

## Files and entry points
**Schema + contracts**
- `src/codeintel/storage/contracts/json_schema.py`
- `src/codeintel/storage/contracts/provider.py`
- `src/codeintel/storage/schema/json_schema.py`

**Dataset registry**
- `src/codeintel/storage/datasets/registry.py`
- `src/codeintel/storage/datasets/catalog.py`

**Views**
- `src/codeintel/storage/views/materialization.py`
- `src/codeintel/storage/views/discovery.py`
- `src/codeintel/storage/views/inventory.py`
- `src/codeintel/storage/views/ibis_views.py`
- `src/codeintel/storage/warehouse.py`

**Query safety + naming**
- `src/codeintel/storage/queries/safe.py`
- `src/codeintel/storage/helpers/table_key.py`

**Coercion + JSON**
- `src/codeintel/storage/query_results.py`
- `src/codeintel/storage/helpers/json.py`

**Gateway + bootstrapping**
- `src/codeintel/storage/gateway/minimal.py`
- `src/codeintel/storage/gateway/factory.py`
- `src/codeintel/storage/gateway/pool.py`
- `src/codeintel/storage/backend/duckdb_session.py`
- `src/codeintel/storage/duckdb_policy_backend.py`

**DDL + metadata**
- `src/codeintel/storage/metadata/sqlglot_ddl.py`
- `src/codeintel/storage/metadata/ddl.py`
- `src/codeintel/storage/metadata/bootstrap.py`

**Errors**
- `src/codeintel/core/errors/storage.py`
- `src/codeintel/storage/exceptions.py`
- `src/codeintel/storage/queries/safe.py`

**Export validation (non-storage)**
- `src/codeintel/build/exports/validation.py`
- `src/codeintel/build/schemas/json_schema_registry.py`

## Data model / API changes
- `StorageError` and `StorageConnectionError` move to `codeintel.core.errors.storage` and become
  the canonical imports for all storage-related call sites.
- `codeintel.storage.exceptions` becomes a thin re-export or is removed once all internal
  consumers migrate.
- `get_json_schema_for_dataset_name` is removed from
  `codeintel.build.schemas.json_schema_registry` and its exports.
- `validate_export_files` and its callers switch to table-key schema lookup, with dataset names
  used only for file selection.

## Workstreams and phases

### Phase 0 - Inventory and invariants (no behavior change)
**Goal**: confirm call sites and establish a migration checklist.

Actions:
- Inventory all uses of dataset-name schema lookup and non-core `StorageError` usage.
- Inventory raw SQL entry points and name-qualification helpers across storage.
- Record view creation call sites outside `materialization.py`.

Deliverables:
- Call-site inventory and migration order.

### Phase 1 - Error taxonomy consolidation
**Goal**: one canonical storage error surface in `codeintel.core.errors.storage`.

Actions:
- Add `StorageError` and `StorageConnectionError` to `codeintel.core.errors.storage` using
  `StorageErrorCode`.
- Update imports and catch blocks across storage, CLI, build, and serving.
- Remove or convert `codeintel.storage.exceptions` to a re-export shim.

Deliverables:
- Unified storage error taxonomy and imports.

### Phase 2 - JSON schema and registry access consolidation
**Goal**: table_key-only schema lookup and canonical registry access patterns.

Actions:
- Remove dataset-name JSON schema lookup from build schema registry and exports.
- Update export validation to resolve schema by table key.
- Ensure DatasetRegistry usage is canonical (`by_name`, `by_table_key`, `resolve_table_key`,
  `jsonl_datasets`, `parquet_datasets`), and add guardrail tests if needed.

Deliverables:
- No dataset-name schema lookup remaining in runtime code.

### Phase 3 - View orchestration single-path
**Goal**: all view compilation/materialization flows through `materialization.py`.

Actions:
- Remove unused view creation helpers (`warehouse.create_or_replace_view`, `_create_view` in
  `ibis_views.py`) after verifying no active call sites.
- Keep view discovery and materialization centralized in `materialization.py` and
  `duckdb_policy_backend.py` delegation.

Deliverables:
- Single view orchestration path with no alternate helpers.

### Phase 4 - SQL ingress safety and name qualification
**Goal**: a single SQL safety perimeter and canonical name handling.

Actions:
- Route all raw SQL usage through `queries/safe.py` checks (e.g., single-statement validation).
- Ensure table key parsing and qualification uses `helpers/table_key.py` only.
- Remove any duplicate safe-query helpers or re-exports.

Deliverables:
- Centralized SQL ingress safety and name qualification.

### Phase 5 - Scalar/JSON coercion and result shaping
**Goal**: consistent result normalization across repositories and helpers.

Actions:
- Standardize on `query_results.py` coercion helpers and `helpers/json.py` for JSON decoding.
- Remove ad-hoc DataFrame-to-dict or scalar conversion patterns from repositories.

Deliverables:
- Single coercion surface for storage query results.

### Phase 6 - Gateway composition and bootstrapping
**Goal**: single composition root for storage gateways and bootstrapping invariants.

Actions:
- Enforce `MinimalStorageGateway` (or equivalent) as the composition root for
  `DuckDBPolicyBackend` and Ibis integration.
- Remove or inline redundant connection bootstrap paths in gateway factory/pool.

Deliverables:
- One gateway composition path for all modes.

### Phase 7 - DDL builder consolidation
**Goal**: only `sqlglot_ddl.py` builds CREATE SCHEMA/INDEX DDL.

Actions:
- Replace local CREATE SCHEMA/INDEX builders with `sqlglot_ddl.py` helpers.
- Update metadata/bootstrap to use the canonical DDL surface.

Deliverables:
- One DDL builder surface in storage.

### Phase 8 - Guardrails and tests
**Goal**: prevent legacy surfaces from reappearing.

Actions:
- Add fast guardrails that assert table_key-only schema lookup and view orchestration path.
- Extend storage-focused tests for SQL safety, coercion invariants, and DDL rendering stability.

Deliverables:
- Guardrail tests and storage-specific coverage for new invariants.

## Acceptance gates
```bash
uv run ruff check src/codeintel/storage src/codeintel/build src/codeintel/core tests
uv run pyright --warnings --pythonversion=3.13 src/codeintel/storage src/codeintel/build src/codeintel/core tests
uv run pyrefly check src/codeintel/storage src/codeintel/build src/codeintel/core tests
uv run pytest -q tests/storage tests/docs_export tests/build/hamilton
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

## Risks and edge cases
- Hidden call sites outside storage may still rely on deprecated surfaces.
- View orchestration consolidation could change ordering or dependency behavior.
- DDL consolidation can change rendering order or optional clauses; validate against
  existing bootstrap tests.
- Error taxonomy consolidation could change exception classes observed by callers.

## Open questions
- None.
