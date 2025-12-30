# Dynamic Schema Derivation Migration Plan

## Purpose

Transition remaining static schema usage in production and test code to dynamic derivation
driven by the Hamilton DAG and the centralized schema provider/inference pipeline.
This aligns view and output schemas with runtime inference and reduces brittle, static
definitions.

## Goals

- Use dynamic schema derivation for docs views and other inferable outputs.
- Ensure serving and storage flows pull schemas from the canonical provider/registry,
  not static manifests where possible.
- Keep static schemas only for infrastructure tables that are not inferable.

## Non-goals

- Removing infrastructure schemas in metadata or Iceberg catalog tables.
- Changing build artifact formats or schema manifest versioning.

## Phase 0: Inventory and invariants

1) Inventory docs view keys and static schema usage
   - Collect `docs.v_*` from `src/codeintel/storage/views/view_ast_map.json`.
   - Scan for `docs.v_*` in `src/codeintel/core/schemas/table_registry.py`.
2) Identify dynamic schema providers
   - Confirm `get_schema_provider()` in `src/codeintel/storage/contracts/schema_provider.py`
     is the canonical provider and includes derived view schemas.
3) Define hard static boundaries
   - Keep static schemas in `src/codeintel/storage/metadata/schema.py` and
     `src/codeintel/storage/iceberg/catalog_schema.py`.

Acceptance criteria:
- List of docs views and static references documented.
- Confirmed which tables remain static by design.

## Phase 1: Docs view schemas (production)

### 1.1 Remove static docs view schemas from registry

Steps:
- Remove any `docs.v_*` entries from `src/codeintel/core/schemas/table_registry.py`
  (at minimum `docs.v_validation_summary`, which is already in SQLGlot view inventory).
- Ensure view schemas are only derived by `derive_view_schemas()` via
  `src/codeintel/storage/contracts/schema_provider.py`.

Acceptance criteria:
- No `docs.v_*` in `table_registry.py`.
- Derived view schemas still resolved via `get_schema_provider()`.

### 1.3 Decommission legacy static docs view paths

Steps:
- Remove any static docs view table schemas from:
  - `src/codeintel/core/schemas/table_registry.py`
  - `src/codeintel/core/schemas/output_registry.py` (if any docs views exist there)
- Remove any test fixtures that build `TableSchema` for `docs.v_*` (see Phase 6).
- Keep only the SQLGlot view builders + `derive_view_schemas` as the source of truth.

Acceptance criteria:
- No static `docs.v_*` schemas in production code.
- No tests depend on static docs view schemas.

### 1.2 Align schema manifest compilation with derived views

Steps:
- In `src/codeintel/build/schemas/compile.py`, ensure `include_views=True` paths
  pull view schemas from the provider (already supported) and do not depend on
  static registry entries.
- Add a regression test: schema manifest includes `docs.v_*` only when the view
  is present in `view_ast_map.json` and derivation succeeds.

Acceptance criteria:
- Schema manifests include derived docs views without static registry entries.

## Phase 2: Serving inventory and planner (production)

### 2.1 SchemaInventory derives views dynamically

Steps:
- Add a helper on `SchemaInventory` in `src/codeintel/serving/semantic/inventory.py`:
  `with_derived_views(provider: SchemaProvider | None, modules=...)`.
  - Build a `MappingSchemaProvider` from `self.schemas`.
  - Use `derive_view_schemas(provider=..., view_keys=..., modules=...)`.
  - Merge derived view schemas into `self.schemas` (do not overwrite existing).
- Update `src/codeintel/serving/db/manager.py` to call the helper after loading
  from registry or manifest so derived docs views are always present.

Acceptance criteria:
- Serving snapshots see docs view schemas even if the registry/manifest lacks them.

### 2.2 Semantic planner uses dynamic inventory

Steps:
- Ensure `SemanticQueryPlanner` in `src/codeintel/serving/semantic/planner.py`
  receives an inventory that already includes derived views (no change in planner
  logic, only inventory assembly).
- Add a targeted test that a view defined in SQLGlot but not in a manifest still
  appears in `SchemaInventory` and is allowed by the planner when `columns_dynamic`
  is true.

Acceptance criteria:
- Planner no longer fails on missing view schema when derivation is possible.

## Phase 3: Search index table schema (production)

### 3.1 Derive `docs.search_documents` schema dynamically

Steps:
- In `src/codeintel/storage/serving/search_index.py`, replace the static
  `_SEARCH_DOCUMENTS_SCHEMA` with dynamic derivation:
  - Build a single `SELECT ...` expression (already in code) and create the table
    using `CREATE TABLE AS SELECT ... WHERE 1=0` so DuckDB derives the schema.
  - Use explicit `CAST` where needed to keep deterministic types.
- Keep a guard to assert columns match expected names if strictness is required,
  but avoid hardcoding types.

Acceptance criteria:
- Search documents table is created from the query projection, not static schema.

## Phase 4: Dynamic column discovery in queries (production)

### 4.1 Use dynamic schemas in snapshot filtering

Steps:
- Update `table_has_rows_for_snapshot` in `src/codeintel/storage/queries/safe.py`
  to resolve columns via `SchemaProvider` or `information_schema.columns`
  (fallback) instead of `contract.schema`.
- Ensure the logic still works when schemas are inferred at runtime.

Acceptance criteria:
- Snapshot filtering no longer depends on static contract schema.

### 4.2 Use dynamic schemas in graph metrics

Steps:
- Update `src/codeintel/analytics/graphs/config_graph_metrics.py` to resolve
  node/edge column names via `SchemaProvider` instead of `contract.schema`.
  - If schema is missing, fall back to existing defaults.

Acceptance criteria:
- Metrics projection survives schema evolution without static coupling.

## Phase 5: Contract validation and gateway alignment (production)

### 5.1 Restrict strict checks to declared outputs

Steps:
- In `src/codeintel/core/schemas/contract_validation.py`, treat views and
  inferred outputs as dynamic:
  - Keep strict checks for declared target outputs only.
  - For views, validate internal consistency if available but do not require
    static schema match.
- In `src/codeintel/storage/gateway/factory.py`, ensure schema drift checks
  consult the runtime schema provider and avoid failing on derived views
  when inference is enabled.

Acceptance criteria:
- Schema validation errors only block on declared target outputs.

## Phase 7: Static schema decommissioning (production + tests)

### 7.1 Static schema removals (production)

These should be removed or narrowed after dynamic derivation is in place:
- `src/codeintel/core/schemas/table_registry.py`
  - Remove all `docs.v_*` entries (covered in Phase 1).
  - Review for any table schemas that are now always inferred via the schema
    registry; migrate them out unless they are true infrastructure tables.
- `src/codeintel/core/schemas/output_registry.py`
  - Remove override schemas for outputs that are inferable by Hamilton
    (keep only non-inferable outputs).
- `src/codeintel/storage/serving/search_index.py`
  - Remove `_SEARCH_DOCUMENTS_SCHEMA` and any schema-type guards once runtime
    derivation is stable.

### 7.2 Static schema removals (tests)

These should be removed or replaced with dynamic derivation:
- Tests that create `TableSchema` for `docs.v_*` views (move to derived view schemas).
- Test harness helpers that emit static `schema_manifest.json` payloads:
  - `tests/_helpers/hamilton_harness_artifacts.py`
  - `tests/_helpers/serving_snapshot_factory.py`

### 7.3 Static schema functionality to decommission

After rollout, decommission these static paths:
- Any view schema generation that reads from static registry entries.
- Any test-only utilities that construct static docs view schemas.
- Any schema validation logic that assumes a static schema for views or inferable outputs.

Acceptance criteria:
- Static view schemas removed from production and test code.
- Schema derivation for views and inferable outputs is the only path in use.

## Phase 6: Test harness and fixtures (tests)

### 6.1 Replace static schema manifest writers

Steps:
- Update `tests/_helpers/hamilton_harness_artifacts.py` and
  `tests/_helpers/serving_snapshot_factory.py` to build schema manifests via
  `compile_schema_manifest` or `SchemaInventory.from_registry(...).with_derived_views(...)`
  rather than hardcoded table lists.
- Remove explicit `TableSchema` fixtures for docs views in tests where possible.

Acceptance criteria:
- Test harness uses dynamic derivation; no static docs view schemas remain in fixtures.

### 6.2 Add regression tests for dynamic views

Steps:
- Add tests that derived docs views appear in manifests and inventories even
  when not declared statically.
- Add tests for serving planner to allow a derived view with dynamic columns.

Acceptance criteria:
- Tests cover dynamic derivation path end-to-end.

## Rollout order

1) Phase 1: Docs view registry removal + manifest compile alignment.
2) Phase 2: Serving inventory dynamic view merge.
3) Phase 3: Search index schema derivation.
4) Phase 4: Dynamic column discovery in queries/metrics.
5) Phase 5: Validation and gateway strictness adjustments.
6) Phase 6: Test harness updates and regression coverage.

## Acceptance checklist (final)

- No `docs.v_*` static schemas in `src/codeintel/core/schemas/table_registry.py`.
- Serving inventory includes derived docs views without relying on static manifests.
- Search index table schema is derived at runtime.
- Query filtering and graph metrics use dynamic schema discovery.
- Validation only blocks on declared outputs; views are dynamic.
- Tests use dynamic schema manifests and pass with derived view schemas.

## Suggested validation commands

- `uv run ruff check`
- `uv run pyright`
- `uv run pyrefly check`
- `uv run pytest tests/serving/semantic -q`
- `uv run pytest tests/storage -q`
