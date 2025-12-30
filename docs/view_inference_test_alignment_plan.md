# View Inference Test Alignment Plan

## Goal

Bring storage/docs-view tests into full alignment with the production view inference pipeline. The tests should rely on SQLGlot view builders + inferred schemas (no view seeding) and assert durable invariants rather than fixed row counts, while keeping failures actionable.

## Scope

Addresses the view-related issues observed in storage tests:

- Docs views fail to materialize when schemas are not applied.
- View schema inference skips `docs.v_data_models*` (TO_JSON / LIST(STRUCT)).
- Schema provider drops derived view schemas when registry provider is active.
- Docs-view-dependent repository tests out of sync with view-first design.
- Hamilton ingestion failures inside `docs_views_ready_gateway`.
- Iceberg cache test expecting view tables in Iceberg catalog.
- PyArrow API mismatch in golden schema formatting.

## Guiding Approach (Best-in-Class)

- Treat views as **derived-only** artifacts in tests, mirroring production.
- Prefer **contract + inference driven** schemas for views.
- Shift tests to **schema + invariant assertions** rather than exact row counts.
- Ensure view materialization is **self-sufficient** (schemas exist, inference enabled).

## Phase 0 — Common Fixtures and Invariants

### 0.1 Add a canonical “view inference ready” fixture

- Create a new fixture in `tests/storage/conftest.py`:
  - `docs_views_inferred_gateway(tmp_path: Path) -> Iterator[StorageGateway]`
  - Behavior:
    - Ensure contract catalog + schema service are loaded.
    - Ensure schemas exist (core/graph/analytics/docs) even when `apply_schema=False`.
    - Call `gateway.policy.ensure_all_views(overwrite=True, strict=True)`.
    - Return gateway only after views materialize.

### 0.2 Define invariant-based assertion helpers

- Add helper(s) under `tests/_helpers/assertions` to validate:
  - View is queryable without errors.
  - Returned columns contain a required subset (schema-driven).
  - When rows exist, all `repo/commit` match the snapshot.
- Example helper signature (to be implemented later):
  - `assert_view_invariants(gateway, table_key, required_columns, *, repo, commit)`

### Acceptance

- A single fixture produces a gateway where view inference + materialization succeed.
- A single helper covers dynamic success criteria for view-backed tests.

## Phase 1 — Schema Materialization Hardening

### 1.1 Ensure schemas exist before view materialization

- Update `src/codeintel/storage/views/materialization.py` or
  `src/codeintel/storage/duckdb_policy_backend.py` to create missing schemas
  before any `CREATE VIEW` statements.
- Required schemas: `core`, `graph`, `analytics`, `docs`.

### 1.2 Keep derived view schemas even when registry provider is active

- Update `src/codeintel/storage/contracts/schema_provider.py`:
  - When schema service is sourced from registry, wrap it in `_ViewSchemaProvider`.
  - This ensures derived `docs.v_*` schemas remain available everywhere.

### Acceptance

- `gateway.policy.ensure_all_views()` works even when `apply_schema=False`.
- Derived view schemas remain visible even with registry-backed providers.

## Phase 2 — Fix View Inference for Data Model Views

### 2.1 Extend inference for TO_JSON / LIST(STRUCT)

- Update `src/codeintel/storage/views/schema_inference.py`:
  - Treat `TO_JSON(...)` results as `JSON`.
  - Treat `LIST(STRUCT(...))` as `JSON` or `LIST<STRUCT>` (choose JSON unless
    nested schema support is needed elsewhere).

### 2.2 (Optional) Add explicit overrides for `docs.v_data_models*`

- If inference still fails, add explicit schema overrides in
  `src/codeintel/core/schemas/output_registry.py` for:
  - `docs.v_data_models`
  - `docs.v_data_models_normalized`

### Acceptance

- `derive_view_schemas` succeeds for both data-model views.
- No fallback `SELECT NULL` relations for these views.

## Phase 3 — Align View-Dependent Tests

### 3.1 Repository tests

- Update the following tests to use `docs_views_inferred_gateway` and invariants:
  - `tests/storage/repositories/test_repositories.py`
  - `tests/storage/repositories/test_modules.py`
  - `tests/storage/repositories/test_functions.py`
  - `tests/storage/repositories/test_subsystems.py`
  - `tests/storage/test_graphs_repository.py`
  - `tests/storage/test_data_models.py`

### 3.2 Replace strict row-count assertions

- Replace fixed expectations like `len(rows) == N` with:
  - `len(rows) >= 1` only if relevant base data seeded.
  - `len(rows) == 0` when no base data is seeded.
  - Schema + snapshot invariants always enforced.

### Acceptance

- All view-backed tests pass when view inference is enabled.
- Failures identify schema or snapshot mismatches rather than counts.

## Phase 4 — Hamilton Ingestion Fixture Adjustments

### 4.1 Identify failing target in `docs_views_ready_gateway`

- Add temporary debug logging in
  `tests/_helpers/orchestration/provisioning.py` to log the failing target
  and error summary when `assert_target_ok` fails.

### 4.2 Choose best resolution

- Option A (preferred): fix the failing target.
- Option B: downgrade `docs_views_ready_gateway` to seed-only without
  invoking the failing target(s), since these tests focus on views.

### Acceptance

- `docs_views_ready_gateway` succeeds without `target_status=failed`.
- Docs view tests no longer fail during fixture setup.

## Phase 5 — Iceberg Cache + Golden Schema Tests

### 5.1 Iceberg cache view mismatch

- Update `tests/storage/test_iceberg_cache.py` to target a base table
  that is actually written to Iceberg (e.g., `docs.demo`), OR
- Update `ServingSnapshotFactory` to materialize views into Iceberg
  (only if that is the new contract).

### 5.2 PyArrow schema formatting compatibility

- Update `tests/_helpers/goldens/table_goldens.py`:
  - Call `schema.to_string()` without `show_metadata` if unsupported.

### Acceptance

- Iceberg cache test uses a valid Iceberg-backed table.
- Golden schema formatting works with the current PyArrow version.

## Rollout Checklist

- [ ] Add `docs_views_inferred_gateway` fixture and invariant assertions.
- [ ] Ensure schemas exist before view materialization.
- [ ] Wrap registry provider with `_ViewSchemaProvider`.
- [ ] Fix inference for `docs.v_data_models*`.
- [ ] Update view-dependent tests to use invariant checks.
- [ ] Resolve `docs_views_ready_gateway` target failure.
- [ ] Fix Iceberg cache test table selection.
- [ ] Make golden schema formatting version-safe.

## Verification Plan

- Run targeted tests:
  - `uv run pytest tests/storage/repositories/test_repositories.py -q`
  - `uv run pytest tests/storage/test_data_models.py -q`
  - `uv run pytest tests/storage/test_docs_views.py -q`
  - `uv run pytest tests/storage/test_graphs_repository.py -q`
  - `uv run pytest tests/storage/test_iceberg_cache.py -q`
  - `uv run pytest tests/storage/test_table_goldens.py -q`

- Then run storage suite:
  - `uv run pytest tests/storage -q`

## Outcome

After this plan, all view-backed tests should use the same inference and
materialization path as production, be resilient to schema evolution, and
fail only on meaningful regressions.
