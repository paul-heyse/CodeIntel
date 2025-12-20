# Storage Refinement - Legacy and Compatibility Cleanup Implementation Plan

**Status**: Implementation plan  
**Last updated**: 2025-12-19  
**Primary scope**: `src/codeintel/storage/**`  
**Secondary scope (required call-site migrations)**: `tests/storage/**` and any non-storage
consumers of storage compatibility APIs (to be inventoried in Phase 0).

## Context
The storage review identified a small set of legacy and compatibility surfaces inside
`src/codeintel/storage` that are no longer aligned with the table_key-first design. These
surfaces keep old call patterns alive (dataset-name JSON schema lookup, DatasetRegistry alias
properties, and stale legacy docs). This plan removes those surfaces in a staged, safe way and
updates storage-local call paths and tests to the canonical APIs.

## Goals
1) Make JSON schema lookup in storage table_key-first and eliminate dataset-name lookup.
2) Remove DatasetRegistry compatibility alias properties in favor of canonical accessors.
3) Remove stale legacy notes and unused compatibility exports within storage.
4) Keep storage behavior stable for active call sites via staged deprecations and test updates.

## Non-goals
- Changing build or serving export behavior (outside storage).
- Refactoring storage gateway architecture or policy backends.
- Large renames of storage public surfaces outside the specific legacy items listed here.

## Scope and inventory (storage-only)

### A) Dataset-name JSON schema lookup (legacy)
- `src/codeintel/storage/contracts/json_schema.py`
  - `get_json_schema_for_dataset_name(...)` is explicitly marked backward compatibility.
- `src/codeintel/storage/contracts/__init__.py`
  - Re-exports the dataset-name lookup.
- `src/codeintel/storage/validation/conformance.py`
  - Calls dataset-name lookup when validating `json_schema_id` datasets.

### B) DatasetRegistry compatibility alias properties
- `src/codeintel/storage/datasets/registry.py`
  - Alias properties: `mapping`, `meta`, `jsonl_mapping`, `parquet_mapping`, `table_for_name`.

### C) Stale legacy notes / compatibility messaging
- `src/codeintel/storage/queries/safe.py`
  - Notes re-export for backward compatibility that may no longer exist.
- `src/codeintel/storage/views/ibis_views.py`
  - Notes "legacy create_* functions" even though no such functions exist in the module.

## Design decisions (confirm up front)
1) **Schema lookup**: storage should resolve JSON schema by table_key. Dataset-name lookup
   should be removed after migration.
2) **DatasetRegistry canonical access**: use `resolve_table_key`, `by_name`, `by_table_key`,
   `jsonl_datasets`, and `parquet_datasets` (no alias properties).
3) **Deprecation window**: decide whether to keep shims for one release or remove immediately
   after migrations are complete.

## Workstreams and phases

### Phase 0 - Inventory and migration map (no behavior change)
**Goal**: identify all consumers of the legacy surfaces and define the migration order.

Actions:
- Locate all references to `get_json_schema_for_dataset_name` and DatasetRegistry alias
  properties across the repo.
- Classify usage as:
  - Storage-internal (in scope here).
  - Non-storage consumer (record for coordinated migration).
- Document any external dependencies that require coordination.

Deliverables:
- Call-site inventory (file list).
- Migration order and deprecation strategy.

### Phase 1 - Canonical JSON schema lookup by table_key
**Goal**: storage uses table_key for schema lookup; dataset-name helper becomes a shim.

Actions:
- Add a canonical helper in storage for table_key JSON schema retrieval.
  - Example name: `get_json_schema_for_table_key(table_key: str)`.
- Update `src/codeintel/storage/validation/conformance.py` to resolve table_key via the
  DatasetRegistry and use the table_key helper.
- Keep `get_json_schema_for_dataset_name` as a shim or deprecation wrapper for one cycle
  (if a deprecation window is selected).
- Update storage contracts exports to include the canonical helper and (optionally) the shim.

Deliverables:
- Canonical table_key schema helper in storage.
- Conformance validation uses table_key path.

### Phase 2 - Remove DatasetRegistry alias properties
**Goal**: remove alias properties and update storage-local tests.

Actions:
- Replace usage of:
  - `mapping` -> `by_name` or `resolve_table_key`.
  - `meta` -> `by_name`.
  - `jsonl_mapping` -> `jsonl_datasets`.
  - `parquet_mapping` -> `parquet_datasets`.
  - `table_for_name` -> `resolve_table_key`.
- Update `tests/storage/test_datasets.py` to validate canonical accessors instead of aliases.
- If a deprecation window is chosen, mark alias properties as deprecated before removal.

Deliverables:
- DatasetRegistry exposes only canonical accessors.
- Storage tests assert canonical APIs.

### Phase 3 - Remove legacy notes and compatibility shims
**Goal**: delete unused compatibility surfaces and clean up stale documentation.

Actions:
- Remove `get_json_schema_for_dataset_name` from storage contracts (and any storage exports)
  once all call sites are migrated.
- Clean up stale legacy notes in:
  - `src/codeintel/storage/queries/safe.py`
  - `src/codeintel/storage/views/ibis_views.py`
- Remove any now-unused exports or helper functions.

Deliverables:
- No dataset-name schema lookup in storage.
- Storage docs reflect current surfaces.

## Acceptance gates
Run storage-focused checks after each phase:
```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q tests/storage/test_datasets.py tests/storage/test_conformance.py
```
If non-storage call sites are migrated as part of this work, expand test scope accordingly.

## Risks and edge cases
- Non-storage call sites still rely on dataset-name schema lookup or alias properties.
  - Mitigation: inventory call sites and use a deprecation window if needed.
- Dataset name to table_key resolution may differ for views or nonstandard datasets.
  - Mitigation: resolve via DatasetRegistry, not via name munging.
- Conformance validation might skip datasets if schema lookup changes incorrectly.
  - Mitigation: add tests to cover table_key lookup paths.

## Open questions
- Should dataset-name schema lookup and DatasetRegistry aliases be removed immediately after
  migration, or kept for one release with warnings?
- Are any external integrations depending on dataset-name JSON schema lookup from storage
  that need coordination?
