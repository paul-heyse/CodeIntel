# Legacy Decommission Plan (Storage + Serving)

## Summary
This plan decommissions legacy and compatibility code paths in storage/serving while
preserving production safety. It incorporates the confirmed facts that:

- Published snapshots are NOT guaranteed to include `docs.search_documents` or
  `metadata.derived_lineage_*` today (publish + lineage sync are best-effort).
- `jsonl` is still part of the public export surface and remains the canonical
  format in the shared export-format registry.

The plan is therefore phased: first make artifacts mandatory and observable, then
remove serving/runtime fallbacks, and finally retire the `jsonl` alias only after
spec and client updates.

## Scope
Legacy items identified for decommissioning:

1) Snapshot pointer backward-compat fields
   - `created_at` fallback for `published_at`
   - inferred `buildspec_path` when missing
   - File: `src/codeintel/serving/db/pointer.py`

2) Serving runtime fallbacks for missing artifacts
   - missing `docs.search_documents` (empty search results)
   - missing `metadata.derived_lineage_*` (empty lineage in describe)
   - Files: `src/codeintel/serving/semantic/kernel.py`,
     `src/codeintel/serving/search/engine.py`,
     `src/codeintel/storage/serving/search_index.py`

3) Contract/schema provider fallbacks
   - fallback to schema service, lazy build import, local schema generation
   - Files: `src/codeintel/storage/contracts/schema_provider.py`,
     `src/codeintel/storage/contracts/provider.py`,
     `src/codeintel/storage/contracts/json_schema.py`,
     `src/codeintel/storage/schema/json_schema.py`

4) File summary fallback to `core.modules` when docs view is missing
   - File: `src/codeintel/storage/repositories/modules.py`

5) Export format aliasing between `jsonl` and `ndjson`
   - Files: `openspec/specs/export-formats/spec.md`,
     `src/codeintel/core/exports/formats.py`,
     `src/codeintel/serving/export/formats.py`

6) Semantic registry default version
   - fallback to `version="v1"` when absent
   - File: `src/codeintel/serving/semantic/registry.py`

## Non-goals
- No changes to build semantics or schema contracts beyond the compatibility
  cleanup described here.
- No changes to unrelated analytics/graphs/ingestion code.

## Decision Log (current facts)
- `docs.search_documents` is intended to be created at publish time but failures
  are logged and snapshots still publish; therefore it is not guaranteed.
  - `src/codeintel/build/serving/publisher.py`
- `metadata.derived_lineage_*` sync is best-effort and failures are logged, so
  lineage is not guaranteed today.
  - `src/codeintel/build/hamilton/native/export/serving_artifacts.py`
- Export format registry treats `jsonl` as canonical with `ndjson` as an alias;
  build outputs are still `jsonl`.
  - `openspec/specs/export-formats/spec.md`
  - `src/codeintel/core/exports/formats.py`

## Phase 0: Inventory + Gate Definition
Goal: establish explicit guarantees and add observability before removing fallbacks.

Tasks:
- Define required serving artifacts for publish:
  - `docs.search_documents` table exists and populated
  - `metadata.derived_lineage_edges` and `metadata.derived_lineage_columns` exist
- Add explicit publish-time validation rules and failure modes.
- Add metrics/logging for publish results and missing artifact reasons.

Deliverables:
- Documented acceptance criteria for "publish readiness".
- Validation checklist for snapshots.

Acceptance criteria:
- A new snapshot is considered "ready" only when required artifacts exist.
- Publish path fails hard (non-zero) if required artifacts are missing.

## Phase 1: Make Search + Lineage Artifacts Mandatory
Goal: enforce `docs.search_documents` and `metadata.derived_lineage_*` in published
snapshots.

Tasks:
1) Publish hardening
   - Move `docs.search_documents` creation + FTS indexing from best-effort logging
     to a required step that fails publish on error.
   - Emit a single structured error message that indicates why search index build
     failed (missing input tables, DuckDB extension error, etc).
   - File: `src/codeintel/build/serving/publisher.py`

2) Lineage hardening
   - Ensure lineage sync runs in the serving artifacts target with strict error
     propagation when `env.repo` + `env.commit` are present.
   - Optionally gate on a feature flag to allow staged rollout.
   - File: `src/codeintel/build/hamilton/native/export/serving_artifacts.py`

3) CI and runtime validation
   - Add a publish validation test that fails if `docs.search_documents` is
     missing or empty.
   - Add a publish validation test that fails if `metadata.derived_lineage_*`
     tables are missing when serving artifacts are built.

Acceptance criteria:
- Publish fails if search index build fails.
- Publish fails if lineage tables are missing for snapshots with repo/commit.
- Serving snapshots consistently contain both artifacts in CI.

## Phase 2: Remove Serving Runtime Fallbacks
Goal: once artifacts are guaranteed, remove silent compatibility paths.

Tasks:
1) Search fallback removal
   - Remove the empty-result fallback when `docs.search_documents` is missing.
   - Fail fast with a clear error if search table is missing.
   - File: `src/codeintel/serving/semantic/kernel.py`

2) Lineage fallback removal
   - Remove conditional table existence checks and empty lineage fallback.
   - Treat missing lineage tables as an error for describe endpoint.
   - File: `src/codeintel/serving/semantic/kernel.py`

Acceptance criteria:
- Serving endpoints fail loudly if required artifacts are missing.
- Runtime code no longer checks for artifact existence; it assumes publish
  validation guarantees.

## Phase 3: Contract/Schema Provider Cleanup
Goal: eliminate legacy fallback paths and unify on the canonical contract catalog.

Tasks:
1) Require contract catalog presence for storage
   - Make contract catalog load mandatory during gateway open when not read-only.
   - Remove lazy imports and fallback providers.
   - File: `src/codeintel/storage/contracts/schema_provider.py`

2) Remove local JSON schema generation fallbacks
   - Require `SchemaService` for JSON schema retrieval where applicable.
   - Remove fallback generation from table schema definitions.
   - Files: `src/codeintel/storage/contracts/json_schema.py`,
     `src/codeintel/storage/schema/json_schema.py`

3) Make contract provider fail fast if catalog missing
   - Remove fallback to on-the-fly contract construction for missing entries.
   - File: `src/codeintel/storage/contracts/provider.py`

Acceptance criteria:
- Storage fails fast when contract catalog is missing in expected contexts.
- No lazy imports from `codeintel.build.*` remain in storage contracts.

## Phase 4: File Summary Fallback Removal
Goal: require docs view `docs.v_file_summary` and remove `core.modules` fallback.

Tasks:
- Remove fallback block in `ModuleRepository.get_file_summary`.
- Ensure build target that creates `docs.v_file_summary` is always present
  in serving snapshots.
- Add test that validates this view is populated.

Acceptance criteria:
- `get_file_summary` returns None only when no docs view row exists.
- No fallback to `core.modules` remains.

## Phase 5: Pointer Schema Strictness
Goal: remove backward-compat fields in serving snapshot pointer.

Tasks:
1) Enforce pointer schema at publish time
   - Require `published_at` and `buildspec_path` in current.json.
   - Update any writer to always emit these fields.

2) Remove fallback decoding
   - Remove `created_at` fallback and inferred `buildspec_path` logic.
   - File: `src/codeintel/serving/db/pointer.py`

Acceptance criteria:
- Loading pointer fails on missing required fields.
- All published pointers include explicit `published_at` and `buildspec_path`.

## Phase 6: JSONL / NDJSON Deprecation
Goal: deprecate `jsonl` only after spec change and client migration.

Tasks:
1) Spec update (required)
   - Propose an openspec change to either:
     - keep `jsonl` canonical and remove `ndjson`, OR
     - flip canonical to `ndjson` and remove `jsonl`.
   - This is a product/API decision and must be versioned.

2) Client and CLI alignment
   - Update CLI help text and exported artifacts to align with chosen canonical.
   - Update tests to use the canonical name only.

3) Runtime handling
   - Add explicit deprecation warnings for the alias for at least one release.
   - Remove alias handling and update registry.

Acceptance criteria:
- Updated spec merged and communicated.
- All build/serving callers use only the canonical format.
- No alias handling remains in registry.

## Rollout Strategy
- Phase 0 -> Phase 1 (hardening) should land first and run in CI for one full
  release cycle with flags/feature gates if necessary.
- Phase 2 (fallback removals) only after Phase 1 acceptance criteria hold across
  CI and an agreed production window.
- Phase 6 requires explicit product/API decision.

## Testing and Validation
- Add tests for publish hardening (search and lineage).
- Add tests for pointer schema strictness.
- Update tests around exports to reflect the final canonical format choice.
- Maintain existing tests that verify serving artifacts and registry outputs.

## Deletion Checklist (per item)
- Pointer fallback:
  - [ ] publish writer always emits required fields
  - [ ] remove fallback parsing
  - [ ] update tests

- Search/lineage fallbacks:
  - [ ] enforce artifact creation in publish/build
  - [ ] add validation tests
  - [ ] remove runtime fallbacks

- Contract/schema fallbacks:
  - [ ] require contract catalog and schema service
  - [ ] remove lazy fallbacks and local generation
  - [ ] update failing tests and docs

- File summary fallback:
  - [ ] guarantee view in serving snapshots
  - [ ] remove fallback branch

- JSONL/NDJSON alias:
  - [ ] update spec
  - [ ] deprecate alias with warnings
  - [ ] remove alias handling

## Risks and Mitigations
- Risk: publish failures due to missing input tables for search/lineage.
  - Mitigation: add preflight checks and explicit error messages.
- Risk: production snapshots created before hardening might break after removing
  fallbacks.
  - Mitigation: introduce a cutover date or minimum snapshot version guard.
- Risk: client breakage from export format changes.
  - Mitigation: staged deprecation and explicit spec update.

## Ownership and Dependencies
- Build pipeline owners for publish hardening.
- Serving owners for runtime fallback removal.
- Schema/contract owners for registry changes.
- API/spec owners for export format decisions.

