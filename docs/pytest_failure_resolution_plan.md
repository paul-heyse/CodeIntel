# Pytest Failure Resolution Plan

## Context and constraints

- Preserve `hamilton.data_saver` as first-class metadata.
- Tests run with fallback disabled unless explicitly testing fallback behavior.
- Catalog identifiers can be normalized to safe identifiers; exact commit ID fidelity is not required.


## Goals

- Eliminate the 92 failures and 24 errors reported in the last full pytest run.
- Keep Hamilton tag semantics stable for production while making tag filtering robust.
- Ensure docs views and contracts are materialized deterministically in tests.
- Restore CLI/test harness behavior with fallback disabled.
- Maintain architectural boundaries (DuckDB usage localized to storage).


## Root-cause clusters (from test summary)

1. **Hamilton tag validation and tag-filter semantics**
   - `hamilton.data_saver.*` tags in tests fail Hamilton decorator validation.
   - Tag filter queries pass boolean/None values to Hamilton, which now requires strings.
   - Affects: `tests/tags/test_tag_filter_discovery_semantics.py` (collection errors),
     many docs/graphs/ingestion tests that rely on tag-based discovery.

2. **Docs view discovery/materialization drift**
   - Missing `docs.v_*` views in repositories and analytics tests.
   - Likely tied to tag-filter failures and inconsistent view materialization in fixtures.

3. **Unsafe DuckDB catalog identifiers**
   - Catalog names like `codeintel-c2` fail identifier validation for DDL.
   - Impacts snapshot gateways and history CLI tests.

4. **Meta catalog attachment gaps**
   - Tests querying `meta.information_schema` fail when meta is not attached.

5. **CLI resolution expectations**
   - CLI tests assume fallback behavior; fallback is disabled by policy.

6. **Contract/schema validation mismatches**
   - Docs export tests fail due to contract/schema drift vs fixtures.

7. **Boundary and performance regressions**
   - DuckDB boundary test failures (usage outside storage).
   - Dataset list performance budget exceeded; schema roundtrip timeouts.

8. **Observability config regressions**
   - OpenTelemetry config tests failing due to changed validation/entrypoints.

9. **Serving and MCP failures**
   - Semantic registry/mcp tool failures likely downstream of tag/view issues.


## Implementation plan

### Phase 1: Tag metadata, validation, and tag-query behavior

**Objective:** Keep `hamilton.data_saver` metadata while conforming to Hamilton tag rules and
ensuring tag filters remain expressive.

**Design basis:**
- Hamilton `tag` decorator now rejects non-string values and reserved prefixes.
- Production nodes use saver tags attached during node creation, not only via `@tag`.
- Tag filters should be tolerant to bool/None inputs and perform post-filtering.

**Changes:**
1. **Normalize tag values to strings for `hamilton.data_saver` and related keys.**
   - Update saver tag emission to use `"true"` / `"false"` (or `"1"` / `"0"`) consistently.
   - Align validators in build/hamilton to treat string values as boolean equivalents.
   - Files: `src/codeintel/build/hamilton/save_to.py`,
     `src/codeintel/build/hamilton/validate.py`,
     `src/codeintel/build/hamilton/dag_catalog_compiler.py`.

2. **Harden tag filters and TagQuery.**
   - Ensure `TagQuery.query()` only passes strings or string lists to Hamilton.
   - When the filter contains bool/None, perform a local post-filter over returned
     variables using their tags.
   - Files: `src/codeintel/core/hamilton/tag_query.py`,
     `src/codeintel/core/hamilton/tag_filters.py`.

3. **Update tests that directly use `@tag` with `hamilton.*` keys.**
   - Replace direct `@tag` usage with a test helper that constructs nodes with tags
     without calling Hamilton’s `tag` validator (e.g., a fixture that builds a driver and
     injects tags).
   - Files: `tests/tags/test_tag_filter_discovery_semantics.py`,
     test helpers in `tests/_helpers/`.

**Acceptance criteria:**
- All tag-related errors disappear.
- `tests/tags/test_tag_filter_discovery_semantics.py` collects and passes.


### Phase 2: Docs view discovery and materialization

**Objective:** Ensure docs views are always discoverable and materialized in tests.

**Design basis:**
- View discovery should not be blocked by tag-filter failures.
- View materialization should be deterministic in fixtures and in storage repos.

**Changes:**
1. **Prefer module-based view discovery in storage.**
   - Use module scanning in `discover_view_builders` by default for storage tests
     (avoid tag-query reliance for view inventory).
   - Files: `src/codeintel/storage/views/discovery.py`,
     `src/codeintel/storage/views/inventory.py`.

2. **Make fixture setup explicit about view materialization.**
   - Ensure `docs_views_ready_gateway` and other test contexts call
     `ensure_all_views(overwrite=True, strict=True)` after seeding.
   - Files: `tests/_helpers/orchestration/provisioning.py`,
     `tests/storage/test_docs_views.py`, related fixtures.

3. **Add fail-fast logging for view compilation.**
   - When view compilation fails, capture the view key and SQL builder error so tests
     fail early with clear context.
   - File: `src/codeintel/storage/views/materialization.py`.

**Acceptance criteria:**
- All missing `docs.v_*` table errors are resolved.
- Storage repository tests for docs views pass.


### Phase 3: Catalog identifier normalization

**Objective:** Normalize catalog names to safe identifiers without breaking snapshot semantics.

**Design basis:**
- Hyphenated commit-based catalog names are not valid identifiers.
- We can normalize to a safe, deterministic catalog ID.

**Changes:**
1. **Introduce catalog normalization helper.**
   - Map any invalid identifier to a stable safe form (e.g., `codeintel_<hash>` or
     `codeintel_<short_commit>` with underscores).
   - Apply when reading `duckdb_default_catalog` and when building DDL qualifiers.
   - Files: `src/codeintel/storage/duckdb/catalog.py`,
     `src/codeintel/storage/schema_roundtrip.py`,
     `src/codeintel/storage/duckdb_policy_backend.py`.

2. **Update tests to assert normalized catalog names.**
   - Adjust snapshot/cli tests to accept normalized identifiers.
   - Files: `tests/cli/test_history_timeseries_cli.py`,
     `tests/storage/test_gateway_factory.py`.

**Acceptance criteria:**
- No `Invalid catalog identifier` errors.
- Snapshot and history CLI tests pass with normalized catalog IDs.


### Phase 4: Meta catalog attachment in tests

**Objective:** Make meta catalog queries reliable in in-memory tests.

**Changes:**
1. **Attach meta DB in schema-seeding tests or helpers.**
   - Update tests to call `attach_meta_database` before querying `meta.*`.
   - Files: `tests/_helpers/test_schema_seeding.py`,
     `tests/_helpers/orchestration/provisioning.py` (if used broadly).

**Acceptance criteria:**
- `test_ensure_production_schemas_is_idempotent` passes.


### Phase 5: CLI tests with fallback disabled

**Objective:** Ensure CLI tests pass without fallback.

**Changes:**
1. **Update CLI test harness to create minimal `codeintel.yaml`.**
   - Ensure CLI tests provide a project root when fallback is disabled.
   - Files: `tests/_helpers/cli.py`, `tests/cli/*`.

2. **Align CLI storage config for parallel runs.**
   - Avoid multiple connections to the same DB with incompatible flags.
   - Files: `tests/cli/test_build_command.py`, `tests/cli/test_build_parallel.py`.

**Acceptance criteria:**
- CLI build/plan tests pass with fallback disabled.


### Phase 6: Contract and docs export alignment

**Objective:** Sync fixtures with current contract schemas and validation behavior.

**Changes:**
1. **Regenerate docs export fixtures to match current schemas.**
   - Update data model fixtures to match `analytics.data_model_fields` and
     `analytics.data_model_relationships`.
   - Files: `tests/docs_export/*`.

2. **Align minimal export tests with schema validation expectations.**
   - Ensure test data satisfies required columns and constraints.
   - Files: `tests/docs_export/test_export_smoke.py`,
     `tests/docs_export/test_graph_validation_export.py`.

**Acceptance criteria:**
- All docs export tests pass without disabling validation.


### Phase 7: Architecture boundary fixes

**Objective:** Restore boundary invariants and duckdb usage locality.

**Changes:**
1. **Move direct DuckDB usage to storage layer or allowlist intentionally.**
   - If build-layer DuckDB usage is required, add explicit boundary exceptions with
     justification.
   - Files: `tests/architecture/test_duckdb_boundaries.py`,
     `tests/storage/test_storage_architecture_invariants.py`,
     impacted build modules.

**Acceptance criteria:**
- Architecture/boundary tests pass.


### Phase 8: Performance and timeout regressions

**Objective:** Restore test performance budgets and timeouts.

**Changes:**
1. **Reduce repeated view materialization in tests.**
   - Cache results within a fixture or re-use a shared gateway where safe.
   - Files: `tests/_helpers/`.

2. **Optimize schema roundtrip and dataset list paths.**
   - Avoid full graph/view compilation when not needed; use cached schema indexes.
   - Files: storage schema roundtrip and dataset listing helpers.

**Acceptance criteria:**
- Performance tests and timeout-bound tests pass.


### Phase 9: Serving/MCP failures

**Objective:** Restore semantic registry compilation and MCP endpoints.

**Changes:**
1. **Ensure semantic registry compilation does not depend on invalid tag filters.**
   - Use the hardened TagQuery and module-based view discovery.
   - Files: `src/codeintel/build/serving/semantic_compile.py`,
     `tests/serving/test_semantic_registry_compiles_from_driver_tags.py`.

2. **Validate MCP service wiring.**
   - Ensure contract catalog and semantic registry are available in serving harness.
   - Files: serving tests under `tests/serving/`.

**Acceptance criteria:**
- All serving/mcp tests pass (200 responses, no ToolError).


## Validation plan

- Run targeted pytest subsets per phase to keep runtime manageable:
  - Tag/query: `tests/tags`, `tests/storage/test_docs_views.py`
  - Views: `tests/storage/repositories`, `tests/storage/test_graphs_repository.py`
  - Catalog: `tests/cli/test_history_timeseries_cli.py`, `tests/storage/test_gateway_factory.py`
  - CLI: `tests/cli/test_build_command.py`, `tests/cli/test_cli_scope_and_plan.py`
  - Docs export: `tests/docs_export`
  - Serving: `tests/serving`
- Full quality gate after fixes:
  - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
  - `uv run pytest -q` (segmented by major directories as per AOP)


## Risks and mitigations

- **Tag value normalization could break downstream expectations.**
  - Mitigation: treat `"true"`/`"false"` as valid booleans everywhere in build/validate.
- **View discovery changes could reorder views.**
  - Mitigation: preserve deterministic sorting by table key.
- **Catalog normalization could mask bugs where full catalog qualification is needed.**
  - Mitigation: log normalization at debug level and retain original catalog in metadata.


## Deliverables

- Updated tag handling, query filtering, and tests.
- Stable view discovery/materialization path.
- Catalog normalization helper with updated tests.
- CLI test harness aligned with fallback disabled.
- Updated docs export fixtures and schema validations.
- Restored performance budgets and architecture boundary compliance.
