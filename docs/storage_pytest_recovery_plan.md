Storage Pytest Recovery Plan

Goals
- Eliminate all current `tests/storage` failures while preserving production behavior.
- Keep Hamilton DAG compilation deterministic and debuggable.
- Ensure missing docs views return empty results with a warning (no hard failures).
- Centralize view creation and remove ad-hoc creation in non-materialization code.
- Harden relation materialization against non-SQL DuckDB relations.

Scope Summary (from `build/test-results/junit.xml`)
- Duplicate contract output: `core.ast_metrics` (DAG compile failure).
- Missing docs views (e.g., `docs.v_function_summary`, `docs.v_subsystem_summary`).
- SQLGlot `TokenError` from parsing `relation.sql_query()` on in-memory relations.
- Schema manifest tests now require `schema_hash` for view schemas.
- Architecture invariant: unexpected `create_view` in `snapshot_service.py`.

Phase 0: Diagnostics and Guardrails
- Add a diagnostic path for duplicate output keys in `src/codeintel/build/hamilton/dag_catalog_compiler.py`:
  - Include `output.key`, `producer_target`, `saver_node`, and module provenance (`module.__file__`).
  - Keep strict behavior (raise) but make the error actionable.
- Add a lightweight “module list” debug utility in test harnesses to dump module paths used for runtime composition.
  - This will confirm whether `codeintel.build.hamilton.native.ingestion` (package) or any plugin module re-exports `t__ast`.

Phase 1: Resolve Duplicate `core.ast_metrics`
- Likely cause: the same nodes are being loaded twice via two module paths:
  - `codeintel.build.hamilton.native.ingestion.extraction_targets` and
  - a package module that re-exports those targets (e.g., `codeintel.build.hamilton.native.ingestion`).
- Fix strategy:
  - In module resolution, exclude package modules that only re-export target symbols.
  - Add a guard to drop modules whose `__file__` ends with `__init__.py` when a sibling module path already provides the concrete targets.
  - Alternatively, add a module-path allowlist for native modules (only leaf `.py` files) and filter package modules from plugin/workspace additions.
- Add a regression test:
  - Assert that `compile_dag_catalog(..., strict=True)` succeeds for a full runtime composition and that `core.ast_metrics` is unique.

Phase 2: Repository Resilience to Missing Docs Views
- Requirement: repository methods should return empty results when docs views are missing, but log a warning.
- Implement in `src/codeintel/storage/repositories/base.py`:
  - In `_relation`, when `docs.*` is missing after `ensure_all_views`, log a warning with `table_key`, `repo`, `commit`.
  - Return an empty relation with correct schema:
    - Use `gateway.schemas.load_table_schema(table_key)` to get schema.
    - Build `SELECT CAST(NULL AS ...) AS col ... WHERE 1=0`.
  - Ensure downstream `_relation_to_reader` validation is satisfied.
- Add tests:
  - When docs views are missing, repository returns empty list and emits a warning (use `caplog`).

Phase 3: Centralize View Creation
- Move direct `create_view` usage out of `src/codeintel/storage/serving/snapshot_service.py`.
- Add a central helper in `src/codeintel/storage/materialization.py` or `DuckDBPolicyBackend`:
  - `create_view_from_relation(con, view_key, relation)` to perform standardized view creation.
- Update `snapshot_service._create_dataset_view` to call the centralized helper.
- Keep `tests/storage/test_storage_architecture_invariants.py` passing by ensuring only the centralized module uses `create_view`.

Phase 4: Harden Relation Materialization (SQLGlot TokenError)
- In `src/codeintel/storage/warehouse.py`:
  - `_relation_select_expr` should not assume `relation.sql_query()` is parseable SQL.
  - On parse failure, register the relation as a temp view and select columns from that view (no SQLGlot parse).
  - Use a safe unique name (existing `register_ephemeral` or a new helper).
- Add regression test:
  - `tests/storage/test_db_helpers.py` case with `insert_rows` should pass without SQLGlot errors.

Phase 5: Update Schema Manifest Fixtures
- In `tests/_helpers/serving_snapshot_factory.py`:
  - Compute `schema_hash` for view schemas using the same hashing logic as production.
  - Ensure `docs.v_demo` (and other view fixtures) include `schema_hash`.
- Add a targeted fixture test for manifest hashing to prevent regressions.

Phase 6: Verification
- Targeted runs:
  1) `uv run pytest -q tests/storage/test_docs_views.py -n 0`
  2) `uv run pytest -q tests/storage/test_db_helpers.py -n 0`
  3) `uv run pytest -q tests/storage/repositories -n 0`
  4) `uv run pytest -q tests/storage/test_table_goldens.py -n 0`
- Full storage suite:
  - `uv run pytest -q tests/storage -n 0`

Success Criteria
- No duplicate output errors during DAG compilation.
- Repositories return empty results (with warnings) when docs views are missing.
- No SQLGlot TokenError during relation materialization.
- Schema manifest fixtures include `schema_hash` and pass goldens.
- Centralized view creation invariant holds.
