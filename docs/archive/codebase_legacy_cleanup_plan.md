# Codebase Legacy Cleanup Plan

## Status

Proposed

## Goals

- Remove dead or legacy modules that are no longer used.
- Eliminate compatibility paths that diverge from the current design basis.
- Keep the codebase lean while preserving behavior that is still in scope.

## Scope

1) Remove unused Hamilton materialization helpers
   - `src/codeintel/build/hamilton/materialization_helpers.py`
   - Update references in documentation or docstrings (notably in
     `src/codeintel/build/hamilton/execution_result.py`).

2) Remove unused analytics tuple writer utilities
   - `AnalyticsTupleWriteOptions` and `write_analytics_tuple_rows` in
     `src/codeintel/build/analytics/utilities/datasets.py`.
   - Remove re-exports from `src/codeintel/build/analytics/utilities/__init__.py`.

3) Remove legacy build tracking migration path
   - `_migrate_impl_kind_columns` in
     `src/codeintel/storage/tracking/build_tracking.py`.
   - Add a one-time migration step to guarantee no remaining legacy columns.

4) Relocate tooling-only contract checker
   - `src/codeintel/build/hamilton/contracts/check_target_contracts.py`.
   - Keep functionality but move to tooling so runtime packages do not carry
     it as public library surface.

5) Remove YAML project config compatibility
   - `config/codeintel.yaml` handling in project detection and runtime resolution.
   - Ensure TOML remains the only supported project config path.

## Non-Goals

- No behavior changes beyond removing unused or legacy code paths.
- No changes to unrelated build or ingestion functionality.
- No production compatibility shims beyond the migration steps below.

## Implementation Sequence

### Phase 0: Pre-flight inventory

- Confirm no live imports or call sites for each target module.
- Record any documentation references that need updates.

Suggested checks:
- `rg "materialization_helpers|executor_materialize" src docs`
- `rg "AnalyticsTupleWriteOptions|write_analytics_tuple_rows" src tests docs`
- `rg "_migrate_impl_kind_columns|impl_kind" src tests`
- `rg "check_target_contracts" src tools`
- `rg "config/codeintel.yaml" src tests docs`

### Phase 1: Remove Hamilton materialization helpers

1) Delete `src/codeintel/build/hamilton/materialization_helpers.py`.
2) Update any docstring mentions or references:
   - `src/codeintel/build/hamilton/execution_result.py`
3) Ensure no exports from package `__init__` files remain.

Acceptance:
- No imports of `materialization_helpers` remain.
- Documentation does not mention the deleted helper.

### Phase 2: Remove analytics tuple writer utilities

1) Delete `AnalyticsTupleWriteOptions` and `write_analytics_tuple_rows` from
   `src/codeintel/build/analytics/utilities/datasets.py`.
2) Remove the re-exports in `src/codeintel/build/analytics/utilities/__init__.py`.
3) Update any tests or documentation if they referenced these utilities.

Acceptance:
- No references to the removed symbols in `src/`, `tests/`, or `docs/`.

### Phase 3: Retire legacy build tracking migration path

1) Add a one-time migration step (script or tool) to validate that:
   - `build_tracking` tables no longer include legacy `plugin` columns.
   - `impl_kind` is the only authoritative column.
2) Run the migration step and capture output in build logs.
3) Remove `_migrate_impl_kind_columns` from
   `src/codeintel/storage/tracking/build_tracking.py`.
4) Remove any tests or code paths that assume the migration exists.

Acceptance:
- No runtime migration logic remains in `build_tracking.py`.
- A migration artifact exists (script or tooling) to guarantee clean state.

### Phase 4: Relocate tooling-only contract checker

1) Move `src/codeintel/build/hamilton/contracts/check_target_contracts.py`
   into `tools/` (or inline into `tools/quality_report.py` if that is the only
   consumer).
2) Update imports in `tools/quality_report.py` to new module path.
3) Remove any public exports for the moved module.

Acceptance:
- No runtime package import for the contract checker.
- `tools/quality_report.py` still runs without changes to output.

### Phase 5: Remove YAML project config compatibility

1) Delete YAML project config path handling in:
   - `src/codeintel/cli/project/_project.py`
   - `src/codeintel/cli/resolution/runtime.py`
2) Remove any YAML parsing dependencies that are now unused.
3) Update docs and tests to reflect TOML-only support:
   - Remove or rewrite any tests that expect `config/codeintel.yaml`.
4) Ensure CLI messaging explains TOML config path expectations.

Acceptance:
- No `config/codeintel.yaml` references remain in `src/`, `tests/`, or `docs/`.
- TOML paths are the only supported config paths.

## Validation

Run quality report after all phases:

```
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
```

Targeted tests (update the list as needed based on edits):
- CLI config and project tests.
- Storage tracking unit tests.
- Tooling quality report tests (if any).

## Risk Mitigations

- Phase 3 explicitly separates migration from runtime behavior to avoid
  non-deterministic state in production code.
- Phase 4 keeps tooling functionality intact while removing runtime coupling.
- Phase 5 is aggressive but aligns with the design-only, single-developer
  context.

## Rollback Plan

- Restore deleted modules from VCS if a hidden runtime dependency is found.
- Reintroduce compatibility paths only if a critical caller is discovered.
