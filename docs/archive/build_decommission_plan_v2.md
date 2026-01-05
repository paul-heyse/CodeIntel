# Build Decommission Plan (Native-Only Build System)

## Goals
- Remove legacy/plugin compatibility surfaces from build execution.
- Make build metadata and run records describe native implementations only.
- Shrink the public API surface in `codeintel.build` to reduce confusion and drift.
- Keep validation, schema, and export paths intact while simplifying configuration.

## Scope
### Decommission targets (from review)
1) `codeintel.build.types` (test-only types, duplicated by ingestion ports/results)
2) `codeintel.build.serving.search_index` (thin wrapper, only referenced in tests)
3) `BuildRunConfig` and run-config layering (`codeintel.build.run_config`, `BuildRunContext.run_config`, `BuildConfigStack.run_overrides`)
4) Plugin naming/semantics in build execution and records (`plugin_name`, `no_plugin` reason, `PluginExecutionError`)
5) CLI plugin subsystem (plugin management commands, manifests, loader/sandbox, completions)
6) Optional: remove facade packages not used in runtime (`codeintel.build.assets.__init__`, `codeintel.build.serving.__init__`)

### Non-goals (explicitly out of scope)
- Removing core plugin types used by ingestion tool plugins (`codeintel.core.plugins.types`, `codeintel.ingestion.engine.plugins`).
- Changing the Hamilton-native target execution logic, schemas, or export formats beyond what is required to remove plugin compatibility.

## Design Decisions (target state)
- Build is native-only. Run records and tracking use `impl_kind` (e.g., "native") instead of `plugin_name`.
- `impl_kind` is the only implementation identity field (no `implementation_id` or equivalent).
- CLI plugin subsystem is removed; core plugin types remain for ingestion tool plugins.
- Configuration overrides are direct (CLI/runtime -> BuildConfig) with no plugin profile stack.
- Planner and error messaging refer to "implementation" rather than "plugin".
- The build public API does not expose unused compatibility shims.

## Dependency Map (key references to update)
- `codeintel.build.types` is only imported in tests:
  - `tests/_helpers/fakes/fake_providers.py`
- `codeintel.build.serving.search_index` is only imported in tests:
  - `tests/build/serving/test_pr90_search_index_builds.py`
- `BuildRunConfig` usage:
  - `src/codeintel/build/run_context.py`
  - `src/codeintel/build/state.py`
  - `src/codeintel/build/config.py` (BuildConfigStack run_overrides)
  - `src/codeintel/build/__init__.py` (lazy export)
- Plugin naming surfaces:
  - `src/codeintel/core/hamilton/records.py` (`TargetRunRecord.plugin_name`)
  - `src/codeintel/storage/tracking/build_tracking.py` (build.run_targets `plugin` column)
  - `src/codeintel/build/hamilton/run_records.py`
  - `src/codeintel/build/hamilton/native/materialization_records.py`
  - `src/codeintel/build/hamilton/executor.py`
  - `src/codeintel/build/assets/emitter.py`
  - `src/codeintel/build/errors.py` (`PluginExecutionError`)
  - `src/codeintel/build/hamilton/planner.py` (`no_plugin` reason)
- CLI plugin subsystem surfaces:
  - `src/codeintel/cli/plugins/*`
  - `src/codeintel/cli/handlers/plugins.py`
  - `src/codeintel/cli/commands/plugins.py`
  - `src/codeintel/cli/completions/completion_model.py`
  - `src/codeintel/cli/rendering/specs.py`
  - `src/codeintel/cli/errors/results.py`
- Default profile source currently comes from the plugin subsystem:
  - `src/codeintel/build/hamilton/execution_options.py`
  - `src/codeintel/cli/project/_project.py`

## Implementation Plan

### Phase 1: Remove test-only wrappers

#### 1.1 Remove `codeintel.build.types`
- Update tests to import equivalent types from ingestion ports or engine results:
  - Replace `codeintel.build.types.ScipIndexResult` with `codeintel.ingestion.engine.results.ScipIndexResult`.
  - Replace `codeintel.build.types.ScipParseResult` with `codeintel.ingestion.engine.results.ScipIndexResult` (if parse result is required, use the canonical ingestion type).
  - Replace `codeintel.build.types.TypeCheckResult` with the canonical ingestion diagnostic/report type.
- Update `tests/_helpers/fakes/fake_providers.py` to use the new type sources.
- Delete `src/codeintel/build/types.py`.

#### 1.2 Remove `codeintel.build.serving.search_index`
- Update tests to import from `codeintel.storage.serving.search_index` directly:
  - `tests/build/serving/test_pr90_search_index_builds.py`
- Delete `src/codeintel/build/serving/search_index.py`.
- Optional cleanup: update downstream consumer labels in `config/registry/dag_output_inventory.yaml` to remove the "serving.search_index" label if it is meant to reflect module paths rather than conceptual consumers.

**Phase 1 acceptance criteria**
- No references remain to `codeintel.build.types` or `codeintel.build.serving.search_index`.
- All tests compile with updated imports.

---

### Phase 2: Remove CLI plugin subsystem

#### 2.1 Remove plugin management commands/handlers
- Delete `src/codeintel/cli/handlers/plugins.py`.
- Delete `src/codeintel/cli/commands/plugins.py`.

#### 2.2 Remove CLI plugin package
- Delete `src/codeintel/cli/plugins/*` (discovery, loader, manifest, registry, sandbox, testing).
- Remove any imports referencing these modules.

#### 2.3 Update CLI registration, completions, and rendering
- Remove plugin command registration from CLI command wiring.
- Remove plugin command from `src/codeintel/cli/completions/completion_model.py`.
- Remove plugin-specific rendering specs in `src/codeintel/cli/rendering/specs.py`.
- Remove plugin-specific error results in `src/codeintel/cli/errors/results.py` if unused after deletion.

#### 2.4 Preserve graph plugins command
- Ensure `graph plugins` remains intact and continues to surface plan metadata.

**Phase 2 acceptance criteria**
- The top-level `codeintel plugins` command is removed from CLI entrypoints and completions.
- CLI plugin subsystem modules are deleted and no longer imported.
- `graph plugins` continues to function.

---

### Phase 3: Remove run-config plugin layering

#### 3.1 Delete `BuildRunConfig`
- Remove `src/codeintel/build/run_config.py`.
- Remove `BuildRunConfig` from `src/codeintel/build/__init__.py` lazy exports.

#### 3.2 Collapse `BuildRunContext` run_config usage
- Remove `run_config` from:
  - `BuildRunContextOverrides`
  - `BuildRunContext` fields
  - `BuildRunContext.build_config_stack(...)`
- Replace `BuildRunContext.build_config_stack(...)` with direct overlay logic based only on CLI/runtime overrides.

#### 3.3 Replace `BuildConfigStack` overrides
- Remove `run_overrides` from `BuildConfigStack`.
- Create a new explicit override type if needed (e.g., `BuildConfigOverrides`) and pass it from CLI execution.
- Ensure `BuildConfig.parameters_for(...)` remains the single source of truth for parameters after merging module/target sections plus CLI overrides.

#### 3.4 Remove plugin profile dependencies
- Replace `DEFAULT_PROFILE_NAME` import in `src/codeintel/build/hamilton/execution_options.py` with a build-native constant.
- Replace `DEFAULT_PROFILE_NAME` import in `src/codeintel/cli/project/_project.py` with the same build-native constant.
- Proposed new constant location: `src/codeintel/build/settings.py` (or `src/codeintel/core/config/defaults.py` if you want a shared runtime constant).

**Phase 3 acceptance criteria**
- No `codeintel.core.plugins.*` imports remain in build configuration or execution paths.
- Build execution resolves profiles without plugin profiles.
- CLI project config uses a build-native default profile constant.

---

### Phase 4: Replace plugin identity with implementation identity

#### 4.1 Update run record models
- Modify `src/codeintel/core/hamilton/records.py`:
  - Replace `plugin_name: str` with `impl_kind: str`.
  - Update all constructors and usages accordingly.

#### 4.2 Update build run record creation
- Update run record creation paths to emit `impl_kind="native"`:
  - `src/codeintel/build/hamilton/run_records.py`
  - `src/codeintel/build/hamilton/native/materialization_records.py`
  - `src/codeintel/build/hamilton/executor.py`

#### 4.3 Update storage tracking schema
- Update `src/codeintel/storage/tracking/build_tracking.py`:
  - Replace `plugin` column usage with `impl_kind`.
  - Adjust insertion and query logic accordingly.
- Add a migration step for existing tables:
  - Rename column `plugin` -> `impl_kind` (or add new column and backfill).
  - Backfill all rows to `impl_kind = "native"`.

#### 4.4 Update asset tracking emission
- Update `src/codeintel/build/assets/emitter.py` to pass `impl_kind` directly (remove `_impl_kind` shim).
- Confirm `src/codeintel/storage/tracking/asset_tracking.py` continues to store `impl_kind` without plugin-derived conversion.

#### 4.5 Update error types and planner reasons
- Remove `PluginExecutionError` in `src/codeintel/build/errors.py` and replace test coverage with a native equivalent if needed.
- Rename planner reason `"no_plugin"` -> `"no_impl"` (or `"missing_target"`) in `src/codeintel/build/hamilton/planner.py` and update any CLI display logic.

**Phase 4 acceptance criteria**
- No `plugin_name` or `plugin` columns remain in build tracking or runtime record types.
- All run records and asset tracking refer to implementation kind.
- Planner output and errors no longer refer to plugins.

---

### Phase 5: Clean plugin language and compatibility references
- Replace remaining docstrings and user-facing messages that mention plugins in:
  - `src/codeintel/build/errors.py`
  - `src/codeintel/build/resources.py`
  - `src/codeintel/build/hashing.py`
  - `src/codeintel/build/hamilton/native/__init__.py`
- Update architecture docs if they still reference plugin execution paths as current behavior.

**Phase 5 acceptance criteria**
- Build docs/messages are consistent with native-only architecture.

---

### Phase 6 (Optional): Shrink build public facade
- Remove `src/codeintel/build/assets/__init__.py` and/or `src/codeintel/build/serving/__init__.py` if they remain unused after earlier phases.
- Confirm no runtime or tests import these facades.

**Phase 6 acceptance criteria**
- Build package exposes only actively used public surfaces.

## Data Migration Plan
- Build run tracking:
  - Add or rename `impl_kind` column in `build.run_targets`.
  - Backfill existing rows with `"native"`.
  - Update any views or analytics that rely on the `plugin` column.
- Asset tracking:
  - No schema change needed if `impl_kind` already exists; update emitter to pass correct values.

## Test Plan (targeted)
- Phase 1:
  - `tests/_helpers/fakes/fake_providers.py`
  - `tests/build/serving/test_pr90_search_index_builds.py`
- Phase 2:
  - `tests/cli/test_help_rendering.py`
  - `tests/cli/test_cli_scope_and_plan.py`
- Phase 3:
  - `tests/build/test_state.py`
  - `tests/build/test_state_computer.py`
  - `tests/build/hamilton/test_pr09_planner.py`
  - `tests/cli/test_typer_cli.py`
- Phase 4:
  - `tests/build/test_errors.py`
  - `tests/build/hamilton/test_pr13_run_targets.py`
  - `tests/build/hamilton/native/test_skip_logic.py`
- Global gates (as defined in AGENTS.md):
  - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
  - Segmented `pytest -q` by impacted directories

## Acceptance Criteria (overall)
- All decommission targets removed or replaced.
- No references to plugin configuration or plugin identities remain in build execution or tracking.
- CLI plugin subsystem is removed; core plugin types remain intact for ingestion.
- Runtime and test imports resolved without the deprecated modules.
- Architecture and docs reflect native-only build design.

## Risks And Mitigations
- **Risk:** Schema changes to run tracking tables break downstream queries.
  - **Mitigation:** Add temporary compatibility views or a migration script that preserves old column names until callers are updated.
- **Risk:** Tests or tools rely on `plugin_name` semantics.
  - **Mitigation:** Search/replace across `src/` and `tests/` before deletion, update test fixtures.
- **Risk:** Profile defaults become inconsistent.
  - **Mitigation:** Introduce a single build-native `DEFAULT_PROFILE_NAME` and reference it from CLI + build execution.

## Sequencing Recommendation
1) Phase 1 (safe deletes) -> Phase 2 (CLI plugin removal) -> Phase 3 (config removal) -> Phase 4 (record schema) -> Phase 5 (cleanup) -> Phase 6 (optional facade shrink)
2) Run targeted tests after each phase; run full quality gates at the end.
