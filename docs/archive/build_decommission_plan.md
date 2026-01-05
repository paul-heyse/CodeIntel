# Build Decommission Plan (DAG-First Cleanup)

## Purpose
Remove dead, compatibility, and legacy build code so the build subsystem is
fully aligned with the DAG-first design basis. The project is in design mode,
so we can aggressively delete transitional paths and tighten the architecture.

## Scope
In scope:
- Dead or unused modules under `src/codeintel/build`.
- Compatibility layers that allow non-DAG or dual-source behavior.
- Legacy catalog and schema modes that conflict with DAG-first governance.

Out of scope:
- Unrelated refactors outside `src/codeintel/build` (except where required
  to remove direct dependencies on the retired code).
- Performance or feature changes not required for decommissioning.

## Inventory of decommission targets

### Dead or unused modules (no import usage detected)
- `src/codeintel/build/hamilton/runtime_typing.py`
- `src/codeintel/build/hamilton/io_registry.py`
- `src/codeintel/build/hamilton/native/analytics/execution_context.py`

### Compatibility or legacy surfaces
- Output inventory dual-source logic:
  - `src/codeintel/build/output_inventory.py`
  - `src/codeintel/build/target_inventory.py`
  - Settings in `src/codeintel/core/config/settings.py`:
    `output_inventory_source`, `output_inventory_strict`
  - Callers (replace with registry/DAG-based outputs):
    - `src/codeintel/build/hamilton/native/outputs.py`
    - `src/codeintel/build/hamilton/run_records.py`
    - `src/codeintel/build/run_context.py`
    - `src/codeintel/build/spec/compile.py`
    - `src/codeintel/build/schemas/contract_service.py`
    - `src/codeintel/build/target_metadata.py`
- Canonical catalog cache layer:
  - `src/codeintel/build/catalogs/*`
  - `src/codeintel/build/target_catalog.py`
  - `src/codeintel/core/registry/service.py` (uses catalog cache APIs)
- Declared-only schema mode:
  - `src/codeintel/build/schemas/provider_declared.py`
  - `src/codeintel/build/schemas/contract_service.py` (`DECLARED_ONLY` mode)
  - Tests relying on declared-only mode:
    - `tests/build/test_contract_resolution_seams.py`
    - `tests/build/hamilton/test_import_time_schema_safety.py`
- Support nodes source toggle:
  - `BuildSettings.support_nodes_source` in `src/codeintel/core/config/settings.py`
  - Usage in `src/codeintel/build/hamilton/driver_factory.py`
- ExecutionResult compatibility shim:
  - `src/codeintel/build/hamilton/execution_result.py` (`ExecutionResultLike`, `to_execution_result`)
  - Callers that adapt legacy compute outputs to `ExecutionResult`

## Sequenced execution plan

### Phase 0: Pre-flight alignment
Tasks:
- Confirm the DAG output inventory is the authoritative source for target
  outputs and contracts (including artifact outputs if needed).
- Decide on artifact metadata source:
  - Option A: Extend `dag_output_inventory.yaml` to include artifact names
    and path templates.
  - Option B: Derive artifact metadata directly from DAG saver tags at runtime
    and keep the YAML focused on table outputs.
- Freeze any compatibility settings (stop using `output_inventory_source` and
  `support_nodes_source` in new code).

Acceptance:
- A single, agreed source of output metadata is documented.
Status:
- Completed: output inventory derived from saver tags; compatibility toggles removed.

### Phase 1: Remove dead modules
Tasks:
- Delete:
  - `src/codeintel/build/hamilton/runtime_typing.py`
  - `src/codeintel/build/hamilton/io_registry.py`
  - `src/codeintel/build/hamilton/native/analytics/execution_context.py`
- Remove any unused exports or references (if any exist) from package
  `__init__` modules.

Tests:
- Run a focused build/hamilton import test subset after removals:
  - `tests/build/hamilton/test_pr74_auto_mode_native_outputs_have_helpers.py`
  - `tests/build/hamilton/test_pr78_graph_validator_finds_duplicate_producers.py`
Status:
- Completed: dead modules deleted and references removed.

### Phase 2: Consolidate output inventory (DAG-only)
Tasks:
- Replace `OutputInventory` usage with a DAG-first inventory API backed by
  the registry service and/or runtime introspection.
- Update:
  - `src/codeintel/build/hamilton/native/outputs.py`
  - `src/codeintel/build/hamilton/run_records.py`
  - `src/codeintel/build/run_context.py`
  - `src/codeintel/build/spec/compile.py`
  - `src/codeintel/build/schemas/contract_service.py`
  - `src/codeintel/build/target_metadata.py`
- Remove:
  - `src/codeintel/build/output_inventory.py`
  - `src/codeintel/build/target_inventory.py`
- Remove settings:
  - `output_inventory_source`
  - `output_inventory_strict`
  from `src/codeintel/core/config/settings.py` and the loader.

Tests:
- `tests/core/test_dag_output_inventory.py`
- `tests/build/hamilton/test_pr72_manifest_v2.py`
- `tests/build/test_registry.py`
Status:
- Completed: output inventory/target inventory removed; outputs derived from saver tags.

### Phase 3: Remove canonical catalog cache
Tasks:
- Replace `load_target_catalog`/`load_contract_catalog` call sites with direct
  registry or DAG-backed services.
- Delete:
  - `src/codeintel/build/catalogs/*`
  - `src/codeintel/build/target_catalog.py`
- Update `src/codeintel/core/registry/service.py` to use the new registry
  source of truth (DAG + inventory).

Tests:
- `tests/build/test_registry.py`
- `tests/build/hamilton/test_pr66_schema_provider_registry.py`
Status:
- Completed: catalog cache removed; registry uses unified contract service.

### Phase 4: Remove declared-only schema mode
Tasks:
- Remove `declared_schema_provider` and `DECLARED_ONLY` mode.
- Simplify contract resolution to the unified provider only.
- Update or remove tests that assert declared-only behavior:
  - `tests/build/test_contract_resolution_seams.py`
  - `tests/build/hamilton/test_import_time_schema_safety.py`

Tests:
- `tests/build/hamilton/test_pr67_row_binding_parity.py`
- `tests/config/test_datasets_contracts.py`
Status:
- Completed: declared-only mode removed; unified provider only.

### Phase 5: Remove support-nodes source toggle
Tasks:
- Remove `support_nodes_source` from `BuildSettings`.
- Update `src/codeintel/build/hamilton/driver_factory.py` to always derive
  support nodes from the DAG.
- Remove loader wiring from `src/codeintel/core/runtime/loader.py`.

Tests:
- `tests/build/hamilton/test_pr12_loader_nodes.py`
- `tests/build/hamilton/test_pr54_schema_registry_in_build.py`
Status:
- Completed: support nodes derived from saver outputs; toggle removed.

### Phase 6: Remove ExecutionResult compatibility shim
Tasks:
- Update graph compute nodes to return `ExecutionResult` directly.
- Remove `ExecutionResultLike` and `to_execution_result`.
- Simplify `src/codeintel/build/hamilton/materialization_helpers.py`.

Tests:
- `tests/build/hamilton/test_pr21_analytics_native_driver.py`
Status:
- Completed: graph modules now emit `ExecutionResult` directly.

### Phase 7: Final cleanup and verification
Tasks:
- Remove any leftover compatibility settings or dead imports.
- Re-run a targeted quality check for files touched.
- Update `docs/dag_end_to_end_migration_plan.md` with the cleanup outcomes.
Status:
- In progress: documentation and quality/test verification.

Tests:
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Focused pytest slices for the updated areas.

## Risks and mitigations
- Artifact metadata gaps: explicitly decide whether artifact metadata lives in
  the inventory or is derived from DAG introspection.
- CLI or serving regressions: validate plan output and serving snapshot tests
  after the inventory refactor.
- Schema drift: keep contracts and DAG outputs aligned before deletion of
  compatibility code.
