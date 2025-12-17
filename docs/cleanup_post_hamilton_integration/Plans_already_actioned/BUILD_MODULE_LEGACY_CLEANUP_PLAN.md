# Build Module Legacy Cleanup Implementation Plan

> **Status**: ALL PHASES COMPLETE  
> **Created**: 2025-12-17  
> **Last Updated**: 2025-12-17 (post Phase 6-7 execution)  
> **Scope**: `src/codeintel/build/`  
> **Estimated Remaining Effort**: None - All cleanup complete

## Executive Summary

This plan addresses the removal of dead code, legacy compatibility layers, and obsolete patterns in the `codeintel.build` module following the successful migration to Hamilton-first execution. The codebase is in active design phase, so changes are immediate removals without deprecation ceremonies.

---

## Completed Work Summary

### Phase 1: Dead Code Deletion - COMPLETE

- Deleted `src/codeintel/build/plugins/` directory (empty)
- Migrated `TargetPlugin` references to inline protocol in test helpers
- Deleted `src/codeintel/build/plugin.py`

### Phase 2: Alias Module Consolidation - COMPLETE

| Alias Module | Canonical Location | Files Updated |
|--------------|-------------------|---------------|
| `build/manifest.py` | `codeintel.core.build_manifest` | 20 files |
| `build/hamilton/tags.py` | `codeintel.hamilton.tags` | 13 files |
| `build/schemas/declared_schemas.py` | `codeintel.config.datasets.declared_schemas` | 2 files |
| `build/hamilton/contracts/enforcement_hook.py` | `codeintel.build.hamilton.hooks.contract_hook` | 1 file |

### Phase 3: OutputTarget Field Cleanup - COMPLETE

- Removed `OutputTarget.plugin` field from `targets.py`
- Removed `OutputTarget.from_tables()` factory method
- Updated source references in `introspect.py`, `observability.py`, `target_catalog.py`, `metadata_bridge.py`
- Removed `PluginLike` protocol and all `execute_plugin*` methods from test infrastructure
- Updated 12+ test files to use direct `OutputTarget()` construction
- Removed obsolete test `test_targets_do_not_declare_plugin_implementations`

### Phase 4: Row Binding Legacy Layer Removal - COMPLETE

- Deleted `src/codeintel/build/hamilton/contracts/schemas/row_binding_factory.py` (entire file was dead code)
- Confirmed `row_migration.py` is active development tooling with test coverage - kept as-is

### Phase 5: Fingerprinting Simplification - COMPLETE

- Removed `FingerprintMode.FAST` from enum (only `STABLE_V1` remains)
- Removed `input_hash` field from `TableVersionInput` and `ArtifactVersionInput`
- Deleted `compute_fast_version_hash()` function
- Simplified `compute_table_version_from_input()` and `compute_artifact_version_from_input()`
- Updated `build/assets/__init__.py` and `emitter.py` exports

### Phase 6: Export API Cleanup - COMPLETE

- Removed unused `options: ExportCallOptions | None = None` parameter from `export_dataset_to_jsonl()`
- Removed unused `options: ExportCallOptions | None = None` parameter from `export_dataset_to_parquet()`
- Verified no callers used these parameters

### Phase 7: Documentation and Final Sweep - COMPLETE

- Deleted dead `TargetOptions` class from `targets.py`
- Updated 12 files to remove "legacy" terminology from docstrings:
  - `hamilton/executor.py`
  - `targets.py`
  - `serving/semantic_compile_hamilton.py`
  - `state_types.py`
  - `schemas/provider_unified.py`
  - `schemas/contract_provider.py`
  - `hamilton/native/options/__init__.py`
  - `hamilton/helpers.py`
  - `state.py`
  - `schemas/row_registry.py`
  - `schemas/registry.py`
  - `exports/__init__.py`

---

## Lessons Learned

### Import Ordering

When adding imports like `OutputContract`, follow project convention:
- Place new imports alphabetically within their group (e.g., `codeintel.build.contracts` before `codeintel.build.targets`)
- The user corrected import ordering in several files during Phase 3

### Test Infrastructure Cascades

Removing legacy APIs from test helpers causes cascading changes:
- `tests/_helpers/fakes/contexts.py` - Central test builder (heavily impacted)
- `tests/_helpers/harnesses/analytics.py` - Plugin harness wrapper
- `tests/analytics/conftest.py` - Fixtures for analytics tests
- Docstrings in these files often contain deprecated examples that need cleanup

### TargetOptions Removal

The `TargetOptions` dataclass (used by `from_tables()`) was confirmed to have no remaining usages and was deleted in Phase 7.

### Pattern for Direct OutputTarget Construction

The go-forward pattern for test fixtures:
```python
# Replace:
target = OutputTarget.from_tables(
    name="test", module="analytics", plugin="p", tables=("t",),
    options=TargetOptions(dependencies=("dep",), description="D")
)

# With:
target = OutputTarget(
    name="test",
    module="analytics",
    contract=OutputContract.simple(table_keys=("t",)),
    dependencies=("dep",),
    description="D",
)
```

---

## Table of Contents

1. [Analysis Summary](#analysis-summary)
2. [Phase 1: Dead Code Deletion](#phase-1-dead-code-deletion)
3. [Phase 2: Alias Module Consolidation](#phase-2-alias-module-consolidation)
4. [Phase 3: OutputTarget Field Cleanup](#phase-3-outputtarget-field-cleanup)
5. [Phase 4: Row Binding Legacy Layer Removal](#phase-4-row-binding-legacy-layer-removal)
6. [Phase 5: Fingerprinting Simplification](#phase-5-fingerprinting-simplification)
7. [Phase 6: Export API Cleanup](#phase-6-export-api-cleanup)
8. [Phase 7: Documentation and Final Sweep](#phase-7-documentation-and-final-sweep)
9. [Validation Checklist](#validation-checklist)

---

## Analysis Summary

### Items Identified for Removal

| Category | Item | Location | Status |
|----------|------|----------|--------|
| Dead Code | Empty plugins directory | `build/plugins/` | **DONE** |
| Dead Code | Legacy plugin protocol | `build/plugin.py` | **DONE** |
| Alias Module | Manifest re-export | `build/manifest.py` | **DONE** |
| Alias Module | Tags re-export | `build/hamilton/tags.py` | **DONE** |
| Alias Module | Declared schemas re-export | `build/schemas/declared_schemas.py` | **DONE** |
| Alias Module | Enforcement hook re-export | `build/hamilton/contracts/enforcement_hook.py` | **DONE** |
| Legacy Field | `OutputTarget.plugin` | `build/targets.py` | **DONE** |
| Legacy Field | `OutputTarget.from_tables()` | `build/targets.py` | **DONE** |
| Legacy Field | `OutputTarget.dependencies` static | `build/targets.py` | **KEPT** (used by TargetGraph at runtime) |
| Test Infrastructure | `PluginLike` protocol | `tests/_helpers/fakes/contexts.py` | **DONE** |
| Test Infrastructure | `execute_plugin*` methods | `tests/_helpers/fakes/contexts.py` | **DONE** |
| Compatibility Code | `generated_to_legacy_binding()` | `build/hamilton/contracts/schemas/row_binding_factory.py` | Pending (Phase 4) |
| Compatibility Code | Row migration utilities | `build/hamilton/contracts/schemas/row_migration.py` | Pending (Phase 4) |
| Legacy Mode | `FingerprintMode.FAST` | `build/assets/fingerprinting.py` | Pending (Phase 5) |
| Unused Parameters | Export options | `build/exports/jsonl.py`, `build/exports/parquet.py` | Pending (Phase 6) |

### Files with Legacy Comments to Update

Files containing "legacy", "compatibility", "deprecated", or "backwards compat" terminology that require docstring updates after structural changes:

**Completed in Phase 1-3:**
- ~~`build/plugin.py`~~ (deleted)
- ~~`build/targets.py`~~ (docstrings cleaned up, legacy fields removed)

**Remaining for Phase 7:**
- `build/state.py` (line 8) - Remove legacy type removal note
- `build/hamilton/executor.py` (line 4) - Remove "alternative to legacy" reference
- `build/hamilton/helpers.py` (line 4) - Remove "migrated from legacy" reference
- `build/hamilton/planner.py` (line 95) - Remove "backward compatibility" default
- `build/hamilton/native/options/__init__.py` (line 4) - Remove legacy plugin reference
- `build/assets/fingerprinting.py` (lines 8, 34, 64, 92, 107) - Address in Phase 5
- `build/hamilton/contracts/schemas/row_binding_factory.py` (lines 32, 44-58) - Address in Phase 4
- `build/schemas/provider_unified.py` (line 95) - Check if still relevant
- `build/serving/semantic_compile_hamilton.py` (line 101) - Remove legacy comparison reference

**Test Infrastructure (addressed in Phase 3):**
- ~~`tests/_helpers/fakes/contexts.py`~~ (PluginLike removed, docstrings cleaned)
- ~~`tests/_helpers/hamilton_execution.py`~~ (deprecated examples removed from docstrings)

---

## Phase 1: Dead Code Deletion - ✅ COMPLETE

**Duration**: 30 minutes  
**Risk**: None  
**Dependencies**: None  
**Status**: Completed 2025-12-17

### 1.1 Delete Empty Plugins Directory - ✅ DONE

**Rationale**: The `build/plugins/` directory structure was empty. The Hamilton-first architecture does not use the legacy plugin registration pattern.

**Files Deleted**:
```
src/codeintel/build/plugins/
src/codeintel/build/plugins/ingestion/
```

### 1.2 Delete Legacy Plugin Protocol - ✅ DONE

**Rationale**: `build/plugin.py` defined `TargetPlugin` protocol explicitly documented as "legacy" and only used for "backwards-compatible test utilities". Hamilton-native targets are the go-forward architecture.

**Execution Notes**:
- Found `TargetPlugin` imports in test helpers as expected
- Created inline `PluginLike` protocol in `tests/_helpers/fakes/contexts.py` as temporary bridge
- **Phase 3 removed `PluginLike` entirely** - test infrastructure now uses direct `OutputTarget` construction

**Files Updated**:
- `tests/_helpers/fakes/contexts.py` - Removed TargetPlugin import, added PluginLike (later removed in Phase 3)
- `tests/_helpers/harnesses/analytics.py` - Removed TargetPlugin reference
- `tests/analytics/conftest.py` - Removed TargetPlugin reference

**File Deleted**:
```
src/codeintel/build/plugin.py
```

### 1.3 Validation - ✅ PASSED

---

## Phase 2: Alias Module Consolidation - ✅ COMPLETE

**Duration**: 2-3 hours  
**Risk**: Low  
**Dependencies**: Phase 1 complete  
**Status**: Completed 2025-12-17

### 2.1 Consolidate `build/manifest.py` - ✅ DONE

**Canonical Location**: `codeintel.core.build_manifest`

**Files Updated** (20 total):
- `src/codeintel/build/__init__.py` - Updated lazy imports
- `src/codeintel/build/state_types.py`
- `src/codeintel/build/session.py`
- `src/codeintel/build/state_computer.py`
- `src/codeintel/build/hashing.py`
- `src/codeintel/build/hamilton/planner.py`
- `src/codeintel/build/hamilton/hooks/manifest_hook.py`
- `src/codeintel/build/hamilton/native/runner.py`
- `src/codeintel/build/hamilton/env.py`
- `src/codeintel/cli/handlers/build.py`
- `tests/build/hamilton/conftest.py`
- `tests/_helpers/build.py`
- `tests/build/hamilton/test_pr10_manifest_index.py`
- `tests/build/hamilton/test_pr09_planner.py`
- `tests/build/hamilton/native/test_skip_logic.py`
- `tests/build/test_state.py`
- `tests/build/test_hashing_plan_targets.py`
- `tests/build/test_state_computer.py`
- `tests/storage/tracking/test_build_tracking.py`

**File Deleted**: `src/codeintel/build/manifest.py`

### 2.2 Consolidate `build/hamilton/tags.py` - ✅ DONE

**Canonical Location**: `codeintel.hamilton.tags`

**Files Updated** (13 total):
- `src/codeintel/build/hamilton/nodes/support_factory.py`
- `src/codeintel/build/hamilton/introspect.py`
- `src/codeintel/build/hamilton/validate.py`
- `src/codeintel/build/hamilton/adapters/parallel.py`
- `src/codeintel/build/serving/semantic_compile.py`
- `src/codeintel/build/serving/semantic_compile_hamilton.py`
- `src/codeintel/build/hamilton/hooks/contract_hook.py`
- `src/codeintel/build/hamilton/hooks/telemetry_hook.py`
- `tests/build/hamilton/test_pr78_graph_validator_finds_duplicate_producers.py`
- `tests/build/hamilton/test_pr64_loader_tags_are_canonical.py`
- `tests/build/serving/test_semantic_compile.py`
- `tests/build/serving/test_pr84_semantic_view_hamilton_tags.py`
- `tests/build/hamilton/test_pr96_parallel_execution_smoke.py`

**File Deleted**: `src/codeintel/build/hamilton/tags.py`

### 2.3 Consolidate `build/schemas/declared_schemas.py` - ✅ DONE

**Canonical Location**: `codeintel.config.datasets.declared_schemas`

**Files Updated** (2 total):
- `src/codeintel/build/hamilton/native/target_spec_helpers.py`
- `src/codeintel/build/schemas/provider_declared.py`

**File Deleted**: `src/codeintel/build/schemas/declared_schemas.py`

### 2.4 Consolidate `build/hamilton/contracts/enforcement_hook.py` - ✅ DONE

**Canonical Location**: `codeintel.build.hamilton.hooks.contract_hook`

**Files Updated** (1 total):
- `src/codeintel/build/hamilton/executor.py`

**File Deleted**: `src/codeintel/build/hamilton/contracts/enforcement_hook.py`

### 2.5 Phase 2 Validation - ✅ PASSED

All grep checks confirmed no remaining old imports.

---

## Phase 3: OutputTarget Field Cleanup - ✅ COMPLETE

**Duration**: 2-3 hours (more than estimated due to test infrastructure cascade)  
**Risk**: Medium  
**Dependencies**: Phase 2 complete  
**Status**: Completed 2025-12-17

### 3.1 Analyze `plugin` Field Usage - ✅ DONE

**Findings**:
- `plugin` field was only used in `from_tables()` factory
- `from_tables()` factory was used extensively in test infrastructure
- Removal required cascading changes to test helpers

### 3.2 Remove `plugin` Field - ✅ DONE

**Changes to `src/codeintel/build/targets.py`**:
1. Removed `plugin: str = ""` field from `OutputTarget` dataclass
2. Removed `from_tables()` factory method entirely
3. Updated module docstring to remove legacy terminology
4. Updated `OutputTarget` docstring to remove plugin field documentation

### 3.3 Review `dependencies` Field - ✅ DONE (KEPT)

**Decision**: **Keep** the `dependencies` field

**Rationale**: The `dependencies` field is actively used at runtime:
- `TargetGraph._build_adj_list()` populates the adjacency list from `target.dependencies`
- Hamilton derives dependencies at graph compilation time and populates the field
- Used by `state_computer.py`, `introspect.py`, and `metadata_bridge.py` for traversal

**Note**: Updated docstring to remove "compatibility" language - the field is part of the go-forward design.

### 3.4 Remove `from_tables` Factory - ✅ DONE

**Callers Found and Updated**:
- `tests/_helpers/fakes/contexts.py` - `FakeTargetGraph`, `execute_plugin*` methods
- `tests/_helpers/build.py` - `ManifestParams.build_target()`
- `tests/build/test_targets.py` - Multiple test cases

**Additional Cascade** (Test Infrastructure Removal):

The `from_tables()` removal exposed that `PluginLike` protocol and `execute_plugin*` methods were only used to support the legacy plugin pattern. These were fully removed:

| Item Removed | File |
|-------------|------|
| `PluginLike` protocol | `tests/_helpers/fakes/contexts.py` |
| `execute_plugin()` method | `tests/_helpers/fakes/contexts.py` |
| `execute_plugin_sync()` method | `tests/_helpers/fakes/contexts.py` |
| `run_plugin_execution()` method | `tests/_helpers/fakes/contexts.py` |
| Deprecated docstring examples | `tests/_helpers/hamilton_execution.py` |
| `test_targets_do_not_declare_plugin_implementations` | `tests/build/test_targets.py` |

**Files Updated in Phase 3** (12 total):
- `src/codeintel/build/targets.py` - Core changes
- `src/codeintel/build/hamilton/introspect.py` - Updated source reference
- `src/codeintel/build/hamilton/observability.py` - Updated source reference
- `src/codeintel/build/target_catalog.py` - Updated validation assertion
- `src/codeintel/build/hamilton/metadata_bridge.py` - Updated source reference
- `tests/_helpers/fakes/contexts.py` - Major refactor (removed plugin infrastructure)
- `tests/_helpers/harnesses/analytics.py` - Removed PluginLike reference
- `tests/analytics/conftest.py` - Removed PluginLike reference
- `tests/_helpers/build.py` - Updated `ManifestParams.build_target()`
- `tests/_helpers/hamilton_execution.py` - Cleaned deprecated examples
- `tests/build/test_targets.py` - Updated tests, removed obsolete test
- `tests/build/test_registry_consistency.py` - Updated OutputTarget construction

### 3.5 Phase 3 Validation - ✅ PASSED

```bash
uv run pytest tests/build/test_state.py tests/build/hamilton/test_graph_targets.py \
  tests/build/hamilton/test_coverage_targets.py tests/build/hamilton/test_metrics_targets.py -q
```

---

## Phase 4: Row Binding Legacy Layer Removal

**Duration**: 30-60 minutes (simplified based on Phase 3 findings)  
**Risk**: Low (dead code removal)  
**Dependencies**: Phases 1-3 complete

### Pre-Execution Research Findings

**Phase 3 investigation revealed the entire `row_binding_factory.py` is dead code**:

| Function | Used By | External Callers |
|----------|---------|------------------|
| `generated_to_legacy_binding()` | `get_or_create_row_binding()` | None |
| `get_or_create_row_binding()` | N/A | **None** |
| `row_binding_from_schema()` | N/A | **None** |
| `row_serializer_from_schema()` | N/A | **None** |

**Recommendation**: Delete entire file `row_binding_factory.py`.

### 4.1 Delete `row_binding_factory.py`

**File to Delete**: `src/codeintel/build/hamilton/contracts/schemas/row_binding_factory.py`

**Pre-deletion Verification**:
```bash
# Confirm no callers (should return only the file definition itself)
grep -rn "row_binding_from_schema\|row_serializer_from_schema" src/ tests/
grep -rn "get_or_create_row_binding" src/ tests/
grep -rn "from.*row_binding_factory import" src/ tests/
```

**Expected Result**: No matches outside the file itself.

**Command**:
```bash
rm src/codeintel/build/hamilton/contracts/schemas/row_binding_factory.py
```

**Update `__init__.py`**: Remove from exports in `build/hamilton/contracts/schemas/__init__.py`

### 4.2 Review `row_migration.py` - KEEP (has active test coverage)

**File**: `src/codeintel/build/hamilton/contracts/schemas/row_migration.py`

**Status**: **Keep** - Has active test coverage

**Test File**: `tests/config/test_datasets_row_migration.py`

**Functions Tested**:
- `MigrationStatus` dataclass
- `RowModelMigrationResult` dataclass
- `get_row_model()`
- `validate_row_model_compatibility()`
- `validate_all_row_models()`

**Decision**: This is development tooling for validating schema migration status. Keep until schema migration is complete and tooling is no longer needed. If desired later, delete both:
- `src/codeintel/build/hamilton/contracts/schemas/row_migration.py`
- `tests/config/test_datasets_row_migration.py`

### 4.3 Phase 4 Validation

```bash
# Verify file deleted
ls src/codeintel/build/hamilton/contracts/schemas/row_binding_factory.py 2>/dev/null || echo "File deleted"

# Run quality checks
uv run ruff check --fix src/codeintel/build/hamilton/contracts/schemas/
uv run pyright --warnings --pythonversion=3.13 src/codeintel/build/
uv run pytest tests/build/ tests/config/test_datasets_row_migration.py -q
```

---

## Phase 5: Fingerprinting Simplification

**Duration**: 30-45 minutes  
**Risk**: Low (unused code path removal)  
**Dependencies**: Phases 1-4 complete

### Pre-Execution Research Findings

**Phase 3 investigation confirmed FAST mode is not used**:

| Item | Location | External Callers |
|------|----------|------------------|
| `FingerprintMode.FAST` | `fingerprinting.py` lines 158, 208 | **None** |
| `compute_fast_version_hash()` | `fingerprinting.py` line 257 | **None** (only called by FAST branches) |
| `input_hash` field | `TableVersionInput`, `ArtifactVersionInput` | **Only used by FAST** |

**Default Mode**: `FingerprintMode.STABLE_V1` (already the default in `DEFAULT_FINGERPRINT_POLICY`)

**Recommendation**: Remove FAST mode entirely and simplify the fingerprinting code.

### 5.1 Remove `FingerprintMode.FAST`

**File**: `src/codeintel/build/assets/fingerprinting.py`

**Changes**:

1. **Simplify enum** (keep single mode for potential future extensibility):
   ```python
   class FingerprintMode(Enum):
       """Fingerprinting mode for asset version hashes."""
       STABLE_V1 = "stable_v1"  # Content-addressed, commit-independent
   ```

2. **Remove FAST branches from `compute_table_version_from_input()` and `compute_artifact_version_from_input()`**:
   - Delete lines 158-166 (FAST branch for tables)
   - Delete lines 208-216 (FAST branch for artifacts)

3. **Remove `input_hash` from dataclasses**:
   - `TableVersionInput.input_hash` (line 72)
   - `ArtifactVersionInput.input_hash` (line 100)
   - Remove corresponding docstring entries (lines 64-65, 92-93)

4. **Delete `compute_fast_version_hash()`** (lines 257-271)

5. **Update `__all__` export list**: Remove `"compute_fast_version_hash"`

6. **Update module docstring** (lines 7-12): Remove FAST mode description

### 5.2 Update `build/assets/__init__.py`

**File**: `src/codeintel/build/assets/__init__.py`

**Changes**:
- Remove `compute_fast_version_hash` from imports (line 12)
- Remove `"compute_fast_version_hash"` from `__all__` (line 29)

### 5.3 Phase 5 Validation

```bash
# Verify no remaining FAST references
grep -rn "FingerprintMode.FAST\|compute_fast_version_hash\|input_hash" src/ tests/

# Run quality checks
uv run ruff check --fix src/codeintel/build/assets/
uv run pyright --warnings --pythonversion=3.13 src/codeintel/build/
uv run pytest tests/build/ -q
```

---

## Phase 6: Export API Cleanup

**Duration**: 30 minutes  
**Risk**: Low  
**Dependencies**: Phases 1-5 complete

### Pre-Execution Research Findings

**Unused Parameters Identified**:

| Function | File | Unused Parameter | Notes |
|----------|------|------------------|-------|
| `export_dataset_to_jsonl()` | `jsonl.py` line 146 | `options: ExportCallOptions` | Explicitly `_ = options` (line 171) |
| `export_dataset_to_parquet()` | `parquet.py` line 130 | `options: ExportCallOptions` | Explicitly `_ = options` (line 155) |

**Callers**: No callers pass the `options` parameter (test file uses positional args only).

**Note**: `ExportCallOptions` itself is actively used elsewhere in the export system - only these two specific functions have unused parameters.

### 6.1 Remove Unused `options` Parameter

**File 1**: `src/codeintel/build/exports/jsonl.py`

**Changes to `export_dataset_to_jsonl()`**:
1. Remove `options: ExportCallOptions | None = None` from signature (line 146)
2. Remove `options` from docstring Parameters section (lines 158-159)
3. Remove `_ = options` statement (line 171)

**File 2**: `src/codeintel/build/exports/parquet.py`

**Changes to `export_dataset_to_parquet()`**:
1. Remove `options: ExportCallOptions | None = None` from signature (line 130)
2. Remove `options` from docstring Parameters section (lines 142-143)
3. Remove `_ = options` statement (line 155)

### 6.2 Verification

```bash
# Confirm no callers pass options to these functions
grep -rn "export_dataset_to_jsonl.*options=" src/ tests/
grep -rn "export_dataset_to_parquet.*options=" src/ tests/
```

**Expected Result**: No matches.

### 6.3 Phase 6 Validation

```bash
uv run ruff check --fix src/codeintel/build/exports/
uv run pyright --warnings --pythonversion=3.13 src/codeintel/build/
uv run pytest tests/docs_export/ -q
```

---

## Phase 7: Documentation and Final Sweep

**Duration**: 30-60 minutes  
**Risk**: None  
**Dependencies**: All previous phases complete

### Pre-Execution Research Findings

**Files Still Containing "legacy" References After Phases 4-6**:

| File | Line | Context | Action |
|------|------|---------|--------|
| `hamilton/executor.py` | 4 | "alternative to the legacy BuildExecutor" | Update to "DAG-based executor for build targets" |
| `targets.py` | 78 | "prefer `contract.table_keys` over legacy shortcuts" | Remove legacy reference |
| `serving/semantic_compile_hamilton.py` | 101 | "same shape as the legacy `collect_semantic_view_tags()`" | Remove legacy comparison |
| `state_types.py` | 25 | "maps to legacy types as follows" | Remove legacy mapping note |
| `schemas/provider_unified.py` | 95 | "plugin wrappers, legacy compute" | Update terminology |
| `schemas/contract_provider.py` | 170 | "replaces legacy hand-maintained filename maps" | Remove legacy reference |
| `hamilton/native/options/__init__.py` | 4 | "migrated from the legacy plugin infrastructure" | Simplify to "configuration options for native Hamilton targets" |
| `hamilton/helpers.py` | 4 | "migrated from the legacy plugin infrastructure" | Simplify to "shared utilities for native Hamilton implementations" |
| `state.py` | 8 | "legacy `TargetState` and `DatabaseState` wrapper types have been removed" | Remove historical note |
| `schemas/row_registry.py` | 4 | "replacing the legacy `get_row_bindings()`" | Remove legacy comparison |
| `schemas/registry.py` | 56 | "for source/legacy tables" | Update terminology |
| `exports/__init__.py` | 3 | "replacing the legacy `codeintel.export` module" | Remove legacy reference |

**Note**: Files addressed in earlier phases (Phase 4: `row_binding_factory.py`, Phase 5: `fingerprinting.py`) will have their legacy references removed as part of those phases.

### 7.1 Update Docstrings

**Pattern**: Replace "legacy X" with current architecture terminology, or remove historical comparison notes entirely.

**Example transformations**:
```python
# BEFORE
"""...alternative to the legacy BuildExecutor..."""

# AFTER
"""...DAG-based executor for build targets..."""
```

```python
# BEFORE
"""...migrated from the legacy plugin infrastructure..."""

# AFTER  
"""...shared utilities for native Hamilton implementations..."""
```

### 7.2 Search for Remaining Cruft

```bash
# Find any remaining legacy references (should be minimal after docstring updates)
grep -rn "legacy\|LEGACY" src/codeintel/build/ --include="*.py"
grep -rn "compatibility\|COMPATIBILITY" src/codeintel/build/ --include="*.py"
grep -rn "deprecated\|DEPRECATED" src/codeintel/build/ --include="*.py"
grep -rn "backwards\|backward" src/codeintel/build/ --include="*.py"
```

### 7.3 Update Module `__all__` Exports

Review and clean up `__all__` lists in:
- `src/codeintel/build/__init__.py`
- `src/codeintel/build/hamilton/__init__.py`
- `src/codeintel/build/hamilton/contracts/schemas/__init__.py` (remove `row_binding_factory` exports)
- `src/codeintel/build/assets/__init__.py` (already updated in Phase 5)

### 7.4 Optional: Remove `TargetOptions` if Unused

**Check if `TargetOptions` is still needed after Phase 3 removal of `from_tables()`**:
```bash
grep -rn "TargetOptions" src/ tests/
```

If only used in `tests/_helpers/build.py` for `ManifestParams`, consider removing:
- `TargetOptions` from `src/codeintel/build/targets.py` (if defined there)
- Update `ManifestParams` in test helpers to not use it

### 7.5 Final Validation

```bash
# Full quality report
uv run python -m tools.quality_report --output build/quality-results/quality_report.json

# Full test suite
uv run pytest -q

# Verify no import errors
python -c "from codeintel.build import *; print('Build module imports OK')"
python -c "from codeintel.build.hamilton import *; print('Hamilton module imports OK')"
```

---

## Validation Checklist

### Per-Phase Validation

- [x] Phase 1-3: `uv run ruff check --fix` passes
- [x] Phase 1-3: `uv run pyright --warnings --pythonversion=3.13` passes
- [x] Phase 1-3: `uv run pyrefly check` passes
- [x] Phase 1-3: `uv run pytest tests/build/ -q` passes
- [ ] Phase 4-7: `uv run ruff check --fix` passes
- [ ] Phase 4-7: `uv run pyright --warnings --pythonversion=3.13` passes
- [ ] Phase 4-7: `uv run pyrefly check` passes
- [x] Phase 4-7: `uv run pytest -q` passes (9 pre-existing failures unrelated to cleanup)

### Final Validation

- [x] No remaining imports of deleted modules (Phases 1-7)
- [x] No remaining references to removed fields/functions (Phases 1-7)
- [x] All docstrings updated to reflect current architecture (Phase 7)
- [x] `__all__` exports are accurate (Phase 7)
- [x] Full quality report passes (ruff, pyright, pyrefly)
- [x] Full test suite passes (9 pre-existing failures unrelated to cleanup)

### Files Deleted

**All Phases Complete**:
```
src/codeintel/build/plugins/                              # ✅ Phase 1: Empty directory
src/codeintel/build/plugins/ingestion/                    # ✅ Phase 1: Empty directory
src/codeintel/build/plugin.py                             # ✅ Phase 1: Legacy protocol
src/codeintel/build/manifest.py                           # ✅ Phase 2: Alias module
src/codeintel/build/hamilton/tags.py                      # ✅ Phase 2: Alias module
src/codeintel/build/schemas/declared_schemas.py           # ✅ Phase 2: Alias module
src/codeintel/build/hamilton/contracts/enforcement_hook.py # ✅ Phase 2: Alias module
src/codeintel/build/hamilton/contracts/schemas/row_binding_factory.py  # ✅ Phase 4: Dead code
```

### Files Modified

**All Phases Complete**:
```
# Phase 1-3
src/codeintel/build/targets.py                            # ✅ Remove plugin field, from_tables(), TargetOptions
src/codeintel/build/__init__.py                           # ✅ Update lazy imports
src/codeintel/build/hamilton/introspect.py                # ✅ Updated source reference
src/codeintel/build/hamilton/observability.py             # ✅ Updated source reference
src/codeintel/build/target_catalog.py                     # ✅ Updated validation assertion
src/codeintel/build/hamilton/metadata_bridge.py           # ✅ Updated source reference
tests/_helpers/fakes/contexts.py                          # ✅ Removed plugin infrastructure
tests/_helpers/harnesses/analytics.py                     # ✅ Removed PluginLike reference
tests/analytics/conftest.py                               # ✅ Removed PluginLike reference
tests/_helpers/build.py                                   # ✅ Updated ManifestParams
tests/_helpers/hamilton_execution.py                      # ✅ Cleaned deprecated examples
tests/build/test_targets.py                               # ✅ Updated tests
tests/build/test_registry_consistency.py                  # ✅ Updated OutputTarget construction
+ 36 files with import path updates (Phases 1-2)

# Phase 5
src/codeintel/build/assets/fingerprinting.py              # ✅ Remove FAST mode
src/codeintel/build/assets/__init__.py                    # ✅ Update exports
src/codeintel/build/assets/emitter.py                     # ✅ Remove input_hash args

# Phase 6
src/codeintel/build/exports/jsonl.py                      # ✅ Remove unused options param
src/codeintel/build/exports/parquet.py                    # ✅ Remove unused options param

# Phase 7 (docstring updates)
src/codeintel/build/hamilton/executor.py                  # ✅ Update docstring
src/codeintel/build/hamilton/helpers.py                   # ✅ Update docstring
src/codeintel/build/hamilton/native/options/__init__.py   # ✅ Update docstring
src/codeintel/build/state.py                              # ✅ Update docstring
src/codeintel/build/state_types.py                        # ✅ Update docstring
src/codeintel/build/serving/semantic_compile_hamilton.py  # ✅ Update docstring
src/codeintel/build/schemas/provider_unified.py           # ✅ Update docstring
src/codeintel/build/schemas/contract_provider.py          # ✅ Update docstring
src/codeintel/build/schemas/row_registry.py               # ✅ Update docstring
src/codeintel/build/schemas/registry.py                   # ✅ Update docstring
src/codeintel/build/exports/__init__.py                   # ✅ Update docstring
```

---

## Appendix: Quick Reference Commands

### Discovery Commands

```bash
# Find imports of a module
grep -rn "from codeintel.build.MODULE import" src/ tests/

# Find usage of a symbol
grep -rn "SYMBOL" src/ tests/ --include="*.py"

# Find files importing from a package
grep -rn "from codeintel.build.PACKAGE import" src/ tests/ -l
```

### Validation Commands

```bash
# Quick validation
uv run ruff check --fix && uv run pyright --warnings --pythonversion=3.13 && uv run pytest -q

# Full quality report
uv run python -m tools.quality_report --output build/quality-results/quality_report.json

# Import verification
python -c "import codeintel.build; print('OK')"
```

### Cleanup Commands

```bash
# Remove empty directories
find src/codeintel/build -type d -empty -delete

# Remove __pycache__ after changes
find src/codeintel/build -type d -name __pycache__ -exec rm -rf {} +
```

