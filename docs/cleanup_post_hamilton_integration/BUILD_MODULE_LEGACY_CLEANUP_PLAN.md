# Build Module Legacy Cleanup Implementation Plan

> **Status**: Ready for Implementation  
> **Created**: 2025-12-17  
> **Scope**: `src/codeintel/build/`  
> **Estimated Effort**: 6-10 hours total

## Executive Summary

This plan addresses the removal of dead code, legacy compatibility layers, and obsolete patterns in the `codeintel.build` module following the successful migration to Hamilton-first execution. The codebase is in active design phase, so changes are immediate removals without deprecation ceremonies.

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
| Dead Code | Empty plugins directory | `build/plugins/` | Remove |
| Dead Code | Legacy plugin protocol | `build/plugin.py` | Remove |
| Alias Module | Manifest re-export | `build/manifest.py` | Consolidate |
| Alias Module | Tags re-export | `build/hamilton/tags.py` | Consolidate |
| Alias Module | Declared schemas re-export | `build/schemas/declared_schemas.py` | Consolidate |
| Alias Module | Enforcement hook re-export | `build/hamilton/contracts/enforcement_hook.py` | Consolidate |
| Legacy Field | `OutputTarget.plugin` | `build/targets.py` | Remove |
| Legacy Field | `OutputTarget.dependencies` static | `build/targets.py` | Review |
| Compatibility Code | `generated_to_legacy_binding()` | `build/hamilton/contracts/schemas/row_binding_factory.py` | Remove if unused |
| Compatibility Code | Row migration utilities | `build/hamilton/contracts/schemas/row_migration.py` | Review |
| Legacy Mode | `FingerprintMode.FAST` | `build/assets/fingerprinting.py` | Remove if unused |
| Unused Parameters | Export options | `build/exports/jsonl.py`, `build/exports/parquet.py` | Clean up |

### Files with Legacy Comments to Update

Files containing "legacy", "compatibility", "deprecated", or "backwards compat" terminology that require docstring updates after structural changes:

- `build/targets.py` (lines 13-15, 81, 84, 156)
- `build/plugin.py` (entire file - removing)
- `build/state.py` (line 8)
- `build/hamilton/executor.py` (line 4)
- `build/hamilton/helpers.py` (line 4)
- `build/hamilton/planner.py` (line 95)
- `build/hamilton/native/options/__init__.py` (line 4)
- `build/assets/fingerprinting.py` (lines 8, 34, 64, 92, 107)
- `build/hamilton/contracts/schemas/row_binding_factory.py` (lines 32, 44-58)
- `build/schemas/provider_unified.py` (line 95)

---

## Phase 1: Dead Code Deletion

**Duration**: 30 minutes  
**Risk**: None  
**Dependencies**: None

### 1.1 Delete Empty Plugins Directory

**Rationale**: The `build/plugins/` directory structure is empty. The Hamilton-first architecture does not use the legacy plugin registration pattern.

**Files to Delete**:
```
src/codeintel/build/plugins/
src/codeintel/build/plugins/ingestion/
```

**Command**:
```bash
rm -rf src/codeintel/build/plugins/
```

### 1.2 Delete Legacy Plugin Protocol

**Rationale**: `build/plugin.py` defines `TargetPlugin` protocol explicitly documented as "legacy" and only used for "backwards-compatible test utilities". Hamilton-native targets are the go-forward architecture.

**File to Delete**:
```
src/codeintel/build/plugin.py
```

**Pre-deletion Check** - Verify no imports remain:
```bash
grep -r "from codeintel.build.plugin import" src/ tests/
grep -r "from codeintel.build import.*TargetPlugin" src/ tests/
grep -r "TargetPlugin" src/ tests/ --include="*.py"
```

**Expected Result**: Only matches in:
- `build/plugin.py` itself (deleting)
- Documentation files (update references)
- Test helper files (update to remove dependency)

**Files Requiring Import Updates** (if any found):
- `tests/_helpers/fakes/contexts.py` - Remove TargetPlugin usage
- `tests/_helpers/harnesses/analytics.py` - Remove TargetPlugin usage
- `tests/analytics/conftest.py` - Remove TargetPlugin usage

**Command**:
```bash
rm src/codeintel/build/plugin.py
```

### 1.3 Validation

```bash
uv run ruff check src/codeintel/build/
uv run pyright --warnings --pythonversion=3.13 src/codeintel/build/
uv run pytest tests/build/ -q
```

---

## Phase 2: Alias Module Consolidation

**Duration**: 2-3 hours  
**Risk**: Low  
**Dependencies**: Phase 1 complete

### 2.1 Consolidate `build/manifest.py`

**Current State**: Re-exports from `codeintel.core.build_manifest`

```python
# src/codeintel/build/manifest.py (CURRENT - TO DELETE)
from codeintel.core.build_manifest import (
    BuildRunRecord,
    BuildStatus,
    OutputManifest,
)
```

**Canonical Location**: `codeintel.core.build_manifest`

**Migration Steps**:

1. **Find all imports**:
   ```bash
   grep -rn "from codeintel.build.manifest import" src/ tests/
   ```

2. **Update each file** - Replace:
   ```python
   # OLD
   from codeintel.build.manifest import BuildRunRecord, OutputManifest
   
   # NEW
   from codeintel.core.build_manifest import BuildRunRecord, OutputManifest
   ```

3. **Known files requiring updates** (from analysis):
   - `src/codeintel/build/hamilton/executor.py`
   - `src/codeintel/build/hamilton/native/target_spec_helpers.py`
   - `src/codeintel/build/hamilton/nodes/support_factory.py`
   - `src/codeintel/build/hamilton/introspect.py`
   - `src/codeintel/build/hamilton/planner.py`
   - `src/codeintel/build/hamilton/validate.py`
   - `src/codeintel/build/hamilton/hooks/manifest_hook.py`
   - `src/codeintel/build/hamilton/native/runner.py`
   - `src/codeintel/build/hamilton/env.py`
   - `src/codeintel/build/session.py`
   - `src/codeintel/build/state_computer.py`
   - `src/codeintel/build/state_types.py`
   - `src/codeintel/build/hashing.py`
   - `src/codeintel/build/__init__.py`
   - `src/codeintel/cli/handlers/build.py`
   - `tests/build/hamilton/conftest.py`
   - `tests/build/hamilton/test_pr10_manifest_index.py`
   - `tests/build/hamilton/test_pr09_planner.py`
   - `tests/build/hamilton/test_pr78_graph_validator_finds_duplicate_producers.py`
   - `tests/build/hamilton/test_pr64_loader_tags_are_canonical.py`
   - `tests/build/hamilton/native/test_skip_logic.py`
   - `tests/build/test_state.py`
   - `tests/build/test_hashing_plan_targets.py`
   - `tests/build/test_state_computer.py`
   - `tests/storage/tracking/test_build_tracking.py`
   - `tests/_helpers/build.py`

4. **Delete alias module**:
   ```bash
   rm src/codeintel/build/manifest.py
   ```

5. **Update `build/__init__.py`** - Remove from lazy imports if present

### 2.2 Consolidate `build/hamilton/tags.py`

**Current State**: Re-exports from `codeintel.hamilton.tags`

**Canonical Location**: `codeintel.hamilton.tags`

**Migration Steps**:

1. **Find all imports**:
   ```bash
   grep -rn "from codeintel.build.hamilton.tags import" src/ tests/
   grep -rn "from codeintel.build.hamilton import tags" src/ tests/
   ```

2. **Update each file** - Replace:
   ```python
   # OLD
   from codeintel.build.hamilton.tags import TAG_TARGET, TAG_TABLE_KEY
   # or
   from codeintel.build.hamilton import tags as ht
   
   # NEW
   from codeintel.hamilton.tags import TAG_TARGET, TAG_TABLE_KEY
   # or
   from codeintel.hamilton import tags as ht
   ```

3. **Delete alias module**:
   ```bash
   rm src/codeintel/build/hamilton/tags.py
   ```

### 2.3 Consolidate `build/schemas/declared_schemas.py`

**Current State**: Re-exports from `codeintel.config.datasets.declared_schemas`

**Canonical Location**: `codeintel.config.datasets.declared_schemas`

**Migration Steps**:

1. **Find all imports**:
   ```bash
   grep -rn "from codeintel.build.schemas.declared_schemas import" src/ tests/
   ```

2. **Update each file** - Replace:
   ```python
   # OLD
   from codeintel.build.schemas.declared_schemas import TABLE_SCHEMAS, COMPOSITE_SCHEMAS
   
   # NEW
   from codeintel.config.datasets.declared_schemas import TABLE_SCHEMAS, COMPOSITE_SCHEMAS
   ```

3. **Known file requiring update**:
   - `src/codeintel/build/schemas/provider_declared.py`

4. **Delete alias module**:
   ```bash
   rm src/codeintel/build/schemas/declared_schemas.py
   ```

### 2.4 Consolidate `build/hamilton/contracts/enforcement_hook.py`

**Current State**: Re-exports `ContractEnforcementHook` from `hooks/contract_hook.py`

**Canonical Location**: `codeintel.build.hamilton.hooks.contract_hook`

**Migration Steps**:

1. **Find all imports**:
   ```bash
   grep -rn "from codeintel.build.hamilton.contracts.enforcement_hook import" src/ tests/
   grep -rn "from codeintel.build.hamilton.contracts import.*enforcement" src/ tests/
   ```

2. **Update each file** - Replace:
   ```python
   # OLD
   from codeintel.build.hamilton.contracts.enforcement_hook import ContractEnforcementHook
   
   # NEW
   from codeintel.build.hamilton.hooks.contract_hook import ContractEnforcementHook
   ```

3. **Delete alias module**:
   ```bash
   rm src/codeintel/build/hamilton/contracts/enforcement_hook.py
   ```

### 2.5 Phase 2 Validation

```bash
# Verify no remaining old imports
grep -r "from codeintel.build.manifest import" src/ tests/
grep -r "from codeintel.build.hamilton.tags import" src/ tests/
grep -r "from codeintel.build.schemas.declared_schemas import" src/ tests/
grep -r "from codeintel.build.hamilton.contracts.enforcement_hook import" src/ tests/

# Run quality checks
uv run ruff check --fix src/ tests/
uv run pyright --warnings --pythonversion=3.13
uv run pytest -q
```

---

## Phase 3: OutputTarget Field Cleanup

**Duration**: 1-2 hours  
**Risk**: Medium  
**Dependencies**: Phase 2 complete

### 3.1 Analyze `plugin` Field Usage

**File**: `src/codeintel/build/targets.py`

**Current Definition** (line ~119):
```python
@dataclass(frozen=True)
class OutputTarget:
    name: str
    module: TargetModule
    plugin: str = ""  # <-- LEGACY FIELD
    ...
```

**Check Usage**:
```bash
grep -rn "\.plugin" src/codeintel/build/ --include="*.py"
grep -rn "plugin=" src/codeintel/build/ --include="*.py" | grep -v "plugin_name"
```

**Expected**: No meaningful usage - field exists for historical compatibility only.

### 3.2 Remove `plugin` Field

**Changes to `src/codeintel/build/targets.py`**:

1. Remove field from dataclass:
   ```python
   # REMOVE THIS LINE
   plugin: str = ""
   ```

2. Remove from `from_tables` factory if it references `plugin`:
   ```python
   # Update factory method to not pass plugin
   ```

3. Update docstring to remove legacy references (lines 81-84):
   ```python
   # REMOVE from docstring:
   # plugin
   #     Legacy implementation identifier (empty in Hamilton-first execution).
   ```

4. Update architecture note (lines 13-15) to remove backward compatibility mention

### 3.3 Review `dependencies` Field

**Current State**: The docstring states dependencies should be derived from Hamilton DAG, but the field remains for "compatibility with historical tooling".

**Decision Point**: 
- If `dependencies` is never statically populated (always empty tuple) → Remove field
- If still used for some tooling → Keep but update docstring

**Check**:
```bash
grep -rn "dependencies=" src/codeintel/build/ --include="*.py"
grep -rn "\.dependencies" src/codeintel/build/ --include="*.py"
```

**Action**: Based on findings:
- If used by `TargetGraph` for dependency traversal → Keep (Hamilton derives these at runtime)
- If only legacy tooling → Remove

### 3.4 Remove `from_tables` Factory if Unused

**Current Definition** (lines 144-175):
```python
@classmethod
def from_tables(
    cls,
    *,
    name: str,
    module: TargetModule,
    plugin: str = "",  # Legacy parameter
    tables: Iterable[str],
    options: TargetOptions | None = None,
) -> OutputTarget:
    """Create an OutputTarget from table keys and optional artifacts.

    This factory provides compatibility for legacy call sites that
    previously passed ``tables=...`` directly to the constructor.
    ...
    """
```

**Check Usage**:
```bash
grep -rn "OutputTarget.from_tables" src/ tests/
grep -rn "\.from_tables(" src/ tests/
```

**Action**: If no callers found, remove the entire factory method.

### 3.5 Phase 3 Validation

```bash
uv run ruff check --fix src/codeintel/build/targets.py
uv run pyright --warnings --pythonversion=3.13 src/codeintel/build/
uv run pytest tests/build/ -q
```

---

## Phase 4: Row Binding Legacy Layer Removal

**Duration**: 1-2 hours  
**Risk**: Medium  
**Dependencies**: Phases 1-3 complete

### 4.1 Analyze `generated_to_legacy_binding()` Usage

**File**: `src/codeintel/build/hamilton/contracts/schemas/row_binding_factory.py`

**Function** (lines 44-63):
```python
def generated_to_legacy_binding(generated: GeneratedRowBinding) -> RowBinding:
    """Convert a GeneratedRowBinding to a legacy RowBinding.

    This adapter allows schema-generated bindings to be used in places
    that still expect the legacy RowBinding dataclass.
    ...
    """
```

**Check Usage**:
```bash
grep -rn "generated_to_legacy_binding" src/ tests/
grep -rn "from.*row_binding_factory import" src/ tests/
```

**Expected Callers**:
- `get_or_create_row_binding()` in same file (line 144)
- Possibly external callers

### 4.2 Decision Matrix for `row_binding_factory.py`

| Scenario | Action |
|----------|--------|
| No external callers of `generated_to_legacy_binding` | Remove function, update `get_or_create_row_binding` to return `GeneratedRowBinding` |
| External callers exist | Update callers to use `GeneratedRowBinding` directly, then remove |

### 4.3 Review `row_migration.py` Necessity

**File**: `src/codeintel/build/hamilton/contracts/schemas/row_migration.py`

**Purpose**: Provides `validate_row_model_compatibility()` for migration validation.

**Decision Point**:
- If schema migration is complete → Remove entire file
- If migration validation tooling is still useful → Keep as development tool

**Check**:
```bash
grep -rn "from.*row_migration import" src/ tests/
grep -rn "validate_row_model_compatibility" src/ tests/
```

### 4.4 Implementation

**If removing `generated_to_legacy_binding`**:

1. Update `get_or_create_row_binding()` to return `GeneratedRowBinding`:
   ```python
   def get_or_create_row_binding(table_key: str) -> GeneratedRowBinding:
       """Get GeneratedRowBinding from schema registry.
       ...
       """
       return get_row_binding(table_key)
   ```

2. Update all callers to expect `GeneratedRowBinding`

3. Remove `generated_to_legacy_binding()` function

4. Update `__all__` export list

**If removing `row_migration.py`**:
```bash
rm src/codeintel/build/hamilton/contracts/schemas/row_migration.py
```

### 4.5 Phase 4 Validation

```bash
uv run ruff check --fix src/codeintel/build/hamilton/contracts/schemas/
uv run pyright --warnings --pythonversion=3.13 src/codeintel/build/
uv run pytest tests/build/ -q
```

---

## Phase 5: Fingerprinting Simplification

**Duration**: 1 hour  
**Risk**: Low  
**Dependencies**: Phases 1-4 complete

### 5.1 Analyze `FingerprintMode.FAST` Usage

**File**: `src/codeintel/build/assets/fingerprinting.py`

**Current Modes**:
```python
class FingerprintMode(Enum):
    FAST = "fast"        # Legacy mode
    STABLE_V1 = "stable_v1"  # Recommended mode
```

**Check Usage**:
```bash
grep -rn "FingerprintMode.FAST" src/ tests/
grep -rn "FingerprintMode\.FAST" src/ tests/
grep -rn '"fast"' src/codeintel/build/assets/
```

### 5.2 Decision Matrix

| Scenario | Action |
|----------|--------|
| No usage of `FAST` mode | Remove `FAST` from enum, simplify policy class |
| Usage exists | Document why, keep or migrate callers |

### 5.3 Implementation (if removing FAST mode)

**Changes to `src/codeintel/build/assets/fingerprinting.py`**:

1. Remove `FAST` from enum:
   ```python
   class FingerprintMode(Enum):
       STABLE_V1 = "stable_v1"
   ```

2. Simplify `FingerprintPolicy.compute_table_version_from_input()`:
   - Remove FAST mode branch
   - Remove `input_hash` parameter from input dataclasses if only used by FAST

3. Update `TableVersionInput` and `ArtifactVersionInput`:
   - Remove `input_hash` field if only used by FAST mode

4. Update docstrings to remove legacy references

5. Remove `compute_fast_version_hash()` if only used by FAST mode

### 5.4 Phase 5 Validation

```bash
uv run ruff check --fix src/codeintel/build/assets/
uv run pyright --warnings --pythonversion=3.13 src/codeintel/build/
uv run pytest tests/build/ -q
```

---

## Phase 6: Export API Cleanup

**Duration**: 1 hour  
**Risk**: Low  
**Dependencies**: Phases 1-5 complete

### 6.1 Identify Unused Parameters

**Files**:
- `src/codeintel/build/exports/jsonl.py`
- `src/codeintel/build/exports/parquet.py`

**Pattern to Find** (from analysis):
```python
# Parameters marked as "unused, for API compatibility"
options: ExportCallOptions  # Check if truly unused
```

**Check Usage**:
```bash
grep -rn "ExportCallOptions" src/ tests/
grep -rn "export_all_jsonl\|export_all_parquet" src/ tests/
```

### 6.2 Implementation

1. Review each export function signature
2. Remove parameters confirmed as unused
3. Update callers if any were passing unused parameters
4. Simplify function bodies

### 6.3 Phase 6 Validation

```bash
uv run ruff check --fix src/codeintel/build/exports/
uv run pyright --warnings --pythonversion=3.13 src/codeintel/build/
uv run pytest tests/build/exports/ -q
```

---

## Phase 7: Documentation and Final Sweep

**Duration**: 30-60 minutes  
**Risk**: None  
**Dependencies**: All previous phases complete

### 7.1 Update Docstrings

Remove/update "legacy", "compatibility", "deprecated", "backwards compat" terminology from:

| File | Lines | Content to Update |
|------|-------|-------------------|
| `build/targets.py` | 7-22 | Architecture note about Hamilton-first |
| `build/state.py` | 8 | Remove legacy type removal note |
| `build/hamilton/executor.py` | 4 | Remove "alternative to legacy" reference |
| `build/hamilton/helpers.py` | 4 | Remove "migrated from legacy" reference |
| `build/hamilton/planner.py` | 95 | Remove "backward compatibility" default |
| `build/hamilton/native/options/__init__.py` | 4 | Remove legacy plugin reference |
| `build/serving/semantic_compile_hamilton.py` | 101 | Remove legacy comparison reference |

### 7.2 Search for Remaining Cruft

```bash
# Find any remaining legacy references
grep -rn "legacy\|LEGACY" src/codeintel/build/ --include="*.py"
grep -rn "compatibility\|COMPATIBILITY" src/codeintel/build/ --include="*.py"
grep -rn "deprecated\|DEPRECATED" src/codeintel/build/ --include="*.py"
grep -rn "backwards\|backward" src/codeintel/build/ --include="*.py"
```

### 7.3 Update Module `__all__` Exports

Review and clean up `__all__` lists in:
- `src/codeintel/build/__init__.py`
- `src/codeintel/build/hamilton/__init__.py`
- `src/codeintel/build/hamilton/contracts/schemas/__init__.py`

### 7.4 Final Validation

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

- [ ] `uv run ruff check --fix` passes
- [ ] `uv run pyright --warnings --pythonversion=3.13` passes
- [ ] `uv run pyrefly check` passes
- [ ] `uv run pytest -q` passes

### Final Validation

- [ ] No remaining imports of deleted modules
- [ ] No remaining references to removed fields/functions
- [ ] All docstrings updated to reflect current architecture
- [ ] `__all__` exports are accurate
- [ ] Full quality report passes
- [ ] Full test suite passes

### Files Deleted (Expected)

```
src/codeintel/build/plugins/                              # Empty directory
src/codeintel/build/plugins/ingestion/                    # Empty directory
src/codeintel/build/plugin.py                             # Legacy protocol
src/codeintel/build/manifest.py                           # Alias module
src/codeintel/build/hamilton/tags.py                      # Alias module
src/codeintel/build/schemas/declared_schemas.py           # Alias module
src/codeintel/build/hamilton/contracts/enforcement_hook.py # Alias module
src/codeintel/build/hamilton/contracts/schemas/row_migration.py  # If migration complete
```

### Files Modified (Expected)

```
src/codeintel/build/targets.py                            # Remove plugin field, update docstrings
src/codeintel/build/__init__.py                           # Update lazy imports
src/codeintel/build/hamilton/__init__.py                  # Update exports
src/codeintel/build/hamilton/contracts/schemas/__init__.py # Update exports
src/codeintel/build/hamilton/contracts/schemas/row_binding_factory.py # Remove legacy conversion
src/codeintel/build/assets/fingerprinting.py              # Simplify modes
src/codeintel/build/exports/jsonl.py                      # Clean API
src/codeintel/build/exports/parquet.py                    # Clean API
src/codeintel/build/state.py                              # Update docstring
src/codeintel/build/hamilton/executor.py                  # Update docstring
src/codeintel/build/hamilton/helpers.py                   # Update docstring
src/codeintel/build/hamilton/planner.py                   # Update docstring
src/codeintel/build/hamilton/native/options/__init__.py   # Update docstring
src/codeintel/build/serving/semantic_compile_hamilton.py  # Update docstring
+ ~25 files with import path updates
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

