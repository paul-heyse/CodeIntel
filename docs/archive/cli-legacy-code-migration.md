# CLI Legacy Code Migration Plan

> **Purpose**: Eliminate backward-compatibility code, deprecated patterns, and legacy shims from the CLI and related modules to achieve a cleaner, more maintainable codebase.

---

## Executive Summary

The codebase contains ~15 instances of backward-compatibility code spread across the CLI, serving, config, and graphs modules. Most are well-documented legacy shims or re-exports maintained during previous migrations. This plan provides a phased approach to complete those migrations and remove the compatibility code.

**Estimated effort**: 3-4 days of focused work across 3 phases.

---

## Phase 1: CLI Module Cleanup (Low Risk)

### 1.1 Clean Up RuntimeCliOptions Deprecation Note

**File**: `src/codeintel/cli/resolution/params.py`

**Current State**:
```python
class RuntimeParams:
    """Canonical runtime parameters from any input source.

    This is THE type for runtime parameters. All other RuntimeCliOptions
    variants are deprecated in favor of this single type.
```

**Action**: Verify no `RuntimeCliOptions` types exist elsewhere, then update docstring:
```python
class RuntimeParams:
    """Canonical runtime parameters from any input source.

    Provides a single, unified type for runtime parameters across all CLI
    operations, replacing the previous multiple variants.
```

**Verification**:
```bash
rg "RuntimeCliOptions" src/ tests/
```

---

### 1.2 Modernize CliResult Serialization

**File**: `src/codeintel/cli/core/results.py`

**Current State**:
```python
# First try to_dict() for backward compatibility with existing result types
to_dict = getattr(data, "to_dict", None)
if callable(to_dict):
    return to_dict()
```

**Action**: 
1. Audit all result types to ensure they use dataclasses
2. Remove the duck-typed `to_dict()` fallback
3. Rely solely on `serialize_result()` from `serialization.py`

**Updated Code**:
```python
@staticmethod
def _serialize_data(data: object) -> object:
    """Serialize data for JSON output.

    Returns
    -------
    object
        Serialized representation of the data.
    """
    # Use generic serializer for dataclasses
    if is_dataclass(data) and not isinstance(data, type):
        return serialize_result(data)
    if hasattr(data, "__dict__") and not isinstance(data, type):
        return data.__dict__
    return data
```

**Verification**:
```bash
rg "def to_dict" src/codeintel/cli/
pytest tests/cli/ -v
```

---

### 1.3 Explicit Resource Requirements in OperationSpec

**File**: `src/codeintel/cli/execution/registry.py`

**Current State**: Resource requirements default to `True` for backward compatibility.

**Action**: 
1. Audit all registered operations
2. Set explicit `require_runtime`, `require_gateway`, `require_graph_runtime` values
3. Change defaults to `False` (explicit is better than implicit)

**Before**:
```python
require_runtime: bool = True
require_gateway: bool = True
require_graph_runtime: bool = False
```

**After**:
```python
require_runtime: bool = False
require_gateway: bool = False
require_graph_runtime: bool = False
```

**Migration Script**: Create a codemod to update all `OperationSpec` instantiations with explicit values.

---

### 1.4 Remove Handler Migration Note

**File**: `src/codeintel/cli/handlers/history.py`

**Current State**:
```python
"""Handlers for history timeseries commands.

Migrate to use HandlerContext and return CliResult.
"""
```

**Action**: The handler already uses `HandlerContext` and `CliResult`. Update docstring:
```python
"""Handlers for history timeseries commands.

Provide analytics aggregation across multiple commit snapshots.
"""
```

---

## Phase 2: MCP Tools Consolidation (Medium Risk)

### 2.1 Remove Legacy MCP Tool Modules

**Files to remove**:
- `src/codeintel/serving/mcp/dataset_tools.py`
- `src/codeintel/serving/mcp/profile_tools.py`
- `src/codeintel/serving/mcp/function_tools.py`

**Prerequisite**: Verify all imports have been migrated to use `tool_builder.py` or `tools_base.py`:
```bash
rg "from codeintel.serving.mcp.dataset_tools import" src/ tests/
rg "from codeintel.serving.mcp.profile_tools import" src/ tests/
rg "from codeintel.serving.mcp.function_tools import" src/ tests/
```

**Action**:
1. Update any remaining imports to use canonical locations
2. Delete the legacy modules
3. Update `__init__.py` exports

---

### 2.2 Remove Backend Type Aliases

**File**: `src/codeintel/serving/mcp/backend.py`

**Current State**:
```python
# Type Aliases for Backward Compatibility
BaseBackend = BaseBackendProtocol
```

**Action**:
1. Find all usages of the alias
2. Update to import from `codeintel.serving.types`
3. Remove the alias section

**Verification**:
```bash
rg "from codeintel.serving.mcp.backend import BaseBackend" src/ tests/
```

---

## Phase 3: Config & Storage Cleanup (Medium Risk)

### 3.1 Remove Tuple Export Aliases

**Files**:
- `src/codeintel/config/datasets/rows/core.py`
- `src/codeintel/config/datasets/rows/graph.py`

**Current State**:
```python
# Export to_tuple as method references for backward compatibility
goid_to_tuple = GoidRow.to_tuple
cfg_block_to_tuple = CFGBlockRow.to_tuple
```

**Action**:
1. Find all usages of the aliased functions
2. Update to call methods directly: `GoidRow.to_tuple(row)` → `row.to_tuple()`
3. Remove the aliases from `__all__` and module body

---

### 3.2 Remove Auto-Pipeline Compatibility Shim

**File**: `src/codeintel/serving/auto_pipeline.py`

**Current State**: Function wraps `_run_prereqs_build` and converts `BuildResult` to `PipelineRunRecord`.

**Action**:
1. Identify all callers of the shim
2. Update to use `BuildResult` directly
3. Remove the compatibility shim function
4. Update return type annotations

---

### 3.3 Remove DEPRECATED tables Field

**File**: `src/codeintel/build/targets.py`

**Current State**:
```python
tables
    DEPRECATED: Use contract.table_keys instead.
    Kept for backward compatibility during migration.
```

**Action**:
1. Find all usages of `.tables` attribute
2. Migrate to `.contract.table_keys`
3. Remove the `tables` field from the dataclass

---

### 3.4 Remove Resource Re-exports

**File**: `src/codeintel/core/plugins/__init__.py`

**Current State**:
```python
# Re-export from resources (for backwards compatibility)
from codeintel.core.resources.registry import (
    ResourceNotFoundError,
    ...
)
```

**Action**:
1. Update imports to use canonical `codeintel.core.resources.registry`
2. Remove re-exports from `__init__.py`

---

## Phase 4: Verification & Documentation

### 4.1 Run Full Test Suite
```bash
uv run pytest -q
```

### 4.2 Type Check
```bash
uv run pyright --warnings --pythonversion=3.13
uv run pyrefly check
```

### 4.3 Lint
```bash
uv run ruff check --fix
```

### 4.4 Update AGENTS.md

Remove any references to deprecated patterns or backward-compatibility notes in the operating protocol.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking external consumers | Run `rg` searches before each removal to find all usages |
| Test failures | Run targeted tests after each change |
| Type errors | Run pyright/pyrefly after each phase |
| Missing re-exports | Grep for import patterns before removing |

---

## Success Criteria

1. ✅ No occurrences of "backward", "compat", "deprecated", "legacy" in docstrings/comments (except genuine deprecation warnings)
2. ✅ All tests pass
3. ✅ Type checkers pass with no warnings
4. ✅ Ruff passes with no violations
5. ✅ All result types use canonical dataclass serialization
6. ✅ All MCP tools use unified `tool_builder.py`
7. ✅ All imports use canonical module locations

---

## Implementation Order

```
Week 1:
  Day 1: Phase 1 (CLI cleanup) - All 4 items
  Day 2: Phase 2.1 (MCP tool modules)
  
Week 2:
  Day 1: Phase 2.2 (Backend aliases) + Phase 3.1 (Tuple aliases)
  Day 2: Phase 3.2-3.4 (Auto-pipeline, tables field, re-exports)
  Day 3: Phase 4 (Verification & Documentation)
```

---

## Appendix: Search Commands for Verification

```bash
# Find all backward-compat references
rg -i "backward|compat|legacy|deprecated" src/

# Find RuntimeCliOptions usages
rg "RuntimeCliOptions" src/ tests/

# Find to_dict method definitions in CLI
rg "def to_dict" src/codeintel/cli/

# Find MCP tool imports
rg "from codeintel.serving.mcp\.(dataset|profile|function)_tools" src/ tests/

# Find tuple alias usages
rg "(goid|cfg_block|cfg_edge|dfg)_to_tuple" src/ tests/

# Find tables attribute usages on build targets
rg "\.tables\b" src/codeintel/build/
```
