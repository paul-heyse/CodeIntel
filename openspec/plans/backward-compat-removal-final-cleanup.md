# Implementation Plan: Final Backward Compatibility Removal Cleanup

## Overview and Context

### Background

This plan represents the final phase of a multi-stage effort to consolidate and rationalize the CodeIntel architecture. Previous phases have:

1. **Unified the registry system** - Made `PipelineStep` implement `RegistrablePlugin` via a `metadata` property, enabling `StepPluginRegistry` to extend `BasePluginRegistry`
2. **Established a layered context hierarchy** - Created `BaseContext` as the root for all execution contexts
3. **Centralized gateway caching** - Moved `gateway_cache.py` from `pipeline` to `storage` layer
4. **Removed deprecated aliases** - Cleaned up `IngestRuntimeScratch`, `IngestResourceHints`, `StepRegistry`, and various executor aliases
5. **Made `DuckDBBackend.service` required** - Removed the deprecated `service_override` pattern

### Current State

Static analysis (pyright and pyrefly) now reports **26 errors** stemming from incomplete migration of call sites after the `DuckDBBackend` refactoring and related changes. These errors fall into five distinct categories:

1. **CLI commands calling `DuckDBBackend` directly** - Need to use bootstrap helpers or explicit service construction
2. **Missing `get_deps` method** - `StepPluginRegistry` lacks a backward-compatible `get_deps` method
3. **Type variance issues** - `DuckDBBackend.service` type is narrower than parent protocol
4. **Incomplete protocol implementation** - `_ModelLike` protocol missing `...` body
5. **Test fixtures** - Missing sqlalchemy (optional dependency) and missing `service` arguments

### Objectives

1. **Zero errors** from pyright, pyrefly, and ruff
2. **No `# type: ignore` or `# noqa` suppressions**
3. **Structural solutions** that address root causes, not symptoms
4. **Maintain backward compatibility** where semantically correct
5. **Clean, idiomatic code** following project conventions

---

## Phase 1: Fix CLI Commands

### 1.1 Fix `cli/commands/ide.py` - Incorrect `build_backend_resource` Call

**Context:**
The `ide.py` command provides IDE hints for files. It was partially migrated to use `build_backend_resource` but with the wrong signature. The function requires a `ServingConfig` as its first positional argument, not `gateway`/`repo`/`commit` as keyword args.

**Current Code (lines 99-104):**
```python
resource = build_backend_resource(
    gateway=gateway,
    repo=runtime.project.repo,
    commit=runtime.cfg.repo.commit,
    options=BootstrapOptions(graph_runtime=graph_runtime),
)
```

**Issue:**
- `build_backend_resource` signature is: `(cfg: ServingConfig, *, gateway, http_client, options)`
- Using `BootstrapOptions` instead of `BackendResourceOptions`
- Passing `repo` and `commit` which are not parameters

**Fix:**
1. Build a `ServingConfig` from the runtime configuration
2. Pass it as the first positional argument
3. Use `BackendResourceOptions` for the options parameter
4. Remove `repo`/`commit` keyword arguments (they come from `ServingConfig`)

**Files Changed:**
- `src/codeintel/cli/commands/ide.py`

**Estimated Lines Changed:** ~15

---

### 1.2 Fix `cli/commands/pipeline.py` - Missing `get_deps` Method

**Context:**
The pipeline CLI's `deps` subcommand shows dependencies for a given step. It calls `REGISTRY.get_deps(step_name)` but `StepPluginRegistry` doesn't expose this method.

**Current Code (line 323):**
```python
direct_deps = tuple(REGISTRY.get_deps(step_name))
```

**Issue:**
- `StepPluginRegistry` has `expand_with_deps` for transitive dependencies
- Steps have `metadata.depends_on` for direct dependencies
- No `get_deps(name)` method exists

**Fix Options:**

**Option A: Add `get_deps` method to `StepPluginRegistry`** (Recommended)
```python
def get_deps(self, name: str) -> tuple[str, ...]:
    """Return direct dependencies for a step."""
    return self[name].metadata.depends_on
```

**Option B: Update CLI to use step metadata directly**
```python
step = REGISTRY[step_name]
direct_deps = step.metadata.depends_on
```

**Recommendation:** Option A provides a cleaner API and maintains symmetry with other registry methods.

**Files Changed:**
- `src/codeintel/pipeline/steps/plugin_registry.py` (add method)
- No changes to `pipeline.py` needed if Option A

**Estimated Lines Changed:** ~12

---

### 1.3 Fix `cli/commands/subsystem.py` - Direct `DuckDBBackend` Construction

**Context:**
The subsystem CLI commands construct `DuckDBBackend` directly without the now-required `service` parameter.

**Current Code (lines 133-138):**
```python
return DuckDBBackend(
    gateway=gateway,
    repo=runtime.project.repo,
    commit=runtime.cfg.repo.commit,
    query_engine=engine,
)
```

**Issue:**
- `DuckDBBackend.__init__` now requires `service: LocalQueryService`
- CLI commands should use bootstrap infrastructure for consistency

**Fix:**
Use `build_backend_resource` from `codeintel.serving.bootstrap`:
1. Build `ServingConfig` from runtime
2. Call `build_backend_resource` with proper options
3. Return `resource.backend`

**Files Changed:**
- `src/codeintel/cli/commands/subsystem.py`

**Estimated Lines Changed:** ~25

---

## Phase 2: Fix Serving Layer Type Issues

### 2.1 Fix `serving/bootstrap.py` - Service Type Variance

**Context:**
`build_backend_resource` creates a `service` that can be either `LocalQueryService` or `HttpQueryService`, but `DuckDBBackend` only accepts `LocalQueryService`.

**Current Code (lines 758-766):**
```python
backend = duckdb_backend_cls(
    service=service,  # <-- LocalQueryService | HttpQueryService
    gateway=gateway,
    ...
)
```

**Issue:**
- `service` variable has type `LocalQueryService | HttpQueryService`
- `DuckDBBackend.__init__` parameter is typed as `LocalQueryService`
- This is intentional design - DuckDB needs local access

**Fix:**
Add type guard to ensure only `LocalQueryService` is passed to `DuckDBBackend`:
```python
if not isinstance(service, LocalQueryService):
    msg = "DuckDBBackend requires LocalQueryService, got HttpQueryService"
    raise TypeError(msg)
backend = duckdb_backend_cls(service=service, ...)
```

**Files Changed:**
- `src/codeintel/serving/bootstrap.py`

**Estimated Lines Changed:** ~8

---

### 2.2 Fix `serving/mcp/backend.py` - Protocol Variance Issue

**Context:**
`DuckDBBackend.service` is typed as `LocalQueryService`, but parent protocol `DatasetBackendMixin` and `AggregatedBackendProtocol` declare it as `QueryService` (the broader type).

**Current Code (line 258):**
```python
service: LocalQueryService
```

**Issue:**
- Read-write attributes cannot be narrowed in subclasses (Liskov substitution)
- `DuckDBBackend` intentionally requires the narrower type
- This is a structural type system constraint

**Fix Options:**

**Option A: Use `ClassVar` annotation** (Not applicable - `service` is instance attribute)

**Option B: Keep broader type, narrow in `__init__`** (Recommended)
```python
# In class body:
service: QueryService  # Broader type for protocol compatibility

# In __init__:
def __init__(self, service: LocalQueryService, ...):
    # Runtime check is redundant but makes intent clear
    self.service = service
```

**Option C: Make parent protocols use `LocalQueryService`** (Breaking change)

**Recommendation:** Option B - keep the attribute type broad for protocol compatibility but enforce the narrow type through `__init__` parameter.

**Files Changed:**
- `src/codeintel/serving/mcp/backend.py`

**Estimated Lines Changed:** ~5

---

### 2.3 Fix `serving/mcp/tool_builder.py` - Missing Return Statement

**Context:**
The `_ModelLike` protocol class has a method declaration without a body.

**Current Code (lines 75-76):**
```python
def model_dump(self) -> dict[str, object]:
    """Serialize model to dictionary."""
```

**Issue:**
- Protocol methods should have `...` as body
- Missing body causes pyright/pyrefly to expect a return statement

**Fix:**
Add ellipsis body to protocol method:
```python
def model_dump(self) -> dict[str, object]:
    """Serialize model to dictionary."""
    ...
```

**Files Changed:**
- `src/codeintel/serving/mcp/tool_builder.py`

**Estimated Lines Changed:** ~1

---

## Phase 3: Fix Test Files

### 3.1 Fix `tests/serving/mcp/test_mcp_backend_comprehensive.py` - DuckDBBackend Calls

**Context:**
This test file has **13 instances** of direct `DuckDBBackend` construction without the required `service` parameter.

**Affected Lines:**
- Line 45, 69, 417, 1085, 1114, 1148, 1168, 1188, 1209, 1230, 1255, 1275, 1300

**Issue:**
- All these tests construct `DuckDBBackend` for isolated testing
- They need to provide a `LocalQueryService` instance

**Fix:**
Update each test to use the `build_duckdb_backend` helper from `tests/_helpers/gateway.py`:

```python
# Before:
backend = DuckDBBackend(
    gateway=provisioned_repo.gateway,
    repo=provisioned_repo.repo,
    commit=provisioned_repo.commit,
)

# After:
from tests._helpers.gateway import build_duckdb_backend, DuckDBBackendOptions

backend = build_duckdb_backend(DuckDBBackendOptions(
    gateway=provisioned_repo.gateway,
    repo=provisioned_repo.repo,
    commit=provisioned_repo.commit,
))
```

**Files Changed:**
- `tests/serving/mcp/test_mcp_backend_comprehensive.py`

**Estimated Lines Changed:** ~65 (13 call sites × ~5 lines each)

---

### 3.2 Fix Test Fixtures - Missing SQLAlchemy Import

**Context:**
Test fixture files in `tests/fixtures/heuristics/` import sqlalchemy, which is not installed.

**Affected Files:**
- `tests/fixtures/heuristics/models_sqlalchemy.py`
- `tests/fixtures/heuristics/service_usage.py`

**Issue:**
- These are fixture files for heuristics testing
- SQLAlchemy is an optional dependency not in the test environment
- The imports fail because the package isn't installed

**Fix Options:**

**Option A: Add SQLAlchemy to dev dependencies** (Adds weight)

**Option B: Guard imports with TYPE_CHECKING** (Recommended)
```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy import Column, ForeignKey, Integer, String
    from sqlalchemy.orm import declarative_base, relationship
```

**Option C: Add pyright/pyrefly ignore for test fixtures** (Not allowed per policy)

**Recommendation:** Option B - these are test fixtures for static analysis patterns, not runtime code.

**Files Changed:**
- `tests/fixtures/heuristics/models_sqlalchemy.py`
- `tests/fixtures/heuristics/service_usage.py`

**Estimated Lines Changed:** ~15

---

## Phase 4: Final Verification

### 4.1 Run Quality Tools

Execute all quality checks in sequence:

```bash
# Format and lint
uv run ruff format src/ tests/
uv run ruff check --fix src/ tests/

# Type checking
uv run pyright --warnings --pythonversion=3.13
uv run pyrefly check
```

**Expected Outcome:** 0 errors, 0 warnings

### 4.2 Run Affected Tests

Execute tests for all modified modules:

```bash
# CLI tests
uv run pytest tests/cli/ -q

# Serving/backend tests
uv run pytest tests/serving/ -q

# MCP tests
uv run pytest tests/mcp/ -q

# Pipeline tests
uv run pytest tests/orchestration/ -q
```

**Expected Outcome:** All tests pass

### 4.3 Full Test Suite Smoke Test

Run the full test suite to ensure no regressions:

```bash
uv run pytest -q --ignore=tests/fixtures/
```

---

## Summary

| Phase | Item | Files | Est. Lines | Priority |
|-------|------|-------|------------|----------|
| 1.1 | Fix ide.py | 1 | ~15 | High |
| 1.2 | Add get_deps | 1 | ~12 | High |
| 1.3 | Fix subsystem.py | 1 | ~25 | High |
| 2.1 | Fix bootstrap.py type guard | 1 | ~8 | High |
| 2.2 | Fix backend.py variance | 1 | ~5 | High |
| 2.3 | Fix tool_builder.py | 1 | ~1 | Medium |
| 3.1 | Fix comprehensive tests | 1 | ~65 | High |
| 3.2 | Fix fixture imports | 2 | ~15 | Low |
| 4.x | Verification | - | - | Required |

**Total Estimated Changes:**
- **Source files:** 5
- **Test files:** 3
- **Total lines:** ~145

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing CLI behavior | Low | High | Existing tests + integration testing |
| Type system edge cases | Medium | Medium | Comprehensive pyright/pyrefly checks |
| Test fixture dependencies | Low | Low | TYPE_CHECKING guards |
| Regression in backend wiring | Low | High | Existing bootstrap tests |

---

## Dependencies

This plan has no external dependencies. All required infrastructure exists:
- `build_duckdb_backend` helper in `tests/_helpers/gateway.py`
- `build_backend_resource` in `codeintel.serving.bootstrap`
- `DuckDBBackendOptions` dataclass for test helper

---

## Constraints

1. **Zero suppressions** - No `# type: ignore` or `# noqa` comments
2. **Structural solutions** - Fix root causes, not symptoms
3. **Backward compatibility** - Maintain existing public APIs where possible
4. **Test coverage** - All changes must be covered by existing or new tests

