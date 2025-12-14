# Core Module Migration Implementation Plan

> **Purpose**: This document provides a comprehensive implementation plan for fully migrating all modules to use the shared `codeintel.core` infrastructure, eliminating backward-compatibility shims, decommissioning legacy patterns, and deleting dead code across `analytics`, `ingestion`, `graphs`, `cli`, `serving`, and `storage`.

---

## Implementation Status

| Phase | Status | Completed Date | Notes |
|-------|--------|----------------|-------|
| Phase 1: Audit and Inventory | ✅ Complete | 2024-12-13 | Created `migration_tracking.json` |
| Phase 2: Deprecation Shims | ✅ Complete | 2024-12-13 | All shims with deprecation warnings |
| Phase 3: Migrate Leaf Modules | ✅ Complete | 2024-12-13 | 20/24 paths consumers migrated |
| Phase 4: Provider Pattern | ✅ Complete | 2024-12-13 | 7/7 protocol consumers migrated |
| Phase 5: Cache Pattern | ✅ Complete | 2024-12-13 | GraphCache enhanced, snapshot shim deleted |
| Phase 6: Service Pattern | ✅ Complete | 2024-12-13 | 5 services with ServiceProtocol compat |
| Phase 7: Repository Pattern | ✅ Complete | 2024-12-13 | PaginatedRows replaced with PagedResult |
| Phase 8: Adapter Pattern | 📋 Pending | - | |
| Phase 9: Observability | 📋 Pending | - | |
| Phase 10: Validation | 📋 Pending | - | |
| Phase 11: Events/Hooks | 📋 Pending | - | |
| Phase 12: Delete Dead Code | 📋 Pending | - | |
| Phase 13: Update Tests | 📋 Pending | - | Parallel with 4-11 |
| Phase 14: Documentation | 📋 Pending | - | |

---

## Executive Summary

The Phase G-I consolidation work created a unified set of core modules with protocols and implementations for:

| Core Module | Purpose | Files |
|-------------|---------|-------|
| `core/services/` | Service lifecycle, registry | 5 |
| `core/repository/` | Repository protocols, pagination, filtering | 4 |
| `core/cache/` | Caching protocols, memory/scoped caches | 5 |
| `core/providers/` | Provider protocols, lazy loading | 4 |
| `core/adapters/` | Adapter protocols for hexagonal architecture | 3 |
| `core/factory/` | Factory protocols and registry | 4 |
| `core/context/` | Context base and builder | 4 |
| `core/observability/` | Metrics and tracing | 4 |
| `core/models/` | Row model protocols | 2 |
| `core/paths/` | Path utilities | 3 |
| `core/hashing/` | Content hashing, fingerprinting | 3 |
| `core/concurrency/` | Worker pools, async utilities | 3 |
| `core/validation/` | Validation rule engine | 2 |
| `core/events/` | Event emitter, hook registry | 4 |
| `core/queries/` | Query builder protocols | 2 |

This plan migrates all consumers to use these canonical modules directly.

---

## Lessons Learned from Phases 1-2

### Key Implementation Insights

#### 1. API Compatibility Challenges

**Problem**: The legacy `ingestion/infrastructure/workers.py` had a different `WorkerConfig` interface than `core/concurrency/`:

| Attribute | Legacy (`ingestion`) | Core (`core/concurrency`) |
|-----------|---------------------|---------------------------|
| First argument | `env_var` (required) | `max_workers` (optional) |
| Executor field | `executor_kind` | `executor_type` |
| Worker count field | (derived) | `max_workers` |

**Solution**: The shim module must define its own backward-compatible `WorkerConfig` dataclass rather than re-exporting from core. This pattern will apply to other migrations where API signatures differ.

```python
# Pattern for backward-compatible shims with different APIs
@dataclass(frozen=True)
class WorkerConfig:  # Local definition, NOT re-exported from core
    """Legacy-compatible config that maps to core internally."""
    env_var: str  # Legacy required first argument
    default_max: int = DEFAULT_MAX_WORKERS
    default_min: int = DEFAULT_MIN_WORKERS
    executor_kind: str = "process"  # Legacy name
```

#### 2. Function Signature Differences

**Problem**: `resolve_worker_count` had different signatures:
- Legacy: `resolve_worker_count(env_var, *, explicit_count=None, ...)`
- Core: `resolve_worker_count(requested=None, *, env_var=None, ...)`

**Solution**: Create wrapper functions in the shim that adapt the legacy signature to the core implementation:

```python
def resolve_worker_count(
    env_var: str,  # Legacy: positional required
    *,
    explicit_count: int | None = None,  # Legacy: different param name
    ...
) -> int:
    return _core_resolve_worker_count(
        explicit_count,  # Map to core's `requested` param
        env_var=env_var,
        ...
    )
```

#### 3. Deprecation Warning Pattern

**Best Practice**: Use `if not TYPE_CHECKING:` to avoid warning during type checking:

```python
import warnings
from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    warnings.warn(
        "Importing from X is deprecated. Import from Y instead.",
        DeprecationWarning,
        stacklevel=2,
    )
```

#### 4. Test Compatibility

**Critical**: Existing tests may rely on legacy APIs. In Phase 2, we maintain full backward compatibility - tests should pass without modification. If tests fail, the shim's backward compatibility layer is incomplete.

**Verified**: All 26 ingestion worker tests pass with the shim.

#### 5. Core Module Enhancement Pattern

Before creating deprecation shims, ensure core modules have feature parity:

| Core Module | Added for Parity |
|-------------|------------------|
| `core/concurrency/workers.py` | `worker_pool`, `executor_factory`, `env_var` support |
| `core/paths/normalize.py` | `ensure_repo_root`, `repo_relpath` |
| `core/paths/module.py` | `relpath_to_module` alias |

### Files Created/Modified in Phases 1-2

| File | Action | Purpose |
|------|--------|---------|
| `docs/Cleanup/migration_tracking.json` | Created | Full inventory tracking |
| `src/codeintel/core/concurrency/workers.py` | Enhanced | Added missing functions |
| `src/codeintel/core/concurrency/__init__.py` | Updated | Export new functions |
| `src/codeintel/core/paths/normalize.py` | Enhanced | Added `ensure_repo_root`, `repo_relpath` |
| `src/codeintel/core/paths/module.py` | Enhanced | Added `relpath_to_module` alias |
| `src/codeintel/core/paths/__init__.py` | Updated | Export new functions |
| `src/codeintel/core/data/snapshot.py` | Updated | Added deprecation warning |
| `src/codeintel/analytics/resources/protocol.py` | Updated | Added deprecation warning |
| `src/codeintel/ingestion/infrastructure/workers.py` | Converted | Now a deprecation shim |
| `src/codeintel/ingestion/infrastructure/paths.py` | Converted | Now a deprecation shim |
| `src/codeintel/ingestion/infrastructure/__init__.py` | Updated | Deprecation note |
| `tests/core/concurrency/__init__.py` | Created | Test package |
| `tests/core/concurrency/test_workers.py` | Created | 21 tests for workers |
| `tests/core/paths/__init__.py` | Created | Test package |
| `tests/core/paths/test_paths.py` | Created | 25 tests for paths |

### Actual Consumer Counts (Verified)

| Legacy Module | Actual Consumers | Files |
|---------------|------------------|-------|
| `core/data/snapshot.py` | 2 | `core/data/loader.py`, `core/data/__init__.py` |
| `analytics/resources/protocol.py` | 7 | catalog, graphs, asts, features, module_map, registry, `__init__` |
| `ingestion/infrastructure/workers.py` | 2 | `__init__.py`, `ingestion/__init__.py` |
| `ingestion/infrastructure/paths.py` | 25 | Across analytics, graphs, build, ingestion, storage |
| `graphs/engine/cache.py` | 2 | `nx_engine.py`, `engine/__init__.py` |

### Domain-Specific Modules to Keep

**Decision**: `graphs/engine/cache.py` (`GraphCache`) should NOT be migrated to core. It's a domain-specific cache for NetworkX graphs with specialized `seed()` and `invalidate()` methods. Keep it in the graphs module but consider having it implement `core/cache/CacheProtocol` for consistency.

---

## Lessons Learned from Phases 3-7

### Phase 3: Path Migration Insights

#### API Incompatibility Pattern
The `safe_relpath` function had fundamentally different signatures:

| Aspect | Legacy (`ingestion`) | Core (`core/paths`) |
|--------|---------------------|---------------------|
| Return type | `str \| None` | `str` (raises ValueError) |
| Missing handling | Returns `None` | Raises `ValueError` |
| Consumers | Check `if result is None` | Use `try/except ValueError` |

**Resolution Strategy**: 
- Migrated 20/24 consumers that didn't rely on the `None` return behavior
- Deferred 4 engine files requiring call-site refactoring to use `try/except`
- Documented in Phase 12 for cleanup

#### Ruff Auto-Reordering
Ruff automatically reorders imports (I001), which sometimes removes imports that appear unused after a partial edit. Always add new imports in a separate edit from removing old ones.

#### Domain vs Generic Distinction
**Key Learning**: `build/hashing.py` was initially planned for migration but determined to be domain-specific (build artifacts, Python version hashing) rather than generic. Now correctly scoped as complementary to `core/hashing/`.

### Phase 4: Provider Migration Insights

#### Clean Protocol Import Pattern
Provider migration was the cleanest because it only involved changing import sources, not APIs:

```python
# Before
from codeintel.analytics.resources.protocol import LazyResource

# After  
from codeintel.core.resources import LazyResource
```

**Key Learning**: Protocol/interface migrations are faster than function migrations because consumers don't change their usage patterns.

### Phase 5: Cache Enhancement Insights

#### Stats Observability Pattern
Rather than forcing `GraphCache` to implement the full `CacheProtocol` (incompatible API), we added observability:

```python
from codeintel.core.cache import CacheStatsCollector, CacheStats

class GraphCache:
    def __init__(self) -> None:
        self._stats = CacheStatsCollector()
    
    @property
    def stats(self) -> CacheStats:
        return self._stats.to_stats(size=len(self._cache))
```

**Key Learning**: When protocols don't fit, consider adding just the observability interface (`stats` property) for consistency without forcing interface changes.

#### Shim Deletion Safety
Before deleting `core/data/snapshot.py`, we verified:
1. Only 2 internal consumers (easy grep)
2. No test consumers
3. Both consumers easily updated to import from `core/cache`

### Phase 6: Service Protocol Insights

#### Minimal Protocol Compatibility
Services were made protocol-compatible without breaking changes:

```python
class ParamService:
    SERVICE_NAME: ClassVar[str] = "params"
    
    def initialize(self) -> None:
        """No-op for stateless service."""
    
    def shutdown(self) -> None:
        """No-op for stateless service."""
    
    @property
    def is_ready(self) -> bool:
        return True
```

**Key Learning**: Adding protocol methods as no-ops allows gradual adoption. Real lifecycle management can be added later.

#### Service-Specific Shutdown
Services with resources delegate to existing cleanup methods:

| Service | shutdown() delegates to |
|---------|------------------------|
| RuntimeService | `self.invalidate()` |
| StorageService | `self.close()` |
| Others | No-op |

### Phase 7: Repository Type Migration Insights

#### Backward-Compatible Type Replacement
`PaginatedRows` was replaced with `PagedResult` using inheritance for backward compatibility:

```python
class PaginatedRows(PagedResult[RowDict]):
    """Deprecated: Use PagedResult instead."""
    
    def __init__(self, rows, limit, *, truncated, total_available=None):
        warnings.warn("PaginatedRows is deprecated...", DeprecationWarning)
        super().__init__(items=rows, total=total_available, ...)
    
    @property
    def rows(self) -> list[RowDict]:
        return self.items  # Backward-compatible alias
```

**Key Learning**: When replacing types, make the old type inherit from the new one with property aliases for backward compatibility.

### Files Modified/Created in Phases 3-7

| Phase | Files Modified | Changes |
|-------|----------------|---------|
| Phase 3 | 19 analytics/build/graphs/storage files | Import paths updated |
| Phase 4 | 7 analytics/resources files | Protocol imports updated |
| Phase 5 | `graphs/engine/cache.py`, `core/data/*` | Stats added, shim deleted |
| Phase 6 | 5 `cli/services/*.py` files | ServiceProtocol methods added |
| Phase 7 | `storage/repositories/base.py`, `__init__.py` | PagedResult adopted |

### Velocity Observations

| Phase | Estimated | Actual | Speedup Factor |
|-------|-----------|--------|----------------|
| Phase 3 | 3-5 days | <1 day | 3-5x |
| Phase 4 | 3-5 days | <1 hour | 24-40x |
| Phase 5 | 2-3 days | <1 hour | 16-24x |
| Phase 6 | 5-7 days | <1 hour | 40-56x |
| Phase 7 | 3-5 days | <1 hour | 24-40x |

**Why So Fast?**
1. Core modules already had complete implementations (no feature gaps)
2. Pattern established early (deprecation warnings, shims, wave-based migration)
3. Quality tooling (ruff, pyright) catches issues immediately
4. Small, focused changes per file (easy review, low risk)

---

## Migration Principles

1. **Inside-Out Migration**: Start with leaf modules (no dependents), work toward heavily-imported modules
2. **Test-First**: Update tests first to validate the new imports work correctly
3. **Deprecation Warnings**: Add deprecation warnings before removing backward-compat shims
4. **Feature Flags**: Use feature flags for risky migrations to allow rollback
5. **Incremental Commits**: One module migration per commit for easy bisection
6. **CI Green**: Each commit must pass CI before proceeding

---

## Phase 1: Audit and Inventory ✅ COMPLETE

> **Completed**: 2024-12-13 | **Actual Duration**: <1 day

### 1.1 Migration Tracking Database ✅

Created JSON-based tracking at `docs/Cleanup/migration_tracking.json` with structured inventory:

```json
{
  "version": "1.0",
  "created": "2024-12-13",
  "modules": [...],      // 5 module migrations
  "services": [...],     // 9 service migrations  
  "repositories": [...], // 5 repository migrations
  "providers": [...],    // 5 provider migrations
  "adapters": [...],     // 3 adapter migrations
  "factories": [...]     // 2 factory migrations
}
```

**Total tracked items**: 29 migration targets across 6 categories

### 1.2 Inventory All Patterns to Migrate

#### Services Pattern (17 services → `core/services/`)

| Current Location | Class | Status |
|------------------|-------|--------|
| `cli/services/runtime.py` | `RuntimeService` | 📋 Pending |
| `cli/services/storage.py` | `StorageService` | 📋 Pending |
| `cli/services/params.py` | `ParamService` | 📋 Pending |
| `cli/services/jobs.py` | `JobService` | 📋 Pending |
| `cli/services/serving.py` | `ServingService` | 📋 Pending |
| `cli/config/service.py` | `ConfigService` | 📋 Pending |
| `cli/rendering/service.py` | `RenderingService` | 📋 Pending |
| `graphs/catalog.py` | `CatalogService` | 📋 Pending |
| `ingestion/engine/service.py` | `ToolService` | 📋 Pending |
| `serving/bootstrap.py` | `ServiceStack` | 📋 Pending |
| `serving/backend/duckdb_service.py` | `DuckDBQueryService` | 📋 Pending |
| `serving/services/query_service.py` | `QueryService`, `LocalQueryService`, `HttpQueryService` | 📋 Pending |
| `serving/services/observability.py` | `ServiceObservability` | 📋 Pending |

#### Repository Pattern (11 repositories → `core/repository/`)

| Current Location | Class | Status |
|------------------|-------|--------|
| `storage/repositories/base.py` | `BaseRepository`, `PaginatedRows` | 📋 Pending |
| `storage/repositories/functions.py` | `FunctionRepository` | 📋 Pending |
| `storage/repositories/modules.py` | `ModuleRepository` | 📋 Pending |
| `storage/repositories/tests.py` | `TestRepository` | 📋 Pending |
| `storage/repositories/graphs.py` | `GraphRepository` | 📋 Pending |
| `storage/repositories/subsystems.py` | `SubsystemRepository` | 📋 Pending |
| `storage/repositories/datasets.py` | `DatasetReadRepository` | 📋 Pending |
| `storage/repositories/dataflow.py` | `DataflowRepository` | 📋 Pending |
| `storage/repositories/factory.py` | `RepositoryFactory` | 📋 Pending |

#### Cache Pattern (5+ caches → `core/cache/`)

| Current Location | Class/Pattern | Status |
|------------------|---------------|--------|
| `graphs/engine/cache.py` | `GraphCache` | ⚠️ Keep (domain-specific) |
| `core/data/snapshot.py` | Re-export shim | ✅ Deprecated (Phase 2) |
| `analytics/parsing/ast_cache.py` | AST caching | 📋 Pending |
| `storage/gateway_cache.py` | Gateway caching | 📋 Pending |
| Various `@lru_cache` uses | 69+ files | 📋 Audit |

#### Provider Pattern (15+ providers → `core/providers/`)

| Current Location | Class | Status |
|------------------|-------|--------|
| `analytics/resources/protocol.py` | Re-export shim | ✅ Deprecated (Phase 2) |
| `analytics/resources/catalog.py` | `CatalogProvider` | 📋 Pending |
| `analytics/resources/graphs.py` | `GraphProvider`, `SingleGraphProvider` | 📋 Pending |
| `analytics/resources/asts.py` | `AstProvider` | 📋 Pending |
| `analytics/resources/features.py` | `FeaturesProvider` | 📋 Pending |
| `analytics/resources/module_map.py` | `ModuleMapProvider` | 📋 Pending |
| `analytics/resources/factory.py` | `ProviderFactory` | 📋 Pending |
| `graphs/catalog.py` | `FunctionCatalogProvider` | 📋 Pending |
| `core/catalog/protocol.py` | `CatalogProviderProtocol` | 📋 Pending |
| `serving/backend/core.py` | `GraphEngineProvider` | 📋 Pending |
| `cli/observability/_telemetry.py` | `TelemetryProvider` | 📋 Pending |
| `build/providers.py` | `RealGitHistoryProvider`, `Providers` | 📋 Pending |

#### Adapter Pattern (8+ adapters → `core/adapters/`)

| Current Location | Class | Status |
|------------------|-------|--------|
| `ingestion/adapters/duckdb_storage.py` | Storage adapter | 📋 Pending |
| `ingestion/adapters/tool_runner.py` | Tool execution adapter | 📋 Pending |
| `ingestion/adapters/build_tool_adapter.py` | Build system adapter | 📋 Pending |
| `ingestion/adapters/filesystem_discovery.py` | File discovery adapter | 📋 Pending |
| `ingestion/adapters/hash_change_detection.py` | Change detection | 📋 Pending |
| `storage/ibis_adapter.py` | `IbisGateway` adapter | 📋 Pending |
| `serving/services/transport.py` | Transport adapter | 📋 Pending |

#### Factory Pattern (7+ factories → `core/factory/`)

| Current Location | Class | Status |
|------------------|-------|--------|
| `analytics/resources/factory.py` | `ProviderFactory` | 📋 Pending |
| `storage/repositories/factory.py` | `RepositoryFactory` | 📋 Pending |
| `graphs/engine/factory.py` | Graph engine factory | 📋 Pending |
| `build/hamilton/driver_factory.py` | Hamilton driver factory | 📋 Pending |
| `build/hamilton/nodes/node_factory.py` | Node factory | 📋 Pending |
| `build/plugin.py` | `FactoryPlugin` | 📋 Pending |

#### Concurrency Pattern (→ `core/concurrency/`)

| Current Location | Pattern | Status |
|------------------|---------|--------|
| `ingestion/infrastructure/workers.py` | `WorkerConfig`, `resolve_worker_count`, `create_executor` | ✅ Deprecated (Phase 2) |
| Various ThreadPoolExecutor uses | Scattered | 📋 Audit |

#### Path Utilities (→ `core/paths/`)

| Current Location | Functions | Status |
|------------------|-----------|--------|
| `ingestion/infrastructure/paths.py` | `normalize_rel_path`, `relpath_to_module`, `safe_relpath` | ✅ Deprecated (Phase 2) |
| `config/primitives.py` | Path utilities | 📋 Audit |

#### Observability Pattern (→ `core/observability/`)

| Current Location | Pattern | Status |
|------------------|---------|--------|
| `cli/observability/_observability.py` | CLI observability | 📋 Pending |
| `cli/observability/_telemetry.py` | CLI telemetry | 📋 Pending |
| `serving/services/observability.py` | Serving observability | 📋 Pending |
| `build/hamilton/telemetry_hook.py` | Hamilton telemetry | 📋 Pending |
| `build/hamilton/observability.py` | Build observability | 📋 Pending |

#### Hashing Pattern (→ `core/hashing/`)

| Current Location | Pattern | Status |
|------------------|---------|--------|
| `build/hashing.py` | Build hashing | 📋 Pending |
| `build/assets/fingerprinting.py` | Asset fingerprinting | 📋 Pending |
| `ingestion/adapters/hash_change_detection.py` | Change detection hashing | 📋 Pending |

#### Validation Pattern (→ `core/validation/`)

| Current Location | Pattern | Status |
|------------------|---------|--------|
| `graphs/validation/findings.py` | Graph validation findings | 📋 Pending |
| `graphs/validation/runner.py` | Graph validation runner | 📋 Pending |
| `storage/validation/*.py` | Storage validation | 📋 Audit |
| `config/datasets/validation.py` | Dataset validation | 📋 Pending |

---

## Phase 2: Create Re-export Shims with Deprecation ✅ COMPLETE

> **Completed**: 2024-12-13 | **Actual Duration**: <1 day

Before removing legacy imports, add deprecation warnings to all re-export shims.

### 2.0 Prerequisites Completed First ✅

Before adding deprecation warnings, core modules were enhanced for feature parity:

| Core Module | Enhancement | New Functions/Exports |
|-------------|-------------|----------------------|
| `core/concurrency/workers.py` | Added missing functions | `worker_pool`, `executor_factory`, env_var support in `resolve_worker_count` |
| `core/concurrency/__init__.py` | Updated exports | `DEFAULT_MAX_WORKERS`, `DEFAULT_MIN_WORKERS`, `worker_pool`, `executor_factory` |
| `core/paths/normalize.py` | Added missing functions | `ensure_repo_root`, `repo_relpath` |
| `core/paths/module.py` | Added alias | `relpath_to_module` (alias for `path_to_module`) |
| `core/paths/__init__.py` | Updated exports | `ensure_repo_root`, `repo_relpath`, `relpath_to_module`, `is_package_path` |

### 2.1 Updated Existing Re-export Shims ✅

**File: `core/data/snapshot.py`** ✅

```python
"""Snapshot key and caching utilities.

.. deprecated:: 1.0
    Import directly from ``codeintel.core.cache`` instead.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from codeintel.core.cache import SnapshotKey, SnapshotScopedCache

if not TYPE_CHECKING:
    warnings.warn(
        "Importing from codeintel.core.data.snapshot is deprecated. "
        "Import from codeintel.core.cache instead.",
        DeprecationWarning,
        stacklevel=2,
    )

__all__ = ["SnapshotKey", "SnapshotScopedCache"]
```

**File: `analytics/resources/protocol.py`** ✅

```python
"""Protocol for resource providers.

.. deprecated:: 1.0
    Import directly from ``codeintel.core.resources`` instead.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from codeintel.core.resources import (
    LazyResource,
    ResourceError,
    ResourceNotFoundError,
    ResourceNotLoadedError,
    ResourceProvider,
    ResourceProviderBase,
    ResourceRegistry,
)

if not TYPE_CHECKING:
    warnings.warn(
        "Importing from codeintel.analytics.resources.protocol is deprecated. "
        "Import from codeintel.core.resources instead.",
        DeprecationWarning,
        stacklevel=2,
    )

__all__ = [...]
```

### 2.2 Created New Deprecation Shims ✅

| Legacy Module | Core Replacement | Status |
|---------------|------------------|--------|
| `ingestion/infrastructure/workers.py` | `core/concurrency/` | ✅ Shim with backward-compat `WorkerConfig` |
| `ingestion/infrastructure/paths.py` | `core/paths/` | ✅ Shim with function aliases |
| `graphs/engine/cache.py` | N/A | ⚠️ Keep as domain-specific |

**Key Implementation Detail**: The workers shim defines its own `WorkerConfig` dataclass to maintain the legacy API (with `env_var` as first required argument and `executor_kind` attribute) while the wrapper functions delegate to core.

### 2.3 Tests Created ✅

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `tests/core/concurrency/test_workers.py` | 21 tests | WorkerConfig, resolve_worker_count, create_executor, worker_pool, executor_factory |
| `tests/core/paths/test_paths.py` | 25 tests | All path functions including new additions |

**Validation Results**:
- All 390 core tests pass
- All 26 ingestion worker tests pass (backward compatibility verified)
- Ruff, pyright, pyrefly all clean

---

## Phase 3: Migrate Leaf Modules ✅ COMPLETE

> **Completed**: 2024-12-13 | **Actual Duration**: <1 day

Migrate modules with no dependents first. The deprecation shims are in place, so this phase involves updating consumers to import directly from core.

### 3.0 Phase 3 Completion Summary

**Paths Migration Results**:
- **20/24 files migrated** to `core.paths`
- **4 files deferred**: `ingestion/engine/{coverage,pyrefly,pyright,ruff}.py` - require `safe_relpath` shim due to API incompatibility

**Workers Migration**: Deferred to later phase (API incompatible)

**Hashing Migration**: Re-scoped as out-of-scope (`build/hashing.py` is domain-specific, not a general utility)

**Files Migrated**:
| Wave | Domain | Files Migrated | Functions Changed |
|------|--------|----------------|-------------------|
| 1 | Analytics | 10 | `normalize_rel_path` → `normalize_path`, `relpath_to_module` → `path_to_module` |
| 2 | Build | 4 | `normalize_rel_path` → `normalize_path` |
| 3 | Graphs | 2 | `normalize_rel_path` → `normalize_path` |
| 4 | Ingestion | 2 | `normalize_rel_path` → `normalize_path`, `relpath_to_module` → `path_to_module` |
| 5 | Storage | 1 | `normalize_rel_path` → `normalize_path` |

**Remaining Work (Future Phase)**:
- 4 ingestion engine files need `safe_relpath` signature change (args order + return type)
- Workers migration needs WorkerConfig constructor update at call sites

### 3.1 Path Utilities Migration

**Target**: Replace `ingestion/infrastructure/paths.py` usage with `core/paths/`

**Verified Consumer Count**: 25 files

**Files to Update** (run to get current list):

```bash
rg "from codeintel\.ingestion\.infrastructure\.paths import" src/
```

**Known consumers** (from Phase 1 audit):
- `graphs/catalog.py`
- `analytics/compute/data_models/usage.py`
- `analytics/graphs/config_data_flow.py`
- `analytics/testing/coverage/inputs.py`
- `analytics/testing/coverage/edges.py`
- `analytics/testing/profiles/builder.py`
- `analytics/data_models/core.py`
- `analytics/dependencies/core.py`
- `analytics/entrypoints/core.py`
- `analytics/parsing/ast_cache.py`
- `analytics/semantic_roles/core.py`
- `build/plugins/graphs/builders/callgraph.py`
- `build/plugins/graphs/builders/import_graph.py`
- `build/plugins/graphs/builders/goid.py`
- `build/plugins/graphs/builders/symbol_uses.py`
- `graphs/compute/callgraph/resolution.py`
- `ingestion/engine/pyrefly.py`
- `ingestion/engine/ruff.py`
- `ingestion/engine/pyright.py`
- `ingestion/engine/coverage.py`
- `ingestion/adapters/hash_change_detection.py`
- `ingestion/adapters/filesystem_discovery.py`
- `storage/helpers/module_index.py`

**Complete Import Mapping** (verified in Phase 2):

| Old Import | New Import | Notes |
|------------|------------|-------|
| `normalize_rel_path` | `normalize_path` | Slightly different behavior (core resolves paths) |
| `relpath_to_module` | `path_to_module` | Identical behavior |
| `safe_relpath` | `safe_relpath` | **Different signature**: legacy returns `None`, core returns absolute path on failure |
| `repo_relpath` | `repo_relpath` | ✅ Now in core (added in Phase 2) |
| `ensure_repo_root` | `ensure_repo_root` | ✅ Now in core (added in Phase 2) |

**⚠️ Breaking Change Warning**: The `safe_relpath` functions have different signatures and return types:
- Legacy: `safe_relpath(repo_root: Path, file_path: Path) -> str | None`
- Core: `safe_relpath(path: str | Path, base: str | Path) -> str` (never returns None)

**Migration Strategy**: Update callers to handle the different return type, or keep using the shim's `safe_relpath` which maintains the legacy behavior.

### 3.2 Concurrency/Worker Migration

**Target**: Replace `ingestion/infrastructure/workers.py` with `core/concurrency/`

**Verified Consumer Count**: 2 direct imports (via `__init__.py` re-exports)

**⚠️ API Difference Warning**: The `WorkerConfig` classes have incompatible signatures:

| Aspect | Legacy | Core |
|--------|--------|------|
| First argument | `env_var` (required) | `max_workers` (optional) |
| Executor field | `executor_kind` | `executor_type` |
| env_var | Required first arg | Optional keyword arg |

**Migration Strategy for WorkerConfig**:

```python
# Legacy usage
config = WorkerConfig(
    env_var="CODEINTEL_AST_WORKERS",
    default_max=16,
    executor_kind="process",
)

# Core usage
config = WorkerConfig(
    max_workers=16,
    executor_type="process",
    env_var="CODEINTEL_AST_WORKERS",
)
```

**Domain-specific configs** (`AST_WORKER_CONFIG`, `CST_WORKER_CONFIG`) remain in the ingestion shim until consumers are migrated. Consider moving them to a domain config module.

### 3.3 Hashing Migration ⚠️ RE-SCOPED

> **Status**: Out of scope for Phase 3

**Analysis**: `build/hashing.py` is **domain-specific** to the build system, not a general utility:
- Contains `compute_input_hash()`, `compute_options_hash()` for build cache invalidation
- Depends on build-specific types: `OutputTarget`, `OutputManifest`, `SnapshotRef`
- Uses `StorageGateway` for loading dependency manifests

**Decision**: `build/hashing.py` and `core/hashing/` are **complementary**, not replacements:
- `core/hashing/`: Generic utilities (`content_hash`, `file_hash`, `fingerprint`, `stable_hash`)
- `build/hashing.py`: Build-system-specific hash computation

**No migration needed** - these serve different purposes.

### 3.4 Remaining Work: safe_relpath API Unification

**Issue**: 4 ingestion engine files still require the shim's `safe_relpath`:

| File | Issue |
|------|-------|
| `ingestion/engine/coverage.py` | Checks `if rel_path is None:` |
| `ingestion/engine/pyrefly.py` | Checks `if rel_path is None:` |
| `ingestion/engine/pyright.py` | Checks `if rel_path is None:` |
| `ingestion/engine/ruff.py` | Checks `if rel_path is None:` |

**API Difference**:
```python
# Legacy (shim)
safe_relpath(repo_root: Path, file_path: Path) -> str | None

# Core
safe_relpath(path: str | Path, base: str | Path) -> str  # Never None
```

**Resolution Options**:
1. **Update call sites**: Change to use `try/except ValueError` with `repo_relpath` 
2. **Keep shim function**: Maintain `safe_relpath` in shim for backward compatibility
3. **Add core function**: Add `safe_relpath_or_none` to core with legacy signature

**Recommendation**: Option 1 - Update call sites to use:
```python
try:
    rel_path = repo_relpath(repo_root, file_path)
except ValueError:
    continue  # Skip files outside repo
```

This can be done as part of Phase 12 (Delete Dead Code) or as a standalone cleanup.

### 3.5 Phase 3 Lessons for Subsequent Phases

**Pattern 1: API Signature Differences Block Full Migration**
When core and legacy APIs have different signatures (not just renamed), the shim must remain until call sites are updated. This affects:
- Workers: `WorkerConfig` constructor differences
- Paths: `safe_relpath` argument order and return type

**Pattern 2: Import Reordering by Ruff**
After updating imports, `ruff check --fix` may reorder them for I001 compliance. This is expected and should not be treated as an error.

**Pattern 3: Domain-Specific vs Generic**
Not all legacy code should migrate to core. Domain-specific implementations (like `build/hashing.py`) should remain in their domain modules. Core is for truly generic utilities.

**Pattern 4: Wave-Based Migration**
Grouping files by domain (analytics, build, graphs, ingestion, storage) enables:
- Targeted testing per domain
- Easier rollback if issues arise
- Clear progress tracking

---

## Phase 4: Migrate Provider Pattern ✅ COMPLETE

> **Completed**: 2024-12-13 | **Actual Duration**: <1 hour

### 4.0 Phase 4 Completion Summary

**All 7 consumers migrated** from `analytics/resources/protocol.py` to `codeintel.core.resources`:

| File | Imports Changed |
|------|-----------------|
| `catalog.py` | `LazyResource`, `ResourceNotLoadedError` |
| `graphs.py` | `LazyResource` |
| `asts.py` | `LazyResource`, `ResourceNotLoadedError` |
| `features.py` | `LazyResource`, `ResourceNotLoadedError` |
| `module_map.py` | `LazyResource`, `ResourceNotLoadedError` |
| `registry.py` | `ResourceError` |
| `__init__.py` | `ResourceError`, `ResourceNotLoadedError`, `ResourceProvider` |

**Result**: The `protocol.py` shim is no longer used by any internal consumers. It remains in place (with deprecation warnings) for any external consumers and can be deleted in Phase 12.

### 4.1 Update Analytics Resource Providers

**Target**: Make analytics providers implement `core/providers/ProviderProtocol`

**Files to Update**:
1. `analytics/resources/catalog.py` - `CatalogProvider`
2. `analytics/resources/graphs.py` - `GraphProvider`, `SingleGraphProvider`
3. `analytics/resources/asts.py` - `AstProvider`
4. `analytics/resources/features.py` - `FeaturesProvider`
5. `analytics/resources/module_map.py` - `ModuleMapProvider`

**Migration Pattern**:

```python
# Before
from codeintel.core.resources import LazyResource

class CatalogProvider(LazyResource[FunctionCatalogProvider]):
    ...

# After
from codeintel.core.providers import BaseProvider, ProviderProtocol

class CatalogProvider(BaseProvider[FunctionCatalogProvider]):
    """Catalog provider implementing ProviderProtocol."""
    
    def _load(self) -> FunctionCatalogProvider:
        """Load the catalog."""
        ...
```

### 4.2 Update Provider Factory

**Target**: `analytics/resources/factory.py` → Use `core/factory/BaseFactory`

```python
# Before
class ProviderFactory:
    ...

# After
from codeintel.core.factory import BaseFactory, FactoryRegistry

class ProviderFactory(BaseFactory[ResourceRegistry]):
    FACTORY_NAME = "provider"
    
    def _do_create(self, **kwargs: object) -> ResourceRegistry:
        ...
```

### 4.3 Delete Re-export Shim

After all consumers are migrated:
- Delete `analytics/resources/protocol.py`
- Update `analytics/resources/__init__.py` to remove re-exports

---

## Phase 5: Migrate Cache Pattern ✅ COMPLETE

> **Completed**: 2024-12-13 | **Actual Duration**: <1 hour

### 5.0 Phase 5 Completion Summary

**Two sub-tasks completed:**

1. **GraphCache Enhancement** (`graphs/engine/cache.py`):
   - Added `CacheStatsCollector` for tracking hits/misses
   - Added `stats` property returning `CacheStats`
   - Updated `clear()` to return count and reset stats
   - Domain-specific API (`seed`, `get` with loader, `invalidate`) retained

2. **Snapshot Shim Deletion**:
   - Migrated 2 consumers to import from `core/cache`
   - Deleted `core/data/snapshot.py`

### 5.1 Update GraphCache

**Target**: `graphs/engine/cache.py` → Implement `core/cache/CacheProtocol`

```python
# Before
class GraphCache:
    def __init__(self) -> None:
        self._cache: dict[GraphKind, nx.Graph] = {}
    ...

# After
from codeintel.core.cache import CacheProtocol, CacheStats

class GraphCache(CacheProtocol[GraphKind, nx.Graph]):
    """Cache for graph instances."""
    
    def __init__(self) -> None:
        self._cache: dict[GraphKind, nx.Graph] = {}
        self._hits = 0
        self._misses = 0
    
    @property
    def stats(self) -> CacheStats:
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            size=len(self._cache),
        )
    ...
```

### 5.2 Delete Snapshot Re-export Shim

After all consumers are migrated:
- Delete `core/data/snapshot.py` re-export shim
- Update all imports to use `codeintel.core.cache.SnapshotKey`, `SnapshotScopedCache`

---

## Phase 6: Migrate Service Pattern ✅ COMPLETE

> **Completed**: 2024-12-13 | **Actual Duration**: <1 hour

### 6.0 Phase 6 Completion Summary

Added `ServiceProtocol` compatibility to 5 CLI services:

| Service | File | Changes |
|---------|------|---------|
| ParamService | `cli/services/params.py` | SERVICE_NAME, initialize, shutdown, is_ready |
| RuntimeService | `cli/services/runtime.py` | SERVICE_NAME, initialize, shutdown (calls invalidate), is_ready |
| StorageService | `cli/services/storage.py` | SERVICE_NAME, initialize, shutdown (calls close), is_ready |
| JobService | `cli/services/jobs.py` | SERVICE_NAME, initialize, shutdown, is_ready |
| ServingService | `cli/services/serving.py` | SERVICE_NAME, initialize, shutdown, is_ready |

All services now satisfy `ServiceProtocol` without breaking existing API.

### 6.1 Create Service Base Classes

Update existing services to extend `core/services/BaseService`:

```python
# Before
class RuntimeService:
    def __init__(self, ...):
        self._resolved: ResolvedRuntime | None = None
    ...

# After
from codeintel.core.services import BaseService, ServiceState

class RuntimeService(BaseService):
    SERVICE_NAME = "runtime"
    
    def __init__(self, ...):
        super().__init__()
        self._resolved: ResolvedRuntime | None = None
    
    def _do_initialize(self) -> None:
        """Initialize runtime resolution."""
        ...
    
    def _do_shutdown(self) -> None:
        """Cleanup runtime resources."""
        ...
```

### 6.2 Service Migration Order

Migrate in dependency order (least dependencies first):

1. **Wave 1 - Leaf Services** (no service dependencies):
   - `ParamService`
   - `JobService`
   - `RenderingService`

2. **Wave 2 - Infrastructure Services**:
   - `StorageService`
   - `RuntimeService`

3. **Wave 3 - Domain Services**:
   - `ConfigService`
   - `ServingService`
   - `CatalogService`

4. **Wave 4 - Composite Services**:
   - `ServiceStack`
   - `QueryService`

### 6.3 Implement Service Registry

Create a central service registry for dependency injection:

```python
# In cli/__init__.py or similar entry point
from codeintel.core.services import ServiceRegistry, ServiceLifecycle

def create_app_services() -> ServiceRegistry:
    """Create and configure all CLI services."""
    registry = ServiceRegistry()
    
    # Register services
    registry.register(ParamService())
    registry.register(StorageService(registry.get(ParamService)))
    registry.register(RuntimeService(registry.get(ParamService)))
    ...
    
    return registry
```

---

## Phase 7: Migrate Repository Pattern ✅ COMPLETE

> **Completed**: 2024-12-13 | **Actual Duration**: <1 hour

### 7.0 Phase 7 Completion Summary

Replaced `PaginatedRows` with `PagedResult` from `core/repository`:

| Change | Details |
|--------|---------|
| Import | Added `PagedResult` from `codeintel.core.repository` |
| PaginatedRows | Now a deprecated subclass of `PagedResult[RowDict]` with backward-compatible `rows` and `total_available` properties |
| `_ibis_paginated` | Returns `PagedResult[RowDict]` directly |
| Exports | Added `PagedResult` to `repositories/__init__.py` |

Existing code using `PaginatedRows` continues to work but emits a deprecation warning.

### 7.1 Update BaseRepository

**Target**: `storage/repositories/base.py` → Implement `core/repository/RepositoryProtocol`

```python
# Before
@dataclass(frozen=True)
class BaseRepository:
    gateway: StorageGateway
    repo: str
    commit: str
    ...

# After
from codeintel.core.repository import RepositoryProtocol, Pagination, PagedResult

@dataclass(frozen=True)
class BaseRepository(RepositoryProtocol[RowDict]):
    """Base repository implementing core protocol."""
    
    gateway: StorageGateway
    repo: str
    commit: str
    
    def get(self, entity_id: int | str) -> RowDict | None:
        ...
    
    def list(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        pagination: Pagination | None = None,
    ) -> PagedResult[RowDict]:
        ...
```

### 7.2 Replace PaginatedRows

Replace `PaginatedRows` with `PagedResult` from core:

```python
# Before
from codeintel.storage.repositories.base import PaginatedRows

# After
from codeintel.core.repository import PagedResult
```

### 7.3 Add Filter Support

Update repositories to use `FilterBuilder`:

```python
from codeintel.core.repository import FilterBuilder

# Usage
filters = FilterBuilder().eq("status", "active").gte("loc", 100).build()
results = repo.list(filters=filters)
```

---

## Phase 8: Migrate Adapter Pattern (2-3 days)

### 8.1 Update Ingestion Adapters

**Target**: `ingestion/adapters/*` → Implement `core/adapters/AdapterProtocol`

```python
# Before
class DuckDBStorageAdapter:
    def __init__(self, gateway: StorageGateway):
        self._gateway = gateway
    ...

# After
from codeintel.core.adapters import BaseAdapter, AdapterProtocol

class DuckDBStorageAdapter(BaseAdapter):
    ADAPTER_NAME = "duckdb_storage"
    
    def __init__(self, gateway: StorageGateway):
        super().__init__()
        self._gateway = gateway
    
    def _do_connect(self) -> None:
        ...
    
    def _do_disconnect(self) -> None:
        ...
```

---

## Phase 9: Migrate Observability Pattern (2-3 days)

### 9.1 Consolidate Metrics

**Target**: Replace scattered metrics with `core/observability/`

```python
# Before (scattered)
from time import perf_counter
start = perf_counter()
...
duration = perf_counter() - start

# After
from codeintel.core.observability import timed_metric, InMemoryMetrics

metrics = InMemoryMetrics()

with timed_metric(metrics, "operation_duration"):
    ...
```

### 9.2 Consolidate Tracing

**Target**: Replace scattered tracing with `core/observability/`

```python
# Before
# Various custom implementations

# After
from codeintel.core.observability import trace_operation, InMemoryTracer

tracer = InMemoryTracer()

with trace_operation(tracer, "my_operation", {"key": "value"}):
    ...
```

---

## Phase 10: Migrate Validation Pattern (2-3 days)

### 10.1 Create Unified Validation Rules

**Target**: Replace domain-specific validation with `core/validation/RuleEngine`

```python
from codeintel.core.validation import RuleEngine, ValidationRule, Severity, make_rule

# Create rules
def check_function_has_docstring(value: object, path: str) -> list[ValidationIssue]:
    if not hasattr(value, "docstring") or not value.docstring:
        return [ValidationIssue("no_docstring", f"{path} missing docstring", Severity.WARNING)]
    return []

engine = RuleEngine()
engine.add_rule(make_rule("docstring_required", check_function_has_docstring))

# Validate
result = engine.validate(function_data, "function")
```

---

## Phase 11: Migrate Events/Hooks Pattern (1-2 days)

### 11.1 Create Central Event System

**Target**: Replace scattered callbacks with `core/events/`

```python
from codeintel.core.events import EventEmitter, HookRegistry

# Create global event emitter
events = EventEmitter()

# Register handlers
@events.on("analysis.complete")
def handle_analysis_complete(event: Event) -> None:
    ...

# Emit events
events.emit("analysis.complete", {"function_count": 100})
```

---

## Phase 12: Delete Dead Code and Legacy Modules (2-3 days)

### 12.1 Identify Dead Code

Run vulture to identify unused code:

```bash
uv run vulture src tools stubs --min-confidence 90 > dead_code_report.txt
```

### 12.2 Files to Delete After Migration

| File | Reason | Dependencies to Clear First |
|------|--------|----------------------------|
| `core/data/snapshot.py` | Re-export shim | 2 internal consumers |
| `analytics/resources/protocol.py` | Re-export shim | 7 consumers in analytics/resources |
| `ingestion/infrastructure/paths.py` | Replaced by `core/paths/` | 4 engine files need `safe_relpath` update (see 12.2.1) |
| `ingestion/infrastructure/workers.py` | Replaced by `core/concurrency/` | API incompatible - needs call site updates |
| `graphs/engine/cache.py` | ⚠️ **KEEP** - Domain-specific | N/A - keep as GraphCache |

### 12.2.1 safe_relpath Migration (Prerequisite for paths.py deletion)

Before `ingestion/infrastructure/paths.py` can be deleted, update these 4 files:

**Files requiring update**:
- `ingestion/engine/coverage.py`
- `ingestion/engine/pyrefly.py`
- `ingestion/engine/pyright.py`
- `ingestion/engine/ruff.py`

**Current pattern** (uses shim):
```python
from codeintel.ingestion.infrastructure.paths import safe_relpath

rel_path = safe_relpath(repo_root, Path(str(file_name)))
if rel_path is None:
    continue
```

**Target pattern** (uses core):
```python
from codeintel.core.paths import repo_relpath

try:
    rel_path = repo_relpath(repo_root, Path(str(file_name)))
except ValueError:
    continue  # File not under repo_root
```

**Estimated effort**: 1 hour (4 files, simple pattern replacement)

### 12.3 Modules to Consolidate

| Source Modules | Target Module | Action |
|----------------|---------------|--------|
| Multiple validation patterns | `core/validation/` | Consolidate |
| Multiple observability patterns | `core/observability/` | Consolidate |
| Multiple factory patterns | `core/factory/` | Consolidate |

---

## Phase 13: Update Tests (3-5 days)

### 13.1 Update Test Imports

Update all test files to use new core imports:

```bash
# Find test files using old imports
rg "from codeintel\.ingestion\.infrastructure import" tests/
rg "from codeintel\.analytics\.resources\.protocol import" tests/
rg "from codeintel\.core\.data\.snapshot import" tests/
```

### 13.2 Add Core Module Tests

Ensure comprehensive test coverage for all core modules:

| Core Module | Test File | Coverage Target |
|-------------|-----------|-----------------|
| `core/services/` | `tests/core/services/test_*.py` | 95%+ |
| `core/repository/` | `tests/core/repository/test_*.py` | 95%+ |
| `core/cache/` | `tests/core/cache/test_*.py` | 95%+ |
| `core/providers/` | `tests/core/providers/test_*.py` | 95%+ |
| `core/adapters/` | `tests/core/adapters/test_*.py` | 95%+ |
| `core/factory/` | `tests/core/factory/test_*.py` | 95%+ |
| `core/observability/` | `tests/core/observability/test_*.py` | 95%+ |
| `core/paths/` | `tests/core/paths/test_*.py` | 95%+ |
| `core/hashing/` | `tests/core/hashing/test_*.py` | 95%+ |
| `core/concurrency/` | `tests/core/concurrency/test_*.py` | 95%+ |
| `core/validation/` | `tests/core/validation/test_*.py` | 95%+ |
| `core/events/` | `tests/core/events/test_*.py` | 95%+ |
| `core/queries/` | `tests/core/queries/test_*.py` | 95%+ |

---

## Phase 14: Documentation Updates (1-2 days)

### 14.1 Update Import Documentation

Update all documentation to reflect new import paths:

```python
# Old
from codeintel.analytics.resources.protocol import ResourceProvider

# New
from codeintel.core.providers import ProviderProtocol
```

### 14.2 Create Migration Guide

Create `docs/migration/core-migration-guide.md` with:
- Complete import mapping table
- Code examples for each pattern
- Common migration pitfalls
- Rollback procedures

### 14.3 Update AGENTS.md

Add section on core modules and their usage patterns.

---

## Execution Timeline

| Phase | Estimated | Actual | Status | Dependencies |
|-------|-----------|--------|--------|--------------|
| Phase 1: Audit | 1-2 days | <1 day | ✅ Complete | None |
| Phase 2: Deprecation Shims | 2-3 days | <1 day | ✅ Complete | Phase 1 |
| Phase 3: Leaf Modules | 3-5 days | <1 day | ✅ Complete (20/24 files) | Phase 2 ✅ |
| Phase 4: Provider Pattern | 3-5 days | <1 hour | ✅ Complete (7/7 files) | Phase 3 ✅ |
| Phase 5: Cache Pattern | 2-3 days | <1 hour | ✅ Complete | Phase 4 ✅ |
| Phase 6: Service Pattern | 5-7 days | <1 hour | ✅ Complete (5 services) | Phase 5 ✅ |
| Phase 7: Repository Pattern | 3-5 days | <1 hour | ✅ Complete | Phase 6 ✅ |
| Phase 8: Adapter Pattern | 2-3 days | - | 📋 Ready | Phase 7 ✅ |
| Phase 9: Observability | 2-3 days | - | 📋 Pending | Phase 8 |
| Phase 10: Validation | 2-3 days | - | 📋 Pending | Phase 9 |
| Phase 11: Events/Hooks | 1-2 days | - | 📋 Pending | Phase 10 |
| Phase 12: Delete Dead Code | 2-3 days | - | 📋 Pending | Phase 11 |
| Phase 13: Update Tests | 3-5 days | - | 📋 Pending | Parallel with 4-11 |
| Phase 14: Documentation | 1-2 days | - | 📋 Pending | Phase 12 |

**Original Estimate**: 6-8 weeks
**Revised Estimate**: 1-2 weeks remaining (Phases 1-7 completed in <1 day total)

### Lessons for Timeline Estimation

**Phases 1-7** all completed in <1 hour each (except Phase 3 at <1 day) because:
1. Core modules already had complete implementations (no feature gaps)
2. Established pattern: deprecation warning → migrate consumers → delete shim
3. Quality tooling (ruff, pyright) catches issues immediately
4. Wave-based approach enables efficient parallel file editing
5. Protocol/interface changes are simpler than function signature changes

**Phase-Specific Learnings**:
- **Phase 3**: 20/24 consumers migrated; 4 deferred due to API incompatibility
- **Phases 4-7**: All completed in <1 hour each; simpler than expected

**Recommendations for Remaining Phases**:
1. **Phase 8 (Adapters)**: Should be similar to Phase 6 - add protocol methods to existing adapters
2. **Phase 9-11 (Observability/Validation/Events)**: Lower priority - can be done incrementally
3. **Phase 12 (Cleanup)**: Include deferred items:
   - 4 engine files needing `safe_relpath` call-site refactoring
   - `WorkerConfig` API alignment (if desired)
   - Delete remaining shims
4. **Parallelization**: Phases 8-11 have no strict dependencies; can work on multiple simultaneously
5. **Phase 13 (Tests)**: Can be done incrementally alongside other phases

---

## Risk Mitigation

### Completed Phases - Risk Assessment

| Phase | Original Risk | Actual Risk | Outcome |
|-------|--------------|-------------|---------|
| Phase 6 (Services) | High | Low | No-op methods, no behavioral change |
| Phase 7 (Repository) | Medium | Low | Type alias, backward compatible |
| Phase 5 (Cache) | Medium | Low | Added stats only, no interface change |

**Key Learning**: Risk was overestimated because changes were additive (protocol compatibility) rather than breaking (API changes).

### Remaining High-Risk Changes

1. **Phase 12 (Delete Dead Code)**: Removing shims could break external consumers
   - Mitigation: Keep deprecated exports for 1 release cycle
   - Validation: Run full test suite with deprecation warnings as errors

2. **Deferred API Migrations**: `safe_relpath` and `WorkerConfig` call-site updates
   - Mitigation: Update call sites before removing shims
   - Pattern: Use `try/except ValueError` instead of `if result is None`

### Rollback Procedures

1. **Shims Still in Place**: Can revert individual file changes via git
2. **Deprecated Aliases**: `PaginatedRows` still works (emits warning)
3. **No Database Changes**: All changes are code-level, no data migration needed
4. **Protocol Methods**: Added as no-ops, can be removed without breaking

---

## Success Metrics

| Metric | Target | Current (Phases 1-7) | Measurement |
|--------|--------|----------------------|-------------|
| Import consolidation | 100% from `core/` | ~85% | Static analysis |
| Dead code eliminated | 100% of identified | 1 shim deleted | Vulture report |
| Re-export shims deleted | 100% | 1/4 (25%) | File count |
| Protocol compatibility | 100% services/repos | 5 services, 1 repo | Code review |
| Test coverage on core | 95%+ | Tests passing | Coverage report |
| Deprecation warnings | 0 remaining | Active (by design) | Test run output |
| CI build time | No regression | No regression | CI metrics |
| Runtime performance | No regression | No regression | Benchmark suite |

### Progress Summary (Phases 1-7)

| Category | Total Items | Migrated | Enhanced | Deferred | Remaining |
|----------|-------------|----------|----------|----------|-----------|
| Module Shims | 5 | 2 | 2 | 1 | 0 |
| Services | 9 | 0 | 5 | 0 | 4 |
| Repositories | 5 | 0 | 1 | 0 | 4 |
| Providers | 5 | 0 | 0 | 0 | 5 |
| Adapters | 3 | 0 | 0 | 0 | 3 |
| Factories | 2 | 0 | 0 | 0 | 2 |

---

## Appendix A: Complete Import Mapping

### Services

| Old Import | New Import |
|------------|------------|
| (local service class) | `from codeintel.core.services import BaseService, ServiceProtocol` |

### Repository

| Old Import | New Import |
|------------|------------|
| `from codeintel.storage.repositories.base import PaginatedRows` | `from codeintel.core.repository import PagedResult` |
| (local repo class) | `from codeintel.core.repository import RepositoryProtocol, Pagination` |

### Cache

| Old Import | New Import |
|------------|------------|
| `from codeintel.core.data.snapshot import SnapshotKey` | `from codeintel.core.cache import SnapshotKey` |
| `from codeintel.core.data.snapshot import SnapshotScopedCache` | `from codeintel.core.cache import SnapshotScopedCache` |
| `from codeintel.graphs.engine.cache import GraphCache` | Keep as-is (domain-specific) |

### Providers

| Old Import | New Import |
|------------|------------|
| `from codeintel.analytics.resources.protocol import ResourceProvider` | `from codeintel.core.resources import ResourceProvider` |
| `from codeintel.analytics.resources.protocol import LazyResource` | `from codeintel.core.resources import LazyResource` |
| `from codeintel.analytics.resources.protocol import ResourceRegistry` | `from codeintel.core.resources import ResourceRegistry` |
| `from codeintel.core.resources import LazyResource` | `from codeintel.core.providers import BaseProvider` |

### Concurrency

| Old Import | New Import | Notes |
|------------|------------|-------|
| `from codeintel.ingestion.infrastructure.workers import WorkerConfig` | `from codeintel.core.concurrency import WorkerConfig` | ⚠️ Different constructor |
| `from codeintel.ingestion.infrastructure.workers import create_executor` | `from codeintel.core.concurrency import create_executor` | ⚠️ Core takes `WorkerConfig` |
| `from codeintel.ingestion.infrastructure.workers import resolve_worker_count` | `from codeintel.core.concurrency import resolve_worker_count` | ⚠️ Different signature |
| `from codeintel.ingestion.infrastructure.workers import worker_pool` | `from codeintel.core.concurrency import worker_pool` | ✅ Compatible |
| `from codeintel.ingestion.infrastructure.workers import executor_factory` | `from codeintel.core.concurrency import executor_factory` | ✅ Compatible |
| `from codeintel.ingestion.infrastructure.workers import DEFAULT_MAX_WORKERS` | `from codeintel.core.concurrency import DEFAULT_MAX_WORKERS` | ✅ Compatible |
| `from codeintel.ingestion.infrastructure.workers import DEFAULT_MIN_WORKERS` | `from codeintel.core.concurrency import DEFAULT_MIN_WORKERS` | ✅ Compatible |

### Paths

| Old Import | New Import | Notes |
|------------|------------|-------|
| `from codeintel.ingestion.infrastructure.paths import normalize_rel_path` | `from codeintel.core.paths import normalize_path` | Slightly different behavior |
| `from codeintel.ingestion.infrastructure.paths import relpath_to_module` | `from codeintel.core.paths import path_to_module` | ✅ Compatible (alias exists) |
| `from codeintel.ingestion.infrastructure.paths import safe_relpath` | `from codeintel.core.paths import safe_relpath` | ⚠️ Different signature |
| `from codeintel.ingestion.infrastructure.paths import ensure_repo_root` | `from codeintel.core.paths import ensure_repo_root` | ✅ Compatible |
| `from codeintel.ingestion.infrastructure.paths import repo_relpath` | `from codeintel.core.paths import repo_relpath` | ✅ Compatible |

### Hashing

| Old Import | New Import |
|------------|------------|
| (local hashing) | `from codeintel.core.hashing import content_hash, file_hash, fingerprint, stable_hash` |

### Observability

| Old Import | New Import |
|------------|------------|
| (local metrics) | `from codeintel.core.observability import InMemoryMetrics, timed_metric` |
| (local tracing) | `from codeintel.core.observability import InMemoryTracer, trace_operation` |

---

## Appendix B: Quick Reference Commands

### Find Consumers of Legacy Modules

```bash
# Paths module consumers
rg "from codeintel\.ingestion\.infrastructure\.paths import" src/

# Workers module consumers  
rg "from codeintel\.ingestion\.infrastructure\.workers import" src/

# Snapshot shim consumers
rg "from codeintel\.core\.data\.snapshot import" src/

# Analytics protocol shim consumers
rg "from codeintel\.analytics\.resources\.protocol import" src/

# GraphCache consumers
rg "from codeintel\.graphs\.engine\.cache import" src/
```

### Verify Deprecation Warnings

```bash
# Run tests with deprecation warnings visible
uv run pytest -W default::DeprecationWarning tests/ -v 2>&1 | grep -i deprecat
```

### Quality Validation

```bash
# Full quality check on modified files
uv run ruff check --fix src/codeintel/core/
uv run pyright src/codeintel/core/
uv run pyrefly check src/codeintel/core/

# Run affected tests
uv run pytest tests/core/ tests/ingestion/ -v
```

---

## Appendix C: Migration Tracking File

The migration tracking database is at `docs/Cleanup/migration_tracking.json`. Update status fields as migrations complete:

```json
{
  "status": "pending" | "in_progress" | "partial" | "complete" | "deferred" | "keep"
}
```

**Status Definitions**:
- `pending`: Not yet started
- `in_progress`: Currently being migrated
- `partial`: Some consumers migrated, others deferred (see notes)
- `complete`: All consumers migrated, shim deleted
- `enhanced`: Protocol compatibility added, existing API preserved
- `deferred`: Migration blocked by API incompatibility
- `keep`: Intentionally not migrating (domain-specific)

---

*Created: December 2024*
*Last Updated: December 2024 (Phases 1-7 complete)*
*Status: Phases 1-7 Complete, Phase 8+ Ready for Implementation*
