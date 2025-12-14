# Core Module Migration Implementation Plan

> **Purpose**: This document provides a comprehensive implementation plan for fully migrating all modules to use the shared `codeintel.core` infrastructure, eliminating backward-compatibility shims, decommissioning legacy patterns, and deleting dead code across `analytics`, `ingestion`, `graphs`, `cli`, `serving`, and `storage`.

---

## Implementation Status

| Phase | Status | Completed Date | Notes |
|-------|--------|----------------|-------|
| Phase 1: Audit and Inventory | ✅ Complete | 2024-12-13 | Created `migration_tracking.json` |
| Phase 2: Deprecation Shims | ✅ Complete | 2024-12-13 | All shims with deprecation warnings |
| Phase 3: Migrate Leaf Modules | 📋 Ready | - | Can begin immediately |
| Phase 4: Provider Pattern | 📋 Pending | - | Depends on Phase 3 |
| Phase 5: Cache Pattern | 📋 Pending | - | |
| Phase 6: Service Pattern | 📋 Pending | - | |
| Phase 7: Repository Pattern | 📋 Pending | - | |
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

## Migration Principles

1. **Inside-Out Migration**: Start with leaf modules (no dependents), work toward heavily-imported modules
2. **Test-First**: Update tests first to validate the new imports work correctly
3. **Deprecation Warnings**: Add deprecation warnings before removing backward-compat shims
4. **Feature Flags**: Use feature flags for risky migrations to allow rollback
5. **Incremental Commits**: One module migration per commit for easy bisection
6. **CI Green**: Each commit must pass CI before proceeding

---

## Phase 1: Audit and Inventory (1-2 days)

### 1.1 Create Migration Tracking Database

Create a tracking table to monitor migration progress:

```sql
CREATE TABLE migration.core_migration_tracking (
    id INTEGER PRIMARY KEY,
    source_module TEXT NOT NULL,
    source_file TEXT NOT NULL,
    target_core_module TEXT NOT NULL,
    pattern_type TEXT NOT NULL,  -- 'service', 'repository', 'cache', etc.
    migration_status TEXT DEFAULT 'pending',
    migrated_at TIMESTAMP,
    notes TEXT
);
```

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
| `graphs/engine/cache.py` | `GraphCache` | 📋 Pending |
| `core/data/snapshot.py` | Re-export shim | 📋 To Delete |
| `analytics/parsing/ast_cache.py` | AST caching | 📋 Pending |
| `storage/gateway_cache.py` | Gateway caching | 📋 Pending |
| Various `@lru_cache` uses | 69+ files | 📋 Audit |

#### Provider Pattern (15+ providers → `core/providers/`)

| Current Location | Class | Status |
|------------------|-------|--------|
| `analytics/resources/protocol.py` | Re-export shim | 📋 To Delete |
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
| `ingestion/infrastructure/workers.py` | `WorkerConfig`, `resolve_worker_count`, `create_executor` | 📋 Pending |
| Various ThreadPoolExecutor uses | Scattered | 📋 Audit |

#### Path Utilities (→ `core/paths/`)

| Current Location | Functions | Status |
|------------------|-----------|--------|
| `ingestion/infrastructure/paths.py` | `normalize_rel_path`, `relpath_to_module`, `safe_relpath` | 📋 Pending |
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

## Phase 2: Create Re-export Shims with Deprecation (2-3 days)

Before removing legacy imports, add deprecation warnings to all re-export shims.

### 2.1 Update Existing Re-export Shims

**File: `core/data/snapshot.py`**

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

**File: `analytics/resources/protocol.py`**

```python
"""Protocol for resource providers.

.. deprecated:: 1.0
    Import directly from ``codeintel.core.resources`` or
    ``codeintel.core.providers`` instead.
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
        "Import from codeintel.core.resources or codeintel.core.providers instead.",
        DeprecationWarning,
        stacklevel=2,
    )

__all__ = [
    "LazyResource",
    "ResourceError",
    "ResourceNotFoundError",
    "ResourceNotLoadedError",
    "ResourceProvider",
    "ResourceProviderBase",
    "ResourceRegistry",
]
```

### 2.2 Create New Re-export Shims for Legacy Modules

Create deprecation shims for all modules that will be replaced by core modules:

| Legacy Module | Core Replacement | Shim Action |
|---------------|------------------|-------------|
| `ingestion/infrastructure/workers.py` | `core/concurrency/` | Add deprecation warning |
| `ingestion/infrastructure/paths.py` | `core/paths/` | Add deprecation warning |
| `graphs/engine/cache.py` | `core/cache/` | Add deprecation warning |

---

## Phase 3: Migrate Leaf Modules (3-5 days)

Migrate modules with no dependents first.

### 3.1 Path Utilities Migration

**Target**: Replace `ingestion/infrastructure/paths.py` usage with `core/paths/`

**Files to Update**:
1. All files importing from `ingestion.infrastructure.paths`
2. Update to import from `codeintel.core.paths`

**Migration Script**:

```bash
# Find all imports of ingestion.infrastructure.paths
rg "from codeintel\.ingestion\.infrastructure\.paths import" src/

# Replace with core imports
# normalize_rel_path -> normalize_path
# relpath_to_module -> path_to_module
# safe_relpath -> safe_relpath (same name)
```

**Mapping**:
| Old Import | New Import |
|------------|------------|
| `normalize_rel_path` | `normalize_path` |
| `relpath_to_module` | `path_to_module` |
| `safe_relpath` | `safe_relpath` |
| `repo_relpath` | (use `normalize_path` + `Path.relative_to`) |
| `ensure_repo_root` | (use `Path.expanduser().resolve()`) |

### 3.2 Concurrency/Worker Migration

**Target**: Replace `ingestion/infrastructure/workers.py` with `core/concurrency/`

**Migration Mapping**:
| Old Import | New Import |
|------------|------------|
| `WorkerConfig` | `codeintel.core.concurrency.WorkerConfig` |
| `resolve_worker_count` | `codeintel.core.concurrency.resolve_worker_count` |
| `create_executor` | `codeintel.core.concurrency.create_executor` |
| `worker_pool` | (context manager, keep or inline) |
| `executor_factory` | (factory pattern, adapt to `core/factory/`) |

### 3.3 Hashing Migration

**Target**: Consolidate hashing patterns to `core/hashing/`

**Files to Update**:
- `build/hashing.py`
- `build/assets/fingerprinting.py`
- `ingestion/adapters/hash_change_detection.py`

**Migration Mapping**:
| Old Pattern | New Import |
|-------------|------------|
| Content hashing | `codeintel.core.hashing.content_hash` |
| File hashing | `codeintel.core.hashing.file_hash` |
| Fingerprinting | `codeintel.core.hashing.fingerprint` |
| Stable hash | `codeintel.core.hashing.stable_hash` |

---

## Phase 4: Migrate Provider Pattern (3-5 days)

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

## Phase 5: Migrate Cache Pattern (2-3 days)

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

## Phase 6: Migrate Service Pattern (5-7 days)

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

## Phase 7: Migrate Repository Pattern (3-5 days)

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

| File | Reason |
|------|--------|
| `core/data/snapshot.py` | Re-export shim → delete after consumers migrated |
| `analytics/resources/protocol.py` | Re-export shim → delete after consumers migrated |
| `ingestion/infrastructure/paths.py` | Replaced by `core/paths/` |
| `ingestion/infrastructure/workers.py` | Replaced by `core/concurrency/` |
| `graphs/engine/cache.py` | Integrated into `core/cache/` |

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

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Audit | 1-2 days | None |
| Phase 2: Deprecation Shims | 2-3 days | Phase 1 |
| Phase 3: Leaf Modules | 3-5 days | Phase 2 |
| Phase 4: Provider Pattern | 3-5 days | Phase 3 |
| Phase 5: Cache Pattern | 2-3 days | Phase 4 |
| Phase 6: Service Pattern | 5-7 days | Phase 5 |
| Phase 7: Repository Pattern | 3-5 days | Phase 6 |
| Phase 8: Adapter Pattern | 2-3 days | Phase 7 |
| Phase 9: Observability | 2-3 days | Phase 8 |
| Phase 10: Validation | 2-3 days | Phase 9 |
| Phase 11: Events/Hooks | 1-2 days | Phase 10 |
| Phase 12: Delete Dead Code | 2-3 days | Phase 11 |
| Phase 13: Update Tests | 3-5 days | Parallel with 4-11 |
| Phase 14: Documentation | 1-2 days | Phase 12 |

**Total Estimated Time**: 6-8 weeks

---

## Risk Mitigation

### High-Risk Changes

1. **Service Pattern Migration**: Core business logic, requires careful testing
   - Mitigation: Feature flag, shadow mode running both old/new

2. **Repository Pattern Migration**: Data access layer, could cause data issues
   - Mitigation: Read-only validation first, then write operations

3. **Cache Pattern Migration**: Performance-critical, could cause regressions
   - Mitigation: A/B testing, performance benchmarks before/after

### Rollback Procedures

1. Keep old modules as deprecated imports for 1 release cycle
2. Use feature flags to disable new implementations
3. Maintain database migration rollback scripts
4. Document manual rollback steps for each phase

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Import consolidation | 100% from `core/` | Static analysis |
| Dead code eliminated | 100% of identified | Vulture report |
| Re-export shims deleted | 100% | File count |
| Test coverage on core | 95%+ | Coverage report |
| Deprecation warnings | 0 remaining | Test run output |
| CI build time | No regression | CI metrics |
| Runtime performance | No regression | Benchmark suite |

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
| `from codeintel.graphs.engine.cache import GraphCache` | `from codeintel.core.cache import CacheProtocol` |

### Providers

| Old Import | New Import |
|------------|------------|
| `from codeintel.analytics.resources.protocol import ResourceProvider` | `from codeintel.core.providers import ProviderProtocol` |
| `from codeintel.core.resources import LazyResource` | `from codeintel.core.providers import BaseProvider` |

### Concurrency

| Old Import | New Import |
|------------|------------|
| `from codeintel.ingestion.infrastructure.workers import WorkerConfig` | `from codeintel.core.concurrency import WorkerConfig` |
| `from codeintel.ingestion.infrastructure.workers import create_executor` | `from codeintel.core.concurrency import create_executor` |

### Paths

| Old Import | New Import |
|------------|------------|
| `from codeintel.ingestion.infrastructure.paths import normalize_rel_path` | `from codeintel.core.paths import normalize_path` |
| `from codeintel.ingestion.infrastructure.paths import relpath_to_module` | `from codeintel.core.paths import path_to_module` |

### Hashing

| Old Import | New Import |
|------------|------------|
| (local hashing) | `from codeintel.core.hashing import content_hash, file_hash, fingerprint` |

### Observability

| Old Import | New Import |
|------------|------------|
| (local metrics) | `from codeintel.core.observability import InMemoryMetrics, timed_metric` |
| (local tracing) | `from codeintel.core.observability import InMemoryTracer, trace_operation` |

---

*Created: December 2024*
*Status: Ready for Implementation*
