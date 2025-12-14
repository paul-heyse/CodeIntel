# Cross-Module Consolidation Opportunities

> **Purpose**: This document identifies additional opportunities to streamline shared functionality across `analytics`, `ingestion`, `graphs`, and `core` modules, building on the Phase 1-7 consolidation work already completed.

## Completed Consolidations (Reference)

### Phase 1-7 Consolidations (Original)

| Area | Core Module | Re-exports From |
|------|-------------|-----------------|
| Storage Ports | `core/ports/storage.py` | `graphs/ports/`, `ingestion/ports/` |
| Plugin Protocols | `core/plugins/types/async_protocol.py` | `ingestion/engine/plugins.py` |
| Graph Resources | `core/resources/graphs.py` | `analytics/resources/graphs.py` |
| Centrality Compute | `core/compute/centrality.py` | `graphs/compute/metrics/centrality.py` |
| Validation Runner | `core/validation/runner.py` | Used by `graphs/validation/` |
| Context Protocols | `core/context/protocol.py` | New unified protocols |
| Safe Queries | `storage/queries/safe.py` | `ingestion/infrastructure/db_queries.py` |

### Phase A-E Consolidations (December 2024) ✅

| Phase | Area | Core Module | Status |
|-------|------|-------------|--------|
| A | Error Taxonomy | `core/errors/` | ✅ Complete |
| A | Problem Details | `core/errors/problem_details.py` | ✅ Complete |
| A | Error Base Classes | `core/errors/base.py` | ✅ Complete |
| A | Execution Errors | `core/errors/execution.py` | ✅ Complete |
| A | Storage Errors | `core/errors/storage.py` | ✅ Complete |
| B | Options Protocol | `core/options/protocol.py` | ✅ Complete |
| B | Base Options | `core/options/base.py` | ✅ Complete |
| C | Function Span | `core/catalog/function_span.py` | ✅ Complete |
| C | Span Index | `core/catalog/span_index.py` | ✅ Complete |
| C | Catalog Protocol | `core/catalog/protocol.py` | ✅ Complete |
| D | Source Span | `core/parsing/source_span.py` | ✅ Complete |
| D | AST Index | `core/parsing/ast_index.py` | ✅ Complete |
| D | Parsed Models | `core/parsing/models.py` | ✅ Complete |
| E | Runtime Protocol | `core/runtime/protocol.py` | ✅ Complete |
| E | Execution Tracking | `core/runtime/tracking.py` | ✅ Complete |

### Phase F1-F3 Consolidations (December 2024) ✅

| Phase | Area | Core Module | Status |
|-------|------|-------------|--------|
| F1 | Result Protocol | `core/results/protocol.py` | ✅ Complete |
| F1 | Base Result | `core/results/base.py` | ✅ Complete |
| F1 | Execution Result | `core/results/execution.py` | ✅ Complete |
| F2 | Serialization Protocol | `core/serialization/protocol.py` | ✅ Complete |
| F2 | Serializable Base | `core/serialization/base.py` | ✅ Complete |
| F2 | Type Converters | `core/serialization/converters.py` | ✅ Complete |
| F3 | Data Loader Protocol | `core/data/protocol.py` | ✅ Complete |
| F3 | Base Data Loader | `core/data/loader.py` | ✅ Complete |
| F3 | Snapshot Utilities | `core/data/snapshot.py` | ✅ Complete |

### Phase G-I Consolidations (December 2024) ✅

| Phase | Area | Core Module | Status |
|-------|------|-------------|--------|
| G1 | Service Protocol | `core/services/protocol.py` | ✅ Complete |
| G1 | Base Service | `core/services/base.py` | ✅ Complete |
| G1 | Service Lifecycle | `core/services/lifecycle.py` | ✅ Complete |
| G1 | Service Registry | `core/services/registry.py` | ✅ Complete |
| G2 | Repository Protocol | `core/repository/protocol.py` | ✅ Complete |
| G2 | Pagination | `core/repository/pagination.py` | ✅ Complete |
| G2 | Filtering | `core/repository/filtering.py` | ✅ Complete |
| G3 | Cache Protocol | `core/cache/protocol.py` | ✅ Complete |
| G3 | Memory Cache | `core/cache/memory.py` | ✅ Complete |
| G3 | Scoped Cache | `core/cache/scoped.py` | ✅ Complete |
| G3 | Cache Keying | `core/cache/keying.py` | ✅ Complete |
| G4 | Provider Protocol | `core/providers/protocol.py` | ✅ Complete |
| G4 | Base Provider | `core/providers/base.py` | ✅ Complete |
| G4 | Lazy Provider | `core/providers/lazy.py` | ✅ Complete |
| H1 | Adapter Protocol | `core/adapters/protocol.py` | ✅ Complete |
| H1 | Base Adapter | `core/adapters/base.py` | ✅ Complete |
| H2 | Factory Protocol | `core/factory/protocol.py` | ✅ Complete |
| H2 | Base Factory | `core/factory/base.py` | ✅ Complete |
| H2 | Factory Registry | `core/factory/registry.py` | ✅ Complete |
| H3 | Base Context | `core/context/base.py` | ✅ Complete |
| H3 | Context Builder | `core/context/builder.py` | ✅ Complete |
| H4 | Observability Protocol | `core/observability/protocol.py` | ✅ Complete |
| H4 | Metrics | `core/observability/metrics.py` | ✅ Complete |
| H4 | Tracing | `core/observability/tracing.py` | ✅ Complete |
| H5 | Row Models | `core/models/rows.py` | ✅ Complete |
| I1 | Path Normalize | `core/paths/normalize.py` | ✅ Complete |
| I1 | Path Module | `core/paths/module.py` | ✅ Complete |
| I2 | Content Hash | `core/hashing/content.py` | ✅ Complete |
| I2 | Fingerprint | `core/hashing/fingerprint.py` | ✅ Complete |
| I3 | Workers | `core/concurrency/workers.py` | ✅ Complete |
| I3 | Async Utils | `core/concurrency/async_utils.py` | ✅ Complete |
| I4 | Validation Rules | `core/validation/rules.py` | ✅ Complete |
| I5 | Event Protocol | `core/events/protocol.py` | ✅ Complete |
| I5 | Event Emitter | `core/events/emitter.py` | ✅ Complete |
| I5 | Hook Registry | `core/events/registry.py` | ✅ Complete |
| I6 | Query Builder | `core/queries/builder.py` | ✅ Complete |

---

## High Priority Consolidation Opportunities (Phase G)

### G1. Service Pattern Standardization

**Current State:**
Multiple "service" classes with similar patterns (15+ services):
- `graphs/catalog.py` - `CatalogService`
- `cli/services/runtime.py` - `RuntimeService`
- `cli/services/storage.py` - `StorageService`
- `cli/services/params.py` - `ParamService`
- `cli/config/service.py` - `ConfigService`
- `cli/services/jobs.py` - `JobService`
- `cli/services/serving.py` - `ServingService`
- `cli/rendering/service.py` - `RenderingService`
- `serving/services/*.py` - Various serving services
- `ingestion/engine/service.py` - `IngestionService`
- `serving/backend/duckdb_service.py` - `DuckDBService`

**Problem:** Services share patterns but don't follow a consistent interface:
- Inconsistent initialization
- Different caching approaches
- No standard lifecycle management (start/stop)
- Varied dependency injection patterns

**Proposed Consolidation:**

```
core/services/
├── __init__.py           # Unified exports
├── protocol.py           # ServiceProtocol with lifecycle
├── base.py               # BaseService with common patterns
├── lifecycle.py          # ServiceLifecycle management
└── registry.py           # ServiceRegistry for DI
```

**Key Protocol:**

```python
@runtime_checkable
class ServiceProtocol(Protocol):
    """Protocol for all service types."""
    
    SERVICE_NAME: ClassVar[str]
    
    def initialize(self) -> None:
        """Initialize the service."""
        ...
    
    def shutdown(self) -> None:
        """Shutdown the service gracefully."""
        ...
    
    @property
    def is_ready(self) -> bool:
        """Whether the service is ready to handle requests."""
        ...
```

**Benefits:**
- Unified service lifecycle management
- Consistent initialization patterns
- Better service discovery and dependency injection
- Standardized health checking

---

### G2. Repository Pattern Enhancement

**Current State:**
Repository implementations in `storage/repositories/`:
- `base.py` - `BaseRepository` with common query patterns
- `functions.py` - `FunctionRepository`
- `modules.py` - `ModuleRepository`
- `tests.py` - `TestRepository`
- `graphs.py` - `GraphRepository`
- `subsystems.py` - `SubsystemRepository`
- `datasets.py` - `DatasetRepository`
- `dataflow.py` - `DataflowRepository`
- `data_models.py` - `DataModelRepository`

**Problem:** Repositories have evolved independently with:
- Different pagination approaches (`PaginatedRows` vs ad-hoc)
- Inconsistent filtering patterns
- No standard aggregate methods
- Varying transaction handling

**Proposed Enhancement:**

```
core/repository/
├── __init__.py           # Unified exports
├── protocol.py           # RepositoryProtocol with CRUD
├── base.py               # BaseRepository with common patterns
├── pagination.py         # Unified pagination types
├── filtering.py          # Filter builder utilities
└── aggregates.py         # Standard aggregate methods
```

**Key Protocol:**

```python
@runtime_checkable
class RepositoryProtocol[T](Protocol):
    """Protocol for repository implementations."""
    
    def get(self, id: int | str) -> T | None:
        """Get a single entity by ID."""
        ...
    
    def list(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        pagination: Pagination | None = None,
    ) -> PagedResult[T]:
        """List entities with filtering and pagination."""
        ...
    
    def count(self, *, filters: Mapping[str, object] | None = None) -> int:
        """Count entities matching filters."""
        ...
    
    def exists(self, id: int | str) -> bool:
        """Check if entity exists."""
        ...
```

---

### G3. Caching Infrastructure Consolidation

**Current State:**
Multiple caching approaches (79+ files with caching):
- `@lru_cache` decorators scattered across modules (69+ files)
- `core/resources/protocol.py` - Resource caching in providers
- `graphs/engine/cache.py` - `GraphCache` for graph-specific caching
- `analytics/parsing/ast_cache.py` - AST caching
- `analytics/functions/config.py` - Config caching
- `storage/gateway_cache.py` - Gateway caching
- `core/data/snapshot.py` - `SnapshotScopedCache` (new)
- Various `_cache: dict` patterns in classes

**Problem:** Caching is implemented inconsistently:
- No unified invalidation strategy
- Different TTL/expiration approaches
- Cache keys computed differently
- No cache metrics/observability
- `GraphCache` and `SnapshotScopedCache` have similar patterns

**Proposed Consolidation:**

```
core/cache/
├── __init__.py           # Unified exports
├── protocol.py           # CacheProtocol
├── memory.py             # In-memory cache with LRU/TTL
├── keying.py             # Cache key generation utilities
├── invalidation.py       # Invalidation strategies
├── scoped.py             # ScopedCache base (snapshot, graph, etc.)
└── metrics.py            # Cache hit/miss metrics
```

**Key Protocol:**

```python
@runtime_checkable
class CacheProtocol[K, V](Protocol):
    """Protocol for cache implementations."""
    
    def get(self, key: K) -> V | None:
        """Get cached value."""
        ...
    
    def set(self, key: K, value: V, *, ttl_s: float | None = None) -> None:
        """Set cached value with optional TTL."""
        ...
    
    def invalidate(self, key: K) -> bool:
        """Invalidate a specific key."""
        ...
    
    def clear(self) -> int:
        """Clear all cached values, return count cleared."""
        ...
    
    @property
    def stats(self) -> CacheStats:
        """Return cache statistics."""
        ...
```

**Benefits:**
- Unified cache invalidation
- Observable cache performance
- Consistent TTL handling
- Reusable scoped caching patterns

---

### G4. Provider Pattern Consolidation

**Current State:**
Multiple provider classes with similar patterns (17+ providers):
- `analytics/resources/catalog.py` - `CatalogProvider`
- `analytics/resources/graphs.py` - `GraphProvider`
- `analytics/resources/asts.py` - `AstProvider`
- `analytics/resources/features.py` - `FeatureProvider`
- `analytics/resources/module_map.py` - `ModuleMapProvider`
- `core/resources/graphs.py` - Core graph provider
- `core/catalog/protocol.py` - `CatalogProviderProtocol`
- `build/providers.py` - Build-time providers
- `serving/backend/core.py` - Backend providers
- `core/plugins/execution/context.py` - Context providers

**Problem:** Providers share the "lazy load and cache" pattern but:
- Different initialization approaches
- No standard refresh/invalidation
- Inconsistent error handling for missing data
- Various caching strategies

**Proposed Consolidation:**

```
core/providers/
├── __init__.py           # Unified exports
├── protocol.py           # ProviderProtocol with get/refresh
├── base.py               # BaseProvider with lazy loading
├── lazy.py               # LazyProvider decorator
└── registry.py           # ProviderRegistry for discovery
```

**Key Protocol:**

```python
@runtime_checkable
class ProviderProtocol[T](Protocol):
    """Protocol for resource providers."""
    
    def get(self) -> T:
        """Get the provided resource (lazy-load if needed)."""
        ...
    
    def refresh(self) -> None:
        """Force refresh of cached resource."""
        ...
    
    @property
    def is_loaded(self) -> bool:
        """Whether resource is currently loaded."""
        ...
```

---

## Medium Priority Consolidation Opportunities (Phase H)

### H1. Adapter Pattern Standardization

**Current State:**
Multiple adapter classes (8+ adapters):
- `ingestion/adapters/duckdb_storage.py` - Storage adapter
- `ingestion/adapters/tool_runner.py` - Tool execution adapter
- `ingestion/adapters/build_tool_adapter.py` - Build system adapter
- `ingestion/adapters/filesystem_discovery.py` - File discovery adapter
- `ingestion/adapters/hash_change_detection.py` - Change detection
- `build/hamilton/io/ibis_adapter.py` - Ibis I/O adapter
- `storage/ibis_adapter.py` - `IbisGateway` adapter
- `serving/services/transport.py` - Transport adapter

**Proposed:** Create `core/adapters/` with base adapter protocols for hexagonal architecture.

---

### H2. Factory Pattern Consolidation

**Current State:**
Multiple factory implementations (7+ factories):
- `analytics/resources/factory.py` - `ResourceFactory`
- `storage/repositories/factory.py` - `RepositoryFactory`
- `graphs/engine/factory.py` - Graph engine factory
- `build/hamilton/driver_factory.py` - Hamilton driver factory
- `build/hamilton/nodes/node_factory.py` - Node factory
- `config/builder.py` - Config builder
- `build/plugin.py` - Plugin factory patterns

**Proposed:** Create `core/factory/` with factory protocols and utilities.

---

### H3. Context Builder Pattern

**Current State:**
Multiple context types and builders (49+ context classes):
- `core/plugins/execution/context.py` - `PluginExecutionContext`, `PluginExecutionContextBuilder`
- `core/execution/context.py` - Execution context
- `build/context.py` - `BuildContext`
- `build/context_base.py` - `ContextBase`
- `cli/context.py` - CLI context
- `serving/context.py` - Serving context
- `serving/mcp/tool_context.py` - MCP tool context
- `analytics/runtime/context.py` - `AnalyticsRuntimeContext`
- Various domain-specific contexts

**Proposed:** Create `core/context/builder.py` with generic builder pattern and `BaseContext`.

---

### H4. Metrics/Observability Consolidation

**Current State:**
- `cli/observability/_observability.py` - CLI observability utilities
- `cli/observability/_telemetry.py` - CLI telemetry
- `serving/services/observability.py` - Serving observability
- `build/hamilton/telemetry_hook.py` - Hamilton telemetry
- `build/hamilton/observability.py` - Build observability
- Various metrics scattered across modules

**Proposed:** Create unified observability infrastructure in `core/observability/`.

---

### H5. Row Model Standardization

**Current State:**
Row type definitions scattered across (11+ files):
- `config/datasets/rows/core.py` - Core row types
- `config/datasets/rows/analytics.py` - Analytics row types
- `config/datasets/rows/graph.py` - Graph row types
- `config/datasets/rows/profiles.py` - Profile row types
- `config/datasets/rows/test.py` - Test row types
- `config/datasets/generated_rows/*.py` - Generated row types
- `core/data_models/rows.py` - Core data model rows

**Problem:** Row definitions are inconsistent and duplicated, with TypedDict definitions not always matching schema definitions.

**Proposed:** Create `core/models/rows.py` with base row protocols and standardized field naming.

---

## Lower Priority Opportunities (Phase I)

### I1. Path Utilities Consolidation

**Current State:**
Path handling scattered across modules:
- `ingestion/infrastructure/paths.py` - `normalize_rel_path`, `relpath_to_module`, `safe_relpath`
- `config/primitives.py` - Path-related utilities
- `config/models.py` - Path normalization
- Various modules with ad-hoc path handling

**Proposed:** Create `core/paths/` with unified path utilities.

---

### I2. Hashing/Fingerprinting Consolidation

**Current State:**
- `build/hashing.py` - Build hashing
- `build/assets/fingerprinting.py` - Asset fingerprinting
- `ingestion/adapters/hash_change_detection.py` - Change detection
- Various content hashing patterns

**Proposed:** Create `core/hashing/` with unified hashing utilities.

---

### I3. Worker/Threading Patterns

**Current State:**
- `ingestion/infrastructure/workers.py` - `WorkerConfig`, `resolve_worker_count`, `create_executor`
- Various async patterns across modules
- ThreadPoolExecutor and ProcessPoolExecutor usage scattered

**Proposed:** Create `core/concurrency/` with worker pool abstractions and async utilities.

---

### I4. Validation Rule Engine

**Current State:**
- `core/validation/runner.py` - Validation runner
- `graphs/validation/findings.py` - Graph validation findings
- `graphs/validation/runner.py` - Graph validation runner
- `storage/validation/*.py` - Storage validation
- `config/datasets/validation.py` - Dataset validation
- `cli/introspection/validation.py` - CLI validation

**Proposed:** Create `core/validation/rules.py` with a composable rule engine for unified validation across all domains.

---

### I5. Event/Hook System

**Current State:**
- Various callback patterns across modules
- `build/hamilton/manifest_hook.py` - Manifest hooks
- `build/hamilton/telemetry_hook.py` - Telemetry hooks
- No unified event system for extensibility

**Proposed:** Create `core/events/` with a lightweight event system for cross-module communication without tight coupling.

---

### I6. Query Builder Consolidation

**Current State:**
- `storage/queries/safe.py` - Safe query building
- `storage/duckdb_policy_backend.py` - Policy-based queries
- Various raw SQL construction patterns
- Ibis query building in `storage/ibis_adapter.py`

**Proposed:** Create `core/queries/` with unified query builder abstractions.

---

## Implementation Roadmap

### Phase F: Result & Serialization ✅ COMPLETED
1. ✅ Created `core/results/` with ResultProtocol, BaseResult, ExecutionResult
2. ✅ Created `core/serialization/` with SerializableProtocol, converters
3. ✅ Created `core/data/` with DataLoaderProtocol, SnapshotKey, SnapshotScopedCache
4. ✅ Updated CLI serialization to re-export from core

### Phase G: Services & Infrastructure ✅ COMPLETED
1. ✅ Created `core/services/` with ServiceProtocol, BaseService, lifecycle, registry
2. ✅ Enhanced repository pattern in `core/repository/` with protocols, pagination, filtering
3. ✅ Consolidated caching in `core/cache/` with CacheProtocol, memory cache, scoped caches
4. ✅ Created `core/providers/` with ProviderProtocol, BaseProvider, lazy decorator

### Phase H: Adapters & Patterns ✅ COMPLETED
1. ✅ Created `core/adapters/` with AdapterProtocol, BaseAdapter
2. ✅ Created `core/factory/` with FactoryProtocol, BaseFactory, CachingFactory, registry
3. ✅ Consolidated context builders in `core/context/` with BaseContext, ContextBuilder
4. ✅ Unified observability infrastructure in `core/observability/` with metrics, tracing
5. ✅ Standardized row models in `core/models/rows.py`

### Phase I: Utilities & Extensions ✅ COMPLETED
1. ✅ Consolidated path utilities in `core/paths/` with normalize, module conversion
2. ✅ Unified hashing infrastructure in `core/hashing/` with content hash, fingerprinting
3. ✅ Standardized worker patterns in `core/concurrency/` with workers, async utilities
4. ✅ Created validation rule engine in `core/validation/rules.py`
5. ✅ Added event/hook system in `core/events/` with emitter, registry
6. ✅ Created query builder abstractions in `core/queries/`

---

## Metrics for Success

| Metric | Target | Status |
|--------|--------|--------|
| Duplicate code reduction | 30% reduction in similar patterns | ✅ Achieved |
| Import depth | Max 3 levels for common types | ✅ Achieved |
| Type coverage | 100% for all consolidated modules | ✅ Achieved |
| Test coverage | 90%+ for core modules | 🔄 In Progress |
| Documentation | All protocols documented with examples | ✅ Achieved |
| Backward compatibility | All existing imports continue to work | ✅ Achieved |
| Core module count | 15+ unified core modules | ✅ 23 complete |

---

## Architecture Principles

The consolidation follows these principles:

1. **Protocol-First Design**: Define protocols before implementations
2. **Backward Compatibility**: Always re-export from original locations
3. **Incremental Migration**: Allow gradual adoption of new patterns
4. **Composition over Inheritance**: Prefer protocol composition
5. **Single Source of Truth**: One canonical location for each concept
6. **Explicit Deprecation**: Use deprecation warnings with migration guides
7. **Lazy Loading**: Resources should be loaded on-demand where possible
8. **Observable by Default**: Include metrics/logging hooks in infrastructure

---

## Core Module Summary

| Module | Purpose | Files | Status |
|--------|---------|-------|--------|
| `core/errors/` | Error taxonomy, Problem Details | 5 | ✅ Complete |
| `core/options/` | Options protocol and base | 3 | ✅ Complete |
| `core/catalog/` | Function spans, span index | 4 | ✅ Complete |
| `core/parsing/` | Source spans, AST index, models | 4 | ✅ Complete |
| `core/runtime/` | Execution tracking, protocols | 3 | ✅ Complete |
| `core/context/` | Context protocols, builder | 4 | ✅ Complete |
| `core/results/` | Result protocol, base types | 4 | ✅ Complete |
| `core/serialization/` | Serialization protocol, converters | 4 | ✅ Complete |
| `core/data/` | Data loading, snapshot caching | 4 | ✅ Complete |
| `core/services/` | Service protocol, lifecycle, registry | 5 | ✅ Complete |
| `core/repository/` | Repository protocol, pagination, filtering | 4 | ✅ Complete |
| `core/cache/` | Cache protocol, memory, scoped, keying | 5 | ✅ Complete |
| `core/providers/` | Provider protocol, base, lazy | 4 | ✅ Complete |
| `core/adapters/` | Adapter protocol, base | 3 | ✅ Complete |
| `core/factory/` | Factory protocol, base, registry | 4 | ✅ Complete |
| `core/observability/` | Metrics, tracing, protocols | 4 | ✅ Complete |
| `core/models/` | Row model protocols | 2 | ✅ Complete |
| `core/paths/` | Path utilities, module conversion | 3 | ✅ Complete |
| `core/hashing/` | Content hash, fingerprinting | 3 | ✅ Complete |
| `core/concurrency/` | Workers, async utilities | 3 | ✅ Complete |
| `core/validation/` | Validation rule engine | 2 | ✅ Complete |
| `core/events/` | Event emitter, hook registry | 4 | ✅ Complete |
| `core/queries/` | Query builder protocol | 2 | ✅ Complete |

---

## Notes

- All consolidations maintain backward compatibility via re-exports
- Each phase includes comprehensive tests
- Migration is incremental with clear deprecation warnings
- Consider using `typing.deprecated` decorator (Python 3.13+) for phased deprecation
- New code should import from `core/` modules directly
- The `core/data/snapshot.py` module provides `SnapshotScopedCache` which can serve as the basis for `core/cache/`

---

*Last Updated: December 2024*
*Author: Cross-Module Consolidation Analysis*
*Phases G, H, I Completed: December 2024*
