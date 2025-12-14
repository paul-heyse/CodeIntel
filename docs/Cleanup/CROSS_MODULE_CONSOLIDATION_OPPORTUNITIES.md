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

---

## High Priority Consolidation Opportunities (Phase F-H)

### F1. Result Type Unification

**Current State:**
Multiple result types with similar patterns but different implementations:
- `core/runtime/tracking.py` - `StepResult`, `StepStatus`
- `core/plugins/types/result.py` - `BasePluginResult`, `PluginResult`, `PluginExecutionRecord`
- `ingestion/engine/results.py` - `DiagnosticReport`, `CoverageReport`, `TestReport`
- `ingestion/engine/infrastructure/runner.py` - `ToolRunResult`
- `cli/core/result_types.py` - `ListResult`, `ActionResult`, `StatusResult`
- `build/result.py` - Build-specific results
- `core/ports/results.py` - `BaseQueryResult`, `BaseBatchResult`

**Problem:** Similar result patterns (success/failure, row counts, duration, artifacts) are implemented differently across modules, making it hard to aggregate and report on execution outcomes consistently.

**Proposed Consolidation:**
```
core/results/
├── __init__.py           # Unified exports
├── protocol.py           # ResultProtocol with success/error/duration
├── base.py               # BaseResult with common factory methods
├── execution.py          # ExecutionResult for steps/plugins
├── query.py              # QueryResult for database operations
├── batch.py              # BatchResult for bulk operations
└── aggregation.py        # ResultAggregator for combining results
```

**Key Protocol:**
```python
@runtime_checkable
class ResultProtocol(Protocol):
    """Unified protocol for all result types."""
    
    @property
    def success(self) -> bool:
        """Whether the operation succeeded."""
        ...
    
    @property
    def error(self) -> str | None:
        """Error message if failed."""
        ...
    
    @property
    def duration_s(self) -> float:
        """Operation duration in seconds."""
        ...
    
    @classmethod
    def ok(cls, **kwargs: object) -> Self:
        """Create a success result."""
        ...
    
    @classmethod
    def fail(cls, error: str, **kwargs: object) -> Self:
        """Create a failure result."""
        ...
```

**Benefits:**
- Consistent result handling across all modules
- Unified aggregation for pipeline/batch operations
- Simpler result composition and chaining
- Better observability integration

---

### F2. Serialization Protocol Standardization

**Current State:**
Many classes implement `to_dict`/`from_dict` with inconsistent patterns:
- `core/errors/problem_details.py` - `ProblemDetail.to_dict()`, `to_json()`
- `core/options/base.py` - `BaseOptions.to_dict()`, `from_dict()`
- `analytics/runtime/graph.py` - `GraphRuntimeOptions.to_dict()`
- `cli/core/results.py` - `@result_type` decorator for auto-serialization
- `config/datasets/schema.py` - Schema serialization
- 74+ files with `to_dict`/`from_dict` patterns

**Problem:** Serialization is implemented ad-hoc, leading to:
- Inconsistent None handling
- Different datetime/path/enum serialization approaches
- No standard validation on deserialization
- Duplicate serialization logic

**Proposed Consolidation:**
```
core/serialization/
├── __init__.py           # Unified exports
├── protocol.py           # SerializableProtocol
├── base.py               # BaseSeriazable mixin
├── converters.py         # Type converters (datetime, Path, Enum)
├── validation.py         # Deserialization validation
└── decorators.py         # @serializable decorator
```

**Key Protocol:**
```python
@runtime_checkable
class SerializableProtocol(Protocol):
    """Protocol for serializable types."""
    
    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary."""
        ...
    
    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Deserialize from dictionary."""
        ...
    
    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize to JSON string."""
        ...
```

**Benefits:**
- Consistent serialization across all types
- Type-safe deserialization with validation
- Reusable converters for complex types
- Better JSON schema generation

---

### F3. Data Loading Pattern Consolidation

**Current State:**
Multiple patterns for loading data from storage:
- `graphs/catalog.py` - `load_function_catalog()`, `load_function_spans()`
- `storage/helpers/module_index.py` - `load_module_map()`
- `analytics/resources/*.py` - Various `_load()` methods
- `graphs/engine/nx_engine.py` - `load_from_db()`
- `analytics/parsing/ast_cache.py` - AST caching with load patterns

**Problem:** Data loading patterns are scattered with different:
- Error handling approaches
- Caching strategies
- Snapshot scoping (repo/commit filtering)
- Null/empty result handling

**Proposed Consolidation:**
```
core/data/
├── __init__.py           # Unified exports
├── protocol.py           # DataLoaderProtocol
├── loader.py             # BaseDataLoader with caching
├── snapshot.py           # SnapshotScopedLoader mixin
└── cache.py              # LoaderCache for shared caching
```

**Key Pattern:**
```python
class DataLoaderProtocol[T](Protocol):
    """Protocol for data loaders with snapshot scoping."""
    
    def load(
        self,
        gateway: StorageGateway,
        *,
        repo: str,
        commit: str,
    ) -> T:
        """Load data for a snapshot."""
        ...
    
    def invalidate(self, *, repo: str | None = None, commit: str | None = None) -> None:
        """Invalidate cached data."""
        ...
```

---

### G1. Service Pattern Standardization

**Current State:**
Multiple "service" classes with similar patterns:
- `graphs/catalog.py` - `CatalogService`
- `cli/services/runtime.py` - `RuntimeService`
- `cli/services/storage.py` - `StorageService`
- `cli/services/params.py` - `ParamService`
- `cli/config/service.py` - `ConfigService`
- `serving/services/*.py` - Various serving services

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

**Problem:** Repositories have evolved independently with:
- Different pagination approaches
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
```

---

### G3. Caching Infrastructure Consolidation

**Current State:**
Multiple caching approaches:
- `@lru_cache` decorators scattered across modules (69+ files)
- `core/resources/protocol.py` - Resource caching in providers
- `graphs/engine/cache.py` - Graph-specific caching
- `analytics/parsing/ast_cache.py` - AST caching
- `analytics/functions/config.py` - Config caching
- `storage/gateway_cache.py` - Gateway caching

**Problem:** Caching is implemented inconsistently:
- No unified invalidation strategy
- Different TTL/expiration approaches
- Cache keys computed differently
- No cache metrics/observability

**Proposed Consolidation:**
```
core/cache/
├── __init__.py           # Unified exports
├── protocol.py           # CacheProtocol
├── memory.py             # In-memory cache with LRU
├── keying.py             # Cache key generation utilities
├── invalidation.py       # Invalidation strategies
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
```

---

## Medium Priority Consolidation Opportunities (Phase H)

### H1. Adapter Pattern Standardization

**Current State:**
- `ingestion/adapters/duckdb_storage.py` - Storage adapter
- `ingestion/adapters/tool_runner.py` - Tool execution adapter
- `ingestion/adapters/build_tool_adapter.py` - Build system adapter
- `build/hamilton/io/ibis_adapter.py` - Ibis I/O adapter
- `storage/ibis_adapter.py` - Ibis gateway adapter

**Proposed:** Create `core/adapters/` with base adapter protocols for hexagonal architecture.

---

### H2. Factory Pattern Consolidation

**Current State:**
- `analytics/resources/factory.py` - Resource factory
- `storage/repositories/factory.py` - Repository factory
- `graphs/engine/factory.py` - Graph engine factory
- `build/hamilton/driver_factory.py` - Hamilton driver factory
- `config/builder.py` - Config builder

**Proposed:** Create `core/factory/` with factory protocols and utilities.

---

### H3. Metrics/Observability Consolidation

**Current State:**
- `cli/observability/_observability.py` - CLI observability
- `cli/observability/_telemetry.py` - CLI telemetry
- Various metrics scattered across modules

**Proposed:** Create unified observability infrastructure in `core/observability/`.

---

### H4. Context Builder Pattern

**Current State:**
- `core/plugins/execution/context.py` - `PluginExecutionContextBuilder`
- `build/context.py` - `BuildContext` construction
- `cli/context.py` - CLI context
- `serving/context.py` - Serving context
- `analytics/runtime/context.py` - Analytics context

**Proposed:** Create `core/context/builder.py` with generic builder pattern.

---

### H5. Row Model Standardization

**Current State:**
Row type definitions scattered across:
- `config/datasets/rows/core.py` - Core row types
- `config/datasets/generated_rows/*.py` - Generated row types
- Various `*Row` TypedDicts across modules

**Problem:** Row definitions are inconsistent and duplicated.

**Proposed:** Create `core/models/rows.py` with base row protocols and standardized field naming.

---

## Lower Priority Opportunities (Phase I)

### I1. Path Utilities Consolidation

**Current State:**
- `ingestion/infrastructure/paths.py` - `normalize_rel_path`, `relpath_to_module`
- Various modules with path handling

**Proposed:** Create `core/paths/` with unified path utilities.

---

### I2. Hashing/Fingerprinting Consolidation

**Current State:**
- `build/hashing.py` - Build hashing
- `build/assets/fingerprinting.py` - Asset fingerprinting
- `ingestion/adapters/hash_change_detection.py` - Change detection

**Proposed:** Create `core/hashing/` with unified hashing utilities.

---

### I3. Worker/Threading Patterns

**Current State:**
- `ingestion/infrastructure/workers.py` - Worker utilities
- Various async patterns across modules

**Proposed:** Create `core/concurrency/` with worker pool abstractions.

---

### I4. Validation Rule Engine

**Current State:**
- `core/validation/runner.py` - Validation runner
- `graphs/validation/findings.py` - Graph validation
- `storage/validation/*.py` - Storage validation
- `config/datasets/validation.py` - Dataset validation

**Proposed:** Create `core/validation/rules.py` with a composable rule engine for unified validation across all domains.

---

### I5. Event/Hook System

**Current State:**
- Various callback patterns across modules
- No unified event system for extensibility

**Proposed:** Create `core/events/` with a lightweight event system for cross-module communication without tight coupling.

---

## Implementation Roadmap

### Phase F: Result & Serialization (High Impact, Medium Effort)
1. ✅ Phase A-E completed
2. Create `core/results/` structure with ResultProtocol
3. Create `core/serialization/` with SerializableProtocol
4. Create `core/data/` with DataLoaderProtocol
5. Update existing result types to implement protocols

### Phase G: Services & Infrastructure (Medium Impact, Medium Effort)
1. Create `core/services/` with ServiceProtocol
2. Enhance repository pattern in `core/repository/`
3. Create `core/cache/` with unified caching

### Phase H: Adapters & Factories (Medium Impact, Low Effort)
1. Create `core/adapters/` with adapter protocols
2. Create `core/factory/` with factory utilities
3. Consolidate observability infrastructure
4. Standardize context builders

### Phase I: Utilities & Extensions (Lower Impact, Low Effort)
1. Consolidate path utilities
2. Unify hashing infrastructure
3. Standardize worker patterns
4. Create validation rule engine
5. Add event/hook system for extensibility

---

## Metrics for Success

| Metric | Target | Status |
|--------|--------|--------|
| Duplicate code reduction | 30% reduction in similar patterns | 🔄 In Progress |
| Import depth | Max 3 levels for common types | ✅ Achieved |
| Type coverage | 100% for all consolidated modules | ✅ Achieved |
| Test coverage | 90%+ for core modules | 🔄 In Progress |
| Documentation | All protocols documented with examples | ✅ Achieved |
| Backward compatibility | All existing imports continue to work | ✅ Achieved |

---

## Architecture Principles

The consolidation follows these principles:

1. **Protocol-First Design**: Define protocols before implementations
2. **Backward Compatibility**: Always re-export from original locations
3. **Incremental Migration**: Allow gradual adoption of new patterns
4. **Composition over Inheritance**: Prefer protocol composition
5. **Single Source of Truth**: One canonical location for each concept
6. **Explicit Deprecation**: Use deprecation warnings with migration guides

---

## Notes

- All consolidations maintain backward compatibility via re-exports
- Each phase includes comprehensive tests
- Migration is incremental with clear deprecation warnings
- Consider using `typing.deprecated` decorator (Python 3.13+) for phased deprecation
- New code should import from `core/` modules directly

---

*Last Updated: December 2024*
*Author: Cross-Module Consolidation Analysis*
