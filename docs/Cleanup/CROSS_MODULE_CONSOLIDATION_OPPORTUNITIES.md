# Cross-Module Consolidation Opportunities

> **Purpose**: This document identifies additional opportunities to streamline shared functionality across `analytics`, `ingestion`, `graphs`, and `core` modules, building on the Phase 1-7 consolidation work already completed.

## Completed Consolidations (Reference)

The following consolidations have already been implemented:

| Area | Core Module | Re-exports From |
|------|-------------|-----------------|
| Storage Ports | `core/ports/storage.py` | `graphs/ports/`, `ingestion/ports/` |
| Plugin Protocols | `core/plugins/types/async_protocol.py` | `ingestion/engine/plugins.py` |
| Graph Resources | `core/resources/graphs.py` | `analytics/resources/graphs.py` |
| Centrality Compute | `core/compute/centrality.py` | `graphs/compute/metrics/centrality.py` |
| Validation Runner | `core/validation/runner.py` | Used by `graphs/validation/` |
| Context Protocols | `core/context/protocol.py` | New unified protocols |
| Safe Queries | `storage/queries/safe.py` | `ingestion/infrastructure/db_queries.py` |

---

## High Priority Consolidation Opportunities

### 1. Error Taxonomy Unification

**Current State:**
- `core/execution/errors.py` - Plugin execution errors
- `cli/errors/taxonomy.py` - CLI error codes with RFC 9457 Problem Details
- `storage/queries/safe.py` - Query-specific errors (QueryError, TableNotFoundError)
- `serving/services/errors.py` - Service layer errors
- `serving/mcp/errors.py` - MCP-specific errors
- `build/errors.py` - Build system errors

**Problem:** Error hierarchies are fragmented across modules, making it difficult to handle errors consistently and provide unified error reporting.

**Proposed Consolidation:**
```
core/errors/
├── __init__.py           # Unified exports
├── base.py               # Base error classes and protocols
├── taxonomy.py           # Error code enums (from CLI)
├── problem_details.py    # RFC 9457 Problem Details factory
├── execution.py          # Plugin/execution errors
├── storage.py            # Storage/query errors
└── service.py            # Service layer errors
```

**Benefits:**
- Consistent error handling across all modules
- Single source of truth for error codes
- Unified Problem Details generation
- Better error correlation in observability

**Migration Path:**
1. Create `core/errors/base.py` with base classes
2. Move taxonomy enums to `core/errors/taxonomy.py`
3. Update existing error modules to inherit from core
4. Add re-exports for backward compatibility

---

### 2. Options/Config Pattern Unification

**Current State:**
Multiple `*Options` dataclasses across modules:
- `core/validation/options.py` - `BaseValidationOptions`
- `core/plugins/execution/options.py` - `PluginExecutionOptions`
- `analytics/runtime/context.py` - `GraphMetricsOptions`, `GraphContext`
- `graphs/validation/findings.py` - `GraphValidationOptions`
- `graphs/engine/factory.py` - `GraphEngineOptions`
- `build/plugins/*/` - Many plugin-specific options classes

**Problem:** Options classes follow different patterns, making it hard to compose and validate configurations consistently.

**Proposed Consolidation:**
```
core/options/
├── __init__.py           # Unified exports
├── protocol.py           # OptionsProtocol with validation
├── base.py               # BaseOptions with common fields
├── composition.py        # Options composition utilities
└── validation.py         # Options validation helpers
```

**Key Protocol:**
```python
@runtime_checkable
class OptionsProtocol(Protocol):
    """Protocol for all options/config classes."""
    
    def validate(self) -> ValidationResult:
        """Validate options and return any issues."""
        ...
    
    def with_defaults(self, defaults: Self) -> Self:
        """Merge with default values."""
        ...
    
    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary."""
        ...
```

**Benefits:**
- Consistent validation across all options
- Composable configuration patterns
- Easier testing and mocking

---

### 3. Catalog/Provider Pattern Consolidation

**Current State:**
- `graphs/catalog.py` - `FunctionCatalog`, `CatalogService`, `FunctionCatalogProvider`
- `analytics/resources/catalog.py` - `CatalogProvider` (wraps graphs catalog)
- `analytics/resources/graphs.py` - `GraphProvider`, `GraphResources`
- `graphs/resources/graphs.py` - `GraphResource` (different implementation)

**Problem:** Two different catalog/provider implementations exist with similar but not identical interfaces.

**Proposed Consolidation:**
```
core/catalog/
├── __init__.py           # Unified exports
├── protocol.py           # CatalogProtocol, CatalogProviderProtocol
├── function_span.py      # FunctionSpan dataclass (shared)
├── span_index.py         # SpanIndex for lookups
└── service.py            # Base CatalogService
```

**Benefits:**
- Single FunctionSpan definition
- Unified catalog protocol
- Consistent provider pattern

---

### 4. AST/Parsing Utilities Consolidation

**Current State:**
- `ingestion/infrastructure/ast_utils.py` - `AstSpanIndex`, `parse_python_module`
- `ingestion/infrastructure/cst_utils.py` - CST parsing utilities
- `analytics/parsing/models.py` - `SourceSpan`, `ParsedFunction`, `ParsedModule`
- `analytics/parsing/function_parsing.py` - Function extraction
- `analytics/parsing/span_resolver.py` - Span resolution

**Problem:** Parsing utilities are split between ingestion (lower-level) and analytics (higher-level) with unclear boundaries.

**Proposed Consolidation:**
```
core/parsing/
├── __init__.py           # Unified exports
├── spans.py              # SourceSpan, SpanIndex (unified)
├── ast_utils.py          # AST parsing utilities
├── models.py             # ParsedFunction, ParsedModule
└── protocols.py          # ParserProtocol, SpanResolverProtocol
```

**Migration Notes:**
- `AstSpanIndex` from ingestion becomes canonical
- `SourceSpan` from analytics becomes canonical
- Both modules re-export from core

---

### 5. Runtime/Executor Pattern Unification

**Current State:**
- `analytics/runtime/graph.py` - `GraphRuntime`
- `ingestion/engine/infrastructure/runner.py` - `ToolRunner`
- `build/hamilton/executor.py` - Hamilton executor
- `build/hamilton/native/executor.py` - Native executor
- `cli/execution/registry.py` - CLI execution registry

**Problem:** Multiple execution/runtime patterns with different interfaces for similar concepts (execute, track, report).

**Proposed Consolidation:**
```
core/runtime/
├── __init__.py           # Unified exports
├── protocol.py           # RuntimeProtocol, ExecutorProtocol
├── tracking.py           # Execution tracking utilities
├── timing.py             # Duration/timing helpers
└── reporting.py          # Execution report types
```

**Key Protocol:**
```python
@runtime_checkable
class ExecutorProtocol[TInput, TOutput](Protocol):
    """Protocol for all executors."""
    
    async def execute(self, input: TInput) -> TOutput:
        """Execute the operation."""
        ...
    
    def track(self, name: str) -> ExecutionTracker:
        """Get a tracker for timing/metrics."""
        ...
```

---

## Medium Priority Consolidation Opportunities

### 6. Adapter Pattern Standardization

**Current State:**
- `ingestion/adapters/duckdb_storage.py` - Storage adapter
- `ingestion/adapters/tool_runner.py` - Tool execution adapter
- `ingestion/adapters/build_tool_adapter.py` - Build system adapter
- `build/hamilton/io/ibis_adapter.py` - Ibis I/O adapter

**Proposed:** Create `core/adapters/` with base adapter protocols.

---

### 7. Factory Pattern Consolidation

**Current State:**
- `analytics/resources/factory.py` - Resource factory
- `storage/repositories/factory.py` - Repository factory
- `graphs/engine/factory.py` - Graph engine factory
- `build/hamilton/driver_factory.py` - Hamilton driver factory

**Proposed:** Create `core/factory/` with factory protocols and utilities.

---

### 8. Metrics/Observability Consolidation

**Current State:**
- `cli/observability/_observability.py` - CLI observability
- `cli/observability/_telemetry.py` - CLI telemetry
- `build/hamilton/observability.py` - Build observability
- `serving/services/observability.py` - Serving observability

**Proposed:** Create unified observability infrastructure in `core/observability/`.

---

### 9. Context Builder Pattern

**Current State:**
- `core/plugins/execution/context.py` - `PluginExecutionContextBuilder`
- `build/context.py` - `BuildContext` construction
- `cli/context.py` - CLI context
- `serving/context.py` - Serving context

**Proposed:** Create `core/context/builder.py` with generic builder pattern.

---

## Lower Priority Opportunities

### 10. Path Utilities Consolidation

**Current State:**
- `ingestion/infrastructure/paths.py` - `normalize_rel_path`
- Various modules with path handling

**Proposed:** Create `core/paths/` with unified path utilities.

---

### 11. Hashing/Fingerprinting Consolidation

**Current State:**
- `build/hashing.py` - Build hashing
- `build/assets/fingerprinting.py` - Asset fingerprinting
- `ingestion/adapters/hash_change_detection.py` - Change detection

**Proposed:** Create `core/hashing/` with unified hashing utilities.

---

### 12. Worker/Threading Patterns

**Current State:**
- `ingestion/infrastructure/workers.py` - Worker utilities
- Various async patterns across modules

**Proposed:** Create `core/concurrency/` with worker pool abstractions.

---

## Implementation Roadmap

### Phase A: Error Taxonomy (High Impact, Medium Effort)
1. Create `core/errors/` structure
2. Move base error classes
3. Migrate taxonomy enums
4. Update all error imports

### Phase B: Options Protocol (High Impact, Low Effort)
1. Create `core/options/protocol.py`
2. Define `OptionsProtocol`
3. Update existing options to implement protocol
4. Add validation utilities

### Phase C: Catalog Consolidation (Medium Impact, Medium Effort)
1. Move `FunctionSpan` to core
2. Create unified `CatalogProtocol`
3. Update graphs and analytics to use core types

### Phase D: Parsing Utilities (Medium Impact, High Effort)
1. Create `core/parsing/` structure
2. Move AST utilities
3. Consolidate span types
4. Update all consumers

### Phase E: Runtime/Executor (Medium Impact, High Effort)
1. Create `core/runtime/` structure
2. Define executor protocols
3. Implement tracking utilities
4. Migrate existing runtimes

---

## Metrics for Success

| Metric | Target |
|--------|--------|
| Duplicate code reduction | 30% reduction in similar patterns |
| Import depth | Max 3 levels for common types |
| Type coverage | 100% for all consolidated modules |
| Test coverage | 90%+ for core modules |
| Documentation | All protocols documented with examples |

---

## Notes

- All consolidations should maintain backward compatibility via re-exports
- Each phase should include comprehensive tests
- Migration should be incremental with clear deprecation warnings
- Consider using `typing.deprecated` decorator for phased deprecation

---

*Last Updated: December 2024*
*Author: Code Consolidation Analysis*
