# Cross-Module Consolidation - Phase 1 Implementation

> **Status**: ✅ Completed  
> **Date**: December 2024  
> **Scope**: Storage ports, plugin protocols, graph resources, compute functions, validation, context protocols, and query utilities

## Executive Summary

This document describes the first phase of cross-module consolidation work that streamlined shared functionality from `analytics`, `ingestion`, and `graphs` into the `core` module. The consolidation eliminates code duplication while maintaining full backward compatibility through re-exports.

---

## Implementation Overview

### Files Created

| File | Purpose |
|------|---------|
| `core/ports/storage.py` | Unified QueryResult, BatchResult, StoragePort types |
| `core/plugins/types/async_protocol.py` | AsyncPluginProtocol for tool plugins |
| `core/resources/graphs.py` | GraphBundle and GraphProviderProtocol |
| `core/compute/__init__.py` | Compute module initialization |
| `core/compute/centrality.py` | Pure centrality metric functions |
| `core/validation/runner.py` | ValidationRunner and CheckProtocol |
| `core/context/__init__.py` | Context module initialization |
| `core/context/protocol.py` | ExecutionContextProtocol and related protocols |
| `storage/queries/__init__.py` | Query utilities module initialization |
| `storage/queries/safe.py` | Safe database query helpers |

### Files Modified

| File | Change |
|------|--------|
| `core/ports/__init__.py` | Added exports for new storage types |
| `core/plugins/types/__init__.py` | Added AsyncPluginProtocol exports |
| `core/plugins/types/protocol.py` | Added "tool" to PluginKind, tool metadata fields |
| `core/resources/__init__.py` | Added graph resource exports |
| `core/validation/__init__.py` | Added ValidationRunner exports |
| `graphs/ports/storage.py` | Converted to re-export from core |
| `ingestion/ports/storage.py` | Converted to re-export with backward-compat wrapper |
| `ingestion/engine/plugins.py` | Updated to use core plugin types |
| `analytics/resources/graphs.py` | GraphResources aliased to GraphBundle |
| `graphs/compute/metrics/centrality.py` | Converted to re-export from core |
| `ingestion/infrastructure/db_queries.py` | Converted to re-export from storage |

---

## Detailed Changes by Phase

### Phase 1: Storage Ports and Result Types

**Problem:** `QueryResult` and `BatchResult` were independently defined in both `graphs/ports/storage.py` and `ingestion/ports/storage.py` with slightly different field names and interfaces.

**Solution:** Created unified types in `core/ports/storage.py`:

```python
# core/ports/storage.py

@dataclass(frozen=True)
class QueryResult:
    """Unified query result with column metadata."""
    rows: tuple[tuple[object, ...], ...]
    columns: tuple[str, ...] = ()
    row_count: int = 0
    
    @classmethod
    def empty(cls) -> QueryResult: ...
    @classmethod
    def from_rows(cls, rows, columns=None) -> QueryResult: ...

@dataclass(frozen=True)
class BatchResult:
    """Unified batch operation result."""
    table: str
    rows_affected: int
    success: bool = True
    error: str | None = None
    duration_s: float = 0.0
    
    # Backward-compatible aliases
    @property
    def table_key(self) -> str: ...  # Alias for table
    @property
    def rows_written(self) -> int: ...  # Alias for rows_affected
    
    @classmethod
    def ok(cls, table, rows_affected, duration_s=0.0) -> BatchResult: ...
    @classmethod
    def fail(cls, table, error) -> BatchResult: ...
    @classmethod
    def from_write(cls, table_key, rows_written, duration_s=0.0) -> BatchResult: ...

@runtime_checkable
class StoragePort(Protocol):
    """Unified storage port protocol."""
    def execute_query(self, sql, params=None) -> QueryResult: ...
    def execute_mutation(self, sql, params=None) -> int: ...
    def write_batch(self, table_key, rows, *, scope=None) -> BatchResult: ...
    # ... additional methods
```

**Backward Compatibility:**
- `graphs/ports/storage.py` now re-exports directly from core
- `ingestion/ports/storage.py` provides a thin wrapper with `table_key`/`rows_written` field names for existing code

---

### Phase 2: Plugin System Unification

**Problem:** Tool plugins in `ingestion/engine/plugins.py` had their own `ToolPluginMetadata` that was separate from the core `PluginMetadata`, preventing unified plugin introspection.

**Solution:** 

1. Created `core/plugins/types/async_protocol.py`:

```python
# core/plugins/types/async_protocol.py

@runtime_checkable
class AsyncPluginProtocol(Protocol):
    """Protocol for async plugin execution (tool plugins)."""
    
    @property
    def metadata(self) -> PluginMetadata: ...
    
    async def execute(self, ctx: PluginExecutionContext) -> PluginResult: ...
    
    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult: ...
```

2. Extended `PluginKind` and `PluginMetadata`:

```python
# core/plugins/types/protocol.py

PluginKind = Literal["builder", "metric", "validation", "analytics", "tool"]

@dataclass(frozen=True)
class PluginMetadata:
    # ... existing fields ...
    
    # Tool plugin fields (optional, used when kind="tool")
    tool_binary: str | None = None
    produces_artifacts: tuple[str, ...] = ()
    consumes_configs: tuple[str, ...] = ()
```

3. Updated `ToolPluginMetadata` with conversion method:

```python
# ingestion/engine/plugins.py

@dataclass(frozen=True)
class ToolPluginMetadata:
    name: str
    produces_artifacts: tuple[str, ...]
    # ... other fields ...
    
    def to_core_metadata(self) -> CorePluginMetadata:
        """Convert to core PluginMetadata for unified introspection."""
        return CorePluginMetadata(
            name=self.name,
            kind="tool",
            stage="pipeline_ingestion",
            tool_binary=self.tool_binary,
            produces_artifacts=self.produces_artifacts,
            # ...
        )
```

---

### Phase 3: Graph Resource Provider Unification

**Problem:** `analytics/resources/graphs.py` defined `GraphResources` and `graphs/resources/graphs.py` defined `GraphResource` with similar but not identical structures.

**Solution:** Created unified `GraphBundle` in `core/resources/graphs.py`:

```python
# core/resources/graphs.py

@dataclass
class GraphBundle:
    """Unified container for all graph types."""
    call_graph: nx.DiGraph | None = None
    import_graph: nx.DiGraph | None = None
    symbol_module_graph: nx.Graph | None = None
    symbol_function_graph: nx.Graph | None = None
    config_module_bipartite: nx.Graph | None = None
    test_function_bipartite: nx.Graph | None = None
    cfg_graph: nx.DiGraph | None = None
    
    @classmethod
    def empty(cls) -> GraphBundle: ...
    
    @property
    def has_call_graph(self) -> bool: ...
    @property
    def available_graphs(self) -> tuple[str, ...]: ...

@runtime_checkable
class GraphProviderProtocol[T_co](Protocol):
    """Protocol for graph resource providers."""
    RESOURCE_NAME: ClassVar[str]
    
    def get(self) -> T_co: ...
    def invalidate(self) -> None: ...
    
    @property
    def call_graph(self) -> nx.DiGraph | None: ...
    @property
    def import_graph(self) -> nx.DiGraph | None: ...
```

**Backward Compatibility:**
- `analytics/resources/graphs.py` aliases `GraphResources = GraphBundle`
- Both types are exported for discoverability

---

### Phase 4: Centrality Compute Function Consolidation

**Problem:** Pure centrality computation functions in `graphs/compute/metrics/centrality.py` could be shared across both graphs and analytics without any I/O dependencies.

**Solution:** Moved all pure functions to `core/compute/centrality.py`:

```python
# core/compute/centrality.py

@dataclass(frozen=True)
class CentralityMetrics:
    """Collection of centrality metrics for a node."""
    pagerank: float
    betweenness: float
    closeness: float
    harmonic: float
    eigenvector: float
    in_degree: int
    out_degree: int
    degree: int

def compute_pagerank(graph, alpha=0.85, max_iter=100, tol=1e-6, weight=None) -> dict[Any, float]: ...
def compute_betweenness(graph, *, normalized=True, k=None, weight=None, seed=None) -> dict[Any, float]: ...
def compute_closeness(graph, *, wf_improved=True) -> dict[Any, float]: ...
def compute_harmonic_centrality(graph) -> dict[Any, float]: ...
def compute_eigenvector_centrality(graph, *, max_iter=100, tol=1e-6, weight=None) -> dict[Any, float]: ...
def compute_degree_centrality(graph) -> dict[Any, float]: ...
def compute_in_degree_centrality(graph) -> dict[Any, float]: ...
def compute_out_degree_centrality(graph) -> dict[Any, float]: ...
def compute_all_centralities(graph, *, alpha=0.85, betweenness_k=None, ...) -> dict[Any, CentralityMetrics]: ...
def centrality_to_rows(metrics, repo, commit) -> list[dict[str, object]]: ...
```

**Backward Compatibility:**
- `graphs/compute/metrics/centrality.py` re-exports all functions from core

---

### Phase 5: Validation Framework Completion

**Problem:** Validation infrastructure was partially consolidated but lacked a unified runner pattern.

**Solution:** Created `core/validation/runner.py`:

```python
# core/validation/runner.py

@runtime_checkable
class CheckProtocol[TContext](Protocol):
    """Protocol for validation check implementations."""
    
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def severity(self) -> ValidationSeverity: ...
    
    def __call__(self, ctx: TContext) -> Sequence[Mapping[str, object]]: ...

@dataclass
class CheckResult[TFinding: Mapping[str, object]]:
    """Result from executing a single check."""
    check_name: str
    findings: list[TFinding] = field(default_factory=list)
    duration_s: float = 0.0
    error: str | None = None
    skipped: bool = False

@dataclass
class ValidationReport[TFinding: Mapping[str, object]]:
    """Aggregate report from a validation run."""
    findings: list[TFinding] = field(default_factory=list)
    check_results: list[CheckResult[TFinding]] = field(default_factory=list)
    total_duration_s: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    # ...
    
    @property
    def has_errors(self) -> bool: ...
    @property
    def passed(self) -> bool: ...

@dataclass
class ValidationRunner[TContext, TFinding: Mapping[str, object]]:
    """Generic runner for executing validation checks."""
    checks: list[CheckProtocol[TContext]] = field(default_factory=list)
    options: BaseValidationOptions | None = None
    
    def register(self, check: CheckProtocol[TContext]) -> None: ...
    def register_all(self, checks: Sequence[CheckProtocol[TContext]]) -> None: ...
    def run(self, ctx: TContext, *, check_filter=None) -> ValidationReport[TFinding]: ...
```

---

### Phase 6: Context Protocol Unification

**Problem:** Multiple context types existed across modules without a common protocol, making it difficult to write code that works with any context.

**Solution:** Created `core/context/protocol.py`:

```python
# core/context/protocol.py

@runtime_checkable
class SnapshotContextProtocol(Protocol):
    """Protocol for contexts providing snapshot information."""
    @property
    def snapshot(self) -> SnapshotRef: ...
    @property
    def repo(self) -> str: ...
    @property
    def commit(self) -> str: ...
    @property
    def repo_root(self) -> Path: ...

@runtime_checkable
class StorageContextProtocol(Protocol):
    """Protocol for contexts providing storage gateway access."""
    @property
    def gateway(self) -> StorageGateway: ...

@runtime_checkable
class ConfigContextProtocol(Protocol):
    """Protocol for contexts providing configuration access."""
    def get_config(self, config_type: type[T]) -> T: ...
    def get_optional_config(self, config_type: type[T]) -> T | None: ...
    def has_config(self, config_type: type[T]) -> bool: ...

@runtime_checkable
class ResourceContextProtocol(Protocol):
    """Protocol for contexts providing resource registry access."""
    def require(self, resource_type: type[T]) -> T: ...
    def require_or_none(self, resource_type: type[T]) -> T | None: ...
    def has_resource(self, resource_type: type) -> bool: ...

@runtime_checkable
class ExecutionContextProtocol(
    SnapshotContextProtocol,
    StorageContextProtocol,
    ConfigContextProtocol,
    ResourceContextProtocol,
    Protocol,
):
    """Complete protocol combining all context capabilities."""
    @property
    def run_id(self) -> str | None: ...
```

---

### Phase 7: Safe Query Utilities Consolidation

**Problem:** Safe database query utilities in `ingestion/infrastructure/db_queries.py` were generally useful but located in an ingestion-specific location.

**Solution:** Moved to `storage/queries/safe.py`:

```python
# storage/queries/safe.py

DUCKDB_QUERY_ERRORS: tuple[type[BaseException], ...] = (...)

class QueryError(Exception): ...
class TableNotFoundError(QueryError): ...
class ColumnNotFoundError(QueryError): ...

@dataclass(frozen=True)
class ForeignKeyRef:
    """Foreign key reference specification for orphan counting."""
    source_table: str
    source_column: str
    ref_table: str
    ref_column: str
    allow_null: bool = True

def safe_count(gateway, table_key) -> int | None: ...
def safe_count_with_scope(gateway, table_key, snapshot) -> int | None: ...
def safe_table_exists(gateway, table_key) -> bool: ...
def safe_get_columns(gateway, table_key) -> set[str]: ...
def safe_count_nulls(gateway, table_key, column) -> int: ...
def safe_min_value(gateway, table_key, column) -> float | None: ...
def safe_max_value(gateway, table_key, column) -> float | None: ...
def safe_count_non_positive(gateway, table_key, column) -> int: ...
def safe_count_duplicates(gateway, table_key, column) -> int: ...
def safe_not_null_fraction(gateway, table_key, column) -> float: ...
def safe_count_orphan_refs(gateway, fk: ForeignKeyRef) -> int: ...
```

**Backward Compatibility:**
- `ingestion/infrastructure/db_queries.py` re-exports everything from `storage/queries/safe.py`

---

## Import Patterns

### Recommended (New Code)

```python
# Storage types
from codeintel.core.ports import QueryResult, BatchResult, StoragePort

# Plugin types
from codeintel.core.plugins.types import AsyncPluginProtocol, PluginMetadata

# Graph resources
from codeintel.core.resources import GraphBundle, GraphProviderProtocol

# Centrality compute
from codeintel.core.compute import compute_pagerank, CentralityMetrics

# Validation
from codeintel.core.validation import ValidationRunner, CheckProtocol

# Context protocols
from codeintel.core.context import ExecutionContextProtocol

# Safe queries
from codeintel.storage.queries import safe_count, safe_table_exists
```

### Legacy (Still Supported)

```python
# These still work via re-exports
from codeintel.graphs.ports.storage import QueryResult, BatchResult
from codeintel.ingestion.ports.storage import BatchResult, IngestStoragePort
from codeintel.graphs.compute.metrics.centrality import compute_pagerank
from codeintel.ingestion.infrastructure.db_queries import safe_count
```

---

## Testing

All changes pass:
- `uv run ruff check` - No lint errors
- `uv run pyright` - No type errors
- Import verification - All new and legacy imports work

---

## Migration Notes

### For New Code
- Import from `core` modules for new code
- Use protocol types for type hints when accepting contexts

### For Existing Code
- No changes required - re-exports maintain compatibility
- Consider migrating to core imports during refactoring

### For Tests
- Test fixtures can use core types directly
- Existing tests continue to work unchanged

---

## Related Documents

- [CROSS_MODULE_CONSOLIDATION_OPPORTUNITIES.md](./CROSS_MODULE_CONSOLIDATION_OPPORTUNITIES.md) - Future consolidation opportunities
- [AGENTS.md](/AGENTS.md) - Agent operating protocol with code standards

---

*Document generated from Phase 1 implementation work*
