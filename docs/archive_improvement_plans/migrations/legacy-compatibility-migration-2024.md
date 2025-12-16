# Legacy Compatibility Code Migration Summary

This document summarizes the comprehensive migration away from legacy compatibility code patterns to the go-forward architecture. The migration was completed in two major phases.

---

## Phase 1: Serving Layer and Core Compatibility Shims

### 1.1 Serving Layer Shims Removed

#### Query Service Shim (`serving/mcp/query_service.py`)
- **Removed**: Backward-compat shim for DuckDB query backend
- **Action**: Deleted shim file, updated 6 consumer files to import from canonical locations

#### Factory Shim (`serving/services/factory.py`)
- **Removed**: Backward-compatibility shim re-exporting from bootstrap/wiring
- **Action**: Deleted shim file, updated 7 consumer files

#### Wiring Shim (`serving/services/wiring.py`)
- **Removed**: Backward-compatibility shim mirroring canonical wiring
- **Action**: Moved `BackendResource` to `bootstrap.py`, deleted wiring.py

#### `OperationSpec` Alias (`serving/registry.py`)
- **Removed**: Legacy `OperationSpec` alias for `Operation`
- **Action**: Replaced with `Operation` in 22 files

#### `ClampResult` Legacy Pagination (`serving/backend/pagination.py`)
- **Removed**: Legacy `ClampResult` and `clamp_offset_value` functions
- **Action**: Replaced with `LimitClamp` and `OffsetClamp` modern types

#### `OperationContract` Wrapper (`serving/backend/operations.py`)
- **Removed**: Compatibility wrapper around `Operation`
- **Action**: Consumers now use `Operation` directly

#### Dict-style Access in MCP Models (`serving/mcp/models.py`)
- **Removed**: `__getitem__` and `get` methods from `MappingModel`/`ViewRow`
- **Action**: Consumers use proper Pydantic model attribute access

#### `problem_detail` Alias (`serving/services/errors.py`)
- **Removed**: `problem_detail` property alias for `detail`
- **Action**: Consumers use `detail` property directly

### 1.2 Analytics Layer Shims Removed

#### Function Re-exports (`analytics/functions/__init__.py`)
- **Removed**: Backward-compatible re-exports of analytics functions
- **Action**: Simplified module, updated 11 consumers to import from canonical locations

#### `graph_runtime` Alias (`analytics/resources/graphs.py`)
- **Removed**: Runtime alias method for compatibility
- **Action**: Consumers use direct `GraphProvider` methods

#### `DEFAULT_ANALYTICS_PLUGINS` Constant (`analytics/core/plugins/registration.py`)
- **Removed**: Backward-compatible constant for plugin names
- **Action**: Consumers use the registry directly

### 1.3 Pipeline Layer Shims Removed

#### Pipeline Step Exports (`pipeline/orchestration/steps.py`)
- **Removed**: `PIPELINE_STEPS`, `PIPELINE_DEPS`, `PIPELINE_SEQUENCE` backward-compatible exports
- **Action**: Consumers use registry-based step discovery

#### `DEFAULT_VALIDATION_SCHEMAS` (`pipeline/export/__init__.py`)
- **Removed**: Backward-compatible constant
- **Action**: Consumers use `default_validation_schemas()` function

### 1.4 Ingestion Layer Shims Removed

#### `ConfigNotFoundError` as `KeyError` (`ingestion/core/execution_context.py`)
- **Removed**: Re-raising `ConfigNotFoundError` as `KeyError` for backward compatibility
- **Action**: Consumers handle `ConfigNotFoundError` directly

---

## Phase 2: Best-in-Class Architecture Migration

### 2.1 Graph Plugin Runner Service (Phase 1)

**Problem**: `build_call_graph()` was a legacy convenience function that obscured plugin orchestration.

**Solution**: Created `GraphPluginRunner` service providing proper abstraction for pipeline steps.

#### New File: `src/codeintel/graphs/plugins/runner.py`

```python
@dataclass
class GraphPluginRunner:
    gateway: StorageGateway
    scratch: GraphRuntimeScratch | None = None

    def build_context(
        self,
        snapshot: SnapshotRef,
        paths: BuildPaths | None = None,
        catalog_provider: FunctionCatalogProvider | None = None,
        ...
    ) -> GraphExecutionContext:
        """Build a GraphExecutionContext with proper resource registration."""
        container = ResourceContainer()
        container.register(StorageResource(self.gateway, snapshot.repo_root))
        if catalog_provider is not None:
            container.register(CatalogResource(catalog_provider))
        return GraphExecutionContext(...)

    def run_plugin(
        self,
        plugin: GraphPluginProtocol,
        ctx: GraphExecutionContext,
        *,
        raise_on_failure: bool = True,
    ) -> GraphPluginResult:
        """Execute a graph plugin with proper error handling."""
        ...
```

#### Files Updated
- `src/codeintel/pipeline/orchestration/steps_graphs.py` - `CallGraphStep` now uses `GraphPluginRunner`
- `tests/_helpers/fixtures.py` - Test helper updated
- `tests/orchestration/test_pipeline_catalog_entrypoint.py` - Integration test updated
- `tests/graphs/test_span_consistency_integration.py` - Integration test updated
- `tests/test_callgraph_alias_resolution.py` - Unit test updated

#### Files Removed
- Removed `build_call_graph()` function from `src/codeintel/graphs/plugins/builders/callgraph.py` (lines 381-419)

### 2.2 Remove `ctx.engine` Fallback (Phase 2)

**Problem**: Graph metrics plugins had a fallback pattern that checked for `GraphResource` first, then fell back to `ctx.engine`. This created implicit coupling and made the code harder to understand.

**Solution**: Added `require_graphs()` helper method and removed the fallback pattern.

#### Changes to `src/codeintel/graphs/core/context.py`

Added:

```python
def require_graphs(self) -> GraphResource:
    """Get graph resource, raising if unavailable."""
    if not self.resources.has(GraphResource.RESOURCE_NAME):
        raise RuntimeError("No GraphResource registered in context")
    return self.require(GraphResource)
```

Removed:
- `_engine: GraphEngine | None = field(default=None, repr=False)`
- `engine` property (lines 250-265)
- `GraphEngine` from TYPE_CHECKING imports

#### Files Updated
- `src/codeintel/graphs/plugins/metrics/core.py` - Replaced fallback pattern with `require_graphs()`
- `src/codeintel/graphs/plugins/metrics/secondary.py` - Updated docstring
- `src/codeintel/graphs/recipes/executor.py` - Removed `_engine` parameter from context creation
- `src/codeintel/graphs/runtime/executor.py` - Removed `_engine` parameter from context creation

### 2.3 Remove Legacy kwargs from SCIP Resolver (Phase 3)

**Problem**: `resolve_scip_inputs()` accepted `**legacy_kwargs` for backward compatibility, making the API unclear and type-unsafe.

**Solution**: Added typed `ScipResolverInput.build()` factory method and removed legacy kwargs support.

#### Changes to `src/codeintel/ingestion/infrastructure_utilities/_scip_resolver.py`

Added:

```python
@classmethod
def build(
    cls,
    *,
    repo: str | None = None,
    commit: str | None = None,
    repo_root: Path | str | None = None,
    build_dir: Path | str | None = None,
    document_output_dir: Path | str | None = None,
    scip_python_bin: str | None = None,
    scip_bin: str | None = None,
    modules: Sequence[ModuleRecord] | None = None,
    cfg: object | None = None,
) -> ScipResolverInput:
    """Construct input with automatic path coercion."""
    ...
```

Removed:
- `_build_inputs_from_legacy()` function
- Helper cast functions (`_cast_optional_str`, `_cast_optional_path`, etc.)
- `**legacy_kwargs` from `resolve_scip_inputs()` signature

#### Files Updated
- `tests/ingestion/test_scip_resolver.py` - All 10 test calls updated to use `ScipResolverInput.build()`

### 2.4 Remove `__getattr__` from Contracts (Phase 4)

**Problem**: `contracts.py` used `__getattr__` to expose module-level constants (`TABLE_SCHEMAS`, `DATASET_CONTRACTS`, etc.) for backward compatibility. This pattern is implicit and makes code harder to analyze.

**Solution**: Migrated all consumers to use explicit accessor functions, then removed `__getattr__`.

#### Accessor Functions (Already Existed)
- `get_table_schemas()` → Returns `dict[str, TableSchema]`
- `get_composite_schemas()` → Returns `dict[str, CompositeSchema]`
- `get_dataset_contracts()` → Returns `dict[str, DatasetContract]`
- `get_dataset_contracts_by_table_key()` → Returns `dict[str, DatasetContract]`
- `get_row_bindings()` → Returns `dict[str, RowBinding]`

#### Source Files Updated (23+ files)
- `src/codeintel/storage/datasets.py`
- `src/codeintel/storage/normalized_macros.py`
- `src/codeintel/storage/ingest_helpers.py`
- `src/codeintel/storage/schema_generation.py`
- `src/codeintel/storage/schemas.py`
- `src/codeintel/serving/backend/datasets.py`

#### Test Files Updated (20+ files)
- `tests/serving/test_operation_spec_alignment.py`
- `tests/serving/test_operation_catalog_alignment.py`
- `tests/storage/test_dataset_catalog.py`
- `tests/analytics/test_writer_guard.py`
- `tests/analytics/test_function_history.py`
- `tests/analytics/test_analytics_rows_contracts.py`
- `tests/_helpers/history.py`
- `tests/storage/test_macro_registry.py`
- `tests/storage/test_normalized_macros_helper.py`
- `tests/storage/test_macro_schemas.py`
- `tests/storage/test_dataflow_graph.py`
- `tests/storage/test_schema_roundtrip.py`
- `tests/docs_export/test_export_edge_columns.py`
- `tests/config/test_dataset_contract.py`
- `tests/config/test_datasets_contracts.py`
- `tests/config/test_composite_schemas.py`
- `tests/config/test_dataset_contract_snapshot.py`

#### Files Removed
- Removed `__getattr__` function from `src/codeintel/config/datasets/contracts.py` (lines 1063-1093)

---

## Migration Patterns Summary

### Pattern 1: Shim Removal
Files that existed solely for backward-compatible re-exports were deleted, and consumers were updated to import from canonical locations.

### Pattern 2: Alias Removal
Type aliases and property aliases that maintained old naming conventions were removed, with direct usage of canonical names.

### Pattern 3: Service Extraction
Convenience functions that obscured proper architecture were replaced with explicit service classes (`GraphPluginRunner`).

### Pattern 4: Fallback Elimination
Fallback patterns (like `ctx.engine` checks) were replaced with explicit requirements (`require_graphs()`).

### Pattern 5: Type-Safe APIs
Loosely-typed APIs (`**kwargs`) were replaced with typed dataclasses and factory methods.

### Pattern 6: Explicit over Implicit
Magic methods like `__getattr__` for module attributes were replaced with explicit accessor functions.

---

## Verification

All legacy patterns confirmed removed:

```bash
# No build_call_graph() calls (only internal _build_call_graph implementation)
grep -r "build_call_graph\(" src/ --include="*.py" # No matches

# No ctx.engine fallback patterns
grep -r "ctx\.engine" src/codeintel/graphs/plugins/metrics/ # No matches

# No **legacy_kwargs
grep -r "\*\*legacy_kwargs" src/ # No matches

# No __getattr__ in contracts.py
grep -r "__getattr__" src/codeintel/config/datasets/contracts.py # No matches
```

---

## Benefits of Migration

1. **Type Safety**: All APIs now have explicit type annotations
2. **Discoverability**: No magic methods; all functionality discoverable via explicit imports
3. **Testability**: Service classes can be easily mocked/stubbed
4. **Maintainability**: Clear ownership and responsibility for each component
5. **IDE Support**: Better autocomplete and static analysis support
6. **Documentation**: Self-documenting APIs with clear signatures

