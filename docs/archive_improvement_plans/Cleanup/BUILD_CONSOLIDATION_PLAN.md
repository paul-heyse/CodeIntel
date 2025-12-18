# Build System Consolidation Plan

> **Status**: Ready for Implementation  
> **Created**: 2025-12-13  
> **Scope**: Consolidating duplicated functionality, streamlining patterns, and enhancing maintainability  
> **Prerequisite**: Completion of BUILD_CLEANUP_PLAN.md Phase 1-5 (dead code removal)

---

## Executive Summary

This plan addresses consolidation opportunities in the build system to:

1. **Eliminate code duplication** - 16+ files duplicate the same helper functions
2. **Unify registries** - Multiple registries for targets, plugins, and native modules
3. **Standardize patterns** - Native targets and plugins follow inconsistent patterns
4. **Reduce boilerplate** - Plugin classes repeat the same structural code
5. **Improve caching** - Centralize session-scoped operations

### Impact Assessment

| Priority | Items | Files Affected | Risk | Effort |
|----------|-------|----------------|------|--------|
| High     | 4     | 25+ files      | Low-Medium | 3-4 days |
| Medium   | 4     | 15+ files      | Low | 2-3 days |
| Low      | 3     | 5-10 files     | Low | 1 day |

### Dependencies

```
Phase 1.1 (Metadata Helper) ─────────────────────────────────┐
Phase 1.2 (Native Registry Integration) ─────────────────────┼──> Phase 2.1 (Complete Registrations)
Phase 1.3 (Plugin Base Enhancement) ─────────────────────────┘
Phase 1.4 (Runner Record Factory) ────────────────────────────────> Phase 2.2 (Native Target Pattern)
Phase 2.3 (Snapshot Filter Helper) ──────────────────────────────> Phase 2.2 (Native Target Pattern)
Phase 2.4 (MaterializationContext Merge) ────────────────────────> Phase 3.1 (Context Finalization)
```

---

## Phase 1: High-Impact Consolidations

### 1.1 Plugin Metadata Conversion Pattern

**Problem**: The `_to_plugin_metadata()` function is duplicated across 16 plugin files, violating DRY principles.

**Files with duplication**:
- `src/codeintel/build/plugins/ingestion/repo_scan.py`
- `src/codeintel/build/plugins/ingestion/ast_extract.py`
- `src/codeintel/build/plugins/ingestion/cst_extract.py`
- `src/codeintel/build/plugins/ingestion/scip_plugin.py`
- `src/codeintel/build/plugins/ingestion/typing_plugin.py`
- `src/codeintel/build/plugins/ingestion/coverage_plugin.py`
- `src/codeintel/build/plugins/ingestion/tests_plugin.py`
- `src/codeintel/build/plugins/ingestion/docstrings_plugin.py`
- `src/codeintel/build/plugins/ingestion/config_plugin.py`
- `src/codeintel/build/plugins/analytics/functions/metrics.py`
- `src/codeintel/build/plugins/analytics/types/coverage.py`
- `src/codeintel/build/plugins/graphs/builders/goid.py`
- `src/codeintel/build/plugins/graphs/builders/callgraph.py`
- `src/codeintel/build/plugins/graphs/builders/import_graph.py`
- `src/codeintel/build/plugins/graphs/builders/cfg_dfg.py`
- `src/codeintel/build/plugins/graphs/builders/symbol_uses.py`

**Existing shared helper** (already exists):
```python
# src/codeintel/build/plugins/analytics/_metadata.py
def to_plugin_metadata(core: CorePluginMetadata) -> PluginMetadata:
    """Convert CorePluginMetadata to the protocol-friendly PluginMetadata."""
    return PluginMetadata(
        name=core.name,
        version=core.version,
        description=core.description,
        kind=cast("PluginKind", core.kind),
        stage=cast("PluginStage", core.stage or "other"),
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
    )
```

**Implementation Steps**:

1. **Relocate shared helper to build-level module**:
   - Move `to_plugin_metadata` from `plugins/analytics/_metadata.py` to `plugins/_metadata.py`
   - This makes it accessible to all plugin domains (ingestion, graphs, analytics)

2. **Update all 16 plugin files**:
   - Remove local `_to_plugin_metadata()` function definition
   - Add import: `from codeintel.build.plugins._metadata import to_plugin_metadata`
   - Update `metadata` property to use `to_plugin_metadata(self._core_metadata)`

3. **Deprecate old location**:
   - Keep `plugins/analytics/_metadata.py` as re-export for backward compatibility
   - Add deprecation comment

**Validation**:
```bash
# Ensure no local _to_plugin_metadata definitions remain
grep -r "def _to_plugin_metadata" src/codeintel/build/plugins/

# Run quality checks
uv run python -m tools.quality_report
uv run pytest -q tests/build/
```

**Risk**: Low - Pure refactoring, no logic changes

---

### 1.2 Native Registry Integration with UnifiedRegistry

**Problem**: Two separate registries exist for tracking implementations:
- `hamilton/native/registry.py` - `NATIVE_TARGETS` tuple
- `unified_registry.py` - `UnifiedRegistry` class

This creates potential for inconsistencies and requires updates in multiple places.

**Current state**:
```python
# hamilton/native/registry.py
NATIVE_TARGETS: Final[tuple[NativeTargetSpec, ...]] = (
    NativeTargetSpec(
        target_name="risk_factors",
        module_path="codeintel.build.hamilton.native.analytics.risk_factors",
    ),
    # ... 8 more entries
)

# unified_registry.py
@dataclass(frozen=True)
class TargetRegistration:
    target: OutputTarget
    plugin_class: type[TargetPlugin] | None = None
    native_module: str | None = None  # Already has native_module field!
```

**Implementation Steps**:

1. **Update `register_*_targets()` functions in `registrations.py`**:
   - For each native target, use `registry.register()` with `native_module` parameter:
   ```python
   registry.register(
       RISK_FACTORS_TARGET,
       native_module="codeintel.build.hamilton.native.analytics.risk_factors",
   )
   ```

2. **Add native module query methods to `UnifiedRegistry`**:
   ```python
   def native_target_names(self) -> frozenset[str]:
       """Return names of targets with native implementations."""
       return frozenset(
           name for name, reg in self._registrations.items()
           if reg.native_module is not None
       )
   
   def is_native_target(self, name: str) -> bool:
       """Check if a target has a native implementation."""
       reg = self._registrations.get(name)
       return reg is not None and reg.native_module is not None
   ```

3. **Create adapter functions in `hamilton/native/registry.py`**:
   ```python
   def native_target_names() -> frozenset[str]:
       """Return target names with native implementations (delegates to UnifiedRegistry)."""
       from codeintel.build.unified_registry import get_unified_registry
       return get_unified_registry().native_target_names()
   
   def is_native_target(target_name: str) -> bool:
       """Check if target has native implementation (delegates to UnifiedRegistry)."""
       from codeintel.build.unified_registry import get_unified_registry
       return get_unified_registry().is_native_target(target_name)
   ```

4. **Deprecate `NATIVE_TARGETS` tuple**:
   - Add deprecation comment
   - Keep for backward compatibility during transition
   - Remove in future cleanup pass

5. **Update `load_native_modules()` to use registry**:
   ```python
   def load_native_modules() -> tuple[ModuleType, ...]:
       """Load all native target modules from the unified registry."""
       registry = get_unified_registry()
       modules: list[ModuleType] = []
       for reg in registry.get_all_registrations():
           if reg.native_module:
               module = importlib.import_module(reg.native_module)
               modules.append(module)
       return tuple(modules)
   ```

**Validation**:
```bash
# Verify native targets are correctly registered
uv run python -c "
from codeintel.build.unified_registry import get_unified_registry
reg = get_unified_registry()
natives = [n for n in reg if reg.get_native_module(n)]
print(f'Native targets: {natives}')
assert 'risk_factors' in natives
"

# Run Hamilton tests
uv run pytest -q tests/build/hamilton/
```

**Risk**: Medium - Changes core registry behavior, needs careful testing

---

### 1.3 Plugin Base Class Enhancement

**Problem**: Every plugin class repeats the same boilerplate:
- `_core_metadata: ClassVar[CorePluginMetadata]` definition
- `metadata` property implementation
- Options resolver handling

**Current pattern** (repeated 40+ times):
```python
class SomePlugin(TargetPlugin):
    plugin_name: ClassVar[str] = "some_plugin"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "..."
    _core_metadata: ClassVar[CorePluginMetadata] = SOME_METADATA
    
    def __init__(self, *, options_resolver: PluginOptionsResolver | None = None) -> None:
        self._options_resolver = options_resolver
    
    @property
    def metadata(self) -> PluginMetadata:
        return _to_plugin_metadata(self._core_metadata)
```

**Implementation Steps**:

1. **Create `MetadataPlugin` base class in `plugin.py`**:
   ```python
   class MetadataPlugin(TargetPlugin):
       """Enhanced plugin base with automatic metadata handling.
       
       Subclasses define `_core_metadata` and get `metadata` property
       automatically, plus standard options resolver handling.
       """
       
       _core_metadata: ClassVar[CorePluginMetadata]
       
       def __init__(
           self,
           *,
           options_resolver: PluginOptionsResolver | None = None,
       ) -> None:
           """Initialize with optional options resolver."""
           self._options_resolver = options_resolver
       
       @property
       def metadata(self) -> PluginMetadata:
           """Return protocol-compatible metadata."""
           from codeintel.build.plugins._metadata import to_plugin_metadata
           return to_plugin_metadata(self._core_metadata)
       
       @property
       def options_resolver(self) -> PluginOptionsResolver | None:
           """Return the options resolver if configured."""
           return self._options_resolver
   ```

2. **Add derived properties from `_core_metadata`**:
   ```python
       @property
       def plugin_name(self) -> str:
           """Return plugin name from core metadata."""
           return self._core_metadata.name
       
       @property
       def plugin_version(self) -> str:
           """Return plugin version from core metadata."""
           return self._core_metadata.version
       
       @property
       def plugin_description(self) -> str:
           """Return plugin description from core metadata."""
           return self._core_metadata.description
   ```

3. **Migrate plugins incrementally** (optional, can be done gradually):
   - Start with new plugins using `MetadataPlugin`
   - Existing plugins continue to work with `TargetPlugin`
   - Migration is backward compatible

**Example migrated plugin**:
```python
class RepoScanPlugin(MetadataPlugin):
    """Scan repository modules and build change-tracker state."""
    
    _core_metadata: ClassVar[CorePluginMetadata] = REPO_SCAN_METADATA
    
    # No __init__, metadata property, or ClassVars needed!
    
    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        ...
```

**Validation**:
```bash
# Verify base class works
uv run python -c "
from codeintel.build.plugin import MetadataPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

class TestPlugin(MetadataPlugin):
    _core_metadata = CorePluginMetadata(
        name='test', version='1.0', description='Test',
        domain=PluginDomain.ANALYTICS, kind='test'
    )
    async def execute(self, ctx): pass

p = TestPlugin()
print(f'Name: {p.plugin_name}')
print(f'Metadata: {p.metadata}')
"
```

**Risk**: Low - Additive change, existing code unaffected

---

### 1.4 Runner Record Factory Consolidation

**Problem**: Three separate functions share overlapping logic:
- `create_success_record()`
- `create_skipped_record()`
- `create_failed_record()`

**Current state in `hamilton/native/runner.py`**:
```python
def create_success_record(target, env, run) -> TargetRunRecord:
    datasets = expected_datasets(target, env.snapshot)
    artifacts = expected_artifacts(target, env.snapshot, path_formatter={...})
    return TargetRunRecord(
        target=target.name,
        plugin_name=f"native:{target.name}",
        status="succeeded",
        ...
    )

def create_skipped_record(target, env, run) -> TargetRunRecord:
    datasets = expected_datasets(target, env.snapshot)  # Duplicated
    artifacts = expected_artifacts(target, env.snapshot, path_formatter={...})  # Duplicated
    return TargetRunRecord(
        target=target.name,
        plugin_name=f"native:{target.name}",  # Duplicated
        status="skipped",
        ...
    )

def create_failed_record(target, input_hash, options_hash, duration_ms, error) -> TargetRunRecord:
    return TargetRunRecord(
        target=target.name,
        plugin_name=f"native:{target.name}",  # Duplicated
        status="failed",
        ...
    )
```

**Implementation Steps**:

1. **Create unified factory function**:
   ```python
   def create_run_record(
       target: OutputTarget,
       status: Literal["succeeded", "skipped", "failed"],
       input_hash: str,
       *,
       env: BuildEnv | None = None,
       run: NativeRunInfo | None = None,
       error: Exception | None = None,
   ) -> TargetRunRecord:
       """Create a TargetRunRecord for any completion status.
       
       Parameters
       ----------
       target
           Target that was executed.
       status
           Completion status: succeeded, skipped, or failed.
       input_hash
           Input hash for this execution.
       env
           Build environment (required for succeeded/skipped).
       run
           Run metadata (required for succeeded/skipped).
       error
           Exception that caused failure (required for failed).
       
       Returns
       -------
       TargetRunRecord
           Record with appropriate datasets/artifacts based on status.
       """
       plugin_name = f"native:{target.name}"
       
       if status == "failed":
           return TargetRunRecord(
               target=target.name,
               plugin_name=plugin_name,
               status="failed",
               input_hash=input_hash,
               options_hash=run.options_hash if run else None,
               duration_ms=run.duration_ms if run else 0.0,
               row_counts={},
               error=str(error) if error else None,
               datasets=(),
               artifacts=(),
           )
       
       if env is None or run is None:
           msg = f"env and run required for status '{status}'"
           raise ValueError(msg)
       
       # Generate expected refs from contract
       datasets = expected_datasets(target, env.snapshot)
       artifacts = expected_artifacts(
           target,
           env.snapshot,
           path_formatter={
               "build_dir": str(env.paths.build_dir),
               "scip_dir": str(env.paths.scip_dir),
               "export_dir": str(env.paths.document_output_dir),
               "repo_root": str(env.snapshot.repo_root),
           },
       )
       
       # Update row counts for success
       if status == "succeeded" and run.row_counts:
           from codeintel.build.hamilton.io.dataset_ref import DatasetRef
           updated_datasets = tuple(
               DatasetRef(
                   table_key=ds.table_key,
                   repo=ds.repo,
                   commit=ds.commit,
                   row_count=run.row_counts.get(ds.table_key, ds.row_count),
               )
               for ds in datasets
           )
           datasets = updated_datasets
       
       return TargetRunRecord(
           target=target.name,
           plugin_name=plugin_name,
           status=status,
           input_hash=input_hash,
           options_hash=run.options_hash,
           duration_ms=run.duration_ms,
           row_counts=run.row_counts or {},
           error=None,
           datasets=datasets,
           artifacts=artifacts,
       )
   ```

2. **Deprecate old functions with forwarding**:
   ```python
   def create_success_record(target, env, run) -> TargetRunRecord:
       """Create successful record. Deprecated: use create_run_record."""
       return create_run_record(target, "succeeded", run.input_hash, env=env, run=run)
   
   def create_skipped_record(target, env, run) -> TargetRunRecord:
       """Create skipped record. Deprecated: use create_run_record."""
       return create_run_record(target, "skipped", run.input_hash, env=env, run=run)
   
   def create_failed_record(target, input_hash, options_hash, duration_ms, error) -> TargetRunRecord:
       """Create failed record. Deprecated: use create_run_record."""
       run = NativeRunInfo(input_hash=input_hash, options_hash=options_hash, duration_ms=duration_ms)
       return create_run_record(target, "failed", input_hash, run=run, error=error)
   ```

3. **Update native modules to use unified factory** (optional):
   - Can be done incrementally as modules are touched

**Validation**:
```bash
uv run pytest -q tests/build/hamilton/
uv run python -c "
from codeintel.build.hamilton.native.runner import create_run_record
# Basic smoke test
"
```

**Risk**: Low - Old functions continue to work via forwarding

---

## Phase 2: Medium-Impact Consolidations

### 2.1 Complete Plugin Registrations in UnifiedRegistry

**Problem**: Current registrations use `register_target_only()` without plugin classes:
```python
registry.register_target_only(MODULES_TARGET)
registry.register_target_only(AST_TARGET)
```

This defeats the purpose of atomic registration.

**Implementation Steps**:

1. **Map targets to their plugin classes**:
   Create a mapping table:
   
   | Target | Plugin Class | Module |
   |--------|-------------|--------|
   | `MODULES_TARGET` | `RepoScanPlugin` | `plugins.ingestion.repo_scan` |
   | `AST_TARGET` | `AstExtractPlugin` | `plugins.ingestion.ast_extract` |
   | `CST_TARGET` | `CstExtractPlugin` | `plugins.ingestion.cst_extract` |
   | ... | ... | ... |

2. **Update `register_ingestion_targets()`**:
   ```python
   def register_ingestion_targets(registry: UnifiedRegistry) -> None:
       from codeintel.build.plugins.ingestion import (
           RepoScanPlugin,
           AstExtractPlugin,
           CstExtractPlugin,
           ScipPlugin,
           TypingPlugin,
           CoveragePlugin,
           TestsPlugin,
           DocstringsPlugin,
           ConfigPlugin,
       )
       
       registry.register(MODULES_TARGET, plugin=RepoScanPlugin)
       registry.register(AST_TARGET, plugin=AstExtractPlugin)
       registry.register(CST_TARGET, plugin=CstExtractPlugin)
       registry.register(SCIP_TARGET, plugin=ScipPlugin)
       registry.register(TYPING_TARGET, plugin=TypingPlugin)
       # ... etc
   ```

3. **Update `register_graph_targets()`**:
   ```python
   def register_graph_targets(registry: UnifiedRegistry) -> None:
       from codeintel.build.plugins.graphs.builders import (
           GoidPlugin,
           CallgraphPlugin,
           ImportGraphPlugin,
           CfgDfgPlugin,
           SymbolUsesPlugin,
       )
       
       registry.register(GOIDS_TARGET, plugin=GoidPlugin)
       registry.register(CALL_GRAPH_TARGET, plugin=CallgraphPlugin)
       # ... etc
   ```

4. **Update `register_analytics_targets()`** with native modules:
   ```python
   def register_analytics_targets(registry: UnifiedRegistry) -> None:
       # Plugin-based targets
       registry.register(FUNCTION_METRICS_TARGET, plugin=FunctionMetricsPlugin)
       
       # Native targets
       registry.register(
           RISK_FACTORS_TARGET,
           native_module="codeintel.build.hamilton.native.analytics.risk_factors",
       )
       registry.register(
           HOTSPOTS_TARGET,
           native_module="codeintel.build.hamilton.native.analytics.hotspots",
       )
       # ... etc
   ```

5. **Add validation to ensure completeness**:
   ```python
   def register_all_targets(registry: UnifiedRegistry) -> None:
       register_ingestion_targets(registry)
       register_graph_targets(registry)
       register_analytics_targets(registry)
       register_export_targets(registry)
       
       # Validate all targets have implementations
       errors = registry.validate()
       if errors:
           msg = f"Registry validation failed: {errors}"
           raise RuntimeError(msg)
   ```

**Validation**:
```bash
uv run python -c "
from codeintel.build.unified_registry import get_unified_registry
reg = get_unified_registry()
print(f'Total targets: {len(reg)}')
with_plugin = sum(1 for r in reg.get_all_registrations() if r.plugin_class)
with_native = sum(1 for r in reg.get_all_registrations() if r.native_module)
print(f'With plugin: {with_plugin}')
print(f'With native: {with_native}')
"
```

**Risk**: Medium - Changes plugin loading path

---

### 2.2 Native Target Pattern Consolidation

**Problem**: All 9 native Hamilton modules follow nearly identical patterns with substantial boilerplate.

**Current pattern** (repeated in each module):
```python
import time
from hamilton.function_modifiers import tag

from codeintel.build.hamilton.native.runner import (
    NativeRunInfo,
    create_failed_record,
    create_skipped_record,
    create_success_record,
    save_manifest,
    should_skip_native_target,
)

@tag(domain="analytics", target="risk_factors", node_type="compute")
def t__risk_factors__compute(...) -> ir.Table:
    """Pure compute node."""
    ...

def t__risk_factors(
    env: BuildEnv,
    graph: TargetGraph,
    t__risk_factors__compute: ir.Table,
) -> TargetRunRecord:
    """Materialize node with skip logic."""
    target = graph.get("risk_factors")
    input_hash = compute_input_hash(...)
    
    if should_skip_native_target(env, target, input_hash):
        return create_skipped_record(target, env, NativeRunInfo(...))
    
    start = time.perf_counter()
    try:
        ctx = MaterializationContext(...)
        ref = materialize_table(ctx, "analytics.table", t__risk_factors__compute)
        duration_ms = (time.perf_counter() - start) * 1000
        
        record = create_success_record(target, env, NativeRunInfo(..., row_counts={...}))
        save_manifest(env, record)
        return record
    except Exception as e:
        return create_failed_record(target, input_hash, None, 0.0, e)
```

**Implementation Steps**:

1. **Create `NativeTargetExecutor` class**:
   ```python
   # hamilton/native/executor.py
   
   @dataclass
   class NativeTargetExecutor:
       """Handles skip-check, timing, and record creation for native targets."""
       
       env: BuildEnv
       target: OutputTarget
       input_hash: str
       options_hash: str | None = None
       
       @classmethod
       def for_target(
           cls,
           env: BuildEnv,
           graph: TargetGraph,
           target_name: str,
       ) -> NativeTargetExecutor:
           """Create executor for a named target."""
           target = graph.get(target_name)
           input_hash = compute_input_hash(target, env.snapshot, env.gateway)
           return cls(env=env, target=target, input_hash=input_hash)
       
       def should_skip(self) -> bool:
           """Check if target can be skipped."""
           return should_skip_native_target(self.env, self.target, self.input_hash)
       
       def skip(self) -> TargetRunRecord:
           """Create skipped record."""
           run = NativeRunInfo(
               input_hash=self.input_hash,
               options_hash=self.options_hash,
               duration_ms=0.0,
           )
           return create_run_record(self.target, "skipped", self.input_hash, env=self.env, run=run)
       
       def execute(
           self,
           compute_fn: Callable[[], dict[str, int]],
       ) -> TargetRunRecord:
           """Execute with timing, error handling, and manifest persistence."""
           start = time.perf_counter()
           try:
               row_counts = compute_fn()
               duration_ms = (time.perf_counter() - start) * 1000
               
               run = NativeRunInfo(
                   input_hash=self.input_hash,
                   options_hash=self.options_hash,
                   duration_ms=duration_ms,
                   row_counts=row_counts,
               )
               record = create_run_record(
                   self.target, "succeeded", self.input_hash, env=self.env, run=run
               )
               save_manifest(self.env, record)
               return record
           except Exception as e:
               duration_ms = (time.perf_counter() - start) * 1000
               run = NativeRunInfo(
                   input_hash=self.input_hash,
                   options_hash=self.options_hash,
                   duration_ms=duration_ms,
               )
               return create_run_record(
                   self.target, "failed", self.input_hash, run=run, error=e
               )
   ```

2. **Simplify native target modules**:
   ```python
   # risk_factors.py (simplified)
   
   @tag(domain="analytics", target="risk_factors", node_type="compute")
   def t__risk_factors__compute(...) -> ir.Table:
       """Pure compute node - unchanged."""
       ...
   
   def t__risk_factors(
       env: BuildEnv,
       graph: TargetGraph,
       t__risk_factors__compute: ir.Table,
   ) -> TargetRunRecord:
       """Materialize with standard executor."""
       executor = NativeTargetExecutor.for_target(env, graph, "risk_factors")
       
       if executor.should_skip():
           return executor.skip()
       
       def compute() -> dict[str, int]:
           ctx = MaterializationContext.from_build_context(...)
           ref = materialize_table(ctx, "analytics.goid_risk_factors", t__risk_factors__compute)
           return {ref.table_key: ref.row_count}
       
       return executor.execute(compute)
   ```

3. **Create helper decorator for even simpler cases** (optional):
   ```python
   def native_target(target_name: str, table_key: str):
       """Decorator for simple native targets with one output table."""
       def decorator(compute_fn):
           def wrapper(env, graph, computed_table):
               executor = NativeTargetExecutor.for_target(env, graph, target_name)
               if executor.should_skip():
                   return executor.skip()
               
               def compute() -> dict[str, int]:
                   ctx = MaterializationContext.from_build_context(...)
                   ref = materialize_table(ctx, table_key, computed_table)
                   return {ref.table_key: ref.row_count}
               
               return executor.execute(compute)
           return wrapper
       return decorator
   ```

**Validation**:
```bash
uv run pytest -q tests/build/hamilton/native/
```

**Risk**: Medium - Core native target logic changes

---

### 2.3 Snapshot Filtering Helper

**Problem**: Multiple native modules repeat identical Ibis snapshot filtering:
```python
modules_filtered = q__core__modules.filter(
    cast("Any", and_predicates(
        q__core__modules.repo == env.snapshot.repo,
        q__core__modules.commit == env.snapshot.commit,
    ))
)
```

**Implementation Steps**:

1. **Create utility module** `hamilton/native/ibis_helpers.py`:
   ```python
   """Ibis helper utilities for native Hamilton targets."""
   
   from __future__ import annotations
   
   from typing import TYPE_CHECKING, Any, cast
   
   from codeintel.core.ibis_typing import and_predicates
   
   if TYPE_CHECKING:
       import ibis.expr.types as ir
       from codeintel.config.primitives import SnapshotRef
   
   
   def filter_for_snapshot(table: ir.Table, snapshot: SnapshotRef) -> ir.Table:
       """Filter an Ibis table to the current snapshot.
       
       Parameters
       ----------
       table
           Ibis table expression with repo and commit columns.
       snapshot
           Snapshot reference providing repo and commit values.
       
       Returns
       -------
       ir.Table
           Filtered table expression.
       
       Examples
       --------
       >>> filtered = filter_for_snapshot(q__core__modules, env.snapshot)
       """
       return table.filter(
           cast("Any", and_predicates(
               table.repo == snapshot.repo,
               table.commit == snapshot.commit,
           ))
       )
   
   
   def filter_tables_for_snapshot(
       snapshot: SnapshotRef,
       **tables: ir.Table,
   ) -> dict[str, ir.Table]:
       """Filter multiple tables to the current snapshot.
       
       Parameters
       ----------
       snapshot
           Snapshot reference.
       **tables
           Named table expressions to filter.
       
       Returns
       -------
       dict[str, ir.Table]
           Mapping of names to filtered table expressions.
       
       Examples
       --------
       >>> filtered = filter_tables_for_snapshot(
       ...     env.snapshot,
       ...     modules=q__core__modules,
       ...     file_state=q__core__file_state,
       ... )
       >>> filtered["modules"]  # Filtered modules table
       """
       return {name: filter_for_snapshot(table, snapshot) for name, table in tables.items()}
   ```

2. **Update native modules to use helper**:
   ```python
   # Before
   modules_filtered = q__core__modules.filter(
       cast("Any", and_predicates(
           q__core__modules.repo == env.snapshot.repo,
           q__core__modules.commit == env.snapshot.commit,
       ))
   )
   
   # After
   from codeintel.build.hamilton.native.ibis_helpers import filter_for_snapshot
   modules_filtered = filter_for_snapshot(q__core__modules, env.snapshot)
   ```

**Validation**:
```bash
# Verify helper works
uv run python -c "
from codeintel.build.hamilton.native.ibis_helpers import filter_for_snapshot
print('Helper imported successfully')
"
```

**Risk**: Low - Pure utility addition

---

### 2.4 MaterializationContext → BuildContext Merge

**Problem**: `MaterializationContext` duplicates fields from `BuildContext`:

```python
# materializer.py
@dataclass(frozen=True)
class MaterializationContext:
    gateway: StorageGateway
    snapshot: SnapshotRef
    validate: bool = False
    owner_target: str | None = None
    input_hash: str | None = None

# context_base.py
@dataclass(frozen=True)
class BuildContext:
    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    session: BuildSession | None = None
```

**Implementation Steps**:

1. **Extend `BuildContext` with materialization options**:
   ```python
   @dataclass(frozen=True)
   class BuildContext:
       gateway: StorageGateway
       snapshot: SnapshotRef
       paths: BuildPaths
       session: BuildSession | None = None
       # Materialization options
       validate_schemas: bool = False
       owner_target: str | None = None
       input_hash: str | None = None
   ```

2. **Update `materialize_table()` to accept `BuildContext`**:
   ```python
   def materialize_table(
       ctx: BuildContext | MaterializationContext,
       table_key: str,
       expr: ir.Table,
   ) -> DatasetRef:
       """Materialize an Ibis expression to a DuckDB table.
       
       Accepts either BuildContext or MaterializationContext for
       backward compatibility.
       """
       # Extract fields from either context type
       gateway = ctx.gateway
       snapshot = ctx.snapshot
       validate = getattr(ctx, "validate_schemas", getattr(ctx, "validate", False))
       owner_target = ctx.owner_target
       input_hash = ctx.input_hash
       
       # ... rest of implementation
   ```

3. **Deprecate `MaterializationContext`**:
   ```python
   @dataclass(frozen=True)
   class MaterializationContext:
       """Deprecated: Use BuildContext instead.
       
       This class is retained for backward compatibility.
       """
       gateway: StorageGateway
       snapshot: SnapshotRef
       validate: bool = False
       owner_target: str | None = None
       input_hash: str | None = None
       
       @classmethod
       def from_build_context(cls, ctx: BuildContext, **kwargs) -> MaterializationContext:
           """Create from BuildContext. Prefer using BuildContext directly."""
           return cls(
               gateway=ctx.gateway,
               snapshot=ctx.snapshot,
               validate=kwargs.get("validate", ctx.validate_schemas),
               owner_target=kwargs.get("owner_target", ctx.owner_target),
               input_hash=kwargs.get("input_hash", ctx.input_hash),
           )
   ```

4. **Update native modules to use `BuildContext`** (incremental):
   - New code uses `BuildContext` directly
   - Existing code continues to work via type union

**Validation**:
```bash
uv run pytest -q tests/build/hamilton/
```

**Risk**: Low - Backward compatible changes

---

## Phase 3: Low-Impact Improvements

### 3.1 Context Hierarchy Finalization

**Problem**: `TargetExecutionContext` and `ExecutionContext` have overlapping functionality:

- `ExecutionContext` in `context_base.py` - base for plugin execution
- `TargetExecutionContext` in `context.py` - actual plugin context with write methods

**Implementation Steps**:

1. **Make `TargetExecutionContext` inherit from `ExecutionContext`**:
   Currently they are separate; unify inheritance.

2. **Move common property implementations to `ExecutionContext`**:
   - `repo`, `commit`, `repo_root`, `build_dir`, `scip_dir`
   - `artifact_path()`

3. **Keep write tracking in `TargetExecutionContext`**:
   - `write_table()`, `write_validated_table()`
   - `_written_tables` tracking

**Validation**:
```bash
uv run pyright src/codeintel/build/context.py
uv run pytest -q tests/build/
```

**Risk**: Low - Internal refactoring

---

### 3.2 Error Handling Completeness

**Problem**: Some places still use generic exceptions where structured errors exist.

**Implementation Steps**:

1. **Audit error sites**:
   ```bash
   grep -r "raise ValueError" src/codeintel/build/
   grep -r "raise RuntimeError" src/codeintel/build/
   ```

2. **Replace with structured errors where appropriate**:
   - `ValueError("Schema not found")` → `SchemaNotFoundError`
   - `RuntimeError("Gateway not available")` → `ResourceError` subclass

3. **Add new error types if needed**:
   - `GatewayNotAvailableError(ResourceError)`
   - `SessionNotInitializedError(ResourceError)`

**Validation**:
```bash
# Ensure all error imports work
uv run python -c "from codeintel.build.errors import *"
```

**Risk**: Low - Improves error reporting

---

### 3.3 Path Resolution Centralization

**Problem**: Path resolution logic is duplicated across contexts.

**Implementation Steps**:

1. **Create `PathResolver` utility**:
   ```python
   # resources.py or new path_resolver.py
   
   @dataclass(frozen=True)
   class PathResolver:
       """Centralized path resolution for build artifacts."""
       
       paths: BuildPaths
       snapshot: SnapshotRef
       
       def artifact_path(self, spec: ArtifactSpec) -> Path:
           """Resolve artifact path from spec."""
           return Path(spec.path_template.format(
               build_dir=self.paths.build_dir,
               scip_dir=self.paths.scip_dir,
               export_dir=self.paths.document_output_dir,
               repo_root=self.snapshot.repo_root,
           ))
       
       def table_export_path(self, table_key: str, format: str = "parquet") -> Path:
           """Generate export path for a table."""
           schema, table = table_key.split(".", 1)
           return self.paths.document_output_dir / schema / f"{table}.{format}"
   ```

2. **Use in context classes**:
   ```python
   @property
   def path_resolver(self) -> PathResolver:
       return PathResolver(self.paths, self.snapshot)
   ```

**Validation**:
```bash
uv run pyright src/codeintel/build/
```

**Risk**: Low - Utility addition

---

## Validation Checklist

After completing all phases, verify:

```bash
# Full quality check
uv run python -m tools.quality_report

# Type checking
uv run pyright --warnings --pythonversion=3.13
uv run pyrefly check

# All tests pass
uv run pytest -q

# Build-specific tests
uv run pytest -q tests/build/

# Hamilton tests
uv run pytest -q tests/build/hamilton/

# Registry consistency
uv run python -c "
from codeintel.build.unified_registry import get_unified_registry
reg = get_unified_registry()
errors = reg.validate()
if errors:
    print('Validation errors:', errors)
    exit(1)
print(f'Registry valid: {len(reg)} targets registered')
"
```

---

## Rollback Plan

Each phase is designed to be independently reversible:

1. **Phase 1.1** (Metadata): Revert import changes in affected files
2. **Phase 1.2** (Native Registry): Keep `NATIVE_TARGETS` tuple as fallback
3. **Phase 1.3** (Plugin Base): `MetadataPlugin` is additive, no migration required
4. **Phase 1.4** (Runner Factory): Old functions forward to new, remove forwarding
5. **Phase 2.x**: Each item has deprecation shims for backward compatibility

---

## Success Metrics

| Metric | Before | Target | Validation |
|--------|--------|--------|------------|
| Duplicated `_to_plugin_metadata` | 16 files | 0 files | `grep` |
| Registry sources | 3 | 1 | Code inspection |
| Plugin boilerplate lines | ~25/plugin | ~5/plugin | Line counts |
| Native module boilerplate | ~50 lines | ~15 lines | Line counts |
| Test coverage | Current | Maintained | pytest-cov |

---

## Timeline Estimate

| Phase | Estimated Effort |
|-------|------------------|
| 1.1 Metadata Helper | 2 hours |
| 1.2 Native Registry | 3 hours |
| 1.3 Plugin Base | 2 hours |
| 1.4 Runner Factory | 1.5 hours |
| 2.1 Complete Registrations | 3 hours |
| 2.2 Native Pattern | 4 hours |
| 2.3 Snapshot Helper | 1 hour |
| 2.4 Context Merge | 2 hours |
| 3.1-3.3 Low Priority | 3 hours |
| **Total** | **~21 hours** |

---

## Appendix: File Reference

### Files to Create

- `src/codeintel/build/plugins/_metadata.py` (move from analytics/)
- `src/codeintel/build/hamilton/native/executor.py`
- `src/codeintel/build/hamilton/native/ibis_helpers.py`

### Files to Modify

| File | Changes |
|------|---------|
| `plugin.py` | Add `MetadataPlugin` base class |
| `unified_registry.py` | Add native module query methods |
| `registrations.py` | Wire up plugin classes and native modules |
| `hamilton/native/registry.py` | Delegate to UnifiedRegistry |
| `hamilton/native/runner.py` | Add unified `create_run_record` factory |
| `hamilton/native/materializer.py` | Accept BuildContext |
| `context_base.py` | Add materialization fields |
| 16 plugin files | Update metadata helper import |
| 9 native modules | Use new executor/helpers |

### Files to Deprecate (keep for compatibility)

- `plugins/analytics/_metadata.py` → re-export from plugins/_metadata.py
- `MaterializationContext` → recommend BuildContext
- `NATIVE_TARGETS` tuple → delegate to UnifiedRegistry
