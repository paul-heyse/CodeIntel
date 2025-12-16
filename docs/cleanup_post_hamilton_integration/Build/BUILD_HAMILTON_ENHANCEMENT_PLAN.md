# Build System Hamilton Enhancement Plan

> **Status**: Draft  
> **Author**: AI Assistant  
> **Date**: 2025-12-16  
> **Prerequisite**: BUILD_CONSOLIDATION_PLAN.md  
> **Scope**: Leveraging Hamilton advanced features to achieve best-in-class architecture

---

## Executive Summary

This document extends the BUILD_CONSOLIDATION_PLAN by identifying how Hamilton's advanced features can be leveraged to further streamline the build system. The current implementation already uses several Hamilton features (`@tag`, `@check_output_custom`, `@schema.output`, lifecycle hooks), but many powerful capabilities remain untapped.

### Current Hamilton Feature Usage

| Feature | Status | Usage |
|---------|--------|-------|
| `@tag` | **Used** | 129 nodes tagged with domain/target/node_type |
| `@check_output_custom` | **Used** | ~6 targets with Pandera-style validators |
| `@schema.output` | **Used** | Schema documentation on compute nodes |
| Lifecycle hooks | **Used** | Manifest, telemetry, contract, progress hooks |
| `NativeTargetExecutor` | **Used** | Consolidated executor pattern |

### Untapped Hamilton Features

| Feature | Potential Impact | Effort |
|---------|-----------------|--------|
| `@cache` | Replace custom skip logic | High |
| `@parameterize` | Reduce ~50% native module boilerplate | High |
| `@config.when` | Environment-specific implementations | Medium |
| `@dataloader`/`@datasaver` | Standardized I/O patterns | Medium |
| Graph Adapters | Parallel target execution | Medium |
| `@extract_fields` | Multi-table target simplification | Low |
| Result Builders | Custom aggregation of TargetRunRecords | Low |

---

## Part 1: Replace Custom Skip Logic with Hamilton Cache

### Current State

Every native target implements custom skip logic via `NativeTargetExecutor.should_skip()`:

```python
# Current pattern in every target
executor = NativeTargetExecutor.for_target(env, graph, "risk_factors")
if executor.should_skip():
    return executor.skip()
```

This involves:
1. Computing input hash via `compute_input_hash()`
2. Checking manifest via `should_skip_native_target()`
3. Creating skip records via `executor.skip()`

### Hamilton's Built-in Caching

Hamilton provides `@cache` with sophisticated data versioning:

```python
from hamilton.function_modifiers import cache

@cache(format="parquet")
def t__risk_factors__compute(
    q__analytics__function_metrics: ir.Table,
    q__graph__call_graph_edges: ir.Table,
) -> ir.Table:
    """Compute risk factors - automatically cached."""
    ...
```

**Benefits:**
1. **Automatic invalidation**: Hamilton computes hashes based on function code + inputs
2. **Format flexibility**: Support for parquet, json, csv, pickle
3. **Transparent skip**: No manual skip logic required
4. **Cross-run persistence**: `.hamilton_cache` directory

### Implementation Strategy

#### Phase 1: Cache Adapter Integration

Create a custom `CacheConfig` that integrates with existing manifest system:

```python
# hamilton/cache/manifest_cache.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hamilton.caching import CacheConfig, ResultStore

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv


@dataclass
class ManifestCacheConfig(CacheConfig):
    """Cache config that integrates with build manifest system.
    
    This bridges Hamilton's cache API with our existing manifest-based
    skip logic, allowing gradual migration while maintaining consistency.
    """
    
    env: BuildEnv
    
    def get_result_store(self) -> ResultStore:
        """Return manifest-backed result store."""
        return ManifestResultStore(self.env)


class ManifestResultStore(ResultStore):
    """Result store backed by target manifests.
    
    Integrates Hamilton caching with existing manifest persistence,
    enabling incremental adoption of @cache decorator.
    """
    
    def __init__(self, env: BuildEnv) -> None:
        self.env = env
        self.manifest_index = env.manifest_index
    
    def get_result(self, key: str, data_version: str) -> tuple[bool, object]:
        """Check manifest for cached result."""
        # Map Hamilton cache key to target manifest
        target_name = self._extract_target_name(key)
        manifest = self.manifest_index.get(target_name)
        
        if manifest and manifest.input_hash == data_version:
            return (True, ManifestCacheHit(manifest))
        return (False, None)
    
    def store_result(self, key: str, data_version: str, result: object) -> None:
        """Store result in manifest system."""
        # Delegate to existing manifest persistence
        ...
```

#### Phase 2: Compute Node Caching

Apply `@cache` to pure compute nodes:

```python
# Before
@tag(domain="analytics", target="risk_factors", node_type="compute")
@check_output_custom(...)
def t__risk_factors__compute(...) -> ir.Table:
    ...

# After
@tag(domain="analytics", target="risk_factors", node_type="compute")
@cache(format="parquet", behavior=CacheBehavior.DEFAULT)
@check_output_custom(...)
def t__risk_factors__compute(...) -> ir.Table:
    ...
```

#### Phase 3: Eliminate Manual Skip Logic

Once caching is integrated, materialize nodes become simpler:

```python
# Before (40+ lines)
def t__risk_factors(env, graph, t__risk_factors__compute):
    executor = NativeTargetExecutor.for_target(env, graph, "risk_factors")
    if executor.should_skip():
        return executor.skip()
    return executor.execute(lambda: ...)

# After (~15 lines)
def t__risk_factors(env, graph, t__risk_factors__compute):
    """Materialize node - caching handled by Hamilton."""
    executor = NativeTargetExecutor.for_target(env, graph, "risk_factors")
    return executor.execute(lambda: ...)  # No skip check needed
```

### Migration Path

1. **Phase 1**: Implement `ManifestCacheConfig` and `ManifestResultStore`
2. **Phase 2**: Add `@cache` to 5 compute nodes as pilot
3. **Phase 3**: Validate caching behavior matches existing skip logic
4. **Phase 4**: Roll out to all native targets
5. **Phase 5**: Remove manual skip logic from `NativeTargetExecutor`

---

## Part 2: Use @parameterize for Similar Targets

### Current State

Many native targets follow nearly identical patterns. For example, ingestion targets:

```python
# ast.py
@tag(domain="ingestion", target="ast", node_type="compute")
def t__ast__extract(env: BuildEnv) -> AstExtractResult:
    modules = collect_modules(env)
    return extract_ast(modules, env.snapshot)

def t__ast(env, graph, t__ast__extract):
    executor = NativeTargetExecutor.for_target(env, graph, "ast")
    ...

# cst.py (almost identical)
@tag(domain="ingestion", target="cst", node_type="compute")
def t__cst__extract(env: BuildEnv) -> CstExtractResult:
    modules = collect_modules(env)
    return extract_cst(modules, env.snapshot)

def t__cst(env, graph, t__cst__extract):
    executor = NativeTargetExecutor.for_target(env, graph, "cst")
    ...
```

### Hamilton's @parameterize

Use `@parameterize` to generate multiple targets from one template:

```python
from hamilton.function_modifiers import parameterize, source, value

# Define extraction configurations
EXTRACT_CONFIGS = {
    "ast": {
        "extractor": value(extract_ast),
        "table_key": value("ingestion.ast_nodes"),
        "result_type": value(AstExtractResult),
    },
    "cst": {
        "extractor": value(extract_cst),
        "table_key": value("ingestion.cst_nodes"),
        "result_type": value(CstExtractResult),
    },
    "docstrings": {
        "extractor": value(extract_docstrings),
        "table_key": value("ingestion.docstrings"),
        "result_type": value(DocstringsResult),
    },
}


@parameterize(**EXTRACT_CONFIGS)
@tag(domain="ingestion", node_type="compute")
def t__{target}__extract(
    env: BuildEnv,
    extractor: Callable,
    result_type: type,
) -> Any:
    """Extract {target} data from repository modules.
    
    This function is parameterized to generate ast, cst, and docstrings
    extraction nodes with consistent behavior.
    """
    modules = collect_modules(env)
    return extractor(modules, env.snapshot)
```

### Benefits

1. **Reduce boilerplate**: 3 files (~300 lines) → 1 file (~80 lines)
2. **Consistent behavior**: Changes apply to all targets
3. **Self-documenting**: Config dict shows all variants
4. **Easy to extend**: Add new extractors by updating dict

### Implementation Strategy

#### Phase 1: Identify Parameterizable Groups

| Group | Targets | Pattern |
|-------|---------|---------|
| Ingestion extractors | ast, cst, docstrings | collect → extract → persist |
| Graph builders | call_graph, import_graph, cfg_dfg | load edges → build graph → persist |
| Metric computations | function_metrics, risk_factors, hotspots | load data → compute → persist |

#### Phase 2: Create Parameterized Templates

```python
# hamilton/native/templates/extraction.py

from hamilton.function_modifiers import parameterize, tag

# Configuration for all extraction targets
EXTRACTION_TARGETS = {
    "ast": {"extractor": extract_ast, "table": "ingestion.ast_nodes"},
    "cst": {"extractor": extract_cst, "table": "ingestion.cst_nodes"},
    "docstrings": {"extractor": extract_docstrings, "table": "ingestion.docstrings"},
    "tests": {"extractor": extract_tests, "table": "ingestion.test_catalog"},
    "config": {"extractor": extract_config, "table": "ingestion.config_entries"},
}


def build_extraction_nodes() -> dict[str, Callable]:
    """Generate parameterized extraction nodes."""
    nodes = {}
    for target_name, config in EXTRACTION_TARGETS.items():
        nodes[f"t__{target_name}__extract"] = create_extract_node(target_name, config)
        nodes[f"t__{target_name}"] = create_materialize_node(target_name, config)
    return nodes
```

#### Phase 3: Consolidate Native Modules

Replace individual target files with template-based generation:

```
hamilton/native/
├── ingestion/
│   ├── __init__.py          # Import from templates
│   ├── templates/
│   │   ├── extraction.py    # @parameterize for ast/cst/docstrings
│   │   └── indexing.py      # @parameterize for scip/typing
│   └── modules.py           # Keep special cases
├── analytics/
│   ├── __init__.py
│   ├── templates/
│   │   ├── metrics.py       # @parameterize for metric targets
│   │   └── coverage.py      # @parameterize for coverage targets
│   └── special/             # Non-parameterizable targets
└── graphs/
    ├── __init__.py
    └── templates/
        └── builders.py      # @parameterize for graph builders
```

### Estimated Reduction

| Domain | Current Files | After Parameterization |
|--------|---------------|------------------------|
| Ingestion | 10 | 4 |
| Analytics | 25 | 8 |
| Graphs | 8 | 3 |
| **Total** | 43 | 15 |

Lines of code reduction: ~60% (estimated 8,000 → 3,200 lines)

---

## Part 3: Use @config.when for Environment Variants

### Current State

Build behavior varies by environment but uses runtime conditionals:

```python
def t__scip__index(env: BuildEnv) -> ScipResult:
    if env.profile.skip_scip:
        return ScipResult.empty()
    if env.profile.scip_binary:
        return run_scip_binary(env.profile.scip_binary)
    return run_default_scip()
```

### Hamilton's @config.when

Use config-based node selection:

```python
from hamilton.function_modifiers import config

@config.when(scip_mode="binary")
def t__scip__index__binary(env: BuildEnv, scip_binary_path: str) -> ScipResult:
    """Index using specified SCIP binary."""
    return run_scip_binary(scip_binary_path)


@config.when(scip_mode="default")
def t__scip__index__default(env: BuildEnv) -> ScipResult:
    """Index using bundled SCIP."""
    return run_default_scip()


@config.when(scip_mode="skip")
def t__scip__index__skip(env: BuildEnv) -> ScipResult:
    """Skip SCIP indexing."""
    return ScipResult.empty()
```

### Benefits

1. **Compile-time selection**: DAG structure reflects config
2. **Type safety**: Each variant has explicit signature
3. **Testability**: Test each variant in isolation
4. **Documentation**: Config options are self-documenting

### Use Cases

| Scenario | Config Key | Variants |
|----------|------------|----------|
| SCIP indexing | `scip_mode` | binary, default, skip |
| Coverage collection | `coverage_mode` | pytest-cov, coverage.py, skip |
| Graph backend | `graph_backend` | networkx, networkx-gpu, igraph |
| Export format | `export_format` | parquet, jsonl, both |

### Implementation

```python
# config/build_config.py

from hamilton.function_modifiers import config

# Define all config variants
BUILD_CONFIG_OPTIONS = {
    "scip_mode": ["binary", "default", "skip"],
    "coverage_mode": ["pytest-cov", "coverage.py", "skip"],
    "graph_backend": ["networkx", "networkx-gpu"],
    "export_format": ["parquet", "jsonl"],
}


def validate_build_config(config: dict) -> dict:
    """Validate build config against known options."""
    for key, value in config.items():
        if key in BUILD_CONFIG_OPTIONS:
            valid = BUILD_CONFIG_OPTIONS[key]
            if value not in valid:
                msg = f"Invalid {key}={value}. Valid: {valid}"
                raise ValueError(msg)
    return config
```

---

## Part 4: Use @dataloader/@datasaver for I/O

### Current State

I/O is handled via `MaterializationContext` and `materialize_table()`:

```python
def t__risk_factors(env, graph, t__risk_factors__compute):
    executor = NativeTargetExecutor.for_target(env, graph, "risk_factors")
    if executor.should_skip():
        return executor.skip()
    
    def compute():
        ctx = MaterializationContext(
            gateway=env.gateway,
            snapshot=env.snapshot,
            validate=True,
            owner_target="risk_factors",
        )
        ref = materialize_table(ctx, "analytics.goid_risk_factors", t__risk_factors__compute)
        return {ref.table_key: ref.row_count}
    
    return executor.execute(compute)
```

### Hamilton's I/O Decorators

Use `@datasaver` for standardized output:

```python
from hamilton.function_modifiers import datasaver
from hamilton.io import utils

@datasaver()
def save__risk_factors(
    t__risk_factors__compute: ir.Table,
    env: BuildEnv,
) -> dict:
    """Save risk factors to analytics schema.
    
    Uses @datasaver for standardized I/O with metadata capture.
    """
    ref = materialize_ibis_table(
        env.gateway,
        "analytics.goid_risk_factors",
        t__risk_factors__compute,
        snapshot=env.snapshot,
    )
    return {
        "table_key": ref.table_key,
        "row_count": ref.row_count,
        "repo": env.snapshot.repo,
        "commit": env.snapshot.commit,
    }
```

### Benefits

1. **Standardized metadata**: Automatic capture of I/O metadata
2. **Composability**: Works with Hamilton's materialization API
3. **Observability**: I/O nodes visible in DAG visualization
4. **Separation**: Clear boundary between compute and I/O

### Custom DataSaver for DuckDB

```python
# hamilton/io/duckdb_saver.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hamilton.io import DataSaver

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from codeintel.storage.gateway import StorageGateway


@dataclass
class DuckDBTableSaver(DataSaver):
    """DataSaver for persisting Ibis expressions to DuckDB.
    
    Integrates with Hamilton's materialization API while using
    our existing storage gateway infrastructure.
    """
    
    table_key: str
    gateway: StorageGateway
    
    @classmethod
    def applicable_types(cls) -> list[type]:
        """Return types this saver handles."""
        import ibis.expr.types as ir
        return [ir.Table]
    
    def save_data(self, data: ir.Table) -> dict:
        """Persist Ibis table to DuckDB."""
        from codeintel.build.hamilton.native.materializer import materialize_table
        
        ref = materialize_table(
            MaterializationContext(gateway=self.gateway),
            self.table_key,
            data,
        )
        return {
            "table_key": ref.table_key,
            "row_count": ref.row_count,
        }
```

---

## Part 5: Use Graph Adapters for Parallel Execution

### Current State

Targets execute sequentially via Hamilton's default executor:

```python
driver = (
    Builder()
    .with_modules(*native_modules)
    .with_config(config)
    .build()
)
results = driver.execute(target_nodes)  # Sequential
```

### Hamilton's Parallel Adapters

Use thread pool for I/O-bound targets:

```python
from hamilton import driver
from hamilton.plugins.h_threadpool import FutureAdapter

# Thread pool for I/O-bound ingestion targets
io_adapter = FutureAdapter(max_workers=4)
ingestion_driver = (
    Builder()
    .with_modules(*ingestion_modules)
    .with_adapter(io_adapter)
    .build()
)

# CPU-bound analytics can use default or process pool
analytics_driver = (
    Builder()
    .with_modules(*analytics_modules)
    .enable_dynamic_execution(allow_experimental_mode=True)
    .with_remote_executor(executors.MultiProcessingExecutor(max_tasks=4))
    .build()
)
```

### Target Classification for Parallelism

| Category | Targets | Adapter |
|----------|---------|---------|
| I/O-bound | scip, typing, coverage, tests | ThreadPool(4) |
| CPU-bound | metrics, risk_factors, hotspots | MultiProcessing(4) |
| Memory-bound | call_graph, import_graph | Sequential |
| Quick | goids, modules | Sequential |

### Implementation

```python
# hamilton/execution/parallel.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from hamilton.plugins.h_threadpool import FutureAdapter

if TYPE_CHECKING:
    from hamilton.driver import Driver


class ExecutionMode(Enum):
    """Target execution mode."""
    
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    MULTIPROCESS = "multiprocess"


@dataclass
class ParallelExecutionConfig:
    """Configuration for parallel target execution."""
    
    io_workers: int = 4
    cpu_workers: int = 4
    enable_multiprocess: bool = False


def build_parallel_driver(
    modules: list,
    config: dict,
    execution_config: ParallelExecutionConfig,
) -> Driver:
    """Build driver with appropriate parallelism.
    
    Selects execution strategy based on target classification
    and system resources.
    """
    builder = driver.Builder().with_modules(*modules).with_config(config)
    
    # Classify targets and select adapter
    io_bound = classify_io_bound_targets(modules)
    
    if io_bound:
        adapter = FutureAdapter(max_workers=execution_config.io_workers)
        builder = builder.with_adapter(adapter)
    
    if execution_config.enable_multiprocess:
        builder = builder.enable_dynamic_execution(allow_experimental_mode=True)
        builder = builder.with_remote_executor(
            executors.MultiProcessingExecutor(max_tasks=execution_config.cpu_workers)
        )
    
    return builder.build()
```

---

## Part 6: Use @extract_fields for Multi-Table Targets

### Current State

Some targets produce multiple outputs:

```python
@dataclass
class DataModelsResult:
    fields: list[FieldRow]
    relationships: list[RelationshipRow]


def t__data_models__compute(env: BuildEnv) -> DataModelsResult:
    """Compute data model fields and relationships."""
    fields, relationships = analyze_data_models(env)
    return DataModelsResult(fields=fields, relationships=relationships)


def t__data_models(env, graph, t__data_models__compute):
    # Materialize both tables
    ctx = MaterializationContext(...)
    ref_fields = materialize_table(ctx, "analytics.data_model_fields", ...)
    ref_rels = materialize_table(ctx, "analytics.data_model_relationships", ...)
    return {ref_fields.table_key: ..., ref_rels.table_key: ...}
```

### Hamilton's @extract_fields

Split dict/TypedDict outputs into separate nodes:

```python
from hamilton.function_modifiers import extract_fields
from typing import TypedDict


class DataModelsOutput(TypedDict):
    fields: ir.Table
    relationships: ir.Table


@tag(domain="analytics", target="data_models", node_type="compute")
@extract_fields({"fields": ir.Table, "relationships": ir.Table})
def t__data_models__compute(env: BuildEnv) -> DataModelsOutput:
    """Compute data model fields and relationships.
    
    Returns dict that is automatically split into:
    - t__data_models__compute.fields
    - t__data_models__compute.relationships
    """
    fields_expr = analyze_fields(env)
    rels_expr = analyze_relationships(env)
    return {"fields": fields_expr, "relationships": rels_expr}


# Now can save each output independently
@datasaver()
def save__data_model_fields(
    t__data_models__compute_fields: ir.Table,
    env: BuildEnv,
) -> dict:
    """Save data model fields."""
    ...


@datasaver()
def save__data_model_relationships(
    t__data_models__compute_relationships: ir.Table,
    env: BuildEnv,
) -> dict:
    """Save data model relationships."""
    ...
```

### Benefits

1. **Fine-grained dependencies**: Downstream nodes can depend on specific outputs
2. **Parallel materialization**: Each output can be saved independently
3. **Selective execution**: Request only needed outputs
4. **Better caching**: Cache each output separately

---

## Part 7: Custom Result Builder for TargetRunRecord Aggregation

### Current State

Build results are collected manually:

```python
results = driver.execute(target_nodes)
records: list[TargetRunRecord] = []
for node, value in results.items():
    if isinstance(value, TargetRunRecord):
        records.append(value)
```

### Hamilton's Result Builders

Create custom builder for aggregating records:

```python
from hamilton.lifecycle import ResultBuilder


class TargetRunRecordBuilder(ResultBuilder):
    """Aggregate TargetRunRecords from Hamilton execution.
    
    Collects all TargetRunRecord outputs and provides summary
    statistics and validation.
    """
    
    def build_result(self, **outputs: dict) -> BuildExecutionResult:
        """Build aggregated execution result."""
        records: list[TargetRunRecord] = []
        other_outputs: dict[str, object] = {}
        
        for name, value in outputs.items():
            if isinstance(value, TargetRunRecord):
                records.append(value)
            else:
                other_outputs[name] = value
        
        return BuildExecutionResult(
            records=records,
            succeeded=[r for r in records if r.status == "succeeded"],
            failed=[r for r in records if r.status == "failed"],
            skipped=[r for r in records if r.status == "skipped"],
            total_duration_ms=sum(r.duration_ms for r in records),
            total_rows=sum(sum(r.row_counts.values()) for r in records),
            other_outputs=other_outputs,
        )


@dataclass
class BuildExecutionResult:
    """Aggregated build execution result."""
    
    records: list[TargetRunRecord]
    succeeded: list[TargetRunRecord]
    failed: list[TargetRunRecord]
    skipped: list[TargetRunRecord]
    total_duration_ms: float
    total_rows: int
    other_outputs: dict[str, object]
    
    @property
    def success_rate(self) -> float:
        """Return success rate as percentage."""
        total = len(self.records)
        return len(self.succeeded) / total * 100 if total else 0.0
    
    def summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"Build completed: {len(self.succeeded)} succeeded, "
            f"{len(self.failed)} failed, {len(self.skipped)} skipped "
            f"({self.total_duration_ms:.1f}ms, {self.total_rows:,} rows)"
        )
```

### Usage

```python
from hamilton import driver
from hamilton import base

adapter = base.SimplePythonGraphAdapter(
    result_builder=TargetRunRecordBuilder()
)
dr = driver.Builder().with_modules(*modules).with_adapter(adapter).build()

result: BuildExecutionResult = dr.execute(target_nodes)
print(result.summary())
# Build completed: 25 succeeded, 0 failed, 3 skipped (12345.6ms, 1,234,567 rows)
```

---

## Part 8: Use Lifecycle Hooks for Cross-Cutting Concerns

### Current Hooks

The codebase already has good hook infrastructure:

```python
# hooks/__init__.py
def build_hooks(run_id, gateway, graph, options=None) -> list:
    hooks = []
    if options.enable_telemetry:
        hooks.append(NodeTelemetryHook(run_id, gateway))
    if options.enable_validation:
        hooks.append(ContractEnforcementHook(graph))
    if options.enable_progress:
        hooks.append(create_progress_hook())
    return hooks
```

### Additional Hooks to Add

#### 1. OpenLineage Adapter for Data Lineage

```python
from hamilton.plugins.h_openlineage import OpenLineageAdapter

lineage_adapter = OpenLineageAdapter(
    namespace="codeintel",
    job_name="build",
    producer="codeintel-build-system",
)
```

#### 2. MLflow Tracker for Metrics

```python
from hamilton.plugins.h_mlflow import MLFlowTracker

mlflow_tracker = MLFlowTracker(
    experiment_name="codeintel-build",
    tracking_uri="http://localhost:5000",
)
```

#### 3. Slack Notifier for Failures

```python
from hamilton.plugins.h_slack import SlackNotifier

slack_notifier = SlackNotifier(
    api_key=os.environ.get("SLACK_API_KEY"),
    channel="#build-alerts",
    notify_on=["failure"],  # Only notify on failures
)
```

### Enhanced Hook Configuration

```python
@dataclass(frozen=True, slots=True)
class HookOptions:
    """Extended hook configuration."""
    
    # Existing options
    strict_contracts: bool = False
    enable_validation: bool = True
    enable_telemetry: bool = True
    enable_progress: bool = False
    enable_timing: bool = False
    progress_desc: str = "Building targets"
    
    # New options from Hamilton advanced features
    enable_openlineage: bool = False
    openlineage_namespace: str = "codeintel"
    enable_mlflow: bool = False
    mlflow_experiment: str = "codeintel-build"
    enable_slack: bool = False
    slack_channel: str = "#build-alerts"
    slack_notify_on: tuple[str, ...] = ("failure",)


def build_hooks(run_id, gateway, graph, options=None) -> list:
    """Build comprehensive hook set."""
    if options is None:
        options = HookOptions()
    
    hooks = []
    
    # Existing hooks
    if options.enable_telemetry:
        hooks.append(NodeTelemetryHook(run_id, gateway))
    if options.enable_validation:
        hooks.append(ContractEnforcementHook(graph, strict=options.strict_contracts))
    if options.enable_progress:
        hooks.append(create_progress_hook(options.progress_desc))
    if options.enable_timing:
        hooks.append(BuildTimingHook())
    
    # New Hamilton-native hooks
    if options.enable_openlineage:
        hooks.append(OpenLineageAdapter(namespace=options.openlineage_namespace))
    if options.enable_mlflow:
        hooks.append(MLFlowTracker(experiment_name=options.mlflow_experiment))
    if options.enable_slack:
        hooks.append(SlackNotifier(
            channel=options.slack_channel,
            notify_on=list(options.slack_notify_on),
        ))
    
    return hooks
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

| Task | Description | Impact |
|------|-------------|--------|
| 1.1 | Implement `ManifestCacheConfig` | Enables @cache integration |
| 1.2 | Add `@cache` to 5 pilot targets | Validate caching approach |
| 1.3 | Create `TargetRunRecordBuilder` | Cleaner result aggregation |

### Phase 2: Parameterization (Week 3-4)

| Task | Description | Impact |
|------|-------------|--------|
| 2.1 | Identify parameterizable target groups | Foundation for consolidation |
| 2.2 | Create extraction template module | Consolidate ast/cst/docstrings |
| 2.3 | Create metrics template module | Consolidate analytics targets |
| 2.4 | Migrate remaining targets | Complete consolidation |

### Phase 3: Advanced Features (Week 5-6)

| Task | Description | Impact |
|------|-------------|--------|
| 3.1 | Add `@config.when` variants | Environment flexibility |
| 3.2 | Implement `@dataloader`/`@datasaver` | Standardized I/O |
| 3.3 | Add parallel execution config | Performance improvement |
| 3.4 | Integrate OpenLineage/MLflow hooks | Observability |

### Phase 4: Cleanup (Week 7-8)

| Task | Description | Impact |
|------|-------------|--------|
| 4.1 | Remove manual skip logic | Code simplification |
| 4.2 | Deprecate `MaterializationContext` | API consolidation |
| 4.3 | Update documentation | Developer experience |
| 4.4 | Performance benchmarks | Validation |

---

## Success Metrics

| Metric | Current | Target | Validation |
|--------|---------|--------|------------|
| Native target files | 43 | 15 | File count |
| Lines of code | ~8,000 | ~3,200 | `wc -l` |
| Skip logic instances | ~48 | 0 | grep |
| Cache hit rate | N/A | >80% | Telemetry |
| Parallel speedup | 1x | 2-4x | Benchmark |
| Test coverage | Current | Maintained | pytest-cov |

---

## Appendix: Hamilton Feature Reference

### Key Imports

```python
from hamilton.function_modifiers import (
    # Metadata
    tag,
    schema,
    
    # Validation
    check_output,
    check_output_custom,
    
    # Caching
    cache,
    
    # Parameterization
    parameterize,
    parameterize_sources,
    parameterize_values,
    source,
    value,
    
    # Output splitting
    extract_fields,
    extract_columns,
    unpack_fields,
    
    # Conditional
    config,
    
    # I/O
    dataloader,
    datasaver,
    load_from,
    save_to,
)

from hamilton.plugins import (
    h_threadpool,
    h_dask,
    h_ray,
    h_openlineage,
    h_mlflow,
    h_slack,
    h_tqdm,
)
```

### Documentation Links

- Function Modifiers: https://hamilton.apache.org/concepts/function-modifiers/
- Caching: https://hamilton.apache.org/how-tos/caching-tutorial/
- Materialization: https://hamilton.staged.apache.org/concepts/materialization/
- Parallel Execution: https://hamilton.apache.org/concepts/parallel-task/
- Lifecycle Hooks: https://hamilton.staged.apache.org/reference/lifecycle-hooks/

---

## Conclusion

By leveraging Hamilton's advanced features, the build system can achieve:

1. **60% code reduction** via `@parameterize` and template consolidation
2. **Simplified caching** via `@cache` replacing manual skip logic
3. **Better observability** via OpenLineage and MLflow integration
4. **Improved performance** via parallel execution adapters
5. **Cleaner I/O** via `@dataloader`/`@datasaver` patterns
6. **Environment flexibility** via `@config.when` variants

This represents a significant evolution toward a truly best-in-class build system that fully leverages the Hamilton framework's capabilities.

