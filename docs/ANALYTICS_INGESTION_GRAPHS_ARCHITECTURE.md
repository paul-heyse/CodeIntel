# CodeIntel Architecture Design Document

> **Comprehensive Technical Reference for the Analytics, Ingestion, and Graphs Modules**

---

## Table of Contents

### Part I: Analytics Module
- [1. Executive Summary](#executive-summary)
- [2. Architectural Overview](#1-architectural-overview)
- [3. Core Component Deep Dive](#2-core-component-deep-dive)
  - [3.1 Plugin Protocol](#21-plugin-protocol-coreplugin_protocolpy)
  - [3.2 Execution Context](#22-execution-context-coreexecution_contextpy)
  - [3.3 Base Plugin Classes](#23-base-plugin-classes-corebasepy)
  - [3.4 Plugin Registry & Execution](#24-plugin-registry--execution-coreregistrypy-coreexecutorpy)
- [4. Resource Provider Layer](#3-resource-provider-layer-resources)
- [5. Pure Compute Layer](#4-pure-compute-layer-compute)
- [6. Persistence Layer](#5-persistence-layer-adapters)
- [7. Registered Plugins & Output Data](#6-registered-plugins--output-data)
- [8. Middleware & Cross-Cutting Concerns](#7-middleware--cross-cutting-concerns)
- [9. Output Contracts & Validation](#8-output-contracts--validation)
- [10. Pipeline Integration](#9-pipeline-integration)
- [11. Data Flow Summary](#10-data-flow-summary)
- [12. Key Design Decisions & Rationale](#11-key-design-decisions--rationale)
- [13. Extension Points](#12-extension-points)
- [14. Testing Strategy](#13-testing-strategy)
- [15. Performance Considerations](#14-performance-considerations)
- [16. Observability](#15-observability)
- [17. File Structure Reference](#16-file-structure-reference)

### Part II: Ingestion Module
- [18. Ingestion Executive Summary](#executive-summary-1)
- [19. Ingestion Architectural Overview](#17-ingestion-architectural-overview)
- [20. Ingestion Core Components](#18-core-component-deep-dive-1)
  - [20.1 Plugin Protocol](#181-plugin-protocol-pluginsprotocolpy)
  - [20.2 Execution Context](#182-execution-context-coreexecution_contextpy-1)
  - [20.3 Base Plugin Classes](#183-base-plugin-classes-corebasepy-1)
  - [20.4 Plugin Registry](#184-plugin-registry-pluginsregistrypy)
- [21. Port-Adapter Architecture](#19-port-adapter-architecture)
- [22. Change Tracker & Incremental Ingestion](#20-change-tracker--incremental-ingestion)
- [23. Resource Provider Layer](#21-resource-provider-layer-resources-1)
- [24. Steps Layer (Pure Domain Logic)](#22-steps-layer-pure-domain-logic)
- [25. Recipe Composition](#23-recipe-composition-recipes)
- [26. Ingest Runs & Observability](#24-ingest-runs--observability)
- [27. Ingestion Registered Plugins](#25-registered-plugins--output-data-1)
- [28. Ingestion Data Flow Summary](#26-ingestion-data-flow-summary)
- [29. Ingestion Key Design Decisions](#27-key-design-decisions--rationale-1)
- [30. Ingestion Extension Points](#28-extension-points-1)
- [31. Ingestion Testing Strategy](#29-testing-strategy-1)
- [32. Ingestion File Structure](#30-file-structure-reference-1)

### Part III: Graphs Module
- [33. Graphs Executive Summary](#graphs-executive-summary)
- [34. Graphs Architectural Overview](#31-graphs-architectural-overview)
- [35. Graphs Core Components](#32-graphs-core-component-deep-dive)
  - [35.1 Graph Plugin Protocol](#321-graph-plugin-protocol-coreprotocolpy)
  - [35.2 Graph Execution Context](#322-graph-execution-context-corecontextpy)
  - [35.3 Graph Plugin Registry](#323-graph-plugin-registry-coreregistrypy)
- [36. Graph Engine Layer](#33-graph-engine-layer)
- [37. Graph Plugin Kinds](#34-graph-plugin-kinds)
- [38. Hexagonal Architecture in Graphs](#35-hexagonal-architecture-in-graphs)
- [39. Graph Recipe Composition](#36-graph-recipe-composition)
- [40. Registered Graph Plugins](#37-registered-graph-plugins--output-data)
- [41. Graphs Data Flow Summary](#38-graphs-data-flow-summary)
- [42. Graphs Key Design Decisions](#39-graphs-key-design-decisions--rationale)
- [43. Graphs Extension Points](#40-graphs-extension-points)
- [44. Graphs File Structure](#41-graphs-file-structure-reference)

---

# Part I: Analytics Module

## Executive Summary

The `analytics` module in CodeIntel is a sophisticated **plugin-based analytics computation engine** that processes code repository snapshots to extract deep insights about code structure, quality, dependencies, and maintainability. It follows a **layered architecture** with clear separation between orchestration, computation, and persistence, enabling both modularity and high-performance batch processing.

---

## 1. Architectural Overview

The analytics system is organized into **five primary layers**:

```
┌─────────────────────────────────────────────────────────────────┐
│                 Pipeline / Orchestration Layer                   │
│          (pipeline_bridge.py, executor.py, registry.py)          │
├─────────────────────────────────────────────────────────────────┤
│                      Plugin Layer                                │
│         (core/plugins/*, thin orchestration wrappers)            │
├─────────────────────────────────────────────────────────────────┤
│                 Resource Provider Layer                          │
│    (resources/graphs.py, catalog.py, asts.py, features.py)       │
├─────────────────────────────────────────────────────────────────┤
│                 Pure Compute Layer                               │
│         (compute/functions/, compute/graphs/, etc.)              │
├─────────────────────────────────────────────────────────────────┤
│                 Persistence / Adapter Layer                      │
│           (adapters/, datasets.py, DuckDB storage)               │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Separation of Concerns**: Computation is pure (no I/O); plugins orchestrate; adapters handle persistence
2. **Lazy Resource Loading**: Expensive resources (graphs, AST maps) load on-demand through providers
3. **Capability-Based Dependencies**: Plugins declare what they provide and require; the system resolves execution order
4. **Contract-Driven Validation**: Output contracts ensure data quality and schema compliance
5. **Middleware Support**: Cross-cutting concerns (logging, metrics, tracing) are pluggable

---

## 2. Core Component Deep Dive

### 2.1 Plugin Protocol (`core/plugin_protocol.py`)

The foundation is the `AnalyticsPluginProtocol`, a runtime-checkable protocol that defines what every plugin must provide:

```python
@runtime_checkable
class AnalyticsPluginProtocol(Protocol):
    @property
    def metadata(self) -> PluginMetadata: ...
    def execute(self, ctx: PluginExecutionContext) -> PluginResult: ...
    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult: ...
```

**Key Metadata Fields** (from `PluginMetadata`):

| Field | Purpose |
|-------|---------|
| `name` | Stable identifier (e.g., `"functions.metrics"`) |
| `stage` | Execution grouping (`function`, `graph`, `test`, `coverage`, etc.) |
| `depends_on` | Explicit dependencies on other plugins |
| `capabilities_provided` | What this plugin provides (e.g., `"analytics.function_metrics"`) |
| `capabilities_required` | What this plugin needs from others |
| `severity` | Error handling (`fatal`, `soft_fail`, `skip_on_error`) |
| `resource_hints` | Runtime budgets (max runtime, memory, GPU preference) |

**Available Plugin Stages**:

```python
PluginStage = Literal[
    "graph",           # Graph-based computations
    "function",        # Function-level analysis
    "function_history",# Historical function metrics
    "test",            # Test-related analytics
    "coverage",        # Code coverage analysis
    "subsystem",       # Architectural grouping
    "data_model",      # Data structure analysis
    "data_model_usage",# Data model usage patterns
    "entrypoints",     # Entry point detection
    "profiles",        # Multi-level profiles
    "history",         # Historical trends
    "semantic",        # Semantic role analysis
    "hotspots",        # Risk hotspot detection
    "risk",            # Risk factor computation
    "cfg",             # Control flow graph
    "dfg",             # Data flow graph
    "symbol",          # Symbol-level analysis
    "config",          # Configuration analysis
    "stats",           # Statistical computations
    "other",           # Uncategorized
]
```

### 2.2 Execution Context (`core/execution_context.py`)

The `PluginExecutionContext` is a **slim, protocol-driven context** that provides plugins access to:

```python
@dataclass
class PluginExecutionContext:
    gateway: StorageGateway       # DuckDB access
    snapshot: SnapshotRef         # repo/commit/repo_root
    run_id: str                   # Unique execution identifier
    scope: AnalyticsScope         # Filtering (paths, modules, time window)
    configs: ConfigProvider       # Typed configuration access
    resources: ResourceRegistry   # Lazy resource providers
    scratch: PluginScratch        # Inter-plugin communication
```

**Resource Access Pattern**:

```python
# Typed resource access
graph_provider = ctx.require(GraphProvider)
call_graph = graph_provider.call_graph

# Config access
config = ctx.get_config(FunctionAnalyticsStepConfig)

# Inter-plugin communication via scratch
ctx.scratch.declare("my_intermediate_result", data)
data = ctx.scratch.consume("prior_plugin_result")
```

### 2.3 Base Plugin Classes (`core/base.py`)

A hierarchy of base classes minimizes boilerplate:

```
BasePlugin (abstract)
├── TableWriterPlugin (auto row-count tracking)
├── ConfigBoundPlugin[TConfig] (typed config injection)
├── CatalogRequiringPlugin (function catalog access)
├── GraphRuntimeRequiringPlugin (graph loading)
├── GraphMetricsPlugin (combines table + graph + catalog)
└── Composite classes:
    ├── ConfiguredTableWriterPlugin[TConfig]
    └── ConfiguredGraphMetricsPlugin[TConfig]
```

**Example Plugin (thin orchestration)**:

```python
@dataclass
class FunctionMetricsPlugin(
    ConfiguredTableWriterPlugin[FunctionAnalyticsStepConfig],
    WithContractValidation,
):
    """Compute function metrics, complexity, and type annotations."""

    # Identification
    plugin_name: ClassVar[str] = "functions.metrics"
    plugin_stage: ClassVar[PluginStage] = "function"
    plugin_version: ClassVar[str] = "3.0.0"

    # Configuration
    config_type: ClassVar[type[FunctionAnalyticsStepConfig]] = FunctionAnalyticsStepConfig

    # Output tables (contracts auto-generated from these)
    output_tables: ClassVar[tuple[str, ...]] = (
        "analytics.function_metrics",
        "analytics.function_types",
    )

    # Capabilities
    provides: ClassVar[tuple[str, ...]] = output_tables
    requires: ClassVar[tuple[str, ...]] = ("core.goids",)

    def compute(self, ctx: PluginExecutionContext) -> Mapping[str, int] | None:
        """Execute function metrics computation."""
        # Get AST data from AstProvider if available
        function_ast_map = None
        if ctx.has_resource_by_name("AstProvider"):
            ast_data = cast("LegacyAstData", ctx.require_by_name("AstProvider"))
            function_ast_map = ast_data.function_ast_map

        opts = FunctionAnalyticsOptions(function_ast_map=function_ast_map)
        result = compute_function_metrics_and_types(ctx.gateway, self.config, options=opts)

        return {
            "analytics.function_metrics": result.get("metrics_rows", 0),
            "analytics.function_types": result.get("types_rows", 0),
        }
```

### 2.4 Plugin Registry & Execution (`core/registry.py`, `core/executor.py`)

The **`PluginRegistry`** manages plugin discovery and dependency resolution:

1. **Registration**: Plugins register by name, with indexes by capability and stage
2. **Planning**: `registry.plan(plugin_names)` performs:
   - Selection based on enabled/disabled lists
   - Dependency resolution (explicit + capability-based)
   - Topological sorting with cycle detection
3. **Execution**: `PluginExecutor` runs plugins in order with:
   - Input validation before each plugin
   - Retry logic with configurable backoff
   - Middleware chain (before/after hooks, error handling)
   - Contract validation after successful execution

**Execution Flow**:

```
plan_analytics_plugin_run(request)
    └── registry.plan(plugin_names)
            └── _resolve_selection()
            └── _resolve_dependencies()
            └── _topological_sort()

run_analytics_plugins(plan, run_context)
    └── executor.execute(ctx, plan)
            └── for plugin in plan.plugins:
                    └── validate_inputs(ctx)
                    └── middleware.before_execute()
                    └── plugin.execute(ctx)
                    └── middleware.after_execute()
                    └── validate_plugin_outputs()
```

**Execution Policy Configuration**:

```python
@dataclass(frozen=True)
class ExecutionPolicy:
    fail_fast: bool = True           # Stop on first failure
    max_retries: int = 0             # Retry attempts for failed plugins
    retry_backoff_ms: int = 100      # Wait between retries
    skip_on_unchanged: bool = False  # Skip if inputs unchanged
    dry_run: bool = False            # Plan but don't execute
    validate_contracts: bool = True  # Validate output contracts
```

---

## 3. Resource Provider Layer (`resources/`)

The **lazy loading pattern** ensures expensive resources are only loaded when needed.

### 3.1 Protocol & Base Classes

```python
@runtime_checkable
class ResourceProvider[T_co](Protocol):
    @property
    def is_loaded(self) -> bool: ...
    def get(self) -> T_co: ...           # Load on first access
    def get_or_none(self) -> T_co | None: ...
    def invalidate(self) -> None: ...    # Force reload

class LazyResource[T](ABC):
    """Base class with standard lazy loading + caching."""
    def _load(self) -> T: ...            # Subclasses implement
    def set_preloaded(self, resource: T) -> None: ...
```

### 3.2 Key Providers

| Provider | What It Provides | When Used |
|----------|------------------|-----------|
| `GraphProvider` | Call graph, import graph, symbol graphs, bipartite graphs | Graph metrics, centrality, subsystems |
| `CatalogProvider` | Function metadata catalog (GOIDs, URNs, locations) | All function-level analytics |
| `AstProvider` | Parsed AST maps per function | Complexity, type annotations, effects |
| `FeaturesProvider` | Extracted AST features | Pattern detection, behavioral analysis |

### 3.3 Graph Runtime (`graph_runtime.py`)

The `GraphRuntime` wraps a `GraphEngine` and provides:

- **Lazy graph loading** with in-memory caching
- **Disk caching** (JSON serialization) for expensive computations
- **Multiple graph types**:

| Graph | Type | Purpose |
|-------|------|---------|
| `call_graph` | DiGraph | Function call relationships |
| `import_graph` | DiGraph | Module import dependencies |
| `symbol_module_graph` | Graph | Symbol-to-module bipartite coupling |
| `symbol_function_graph` | Graph | Symbol-to-function bipartite coupling |
| `config_module_bipartite` | Graph | Config key to module mapping |
| `test_function_bipartite` | Graph | Test to function coverage mapping |
| `cfg_graph` | DiGraph | Control flow graph (optional) |

- **Runtime pooling** (`GraphRuntimePool`) with LRU eviction and TTL expiry

**Graph Runtime Options**:

```python
@dataclass(frozen=True)
class GraphRuntimeOptions:
    snapshot: SnapshotRef | None = None
    backend: GraphBackendConfig | None = None
    graphs: GraphKind = GraphKind.ALL
    eager: bool = False              # Load all graphs immediately
    validate: bool = False           # Validate graph integrity
    cache_key: str | None = None
    graph_cache_dir: Path | None = None
    features: GraphFeatureFlags = field(default_factory=GraphFeatureFlags)
```

---

## 4. Pure Compute Layer (`compute/`)

The compute layer contains **pure functions** with no I/O or side effects. This enables:

- Easy unit testing
- Potential parallelization
- Clear separation from persistence

### 4.1 Compute Domains

| Domain | Module | Computations |
|--------|--------|-------------|
| Functions | `functions/complexity.py` | Cyclomatic complexity, nesting depth, statement counts |
| Functions | `functions/typedness.py` | Type annotation coverage analysis |
| Functions | `functions/loc.py` | Lines of code metrics |
| Functions | `functions/signatures.py` | Function signature parsing |
| Graphs | `graphs/centrality.py` | PageRank, betweenness, closeness centrality |
| Graphs | `graphs/statistics.py` | Graph statistics (node/edge counts, density) |
| Profiles | `profiles/aggregation.py` | Multi-metric profile aggregation |
| Profiles | `profiles/features.py` | Feature extraction for ML/analysis |
| Dependencies | `dependencies/classification.py` | Dependency type classification |
| Dependencies | `dependencies/detection.py` | Dependency detection algorithms |
| Semantic | `semantic_roles/classification.py` | Function role inference (handler, utility, etc.) |

### 4.2 Example: Complexity Computation

```python
@dataclass(frozen=True)
class ComplexityMetrics:
    """Immutable container for function complexity metrics."""
    cyclomatic: int              # McCabe cyclomatic complexity
    max_nesting_depth: int       # Maximum nesting level
    return_count: int            # Number of return statements
    yield_count: int             # Number of yield/yield from statements
    raise_count: int             # Number of raise statements
    stmt_count: int              # Statements in function body
    decorator_count: int         # Decorators applied
    has_docstring: bool          # Whether function has docstring
    is_async: bool               # Async function flag
    is_generator: bool           # Generator function flag
    complexity_bucket: str       # "low", "medium", or "high"

def compute_complexity(node: ast.AST) -> ComplexityMetrics:
    """Compute complexity metrics for a function AST node."""
    # Pure function - no I/O, no side effects
    ...
```

---

## 5. Persistence Layer (`adapters/`)

Adapters handle all database I/O, mapping between domain objects and DuckDB tables.

### 5.1 Adapter Patterns

```python
# Base pattern for load + persist
class AnalyticsAdapter[RowT](ABC):
    def load(self) -> Iterator[RowT]: ...
    def persist(self, rows: Sequence[RowT]) -> int: ...

# Split input/output types (preferred for new adapters)
class ComputeAdapter[InputT, OutputT](ABC):
    def load_inputs(self) -> Iterator[InputT]: ...   # Source data
    def load_outputs(self) -> Iterator[OutputT]: ... # Existing results
    def persist(self, rows: Sequence[OutputT]) -> int: ...

# Batch adapter with delete-before-insert
class BatchAdapter[RowT](AnalyticsAdapter[RowT], ABC):
    @property
    def table_name(self) -> str: ...
    def delete_scope(self) -> DeleteScope: ...
    def persist_batch(self, rows, *, delete_before: bool = True) -> int: ...
```

### 5.2 Available Adapters

| Adapter | Tables | Purpose |
|---------|--------|---------|
| `FunctionMetricsAdapter` | `analytics.function_metrics` | Complexity, size metrics |
| `FunctionTypesAdapter` | `analytics.function_types` | Type annotation data |
| `ProfilesAdapter` | `analytics.profiles_*` | Multi-level profiles |
| `SubsystemsAdapter` | `analytics.subsystems`, `subsystem_modules`, `subsystem_functions` | Architectural groupings |
| `EntrypointsAdapter` | `analytics.entrypoints` | Entry point detection |
| `SemanticRolesAdapter` | `analytics.semantic_roles` | Function role classification |
| `DependenciesAdapter` | `analytics.dependencies` | External dependency tracking |
| `DataModelsAdapter` | `analytics.data_models_*` | Data structure usage |

---

## 6. Registered Plugins & Output Data

### 6.1 All Registered Plugins

| Plugin | Stage | Description |
|--------|-------|-------------|
| `functions.metrics` | function | Compute complexity and type coverage metrics |
| `functions.ast_features` | function | Extract AST-based features |
| `functions.effects` | function | Analyze side effects (I/O, state mutation) |
| `functions.contracts` | function | Extract pre/post conditions |
| `functions.history` | function_history | Historical function metrics |
| `coverage.functions` | coverage | Test coverage per function |
| `coverage.test_edges` | coverage | Test-to-function mapping |
| `tests.profile` | test | Test profile analysis |
| `tests.behavioral_coverage` | test | Behavioral coverage analysis |
| `hotspots.build` | hotspots | Risk hotspot identification |
| `subsystems.build` | subsystem | Infer architectural boundaries |
| `semantic_roles.compute` | semantic | Function role classification |
| `data_models.build` | data_model | Data structure definitions |
| `data_model_usage.build` | data_model_usage | Usage pattern analysis |
| `entrypoints.build` | entrypoints | Entry point detection |
| `dependencies.external` | function | External dependency tracking |
| `profiles.build` | profiles | Multi-level aggregated profiles |
| `history.timeseries` | history | Historical trend analysis |
| `risk_factors.build` | risk | Risk scoring computation |
| `config_data_flow.compute` | config | Config propagation analysis |
| `core_graph_metrics` | graph | Centrality and degree metrics |

### 6.2 Output Tables (Data Flow Destinations)

| Table | Plugin | Contents |
|-------|--------|----------|
| `analytics.function_metrics` | `functions.metrics` | Complexity, LOC, nesting depth |
| `analytics.function_types` | `functions.metrics` | Type annotation coverage |
| `analytics.function_effects` | `functions.effects` | Side effects (I/O, state mutation) |
| `analytics.function_contracts` | `functions.contracts` | Pre/post conditions |
| `analytics.function_ast_features` | `functions.ast_features` | Pattern-based features |
| `analytics.graph_metrics_functions` | `core_graph_metrics` | Centrality, degree metrics |
| `analytics.graph_metrics_modules` | `core_graph_metrics` | Module-level graph metrics |
| `analytics.coverage_functions` | `coverage.functions` | Test coverage per function |
| `analytics.coverage_test_edges` | `coverage.test_edges` | Test-to-function mapping |
| `analytics.hotspots` | `hotspots.build` | Risk hotspot identification |
| `analytics.subsystems` | `subsystems.build` | Inferred architectural boundaries |
| `analytics.subsystem_modules` | `subsystems.build` | Module-to-subsystem mapping |
| `analytics.subsystem_functions` | `subsystems.build` | Function-to-subsystem mapping |
| `analytics.semantic_roles` | `semantic_roles.compute` | Function role classification |
| `analytics.entrypoints` | `entrypoints.build` | Entry point detection |
| `analytics.entrypoint_tests` | `entrypoints.build` | Tests for entry points |
| `analytics.profiles_*` | `profiles.build` | Multi-level aggregated profiles |
| `analytics.history_timeseries` | `history.timeseries` | Historical metrics |
| `analytics.risk_factors` | `risk_factors.build` | Risk scoring |
| `analytics.data_models` | `data_models.build` | Data structure definitions |
| `analytics.data_model_usage` | `data_model_usage.build` | Usage patterns |
| `analytics.config_data_flow` | `config_data_flow.compute` | Config propagation |
| `analytics.external_deps` | `dependencies.external` | External dependencies |

---

## 7. Middleware & Cross-Cutting Concerns

### 7.1 Middleware Protocol

```python
@runtime_checkable
class PluginMiddleware(Protocol):
    @property
    def name(self) -> str: ...
    
    def before_execute(self, ctx: PluginExecutionContext, 
                       plugin: AnalyticsPluginProtocol) -> None: ...
    
    def after_execute(self, ctx: PluginExecutionContext,
                      plugin: AnalyticsPluginProtocol,
                      result: PluginResult) -> PluginResult: ...
    
    def on_error(self, ctx: PluginExecutionContext,
                 plugin: AnalyticsPluginProtocol,
                 error: Exception) -> Exception | None: ...
```

### 7.2 Middleware Chain

The `MiddlewareChain` composes multiple middleware:

- `before_execute`: Called in order
- `after_execute`: Called in **reverse** order
- `on_error`: Called in order, can suppress/transform errors

### 7.3 Built-in Middleware

| Middleware | Purpose |
|------------|---------|
| `LoggingMiddleware` | Structured logging of execution start/end/errors |
| `MetricsMiddleware` | Prometheus metrics (duration, success/failure counters) |
| `TracingMiddleware` | OpenTelemetry spans for distributed tracing |

---

## 8. Output Contracts & Validation

The contract system (`core/contracts.py`) ensures data quality:

```python
@dataclass(frozen=True)
class OutputContractSpec:
    table: str
    min_rows: int | None = None
    required_columns: tuple[str, ...] = ()
    column_constraints: tuple[ColumnConstraint, ...] = ()
    description: str = ""
    severity: Literal["error", "warning"] = "error"
```

### 8.1 Constraint Types

| Type | Purpose |
|------|---------|
| `not_null` | Column must not contain NULLs |
| `min_value` | Numeric lower bound |
| `max_value` | Numeric upper bound |
| `min_fraction_not_null` | Minimum non-null fraction (0.0-1.0) |
| `in_set` | Value must be in allowed set |
| `regex` | String pattern matching |

### 8.2 Contract Validation Flow

```python
# Contracts are auto-generated from output_tables or explicit
@property
def output_contracts(self) -> tuple[OutputContractSpec, ...]:
    return (
        OutputContractSpec(
            table="analytics.graph_metrics_functions",
            min_rows=1,
            required_columns=("repo", "commit", "node_id", "out_degree"),
        ),
    )

# Validation runs after successful plugin execution
validator = ContractValidator(gateway)
result = validator.validate(contracts, snapshot)
if not result.valid:
    for violation in result.violations:
        log.error(violation.message)
```

---

## 9. Pipeline Integration

The `pipeline_bridge.py` module connects analytics to the broader orchestration system:

### 9.1 Planning

```python
@dataclass(frozen=True)
class AnalyticsPlanRequest:
    plugin_names: Sequence[str]
    policy: GraphPluginPolicy
    repo: str
    commit: str
    scope: GraphRunScope
    prior_manifest: Mapping[str, Mapping[str, object]] | None
    cfg_options: Mapping[str, dict[str, object]] | None
    runtime_options: Mapping[str, dict[str, object]] | None
    run_id: str

def plan_analytics_plugin_run(request: AnalyticsPlanRequest) -> AnalyticsPluginExecutionPlan:
    """Creates an execution plan with dependency resolution."""
```

### 9.2 Execution

```python
@dataclass(frozen=True)
class AnalyticsRunContext:
    gateway: StorageGateway
    graph_runtime: GraphRuntime | None
    cfgs: Mapping[str, Any]
    extra: Mapping[str, Any]
    catalog_provider: FunctionCatalogProvider | None = None
    snapshot: SnapshotRef | None = None

def run_analytics_plugins(
    plan: AnalyticsPluginExecutionPlan,
    run_context: AnalyticsRunContext,
    enable_middleware: bool = True,
) -> AnalyticsRunReport:
    """Executes all plugins and returns a telemetry report."""
```

### 9.3 Output Report

```python
@dataclass(frozen=True)
class AnalyticsRunReport:
    repo: str
    commit: str
    run_id: str
    scope: AnalyticsScope
    records: tuple[AnalyticsRunRecord, ...]  # Per-plugin execution records
    plan: AnalyticsPlanInfo                   # Ordered steps, skipped, dep graph
    tags: Mapping[str, str]

@dataclass(frozen=True)
class AnalyticsRunRecord:
    name: str
    kind: str
    status: AnalyticsStatus  # "succeeded" | "failed" | "skipped"
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    attempts: int
    partial: bool
    error: str | None
    meta: Mapping[str, Any]  # row_counts, contract results, etc.
```

---

## 10. Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SOURCE DATA                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  core.goids  │  │ core.modules │  │ core.calls   │  ...          │
│  │ (functions)  │  │  (imports)   │  │ (references) │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
└─────────┼──────────────────┼──────────────────┼─────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RESOURCE PROVIDERS                                │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │
│  │ CatalogProvider│  │ GraphProvider  │  │  AstProvider   │         │
│  │ (lazy load)    │  │ (lazy graphs)  │  │ (lazy AST)     │         │
│  └────────────────┘  └────────────────┘  └────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PLUGIN EXECUTION                                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Plugin Registry → Plan → Executor → Middleware Chain          │   │
│  │   ↓                                                           │   │
│  │ functions.metrics → graph_metrics → subsystems → profiles → … │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PURE COMPUTATION                                  │
│  compute/functions/    compute/graphs/    compute/profiles/          │
│  (complexity, types)   (centrality)       (aggregation)              │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PERSISTENCE (ADAPTERS)                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ DuckDB Tables in `analytics.*` schema                          │ │
│  │ • function_metrics  • graph_metrics_*  • subsystems           │ │
│  │ • coverage_*        • profiles_*       • semantic_roles        │ │
│  │ • hotspots          • entrypoints      • data_models          │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DOWNSTREAM CONSUMERS                              │
│  • Serving layer (HTTP APIs, search, recommendations)                │
│  • Graph plugins (further graph analysis)                            │
│  • Reports & dashboards                                              │
│  • MCP server (AI agent access)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 11. Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **Plugin-based architecture** | Enables independent development, testing, and deployment of analytics features |
| **Capability-based dependencies** | Loose coupling; plugins don't need to know about each other directly |
| **Lazy resource loading** | Graphs and AST maps are expensive; load only when needed |
| **Pure compute layer** | Testable, parallelizable, no hidden state |
| **Contract validation** | Data quality assurance; catches schema drift and computation bugs |
| **Middleware chain** | Cross-cutting concerns (logging, metrics) without polluting plugin code |
| **DuckDB as storage** | Fast analytical queries, columnar storage, embedded (no external service) |
| **Typed configs** | Type-safe configuration access prevents runtime errors |
| **Graph runtime pooling** | Avoids redundant graph loading across plugins in same run |
| **Immutable dataclasses** | Thread-safe, hashable results for caching and comparison |

---

## 12. Extension Points

To add a new analytics capability:

### Step 1: Create Pure Computation

```python
# compute/<domain>/<feature>.py
@dataclass(frozen=True)
class MyMetrics:
    """Immutable result container."""
    metric_a: float
    metric_b: int

def compute_my_metrics(data: SomeInput) -> MyMetrics:
    """Pure function - no I/O."""
    return MyMetrics(metric_a=..., metric_b=...)
```

### Step 2: Create Adapter

```python
# adapters/<domain>.py
class MyAdapter(BatchAdapter["MyRow"]):
    @property
    def table_name(self) -> str:
        return "analytics.my_table"
    
    def load(self) -> Iterator[MyRow]: ...
    def persist(self, rows: Sequence[MyRow]) -> int: ...
```

### Step 3: Create Plugin

```python
# core/plugins/<domain>/<feature>.py
@dataclass
class MyPlugin(ConfiguredTableWriterPlugin[MyStepConfig]):
    plugin_name: ClassVar[str] = "my.feature"
    plugin_stage: ClassVar[PluginStage] = "function"
    output_tables: ClassVar[tuple[str, ...]] = ("analytics.my_table",)
    config_type: ClassVar[type[MyStepConfig]] = MyStepConfig
    
    def compute(self, ctx: PluginExecutionContext) -> Mapping[str, int] | None:
        # Orchestrate: load resources → compute → persist
        ...
```

### Step 4: Register Plugin

```python
# core/plugins/registration.py
MY_PLUGIN = MyPlugin()
ALL_PLUGINS = (..., MY_PLUGIN)
```

### Step 5: Define Configuration (if needed)

```python
# config/steps_analytics.py
@dataclass(frozen=True)
class MyStepConfig:
    snapshot: SnapshotRef
    option_a: bool = True
    threshold: float = 0.5
```

---

## 13. Testing Strategy

### Unit Tests for Compute Layer

```python
def test_complexity_simple_function():
    source = "def f(x): return x if x > 0 else -x"
    func = ast.parse(source).body[0]
    metrics = compute_complexity(func)
    assert metrics.cyclomatic == 2
    assert metrics.complexity_bucket == "low"
```

### Integration Tests with Test Harness

```python
def test_my_plugin(tmp_path, analytics_gateway):
    result = (
        PluginTestHarness.for_plugin(MyPlugin())
        .with_gateway(analytics_gateway)
        .with_config(MyStepConfig(...))
        .execute()
    )
    assert result.success
    assert result.row_counts["analytics.my_table"] > 0
```

### Contract Validation Tests

```python
def test_output_contracts():
    contracts = build_plugin_output_contracts(MyPlugin())
    assert len(contracts) == 1
    assert contracts[0].contracts[0].table == "analytics.my_table"
```

---

## 14. Performance Considerations

### Graph Runtime Pooling

```python
pool = GraphRuntimePool(max_size=4, ttl_seconds=300)
runtime = pool.get(gateway, options)  # Reuses cached runtime
```

### Disk Caching for Graphs

```python
options = GraphRuntimeOptions(
    snapshot=snapshot,
    graph_cache_dir=Path("/tmp/graph_cache"),
)
# Graphs serialized to JSON for subsequent runs
```

### Batch Processing

- Adapters support `persist_batch()` with delete-before-insert
- Row-level parallelism within plugins where safe
- Stage-level parallelism for independent plugin groups

---

## 15. Observability

### Logging

```python
log.info(
    "graph_runtime.ensure.call_graph nodes=%d edges=%d cache_hit=%s",
    node_count, edge_count, cache_hit
)
```

### Metrics (via Middleware)

- `analytics_plugin_duration_seconds` (histogram)
- `analytics_plugin_success_total` (counter)
- `analytics_plugin_failure_total` (counter)

### Tracing (via Middleware)

- OpenTelemetry spans for each plugin execution
- Correlation IDs propagated through context

---

## 16. File Structure Reference

```
src/codeintel/analytics/
├── __init__.py
├── core/
│   ├── base.py              # Base plugin classes
│   ├── contracts.py         # Output contract validation
│   ├── execution_context.py # Plugin execution context
│   ├── executor.py          # Plugin executor
│   ├── pipeline_bridge.py   # Pipeline integration
│   ├── plugin_protocol.py   # Core protocol definitions
│   ├── registry.py          # Plugin registry
│   ├── traits.py            # Mixin traits
│   └── plugins/             # All registered plugins
│       ├── registration.py  # Plugin registration
│       ├── middleware/      # Logging, metrics, tracing
│       ├── functions/       # Function-level plugins
│       ├── graphs/          # Graph-based plugins
│       ├── coverage/        # Coverage plugins
│       └── ...
├── resources/
│   ├── protocol.py          # ResourceProvider protocol
│   ├── registry.py          # ResourceRegistry
│   ├── graphs.py            # GraphProvider
│   ├── catalog.py           # CatalogProvider
│   ├── asts.py              # AstProvider
│   └── features.py          # FeaturesProvider
├── compute/
│   ├── functions/           # Pure function computations
│   ├── graphs/              # Pure graph computations
│   ├── profiles/            # Pure profile computations
│   └── ...
├── adapters/
│   ├── base.py              # Adapter base classes
│   ├── functions.py         # Function adapters
│   ├── profiles.py          # Profile adapters
│   └── ...
├── graph_runtime.py         # GraphRuntime + pooling
├── runtime_manifest.py      # Run report schemas
└── datasets.py              # Dataset contracts
```

---

---

# Part II: Ingestion Module

## Executive Summary

The `ingestion` module in CodeIntel is the **data acquisition layer** responsible for parsing repositories, extracting structural information, and populating the core tables that feed into analytics. It follows a **port-adapter architecture** with a plugin-based orchestration layer, enabling clean separation between domain logic (steps), external interfaces (ports), and concrete implementations (adapters).

---

## 17. Ingestion Architectural Overview

The ingestion system is organized into **six primary layers**:

```
┌─────────────────────────────────────────────────────────────────┐
│                 Recipe / Orchestration Layer                     │
│           (recipes/executor.py, ingest_runs.py)                  │
├─────────────────────────────────────────────────────────────────┤
│                      Plugin Layer                                │
│         (plugins/*, registry.py, protocol.py)                    │
├─────────────────────────────────────────────────────────────────┤
│                 Resource Provider Layer                          │
│         (resources/modules.py, tracker.py, tools.py)             │
├─────────────────────────────────────────────────────────────────┤
│              Pure Domain Steps Layer                             │
│      (steps/ast_extract.py, steps/scip_ingest.py, etc.)          │
├─────────────────────────────────────────────────────────────────┤
│                    Ports Layer (Interfaces)                      │
│         (ports/discovery.py, storage.py, tools.py)               │
├─────────────────────────────────────────────────────────────────┤
│                 Adapters Layer (Implementations)                 │
│   (adapters/duckdb_storage.py, filesystem_discovery.py, etc.)    │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Port-Adapter Pattern**: Clean interfaces (ports) with swappable implementations (adapters)
2. **Incremental Processing**: Smart change detection avoids re-processing unchanged files
3. **Plugin Composition**: Declarative recipes compose plugins into execution pipelines
4. **Tool Integration**: External tools (scip-python, pyright, pytest) are abstracted behind ports
5. **Staged Execution**: Plugins execute in ordered stages (scan → parse → index → enrich → validate)

---

## 18. Core Component Deep Dive

### 18.1 Plugin Protocol (`plugins/protocol.py`)

The foundation is the `IngestPluginProtocol`, defining what every ingestion plugin must provide:

```python
@runtime_checkable
class IngestPluginProtocol(Protocol):
    @property
    def metadata(self) -> IngestPluginMetadata: ...
    def execute(self, ctx: IngestExecutionContext) -> IngestPluginResult: ...
```

**Key Metadata Fields** (from `IngestPluginMetadata`):

| Field | Purpose |
|-------|---------|
| `name` | Stable identifier (e.g., `"ast_extract"`) |
| `stage` | Execution grouping (`scan`, `parse`, `index`, `enrich`, `validate`) |
| `severity` | Error handling (`fatal`, `soft_fail`, `skip_on_error`) |
| `depends_on` | Explicit plugin dependencies |
| `provides` | Capabilities this plugin provides |
| `requires` | Capabilities this plugin needs from others |
| `produces_tables` | Tables this plugin writes to |
| `tool_dependencies` | External tools required (e.g., `"scip"`, `"pyright"`) |
| `supports_incremental` | Whether incremental ingestion is supported |
| `resource_hints` | Runtime budgets (max runtime, memory, CPU/IO intensive flags) |
| `isolation_kind` | Process/thread/none isolation for execution |

**Available Plugin Stages**:

```python
IngestStage = Literal[
    "scan",      # Repository scanning, module discovery
    "parse",     # AST/CST parsing, source extraction
    "index",     # SCIP indexing, symbol analysis
    "enrich",    # Type inference, coverage, diagnostics
    "validate",  # Contract validation, consistency checks
]
```

**Error Severity Levels**:

```python
IngestSeverity = Literal[
    "fatal",         # Stop pipeline on failure
    "soft_fail",     # Log error but continue
    "skip_on_error", # Skip silently on error
]
```

### 18.2 Execution Context (`core/execution_context.py`)

The `IngestExecutionContext` provides plugins access to all necessary resources:

```python
@dataclass
class IngestExecutionContext:
    gateway: StorageGateway        # DuckDB access
    snapshot: SnapshotRef          # repo/commit/repo_root
    paths: BuildPaths              # Build directory configuration
    tools: ToolsConfig             # Tool paths (scip, pyright, etc.)
    code_profile: ScanProfile      # File patterns for code scanning
    config_profile: ScanProfile    # File patterns for config scanning
    resources: ResourceRegistry    # Lazy resource providers
    scratch: IngestRuntimeScratch  # Inter-plugin communication
    configs: dict[type, object]    # Typed configuration instances
    plugin_name: str | None        # Current executing plugin
    run_id: str | None             # Unique execution identifier
```

**Resource Access Pattern**:

```python
# Typed resource access
modules_provider = ctx.require(ModuleProvider)
modules = modules_provider.get()

# Config access
config = ctx.get_config(AstExtractConfig)

# Inter-plugin communication via scratch
ctx.scratch.declare("discovered_modules", modules)
prior_data = ctx.scratch.consume("change_tracker")
```

### 18.3 Base Plugin Classes (`core/base.py`)

A hierarchy of base classes minimizes boilerplate:

```
BaseIngestPlugin (abstract)
├── TableWriterIngestPlugin (auto row-count tracking)
├── ConfiguredIngestPlugin[TConfig] (typed config injection)
├── ToolDependentIngestPlugin (validates tool availability)
├── TrackerRequiringPlugin (requires change tracker)
└── Composite classes:
    └── ConfiguredTableWriterPlugin[TConfig]
```

**Example Plugin (SCIP Indexer)**:

```python
@dataclass
class ScipIngestPlugin(
    TrackerRequiringPlugin,
    ToolDependentIngestPlugin,
    WithDependencyData,
    WithToolDependencies,
):
    """Run scip-python and persist symbols and GOID crosswalk."""

    # Identification
    plugin_name: ClassVar[str] = "scip_ingest"
    plugin_stage: ClassVar[IngestStage] = "index"
    plugin_version: ClassVar[str] = "2.0.0"

    # Output tables
    output_tables: ClassVar[tuple[str, ...]] = (
        "index.scip",
        "core.scip_symbols",
        "core.goid_crosswalk",
    )

    # Dependencies
    depends_on: ClassVar[tuple[str, ...]] = ("repo_scan",)
    requires: ClassVar[tuple[str, ...]] = ("change_tracker",)
    tool_dependencies: ClassVar[tuple[str, ...]] = ("scip",)
    supports_incremental: ClassVar[bool] = True

    # Resource hints
    resource_hints: ClassVar[IngestResourceHints] = IngestResourceHints(
        cpu_intensive=True,
        io_intensive=True,
        max_runtime_ms=300000,
    )

    def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int] | None:
        """Execute SCIP indexing."""
        # Get tool service from provider
        tools_provider = ctx.require(ToolsProvider)
        service = tools_provider.get()

        # Create adapters
        storage = DuckDBStorageAdapter(ctx.gateway)
        tool = ToolRunnerAdapter(service)

        # Execute step
        step = ScipIngestStep(storage=storage, tools=tool)
        result = asyncio.run(step.execute_async(modules, config))

        return None  # Auto-compute row counts
```

### 18.4 Plugin Registry (`plugins/registry.py`)

The `IngestPluginRegistry` manages plugin discovery and dependency resolution:

1. **Registration**: Plugins register by name, with indexes by capability, stage, and produced tables
2. **Planning**: `registry.plan(options)` performs:
   - Selection based on enabled/disabled lists
   - Tool availability checking
   - Dependency resolution (explicit + capability-based)
   - Topological sorting with cycle detection
3. **Entry-Point Discovery**: Auto-discovers plugins from `codeintel.ingest_plugins` entry points

**Plan Options**:

```python
@dataclass(frozen=True)
class PlanOptions:
    plugin_names: Sequence[str] | None = None     # Explicit plugin list
    enabled: Sequence[str] | None = None          # Override enabled plugins
    disabled: Sequence[str] | None = None         # Plugins to exclude
    defaults: Sequence[str] | None = None         # Default plugins
    check_tools: bool = False                     # Verify tool availability
    available_tools: Sequence[str] | None = None  # Available tools
```

**Execution Plan**:

```python
@dataclass(frozen=True)
class IngestPluginPlan:
    plugins: tuple[IngestPluginProtocol, ...]   # Ordered plugins to execute
    plan_id: str                                 # Unique plan identifier
    skipped_plugins: tuple[IngestPluginSkip, ...] # Plugins that were skipped
    dep_graph: dict[str, tuple[str, ...]]        # Dependency graph
```

---

## 19. Port-Adapter Architecture

The ingestion system uses the **port-adapter pattern** to separate interfaces from implementations.

### 19.1 Ports (Interfaces)

| Port | Purpose | Methods |
|------|---------|---------|
| `ModuleDiscoveryPort` | Source file enumeration | `discover_modules()`, `read_module_source()` |
| `IngestStoragePort` | Database operations | `write_batch()`, `delete_by_paths()`, `execute_query()` |
| `IngestToolPort` | External tool execution | `run_scip()`, `run_pyright()`, `run_pytest()` |
| `ChangeDetectionPort` | Incremental change tracking | `compute_changes()` |

**Module Discovery Port**:

```python
@runtime_checkable
class ModuleDiscoveryPort(Protocol):
    def discover_modules(
        self,
        repo_root: Path,
        profile: ScanProfile,
    ) -> Sequence[ModuleRecord]: ...

    def read_module_source(self, record: ModuleRecord) -> str | None: ...

    def file_exists(self, path: Path) -> bool: ...
```

**Storage Port**:

```python
class IngestStoragePort(Protocol):
    def write_batch(
        self,
        table_key: str,
        rows: Sequence[Sequence[object]],
        *,
        scope: str | None = None,
    ) -> BatchResult: ...

    def delete_by_paths(
        self,
        table_key: str,
        paths: Sequence[str],
        *,
        path_column: str = "rel_path",
    ) -> int: ...

    def execute_query(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> QueryResult: ...
```

### 19.2 Adapters (Implementations)

| Adapter | Port | Implementation Details |
|---------|------|------------------------|
| `DuckDBStorageAdapter` | `IngestStoragePort` | Macro-based batch inserts, prepared statements fallback |
| `FilesystemDiscoveryAdapter` | `ModuleDiscoveryPort` | Filesystem scanning with glob patterns |
| `HashChangeDetectionAdapter` | `ChangeDetectionPort` | Blake2b hash-based change detection |
| `ToolRunnerAdapter` | `IngestToolPort` | External process execution via `ToolService` |

**DuckDB Storage Adapter**:

```python
class DuckDBStorageAdapter:
    """DuckDB storage adapter with macro-based batch inserts."""

    def write_batch(
        self,
        table_key: str,
        rows: Sequence[Sequence[object]],
        *,
        scope: str | None = None,
    ) -> BatchResult:
        # Use macro-based insertion for large batches
        if len(rows) > SMALL_BATCH_THRESHOLD:
            return self._ingest_via_macro(table_key, rows)
        # Fall back to prepared statements for small batches
        return self._fallback_prepared_insert(table_key, rows)
```

---

## 20. Change Tracker & Incremental Ingestion

The `ChangeTracker` provides unified change detection for efficient incremental processing.

### 20.1 Core Components

```python
@dataclass
class ChangeTracker:
    """Single source of truth for change detection across ingest steps."""
    
    gateway: StorageGateway           # Storage access
    change_request: ChangeRequest     # Request parameters
    modules: Sequence[ModuleRecord]   # All modules in scope
    change_set: ChangeSet             # Computed changes
    policy: IncrementalIngestPolicy   # Behavior tuning
```

**Change Set**:

```python
@dataclass(frozen=True)
class ChangeSet:
    added: Sequence[ModuleRecord]     # New files
    modified: Sequence[ModuleRecord]  # Changed files
    deleted: Sequence[ModuleRecord]   # Removed files
    unchanged: Sequence[ModuleRecord] # No changes detected
```

**Incremental Policy**:

```python
@dataclass(frozen=True)
class IncrementalIngestPolicy:
    max_changed_ratio: float = 0.7      # Trigger full rebuild if exceeded
    max_deleted_ratio: float = 0.7      # Trigger full rebuild if exceeded
    min_total_modules_for_ratio: int = 20
    log_every: int = 100                # Progress logging interval
    flush_every: int = 500              # Batch flush interval
```

### 20.2 Dataset Views

Plugins request dataset-specific views of changes:

```python
def view_for_dataset(
    self,
    *,
    dataset_name: str,
    module_filter: ModuleFilter | None = None,
) -> ChangeTrackerDatasetView:
    """Compute dataset-scoped changes with full rebuild policy applied."""
```

**Dataset View**:

```python
class ChangeTrackerDatasetView(NamedTuple):
    to_reparse: list[ModuleRecord]        # Modules needing processing
    deleted_paths: list[str]              # Paths to delete
    total_modules_considered: int         # Total scope size
    changed_modules_count: int            # Number changed
    deleted_modules_count: int            # Number deleted
    use_full_rebuild: bool                # Whether full rebuild is active
```

### 20.3 Incremental Ingest Protocol

For datasets supporting incremental updates:

```python
@runtime_checkable
class IncrementalIngestOps(Protocol[RowT]):
    """Operations required to incrementally ingest a dataset."""
    
    dataset_name: ClassVar[str]

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool: ...

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None: ...

    @staticmethod
    def process_module(module: ModuleRecord) -> Iterable[RowT]: ...

    def insert_rows(self, gateway: StorageGateway, rows: Sequence[RowT]) -> None: ...
```

---

## 21. Resource Provider Layer (`resources/`)

### 21.1 Protocol & Base Classes

```python
@runtime_checkable
class ResourceProvider[T_co](Protocol):
    @property
    def is_loaded(self) -> bool: ...
    @property
    def resource_name(self) -> str: ...
    def get(self) -> T_co: ...            # Load on first access
    def get_or_none(self) -> T_co | None: ...
    def invalidate(self) -> None: ...     # Force reload

class LazyResource[T](ABC):
    """Base class with standard lazy loading + caching."""
    def _load(self) -> T: ...             # Subclasses implement
    def set_preloaded(self, resource: T) -> None: ...
```

### 21.2 Key Providers

| Provider | What It Provides | When Used |
|----------|------------------|-----------|
| `ModuleProvider` | Discovered source modules | All parsing plugins |
| `TrackerProvider` | Change tracker instance | Incremental ingestion |
| `ToolsProvider` | Tool service for external execution | SCIP, pyright, pytest |
| `StorageProvider` | Storage adapter access | All data persistence |

---

## 22. Steps Layer (Pure Domain Logic)

Steps contain the pure domain logic, operating on ports rather than concrete implementations.

### 22.1 Available Steps

| Step | Tables | Purpose |
|------|--------|---------|
| `RepoScanStep` | `core.modules`, `core.file_state` | Repository scanning, change detection |
| `AstExtractStep` | `core.ast_nodes`, `core.ast_metrics` | Python AST parsing |
| `CstExtractStep` | `core.cst_nodes` | LibCST concrete syntax tree extraction |
| `ScipIngestStep` | `core.scip_symbols`, `core.goid_crosswalk` | SCIP symbol indexing |
| `TypingIngestStep` | `analytics.typedness`, `analytics.static_diagnostics` | Type annotation analysis |
| `CoverageIngestStep` | `analytics.coverage_lines`, `analytics.test_coverage_edges` | Coverage data ingestion |
| `TestsIngestStep` | `analytics.test_catalog` | Test results ingestion |
| `DocstringsExtractStep` | `core.docstrings` | Docstring parsing |
| `ConfigIngestStep` | `core.config_entries` | Configuration file flattening |

### 22.2 Step Result

```python
@dataclass
class StepResult:
    rows_written: int = 0
    table_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def success(self) -> bool:
        return not self.errors and not self.skipped
```

---

## 23. Recipe Composition (`recipes/`)

Recipes provide declarative pipeline composition.

### 23.1 Recipe Definition

```python
@dataclass(frozen=True)
class IngestRecipe:
    """Declarative recipe for ingestion pipeline composition."""
    
    name: str                                  # Unique identifier
    description: str = ""                      # Human-readable description
    version: str = "1.0.0"                     # Recipe version
    stages: tuple[RecipeStage, ...] = ()       # Execution stages
    options: RecipeOptions = RecipeOptions()   # Global options
    disabled_plugins: tuple[str, ...] = ()     # Plugins to exclude
    enabled_plugins: tuple[str, ...] | None = None  # Override defaults
    includes: tuple[str, ...] = ()             # Other recipes to include
    tags: tuple[str, ...] = ()                 # Classification tags
```

**Recipe Stage**:

```python
@dataclass(frozen=True)
class RecipeStage:
    name: str                          # Stage identifier
    plugins: tuple[str, ...]           # Plugins to execute
    parallel: bool = False             # Allow parallel execution
    required: bool = True              # Must succeed to continue
    timeout_s: int | None = None       # Maximum execution time
    description: str = ""              # Stage description
```

### 23.2 Recipe Options

```python
@dataclass(frozen=True)
class RecipeOptions:
    enable_incremental: bool = True      # Enable incremental ingestion
    enable_contracts: bool = True        # Validate output contracts
    max_parallel_plugins: int = 4        # Parallel plugin limit
    fail_fast: bool = True               # Stop on first failure
    continue_on_soft_fail: bool = True   # Continue after soft failures
    dry_run: bool = False                # Validate without executing
```

### 23.3 Recipe DSL

```python
from codeintel.ingestion.recipes import recipe, stage, StageSpec, RecipeSpec

# Define a custom recipe
my_recipe = recipe(
    "my_pipeline",
    stages=[
        stage("scan", ["repo_scan"]),
        stage("parse", ["ast_extract", "cst_extract"], StageSpec(parallel=True)),
        stage("index", ["scip_ingest"]),
        stage("enrich", ["typing_ingest", "coverage_ingest"]),
    ],
    spec=RecipeSpec(
        description="Custom ingestion pipeline",
        options=RecipeOptions(enable_incremental=True),
    ),
)
```

### 23.4 Builtin Recipes

```python
DEFAULT_INGEST_PLUGINS: tuple[str, ...] = (
    "repo_scan",
    "scip_ingest",
    "cst_extract",
    "ast_extract",
    "typing_ingest",
    "coverage_ingest",
    "tests_ingest",
    "docstrings_ingest",
    "config_ingest",
)
```

---

## 24. Ingest Runs & Observability

### 24.1 Ingest Run Record

```python
@dataclass
class IngestRun:
    """Structured record describing a single ingestion step execution."""
    
    run_id: str
    repo: str
    commit: str
    step: str
    datasets: tuple[str, ...]
    mode: IngestRunMode           # FULL | INCREMENTAL | UNKNOWN
    started_at: datetime
    finished_at: datetime | None
    duration_s: float | None

    rows_before: Mapping[str, int]
    rows_after: Mapping[str, int]
    rows_inserted: int
    rows_deleted: int

    status: IngestRunStatus       # OK | SKIPPED | ERROR
    error_kind: str | None
    error_message: str | None

    # Incremental metrics
    modules_total: int | None
    modules_changed: int | None
    modules_deleted: int | None
    use_full_rebuild: bool | None
```

### 24.2 Sinks

| Sink | Purpose |
|------|---------|
| `JsonlIngestRunSink` | Append JSONL records to disk |
| `DuckDBIngestRunSink` | Persist to `core.ingest_runs` table |
| `OtelIngestRunSink` | Emit OpenTelemetry metrics |
| `MultiSink` | Fan out to multiple sinks |

### 24.3 Error Classification

```python
def classify_error(exc: BaseException) -> str:
    """Map exceptions into coarse error kinds for dashboards."""
    if isinstance(exc, ToolNotFoundError):
        return "tool_not_found"
    if isinstance(exc, ToolExecutionError):
        if "timeout" in str(exc).lower():
            return "tool_timeout"
        return "tool_execution"
    if isinstance(exc, DuckDBError):
        return "db_error"
    if isinstance(exc, ValueError):
        return "parse_error"
    return exc.__class__.__name__
```

---

## 25. Registered Plugins & Output Data

### 25.1 All Registered Plugins

| Plugin | Stage | Description |
|--------|-------|-------------|
| `repo_scan` | scan | Repository scanning, module discovery, change detection |
| `ast_extract` | parse | Python AST parsing and metrics extraction |
| `cst_extract` | parse | LibCST concrete syntax tree extraction |
| `scip_ingest` | index | SCIP symbol indexing and GOID crosswalk |
| `typing_ingest` | enrich | Type annotation analysis via pyright |
| `coverage_ingest` | enrich | Coverage.py data ingestion |
| `tests_ingest` | enrich | Pytest JSON report ingestion |
| `docstrings_ingest` | enrich | Docstring parsing and persistence |
| `config_ingest` | enrich | Configuration file flattening |

### 25.2 Output Tables (Data Flow Destinations)

| Table | Plugin | Contents |
|-------|--------|----------|
| `core.modules` | `repo_scan` | Discovered source modules |
| `core.file_state` | `repo_scan` | File hashes for change detection |
| `core.ast_nodes` | `ast_extract` | Parsed AST nodes |
| `core.ast_metrics` | `ast_extract` | Basic AST-derived metrics |
| `core.cst_nodes` | `cst_extract` | LibCST syntax nodes |
| `core.scip_symbols` | `scip_ingest` | SCIP symbol information |
| `core.goid_crosswalk` | `scip_ingest` | Symbol to GOID mapping |
| `core.goids` | `scip_ingest` | Global object identifiers |
| `core.docstrings` | `docstrings_ingest` | Extracted docstrings |
| `analytics.typedness` | `typing_ingest` | Type annotation coverage |
| `analytics.static_diagnostics` | `typing_ingest` | Pyright diagnostics |
| `analytics.coverage_lines` | `coverage_ingest` | Line-level coverage |
| `analytics.test_coverage_edges` | `coverage_ingest` | Test-to-function mapping |
| `analytics.test_catalog` | `tests_ingest` | Test function catalog |

---

## 26. Ingestion Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SOURCE REPOSITORY                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Python Files │  │ Config Files │  │ Coverage Data│  ...          │
│  │   (*.py)     │  │ (*.yml,etc.) │  │ (.coverage)  │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
└─────────┼──────────────────┼──────────────────┼─────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MODULE DISCOVERY                                │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ FilesystemDiscoveryAdapter → ModuleRecord list                 │ │
│  │ (glob patterns, ScanProfile filtering)                         │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     CHANGE DETECTION                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ HashChangeDetectionAdapter → ChangeSet (added/modified/deleted)│ │
│  │ (Blake2b hashes, file_state comparison)                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PLUGIN EXECUTION                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Recipe → Plan → Executor (staged execution)                   │   │
│  │   ↓                                                           │   │
│  │ scan: repo_scan                                               │   │
│  │ parse: ast_extract, cst_extract                               │   │
│  │ index: scip_ingest                                            │   │
│  │ enrich: typing_ingest, coverage_ingest, tests_ingest, ...     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP EXECUTION                                   │
│  steps/ast_extract.py    steps/scip_ingest.py    steps/typing.py    │
│  (pure domain logic operating on ports)                              │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PERSISTENCE (ADAPTERS)                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ DuckDBStorageAdapter (macro-based batch inserts)               │ │
│  │ Tables in `core.*` schema:                                     │ │
│  │ • modules    • ast_nodes    • cst_nodes    • goids             │ │
│  │ • scip_symbols  • docstrings  • file_state  • goid_crosswalk   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Tables in `analytics.*` schema:                                │ │
│  │ • typedness  • static_diagnostics  • coverage_lines            │ │
│  │ • test_catalog  • test_coverage_edges                          │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   DOWNSTREAM CONSUMERS                               │
│  • Analytics module (function metrics, graph analysis)               │
│  • Graph construction (call graphs, import graphs)                   │
│  • Serving layer (search, symbol lookup)                             │
│  • MCP server (AI agent access)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 27. Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **Port-Adapter pattern** | Testability with fake filesystems, swappable storage backends |
| **Plugin-based architecture** | Independent development, testing, and deployment of ingestion features |
| **Incremental processing** | Performance - avoid re-processing unchanged files |
| **Staged execution** | Clear ordering constraints (can't index before parsing) |
| **Recipe composition** | Declarative pipelines, easy customization and extension |
| **External tool abstraction** | Swap SCIP/pyright implementations, handle failures gracefully |
| **Macro-based batch inserts** | DuckDB performance optimization for large datasets |
| **Change tracker unification** | Single source of truth for incremental decisions across plugins |
| **Runtime scratch space** | Inter-plugin communication without database round-trips |

---

## 28. Extension Points

To add a new ingestion capability:

### Step 1: Create Step (Pure Domain Logic)

```python
# steps/my_extract.py
class MyExtractStep:
    def __init__(self, storage: IngestStoragePort) -> None:
        self._storage = storage

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        config: MyExtractConfig,
    ) -> StepResult:
        """Execute extraction using ports (no concrete dependencies)."""
        rows = []
        for module in modules:
            # Process each module
            rows.extend(self._process_module(module))
        
        result = self._storage.write_batch("core.my_table", rows)
        return StepResult.ok(rows_written=result.rows_written)
```

### Step 2: Create Plugin

```python
# plugins/my_extract.py
@dataclass
class MyExtractPlugin(ConfiguredTableWriterPlugin[MyExtractConfig]):
    plugin_name: ClassVar[str] = "my_extract"
    plugin_stage: ClassVar[IngestStage] = "parse"
    output_tables: ClassVar[tuple[str, ...]] = ("core.my_table",)
    config_type: ClassVar[type[MyExtractConfig]] = MyExtractConfig
    depends_on: ClassVar[tuple[str, ...]] = ("repo_scan",)

    def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int] | None:
        # Get modules from provider
        modules_provider = ctx.require(ModuleProvider)
        modules = modules_provider.get()

        # Create adapter and step
        storage = DuckDBStorageAdapter(ctx.gateway)
        step = MyExtractStep(storage=storage)
        
        # Execute
        result = step.execute(modules, self.config)
        return result.table_counts
```

### Step 3: Register Plugin

```python
# plugins/registry.py
def _register_builtin_plugins(self) -> None:
    from codeintel.ingestion.plugins.my_extract import MyExtractPlugin
    # ... existing plugins ...
    plugins.append(MyExtractPlugin())
```

### Step 4: Add to Default Recipe (Optional)

```python
# recipes/builtin.py
DEFAULT_INGEST_PLUGINS = (
    "repo_scan",
    # ... existing ...
    "my_extract",  # Add your plugin
)
```

---

## 29. Testing Strategy

### Unit Tests for Steps

```python
def test_my_extract_step():
    # Create fake storage port
    storage = FakeStorageAdapter()
    step = MyExtractStep(storage=storage)
    
    modules = [ModuleRecord(...), ...]
    result = step.execute(modules, MyExtractConfig(...))
    
    assert result.success
    assert storage.written_rows["core.my_table"] > 0
```

### Integration Tests with Harness

```python
def test_my_extract_plugin(tmp_path, ingest_gateway):
    result = (
        IngestPluginTestHarness.for_plugin(MyExtractPlugin())
        .with_gateway(ingest_gateway)
        .with_modules([ModuleRecord(...)])
        .execute()
    )
    assert result.success
    assert result.row_counts["core.my_table"] > 0
```

### Incremental Ingest Tests

```python
def test_incremental_detects_changes(tmp_path, gateway):
    # First run - full ingest
    tracker = ChangeTracker.create(gateway, request, modules)
    assert tracker.change_set.added == modules
    
    # Second run - no changes
    tracker2 = ChangeTracker.create(gateway, request, modules)
    assert len(tracker2.change_set.added) == 0
    assert len(tracker2.change_set.unchanged) == len(modules)
```

---

## 30. File Structure Reference

```
src/codeintel/ingestion/
├── __init__.py              # Public API exports
├── change_tracker.py        # Unified change detection
├── common.py                # Shared utilities
├── ingest_runs.py           # Run record management
├── ingest_service.py        # Service facade
├── tool_service.py          # External tool management
├── core/
│   ├── base.py              # Base plugin classes
│   ├── execution_context.py # Plugin execution context
│   ├── traits.py            # Mixin traits
│   └── middleware/          # Logging, metrics, tracing
├── plugins/
│   ├── protocol.py          # Plugin protocol definitions
│   ├── registry.py          # Plugin registry
│   ├── contracts.py         # Output contracts
│   ├── config_factory.py    # Config building
│   ├── repo_scan.py         # Repository scanner plugin
│   ├── ast_extract.py       # AST extraction plugin
│   ├── cst_extract.py       # CST extraction plugin
│   ├── scip_plugin.py       # SCIP indexer plugin
│   ├── typing_plugin.py     # Type analysis plugin
│   ├── coverage_plugin.py   # Coverage plugin
│   ├── tests_plugin.py      # Test results plugin
│   ├── docstrings_plugin.py # Docstrings plugin
│   └── config_plugin.py     # Config files plugin
├── steps/
│   ├── base.py              # Step base types
│   ├── repo_scan.py         # Repository scanning step
│   ├── ast_extract.py       # AST extraction step
│   ├── cst_extract.py       # CST extraction step
│   ├── scip_ingest.py       # SCIP indexing step
│   ├── typing_ingest.py     # Type analysis step
│   ├── coverage_ingest.py   # Coverage step
│   ├── tests_ingest.py      # Test results step
│   ├── docstrings_extract.py# Docstrings step
│   └── config_ingest.py     # Config files step
├── ports/
│   ├── discovery.py         # Module discovery port
│   ├── storage.py           # Storage port
│   ├── tools.py             # Tool execution port
│   └── change_detection.py  # Change detection port
├── adapters/
│   ├── duckdb_storage.py    # DuckDB storage adapter
│   ├── filesystem_discovery.py # Filesystem adapter
│   ├── hash_change_detection.py # Hash-based change detection
│   └── tool_runner.py       # Tool execution adapter
├── recipes/
│   ├── dsl.py               # Recipe DSL definitions
│   ├── executor.py          # Recipe executor
│   └── builtin.py           # Built-in recipes
├── resources/
│   ├── protocol.py          # Resource provider protocol
│   ├── registry.py          # Resource registry
│   ├── modules.py           # Module provider
│   ├── tracker.py           # Tracker provider
│   └── tools.py             # Tools provider
└── infrastructure_utilities/
    ├── source_scanner.py    # File discovery utilities
    ├── tool_runner.py       # Tool execution utilities
    ├── workers.py           # Worker pool configuration
    └── paths.py             # Path utilities
```

---

---

# Part III: Graphs Module

## Graphs Executive Summary

The `graphs` module in CodeIntel is the **graph construction and analysis layer** responsible for building code relationship graphs (call graphs, import graphs, symbol graphs, control/data flow graphs) and computing graph-based metrics. It follows a **hexagonal architecture** with ports, adapters, and pure compute functions, enabling clean separation between I/O operations and stateless computation.

---

## 31. Graphs Architectural Overview

The graphs system is organized into **seven primary layers**:

```
┌─────────────────────────────────────────────────────────────────┐
│                 Recipe / Orchestration Layer                     │
│             (recipes/executor.py, recipes/dsl.py)                │
├─────────────────────────────────────────────────────────────────┤
│                      Plugin Layer                                │
│     (plugins/builders/*, plugins/metrics/*, plugins/validation/) │
├─────────────────────────────────────────────────────────────────┤
│                 Resource Provider Layer                          │
│        (resources/storage.py, graphs.py, catalog.py)             │
├─────────────────────────────────────────────────────────────────┤
│              Pure Compute Layer (Stateless)                      │
│        (compute/callgraph.py, compute/metrics/*)                 │
├─────────────────────────────────────────────────────────────────┤
│                    Ports Layer (Interfaces)                      │
│         (ports/storage.py, parsing.py, catalog.py)               │
├─────────────────────────────────────────────────────────────────┤
│                 Adapters Layer (Implementations)                 │
│  (adapters/callgraph_persistence.py, duckdb_storage.py, etc.)    │
├─────────────────────────────────────────────────────────────────┤
│                    Graph Engine Layer                            │
│   (engine/nx_engine.py, engine/protocol.py, engine/cache.py)     │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Hexagonal Architecture**: Ports define interfaces; adapters provide implementations; compute is pure
2. **Three Plugin Kinds**: Builders (construct graphs), Metrics (analyze graphs), Validation (verify integrity)
3. **Graph Engine Abstraction**: Backend-agnostic interface supporting NetworkX (CPU/GPU)
4. **Staged Execution**: Builders run before metrics; metrics run before validation
5. **Recipe Composition**: Declarative pipelines compose plugins into workflows

---

## 32. Graphs Core Component Deep Dive

### 32.1 Graph Plugin Protocol (`core/protocol.py`)

The foundation is the `GraphPluginProtocol`, defining what every graph plugin must provide:

```python
@runtime_checkable
class GraphPluginProtocol(Protocol):
    @property
    def metadata(self) -> GraphPluginMetadata: ...
    def execute(self, ctx: GraphExecutionContext) -> GraphPluginResult: ...
```

**Key Metadata Fields** (from `GraphPluginMetadata`):

| Field | Purpose |
|-------|---------|
| `name` | Stable identifier (e.g., `"callgraph_builder"`) |
| `kind` | Plugin kind (`builder`, `metric`, `validation`) |
| `stage` | Execution grouping within kind |
| `severity` | Error handling (`fatal`, `soft_fail`, `skip_on_error`) |
| `depends_on` | Explicit plugin dependencies |
| `provides` | Capabilities this plugin provides |
| `requires` | Capabilities this plugin needs from others |
| `produces_tables` | DuckDB tables this plugin writes to |
| `produces_graphs` | `GraphKind` values this plugin builds (for builders) |
| `requires_graphs` | `GraphKind` values this plugin needs (for metrics) |
| `resource_hints` | Runtime budgets (max runtime, memory, CPU/IO intensive) |
| `cache_populates` | Cache keys this plugin populates |
| `cache_consumes` | Cache keys this plugin consumes |

**Plugin Kinds**:

```python
GraphPluginKind = Literal["builder", "metric", "validation"]
```

**Plugin Stages**:

```python
GraphPluginStage = Literal[
    # Builder stages
    "goid",        # GOID population
    "edges",       # Edge construction (call, import)
    "structure",   # Structural graphs (CFG, DFG)
    
    # Metric stages
    "core",        # Core centrality metrics
    "cfg",         # CFG-based metrics
    "dfg",         # DFG-based metrics
    "test",        # Test-related metrics
    "symbol",      # Symbol-based metrics
    "subsystem",   # Subsystem/architectural metrics
    "config",      # Config-related metrics
    "stats",       # General statistics
    
    # Validation stage
    "validation",  # Graph integrity checks
]
```

### 32.2 Graph Execution Context (`core/context.py`)

The `GraphExecutionContext` provides plugins access to resources via dependency injection:

```python
@dataclass
class GraphExecutionContext:
    snapshot: SnapshotRef              # repo/commit/repo_root
    resources: ResourceContainer       # DI container for resources
    _gateway: StorageGateway | None    # Direct storage access (fallback)
    _engine: GraphEngine | None        # Direct engine access (fallback)
    _catalog_provider: FunctionCatalogProvider | None
    paths: BuildPaths | None           # Build directory configuration
    scratch: GraphRuntimeScratch       # Inter-plugin communication
    options: object | None             # Plugin-specific options
    plugin_name: str | None            # Current executing plugin
    run_id: str | None                 # Unique execution identifier
    scope: GraphRunScope | None        # Scoping for incremental execution
```

**Resource Access Pattern**:

```python
# Typed resource access via DI container
storage = ctx.require(StorageResource)
gateway = storage.gateway

# Engine access
engine = ctx.engine
call_graph = engine.call_graph()

# Inter-plugin communication via scratch
ctx.scratch.declare("goid_map", mapping)
prior_data = ctx.scratch.consume("scip_data")
```

### 32.3 Graph Plugin Registry (`core/registry.py`)

The `GraphPluginRegistry` manages plugin discovery and dependency resolution:

1. **Registration**: Plugins register by name, with indexes by kind, stage, capability, and table
2. **Planning**: `registry.plan(plugin_names)` performs:
   - Selection based on enabled/disabled lists
   - Dependency resolution (explicit + capability-based)
   - Topological sorting with cycle detection
3. **Entry-Point Discovery**: Auto-discovers plugins from `codeintel.graph_plugins` entry points

**Execution Plan**:

```python
@dataclass(frozen=True)
class GraphPluginPlan:
    plugins: tuple[GraphPluginProtocol, ...]   # Ordered plugins to execute
    plan_id: str                                # Unique plan identifier
    skipped_plugins: tuple[GraphPluginSkip, ...] # Plugins that were skipped
    dep_graph: dict[str, tuple[str, ...]]       # Dependency graph
```

---

## 33. Graph Engine Layer

The `GraphEngine` protocol provides a backend-agnostic interface for building and caching graphs.

### 33.1 Graph Engine Protocol (`engine/protocol.py`)

```python
class GraphEngine(Protocol):
    gateway: StorageGateway

    @property
    def use_gpu(self) -> bool: ...

    def call_graph(self) -> nx.DiGraph: ...
    def import_graph(self) -> nx.DiGraph: ...
    def symbol_module_graph(self) -> nx.Graph: ...
    def symbol_function_graph(self) -> nx.Graph: ...
    def config_module_bipartite(self) -> nx.Graph: ...
    def test_function_bipartite(self) -> nx.Graph: ...

    @property
    def snapshot(self) -> SnapshotRef: ...
```

### 33.2 Graph Kinds (`engine/protocol.py`)

```python
class GraphKind(Flag):
    NONE = 0
    CALL_GRAPH = auto()              # Function call relationships
    IMPORT_GRAPH = auto()            # Module import dependencies
    CFG_GRAPH = auto()               # Control flow graphs
    SYMBOL_MODULE_GRAPH = auto()     # Symbol-to-module bipartite
    SYMBOL_FUNCTION_GRAPH = auto()   # Symbol-to-function bipartite
    CONFIG_MODULE_BIPARTITE = auto() # Config key to module mapping
    TEST_FUNCTION_BIPARTITE = auto() # Test to function coverage mapping
    SYMBOL = SYMBOL_MODULE_GRAPH | SYMBOL_FUNCTION_GRAPH
    ALL = (CALL_GRAPH | IMPORT_GRAPH | CFG_GRAPH | SYMBOL 
           | CONFIG_MODULE_BIPARTITE | TEST_FUNCTION_BIPARTITE)
```

### 33.3 NetworkX Engine (`engine/nx_engine.py`)

The `NxGraphEngine` is the primary implementation:

- **Lazy Loading**: Graphs loaded on first access, cached in memory
- **Disk Caching**: Optional JSON serialization for expensive computations
- **GPU Backend**: Optional `nx-cugraph` backend for NVIDIA GPU acceleration
- **Graph Views**: Filtered views without full graph copies

---

## 34. Graph Plugin Kinds

### 34.1 Builder Plugins

Builders construct graph structures from source data:

| Plugin | Stage | Produces | Description |
|--------|-------|----------|-------------|
| `goid_builder` | goid | GOIDs | Populate global object identifiers |
| `callgraph_builder` | edges | `CALL_GRAPH` | Build function call relationships |
| `import_graph_builder` | edges | `IMPORT_GRAPH` | Build module import dependencies |
| `cfg_dfg_builder` | structure | `CFG_GRAPH` | Build control/data flow graphs |
| `symbol_uses_builder` | edges | `SYMBOL_*_GRAPH` | Build symbol usage relationships |

### 34.2 Metric Plugins

Metrics analyze constructed graphs:

| Plugin | Stage | Requires | Description |
|--------|-------|----------|-------------|
| `core_graph_metrics` | core | `CALL_GRAPH` | PageRank, betweenness, closeness centrality |
| `graph_metrics_functions_ext` | core | `CALL_GRAPH` | Extended function-level metrics |
| `graph_metrics_modules_ext` | core | `IMPORT_GRAPH` | Extended module-level metrics |
| `cfg_metrics` | cfg | `CFG_GRAPH` | Control flow complexity metrics |
| `dfg_metrics` | dfg | `CFG_GRAPH` | Data flow analysis metrics |
| `symbol_graph_metrics_*` | symbol | `SYMBOL_*_GRAPH` | Symbol coupling metrics |
| `subsystem_graph_metrics` | subsystem | `CALL_GRAPH` | Architectural boundary metrics |
| `graph_stats` | stats | `ALL` | General graph statistics |

### 34.3 Validation Plugins

Validation checks graph integrity:

| Plugin | Stage | Description |
|--------|-------|-------------|
| `graph_validation` | validation | Comprehensive integrity checks |

---

## 35. Hexagonal Architecture in Graphs

### 35.1 Ports (Interfaces)

| Port | Purpose | Methods |
|------|---------|---------|
| `StoragePort` | Database operations | `write_batch()`, `execute_query()` |
| `ParsingPort` | Source code parsing | `parse_module()`, `parse_file()` |
| `CatalogPort` | Function metadata | `get_functions()`, `get_spans()` |
| `EnginePort` | Graph access | `call_graph()`, `import_graph()` |

### 35.2 Adapters (Implementations)

| Adapter | Port | Implementation |
|---------|------|----------------|
| `DuckDBStorageAdapter` | `StoragePort` | DuckDB-specific operations |
| `CallgraphPersistenceAdapter` | N/A | Call graph edge persistence |
| `LibCSTParsingAdapter` | `ParsingPort` | LibCST-based parsing |

### 35.3 Resource Providers

```python
@runtime_checkable
class ResourceProvider(Protocol[T_co]):
    @property
    def resource_name(self) -> str: ...
    def get(self) -> T_co: ...
    def invalidate(self) -> None: ...
```

| Provider | Resource | Purpose |
|----------|----------|---------|
| `StorageResource` | `StorageGateway` | Database access |
| `GraphResource` | `GraphEngine` | Graph engine access |
| `CatalogResource` | `FunctionCatalog` | Function metadata |

### 35.4 Pure Compute Layer (`compute/`)

All computations are pure functions with no I/O:

| Module | Functions | Purpose |
|--------|-----------|---------|
| `compute/callgraph.py` | `collect_aliases()`, `resolve_callee()` | Edge resolution |
| `compute/metrics/centrality.py` | `compute_pagerank()`, `compute_betweenness()` | Centrality metrics |
| `compute/metrics/components.py` | `compute_sccs()`, `detect_cycles()` | Graph structure analysis |
| `compute/metrics/coupling.py` | `compute_coupling_metrics()` | Module coupling |

**Example: Centrality Computation**:

```python
@dataclass(frozen=True)
class CentralityMetrics:
    """Immutable container for centrality metrics."""
    pagerank: float
    betweenness: float
    closeness: float
    in_degree: int
    out_degree: int
    degree: int

def compute_pagerank(
    graph: nx.DiGraph,
    alpha: float = 0.85,
    max_iter: int = 100,
) -> dict[Any, float]:
    """Pure function - no I/O, no side effects."""
    if graph.number_of_nodes() == 0:
        return {}
    return nx.pagerank(graph, alpha=alpha, max_iter=max_iter)
```

---

## 36. Graph Recipe Composition

Recipes provide declarative pipeline composition for graph workflows.

### 36.1 Recipe Definition

```python
@dataclass(frozen=True)
class GraphRecipe:
    name: str                          # Recipe identifier
    description: str                   # Human-readable description
    stages: tuple[GraphStage, ...]     # Ordered stages to execute
    options: GraphRecipeOptions        # Global options
    version: str = "1.0"               # Recipe version
```

**Graph Stage**:

```python
@dataclass(frozen=True)
class GraphStage:
    name: str                    # Stage identifier
    plugins: tuple[str, ...]     # Plugins to execute
    parallel: bool = False       # Allow parallel execution
    fail_fast: bool = True       # Abort on first failure
    optional: bool = False       # Can be skipped
```

### 36.2 Builtin Recipes

```python
# Full pipeline: builders → metrics → validation
FULL_GRAPH_RECIPE = graph_recipe(
    "full",
    description="Complete graph construction and analysis",
    stages=[
        graph_stage("build_goids", ["goid_builder"]),
        graph_stage("build_edges", ["callgraph_builder", "import_graph_builder"]),
        graph_stage("build_structure", ["cfg_dfg_builder", "symbol_uses_builder"]),
        graph_stage("compute_metrics", [...], parallel=True),
        graph_stage("validate", ["graph_validation"], optional=True),
    ],
)

# Builders only
BUILDERS_ONLY_RECIPE = graph_recipe(
    "builders",
    stages=[
        graph_stage("build", DEFAULT_BUILDER_PLUGINS),
    ],
)

# Metrics only (assumes graphs already built)
METRICS_ONLY_RECIPE = graph_recipe(
    "metrics",
    stages=[
        graph_stage("compute", DEFAULT_METRIC_PLUGINS, parallel=True),
    ],
)
```

### 36.3 Recipe Execution

```python
from codeintel.graphs.recipes import execute_graph_recipe, FULL_GRAPH_RECIPE

result = execute_graph_recipe(
    FULL_GRAPH_RECIPE,
    gateway=gateway,
    snapshot=snapshot,
)

if result.success:
    print(f"Built {result.total_rows} rows in {result.duration_s:.2f}s")
```

---

## 37. Registered Graph Plugins & Output Data

### 37.1 Default Plugin Sets

```python
DEFAULT_BUILDER_PLUGINS = (
    "goid_builder",
    "callgraph_builder",
    "import_graph_builder",
    "cfg_dfg_builder",
    "symbol_uses_builder",
)

DEFAULT_METRIC_PLUGINS = (
    "core_graph_metrics",
    "graph_metrics_functions_ext",
    "graph_metrics_modules_ext",
    "test_graph_metrics",
    "cfg_metrics",
    "dfg_metrics",
    "symbol_graph_metrics_modules",
    "symbol_graph_metrics_functions",
    "config_graph_metrics",
    "subsystem_graph_metrics",
    "subsystem_agreement",
    "graph_stats",
)

DEFAULT_VALIDATION_PLUGINS = ("graph_validation",)
```

### 37.2 Output Tables

| Table | Plugin | Contents |
|-------|--------|----------|
| `graph.call_graph_nodes` | `callgraph_builder` | Call graph node metadata |
| `graph.call_graph_edges` | `callgraph_builder` | Function call relationships |
| `graph.import_graph_edges` | `import_graph_builder` | Module import dependencies |
| `graph.cfg_blocks` | `cfg_dfg_builder` | Control flow graph blocks |
| `graph.cfg_edges` | `cfg_dfg_builder` | Control flow graph edges |
| `graph.dfg_edges` | `cfg_dfg_builder` | Data flow graph edges |
| `graph.symbol_use_edges` | `symbol_uses_builder` | Symbol usage relationships |
| `analytics.graph_metrics_functions` | `core_graph_metrics` | Function centrality metrics |
| `analytics.graph_metrics_modules` | `core_graph_metrics` | Module centrality metrics |
| `analytics.graph_validation_findings` | `graph_validation` | Validation findings |

---

## 38. Graphs Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SOURCE DATA (from Ingestion)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  core.goids  │  │ core.modules │  │ core.ast_*   │  ...          │
│  │ (identifiers)│  │  (imports)   │  │ (structure)  │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
└─────────┼──────────────────┼──────────────────┼─────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BUILDER PLUGINS (Stage 1)                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ goid_builder → callgraph_builder → import_graph_builder        │ │
│  │     → cfg_dfg_builder → symbol_uses_builder                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       GRAPH ENGINE                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │
│  │   Call Graph   │  │  Import Graph  │  │  Symbol Graph  │  ...    │
│  │   (DiGraph)    │  │   (DiGraph)    │  │   (Bipartite)  │         │
│  └────────────────┘  └────────────────┘  └────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    METRIC PLUGINS (Stage 2)                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ core_graph_metrics → cfg_metrics → symbol_graph_metrics → ...│   │
│  │ (PageRank, betweenness, closeness, coupling, complexity)     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PURE COMPUTE LAYER                                 │
│  compute/metrics/centrality.py    compute/metrics/components.py     │
│  (stateless functions, no I/O)                                       │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   VALIDATION PLUGINS (Stage 3)                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ graph_validation: integrity checks, anomaly detection          │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PERSISTENCE (ADAPTERS)                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ DuckDB Tables:                                                 │ │
│  │ • graph.call_graph_*  • graph.import_graph_*  • graph.cfg_*    │ │
│  │ • graph.dfg_*  • analytics.graph_metrics_*                     │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   DOWNSTREAM CONSUMERS                               │
│  • Analytics plugins (subsystems, profiles, risk factors)            │
│  • Serving layer (search, recommendations)                           │
│  • Reports & dashboards                                              │
│  • MCP server (AI agent access)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 39. Graphs Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **Hexagonal architecture** | Clean separation between I/O (adapters) and logic (compute) |
| **Three plugin kinds** | Clear separation of concerns: build → analyze → validate |
| **Graph engine abstraction** | Backend-agnostic; swap CPU/GPU implementations |
| **Pure compute layer** | Testable, parallelizable, no hidden state |
| **Recipe composition** | Declarative pipelines, easy customization |
| **Resource injection** | Type-safe DI, testable with fakes |
| **Lazy graph loading** | Load only needed graphs, cache in memory |
| **NetworkX as foundation** | Rich algorithm library, GPU backend available |

---

## 40. Graphs Extension Points

To add a new graph capability:

### Step 1: Create Pure Compute Function

```python
# compute/my_analysis.py
@dataclass(frozen=True)
class MyMetrics:
    """Immutable result container."""
    score: float
    count: int

def compute_my_metrics(graph: nx.DiGraph) -> dict[int, MyMetrics]:
    """Pure function - no I/O."""
    result = {}
    for node in graph.nodes():
        result[node] = MyMetrics(score=..., count=...)
    return result
```

### Step 2: Create Plugin

```python
# plugins/metrics/my_metrics.py
from codeintel.graphs.core import make_metric_plugin, ComputationResult

def _compute(ctx: GraphExecutionContext) -> ComputationResult:
    # Get resources
    storage = ctx.require(StorageResource)
    engine = ctx.engine
    
    # Get graph
    call_graph = engine.call_graph()
    
    # Compute (pure function)
    metrics = compute_my_metrics(call_graph)
    
    # Persist via adapter
    rows = metrics_to_rows(metrics, ctx.repo, ctx.commit)
    storage.gateway.execute_batch("analytics.my_metrics", rows)
    
    return ComputationResult.ok(row_counts={"analytics.my_metrics": len(rows)})

my_metrics_plugin = make_metric_plugin(
    name="my_metrics",
    computation=_compute,
    stage="core",
    requires_graphs=(GraphKind.CALL_GRAPH,),
    produces_tables=("analytics.my_metrics",),
)
```

### Step 3: Register Plugin

```python
# Plugin auto-registers via @graph_plugin decorator or explicit registration
from codeintel.graphs.core import register_graph_plugin
register_graph_plugin(my_metrics_plugin)
```

---

## 41. Graphs File Structure Reference

```
src/codeintel/graphs/
├── __init__.py              # Public API exports
├── catalog.py               # Function catalog (spans, metadata, service)
├── engine_factory.py        # Engine factory functions
├── nx_backend.py            # NetworkX GPU backend configuration
├── core/
│   ├── protocol.py          # GraphPluginProtocol, metadata, decorators
│   ├── context.py           # GraphExecutionContext, scratch
│   ├── result.py            # GraphPluginResult, run records
│   ├── registry.py          # GraphPluginRegistry
│   ├── computation.py       # ComputationResult helpers
│   ├── factories.py         # Plugin factory functions
│   └── adapters.py          # Core adapter utilities
├── engine/
│   ├── protocol.py          # GraphEngine protocol, GraphKind
│   ├── nx_engine.py         # NetworkX implementation
│   ├── cache.py             # Graph caching utilities
│   └── views.py             # Graph view helpers
├── plugins/
│   ├── builders/            # Graph builder plugins
│   │   ├── goid.py          # GOID builder
│   │   ├── callgraph.py     # Call graph builder
│   │   ├── import_graph.py  # Import graph builder
│   │   ├── cfg_dfg.py       # CFG/DFG builder
│   │   └── symbol_uses.py   # Symbol uses builder
│   ├── metrics/             # Graph metric plugins
│   │   ├── core.py          # Core centrality metrics
│   │   ├── secondary.py     # Extended metrics
│   │   └── _runtime.py      # Runtime helpers
│   └── validation.py        # Graph validation plugin
├── compute/
│   ├── callgraph.py         # Pure call graph computation
│   ├── imports.py           # Pure import graph computation
│   ├── cfg.py               # Pure CFG computation
│   ├── dfg.py               # Pure DFG computation
│   ├── symbols.py           # Pure symbol computation
│   └── metrics/             # Pure metric computation
│       ├── centrality.py    # Centrality algorithms
│       ├── components.py    # Component analysis
│       └── coupling.py      # Coupling metrics
├── ports/
│   ├── storage.py           # Storage port protocol
│   ├── parsing.py           # Parsing port protocol
│   ├── catalog.py           # Catalog port protocol
│   └── engine.py            # Engine port protocol
├── adapters/
│   ├── duckdb_storage.py    # DuckDB storage adapter
│   └── callgraph_persistence.py  # Call graph persistence
├── resources/
│   ├── protocol.py          # ResourceProvider protocol
│   ├── container.py         # ResourceContainer (DI)
│   ├── storage.py           # StorageResource
│   ├── graphs.py            # GraphResource
│   └── catalog.py           # CatalogResource
├── recipes/
│   ├── dsl.py               # Recipe DSL definitions
│   ├── executor.py          # Recipe executor
│   └── builtin.py           # Builtin recipes (full, builders, metrics)
└── validation/
    ├── checks.py            # Individual validation checks
    ├── findings.py          # Validation finding types
    └── orchestrator.py      # Validation orchestration
```

---

*This document is maintained as part of the CodeIntel project and reflects the current analytics, ingestion, and graphs architecture as of version 3.0.*

