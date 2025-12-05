# Executor Architecture Consolidation

> **Status:** Phase 1 Complete | **Date:** December 5, 2025  
> **Scope:** Unified base classes for plugin executors across analytics, graphs, and ingestion domains

---

## Executive Summary

This document details the consolidation of the executor architecture across the CodeIntel codebase. The work unified three parallel executor implementations (analytics, graphs, ingestion) into a shared base class hierarchy, establishing consistent patterns for retry handling, telemetry integration, and execution reporting.

---

## Part 1: Work Completed

### 1.1 Context and Motivation

Prior to this consolidation, the codebase had three separate executor implementations:
- **Analytics:** `codeintel.analytics.core.executor.PluginExecutor`
- **Graphs:** `codeintel.graphs.runtime.executor` (function-based `run_graph_plugins`)
- **Ingestion:** `codeintel.ingestion.recipes.executor.RecipeExecutor`

Each implementation had its own:
- Execution policy configuration
- Executor context structure
- Execution report format
- Retry logic implementation
- Telemetry integration

This led to:
- ~600 lines of duplicated code
- Inconsistent behavior across domains
- Difficult maintenance and testing
- No shared infrastructure for new features

### 1.2 New Core Infrastructure Created

#### 1.2.1 `BaseExecutionPolicy` (`src/codeintel/core/plugins/policy.py`)

Unified execution policy consolidating common fields from all three domains:

```python
@dataclass(frozen=True)
class BaseExecutionPolicy:
    """Common execution policy for all plugin executors."""
    fail_fast: bool = True
    max_retries: int = 0
    retry_backoff_ms: int = 100
    skip_on_unchanged: bool = False
    dry_run: bool = False
    enable_parallel: bool = False
    max_workers: int = 4
    timeout_ms: int | None = None
    validate_contracts: bool = False
    
    def to_retry_policy(self) -> RetryPolicy:
        """Convert to tenacity RetryPolicy from core.runtime.retry."""
        ...
```

**Key Features:**
- Converts to `tenacity`-based `RetryPolicy` for retry execution
- Frozen dataclass for immutability
- Sensible defaults matching existing behavior

#### 1.2.2 `BaseExecutorContext` (`src/codeintel/core/plugins/executor_context.py`)

Unified executor context (distinct from plugin execution context):

```python
@dataclass
class BaseExecutorContext:
    """Common executor context for all domains."""
    gateway: StorageGateway
    snapshot: SnapshotRef
    run_context: RunContext | None = None
    telemetry: RuntimeTelemetry = field(default_factory=get_runtime_telemetry)
    
    @property
    def effective_run_id(self) -> str:
        """Get run ID from run_context or empty string."""
        ...
```

**Key Features:**
- Provides gateway and snapshot access
- Integrates with unified `RunContext` for correlation
- Lazy telemetry initialization

#### 1.2.3 `BaseExecutionReport` (`src/codeintel/core/plugins/report.py`)

Unified execution report with common metrics:

```python
@dataclass(frozen=True)
class BaseExecutionReport:
    """Common execution report for all domains."""
    run_id: str
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    records: tuple[PluginExecutionRecord, ...]
    fatal_error: bool = False
    
    @property
    def success_count(self) -> int: ...
    @property
    def failure_count(self) -> int: ...
    @property
    def skip_count(self) -> int: ...
    @property
    def status(self) -> Literal["succeeded", "failed", "partial"]: ...
```

**Key Features:**
- Computed properties for metrics aggregation
- Frozen for immutability
- Status derivation logic centralized

#### 1.2.4 `BasePluginExecutor` (`src/codeintel/core/plugins/executor.py`)

Abstract base executor with shared execution logic:

```python
class BasePluginExecutor[P, C, EC, R](ABC, Generic[P, C, EC, R]):
    """Abstract base executor with retry, telemetry, and recording."""
    
    def __init__(
        self,
        policy: BaseExecutionPolicy | None = None,
        telemetry: RuntimeTelemetry | None = None,
    ) -> None: ...
    
    @abstractmethod
    def _build_plugin_context(self, executor_ctx: EC, plugin: P, scratch: PluginScratch) -> C:
        """Build domain-specific plugin execution context."""
        ...
    
    @abstractmethod
    def _build_report(self, run_id: str, records: list[PluginExecutionRecord], ...) -> R:
        """Build domain-specific execution report."""
        ...
    
    def execute_plan(self, executor_ctx: EC, plan: PluginPlan[P], ...) -> R:
        """Execute all plugins in plan with retry and telemetry."""
        ...
    
    def _execute_single_plugin(self, plugin: P, ctx: C, run_id: str) -> PluginExecutionRecord:
        """Execute one plugin with retry using core.runtime.retry."""
        ...
```

**Key Features:**
- Generic over plugin protocol, contexts, and report types
- Uses `tenacity` via `core.runtime.retry` for retry logic
- Integrates `RuntimeTelemetry` for OTel/Prometheus spans
- Hook methods for domain-specific customization

#### 1.2.5 Tracking Helpers (`src/codeintel/core/plugins/tracking.py`)

Extracted shared pipeline run/step recording logic:

```python
def record_plugin_steps(
    runs: PipelineRunTracking,
    run_id: str,
    module: ModuleKind,
    records: Sequence[PluginExecutionRecord],
    get_stage: Callable[[str], str],
) -> None:
    """Record plugin execution records as pipeline steps."""
    ...

def complete_run_from_records(
    runs: PipelineRunTracking,
    run_id: str,
    records: Sequence[PluginExecutionRecord],
    fatal_error: bool = False,
) -> None:
    """Complete a pipeline run based on execution records."""
    ...
```

### 1.3 Domain Executor Migrations

#### Analytics Executor
- **File:** `src/codeintel/analytics/core/executor.py`
- **Changes:**
  - Created `AnalyticsExecutorContext` extending `BaseExecutorContext`
  - Uses `BaseExecutionPolicy` (via `policy` property)
  - Integrated `RuntimeTelemetry` for span management
  - Kept domain-specific middleware chain and contract validation

#### Graphs Executor
- **File:** `src/codeintel/graphs/runtime/executor.py`
- **Changes:**
  - Created `GraphExecutorContext` extending `BaseExecutorContext`
  - Renamed `GraphRunReport` to align with base pattern
  - Integrated `RuntimeTelemetry` for span management
  - Maintained parallel execution infrastructure

#### Ingestion Executor
- **File:** `src/codeintel/ingestion/recipes/executor.py`
- **Changes:**
  - Integrated `RuntimeTelemetry` for run-level metrics
  - Added `_record_telemetry()` helper for metrics recording
  - Maintained recipe stage execution pattern

### 1.4 Protocol Updates

#### Analytics Plugin Protocol
- **File:** `src/codeintel/analytics/core/protocol.py`
- **Changes:**
  - Enhanced documentation explaining structural compatibility with `PluginProtocol`
  - Added `is_analytics_plugin()` and `is_core_compatible()` helper functions
  - Re-exported `PluginProtocol` for explicit documentation

#### Context Inheritance
- **Analytics:** `analytics.core.context.PluginExecutionContext` extends `core.plugins.context.PluginExecutionContext`
- **Ingestion:** `IngestExecutionContext` extends `core.plugins.context.PluginExecutionContext`
- **All domains:** Use same `PluginMetadata` from core

### 1.5 Tests Created

- **File:** `tests/core/plugins/test_base_executor.py`
- **Coverage:** 18 passing tests verifying:
  - `BaseExecutionPolicy` default and custom values
  - `BaseExecutorContext` `effective_run_id` property
  - `BaseExecutionReport` status calculation (succeeded, failed, partial)
  - `PluginScratch` utilities (declare, consume, cleanup)
  - `RuntimeTelemetry` span lifecycle
  - Real executor implementations use base infrastructure

### 1.6 Quality Gates

All quality gates pass:
- ✅ `ruff check` — zero errors
- ✅ `pyright --warnings --pythonversion=3.13` — zero errors
- ✅ `pyrefly check` — zero errors
- ✅ `pytest` — all tests passing

---

## Part 2: Immediate Cleanup & Migration Work

### 2.1 Duplicate `PluginExecutionRecord` Definitions to Consolidate

**Current State:** Three separate `PluginExecutionRecord` classes exist:
- `src/codeintel/core/plugins/result.py` (canonical)
- `src/codeintel/ingestion/recipes/executor.py` (duplicate)
- `src/codeintel/ingestion/runtime/executor.py` (duplicate)

**Action Required:**

| File | Action |
|------|--------|
| `src/codeintel/ingestion/recipes/executor.py` | Delete duplicate class, import from `core.plugins.result` |
| `src/codeintel/ingestion/runtime/executor.py` | Delete duplicate class, import from `core.plugins.result` |

**Impact:** ~100 lines removed, single source of truth for execution records.

---

### 2.2 Telemetry Implementation Consolidation

**Current State:** Three telemetry implementations with minimal differentiation:
- `src/codeintel/core/runtime/telemetry.py` — `RuntimeTelemetry` (base)
- `src/codeintel/graphs/runtime/telemetry.py` — `GraphRuntimeTelemetry` (extends base)
- `src/codeintel/ingestion/runtime/telemetry.py` — `IngestRuntimeTelemetry` (extends base)

**Action Required:**

| File | Action |
|------|--------|
| `src/codeintel/graphs/runtime/telemetry.py` | Evaluate if `GraphRuntimeTelemetry` adds value beyond config; if not, use base directly |
| `src/codeintel/ingestion/runtime/telemetry.py` | Evaluate if `IngestRuntimeTelemetry` adds value beyond config; if not, use base directly |
| All call sites | Update to use `get_runtime_telemetry()` from core if domain-specific versions are removed |

**Recommendation:** Keep domain-specific config classes (`GraphTelemetryConfig`, `IngestTelemetryConfig`) but use `RuntimeTelemetry` directly unless domain-specific span attributes are required.

---

### 2.3 Recipe Executor Consolidation

**Current State:** Three separate recipe executor implementations:
- `src/codeintel/analytics/recipes/executor.py` — `RecipeExecutor`
- `src/codeintel/graphs/recipes/executor.py` — `RecipeExecutor`  
- `src/codeintel/ingestion/recipes/executor.py` — `RecipeExecutor`

**Action Required:**

1. **Create `BaseRecipeExecutor`** in `src/codeintel/core/recipes/executor.py`:
   - Extract common stage execution logic
   - Extract common parallel execution infrastructure
   - Extract common result aggregation

2. **Migrate domain executors:**

| File | Action |
|------|--------|
| `src/codeintel/analytics/recipes/executor.py` | Extend `BaseRecipeExecutor`, keep analytics-specific logic |
| `src/codeintel/graphs/recipes/executor.py` | Extend `BaseRecipeExecutor`, keep graph-specific logic |
| `src/codeintel/ingestion/recipes/executor.py` | Extend `BaseRecipeExecutor`, keep ingestion-specific logic |

**Estimated Impact:** ~300-400 lines of duplicate code removed.

---

### 2.4 Tracking Helper Deduplication

**Current State:** Multiple implementations of step recording:
- `src/codeintel/core/plugins/tracking.py` — `record_plugin_steps()`, `complete_run_from_records()` (new, canonical)
- `src/codeintel/graphs/runtime/executor.py` — `_record_graph_steps()` (domain-specific)
- `src/codeintel/ingestion/recipes/executor.py` — inline recording logic

**Action Required:**

| File | Action |
|------|--------|
| `src/codeintel/graphs/runtime/executor.py` | Migrate `_record_graph_steps()` to use core `record_plugin_steps()` |
| `src/codeintel/ingestion/recipes/executor.py` | Use core tracking helpers |
| `src/codeintel/analytics/core/pipeline_bridge.py` | Use core tracking helpers |

---

### 2.5 Export Updates Needed

**Files requiring export updates:**

| File | Changes Needed |
|------|----------------|
| `src/codeintel/core/plugins/__init__.py` | Verify all new base classes are exported |
| `src/codeintel/analytics/core/__init__.py` | Export `AnalyticsExecutorContext`, `AnalyticsExecutionReport` |
| `src/codeintel/graphs/runtime/__init__.py` | Export `GraphExecutorContext`, consider exporting executor class |
| `src/codeintel/ingestion/recipes/__init__.py` | Export `IngestExecutorContext`, `IngestExecutionReport` if created |

---

### 2.6 Old/Legacy Files to Review for Deletion

| File | Status | Action |
|------|--------|--------|
| `src/codeintel/ingestion/runtime/executor.py` | Contains duplicate `PluginExecutionRecord` | Consolidate or delete if superseded by `recipes/executor.py` |
| `src/codeintel/core/recipes/model.py` | May have overlapping definitions | Review for consolidation with new base classes |

---

### 2.7 Test File Updates

| File | Action |
|------|--------|
| `tests/core/plugins/test_base_executor.py` | ✅ Created |
| `tests/analytics/core/test_executor.py` | Update to verify integration with base classes |
| `tests/graphs/runtime/test_executor.py` | Update to verify integration with base classes |
| `tests/ingestion/recipes/test_executor.py` | Update to verify integration with base classes |

---

## Part 3: Substantial Opportunities for Investigation

### 3.1 Unified Plugin Registry Architecture

**Observation:** Three separate plugin registries exist:
- `codeintel.core.plugins.registry.BasePluginRegistry`
- `codeintel.analytics.core.registry.PluginRegistry`
- `codeintel.graphs.core.registry.GraphPluginRegistry`

**Opportunity:** Create a unified `PluginRegistry[P]` generic base that all domains extend, with:
- Common plugin discovery and registration
- Common dependency resolution (topological sort already in analytics)
- Common validation infrastructure
- Domain-specific metadata fields via generics

**Estimated Benefit:** 
- Single source of truth for plugin lifecycle
- Consistent plugin introspection across domains
- Simplified testing with unified mocking

---

### 3.2 Plugin Execution Context Unification

**Observation:** Multiple execution context hierarchies:
- Core: `PluginExecutionContext`, `PluginExecutionContextBuilder`
- Analytics: `analytics.PluginExecutionContext` (extends core)
- Graphs: `GraphPluginExecutionContext`
- Ingestion: `IngestExecutionContext` (extends core)

**Opportunity:** Formalize a context protocol with:
- Required capabilities (gateway, snapshot, run_id)
- Optional capabilities (scope, profiles, tools)
- Resource registry pattern (already well-implemented)

**Design Consideration:**
```python
@runtime_checkable
class ExecutionContextProtocol(Protocol):
    """Unified protocol for all execution contexts."""
    gateway: StorageGateway
    snapshot: SnapshotRef
    run_id: str
    resources: ResourceRegistry
    
    @property
    def effective_run_id(self) -> str: ...
```

---

### 3.3 Contract Validation Framework

**Observation:** Contract validation exists in analytics but not consistently across domains:
- `codeintel.analytics.core.contracts` — contract checkers
- `codeintel.storage.validation.contract` — data validation

**Opportunity:** Create a unified contract framework:
1. Pre-execution input contracts (table existence, column types)
2. Post-execution output contracts (row counts, schema compliance)
3. Cross-plugin contracts (capability dependencies)

**Integration Points:**
- Hook into `BasePluginExecutor._validate_plugin_inputs()`
- Hook into plugin metadata's `inputs` and `outputs` specs
- Generate contract reports as part of execution reports

---

### 3.4 Observability Dashboard Integration

**Observation:** Telemetry is well-structured with OTel + Prometheus:
- Spans for plugin execution
- Metrics for duration and counts
- Structured logging

**Opportunity:** Build a real-time observability layer:
1. **Pipeline dashboard** showing:
   - Active runs with progress
   - Historical run metrics
   - Plugin performance heatmaps
   
2. **Alerting integration** for:
   - Plugin execution failures
   - Duration anomalies
   - Retry exhaustion

3. **Trace correlation** across:
   - Ingestion → Graphs → Analytics pipeline
   - HTTP request → pipeline execution → storage writes

---

### 3.5 Incremental Execution Infrastructure

**Observation:** Incremental support exists but is inconsistent:
- `PluginMetadata.supports_incremental`
- `ChangeTracker` in ingestion
- Skip logic in executors

**Opportunity:** Build a robust incremental execution system:
1. **Input hash computation** (already in graphs via `compute_input_hash`)
2. **Output manifest tracking** (already in graphs via `GraphPluginManifest`)
3. **Dependency-aware invalidation** — if upstream changed, invalidate downstream
4. **Checkpointing** — persist intermediate state for resume-on-failure

**Design:**
```python
class IncrementalExecutionPolicy:
    """Policy for incremental plugin execution."""
    hash_inputs: bool = True
    track_manifests: bool = True
    invalidate_dependents: bool = True
    checkpoint_frequency: int | None = None
```

---

### 3.6 Plugin Isolation & Resource Limits

**Observation:** Plugin metadata supports isolation:
- `PluginMetadata.isolation_kind` — "none", "thread", "process"
- `PluginMetadata.resource_hints` — memory, CPU hints

**Opportunity:** Implement actual resource isolation:
1. **Thread isolation** — ThreadPoolExecutor with timeout (partially implemented)
2. **Process isolation** — ProcessPoolExecutor for memory-hungry plugins
3. **Container isolation** — Docker-based isolation for untrusted plugins
4. **Resource limits** — cgroups/ulimits for memory/CPU caps

---

### 3.7 Plugin Development Toolkit

**Observation:** Plugin creation requires understanding multiple modules and patterns.

**Opportunity:** Create a plugin development toolkit:
1. **Plugin scaffolding CLI:**
   ```bash
   codeintel plugin new --domain analytics --name my_plugin
   ```
2. **Plugin testing harness** (partially exists in `tests/_helpers/plugin_harness.py`)
3. **Plugin documentation generator** from metadata
4. **Plugin compliance checker** verifying all required protocols

---

### 3.8 Async Execution Support

**Observation:** Current executors are synchronous with thread-based parallelism.

**Opportunity:** Add first-class async support:
1. **Async plugin protocol:**
   ```python
   class AsyncPluginProtocol(Protocol):
       async def execute(self, ctx: PluginExecutionContext) -> PluginResult: ...
   ```
2. **Async executor** using `asyncio.TaskGroup`
3. **Mixed execution** — run sync plugins in thread pool, async plugins natively
4. **Better cancellation** via async context managers

---

### 3.9 Configuration Schema Generation

**Observation:** Plugin configs are dataclasses with runtime validation.

**Opportunity:** Generate JSON Schema from configs:
1. Auto-generate JSON Schema from `pydantic` or `dataclass` configs
2. Expose schemas via API for IDE autocomplete
3. Validate configs before pipeline execution
4. Generate documentation from schemas

---

### 3.10 Event-Driven Plugin Communication

**Observation:** Plugins communicate via scratch store and database writes.

**Opportunity:** Add event bus for loose coupling:
1. **Plugin events:**
   - `PluginStarted`, `PluginCompleted`, `PluginFailed`
   - `DataWritten(table, row_count)`
   - `GraphBuilt(graph_kind)`
   
2. **Event handlers:**
   - Real-time progress reporting
   - Dependent plugin triggering
   - External webhook notifications

---

## Part 4: Summary Priority Matrix

| Category | Item | Priority | Effort |
|----------|------|----------|--------|
| **Cleanup** | Duplicate PluginExecutionRecord | 🔴 High | Low |
| **Cleanup** | Tracking helper deduplication | 🔴 High | Low |
| **Cleanup** | Export updates | 🟡 Medium | Low |
| **Migration** | Recipe executor consolidation | 🟡 Medium | Medium |
| **Migration** | Telemetry consolidation | 🟢 Low | Low |
| **Enhancement** | Unified plugin registry | 🟡 Medium | High |
| **Enhancement** | Contract validation framework | 🟡 Medium | Medium |
| **Enhancement** | Incremental execution | 🔴 High | High |
| **Enhancement** | Async execution support | 🟢 Low | High |
| **Tooling** | Plugin development toolkit | 🟢 Low | Medium |
| **Observability** | Dashboard integration | 🟢 Low | Medium |

---

## Appendix: Files Changed in Phase 1

### New Files Created
| File | Purpose |
|------|---------|
| `src/codeintel/core/plugins/policy.py` | `BaseExecutionPolicy` |
| `src/codeintel/core/plugins/executor_context.py` | `BaseExecutorContext` |
| `src/codeintel/core/plugins/report.py` | `BaseExecutionReport` |
| `src/codeintel/core/plugins/executor.py` | `BasePluginExecutor` |
| `src/codeintel/core/plugins/tracking.py` | `record_plugin_steps`, `complete_run_from_records` |
| `tests/core/plugins/test_base_executor.py` | Unit tests |

### Files Modified
| File | Changes |
|------|---------|
| `src/codeintel/core/plugins/__init__.py` | Export new base classes |
| `src/codeintel/analytics/core/executor.py` | Integrated base classes and telemetry |
| `src/codeintel/analytics/core/protocol.py` | Enhanced documentation, added helpers |
| `src/codeintel/analytics/core/context.py` | Enhanced documentation |
| `src/codeintel/graphs/runtime/executor.py` | Created `GraphExecutorContext`, integrated telemetry |
| `src/codeintel/ingestion/recipes/executor.py` | Integrated telemetry, refactored complexity |
| `src/codeintel/ingestion/core/execution_context.py` | Enhanced documentation |
| `src/codeintel/pipeline/orchestration/core.py` | Enhanced documentation |
| `pyproject.toml` | Added per-file-ignores for base executor |

