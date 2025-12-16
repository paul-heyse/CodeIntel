# Hamilton Integration Implementation Report (Phase 0)

> **Status**: Phase 0 Complete, Phase 0 optional items Complete  
> **Date**: December 2024  
> **Next**: See [Phase 1 Implementation Report](Hamilton_phase1_implementation_report.md) for production-grade features

---

## Executive Summary

This document details the production code implemented to integrate Hamilton as the build orchestration layer for CodeIntel. Hamilton replaces the legacy `BuildExecutor` with a DAG-based execution model that provides:

- **Dependency-driven execution** via Hamilton's automatic DAG resolution
- **Skip-if-unchanged caching** via existing manifest infrastructure
- **Type-safe dataset references** via `DatasetRef` and `IbisIOConfig`
- **Contract validation** via Pandera integration with `SCHEMA_REGISTRY`
- **Dynamic node generation** from `TargetGraph` metadata

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Hamilton Build System                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │   BuildEnv       │  │ HamiltonRuntime  │  │ HamiltonBuildExecutor│   │
│  │ (Single Input)   │  │ (Driver+Graph)   │  │  (Orchestrator)      │   │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘   │
│           │                     │                       │               │
│           ▼                     ▼                       ▼               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Hamilton Driver                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │   │
│  │  │ Target Nodes│  │Dataset Nodes│  │  Generated Nodes        │   │   │
│  │  │ (Phase 0)   │  │(Phase 0 opt)|  │  (Node Factory)         │   │   │
│  │  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘   │   │
│  └─────────┼────────────────┼──────────────────────┼────────────────┘   │
│            │                │                      │                    │
│            ▼                ▼                      ▼                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      IO Layer (Phase 0 optional)                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │   │
│  │  │ DatasetRef   │  │ IbisIOConfig │  │ Pandera Contracts    │    │   │
│  │  │ (References) │  │ (IO Adapter) │  │ (SCHEMA_REGISTRY)    │    │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘    │   │
│  └─────────┼─────────────────┼─────────────────────┼────────────────┘   │
│            │                 │                     │                    │
│            ▼                 ▼                     ▼                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Storage Layer (Existing)                      │   │
│  │  ┌────────────────────────────────┐  ┌───────────────────────┐   │   │
│  │  │          IbisGateway           │  │    SCHEMA_REGISTRY    │   │   │
│  │  │  (DuckDB + SQLGlot writes)     │  │    (Pandera schemas)  │   │   │
│  │  └────────────────────────────────┘  └───────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
src/codeintel/build/hamilton/
├── __init__.py                 # Package exports
├── env.py                      # BuildEnv dataclass (single DAG input)
├── naming.py                   # Node naming conventions (t__, d__ prefixes)
├── metadata_bridge.py          # CanonicalPluginMeta extraction
├── manifest_hook.py            # Skip logic and manifest persistence
├── driver_factory.py           # HamiltonRuntime construction
├── executor.py                 # HamiltonBuildExecutor drop-in replacement
├── io/
│   ├── __init__.py             # IO package exports
│   ├── dataset_ref.py          # DatasetRef type and utilities
│   └── ibis_adapter.py         # Ibis-native dataloader/datasaver
├── contracts/
│   ├── __init__.py             # Contracts package exports
│   └── pandera_hook.py         # SCHEMA_REGISTRY integration
└── nodes/
    ├── __init__.py             # Nodes package exports
    ├── targets_phase0.py       # Explicit Phase 0 node definitions
    ├── dataset_nodes.py        # Dataset extraction nodes
    └── node_factory.py         # Dynamic node generation
```

---

## Phase 0: Core Infrastructure

### 1. BuildEnv (`env.py`)

The single input dataclass that provides all dependencies to Hamilton nodes.

```python
@dataclass(frozen=True)
class BuildEnv:
    """Unified environment for Hamilton node execution.
    
    Attributes
    ----------
    gateway : StorageGateway
        Database access for reads/writes.
    snapshot : SnapshotRef
        Repository snapshot (repo, commit).
    paths : BuildPaths
        Filesystem paths for build artifacts.
    providers : ResourceProviders
        Lazy-loaded resource providers.
    config : BuildConfig
        Build configuration and parameters.
    profile : str
        Active build profile name.
    """
    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    providers: ResourceProviders
    config: BuildConfig
    profile: str = "default"
```

### 2. Naming Conventions (`naming.py`)

Stable naming rules for converting logical identifiers to Hamilton node names.

```python
def to_node_name(logical_name: str, *, prefix: str) -> str:
    """Convert logical ID to valid Python identifier.
    
    - Dots and slashes → double underscores (structural separators)
    - Hyphens → single underscores
    - 3+ underscores → collapsed to double
    
    Examples
    --------
    >>> to_node_name("analytics.function_metrics", prefix="t")
    't__analytics__function_metrics'
    
    >>> to_node_name("graph/call_graph", prefix="d")
    'd__graph__call_graph'
    """

def target_node(target_name: str) -> str:
    """Convert target name to Hamilton node (t__ prefix)."""

def dataset_node(dataset_key: str) -> str:
    """Convert dataset key to Hamilton node (d__ prefix)."""

def node_to_target(node_name: str) -> str | None:
    """Extract original target name from node name."""
```

### 3. Metadata Bridge (`metadata_bridge.py`)

Unified metadata extraction from plugins and targets.

```python
@dataclass(frozen=True)
class CanonicalPluginMeta:
    """Normalized plugin metadata for Hamilton observability.
    
    Attributes
    ----------
    name : str
        Plugin identifier (e.g., "analytics.function_metrics").
    version : str
        Semantic version string.
    domain : str
        Execution domain (ingestion, graphs, analytics).
    description : str
        Human-readable description.
    requires : tuple[str, ...]
        Required upstream targets.
    provides : tuple[str, ...]
        Provided output targets.
    produces_tables : tuple[str, ...]
        DuckDB tables written.
    consumes_tables : tuple[str, ...]
        DuckDB tables read.
    options_model : Type[Any] | None
        Pydantic model for configuration.
    """

def from_target(target: OutputTarget) -> CanonicalPluginMeta:
    """Derive metadata from OutputTarget definition."""

def from_plugin_or_target(*, plugin: Any, target: OutputTarget) -> CanonicalPluginMeta:
    """Extract metadata from plugin (preferred) or target (fallback)."""
```

### 4. Manifest Hook (`manifest_hook.py`)

Skip/cache logic reusing existing build infrastructure.

```python
@dataclass(frozen=True)
class TargetRunRecord:
    """Record of Hamilton node execution.
    
    Attributes
    ----------
    target : str
        Target name executed.
    plugin_name : str
        Plugin that produced output.
    status : str
        "succeeded", "failed", or "skipped".
    input_hash : str | None
        Content-addressable input hash.
    options_hash : str | None
        Configuration options hash.
    duration_ms : float
        Execution time in milliseconds.
    row_counts : Mapping[str, int]
        Table key → row count mapping.
    error : str | None
        Error message if failed.
    datasets : tuple[DatasetRef, ...]
        Dataset references produced (Phase 0 optional).
    """
    
    def get_dataset(self, table_key: str) -> DatasetRef | None:
        """Get specific dataset ref by table key."""

def should_skip(
    *, gateway, target, repo, commit, input_hash
) -> bool:
    """Check if target can be skipped based on prior manifest."""

def save_manifest(*, gateway, request: ManifestSaveRequest) -> None:
    """Persist manifest record for completed target."""
```

### 5. Driver Factory (`driver_factory.py`)

Hamilton Driver construction with Phase 0 or generated nodes.

```python
@dataclass(frozen=True)
class HamiltonRuntime:
    """Bundled Driver and TargetGraph.
    
    Attributes
    ----------
    dr : hamilton.driver.Driver
        Configured Hamilton Driver.
    graph : TargetGraph
        Target graph for metadata lookup.
    """

def build_driver(
    *,
    config: dict[str, Any] | None = None,
    use_generated: bool = False,
) -> HamiltonRuntime:
    """Build Hamilton Driver for build execution.
    
    Parameters
    ----------
    config
        Configuration dict passed to Hamilton Driver.
    use_generated
        If True, use dynamically generated nodes from TargetGraph.
        If False (default), use explicit Phase 0 nodes.
    """
```

### 6. Executor (`executor.py`)

Drop-in replacement for legacy `BuildExecutor`.

```python
@dataclass(frozen=True)
class HamiltonBuildResult:
    """Result of Hamilton-based build execution.
    
    Attributes
    ----------
    success : bool
        Overall success status.
    records : tuple[TargetRunRecord, ...]
        Execution records for each target.
    duration_ms : float
        Total execution time.
    error : str | None
        Error message if failed.
    """

class HamiltonBuildExecutor:
    """Hamilton-based build executor.
    
    Replaces legacy BuildExecutor with Hamilton DAG execution.
    """
    
    def run(
        self,
        env: BuildEnv,
        targets: list[str],
    ) -> HamiltonBuildResult:
        """Execute targets via Hamilton Driver."""
```

### 7. Phase 0 Target Nodes (`nodes/targets_phase0.py`)

Explicit Hamilton nodes for the risk_factors execution chain.

```python
# Node dependency chain:
# modules → scip → goids → call_graph → risk_factors
#        → ast  ↗      → function_metrics ↗

@tag(domain="ingestion", target="modules")
def t__modules(env: BuildEnv, graph: TargetGraph) -> TargetRunRecord:
    """Execute modules target (repository scan)."""

@tag(domain="ingestion", target="scip")
def t__scip(env: BuildEnv, graph: TargetGraph, t__modules: TargetRunRecord) -> TargetRunRecord:
    """Execute scip target (SCIP index ingestion)."""

@tag(domain="ingestion", target="ast")
def t__ast(env: BuildEnv, graph: TargetGraph, t__modules: TargetRunRecord) -> TargetRunRecord:
    """Execute ast target (AST extraction)."""

@tag(domain="graphs", target="goids")
def t__goids(env: BuildEnv, graph: TargetGraph, t__scip: TargetRunRecord, t__ast: TargetRunRecord) -> TargetRunRecord:
    """Execute goids target (GOID resolution)."""

@tag(domain="graphs", target="call_graph")
def t__call_graph(env: BuildEnv, graph: TargetGraph, t__goids: TargetRunRecord, t__scip: TargetRunRecord) -> TargetRunRecord:
    """Execute call_graph target (function call graph)."""

@tag(domain="analytics", target="function_metrics")
def t__function_metrics(env: BuildEnv, graph: TargetGraph, t__goids: TargetRunRecord, t__ast: TargetRunRecord) -> TargetRunRecord:
    """Execute function_metrics target (structural metrics)."""

@tag(domain="analytics", target="risk_factors")
def t__risk_factors(env: BuildEnv, graph: TargetGraph, t__function_metrics: TargetRunRecord, t__call_graph: TargetRunRecord) -> TargetRunRecord:
    """Execute risk_factors target (composite risk factors)."""

# Internal helper used by all nodes
def _run_target(*, env: BuildEnv, graph: TargetGraph, target_name: str) -> TargetRunRecord:
    """Execute target plugin with skip logic and manifest persistence."""
```

---

## Phase 0 optional: IO Adapters & Contracts

### 1. DatasetRef (`io/dataset_ref.py`)

Type-safe dataset references for DAG lineage.

```python
@dataclass(frozen=True)
class DatasetRef:
    """Lightweight reference to a DuckDB table.
    
    Flows through the Hamilton DAG without materializing data.
    
    Attributes
    ----------
    table_key : str
        Fully-qualified table name (e.g., "analytics.function_metrics").
    schema_version : str | None
        Schema version for compatibility tracking.
    row_count : int | None
        Row count if known from prior computation.
    source_target : str | None
        Target that produced this dataset.
    metadata : dict[str, object]
        Additional metadata for observability.
    """
    
    @property
    def schema_name(self) -> str:
        """Extract schema portion (e.g., 'analytics')."""
    
    @property
    def table_name(self) -> str:
        """Extract table portion (e.g., 'function_metrics')."""
    
    def with_row_count(self, count: int) -> DatasetRef:
        """Return new ref with updated row count."""
    
    def with_metadata(self, key: str, value: object) -> DatasetRef:
        """Return new ref with additional metadata."""

def refs_from_target_result(
    target_name: str,
    table_keys: tuple[str, ...],
    row_counts: dict[str, int] | None = None,
) -> dict[str, DatasetRef]:
    """Create DatasetRef instances from target execution result."""

def refs_to_tuple(refs: dict[str, DatasetRef]) -> tuple[DatasetRef, ...]:
    """Convert dict of refs to immutable tuple."""
```

### 2. Ibis IO Adapter (`io/ibis_adapter.py`)

Ibis-native IO operations via `IbisGateway`.

```python
@dataclass(frozen=True)
class IbisIOConfig:
    """Configuration for Ibis IO operations.
    
    Attributes
    ----------
    gateway : StorageGateway
        Storage gateway (use gateway.ibis for operations).
    validate_schema : bool
        Whether to validate against Pandera schema.
    """

def load_ibis_table(
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> tuple[ir.Table, dict[str, Any]]:
    """Load table as Ibis expression via IbisGateway.table()."""

def load_table_as_dataframe(
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load table as pandas DataFrame."""

def save_ibis_expression(
    output: ir.Table,
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save Ibis expression via IbisGateway.write() (INSERT...SELECT)."""

def save_dataframe(
    df: pd.DataFrame,
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save DataFrame via IbisGateway.write() (INSERT...VALUES)."""

def save_rows(
    rows: Sequence[tuple[object, ...]],
    columns: Sequence[str],
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save row tuples via IbisGateway.write()."""

def upsert_dataframe(
    df: pd.DataFrame,
    dataset_ref: DatasetRef,
    conflict_columns: Sequence[str],
    update_columns: Sequence[str],
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Upsert DataFrame via IbisGateway.upsert() (INSERT...ON CONFLICT)."""
```

**Key Design Principle**: All DuckDB operations go through `IbisGateway`, which internally delegates to `DuckDBPolicyBackend` for SQLGlot-based SQL generation. Hamilton adapters never access `DuckDBPolicyBackend` directly.

### 3. Pandera Contract Hook (`contracts/pandera_hook.py`)

Integration with `SCHEMA_REGISTRY` for data quality validation.

```python
def get_pandera_schema(table_key: str) -> pa.DataFrameSchema | None:
    """Retrieve Pandera schema from SCHEMA_REGISTRY."""

def validate_dataframe(df: pd.DataFrame, table_key: str) -> pd.DataFrame:
    """Validate DataFrame against registered schema.
    
    Raises
    ------
    ValueError
        If no schema registered or validation fails.
    """

def with_contract(table_key: str) -> Callable[[F], F]:
    """Decorator to validate function output against Pandera schema.
    
    Example
    -------
    @with_contract("analytics.function_metrics")
    def compute_metrics(data: pd.DataFrame) -> pd.DataFrame:
        return process(data)
    """

def validate_dataset_ref(
    ref: DatasetRef,
    gateway: StorageGateway,
) -> tuple[bool, str | None]:
    """Validate DatasetRef's underlying table against schema.
    
    Returns
    -------
    tuple[bool, str | None]
        (is_valid, error_message) tuple.
    """

def contract_status_for_table(table_key: str) -> dict[str, Any]:
    """Get contract status info (has_schema, columns, etc.)."""
```

### 4. Dataset Extraction Nodes (`nodes/dataset_nodes.py`)

Nodes that expose individual datasets from target results.

```python
def extract_datasets_from_record(
    record: TargetRunRecord,
    table_keys: tuple[str, ...],
) -> dict[str, DatasetRef]:
    """Extract DatasetRef instances from TargetRunRecord."""

@tag(domain="graphs", dataset="call_graph_edges")
def d__graph__call_graph_edges(t__call_graph: TargetRunRecord) -> DatasetRef:
    """Extract call_graph_edges dataset from call_graph target."""

@tag(domain="graphs", dataset="call_graph_nodes")
def d__graph__call_graph_nodes(t__call_graph: TargetRunRecord) -> DatasetRef:
    """Extract call_graph_nodes dataset from call_graph target."""

@tag(domain="analytics", dataset="function_metrics")
def d__analytics__function_metrics(t__function_metrics: TargetRunRecord) -> DatasetRef:
    """Extract function_metrics dataset from function_metrics target."""

@tag(domain="analytics", dataset="risk_factors")
def d__analytics__risk_factors(t__risk_factors: TargetRunRecord) -> DatasetRef:
    """Extract risk_factors dataset from risk_factors target."""
```

### 5. Node Factory (`nodes/node_factory.py`)

Dynamic node generation from `TargetGraph` metadata.

```python
def build_target_module(
    *,
    include_targets: set[str] | None = None,
    exclude_targets: set[str] | None = None,
) -> ModuleType:
    """Generate module containing Hamilton nodes for all targets.
    
    Creates Python functions dynamically from TargetGraph with
    proper signatures and dependencies for Hamilton execution.
    
    Parameters
    ----------
    include_targets
        Only generate nodes for these targets.
    exclude_targets
        Exclude these targets from generation.
    
    Returns
    -------
    ModuleType
        Module with generated node functions and TARGET_TO_NODE mapping.
    """

def get_generated_module() -> ModuleType:
    """Get or create cached generated nodes module."""

def clear_generated_module_cache() -> None:
    """Clear cached module (for testing or TargetGraph changes)."""
```

---

## Package Exports

### Main Package (`__init__.py`)

```python
__all__ = [
    # Phase 0
    "BuildEnv",
    "CanonicalPluginMeta",
    "HamiltonBuildExecutor",
    "HamiltonBuildResult",
    "TargetRunRecord",
    "target_node",
    "dataset_node",
    "to_node_name",
    
    # Phase 0 optional items implemented
    "DatasetRef",
    "IbisIOConfig",
    "refs_from_target_result",
    "get_pandera_schema",
    "validate_dataframe",
    "validate_dataset_ref",
    "with_contract",
]
```

---

## Integration Points

### CLI Integration

The `--engine` flag on build commands allows selecting between legacy and Hamilton execution:

```bash
# Legacy executor (default)
codeintel build run --targets risk_factors

# Hamilton executor
codeintel build run --targets risk_factors --engine hamilton
```

### Storage Layer Integration

Hamilton IO adapters integrate exclusively via `IbisGateway`:

| Operation | IbisGateway Method | Internal Backend |
|-----------|-------------------|------------------|
| Read table | `gateway.ibis.table()` | Ibis 11 query |
| Write Ibis expr | `gateway.ibis.write()` | SQLGlot INSERT...SELECT |
| Write DataFrame | `gateway.ibis.write()` | DuckDBPolicyBackend.bulk_insert() |
| Write tuples | `gateway.ibis.write()` | DuckDBPolicyBackend.bulk_insert() |
| Upsert | `gateway.ibis.upsert()` | DuckDBPolicyBackend.upsert() |

### Schema Registry Integration

Contract validation uses `SCHEMA_REGISTRY` as the single source of truth:

```python
from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY

# Get Pandera schema
schema = SCHEMA_REGISTRY.get("analytics.function_metrics")
if schema:
    validated_df = schema.pandera_schema.validate(df)
```

---

## Future Work

> **Note**: Many items below are now implemented in Phase 1. See [Phase 1 Implementation Report](Hamilton_phase1_implementation_report.md).

1. ~~**Generated Nodes Default**: Switch from Phase 0 explicit nodes to generated nodes~~ ✅ Phase 1
2. **Hamilton UI Integration**: Enable Hamilton's observability dashboard
3. **Parallel Execution**: Configure Hamilton's parallelization for independent targets
4. **Policy Profiles**: Implement config-driven DAG variants via Hamilton's `@config` decorators
5. **Content-Based Caching**: Leverage Hamilton's built-in caching alongside manifest-based skip logic

### Phase 1 Delivered Features

- ✅ Hamilton Node Mode (phase0/generated) with correct target↔node mapping
- ✅ Full dependency closure execution with accurate computed/skipped/failed tracking
- ✅ Upstream failure gating (downstream targets skip if upstream fails)
- ✅ CLI parity with `--force` flag support
- ✅ Run tracking integration with `build.runs`
- ✅ Universal dataset lineage with auto-populated `TargetRunRecord.datasets`
- ✅ DAG observability with `codeintel build graph` command

---

## References

- [Hamilton Documentation](https://hamilton.dagworks.io/)
- [Phase 0 Specification](Hamilton_apache_phase0.md)
- [Phase 0 optional items IO Contracts Plan](Hamilton_phase1_io_contracts.md)
- [Ibis 11 Migration Notes](../migrations/ibis-11-migration.md)
- [SCHEMA_REGISTRY Architecture](../../openspec/plans/pandera-schema-unification-architecture.md)

