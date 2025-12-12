# Hamilton Phase 1: IO Adapters, Dataset Contracts, and Pandera Integration

> **Purpose**: This plan details the implementation of Hamilton IO adapters, dataset-first DAG design, and Pandera contract integration that builds on Phase 0 and aligns with the Ibis/Pandera/SQLGlot architecture.

---

## Table of Contents

1. [Overview and Goals](#overview-and-goals)
2. [Architecture Integration Points](#architecture-integration-points)
3. [Module Structure](#module-structure)
4. [Implementation Tasks](#implementation-tasks)
   - [Task 1: DatasetRef Type System](#task-1-datasetref-type-system)
   - [Task 2: Ibis-Native DataLoader/DataSaver](#task-2-ibis-native-dataloaderdatasaver)
   - [Task 3: Dataset Extraction Nodes](#task-3-dataset-extraction-nodes)
   - [Task 4: Pandera Contract Integration](#task-4-pandera-contract-integration)
   - [Task 5: Node Factory for Generated Nodes](#task-5-node-factory-for-generated-nodes)
5. [Testing Strategy](#testing-strategy)
6. [Migration Path](#migration-path)
7. [Acceptance Criteria](#acceptance-criteria)

---

## Overview and Goals

### Phase 1 Scope

Phase 1 extends the Hamilton integration to make **datasets first-class citizens** in the build DAG while leveraging the existing Ibis/Pandera/SQLGlot infrastructure:

| Component | Purpose | Integration Point |
|-----------|---------|-------------------|
| `DatasetRef` | Type-safe dataset references in DAG | Bridges Hamilton nodes to DuckDB tables |
| `@dataloader/@datasaver` | Ibis-native IO | Uses `IbisGateway` for table access |
| `@extract_fields` | Dataset lineage | Exposes per-table outputs from target nodes |
| `@check_output` | Pandera validation | Uses `SCHEMA_REGISTRY` for contracts |
| `node_factory.py` | Generated nodes | Scales beyond explicit Phase 0 nodes |

### Design Principles

1. **Ibis as Query Layer**: All table reads/writes go through `IbisGateway` or `DuckDBPolicyBackend`
2. **Pandera as Contract Source**: Dataset schemas come from `SCHEMA_REGISTRY`, not inline definitions
3. **SQLGlot for SQL Generation**: Bulk operations use `DuckDBPolicyBackend` (SQLGlot-based)
4. **No Parallel Systems**: Reuse existing storage abstractions, don't create Hamilton-specific ones

---

## Architecture Integration Points

### Existing Infrastructure to Leverage

```
┌─────────────────────────────────────────────────────────────────┐
│                      Hamilton DAG Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Target Nodes│  │Dataset Nodes│  │ @check_output Pandera   │  │
│  │ (Phase 0)   │  │(@extract)   │  │                         │  │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘  │
│         │                │                      │               │
│         ▼                ▼                      ▼               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Hamilton IO Adapters (New)                     ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  ││
│  │  │ @dataloader  │  │ @datasaver   │  │  DatasetRef      │  ││
│  │  │ load_table() │  │ save_table() │  │  (typed refs)    │  ││
│  │  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  ││
│  └─────────┼─────────────────┼──────────────────┼─────────────┘│
└────────────┼─────────────────┼──────────────────┼──────────────┘
             │                 │                  │
             ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Existing Storage Layer                         │
│                                                                  │
│  ┌────────────────────────────────────────┐  ┌───────────────┐  │
│  │            IbisGateway                 │  │SCHEMA_REGISTRY│  │
│  │  ┌─────────────────────────────────┐   │  │   (Pandera)   │  │
│  │  │  Reads: table() / read() / view()│   │  └───────────────┘  │
│  │  │  Writes: write() / insert() /   │   │                     │
│  │  │          upsert() / delete()    │   │                     │
│  │  └─────────────┬───────────────────┘   │                     │
│  │                │                       │                     │
│  │                ▼ (internal)            │                     │
│  │  ┌─────────────────────────────────┐   │                     │
│  │  │     DuckDBPolicyBackend         │   │                     │
│  │  │  (SQLGlot SQL generation)       │   │                     │
│  │  │  • bulk_insert()                │   │                     │
│  │  │  • upsert()                     │   │                     │
│  │  └─────────────────────────────────┘   │                     │
│  └────────────────────┬───────────────────┘                     │
│                       │                                         │
│                       ▼                                         │
│             ┌──────────────────┐                                │
│             │     DuckDB       │                                │
│             └──────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

**Key Architecture Point**: `DuckDBPolicyBackend` is an *internal implementation detail*
of `IbisGateway`. Hamilton adapters should only interact with `IbisGateway` methods:
- `gateway.ibis.table()` / `read()` / `view()` for reads
- `gateway.ibis.write()` / `insert()` / `upsert()` for writes

### Key Classes to Integrate With

| Class | Location | Purpose |
|-------|----------|---------|
| `IbisGateway` | `codeintel.storage.ibis_adapter` | **Unified data access layer** - reads via Ibis, writes via SQLGlot |
| `SCHEMA_REGISTRY` | `codeintel.config.datasets.schema_registry` | Pandera schema lookup |
| `DatasetSchema` | `codeintel.config.datasets.schema` | Unified schema wrapper |
| `StorageGateway` | `codeintel.storage.gateway` | Gateway that exposes `.ibis` accessor |

> **Important**: `DuckDBPolicyBackend` is used **internally by `IbisGateway`** for write operations.
> Hamilton adapters should use `IbisGateway.write()` / `insert()` / `upsert()`, NOT direct `DuckDBPolicyBackend` access.

---

## Module Structure

```
src/codeintel/build/hamilton/
  io/
    __init__.py                    # Package exports
    dataset_ref.py                 # DatasetRef type and utilities
    ibis_adapter.py                # @dataloader/@datasaver implementations
    materialization.py             # Materialization strategies
  contracts/
    __init__.py                    # Package exports
    pandera_hook.py                # @check_output integration with SCHEMA_REGISTRY
    validation.py                  # Contract validation utilities
  nodes/
    dataset_nodes.py               # @extract_fields dataset node generators
    node_factory.py                # Dynamic node generation from TargetGraph
```

---

## Implementation Tasks

### Task 1: DatasetRef Type System

**Purpose**: Create a type-safe reference type for datasets that can flow through the Hamilton DAG without materializing data.

#### 1.1 DatasetRef Dataclass

```python
# src/codeintel/build/hamilton/io/dataset_ref.py
"""Type-safe dataset references for Hamilton DAG.

DatasetRef provides a lightweight reference to a DuckDB table that can
flow through the Hamilton DAG. The actual data is not materialized until
explicitly requested via the IO adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.config.datasets.schema import DatasetSchema


@dataclass(frozen=True)
class DatasetRef:
    """Reference to a dataset in the build DAG.

    This is a lightweight handle that identifies a table without loading data.
    Used to establish lineage relationships in the Hamilton DAG.

    Attributes
    ----------
    table_key
        Fully-qualified table name (e.g., "analytics.function_metrics").
    schema_version
        Optional schema version for compatibility tracking.
    row_count
        Optional row count if known from prior computation.
    source_target
        Target that produced this dataset (for lineage).

    Examples
    --------
    >>> ref = DatasetRef(
    ...     table_key="analytics.function_metrics",
    ...     source_target="function_metrics",
    ...     row_count=1500,
    ... )
    >>> ref.schema_name
    'analytics'
    >>> ref.table_name
    'function_metrics'
    """

    table_key: str
    schema_version: str | None = None
    row_count: int | None = None
    source_target: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def schema_name(self) -> str:
        """Extract schema name from table key."""
        parts = self.table_key.split(".", 1)
        return parts[0] if len(parts) > 1 else "main"

    @property
    def table_name(self) -> str:
        """Extract table name from table key."""
        parts = self.table_key.split(".", 1)
        return parts[1] if len(parts) > 1 else parts[0]

    def with_row_count(self, count: int) -> DatasetRef:
        """Return a new ref with updated row count."""
        return DatasetRef(
            table_key=self.table_key,
            schema_version=self.schema_version,
            row_count=count,
            source_target=self.source_target,
            metadata=self.metadata,
        )


def refs_from_target_result(
    target_name: str,
    table_keys: tuple[str, ...],
    row_counts: dict[str, int] | None = None,
) -> dict[str, DatasetRef]:
    """Create DatasetRef instances from a target execution result.

    Parameters
    ----------
    target_name
        Name of the target that produced these datasets.
    table_keys
        Table keys produced by the target.
    row_counts
        Optional mapping of table key to row count.

    Returns
    -------
    dict[str, DatasetRef]
        Mapping of table key to DatasetRef.
    """
    counts = row_counts or {}
    return {
        key: DatasetRef(
            table_key=key,
            source_target=target_name,
            row_count=counts.get(key),
        )
        for key in table_keys
    }
```

#### 1.2 Integration with TargetRunRecord

Update `TargetRunRecord` to include dataset references:

```python
# Addition to manifest_hook.py

@dataclass(frozen=True)
class TargetRunRecord:
    # ... existing fields ...
    
    # New field for Phase 1
    datasets: tuple[DatasetRef, ...] = ()
    
    def get_dataset(self, table_key: str) -> DatasetRef | None:
        """Get a specific dataset ref by table key."""
        for ds in self.datasets:
            if ds.table_key == table_key:
                return ds
        return None
```

---

### Task 2: Ibis-Native DataLoader/DataSaver

**Purpose**: Implement Hamilton IO modifiers that use `IbisGateway` for table access.

#### 2.1 Ibis DataLoader

```python
# src/codeintel/build/hamilton/io/ibis_adapter.py
"""Ibis-native IO adapters for Hamilton materialization.

These adapters integrate Hamilton's @dataloader/@datasaver pattern with
the existing IbisGateway infrastructure for DuckDB access.

IMPORTANT: All DuckDB operations go through IbisGateway:
- Reads: IbisGateway.table() / read() / view()
- Writes: IbisGateway.write() / insert() / upsert()

IbisGateway internally delegates writes to DuckDBPolicyBackend (SQLGlot).
Do NOT use DuckDBPolicyBackend directly from Hamilton adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hamilton.function_modifiers import dataloader, datasaver

from codeintel.build.hamilton.io.dataset_ref import DatasetRef

if TYPE_CHECKING:
    import pandas as pd
    import ibis.expr.types as ir

    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class IbisIOConfig:
    """Configuration for Ibis IO operations.

    Attributes
    ----------
    gateway
        Storage gateway for database access (use gateway.ibis for operations).
    validate_schema
        Whether to validate against Pandera schema on load/save.
    """

    gateway: StorageGateway
    validate_schema: bool = True


@dataloader()
def load_ibis_table(
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> tuple[ir.Table, dict[str, Any]]:
    """Load a table as an Ibis expression.

    Uses IbisGateway.table() which handles qualified name splitting
    correctly for Ibis 11.

    Parameters
    ----------
    dataset_ref
        Reference to the table to load.
    io_config
        IO configuration with gateway access.

    Returns
    -------
    tuple[ir.Table, dict[str, Any]]
        Ibis table expression and metadata dict.
    """
    # Use IbisGateway.table() for reads - handles qualified names correctly
    table = io_config.gateway.ibis.table(dataset_ref.table_key)
    
    metadata = {
        "source": "duckdb",
        "table_key": dataset_ref.table_key,
        "schema": dataset_ref.schema_name,
        "table": dataset_ref.table_name,
    }
    
    return table, metadata


@datasaver()
def save_ibis_expression(
    output: ir.Table,
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save an Ibis expression to DuckDB.

    Uses IbisGateway.write() which generates INSERT...SELECT via SQLGlot.

    Parameters
    ----------
    output
        Ibis table expression to save.
    dataset_ref
        Reference specifying where to save.
    io_config
        IO configuration with gateway access.

    Returns
    -------
    dict[str, Any]
        Metadata about the save operation.
    """
    # Use IbisGateway.write() for Ibis expression writes
    # This generates INSERT...SELECT via SQLGlot internally
    result = io_config.gateway.ibis.write(
        dataset_ref.table_key,
        output,
    )
    
    return {
        "saved_to": "duckdb",
        "table_key": dataset_ref.table_key,
        "row_count": result.rows_affected,
        "method": result.method,
    }


def load_table_as_dataframe(
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a table as a pandas DataFrame.

    Convenience wrapper that executes the Ibis expression.

    Parameters
    ----------
    dataset_ref
        Reference to the table to load.
    io_config
        IO configuration with gateway access.

    Returns
    -------
    tuple[DataFrame, dict[str, Any]]
        Pandas DataFrame and metadata.
    """
    table, metadata = load_ibis_table(dataset_ref, io_config)
    df = table.execute()
    metadata["format"] = "pandas"
    return df, metadata
```

#### 2.2 Bulk Operations via IbisGateway

All bulk insert/upsert operations go through `IbisGateway.write()` which internally
delegates to `DuckDBPolicyBackend` for SQLGlot-based SQL generation:

```python
# Addition to ibis_adapter.py

from collections.abc import Sequence


@datasaver()
def save_dataframe(
    df: pd.DataFrame,
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save a pandas DataFrame to DuckDB.

    Uses IbisGateway.write() which internally uses DuckDBPolicyBackend
    for efficient INSERT...VALUES via SQLGlot.

    Parameters
    ----------
    df
        DataFrame to save.
    dataset_ref
        Target table reference.
    io_config
        IO configuration.

    Returns
    -------
    dict[str, Any]
        Write operation metadata.
    """
    # IbisGateway.write() accepts DataFrames directly
    # Internally uses DuckDBPolicyBackend.bulk_insert()
    result = io_config.gateway.ibis.write(
        dataset_ref.table_key,
        df,
    )
    
    return {
        "operation": "insert_values",
        "table_key": dataset_ref.table_key,
        "row_count": result.rows_affected,
        "method": result.method,
    }


@datasaver()
def save_rows(
    rows: Sequence[tuple[object, ...]],
    columns: Sequence[str],
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save row tuples to DuckDB.

    Uses IbisGateway.write() which internally uses DuckDBPolicyBackend
    for efficient INSERT...VALUES via SQLGlot.

    Parameters
    ----------
    rows
        Sequence of row tuples.
    columns
        Column names matching row tuple positions.
    dataset_ref
        Target table reference.
    io_config
        IO configuration.

    Returns
    -------
    dict[str, Any]
        Write operation metadata.
    """
    # IbisGateway.write() accepts tuples directly
    # Internally uses DuckDBPolicyBackend.bulk_insert()
    result = io_config.gateway.ibis.write(
        dataset_ref.table_key,
        rows,
        columns=columns,
    )
    
    return {
        "operation": "insert_values",
        "table_key": dataset_ref.table_key,
        "row_count": result.rows_affected,
        "method": result.method,
    }


@datasaver()
def upsert_dataframe(
    df: pd.DataFrame,
    dataset_ref: DatasetRef,
    conflict_columns: Sequence[str],
    update_columns: Sequence[str],
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Upsert a DataFrame using INSERT...ON CONFLICT.

    Uses IbisGateway.upsert() which internally uses DuckDBPolicyBackend
    for SQLGlot-based UPSERT generation.

    Parameters
    ----------
    df
        DataFrame to upsert.
    dataset_ref
        Target table reference.
    conflict_columns
        Columns defining uniqueness constraint.
    update_columns
        Columns to update on conflict.
    io_config
        IO configuration.

    Returns
    -------
    dict[str, Any]
        Upsert operation metadata.
    """
    # IbisGateway.upsert() handles ON CONFLICT semantics
    # Internally uses DuckDBPolicyBackend.upsert()
    result = io_config.gateway.ibis.upsert(
        dataset_ref.table_key,
        df,
        columns=list(df.columns),
        conflict_columns=conflict_columns,
        update_columns=update_columns,
    )
    
    return {
        "operation": "upsert",
        "table_key": dataset_ref.table_key,
        "row_count": result.rows_affected,
        "method": result.method,
    }
```

---

### Task 3: Dataset Extraction Nodes

**Purpose**: Use `@extract_fields` to expose individual datasets from target nodes for lineage tracking.

#### 3.1 Dataset Node Generator

```python
# src/codeintel/build/hamilton/nodes/dataset_nodes.py
"""Dataset extraction nodes for Hamilton lineage.

This module generates nodes that expose individual datasets from target
execution results, enabling fine-grained lineage tracking in the DAG.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hamilton.function_modifiers import extract_fields, tag

from codeintel.build.hamilton.io.dataset_ref import DatasetRef, refs_from_target_result
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.naming import dataset_node
from codeintel.build.registry import ALL_TARGETS

if TYPE_CHECKING:
    pass


def _build_extract_dict(table_keys: tuple[str, ...]) -> dict[str, type]:
    """Build the extract_fields type dict from table keys."""
    return {dataset_node(key): DatasetRef for key in table_keys}


# Example: Manually define dataset extraction for call_graph
# This pattern will be generalized in node_factory.py

@extract_fields(_build_extract_dict(("graph.call_graph_edges", "graph.call_graph_nodes")))
@tag(domain="graphs", produces="datasets")
def d__call_graph(t__call_graph: TargetRunRecord) -> dict[str, DatasetRef]:
    """Extract dataset refs from call_graph target result.

    Parameters
    ----------
    t__call_graph
        Execution record from call_graph target.

    Returns
    -------
    dict[str, DatasetRef]
        Mapping of dataset node names to refs.
    """
    table_keys = ("graph.call_graph_edges", "graph.call_graph_nodes")
    refs = refs_from_target_result(
        target_name="call_graph",
        table_keys=table_keys,
        row_counts=dict(t__call_graph.row_counts),
    )
    # Convert to node names for extract_fields
    return {dataset_node(key): ref for key, ref in refs.items()}


@extract_fields(_build_extract_dict(("analytics.function_metrics",)))
@tag(domain="analytics", produces="datasets")
def d__function_metrics(t__function_metrics: TargetRunRecord) -> dict[str, DatasetRef]:
    """Extract dataset refs from function_metrics target result."""
    table_keys = ("analytics.function_metrics",)
    refs = refs_from_target_result(
        target_name="function_metrics",
        table_keys=table_keys,
        row_counts=dict(t__function_metrics.row_counts),
    )
    return {dataset_node(key): ref for key, ref in refs.items()}


@extract_fields(_build_extract_dict(("analytics.risk_factors",)))
@tag(domain="analytics", produces="datasets")
def d__risk_factors(t__risk_factors: TargetRunRecord) -> dict[str, DatasetRef]:
    """Extract dataset refs from risk_factors target result."""
    table_keys = ("analytics.risk_factors",)
    refs = refs_from_target_result(
        target_name="risk_factors",
        table_keys=table_keys,
        row_counts=dict(t__risk_factors.row_counts),
    )
    return {dataset_node(key): ref for key, ref in refs.items()}
```

---

### Task 4: Pandera Contract Integration

**Purpose**: Integrate Hamilton's `@check_output` with the existing `SCHEMA_REGISTRY` and Pandera schemas.

#### 4.1 Pandera Hook Implementation

```python
# src/codeintel/build/hamilton/contracts/pandera_hook.py
"""Pandera contract integration for Hamilton nodes.

This module provides utilities to attach Pandera validation to Hamilton
node outputs using the existing SCHEMA_REGISTRY as the schema source.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from hamilton.function_modifiers import check_output

from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY

if TYPE_CHECKING:
    import pandas as pd
    import pandera as pa

    from codeintel.build.hamilton.io.dataset_ref import DatasetRef

F = TypeVar("F", bound=Callable[..., Any])


def get_pandera_schema(table_key: str) -> pa.DataFrameSchema | None:
    """Retrieve Pandera schema from registry.

    Parameters
    ----------
    table_key
        Fully-qualified table name.

    Returns
    -------
    pa.DataFrameSchema | None
        Pandera schema if registered, None otherwise.
    """
    dataset_schema = SCHEMA_REGISTRY.get(table_key)
    if dataset_schema is None:
        return None
    return dataset_schema.pandera_schema


def validate_dataframe(df: pd.DataFrame, table_key: str) -> pd.DataFrame:
    """Validate a DataFrame against its registered schema.

    Parameters
    ----------
    df
        DataFrame to validate.
    table_key
        Table key for schema lookup.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame (may be coerced).

    Raises
    ------
    pa.errors.SchemaError
        If validation fails.
    ValueError
        If no schema is registered for the table key.
    """
    schema = get_pandera_schema(table_key)
    if schema is None:
        msg = f"No Pandera schema registered for {table_key}"
        raise ValueError(msg)
    return schema.validate(df)


def with_contract(table_key: str) -> Callable[[F], F]:
    """Decorator to attach Pandera validation to a node output.

    Uses @check_output from Hamilton with the schema from SCHEMA_REGISTRY.

    Parameters
    ----------
    table_key
        Table key for schema lookup.

    Returns
    -------
    Callable[[F], F]
        Decorated function with Pandera validation.

    Examples
    --------
    >>> @with_contract("analytics.function_metrics")
    ... def compute_metrics(data: pd.DataFrame) -> pd.DataFrame:
    ...     return process(data)
    """
    schema = get_pandera_schema(table_key)
    
    def decorator(func: F) -> F:
        if schema is None:
            # No schema registered - return unchanged
            return func
        
        # Apply Hamilton's check_output with Pandera schema
        return check_output(schema=schema)(func)
    
    return decorator


def validate_dataset_ref(
    ref: DatasetRef,
    gateway: Any,
) -> tuple[bool, str | None]:
    """Validate a DatasetRef's underlying table against its schema.

    Parameters
    ----------
    ref
        Dataset reference to validate.
    gateway
        Storage gateway for table access.

    Returns
    -------
    tuple[bool, str | None]
        (is_valid, error_message) tuple.
    """
    schema = get_pandera_schema(ref.table_key)
    if schema is None:
        return True, None  # No schema = no validation
    
    try:
        # Load table and validate
        table = gateway.ibis.table(ref.table_key)
        df = table.execute()
        schema.validate(df)
        return True, None
    except Exception as e:
        return False, str(e)
```

#### 4.2 Contract-Aware Dataset Nodes

```python
# Addition to dataset_nodes.py

from codeintel.build.hamilton.contracts.pandera_hook import with_contract


@with_contract("analytics.function_metrics")
def validated_function_metrics(
    d__analytics__function_metrics: DatasetRef,
    io_config: IbisIOConfig,
) -> pd.DataFrame:
    """Load and validate function_metrics dataset.

    This node loads the dataset and validates it against its Pandera schema.
    Downstream nodes can depend on this for validated data.

    Parameters
    ----------
    d__analytics__function_metrics
        Dataset reference from extraction node.
    io_config
        IO configuration for table access.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame.
    """
    df, _ = load_table_as_dataframe(d__analytics__function_metrics, io_config)
    return df
```

---

### Task 5: Node Factory for Generated Nodes

**Purpose**: Dynamically generate Hamilton nodes from the TargetGraph, scaling beyond explicit definitions.

#### 5.1 Node Factory Implementation

```python
# src/codeintel/build/hamilton/nodes/node_factory.py
"""Dynamic node generation from TargetGraph metadata.

This module generates Hamilton nodes programmatically from the target
graph, enabling automatic coverage of all targets without manual
node definitions.
"""

from __future__ import annotations

import inspect
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.metadata_bridge import from_target
from codeintel.build.hamilton.naming import target_node
from codeintel.build.hamilton.nodes.targets_phase0 import _run_target
from codeintel.build.registry import get_target_graph

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.targets import TargetGraph


def _create_node_function(
    target_name: str,
    dep_node_names: list[str],
    domain: str,
) -> Callable[..., TargetRunRecord]:
    """Create a Hamilton node function for a target.

    Parameters
    ----------
    target_name
        Name of the target to wrap.
    dep_node_names
        Hamilton node names of dependencies.
    domain
        Domain for tagging (ingestion, graphs, analytics).

    Returns
    -------
    Callable[..., TargetRunRecord]
        Node function with correct signature for Hamilton.
    """
    def node_fn(**kwargs: Any) -> TargetRunRecord:
        env: BuildEnv = kwargs["env"]
        graph: TargetGraph = kwargs["graph"]
        # Dependencies are received but used only for DAG ordering
        return _run_target(env=env, graph=graph, target_name=target_name)

    # Build signature that Hamilton can inspect
    params = [
        inspect.Parameter("env", inspect.Parameter.KEYWORD_ONLY, annotation="BuildEnv"),
        inspect.Parameter("graph", inspect.Parameter.KEYWORD_ONLY, annotation="TargetGraph"),
    ]
    # Add dependency parameters
    for dep_name in dep_node_names:
        params.append(
            inspect.Parameter(
                dep_name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=TargetRunRecord,
            )
        )

    node_fn.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    node_fn.__name__ = target_node(target_name)
    node_fn.__doc__ = f"Execute the {target_name} target ({domain})."

    # Apply tag decorator
    return tag(domain=domain, target=target_name)(node_fn)


def build_target_module(
    *,
    include_targets: set[str] | None = None,
    exclude_targets: set[str] | None = None,
) -> ModuleType:
    """Generate a module containing Hamilton nodes for all targets.

    Parameters
    ----------
    include_targets
        If provided, only generate nodes for these targets.
    exclude_targets
        If provided, exclude these targets from generation.

    Returns
    -------
    ModuleType
        Module containing generated node functions.

    Examples
    --------
    >>> module = build_target_module(exclude_targets={"export_jsonl"})
    >>> driver = hamilton.driver.Driver({}, module)
    """
    graph = get_target_graph()
    include = include_targets or {t.name for t in graph.all_targets}
    exclude = exclude_targets or set()

    # Create module
    module = ModuleType("codeintel.build.hamilton.nodes.generated")
    module.__doc__ = "Auto-generated Hamilton nodes from TargetGraph."

    # Track generated node names for TARGET_TO_NODE mapping
    target_to_node: dict[str, str] = {}

    for target in graph.all_targets:
        if target.name not in include or target.name in exclude:
            continue

        # Get metadata for domain
        meta = from_target(target)

        # Map dependencies to node names
        dep_node_names = [target_node(dep) for dep in target.dependencies]

        # Create and register node function
        node_fn = _create_node_function(
            target_name=target.name,
            dep_node_names=dep_node_names,
            domain=meta.domain,
        )

        node_name = target_node(target.name)
        setattr(module, node_name, node_fn)
        target_to_node[target.name] = node_name

    # Attach mapping for executor lookups
    module.TARGET_TO_NODE = target_to_node  # type: ignore[attr-defined]

    return module


def get_generated_module() -> ModuleType:
    """Get or create the generated nodes module.

    Returns
    -------
    ModuleType
        Cached generated module.
    """
    # Could add caching here if needed
    return build_target_module()
```

#### 5.2 Updated Driver Factory

```python
# Update to driver_factory.py

from codeintel.build.hamilton.nodes import targets_phase0
from codeintel.build.hamilton.nodes.node_factory import build_target_module


def build_driver(
    *,
    config: dict[str, Any],
    use_generated: bool = False,
) -> HamiltonRuntime:
    """Build a Hamilton Driver for build execution.

    Parameters
    ----------
    config
        Configuration dict passed to Hamilton Driver.
    use_generated
        If True, use dynamically generated nodes instead of Phase 0 explicit nodes.

    Returns
    -------
    HamiltonRuntime
        Runtime containing Driver and TargetGraph.
    """
    graph = get_target_graph()
    
    if use_generated:
        nodes_module = build_target_module()
    else:
        nodes_module = targets_phase0
    
    dr = driver.Driver(config, nodes_module)
    return HamiltonRuntime(dr=dr, graph=graph)
```

---

## Testing Strategy

All tests follow the Testing Charter (no monkeypatching, production-parity):

### Test Files

```
tests/build/hamilton/
  __init__.py
  test_dataset_ref.py           # DatasetRef type and utilities
  test_ibis_adapter.py          # IO adapters with real DuckDB
  test_pandera_integration.py   # Contract validation with SCHEMA_REGISTRY
  test_node_factory.py          # Generated nodes from TargetGraph
  test_dataset_nodes.py         # @extract_fields dataset extraction
```

### Test Fixtures

```python
# tests/_helpers/hamilton_fixtures.py additions

@pytest.fixture
def io_config(analytics_gateway: StorageGateway) -> IbisIOConfig:
    """Create IO config for Hamilton adapter tests."""
    return IbisIOConfig(
        gateway=analytics_gateway,
        validate_schema=True,
    )


@pytest.fixture
def seeded_dataset_ref(seeded_analytics_gateway: StorageGateway) -> DatasetRef:
    """Create a DatasetRef pointing to seeded test data."""
    return DatasetRef(
        table_key="analytics.function_metrics",
        source_target="function_metrics",
    )
```

### Example Test Cases

```python
# tests/build/hamilton/test_ibis_adapter.py

class TestIbisDataLoader:
    """Tests for Ibis-based data loading."""

    @staticmethod
    def test_load_returns_ibis_table(
        seeded_dataset_ref: DatasetRef,
        io_config: IbisIOConfig,
    ) -> None:
        """Verify load_ibis_table returns an Ibis table expression."""
        table, metadata = load_ibis_table(seeded_dataset_ref, io_config)
        
        # Verify it's an Ibis table
        if not hasattr(table, "execute"):
            pytest.fail("Result is not an Ibis table")
        
        # Verify metadata
        if metadata.get("table_key") != seeded_dataset_ref.table_key:
            pytest.fail("Metadata table_key mismatch")

    @staticmethod
    def test_load_respects_qualified_names(
        io_config: IbisIOConfig,
    ) -> None:
        """Verify qualified table names are handled correctly (Ibis 11)."""
        ref = DatasetRef(table_key="analytics.function_metrics")
        table, _ = load_ibis_table(ref, io_config)
        
        # Should not raise - Ibis 11 handles qualified names via database param
        df = table.limit(1).execute()
        if df is None:
            pytest.fail("Table load failed")
```

---

## Migration Path

### Phase 1a: Foundation (Week 1-2)

1. Implement `DatasetRef` type system
2. Add `IbisIOConfig` and basic loaders
3. Update `TargetRunRecord` with dataset refs
4. Add tests for new types

### Phase 1b: Dataset Nodes (Week 2-3)

1. Implement `@extract_fields` dataset nodes for Phase 0 targets
2. Add dataset nodes to driver factory
3. Verify lineage in Hamilton DAG visualization

### Phase 1c: Contracts (Week 3-4)

1. Implement Pandera hook with `SCHEMA_REGISTRY` integration
2. Add `@check_output` to critical dataset nodes
3. Create validated dataset nodes for serving layer

### Phase 1d: Node Factory (Week 4-5)

1. Implement `node_factory.py` for dynamic generation
2. Add `--use-generated` flag to CLI
3. Validate generated nodes match Phase 0 behavior

### Phase 1e: Full Rollout (Week 5-6)

1. Switch default to generated nodes
2. Deprecate Phase 0 explicit nodes (keep for reference)
3. Documentation and runbook updates

---

## Acceptance Criteria

### Functional Requirements

1. **DatasetRef Integration**
   - [ ] DatasetRef flows through Hamilton DAG
   - [ ] Target nodes produce DatasetRef tuples
   - [ ] Dataset nodes extract individual refs via @extract_fields

2. **Ibis IO Adapters**
   - [ ] `load_ibis_table` returns Ibis table expression
   - [ ] `save_ibis_table` persists to DuckDB correctly
   - [ ] Qualified names work with Ibis 11 patterns

3. **Pandera Contracts**
   - [ ] `@check_output` validates against SCHEMA_REGISTRY
   - [ ] Validation errors are surfaced correctly
   - [ ] Contract nodes can be added to critical paths

4. **Node Factory**
   - [ ] Generated nodes match Phase 0 explicit nodes
   - [ ] All targets in TargetGraph have generated nodes
   - [ ] CLI supports `--use-generated` flag

### Quality Requirements

1. **Type Safety**
   - [ ] `pyright --strict` passes with 0 errors
   - [ ] `pyrefly check` passes with 0 errors
   - [ ] No `type: ignore` or `noqa` suppressions

2. **Code Quality**
   - [ ] `ruff check` passes with 0 errors
   - [ ] All new code has NumPy-style docstrings
   - [ ] Cyclomatic complexity ≤ 10

3. **Test Coverage**
   - [ ] All new modules have test files
   - [ ] Tests use real DuckDB (no mocking)
   - [ ] Edge cases covered (empty tables, missing schemas)

---

## Dependencies

### Required Existing Modules

- `codeintel.storage.ibis_adapter.IbisGateway` - Unified data access (wraps DuckDBPolicyBackend internally)
- `codeintel.storage.gateway.StorageGateway` - Gateway with `.ibis` accessor
- `codeintel.config.datasets.schema_registry.SCHEMA_REGISTRY` - Pandera schema lookup
- `codeintel.build.hamilton.*` (Phase 0 infrastructure)

> **Note**: Do NOT import `DuckDBPolicyBackend` directly in Hamilton adapters.
> Use `gateway.ibis.write()` which delegates to it internally.

### External Dependencies

- `sf-hamilton>=1.89.0` (already in pyproject.toml)
- `ibis-framework[duckdb]>=11.0.0`
- `pandera>=0.18.0`

---

## References

- [Hamilton IO Modifiers](https://hamilton.dagworks.io/en/latest/concepts/io/)
- [Hamilton check_output](https://hamilton.dagworks.io/en/latest/reference/decorators/check_output/)
- [Pandera DataFrame Schemas](https://pandera.readthedocs.io/en/stable/dataframe_schemas.html)
- [Ibis 11 Migration Notes](docs/migrations/ibis-11-migration.md)
- [SCHEMA_REGISTRY Architecture](openspec/plans/pandera-schema-unification-architecture.md)

