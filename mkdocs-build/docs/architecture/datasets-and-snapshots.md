# Datasets and Snapshots

This page documents the snapshot model and dataset contract system used
throughout CodeIntel.

## Snapshot Model

A **snapshot** represents a point-in-time view of a repository for analysis.

### SnapshotRef

The core identity type for snapshots:

```python
from codeintel.config import SnapshotRef

snapshot = SnapshotRef(
    repo="my-org/my-repo",
    commit="abc123def",
    repo_root=Path("/path/to/repo"),
)
```

| Field | Description |
|-------|-------------|
| `repo` | Repository identifier (org/name format) |
| `commit` | Git commit SHA or reference |
| `repo_root` | Local filesystem path to repository |

### BuildPaths

Paths for build artifacts:

```python
from codeintel.config import BuildPaths

paths = BuildPaths.for_repo_root(repo_root)
# paths.build_dir -> repo_root/build
# paths.db_path -> repo_root/build/db/codeintel.duckdb
```

## Dataset Contracts

All persistent data has explicit **dataset contracts** that define:

- Schema (column names, types, constraints)
- Ownership (which module produces the data)
- Lifecycle (when data is created/updated)
- Validation rules

### Contract Definition

Contracts are defined in `codeintel.config.datasets`:

```python
from codeintel.config.datasets import DatasetContract

CONTRACT = DatasetContract(
    name="analytics.function_metrics",
    schema=FUNCTION_METRICS_SCHEMA,
    owner="analytics.plugins.function_metrics",
)
```

### Schema Validation

Schemas are validated at:

1. **Write time**: Data must match contract schema
2. **Read time**: Contract presence verified
3. **Build time**: Schema alignment checked

## Storage Gateway

The `StorageGateway` provides typed access to all datasets:

```python
from codeintel.storage.gateway import open_gateway

gateway = open_gateway(config)

# Access datasets by namespace
functions = gateway.analytics.functions()
call_graph = gateway.graph.call_graph()
```

## Key Datasets

### Core Schema

| Dataset | Description |
|---------|-------------|
| `core.goids` | Global object identifiers |
| `core.modules` | Module index |

### Analytics Schema

| Dataset | Description |
|---------|-------------|
| `analytics.functions` | Function metadata |
| `analytics.function_metrics` | Complexity, size metrics |
| `analytics.profiles` | Module profiles |
| `analytics.risk_factors` | Risk assessment |

### Graph Schema

| Dataset | Description |
|---------|-------------|
| `graph.call_graph` | Function call relationships |
| `graph.import_graph` | Module import relationships |
| `graph.cfg` | Control flow graphs |
| `graph.dfg` | Data flow graphs |

## Lifecycle

1. **Ingestion** creates base datasets (AST, modules, functions)
2. **Graphs** builds relationship datasets (call graph, imports)
3. **Analytics** computes derived datasets (metrics, profiles, risk)

