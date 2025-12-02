Here’s a concrete, file-by-file implementation plan for **Epic 1 – Contract-driven dataflow graph** that you can hand directly to an implementation agent (or follow yourself).

I’ll structure it as:

1. `config.dataset_contract`: define dataflow model + dataset/docs edges.
2. `storage.metadata_bootstrap` + `storage.repositories`: persist graph into `metadata.dataset_dataflow_*`.
3. `serving.registry`: add operation + graph-level nodes/edges and a combined graph builder.
4. `serving.http.routes.meta`: add `/meta/dataflow`.
5. `serving.mcp.meta_tools`: add MCP tools (`explain_dataset`, `explain_operation`, `explain_path`).
6. Tests: new + extended tests to lock in contracts.

Where I show “full” snippets, they’re meant to be dropped in as-is; where I show partials, they’re focused diffs around the parts that change.

---

## 1. `config.dataset_contract`: core dataflow model + dataset/docs edges

### 1.1 Add `DataflowNode` / `DataflowEdge` types

In **`config/config/dataset_contract.py`**, near the top where you define `RowBinding` / `DatasetContract`, add two new dataclasses and some type aliases.

Right after `RowBinding` / `DatasetContract` is defined (and you already import `Literal`, `Final`, `TypedDict`, etc.), insert:

```python
# ---------------------------------------------------------------------------
# Section 0.x: Dataflow graph primitives
# ---------------------------------------------------------------------------

NodeKind = Literal["table", "view", "operation", "graph"]
EdgeType = Literal["builds", "reads", "exposes", "depends_on"]


@dataclass(frozen=True)
class DataflowNode:
    """
    Node in the logical dataflow graph for CodeIntel datasets, views, and runtimes.

    Attributes
    ----------
    id
        Stable identifier for this node, e.g.:

        - "analytics.function_metrics"
        - "docs.v_function_summary"
        - "profiles.function"
        - "graph.callgraph"

    kind
        High-level category: "table", "view", "operation", or "graph".

    family
        Optional dataset family, e.g. "core", "analytics", "docs".

    owner_package
        Optional owning package from the dataset contract (core, analytics, graphs, qa, docs).

    description
        Optional human-readable description.
    """

    id: str
    kind: NodeKind
    family: str | None = None
    owner_package: Literal["core", "analytics", "graphs", "qa", "docs"] | None = None
    description: str | None = None


@dataclass(frozen=True)
class DataflowEdge:
    """
    Directed edge in the dataflow graph.

    Attributes
    ----------
    src
        Source node id (upstream dataset, view, or runtime).
    dst
        Destination node id (downstream dataset, view, or operation).
    edge_type
        Relationship type:

        - "builds"      : upstream dataset/view used to build downstream dataset/view.
        - "reads"       : dataset read by an operation.
        - "exposes"     : dataset exposed by an operation as an API surface.
        - "depends_on"  : operation depends on a logical graph runtime.
    """

    src: str
    dst: str
    edge_type: EdgeType
```

This gives all layers a shared, typed representation for the dataflow graph.

> **Convention:**
> For datasets/views, we’ll treat `DataflowNode.id == DatasetContract.table_key` (e.g. `"analytics.function_profile"`, `"docs.v_function_summary"`).
> For operations, `id == OperationSpec.id` (e.g. `"profiles.function"`).
> For graph runtimes, `id == f"graph.{graph_name}"` (e.g. `"graph.callgraph"`).

---

### 1.2 Dataset nodes: `iter_dataset_nodes`

At the **bottom** of `dataset_contract.py`, after the `DATASET_CONTRACTS` and `*_BY_DATASET_NAME` dicts are defined, add dataset-node builders.

First, ensure you have the imports:

```python
from collections.abc import Iterable, Iterator
```

Then add:

```python
# ---------------------------------------------------------------------------
# Section N: Dataflow graph builders (dataset + docs layer)
# ---------------------------------------------------------------------------


def iter_dataset_nodes() -> Iterator[DataflowNode]:
    """
    Yield DataflowNode entries for every DatasetContract.

    Node IDs
    --------
    We use DatasetContract.table_key as the canonical id, e.g.:

        - "analytics.function_profile"
        - "graph.call_graph_edges"
        - "docs.v_function_summary"
    """
    for contract in DATASET_CONTRACTS.values():
        kind: NodeKind = "view" if contract.is_view else "table"
        yield DataflowNode(
            id=contract.table_key,
            kind=kind,
            family=contract.family,
            owner_package=contract.owner_package,
            description=contract.description,
        )
```

---

### 1.3 Composite/profile edges from `COMPOSITE_SCHEMAS`

Add an edge builder that reflects the `CompositeSchema.composed_of` relationships (source tables → profile tables):

```python
def iter_composite_edges() -> Iterator[DataflowEdge]:
    """
    Yield "builds" edges for profile datasets defined in COMPOSITE_SCHEMAS.

    Each edge represents a source analytics table contributing to a profile table:

        analytics.function_metrics --> analytics.function_profile
        analytics.function_types   --> analytics.function_profile
        ...
    """
    for table_key, composition in COMPOSITE_SCHEMAS.items():
        target = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
        if target is None:
            # Should not happen; COMPOSITE_SCHEMAS keys should align with TABLE_SCHEMAS
            continue

        dst_id = target.table_key
        for src_table_key in composition.composed_of:
            upstream = DATASET_CONTRACTS_BY_TABLE_KEY.get(src_table_key)
            if upstream is None:
                # Misconfiguration: allow missing (for forward compatibility) but skip edge
                continue
            yield DataflowEdge(
                src=upstream.table_key,
                dst=dst_id,
                edge_type="builds",
            )
```

This encodes “analytics.* → analytics.function_profile” etc. without touching DuckDB.

---

### 1.4 Dataset dependency edges from `DatasetContract.upstream_dependencies`

You already populate `DatasetContract.upstream_dependencies` from `_DEPENDENCIES_BY_DATASET_NAME`. Add a second builder that uses those:

```python
def iter_dependency_edges() -> Iterator[DataflowEdge]:
    """
    Yield "builds" edges from DatasetContract.upstream_dependencies.

    These dependencies capture higher-level compositions like:

        call_graph_edges  --> function_profile
        symbol_use_edges  --> function_profile
        test_profile      --> behavioral_coverage
        data_model_fields --> data_model_relationships
    """
    name_to_contract = DATASET_CONTRACTS

    for contract in name_to_contract.values():
        if not contract.upstream_dependencies:
            continue

        dst_id = contract.table_key
        for upstream_name in contract.upstream_dependencies:
            upstream = name_to_contract.get(upstream_name)
            if upstream is None:
                # Dependency refers to a dataset not in this registry; skip for now.
                continue

            yield DataflowEdge(
                src=upstream.table_key,
                dst=dst_id,
                edge_type="builds",
            )
```

You’ll end up with both:

* fine-grained, table-level edges from `COMPOSITE_SCHEMAS`, and
* coarser, conceptual edges from `upstream_dependencies`.

That’s good: one is “physical composition”, the other is “logical lineage”.

---

### 1.5 Docs view edges: alias + (optionally) derived views

We want docs views to show up in the graph as “derived from analytics.* tables”.

There are two sources:

1. **Alias docs views** in `storage.views.ALIAS_DOCS_VIEWS` (straight alias).
2. **Derived docs views** in `DERIVED_DOCS_VIEWS` (e.g. `docs.v_function_summary`).

We’ll cover (1) now, and make (2) easy to extend.

#### 1.5.1 Extend `_DEPENDENCIES_BY_DATASET_NAME` for key docs views (optional)

In `dataset_contract.py`, you already have:

```python
_DEPENDENCIES_BY_DATASET_NAME: Final[dict[str, tuple[str, ...]]] = {
    "function_profile": ("call_graph_edges", "symbol_use_edges"),
    "file_profile": ("call_graph_edges",),
    "module_profile": ("call_graph_edges", "symbol_use_edges"),
    "test_profile": ("test_coverage_edges",),
    "behavioral_coverage": ("test_profile",),
    "data_model_relationships": ("data_model_fields",),
}
```

You can (optionally) enrich this with docs views to get richer lineage. For example:

```python
_DEPENDENCIES_BY_DATASET_NAME: Final[dict[str, tuple[str, ...]]] = {
    "function_profile": ("call_graph_edges", "symbol_use_edges"),
    "file_profile": ("call_graph_edges",),
    "module_profile": ("call_graph_edges", "symbol_use_edges"),
    "test_profile": ("test_coverage_edges",),
    "behavioral_coverage": ("test_profile",),
    "data_model_relationships": ("data_model_fields",),

    # Docs views – *conceptual* lineage (not 1:1 with view SQL, but helpful)
    "v_function_summary": (
        "function_metrics",
        "function_types",
        "coverage_functions",
        "goid_risk_factors",
    ),
    # similar entries for v_function_architecture, v_subsystem_summary, etc.
}
```

You don’t need to fill *every* docs view immediately; start with ones that are heavily used (`v_function_summary`, `v_subsystem_summary`) and extend gradually.

This will automatically feed into `iter_dependency_edges()`.

#### 1.5.2 Add `iter_docs_view_edges` using the alias map

At the bottom, alongside the other dataflow builders, add:

```python
def iter_docs_view_alias_edges() -> Iterator[DataflowEdge]:
    """
    Yield "builds" edges for docs views that are pure aliases.

    These are views like:

        docs.v_function_profile -> analytics.function_profile

    which we treat as:

        analytics.function_profile --> docs.v_function_profile  (builds)
    """
    # Local import to avoid import cycles during package initialization
    from codeintel.storage.views import ALIAS_DOCS_VIEWS

    for view_key, table_key in ALIAS_DOCS_VIEWS.items():
        # view_key like "docs.v_function_profile"
        # table_key like "analytics.function_profile"
        yield DataflowEdge(
            src=table_key,
            dst=view_key,
            edge_type="builds",
        )
```

We *don’t* create `DataflowNode` entries for alias views here yet; that’s fine, because alias views are not first-class dataset contracts — they’re mostly convenience wrappers. If you want them as nodes later, you can add a small `iter_docs_view_alias_nodes()` that yields `DataflowNode(id=view_key, kind="view")`.

---

### 1.6 Aggregate contract-level dataflow graph

Finally, expose a single helper that builds the full **dataset/docs** graph, ready for storage/serving to consume:

```python
def build_contract_dataflow_graph() -> tuple[list[DataflowNode], list[DataflowEdge]]:
    """
    Build the dataset/docs layer of the dataflow graph from static contracts.

    Returns
    -------
    nodes
        Dataflow nodes for all datasets/views (DatasetContract-based).
    edges
        Dataflow edges derived from composite schemas, explicit dependencies,
        and docs-view aliases.
    """
    nodes = list(iter_dataset_nodes())

    edges_iterables: list[Iterable[DataflowEdge]] = [
        iter_composite_edges(),
        iter_dependency_edges(),
        iter_docs_view_alias_edges(),
    ]

    # Flatten + dedupe edges
    seen: set[tuple[str, str, str]] = set()
    edges: list[DataflowEdge] = []
    for edges_iter in edges_iterables:
        for edge in edges_iter:
            key = (edge.src, edge.dst, edge.edge_type)
            if key in seen:
                continue
            seen.add(key)
            edges.append(edge)

    return nodes, edges
```

This is the **“thin adapter hook”** that storage + serving will build upon: the **core contract graph** lives entirely in `config.dataset_contract`, with no knowledge of OperationSpec.

---

## 2. `storage`: persist graph into `metadata.dataset_dataflow_*`

### 2.1 Add metadata tables in `storage/storage/metadata_bootstrap.py`

In `METADATA_SCHEMA_DDL_REST`, you already define:

* `metadata.datasets`
* `metadata.dataset_rows` macro
* other macros & helpers

Extend `METADATA_SCHEMA_DDL_REST` with two tables (and indexes) for the dataflow graph.

Find the tuple:

```python
METADATA_SCHEMA_DDL_REST: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS metadata.datasets (
        table_key        TEXT PRIMARY KEY,
        name             TEXT NOT NULL,
        is_view          BOOLEAN NOT NULL,
        jsonl_filename   TEXT,
        parquet_filename TEXT,
        family           TEXT,
        description      TEXT
    );
    """,
    ...
)
```

Append new DDL strings:

```python
METADATA_SCHEMA_DDL_REST: tuple[str, ...] = (
    ...
    """
    CREATE TABLE IF NOT EXISTS metadata.dataset_dataflow_nodes (
        id            TEXT PRIMARY KEY,
        kind          TEXT NOT NULL,
        family        TEXT,
        owner_package TEXT,
        description   TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS metadata.dataset_dataflow_edges (
        src       TEXT NOT NULL,
        dst       TEXT NOT NULL,
        edge_type TEXT NOT NULL,
        PRIMARY KEY (src, dst, edge_type)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_dataset_dataflow_edges_src
        ON metadata.dataset_dataflow_edges (src);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_dataset_dataflow_edges_dst
        ON metadata.dataset_dataflow_edges (dst);
    """,
    ...
)
```

No behavior changes yet; just adds tables.

> The existing `apply_metadata_ddl(con)` will pick these up automatically.

---

### 2.2 Sync function: `sync_dataset_dataflow_graph`

Still in `metadata_bootstrap.py`, import the builder we just added:

```python
from codeintel.config.dataset_contract import (
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
    build_contract_dataflow_graph,
)
```

Then, near `bootstrap_metadata_datasets`, add:

```python
def sync_dataset_dataflow_graph(con: DuckDBPyConnection) -> None:
    """
    Refresh metadata.dataset_dataflow_nodes and metadata.dataset_dataflow_edges
    based on the current dataset contract.

    Safe to run repeatedly; performs a full replace via DELETE + INSERT.
    """
    nodes, edges = build_contract_dataflow_graph()

    con.execute("DELETE FROM metadata.dataset_dataflow_nodes")
    con.execute("DELETE FROM metadata.dataset_dataflow_edges")

    if nodes:
        con.executemany(
            """
            INSERT INTO metadata.dataset_dataflow_nodes (
                id, kind, family, owner_package, description
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (node.id, node.kind, node.family, node.owner_package, node.description)
                for node in nodes
            ],
        )

    if edges:
        con.executemany(
            """
            INSERT INTO metadata.dataset_dataflow_edges (
                src, dst, edge_type
            )
            VALUES (?, ?, ?)
            """,
            [(edge.src, edge.dst, edge.edge_type) for edge in edges],
        )
```

---

### 2.3 Call sync from `bootstrap_metadata_datasets`

In `bootstrap_metadata_datasets`, after you already call:

```python
    _assert_macro_coverage()
    apply_metadata_ddl(con)
    _register_macros(con)
    _register_dataset_schema_hashes(con)
    validate_failures = validate_metadata_datasets(con)
    ...
    _upsert_dataset_row(...)
```

append a call to `sync_dataset_dataflow_graph(con)` at the end of the function:

```python
    # Existing dataset registry population & validation
    ...
    gateway_datasets = con.execute("SELECT COUNT(*) FROM metadata.datasets").fetchone()[0]
    if gateway_datasets == 0:
        raise RuntimeError("metadata.datasets is unexpectedly empty after bootstrap")

    # New: materialize the contract-level dataflow graph into metadata tables.
    sync_dataset_dataflow_graph(con)
```

(Adjust the exact placement so it happens **after** `apply_metadata_ddl` and after you’ve ensured metadata schema exists; but it **doesn’t** depend on actual table contents, just the Python contracts.)

---

### 2.4 Repository: `DataflowRepository`

Create a new file **`storage/storage/repositories/dataflow.py`**:

```python
"""Repository for dataset-level dataflow graph in the metadata schema."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.storage.repositories.base import BaseRepository, RowDict, fetch_all_dicts


@dataclass(frozen=True)
class DataflowRepository(BaseRepository):
    """Read-only access to metadata.dataset_dataflow_nodes and edges."""

    def list_nodes(self) -> list[RowDict]:
        """
        Return all dataflow nodes.

        Returns
        -------
        list[RowDict]
            Rows with keys: id, kind, family, owner_package, description.
        """
        sql = """
        SELECT id, kind, family, owner_package, description
        FROM metadata.dataset_dataflow_nodes
        ORDER BY id
        """
        return fetch_all_dicts(self.con, sql, [])

    def list_edges(self, *, src: str | None = None, dst: str | None = None) -> list[RowDict]:
        """
        Return dataflow edges, optionally filtered by src/dst.

        Parameters
        ----------
        src, dst
            Optional filters for edge endpoints.

        Returns
        -------
        list[RowDict]
            Rows with keys: src, dst, edge_type.
        """
        sql = """
        SELECT src, dst, edge_type
        FROM metadata.dataset_dataflow_edges
        """
        params: list[object] = []
        predicates: list[str] = []

        if src is not None:
            predicates.append("src = ?")
            params.append(src)
        if dst is not None:
            predicates.append("dst = ?")
            params.append(dst)

        if predicates:
            sql += " WHERE " + " AND ".join(predicates)

        sql += " ORDER BY src, dst, edge_type"
        return fetch_all_dicts(self.con, sql, params)
```

Then register it in **`storage/storage/repositories/__init__.py`**:

```python
from codeintel.storage.repositories.dataflow import DataflowRepository
...
__all__ = [
    ...
    "DataflowRepository",
    ...
]
```

Now you have a clean entry point for any graph-aware tooling (including serving) to fetch dataset-level graph from DuckDB.

---

## 3. `serving.registry`: operation + graph nodes/edges + combined graph

### 3.1 Extend `OperationSpec` to support `exposed_datasets` (optional but nice)

In **`serving/serving/registry.py`**, after `required_graphs`, add a new field with a default:

```python
@dataclass(frozen=True)
class OperationSpec:
    """Cross-transport description of a single serving operation."""

    id: str
    category: str
    summary: str
    description: str | None
    http_method: Literal["GET", "POST"] | None
    http_path: str | None
    tool_name: str | None
    output_model_name: str
    backend_method: str
    required_datasets: Sequence[str]
    required_graphs: Sequence[str]
    default_limit: int | None
    max_limit: int | None
    # New: for dataflow graph
    exposed_datasets: Sequence[str] = ()
```

Because it has a default and is at the end, all existing OperationSpec instantiations remain valid without changes.

Later, you can go back and set `exposed_datasets` for specific operations, e.g.:

```python
"datasets.rows": OperationSpec(
    id="datasets.rows",
    ...
    required_datasets=(),
    required_graphs=(),
    default_limit=BackendLimits.DEFAULT_DATASET_LIMIT,
    max_limit=BackendLimits.MAX_DATASET_LIMIT,
    exposed_datasets=("*",),  # or leave empty and rely on query params
),
```

For now, leaving them default is fine; we’ll still get “reads” edges from `required_datasets`.

---

### 3.2 Import dataflow primitives and contracts

At the top of `registry.py`, import the dataflow types + contracts:

```python
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from codeintel.config.dataset_contract import (
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
    DataflowEdge,
    DataflowNode,
)
from codeintel.serving.backend import BackendLimits
...
```

We’ll add graph builders below.

---

### 3.3 Helper: resolve dataset identifier → canonical table_key

Add a small helper:

```python
def _resolve_dataset_identifier(identifier: str) -> str | None:
    """
    Resolve a dataset identifier used in OperationSpec into a canonical table_key.

    OperationSpec.required_datasets may refer to either:
    - DatasetContract.name (e.g. "call_graph_nodes"), or
    - DatasetContract.table_key (e.g. "docs.v_subsystem_summary").

    This helper normalizes both to DatasetContract.table_key, which is used
    as the DataflowNode.id for datasets/views.
    """
    contract = DATASET_CONTRACTS.get(identifier)
    if contract is not None:
        return contract.table_key

    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(identifier)
    if contract is not None:
        return contract.table_key

    return None
```

---

### 3.4 Operation + graph nodes

After `iter_operation_specs` / `get_operation_spec`, add:

```python
from itertools import chain


def iter_operation_nodes() -> list[DataflowNode]:
    """
    Return DataflowNode entries for all serving operations.

    Node IDs match OperationSpec.id, and are tagged as kind="operation".
    """
    nodes: list[DataflowNode] = []
    for spec in _OPERATION_SPECS.values():
        nodes.append(
            DataflowNode(
                id=spec.id,
                kind="operation",
                family="serving",
                owner_package=None,
                description=spec.summary,
            )
        )
    return nodes


def iter_graph_nodes() -> list[DataflowNode]:
    """
    Return DataflowNode entries for logical graph runtimes referred to by operations.

    IDs use the "graph.<name>" convention, e.g. "graph.callgraph".
    """
    names: set[str] = set()
    for spec in _OPERATION_SPECS.values():
        for graph_name in spec.required_graphs:
            names.add(graph_name)

    nodes: list[DataflowNode] = []
    for graph_name in sorted(names):
        nodes.append(
            DataflowNode(
                id=f"graph.{graph_name}",
                kind="graph",
                family="graph",
                owner_package="graphs",
                description=f"Logical {graph_name} graph runtime (call/import/etc.)",
            )
        )
    return nodes
```

---

### 3.5 Operation edges: dataset + graph

Add edge builders:

```python
def iter_operation_dataset_edges() -> list[DataflowEdge]:
    """
    Build edges from datasets to operations based on required_datasets/exposed_datasets.

    - required_datasets -> "reads" edges
    - exposed_datasets  -> "exposes" edges
    """
    edges: list[DataflowEdge] = []

    for spec in _OPERATION_SPECS.values():
        # required_datasets: dataset -> operation (reads)
        for ds_identifier in spec.required_datasets:
            table_key = _resolve_dataset_identifier(ds_identifier)
            if table_key is None:
                continue
            edges.append(
                DataflowEdge(
                    src=table_key,
                    dst=spec.id,
                    edge_type="reads",
                )
            )

        # exposed_datasets: dataset -> operation (exposes)
        for ds_identifier in spec.exposed_datasets:
            table_key = _resolve_dataset_identifier(ds_identifier)
            if table_key is None:
                continue
            edges.append(
                DataflowEdge(
                    src=table_key,
                    dst=spec.id,
                    edge_type="exposes",
                )
            )

    return edges


def iter_operation_graph_edges() -> list[DataflowEdge]:
    """
    Build edges from logical graph runtimes to operations (depends_on).
    """
    edges: list[DataflowEdge] = []
    for spec in _OPERATION_SPECS.values():
        for graph_name in spec.required_graphs:
            graph_id = f"graph.{graph_name}"
            edges.append(
                DataflowEdge(
                    src=graph_id,
                    dst=spec.id,
                    edge_type="depends_on",
                )
            )
    return edges
```

---

### 3.6 Combined serving-level graph: `build_serving_dataflow_graph`

Finally, add a combined builder that merges:

* dataset/docs graph from `build_contract_dataflow_graph`, and
* operation + graph nodes/edges.

At the top of `registry.py`, also import the contract-level graph builder:

```python
from codeintel.config.dataset_contract import (
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
    DataflowEdge,
    DataflowNode,
    build_contract_dataflow_graph,
)
```

Then, near the other public helpers:

```python
def build_serving_dataflow_graph() -> tuple[list[DataflowNode], list[DataflowEdge]]:
    """
    Build a combined dataflow graph for datasets/docs/views, operations, and graphs.

    Returns
    -------
    nodes
        Dataflow nodes for:
        - all DatasetContracts (tables + docs views)
        - all OperationSpecs
        - all graph runtimes referred to by OperationSpecs
    edges
        Dataflow edges for:
        - dataset -> dataset/docs ("builds")
        - dataset -> operation ("reads", "exposes")
        - graph   -> operation ("depends_on")
    """
    ds_nodes, ds_edges = build_contract_dataflow_graph()
    op_nodes = iter_operation_nodes()
    graph_nodes = iter_graph_nodes()

    op_ds_edges = iter_operation_dataset_edges()
    op_graph_edges = iter_operation_graph_edges()

    # Deduplicate nodes by (id, kind)
    node_map: dict[tuple[str, NodeKind], DataflowNode] = {}
    for node in chain(ds_nodes, op_nodes, graph_nodes):
        node_map[(node.id, node.kind)] = node

    nodes = list(node_map.values())

    # Deduplicate edges by (src, dst, edge_type)
    seen_edges: set[tuple[str, str, str]] = set()
    edges: list[DataflowEdge] = []
    for edge in chain(ds_edges, op_ds_edges, op_graph_edges):
        key = (edge.src, edge.dst, edge.edge_type)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append(edge)

    return nodes, edges
```

Update `__all__` at the bottom to export the new helpers:

```python
__all__ = [
    "DatasetMeta",
    "OperationSpec",
    "build_dataset_meta",
    "get_operation_spec",
    "iter_operation_specs",
    "iter_operation_nodes",
    "iter_graph_nodes",
    "build_serving_dataflow_graph",
]
```

---

## 4. `serving.http.routes.meta`: add `/meta/dataflow`

### 4.1 Pydantic models for the response

In **`serving/serving/mcp/models.py`**, add simple payload models.

Somewhere near `DatasetMetaResponse` / `OperationMetaResponse`, add:

```python
class DataflowNodePayload(BaseModel):
    """HTTP/MCP payload representing a single dataflow node."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    id: str
    kind: Literal["table", "view", "operation", "graph"]
    family: str | None = None
    owner_package: str | None = None
    description: str | None = None


class DataflowEdgePayload(BaseModel):
    """HTTP/MCP payload representing a dataflow edge."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    src: str
    dst: str
    edge_type: Literal["builds", "reads", "exposes", "depends_on"]


class DataflowGraphResponse(BaseModel):
    """Bundle of dataflow nodes and edges."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    nodes: list[DataflowNodePayload]
    edges: list[DataflowEdgePayload]
```

Also export them via `__all__` if you maintain one, or just rely on direct imports.

---

### 4.2 HTTP route

In **`serving/serving/http/routes/meta.py`**, extend imports:

```python
from codeintel.serving.mcp.models import (
    DatasetMetaResponse,
    OperationMetaResponse,
    DataflowGraphResponse,
    DataflowNodePayload,
    DataflowEdgePayload,
)
from codeintel.serving.registry import build_dataset_meta, build_serving_dataflow_graph, iter_operation_specs
```

Then inside `build_meta_router()`, add:

```python
    @router.get(
        f"{LOG_ROUTE_PREFIX}/dataflow",
        response_model=DataflowGraphResponse,
        summary="Return a dataflow graph for datasets, docs views, operations, and graphs.",
    )
    def get_dataflow_graph() -> DataflowGraphResponse:
        """
        Return the combined dataflow graph for this CodeIntel deployment.

        This uses the static dataset contract and OperationSpec registry, and
        does not depend on the active DuckDB state.
        """
        nodes, edges = build_serving_dataflow_graph()

        node_payloads = [
            DataflowNodePayload(
                id=node.id,
                kind=node.kind,
                family=node.family,
                owner_package=node.owner_package,
                description=node.description,
            )
            for node in nodes
        ]
        edge_payloads = [
            DataflowEdgePayload(
                src=edge.src,
                dst=edge.dst,
                edge_type=edge.edge_type,
            )
            for edge in edges
        ]

        return DataflowGraphResponse(nodes=node_payloads, edges=edge_payloads)
```

This gives you a **fully introspectable dataflow graph** via HTTP: `GET /meta/dataflow`.

---

## 5. `serving.mcp.meta_tools`: MCP tools for dataflow introspection

### 5.1 Extend imports

In **`serving/serving/mcp/meta_tools.py`**, augment imports:

```python
from typing import cast

from mcp.server.fastmcp import FastMCP

from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.models import (
    DatasetMetaResponse,
    OperationMetaResponse,
    ProblemDetail,
    DataflowGraphResponse,
    DataflowNodePayload,
    DataflowEdgePayload,
)
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import build_dataset_meta, build_serving_dataflow_graph, iter_operation_specs
from codeintel.serving.services.query_service import QueryService
```

We’ll reuse `build_serving_dataflow_graph()` here.

---

### 5.2 Precompute graph inside `register_meta_tools`

You already have something like:

```python
def register_meta_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    limits = _get_limits(backend)
    service = _get_service(backend)
    ...
```

Right after `limits` / `service` are computed, add:

```python
    # Precompute static dataflow graph for this backend.
    # It’s cheap and deterministic, so per-process caching is fine.
    df_nodes, df_edges = build_serving_dataflow_graph()
    df_node_by_id = {node.id: node for node in df_nodes}

    incoming: dict[str, list[DataflowEdge]] = {}
    outgoing: dict[str, list[DataflowEdge]] = {}
    for edge in df_edges:
        outgoing.setdefault(edge.src, []).append(edge)
        incoming.setdefault(edge.dst, []).append(edge)
```

(If you prefer, you can build these structures lazily in each tool instead.)

---

### 5.3 Tool: `explain_dataset`

Still inside `register_meta_tools`, add:

```python
    @mcp.tool()
    @_wrap  # reuse your existing wrapper for ProblemDetail handling
    def explain_dataset(node_id: str) -> list[dict]:
        """
        Explain a dataset/docs view node in the dataflow graph.

        node_id should typically be a table_key like "analytics.function_profile"
        or "docs.v_function_summary".
        """
        node = df_node_by_id.get(node_id)
        if node is None or node.kind not in ("table", "view"):
            detail = f"Unknown dataset/docs node_id: {node_id}"
            raise ProblemDetail.from_domain(
                ProblemDetail(
                    title="UnknownDataset",
                    detail=detail,
                    status=404,
                )
            )

        node_payload = DataflowNodePayload(
            id=node.id,
            kind=node.kind,
            family=node.family,
            owner_package=node.owner_package,
            description=node.description,
        )

        incoming_edges = [
            DataflowEdgePayload(src=e.src, dst=e.dst, edge_type=e.edge_type)
            for e in incoming.get(node.id, [])
        ]
        outgoing_edges = [
            DataflowEdgePayload(src=e.src, dst=e.dst, edge_type=e.edge_type)
            for e in outgoing.get(node.id, [])
        ]

        response = {
            "node": node_payload.model_dump(),
            "incoming_edges": [e.model_dump() for e in incoming_edges],
            "outgoing_edges": [e.model_dump() for e in outgoing_edges],
        }
        return [response]
```

This lets MCP clients ask “what builds `analytics.function_profile`?” or “what reads `docs.v_subsystem_summary`?”.

---

### 5.4 Tool: `explain_operation`

Add a similar tool for operations:

```python
    @mcp.tool()
    @_wrap
    def explain_operation(operation_id: str) -> list[dict]:
        """
        Explain an OperationSpec node in the dataflow graph.

        operation_id must match OperationSpec.id, e.g. "profiles.function".
        """
        node = df_node_by_id.get(operation_id)
        if node is None or node.kind != "operation":
            detail = f"Unknown operation id: {operation_id}"
            raise ProblemDetail.from_domain(
                ProblemDetail(
                    title="UnknownOperation",
                    detail=detail,
                    status=404,
                )
            )

        node_payload = DataflowNodePayload(
            id=node.id,
            kind=node.kind,
            family=node.family,
            owner_package=node.owner_package,
            description=node.description,
        )

        incoming_edges = [
            DataflowEdgePayload(src=e.src, dst=e.dst, edge_type=e.edge_type)
            for e in incoming.get(node.id, [])
        ]
        outgoing_edges = [
            DataflowEdgePayload(src=e.src, dst=e.dst, edge_type=e.edge_type)
            for e in outgoing.get(node.id, [])
        ]

        response = {
            "node": node_payload.model_dump(),
            "incoming_edges": [e.model_dump() for e in incoming_edges],
            "outgoing_edges": [e.model_dump() for e in outgoing_edges],
        }
        return [response]
```

This is your “explain what this operation depends on and what it exposes” surface.

---

### 5.5 Tool: `explain_path` (shortest path)

Finally, a small graph traversal helper to show the chain between two nodes.

Add:

```python
    @mcp.tool()
    @_wrap
    def explain_path(src_id: str, dst_id: str, max_hops: int = 6) -> list[dict]:
        """
        Return a shortest path between two dataflow nodes, if one exists.

        Parameters
        ----------
        src_id
            Source node id, e.g. "analytics.function_metrics".
        dst_id
            Destination node id, e.g. "profiles.function".
        max_hops
            Optional maximum path length to search (defaults to 6).
        """
        from collections import deque

        if src_id not in df_node_by_id:
            raise ProblemDetail.from_domain(
                ProblemDetail(
                    title="UnknownNode",
                    detail=f"Unknown src_id: {src_id}",
                    status=404,
                )
            )
        if dst_id not in df_node_by_id:
            raise ProblemDetail.from_domain(
                ProblemDetail(
                    title="UnknownNode",
                    detail=f"Unknown dst_id: {dst_id}",
                    status=404,
                )
            )

        # BFS on unweighted directed graph
        queue: deque[str] = deque([src_id])
        parent: dict[str, DataflowEdge | None] = {src_id: None}
        found = False

        while queue and not found:
            current = queue.popleft()
            depth = 0
            # reconstruct depth by walking parents (cheap with small graphs)
            cursor = current
            while parent[cursor] is not None:
                depth += 1
                cursor = parent[cursor].src  # type: ignore[assignment]
            if depth >= max_hops:
                continue

            for edge in outgoing.get(current, []):
                if edge.dst in parent:
                    continue
                parent[edge.dst] = edge
                if edge.dst == dst_id:
                    found = True
                    break
                queue.append(edge.dst)

        if not found:
            return [
                {
                    "path": [],
                    "message": f"No path from {src_id} to {dst_id} within {max_hops} hops.",
                }
            ]

        # Reconstruct path
        edges_in_path: list[DataflowEdge] = []
        node_id = dst_id
        while parent[node_id] is not None:
            edge = parent[node_id]
            assert edge is not None
            edges_in_path.append(edge)
            node_id = edge.src
        edges_in_path.reverse()

        nodes_in_path = [df_node_by_id[src_id]]
        for edge in edges_in_path:
            nodes_in_path.append(df_node_by_id[edge.dst])

        node_payloads = [
            DataflowNodePayload(
                id=node.id,
                kind=node.kind,
                family=node.family,
                owner_package=node.owner_package,
                description=node.description,
            ).model_dump()
            for node in nodes_in_path
        ]
        edge_payloads = [
            DataflowEdgePayload(
                src=edge.src,
                dst=edge.dst,
                edge_type=edge.edge_type,
            ).model_dump()
            for edge in edges_in_path
        ]

        return [
            {
                "nodes": node_payloads,
                "edges": edge_payloads,
            }
        ]
```

Finally, confirm `__all__` still exports `register_meta_tools` only; no change needed for tools themselves.

---

## 6. Tests

### 6.1 `tests/storage/test_dataflow_graph.py`: contract consistency

Create a new file **`tests/tests/storage/test_dataflow_graph.py`**:

```python
"""Tests for the contract-driven dataset/dataflow graph."""

from __future__ import annotations

import pytest

from codeintel.config.dataset_contract import (
    COMPOSITE_SCHEMAS,
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
    build_contract_dataflow_graph,
)
from codeintel.storage.gateway import open_memory_gateway
from codeintel.storage.metadata_bootstrap import bootstrap_metadata_datasets


def _require(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def test_contract_dataflow_includes_all_datasets() -> None:
    """Every DatasetContract must appear as a DataflowNode."""
    nodes, _ = build_contract_dataflow_graph()
    node_ids = {node.id for node in nodes}

    for contract in DATASET_CONTRACTS.values():
        _require(
            contract.table_key in node_ids,
            f"DatasetContract {contract.name} missing node for {contract.table_key}",
        )


def test_composite_edges_align_with_composite_schemas() -> None:
    """COMPOSITE_SCHEMAS must be fully represented in the dataflow graph."""
    _, edges = build_contract_dataflow_graph()
    builds_edges = {(e.src, e.dst) for e in edges if e.edge_type == "builds"}

    for table_key, composite in COMPOSITE_SCHEMAS.items():
        target = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
        if target is None:
            pytest.fail(f"CompositeSchema target {table_key} missing DatasetContract")
        dst_id = table_key
        for src_table_key in composite.composed_of:
            _require(
                (src_table_key, dst_id) in builds_edges,
                f"Missing builds edge {src_table_key} -> {dst_id} in composite graph",
            )


def test_metadata_dataflow_tables_populated() -> None:
    """
    bootstrap_metadata_datasets must populate metadata.dataset_dataflow_* tables.
    """
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=False)
    try:
        bootstrap_metadata_datasets(gateway.con, include_views=True)
        node_count = gateway.con.execute(
            "SELECT COUNT(*) FROM metadata.dataset_dataflow_nodes"
        ).fetchone()[0]
        edge_count = gateway.con.execute(
            "SELECT COUNT(*) FROM metadata.dataset_dataflow_edges"
        ).fetchone()[0]

        _require(node_count > 0, "Expected at least one dataflow node")
        _require(edge_count > 0, "Expected at least one dataflow edge")
    finally:
        gateway.close()
```

---

### 6.2 `tests/serving/test_operation_spec_alignment.py`: ensure dataset ids are resolvable

Extend **`tests/tests/serving/test_operation_spec_alignment.py`** to verify that every `required_datasets` entry maps to a contract:

```python
from codeintel.config.dataset_contract import DATASET_CONTRACTS, DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.serving.registry import iter_operation_specs


def test_required_datasets_resolve_to_dataset_contracts() -> None:
    """Every OperationSpec.required_datasets entry must map to a DatasetContract."""
    dataset_names = set(DATASET_CONTRACTS.keys())
    table_keys = set(DATASET_CONTRACTS_BY_TABLE_KEY.keys())

    for spec in iter_operation_specs():
        for ds_id in spec.required_datasets:
            if ds_id in dataset_names or ds_id in table_keys:
                continue
            pytest.fail(
                f"OperationSpec {spec.id} refers to unknown dataset identifier: {ds_id}"
            )
```

Optionally, add a small test to assert the presence of “reads” edges, but that’s already indirectly covered by this and the dataflow tests.

---

### 6.3 `tests/server/test_meta_dataflow.py`: HTTP endpoint

Add a new test file **`tests/tests/server/test_meta_dataflow.py`**:

```python
"""HTTP tests for the /meta/dataflow endpoint."""

from __future__ import annotations

from http import HTTPStatus

from fastapi.testclient import TestClient

from codeintel.serving.http.fastapi import create_app
from tests._helpers.gateway import build_backend_resource


def test_meta_dataflow_endpoint_smoke(backend_resource: build_backend_resource) -> None:
    """
    /meta/dataflow should return a non-empty graph.
    """
    app = create_app(backend_resource)
    client = TestClient(app)

    resp = client.get("/meta/dataflow")
    assert resp.status_code == HTTPStatus.OK

    payload = resp.json()
    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or []

    assert len(nodes) > 0, "Expected /meta/dataflow to return nodes"
    assert len(edges) > 0, "Expected /meta/dataflow to return edges"
```

(Adjust fixture names to match your existing FastAPI tests—e.g. reusing the `backend` / `backend_resource` fixture from `test_fastapi.py`.)

---

### 6.4 `tests/mcp/test_meta_dataflow_tools.py`: MCP tools

Add a basic smoke test for the new MCP tools:

```python
"""MCP smoke tests for dataflow introspection tools."""

from __future__ import annotations

from typing import Any, cast

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.meta_tools import register_meta_tools
from codeintel.storage.gateway import open_memory_gateway
from codeintel.storage.metadata_bootstrap import bootstrap_metadata_datasets


def test_explain_dataset_tool_smoke() -> None:
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=False)
    try:
        bootstrap_metadata_datasets(gateway.con, include_views=True)
        backend = DuckDBBackend(gateway=gateway, repo="r", commit="c")

        mcp = FastMCP("test")
        register_meta_tools(mcp, cast("Any", backend))

        tools = cast("list[Any]", getattr(mcp, "tools", []))
        tool_by_name = {cast("str", getattr(t, "name", "")): t for t in tools}

        assert "explain_dataset" in tool_by_name

        # Just assert the tool callable can be invoked; behavior is tested elsewhere.
        explain = tool_by_name["explain_dataset"]
        # If your FastMCP wrapper uses a different interface, adapt this call.
        result = explain({"node_id": "analytics.function_metrics"})  # type: ignore[call-arg]
        assert isinstance(result, (dict, list))
    finally:
        gateway.close()
```

(You’ll probably tweak the exact way you invoke FastMCP tools based on how they’re wrapped in your existing tests, but this gives the shape.)

---

If you’d like, next step I can zoom in on **one slice** of this epic and turn it into a pure “ordered patch plan” (per-file before/after diffs) — for example:

* just the **config.dataset_contract** changes, or
* just the **metadata_bootstrap + DataflowRepository** wiring.

But as-is, this plan should be enough to drive a very straightforward implementation of a **first-class, contract-driven dataflow graph** across `config`, `storage`, and `serving`.


# zoom in on **metadata_bootstrap + DataflowRepository** wiring. #

Perfect, let’s zoom in on just the **storage wiring** slice:
`storage/metadata_bootstrap.py` + `storage/repositories/dataflow.py` (+ a tiny `__init__` + test hook).

I’ll give you an **ordered patch plan** with concrete before/after diffs you can hand straight to an implementation agent.

---

## Preconditions

This patch plan assumes you’ve already (or will) add in `config/dataset_contract.py`:

* `DataflowNode`, `DataflowEdge`
* `build_contract_dataflow_graph() -> tuple[list[DataflowNode], list[DataflowEdge]]`

If that’s not in yet, just treat references to `build_contract_dataflow_graph` as “will be added in the config epic”.

---

## File 1 — `storage/metadata_bootstrap.py`

### 1.1 Import the dataflow graph builder

**Goal:** Let metadata bootstrap call your pure-config graph builder.

**Before** (around top of file):

```python
from duckdb import DuckDBPyConnection

from codeintel.config.dataset_contract import DATASET_CONTRACTS, DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.storage.normalized_macros import render_macro
from codeintel.storage.sql_helpers import safe_macro_call
from codeintel.storage.views import create_all_views
```

**After:**

```python
from duckdb import DuckDBPyConnection

from codeintel.config.dataset_contract import (
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
    build_contract_dataflow_graph,
)
from codeintel.storage.normalized_macros import render_macro
from codeintel.storage.sql_helpers import safe_macro_call
from codeintel.storage.views import create_all_views
```

---

### 1.2 Add metadata tables for the dataflow graph

**Goal:** Create two new metadata tables + indexes:

* `metadata.dataset_dataflow_nodes`
* `metadata.dataset_dataflow_edges`

These should be part of the same DDL batch as the other metadata tables.

Find where `METADATA_SCHEMA_DDL_REST` is extended:

```python
METADATA_SCHEMA_DDL_REST: tuple[str, ...] = (
    ...
    """
    CREATE OR REPLACE MACRO metadata.normalized_behavioral_coverage(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.test_goid_h128 AS BIGINT) AS test_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
)
 
METADATA_SCHEMA_DDL_REST += tuple(AUTO_NORMALIZED_MACRO_DDLS)

METADATA_SCHEMA_DDL: tuple[str, ...] = (
    METADATA_SCHEMA_DDL_BASE + INGEST_MACRO_DDLS + METADATA_SCHEMA_DDL_REST
)
```

**Patch it** to append your new table DDLs right after the line that adds `AUTO_NORMALIZED_MACRO_DDLS`:

```diff
-METADATA_SCHEMA_DDL_REST += tuple(AUTO_NORMALIZED_MACRO_DDLS)
+METADATA_SCHEMA_DDL_REST += tuple(AUTO_NORMALIZED_MACRO_DDLS)
+
+METADATA_SCHEMA_DDL_REST += (
+    """
+    CREATE TABLE IF NOT EXISTS metadata.dataset_dataflow_nodes (
+        id            TEXT PRIMARY KEY,
+        kind          TEXT NOT NULL,
+        family        TEXT,
+        owner_package TEXT,
+        description   TEXT
+    );
+    """,
+    """
+    CREATE TABLE IF NOT EXISTS metadata.dataset_dataflow_edges (
+        src       TEXT NOT NULL,
+        dst       TEXT NOT NULL,
+        edge_type TEXT NOT NULL,
+        PRIMARY KEY (src, dst, edge_type)
+    );
+    """,
+    """
+    CREATE INDEX IF NOT EXISTS idx_dataset_dataflow_edges_src
+        ON metadata.dataset_dataflow_edges (src);
+    """,
+    """
+    CREATE INDEX IF NOT EXISTS idx_dataset_dataflow_edges_dst
+        ON metadata.dataset_dataflow_edges (dst);
+    """,
+)
```

`apply_metadata_ddl` will now create these tables along with the rest of the metadata schema.

---

### 1.3 Add `sync_dataset_dataflow_graph(...)`

**Goal:** A small helper that:

* calls `build_contract_dataflow_graph()` from config
* **replaces** the contents of `metadata.dataset_dataflow_nodes` / `edges`.

Find the existing `_upsert_dataset_row` helper and the start of `bootstrap_metadata_datasets`:

```python
def _upsert_dataset_row(con: DuckDBPyConnection, payload: _DatasetUpsert) -> None:
    ...
    con.execute(
        """
        INSERT INTO metadata.datasets (
            ...
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(table_key) DO UPDATE SET
            ...
        """,
        [...],
    )


def bootstrap_metadata_datasets(
    con: DuckDBPyConnection,
    *,
    jsonl_filenames: Mapping[str, str] | None = None,
    parquet_filenames: Mapping[str, str] | None = None,
    include_views: bool = True,
) -> None:
```

**Insert this new function between them:**

```diff
 def _upsert_dataset_row(con: DuckDBPyConnection, payload: _DatasetUpsert) -> None:
     ...
     con.execute(
         ...
     )
 
 
+def sync_dataset_dataflow_graph(con: DuckDBPyConnection) -> None:
+    """
+    Refresh dataset-level dataflow graph metadata tables based on static contracts.
+
+    This uses build_contract_dataflow_graph() from config.dataset_contract and
+    fully replaces the contents of metadata.dataset_dataflow_nodes and
+    metadata.dataset_dataflow_edges.
+    """
+    nodes, edges = build_contract_dataflow_graph()
+
+    # Clear existing graph state. This is safe because all information is
+    # derived from static Python contracts.
+    con.execute("DELETE FROM metadata.dataset_dataflow_nodes")
+    con.execute("DELETE FROM metadata.dataset_dataflow_edges")
+
+    if nodes:
+        con.executemany(
+            """
+            INSERT INTO metadata.dataset_dataflow_nodes (
+                id,
+                kind,
+                family,
+                owner_package,
+                description
+            )
+            VALUES (?, ?, ?, ?, ?)
+            """,
+            [
+                (node.id, node.kind, node.family, node.owner_package, node.description)
+                for node in nodes
+            ],
+        )
+
+    if edges:
+        con.executemany(
+            """
+            INSERT INTO metadata.dataset_dataflow_edges (
+                src,
+                dst,
+                edge_type
+            )
+            VALUES (?, ?, ?)
+            """,
+            [(edge.src, edge.dst, edge.edge_type) for edge in edges],
+        )
+
+
 def bootstrap_metadata_datasets(
     con: DuckDBPyConnection,
     *,
     jsonl_filenames: Mapping[str, str] | None = None,
```

---

### 1.4 Call `sync_dataset_dataflow_graph` from `bootstrap_metadata_datasets`

**Goal:** Whenever you bootstrap metadata, you also refresh the dataflow graph tables.

Find the tail of `bootstrap_metadata_datasets`:

```python
    jsonl_mapping = dict(jsonl_filenames or {})
    parquet_mapping = dict(parquet_filenames or {})

    for name, contract in sorted(DATASET_CONTRACTS.items(), key=lambda item: item[1].table_key):
        if contract.is_view and not include_views:
            continue

        table_key = contract.table_key
        schema_prefix, _ = table_key.split(".", maxsplit=1)
        jsonl_filename = jsonl_mapping.get(table_key) or contract.jsonl_filename
        parquet_filename = parquet_mapping.get(table_key) or contract.parquet_filename

        _upsert_dataset_row(
            con,
            _DatasetUpsert(
                table_key=table_key,
                name=name,
                is_view=contract.is_view,
                jsonl_filename=jsonl_filename,
                parquet_filename=parquet_filename,
                family=contract.family or schema_prefix,
                description=contract.description,
            ),
        )
```

**Append this at the very end of the function:**

```diff
         _upsert_dataset_row(
             con,
             _DatasetUpsert(
                 table_key=table_key,
                 name=name,
                 is_view=contract.is_view,
                 jsonl_filename=jsonl_filename,
                 parquet_filename=parquet_filename,
                 family=contract.family or schema_prefix,
                 description=contract.description,
             ),
         )
+
+    # Materialize the contract-driven dataflow graph into metadata tables.
+    sync_dataset_dataflow_graph(con)
```

Now any code path that calls `bootstrap_metadata_datasets` will end up with **fresh** dataflow nodes/edges.

---

## File 2 — `storage/repositories/dataflow.py` (new)

**Goal:** Provide a typed, repository-style interface for reading the graph from DuckDB.

Create a new file: **`storage/repositories/dataflow.py`** with the following contents:

```python
"""Repository for dataset-level dataflow metadata."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.storage.repositories.base import BaseRepository, RowDict, fetch_all_dicts


@dataclass(frozen=True)
class DataflowRepository(BaseRepository):
    """
    Read-only access to metadata.dataset_dataflow_nodes and
    metadata.dataset_dataflow_edges.

    All methods follow the standard repository patterns used in the rest
    of the storage layer.
    """

    def list_nodes(self) -> list[RowDict]:
        """
        Return all dataflow nodes.

        Returns
        -------
        list[RowDict]
            Each row has keys: id, kind, family, owner_package, description.
        """
        sql = """
        SELECT id, kind, family, owner_package, description
        FROM metadata.dataset_dataflow_nodes
        ORDER BY id
        """
        return fetch_all_dicts(self.con, sql, [])

    def list_edges(self, *, src: str | None = None, dst: str | None = None) -> list[RowDict]:
        """
        Return dataflow edges, optionally filtered by src/dst.

        Parameters
        ----------
        src
            Optional source-node filter.
        dst
            Optional destination-node filter.

        Returns
        -------
        list[RowDict]
            Each row has keys: src, dst, edge_type.
        """
        sql = """
        SELECT src, dst, edge_type
        FROM metadata.dataset_dataflow_edges
        """
        params: list[object] = []
        predicates: list[str] = []

        if src is not None:
            predicates.append("src = ?")
            params.append(src)
        if dst is not None:
            predicates.append("dst = ?")
            params.append(dst)

        if predicates:
            sql += " WHERE " + " AND ".join(predicates)

        sql += " ORDER BY src, dst, edge_type"
        return fetch_all_dicts(self.con, sql, params)
```

Usage stays consistent with other repos:

```python
repo = DataflowRepository(gateway, repo="my/repo", commit="deadbeef")
nodes = repo.list_nodes()
edges = repo.list_edges(src="analytics.function_profile")
```

---

## File 3 — `storage/repositories/__init__.py`

**Goal:** Re-export `DataflowRepository` alongside your other storage repositories.

Current file:

```python
"""Repository layer for DuckDB persistence."""

from codeintel.storage.repositories.base import (
    BaseRepository,
    PaginatedRows,
    RowDict,
    fetch_all_dicts,
    fetch_one_dict,
    fetch_paginated,
    row_exists,
)
from codeintel.storage.repositories.data_models import DataModelRepository
from codeintel.storage.repositories.datasets import DatasetReadRepository
from codeintel.storage.repositories.functions import FunctionRepository
from codeintel.storage.repositories.graphs import GraphRepository
from codeintel.storage.repositories.modules import ModuleRepository
from codeintel.storage.repositories.subsystems import SubsystemRepository
from codeintel.storage.repositories.tests import TestRepository

__all__ = [
    "BaseRepository",
    "DataModelRepository",
    "DatasetReadRepository",
    "FunctionRepository",
    "GraphRepository",
    "ModuleRepository",
    "PaginatedRows",
    "RowDict",
    "SubsystemRepository",
    "TestRepository",
    "fetch_all_dicts",
    "fetch_one_dict",
    "fetch_paginated",
    "row_exists",
]
```

**Patch imports & `__all__`:**

```diff
 from codeintel.storage.repositories.base import (
     BaseRepository,
     PaginatedRows,
     RowDict,
     fetch_all_dicts,
     fetch_one_dict,
     fetch_paginated,
     row_exists,
 )
 from codeintel.storage.repositories.data_models import DataModelRepository
+from codeintel.storage.repositories.dataflow import DataflowRepository
 from codeintel.storage.repositories.datasets import DatasetReadRepository
 from codeintel.storage.repositories.functions import FunctionRepository
 from codeintel.storage.repositories.graphs import GraphRepository
 from codeintel.storage.repositories.modules import ModuleRepository
 from codeintel.storage.repositories.subsystems import SubsystemRepository
 from codeintel.storage.repositories.tests import TestRepository
 
 __all__ = [
     "BaseRepository",
     "DataModelRepository",
+    "DataflowRepository",
     "DatasetReadRepository",
     "FunctionRepository",
     "GraphRepository",
     "ModuleRepository",
     "PaginatedRows",
     "RowDict",
```

Now anything that already imports `codeintel.storage.repositories` can pull in `DataflowRepository` directly.

---

## File 4 — (Optional but recommended) `tests/storage/test_metadata_bootstrap.py`

**Goal:** Add a **storage-level** smoke test that proves:

* `bootstrap_metadata_datasets` creates & populates the dataflow tables.
* `DataflowRepository` can read from them.

In `tests/storage/test_metadata_bootstrap.py`, the imports currently look like:

```python
import pytest

from codeintel.storage.datasets import load_dataset_registry
from codeintel.storage.gateway import open_memory_gateway
```

**Extend them:**

```diff
 import pytest
 
 from codeintel.storage.datasets import load_dataset_registry
 from codeintel.storage.gateway import open_memory_gateway
+from codeintel.storage.metadata_bootstrap import bootstrap_metadata_datasets
+from codeintel.storage.repositories import DataflowRepository
```

At the bottom of the file you currently have a test that ends with:

```python
    view_dataset = registry.by_name.get("v_function_summary")
    ...
    _require(view_dataset.family == "docs", f"Unexpected docs family: {view_dataset.family}")

    gateway.close()
```

**Append a new test after that:**

```python
def test_dataflow_metadata_populated() -> None:
    """
    bootstrap_metadata_datasets should populate dataset_dataflow_* tables and
    DataflowRepository should be able to read from them.
    """
    gateway = open_memory_gateway(
        apply_schema=True,
        ensure_views=True,
        validate_schema=False,
        repo="test/repo",
        commit="deadbeef",
    )
    try:
        bootstrap_metadata_datasets(gateway.con, include_views=True)
        repo = DataflowRepository(gateway, "test/repo", "deadbeef")

        nodes = repo.list_nodes()
        edges = repo.list_edges()

        _require(nodes, "Expected at least one dataflow node")
        _require(edges, "Expected at least one dataflow edge")
    finally:
        gateway.close()
```

That gives you a simple, stable assertion that your storage wiring is correct.

---


