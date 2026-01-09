# Rustworkx Graph Outputs + Metadata (Best-in-Class Plan)

## Goals

- Surface the full range of rustworkx-friendly graph data and metadata for analytics.
- Make every output cluster optional via runtime toggles, with sensible defaults.
- Keep payloads and metadata versioned, schema-driven, and contract-aligned.

## Output toggle model (clustered)

### Toggle groups (recommended)

- `core_metadata` (always on): graph identity, determinism, ordering, weight policy, payload versions.
- `graph_stats`: counts, density, component counts, DAG/cycle flags.
- `node_payloads`: per-node metadata and metrics payload enrichment.
- `edge_payloads`: per-edge metadata and metrics payload enrichment.
- `algorithms_basic`: degree, components, layers, SCC/WCC.
- `algorithms_advanced`: centrality, paths, clustering/community, dominance/CFG/DFG extras.
- `serialization_exports`: node-link JSON, GraphML, DOT.
- `materialized_tables`: persist derived tables for analytics joins.

### Toggle surface (runtime settings)

```python
# config/primitives.py
@dataclass(frozen=True)
class GraphOutputToggles:
    core_metadata: bool = True
    graph_stats: bool = True
    node_payloads: bool = True
    edge_payloads: bool = True
    algorithms_basic: bool = True
    algorithms_advanced: bool = False
    serialization_exports: bool = False
    materialized_tables: bool = True
```

```python
# build/graphs/runtime/context.py
@dataclass(frozen=True)
class GraphMetricsOptions:
    output_toggles: GraphOutputToggles = GraphOutputToggles()
```

## Comprehensive schema (graph + node + edge)

### Graph-level metadata schema (graph.attrs)

```python
@dataclass(frozen=True, slots=True)
class GraphMetadata:
    # Identity + lineage
    graph_kind: str
    repo: str
    commit: str
    run_id: str | None
    build_timestamp: str | None
    dataset_root: str | None
    source_tables: tuple[str, ...]

    # Engine + versions
    engine: str
    cache_version: str
    node_payload_version: str
    edge_payload_version: str

    # Determinism + ordering
    determinism_tier: str
    ordering_keys: tuple[str, ...] | None
    tie_breaker_keys: tuple[str, ...] | None
    scan_profile: str | None
    runtime_profile: str | None

    # Semantics
    weight_policy: str
    weight_semantics: str
    is_directed: bool
    is_multigraph: bool

    # Summary stats (optional)
    node_count: int | None = None
    edge_count: int | None = None
    density: float | None = None
    component_count: int | None = None
    scc_count: int | None = None
    has_cycles: bool | None = None
```

### Node payload schema (node payloads)

```python
@dataclass(frozen=True, slots=True)
class GraphNodePayload:
    node_id: object
    node_kind: str | None
    label: str | None
    path: str | None
    span: tuple[int, int, int, int] | None  # start_line, start_col, end_line, end_col
    namespace: str | None
    module: str | None
    symbol_kind: str | None
    block_kind: str | None
    synthetic: bool | None
    metrics: dict[str, float | int | bool] | None
```

### Edge payload schema (edge payloads)

```python
@dataclass(frozen=True, slots=True)
class GraphEdgePayload:
    weight: float
    edge_kind: str | None
    count: int | None
    callsite: tuple[str, int, int] | None  # path, line, col
    symbol_ref: str | None
    config_key: str | None
    synthetic: bool | None
    metrics: dict[str, float | int | bool] | None
```

## Scope items

### 1) Graph output schema + versioning

**Intent**: Define graph/node/edge payload schemas and version tags; unify encoding/decoding.

**Code pattern**
```python
# build/graphs/rx/payloads.py
NODE_PAYLOAD_VERSION = "v2"
EDGE_PAYLOAD_VERSION = "v1"

def encode_edge_payload(payload: GraphEdgePayload) -> dict[str, object]:
    return dataclasses.asdict(payload)
```

**Targets**
- `src/codeintel/build/graphs/rx/metadata.py`
- `src/codeintel/build/graphs/rx/payloads.py`
- `src/codeintel/build/graphs/rx/store.py`

**Checklist**
- Add edge payload version constant and encoder/decoder.
- Extend `GraphMetadata` with schema/version fields.
- Ensure `apply_graph_metadata` writes versioned metadata.

### 2) Output toggles and runtime wiring

**Intent**: Add `GraphOutputToggles` and plumb it through runtime context.

**Code pattern**
```python
# config/primitives.py
class GraphFeatureFlags:
    graph_outputs: GraphOutputToggles | None = None

# build/graphs/runtime/context.py
def graph_metrics_options_from_features(features: GraphFeatureFlags) -> GraphMetricsOptions:
    outputs = features.graph_outputs or GraphOutputToggles()
    return GraphMetricsOptions(output_toggles=outputs)
```

**Targets**
- `src/codeintel/config/primitives.py`
- `src/codeintel/build/graphs/runtime/context.py`
- `src/codeintel/build/graphs/runtime/__init__.py`
- `src/codeintel/build/analytics/graphs/context_helpers.py`
- `src/codeintel/build/analytics/cfg_dfg/compute.py`
- `src/codeintel/build/hamilton/native/analytics/cfg_dfg_metrics.py`

**Checklist**
- Add env parsing for output toggles.
- Expose toggles on `GraphMetricsOptions`.
- Thread toggles into GraphContext in both Hamilton and pure paths.

### 3) Graph metadata enrichment at build time

**Intent**: Apply full `GraphMetadata` on graph construction.

**Code pattern**
```python
metadata = GraphMetadata(
    graph_kind="import_graph",
    repo=repo,
    commit=commit,
    source_tables=("graph.import_graph_edges",),
    weight_policy=store.weight_policy.name,
    determinism_tier=determinism,
    ordering_keys=ordering_keys,
)
apply_graph_metadata(store.graph, metadata)
```

**Targets**
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/graphs/compute/imports.py`
- `src/codeintel/build/graphs/runtime/runtime.py`

**Checklist**
- Add helper for metadata assembly (repo/commit/run_id/etc.).
- Populate determinism + ordering keys from schema service.
- Include summary stats if `graph_stats` toggle is enabled.

### 4) Node/edge payload enrichment

**Intent**: Normalize node and edge payloads with structured metadata and metrics.

**Code pattern**
```python
EdgeBuildSpec(
    directed=True,
    weight_policy=policy,
    numeric_policy=numeric,
    node_attrs_fn=lambda node_id, kind: {"node_kind": kind},
)
```

**Targets**
- `src/codeintel/build/graphs/rx/build_from_edges.py`
- `src/codeintel/build/graphs/rx/store.py`
- `src/codeintel/build/graphs/rx/payloads.py`
- Graph producers in `src/codeintel/build/graphs/compute/*`

**Checklist**
- Standardize node/edge payload shapes.
- Add optional metrics attachment when toggles are on.
- Ensure payload version tags are attached to metadata.

### 5) Derived analytics outputs gated by toggles

**Intent**: Compute advanced metrics only when enabled.

**Code pattern**
```python
if ctx.output_toggles.algorithms_advanced:
    centrality = compute_centrality(store)
else:
    centrality = {}
```

**Targets**
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/analytics/graphs/graph_metrics_ext.py`
- `src/codeintel/build/analytics/cfg_dfg/*`
- `src/codeintel/build/graphs/compute/metrics/*`

**Checklist**
- Gate heavy algorithms behind advanced toggle.
- Keep basic outputs available with `algorithms_basic`.
- Maintain deterministic ordering for all outputs.

### 6) Serialization exports

**Intent**: Optionally export node-link JSON / GraphML / DOT with metadata.

**Code pattern**
```python
if ctx.output_toggles.serialization_exports:
    payload = dumps_node_link_json(store.graph)
```

**Targets**
- `src/codeintel/build/graphs/rx/serialization.py`
- `src/codeintel/build/graphs/runtime/runtime.py`
- `src/codeintel/storage/*` (optional persistence)

**Checklist**
- Add export helpers for each format.
- Include metadata and payload versions in exports.
- Ensure ordering and determinism are preserved.

### 7) Materialized tables for analytics joins

**Intent**: Emit derived tables (components, paths, centrality) under toggles.

**Code pattern**
```python
if ctx.output_toggles.materialized_tables:
    table = table_for_rows("analytics.graph_centrality", rows)
```

**Targets**
- `src/codeintel/build/analytics/compute/row_builders/*`
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/core/schemas/output_registry.py`

**Checklist**
- Define schema entries for new outputs.
- Emit tables only when toggle is on.
- Add contract alignment in finalize gates.

## Suggested rollout order

1) Schema + toggles (Scope 1 + 2).
2) Graph metadata enrichment (Scope 3).
3) Node/edge payload enrichment (Scope 4).
4) Derived outputs + serialization (Scope 5 + 6 + 7).

