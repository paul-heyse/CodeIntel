# Rustworkx Best-In-Class Structural Improvements Plan

## Goals
- Harden the rustworkx store layer against unsafe mutations.
- Make edge weight aggregation configurable by graph kind.
- Persist and validate graph metadata through node-link JSON caches.
- Replace DFG depth traversal with rustworkx BFS visitor semantics.
- Reduce internal duplication and graph re-materialization costs.

## Non-Goals
- Change analytics table schemas or row shapes.
- Introduce new metrics beyond current parity.
- Add GPU backends or non-rustworkx engines.

## Scope Items

### 1) Guarded mutation API for `RxGraphStore`
**Intent**
Ensure all node/edge changes go through store methods so ID/index mappings,
node attributes, and versioning stay consistent.

**Plan**
- Introduce explicit mutation methods on `RxGraphStore` for add/remove/update.
- Track a `_version` counter and invalidate derived caches on mutation.
- Provide a read-only `graph_view` for algorithms and reserve `_graph` for internal use.
- Add invariant checks in debug paths and unit tests to verify mapping integrity.

**Code pattern**
```python
@dataclass(slots=True)
class RxGraphStore:
    _graph: RxGraph
    id_to_index: dict[Hashable, int]
    index_to_id: dict[int, Hashable]
    node_attrs: dict[Hashable, dict[str, object]]
    is_directed: bool
    _version: int = 0

    @property
    def graph_view(self) -> RxGraph:
        return self._graph

    def add_node(self, node_id: Hashable, attrs: Mapping[str, object] | None = None) -> int:
        index = self.ensure_node(node_id)
        if attrs:
            self.set_node_attrs(node_id, attrs)
        self._version += 1
        return index

    def remove_node(self, node_id: Hashable) -> bool:
        index = self.id_to_index.pop(node_id, None)
        if index is None:
            return False
        self._graph.remove_node(index)
        self.index_to_id.pop(index, None)
        self.node_attrs.pop(node_id, None)
        self._version += 1
        return True

    def remove_edge(self, src_id: Hashable, dst_id: Hashable) -> bool:
        src_idx = self.id_to_index.get(src_id)
        dst_idx = self.id_to_index.get(dst_id)
        if src_idx is None or dst_idx is None:
            return False
        if not self._graph.has_edge(src_idx, dst_idx):
            return False
        self._graph.remove_edge(src_idx, dst_idx)
        self._version += 1
        return True
```

**Target files**
- `src/codeintel/build/graphs/rx/store.py`
- `tests/graphs/test_rx_store.py`

**Status**
Completed.

**Implementation notes**
- Added guarded mutation methods (`add_node`, `remove_node`, `remove_edge`, `set_edge_weight`) plus
  versioned cache invalidation in `src/codeintel/build/graphs/rx/store.py`.
- Exposed `graph_view` accessor and `_view_cache`; underlying field remains `graph` (no `_graph`
  rename) to avoid a broad refactor.
- Updated fixtures and tests to use the guarded setters in
  `tests/_helpers/fixtures/graphs.py` and `tests/graphs/test_rx_store.py`.

### 2) Per-graph weight aggregation policy
**Intent**
Make edge aggregation behavior explicit per `GraphKind` (sum, max, replace),
while keeping NaN handling consistent with existing normalization rules.

**Plan**
- Introduce a `GraphWeightPolicy` in a dedicated module.
- Add a `weight_policy` attribute to `RxGraphStore`.
- Resolve policies per graph kind during graph construction.
- Replace ad-hoc edge weight increments with policy-driven aggregation.

**Code pattern**
```python
@dataclass(frozen=True, slots=True)
class GraphWeightPolicy:
    default_weight: float
    combine: Callable[[float, float], float]
    nan_policy: NanPolicy = "keep"

    def coerce(self, payload: object | None) -> float:
        return edge_weight_from_payload(payload, nan_policy=self.nan_policy)


GRAPH_KIND_POLICIES: dict[GraphKind, GraphWeightPolicy] = {
    GraphKind.CALL_GRAPH: GraphWeightPolicy(default_weight=1.0, combine=operator.add),
    GraphKind.IMPORT_GRAPH: GraphWeightPolicy(default_weight=1.0, combine=operator.add),
    GraphKind.SYMBOL_MODULE_GRAPH: GraphWeightPolicy(default_weight=1.0, combine=operator.add),
    GraphKind.SYMBOL_FUNCTION_GRAPH: GraphWeightPolicy(default_weight=1.0, combine=operator.add),
    GraphKind.CONFIG_MODULE_BIPARTITE: GraphWeightPolicy(default_weight=1.0, combine=operator.add),
}


def add_weighted_edge(
    store: RxGraphStore,
    source: Hashable,
    target: Hashable,
) -> None:
    policy = store.weight_policy
    src_idx = store.ensure_node(source)
    dst_idx = store.ensure_node(target)
    if store.graph_view.has_edge(src_idx, dst_idx):
        current = policy.coerce(store.graph_view.get_edge_data(src_idx, dst_idx))
        updated = policy.combine(current, policy.default_weight)
        store.graph_view.update_edge(src_idx, dst_idx, updated)
        return
    store.graph_view.add_edge(src_idx, dst_idx, policy.default_weight)
```

**Target files**
- `src/codeintel/build/graphs/rx/policies.py` (new)
- `src/codeintel/build/graphs/rx/store.py`
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/runtime/runtime.py`
- `tests/graphs/test_rx_store.py`

**Status**
Completed.

**Implementation notes**
- Introduced `GraphWeightPolicy` and `GraphNumericPolicy` in
  `src/codeintel/build/graphs/rx/policies.py`, with per-kind policy resolution.
- `weight_policy_for_kind` resolves by GraphKind name tokens (string mapping) to avoid a
  module import cycle.
- Builders/views now construct stores with the per-kind policy; runtime validates cached
  graphs against the policy name.

### 3) Graph metadata in `graph.attrs` + cache version bump
**Intent**
Embed cache metadata into rustworkx graphs and reject incompatible cache payloads.

**Plan**
- Define a `GraphMetadata` shape and store it in `graph.attrs`.
- Bump `GRAPH_CACHE_VERSION` and validate metadata on load.
- Require metadata presence during serialization for cache writes.

**Code pattern**
```python
@dataclass(frozen=True, slots=True)
class GraphMetadata:
    cache_version: str
    engine: str
    graph_kind: str
    weight_policy: str
    node_payload_version: str

    def as_attrs(self) -> dict[str, object]:
        return {
            "cache_version": self.cache_version,
            "engine": self.engine,
            "graph_kind": self.graph_kind,
            "weight_policy": self.weight_policy,
            "node_payload_version": self.node_payload_version,
        }


def apply_graph_metadata(store: RxGraphStore, metadata: GraphMetadata) -> None:
    store.graph_view.attrs = metadata.as_attrs()
```

**Target files**
- `src/codeintel/build/graphs/rx/metadata.py` (new)
- `src/codeintel/build/graphs/rx/serialization.py`
- `src/codeintel/build/graphs/runtime/runtime.py` (bump `GRAPH_CACHE_VERSION`)
- `src/codeintel/build/graphs/engine/cache.py`
- `tests/graphs/test_rx_serialization.py`

**Status**
Completed (engine cache file unchanged; runtime cache path now enforces metadata).

**Implementation notes**
- Added `GraphMetadata` and `GraphAttrs` protocol in `src/codeintel/build/graphs/rx/metadata.py`.
- Runtime now writes metadata into `graph.attrs`, bumps `GRAPH_CACHE_VERSION` to `v4`,
  and validates on read in `src/codeintel/build/graphs/runtime/runtime.py`.
- Serialization now requires metadata for cache writes and is covered by
  `tests/graphs/test_rx_serialization.py`.

### 4) Unify rustworkx graph conversion
**Intent**
Avoid duplicated conversion logic and ensure node payload decoding is consistent.

**Plan**
- Remove `_store_from_rx` from algorithm helpers.
- Make `store_from_rx` the single conversion path.
- Update `ensure_store` and direct conversions to use the unified function.

**Code pattern**
```python
from codeintel.build.graphs.rx.convert import store_from_rx


def ensure_store(graph: GraphInput) -> RxGraphStore:
    if isinstance(graph, RxGraphStore):
        return graph
    if isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        return store_from_rx(graph)
    raise TypeError(f"Unsupported graph input: {type(graph).__name__}")
```

**Target files**
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/rx/convert.py`

**Status**
Completed.

**Implementation notes**
- Removed ad-hoc conversion in algorithms and routed all conversions through
  `store_from_rx` in `src/codeintel/build/graphs/rx/convert.py`.

### 5) BFS visitor for DFG path lengths
**Intent**
Use rustworkx BFS visitor traversal with pruning for max-depth bounds.

**Plan**
- Implement a BFS visitor that records distances and prunes at depth.
- Keep the existing max-depth semantics (allow depth `max_depth + 1`).
- Validate results against existing DFG tests.

**Code pattern**
```python
from rustworkx import visit


@dataclass
class DepthVisitor(visit.BFSVisitor):
    distances: dict[int, int]
    max_depth: int

    def discover_vertex(self, v: int) -> None:
        self.distances.setdefault(v, 0)

    def tree_edge(self, edge: tuple[int, int]) -> None:
        parent, child = edge
        dist = self.distances.get(parent, 0) + 1
        if dist > self.max_depth + 1:
            raise visit.PruneSearch()
        self.distances[child] = dist


def compute_dfg_path_lengths(graph: GraphInput, *, max_depth: int = 100) -> dict[Any, DFGPathStats]:
    store = ensure_directed_store(graph)
    directed = cast("rx.PyDiGraph", store.graph_view)
    result: dict[Any, DFGPathStats] = {}
    for node_id in store.node_ids():
        node_idx = store.id_to_index[node_id]
        distances: dict[int, int] = {node_idx: 0}
        visitor = DepthVisitor(distances=distances, max_depth=max_depth)
        rx.bfs_search(directed, [node_idx], visitor)
        bounded = [d for idx, d in distances.items() if idx != node_idx]
        # ... aggregate to DFGPathStats
```

**Target files**
- `src/codeintel/build/graphs/compute/metrics/dfg.py`
- `tests/graphs/test_compute_metrics_dfg.py`

**Status**
Completed.

**Implementation notes**
- Implemented BFS visitor traversal with depth pruning and preserved the
  `max_depth + 1` semantics in `src/codeintel/build/graphs/compute/metrics/dfg.py`.

### 6) Cached derived graph views
**Intent**
Avoid repeated `to_undirected()`/`to_directed()` conversions across metrics.

**Plan**
- Cache derived views keyed by store version.
- Invalidate cache on mutation.
- Replace direct conversions in metrics with `store.as_undirected()` and
  `store.as_directed()`.

**Code pattern**
```python
@dataclass(slots=True)
class RxGraphStore:
    _view_cache: dict[str, tuple[int, RxGraphStore]] = field(default_factory=dict)

    def as_undirected(self) -> RxGraphStore:
        cached = self._view_cache.get("undirected")
        if cached is not None and cached[0] == self._version:
            return cached[1]
        undirected = store_from_rx(self.graph_view.to_undirected())
        self._view_cache["undirected"] = (self._version, undirected)
        return undirected
```

**Target files**
- `src/codeintel/build/graphs/rx/store.py`
- `src/codeintel/build/graphs/compute/metrics/statistics.py`
- `src/codeintel/build/graphs/compute/metrics/components.py`
- `src/codeintel/build/graphs/compute/metrics/structural.py`

**Status**
Completed.

**Implementation notes**
- Added versioned derived-view caching in `src/codeintel/build/graphs/rx/store.py`.
- All helper conversions now route through `store.as_undirected()` / `store.as_directed()`
  via the rustworkx algorithm wrappers.

### 7) Policy consolidation for numeric normalization
**Intent**
Centralize NaN policy and tolerance handling so metrics can share configuration
and remain deterministic across algorithms.

**Plan**
- Introduce a `GraphNumericPolicy` with `nan_policy` and per-metric tolerances.
- Pass it through rustworkx wrappers and custom algorithms.
- Replace per-module constants with policy accessors.

**Code pattern**
```python
@dataclass(frozen=True, slots=True)
class GraphNumericPolicy:
    nan_policy: NanPolicy = "keep"
    harmonic_tol: tuple[float, float] = (1e-9, 1e-6)
    clustering_tol: tuple[float, float] = (1e-9, 1e-6)
```

**Target files**
- `src/codeintel/build/graphs/rx/normalize.py`
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/compute/metrics/*`

**Status**
Completed.

**Implementation notes**
- `GraphNumericPolicy` now drives NaN handling and tolerances in
  `src/codeintel/build/graphs/rx/algos.py`.
- Projection stores carry the numeric + weight policy from their source store for
  consistent normalization.

## Acceptance Criteria
- Node/edge mutations only occur through `RxGraphStore` methods. (Implemented.)
- Edge aggregation behavior is explicit per `GraphKind`. (Implemented.)
- Cache payloads include metadata and reject mismatches. (Implemented.)
- DFG path lengths match existing tests under BFS visitor traversal. (Implemented.)
- Derived graph views are cached and invalidated on mutation. (Implemented.)

## Validation Notes
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
  runs successfully for guardrails and types, but still reports unrelated Ruff/Pyrefly
  issues in Hamilton modules.
- Targeted pytest run is currently blocked by a missing SchemaService configuration.
