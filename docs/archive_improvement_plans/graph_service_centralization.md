Here’s a concrete implementation plan for turning `graph_service` into *the* canonical hub for all NetworkX analytics in CodeIntel, with example code tied to your actual files.

I’ll treat the current code as the “baseline” (since you already implemented a lot of this) and then spell out:

1. What GraphService owns today
2. What we want it to own (target responsibilities)
3. The changes to make in `graph_service.py`
4. The way each analytics module should use it
5. How to test and guide future AI agents

---

## 1. Baseline: what `graph_service` already does

You already have an excellent skeleton in `src/codeintel/analytics/graph_service.py` :

* **GraphContext**: common parameters: repo/commit, sampling caps, eigen iteration, seeds, pagerank/betweenness weights, with `resolved_now()` helper for timestamps .

* **build_graph_context**: builds a `GraphContext` from `GraphMetricsConfig` with caps for betweenness and eigen iterations .

* **GraphBundle**: named lazy loaders for graphs (call/import/symbol/etc.), memoized in `_cache` .

* **Metrics dataclasses** :

  * `NeighborStats`
  * `CentralityBundle`
  * `ComponentBundle`
  * `ProjectionMetrics` (degree/weighted_degree/clustering/betweenness)
  * `StructuralMetrics` (clustering/triangles/core_number/constraint/effective_size/community_id)

* **Utilities**: id normalization, float coercion, logging helpers, betweenness sampling .

* **Core algorithms**:

  * `neighbor_stats` for fan-in/out and weighted degree 
  * `centrality_directed` (PR + BC + closeness + harmonic + optional eigen) 
  * `centrality_undirected` (PR + BC + closeness + harmonic + eigen + optional structuralholes) 
  * `component_metadata` (weak comps + SCCs + condensation layers) 
  * `component_ids_undirected` (comp ids/sizes) 
  * `_dag_layers` for condensation layer depth on a DAG 
  * `projection_metrics` for bipartite projection metrics (degree, weighted_degree, clustering, betweenness) 
  * `community_ids` using `community.asyn_lpa_communities` 
  * `structural_metrics` for undirected graphs (core, clustering, triangles, constraint, effective size, communities) 

And these are already used in:

* `analytics.graph_metrics` (call/import graph metrics) 
* `analytics.graph_metrics_ext` (function call-graph extended metrics) 
* `analytics.module_graph_metrics_ext` (module import-graph extended metrics) 
* `analytics.symbol_graph_metrics` (symbol module/function graphs) 
* `analytics.test_graph_metrics` (test–function bipartite projections ⟶ uses `projection_metrics`) 
* `analytics.config_graph_metrics` (uses `centrality_undirected`, `community_ids`, `log_empty_graph`, `log_projection_skipped`) 
* `analytics.cfg_dfg_metrics` (uses `GraphContext` + `centrality_directed` for CFG/DFG) 
* `analytics.subsystem_graph_metrics` (uses `GraphContext` + `centrality_directed`) 

So GraphService is already the main “primitive layer.” The remaining work is to:

* Add **global graph stats** primitives (for what `graph_stats.py` does now).
* Normalize **projection logic** for bipartite graphs so test/config/graph_stats share the same helpers.
* Tighten **usage patterns** in the analytics modules so all heavy NX calls go through GraphService.

---

## 2. Target responsibilities for `GraphService`

Conceptually, we want `graph_service` to own **all reusable NetworkX algorithms**; analytics modules:

> “Pick the graph(s), call GraphService, turn its dictionaries into rows.”

Concretely, GraphService should provide:

1. **Context & bundling**

   * `GraphContext` and `build_graph_context` for sampling/weights.
   * `GraphBundle` for memoized graph loading.

2. **Node-level metrics**

   * Directed: `centrality_directed`, `neighbor_stats`, `component_metadata`.
   * Undirected: `centrality_undirected`, `structural_metrics`, `component_ids_undirected`.

3. **Bipartite projections**

   * `projection_metrics` for “project a partition and compute degree/weighted_degree/clustering/betweenness”.

4. **Global graph statistics**

   * A new `GlobalGraphStats` dataclass + `global_graph_stats` function to compute what `graph_stats.py` is doing today: node/edge counts, weak/SCC counts, condensation layer count, avg clustering, diameter, avg SPL.

5. **Community detection helpers**

   * `community_ids` (already there) reused by symbol/config/test analytics and possibly global stats.

Everything else (config-specific shapes, row-building, risk-weighting, etc.) stays in `analytics/*.py`.

---

## 3. Extending `graph_service.py`

### 3.1. Add `GlobalGraphStats` and `global_graph_stats`

Right now `graph_stats.py` implements `_diameter_and_spl` and `_component_layers` locally  . We want to move that logic into GraphService and expose a neat API.

**Step 1 – import approximation and exceptions**

At the top of `graph_service.py` you already import NX exceptions and algorithms; add approximation:

```python
# existing imports
import networkx as nx
from networkx.algorithms import bipartite, community, structuralholes
from networkx.exception import NetworkXAlgorithmError, PowerIterationFailedConvergence
# add:
from networkx.algorithms import approximation
```

**Step 2 – add a dataclass**

Append near the other dataclasses:

```python
@dataclass(frozen=True)
class GlobalGraphStats:
    """Whole-graph summary statistics shared across analytics modules."""

    node_count: int
    edge_count: int
    weak_component_count: int
    scc_count: int
    component_layers: int | None
    avg_clustering: float
    diameter_estimate: float | None
    avg_shortest_path_estimate: float | None
```

**Step 3 – move `_diameter_and_spl` into GraphService**

Take the logic from `analytics.graph_stats._diameter_and_spl`  and adapt:

```python
def _diameter_and_spl(graph: nx.Graph | nx.DiGraph) -> tuple[float | None, float | None]:
    """
    Estimate diameter and average shortest path length on the largest connected component.

    Returns
    -------
    tuple[float | None, float | None]
        (diameter_estimate, avg_shortest_path_estimate).
    """
    if graph.number_of_nodes() == 0:
        return None, None
    undirected = graph.to_undirected()
    components = list(nx.connected_components(undirected))
    if not components:
        return None, None
    largest = undirected.subgraph(max(components, key=len)).copy()
    try:
        diameter = float(approximation.diameter(largest))
    except (NetworkXAlgorithmError, nx.NetworkXError):  # be generous
        diameter = None
    try:
        avg_spl = float(nx.average_shortest_path_length(largest))
    except (NetworkXAlgorithmError, nx.NetworkXError):
        avg_spl = None
    return diameter, avg_spl
```

Note: `graph_stats.py` currently catches `NetworkXError`; we include both.

**Step 4 – move `_component_layers` into GraphService**

Right now it’s in `graph_stats.py` and uses condensation to count DAG layers :

```python
def _component_layers(graph: nx.Graph | nx.DiGraph) -> int | None:
    """
    Return the number of condensation layers for directed graphs.

    Parameters
    ----------
    graph : nx.Graph | nx.DiGraph
        Graph to analyze.

    Returns
    -------
    int | None
        Layer count when the graph is directed; otherwise None.
    """
    if graph.number_of_nodes() == 0 or not isinstance(graph, nx.DiGraph):
        return None
    condensation = nx.condensation(graph)
    if condensation.number_of_nodes() == 0:
        return 0
    layers: dict[int, int] = {
        node: 0 for node in condensation.nodes if condensation.in_degree(node) == 0
    }
    for node in nx.topological_sort(condensation):
        base = layers.get(node, 0)
        for succ in condensation.successors(node):
            layers[succ] = max(layers.get(succ, 0), base + 1)
    return max(layers.values(), default=0) + 1
```

You already have `_dag_layers` for DAGs of SCCs; `_component_layers` is the higher-level “how many layers does condensation have” measure.

**Step 5 – add `global_graph_stats`**

Now define the main helper:

```python
def global_graph_stats(graph: nx.Graph | nx.DiGraph) -> GlobalGraphStats:
    """
    Compute whole-graph summary statistics for use in analytics.graph_stats.

    Parameters
    ----------
    graph : nx.Graph | nx.DiGraph
        Graph to summarize.

    Returns
    -------
    GlobalGraphStats
        Node/edge counts, component statistics, clustering and distance metrics.
    """
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    if isinstance(graph, nx.DiGraph):
        weak_count = nx.number_weakly_connected_components(graph)
        scc_count = sum(1 for _ in nx.strongly_connected_components(graph))
    else:
        weak_count = nx.number_connected_components(graph)
        scc_count = weak_count

    layers = _component_layers(graph)
    diameter, avg_spl = _diameter_and_spl(graph)
    avg_clustering = (
        float(nx.average_clustering(graph.to_undirected())) if node_count > 0 else 0.0
    )

    return GlobalGraphStats(
        node_count=node_count,
        edge_count=edge_count,
        weak_component_count=weak_count,
        scc_count=scc_count,
        component_layers=layers,
        avg_clustering=avg_clustering,
        diameter_estimate=diameter,
        avg_shortest_path_estimate=avg_spl,
    )
```

After this, `analytics.graph_stats` can shrink to: “load graphs, call `global_graph_stats`, build rows.”

---

### 3.2. (Optional) Extend `ProjectionMetrics` to carry closeness and communities

You already have:

* `ProjectionMetrics` in GraphService = degree, weighted_degree, clustering, betweenness .
* `projection_metrics` which uses bipartite projection + betweenness + clustering .
* `test_graph_metrics` uses this directly for tests/functions projections .

`config_graph_metrics` defines its *own* `ProjectionMetrics` dataclass with degree, weighted-degree, betweenness, **closeness**, and a **community id map**, and its own `_projection_metrics` that calls `centrality_undirected` and `community_ids`  .

If you want *one* canonical API for projections, you can:

1. Expand GraphService’s `ProjectionMetrics`:

   ```python
   @dataclass(frozen=True)
   class ProjectionMetrics:
       """Centrality bundle for projected bipartite graphs."""

       degree: dict[Any, int]
       weighted_degree: dict[Any, float]
       clustering: dict[Any, float]
       betweenness: dict[Any, float]
       closeness: dict[Any, float]
       community_id: dict[Any, int]
   ```

2. Update `projection_metrics` to fill them:

   ```python
   def projection_metrics(
       bipartite_graph: nx.Graph,
       nodes: Iterable[Any],
       ctx: GraphContext,
       *,
       weight: str | None = None,
   ) -> ProjectionMetrics:
       ...
       proj = bipartite.weighted_projected_graph(bipartite_graph, nodes_set)
       degree_view = proj.degree(weight=None)
       weighted_view = proj.degree(weight=weight)
       degree = {node: int(deg) for node, deg in degree_view}
       weighted_degree = {node: float(deg) for node, deg in weighted_view}

       clustering_val = nx.clustering(proj, weight=weight) if proj.number_of_nodes() > 0 else {}
       clustering = clustering_val if isinstance(clustering_val, dict) else {}
       betweenness = (
           nx.betweenness_centrality(
               proj,
               weight=weight,
               k=_betweenness_sample(proj, ctx),
               seed=ctx.seed,
           )
           if proj.number_of_nodes() > 0
           else {}
       )
       closeness = (
           {node: float(val) for node, val in nx.closeness_centrality(proj).items()}
           if proj.number_of_nodes() > 0
           else {}
       )
       communities = community_ids(proj, weight=weight)

       return ProjectionMetrics(
           degree=degree,
           weighted_degree=weighted_degree,
           clustering=clustering,
           betweenness=betweenness,
           closeness=closeness,
           community_id=communities,
       )
   ```

3. Adjust use sites:

   * `test_graph_metrics` keeps using `degree`, `weighted_degree`, `clustering`, `betweenness` and just ignores the new fields.
   * `config_graph_metrics` drops its local `ProjectionMetrics` dataclass and `_projection_metrics` and instead uses GraphService’s `ProjectionMetrics` plus `projection_metrics(proj, ...)`.

This way, *all* projection metrics (`test_graph_metrics`, `config_graph_metrics`, future symbol/config/test stuff) reuse one primitive.

If you’d rather not change the existing dataclass shape, you can leave GraphService as-is and just have `config_graph_metrics` call:

* `proj = nx.bipartite.weighted_projected_graph(...)`
* `centrality_undirected(proj, ctx)`
* `community_ids(proj)`

…which it already does via its local `_projection_metrics` . That’s functionally fine; the “algorithms” are still centralized in GraphService.

---

## 4. Refactoring the analytics modules to rely on GraphService

### 4.1. `analytics/graph_stats.py`

This is the clearest “global stats” consumer.

**Today** it has `_diameter_and_spl`, `_component_layers`, `_safe_project`, and a manual loop computing stats for each graph  .

After the GraphService extension above, you can:

1. Remove `_diameter_and_spl` and `_component_layers` from `graph_stats.py`.

2. Import `global_graph_stats`:

   ```python
   from codeintel.analytics.graph_service import GraphContext, global_graph_stats
   ```

3. In `compute_graph_stats`, replace this block :

   ```python
   for name, graph in graphs.items():
       weak_count = (
           nx.number_weakly_connected_components(graph)
           if isinstance(graph, nx.DiGraph)
           else nx.number_connected_components(graph)
       )
       scc_count = (
           sum(1 for _ in nx.strongly_connected_components(graph))
           if isinstance(graph, nx.DiGraph)
           else weak_count
       )
       diameter, avg_spl = _diameter_and_spl(graph)
       layers = _component_layers(graph)
       rows.append(
           (
               name,
               repo,
               commit,
               graph.number_of_nodes(),
               graph.number_of_edges(),
               weak_count,
               scc_count,
               layers,
               nx.average_clustering(graph.to_undirected())
               if graph.number_of_nodes() > 0
               else 0.0,
               diameter,
               avg_spl,
               now,
           )
       )
   ```

   with:

   ```python
   for name, graph in graphs.items():
       stats = global_graph_stats(graph)
       rows.append(
           (
               name,
               repo,
               commit,
               stats.node_count,
               stats.edge_count,
               stats.weak_component_count,
               stats.scc_count,
               stats.component_layers,
               stats.avg_clustering,
               stats.diameter_estimate,
               stats.avg_shortest_path_estimate,
               now,
           )
       )
   ```

`_safe_project` can either stay here (as it’s very graph_stats-specific) or be moved into GraphService as a second, more defensive projection helper. It’s already fairly specialized to “config projections” though .

---

### 4.2. `analytics/test_graph_metrics.py`

You’ve already wired this nicely:

* It uses `GraphContext` and `projection_metrics` from GraphService .
* It still imports `bipartite` locally only for `degree_centrality` in `_bipartite_degrees` .

**Optional improvements**:

* If you extend `ProjectionMetrics` to include closeness and community IDs, you can use those in the future if you decide to store richer test-level metrics.
* If you want to remove *all* direct uses of `networkx.algorithms.bipartite` from analytics modules, you could add a `bipartite_degrees` helper to GraphService, but what you have now is already pretty clean.

---

### 4.3. `analytics/config_graph_metrics.py`

This module already uses GraphService for centrality & logging, but it re-defines its own `ProjectionMetrics` and `_projection_metrics`  .

**Option A – Keep local ProjectionMetrics, but use GraphService primitives**

This is already the case:

* `_projection_metrics` calls `centrality_undirected(proj, ctx)` and `community_ids(proj, weight=ctx.pagerank_weight)`; only the dataclass is local.

This is acceptable—and consistent with the “GraphService owns algorithms” principle.

**Option B – Fully reuse GraphService’s ProjectionMetrics**

If you extend GraphService’s `ProjectionMetrics` (degree, weighted_degree, clustering, betweenness, closeness, community_id), you can:

1. Delete the local `@dataclass ProjectionMetrics` in `config_graph_metrics.py` .
2. Replace `_projection_metrics` with a simple wrapper around GraphService:

   ```python
   from codeintel.analytics.graph_service import GraphContext, projection_metrics

   ...

   def _projection_metrics(proj: nx.Graph, ctx: GraphContext) -> ProjectionMetrics:
       # If proj was already built by _build_projection
       return projection_metrics(proj, proj.nodes(), ctx, weight=ctx.pagerank_weight)
   ```

   And then `_projection_rows` stays as-is, just reading `.degree`, `.weighted_degree`, `.betweenness`, `.clo` (which would become `.closeness`), and `.comm_map` (which would become `.community_id`) from the GraphService dataclass.

   That’s a slightly larger refactor but makes the GraphService API the single source of truth for projection metrics.

---

### 4.4. `analytics/graph_metrics_ext.py` & `analytics/module_graph_metrics_ext.py`

Both already use GraphService’s centralities and structural metrics:

* `graph_metrics_ext` uses `centrality_directed`, `component_metadata`, `structural_metrics`, `to_decimal_id` .
* `module_graph_metrics_ext` uses `centrality_directed`, `component_metadata`, `structural_metrics` .

No further refactor needed here; they’re already ideal consumers: pick a graph, call GraphService, build rows.

---

### 4.5. `analytics/symbol_graph_metrics.py`

This is also already a very clean client:

* It uses `GraphBundle` for caching, then `centrality_undirected`, `structural_metrics`, `component_ids_undirected`, `log_empty_graph`, and `to_decimal_id` .

Nothing to change; this module is the textbook example of “GraphService + nx_views + row-building only.”

---

### 4.6. `analytics/cfg_dfg_metrics.py`

CFG/DFG metrics are per-function and use some specialized algorithms:

* `_compute_cfg_centralities` uses `nx.immediate_dominators`, `nx.dominance_frontiers`, and `centrality_directed` with a local `GraphContext` .
* `_dfg_centralities` uses `centrality_directed` and returns betweenness + eigenvector maps .
* A bunch of other helpers measure path counts, branching, loop sizes, etc.

That’s all appropriate: CFG/DFG graphs are small, fairly unique, and you already call GraphService for centralities. No need to force the rest into GraphService.

If you want to be strict, you could:

* Add `cfg_centralities` and `dfg_centralities` as named helpers to GraphService, but they’re so tailored to CFG/DFG that keeping them in `cfg_dfg_metrics.py` is fine.

---

### 4.7. `analytics/subsystem_graph_metrics.py`

This file already uses `GraphContext` + `centrality_directed` for subsystem-level import graph metrics .

The only local “algorithm” is `_dag_layers` over subsystems, but GraphService has `_dag_layers` for SCC condensation already .

If you want to reuse that:

* `_layer_by_subsystem` currently computes layers via `nx.condensation(subsystem_graph)` and its own DAG traversal over SCCs .
* That’s essentially the same pattern as `_dag_layers`; you can leave it local, since it’s specific to “subsystem graph” rather than a generic graph.

---

## 5. Usage pattern for future AI agents (the “contract”)

It’s worth codifying a short set of rules for agents in comments / docs:

1. **Never call NetworkX algorithms directly in analytics modules**
   If you need:

   * call/import graph centrality ⟶ `centrality_directed` + `neighbor_stats` + `component_metadata`.
   * undirected graph metrics ⟶ `centrality_undirected` + `structural_metrics` + `component_ids_undirected`.
   * bipartite projection ⟶ `projection_metrics`.
   * global stats ⟶ `global_graph_stats`.
   * components or condensation layers ⟶ `component_metadata` / `component_ids_undirected` / `_component_layers`.

2. **Always get graphs from `nx_views` or AnalyticsContext**

   * `load_call_graph`, `load_import_graph`, `load_symbol_*`, `load_test_function_bipartite`, `load_config_module_bipartite` 
   * Reuse `AnalyticsContext.call_graph/import_graph` when available, exactly like `graph_metrics` does via `GraphBundle` .

3. **Use `GraphContext` everywhere**

   * Either call `build_graph_context(GraphMetricsConfig)` or construct a `GraphContext` manually with reasonable defaults (betweenness sample caps, eigen iteration caps), as you do in `graph_metrics_ext`, `module_graph_metrics_ext`, `symbol_graph_metrics`, and `test_graph_metrics`    .

4. **Keep analytics modules as “selection + row builders”**

   * All heavy NX work goes into GraphService.
   * Analytics modules:

     * choose which graph(s) to load,
     * possibly perform simple filtering / partitioning (e.g. tests vs funcs, subsystems vs modules),
     * call GraphService functions,
     * format the metrics into table rows and write via DuckDB.

---

## 6. Implementation order and testing

A sensible sequence for actually doing this in your repo:

1. **Implement `GlobalGraphStats` and `global_graph_stats` in `graph_service.py`** (plus `_diameter_and_spl`, `_component_layers` helpers).
2. **Refactor `analytics.graph_stats`** to use `global_graph_stats` and delete duplicated code.
3. **(Optional) Expand `ProjectionMetrics`** and update `projection_metrics`, `test_graph_metrics`, and `config_graph_metrics` to use it.
4. **Add tests**:

   * Unit test GraphService functions (especially `global_graph_stats` and `projection_metrics`) on small fixtures: path graph, cycle, 2-component graph, bipartite toy graph, etc.
   * Regression tests for `analytics.graph_stats` to ensure row values don’t change unexpectedly.

Once this is in place, `GraphService` really is the “one graph analytics spine” you wanted: every heavy NetworkX operation the system cares about is funneled through a small, well-documented, strongly-typed set of functions.
