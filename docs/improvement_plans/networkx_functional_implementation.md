You’re already in a really good place: the repo is fully typed, Pydantic is the canonical model layer, Prefect is orchestrating the pipeline, and NetworkX is already used in `src/codeintel/analytics/graph_metrics.py` for centrality + SCC/condensation on call/import graphs.

Below are **concrete, file‑level places** where you can lean harder on NetworkX in ways that *add new capability / robustness*, not just refactors.

---

## 0. Overall pattern I’d aim for

Right now, each analytics module rebuilds its own little adjacency structures from DuckDB. The “best‑in‑class” move is to standardize a *single* way to materialize graphs in memory, and then reuse them across metrics, subsystems, and validations.

I’d introduce a small helper module:

> `src/codeintel/graphs/nx_views.py`

with things like:

```python
# src/codeintel/graphs/nx_views.py
from __future__ import annotations

import duckdb
import networkx as nx

def load_call_graph(con: duckdb.DuckDBPyConnection, repo: str, commit: str) -> nx.DiGraph:
    rows = con.execute("""
        SELECT caller_goid_h128, callee_goid_h128
        FROM graph.call_graph_edges
        WHERE callee_goid_h128 IS NOT NULL
          AND repo = ? AND commit = ?
    """, [repo, commit]).fetchall()

    g = nx.DiGraph()
    for caller, callee in rows:
        if caller is None or callee is None:
            continue
        g.add_edge(int(caller), int(callee), weight=1)

    return g


def load_import_graph(con: duckdb.DuckDBPyConnection, repo: str, commit: str) -> nx.DiGraph:
    rows = con.execute("""
        SELECT src_module, dst_module
        FROM graph.import_graph_edges
        WHERE repo = ? AND commit = ?
    """, [repo, commit]).fetchall()

    g = nx.DiGraph()
    for src, dst in rows:
        if src is None or dst is None:
            continue
        s = str(src)
        d = str(dst)
        g.add_edge(s, d, weight=g.get_edge_data(s, d, {}).get("weight", 0) + 1)
    return g
```

Then:

* `analytics.graph_metrics`, `analytics.subsystems`, and any future graph‑heavy analytics all use these functions instead of each building their own dicts.
* `graphs.validation` can also lean on these for richer checks (more on that later).

What follows are specific places to plug this in.

---

## 1. Subsystem inference: move clustering fully onto NetworkX

**File:** `src/codeintel/analytics/subsystems.py`
Key functions: `build_subsystems`, `_build_weighted_adjacency`, `_label_propagation`, `_reassign_small_clusters`, `_limit_clusters`, `_subsystem_edge_stats`.

Right now, this module:

1. Loads modules and tags (`_load_modules`).
2. Builds a **custom weighted adjacency dict** from imports, symbol uses, and config (`_build_weighted_adjacency`).
3. Runs a homegrown **label‑propagation clustering** on that adjacency (`_label_propagation`) with seeding from tags.
4. Post‑processes labels: `*_small_clusters`, `_limit_clusters`.
5. Computes per‑subsystem edge stats via manual iteration over `import_edges` (`_subsystem_edge_stats`).

You can keep the *semantics* but stand on NetworkX for the heavy lifting:

### 1.1 Build a proper weighted graph for subsystems

Today `_build_weighted_adjacency` returns `dict[str, dict[str, float]]`. I’d change it to *also* build a NetworkX graph:

```python
# subsystems.py

import networkx as nx

def _build_weighted_graph(
    con: duckdb.DuckDBPyConnection, cfg: SubsystemsConfig, modules: set[str]
) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(modules)

    # Imports
    rows = con.execute("""
        SELECT src_module, dst_module
        FROM graph.import_graph_edges
        WHERE repo = ? AND commit = ?
    """, [cfg.repo, cfg.commit]).fetchall()
    for src, dst in rows:
        if src is None or dst is None:
            continue
        a = str(src)
        b = str(dst)
        if a in modules and b in modules:
            w = g.get_edge_data(a, b, {}).get("weight", 0.0) + cfg.import_weight
            g.add_edge(a, b, weight=w)

    # Symbol uses / config coupling – same pattern as your current `_add_weight` logic
    # using symbol edges + config_values, just expressed via `g.add_edge`.

    return g
```

You can keep `_build_weighted_adjacency` as a thin adapter if you still want the dict structure for compatibility:

```python
def _build_weighted_adjacency(...):
    g = _build_weighted_graph(con, cfg, modules)
    adj: dict[str, dict[str, float]] = defaultdict(dict)
    for u, v, data in g.edges(data=True):
        adj[u][v] = data["weight"]
        adj[v][u] = data["weight"]
    return adj
```

But most of the later bits can now work directly on `g`.

### 1.2 Replace custom label propagation with a NetworkX‑backed variant

Your `_label_propagation` is a seeded, synchronous label propagation:

```python
def _label_propagation(
    modules: set[str],
    adjacency: dict[str, dict[str, float]],
    seed_labels: dict[str, str],
    max_iters: int = 20,
) -> dict[str, str]:
    labels = {module: seed_labels.get(module, module) for module in modules}
    frozen = set(seed_labels)
    ...
```

NetworkX has `asyn_lpa_communities(G, weight='weight', seed=...)`. It doesn’t support *frozen* seeds out of the box, but you can still:

* Use NetworkX’s graph to simplify neighbor/weight logic.
* Keep your “freeze seeded nodes” semantics.

For example:

```python
def _label_propagation_nx(
    g: nx.Graph,
    seed_labels: dict[str, str],
    max_iters: int = 20,
) -> dict[str, str]:
    labels: dict[str, str] = {n: seed_labels.get(n, n) for n in g.nodes}
    frozen: set[str] = set(seed_labels)

    for _ in range(max_iters):
        changed = False
        # deterministic order
        for node in sorted(g.nodes):
            if node in frozen:
                continue
            weights: dict[str, float] = defaultdict(float)
            for neighbor, data in g[node].items():
                neighbor_label = labels.get(neighbor)
                if neighbor_label is None:
                    continue
                w = data.get("weight", 1.0)
                weights[neighbor_label] += w
            if not weights:
                continue
            best_label = max(weights.items(), key=lambda item: (item[1], item[0]))[0]
            if labels[node] != best_label:
                labels[node] = best_label
                changed = True
        if not changed:
            break
    return labels
```

This gives you:

* **Less custom adjacency surgery** (all neighbor iteration comes from `g[node]`).
* Easy path to experiment with other community algorithms (e.g., `nx.algorithms.community.greedy_modularity_communities`) while sharing the same `g`.

You can slot this into `build_subsystems` as:

```python
modules, tags_by_module = _load_modules(con, cfg)
g = _build_weighted_graph(con, cfg, modules)

seed_labels = _seed_labels_from_tags(tags_by_module)
labels = _label_propagation_nx(g, seed_labels, max_iters=cfg.max_label_iters)
labels = _reassign_small_clusters(labels, _graph_to_adjacency(g), cfg.min_cluster_size)
labels = _limit_clusters(labels, _graph_to_adjacency(g), cfg.max_clusters)
clusters = _clusters_from_labels(labels)
```

Where `_graph_to_adjacency` is optional compatibility glue.

### 1.3 Use NetworkX for subsystem edge stats & entrypoints

`_subsystem_edge_stats` currently walks a `dict[(src, dst) -> count]` of import edges to compute `internal_edges`, `external_edges`, and `fan_in/out`. That’s perfect for:

```python
def _subsystem_edge_stats(
    members: list[str],
    labels: dict[str, str],
    import_graph: nx.DiGraph,
) -> SubsystemEdgeStats:
    member_set = set(members)
    label = labels[members[0]]

    internal_edges = 0
    external_edges = 0
    fan_in: set[str] = set()
    fan_out: set[str] = set()

    for u, v, data in import_graph.edges(data=True):
        w = int(data.get("weight", 1))
        lu = labels.get(u)
        lv = labels.get(v)
        if lu is None or lv is None:
            continue
        if u in member_set and v in member_set:
            internal_edges += w
        elif lu == label and lv != label:
            external_edges += w
            fan_out.add(lv)
        elif lv == label and lu != label:
            external_edges += w
            fan_in.add(lu)

    return SubsystemEdgeStats(
        internal_edges=internal_edges,
        external_edges=external_edges,
        fan_in=fan_in,
        fan_out=fan_out,
    )
```

You can get `import_graph` either from `graphs.nx_views.load_import_graph` or build it alongside the subsystem affinity graph.

Similarly, `_entrypoints_for_cluster` (currently heuristics over fan‑in/out + tags) can be simplified and strengthened:

* Use `import_graph.in_degree()` / `out_degree()` to compute external vs internal degrees.
* Use `nx.algorithms.centrality.betweenness_centrality` on the **condensed subsystem graph** to pick entrypoints that sit on cross‑subsystem bridges.

Net effect: subsystem logic is now entirely expressed in terms of **two NetworkX graphs**:

* A weighted undirected “affinity” `nx.Graph` for clustering.
* The directed import `nx.DiGraph` for entrypoints + edge stats.

---

## 2. Import graph cycles & layering: drop your custom Tarjan for NetworkX

**File:** `src/codeintel/graphs/import_graph.py`
Key functions: `_tarjan_scc`, `build_import_graph`.

You currently define `_tarjan_scc(graph: dict[str, set[str]]) -> dict[str, int]` and then (inside `build_import_graph`) compute cycle groups per module to populate `graph.import_graph_edges.cycle_group`. That’s exactly what NetworkX’s SCC and condensation utilities are built for, and you’re *already* using them in `analytics/graph_metrics._component_metadata`.

### 2.1 Replace `_tarjan_scc` entirely

Instead of maintaining your own Tarjan, you can do:

```python
import networkx as nx

def _tarjan_scc(graph: dict[str, set[str]]) -> dict[str, int]:
    g = nx.DiGraph()
    for src, dsts in graph.items():
        for dst in dsts:
            g.add_edge(src, dst)

    components = list(nx.strongly_connected_components(g))
    return {node: idx for idx, comp in enumerate(components) for node in comp}
```

This gives you:

* Shared semantics with `analytics.graph_metrics._component_metadata`.
* No custom SCC implementation to test/maintain.

### 2.2 (Optional) Introduce per‑module layers here too

You’re already computing an **import layer** in `analytics.graph_metrics._compute_module_graph_metrics` via condensation + `_dag_layers`. It may be worthwhile (if you want the layer to be available “earlier”) to compute module layers in `build_import_graph` as well and persist them into `graph.import_graph_edges` or a small `graph.import_modules` helper table.

Pattern:

```python
condensed = nx.condensation(g, scc=components)
layer_by_comp = _dag_layers(condensed)   # you already have this in graph_metrics.py
layer_by_module = {node: layer_by_comp[comp_index[node]] for node in g.nodes}
```

Then include `module_layer` in the rows you build for `ImportEdgeRow`. This gives downstream analytics (including subsystems) a ready classification of “core infra vs leaf module” without needing to recompute condensation at query time.

---

## 3. CFG / DFG: use NetworkX for intra‑function graph algorithms

**File:** `src/codeintel/graphs/cfg_builder.py`
Key classes/functions: `CFGBuilder`, `DFGBuilder`, `_build_cfg_for_function`, `build_cfg_and_dfg`.

Right now:

* `CFGBuilder` manages its own lists of `Block` and `Edge` dataclasses.
* Control‑flow constructs (`visit_If`, loops, `try/except/finally`, etc.) are encoded into block and edge lists.
* Data‑flow is built by `DFGBuilder` walking AST and block metadata.

You already persist CFG/DFG to DuckDB and later analytics can query them. But for **advanced control‑flow/data‑flow questions** (dominators, loop nesting, unreachable blocks, longest paths, etc.), NetworkX is a natural fit.

### 3.1 Add `to_networkx` views in CFGBuilder / DFGBuilder

Without changing your storage format, you can add:

```python
import networkx as nx

@dataclass
class CFGBuilder:
    ...
    blocks: list[Block] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def as_nx_digraph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for block in self.blocks:
            g.add_node(block.idx, kind=block.kind, start_line=block.start_line, end_line=block.end_line)
        for edge in self.edges:
            g.add_edge(edge.src_idx, edge.dst_idx, edge_type=edge.edge_type)
        return g
```

Then in `_build_cfg_for_function` (or right after you’ve built the blocks/edges), you can:

* Run invariants using NetworkX:

  * `nx.is_weakly_connected(cfg_graph)` for reachability.
  * `nx.algorithms.dag_longest_path_length(cfg_graph)` for “max straight‑line path length.”
* Optionally compute **dominators**, which NetworkX already provides via `nx.immediate_dominators(cfg_graph, entry_block_idx)`.

You can either:

* Store some of these as per‑function metrics (new `analytics.cfg_metrics` table).
* Or emit them into `function_profile` as extra columns (e.g., `cfg_longest_path`, `loop_nesting_depth`).

### 3.2 Use NetworkX in DFG for richer data‑flow

For `DFGBuilder`, you can do the same:

```python
@dataclass
class DFGBuilder:
    edges: list[DFGEdge] = field(default_factory=list)
    ...

    def as_nx_digraph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for edge in self.edges:
            g.add_edge(
                edge.src_block_idx,
                edge.dst_block_idx,
                src_symbol=edge.src_symbol,
                dst_symbol=edge.dst_symbol,
                use_kind=edge.use_kind,
            )
        return g
```

With that, you can:

* Use NetworkX’s `all_simple_paths` to answer “what are the possible def→use paths for symbol X?”
* Compute per‑function **data‑flow complexity** metrics (e.g., number of simple paths between entry and exit, number of edges per block) and feed them into `goid_risk_factors` as additional signals.

This is not just refactoring: it opens up a *whole class* of analyses (dominance, SSA‑like reasoning, path explosion hints) with very little extra code.

---

## 4. Richer graph validations using NetworkX

**File:** `src/codeintel/graphs/validation.py`
Key function: `run_graph_validations`.

This module currently does relatively lightweight checks:

* Missing GOIDs vs function counts per file.
* Callsite span mismatches, etc.

With the `graphs.nx_views` helpers, you can add **structural validations** that are hard to express in SQL but trivial in NetworkX:

### 4.1 Call graph sanity checks

Using `load_call_graph(con, repo, commit)`:

* Detect isolated nodes: `len([n for n in g.nodes if g.degree(n) == 0])`.
* Detect large SCCs: `nx.strongly_connected_components(g)` and flag components above a threshold as “suspicious recursion cluster.”
* Detect “megahub” functions: high out‑degree or in‑degree beyond a configured threshold.

These can be logged into `analytics.graph_validation` with structured reasons, just like your existing warnings.

### 4.2 Import graph anti‑patterns

Using `load_import_graph`:

* Identify **cycles** that cross top‑level packages and flag them (e.g., `foo.api` importing `foo.db` and vice versa).
* Flag modules that act as **bridges between subsystems** (high betweenness centrality across `subsystem_id` boundaries) and surface them as “architectural joints” where refactors are risky.

Implementation‑wise, this is just a new set of helper functions in `validation.py` that take a `nx.DiGraph` (import or call) and append findings to your existing `rows: list[dict[str, Any]]` before writing to `FINDINGS_TABLE`.

---

## 5. Optional: targeted NetworkX usage in the server layer for rich navigation

**Files:**

* `src/codeintel/server/datasets.py`
* `src/codeintel/server/fastapi.py`

Your FastAPI backend currently exposes mostly **table/view‑shaped** queries (things under `DOCS_VIEWS`, various profiles, etc.). For “best‑in‑class” navigation UX, you might add *a couple* of graph‑aware endpoints that internally use NetworkX:

* `GET /call-graph/neighborhood?goid=...&radius=N`

  * Implementation: load a small subgraph around the given GOID using the DB (`WHERE caller=... OR callee=...` plus iterative expansion), reify it as a `DiGraph`, then use `nx.ego_graph` to grab a radius‑N neighborhood. Return nodes + edges (and maybe metrics like in/out degree).
* `GET /import-graph/subsystem-boundary?subsystem_id=...`

  * Implementation: build a `DiGraph` of modules, color nodes by `subsystem_id`, and return all edges crossing the boundary.

These endpoints would:

* Be disabled by default or guarded by config (to avoid accidentally loading huge graphs).
* Make it trivial for MCP tools / UIs to ask “show me everything 2 hops out from this function” without hand‑rolling BFS in SQL each time.

From an implementation point of view, you’d add small helpers in `server/fastapi.py` that:

1. Use your existing DuckDB connection management.
2. Call into `graphs.nx_views` to build a tiny `nx.Graph` focused on the region of interest.
3. Serialize the resulting neighborhood to JSON.

---

## 6. Summary of concrete NetworkX touchpoints

Putting it all in one place, the main “real” extensions I’d prioritize are:

1. **Introduce a shared NetworkX view module**

   * `src/codeintel/graphs/nx_views.py` – central helpers to build call/import graphs from DuckDB into `nx.DiGraph`/`nx.Graph`.

2. **Subsystems clustering & stats on NetworkX**

   * `src/codeintel/analytics/subsystems.py`

     * Replace dict‑only adjacency with `_build_weighted_graph`.
     * Implement `_label_propagation_nx` using NetworkX neighbor iteration + weights (keeping seeded labels).
     * Use NetworkX import graph for `_subsystem_edge_stats` and potentially `_entrypoints_for_cluster`.

3. **Simplify import graph SCCs & layering**

   * `src/codeintel/graphs/import_graph.py`

     * Replace `_tarjan_scc` with `nx.strongly_connected_components`.
     * Optionally compute module layers via condensation and store them in your import‑graph tables for reuse.

4. **Add CFG/DFG graph views & metrics**

   * `src/codeintel/graphs/cfg_builder.py`

     * Add `as_nx_digraph` to `CFGBuilder` and `DFGBuilder`.
     * Use NetworkX to compute dominators, longest paths, and loop structure and feed those into a small `analytics.cfg_metrics` or directly into `function_profile`.

5. **Richer graph validations**

   * `src/codeintel/graphs/validation.py`

     * Add structural checks using NetworkX (big SCCs, hubs, bridges) and record findings in `analytics.graph_validation`.

6. **(Optional) Graph‑aware server endpoints**

   * `src/codeintel/server/fastapi.py`, `server/datasets.py`

     * Small, bounded NetworkX queries for “N‑hop neighborhood” style APIs.

All of these keep the current DuckDB‑first design intact and use NetworkX as an **in‑memory analytics layer** on top, which is exactly where it shines and where you get the biggest functional payoff for the complexity you’re taking on.


# metadata from networkx implementation plan #

Short version: you can treat **every graph you already export** as a NetworkX playground and add a *lot* of extra structure: richer centralities, components, communities, dominance, structural holes, bipartite projections, etc. All of that can be persisted as new `analytics.*` tables keyed by GOIDs/modules/tests/config keys so an LLM can pull whatever it needs. I’ll walk graph‑by‑graph and spell out concrete metrics + suggested tables.

I’ll lean on:

* Your revised metadata overview for what you already emit. 
* The NetworkX 3.6 capabilities summary (centrality, communities, dominance, structural holes, bipartite, etc.). 

---

## 0. General pattern for all of this

For each logical graph you already have in DuckDB:

* Materialize it as a `nx.Graph` / `nx.DiGraph` / bipartite graph in memory.
* Run one or more NetworkX algorithms.
* Persist results into new `analytics.*` tables:

  * one row per **node**: extra node metrics
  * optionally one row per **edge**: edge‑level metrics (e.g., edge betweenness)

You already have this pattern for `graph_metrics_functions.*` and `graph_metrics_modules.*` (fan‑in/out, degrees, PageRank, layers, symbol coupling). 
Everything below is essentially “version 2” of that idea across *all* your graphs.

For naming, I’ll write things like `analytics.graph_metrics_functions_ext` or “add columns to `graph_metrics_functions.*`”; you can decide whether to extend or create siblings.

---

## 1. Function call graph → **much richer function‑level graph metrics**

**Existing graph**
`graph.call_graph_nodes.*`, `graph.call_graph_edges.*` with directed `caller → callee` edges keyed by function GOID. 

**Existing metrics**
`graph_metrics_functions.*` already has: `call_fan_in`, `call_fan_out`, in/out‑degree, PageRank, layer index, and leaf flag. 

### 1.1 Additional node centralities

NetworkX has a big centrality module: betweenness, closeness, eigenvector, Katz, harmonic, current‑flow, load, etc. 

For each function node, add:

* `call_bc` – betweenness centrality (how often the function lies on shortest paths between others).
* `call_bc_approx` – optional sampled betweenness (store `k` sample size + flag `approx = true`).
* `call_closeness` – closeness centrality (average distance to all others in the same weakly‑connected component).
* `call_eigenvector` – eigenvector centrality (importance via important callees/callers).
* `call_katz` – Katz centrality (path‑weighted influence; nice for graphs with cycles).
* `call_harmonic` – harmonic centrality (variant of closeness more robust on disconnected graphs).
* `call_subgraph_centrality` / `call_estrada_index_contrib` – subgraph centrality contributions (how much each node contributes to the Estrada index).

All of these live naturally alongside your existing PageRank, and they’re gold for an LLM trying to decide “which functions are structurally core vs peripheral.” 

### 1.2 Structural roles & connectivity

From the connectivity/components algorithms. 

Per function:

* `call_component_id` – ID of its weakly connected component.
* `call_component_size` – size of that component.
* `call_scc_id` – strongly‑connected component (SCC) ID (i.e., recursion cluster).
* `call_scc_size` – size of its SCC.
* `call_is_recursive` – `scc_size > 1` or self‑loop.

From articulation/bridge detection:

* `call_is_articulation` – true if removing this node disconnects part of the graph (articulation point).
* `call_articulation_impact` – how many nodes become unreachable when it’s removed (rough measure of “architectural joint” functions).

Those make it easy to ask: “Which functions are recursive hubs?” or “What single function splits this entire subsystem if refactored wrong?”

### 1.3 Path‑based characteristics

Using shortest‑path and simple‑path algorithms. 

Per function:

* `call_min_distance_to_public_api` – shortest path length to any node tagged as public API (function/node tagged via `tags` or subsystem entrypoints). 
* `call_min_distance_to_tests` – via mixed graph of call graph + test coverage edges (see §5).
* `call_descendant_count` / `call_ancestor_count` – total unique callees/callers reachable along directed paths (within some depth limit).
* `call_longest_path_len_from_here` – max path length starting at the function (bounded for performance).

All of these help an agent answer “if I change this function, how far does the blast radius propagate?” without doing path searches on the fly.

### 1.4 Local clustering & motifs

From clustering, triads, and small‑world algorithms. 

Per function:

* `call_clustering_coeff` – local clustering coefficient on an undirected view of the call graph.
* `call_triangle_count` – number of length‑3 cycles (function A ↔ B ↔ C ↔ A).
* `call_motif_triads` – optional triad type counts (if you care about specific directed motifs).

These capture “tangles” of functions that call each other tightly, beyond what your SCC grouping alone reveals.

### 1.5 Community detection on the call graph

From the Communities & Clique modules. 

* Run a community detection algorithm (label propagation, modularity‑based, etc.) on the **undirected** view of your call graph.
* Persist:

  * `call_community_id` per function.
  * `analytics.call_communities` table:

    ```text
    community_id, repo, commit,
    member_count,
    avg_loc, avg_complexity,
    avg_risk_score, modules_json
    ```

This gives a second clustering axis (function‑level communities) that complements subsystems (module‑level clusters). 

---

## 2. Module import graph → **deeper module architecture metrics**

**Existing graph**
`graph.import_graph_edges.*` with directed `src_module → dst_module`, fan‑in/out, and SCC `cycle_group`. 

**Existing metrics**
`graph_metrics_modules.*` already has import fan‑in/out, degrees, import PageRank, optional call PageRank rollup, symbol coupling, layer index (in condensed import DAG), and cycle group. 

### 2.1 More centralities (module‑level)

Use the same centrality toolkit as for functions, but now over the module import graph. 

Add per module:

* `import_bc`, `import_bc_approx` – betweenness centrality.
* `import_closeness` – closeness in the import graph.
* `import_eigenvector`, `import_katz`.
* `import_harmonic`.
* `import_load_centrality` – load centrality (flow‑based influence).

Interpretation for LLMs:

* High `import_bc` = architecture “bridge module” that sits between subsystems.
* High `import_eigenvector` = central shared infra module everyone uses.
* Low centralities with low fan‑in/out = leaf modules safe to refactor in isolation.

### 2.2 Cores, shells, and rich club

NetworkX has cores, rich‑club, and “degree‑based” structure measures. 

Per module:

* `import_k_core` – highest k such that module is in the k‑core of the import graph.
* `import_core_shell` – shell index (re‑express `k_core` as a small integer layer).
* `import_rich_club_flag` – boolean if module is in top X% degree/rich‑club.

Global graph metrics table (e.g. `analytics.import_graph_stats`):

* `rich_club_coefficient` for high‑degree nodes.
* `degree_assortativity` (do high‑degree modules import other high‑degree modules?).

These give your architecture views a notion of “core infra” vs “periphery” beyond PageRank and layer index.

### 2.3 Structural holes and brokerage

NetworkX has a Structural Holes module with measures like **constraint** and **effective size**. 

Per module:

* `import_constraint` – how constrained a module is by its neighbors (low constraint = “broker” sitting between otherwise disconnected groups).
* `import_effective_size` – how many *unique* neighbors it effectively connects.

These are great signals for “this module is the one gluing together many loosely connected subsystems,” which is very useful for a planning agent.

### 2.4 Import communities & subsystem cross‑check

You already have subsystems (`subsystems.*`, `subsystem_modules.*`) derived from graph metrics + risk. 

You can run *pure* graph‑based communities on the import graph:

* Per module: `import_community_id`.
* `analytics.import_communities` table summarizing each community.

Then:

* Compare graph‑only communities to your existing `subsystem_id` to derive:

  * `subsystem_community_agreement` (fraction of modules whose community matches their subsystem).
  * “Suspicious” modules where community & subsystem strongly disagree (aliased modules, mis‑tagging, etc.).

Surface that via `graph_validation.*` as architecture warnings. 

### 2.5 Subsystem‑level graph metrics

Treat subsystems as nodes in a coarse graph:

* Build `G_subsystems` where nodes are `subsystem_id` and edges indicate imports between their member modules.
* Compute:

  * `subsystem_import_bc`, `subsystem_import_pagerank`.
  * `subsystem_layer_index` (DAG layer over condensed subsystem graph).
  * `subsystem_in_degree`, `out_degree`.

Persist into `analytics.subsystems` as extra columns (e.g. `graph_import_bc`, `graph_layer_index`). 

This gives you architecture‑level “which subsystem is at the bottom of the dependency stack?”

---

## 3. CFG / DFG → **intra‑function control/data‑flow richness**

You already export CFG blocks/edges and DFG edges keyed by `function_goid_h128` and block indices. 

NetworkX has dominance, DAG, paths, and centrality algorithms that map nicely onto per‑function graphs. 

### 3.1 Per‑function CFG graph metrics

For each function’s CFG (nodes = `block_idx`, edges = `src_block_idx → dst_block_idx`):

* **Dominance** (using the Dominance module):

  * Per block: `dom_depth`, `dom_children_count` (size of dominated subtree).
  * Flags: `is_loop_header`, `is_post_dominator` for typical loop/exit blocks.

* **Path metrics**:

  * `cfg_longest_path_len` – longest path length from entry to exit.
  * `cfg_avg_shortest_path_len` – average shortest path between blocks in the same component.
  * `cfg_branching_factor` – average `out_degree` of non‑exit blocks (you already store `in_degree`/`out_degree` in blocks; just aggregate). 

* **Centralities**:

  * `cfg_block_bc` – betweenness centrality per block (highlight “decision hubs”).
  * Aggregate into `cfg_bc_max`, `cfg_bc_mean` per function.

Persist:

* `analytics.cfg_block_metrics` keyed by `(function_goid_h128, block_idx)`
* `analytics.cfg_function_metrics` keyed by `function_goid_h128` with rollups.

These are extremely interpretable to an LLM: “this function has deeply nested control flow, long paths, and critical decision blocks.”

### 3.2 Per‑function DFG graph metrics

For each function’s DFG (`src_block_idx → dst_block_idx` edges with symbols):

* Node‑level centrality for blocks based on data‑flow (which blocks’ outputs are consumed widely).
* Simple path counts up to a small N to approximate “data‑flow complexity” (number of different data paths from entry to exit).

Store:

* `dfg_block_bc`, `dfg_block_influential` flags; aggregated `dfg_complexity_score` per function.

You can join these back into `function_profile.*` as `cfg_complexity_score` / `dfg_complexity_score` and even factor them into `risk_score`. 

---

## 4. Symbol‑use graph → **semantic coupling metrics**

`symbol_use_edges.*` gives you SCIP symbol def→use edges between files/modules. 

### 4.1 Module‑level symbol coupling graph

Build a weighted undirected graph where nodes are modules and an edge weight is proportional to how many symbols they share:

* From each `symbol_use_edges` row:

  * Map `def_path` + `use_path` → `def_module`, `use_module` (join via `modules.jsonl`). 
  * Increment `weight[def_module, use_module]`.

Then compute:

* `symbol_bc`, `symbol_closeness`, `symbol_eigenvector` per module on this graph.
* `symbol_constraint`, `symbol_effective_size` (structural holes) – “semantic brokers”.
* Symbol‑based communities: `symbol_community_id`.

You already have a scalar `symbol_coupling` in `graph_metrics_modules.*`; this is the “full graph‑theoretic version” of that idea. 

Persist as `analytics.symbol_graph_metrics_modules`.

### 4.2 Function‑level symbol coupling

If/when you add GOID‑level symbol edges (via `goid_crosswalk.scip_symbol`), you can duplicate the above at the function level:

* `analytics.symbol_graph_metrics_functions` keyed by `function_goid_h128`.

---

## 5. Test ↔ function bipartite graph → **test architecture metrics**

You have a perfect bipartite graph latent in `test_coverage_edges.*`: tests ↔ functions (GOIDs). 

NetworkX has a Bipartite module plus communities & centralities applicable to bipartite graphs. 

### 5.1 Node metrics on the bipartite graph

Construct a bipartite graph:

* Left set: `test_id` (optionally attach `test_goid_h128`). 
* Right set: `function_goid_h128`.
* Edge attribute: `coverage_ratio` from `test_coverage_edges.*`.

Per test node:

* `test_degree` – how many functions it covers.
* `test_weighted_degree` – sum of `coverage_ratio` it provides.
* `test_bc` – betweenness centrality within the bipartite graph (tests that connect otherwise disjoint function clusters).
* `test_target_risk_sum` – sum of risk scores of functions it covers (join `goid_risk_factors.*`). 

Per function node:

* `test_graph_degree` – number of tests covering it (already partly `tests_touching` in `function_profile`, but now in graph context). 
* `test_graph_bc` – how central it is in test coverage structure.

Persist:

* `analytics.test_graph_metrics_tests` keyed by `test_id`.
* `analytics.test_graph_metrics_functions` keyed by `function_goid_h128`.

### 5.2 Projected graphs: test–test and function–function similarity

Use bipartite projection functions to build:

* **Test–test graph**: tests connected with weight = number of functions they both cover (or total shared coverage).

  * Community detection → `test_suite_id`.
  * Node metrics → `test_cluster_centrality` (tests representative of a suite).

* **Function–function graph**: functions connected with weight = number of tests that cover both.

  * Communities → “co‑tested” clusters; great for impact / regression analysis.
  * Metrics like `co_tested_degree` highlight functions that tend to move together under the same tests.

Persist:

* `analytics.test_similarity_edges_tests`
* `analytics.test_similarity_edges_functions`
* Optionally `analytics.test_communities`, `analytics.function_test_communities`.

For an LLM agent doing change‑impact planning, this is hugely useful: “if I change function X, which other functions tend to be tested together with it?”

---

## 6. Config ↔ module bipartite graph → **configuration architecture**

`config_values.*` already tells you, for each config key, which modules reference it. 

Turn that into another bipartite graph:

* Left nodes: config keys (`key`).
* Right nodes: modules (`reference_modules`).

### 6.1 Node metrics

Per config key:

* `config_degree` – number of modules referencing it (`reference_count` you already have; now in graph context). 
* `config_bc`, `config_closeness` – central config values.
* `config_effective_size`, `config_constraint` – keys that bridge multiple module groups (e.g., global feature flags).

Per module:

* `config_degree` – number of distinct keys it touches.
* `config_centrality` – how central it is in config graph.

Persist as `analytics.config_graph_metrics_keys` / `analytics.config_graph_metrics_modules`.

### 6.2 Projected graphs

* Key–key graph: two keys are linked when they’re used in the same module.

  * Use communities to identify “config families” (e.g., `service.database.*` cluster together).
* Module–module config graph: modules linked when they share config usage.

LLMs can use this to answer “which modules are coupled through shared configuration?”—often more important than call dependencies.

---

## 7. Subsystem graph & cross‑subsystem relationships

You already have subsystem summaries and membership. 

Build a subsystem‑level graph:

* Nodes: `subsystem_id`.
* Directed edges: from A to B if any member module of A imports a member module of B.

Compute:

* `subsystem_bc`, `subsystem_pagerank`, `subsystem_closeness`.
* `subsystem_k_core`, `subsystem_layer`.
* `subsystem_import_count` (inter‑subsystem edge count).

Add to `subsystems.*`:

* `graph_centrality_bc`, `graph_centrality_closeness`, `graph_centrality_layer`, etc.

Now an LLM can query: “Give me the most central, high‑risk subsystems with low coverage” in a single join across `subsystems.*`, `graph_metrics_modules.*`, and risk rollups. 

---

## 8. Global graph‑level stats & validation

For each major graph (call, import, symbol, CFG, DFG, test‑function, config‑module):

* Global stats table `analytics.graph_stats`:

  ```text
  graph_name, repo, commit,
  node_count, edge_count,
  connected_components, scc_count,
  diameter_estimate, avg_shortest_path_estimate,
  avg_clustering_coeff,
  small_world_sigma,
  created_at
  ```

Use NetworkX’s components, clustering, small‑world, and efficiency modules. 

You can also extend `graph_validation.*` with **structural warnings**, e.g.:

* Modules whose import graph betweenness or structural‑holes score is extremely high.
* Functions whose CFG has pathological depth or longest path.

---

## 9. How this surfaces to the LLM

Once you’ve computed these metrics and written them into new `analytics.*` tables:

* Add them as joins to your `docs.v_*` views:

  * `docs.v_function_architecture` → join `graph_metrics_functions_ext`, CFG/DFG metrics, test graph metrics. 
  * `docs.v_module_architecture` → join extended module graph metrics (import, symbol, config). 
  * `docs.v_subsystem_summary` → join subsystem‑level graph metrics.

* Expose a couple of “generic graph metrics” endpoints in the server/MCP layer so the agent can ask things like:

  * “Give me the top 20 functions by call betweenness centrality.”
  * “List config keys with highest constraint/effective size.”
  * “Which tests cover the highest sum of high‑risk functions?”

Because your storage is already structured and DuckDB‑backed, adding these metrics is mostly about:

1. Materializing each graph into NetworkX.
2. Running the relevant algorithms.
3. Writing the results back with stable join keys (GOID, module, test_id, key, subsystem_id).

Everything else (views, MCP queries, LLM prompts) can build on top of that.


# meta data detailed implementation plan graph metrics #

Perfect, let’s make this concrete.

Below I’ll:

1. Specify **schemas** for:

   * `analytics.graph_metrics_functions_ext`
   * `analytics.test_graph_metrics_tests`
   * `analytics.test_graph_metrics_functions`
2. Sketch **DuckDB + NetworkX + Prefect** code to populate them, wired to the tables you already have (`call_graph_*`, `test_coverage_edges.*`, `test_catalog.*`, etc.).

I’ll keep names/types consistent with your existing analytics tables and views. 

---

## 1. `analytics.graph_metrics_functions_ext`

### 1.1 Purpose

Additional call‑graph metrics per function GOID, computed with NetworkX:

* richer centralities (betweenness, closeness, eigenvector, harmonic)
* local clustering
* k‑core index
* articulation‑point & bridge flags

This table is **one row per function**, keyed by `function_goid_h128`, `repo`, `commit`, and is designed to be joined onto `analytics.graph_metrics_functions.*` and `function_profile.*`. 

### 1.2 Schema

DuckDB table: `analytics.graph_metrics_functions_ext`

```sql
CREATE TABLE IF NOT EXISTS analytics.graph_metrics_functions_ext (
    function_goid_h128  DECIMAL(38,0) NOT NULL,
    repo                VARCHAR       NOT NULL,
    commit              VARCHAR       NOT NULL,

    -- Centralities over call graph (using DiGraph, but some on undirected view)
    call_betweenness        DOUBLE,  -- betweenness_centrality on DiGraph
    call_closeness          DOUBLE,  -- closeness_centrality on DiGraph (or largest WCC)
    call_eigenvector        DOUBLE,  -- eigenvector_centrality on undirected view
    call_harmonic           DOUBLE,  -- harmonic_centrality on DiGraph

    -- Local structure
    call_core_number        INTEGER, -- k-core index on undirected view
    call_clustering_coeff   DOUBLE,  -- clustering coefficient on undirected view
    call_triangle_count     BIGINT,  -- triangles(G_und)[node]

    -- Structural roles
    call_is_articulation    BOOLEAN, -- articulation_points on undirected view
    call_articulation_impact INTEGER, -- optional: estimated nodes cut off if removed
    call_is_bridge_endpoint BOOLEAN, -- incident to at least one bridge edge

    -- Graph/component context
    call_component_id       INTEGER, -- weakly connected component id
    call_component_size     INTEGER, -- size of that component
    call_scc_id             INTEGER, -- strongly connected component id
    call_scc_size           INTEGER, -- size of SCC

    created_at              TIMESTAMP NOT NULL,

    PRIMARY KEY (function_goid_h128, repo, commit)
);
```

This **does not duplicate** columns that already exist in `graph_metrics_functions.*` (fan‑in/out, degrees, PageRank, `call_layer`, `call_is_leaf`).

### 1.3 Building the NetworkX graph

We’ll use your call‑graph edges (`graph.call_graph_edges.*`) keyed by `caller_goid_h128`, `callee_goid_h128`. 

```python
# src/codeintel/analytics/nx_views.py (new helper module)
from __future__ import annotations

from decimal import Decimal
from typing import Iterable

import duckdb
import networkx as nx


def load_call_graph(
    con: duckdb.DuckDBPyConnection, repo: str, commit: str
) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph of the call graph for a given repo/commit.

    Nodes: function_goid_h128 (Decimal -> int)
    Edges: caller -> callee (only where callee is resolved)
    """
    rows: Iterable[tuple[Decimal, Decimal | None]] = con.execute(
        """
        SELECT caller_goid_h128, callee_goid_h128
        FROM graph.call_graph_edges
        WHERE repo = ? AND commit = ? AND callee_goid_h128 IS NOT NULL
        """,
        [repo, commit],
    ).fetchall()

    G = nx.DiGraph()
    for caller, callee in rows:
        if caller is None or callee is None:
            continue
        u = int(caller)
        v = int(callee)
        # multiple edges: treat as simple graph with multiplicity in "weight"
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G
```

> You can also add all nodes from `graph.call_graph_nodes` to ensure isolated functions show up in metrics. 

### 1.4 Computing metrics with NetworkX

All algorithms below are standard in the NetworkX 3.6 centrality/structure modules. 

```python
# src/codeintel/analytics/graph_metrics_functions_ext.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict, Any

import duckdb
import networkx as nx

from .nx_views import load_call_graph


def compute_function_graph_metrics_ext_for_repo(
    con: duckdb.DuckDBPyConnection, repo: str, commit: str
) -> list[dict[str, Any]]:
    G = load_call_graph(con, repo, commit)
    G_und = G.to_undirected()

    now = datetime.utcnow()

    # Centralities (you may want approximate betweenness for large graphs)
    bet = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()))
    clo = nx.closeness_centrality(G)
    # Use undirected view for eigenvector/harmonic for stability
    eig = nx.eigenvector_centrality(G_und, max_iter=200)
    har = nx.harmonic_centrality(G)

    core = nx.core_number(G_und) if G_und.number_of_nodes() > 0 else {}
    clustering = nx.clustering(G_und)
    triangles = nx.triangles(G_und)

    # Components
    comp_id: dict[int, int] = {}
    comp_size: dict[int, int] = {}
    for cid, nodes in enumerate(nx.weakly_connected_components(G)):
        size = len(nodes)
        for n in nodes:
            comp_id[n] = cid
            comp_size[n] = size

    scc_id: dict[int, int] = {}
    scc_size: dict[int, int] = {}
    for sid, nodes in enumerate(nx.strongly_connected_components(G)):
        size = len(nodes)
        for n in nodes:
            scc_id[n] = sid
            scc_size[n] = size

    # Articulation points & bridges on undirected view
    arts = set(nx.articulation_points(G_und))
    bridges = set(nx.bridges(G_und))
    bridge_incident: dict[int, int] = {n: 0 for n in G_und.nodes}
    for u, v in bridges:
        bridge_incident[u] += 1
        bridge_incident[v] += 1

    # Optional: crude articulation impact = component_size - largest WCC size if node removed
    # (You can refine; here we just mark big articulation points specially.)
    # For a first pass, you may store 0 and fill later if needed.

    rows: list[dict[str, Any]] = []
    for n in G.nodes:
        # NetworkX node id -> Decimal for DuckDB compatibility
        goid = Decimal(n)

        rows.append(
            {
                "function_goid_h128": goid,
                "repo": repo,
                "commit": commit,
                "call_betweenness": bet.get(n, 0.0),
                "call_closeness": clo.get(n, 0.0),
                "call_eigenvector": eig.get(n, 0.0),
                "call_harmonic": har.get(n, 0.0),
                "call_core_number": core.get(n),
                "call_clustering_coeff": clustering.get(n, 0.0),
                "call_triangle_count": int(triangles.get(n, 0)),
                "call_is_articulation": n in arts,
                "call_articulation_impact": None,  # fill if you implement impact
                "call_is_bridge_endpoint": bridge_incident.get(n, 0) > 0,
                "call_component_id": comp_id.get(n),
                "call_component_size": comp_size.get(n),
                "call_scc_id": scc_id.get(n),
                "call_scc_size": scc_size.get(n),
                "created_at": now,
            }
        )

    return rows
```

### 1.5 Writing into DuckDB

Assuming you already have a central connection helper (e.g. `storage.get_connection()`), you can register the rows as a DuckDB relation and insert/replace:

```python
import duckdb
import pandas as pd


def upsert_graph_metrics_functions_ext(
    con: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]
) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    con.register("metrics_ext_df", df)

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS analytics.graph_metrics_functions_ext AS
        SELECT * FROM metrics_ext_df
        """
    )

    # If table already exists, replace rows for this repo+commit
    con.execute(
        """
        DELETE FROM analytics.graph_metrics_functions_ext
        WHERE repo = ? AND commit = ?
        """,
        [rows[0]["repo"], rows[0]["commit"]],
    )
    con.execute(
        """
        INSERT INTO analytics.graph_metrics_functions_ext
        SELECT * FROM metrics_ext_df
        """
    )
    con.unregister("metrics_ext_df")
```

### 1.6 Prefect wiring

```python
# src/codeintel/orchestration/flows/graph_metrics_ext.py
from __future__ import annotations

from pathlib import Path

import duckdb
from prefect import flow, task

from codeintel.storage import get_connection  # your existing helper
from codeintel.analytics.graph_metrics_functions_ext import (
    compute_function_graph_metrics_ext_for_repo,
    upsert_graph_metrics_functions_ext,
)


@task
def compute_and_store_function_graph_metrics_ext(
    db_path: Path, repo: str, commit: str
) -> None:
    con = get_connection(db_path)
    rows = compute_function_graph_metrics_ext_for_repo(con, repo, commit)
    upsert_graph_metrics_functions_ext(con, rows)
    con.close()


@flow
def enrich_graph_metrics_ext(db_path: Path, repo: str, commit: str) -> None:
    compute_and_store_function_graph_metrics_ext(db_path, repo, commit)
```

You can call this `enrich_graph_metrics_ext` flow from your existing `enrich_pipeline all` or from the same orchestration stage that currently runs `graph_metrics_functions`. 

---

## 2. Test bipartite metrics: `analytics.test_graph_metrics_*`

We’ll use `analytics.test_coverage_edges.*` and `analytics.test_catalog.*` to build a bipartite graph between tests and functions, then compute metrics on both partitions. 

### 2.1 Graph construction

* Nodes:

  * Tests: keyed by `test_id` (string).
  * Functions: keyed by `function_goid_h128` (Decimal).
* Edges:

  * From `test_coverage_edges.*`: `(test_id, function_goid_h128)` with attributes:

    * `coverage_ratio`
    * `covered_lines`
    * `executable_lines` 

```python
# src/codeintel/analytics/nx_views.py (add this)
from typing import Tuple

import networkx as nx
from decimal import Decimal


def load_test_function_bipartite(
    con: duckdb.DuckDBPyConnection, repo: str, commit: str
) -> nx.Graph:
    """
    Bipartite graph: tests <-> functions.

    Left partition: ("t", test_id)
    Right partition: ("f", function_goid_h128 as int)
    """
    rows: list[Tuple[str, Decimal, float | None]] = con.execute(
        """
        SELECT test_id, function_goid_h128, COALESCE(coverage_ratio, 0.0)
        FROM analytics.test_coverage_edges
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()

    B = nx.Graph()
    for test_id, func_goid, cov in rows:
        t_node = ("t", test_id)
        f_node = ("f", int(func_goid))
        if not B.has_node(t_node):
            B.add_node(t_node, bipartite=0)
        if not B.has_node(f_node):
            B.add_node(f_node, bipartite=1)

        # accumulate weight by coverage contribution
        if B.has_edge(t_node, f_node):
            B[t_node][f_node]["weight"] += float(cov)
        else:
            B.add_edge(t_node, f_node, weight=float(cov))
    return B
```

### 2.2 Schemas

#### 2.2.1 `analytics.test_graph_metrics_tests`

Per‑test metrics.

```sql
CREATE TABLE IF NOT EXISTS analytics.test_graph_metrics_tests (
    test_id            VARCHAR NOT NULL,
    repo               VARCHAR NOT NULL,
    commit             VARCHAR NOT NULL,

    -- Basic graph stats
    degree             INTEGER,  -- number of functions this test executes
    weighted_degree    DOUBLE,   -- sum of edge weights (e.g. coverage_ratio)
    degree_centrality  DOUBLE,   -- bipartite degree centrality on test side

    -- Projection-based stats (optional, can be null if projection not computed)
    proj_degree        INTEGER,  -- degree in test-test projection
    proj_weight        DOUBLE,   -- sum of weights in projection
    proj_clustering    DOUBLE,   -- clustering in test-test projection
    proj_betweenness   DOUBLE,   -- betweenness in test-test projection

    created_at         TIMESTAMP NOT NULL,

    PRIMARY KEY (test_id, repo, commit)
);
```

#### 2.2.2 `analytics.test_graph_metrics_functions`

Per‑function metrics from the same bipartite / projected graphs.

```sql
CREATE TABLE IF NOT EXISTS analytics.test_graph_metrics_functions (
    function_goid_h128  DECIMAL(38,0) NOT NULL,
    repo                VARCHAR       NOT NULL,
    commit              VARCHAR       NOT NULL,

    -- How this function sits in test coverage graph
    tests_degree        INTEGER,  -- number of tests executing it
    tests_weighted_degree DOUBLE, -- sum of per-edge coverage weights
    tests_degree_centrality DOUBLE,

    -- Projection-based stats (function-function via shared tests)
    proj_degree         INTEGER,
    proj_weight         DOUBLE,
    proj_clustering     DOUBLE,
    proj_betweenness    DOUBLE,

    created_at          TIMESTAMP NOT NULL,

    PRIMARY KEY (function_goid_h128, repo, commit)
);
```

These are cleanly joinable onto `function_profile.*` (via `function_goid_h128`) and `test_catalog.*` (via `test_id`). 

### 2.3 Computing metrics with NetworkX bipartite tools

NetworkX 3.6 has a dedicated **Bipartite** module for degree centrality and projections. 

```python
# src/codeintel/analytics/test_graph_metrics.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Tuple

import duckdb
import networkx as nx
from networkx.algorithms import bipartite

from .nx_views import load_test_function_bipartite


def compute_test_graph_metrics_for_repo(
    con: duckdb.DuckDBPyConnection, repo: str, commit: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    B = load_test_function_bipartite(con, repo, commit)

    now = datetime.utcnow()

    # Partitions
    tests = {n for n, data in B.nodes(data=True) if data.get("bipartite") == 0}
    funcs = set(B) - tests

    # Basic degrees
    deg = dict(B.degree(weight=None))
    wdeg = dict(B.degree(weight="weight"))

    # Bipartite degree centralities
    test_dc = bipartite.degree_centrality(B, funcs)  # centrality for tests
    func_dc = bipartite.degree_centrality(B, tests)  # centrality for funcs

    # Projections
    G_tests = bipartite.weighted_projected_graph(B, tests)
    G_funcs = bipartite.weighted_projected_graph(B, funcs)

    # Test-side projection metrics
    test_proj_deg = dict(G_tests.degree(weight=None))
    test_proj_wdeg = dict(G_tests.degree(weight="weight"))
    test_proj_clustering = nx.clustering(G_tests, weight="weight") if G_tests else {}
    test_proj_bet = (
        nx.betweenness_centrality(G_tests, weight="weight", k=min(200, G_tests.number_of_nodes()))
        if G_tests.number_of_nodes() > 0
        else {}
    )

    # Function-side projection metrics
    func_proj_deg = dict(G_funcs.degree(weight=None))
    func_proj_wdeg = dict(G_funcs.degree(weight="weight"))
    func_proj_clustering = nx.clustering(G_funcs, weight="weight") if G_funcs else {}
    func_proj_bet = (
        nx.betweenness_centrality(G_funcs, weight="weight", k=min(200, G_funcs.number_of_nodes()))
        if G_funcs.number_of_nodes() > 0
        else {}
    )

    test_rows: list[dict[str, Any]] = []
    func_rows: list[dict[str, Any]] = []

    for n in tests:
        prefix, test_id = n  # ("t", test_id)
        test_rows.append(
            {
                "test_id": test_id,
                "repo": repo,
                "commit": commit,
                "degree": int(deg.get(n, 0)),
                "weighted_degree": float(wdeg.get(n, 0.0)),
                "degree_centrality": float(test_dc.get(n, 0.0)),
                "proj_degree": int(test_proj_deg.get(n, 0)),
                "proj_weight": float(test_proj_wdeg.get(n, 0.0)),
                "proj_clustering": float(test_proj_clustering.get(n, 0.0)),
                "proj_betweenness": float(test_proj_bet.get(n, 0.0)),
                "created_at": now,
            }
        )

    for n in funcs:
        prefix, func_id_int = n  # ("f", int(function_goid_h128))
        func_rows.append(
            {
                "function_goid_h128": Decimal(func_id_int),
                "repo": repo,
                "commit": commit,
                "tests_degree": int(deg.get(n, 0)),
                "tests_weighted_degree": float(wdeg.get(n, 0.0)),
                "tests_degree_centrality": float(func_dc.get(n, 0.0)),
                "proj_degree": int(func_proj_deg.get(n, 0)),
                "proj_weight": float(func_proj_wdeg.get(n, 0.0)),
                "proj_clustering": float(func_proj_clustering.get(n, 0.0)),
                "proj_betweenness": float(func_proj_bet.get(n, 0.0)),
                "created_at": now,
            }
        )

    return test_rows, func_rows
```

### 2.4 Upsert helpers

```python
import pandas as pd


def upsert_test_graph_metrics(
    con: duckdb.DuckDBPyConnection,
    repo: str,
    commit: str,
    test_rows: list[dict[str, Any]],
    func_rows: list[dict[str, Any]],
) -> None:
    if test_rows:
        df_tests = pd.DataFrame(test_rows)
        con.register("test_metrics_df", df_tests)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics.test_graph_metrics_tests AS
            SELECT * FROM test_metrics_df
            """
        )
        con.execute(
            "DELETE FROM analytics.test_graph_metrics_tests WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        con.execute(
            "INSERT INTO analytics.test_graph_metrics_tests SELECT * FROM test_metrics_df"
        )
        con.unregister("test_metrics_df")

    if func_rows:
        df_funcs = pd.DataFrame(func_rows)
        con.register("func_metrics_df", df_funcs)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics.test_graph_metrics_functions AS
            SELECT * FROM func_metrics_df
            """
        )
        con.execute(
            "DELETE FROM analytics.test_graph_metrics_functions WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        con.execute(
            "INSERT INTO analytics.test_graph_metrics_functions SELECT * FROM func_metrics_df"
        )
        con.unregister("func_metrics_df")
```

### 2.5 Prefect flow

```python
# src/codeintel/orchestration/flows/test_graph_metrics.py
from __future__ import annotations

from pathlib import Path

import duckdb
from prefect import flow, task

from codeintel.storage import get_connection
from codeintel.analytics.test_graph_metrics import (
    compute_test_graph_metrics_for_repo,
    upsert_test_graph_metrics,
)


@task
def compute_and_store_test_graph_metrics(
    db_path: Path, repo: str, commit: str
) -> None:
    con = get_connection(db_path)
    test_rows, func_rows = compute_test_graph_metrics_for_repo(con, repo, commit)
    upsert_test_graph_metrics(con, repo, commit, test_rows, func_rows)
    con.close()


@flow
def enrich_test_graph_metrics(db_path: Path, repo: str, commit: str) -> None:
    compute_and_store_test_graph_metrics(db_path, repo, commit)
```

You can schedule `enrich_test_graph_metrics` alongside your existing `test-analytics` step, since it depends on `test_coverage_edges.*` and `test_catalog.*` being populated. 

---

## 3. Hooking into docs views (LLM‑facing)

Once these tables exist, you can expose them to the LLM via your `docs.*` views.

### 3.1 Extend `docs.v_function_architecture`

Add joins to both `graph_metrics_functions_ext` and `test_graph_metrics_functions`:

```sql
CREATE OR REPLACE VIEW docs.v_function_architecture AS
SELECT
    fp.*,
    gmf.call_pagerank,
    gmf.call_layer,
    gmf.call_is_leaf,
    gmf_ext.call_betweenness,
    gmf_ext.call_closeness,
    gmf_ext.call_eigenvector,
    gmf_ext.call_harmonic,
    gmf_ext.call_core_number,
    gmf_ext.call_clustering_coeff,
    gmf_ext.call_triangle_count,
    gmf_ext.call_is_articulation,
    gmf_ext.call_is_bridge_endpoint,
    tgf.tests_degree,
    tgf.tests_weighted_degree,
    tgf.tests_degree_centrality,
    tgf.proj_degree        AS tests_co_tested_degree,
    tgf.proj_betweenness   AS tests_co_tested_betweenness
FROM analytics.function_profile fp
LEFT JOIN analytics.graph_metrics_functions gmf
  ON gmf.function_goid_h128 = fp.function_goid_h128
LEFT JOIN analytics.graph_metrics_functions_ext gmf_ext
  ON gmf_ext.function_goid_h128 = fp.function_goid_h128
LEFT JOIN analytics.test_graph_metrics_functions tgf
  ON tgf.function_goid_h128 = fp.function_goid_h128;
```

Now an LLM agent gets all the **call‑graph** and **test‑graph** structure for a function from a single row.

### 3.2 Test‑oriented docs view

Optionally add a new view `docs.v_test_architecture` to surface per‑test metrics:

```sql
CREATE OR REPLACE VIEW docs.v_test_architecture AS
SELECT
    tc.*,
    tgm.degree                  AS function_count,
    tgm.weighted_degree         AS total_coverage_weight,
    tgm.degree_centrality,
    tgm.proj_degree             AS test_suite_degree,
    tgm.proj_betweenness        AS test_suite_betweenness
FROM analytics.test_catalog tc
LEFT JOIN analytics.test_graph_metrics_tests tgm
  ON tc.test_id = tgm.test_id;
```

Now agents can ask “which tests are structurally most central?” or “which tests cover many high‑risk functions?” via simple SQL joins with `goid_risk_factors.*`. 

---

# networkx metadata for cfg/dfg #

Nice, this is the fun part. Let’s give CFG/DFG the same treatment we gave call/test graphs.

I’ll do:

1. Schemas for:

   * `analytics.cfg_block_metrics`
   * `analytics.cfg_function_metrics`
   * `analytics.dfg_block_metrics`
   * `analytics.dfg_function_metrics`
2. Sketch code: DuckDB + NetworkX + Prefect to populate them.

All of this sits on top of your existing `graph.cfg_blocks.*` / `graph.cfg_edges.*` and `graph.dfg_edges.*` tables.
For algorithms I rely on NetworkX’s dominance, centrality, components, and DAG/shortest‑path modules. 

---

## 1. CFG metrics

### 1.1 Per‑block CFG metrics

**Table:** `analytics.cfg_block_metrics`

One row per `(function_goid_h128, block_idx)`.

```sql
CREATE TABLE IF NOT EXISTS analytics.cfg_block_metrics (
    function_goid_h128     DECIMAL(38,0) NOT NULL,
    repo                   TEXT          NOT NULL,
    commit                 TEXT          NOT NULL,
    block_idx              INTEGER       NOT NULL,

    -- Local structure (building on cfg_blocks.in_degree/out_degree)
    is_entry               BOOLEAN,   -- from cfg_blocks.kind = 'entry'
    is_exit                BOOLEAN,   -- kind = 'exit'
    is_branch              BOOLEAN,   -- out_degree > 1
    is_join                BOOLEAN,   -- in_degree > 1

    -- Dominator tree metrics (NetworkX Dominance)
    dom_depth              INTEGER,   -- depth from entry in immediate dominator tree
    dominates_exit         BOOLEAN,   -- true if all paths to exit go through this block

    -- Centralities on per-function CFG
    bc_betweenness         DOUBLE,    -- betweenness_centrality(G_cfg)
    bc_closeness           DOUBLE,    -- closeness_centrality(G_cfg)
    bc_eigenvector         DOUBLE,    -- eigenvector_centrality on undirected view

    -- Loop / cycle structure
    in_loop_scc            BOOLEAN,   -- in SCC of size > 1
    loop_header            BOOLEAN,   -- heuristic: entry of SCC or backedge target
    loop_nesting_depth     INTEGER,   -- nesting depth of loops containing this block

    created_at             TIMESTAMP  NOT NULL,
    metrics_version        INTEGER    DEFAULT 1,

    PRIMARY KEY (function_goid_h128, repo, commit, block_idx)
);
```

You already have block kind + degrees in `graph.cfg_blocks.*`, so these metrics are strictly additive. 

### 1.2 Per‑function CFG metrics

**Table:** `analytics.cfg_function_metrics`

One row per function GOID, summarizing the CFG shape and complexity.

```sql
CREATE TABLE IF NOT EXISTS analytics.cfg_function_metrics (
    function_goid_h128          DECIMAL(38,0) NOT NULL,
    repo                        TEXT          NOT NULL,
    commit                      TEXT          NOT NULL,
    rel_path                    TEXT          NOT NULL,
    module                      TEXT,
    qualname                    TEXT,

    -- Size
    cfg_block_count             INTEGER,   -- number of blocks
    cfg_edge_count              INTEGER,   -- number of edges
    cfg_entry_block_idx         INTEGER,
    cfg_exit_block_idx          INTEGER,

    -- Graph structure
    cfg_is_dag                  BOOLEAN,   -- is_directed_acyclic_graph
    cfg_scc_count               INTEGER,   -- strongly connected components
    cfg_has_cycles              BOOLEAN,

    -- Path metrics
    cfg_longest_path_len        INTEGER,   -- approx longest path (blocks) entry->exit
    cfg_avg_shortest_path_len   DOUBLE,    -- within reachable region from entry

    -- Branching / linearity
    cfg_branching_factor_mean   DOUBLE,    -- avg out_degree over non-exit blocks
    cfg_branching_factor_max    INTEGER,
    cfg_linear_block_fraction   DOUBLE,    -- fraction of blocks with in=1 & out=1

    -- Dominance summary
    cfg_dom_tree_height         INTEGER,   -- max dom_depth from entry
    cfg_dominance_frontier_size_mean DOUBLE,
    cfg_dominance_frontier_size_max  INTEGER,

    -- Loop metrics
    cfg_loop_count              INTEGER,   -- simple cycle count or SCC-based
    cfg_loop_nesting_depth_max  INTEGER,

    -- Centrality aggregates over blocks
    cfg_bc_betweenness_max      DOUBLE,
    cfg_bc_betweenness_mean     DOUBLE,
    cfg_bc_closeness_mean       DOUBLE,
    cfg_bc_eigenvector_max      DOUBLE,

    created_at                  TIMESTAMP  NOT NULL,
    metrics_version             INTEGER    DEFAULT 1,

    PRIMARY KEY (function_goid_h128, repo, commit)
);
```

These join nicely onto `analytics.function_profile` and `docs.v_function_architecture` via `function_goid_h128`. 

---

## 2. DFG metrics

The DFG edges are per‑function `(function_goid_h128, src_block_idx, dst_block_idx, src_symbol, dst_symbol, via_phi, use_kind)`. 

We’ll treat each function’s DFG as a directed multigraph over **blocks**, with edge attributes describing symbols and phi usage.

### 2.1 Per‑block DFG metrics

**Table:** `analytics.dfg_block_metrics`

```sql
CREATE TABLE IF NOT EXISTS analytics.dfg_block_metrics (
    function_goid_h128     DECIMAL(38,0) NOT NULL,
    repo                   TEXT          NOT NULL,
    commit                 TEXT          NOT NULL,
    block_idx              INTEGER       NOT NULL,

    -- Local data-flow structure
    dfg_in_degree          INTEGER,  -- number of incoming data edges
    dfg_out_degree         INTEGER,  -- number of outgoing data edges
    dfg_phi_in_degree      INTEGER,  -- incoming phi edges
    dfg_phi_out_degree     INTEGER,  -- outgoing phi edges

    -- Centralities on per-function DFG
    dfg_bc_betweenness     DOUBLE,
    dfg_bc_closeness       DOUBLE,
    dfg_bc_eigenvector     DOUBLE,

    -- Participation in long chains / SCCs
    dfg_in_chain           BOOLEAN,  -- on some longest data-flow path
    dfg_in_scc             BOOLEAN,  -- SCC size > 1 (data cycles)

    created_at             TIMESTAMP NOT NULL,
    metrics_version        INTEGER   DEFAULT 1,

    PRIMARY KEY (function_goid_h128, repo, commit, block_idx)
);
```

### 2.2 Per‑function DFG metrics

**Table:** `analytics.dfg_function_metrics`

```sql
CREATE TABLE IF NOT EXISTS analytics.dfg_function_metrics (
    function_goid_h128        DECIMAL(38,0) NOT NULL,
    repo                      TEXT          NOT NULL,
    commit                    TEXT          NOT NULL,
    rel_path                  TEXT          NOT NULL,
    module                    TEXT,
    qualname                  TEXT,

    -- Size
    dfg_block_count           INTEGER,   -- distinct blocks in DFG
    dfg_edge_count            INTEGER,   -- total DFG edges
    dfg_phi_edge_count        INTEGER,   -- via_phi = true
    dfg_symbol_count          INTEGER,   -- distinct src_symbol/dst_symbol

    -- Connectivity
    dfg_component_count       INTEGER,
    dfg_scc_count             INTEGER,
    dfg_has_cycles            BOOLEAN,

    -- Path / chain structure
    dfg_longest_chain_len     INTEGER,   -- approx longest dependency chain (blocks)
    dfg_avg_shortest_path_len DOUBLE,

    -- Degree/branching
    dfg_avg_in_degree         DOUBLE,
    dfg_avg_out_degree        DOUBLE,
    dfg_max_in_degree         INTEGER,
    dfg_max_out_degree        INTEGER,
    dfg_branchy_block_fraction DOUBLE,   -- blocks with out_degree > 1

    -- Centrality aggregates
    dfg_bc_betweenness_max    DOUBLE,
    dfg_bc_betweenness_mean   DOUBLE,
    dfg_bc_eigenvector_max    DOUBLE,

    created_at                TIMESTAMP  NOT NULL,
    metrics_version           INTEGER    DEFAULT 1,

    PRIMARY KEY (function_goid_h128, repo, commit)
);
```

Again, designed to join directly into `docs.v_function_architecture`.

---

## 3. Implementation sketch (NetworkX + DuckDB + Prefect)

Below is reasonably close to drop‑in code. It assumes:

* `get_connection(db_path: str) -> duckdb.DuckDBPyConnection` exists.
* You’re using Prefect 2 (`@task`/`@flow`).

### 3.1 Loading CFG & DFG from DuckDB

```python
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple

import duckdb
import networkx as nx
import pandas as pd


@dataclass
class RepoContext:
    repo: str
    commit: str
```

#### CFG

```python
def load_cfg_rows(
    con: duckdb.DuckDBPyConnection,
    ctx: RepoContext,
) -> tuple[
    Dict[int, List[Tuple[int, str, int, int]]],
    Dict[int, List[Tuple[int, int, str]]],
]:
    """
    Return (blocks_by_fn, edges_by_fn).

    blocks_by_fn[fn_goid] = [(block_idx, kind, in_degree, out_degree), ...]
    edges_by_fn[fn_goid]  = [(src_block_idx, dst_block_idx, edge_type), ...]
    """
    block_rows = con.execute(
        """
        SELECT function_goid_h128::BIGINT AS fn,
               block_idx,
               kind,
               in_degree,
               out_degree
        FROM graph.cfg_blocks
        WHERE repo = ? AND commit = ?
        """,
        [ctx.repo, ctx.commit],
    ).fetchall()

    edge_rows = con.execute(
        """
        SELECT function_goid_h128::BIGINT AS fn,
               src_block_idx,
               dst_block_idx,
               edge_type
        FROM graph.cfg_edges
        WHERE repo = ? AND commit = ?
        """,
        [ctx.repo, ctx.commit],
    ).fetchall()

    blocks_by_fn: Dict[int, List[Tuple[int, str, int, int]]] = defaultdict(list)
    edges_by_fn: Dict[int, List[Tuple[int, int, str]]] = defaultdict(list)

    for fn, idx, kind, indeg, outdeg in block_rows:
        blocks_by_fn[int(fn)].append((int(idx), kind, int(indeg), int(outdeg)))

    for fn, src, dst, etype in edge_rows:
        edges_by_fn[int(fn)].append((int(src), int(dst), etype))

    return blocks_by_fn, edges_by_fn
```

#### DFG

```python
def load_dfg_rows(
    con: duckdb.DuckDBPyConnection,
    ctx: RepoContext,
) -> Dict[int, List[Tuple[int, int, str, str, bool, str]]]:
    """
    dfg_by_fn[fn_goid] = [
        (src_block_idx, dst_block_idx, src_symbol, dst_symbol, via_phi, use_kind),
        ...
    ]
    """
    rows = con.execute(
        """
        SELECT function_goid_h128::BIGINT AS fn,
               src_block_idx,
               dst_block_idx,
               src_symbol,
               dst_symbol,
               via_phi,
               use_kind
        FROM graph.dfg_edges
        WHERE repo = ? AND commit = ?
        """,
        [ctx.repo, ctx.commit],
    ).fetchall()

    dfg_by_fn: Dict[int, List[Tuple[int, int, str, str, bool, str]]] = defaultdict(list)
    for fn, src, dst, src_sym, dst_sym, via_phi, use_kind in rows:
        dfg_by_fn[int(fn)].append(
            (int(src), int(dst), src_sym, dst_sym, bool(via_phi), use_kind)
        )
    return dfg_by_fn
```

---

### 3.2 Per‑function CFG metrics (including block metrics)

```python
def compute_cfg_metrics_for_repo(
    con: duckdb.DuckDBPyConnection,
    ctx: RepoContext,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks_by_fn, edges_by_fn = load_cfg_rows(con, ctx)

    # Basic function metadata for joins
    fn_meta = con.execute(
        """
        SELECT function_goid_h128::BIGINT AS goid,
               repo, commit, rel_path, module, qualname
        FROM analytics.function_profile
        WHERE repo = ? AND commit = ?
        """,
        [ctx.repo, ctx.commit],
    ).fetchall()

    now = datetime.now(timezone.utc)
    fn_records: list[dict] = []
    block_records: list[dict] = []

    for goid, repo, commit, rel_path, module, qualname in fn_meta:
        fn = int(goid)
        blocks = blocks_by_fn.get(fn, [])
        edges = edges_by_fn.get(fn, [])

        if not blocks:
            continue

        # Build CFG graph
        G = nx.DiGraph()
        entry_idx = None
        exit_idx = None

        for idx, kind, indeg, outdeg in blocks:
            G.add_node(idx, kind=kind, in_degree=indeg, out_degree=outdeg)
            if kind == "entry":
                entry_idx = idx
            elif kind == "exit":
                exit_idx = idx

        for src, dst, edge_type in edges:
            G.add_edge(src, dst, edge_type=edge_type)

        if entry_idx is None:
            # Fallback: pick node with minimum in-degree as entry
            entry_idx = min(G.nodes, key=lambda n: G.in_degree(n))
        if exit_idx is None:
            # Fallback: node with out-degree 0
            exit_candidates = [n for n in G.nodes if G.out_degree(n) == 0]
            exit_idx = exit_candidates[0] if exit_candidates else entry_idx

        # Structural properties
        is_dag = nx.is_directed_acyclic_graph(G)
        sccs = list(nx.strongly_connected_components(G))
        scc_count = len(sccs)
        has_cycles = not is_dag

        # Path metrics
        try:
            if is_dag:
                # Restrict to reachable from entry
                reachable = nx.descendants(G, entry_idx) | {entry_idx}
                H = G.subgraph(reachable).copy()
                longest_path_len = nx.dag_longest_path_length(H)
            else:
                # Approximate: DAG on SCC condensation
                C = nx.condensation(G)  # DAG of SCCs
                longest_path_len = nx.dag_longest_path_length(C)
        except Exception:
            longest_path_len = None

        # Average shortest path within reachable region from entry
        try:
            lengths = nx.single_source_shortest_path_length(G, entry_idx)
            avg_spl = (
                sum(lengths.values()) / max(len(lengths), 1)
                if lengths
                else None
            )
        except Exception:
            avg_spl = None

        # Branching / linearity
        non_exit_nodes = [n for n in G.nodes if G.out_degree(n) > 0]
        if non_exit_nodes:
            out_degrees = [G.out_degree(n) for n in non_exit_nodes]
            branching_mean = sum(out_degrees) / len(out_degrees)
            branching_max = max(out_degrees)
        else:
            branching_mean = 0.0
            branching_max = 0

        linear_blocks = [
            n
            for n in G.nodes
            if G.in_degree(n) == 1 and G.out_degree(n) == 1
        ]
        linear_fraction = len(linear_blocks) / len(G.nodes)

        # Dominance (requires entry)
        dom_depth: dict[int, int] = {}
        dom_frontier_sizes: dict[int, int] = {}
        dom_tree_height = None
        try:
            idom = nx.immediate_dominators(G, entry_idx)
            # Compute depths by walking to entry
            for n in G.nodes:
                d = 0
                cur = n
                while cur != entry_idx and cur in idom:
                    cur = idom[cur]
                    d += 1
                dom_depth[n] = d
            dom_tree_height = max(dom_depth.values()) if dom_depth else None

            # Dominance frontiers
            df = nx.dominance_frontiers(G, entry_idx)
            for n, frontier in df.items():
                dom_frontier_sizes[n] = len(frontier)
        except Exception:
            # Keep defaults (None/0)
            pass

        df_sizes = list(dom_frontier_sizes.values())
        df_mean = sum(df_sizes) / len(df_sizes) if df_sizes else 0.0
        df_max = max(df_sizes) if df_sizes else 0

        # Centralities on blocks
        try:
            bc = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        except Exception:
            bc = {n: 0.0 for n in G.nodes}
        try:
            closeness = nx.closeness_centrality(G)
        except Exception:
            closeness = {n: 0.0 for n in G.nodes}
        try:
            eig = nx.eigenvector_centrality(G.to_undirected(), max_iter=200)
        except Exception:
            eig = {n: 0.0 for n in G.nodes}

        bc_vals = list(bc.values())
        cfg_bc_max = max(bc_vals) if bc_vals else 0.0
        cfg_bc_mean = sum(bc_vals) / len(bc_vals) if bc_vals else 0.0
        eig_max = max(eig.values()) if eig else 0.0
        closeness_mean = (
            sum(closeness.values()) / len(closeness) if closeness else 0.0
        )

        # Loop metrics using SCCs
        loop_sccs = [s for s in sccs if len(s) > 1]
        loop_count = len(loop_sccs)
        # Very rough loop nesting: max number of SCCs on a path in condensation DAG
        try:
            C = nx.condensation(G)
            loop_scc_nodes = {i for i, comp in enumerate(sccs) if len(comp) > 1}
            # Longest path restricted to loop SCC nodes: approximation
            loop_depth = 0
            for n in loop_scc_nodes:
                lengths = nx.single_source_shortest_path_length(C, n)
                depth_here = max(
                    (d for m, d in lengths.items() if m in loop_scc_nodes), default=0
                )
                loop_depth = max(loop_depth, depth_here)
            loop_nesting_depth = loop_depth
        except Exception:
            loop_nesting_depth = None

        # Per-block metrics rows
        for idx, kind, indeg, outdeg in blocks:
            block_records.append(
                {
                    "function_goid_h128": fn,
                    "repo": repo,
                    "commit": commit,
                    "block_idx": idx,
                    "is_entry": idx == entry_idx,
                    "is_exit": idx == exit_idx,
                    "is_branch": outdeg > 1,
                    "is_join": indeg > 1,
                    "dom_depth": dom_depth.get(idx),
                    "dominates_exit": None,  # you can refine using dominance_frontiers
                    "bc_betweenness": bc.get(idx),
                    "bc_closeness": closeness.get(idx),
                    "bc_eigenvector": eig.get(idx),
                    "in_loop_scc": any(idx in s for s in loop_sccs),
                    "loop_header": False,  # optional extra logic
                    "loop_nesting_depth": None,  # could project loop_nesting_depth
                    "created_at": now,
                    "metrics_version": 1,
                }
            )

        fn_records.append(
            {
                "function_goid_h128": fn,
                "repo": repo,
                "commit": commit,
                "rel_path": rel_path,
                "module": module,
                "qualname": qualname,
                "cfg_block_count": G.number_of_nodes(),
                "cfg_edge_count": G.number_of_edges(),
                "cfg_entry_block_idx": entry_idx,
                "cfg_exit_block_idx": exit_idx,
                "cfg_is_dag": is_dag,
                "cfg_scc_count": scc_count,
                "cfg_has_cycles": has_cycles,
                "cfg_longest_path_len": longest_path_len,
                "cfg_avg_shortest_path_len": avg_spl,
                "cfg_branching_factor_mean": branching_mean,
                "cfg_branching_factor_max": branching_max,
                "cfg_linear_block_fraction": linear_fraction,
                "cfg_dom_tree_height": dom_tree_height,
                "cfg_dominance_frontier_size_mean": df_mean,
                "cfg_dominance_frontier_size_max": df_max,
                "cfg_loop_count": loop_count,
                "cfg_loop_nesting_depth_max": loop_nesting_depth,
                "cfg_bc_betweenness_max": cfg_bc_max,
                "cfg_bc_betweenness_mean": cfg_bc_mean,
                "cfg_bc_closeness_mean": closeness_mean,
                "cfg_bc_eigenvector_max": eig_max,
                "created_at": now,
                "metrics_version": 1,
            }
        )

    return (
        pd.DataFrame.from_records(block_records),
        pd.DataFrame.from_records(fn_records),
    )
```

---

### 3.3 Per‑function DFG metrics (including block metrics)

```python
def compute_dfg_metrics_for_repo(
    con: duckdb.DuckDBPyConnection,
    ctx: RepoContext,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dfg_by_fn = load_dfg_rows(con, ctx)

    fn_meta = con.execute(
        """
        SELECT function_goid_h128::BIGINT AS goid,
               repo, commit, rel_path, module, qualname
        FROM analytics.function_profile
        WHERE repo = ? AND commit = ?
        """,
        [ctx.repo, ctx.commit],
    ).fetchall()

    now = datetime.now(timezone.utc)
    block_records: list[dict] = []
    fn_records: list[dict] = []

    for goid, repo, commit, rel_path, module, qualname in fn_meta:
        fn = int(goid)
        edges = dfg_by_fn.get(fn, [])
        if not edges:
            continue

        G = nx.DiGraph()
        phi_edges = 0
        symbols: set[str] = set()

        for src, dst, src_sym, dst_sym, via_phi, use_kind in edges:
            G.add_edge(
                src,
                dst,
                src_symbol=src_sym,
                dst_symbol=dst_sym,
                via_phi=via_phi,
                use_kind=use_kind,
            )
            symbols.add(src_sym)
            symbols.add(dst_sym)
            if via_phi:
                phi_edges += 1

        # Per block degrees
        dfg_in_deg = dict(G.in_degree())
        dfg_out_deg = dict(G.out_degree())

        dfg_phi_in = {n: 0 for n in G.nodes}
        dfg_phi_out = {n: 0 for n in G.nodes}
        for src, dst, data in G.edges(data=True):
            if data.get("via_phi"):
                dfg_phi_out[src] += 1
                dfg_phi_in[dst] += 1

        # Connectivity / SCCs
        comps = list(nx.weakly_connected_components(G))
        comp_count = len(comps)
        sccs = list(nx.strongly_connected_components(G))
        scc_count = len(sccs)
        has_cycles = any(len(s) > 1 for s in sccs)

        # Path metrics
        try:
            if nx.is_directed_acyclic_graph(G):
                longest_chain = nx.dag_longest_path_length(G)
            else:
                C = nx.condensation(G)
                longest_chain = nx.dag_longest_path_length(C)
        except Exception:
            longest_chain = None

        # Approx avg shortest path over all nodes
        try:
            lengths_all: list[int] = []
            for n in G.nodes:
                lengths = nx.single_source_shortest_path_length(G, n)
                lengths_all.extend(lengths.values())
            avg_spl = (
                sum(lengths_all) / len(lengths_all)
                if lengths_all
                else None
            )
        except Exception:
            avg_spl = None

        # Centralities
        try:
            bc = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        except Exception:
            bc = {n: 0.0 for n in G.nodes}
        try:
            eig = nx.eigenvector_centrality(G.to_undirected(), max_iter=200)
        except Exception:
            eig = {n: 0.0 for n in G.nodes}

        bc_vals = list(bc.values())
        bc_max = max(bc_vals) if bc_vals else 0.0
        bc_mean = sum(bc_vals) / len(bc_vals) if bc_vals else 0.0
        eig_max = max(eig.values()) if eig else 0.0

        # Degree aggregates
        indeg_vals = list(dfg_in_deg.values())
        outdeg_vals = list(dfg_out_deg.values())
        avg_in = sum(indeg_vals) / len(indeg_vals) if indeg_vals else 0.0
        avg_out = sum(outdeg_vals) / len(outdeg_vals) if outdeg_vals else 0.0
        max_in = max(indeg_vals) if indeg_vals else 0
        max_out = max(outdeg_vals) if outdeg_vals else 0
        branchy_fraction = (
            sum(1 for v in outdeg_vals if v > 1) / len(outdeg_vals)
            if outdeg_vals
            else 0.0
        )

        # Per-block rows
        for n in G.nodes:
            block_records.append(
                {
                    "function_goid_h128": fn,
                    "repo": repo,
                    "commit": commit,
                    "block_idx": n,
                    "dfg_in_degree": dfg_in_deg.get(n, 0),
                    "dfg_out_degree": dfg_out_deg.get(n, 0),
                    "dfg_phi_in_degree": dfg_phi_in.get(n, 0),
                    "dfg_phi_out_degree": dfg_phi_out.get(n, 0),
                    "dfg_bc_betweenness": bc.get(n),
                    "dfg_bc_closeness": None,  # optional
                    "dfg_bc_eigenvector": eig.get(n),
                    "dfg_in_chain": None,      # could mark nodes on some longest path
                    "dfg_in_scc": any(n in s for s in sccs if len(s) > 1),
                    "created_at": now,
                    "metrics_version": 1,
                }
            )

        fn_records.append(
            {
                "function_goid_h128": fn,
                "repo": repo,
                "commit": commit,
                "rel_path": rel_path,
                "module": module,
                "qualname": qualname,
                "dfg_block_count": G.number_of_nodes(),
                "dfg_edge_count": G.number_of_edges(),
                "dfg_phi_edge_count": phi_edges,
                "dfg_symbol_count": len(symbols),
                "dfg_component_count": comp_count,
                "dfg_scc_count": scc_count,
                "dfg_has_cycles": has_cycles,
                "dfg_longest_chain_len": longest_chain,
                "dfg_avg_shortest_path_len": avg_spl,
                "dfg_avg_in_degree": avg_in,
                "dfg_avg_out_degree": avg_out,
                "dfg_max_in_degree": max_in,
                "dfg_max_out_degree": max_out,
                "dfg_branchy_block_fraction": branchy_fraction,
                "dfg_bc_betweenness_max": bc_max,
                "dfg_bc_betweenness_mean": bc_mean,
                "dfg_bc_eigenvector_max": eig_max,
                "created_at": now,
                "metrics_version": 1,
            }
        )

    return (
        pd.DataFrame.from_records(block_records),
        pd.DataFrame.from_records(fn_records),
    )
```

---

### 3.4 Writing to DuckDB

```python
def upsert_cfg_metrics(
    con: duckdb.DuckDBPyConnection,
    ctx: RepoContext,
    df_blocks: pd.DataFrame,
    df_functions: pd.DataFrame,
) -> None:
    con.execute(
        "DELETE FROM analytics.cfg_block_metrics WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    )
    con.execute(
        "DELETE FROM analytics.cfg_function_metrics WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    )

    if not df_blocks.empty:
        con.register("tmp_cfg_block_metrics", df_blocks)
        con.execute(
            "INSERT INTO analytics.cfg_block_metrics SELECT * FROM tmp_cfg_block_metrics"
        )
        con.unregister("tmp_cfg_block_metrics")

    if not df_functions.empty:
        con.register("tmp_cfg_fn_metrics", df_functions)
        con.execute(
            "INSERT INTO analytics.cfg_function_metrics SELECT * FROM tmp_cfg_fn_metrics"
        )
        con.unregister("tmp_cfg_fn_metrics")


def upsert_dfg_metrics(
    con: duckdb.DuckDBPyConnection,
    ctx: RepoContext,
    df_blocks: pd.DataFrame,
    df_functions: pd.DataFrame,
) -> None:
    con.execute(
        "DELETE FROM analytics.dfg_block_metrics WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    )
    con.execute(
        "DELETE FROM analytics.dfg_function_metrics WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    )

    if not df_blocks.empty:
        con.register("tmp_dfg_block_metrics", df_blocks)
        con.execute(
            "INSERT INTO analytics.dfg_block_metrics SELECT * FROM tmp_dfg_block_metrics"
        )
        con.unregister("tmp_dfg_block_metrics")

    if not df_functions.empty:
        con.register("tmp_dfg_fn_metrics", df_functions)
        con.execute(
            "INSERT INTO analytics.dfg_function_metrics SELECT * FROM tmp_dfg_fn_metrics"
        )
        con.unregister("tmp_dfg_fn_metrics")
```

---

### 3.5 Prefect flow wiring

```python
from prefect import task, flow


@task
def cfg_dfg_metrics_task(db_path: str, repo: str, commit: str) -> None:
    con = duckdb.connect(db_path)
    ctx = RepoContext(repo=repo, commit=commit)

    cfg_block_df, cfg_fn_df = compute_cfg_metrics_for_repo(con, ctx)
    upsert_cfg_metrics(con, ctx, cfg_block_df, cfg_fn_df)

    dfg_block_df, dfg_fn_df = compute_dfg_metrics_for_repo(con, ctx)
    upsert_dfg_metrics(con, ctx, dfg_block_df, dfg_fn_df)

    con.close()


@flow
def enrich_cfg_dfg_metrics(db_path: str, repo: str, commit: str) -> None:
    cfg_dfg_metrics_task(db_path, repo, commit)
```

You’d hook `enrich_cfg_dfg_metrics` into the same orchestration stage that currently runs call‑graph metrics and profiles, since it depends only on `graph.cfg_*`, `graph.dfg_edges.*`, and `analytics.function_profile.*` being present.

---


 



