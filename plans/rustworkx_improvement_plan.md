
I dug through your rustworkx layer under `src/codeintel/build/graphs/` (especially `graphs/rx/*`, the graph loaders in `graphs/engine/views.py`, and the metric implementations in `graphs/compute/*`). You’re already doing several “best practice” things (notably the *stable ID ↔ node index* mapping in `RxGraphStore`, plus deterministic post-processing of algorithm outputs). The biggest opportunities I see are:

1. **Replace a few hand-rolled graph operations with first-class rustworkx APIs (notably `condensation()`).**
2. **Fix / harden Node-Link JSON serialization so graph caching is actually lossless.**
3. **Move graph construction to “batch/bulk add” patterns (and pre-aggregate edges) to reduce Python↔Rust crossings.**
4. **Consolidate duplicated “graph helper” logic (neighbor maps, component sorting, edge-weight extraction) into shared helpers.**
5. **Make weight semantics explicit (strength vs cost) and treat multithreading knobs as part of an “algorithm contract”.**

Below is a concrete, code-focused plan.

---

## 1) Eliminate bespoke condensation DAG code with `rustworkx.condensation()`

You currently reimplement condensation/quotient-graph construction in multiple places (e.g., `graphs/compute/metrics/cfg.py`, `graphs/compute/metrics/components.py`, `graphs/compute/metrics/statistics.py`, and `graphs/compute/imports.py`). As of rustworkx 0.17.1 there’s a built-in `condensation()` that works for both directed and undirected graphs and returns a condensed graph where nodes are SCCs (directed) or connected components (undirected); it also attaches a `node_map` mapping original node indices → condensed node indices.

### What to do

* Create **one** shared helper in `graphs/rx/algos.py` (or a new `graphs/rx/components.py`) that:

  * calls `rx.condensation(store.graph)`
  * exposes:

    * the condensation graph/store
    * a stable **membership mapping** from your domain IDs → condensed component ID
    * optional: a stable ordering/renumbering of components (so IDs don’t vary if rustworkx’s internal ordering changes)

### Why this is “best in class”

* Removes repeated bespoke code (and bugs).
* Uses a Rust implementation rather than Python loops.
* Lets you share one stable “component contract” (IDs, ordering, mapping) across CFG, import graphs, stats, etc.

### Important determinism note

The `node_map` object is a `NodeMap`, and rustworkx explicitly warns it’s **unordered** when iterated; if you ever iterate it directly, you must sort for stable order.
(If instead you only do `node_map[idx]` lookups for each original node index, you’re fine.)

---

## 2) Fix Node-Link JSON serialization (this is likely a correctness + cache hit-rate unlock)

Right now `src/codeintel/build/graphs/rx/serialization.py` uses:

* `rx.node_link_json(graph)` (no extractors)
* `rx.parse_node_link_json(data)` (no extractors)

But rustworkx’s Node-Link JSON **does not attempt to serialize arbitrary Python payloads** unless you provide `graph_attrs`, `node_attrs`, and `edge_attrs` functions that return `dict[str, str]`.
Your attached rustworkx deep dive explicitly calls out the same “data:null” pitfall and treats extractors as effectively mandatory for lossless serialization. 

### Why this matters in *your* codebase

Your `RxGraphStore.from_rx_graph()` expects node payloads shaped like your `encode_node_payload(...)` dict. If Node-Link JSON round-trips nodes/edges with `data: null`, the loaded graph won’t contain your `{"id": ..., "attrs": ...}` payloads, and store reconstruction (and cache reuse) becomes unreliable.

### Best-practice implementation pattern

Use Node-Link JSON as intended: store your full payload as a JSON string inside the required `dict[str,str]` envelope.

**Write side (sketch):**

```python
import json
import rustworkx as rx

def dumps_node_link_json(graph: rx.PyGraph | rx.PyDiGraph) -> str:
    return rx.node_link_json(
        graph,
        graph_attrs=lambda attrs: {str(k): str(v) for k, v in (attrs or {}).items()},
        node_attrs=lambda payload: {"payload": json.dumps(payload, separators=(",", ":"), sort_keys=True)},
        edge_attrs=lambda payload: {"payload": json.dumps(payload, separators=(",", ":"), sort_keys=True)},
    )
```

**Read side (sketch):**

```python
def loads_node_link_json(data: str) -> rx.PyGraph | rx.PyDiGraph:
    return rx.parse_node_link_json(
        data,
        graph_attrs=lambda d: dict(d),  # keep dict[str,str] or convert to your GraphMetadata type
        node_attrs=lambda d: json.loads(d["payload"]),
        edge_attrs=lambda d: json.loads(d["payload"]),
    )
```

This matches rustworkx’s documented contract: Node-Link JSON extractors/adapters are responsible for mapping payloads ↔ string dicts. 

---

## 3) Make graph building fast: batch adds + preaggregation (reduce Python↔Rust crossings)

Your loaders in `graphs/engine/views.py` build graphs by iterating row-by-row and calling `store.add_edge(...)` / `store.add_weighted_edge(...)` for every record. That is simple and correct, but it’s exactly the “Python↔Rust crossing” pattern rustworkx tells you to avoid at scale: prefer `add_nodes_from`/`add_edges_from` (and `node_count_hint`/`edge_count_hint`). 

### Best-in-class ingestion pattern for your use case

For each edge dataset (call graph edges, import edges, symbol use edges, etc.):

1. **Pre-aggregate edges in Arrow** (or DuckDB) to get unique `(src_id, dst_id, weight)` tuples:

   * For COUNT semantics: group by `(src, dst)` and count
   * For SUM semantics: group by and sum
2. Build a stable list of unique node IDs once.
3. Bulk add nodes: `NodeIndices = graph.add_nodes_from([...payloads...])`
4. Bulk add edges: `graph.add_edges_from([(src_idx, dst_idx, weight), ...])`

You can still keep `RxGraphStore` as the canonical interface; just change *how it populates* the internal rustworkx graph.

### Why this reduces bespoke code too

Once you have a single “edge-table → RxGraphStore” builder, your view methods like `load_call_graph`, `load_import_graph`, `load_cfg_graph`, `load_symbol_graph` can collapse to:

* get filtered table/record batches
* call `build_store_from_edges(table, ...)`

…and almost all the per-graph bespoke loops disappear.

---

## 4) Consolidate duplicated helpers (your code already *wants* a shared “graph contract” layer)

You have the right instincts in `graphs/rx/normalize.py` (`stable_key`, `sorted_mapping`, etc.). But there’s still noticeable duplication:

* Multiple “neighbor map” implementations (e.g., `graphs/rx/algos.py` vs `graphs/compute/metrics/community.py`)
* Multiple “component sorting key” helpers (`cfg.py`, `imports.py`, `community.py`)
* Multiple “edge weight extraction” helpers (some use `edge_list()+get_edge_data`, some zip `edge_list()` with `edges()`, etc.)

### Best-in-class move

Create a single `graphs/rx/helpers.py` (or promote existing private helpers in `rx/algos.py` to public) containing:

* `neighbors_by_index(store) -> dict[int, list[int]]` (sorted deterministically)
* `component_sort_key(store, component_indices) -> tuple[str,str]`
* `edge_weights_undirected(store) -> dict[tuple[int,int], float]`
* `iter_weighted_edges(store) -> Iterable[tuple[int,int,float]]`

Then:

* `community.py` uses the shared neighbor/component helpers (no private re-implementations).
* `cfg.py` and `imports.py` use the same component ordering logic, so “component id 0” means the same conceptual component across metrics.

This aligns with your attached guidance: treat determinism and normalization rules as part of a pinned “algorithm contract” layer. 

---

## 5) Make weight semantics explicit (strength vs cost) and route algorithms accordingly

Your current architecture *supports* weights (via `GraphWeightPolicy` and `edge_weight_from_payload`), but rustworkx’s algorithm surface is mixed:

* Some algorithms **ignore weights entirely** (by design). 
* `betweenness_centrality()` and `closeness_centrality()` are **unweighted by signature**. 
* Weighted closeness exists as a separate function and assumes weights represent **connection strength**, not distance/cost; if weights are costs, rustworkx calls not inverting them a “logical error”. 
* Weighted betweenness is still not provided in the public API (hence your custom implementation), and there’s an open issue about it.

### Best-in-class move

Add an explicit, small “weight semantics” enum (or config) used by any metric that consumes weights:

* `WeightSemantics.STRENGTH` (e.g., call counts, co-usage frequency)
* `WeightSemantics.COST` (e.g., latency, penalties, distances)

Then define one canonical conversion:

* `strength -> cost` (e.g., `cost = 1 / max(strength, eps)`)

This prevents future bugs where a developer passes call-count weights into a shortest-path algorithm and accidentally interprets “more calls” as “farther away”.

---

## 6) Treat parallelism knobs as API surface (determinism + ops)

rustworkx centrality algorithms are multithreaded with a `parallel_threshold`, and thread count can be controlled with `RAYON_NUM_THREADS`. 

Also, rustworkx itself notes that parallel execution can affect which “tie” you get (example called out for `longest_simple_path`). 

### Best-in-class move

* Introduce a single `GraphAlgoConfig` (maybe on `GraphRuntimeOptions`) that sets:

  * default `parallel_threshold` per algorithm family
  * desired `RAYON_NUM_THREADS` (or at least documents it clearly in your runtime)
* Normalize outputs deterministically anyway (you already do a lot of this via sorting and `stable_key`), and keep doing it.

---

## 7) Use rustworkx return types and iterators more directly (small wins, less code)

Rustworkx returns specialized containers (e.g., “read-only dict-like” centrality mappings). Your attached guide recommends normalizing these immediately for contracts/serialization (e.g., `dict(rx.pagerank(...))`). 

Similarly, when you need edge endpoints + payloads together, prefer the API that returns them together (e.g., `weighted_edge_list()`), rather than `edge_list()` + `get_edge_data()` loops, which are slower and noisier. 

This is mostly a cleanup/refactor, but it’s very aligned with “streamlined and maintainable”.

---

## 8) Community detection: be honest about what rustworkx provides today

Your `detect_communities_louvain()` and `detect_communities_label_propagation()` are *not* actually Louvain / label propagation—they’re deterministic heuristics built around bridge removal and component splitting. That might be fine operationally, but I’d rename them to match reality unless you intend to match the true algorithms.

Also: rustworkx does not currently ship Louvain or label propagation community detection in its main algorithm set; there are open issues requesting them.

**Best-in-class approach:**

* Either:

  * Rename to something like `detect_communities_bridge_split(...)`, keep it, document it, and benchmark it.
* Or:

  * Pull in a dedicated community lib (NetworkX’s implementations, `python-louvain`, Leiden, etc.) and treat rustworkx as your base graph + conversion source.

---

## 9) Add the “micro-contract” tests that keep rustworkx codebases healthy

Your attached rustworkx doc includes exactly the sort of high-signal tests that prevent regressions:

* indices can be reused after deletion (don’t treat indices as stable IDs) 
* Node-Link JSON requires `dict[str,str]` extractors; invalid values should raise 
* negative/NaN weights should raise for shortest paths (expect `ValueError`) 

Adding these as small pytest fixtures will keep future refactors (especially your serialization + ingestion refactor) safe.

---

# If you want the “highest ROI” order of operations

1. **Fix Node-Link JSON (lossless) serialization** → unlock reliable caching & reproducibility. 
2. **Replace bespoke condensation code with `rx.condensation()`** in all current call sites.
3. **Implement batch graph construction** (preaggregate edges + `add_nodes_from`/`add_edges_from`) and update `views.py` loaders. 
4. **Consolidate helper duplication** (neighbors, component sort keys, edge-weight iterators).
5. **Codify weight semantics + parallelism knobs** (a tiny “algorithm contract” layer). 

---

If you want, I can draft a concrete patch plan (file-by-file) for:

* `graphs/rx/serialization.py` (lossless node-link JSON round-trip),
* a new `graphs/rx/condensation.py` helper that returns (condensed_store, membership_map),
* and a new `graphs/rx/build_from_edges.py` builder that your `views.py` loaders call.

Below is a **concrete, file-by-file patch plan** to implement the three deliverables:

1. **`graphs/rx/serialization.py`**: *lossless node-link JSON round-trip* (including graph attrs + node payloads + edge payloads)
2. **new `graphs/rx/condensation.py`**: a shared helper that returns **`(condensed_store, membership_map)`**
3. **new `graphs/rx/build_from_edges.py`**: a shared builder that your **`graphs/engine/views.py`** loaders call (reducing bespoke loops + enabling “bulk add edges”)

I’ll include the key integration edits (views + call sites) so this actually lands end-to-end.

---

## 1) Patch plan: `src/codeintel/build/graphs/rx/serialization.py`

### Why change it

Right now you call `rustworkx.node_link_json(graph)` and `rustworkx.parse_node_link_json(data)` **without** the `graph_attrs/node_attrs/edge_attrs` callbacks. Per rustworkx docs, those callbacks are how you get payloads into the `"data"` fields (and they must be `dict[str, str]`). ([Rustworkx][1])

Without callbacks, users observe `"data": null` for nodes/edges in node-link JSON—exactly the behavior described in rustworkx issue #1298. ([GitHub][2])

In your codebase, that means:

* disk graph cache JSON loses node payloads / edge weights
* `store_from_rx(loads_node_link_json(...))` fails or yields junk
* caches miss constantly

### Design goal

Make `dumps_node_link_json()` + `loads_node_link_json()` **lossless** for:

* graph attrs (your `apply_graph_metadata(...)` output included)
* node payload objects (your encoded `{id, attrs}` structure)
* edge payload objects (your numeric weights)

…and still comply with rustworkx’s requirement that the `data` field is a `dict[str, str]`. ([Rustworkx][1])

### Proposed encoding format (“CodeIntel envelope”)

Because rustworkx requires `dict[str, str]`, the simplest robust approach is:

* Put the *real payload* into **msgpack bytes**, then **base64** it into a string.
* Store it under one stable key, e.g. `"ci_b64_msgpack"`.

You already have reliable payload helpers:

* `codeintel.core.serialization.payload.encode_payload()` / `decode_payload()`

So you don’t need to invent a new serializer—just wrap it.

### Concrete changes in `serialization.py`

#### A) Add private constants + helpers

Add near top of file:

* `CI_DATA_KEY = "ci_b64_msgpack"`
* `_pack(obj) -> dict[str, str]`
* `_unpack(data: dict[str, str] | None) -> object | None`

Implementation sketch:

* `_pack(obj)`:

  * if obj is `None`, return `{}` (or `{CI_DATA_KEY: ""}`)
  * else `raw = encode_payload(obj)` → bytes
  * `b64 = base64.b64encode(raw).decode("ascii")`
  * return `{CI_DATA_KEY: b64}`

* `_unpack(data)`:

  * if `not data` or `CI_DATA_KEY not in data`: return `data` (back-compat/fallback)
  * decode base64 → bytes → `decode_payload(bytes)`

This yields:

* strict compliance (`dict[str, str]`)
* round-trip of nested dict/list/ints/floats/bools/None (and your sanitized behavior for out-of-range ints etc.)

#### B) Update `dumps_node_link_json(graph, require_metadata=...)`

Replace:

```py
return rx.node_link_json(graph)
```

with:

```py
return rx.node_link_json(
    graph,
    graph_attrs=lambda attrs: _pack(attrs),
    node_attrs=lambda node_payload: _pack(node_payload),
    edge_attrs=lambda edge_payload: _pack(edge_payload),
)
```

Rustworkx explicitly documents these hooks and the “dict[str,str]” requirement. ([Rustworkx][1])

Also keep your existing `require_metadata` behavior as-is (that part is good).

#### C) Update `loads_node_link_json(data: str)`

Replace:

```py
return rx.parse_node_link_json(data)
```

with:

```py
return rx.parse_node_link_json(
    data,
    graph_attrs=lambda d: _unpack(d) or {},
    node_attrs=lambda d: _unpack(d),
    edge_attrs=lambda d: _unpack(d),
)
```

Notes:

* For `graph_attrs`, ensure you return a dict (not `None`) so `graph.attrs` stays a mapping.
* For node/edge, return whatever you packed (dict payload, float weight, etc.)

#### D) Ensure file helpers use the new logic

* `write_node_link_json(path, graph, ...)` should call the updated `rx.node_link_json(..., path=...)` **with the same callbacks**.
* `read_node_link_json(path)` can continue reading text then calling `loads_node_link_json`.

#### E) Backward compatibility (optional but recommended)

In `_unpack()`, if `CI_DATA_KEY` is missing:

* return `data` untouched (so you can still parse JSON from other sources)
* but your `store_from_rx` path expects your node payload shape; old cached JSON that had `data: null` will still be unusable—but your runtime already treats cache failures as a miss and rebuilds.

#### F) Add targeted tests (pytest)

Add a new test module:
`src/codeintel/build/graphs/rx/tests/test_serialization.py` (or `tests/…` depending on your layout)

Minimum tests:

1. **Round-trip preserves node IDs + attrs + edge weights**

   * build a small `RxGraphStore.directed()`
   * set node attrs via `store.set_node_attrs(...)`
   * add weighted edges
   * `apply_graph_metadata(store.graph, ...)`
   * `payload = dumps_node_link_json(store.graph, require_metadata=True)`
   * `g2 = loads_node_link_json(payload)`
   * `store2 = store_from_rx(g2, weight_policy=...)`
   * assert:

     * same `node_ids()` set
     * same `node_attrs` for a couple nodes
     * same edge weights for known pairs
     * `metadata_from_graph(g2)` matches

2. **Graph attrs survive serialization**

   * ensure `metadata_from_graph(g2)` is non-null and contains expected fields

This directly validates the rustworkx callback-based serialization behavior that otherwise yields null data. ([GitHub][2])

---

## 2) Patch plan: new `src/codeintel/build/graphs/rx/condensation.py`

### Why change it

You currently implement SCC condensation logic in multiple places:

* `graphs/compute/metrics/cfg.py` (`_condensation_store`)
* `graphs/compute/metrics/statistics.py` (`_condensation_graph`)
* `graphs/compute/metrics/components.py` (`find_strongly_connected`)
* `graphs/compute/imports.py` (`compute_scc`)

This is classic “same algorithm, reimplemented 4 ways”.

### What “best-in-class rustworkx” looks like here

Rustworkx now provides a universal `condensation()` function (directed: SCC quotient DAG; undirected: connected-component quotient). The **returned graph includes a `node_map` mapping original node indices → condensed node indices**. ([Rustworkx][3])

Even if you still want **stable component numbering** (you do, based on current sorting logic), you can centralize that policy in one helper.

### Proposed public API

Create:

```py
def condense_store(
    store: RxGraphStore,
    *,
    stable: bool = True,
    count_intercomponent_edges: bool = True,
) -> tuple[RxGraphStore, dict[Hashable, int]]:
    ...
```

Return:

* `condensed_store`: nodes are component IDs (`int` 0..k-1)
* `membership_map`: `{original_node_id -> component_id}`

### Implementation plan inside `condensation.py`

#### A) Compute components deterministically

For **directed** graphs:

* call `rx.strongly_connected_components(directed_graph)` (already in use)
* convert to `list[set[int]]`
* if `stable=True`:

  * sort components by “smallest member node-id stable_key”
  * (this reproduces your existing behavior)

For **undirected** graphs:

* use `rx.connected_components(graph)` (if you need it; optional)

#### B) Build `membership_map` in *node-id space*

Use `store.index_to_id[idx]` and assign comp_id.

This is what your callers want; they shouldn’t care about rustworkx node indices.

#### C) Build the condensed store

Two options:

**Option 1 (keep current semantics exactly):**

* Create `RxGraphStore.directed(node_hint=n_components, ...)`
* Ensure nodes 0..n-1 exist
* Iterate original edges:

  * map `(u_idx, v_idx)` → `(cu, cv)`
  * if `cu != cv`: `condensed_store.add_weighted_edge(cu, cv, weight=1.0)`
* This preserves your current meaning of condensed edge weight = **count of inter-component edges**.

**Option 2 (use rustworkx.condensation for structure, then re-weight):**

* Use `rx.condensation(store.graph, sccs=sorted_components)` to get the quotient DAG structure (Tarjan-based). ([Rustworkx][3])
* Still iterate original edges to accumulate edge weights, but you can skip `has_edge` checks because the quotient already knows which edges exist.

Given you already do (Option 1) everywhere, I’d implement Option 1 first (lowest risk), and optionally refactor to Option 2 later.

#### D) Add docstring + type clarity

* Explicitly document determinism guarantees when `stable=True`.
* Document weight semantics (“count_intercomponent_edges”).

#### E) Tests

Add `src/codeintel/build/graphs/rx/tests/test_condensation.py`:

* Build a small directed graph with a known SCC (cycle of 3) + edges out.
* Assert:

  * membership_map groups cycle nodes into same component id
  * condensed_store is acyclic
  * condensed edge weights match expected counts

### Update call sites to use the helper (remove duplication)

Make follow-on edits:

1. `graphs/compute/metrics/cfg.py`

   * replace `_condensation_store(store)` with `condense_store(store)[0]` (and use membership_map if needed later)

2. `graphs/compute/metrics/statistics.py`

   * replace `_condensation_graph(store)` with the helper
   * remove duplicated sorting + edge counting logic

3. `graphs/compute/metrics/components.py`

   * inside `find_strongly_connected`, call helper and (if needed) reconstruct the `components` output from `membership_map` (or add a small helper to invert membership)

4. `graphs/compute/imports.py`

   * `compute_scc(...)` can become:

     * build store
     * `_, membership = condense_store(store)`
     * return membership (module → comp_id)

This one change will delete a lot of bespoke SCC “glue code”.

---

## 3) Patch plan: new `src/codeintel/build/graphs/rx/build_from_edges.py`

### Why change it

Your `graphs/engine/views.py` loaders repeat the same patterns:

* initialize store
* iterate Arrow tuples row-by-row
* normalize IDs
* `add_weighted_edge(...)`
* (sometimes) set node attrs

This is:

* duplicated (call graph, import graph, symbol graphs, config bipartite)
* slower than it needs to be (per-edge Python calls + `has_edge`/`get_edge_data` hot loop)

### “Best-in-class rustworkx” for building graphs

Rustworkx graph objects support **bulk edge insertion** (`add_edges_from`, `extend_from_edge_list`, etc.). The big performance win is:

1. **aggregate** duplicates in Python (or Arrow) once
2. **bulk add** unique edges into the graph
3. avoid `has_edge/get_edge_data/update_edge` per row

### Proposed API

Create a small config dataclass + two entry points:

```py
@dataclass(frozen=True, slots=True)
class EdgeBuildSpec:
    directed: bool
    src_fn: Callable[[object], Hashable] = default
    dst_fn: Callable[[object], Hashable] = default
    weight_fn: Callable[[object], float] = default_weight
    node_attrs_fn: Callable[[Hashable, str], dict[str, object]] | None = None
    # where side is "src"/"dst" for bipartite tagging

def build_store_from_edge_tuples(
    edges: Iterable[tuple[object, object] | tuple[object, object, object]],
    *,
    spec: EdgeBuildSpec,
    stable_nodes: bool = True,
    aggregate_edges: bool = True,
) -> RxGraphStore: ...
```

Optional “Arrow-native” entry point (phase 2):

```py
def build_store_from_reader(
    reader: pa.RecordBatchReader,
    *,
    src_col: str,
    dst_col: str,
    weight_col: str | None = None,
    spec: EdgeBuildSpec,
    ...
) -> RxGraphStore: ...
```

### Implementation details (v1: tuples-based, low-risk)

#### A) Single pass: collect + aggregate

* `nodes: set[Hashable]`
* `weights: dict[tuple[Hashable, Hashable], float]` (or `Counter` if always 1.0)

For each row:

* `src = spec.src_fn(row[0])`
* `dst = spec.dst_fn(row[1])`
* `w = spec.weight_fn(row[2]) if len(row) > 2 else 1.0`
* `weights[(src, dst)] += w` (if aggregate)
* add `src,dst` to nodes
* optional node attrs tagging:

  * if `spec.node_attrs_fn`: store a side-aware attrs mapping per node id (or call `store.set_node_attrs` after store exists)

#### B) Deterministic node indexing (important)

If `stable_nodes=True`, do:

* `node_list = sorted(nodes, key=stable_key)` (you already have `stable_key` in rx/normalize.py)

Then:

* `store = RxGraphStore.directed(node_hint=len(node_list), edge_hint=len(weights))` (or undirected)
* `for node_id in node_list: store.ensure_node(node_id)`

#### C) Bulk add edges

Convert to rustworkx index edges once:

* `edge_triples = [(store.id_to_index[s], store.id_to_index[d], store.weight_policy.normalize_weight(w)) ...]`
* `store.graph.add_edges_from(edge_triples)`
* then `store._touch()` once (yes it’s private; acceptable inside the rx package, but you can also add a small public `touch()` if you prefer)

This avoids per-edge `has_edge/get_edge_data/update_edge` overhead.

### Update `graphs/engine/views.py` loaders to call it

File: `src/codeintel/build/graphs/engine/views.py`

#### A) `load_call_graph()`

Current loop:

* iter tuples
* stable_decimal_id per row
* add_weighted_edge(..., 1.0)

Replace with:

* `edges = GraphViewFactory.iter_tuples(reader, columns=["caller_id", "callee_id"])`
* `spec = EdgeBuildSpec(directed=True, src_fn=stable_decimal_id, dst_fn=stable_decimal_id)`
* `return build_store_from_edge_tuples(edges, spec=spec, aggregate_edges=True, stable_nodes=True)`

#### B) `load_import_graph()`

Replace similarly, with `directed=True`, `src_fn=stable_decimal_id`, `dst_fn=stable_decimal_id`

#### C) `load_symbol_module_graph()` and `load_symbol_function_graph()`

* `directed=False`
* `src_fn=str`, `dst_fn=str`

#### D) `load_config_module_bipartite()`

Use node attrs tagging via `node_attrs_fn`:

* `node_attrs_fn(node_id, side)` returns:

  * `{"partition": "config_key"}` for side `"src"`
  * `{"partition": "module_path"}` for side `"dst"`

Then after building the store, set node attrs:

* either during ensure_node (requires builder to apply attrs as it adds nodes)
* or in a second pass over nodes collected during scan

### Tests

Add `src/codeintel/build/graphs/rx/tests/test_build_from_edges.py`:

* directed: duplicated edges collapse to single edge with summed weight
* stable_nodes=True yields deterministic `store.node_ids()` order
* node_attrs_fn applies partitions correctly for bipartite

### Export from `graphs/rx/__init__.py`

Update:

* add `condense_store`
* add `EdgeBuildSpec`, `build_store_from_edge_tuples`, (optionally) `build_store_from_reader`

---

## 4) Small but necessary follow-ups

### Update `src/codeintel/build/graphs/rx/__init__.py`

Add exports so callers don’t import deep paths.

### Run-time impact / compatibility

* **Graph cache JSON format changes**, but this is good: your caches will finally be usable.
* Old cache files that were written with null `"data"` won’t load into `RxGraphStore` (they never really did). They’ll be treated as misses and overwritten.

### Why this is “best-in-class rustworkx”

* Serialization now uses rustworkx’s **intended** node-link mechanism (callbacks) rather than assuming payloads serialize automatically. ([GitHub][2])
* Condensation logic is centralized and aligned with rustworkx’s modern condensation support and node index mapping conventions. ([Rustworkx][3])
* Graph building uses **bulk edge insertion** patterns and isolates “ID normalization + aggregation policy” into a single reusable builder.

---



[1]: https://www.rustworkx.org/apiref/rustworkx.node_link_json.html?utm_source=chatgpt.com "rustworkx.node_link_json"
[2]: https://github.com/Qiskit/rustworkx/issues/1298 "Serializing data with node_link_json returns \"data\": null · Issue #1298 · Qiskit/rustworkx · GitHub"
[3]: https://www.rustworkx.org/release_notes.html?utm_source=chatgpt.com "Release Notes - rustworkx 0.17.1"

Below is a **detailed, file-by-file patch plan** for the remaining “highest ROI” items (beyond the three helpers we already planned: lossless node-link JSON, `graphs/rx/condensation.py`, and `graphs/rx/build_from_edges.py`).

I’m structuring this as a sequence of small PRs so you can land it safely, keep outputs stable, and make performance wins measurable.

---

## PR 4 — Replace bespoke condensation code with `rx.condensation()` at all call sites

**Goal**

* Eliminate repeated SCC → condensation implementations.
* Standardize on **rustworkx’s `condensation()`** (added in rustworkx 0.17.1) and your new helper that returns:

  * `condensed_store: RxGraphStore`
  * `membership_map: dict[node_id, component_id]`
    (release notes: condensation + node_map attribute) ([Rustworkx][1])
* Preserve your current semantics: stable component ordering + stable membership IDs.

### Files to change

### 1) `src/codeintel/build/graphs/compute/metrics/statistics.py`

**Current**

* `_condensation_graph()` builds SCCs manually and builds a `rx.PyDiGraph`.
* `compute_condensation_layer_count()` calls `_layer_count()` on that graph.

**Patch**

1. Replace `_condensation_graph()` with a call to your helper:

```python
from codeintel.build.graphs.rx.condensation import condensation_store  # new helper

def compute_condensation_layer_count(store: GraphStore) -> int:
    store = ensure_directed_store(store)
    condensed_store, _membership = condensation_store(store)
    return _layer_count(condensed_store.graph)
```

2. Delete the now-unneeded:

   * `_condensation_graph`
   * `_component_membership`
   * `_component_sort_key`
   * `_directed_graph` (this is duplicated elsewhere; we’ll consolidate in PR 6)

**Definition of done**

* Layer count matches old implementation for representative graphs.
* No change to output type/shape.

---

### 2) `src/codeintel/build/graphs/compute/metrics/cfg.py`

**Current**

* `_condensation_store()` manually computes SCCs and builds a new `RxGraphStore`.
* `compute_cfg_longest_path()` calls `rx.dag_longest_path_length()` on that store’s `.graph`.

**Patch**

1. Import your helper and replace:

```python
from codeintel.build.graphs.rx.condensation import condensation_store

def compute_cfg_longest_path(store: GraphStore) -> int:
    store = ensure_directed_store(store)
    condensed, _membership = condensation_store(store)
    return rx.dag_longest_path_length(condensed.graph)
```

2. Delete:

* `_condensation_store`
* `_component_membership`
* `_component_sort_key`

3. Replace local `_ensure_directed_store()` and `_directed_graph()` with the shared versions (PR 6), **or** if you want to keep PR 4 narrowly scoped:

* leave them for now, but remove their usage from condensation code paths.

**Definition of done**

* Longest path matches old implementation on same input graphs.
* No change in result definition.

---

### 3) `src/codeintel/build/graphs/compute/metrics/components.py`

**Current**

* `find_strongly_connected(..., compute_condensation=True)` builds:

  * `condensed_store` manually
  * `node_to_component` via a loop

**Patch**

1. Replace the custom condensation-building block with:

```python
from codeintel.build.graphs.rx.condensation import condensation_store

if compute_condensation:
    condensed_store, node_to_component = condensation_store(store)
else:
    condensed_store = None
    node_to_component = _component_membership(store, components)  # keep existing for now
```

2. Optional (recommended): even when `compute_condensation=False`, you can still compute membership from the helper *without* building the full store if you expose a lightweight helper mode:

* `condensation_membership(store) -> membership_map`
* But if you don’t want to expand helper scope, keep the existing membership function.

3. Remove the duplicated condensation-store construction logic.

**Definition of done**

* `StronglyConnectedComponents.components` ordering matches prior stable behavior.
* `node_to_component` mapping is stable/deterministic across runs.

---

### 4) (Optional but high ROI) `src/codeintel/build/graphs/compute/imports.py`

This isn’t using “condensation” directly, but it *is* reimplementing SCC + stable ordering + a layer DAG.

**Patch (optional, but I’d do it)**

1. Replace `compute_scc()` internals with:

* build `RxGraphStore`
* call `condensation_store(store)` to get stable membership ids
* invert membership into SCC groups

2. Replace `compute_layers()` to operate on the condensed DAG returned by `condensed_store`, rather than rebuilding adjacency sets manually.

If you decide to keep your current layer algorithm, you can still reduce sprawl by:

* using `condensed_store.graph.edge_list()` to build adjacency.

---

### Tests to add for PR 4

Create: `tests/test_rx_condensation_call_sites.py`

* Build a handful of small directed graphs with:

  * self-loops
  * multiple SCCs
  * parallel edges in source data (your store combines weights)
* Assert:

  * `compute_cfg_longest_path` same as before
  * `compute_condensation_layer_count` same as before
  * `find_strongly_connected(... compute_condensation=True)` returns the same component partition + stable ordering

---

## PR 5 — Batch graph construction in loaders: preaggregate edges + `add_nodes_from`/`add_edges_from`

**Goal**

* Replace edge-by-edge mutation (`has_edge` + `add_edge`) with **bulk edge insertion**.
* Reduce Python overhead in all parquet/arrow-driven loaders (biggest runtime win).
* Preserve weight semantics: because you construct graphs with `multigraph=False`, parallel edges are not allowed (duplicates require preaggregation). ([Rustworkx][2])
* Use rustworkx bulk APIs:

  * `add_nodes_from` returns created indices (lets you build id→index in one pass). ([Rustworkx][3])
  * `add_edges_from` adds edges from tuples of `(parent, child, obj)`. ([Rustworkx][4])

### Core implementation approach

Instead of rewriting all loaders to “collect everything then build once” (memory risk), implement a **streaming bulk inserter**:

* Accumulate edges in a dict `{(u_id, v_id): weight}`
* Periodically flush:

  * ensure any missing nodes (in bulk)
  * translate IDs → indices
  * `graph.add_edges_from([...])`
* Clear accumulator and continue

### Files to change

### 1) `src/codeintel/build/graphs/rx/build_from_edges.py` (already planned)

**Add: `BulkEdgeInserter` (or `RxBulkInserter`)**

Key features:

* Works with an existing `RxGraphStore`
* Accepts node ids + weight
* Applies `weight_policy.normalize_weight()`
* Applies `weight_policy.combine_weights()` on duplicates
* Periodically flushes

Skeleton:

```python
class BulkEdgeInserter(Generic[NodeIdT]):
    def __init__(self, store: RxGraphStore[NodeIdT], *, flush_every: int = 250_000):
        self.store = store
        self.flush_every = flush_every
        self._edges: dict[tuple[NodeIdT, NodeIdT], float] = {}
        self._touched_nodes: set[NodeIdT] = set()

    def add(self, u: NodeIdT, v: NodeIdT, w: float = 1.0) -> None:
        w = self.store.weight_policy.normalize_weight(w)
        self._touched_nodes.add(u); self._touched_nodes.add(v)
        key = (u, v) if self.store.is_directed else _undirected_key(u, v)
        prev = self._edges.get(key)
        self._edges[key] = w if prev is None else self.store.weight_policy.combine_weights(prev, w)

        if len(self._edges) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._edges:
            return

        # 1) add missing nodes in bulk
        new_nodes = [n for n in self._touched_nodes if n not in self.store.id_to_index]
        if new_nodes:
            new_nodes.sort(key=stable_key)
            payloads = [encode_node_payload(n, self.store.node_attrs.get(n)) for n in new_nodes]
            new_idxs = self.store.graph.add_nodes_from(payloads)
            for n, idx in zip(new_nodes, new_idxs):
                self.store.id_to_index[n] = idx
                self.store.index_to_id[idx] = n

        # 2) add edges in bulk
        edge_triples = [
            (self.store.id_to_index[u], self.store.id_to_index[v], w)
            for (u, v), w in self._edges.items()
        ]
        self.store.graph.add_edges_from(edge_triples)

        # 3) touch once
        self.store._touch()

        self._edges.clear()
        self._touched_nodes.clear()
```

**Important note**

* Your graphs are `multigraph=False` (parallel edges disabled), so your preaggregation is the mechanism that preserves “sum duplicate edges” semantics. ([Rustworkx][2])

---

### 2) `src/codeintel/build/graphs/engine/views.py`

Update each loader that currently does:

* loop record batches
* `add_weighted_edge(store, src, dst, w)`
* (often `has_edge` check inside the helper)

**Change to:**

* create `BulkEdgeInserter(store)`
* call `bulk.add(src, dst, w)` in the loop
* call `bulk.flush()` at the end (and maybe per-recordbatch flush for memory)

#### A) `load_call_graph()`

In `_load_edges_from_batches(...)`:

* replace direct `add_weighted_edge` calls with bulk inserter.

Also: keep `load_graph_nodes(...)` unchanged for now; it will update node attrs after edges load.

#### B) `load_import_graph()`

Same.

#### C) `load_symbol_module_graph()` / `load_symbol_function_graph()`

Same.

#### D) `_populate_config_graph()`

This is slightly different because one row can generate many edges.
Still works: call `bulk.add("config:...", module, 1.0)`.

**Definition of done**

* Graph outputs (nodes/edges/weights) match prior behavior.
* Loader runtime improves (expect large wins on call/import graphs).

---

### 3) (Nice follow-up) `src/codeintel/build/graphs/builders.py`

Optionally:

* add a `bulk_add_weighted_edge(...)` wrapper that calls bulk inserter for “non-arrow” call sites (analytics code that constructs graphs in Python loops).
* Helps you migrate remaining bespoke loops later with minimal diff.

---

### Tests/benchmarks for PR 5

Create:

* `tests/test_bulk_edge_inserter_equivalence.py`

Test strategy:

* Build the same `RxGraphStore` two ways:

  1. old `store.add_weighted_edge(...)` calls
  2. new `BulkEdgeInserter.add(...); flush()`
* Compare:

  * node id sets
  * edge sets keyed by `(u_id, v_id)`
  * weights per edge (exact match)

Optional microbenchmark (pytest-benchmark) if you want.

---

## PR 6 — Consolidate helper duplication (neighbors, component sort keys, edge-weight iterators)

**Goal**

* Remove the repeated “same helper” logic spread across:

  * `graphs/rx/algos.py`
  * `graphs/compute/metrics/{cfg,statistics,community,components}.py`
  * `graphs/compute/imports.py`
* Make it easy to write new metrics without copying boilerplate.

### What to standardize

1. **Component stable ordering**

   * You currently reimplement `_component_sort_key(...)` in multiple modules.

2. **Directed/undirected “graph view” helpers**

   * Multiple `_directed_graph(store)`/`_undirected_graph(store)` duplicates.
   * You already have `ensure_directed_store()` / `ensure_undirected_store()` in `graphs/rx/algos.py`.

3. **Fast edge iteration**

   * Many modules do `edge_list()` then `get_edge_data()` per edge (slow).
   * You already have an optimized pattern in `graphs/rx/algos.py` using `zip(graph.edge_list(), graph.edges())` to avoid repeated lookups.

4. **Neighbor map creation**

   * `community.py` duplicates logic already present privately in `rx/algos.py`.

### Files to change

### 1) New: `src/codeintel/build/graphs/rx/components.py`

Add canonical helpers:

```python
def component_sort_key(store: RxGraphStore, comp: set[int]) -> tuple:
    # stable: based on min stable_key of member node_ids
    return min((stable_key(store.index_to_id[i]),) for i in comp)

def sort_components(store: RxGraphStore, comps: Iterable[set[int]]) -> list[set[int]]:
    return sorted(comps, key=lambda c: component_sort_key(store, c))
```

Also add:

* `invert_membership_map(membership_map) -> list[list[node_id]]` in stable component-id order

---

### 2) New: `src/codeintel/build/graphs/rx/iterators.py`

Add efficient, reusable iterators:

```python
def iter_edge_payloads(graph: rx.PyDiGraph | rx.PyGraph):
    # yields (u_idx, v_idx, payload) without get_edge_data()
    for (u, v), payload in zip(graph.edge_list(), graph.edges()):
        yield u, v, payload
```

Add:

* `iter_edge_weights(store, *, nan_policy=None) -> Iterable[(u_idx, v_idx, float)]`
* `edge_weight_map(store, *, undirected_keys=False, transform=None) -> dict[...]`

This becomes the single source of truth used by metrics.

---

### 3) `src/codeintel/build/graphs/rx/algos.py`

**Patch**

* Replace private helpers with imports from the new shared modules, or:

  * promote them to public and re-export from `rx/algos.py` for convenience.

For example:

* `_edge_weight_map` → public `edge_weight_map(...)` backed by `iterators.edge_weight_map`
* `_neighbor_map` → public `neighbor_map(...)`

---

### 4) Update metrics modules to delete duplicates

#### A) `src/codeintel/build/graphs/compute/metrics/community.py`

* Remove its `_neighbor_map`
* Import from `codeintel.build.graphs.rx.algos` (or `rx/iterators.py`) instead
* Remove `_component_sort_key` and use `rx/components.py`

#### B) `cfg.py`, `statistics.py`, `components.py`, `imports.py`

* Remove their `_component_sort_key` implementations
* Replace with `rx.components.sort_components(...)`
* Replace local `_directed_graph` / `_undirected_graph` with:

  * `ensure_directed_store(store).graph` (and cast if needed)

**Definition of done**

* You delete 4–8 repeated helper blocks.
* All affected metrics behave identically.

---

### Tests for PR 6

Create:

* `tests/test_rx_iterators_and_component_sort.py`
* Assert:

  * `iter_edge_payloads` yields same endpoints/payloads as repeated `get_edge_data` for small graphs
  * component sorting is deterministic across node insertion order

---

## PR 7 — Codify weight semantics + parallelism knobs (tiny “algorithm contract” layer)

**Goal**

* Make it explicit *what* edge weights mean for each algorithm (strength vs distance vs count).
* Provide consistent, centralized knobs for rustworkx parallelism:

  * per-call `parallel_threshold`
  * environment `RAYON_NUM_THREADS` thread count (documented in rustworkx docs). ([Rustworkx][5])

### Why this matters for “best-in-class”

Right now, “weight” is just a column name. But different algorithms interpret weight differently:

* shortest-path-based metrics treat weight as **distance/cost**
* many graph analytics treat weight as **strength/capacity**
* rustworkx has clear parallel behavior controlled by `parallel_threshold` and `RAYON_NUM_THREADS`. ([Rustworkx][5])

### Files to change

### 1) New: `src/codeintel/build/graphs/rx/contracts.py`

Add two small concepts:

#### A) Weight semantics

```python
from dataclasses import dataclass
from typing import Literal, Callable

WeightSemantics = Literal["distance", "strength"]

@dataclass(frozen=True, slots=True)
class WeightContract:
    semantics: WeightSemantics = "distance"
    epsilon: float = 1e-12  # used when inverting

    def transform(self, w: float) -> float:
        if self.semantics == "distance":
            return w
        # strength -> distance
        return 1.0 / max(w, self.epsilon)
```

#### B) Parallelism config

```python
@dataclass(frozen=True, slots=True)
class ParallelismContract:
    parallel_threshold: int | None = None  # None => rustworkx default
    rayon_num_threads: int | None = None   # env var
```

Add a helper:

```python
def configure_rayon_threads(contract: ParallelismContract) -> None:
    if contract.rayon_num_threads is not None:
        os.environ["RAYON_NUM_THREADS"] = str(contract.rayon_num_threads)
```

> Note: rustworkx documents `parallel_threshold` and `RAYON_NUM_THREADS` for multithreaded centrality functions. ([Rustworkx][5])

---

### 2) `src/codeintel/build/graphs/rx/algos.py`

**Patch**

* Extend your existing option dataclasses to include contracts.

Example: `BetweennessOptions`

```python
@dataclass(frozen=True, slots=True)
class BetweennessOptions:
    normalized: bool = True
    endpoints: bool = False
    k: int | None = None
    seed: int = 42
    weight: str | None = None

    weight_contract: WeightContract = WeightContract("distance")
    parallel: ParallelismContract = ParallelismContract()
```

Then:

* when using rustworkx built-ins (`rx.digraph_betweenness_centrality`) pass:

  * `parallel_threshold=options.parallel.parallel_threshold` if not None
    (rustworkx docs describe this param and default behavior) ([Rustworkx][5])
* when using your custom weighted Brandes:

  * apply `weight_contract.transform(w)` when building the weight map

Concretely: update your edge weight map creation helpers to accept an optional transform callable.

---

### 3) `src/codeintel/build/graphs/runtime/context.py`

**Patch**
Add fields to `GraphMetricsOptions` (defaults preserve existing behavior):

```python
@dataclass(frozen=True, slots=True)
class GraphMetricsOptions:
    ...
    # rustworkx parallelism
    rx_parallel_threshold: int | None = None
    rx_rayon_num_threads: int | None = None

    # weight meaning (defaults keep existing outputs)
    betweenness_weight_semantics: Literal["distance", "strength"] = "distance"
```

Then in `GraphContext`, store them similarly.

---

### 4) `src/codeintel/build/graphs/runtime/runtime.py` (or earliest init point)

**Patch**
At runtime startup (before heavy graph metrics):

* call `configure_rayon_threads(...)` once, if configured

This prevents “hidden performance differences” between environments.

---

### 5) `src/codeintel/core/compute/centrality.py`

**Patch**
When creating options, thread through contract from context:

```python
options = BetweennessOptions(
    normalized=ctx.betweenness_normalize,
    k=ctx.betweenness_sample_k,
    weight=ctx.betweenness_weight,
    seed=ctx.betweenness_seed,
    weight_contract=WeightContract(ctx.betweenness_weight_semantics),
    parallel=ParallelismContract(
        parallel_threshold=ctx.rx_parallel_threshold,
        rayon_num_threads=ctx.rx_rayon_num_threads,
    ),
)
```

**No behavior change by default**, because:

* semantics stays `"distance"`
* `parallel_threshold=None` means rustworkx default (50 for these centrality functions) ([Rustworkx][5])

---

### Tests for PR 7

Create:

* `tests/test_rx_contracts.py`

Cases:

* `WeightContract("distance").transform(w) == w`
* `WeightContract("strength")` correctly inverts (with epsilon handling)
* passing `parallel_threshold` through to wrappers doesn’t change result values (small graphs)

---

## Recommended landing order (to keep risk low)

1. **PR 4** (condensation call-site refactor) — eliminates bespoke code, minimal runtime risk
2. **PR 5** (bulk loader insertion) — biggest performance win, still semantically identical
3. **PR 6** (helper consolidation) — reduces sprawl, makes future metrics cheap to add
4. **PR 7** (contracts) — gives you “best-in-class” configurability without changing defaults

---

If you want, I can also turn this patch plan into a **literal checklist of diffs** (imports to add/remove, exact functions to delete, and exact call-site replacements per file), but the above should be enough for a clean PR sequence and safe review.

[1]: https://www.rustworkx.org/release_notes.html?utm_source=chatgpt.com "Release Notes - rustworkx 0.17.1"
[2]: https://www.rustworkx.org/apiref/rustworkx.PyDiGraph.add_edge.html "rustworkx.PyDiGraph.add_edge - rustworkx 0.17.1"
[3]: https://www.rustworkx.org/apiref/rustworkx.PyDiGraph.add_nodes_from.html "rustworkx.PyDiGraph.add_nodes_from - rustworkx 0.17.1"
[4]: https://www.rustworkx.org/apiref/rustworkx.PyDiGraph.add_edges_from.html "rustworkx.PyDiGraph.add_edges_from - rustworkx 0.17.1"
[5]: https://www.rustworkx.org/apiref/rustworkx.digraph_betweenness_centrality.html?utm_source=chatgpt.com "rustworkx.digraph_betweenness_centrality"

Below is a **literal, file-by-file diff checklist** (imports to add/remove, functions to delete, and the exact call-site replacements) for the “highest ROI” rustworkx improvements we discussed—organized as a **clean PR sequence**.

Where I reference rustworkx APIs/behavior (serialization, condensation, parallelism), I’m grounding those details in the upstream docs/release notes. ([Rustworkx][1])

---

## PR 1 — Fix lossless Node-Link JSON serialization (unblocks caching & reproducibility)

### 1) Edit `src/codeintel/build/graphs/rx/serialization.py`

**Imports**

* **ADD**

  * `import json`
  * `from typing import Any`
* **KEEP** existing imports (`rustworkx as rx`, `Path`, etc.)
* **REMOVE** nothing (unless any now-unused imports appear after refactor)

**Add these new private helpers (new code)**

* `def _json_dumps(obj: Any) -> str:` (stable JSON: `sort_keys=True`, compact separators)
* `def _json_loads(s: str) -> Any:`
* `def _encode_str_dict(obj: Any) -> dict[str, str]:`

  * returns `{"payload": _json_dumps(obj)}`
* `def _decode_str_dict(d: dict[str, str]) -> Any:`

  * reads `d["payload"]` and JSON-decodes it

**Change `dumps_node_link_json()`**

* **REPLACE**:

  * `rx.node_link_json(graph)`
* **WITH**:

  * `rx.node_link_json(graph, graph_attrs=..., node_attrs=..., edge_attrs=...)`
* Concrete wiring:

  * `graph_attrs` callable: takes `graph.attrs` and returns `dict[str,str]` (values must be strings) ([Rustworkx][1])
  * `node_attrs` callable: takes *node payload* and returns `dict[str,str]` ([Rustworkx][1])
  * `edge_attrs` callable: takes *edge payload* and returns `dict[str,str]` ([Rustworkx][1])

**Recommended encoding**

* Graph attrs:

  * For each `k,v` in `graph.attrs`: stringify via `str(v)` **or** JSON-stringify via `_json_dumps(v)` (future-proof if attrs become non-strings)
* Node payload:

  * Your payload is already dict-like (`encode_node_payload` → `{"id": ..., "attrs": ...}`), but rustworkx requires string values; store it under a single string key:

    * `return {"payload": _json_dumps(payload)}`
* Edge payload:

  * Your edges store the numeric weight (float/int). Store:

    * `return {"weight": str(payload)}`
  * Or store `"payload": _json_dumps(payload)` if you want symmetry.

**Why this exact change**

* `node_link_json()` only serializes node/edge “data” via the callables, and those callables must return `dict[str,str]`. ([Rustworkx][1])
* Without supplying these, it’s easy to end up with missing/`null` payloads in output (and this has been a known failure mode). ([GitHub][2])

**Change `loads_node_link_json()`**

* **REPLACE**:

  * `rx.parse_node_link_json(payload)`
* **WITH**:

  * `rx.parse_node_link_json(payload, graph_attrs=..., node_attrs=..., edge_attrs=...)`
* Decoder callables:

  * `graph_attrs`: receives `dict[str,str]` → returns `dict[str,object]` for `graph.attrs`
  * `node_attrs`: receives `dict[str,str]` → returns the original node payload (dict)
  * `edge_attrs`: receives `dict[str,str]` → returns numeric weight (float)

**Keep the public API the same**

* Keep: `read_node_link_json()`, `write_node_link_json()` signatures unchanged.
* Ensure `require_metadata` still works.

---

### 2) Edit `src/codeintel/build/graphs/runtime/runtime.py`

**Bump cache version (strongly recommended)**

* **CHANGE**:

  * `GRAPH_CACHE_VERSION = "v4"`
* **TO**:

  * `GRAPH_CACHE_VERSION = "v5"`
* Rationale: old caches (if any) written with missing node/edge payloads should be treated as invalid; version bump prevents ambiguous mixed-format reuse.

No other call-site changes required if runtime already calls `dumps_node_link_json()` / `loads_node_link_json()`.

---

### 3) Add tests (if you want PRs to be “safe by default”)

**ADD** `tests/test_rx_node_link_roundtrip.py` (or `tests/graphs/rx/test_serialization.py`)

Test assertions:

* Build a tiny `RxGraphStore` with:

  * at least 2 nodes with non-empty attrs
  * at least 1 edge with weight ≠ 1
  * metadata applied (`apply_graph_metadata`)
* Serialize → parse → `RxGraphStore.from_rx_graph()`
* Assert:

  * node ids preserved
  * node attrs preserved
  * edge weights preserved
  * graph attrs/metadata preserved

---

## PR 2 — Replace bespoke condensation code with `rx.condensation()` everywhere

rustworkx provides `condensation()` which returns a condensed graph plus a `node_map` mapping each original node index → condensed node index.

### 1) ADD `src/codeintel/build/graphs/rx/condensation.py`

**New module exports**

* `def condense_store(store: RxGraphStore, *, deterministic: bool = True) -> tuple[RxGraphStore, dict[Any, int]]:`

**Implementation requirements (contract)**

* Ensure directed input:

  * If `not store.is_directed`, call `store.as_directed()`
* Call:

  * `condensed_rx = rx.condensation(directed_store.graph)`
* Extract mapping:

  * `node_map = condensed_rx.node_map` (per release notes)
* Build a stable `membership_map: dict[node_id, component_id]`

  * Convert original node indices → node ids using `store.index_to_id`
  * If `deterministic=True`, **relabel component ids** by stable representative (ex: smallest `stable_key(node_id)` in each SCC)
* Return:

  * a condensed `RxGraphStore` with nodes = component ids and edges = condensation edges
  * the stable `membership_map`

---

### 2) Edit `src/codeintel/build/graphs/compute/metrics/statistics.py`

**DELETE these functions**

* `def _condensation_graph(...)`
* `def _component_membership(...)`

(Confirm they’re not imported elsewhere; they’re private and used locally.)

**Update call site**

* In `compute_condensation_layer_count()`:

  * **REPLACE**:

    * `condensed = _condensation_graph(store)`
  * **WITH** (either option):

    * Option A (use helper):

      * `condensed_store, _ = condense_store(store, deterministic=False)`
      * `layer_count = _layer_count(condensed_store.graph)`
    * Option B (direct rustworkx call, simplest):

      * `condensed = rx.condensation(_directed_graph(store))`
      * `layer_count = _layer_count(condensed)`

(For “layer count” determinism doesn’t matter.)

---

### 3) Edit `src/codeintel/build/graphs/compute/metrics/components.py`

**Imports**

* **ADD**:

  * `from codeintel.build.graphs.rx.condensation import condense_store`
* **REMOVE** anything that becomes unused after deleting bespoke condensation helpers.

**Replace condensation construction inside `find_strongly_connected()`**

* **FIND** the block that currently:

  * constructs `condensed_store = RxGraphStore.directed(...)`
  * loops edges and adds via `add_weighted_edge`
  * builds `membership` and `layers`
* **REPLACE** with:

  * `condensed_store, membership = condense_store(store, deterministic=True)`
  * `layers = rx.topological_generations(condensed_store.graph)` (existing approach)
  * Build your per-node `layer_by_node` using `membership[node_id]`

**Optional cleanups (can be PR 4 instead)**

* `_component_sort_key`, `_sort_components` duplication can be removed later; don’t mix if you want small PRs.

---

### 4) Edit `src/codeintel/build/graphs/compute/metrics/cfg.py`

**Imports**

* **ADD**:

  * `from codeintel.build.graphs.rx.condensation import condense_store`
  * `from codeintel.build.graphs.rx.algos import ensure_directed_store`
* **REMOVE** now-unused:

  * `stable_key` if only used by deleted helpers
  * `edge_weight_from_payload` if only used by deleted helpers
  * local `rustworkx as rx` stays (still used)

**DELETE these helper functions**

* `_ensure_directed_store` (you already have `ensure_directed_store`)
* `_directed_graph`
* `_component_sort_key`
* `_sorted_components`
* `_condensation_store`

**Call-site replacement**

* In `compute_cfg_longest_path()` (and/or wherever you compute condensation depth):

  * **REPLACE**:

    * `directed = _ensure_directed_store(graph)`
    * `condensed = _condensation_store(directed)`
  * **WITH**:

    * `directed = ensure_directed_store(graph)`
    * `condensed, _ = condense_store(directed, deterministic=False)` (determinism not required for depth)
* Use `condensed.graph` wherever you previously used the directed graph.

---

### 5) Optional: Edit `src/codeintel/build/graphs/compute/imports.py`

If you want SCC computations to share the same deterministic SCC id scheme:

* **REPLACE** bespoke SCC ordering with:

  * `_, membership = condense_store(graph, deterministic=True)`
* Invert membership to rebuild `sccs: list[set[Any]]` if needed.

---

## PR 3 — Batch graph construction (preaggregate edges + `add_nodes_from`/`add_edges_from`) and update loaders

rustworkx provides `add_nodes_from()` for bulk node insertion (returns created indices). ([Rustworkx][3])
(And `add_edges_from()` similarly exists for bulk edges.)

### 1) ADD `src/codeintel/build/graphs/rx/build_from_edges.py`

**New APIs**

* `def preaggregate_edges(edges: Iterable[tuple[Any, Any]], *, directed: bool, combine: Callable[[float, float], float]) -> dict[tuple[Any, Any], float]:`

  * Uses dict accumulation, calling `combine(prev, 1.0)` for each raw edge
  * Canonicalize undirected endpoints using `stable_key()` ordering
* `def build_store_from_edge_weights(edge_weights: Mapping[tuple[Any, Any], float], *, directed: bool, node_attrs: Mapping[Any, dict[str, Any]] | None = None, weight_policy: GraphWeightPolicy = DEFAULT_WEIGHT_POLICY, numeric_policy: GraphNumericPolicy = DEFAULT_NUMERIC_POLICY, node_count_hint: int | None = None, edge_count_hint: int | None = None) -> RxGraphStore:`

  * Determine node set from:

    * keys of `node_attrs` union endpoints in `edge_weights`
  * Sort node ids deterministically
  * Build payload list via `encode_node_payload(node_id, attrs)`
  * Bulk add nodes via `graph.add_nodes_from(payloads)` ([Rustworkx][3])
  * Bulk add edges via `graph.add_edges_from([(src_idx, dst_idx, weight), ...])`

This becomes the single “best practice” entry point for graph construction.

---

### 2) Edit `src/codeintel/build/graphs/engine/views.py`

This is the big ROI: remove per-edge `store.add_edge()` calls.

**Imports**

* **REMOVE**:

  * `from codeintel.build.graphs.builders import add_weighted_edge`
* **ADD**:

  * `from codeintel.build.graphs.rx.build_from_edges import preaggregate_edges, build_store_from_edge_weights`

Now patch each view builder:

#### A) `call_graph_view(...)`

**REPLACE**:

* `store = RxGraphStore.directed(...)`
* the `for row in edge_view.iter_tuples(): add_weighted_edge(...)` loop

**WITH**:

* build `edges: list[tuple[Any, Any]]` (or stream into dict)
* `edge_weights = preaggregate_edges(edges, directed=True, combine=DEFAULT_WEIGHT_POLICY.combine_weights)`
* `store = build_store_from_edge_weights(edge_weights, directed=True, node_attrs=call_graph_nodes, weight_policy=weight_policy_for_kind(GraphKind.CALL_GRAPH))`

#### B) `import_graph_view(...)`

**REPLACE**:

* `store = RxGraphStore.directed(...)`
* the big loop that:

  * ensures nodes
  * sets attrs
  * `add_weighted_edge(...)`

**WITH**:

* `node_attrs: dict[module_id, attrs]` accumulated from rows
* `edges: list[(src_module, dst_module)]`
* `edge_weights = preaggregate_edges(... directed=True ...)`
* `store = build_store_from_edge_weights(edge_weights, directed=True, node_attrs=node_attrs, weight_policy=weight_policy_for_kind(GraphKind.IMPORT_GRAPH))`

(Keep your layer fallback logic; just apply it when populating `node_attrs`.)

#### C) `config_key_module_bipartite_view(...)`

Same pattern:

* Accumulate `node_attrs` for both (“key”, key) and (“module”, module)
* Accumulate edges list
* Preaggregate
* Build store in one call

#### D) `symbol_module_graph_view(...)`, `symbol_module_undirected_graph_view(...)`

Same:

* Accumulate node attrs by module id
* Accumulate edges between use_module and def_module (skip self-edge in undirected)
* Preaggregate (undirected canonicalization for undirected)
* Build store

#### E) `symbol_function_graph_view(...)`

Same:

* node attrs keyed by goid
* edges between use_goid and def_goid
* preaggregate directed
* build

---

### 3) Edit `src/codeintel/build/hamilton/native/analytics/config_graphs.py` (optional but consistent)

This file currently duplicates “loader-style” graph building.

**Imports**

* **ADD**:

  * `from codeintel.build.graphs.rx.build_from_edges import preaggregate_edges, build_store_from_edge_weights`
* **REMOVE**:

  * anything only used by `_add_call_graph_edges/_add_call_graph_nodes` if you delete them

**DELETE**

* `_add_call_graph_edges`
* `_add_call_graph_nodes`

**REPLACE** `_call_graph_from_frames(...)` internals

* Instead of constructing store then adding edges:

  * build `node_attrs` from `call_graph_nodes`
  * build `edges` from `call_graph_edges`
  * `edge_weights = preaggregate_edges(..., directed=True, combine=...)`
  * `return build_store_from_edge_weights(...)`

---

## PR 4 — Consolidate helper duplication (neighbors, component sort keys, edge-weight iteration)

This PR is “pay down the tax” after PR2/PR3.

### 1) Edit `src/codeintel/build/graphs/rx/store.py`

**ADD methods**

* `def iter_edge_payloads(self) -> Iterable[tuple[int, int, object]]:`

  * yield `(u, v, payload)` by zipping `graph.edge_list()` with `graph.edges()`
* `def iter_edge_weights(self, *, nan_policy: NanPolicy | None = None) -> Iterable[tuple[int, int, float]]:`

  * uses `edge_weight_from_payload(payload, nan_policy=...)` or your `weight_policy.normalize_weight(payload)`

This removes repeated `get_edge_data()` loops and normalizes how you read edge weights.

---

### 2) ADD `src/codeintel/build/graphs/rx/components_helpers.py` (or `graphs/rx/helpers.py`)

**ADD shared helpers**

* `def component_sort_key(store: RxGraphStore, component: set[int]) -> tuple:`

  * use representative id derived from `stable_key(store.index_to_id[min(component)])`
* `def sort_components(store, components) -> list[set[int]]:`

---

### 3) Edit `src/codeintel/build/graphs/compute/metrics/centrality.py`

**Replace bespoke edge weight iteration**

* In `neighbor_stats()`:

  * **REPLACE** the loop:

    * `for left, right in store.graph.edge_list(): store.graph.get_edge_data(left, right)`
  * **WITH**:

    * `for left, right, payload in store.iter_edge_payloads():`
    * weight via `store.weight_policy.normalize_weight(payload)` (or `edge_weight_from_payload`)

This makes neighbor stats consistent and faster.

---

### 4) Edit `src/codeintel/build/graphs/compute/metrics/components.py`, `cfg.py`, `statistics.py`

**Remove duplicated sort-key helpers**

* **DELETE** local:

  * `_component_sort_key`
  * `_sort_components` / `_sorted_components`
* **REPLACE** with import from your new helper module

**Remove `_directed_graph/_undirected_graph` duplication (optional)**

* Instead of local `cast` helpers:

  * use:

    * `directed = store if store.is_directed else store.as_directed()`
    * `graph = directed.graph`
  * and keep the casts close to use sites if mypy needs it.

---

## PR 5 — Codify weight semantics + parallelism knobs (“algorithm contract”)

rustworkx centrality functions support `parallel_threshold` and can use `RAYON_NUM_THREADS` to tune threading. ([Rustworkx][4])

### 1) Edit `src/codeintel/build/graphs/runtime/context.py`

**Add knobs to both GraphMetricsOptions + GraphContext**

* **ADD fields** (same defaults in both):

  * `rx_parallel_threshold: int = 50`
  * `rx_rayon_num_threads: int | None = None`

**Update context normalization**

* In `_base_context()`, `_normalize_context()`, etc.:

  * propagate these options from `GraphMetricsOptions` → `GraphContext`

**Update JSON read/write**

* If you serialize options to JSON anywhere in this module, ensure new fields are included (or safely ignored when missing).

---

### 2) Edit `src/codeintel/build/graphs/rx/algos.py`

**Betweenness**

* In `@dataclass BetweennessOptions`:

  * **ADD**:

    * `parallel_threshold: int = 50`
* In `_betweenness_builtin_by_id(...)`:

  * **PASS THROUGH**:

    * `parallel_threshold=resolved.parallel_threshold`
  * to:

    * `rx.digraph_betweenness_centrality(...)`
    * `rx.graph_betweenness_centrality(...)` ([Rustworkx][4])

**Closeness**

* In `closeness_by_id(...)`:

  * **ADD param**:

    * `parallel_threshold: int = 50`
  * **PASS THROUGH** to:

    * `rx.digraph_closeness_centrality(..., parallel_threshold=parallel_threshold)`
    * `rx.graph_closeness_centrality(..., parallel_threshold=parallel_threshold)` ([Rustworkx][5])
    * `rx.digraph_newman_weighted_closeness_centrality(..., parallel_threshold=parallel_threshold)`
    * `rx.graph_newman_weighted_closeness_centrality(..., parallel_threshold=parallel_threshold)` ([Rustworkx][6])

**Optional “contract layer” clarity**

* Add a tiny internal helper:

  * `def _weighted(weight: str | None) -> bool: return weight is not None`
* Replace repeated `if weight is not None` checks with `_weighted(weight)` for consistency.

**Thread count knob**

* (Optional but useful) Add:

  * `@contextmanager def rx_threads(num: int | None): ...`
  * temporarily set `os.environ["RAYON_NUM_THREADS"]`
* Use it in the few most expensive calls (closeness/betweenness). (Docs explicitly call out `RAYON_NUM_THREADS`.) ([Rustworkx][4])

---

### 3) Edit `src/codeintel/core/compute/centrality.py`

**Plumb ctx knobs into algos**

* Update:

  * `compute_betweenness(...)`:

    * when creating `BetweennessOptions`, set `parallel_threshold=ctx.rx_parallel_threshold` (or default if no ctx passed)
  * `compute_closeness(...)`:

    * pass `parallel_threshold=ctx.rx_parallel_threshold` into `closeness_by_id(...)`

---

### 4) Edit `src/codeintel/build/graphs/compute/metrics/centrality.py`

**Use ctx knobs**

* In `centrality_directed()` and `centrality_undirected()`:

  * pass `ctx.rx_parallel_threshold` down into `compute_closeness/compute_betweenness`

---

## Quick “ripgrep-driven” checklist (helps reviewers verify the PR)

Run these searches and ensure each yields the expected deltas:

1. **Serialization**

* `rg "rx\.node_link_json\(" src/codeintel/build/graphs/rx/serialization.py`

  * should show callables passed
* `rg "rx\.parse_node_link_json\(" src/codeintel/build/graphs/rx/serialization.py`

  * should show decoder callables

2. **Condensation**

* `rg "strongly_connected_components|_condensation_graph|_condensation_store" src/codeintel/build/graphs/compute/metrics`

  * should be largely removed/replaced with `condense_store(...)` / `rx.condensation(...)`

3. **Batch building**

* `rg "add_weighted_edge\(" src/codeintel/build/graphs/engine/views.py`

  * should be zero occurrences
* `rg "build_store_from_edge_weights|preaggregate_edges" src/codeintel/build/graphs/engine/views.py`

  * should appear in each graph view builder

4. **Duplicate helpers**

* `rg "def _component_sort_key|def _sorted_components|def _sort_components" src/codeintel/build/graphs/compute/metrics`

  * should shrink significantly after PR4

5. **Parallel knobs**

* `rg "parallel_threshold" src/codeintel/build/graphs/rx/algos.py`

  * should appear in betweenness/closeness calls

---

If you want, I can also provide **a “reviewer map”** (per PR: risk notes, expected perf wins, and a minimal benchmark harness to compare old vs new construction time and cache hit rate).

[1]: https://www.rustworkx.org/apiref/rustworkx.node_link_json.html "rustworkx.node_link_json - rustworkx 0.17.1"
[2]: https://github.com/Qiskit/rustworkx/issues/1298?utm_source=chatgpt.com "Serializing data with node_link_json returns \"data\": null"
[3]: https://www.rustworkx.org/apiref/rustworkx.PyDiGraph.add_nodes_from.html "rustworkx.PyDiGraph.add_nodes_from - rustworkx 0.17.1"
[4]: https://www.rustworkx.org/apiref/rustworkx.digraph_betweenness_centrality.html "rustworkx.digraph_betweenness_centrality - rustworkx 0.17.1"
[5]: https://www.rustworkx.org/apiref/rustworkx.closeness_centrality.html?utm_source=chatgpt.com "rustworkx.closeness_centrality"
[6]: https://www.rustworkx.org/apiref/rustworkx.graph_newman_weighted_closeness_centrality.html "rustworkx.graph_newman_weighted_closeness_centrality - rustworkx 0.17.1"
