The current architecture is still “repo → rich knowledge graph → docs views for LLMs,” but with one big new spine through the middle: **NetworkX‑powered graph analytics**. NetworkX now sits between the raw `graph.*` tables and the `analytics.*` / `docs.*` layers, and is used consistently across call graphs, imports, tests, CFGs, DFGs, and subsystems.

I’ll describe the system end‑to‑end, but I’ll call out where NetworkX now “owns” the logic.

---

## 1. Big picture

For a given `<repo, commit>` the system:

1. **Scans & indexes** the repo (AST/CFG/DFG, SCIP symbols, tests, coverage, config).
2. Assigns **GOIDs** to every entity and normalizes all sources into a DuckDB schema (`core.*`, `graph.*`, `analytics.*`). 
3. Builds **graphs**: call graph, import graph, symbol‑use graph, per‑function CFG/DFG, test⇄function bipartite graph.
4. Uses **NetworkX** to compute a huge set of graph‑theoretic metrics for each graph and writes them back into `analytics.*`.
5. Projects those datasets into **docs views** (`docs.*`) like `v_function_architecture`, `v_module_architecture`, and block‑level CFG/DFG views so an LLM can query a single row and see “everything interesting” about a function, module, or block.

Orchestration is handled by Prefect flows; all heavy lifting happens inside `src/codeintel` subpackages (ingestion, graphs, analytics, docs_export, etc.).

---

## 2. Core data model: GOIDs + DuckDB + graph overlays

### 2.1 GOIDs as universal keys

Every entity (module, function, method, class, basic block) gets a **128‑bit GOID** (`goid_h128`) and a human‑readable URN. This is built once from AST spans and stored in `core.goids`. 

Key tables:

* `core.goids`: one row per entity with repo, commit, path, kind, qualname, span.
* `core.goid_crosswalk`: links a GOID to all evidence sources that mention it (AST path, SCIP symbol, CST node, embedding chunk, etc.). 

All higher‑level tables refer to GOIDs, not raw paths; this is what lets graphs, coverage, tests, and config all line up cleanly.

### 2.2 Storage schemas

DuckDB schemas are grouped by concern:

* `core.*` – canonical registries and crosswalks (`goids`, `modules`, `repo_map`, etc.).
* `graph.*` – raw structural graphs (call, import, CFG, DFG, test coverage, SCIP symbol uses).
* `analytics.*` – derived metrics: coverage, risk, hotspots, typedness, graph metrics, subsystems, validation findings.
* `docs.*` – denormalized “architecture views” (function/module/subsystem/CFG/DFG) consumed by the MCP server and LLM agents.

### 2.3 NetworkX graph overlays

For every major graph table, the code builds a **NetworkX view**:

* Call graph: `DiGraph` over `graph.call_graph_edges` keyed by function GOIDs.
* Module import graph: `DiGraph` over `graph.import_graph_edges` keyed by module names or GOIDs.
* Test graph: bipartite graph between test GOIDs and function GOIDs using coverage edges.
* CFGs/DFGs: per‑function `DiGraph` for blocks (`graph.cfg_blocks`/`graph.cfg_edges`) and data‑flow edges (`graph.dfg_edges`).
* Symbol‑use graph: directed graph over SCIP symbols / GOIDs when needed for local analytics.

These views are thin helpers that:

1. Query DuckDB for the relevant `graph.*` rows.
2. Construct a `Graph`/`DiGraph`/bipartite graph with attributes such as `weight`, `kind`, or `source`.
3. Hand that `G` to NetworkX algorithms to compute metrics, which are then written to `analytics.*`.

---

## 3. Ingestion & indexing

The ingestion layer is mostly unchanged conceptually, but refactored for clarity and stronger typing. It still does the heavy lifting to create the raw data that all graphs sit on.

### 3.1 AST, CFG, and DFG

* Python source files are parsed into an AST and then into **basic blocks** and edges for control‑flow and data‑flow.
* Blocks and their edges are stored in:

  * `graph.cfg_blocks` (per‑function blocks, labels, spans, block kind, in/out degree)
  * `graph.cfg_edges` (block‑to‑block edges with kinds like normal/exception).
  * `graph.dfg_edges` (data‑flow edges between definitions/uses across blocks and statements). 

Each block is also given a GOID (`kind='block'`) so CFG/DFG analytics can tie back into the same entity universe.

### 3.2 SCIP, tests, coverage, config

* **SCIP ingestion** maps symbols and references onto GOIDs and fills `core.goid_crosswalk.scip_symbol`.
* **Test indexing** builds `analytics.test_catalog` while coverage processing populates `graph.test_coverage_edges` linking tests to covered functions/lines. 
* **Config ingestion** (e.g., pyproject, CI config, service manifests) yields config entities and references that can be linked into graphs and risk/ownership analytics.

All of this is orchestrated via Prefect flows so each piece can be re‑run independently, but the architecture is the same: normalize each external source into DuckDB tables keyed by GOID.

---

## 4. Graph construction layer (graph.*)

Once raw evidence exists, the graphs layer normalizes it into the structural graphs NetworkX will consume.

### 4.1 Call graph

* Builder walks AST/SCIP to find **call sites**, then resolves them to callee GOIDs via the GOID crosswalk.
* Stores nodes in `graph.call_graph_nodes` and edges in `graph.call_graph_edges` with `caller_goid_h128`, `callee_goid_h128`, plus edge attributes like callsite location and call kind. 

This table is intentionally low‑level and may contain multiple edges per logical caller/callee pair (different callsites). The NetworkX layer can aggregate or weight those as needed.

### 4.2 Module import graph

* Scans syntactic imports and resolves them to module GOIDs / module paths.
* Stores in `graph.import_graph_edges` capturing directed edges `src_module → dst_module`.

### 4.3 CFG & DFG graphs

From the ingestion step:

* `graph.cfg_blocks` + `graph.cfg_edges` represent per‑function control‑flow.
* `graph.dfg_edges` represent data‑flow between definitions and uses, usually annotated with variable/symbol information. 

These are already in graph form; the NetworkX layer mostly just turns them into in‑memory `DiGraph` objects and adds attributes.

### 4.4 Test & symbol graphs

* `graph.test_coverage_edges` define a bipartite graph between `test_goid_h128` and `function_goid_h128` with attributes like coverage weight.
* SCIP symbol use edges can be turned into a symbol‑definition graph when needed. 

---

## 5. NetworkX analytics layer (analytics.*)

This is the big change: instead of ad‑hoc degree counts or simple SQL, graph understanding is now **centralized in NetworkX**. The implementation plan doc describes the intended metrics; the current code implements that plan (with some deviations) by using NetworkX’s centrality, components, dominance, and bipartite modules.

### 5.1 Call‑graph metrics (functions & modules)

From the function call graph:

* Build a `DiGraph` whose nodes are function GOIDs and edges are calls.

* Compute:

  * Fan‑in/fan‑out, in/out degree, strongly/weakly connected components.
  * PageRank, betweenness centrality, closeness, eigenvector, harmonic centralities.
  * k‑core indices and local clustering coefficients (on an undirected view).
  * Articulation points and bridge endpoints (functions whose removal separates components).

* Store these metrics in `analytics.graph_metrics_functions` and extended tables, keyed by `function_goid_h128`.

For modules, a similar pattern runs over the import graph, producing `analytics.graph_metrics_modules` with module‑level fan‑in/out, PageRank, centralities, and coupling metrics. 

### 5.2 Subsystem inference

The module import graph plus symbol‑based edges are used to infer **subsystems**:

1. Build a weighted `Graph` whose nodes are modules; edges capture imports, shared symbols, and configuration hints.
2. Seed clusters using tags and config, then apply NetworkX community/label‑propagation algorithms to partition modules into subsystems.
3. Emit:

   * `subsystems.*` – one row per subsystem with aggregate metrics (size, risk, test coverage, etc.).
   * `subsystem_modules.*` – module membership, roles (entrypoint, core, leaf). 

NetworkX’s community and centrality metrics help distinguish “core” modules from peripheral ones and identify good boundaries between subsystems.

### 5.3 Test graph metrics

From the bipartite test⇄function graph:

* Build a bipartite graph in NetworkX (`B`) with node sets for tests and functions.
* Compute per‑test metrics (node degrees, centrality) and the **function‑projection** graph (functions are connected if they are co‑tested in many tests).
* Store metrics in:

  * `analytics.test_graph_metrics_tests` – how broadly each test touches the codebase.
  * `analytics.test_graph_metrics_functions` – which functions are tested by many tests and how they cluster together in the co‑test graph. 

These feed into test impact analysis and risk scoring.

### 5.4 CFG & DFG metrics

For each function:

1. Build a NetworkX `DiGraph` for its CFG from `graph.cfg_blocks`/`graph.cfg_edges`.

2. Use NetworkX dominance algorithms and SCCs to derive:

   * Dominator tree depth per block, whether a block dominates the exit.
   * Loop headers, loop membership, loop nesting depth (from SCCs/backedges).
   * Per‑block betweenness/closeness/eigenvector centrality.

3. Aggregate to `analytics.cfg_block_metrics` and `analytics.cfg_function_metrics` (block‑level and function‑level complexity, loop structure, path lengths). 

Similarly, for DFGs:

* Build a per‑function data‑flow graph; compute connectivity, central data nodes, and counts that reflect data complexity (fan‑out of definitions, depth of data chains, etc.).
* Store in `analytics.dfg_block_metrics` / `analytics.dfg_function_metrics`.

### 5.5 Graph validation

NetworkX also powers a **validation layer**:

* Check for unreachable nodes, broken edges, missing GOIDs, and inconsistent components.
* Emit findings into `analytics.graph_validation.*` (e.g., orphan functions/modules, inconsistent call edges, CFG/DFG anomalies).

This helps catch indexing bugs early and is surfaced in tooling as diagnostics.

### 5.6 Non‑graph analytics

Other analytics (coverage rollups, typedness, hotspots, risk scoring) are still driven by SQL and Python/Pydantic models, but now *consume* the NetworkX metrics as inputs: e.g., risk and hotspot scoring can weight call‑graph centrality and CFG complexity alongside coverage and churn.

---

## 6. Docs views: architecture profiles for LLMs (docs.*)

The `docs.*` schema exposes **denormalized views** that give an LLM a single row per entity with everything it needs. These views are built directly in DuckDB but are conceptually part of the docs_export layer.

### 6.1 Function architecture view

`docs.v_function_architecture` (updated) joins:

* `analytics.function_profile` – core identity, path, qualname, LOC, cyclomatic complexity, typedness, coverage, risk scores.
* `analytics.graph_metrics_functions` + extended call‑graph metrics – PageRank, degrees, centralities, k‑core, clustering, articulation roles, component/SCC info.
* `analytics.test_graph_metrics_functions` – how heavily and broadly a function is tested; co‑test clustering metrics.
* `analytics.cfg_function_metrics` / `analytics.dfg_function_metrics` – control‑flow and data‑flow complexity and shape.
* Subsystem info from `subsystem_modules.*` / `subsystems.*` – which subsystem a function sits in and the subsystem’s overall risk/coverage.

The result: one row per function GOID that includes **static structure, dynamic/testing signals, and NetworkX graph structure**.

### 6.2 Module & subsystem views

* `docs.v_module_architecture` merges module profile, import graph metrics, subsystem membership, and coverage/risk rollups.
* `docs.v_subsystem_summary` summarizes each subsystem (size, modules, risk posture, key entrypoints, call/import centrality).

### 6.3 CFG/DFG block views

To zoom into a single function’s internals, there are block‑level views:

* `docs.v_cfg_block_architecture` – one row per basic block with function context, block span and kind, CFG in/out degree, and per‑block CFG metrics (dominance, centrality, loop membership), plus function‑level CFG context.
* `docs.v_dfg_block_architecture` – similar, but focused on data‑flow: what variables/definitions a block produces/consumes, DFG centrality, and data‑complexity summaries.

These allow an LLM agent to query “show me the structure of this function” and get detailed, graph‑aware information at the block level without recomputing any graphs.

---

## 7. Export & serving

### 7.1 Docs export

A dedicated `docs_export` layer runs DuckDB queries to materialize:

* Parquet and JSONL for each dataset in `core.*`, `graph.*`, `analytics.*`, and selected `docs.*` views.
* A `generate_documents.sh` (or equivalent entrypoint) drives this process so downstream tools only ever see the exported artifacts, not DuckDB directly.

### 7.2 Server & MCP / tools

The server and MCP layers:

* Open DuckDB or read the exported Parquet/JSONL.
* Expose higher‑level APIs for agents and UIs, typically built directly on the `docs.*` views (e.g. “fetch function architecture for GOID X”). 

Because all graph metrics are precomputed via NetworkX, these APIs can answer quite sophisticated questions without having to run heavy algorithms at request time.

---

## 8. Subpackages in `src/codeintel` and their roles

At a high level (names/focus maintained, internals refactored):

* **`config/`** – config models, feature flags, and tuning knobs (e.g., which metrics to compute, thresholds for risk) as Pydantic types.
* **`ingestion/`** – AST/CFG/DFG builders, SCIP ingestion, tests/coverage, config readers; all writing to `core.*` and `graph.*`.
* **`graphs/`** – glue between `graph.*` tables and NetworkX: helpers to build `DiGraph`/bipartite graphs for each structure.
* **`analytics/`** – Prefect tasks that call into NetworkX to compute metrics, plus non‑graph analytics (coverage, typedness, risk, hotspots). They write `analytics.*` tables.
* **`docs_export/`** – DuckDB view definitions and export routines for `docs.*`.
* **`orchestration/`** – Prefect flows that wire ingestion → graphs → analytics → docs_export into pipelines.
* **`server/`, `mcp/`, `services/`** – serving layer and MCP integration, reading `docs.*` outputs to answer queries.
* **`models/`, `utils/`, `stubs/`** – shared Pydantic models, type stubs, and utilities.
* **`cli/`** – entrypoints for running the pipeline and targeted commands from the command line.

---

## 9. How it feels to use, end‑to‑end

For a caller (either a CLI user or an MCP agent), the workflow is conceptually:

1. **Run the enrichment pipeline** for a repo/commit via a Prefect flow.
2. This runs ingestion → GOID assignment → graph construction → NetworkX analytics → docs export.
3. After that, everything you do goes through **docs views** and **analytics tables**: you never rebuild graphs in‑memory; the heavy NetworkX results are already stored.

The net effect of the refactor is that:

* The **data model** (GOIDs, graphs, analytics, docs) is stable and well‑documented.
* The **implementation** of graph understanding is now clearly and consistently delegated to **NetworkX**, giving you a rich set of graph‑theoretic signals for every entity that an LLM can selectively query and use.

# networkx deep dive - graph calcs and high level architectural analysis #

The way your system uses NetworkX now is basically:

> **“Take every structural view of the repo → turn it into a NetworkX graph → run a battery of algorithms → write the results back into DuckDB as `analytics.*` tables → expose them to LLMs via `docs.*` views.”**

I’ll walk through that in detail, graph by graph:

* what *inputs* you feed into NetworkX,
* which NetworkX *algorithms/modules* you use,
* and which *metrics/tables* you produce.

---

## 1. The general NetworkX pattern in your code

Across the codebase, the NetworkX usage follows one consistent pattern

1. **Load graph data from DuckDB**
   Each NetworkX job starts by loading rows from `graph.*` or `analytics.*` tables (e.g. `graph.call_graph_edges`, `graph.import_graph_edges`, `graph.cfg_blocks`, `graph.dfg_edges`, `analytics.test_coverage_edges`).

2. **Build an in‑memory NetworkX graph**
   Depending on what you’re modelling, you build one of:

   * `nx.DiGraph` – directed call graphs, import graphs, CFG/DFG per function. 
   * `nx.Graph` – undirected module affinity graphs, co‑tested function graphs.
   * Bipartite `nx.Graph` – test ↔ function and config ↔ module bipartite graphs (with bipartite node attributes).
   * Occasionally, condensation graphs (`nx.condensation`) for DAG‑style analyses over SCCs.

   Nodes are GOIDs or other stable keys (module name, test ID, config key), edges carry attributes like weights or coverage ratios.

3. **Run NetworkX algorithms**
   You call into NetworkX’s centrality, components, dominance, bipartite, and structural‑holes modules (all documented in the 3.6 reference) to compute metrics. 

4. **Write metrics back into DuckDB**
   Results are persisted in `analytics.*` tables like:

   * `analytics.graph_metrics_functions_ext`
   * `analytics.graph_metrics_modules`
   * `analytics.cfg_block_metrics`, `analytics.cfg_function_metrics`
   * `analytics.dfg_block_metrics`, `analytics.dfg_function_metrics`
   * `analytics.test_graph_metrics_tests`, `analytics.test_graph_metrics_functions`
   * plus symbol/config/subsystem‑level metrics. 

5. **Expose via docs views**
   The `docs.*` views (especially `docs.v_function_architecture`, `docs.v_cfg_block_architecture`, `docs.v_dfg_block_architecture`) join these metrics onto function/module/subsystem profiles so the MCP/LLM layer sees them as regular columns. 

---

## 2. Function call‑graph analytics

### 2.1 Inputs

From DuckDB:

* `graph.call_graph_edges`

  * `caller_goid_h128` → caller function GOID
  * `callee_goid_h128` → callee function GOID (resolved)
  * repo/commit, plus callsite attributes. 

Optionally:

* `graph.call_graph_nodes` to ensure isolated functions are still included.
* `analytics.function_profile` for joining back metadata (loc, complexity, risk, etc.).

### 2.2 NetworkX graph

You build a **directed call graph**:

```text
G_call = nx.DiGraph()
nodes: function_goid_h128 as ints
edges: caller → callee, with edge["weight"] = number of callsites
```

Multiple callsites between the same functions are coalesced into a single directed edge with an integer `weight` attribute. 

You also construct an **undirected view** for some symmetric metrics:

```python
G_und = G_call.to_undirected()
```

### 2.3 NetworkX algorithms used

From the NetworkX **centrality**, **components**, **clustering**, **cores**, and **bridges** modules  :

* **Centralities** (on `G_call` / `G_und`):

  * `nx.betweenness_centrality(G_call, k=...)`
  * `nx.closeness_centrality(G_call)`
  * `nx.eigenvector_centrality(G_und, max_iter=...)`
  * `nx.harmonic_centrality(G_call)`

* **Components & SCCs**:

  * `nx.weakly_connected_components(G_call)`
  * `nx.strongly_connected_components(G_call)`

* **Local structure & cores**:

  * `nx.core_number(G_und)`
  * `nx.clustering(G_und)`
  * `nx.triangles(G_und)`

* **Bridges / articulation points**:

  * `nx.articulation_points(G_und)`
  * `nx.bridges(G_und)`

These functions are all part of NetworkX’s 3.6 algorithm reference for centrality, components, clustering, and structural roles. 

### 2.4 Outputs: `analytics.graph_metrics_functions_ext`

For each function node, you write one row with metrics like  :

* **Centrality metrics**

  * `call_betweenness` – betweenness centrality (how often the function sits on shortest paths).
  * `call_closeness` – closeness centrality (inverse of average distance to other nodes).
  * `call_eigenvector` – eigenvector centrality (importance via important neighbors).
  * `call_harmonic` – harmonic centrality (closeness variant robust across components).

* **Local structure**

  * `call_core_number` – k‑core index from `nx.core_number`.
  * `call_clustering_coeff` – local clustering coefficient (triangle density).
  * `call_triangle_count` – exact triangle count from `nx.triangles`.

* **Structural role**

  * `call_is_articulation` – true if the node is an articulation point in `G_und`.
  * `call_is_bridge_endpoint` – true if the node touches a bridge edge.
  * (Optionally) `call_articulation_impact` – estimated “damage” if removed.

* **Graph context**

  * `call_component_id`, `call_component_size` – from weakly connected components.
  * `call_scc_id`, `call_scc_size` – from strongly connected components.

These live alongside “base” call metrics (fan‑in/out, PageRank, layered position) from `analytics.graph_metrics_functions`, giving you a very rich function‑level call topology. 

---

## 3. Module import‑graph analytics

### 3.1 Inputs

From DuckDB:

* `graph.import_graph_edges`

  * `src_module` → importing module
  * `dst_module` → imported module
  * repo/commit, plus cycle metadata. 

Also `analytics.module_profile` and `analytics.subsystem_modules` for context.

### 3.2 NetworkX graph

You build an **import graph**:

```text
G_import = nx.DiGraph()
nodes: module identifiers (usually module path as string)
edges: src_module → dst_module (import relationship)
```

Weights may accumulate if there are multiple import edges between the same modules.

You also derive:

* SCCs via `nx.strongly_connected_components(G_import)` and a condensation DAG via `nx.condensation(G_import)`.
* An undirected view for some structural metrics.

### 3.3 NetworkX algorithms used

From **centrality**, **components**, **cores**, and **structural holes** modules  :

* **Centrality**:

  * `nx.betweenness_centrality(G_import)`
  * `nx.closeness_centrality(G_import)`
  * `nx.eigenvector_centrality(G_import.to_undirected())`
  * `nx.harmonic_centrality(G_import)`

* **Cores & rich‑club**:

  * `nx.core_number(G_import.to_undirected())` to classify core vs shell modules.

* **Structural holes** (brokerage):

  * `nx.constraint(G_import.to_undirected())`
  * `nx.effective_size(G_import.to_undirected())`

* **SCCs & condensation**:

  * `nx.strongly_connected_components(G_import)` – cycle groups.
  * `nx.condensation(G_import)` – module‑component DAG; used to compute import layers via a DAG‑layering helper.

### 3.4 Outputs: `analytics.graph_metrics_modules` (extended)

For each module you store metrics including: 

* Import fan‑in/fan‑out counts and degrees.
* `import_pagerank` (link‑analysis centrality).
* `import_bc`, `import_closeness`, `import_eigenvector`, `import_harmonic`.
* `import_core_number`, and a derived shell index.
* `import_constraint`, `import_effective_size` – “broker modules.”
* `import_layer` – DAG layer from condensation graph (rough dependency depth).
* `cycle_group` and `in_cycle` – import SCC membership.

These feed directly into module‑level docs views and subsystem inference.

---

## 4. Subsystem graph & community detection

### 4.1 Inputs

* `analytics.module_profile` – the pool of modules to cluster.
* `graph.import_graph_edges` – import relationships.
* Symbol and config coupling (symbol uses and config values) that you fold into weights. 
* Tags and hints from metadata for seeding.

### 4.2 NetworkX graphs

Two main graphs:

1. **Module affinity graph** – undirected, weighted:

   ```text
   G_affinity = nx.Graph()
   nodes: modules
   edges: weight = f(imports, shared symbols, shared config)
   ```

   Built by accumulating contributions from:

   * Import graph,
   * Symbol‑use overlaps,
   * Config co‑usage. 

2. **Subsystem interaction graph** – directed, coarse‑grained:

   ```text
   G_subsystems = nx.DiGraph()
   nodes: subsystem_id
   edges: A → B if any module in A imports any in B; weight = number of such edges
   ```

### 4.3 NetworkX algorithms used

On `G_affinity`:

* A **seeded label‑propagation** algorithm implemented in your code (not the built‑in `asyn_lpa_communities`, but conceptually similar), using adjacency from `G_affinity.node[neighbors]` and edge weights.

  * Modules with strong tags (e.g., `subsystem:foo`) are frozen seeds.
  * Other modules iteratively adopt the label that maximizes total incident edge weight from neighbors.

On `G_subsystems`:

* Centrality and layering:

  * `nx.betweenness_centrality(G_subsystems)`
  * `nx.closeness_centrality(G_subsystems)`
  * `nx.condensation(G_subsystems)` + DAG layering for subsystem depth.

### 4.4 Outputs

* `analytics.subsystems`:

  * Subsystem ID, name, and aggregates:

    * number of modules, numbers of functions, coverage rollups,
    * import fan‑in/out at subsystem level,
    * `subsystem_bc`, `subsystem_closeness`, `subsystem_layer_index`.

* `analytics.subsystem_modules`:

  * `(subsystem_id, module)` membership rows,
  * annotations for “entrypoints” and “bridge modules” based on import centrality and boundary edges. 

Subsystem membership then flows into `docs.v_function_architecture` and `docs.v_module_architecture`.

---

## 5. Test ↔ function bipartite analytics

### 5.1 Inputs

From `analytics.test_coverage_edges` and `analytics.test_catalog`:

* For each coverage edge:

  * `test_id`
  * `function_goid_h128`
  * `coverage_ratio`, `covered_lines`, `executable_lines` 

### 5.2 NetworkX graph

You build a **bipartite graph**:

```text
B = nx.Graph()
nodes:
  ("t", test_id) with bipartite = 0
  ("f", int(function_goid_h128)) with bipartite = 1

edges:
  ("t", test_id) -- ("f", func_goid) with weight = coverage_ratio (accumulated)
```

This uses the NetworkX bipartite graph pattern (undirected `Graph` with a `bipartite` node attribute). 

You then derive **projected graphs**:

* `G_tests` – test ↔ test graph, weighted by number/coverage of shared functions.
* `G_funcs` – function ↔ function graph, weighted by number/coverage of shared tests.

### 5.3 NetworkX algorithms used

From the **bipartite** and **centrality** modules  :

* On the bipartite graph `B`:

  * `bipartite.degree_centrality(B, funcs)` – degree centrality for tests.
  * `bipartite.degree_centrality(B, tests)` – degree centrality for functions.

* Projections:

  * `bipartite.weighted_projected_graph(B, tests)`
  * `bipartite.weighted_projected_graph(B, funcs)`

* On the projected graphs:

  * `nx.degree(G_tests, weight=None/"weight")`
  * `nx.degree(G_funcs, weight=None/"weight")`
  * `nx.clustering(G_tests, weight="weight")`
  * `nx.clustering(G_funcs, weight="weight")`
  * `nx.betweenness_centrality(G_tests, weight="weight", k=...)`
  * `nx.betweenness_centrality(G_funcs, weight="weight", k=...)`

### 5.4 Outputs: `analytics.test_graph_metrics_*`

You persist two main tables: 

**Per test** – `analytics.test_graph_metrics_tests`:

* `degree` – number of functions this test exercises.
* `weighted_degree` – total coverage weight over those edges.
* `degree_centrality` – bipartite degree centrality.
* `proj_degree` – degree in the test↔test projection graph.
* `proj_weight` – weighted degree in that projection.
* `proj_clustering` – local clustering coefficient in test graph.
* `proj_betweenness` – test’s betweenness centrality among tests.

**Per function** – `analytics.test_graph_metrics_functions`:

* `tests_degree` – how many tests cover the function.
* `tests_weighted_degree` – coverage‑weighted test load.
* `tests_degree_centrality` – bipartite centrality for the function node.
* `proj_degree` / `proj_weight` – degree and strength in co‑tested functions graph.
* `proj_clustering` / `proj_betweenness` – structural role among co‑tested functions.

These are joined into `docs.v_function_architecture` (for function rows) and `docs.v_test_architecture` (for test‑centric views). 

---

## 6. CFG (control‑flow graph) analytics

### 6.1 Inputs

From `graph.cfg_blocks` and `graph.cfg_edges`: 

* Blocks per function:

  * `function_goid_h128`
  * `block_idx`, `kind` (`entry`, `exit`, etc.)
  * `in_degree`, `out_degree`
  * `start_line`, `end_line`, labels.

* Edges per function:

  * `src_block_idx`
  * `dst_block_idx`
  * `edge_type` (normal, exceptional, etc.)

Plus function metadata from `analytics.function_profile`.

### 6.2 NetworkX graph

For each function, you build a **per‑function CFG**:

```text
G_cfg = nx.DiGraph()
nodes: block_idx
node attributes: kind, start/end lines, in_degree, out_degree

edges: src_block_idx → dst_block_idx
edge attributes: edge_type
```

You identify entry/exit blocks using `kind` or heuristic fallback (min in‑degree, out_degree==0).

### 6.3 NetworkX algorithms used

From the **domination**, **components**, **DAG**, **shortest paths**, and **centrality** modules  :

* **Graph structure**:

  * `nx.is_directed_acyclic_graph(G_cfg)`
  * `nx.strongly_connected_components(G_cfg)` → loop SCCs.
  * `nx.condensation(G_cfg)` → SCC DAG.

* **Dominance** (Dominance module):

  * `nx.immediate_dominators(G_cfg, entry_block_idx)`
  * `nx.dominance_frontiers(G_cfg, entry_block_idx)`

* **Path metrics**:

  * If DAG: `nx.dag_longest_path_length(G_cfg)` over reachable subgraph from entry.
  * Else: `nx.dag_longest_path_length(nx.condensation(G_cfg))` as an approximation.
  * Single‑source shortest paths from entry:

    * `nx.single_source_shortest_path_length(G_cfg, entry_block_idx)`

* **Centralities** (per block):

  * `nx.betweenness_centrality(G_cfg, k=...)`
  * `nx.closeness_centrality(G_cfg)`
  * `nx.eigenvector_centrality(G_cfg.to_undirected(), max_iter=...)`

### 6.4 Outputs: `analytics.cfg_block_metrics` & `analytics.cfg_function_metrics`

Per block (`analytics.cfg_block_metrics`): 

* Local structure:

  * `is_entry`, `is_exit`
  * `is_branch` (out_degree > 1)
  * `is_join` (in_degree > 1)

* Dominance:

  * `dom_depth` – depth in the dominator tree.
  * `dominates_exit` – flag for blocks that dominate the exit.

* Centrality:

  * `bc_betweenness`
  * `bc_closeness`
  * `bc_eigenvector`

* Loop structure:

  * `in_loop_scc` – block in SCC of size > 1.
  * `loop_header` – candidate loop header (heuristic).
  * `loop_nesting_depth` – depth of nested loops containing this block.

Per function (`analytics.cfg_function_metrics`):

* Size & structure:

  * `cfg_block_count`, `cfg_edge_count`
  * `cfg_entry_block_idx`, `cfg_exit_block_idx`
  * `cfg_is_dag`, `cfg_scc_count`, `cfg_has_cycles`

* Paths & branching:

  * `cfg_longest_path_len`
  * `cfg_avg_shortest_path_len`
  * `cfg_branching_factor_mean`, `cfg_branching_factor_max`
  * `cfg_linear_block_fraction` (blocks with in=1 & out=1)

* Dominance summary:

  * `cfg_dom_tree_height`
  * `cfg_dominance_frontier_size_mean`, `cfg_dominance_frontier_size_max`

* Loop & centrality aggregates:

  * `cfg_loop_count`, `cfg_loop_nesting_depth_max`
  * `cfg_bc_betweenness_max`, `cfg_bc_betweenness_mean`
  * `cfg_bc_closeness_mean`, `cfg_bc_eigenvector_max`

These are joined into `docs.v_function_architecture` and `docs.v_cfg_block_architecture`, giving an LLM detailed control‑flow shape per function and per block. 

---

## 7. DFG (data‑flow graph) analytics

### 7.1 Inputs

From `graph.dfg_edges`: 

* For each edge:

  * `function_goid_h128`
  * `src_block_idx`, `dst_block_idx`
  * `src_symbol`, `dst_symbol`
  * `via_phi` (SSA‑like phi edge)
  * `use_kind` (type of use).

Plus function metadata from `analytics.function_profile`.

### 7.2 NetworkX graph

For each function, you build a **data‑flow graph**:

```text
G_dfg = nx.DiGraph()
nodes: block_idx
edges: src_block_idx → dst_block_idx
edge attributes: src_symbol, dst_symbol, via_phi, use_kind
```

### 7.3 NetworkX algorithms used

From **components**, **DAG**, **shortest‑paths**, **centrality** modules  :

* **Connectivity & SCCs**:

  * `nx.weakly_connected_components(G_dfg)`
  * `nx.strongly_connected_components(G_dfg)`

* **Path metrics**:

  * DAG case: `nx.dag_longest_path_length(G_dfg)`

  * Otherwise: `nx.dag_longest_path_length(nx.condensation(G_dfg))`

  * Approx average shortest path:

    * For each node, `nx.single_source_shortest_path_length(G_dfg, node)` and aggregate.

* **Centralities**:

  * `nx.betweenness_centrality(G_dfg, k=...)`
  * `nx.eigenvector_centrality(G_dfg.to_undirected(), max_iter=...)`
  * (Optionally) closeness centrality per block.

### 7.4 Outputs: `analytics.dfg_block_metrics` & `analytics.dfg_function_metrics`

Per block (`analytics.dfg_block_metrics`): 

* Local degrees:

  * `dfg_in_degree`, `dfg_out_degree`
  * `dfg_phi_in_degree`, `dfg_phi_out_degree`

* Centrality:

  * `dfg_bc_betweenness`
  * `dfg_bc_closeness` (optional)
  * `dfg_bc_eigenvector`

* Structural placement:

  * `dfg_in_chain` – whether it lies on a longest dataflow chain.
  * `dfg_in_scc` – participates in cyclic data‑flow.

Per function (`analytics.dfg_function_metrics`):

* Size & symbol richness:

  * `dfg_block_count`, `dfg_edge_count`
  * `dfg_phi_edge_count`
  * `dfg_symbol_count` – distinct src/dst symbols.

* Connectivity:

  * `dfg_component_count`, `dfg_scc_count`, `dfg_has_cycles`

* Paths & branching:

  * `dfg_longest_chain_len`
  * `dfg_avg_shortest_path_len`
  * `dfg_avg_in_degree`, `dfg_avg_out_degree`
  * `dfg_max_in_degree`, `dfg_max_out_degree`
  * `dfg_branchy_block_fraction`

* Centrality aggregates:

  * `dfg_bc_betweenness_max`, `dfg_bc_betweenness_mean`
  * `dfg_bc_eigenvector_max`

These flow into `docs.v_function_architecture` and `docs.v_dfg_block_architecture`. 

---

## 8. Symbol‑use graph analytics

### 8.1 Inputs

From symbol ingestion:

* `symbol_use_edges` – SCIP symbol definition → use edges, with file/line context. 
* Module mapping from `core.modules` / `analytics.module_profile`.

### 8.2 NetworkX graph

You construct a **module‑level symbol coupling graph**:

```text
G_sym = nx.Graph()
nodes: modules
edges: u -- v
  weight = count or score of shared symbols between modules u and v
```

Weights are accumulated by scanning definition→use edges and mapping paths to modules.

### 8.3 NetworkX algorithms used

From **centrality**, **structural holes**, **communities** modules  :

* `nx.betweenness_centrality(G_sym)`
* `nx.closeness_centrality(G_sym)`
* `nx.eigenvector_centrality(G_sym)`
* `nx.constraint(G_sym)`, `nx.effective_size(G_sym)` – symbol brokers.
* (Optional) community detection, e.g. label propagation or modularity‑based communities.

### 8.4 Outputs

* `analytics.symbol_graph_metrics_modules`:

  * Per module: symbol‑based centralities, constraint/effective size, community ID.

This complements import‑graph metrics with **semantic coupling**.

---

## 9. Configuration bipartite analytics

### 9.1 Inputs

From config ingestion:

* `config_values` – which config keys appear in which modules. 

### 9.2 NetworkX graph

A **config ↔ module bipartite graph**:

```text
B_cfg = nx.Graph()
left nodes: ("k", key)
right nodes: ("m", module)
edges: ("k", key) -- ("m", module)
```

### 9.3 NetworkX algorithms used

Again from **bipartite**, **centrality**, **structural holes**:

* Degrees and bipartite degree centrality.

* Projections:

  * key ↔ key graph (keys used together),
  * module ↔ module graph (modules sharing keys).

* Centrality & brokerage on those projections:

  * `nx.betweenness_centrality`, `nx.closeness_centrality`, `nx.eigenvector_centrality`.
  * `nx.constraint`, `nx.effective_size`.

### 9.4 Outputs

* `analytics.config_graph_metrics_keys`:

  * Per config key: how many modules use it, graph centralities, brokerage scores.

* `analytics.config_graph_metrics_modules`:

  * Per module: config centrality degrees, which keys it shares with other modules.

These help identify config hotspots and hidden coupling through configuration.

---

## 10. Global graph stats & validation

### 10.1 Inputs

For each main graph:

* `G_call`, `G_import`, `G_sym`, `G_subsystems`, `G_cfg`, `G_dfg`, `B_tests`, `B_cfg`.

### 10.2 NetworkX algorithms used

From **components**, **clustering**, **small‑world**, **efficiency** modules  :

* Node/edge counts, component counts:

  * `len(G.nodes)`, `len(G.edges)`
  * `nx.number_connected_components` / `nx.number_weakly_connected_components`
  * `nx.number_strongly_connected_components`

* Path‑length approximations:

  * Sampled shortest paths via `nx.single_source_shortest_path_length` from a few seeds.

* Clustering coefficients:

  * `nx.average_clustering(G.to_undirected())`

* Small‑world metrics:

  * `nx.sigma(G)` or equivalent if enabled.

### 10.3 Outputs

* `analytics.graph_stats`:

  * `graph_name`, `repo`, `commit`
  * `node_count`, `edge_count`
  * `connected_components`, `scc_count`
  * `diameter_estimate`, `avg_shortest_path_estimate`
  * `avg_clustering_coeff`, `small_world_sigma`

* `analytics.graph_validation.*`:

  * Findings derived from structural checks (large SCCs, hubs, weird degree distributions, unreachable code).

These stats are mostly for health checks and “shape of repo” dashboards.

---

## 11. How all this is used downstream

All of these NetworkX‑derived metrics are:

1. **Persisted** into `analytics.*` tables keyed by GOID/module/test/config/subsystem. 

2. **Joined** into `docs.*` views:

   * `docs.v_function_architecture` – function row includes call centralities, test centralities, CFG/DFG metrics, subsystem context.
   * `docs.v_cfg_block_architecture` – block row includes CFG position + centralities.
   * `docs.v_dfg_block_architecture` – block row includes DFG position + centralities.

3. **Consumed** by the MCP/LLM layer, which can now:

   * Ask for “most central high‑risk functions,” “bridge modules between subsystems,” “tests that cover functions with high betweenness & low coverage,” etc., using simple SQL.
   * Plan refactors based on graph structure (calling patterns, import layers, loop depth, data‑flow complexity) rather than just LOC/complexity.

So NetworkX is now the “graph brain” of your system: it takes the structural views you already had in `graph.*`, runs a broad slice of the 3.6 algorithm surface (centrality, components, dominance, bipartite projections, structural holes, clustering), and turns them into durable metadata that an LLM can treat as just more columns in a table.



# All other analysis and derived metrics #$

Gotcha — so now we zoom out from “NetworkX‑specific graph stuff” and walk *all* the other analytics passes and derived tables that sit **after ingest**: function metrics, coverage/tests, typedness, hotspots, config/static diagnostics, risk factors, profiles, subsystems rollups, validations, and the docs views.

Below I’ll go category‑by‑category in the same order as your README + architecture doc, and for each one explain:

* **Inputs** (which tables / signals it reads),
* **Transformations / algorithms**, and
* **Outputs** (the actual table + how downstream code uses it).

---

## 1. AST‑derived analytics: `ast_metrics.*` and `hotspots.jsonl`

### 1.1 `ast_metrics.*` (per‑file structure)

**Inputs**

* Raw AST from the AST indexer (`ast_nodes.*`).
* Repo + commit context from `repo_map.json`.

**Mechanism**

The `CodeIntel.services.enrich.function_metrics` / AST analytics step walks each file’s AST and computes counts and structural properties:

* total AST node count
* number of functions / classes
* max depth
* per‑file complexity (derived from node kinds, branches, etc.)

This is essentially a “static structure summary” for each file, independent of GOIDs.

**Outputs**

`ast_metrics.*` is used by:

* `hotspots.jsonl` (see below) as the **complexity input**.
* `file_profile.*` as `node_count`, `function_count`, `class_count`, `ast_complexity`.

---

### 1.2 `hotspots.jsonl` (churn + complexity)

**Inputs**

* `ast_metrics.*` for per‑file complexity.
* Git history (commits, authors, diff stats) for the configured time window.

**Mechanism**

`CodeIntel.services.enrich.analytics.hotspots` joins git history with AST metrics and computes: 

* `commit_count` — commits touching the file in the window.
* `author_count` — distinct authors.
* `lines_added` / `lines_deleted` — churn.
* `complexity` — from `ast_metrics`.
* `score` — a composite “hotspot” score, typically some monotone function of:

  * higher complexity → higher score
  * more churn/authors → higher score

The exact formula isn’t specified in the README, but the score is explicitly “composite” and used for ranking hotspots. 

**Outputs / usage**

* `hotspot_score` is pulled into `goid_risk_factors.*` as `hotspot_score` and used as one of the risk components. 
* Rolled up to `file_profile.*` as `hotspot_score`. 
* Rolled up again to `module_profile.*` (e.g. “module contains multiple hot files”). 

---

## 2. Typedness & static diagnostics

### 2.1 `typedness.jsonl` (per‑file type coverage)

**Inputs**

* Pyright + Pyrefly runs (errors + type coverage).
* Repo paths from `repo_map.json`.

**Mechanism**

`CodeIntel.services.enrich.exports.write_typedness_output` aggregates “file‑level typing health”: 

* `annotation_ratio` is a small JSON object like `{"params": float, "returns": float}`.
* `type_error_count` is the max of Pyrefly/Pyright error counts for the file.
* `untyped_defs` counts top‑level functions without full annotations.
* `overlay_needed` is a heuristic flag (“should we add stub overlays?”).

**Outputs / usage**

* Joined into `goid_risk_factors.*` as `typedness_bucket`, `file_typed_ratio`, `typedness_source`. 
* Used in `function_profile.*` as `typedness_bucket`, `file_typed_ratio`, and `static_error_count`.
* Rolled up to `file_profile.*` (`annotation_ratio`, `type_error_count`). 

---

### 2.2 `static_diagnostics.*` (per‑file static errors)

**Inputs**

* `FileTypeSignals` captured during pipeline preparation.
* Pyright + Pyrefly error counts.

**Mechanism**

`write_static_diagnostics_output` writes: 

* `pyrefly_errors`, `pyright_errors`
* `total_errors = pyrefly_errors + pyright_errors`
* `has_errors = total_errors > 0`

**Outputs / usage**

* Joined into `goid_risk_factors.*` as `static_error_count` + `has_static_errors`. 
* Joined into `file_profile.*` as `static_error_count`, `has_static_errors`. 
* Strong risk signal for `risk_score` and `risk_level`.

---

## 3. Per‑function structural metrics: `function_metrics.*`

**Inputs**

* GOID registry (`goids.parquet`) for function spans and identity. 
* AST nodes & metrics for the function body.

**Mechanism**

`CodeIntel.services.enrich.function_metrics` (CLI `function-metrics`) walks each function/method GOID and computes: 

* Identity: `function_goid_h128`, `urn`, `repo`, `commit`, `rel_path`, `language`, `kind`, `qualname`.

* Span: `start_line`, `end_line`, with

  * `loc = end_line - start_line + 1`
  * `logical_loc` = non‑blank/non‑comment lines.

* Signature metrics:

  * `param_count`, `positional_params`, `keyword_only_params`, `has_varargs`, `has_varkw`.

* Structural metrics (not fully shown in snippet, but described):

  * `cyclomatic_complexity`
  * nesting depth (if tracked)
  * counts of `return`/`yield`/`raise`
  * docstring presence.

**Outputs / usage**

* Directly surfaced in `function_profile.*` (loc, logical_loc, complexity, param counts, return_type).
* Used heavily by `goid_risk_factors.*` as complexity + size inputs and for the `complexity_bucket`. 

---

## 4. Coverage & tests analytics

### 4.1 Line‑level coverage: `coverage_lines.*`

**Inputs**

* Coverage.py `.coverage` DB with dynamic contexts enabled (`dynamic_context = test_function`).
* Repo mapping for paths.

**Mechanism**

`CodeIntel.cli.enrich_analytics coverage-detailed` reads the coverage DB and writes one row per `(file, line)` with:

* `is_executable` — whether coverage thinks it’s a statement.
* `is_covered` — whether it ran.
* `hits` — best‑effort hit count (usually 1 for “covered at least once”).
* `context_count` — how many distinct coverage contexts (tests) hit this line.

**Outputs / usage**

* Aggregated by GOID span into `coverage_functions.*`.
* Can be joined directly for “highlight uncovered lines in this GOID”.

---

### 4.2 Function coverage: `coverage_functions.*`

**Inputs**

* `coverage_lines.*` for line‑level coverage.
* `goids` for `(function_goid_h128, rel_path, start_line, end_line)`.

**Mechanism**

Group `coverage_lines` by function span and compute: 

* `executable_lines` — count of `is_executable` lines in the span.
* `covered_lines` — count of `is_covered` lines.
* `coverage_ratio = covered_lines / executable_lines` (nullable).
* `tested = covered_lines > 0`.
* `untested_reason` (`no_executable_code`, `no_tests`, or empty).

**Outputs / usage**

* Joined into `goid_risk_factors.*` as `executable_lines`, `covered_lines`, `coverage_ratio`, `tested`. 
* Joined into `function_profile.*` as `coverage_ratio`, `tested`.

---

### 4.3 Test catalog: `test_catalog.*`

**Inputs**

* Pytest JSON report (via `pytest-json-report`).
* GOID crosswalk for tests (mapping test functions to GOIDs).

**Mechanism**

`CodeIntel.cli.enrich_analytics test-analytics` writes one row per pytest nodeid with:

* Identity: `test_id`, optional `test_goid_h128` + `urn`, `rel_path`, `qualname`, `kind`.
* Execution data: `status`, `duration_ms`.
* Metadata: `markers`, `parametrized`, `flaky`.

**Outputs / usage**

* Base “tests dimension” for impact analysis.
* Joined with `test_coverage_edges.*` to answer “which tests hit this GOID?”.
* Used in `function_profile.*` for `tests_touching`, `failing_tests`, `slow_tests`, `last_test_status`.

---

### 4.4 Test ↔ function edges: `test_coverage_edges.*`

**Inputs**

* Coverage DB with dynamic contexts (same as coverage_lines).
* `test_catalog.*` to resolve nodeids.

**Mechanism**

`test-analytics` builds a bipartite edge table `test_coverage_edges.*` where each row is:

* `test_id`, optional `test_goid_h128`.
* `function_goid_h128`, `urn`, `rel_path`, `qualname`.
* `covered_lines`, `executable_lines`, `coverage_ratio` — *for that test → function pair*.
* `last_status` of the test.

Essentially: “this test executed N of M lines in this function”.

**Outputs / usage**

* Summed over `test_id` for each function to derive `tests_touching`, `failing_tests`, `slow_tests` for `function_profile.*`.
* Serves as the input graph for the NetworkX **test bipartite + projections** we already discussed (`test_graph_metrics_*`).

---

## 5. Config & configuration blast‑radius: `config_values.*`

**Inputs**

* Output of `index_config_files` — parsed config files (YAML/TOML/JSON/etc.).
* Output of `prepare_config_state` — where config keys are referenced in code.

**Mechanism**

`write_config_output` flattens per‑file config analysis into per‑key records: 

* `config_path`, `format`.
* `key` — normalized config key path (`"service.database.host"`).
* `reference_paths` — list of code files referencing it.
* `reference_modules` — modules for those files.
* `reference_count` — distinct referencing files.

**Outputs / usage**

* Directly consumable for “if I change this config, where does it flow?”.
* Joined onto modules to find “config‑heavy” modules.
* Feeds into the NetworkX **config ↔ module** graph metrics we described, but the base dataset is this table.

---

## 6. GOID risk factors: `goid_risk_factors.*`

**Inputs**

The CLI `enrich_analytics risk-factors` is the core aggregator and joins:

* `function_metrics.*` (size, complexity, parameters).
* `function_types.*` + `typedness.jsonl` (annotation coverage, return types).
* `coverage_functions.*` (coverage, tested flag).
* `test_catalog.*` + `test_coverage_edges.*` (test counts and status).
* `hotspots.jsonl` (file‑level hotspot score).
* `static_diagnostics.*` (error counts).
* Potentially call graph degrees (fan‑in/out) as extra signals.

**Mechanism**

For each function GOID, it builds a feature vector and computes:

* Raw features like `loc`, `logical_loc`, `cyclomatic_complexity`, `complexity_bucket`, `typedness_bucket`, `file_typed_ratio`, `hotspot_score`, `static_error_count`, `executable_lines`, `covered_lines`, `coverage_ratio`, `tested`, `test_count`. 
* Then a **heuristic risk model**:

  * `risk_component_*` fields for coverage, complexity, static diagnostics, hotspots, typedness, etc.
  * `risk_score` — a 0–1 aggregate of these components.
  * `risk_level` — a categorical bucket (`low` / `medium` / `high`), derived from `risk_score` thresholds.

The README explicitly calls `risk_score` “heuristic” and lists these components, so the combination is under your control in code, but conceptually it’s *“higher when code is complex, untyped, untested, hot, and error‑prone”.*

**Outputs / usage**

* `goid_risk_factors.*` is the *primary* risk source for:

  * `function_profile.risk_score`, `risk_level`, `risk_component_*`. 
  * `module_profile.*` risk rollups (`high_risk_function_count`). 
  * `subsystems.*` risk rollups (e.g., `high_risk_function_count` per subsystem).

---

## 7. Profiles: `function_profile.*`, `file_profile.*`, `module_profile.*`

All three are built by `analytics/profiles.*` via `ProfilesStep` in orchestration. 

### 7.1 `function_profile.*` (main per‑function view)

**Inputs**

* Identity + structural: `function_metrics.*`.
* Types & typedness: `function_types.*`, `typedness.jsonl`, `static_diagnostics.*`.
* Coverage & tests: `coverage_functions.*`, `test_coverage_edges.*`, `test_catalog.*`.
* Call graph degrees: `graph.call_graph_edges.*` / `graph_metrics_functions.*`.
* Risk: `goid_risk_factors.*`.
* Docstrings, tags, owners: from AST metadata + module registry.

**Mechanism**

For each GOID, `build_function_profile` denormalizes all of that into a single record:

* Structural columns: `loc`, `logical_loc`, `cyclomatic_complexity`, `param_count`, `keyword_params`, `vararg`, `kwarg`, `total_params`, `return_type`. 
* Typedness: `typedness_bucket`, `file_typed_ratio`, `static_error_count`.
* Coverage & tests: `coverage_ratio`, `tested`, `tests_touching`, `failing_tests`, `slow_tests`, `last_test_status`.
* Call graph: `call_fan_in`, `call_fan_out`, `call_is_leaf`. (These are pulled from graph metrics / call graph edges.)
* Risk: `risk_score`, `risk_level`, `risk_component_*` (mirroring `goid_risk_factors`).
* Documentation & metadata: `doc_short`, `doc_long`, `tags`, `owners`.

**Outputs / usage**

* Directly used by `docs.v_function_architecture` (and your beefed‑up version that also joins graph/CFG/DFG/test‑graph metrics).
* First stop for MCP/LLM agents when they want “everything we know about this function”.

---

### 7.2 `file_profile.*` (per‑file aggregation)

**Inputs**

* `function_profile.*` (all functions in the file).
* `ast_metrics.*`.
* `hotspots.jsonl`.
* `typedness.jsonl`.
* `static_diagnostics.*`.

**Mechanism**

`build_file_profile` groups `function_profile` by `rel_path` and aggregates: 

* Counts: `total_functions`, `public_functions`.
* AST metrics: `node_count`, `function_count`, `class_count`, `ast_complexity`.
* Risk & complexity: `hotspot_score`, `avg_loc`, `max_loc`, `avg_cyclomatic_complexity`.
* Typedness & diagnostics: `annotation_ratio`, `type_error_count`, `static_error_count`, `has_static_errors`.
* Ownership and tags at file/module level.

**Outputs / usage**

* Drives file‑level prioritization (“which files are most risky / hot / untyped?”).
* Rolls into `module_profile.*` as aggregate size and risk signals. 

---

### 7.3 `module_profile.*` (per‑module aggregation)

**Inputs**

* `file_profile.*` (all files in module).
* `function_profile.*` / `goid_risk_factors.*` for function‑level risk.
* `import_graph_edges.*` for import fan‑in/out and `cycle_group`.

**Mechanism**

For each module (from `modules` / `repo_map.json`), it aggregates:

* Size / complexity: `file_count`, `total_loc`, `function_count`, `avg_file_complexity`, `max_file_complexity`.
* Risk & coverage: `high_risk_function_count`, `module_coverage_ratio` (averaging or weighting from `coverage_functions.*`).
* Import structure: `import_fan_in`, `import_fan_out`, `cycle_group`, `in_cycle` (from `import_graph_edges.*`).
* Ownership: `tags`, `owners`.

**Outputs / usage**

* Core input to module‑level architecture view `docs.v_module_architecture`.
* Input to subsystem summarization (size/risk/coverage per subsystem). 

---

## 8. Graph metrics & subsystems (non‑NetworkX bits)

You already asked for the NetworkX specifics; here I’ll focus on the *rest of the mechanism*.

### 8.1 Graph metrics: `graph_metrics_functions.*`, `graph_metrics_modules.*`

**Inputs**

* Call graph: `graph.call_graph_nodes.*`, `graph.call_graph_edges.*`.
* Import graph: `graph.import_graph_edges.*`.
* Symbol use graph: `symbol_use_edges.*` (for symbol coupling).

**Mechanism**

The graph‑metrics analytics pass builds function‑ and module‑level feature vectors:

* Degrees: `in_degree`, `out_degree`, `total_degree`.
* Fan‑in/fan‑out variants.
* Centralities: PageRank and possibly simple betweenness/closeness (even before the richer NetworkX work).
* Structural flags: `is_leaf`, `in_cycle`, `cycle_group`, approximate layers in the DAG condensation.
* Coupling scores: based on symbol‑use edges and shared imports.

Your newer NetworkX metrics (extended centralities, SCC, k‑core, etc.) are layered on top of these, using the `networkx` library as per your `networkx.md` reference and the implementation plan we already walked through.

**Outputs / usage**

* Joined into `docs.v_function_architecture` and `docs.v_module_architecture` as the “where does this live in the graph?” columns.

---

### 8.2 Subsystems: `subsystems.*`, `subsystem_modules.*`

This is the “cluster modules into higher‑level components” stage.

**Inputs**

* Import graph: `import_graph_edges.*`. 
* Module attributes: `module_profile.*` (size, risk, coverage, tags/owners). 
* Module graph metrics: `graph_metrics_modules.*`.

**Mechanism (in two phases)**

1. **Clustering (subsystem assignment)**

   * Build a module graph from `import_graph_edges.*` with weights reflecting import frequency, call density, or symbol coupling.
   * Optionally condense SCCs using `cycle_group` (modules in import cycles are strongly tied).
   * Run a clustering / label propagation algorithm (currently NetworkX‑backed) over a weighted, undirected “affinity” graph to assign each module a `subsystem_id`.
   * Emit `subsystem_modules.*` with fields like `subsystem_id`, `module`, `path`, `is_entrypoint?`, `in_cycle?`.

2. **Subsystem summarization**

   Aggregate per‑module data into `subsystems.*`:

   * Membership & size: `module_count`, `file_count`, `total_loc`, `function_count`, `class_count`.
   * Risk & coverage: `high_risk_function_count`, aggregated `risk_level`/score, `avg_module_coverage_ratio`, `tested_function_ratio`.
   * Entry‑point detection: modules with high **external fan‑in** (from other subsystems) + API‑ish tags become `entrypoint_modules`/`entrypoint_examples`.

**Outputs / usage**

* `docs.v_subsystem_summary` — one row per subsystem summarizing size, risk, coverage, entrypoints.
* `docs.v_module_with_subsystem` — module rows annotated with subsystem metadata.
* `docs.v_module_architecture` and `docs.v_function_architecture` pull subsystem fields into module/function rows for LLM consumption.

---

## 9. Graph validation: `graph_validation.*`

**Inputs**

* GOID registry (`goids.*`).
* AST functions (`ast_nodes.*` / `function_metrics.*`).
* Call graph (`call_graph_*`).
* Modules & import graph.

**Mechanism**

`graphs.validation.run_graph_validations` runs a series of health checks right after graphs are built: 

* `missing_function_goids` — AST functions that never got a GOID.
* `callsite_span_mismatch` — call graph edges where the callsite line falls outside the caller’s function span.
* `orphan_module` — modules with no GOIDs at all.
* Potentially others (naming suggests more checks; the README lists these examples).

Each finding becomes a row with:

* `check_name`, `severity` (`info`/`warning`/`error`),
* `path`, `detail`,
* `context` JSON for structured details,
* `created_at`. 

**Outputs / usage**

* Not strictly required for analytics, but extremely useful for tooling to surface “analysis warnings”.
* Lets you gate or flag parts of the repo where the graphs or GOIDs are known to be incomplete or suspicious.

---

## 10. Docs views: `docs.v_*` (LLM‑facing denormalization)

These are DuckDB views, not physical tables, but they’re *the* public contract for your server/MCP.

From the README:

* `docs.v_function_summary` — joins `goid_risk_factors.*`, `function_metrics.*`, `function_types.*`, docstrings, tags, owners.
* `docs.v_call_graph_enriched` — call graph edges enriched with caller/callee function metadata + risk scores + evidence JSON.
* `docs.v_function_architecture` — your “monster row per function” view joining `function_profile.*`, function graph metrics, module profile, subsystem context, and now CFG/DFG + test‑graph metrics.
* `docs.v_module_architecture` — module‑level view joining `module_profile.*`, `graph_metrics_modules.*`, subsystem membership.
* `docs.v_subsystem_summary` — one row per subsystem pulled from `subsystems.*`.
* `docs.v_module_with_subsystem` — convenience view combining modules with their subsystem metadata.

On top of those, you now also have:

* `docs.v_cfg_block_architecture` and `docs.v_dfg_block_architecture` for block‑level zoom‑in on CFG/DFG structure, both keyed by `(function_goid_h128, block_idx)` and joining block metrics with function context.

These are what your FastAPI server and MCP tools actually query; all the heavy lifting we’ve just walked through is designed to feed into these docs views so an LLM never has to recompute any of it.

---

If you’d like, the next natural zoom‑in would be to:

* walk **risk_score** end‑to‑end (showing how each `risk_component_*` is computed from its source tables), or
* spell out the exact **join paths** and keys for a couple of “canonical queries” your agent will issue (e.g. “high‑risk, low‑coverage API entrypoints in a given subsystem”).


# ingestion scope #

Here I’m treating “ingestion scope” as **everything from “point at a repo + external tools” up to “we have core.* and graph.* tables + raw evidence tables in DuckDB”** – i.e., before graph metrics, risk scores, profiles, docs views, etc.

I’ll walk phase‑by‑phase and call out:

* **Inputs** (files, tools, tables)
* **Mechanics / code responsibility**
* **Outputs** (exact tables / JSONL)

---

## 1. Repo scan & module registry

### 1.1 `services.enrich.scan` → `repo_map.json`, `modules.jsonl`

**Inputs**

* Repo root path + commit.
* Project config (ignore patterns, overlays, etc.).

**Mechanics**

`CodeIntel.services.enrich.scan` walks the filesystem, identifies Python packages/modules, and builds two main artifacts:

1. **Module rows**

   * For each discovered Python module (essentially a dotted name + file path) it calls `build_module_row` and writes a JSONL row.
   * It also applies tagging rules (from `tags_index.yaml`) to attach **tags** and **owners** per module.

2. **Repo map**

   * One JSON document describing:

     * `repo` and `commit`.
     * Mapping of **module → path**.
     * Overlay and checksum info.
   * Downstream, this is used to:

     * Join SCIPs and coverage lines to canonical repo paths.
     * Drive import‑graph construction and module‑level analytics.

**Outputs**

* `enriched/modules/modules.jsonl` → becomes `core.modules` / exported `modules.jsonl` in Document Output. Columns: `module`, `path`, `repo`, `commit`, `language`, `tags`, `owners`.
* `enriched/repo_map/repo_map.json` → exported as `repo_map.json`.

These are the “what exists in the repo?” and “where is it?” baselines.

---

## 2. AST & CST ingestion

### 2.1 AST indexer → `ast_nodes.*` (+ `ast_metrics.*`)

**Inputs**

* Python source files from the scan.
* Repo + commit context.

**Mechanics**

`CodeIntel.enrich.ast_indexer.AstIndexer` parses each Python file into LibCST/AST and walks it, emitting **one row per syntactic node**:

* Parse phase:

  * Read file text.
  * Build LibCST tree.
* Index phase:

  * For each node of interest (module, class, function, async function, comprehension, etc.), record:

    * `node_type`
    * `name` / `qualname`
    * `start_line`, `end_line`
    * `parent_qualname`
    * `docstring`, `decorators`, etc.

`ast_metrics` is technically analytics, but it’s computed “right on the heels” of AST indexing and is a direct aggregation of counts and depth over `ast_nodes.*` (node counts, function/class counts, depth/complexity). These feed risk and profiles later.

**Outputs**

* `core.ast_nodes` (+ exported `ast_nodes.parquet`/`ast_nodes.jsonl`).
* `analytics.ast_metrics` (+ JSONL).

---

### 2.2 CST builder → `cst_nodes.*`

**Inputs**

* Same Python files as AST.
* Repo/commit context.

**Mechanics**

The `cst_cli` wrapper under `ingestion/` builds a **concrete syntax tree** index:

* Uses LibCST to retain exact formatting and tokens.
* Emits per‑node rows with:

  * `node_id` (stable internal identifier)
  * `kind`
  * `file_path`
  * `start_line`, `end_line`
  * `parent_id`
  * a small `text_preview` / serialized node snippet.

This is intentionally richer than AST (keeps whitespace/comments) and is wired into the GOID crosswalk as a future join target via `cst_node_id`.

**Outputs**

* `core.cst_nodes` (+ JSONL mirror).

---

## 3. GOID registry & crosswalk

### 3.1 GOID registry → `core.goids`

**Inputs**

* `ast_nodes.*` (functions, methods, classes, modules, CFG blocks).
* Repo + commit name.

**Mechanics**

`CodeIntel.enrich.goid_builder.GOIDBuilder` walks the AST index and for each entity that should have a stable identity, it constructs a **GOID descriptor**:

* Fields that feed the GOID:

  * `repo`
  * `commit`
  * `language` (here: `python`)
  * `rel_path`
  * `kind` (`module`, `class`, `function`, `method`, `cfg_block`, etc.)
  * `qualname` (normalized, dotted)
  * `start_line`, `end_line`
* It then computes a **128‑bit hash** (`goid_h128`) and a human‑readable `urn`, and writes one row per entity into `core.goids`.

This becomes **the primary key everywhere** (call graph, CFG/DFG, coverage, tests, metrics).

**Outputs**

* `core.goids` / `goids.parquet` / `goids.jsonl`. Columns: `goid_h128`, `urn`, and the descriptor fields above.

---

### 3.2 GOID crosswalk → `core.goid_crosswalk`

**Inputs**

* GOID registry.
* AST data, repo map, SCIP index, CST index, etc.

**Mechanics**

The crosswalk is a **join hub**. For each GOID, it records all the “addresses” where that entity shows up:

* Source‑side info:

  * `urn` + `goid_h128`
  * `repo`, `commit`, `language`
  * `rel_path`, `start_line`, `end_line`
  * `module`, `qualname`
* Evidence hooks:

  * `scip_symbol` (set during SCIP ingestion)
  * `cst_node_id`
  * `chunk_id` (for embeddings)
  * other tool‑specific IDs as needed.

It’s populated incrementally:

1. From AST/GOID build: fill the “AST side” (paths, spans, qualnames).
2. From SCIP ingestion: fill `scip_symbol` where ranges match.
3. From any embedding/run: fill `chunk_id`.

**Outputs**

* `core.goid_crosswalk` (+ JSONL).

This is what lets you go “from coverage line → GOID”, “from SCIP symbol → GOID”, etc.

---

## 4. SCIP symbol ingestion & symbol‑use edges

### 4.1 Running `scip-python` → `index.scip.json`

**Inputs**

* Repo root.
* scip‑python binary path (from config).

**Mechanics**

A wrapper in `ingestion/` or `utils/`:

1. Runs `scip-python index ../src` to produce a binary `.scip` index.
2. Runs `scip print --json` to obtain `index.scip.json`.

This JSON holds, per file:

* `relative_path`
* `occurrences` – symbol IDs + ranges + roles (def/ref)
* `symbols` – metadata for each symbol.

**Outputs**

* `index.scip.json` in Document Output.

---

### 4.2 Filling `goid_crosswalk.scip_symbol`

**Inputs**

* `index.scip.json`
* `core.goid_crosswalk` (AST spans).

**Mechanics**

An enrichment function:

* For each **definition** occurrence in SCIP:

  * Look up the matching GOID via `(rel_path, start_line..end_line)`.
  * When matched, write the SCIP symbol string into `goid_crosswalk.scip_symbol`.

Ambiguities (overlapping spans, macros, etc.) are handled conservatively (no write if not sure). This is the link that lets you connect symbol‑use edges back to **functions / modules** via GOID.

---

### 4.3 Symbol‑use graph → `graph.symbol_use_edges`

**Inputs**

* `index.scip.json` (def and ref occurrences).
* `modules.jsonl` (to map file paths → modules).
* `goid_crosswalk` (to connect SCIP symbols to GOIDs).

**Mechanics**

`uses_builder.build_use_graph`:

1. For each symbol `S`:

   * Identify the defining file (`def_path`) and the set of using files (`use_path`).
2. For each def→use pair:

   * Compute booleans:

     * `same_file` (`def_path == use_path`)
     * `same_module` (join via `modules.jsonl`).
3. Optionally, join back to GOIDs using `goid_crosswalk.scip_symbol`.

**Outputs**

* `graph.symbol_use_edges` (+ JSONL). Columns (simplified): `symbol`, `def_path`, `use_path`, `same_file`, `same_module`, plus possible GOID references for def/use.

This is the **semantic coupling** baseline later used to derive `symbol_coupling` and module‑level symbol graphs.

---

## 5. Structural graph ingestion (call graph, CFG/DFG, imports)

### 5.1 Call graph → `graph.call_graph_nodes`, `graph.call_graph_edges`

**Inputs**

* AST (`ast_nodes` + extra call‑site extraction).
* GOIDs (functions / methods).
* SCIP symbol data for resolution hints.

**Mechanics**

`graphs/CallGraphBuilder` walks the AST of each function and finds call expressions:

1. For each call expression:

   * Determine caller GOID from current function context (`caller_goid_h128`).
   * Attempt to resolve the callee:

     * Using AST name resolution (local imports).
     * Using SCIP symbol mapping (if available via `goid_crosswalk.scip_symbol`).
   * Record:

     * `caller_goid_h128`
     * `callee_goid_h128` (nullable when unresolved)
     * `callsite_path`, `callsite_line`, `callsite_col`.
2. Deduplicate edges per `(caller_goid_h128, callee_goid_h128, callsite_path, callsite_line, callsite_col)` and sort for deterministic output.

Nodes table just ensures that all GOID’d functions appear even if they have no edges.

**Outputs**

* `graph.call_graph_edges` (+ JSONL) – directed `caller → callee` edges, with callsite metadata and optional SCIP evidence.
* `graph.call_graph_nodes` – node list keyed by `function_goid_h128`.

These are the raw edges your later `graph_metrics_functions.*` feed off.

---

### 5.2 Control‑flow graph (CFG) → `graph.cfg_blocks`, `graph.cfg_edges`

**Inputs**

* Per‑function AST (via `collect_function_info`).
* Function GOIDs.

**Mechanics**

`CodeIntel.enrich.cfg.CFGBuilder` does, for each function:

1. **Block splitting**

   * Walk the function’s AST.
   * Cut basic blocks at control constructs:

     * `if` / `elif` / `else`
     * loops
     * `break`, `continue`
     * `try` / `except` / `finally`
   * Synthesize an **entry** and **exit** block.
   * Assign each block a stable `block_idx` and `block_id` (`<function-goid>:block<idx>`).

2. **Block serialization**

   * For each block, record:

     * `function_goid_h128`
     * `block_idx`, `block_id`
     * `file_path`, `start_line`, `end_line`
     * `kind` (`entry`, `body`, `exit`, `handler`)
     * `stmts_json` – serialized AST metadata of the statements.
     * `in_degree`, `out_degree` – computed while building.

3. **Edge construction**

   * For each control edge:

     * Write `src_block_idx`, `dst_block_idx`.
     * Tag `edge_type` as one of:

       * `fallthrough`, `true`, `false`, `loop`, `exception`.
     * For conditional edges, serialize the guard expression into `cond_json`.

**Outputs**

* `graph.cfg_blocks` – block‑level structure.
* `graph.cfg_edges` – intra‑procedural control edges.

---

### 5.3 Data‑flow graph (DFG) → `graph.dfg_edges`

**Inputs**

* CFG blocks & edges.
* Per‑block AST (same function data).

**Mechanics**

After CFG is built, the same builder performs a **def‑use walk** over variables:

1. Track **definitions** of variables per block.
2. For each use, connect it back to the defining block:

   * `src_block_idx` = definition’s block.
   * `dst_block_idx` = use’s block.
3. Recognize phi‑like situations:

   * For variables merged from multiple predecessors, mark `via_phi = true`.
4. Record symbol names and use kind.

**Outputs**

* `graph.dfg_edges` (+ JSONL). Columns:

  * `function_goid_h128`
  * `src_block_idx`, `dst_block_idx`
  * `src_symbol`, `dst_symbol`
  * `via_phi` (bool)
  * `use_kind` (`read`, `write`, `update`, …)

This is strictly **intra‑procedural**; inter‑procedural flow is represented via call and import graphs.

---

### 5.4 Import graph → `graph.import_graph_edges`

**Inputs**

* LibCST imports from AST index.
* `modules.jsonl` (for resolving modules).

**Mechanics**

`graph.io.write_import_edges` builds a **module‑level import graph**:

1. For each module, collect the set of modules it imports.
2. Emit directed edges:

   * `src_module` (importing module)
   * `dst_module` (imported module)
3. Precompute:

   * `src_fan_out` – number of imported modules.
   * `dst_fan_in` – number of importers.
   * `cycle_group` – SCC ID for the strongly connected component containing the edge nodes.

**Outputs**

* `graph.import_graph_edges` (+ JSONL). This is your **raw module dependency topology**.

---

## 6. Config & static config ingestion → `analytics.config_values.*`

**Inputs**

* Config sources (YAML/TOML/env/ini/etc.), provided via pipeline config.
* Codebase AST (to find uses).
* `modules.jsonl` for module names.

**Mechanics**

The config indexer pipelines (`index_config_files`, `prepare_config_state`) do two things:

1. Parse config files into key/value pairs with:

   * `key_path` (e.g. `service.database.url`)
   * `raw_value`, `value_type`
2. Scan code to find **reads/writes** of config keys:

   * Map each usage to a module/file.
   * Count references per module/key.

It writes a **per‑key, per‑module** view of config usage.

**Outputs**

* `analytics.config_values.*` (+ JSONL). Columns (summarized in README):

  * `config_key`, `module`, `rel_path`
  * usage counts, read/write flags
  * possibly owning repo/commit.

These later get used to compute config graphs and risk around “config hot spots”, but at ingestion it’s just the raw map.

---

## 7. Coverage & test ingestion

### 7.1 Coverage lines → `analytics.coverage_lines.*`

**Inputs**

* Coverage DB / XML from your test runner (typically `coverage.py`).
* `goid_crosswalk` (to map lines → GOIDs).

**Mechanics**

The `coverage-detailed` analytics CLI (wired via `enrich_analytics`) is technically “analytics”, but at this stage it’s doing pure ingestion from an external tool:

1. Read raw coverage data: per file, which lines were executed.
2. For each `(file, line)`:

   * Use `goid_crosswalk` ranges to find owning function GOID (if any).
3. Emit one row per `(file_path, line_no)` with:

   * `is_covered` bool
   * owning GOID where available.

**Outputs**

* `analytics.coverage_lines.*` (+ JSONL) – file‑ & line‑level ground truth coverage.

### 7.2 Coverage functions / test catalog / test‑function edges

These are “analytics” by naming, but still primarily ingestion of external artifacts:

**Inputs**

* Raw coverage lines (above).
* Test runner outputs (pytest JSON, JUnit XML, etc.).
* `goids` and `goid_crosswalk` for mapping.

**Mechanics**

Test analytics pipeline (`test-analytics`) does:

1. Build `test_catalog.*`

   * One row per test:

     * `test_id` (string)
     * `path`, `module`, `nodeid`
     * classification flags (unit/integration/slow/xfail, etc.).

2. Build `test_coverage_edges.*`

   * Construct a **bipartite graph**: tests ↔ function GOIDs.
   * For each test and function:

     * Join coverage lines to functions to compute `coverage_ratio` (fraction of lines hit).
     * Emit edge: `(test_id, function_goid_h128, coverage_ratio)`.

3. Build `coverage_functions.*`

   * For each function GOID:

     * Aggregate coverage lines to get a single `coverage_ratio`.
     * Boolean `tested` if any coverage.
     * Counts of tests touching, failing tests, etc.

**Outputs**

* `analytics.test_catalog.*`
* `analytics.test_coverage_edges.*`
* `analytics.coverage_functions.*`

All of these later get joined into `function_profile`, `module_profile`, and doc views.

---

## 8. Typedness & static diagnostics ingestion

Even though these are under `analytics.*`, they’re very much “read external tool outputs and structure them”.

### 8.1 Typedness → `analytics.typedness.*`, `analytics.function_types.*`

**Inputs**

* AST index (function signatures).
* Type annotations present in the code.

**Mechanics**

`services.enrich.function_types` walks AST function defs and computes:

* For each function:

  * Count `total_params` (excluding `self`/`cls`).
  * Count `annotated_params` vs `unannotated_params`.
  * Check for `return_annotation`, record `return_type`.
  * Derive:

    * `param_typed_ratio`
    * `fully_typed`, `partial_typed`, `untyped`
    * `typedness_bucket` (`typed` / `partial` / `untyped`)
    * `typedness_source` (`annotations`, `mixed`, `unknown`).

Aggregates are then rolled up per file into `typedness.*` (file‑level typedness ratios).

**Outputs**

* `analytics.function_types.*` (per‑function typedness ground truth).
* `analytics.typedness.*` (per‑file summary).

---

### 8.2 Static diagnostics → `analytics.static_diagnostics.*`

**Inputs**

* Output from your static type checkers (pyright / pyrefly).
* File mapping from scan.

**Mechanics**

`services.enrich.exports.write_typedness_output` and related functions:

1. Run type checkers (usually from Prefect step / CLI).
2. Parse their diagnostics into a normalized per‑file schema:

   * `file_path`
   * `error_count`, `warning_count`, maybe `info_count`
   * type of diagnostic (type error, import error, etc.).

This is essentially a **structured export** of type checker output keyed by repo/commit/path.

**Outputs**

* `analytics.static_diagnostics.*` (+ JSONL).

---

## 9. Where ingestion stops and analytics begin

Putting it in one mental picture:

* **Ingestion & indexing**

  * Repo scan → `repo_map`, `modules`.
  * AST/CST → `ast_nodes`, `cst_nodes`, `ast_metrics`.
  * GOIDs → `goids`, `goid_crosswalk`.
  * SCIP index → `index.scip.json`, `goid_crosswalk.scip_symbol`.
  * Graphs → `call_graph_*`, `cfg_*`, `dfg_edges`, `import_graph_edges`, `symbol_use_edges`.
  * External evidence:

    * Config → `config_values.*`.
    * Coverage → `coverage_lines.*` (+ `coverage_functions.*`).
    * Tests → `test_catalog.*`, `test_coverage_edges.*`.
    * Typedness → `function_types.*`, `typedness.*`.
    * Static diagnostics → `static_diagnostics.*`.

* **Analytics / derived**

  * Function metrics, risk factors, profiles.
  * Graph metrics & subsystems.
  * CFG/DFG graph metrics, test graph metrics, config graph metrics (the NetworkX heavy stuff).
  * Docs views (`docs.v_*`).

Everything above in sections 1–8 is the “raw evidence ingestion” side: given a repo and external tools, these are the exact mechanisms that turn that into **structured rows in DuckDB** that your later analytics and LLM‑facing layers build on top of.







