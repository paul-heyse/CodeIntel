Here’s a narrative of the **`src/codeintel` architecture** as it looks now, tying it back to the `.jsonl` / `.parquet` artifacts in your Document Output.

I’ll walk it as:

1. Big‑picture dataflow
2. Core storage + IDs
3. Ingestion & indexing layer
4. Graph construction layer
5. Analytics & “architecture understanding”
6. Export & serving
7. How each `src/codeintel` subpackage fits

---

## 1. Big‑picture mental model

At a high level, `codeintel` is a **pipeline that turns a repo at `<repo, commit>` into a knowledge graph of the codebase** stored in DuckDB, then exported as Parquet/JSONL for LLMs and tools. 

The main phases are:

1. **Scan & index** the repo (AST, CST, config, SCIP, coverage, tests).
2. **Assign canonical IDs (GOIDs)** to all entities and crosswalk them to every evidence source. 
3. **Build graphs**: call graph, control‑flow graph (CFG), data‑flow graph (DFG), import graph, SCIP symbol‑use graph.
4. **Run analytics**: metrics, typedness, coverage, test impact, config usage, risk scores, architecture metrics, subsystems, and the function/file/module profiles that aggregate all of this.
5. **Export** everything to `Document Output/` as Parquet + JSONL via the `docs_export` layer and `generate_documents.sh`.

The CLI + orchestration code wires these phases together into single commands like `enrich_pipeline all` or focused analytics commands such as `coverage-detailed`, `risk-factors`, `test-analytics`, etc.

---

## 2. Core storage & identifiers

Everything revolves around **DuckDB** and **GOIDs**.

### 2.1 GOID registry (`core.goids`)

* Built by `CodeIntel.enrich.goid_builder.GOIDBuilder`, which walks the AST index and assigns each entity (module, function, method, class, CFG block) a **128‑bit hash ID** (`goid_h128`). 
* GOID encodes: repo, commit, language, relative path, entity kind, normalized qualname, and span.
* This ID is the foreign key used across call graphs, CFG/DFG, coverage, tests, analytics, etc.

### 2.2 GOID crosswalk (`core.goid_crosswalk`)

* A “join hub” mapping a GOID URN back to all the **evidence sources** that mentioned that thing: AST path + line range, module path, SCIP symbol, CST node ID, chunk ID used for embeddings, etc. 

* This is what lets you go:

  > “Given a SCIP symbol / CST node / coverage line / chunk ID, which function GOID is this?”

* In particular, `scip_symbol` is filled when SCIP ingestion correlates SCIP definitions to GOID spans; that’s what links symbol‑use edges back to concrete functions.

### 2.3 Storage schema

Although the schema definitions live in the `storage/` package, the README reflects how they’re organized:

* **`core.*`** – entity registries and crosswalks (`goids`, `goid_crosswalk`, `modules`, `repo_map`, etc.).
* **`graph.*`** – structural graphs (call graph, CFG/DFG, import graph, SCIP symbol uses).
* **`analytics.*`** – metrics, coverage, risk, tests, hotspots, graph metrics, subsystems, config, static diagnostics.
* **`docs.*`** – higher‑level views and denormalized profiles used directly by LLMs and UIs (e.g., `docs.v_function_architecture`, `docs.v_module_architecture`, subsystem summaries, function/file/module profiles).

---

## 3. Ingestion & indexing layer

This layer lives mostly in **`ingestion/`, `services/`, and `enrich`** modules and produces the low‑level artifacts.

### 3.1 Repo scan & module registry

* `CodeIntel.services.enrich.scan` walks the repo and produces:

  * `repo_map.json` – repo ID, commit, module→path map, overlay configuration.
  * `modules.jsonl` – one row per module with tags/owners metadata.

* These feed import graph construction, module‑level analytics, and architecture views.

### 3.2 AST extraction & AST metrics

* `CodeIntel.enrich.ast_indexer.AstIndexer` parses every Python file into LibCST/AST and writes `ast_nodes.*` (one row per node with type, qualname, span, parent, docstring). 
* `CodeIntel.enrich.analytics.ast_metrics` aggregates that into `ast_metrics.*` (node counts, function/class counts, depth, complexity). 

These AST datasets are prerequisites for GOIDs, call graph, CFG/DFG, hotspots, function metrics, and more.

### 3.3 CST extraction

* `CodeIntel.cst_build.cst_cli` emits `cst_nodes.jsonl`, capturing exact syntax spans, tokens, parent stacks, and text previews. 
* Future fields (`cst_node_id` in `goid_crosswalk`) are reserved to join GOIDs back to CST with full fidelity.

### 3.4 GOID registry & crosswalk

* `GOIDBuilder` uses `ast_nodes.*` to emit `goids.*` and `goid_crosswalk.*`.
* Each AST entity becomes:

  * A `goid_h128` row in `core.goids`.
  * One or more crosswalk rows, giving its file path, module path, AST qualname, span, and placeholders for SCIP/CST/chunk IDs.

This is what lets all later stages be “GOID‑first.”

### 3.5 SCIP index ingestion

* `scip-python` is run against the repo to produce a binary `.scip` index and `index.scip.json`; CodeIntel treats that JSON as the **language‑agnostic symbol graph**. 
* During ingestion (implemented under `ingestion/scip_*`), CodeIntel:

  * Writes `index.scip.json` to the build directory and Document Output.
  * Registers it in DuckDB (e.g., a `scip_index_view`).
  * Walks SCIP documents to align definition locations (file + line) with GOID spans and **updates `core.goid_crosswalk.scip_symbol`**.

That SCIP ↔ GOID bridge is what allows `symbol_use_edges` to be joined back to concrete functions.

### 3.6 Config, static config, and config usage

* A config indexer (under `config/` + `services/enrich`) parses YAML/TOML/JSON/INI/ENV files and normalizes keys, producing `config_values.*`. 
* Each row says: *this config key* (e.g., `service.database.host`) appears in *this config file* and is referenced by *these code files/modules*.

This gives you a “config dependency graph” that can be joined to modules, call graph, and risk metrics.

### 3.7 Coverage & test ingestion

The CLI entrypoints `coverage-detailed` and `test-analytics` orchestrate coverage + test ingestion.

* **Line coverage** (`coverage_lines.*`): reads Coverage.py DB with dynamic contexts enabled. 
* **Function coverage** (`coverage_functions.*`): groups coverage lines by GOID spans, giving per‑function coverage ratios.
* **Test catalog** (`test_catalog.*`): parses pytest JSON report and joins tests to GOIDs when possible.
* **Test coverage edges** (`test_coverage_edges.*`): bipartite edges linking tests to the functions they executed (with per‑edge coverage ratios).

### 3.8 Static diagnostics & typedness

* **File‑level typedness** (`typedness.jsonl`) comes from `CodeIntel.services.enrich.exports.write_typedness_output`, combining error counts from Pyrefly/Pyright and annotation ratios. 
* **Static diagnostics** (`static_diagnostics.*`) aggregate type‑checker error counts per file based on `FileTypeSignals`.

These feed into risk scoring and typedness analytics.

---

## 4. Graph construction layer

Once AST, GOIDs, and SCIP are in place, the **graph builders** in the `graphs/` (and related) package construct various graph tables.

### 4.1 Call graph (`graph.call_graph_nodes`, `graph.call_graph_edges`)

* Builder: `CodeIntel.enrich.callgraph.CallGraphBuilder`. 

* Inputs:

  * AST & scope info per file (`collect_python_files`).
  * Import resolution via `_ImportResolver`.
  * Optional SCIP signal to improve matches across modules.

* Outputs:

  * **Nodes** keyed by `goid_h128` with callable kind, arity, `is_public`, and file path. 
  * **Edges** keyed by `(caller, callee, line, column)` with resolution provenance (`resolved_via` = `scip`/`scope`/`heuristic`), confidence score, and `evidence_json` for extra context. 

This is the structural backbone for architecture metrics and impact analysis.

### 4.2 Control‑flow graph (CFG) (`graph.cfg_blocks`, `graph.cfg_edges`)

* Builder: `CodeIntel.enrich.cfg.CFGBuilder`. 

* For each function GOID, it:

  * Splits the function into **basic blocks** on control constructs.
  * Synthesizes entry/exit blocks.
  * Assigns each block a `block_idx` and canonical `block_id`.

* Outputs:

  * `cfg_blocks`: per‑block span, label, kind, and degree counts, plus serialized AST of statements.
  * `cfg_edges`: edges between blocks with `edge_type` (`fallthrough`, `true`, `false`, `loop`, `exception`) and optional guard AST.

### 4.3 Data‑flow graph (DFG) (`graph.dfg_edges`)

* Same builder (`CFGBuilder`) does a def‑use walk over each function, creating intra‑procedural data‑flow edges: **definitions to uses**, optionally via phi‑like merges. 
* Each edge: `(function_goid_h128, src_block_idx, dst_block_idx, src_symbol, dst_symbol, via_phi, use_kind)`. 

### 4.4 Import graph (`graph.import_graph_edges`)

* Built from LibCST import metadata via `CodeIntel.enrich.graph.io.write_import_edges`.
* Edges:

  * `src_module` → `dst_module` with `src_fan_out`, `dst_fan_in`, and SCC `cycle_group`. 

This becomes the **module‑level architecture** graph.

### 4.5 SCIP symbol‑use graph (`graph.symbol_use_edges`)

* Builder: `CodeIntel.uses_builder.build_use_graph`, using `index.scip.json` as the source of definitions and references.
* Each edge row says: symbol S defined in `def_path` is used in `use_path`, with flags `same_file` and `same_module`. 
* Combined with `goid_crosswalk.scip_symbol`, this gives you **def→use relationships tied back to concrete functions and modules**.

---

## 5. Analytics & architecture understanding

This layer lives mainly in **`analytics/` and `services/analytics`**, and it’s where the “raw graphs” are turned into something architecture‑aware and risk‑aware.

### 5.1 AST‑derived analytics

* `ast_metrics.*` – per‑file node counts, class/function counts, depth, complexity. 
* `hotspots.jsonl` – git‑churn + AST‑complexity based “hot” files. 

These feed into risk and file/module profiles.

### 5.2 Typedness & type analytics

* File‑level `typedness.*` and per‑function `function_types.*` capture parameter and return annotation coverage, typedness buckets, and return types.

Used by risk factors and profiles to flag untyped areas.

### 5.3 Function metrics

* `function_metrics.*` computes per‑function LOC, complexity, nesting depth, counts of returns/raises/yields, docstring presence, etc. 

This is both a direct signal (“which functions are structurally complex?”) and an input to risk scoring and architecture clustering.

### 5.4 Coverage + tests analytics

* `coverage_functions.*` is the canonical per‑function coverage view.
* `test_catalog.*` + `test_coverage_edges.*` let you traverse from tests to functions, including how much of a function each test covers.

These are the dynamic‑behavior side of the architecture picture.

### 5.5 Config & static diagnostics

* `config_values.*` expose per‑key config usage and reference counts across modules. 
* `static_diagnostics.*` holds aggregated type‑checker error counts per file.

These are used as risk inputs and for “blast radius” queries.

### 5.6 Graph metrics & subsystems (new architecture datasets)

From the README “New architecture datasets”: 

* **Graph metrics** (`graph_metrics_functions.*`, `graph_metrics_modules.*`):

  * Compute graph‑theoretic features over **call and import graphs**: fan‑in/fan‑out, degree counts, PageRank/centralities, cycle membership, layering, symbol coupling, etc.
  * Stored under `analytics.*` and exposed via `docs.v_function_architecture` and `docs.v_module_architecture`.

* **Subsystems** (`subsystems.*`, `subsystem_modules.*`):

  * Cluster modules into **subsystems** based on imports/usage and overlay them with risk and coverage rollups.
  * Exposed via `docs.v_subsystem_summary` and `docs.v_module_with_subsystem`.

These are the main “architecture‑aware” features of the new code: they turn raw graphs into **higher‑level components and dependency structure**, summarized for LLM consumption.

### 5.7 Risk factors

* `goid_risk_factors.*` aggregates per‑function risk signals from:

  * function metrics & types
  * coverage & tests
  * hotspots & typedness
  * static diagnostics

  via the `risk-factors` analytics CLI.

* Output includes `risk_score` (0–1), `risk_level` (low/medium/high), and tagged components.

### 5.8 Profiles (function / file / module)

These are built by `analytics/profiles.*` and orchestrated by `ProfilesStep` in `orchestration/steps.py`.

* **`function_profile.*`**:

  * Denormalized per‑function record: metrics, typedness, coverage, tests, call graph fan‑in/out, risk scores, docstrings, tags/owners. 

* **`file_profile.*`**:

  * Per‑file aggregate over function profiles plus AST metrics, hotspots, typedness, static diagnostics, coverage, and ownership. 

* **`module_profile.*`**:

  * Module‑level view aggregating function/file profiles plus import graph fan‑in/out and cycle info.

Together with graph metrics & subsystems, these profiles are the **primary “ready‑to-feed‑to‑LLM” architecture surfaces**.

---

## 6. Export & serving

### 6.1 Docs export & Document Output

* A dedicated `docs_export/` package plus `generate_documents.sh` handle:

  1. Copying Parquet tables out of the DuckDB catalog into `enriched/...`.
  2. Creating **JSONL mirrors** with `duckdb COPY` for each graph/analytics table.

* The README you attached is effectively the **contract** for those exported datasets: column schema, purpose, and origin module for each.

### 6.2 Server & MCP integration

From the directory layout:

* `server/` exposes the DuckDB database and docs views over an HTTP or RPC API, so UIs and tools can query function profiles, call graphs, architecture metrics, etc.
* `mcp/` wraps those queries in **Model Context Protocol** endpoints so ChatGPT (or other LLM tools) can fetch code intel on demand.
* `stubs/` provides type stubs for internal APIs so that static checkers and LSPs can reason about CodeIntel itself.

(These roles are inferred from naming and the dataset contracts, rather than spelled out verbatim in the README.)

---

## 7. How each `src/codeintel` subpackage fits

Based on the dataset origins and naming, this is how the subdirectories you zipped line up with the architecture:

* **`analytics/`**
  Houses cross‑cutting analytics and aggregations:

  * Graph metrics & subsystem discovery for architecture views. 
  * Profiles builders (`profiles.build_function_profile`, `build_file_profile`, `build_module_profile`).
  * Possibly helpers used by risk‑factors and other CLI analytics.

* **`graphs/`**
  Implementation of graph builders:

  * Call graph (`CallGraphBuilder`) and its IO. 
  * CFG/DFG logic (`CFGBuilder`).
  * Import graph writer (`graph.io.write_import_edges`). 
  * SCIP symbol‑use builder (`uses_builder.build_use_graph`).

* **`ingestion/`**
  Low‑level ingestion of raw evidence:

  * AST/CST pipelines (wrapping `AstIndexer`, `cst_cli`).
  * SCIP ingestion and crosswalk updates.
  * Coverage ingestion (`coverage-detailed`) and config indexing (`index_config_files`, `prepare_config_state`).

* **`services/`**
  “Domain services” that combine ingestion + storage:

  * `services.enrich.scan`, `services.enrich.analytics.hotspots`, `services.enrich.exports.write_typedness_output`, `services.enrich.function_metrics`, `services.enrich.function_types`, etc.
  * These typically read core/graph tables and write analytics tables.

* **`cli/`**
  CLI entrypoints and command groups:

  * `enrich_pipeline all` – full pipeline (AST, GOIDs, graphs, analytics). 
  * `enrich goids|callgraph|cfg|dfg` – focused structural passes. 
  * `enrich_analytics coverage-detailed|test-analytics|risk-factors` – analytics passes.

* **`orchestration/`**
  Pipeline wiring and “steps”:

  * Defines `ProfilesStep` and likely siblings that encapsulate each major stage (scan, goids, graphs, analytics, docs export).
  * Coordinates shared context (repo, commit, paths, tools) across steps.

* **`docs_export/`**

  * Responsible for exporting DuckDB tables to Parquet/JSONL and defining `docs.*` views like `v_function_architecture`, `v_module_architecture`, `v_subsystem_summary`, and `v_module_with_subsystem`.

* **`config/`**

  * Holds config models for the pipeline (paths, repo identifiers, coverage & test sources, tool locations) and helpers used by CLI/orchestration.
  * Works together with the config indexer that feeds `config_values.*`. 

* **`storage/`**

  * DuckDB connection management, table schemas (columns, types, PKs, indexes), and generic I/O helpers used across ingestion, graphs, and analytics.
  * Encapsulates where the metadata DB lives on disk and how it’s opened/closed.

* **`models/`**

  * Typed models / dataclasses for config objects, table rows, and in‑memory representations of AST/graph elements (e.g., GOID descriptors, function profile DTOs).

* **`server/`**

  * HTTP/RPC server exposing docs/analytics views and graph traversals to external tools.

* **`mcp/`**

  * MCP tool shims that call into `server/` or directly into `storage/` + `analytics/` to respond to LLM tool calls.

* **`stubs/`**

  * Type stubs for CodeIntel’s own public API surface or dependencies used in the pipeline.

* **`utils/`**

  * Shared utilities: logging, path handling, subprocess helpers (e.g., running `scip-python`), concurrency helpers, small abstraction glue.

---

## 8. End‑to‑end: what happens on a full run

Putting it all together, a typical **“full architecture export”** looks like:

1. **CLI** (`cli/`) parses flags and builds a configuration object (repo root, commit, build dir, output dir, tool paths).

2. **Orchestration** (`orchestration/`) runs a series of steps:

   * Scan repo (`services.enrich.scan`) → `repo_map`, `modules`.
   * AST + CST extraction (`AstIndexer`, `cst_cli`) → `ast_nodes`, `ast_metrics`, `cst_nodes`.
   * GOID registry + crosswalk (`GOIDBuilder`) → `goids`, `goid_crosswalk`.
   * SCIP ingestion → `index.scip.json` + `goid_crosswalk.scip_symbol`.
   * Graph builders → call graph, CFG/DFG, import graph, symbol_use_edges.
   * Analytics & risk steps → typedness, function metrics/types, hotspots, coverage, tests, config, static diagnostics, risk_factors.
   * Architecture metrics & subsystems → `graph_metrics_*`, `subsystems_*` and docs architecture views. 
   * Profiles step → `function_profile`, `file_profile`, `module_profile`.

3. **Docs export** (`docs_export/`) copies all relevant tables to Parquet and JSONL in `Document Output/` with exactly the schemas described in `README_METADATA.md`.

4. **Server/MCP** optionally loads that DB and exposes queries into these views to your tools and agents.

---

If you’d like, I can next zoom in on any specific layer (for example, “how subsystems are inferred from graph metrics” or “how the architecture views `docs.v_*` are constructed from the underlying tables”) and spell out that evidence path the way we did for SCIP and call‑graph edges.


# subsystem inference from graph metrics #

Subsystems are inferred by taking the **module‑level import graph**, enriching it with **graph metrics and risk data**, and then clustering modules into “architectural neighborhoods”. I’ll walk through what we can see concretely from your metadata, and then how the clustering is likely implemented mechanically.

> One important caveat: I don’t have the actual `subsystems` Python code in front of me, only the exported tables + README. So I’ll describe:
>
> * what we know for sure from the exported schemas, and
> * what the inference pipeline *must* be doing conceptually, based on those schemas and the other graph metrics.

---

## 1. What “subsystems” are in CodeIntel’s data model

From the README:

* **Graph metrics** (`graph_metrics_functions.*`, `graph_metrics_modules.*`) are *graph‑theoretic signals over call and import graphs*—fan‑in/out, degree counts, PageRank/centralities, cycle membership, layers, symbol coupling—stored in `analytics.*` and surfaced via `docs.v_function_architecture` / `docs.v_module_architecture`. 
* **Subsystems** (`subsystems.*`, `subsystem_modules.*`) are *inferred architectural clusters of modules plus risk rollups, membership, and entrypoint hints*, exposed via `docs.v_subsystem_summary` and `docs.v_module_with_subsystem`. 

So from the outside you have:

* A **module graph** from `import_graph_edges.*` (directed edges, SCC `cycle_group`, fan‑in/out). 
* **Per‑module analytics** from `module_profile.*` (coverage/risk/import fan‑in/out, cycle membership, tags/owners).
* **Per‑module graph metrics** from `graph_metrics_modules.*` (centralities, coupling, layers, etc.). 
* **Subsystem assignments** in `subsystem_modules.*` and subsystem‑level rollups in `subsystems.*`. 

Mechanically, subsystem inference is “just” the pass that uses the first three to produce the last two.

---

## 2. Inputs to the subsystem builder

### 2.1 Import graph edges

`import_graph_edges.*` is the normalized directed module import graph: 

* `src_module`, `dst_module`
* `src_fan_out`, `dst_fan_in`
* `cycle_group` (SCC ID)

This gives you the **raw topology** of module dependencies and strongly‑connected components (SCCs), which are a natural starting point for subsystems: in practice, modules in the same import cycle almost always belong to the same architectural “lump”.

### 2.2 Module profiles

`module_profile.*` aggregates function/file metrics and import graph data per module:

* Size / complexity: `file_count`, `total_loc`, `function_count`, `avg_file_complexity`, `max_file_complexity`
* Risk & coverage: `high_risk_function_count`, `module_coverage_ratio`
* Import structure: `import_fan_in`, `fan_out`, `cycle_group`, `in_cycle`
* Ownership: `tags`, `owners`

These give you **attributes** for each node in the module graph.

### 2.3 Module graph metrics

`graph_metrics_modules.*` isn’t fully expanded in the README, but the description is explicit: **fan‑in/out, degree counts, PageRank/centralities, cycle membership, layers, symbol coupling** over the import and call graphs. 

So per module you likely have:

* Degrees: in/out/total degree
* Centralities: PageRank, maybe betweenness/closeness
* Structural flags: “leaf module”, “hub”, “bridge”
* Coupling scores: how tightly it’s connected to neighboring modules, possibly weighted by call frequency or symbol sharing

All of that gives you a **feature vector** per module that describes where it sits in the dependency topology.

### 2.4 Optional extra signals

Based on the other tables, the subsystem builder can also join in:

* **Hotspots** (`hotspots.jsonl` – churn + complexity) per file, rolled up into module_profile.
* **Risk factors** (`goid_risk_factors.*`), which roll up per‑function risk from coverage/complexity/tests/typedness and are then aggregated per module in `module_profile`.
* **Tags/owners** (tags_index and modules registry) for boundaries like “api”, “infra”, “ml”, etc.

Those don’t define subsystems, but they’re used when summarizing and ranking them.

---

## 3. Conceptual algorithm: from graph metrics to subsystems

From the metadata, the subsystem pass is doing three conceptual things:

1. **Condense & weight the module graph** using graph metrics.
2. **Cluster** that graph into cohesive groups (subsystems).
3. **Summarize** each group with risk/topology/entrypoint info.

I’ll break those down in the order they’d typically be implemented.

### 3.1 Build a weighted module graph

Starting from `import_graph_edges.*`, the builder constructs an in‑memory graph:

* Nodes = modules from `core.modules` / `module_profile.*`
* Edges = `src_module -> dst_module` from `import_graph_edges.*` with attributes:

  * `weight` (could be 1, or derived from number of imports or call density)
  * `cycle_group` and maybe “same cycle” flags

Then it augments nodes with graph metrics:

* Join `graph_metrics_modules.*` onto module nodes (centralities, coupling).
* Join `module_profile.*` (risk, coverage, size, file count, fan‑in/out, tags, owners).

This lets the builder answer questions like:

* “Which modules form tight, highly coupled components?”
* “Which modules act as shared infrastructure or gateways (high centrality)?”
* “What’s the risk profile of each module?”

Even without seeing the exact code, we know this enrichment must happen, because the final `subsystems.*` table includes **risk rollups and entrypoint hints**, which can only be computed if the subsystems pass already has those signals.

### 3.2 Pre‑group obvious clusters via cycles and connectivity

Before you run any fancier clustering, the pipeline can exploit structural facts already materialized:

* `cycle_group` on `import_graph_edges.*` and `module_profile.*` gives SCC membership.

A typical pattern here is:

1. Treat each **strongly connected component** (cycle group) as a “proto‑subsystem”.
2. Optionally **contract** each SCC into a super‑node in a condensed DAG, with edges between SCC super‑nodes representing inter‑subsystem imports.
3. Work with that condensed graph for higher‑level clustering.

That avoids having a single call to an SCC splitting apart deeply entangled modules, which would not be architecturally honest.

### 3.3 Compute similarity / affinity between modules or SCCs

Given a (possibly condensed) module graph with metrics, the builder needs a notion of “these belong together”. It can derive various **affinity signals** (the exact formula isn’t documented, but the sources are):

* **Import proximity:** “distance” along the import graph, or number of shared neighbors.
* **Coupling strength:** symbol‑level usage from `symbol_use_edges.*` (shared symbols, same modules), plus call graph fan‑in/fan‑out.
* **Topological similarity:** similar degrees and centrality values from `graph_metrics_modules.*` (e.g., group together a chain of leaf modules in the same area). 
* **Shared tags/owners:** two modules both tagged `api` or owned by the same team are more likely to be in the same subsystem.

Mechanically, this usually looks like:

* For each pair of directly linked modules (or SCCs), compute one or more affinity scores:

  * `affinity_import = 1` if the edge is inside a cycle, smaller otherwise.
  * `affinity_calls = normalized call graph edge count` between them.
  * `affinity_tags = 1` if they share tags/owners, 0 otherwise.
* Combine into a single **edge weight** (e.g., weighted sum or product).

The result is a weighted undirected graph (even though imports are directed), where heavier edges mean “these modules probably belong in the same subsystem”.

### 3.4 Cluster the weighted graph into subsystems

The builder then runs some clustering/community detection algorithm over that weighted graph to produce **cluster IDs** (`subsystem_id`) that define:

* A set of modules that are:

  * tightly connected,
  * mostly reusing each other, and
  * relatively loosely connected to the rest of the graph.

The README doesn’t say *which* algorithm is used (could be Louvain, Leiden, hierarchical clustering, spectral methods, etc.), so I can’t claim a specific one. But the contract (`subsystems.*` + `subsystem_modules.*`) implies:

* Each module is assigned to **at most one** subsystem, otherwise `subsystem_modules.*` would need multi‑membership semantics.
* The clustering is **module‑centric**, not function‑centric (the tables and docs views are explicitly module‑level).

When the clustering is done, the pipeline writes:

* `subsystem_modules.*`: rows like

  ```text
  subsystem_id, module, path, is_entrypoint?, in_cycle?, ...
  ```

* which are then joined into `docs.v_module_with_subsystem`.

---

## 4. Subsystem summarization: building `subsystems.*`

Once each module has a `subsystem_id`, the builder aggregates per‑module data into **subsystem‑level summaries**, which is where “risk rollups, membership, and entrypoint hints” come from.

### 4.1 Membership & basic size metrics

Group `subsystem_modules.*` by `subsystem_id` and aggregate:

* `module_count`
* `file_count` (sum of `module_profile.file_count`)
* `total_loc` / `total_logical_loc`
* `function_count`, `class_count`

These let `docs.v_subsystem_summary` answer “how big is this subsystem?” directly.

### 4.2 Risk & coverage rollups

Join `subsystem_modules.*` → `module_profile.*` and `function_profile.*` / `goid_risk_factors.*` to compute:

* `high_risk_function_count` (sum over modules)
* Subsystem‑level `risk_score` or buckets (max or weighted average of module risk metrics)
* Coverage view: `avg_module_coverage_ratio`, `tested_function_ratio`, etc.

This is how subsystems end up with “risk rollups” in `subsystems.*`.

### 4.3 Entry‑point detection

The README calls out “entrypoint hints” on subsystems. 

Given the available data, the logic for “entrypoint module(s)” almost certainly looks something like:

* For each subsystem:

  * Consider its modules’ **import fan‑in/out** and **centralities**.
  * Classify:

    * Modules with **high external fan‑in** (imported by many modules outside the subsystem) as *“public surface / imports entrypoints”*.
    * Modules with **high centrality within the subsystem** but low external fan‑out as internal “core”.
    * Modules tagged `api`, `cli`, `routes`, etc. as HTTP/CLI entrypoints.

Concretely, this can be implemented as:

1. For each module:

   * `internal_fan_in` / `internal_fan_out` vs `external_fan_in` / `external_fan_out`, computed by splitting `import_graph_edges.*` edges into “inside subsystem” vs “to other subsystems”.
   * Combine with PageRank or similar centrality from `graph_metrics_modules.*`.
2. Mark modules that satisfy some heuristic (e.g., high external fan‑in + API tag) as `is_entrypoint = true` in `subsystem_modules.*`.

Those flags then get lifted into `subsystems.*` as:

* `entrypoint_modules` (list)
* `entrypoint_examples` or similar “hint” fields.

Even though the exact thresholds aren’t documented, the presence of fan‑in/out, centralities, tags, and the “entrypoint hints” wording strongly suggest this pattern.

---

## 5. How this shows up in the docs views

Finally, the **docs views** stitch subsystems into the rest of the architecture story:

* `docs.v_module_architecture`:

  * module‑level row including import/topology metrics, risk, and *which subsystem* the module belongs to (join `module_profile.*` + `graph_metrics_modules.*` + `subsystem_modules.*`).
* `docs.v_subsystem_summary`:

  * one row per subsystem with:

    * membership size,
    * aggregate risk/coverage,
    * tags/owners rollups,
    * entrypoint hints (names of likely public modules).
* `docs.v_module_with_subsystem`:

  * convenience view for UIs: each module row is annotated with its subsystem metadata and key metrics.

These are what your MCP tools and UI will actually query when asking questions like:

> “Which subsystems have high risk and low coverage?”
> “Where does this module sit in the architecture, and what subsystem is it part of?”
> “What’s the likely ingress point into subsystem X?”

---

## 6. Summary

Mechanistically, subsystem inference in this architecture is:

1. Use `import_graph_edges.*` to build the **module import graph** and SCCs. 
2. Enrich each module node with **graph metrics** (`graph_metrics_modules.*`), import fan‑in/out, size, coverage, risk, tags, and ownership (`module_profile.*`, `goid_risk_factors.*`).
3. Build a **weighted undirected “affinity” graph** where heavier edges indicate stronger architectural coupling.
4. Run a **clustering/community detection algorithm** over that graph to assign each module to a `subsystem_id`, yielding `subsystem_modules.*`. 
5. Aggregate per‑module data into **subsystem‑level summaries** with:

   * size/complexity,
   * risk and coverage rollups,
   * entrypoint candidates based on external fan‑in and tags.
6. Expose everything through `docs.v_function_architecture`, `docs.v_module_architecture`, `docs.v_subsystem_summary`, and `docs.v_module_with_subsystem` for LLMs and tools. 

# docs.v_* overview and data and analytics reshaping #

The `docs.v_*` views are basically the **presentation layer** of CodeIntel: they sit on top of the “raw” analytics tables and reshape them into things that are easy for UIs and LLM tools to consume.

I’ll walk through:

1. The general pattern for `docs.*` views
2. `docs.v_function_summary`
3. `docs.v_call_graph_enriched`
4. The *architecture* views:

   * `docs.v_function_architecture`
   * `docs.v_module_architecture`
   * `docs.v_subsystem_summary`
   * `docs.v_module_with_subsystem` 

Where I don’t have the exact SQL, I’ll spell out the mechanics (which tables join where, and what the view is conceptually doing) rather than claiming precise column lists.

---

## 1. General pattern: how `docs.*` views are built

All of the `docs.*` views follow the same strategy:

* **Base tables** live in `core.*`, `graph.*`, and `analytics.*`.
* **Profiles** (`function_profile.*`, `file_profile.*`, `module_profile.*`) already denormalize a lot of signals per entity. 
* **Graph metrics** (`graph_metrics_functions.*`, `graph_metrics_modules.*`) and **subsystems** (`subsystems.*`, `subsystem_modules.*`) add architecture / topology context. 
* The code in `storage/views.py` then defines **SQL views** under the `docs` schema, each of which:

  * Picks one primary table (e.g. `analytics.function_profile`) as the “row spine”.
  * LEFT JOINs in any related tables on stable keys:

    * `function_goid_h128` for function‑level joins
    * `module` for module‑level joins
    * `subsystem_id` for subsystem joins
  * Drops backend‑only fields, renames columns to friendlier names, and sometimes computes small derived flags.

From the outside, you never see the underlying `analytics.*` tables directly—you query the `docs.*` view, which is designed as the stable contract for tools and LLMs. 

---

## 2. `docs.v_function_summary`

**Purpose:** one row per function GOID, combining “everything you’d want to know” about that function into a single record.

**Underlying tables:**

* `analytics.goid_risk_factors` – the core per‑function risk record (coverage, tests, typedness, hotspots, static diagnostics, risk_score/level). 
* `analytics.function_metrics` – structural metrics: LOC, complexity, parameter counts, etc. 
* `analytics.function_types` – typedness and return‑type details. 
* `core.docstrings` / `ast_nodes` – docstring summaries and long text. 
* `analytics.tags_index` / `modules` – tags and owners. 

**Mechanics (conceptually):**

```sql
CREATE VIEW docs.v_function_summary AS
SELECT
  r.function_goid_h128,
  r.urn,
  r.repo,
  r.commit,
  r.rel_path,
  r.language,
  r.kind,
  r.qualname,
  -- structural metrics
  m.loc,
  m.logical_loc,
  m.cyclomatic_complexity,
  m.complexity_bucket,
  -- typedness
  t.typedness_bucket,
  t.return_type,
  t.param_typed_ratio,
  -- coverage / tests / risk
  r.coverage_ratio,
  r.tested,
  r.test_count,
  r.failing_test_count,
  r.hotspot_score,
  r.static_error_count,
  r.risk_score,
  r.risk_level,
  -- docs + ownership
  d.short_desc      AS doc_short,
  d.long_desc       AS doc_long,
  mod.tags,
  mod.owners
FROM analytics.goid_risk_factors r
LEFT JOIN analytics.function_metrics m
  ON m.function_goid_h128 = r.function_goid_h128
LEFT JOIN analytics.function_types t
  ON t.function_goid_h128 = r.function_goid_h128
LEFT JOIN core.docstrings d
  ON d.rel_path = r.rel_path AND d.qualname = r.qualname
LEFT JOIN analytics.module_profile mod
  ON mod.module = d.module;
```

*(Column names and exact joins vary in the real code, but mechanically this is what’s happening.)*

This view is the **core “function lens”** many tools use before you get into architecture‑specific information.

---

## 3. `docs.v_call_graph_enriched`

**Purpose:** enrich raw call graph edges with human‑readable caller/callee info and risk context.

**Underlying tables:**

* `graph.call_graph_edges` – raw edges (`caller_goid_h128`, `callee_goid_h128`, callsite path/line/col, resolution metadata, evidence_json). 
* `core.goids` – for both caller and callee: URN, rel_path, qualname. 
* `analytics.goid_risk_factors` – per‑function risk for caller and callee. 

**Mechanics:**

```sql
CREATE VIEW docs.v_call_graph_enriched AS
SELECT
  e.caller_goid_h128,
  gc.urn           AS caller_urn,
  gc.rel_path      AS caller_rel_path,
  gc.qualname      AS caller_qualname,
  rc.risk_level    AS caller_risk_level,
  rc.risk_score    AS caller_risk_score,

  e.callee_goid_h128,
  gcallee.urn      AS callee_urn,
  gcallee.rel_path AS callee_rel_path,
  gcallee.qualname AS callee_qualname,
  rcallee.risk_level AS callee_risk_level,
  rcallee.risk_score AS callee_risk_score,

  e.callsite_path,
  e.callsite_line,
  e.callsite_col,
  e.language,
  e.kind,
  e.resolved_via,
  e.confidence,
  e.evidence_json
FROM graph.call_graph_edges e
LEFT JOIN core.goids gc
  ON gc.goid_h128 = e.caller_goid_h128
LEFT JOIN analytics.goid_risk_factors rc
  ON rc.function_goid_h128 = e.caller_goid_h128
LEFT JOIN core.goids gcallee
  ON gcallee.goid_h128 = e.callee_goid_h128
LEFT JOIN analytics.goid_risk_factors rcallee
  ON rcallee.function_goid_h128 = e.callee_goid_h128;
```

This is the view you’d use to ask things like “show me all high‑risk callees of this function” or “what does this unresolved call look like, with SCIP evidence attached?”.

---

## 4. `docs.v_function_architecture`

**Purpose:** give a **function‑level view of architecture**, combining function profile, graph metrics, subsystem context, and module profile in one row. 

> From the README: graph metrics are “stored under `analytics.*` and surfaced via `docs.v_function_architecture`”, and subsystems are “exposed via `docs.v_subsystem_summary` and `docs.v_module_with_subsystem`.” 

**Underlying tables:**

* `analytics.function_profile` – spine row: function metrics, coverage, tests, typedness, risk, docs, tags/owners. 
* `analytics.graph_metrics_functions` – per‑function call‑graph metrics (fan‑in/out, centralities, maybe depth/layer). 
* `analytics.module_profile` – module‑level risk/coverage/import metrics, for context. 
* `analytics.subsystem_modules` + `analytics.subsystems` – mapping modules → subsystems and subsystem summaries (used to attach subsystem_id and high‑level subsystem risk). 

**Join keys:**

* `function_profile.function_goid_h128` ↔ `graph_metrics_functions.function_goid_h128`
* `function_profile.module` ↔ `module_profile.module`
* `module_profile.module` ↔ `subsystem_modules.module`
* `subsystem_modules.subsystem_id` ↔ `subsystems.subsystem_id`

**Mechanics (conceptually):**

```sql
CREATE VIEW docs.v_function_architecture AS
SELECT
  f.function_goid_h128,
  f.urn,
  f.rel_path,
  f.module,
  f.qualname,

  -- basic structural + risk info from function_profile
  f.loc,
  f.logical_loc,
  f.cyclomatic_complexity,
  f.risk_score,
  f.risk_level,
  f.coverage_ratio,
  f.tested,
  f.tests_touching,
  f.failing_tests,

  -- call-graph topology from graph_metrics_functions
  gf.call_fan_in,
  gf.call_fan_out,
  gf.call_pagerank,
  gf.call_is_leaf,
  gf.layer_index,

  -- module context (size / risk / imports)
  mp.module_coverage_ratio,
  mp.import_fan_in,
  mp.import_fan_out,
  mp.in_cycle,
  mp.cycle_group,

  -- subsystem context
  sm.subsystem_id,
  s.name             AS subsystem_name,
  s.risk_level       AS subsystem_risk_level,
  s.module_count     AS subsystem_module_count,

  -- ownership
  f.tags,
  f.owners,

  f.created_at       AS snapshot_at
FROM analytics.function_profile f
LEFT JOIN analytics.graph_metrics_functions gf
  ON gf.function_goid_h128 = f.function_goid_h128
LEFT JOIN analytics.module_profile mp
  ON mp.module = f.module
LEFT JOIN analytics.subsystem_modules sm
  ON sm.module = f.module
LEFT JOIN analytics.subsystems s
  ON s.subsystem_id = sm.subsystem_id;
```

Practically: this is the view you query when you want to understand **“where does this function sit in the architecture, how important is it, and how risky is it?”**, without manually joining 4–5 tables.

---

## 5. `docs.v_module_architecture`

**Purpose:** the **module‑centric analogue** of `v_function_architecture`—summarizing topology, risk, coverage, and subsystem membership for each module. 

**Underlying tables:**

* `analytics.module_profile` – spine row: size, complexity, coverage, risk, import fan‑in/out, cycle_group, tags/owners. 
* `analytics.graph_metrics_modules` – per‑module graph metrics from call/import graphs (in/out degree, centrality, layer index, coupling). 
* `analytics.subsystem_modules` + `analytics.subsystems` – subsystem membership and rollups. 

**Join keys:**

* `module_profile.module` ↔ `graph_metrics_modules.module`
* `module_profile.module` ↔ `subsystem_modules.module`
* `subsystem_modules.subsystem_id` ↔ `subsystems.subsystem_id`

**Mechanics:**

```sql
CREATE VIEW docs.v_module_architecture AS
SELECT
  mp.module,
  mp.path,
  mp.file_count,
  mp.total_loc,
  mp.function_count,
  mp.class_count,

  -- risk & coverage
  mp.high_risk_function_count,
  mp.module_coverage_ratio,

  -- import graph topology
  mp.import_fan_in,
  mp.import_fan_out,
  mp.cycle_group,
  mp.in_cycle,

  -- graph metrics (call + import centralities, coupling)
  gm.import_pagerank,
  gm.call_pagerank,
  gm.symbol_coupling,
  gm.layer_index,

  -- subsystem
  sm.subsystem_id,
  s.name           AS subsystem_name,
  s.risk_level     AS subsystem_risk_level,
  s.module_count   AS subsystem_size,

  -- ownership
  mp.tags,
  mp.owners,
  mp.created_at    AS snapshot_at
FROM analytics.module_profile mp
LEFT JOIN analytics.graph_metrics_modules gm
  ON gm.module = mp.module
LEFT JOIN analytics.subsystem_modules sm
  ON sm.module = mp.module
LEFT JOIN analytics.subsystems s
  ON s.subsystem_id = sm.subsystem_id;
```

This view is what you’d hit for questions like:

* “Which modules are central hubs with lots of dependents?”
* “Which subsystem does this module belong to, and how risky is it overall?”
* “Show me modules in cycles with high risk and low coverage.”

---

## 6. `docs.v_subsystem_summary`

**Purpose:** one row per subsystem, aggregating all the module‑ and function‑level signals into a **component‑level summary**. 

**Underlying tables:**

* `analytics.subsystems` – spine row: ID, name/label (if any), risk rollups, maybe pre‑computed coverage and size metrics. 
* `analytics.subsystem_modules` – membership list (which modules are in each subsystem, plus entrypoint flags). 
* `analytics.module_profile` – per‑module size, coverage, risk, fan‑in/out. 
* (Optionally) `analytics.function_profile` / `goid_risk_factors` for function‑level rollups.

**Join keys:**

* `subsystems.subsystem_id` ↔ `subsystem_modules.subsystem_id`
* `subsystem_modules.module` ↔ `module_profile.module`

**Mechanics:**

```sql
CREATE VIEW docs.v_subsystem_summary AS
WITH membership AS (
  SELECT
    sm.subsystem_id,
    COUNT(*)                    AS module_count,
    SUM(mp.file_count)          AS file_count,
    SUM(mp.total_loc)           AS total_loc,
    SUM(mp.high_risk_function_count) AS high_risk_function_count,
    AVG(mp.module_coverage_ratio)     AS avg_coverage,
    SUM(CASE WHEN sm.is_entrypoint THEN 1 ELSE 0 END) AS entrypoint_module_count,
    ARRAY_AGG(CASE WHEN sm.is_entrypoint THEN sm.module END IGNORE NULLS)
      AS entrypoint_modules
  FROM analytics.subsystem_modules sm
  LEFT JOIN analytics.module_profile mp
    ON mp.module = sm.module
  GROUP BY sm.subsystem_id
)
SELECT
  s.subsystem_id,
  s.name,
  s.risk_level,
  s.risk_score,
  m.module_count,
  m.file_count,
  m.total_loc,
  m.high_risk_function_count,
  m.avg_coverage,
  m.entrypoint_module_count,
  m.entrypoint_modules,
  s.created_at AS snapshot_at
FROM analytics.subsystems s
LEFT JOIN membership m
  ON m.subsystem_id = s.subsystem_id;
```

This is the view your tools would use to show the **macro‑architecture**: a list of subsystems, their risk and coverage status, size, and suggested entrypoints.

---

## 7. `docs.v_module_with_subsystem`

**Purpose:** convenience view: **module profile + subsystem annotations** in one place, for UIs that are primarily module‑centric. 

**Underlying tables:**

* `analytics.module_profile` – module‑level metrics.
* `analytics.subsystem_modules` – subsystem membership.
* `analytics.subsystems` – subsystem summary (for name/risk).

**Join keys:**

* `module_profile.module` ↔ `subsystem_modules.module`
* `subsystem_modules.subsystem_id` ↔ `subsystems.subsystem_id`

**Mechanics:**

```sql
CREATE VIEW docs.v_module_with_subsystem AS
SELECT
  mp.*,
  sm.subsystem_id,
  s.name       AS subsystem_name,
  s.risk_level AS subsystem_risk_level
FROM analytics.module_profile mp
LEFT JOIN analytics.subsystem_modules sm
  ON sm.module = mp.module
LEFT JOIN analytics.subsystems s
  ON s.subsystem_id = sm.subsystem_id;
```

Functionally, this is:

* “Give me module_profile, but also tell me what subsystem this is in and how risky that subsystem is.”

---

## 8. How these views fit together

If you zoom out, the architecture views form a tidy hierarchy:

* **Function level:** `docs.v_function_architecture`

  * “What’s the role and risk of this individual function, and how is it placed in the call/import architecture?”

* **File level:** `file_profile.*` (no explicit docs view name, but used under the hood) 

* **Module level:** `docs.v_module_architecture` and `docs.v_module_with_subsystem`

  * “What does this module own (files/functions), what’s its import/call topology, and what subsystem is it in?”

* **Subsystem level:** `docs.v_subsystem_summary`

  * “What are the major components of the codebase, and which ones are big, risky, or under‑tested?”

And orthogonally you have **edges**:

* `docs.v_call_graph_enriched` – function‑to‑function edges with risk and evidence.
* Underlying `import_graph_edges.*` – module‑to‑module edges that feed the module/subsystem views. 

All of this is designed so that a client can:

* Start from a subsystem (`v_subsystem_summary`) → drill into modules (`v_module_architecture`) → drill into functions (`v_function_architecture`) → inspect specific caller/callee links (`v_call_graph_enriched`), without ever manually touching the raw `analytics.*` tables.

---
# Detailed mechanistic overview of graph metrics and other analytics #

Got you — let’s zoom all the way in on **how the graph metrics and the remaining analytics are actually produced**, step by step.

I’ll group things so it’s easier to orient:

1. Graph metrics (functions + modules)
2. AST‑driven analytics (ast_metrics, function_metrics, hotspots)
3. Typedness + static diagnostics (typedness, function_types, static_diagnostics)
4. Coverage + tests (coverage_lines, coverage_functions, test_catalog, test_coverage_edges)
5. Risk aggregation (goid_risk_factors)
6. Profiles (function_profile, file_profile, module_profile)
7. Config + tags (config_values, tags_index)
8. Symbol‑use analytics (symbol_use_edges)

I’ll keep the focus on *mechanics*: inputs → transforms → outputs, and I’ll call out where I’m inferring behavior from the dataset contracts vs. what’s explicitly stated in `README_METADATA.md`.

---

## 1. Graph metrics

### 1.1 Function graph metrics (`graph_metrics_functions.*`)

**Purpose**: add call‑graph‑level topology signals for each function: fan‑in/out, degrees, PageRank/centrality, layers, maybe cycle membership. These are later surfaced via `docs.v_function_architecture`.

**Inputs**

* `graph.call_graph_edges.*` – static call edges with caller/callee GOIDs.
* `graph.call_graph_nodes.*` – callable nodes; at minimum provides `goid_h128`, kind, and `rel_path`.

**Mechanics (algorithm)**

1. **Construct adjacency lists**

   From `call_graph_edges`:

   * For each edge row with non‑null callee:

     * `caller = caller_goid_h128`
     * `callee = callee_goid_h128`

   * Build:

     ```text
     out_neighbors[caller] += {callee}
     in_neighbors[callee]  += {caller}
     ```

     Using sets to dedupe multiple calls between same pair.

2. **Compute fan‑in / fan‑out**

   For each function `f`:

   * `call_fan_out = |out_neighbors[f]|`
   * `call_fan_in  = |in_neighbors[f]|`

   These are exported directly in `function_profile` and/or `graph_metrics_functions`.

3. **Compute degree counts**

   A function’s degree is:

   * `in_degree = call_fan_in`
   * `out_degree = call_fan_out`
   * `total_degree = in_degree + out_degree`

   These may be stored explicitly as `call_in_degree`, `call_out_degree`, etc., in the graph metrics table.

4. **Run centrality (PageRank)**

   The README explicitly says graph metrics include “PageRank/centralities”.

   Mechanically that looks like:

   * Treat each function GOID as a node in a directed graph.

   * Use the adjacency lists to run a PageRank‑style power iteration:

     * Initialize `rank[v] = 1 / N` for all nodes.

     * Iterate:

       ```text
       rank_new[v] = (1 - d)/N
                     + d * sum(rank[u] / out_degree[u] for u in in_neighbors[v])
       ```

       where `d` is a damping factor (usually ~0.85).

     * Stop when max change < ε or after K iterations.

   * Persist `rank[v]` as `call_pagerank` or similar.

   The exact library (networkx vs. custom) isn’t stated, but the resulting values and the wording in the README imply a standard PageRank‑style centrality.

5. **Derive call graph layers**

   The README mentions “layers” as part of graph metrics.

   A typical implementation:

   * Collapse **strongly connected components** (SCCs) of the call graph into supernodes (SCC ID per function).
   * Topologically sort this SCC graph.
   * Assign `layer_index` = topological order index for each SCC.
   * Propagate that back to each function as `call_layer`.

   This gives you:

   * Low layer index: “entry-ish” functions with few or no upstream callers.
   * Higher layer index: deeper internals called by many others.

6. **Persist to `analytics.graph_metrics_functions`**

   Build one row per function GOID with columns such as:

   * `function_goid_h128`
   * `call_fan_in`, `call_fan_out`
   * `call_in_degree`, `call_out_degree`, `total_degree`
   * `call_pagerank`
   * `call_layer`
   * (maybe) `call_is_leaf` (`call_fan_out == 0`)

   These values are then joined into `docs.v_function_architecture`.

---

### 1.2 Module graph metrics (`graph_metrics_modules.*`)

**Purpose**: the same idea, but at **module** level over the `import_graph_edges` graph; used by `docs.v_module_architecture`.

**Inputs**

* `graph.import_graph_edges.*` – module import edges with `src_module`, `dst_module`, and `cycle_group`, plus precomputed `src_fan_out` / `dst_fan_in`.
* Optionally: aggregated function call graph data per module (to mix call‑graph centrality into module metrics).

**Mechanics**

1. **Build import adjacency**

   For each row:

   * `src = src_module`
   * `dst = dst_module`

   Build:

   ```text
   imports_out[src] += {dst}
   imports_in[dst]  += {src}
   ```

2. **Import fan‑in / fan‑out**

   * `import_fan_out = |imports_out[module]|`
   * `import_fan_in  = |imports_in[module]|`

   These values are also computed directly in `import_graph_edges` as `src_fan_out` / `dst_fan_in`, but `graph_metrics_modules` can store them in a per‑module table alongside additional metrics.

3. **Centrality on import graph**

   Run PageRank or a similar centrality algorithm over this directed graph:

   * Output `import_pagerank`, and potentially other centralities like in‑/out‑degree centrality.

4. **Compute hierarchical layers**

   Similar process to functions:

   * Collapse SCCs into supernodes (`cycle_group` from `import_graph_edges` is the SCC ID).
   * Topologically sort; assign `import_layer` per SCC.
   * Propagate `import_layer` to each module in the SCC.

5. **Compute coupling**

   The README mentions “symbol coupling” as part of graph metrics.

   Likely mechanics:

   * Use `symbol_use_edges.*` (SCIP def→use edges) to compute module‑to‑module coupling:

     * Map `def_path` / `use_path` to modules.
     * For each `(def_module, use_module)` pair, add to a `coupling_score[def_module]` and/or `coupling_score[use_module]`.

   * Normalize scores (e.g., by number of functions or total symbols) and persist as `symbol_coupling` in `graph_metrics_modules`.

6. **Persist**

   One row per module:

   * `module`
   * `import_pagerank`, `import_in_degree`, `import_out_degree`, `import_layer`
   * `symbol_coupling`
   * Possibly: aggregated call‑graph centrality across its functions

   Joined into `docs.v_module_architecture` and used for subsystem inference.

---

## 2. AST‑driven analytics

### 2.1 `ast_metrics.*`

**Purpose**: per‑file AST summary: node counts, class/function counts, depth, complexity.

**Inputs**

* `ast_nodes.*` – one row per AST node, with node type, parents, line spans.

**Mechanics**

During AST extraction (`AstIndexer`):

1. For each file:

   * Increment `node_count` for every node.
   * When `node_type` is `FunctionDef`/`AsyncFunctionDef`, increment `function_count`.
   * When `node_type` is `ClassDef`, increment `class_count`.

2. Track depth:

   * Maintain current depth during traversal; track `max_depth` per file.
   * Compute `avg_depth` as the average depth over all nodes.

3. Compute complexity:

   * For each control‑flow construct (if/elif/for/while/try/except, boolean operators, comprehensions), increment a `decision_points` counter.
   * Set `complexity = decision_points` or `1 + decision_points`.

4. Emit `ast_metrics` row:

   * `rel_path`, `node_count`, `function_count`, `class_count`, `avg_depth`, `max_depth`, `complexity`, `generated_at`.

---

### 2.2 `function_metrics.*`

**Purpose**: per‑function structural metrics keyed by GOID; this is your “static complexity” view.

**Inputs**

* AST for each function (via `ast_nodes`).
* GOID spans in `goids.*` (to know the function’s file & line range).

**Mechanics**

For each GOID whose `kind` is `function` or `method`:

1. **Locate AST subtree**

   * Use `rel_path`, `start_line`, `end_line` from `goids` to find the matching `FunctionDef` / `AsyncFunctionDef` node in `ast_nodes`.

2. **Compute lines of code**

   * `loc = end_line - start_line + 1`
   * `logical_loc` = count of non‑blank, non‑comment lines within that span (requires scanning source text).

3. **Parameter metrics**

   * Inspect `function.args`:

     * Count positional‑only, positional, keyword‑only parameters.
     * Detect `*args` (`has_varargs`) and `**kwargs` (`has_varkw`).

4. **Control‑flow & complexity**

   * Walk the body, counting:

     * `return` statements → `return_count`
     * `yield` / `yield from` → `yield_count` & `is_generator`
     * `raise` → `raise_count`

   * Complexity:

     * Start with 1, add 1 for each decision point (if, for, while, Boolean op with multiple operands, except, comprehensions, etc.) → `cyclomatic_complexity`.

   * Max nesting depth:

     * Track a depth counter while walking nested control structures → `max_nesting_depth`.

5. **Other flags**

   * `is_async` – from node type.
   * `decorator_count` – length of decorator list.
   * `has_docstring` – check first body statement for `ast.Expr` string.

6. **Bucket complexity**

   * `complexity_bucket` based on thresholds (like low/medium/high).

7. **Persist row**

   Everything above is written with keys describing GOID, repo, commit, path, qualname, etc.

---

### 2.3 `hotspots.jsonl`

**Purpose**: identify “hot” files based on **git churn + complexity**.

**Inputs**

* Git history data (via `git log` / diff parsing).
* `ast_metrics.*` (per‑file complexity).

**Mechanics**

For each file:

1. Run `git log` within a configured time window (or N commits):

   * Count commits touching the file → `commit_count`.
   * Count distinct authors → `author_count`.
   * Summarize diffs:

     * Sum insertions → `lines_added`.
     * Sum deletions → `lines_deleted`.

2. Look up `complexity` from `ast_metrics` for that file.

3. Compute hotspot `score`:

   * Some function of churn + complexity; e.g.,

     ```text
     score = w1 * normalized_complexity
           + w2 * log1p(commit_count)
           + w3 * log1p(author_count)
     ```

   * Exact formula isn’t documented, but README clearly labels the output as a “composite hotspot score used for ranking.”

4. Emit one row per `rel_path` with all metrics.

Hotspot scores are reused later in `goid_risk_factors` and `file_profile` / `module_profile`.

---

## 3. Typedness + static diagnostics

### 3.1 `function_types.*`

**Purpose**: per‑function typedness + signature details: parameter annotations, return type, typedness buckets.

**Inputs**

* AST / LibCST for function signature and type annotations.
* GOID spans to locate functions.
* (Optionally) stub overlays or type comments.

**Mechanics**

For each function GOID:

1. Locate the AST node (as in `function_metrics`).

2. Count “typed” parameters:

   * Skip `self` / `cls` for methods.

   * For each remaining parameter:

     * If `annotation` expression is present → count as annotated; record `param_types[name] = annotation_text`.
     * Otherwise, unannotated.

   * Set:

     * `total_params`
     * `annotated_params`
     * `unannotated_params = total - annotated`

3. Return typing:

   * If function has `returns` annotation, set:

     * `return_type` = rendered annotation
     * `has_return_annotation = True`
     * `return_type_source = "annotation"`

   * Else:

     * `return_type = null`
     * `return_type_source = "unknown"`

4. Ratios & buckets:

   * `param_typed_ratio = annotated_params / total_params` (or 1.0 if no params).

   * Flags:

     * `fully_typed`: all params + return annotated.
     * `partial_typed`: some but not all.
     * `untyped`: no annotations.

   * `typedness_bucket` = `typed` / `partial` / `untyped`.

5. Persist as `function_types` row with GOID + path/qualname metadata.

---

### 3.2 File‑level `typedness.jsonl`

**Purpose**: file‑level view of typedness + static error counts, used for risk and overlays.

**Inputs**

* `function_types.*` / AST‑based annotation counts.
* Static type checker diagnostics: Pyrefly and Pyright.

**Mechanics**

For each file:

1. Combine type checker errors:

   * `pyrefly_errors` and `pyright_errors` from the underlying run (through some `FileTypeSignals` structure).
   * `type_error_count = max(pyrefly_errors, pyright_errors)` or similar.

2. Compute `annotation_ratio`:

   * Aggregate `total_params` / `annotated_params` across all functions in the file.
   * Build `{ "params": ratio, "returns": ratio_or_flag }`.

3. Count untyped functions:

   * Number of `function_types` rows in the file marked `untyped` → `untyped_defs`.

4. Overlay recommendation:

   * `overlay_needed = True` when `untyped_defs` is high or `type_error_count` > 0 (heuristic).

5. Emit `typedness` row with `path`, `type_error_count`, `annotation_ratio`, `untyped_defs`, `overlay_needed`.

---

### 3.3 `static_diagnostics.*`

**Purpose**: simple per‑file totals for static type errors.

**Inputs**

* Raw outputs from Pyrefly and Pyright, aggregated per file into `FileTypeSignals`.

**Mechanics**

For each file:

* Read `pyrefly_errors`, `pyright_errors`.
* Compute `total_errors = pyrefly_errors + pyright_errors`.
* `has_errors = total_errors > 0`.

Persist row with `rel_path` and these counts. These fields are later joined into `goid_risk_factors` and profiles.

---

## 4. Coverage + test analytics

### 4.1 Line coverage (`coverage_lines.*`)

**Purpose**: raw line‑level execution data from `coverage.py` runs; this is the primitive dynamic signal.

**Inputs**

* `.coverage` database with `dynamic_context = test_function` to capture per‑test context.

**Mechanics**

For each measured file:

1. Iterate all executable lines recorded by coverage:

   * `is_executable` from coverage (line present in analysis).
   * `is_covered` from visited lines.
   * `hits` = count (>=1 when covered).

2. For dynamic contexts:

   * `context_count` = number of distinct coverage contexts that touched the line (e.g., test functions).

3. Persist one row per `(rel_path, line)` with repo + commit from `repo_map`.

---

### 4.2 Function coverage (`coverage_functions.*`)

**Purpose**: per‑function coverage computed by grouping `coverage_lines` over GOID spans.

**Inputs**

* `goids.*` – gives `(rel_path, start_line, end_line)` for each function GOID.
* `coverage_lines.*` – line coverage for those paths.

**Mechanics**

For each function GOID:

1. Filter `coverage_lines` where:

   * `rel_path` matches.
   * `line` in `[start_line, end_line]`.

2. Aggregate:

   * `executable_lines = count(is_executable = TRUE)`.
   * `covered_lines    = count(is_covered = TRUE)`.

3. Derived fields:

   * If `executable_lines > 0`:

     * `coverage_ratio = covered_lines / executable_lines`.
     * `tested = covered_lines > 0`.
     * `untested_reason = ""` or `no_tests`.

   * Else:

     * `coverage_ratio = NULL`.
     * `tested = FALSE`.
     * `untested_reason = "no_executable_code"`.

4. Emit `coverage_functions` row with GOID and these metrics.

---

### 4.3 Test catalog (`test_catalog.*`)

**Purpose**: canonical set of pytest tests with metadata; this is the test‑side of test–function relationships.

**Inputs**

* `pytest-json-report` output (full run).
* GOID lookups for test functions.

**Mechanics**

For each test node in the pytest JSON:

1. Base fields:

   * `test_id` = pytest nodeid (`path::TestClass::test_func[param]`).
   * `status`, `duration_ms`, `markers`, `parametrized` (from JSON).
   * `flaky` = presence of `flaky` marker.

2. Match to GOID:

   * Use the path and qualname to map the test function to a GOID in `goids.*`.
   * If found, set `test_goid_h128` and `urn`. Otherwise leave null.

3. Persist row per test.

---

### 4.4 Test coverage edges (`test_coverage_edges.*`)

**Purpose**: bipartite edges linking tests to functions they executed.

**Inputs**

* `coverage_lines.*` with dynamic context per test.
* `test_catalog.*` – resolved test IDs.
* `goids.*` – function spans.

**Mechanics**

1. For each coverage context (test function):

   * For each covered line, look up which function GOID spans that `(rel_path, line)` — using GOID spans.

2. For each distinct `(test_id, function_goid_h128)` pair:

   * Aggregate:

     * `covered_lines` = count of lines covered in that function by that test.
     * `executable_lines` from `coverage_functions` for that GOID.

   * Compute `coverage_ratio = covered_lines / executable_lines` if `executable_lines > 0`.

3. Attach test metadata:

   * `last_status` from `test_catalog` (pass/fail/error).

4. Persist `test_coverage_edges` row with GOID and test identifiers.

These edges are where “which tests touch this function?” and “which functions does this test cover?” come from.

---

## 5. Risk aggregation (`goid_risk_factors.*`)

**Purpose**: one record per function GOID combining *everything* we’ve talked about so far into a single risk vector.

**Inputs**

* `function_metrics.*` – structure/complexity.
* `function_types.*` and `typedness.*` – typedness of function/file.
* `hotspots.*` – file‑level churn metrics.
* `static_diagnostics.*` – errors per file.
* `coverage_functions.*` – per‑function coverage.
* `test_coverage_edges.*` + `test_catalog.*` – test counts, failing tests, last status.
* Module tags/owners from `modules.*` / `tags_index`.

**Mechanics**

For each function GOID:

1. Join all relevant tables by GOID or path/qualname, pulling:

   * `loc`, `logical_loc`, `cyclomatic_complexity`, `complexity_bucket`.
   * `typedness_bucket`, `typedness_source`, plus file `annotation_ratio`.
   * `hotspot_score` from the file containing the function.
   * `static_error_count` and `has_static_errors` from `static_diagnostics`.
   * `executable_lines`, `covered_lines`, `coverage_ratio`, `tested` from `coverage_functions`.
   * From `test_coverage_edges` + `test_catalog`:

     * `test_count`, `failing_test_count`, `last_test_status`.

2. Compute a numeric `risk_score`:

   * Weighted combination of components, e.g.:

     * Up‑weights high complexity, low coverage, high hotspot_score, static errors, untypedness.
     * Down‑weights well‑covered, typed, low‑churn functions.

   * Exact formula isn’t in the README but the schema clearly labels `risk_component_*` weights in `function_profile` and a final `risk_score` / `risk_level` here.

3. Bucket into `risk_level`: `low`, `medium`, `high` based on numeric thresholds.

4. Attach tags and owners, and persist.

This table is the core feed for `function_profile` and `v_function_architecture`.

---

## 6. Profiles

Profiles are *denormalized* views built in a `ProfilesStep` that sit between raw analytics and the docs views.

### 6.1 `function_profile.*`

**Purpose**: “ready to use” record per function — metrics + coverage + tests + risk + docs + call graph degrees.

**Inputs**

* `goid_risk_factors.*` – main risk vector.
* `function_metrics.*`, `function_types.*`.
* `coverage_functions.*`, `test_coverage_edges.*`, `test_catalog.*`.
* `call_graph_edges.*` / `call_graph_nodes.*` – for call fan‑in/out and leaf flag.
* `core.docstrings` – doc_short/doc_long.
* `modules.*` / `tags_index` – tags/owners.

**Mechanics**

For each GOID:

1. Start with `goid_risk_factors` row.

2. Add:

   * Detailed param counts, varargs flags from `function_metrics`.
   * Return type and param typed ratio from `function_types`.
   * Coverage / tests numbers: `tested`, `tests_touching`, `failing_tests`, `slow_tests` etc.
   * Call fan‑in/out and `call_is_leaf` via counts over `call_graph_edges`.
   * `doc_short` / `doc_long` from docstrings.
   * Tags/owners from module metadata.

3. Bring through `risk_score`, `risk_level` and individual risk component weights.

4. Persist one row per function GOID.

This is what `docs.v_function_architecture` starts from, then adds graph metrics + subsystem context.

---

### 6.2 `file_profile.*`

**Purpose**: per‑file aggregate across AST metrics, typedness, hotspots, function risk, coverage, and ownership.

**Inputs**

* `ast_metrics.*` – AST size/complexity.
* `typedness.*` – file typedness.
* `static_diagnostics.*`.
* `hotspots.*`.
* `function_profile.*` – underlying function risks and coverage.
* Tags/owners from module metadata.

**Mechanics**

For each `rel_path`:

1. Join `ast_metrics`, `typedness`, `static_diagnostics`, `hotspots`.

2. Aggregate over functions in that file (group by `rel_path` on `function_profile`):

   * `total_functions`, `public_functions` (based on naming or call graph nodes).
   * LOC + complexity aggregates: `avg_loc`, `max_loc`, `avg_cyclomatic_complexity`.
   * `high_risk_function_count`.
   * File coverage: e.g., sum covered/executable lines across functions → `file_coverage_ratio`.
   * `tested_function_count` and `tests_touching` (sum of tests executing functions in the file).

3. Add tags/owners (from module).

4. Persist row with all these metrics.

---

### 6.3 `module_profile.*`

**Purpose**: the module/package‑level aggregation; this is the main basis for module architecture and subsystems.

**Inputs**

* `function_profile.*` and `file_profile.*`.
* `graph.import_graph_edges.*` – for `import_fan_in`/`fan_out`, `cycle_group`, `in_cycle`.
* Module metadata (`modules.*`, tags/owners).

**Mechanics**

For each `module`:

1. Aggregate across files mapped to that module:

   * `file_count`, `node_count`, `function_count`, `class_count`.
   * `total_loc`, `total_logical_loc`.
   * `avg_file_complexity`, `max_file_complexity`.

2. Aggregate function risk and coverage:

   * `high_risk_function_count`.
   * `module_coverage_ratio` (e.g. tested functions / total functions).

3. Join import graph:

   * `import_fan_in`, `import_fan_out`, `cycle_group`, `in_cycle`.

4. Attach tags/owners and a representative path from `core.modules`.

5. Persist one row per module.

These rows are then further decorated with graph metrics and subsystem info in `docs.v_module_architecture` and `docs.v_module_with_subsystem`.

---

## 7. Config + tags

### 7.1 `config_values.*`

**Purpose**: flatten structured config files and map each key to the codepaths/modules that reference it.

**Inputs**

* Config indexer output from `index_config_files` (parsing YAML/TOML/JSON/INI/ENV).
* Reference discovery from `prepare_config_state` (searching code for these keys).

**Mechanics**

1. Scan for config files:

   * Walk repo, matching config patterns (`*.yaml`, `*.toml`, `*.json`, `.env`, etc.).
   * For each file, parse into a nested map.

2. Flatten keys:

   * Turn nested keys into dot paths: `service.database.host`.
   * For each `(config_path, key)` pair, record `format` and normalized key string.

3. Find references:

   * Look through code (maybe via simple string search or more structured scanning) for occurrences of the key or a normalized variant.
   * For each reference, record the file path and associated module name.

4. Aggregate:

   * `reference_paths` = sorted list of files referencing the key.
   * `reference_modules` = module names for those files.
   * `reference_count` = length of the set.

5. Persist one row per key per config file.

---

### 7.2 `tags_index.yaml` / tags signals in profiles

**Purpose**: apply semantic tags (e.g., `api`, `infra`, `ml`) and infer ownership; these are then propagated into `modules`, `function_profile`, `file_profile`, `module_profile`.

**Inputs**

* Tag rules (YAML): patterns of includes/excludes per tag.
* `modules.*` registry mapping modules to paths.

**Mechanics**

1. For each tag rule:

   * Evaluate `includes` glob patterns against file paths.
   * Apply `excludes` to filter out matches.

2. For each matched path:

   * Map to module(s) via `repo_map.modules`.
   * Add tag to the module/file.

3. Optionally infer owners:

   * Rules can attach owners (teams/emails) to tags; these propagate to modules/files/functions.

4. Persist `tags_index.yaml` listing rules and resolved `matches`, and propagate tags/owners columns into:

   * `modules.jsonl`
   * `function_profile`, `file_profile`, `module_profile`

These tags feed risk and architecture views (and allow subsystems to be described in human terms).

---

## 8. Symbol‑use analytics (`symbol_use_edges.*`)

**Purpose**: SCIP‑based def→use relationships; used for coupling metrics and for SCPI evidence in call graph edges (we deep‑dived earlier, but here’s the short pipeline).

**Inputs**

* `index.scip.json` – SCIP index with documents, occurrences, and symbols.

**Mechanics**

1. Parse SCIP documents:

   * For each doc: `relative_path` and `occurrences`.

2. Build map `symbol → def_path`:

   * For occurrences where role bit includes “definition”, record the path as `def_path`. First definition wins.

3. Build def→use edges:

   * For each doc + occurrence where role bits include “use” (import, read, write):

     * `use_path = relative_path`
     * Look up `def_path` for the `symbol`. If found:

       * Emit `(symbol, def_path, use_path)`.

4. Decorate with flags:

   * `same_file = def_path == use_path`.
   * `same_module` = modules derived from both paths match.

5. Persist as `symbol_use_edges.*`.

These edges are then:

* Used directly by tools (“where is symbol S used?”).
* Aggregated into module‑level coupling in `graph_metrics_modules`.
* Used to derive SCIP candidates for unresolved call graph edges (the path we walked in detail earlier).

---

## Wrap‑up

So, in terms of *mechanistic* flow:

* **Graphs** (call, CFG, DFG, import, symbol uses) are built first.
* **Graph metrics** run over those graphs to provide centrality, degree, and layering at function + module level.
* **AST analytics** (ast_metrics, function_metrics, hotspots) and **typedness / diagnostics** flesh out the static perspective.
* **Coverage & test analytics** introduce dynamic behavior.
* **Risk aggregation** (`goid_risk_factors`) merges all those signals per function.
* **Profiles** (function/file/module) denormalize everything into three clean “shapes”.
* **Subsystems + graph metrics** then lift this up to **architecture level**, and **docs views** (`docs.v_*`) present it all in a consumable form for LLMs and tools.





