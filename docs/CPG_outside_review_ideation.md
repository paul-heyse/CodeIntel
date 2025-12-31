
According to a document from **December 27, 2025**, DuckDB already has very strong, “native-feeling” integration with **PyArrow** (and Polars), including scanning `pyarrow.Table`/`Dataset`/`Scanner` objects, and using Arrow schema/metadata patterns for drift management—so the **Arrow facts → DuckDB serving** direction you’ve pivoted back to is absolutely a solid foundation. 

With that as the downstream target, here are the *specific Hamilton-DAG-focused* modifications I’d make to get you to a **best-in-class, robust, maximally inference-driven** pipeline where the only hard contract is “produce a CPG”.

---

## 1) Make the CPG contract explicit, and make everything else “internal implementation”

Right now your drift (“declared outputs” vs “inputs”) is a symptom of not having a single, enforced contract boundary.

### What to do

1. Pick a *small* set of canonical “semantic outputs” that define the contract. In your case this is naturally:

   * `cpg_nodes` (Arrow table)
   * `cpg_edges` (Arrow table)
   * optionally: `cpg_health_metrics` (Arrow table or dict)

2. Tag **only those** as “semantic outputs” with a stable `semantic_id` and other metadata (entity/grain/schema_ref/version).

Hamilton supports making the DAG self-describing via tags and then compiling a registry by scanning `list_available_variables(tag_filter=...)`.

3. Add CI invariants:

   * a **semantic registry snapshot** (JSON) built from tags
   * an **execution graph snapshot** (`export_execution(...)`) for your canonical outputs

This is explicitly recommended as a “cheap and strong” way to prevent accidental semantic drift even when internal DAG nodes evolve.

### Why this helps your “max inference” goal

It gives you permission to refactor everything behind the contract without breaking agent-facing behaviors—while still detecting “you broke the CPG contract” immediately.

---

## 2) Align your DAG module structure to the CPG construction order (and keep stages narrow)

Your attached CPG design doc already gives you a very clean stage decomposition and a “construction order” that works well in practice:

1. LineIndex + spans
2. CST + AST → syntax nodes/edges
3. SCIP ingest + weld occurrences → symbol graph
4. per-function CFG
5. per-function def/use → DDG
6. per-function postdom → CDG
7. union → PDG/CPG
8. call graph + call wiring
   …

### What to change in Hamilton

* Put each stage in its own module (`syntax_layer.py`, `symbol_layer.py`, `cfg_layer.py`, `dfg_layer.py`, `cpg_union.py`, etc.).
* Each stage should:

  * accept **only** the upstream stage outputs it truly needs (Arrow tables + minimal scalars like repo_id/commit)
  * emit **Arrow tables** (facts) plus optional lightweight summaries/metrics

This reduces “input/output drift” because each stage boundary is explicit and testable.

---

## 3) Make “Repo snapshot” and “LineIndex/span” first-class nodes (they become your universal join key)

Your CPG doc makes the key weld explicit:

* SCIP occurrences must be mapped to **byte offsets** using a `LineIndex` + encoding rules, then welded to the **best matching syntax leaf**.

### What to do

Create foundational nodes that everything depends on:

* `repo_snapshot`: immutable description of the repo state being analyzed (commit, file list, content hashes, etc.)
* `file_bytes` (or `file_text`) per file from that snapshot
* `line_index_by_file`: mapping line/col → byte offset (for SCIP ranges)
* `syntax_nodes`, `syntax_edges` from CST+AST parsing, with spans in **byte offsets**

Then make the SCIP weld depend on:

* `line_index_by_file`
* `syntax_nodes` (and possibly a leaf index)
* decoded SCIP occurrences

This also sets you up for incremental caching correctly (next section).

---

## 4) Encode incremental recompute boundaries into the DAG (file-level + function-level)

Your doc is very explicit on the boundaries you want:

* parse: **file**
* CFG/DFG/CDG/PDG: **function/method body**
* interproc: affected callgraph neighborhood 

### What to change in Hamilton

#### A) Ensure the DAG has “units” that can be cached at those grains

Even if you don’t go full dynamic execution yet, you can still structure it so the expensive pieces sit behind stable nodes like:

* `syntax_facts_by_file` (dict[file_id → Arrow tables])
* `cfg_edges_by_function`
* `ddg_edges_by_function`
* `cdg_edges_by_function`

#### B) Turn on opt-in caching and cache only the expensive “facts”

Hamilton caching supports opt-in caching via `default_behavior="disable"` and then explicitly opting nodes in.

Also: be strict about ephemeral runtime objects (connections/clients) being `IGNORE`, so they don’t poison cache keys.

#### C) Make “what changed” an input so caching becomes correct

If `repo_snapshot` includes commit/hash, then downstream caches become valid by construction (the cache key changes when the snapshot changes). This is exactly the pattern described in the cache policy guidance. 

---

## 5) Standardize the DAG on PyArrow “fact tables”, and use Arrow schema metadata for contract/versioning

You want “no upfront schema declarations”, but you still need:

* stable join keys
* a way to tolerate new fields (properties)

### Recommended Arrow shape (practical + flexible)

Use a **minimal required column set** for nodes/edges, and keep everything else as either:

* `properties` (JSON string) OR
* `properties` (Arrow `map<string, string>` / `map<string, large_string>`) if you’re confident DuckDB + your tooling handle it cleanly.

Your CPG doc already suggests a minimal node/edge record shape like:

* `SyntaxNode(node_id, file_id, kind, span, …)`
* `SyntaxEdge(src, dst, label, order, …)` 

### Arrow schema drift management tools you should actually use

* **Schema union**: `pyarrow.unify_schemas([...], promote_options=...)` is the correct way to handle heterogeneous batches/fragments. 
* **Schema metadata**: `schema.with_metadata(...)` and `schema.remove_metadata()` are explicitly called out as the right lever for “contract tags”. 
* **Validation**: use `pyarrow.Table.validate(full=...)` in a few critical nodes (end of each stage) to fail fast on broken Arrow invariants. 

### Practical “max inference” interpretation

* You do **not** predeclare a full Arrow schema.
* You do:

  * infer Arrow tables from produced facts
  * attach schema metadata like `{contract: "cpg_nodes_v1", produced_by: "...", repo: "...", commit: "..."}`

This gives you dynamic evolution **with an auditable contract trail**.

---

## 6) Make Hamilton catch drift automatically: validation + snapshots + tooling

Hamilton gives you the hooks you need to stop drift from becoming runtime chaos:

* `validate_execution(...)` (detect issues without executing)
* `validate_inputs(...)` (check required inputs, avoid config collisions)
* `has_cycles(...)` 

And for I/O you can dry-run:

* `validate_materialization(...)` 

### Concrete modifications

* Add a unit test that:

  * builds the driver
  * calls `validate_execution(final_vars=[...])` for your canonical semantic outputs (`cpg_nodes`, `cpg_edges`)
  * exports an execution snapshot for those outputs
* Add a CI step using Hamilton CLI:

  * `hamilton diff ...` and/or `hamilton version ...` to surface DAG changes in PRs 

This directly addresses your “declared outputs vs input” drift.

---

## 7) Keep side effects out of “logic nodes”: use materializers for DuckDB writes / parquet datasets

Hamilton explicitly supports treating I/O via materializers (DataLoader/DataSaver) and a `.materialize()` path when you mainly care about persistence.

### What to change

* Make your logic DAG produce:

  * `cpg_nodes: pa.Table`
  * `cpg_edges: pa.Table`
* Then add materializer nodes like:

  * `materialize_cpg_to_parquet_dataset`
  * `materialize_cpg_to_duckdb_tables`
  * `materialize_latest_views`

…and run those via `Driver.materialize(...)` in production.

This keeps your DAG “pure” and massively reduces the surface area for drift.

---

## 8) Add “health metrics” and “confidence” as first-class outputs

Your CPG doc recommends:

* track `confidence`, `reason`, candidate sets
* automated checks like “% SCIP occurrences mapped”, CFG sanity, symbol stability

### What to change

Add nodes such as:

* `health_scip_weld_metrics`
* `health_cfg_metrics`
* `health_symbol_metrics`

Then optionally union them into a single `cpg_health_metrics` table that you can store in DuckDB alongside the CPG.

This is exactly the sort of “robustness layer” that makes an inference-driven system safe to operate.

---

## 9) Optional but high leverage: parallelize per-file/per-function work (without making the DAG weird)

You have two main approaches:

### A) Simple: Graph adapters for parallelism

Hamilton supports parallel adapters like a threadpool (`FutureAdapter`) for I/O-bound work. 

### B) More structured: Dynamic execution (Parallelizable/Collect)

If you decide to do dynamic, Hamilton supports it but notes caveats:

* enable dynamic execution explicitly
* only one level of Parallelizable→Collect (no nesting)
* multiprocessing uses pickle serialization; Dask/Ray can be better at scale

Given your pipeline shape (file → functions), this can be a big win later, but you can postpone it until correctness is solid.

---

# A concrete “modification checklist” you can apply immediately

If you do nothing else, do these in order:

1. **Define semantic outputs**: `cpg_nodes`, `cpg_edges`, `cpg_health_metrics` and tag them with stable `semantic_id`.
2. **Restructure modules** to match the CPG construction order stages.
3. **Make `repo_snapshot` + `LineIndex`** foundational dependencies of syntax + SCIP weld.
4. **Add `Driver.validate_execution`** tests for the semantic outputs (this will surface your drift immediately). 
5. **Standardize stage outputs as Arrow tables**, and add `Table.validate()` at stage boundaries. 
6. **Use Arrow schema metadata** + schema union as your drift management mechanism, not manual schemas. 
7. **Move I/O into materializers**, and use `validate_materialization()` to dry-run.
8. **Add health metrics + confidence** outputs; store them. 

---

## If you want a “north star” implementation pattern (very close to your philosophy)

* **Only contract**: `semantic_id = "cpg.nodes"` and `"cpg.edges"` (and maybe `"cpg.health"`)
* **All intermediate schemas**: inferred Arrow, versioned via schema metadata + tags
* **All intermediate stability**: enforced via Hamilton validation + snapshots
* **All recompute**: bounded by file/function grain, accelerated via caching
* **All serving**: DuckDB scans Arrow/Parquet, MCP tools query DuckDB

This stays maximally inference-driven, but is still *operationally hard to break*.

If you paste (or describe) the *current* names of your DAG’s intended final outputs and the module layout you currently have, I can map the above directly onto your current structure (e.g., “rename X to Y, split module A into stages B/C, move I/O from node N into materializer M, add tags here”).

According to a document from your attached **CPG_construction.md** (the file itself doesn’t include a date header), the “best-in-class” path is to build **layered graphs (syntax → symbols → CFG → dataflow → dependence → interproc)** and then **weld them into one property graph** using **stable IDs + spans + symbols** as the join glue. 

Below is a concrete “north star” design + the specific **breaking changes** I’d make to your Hamilton DAG + storage layout so you end up with:

* **Hamilton** = the orchestrator + “semantic contract enforcer”
* **PyArrow/Parquet datasets** = the primary artifact store for *all* CPG layers (schema inferred/materialized, versioned by run inputs)
* **DuckDB** = only the **metadata + query serving** layer (plus optional “latest snapshot” materialization), with **simple joins** to reconstruct the full CPG

---

## 1) Lock down the *only* true contracts: snapshot + span + deterministic IDs

Even if you want “max inference / no upfront schema”, there are a few invariants you must treat as non-negotiable, otherwise you can’t reliably stitch layers together.

### 1.1 Repo snapshot contract (input determinism)

Treat every analysis run as operating on an immutable snapshot: repo root + commit (or content hash), plus per-file bytes and hashes.
This is the single biggest change that makes caching, incremental recompute, and “joins that don’t lie” possible.

**Breaking-change recommendation**

* Make `repo_snapshot` a first-class Hamilton node that returns:

  * `repo_id` (stable identifier)
  * `commit` or `snapshot_hash`
  * `python_version`, `platform`, dependency lock hash (if relevant)
* Make *every* downstream node depend (directly or indirectly) on `repo_snapshot`, so Hamilton’s data-versioning/caching can be correct.

### 1.2 Span is the universal join key (and keep byte offsets)

Everything must anchor to a shared coordinate system (file_id + span) and you should store both line/col and byte offsets to avoid encoding mismatch bugs.

**Breaking-change recommendation**

* Standardize a `LineIndex` artifact per file (line → byte/char offsets) and make it an explicit DAG output used by:

  * CST/AST span extraction
  * SCIP range → byte mapping
  * “occurrence → syntax node” welding

This is literally step 1 in the recommended construction order.

---

## 2) Adopt a “DuckDB/Arrow-friendly” CPG storage model (nodes + edges + a few satellites)

Your downstream goal (LLM/agent queries) wants a representation that is:

* easy to filter (by file, symbol, kind, label)
* easy to join (node_id/symbol_id)
* easy to extend (new edge labels, new props)

### 2.1 Minimum viable *tables* (Arrow datasets) for a Python CPG

From the doc: emit **syntax nodes/edges**, then **symbol graph + occurrences**, then **CFG/DDG/CDG**, and finally **call wiring edges** like `ARG_TO_PARAM` and `RET_TO_CALL`.

I would standardize these datasets (each as Parquet/Arrow dataset):

1. `files`

* `file_id` (string)
* `repo_id`, `commit`
* `path` (string)
* `content_hash` (string)
* `bytes_len` (int64)
* `encoding`, `newline_mode`

2. `syntax_nodes`

* `node_id` (string, deterministic)
* `file_id`
* `kind` (string) — include: module, class, function, param, identifier, call, import, return, predicate, etc.
* `start_byte`, `end_byte` (int64)
* `start_line`, `start_col`, `end_line`, `end_col` (int32)
* `code_snippet_hash` (string) (optional but useful)

3. `edges`
   A *single unified* edge table is ideal for DuckDB:

* `src_node_id`, `dst_node_id`
* `label` (string) — e.g. `AST`, `CONTAINS`, `CFG`, `DDG`, `CDG`, `CALLS`, `IMPORTS`, `INHERITS`, `ARG_TO_PARAM`, `RET_TO_CALL`, etc.
* `file_id` (denormalized for faster filtering)
* optional: `order`, `position`, `branch`, `symbol_id`, `confidence`, `reason`

4. `symbols`

* `symbol_id` (string, stable SCIP symbol identity)
* `kind`, `display_name`, `docs_hash`, etc.

5. `occurrences`

* `node_id`, `symbol_id`
* `role` (DEF/REF/…)
* occurrence span fields (byte and/or line/col)

6. `crosswalks` (explicit “weld tables”)
   The doc is explicit that robustness comes from crosswalk tables: CST↔AST, SCIP occurrence↔syntax node, syntax node↔symbol, etc..

So create datasets like:

* `cst_ast_crosswalk(node_id_cst, node_id_ast, confidence, reason)`
* `occurrence_syntax_crosswalk(occurrence_id, node_id, confidence, reason)`

### 2.2 “Max inference” without losing queryability

You can avoid declaring schemas *upfront* while still preventing “schema drift chaos”:

* Keep the above **required columns** as your *semantic contract*.
* Put everything else into:

  * either a `props_json` (string) column, or
  * an Arrow `map<string, string>` column (works if you accept stringified values)

And for “best-in-class Python reality”, explicitly track uncertainty as properties (`confidence`, `reason`, candidate sets).

---

## 3) Rebuild the Hamilton DAG as a layered “CPG factory” with dynamic execution

### 3.1 Structure the DAG by the construction order (make it obvious)

The doc gives a concrete order; turn that into Hamilton modules 1:1:

* `snapshot.py` — repo snapshot + file inventory
* `line_index.py` — per-file line indexes + span normalization
* `syntax.py` — CST + AST extraction + syntax nodes/edges + CST↔AST crosswalk
* `scip.py` — decode SCIP + occurrences + weld to syntax nodes + symbol graph
* `cfg.py` — per-function CFG edges
* `dataflow.py` — defs/uses + reaching defs / DDG edges (tie to symbols where possible)
* `cdg.py` — postdominators + CDG edges
* `interproc.py` — call graph + call wiring (`ARG_TO_PARAM`, `RET_TO_CALL`) + summaries
* `assemble.py` — unify edge tables + emit “CPG datasets”
* `health.py` — validation metrics

That gives you a DAG where “drift” (inputs/outputs mismatch) is much harder to hide.

### 3.2 Use Hamilton dynamic execution for file/function granularity

You want recompute boundaries at smallest practical unit: parse at file level; CFG/DFG/CDG/PDG at function/method level.

Hamilton supports this with dynamic execution, but you must design around its caveats:

* enable via `Builder.enable_dynamic_execution(allow_experimental_mode=True)`
* dynamic execution uses task-based executor
* caveats: no nested Parallelizable/Collect, only one Collect input per function, multiprocessing serialization pitfalls. 

**Breaking-change recommendation**

* Stage A (parse) = `Parallelizable[file_record] → per_file_syntax → Collect`
* Stage C/D/E (CFG/DFG/CDG) = `Parallelizable[function_descriptor] → per_function_cfg/dfg/cdg → Collect`
* Do **not** nest these; instead materialize `function_descriptor` as a dataset between stages.

### 3.3 Make Arrow Tables the *native* node outputs

Standardize: “Hamilton nodes that represent facts return Arrow Tables”.

If you ever produce Polars objects in intermediate steps, convert through Arrow early. PyArrow’s dataframe interchange entrypoint is the canonical pattern (`pyarrow.interchange.from_dataframe`). 

---

## 4) Replace pyiceberg’s “history” with Hamilton caching + dataset manifests in DuckDB

You originally liked pyiceberg for change history. You can get 80–90% of that value with less complexity by combining:

1. **Hamilton caching/data versioning** for compute lineage
   Hamilton’s caching system keys results by inputs + code version hashes, and it supports opting nodes into caching and using Parquet for tabular facts. 

2. A **dataset manifest table** in DuckDB (run_id → dataset paths + schema hashes)

3. Optionally ingest Hamilton cache logs/lineage JSONL into DuckDB for “run graph” introspection (this pattern is already laid out in the Hamilton caching operational doc).

### 4.1 Caching rules to encode in the DAG

Use the “matrix” approach:

* ephemeral runtime objects: `IGNORE` (don’t pollute cache keys)
* big fact tables (syntax nodes, edges, symbols): cache as Parquet (`DEFAULT`, `format="parquet"`)

This is exactly the intended operational posture: opt-in caching, and cache “large tabular facts” as Parquet.

**Breaking-change recommendation**

* Make `default_behavior="disable"` globally (so you don’t cache everything by accident)
* Add `@cache(..., format="parquet")` to:

  * `syntax_nodes`, `edges`, `symbols`, `occurrences`, per-function CFG/DDG/CDG outputs, etc.

---

## 5) Use PyArrow to materialize schemas and keep “inference-driven” behavior safe

You want:

* no upfront schemas
* but you do want “schema materialization” so datasets are stable and queryable

That’s a great fit for an **“infer then freeze”** approach:

1. Write Parquet fragments as you compute them (file-level or function-level partitions)
2. Build a dataset factory and **inspect** to infer a unified schema
3. Finish the dataset with that schema and record the schema hash in DuckDB

This gives you inference-driven development without runtime surprises.

---

## 6) Make DuckDB purely the serving layer (plus metadata)

### 6.1 What DuckDB should store vs reference

**Store in DuckDB**

* runs table (`run_id`, repo_id, commit, started_at, status)
* dataset manifest table:

  * dataset name (`syntax_nodes`, `edges`, `symbols`, …)
  * path to parquet dataset
  * schema hash / schema JSON
  * row counts, basic stats

**Reference (as external)**

* the large Parquet datasets themselves
  DuckDB can query Parquet directly; you can create views over `parquet_scan(...)` paths for “latest run”.

### 6.2 The “simple joins” to reconstruct the CPG

If you follow the storage model above, serving a CPG is basically:

* nodes: `syntax_nodes`
* edges: `edges`
* symbols: `symbols`
* occurrences: `occurrences`

and queries become extremely simple:

* `edges.src_node_id → syntax_nodes.node_id`
* `edges.dst_node_id → syntax_nodes.node_id`
* `occurrences.node_id → syntax_nodes.node_id`
* `occurrences.symbol_id → symbols.symbol_id`

Because the doc’s weld is: occurrence → syntax node, and then symbol identity enables cross-file joins.

---

## 7) The concrete changes I’d make (a “migration checklist”)

### Phase 1 — Re-platform the DAG around Arrow-first “facts”

1. **Refactor every “fact-producing node”** to return `pyarrow.Table`
2. Introduce shared helper utilities:

   * `to_arrow_table(records: list[dict]) -> pa.Table` (inferred)
   * `write_dataset(table, path, partitioning=[...])`
   * `unify_schema(dataset_path) -> pa.Schema` + schema hash
3. Add `health` nodes that compute the doc’s recommended metrics:

   * % SCIP occurrences successfully mapped to syntax nodes
   * CFG sanity checks
   * symbol uniqueness/stability checks

### Phase 2 — Introduce the “layered CPG” pipeline as first-class modules

4. Implement Stage A exactly:

   * build `LineIndex`
   * parse CST + AST
   * emit `syntax_nodes`, `syntax_edges`, `cst_ast_crosswalk`
5. Implement Stage B:

   * ingest SCIP
   * weld occurrences → syntax nodes using span mapping + deterministic tie-breakers
   * emit `symbols`, `symbol_edges`, `occurrences`
6. Implement per-function CFG (Stage C) and emit `CFG` edges connecting *existing syntax nodes*
7. Implement DDG (Stage D) with explicit “store uncertainty” properties
8. Implement CDG + PDG union (Stage E/F) and then unify into one edge table with `label`

### Phase 3 — Interprocedural “agent-grade” stitching

9. Add call graph edges (`CALLS`, plus `CALLS_UNKNOWN` with candidates)
10. Add call wiring edges:

* `ARG_TO_PARAM`
* `RET_TO_CALL`

11. Add summaries for scalable interproc reasoning

### Phase 4 — Serving + metadata

12. Add DuckDB “dataset registry” and “latest pointers”:

* `semantic.latest_run(repo_id) -> run_id`
* views: `semantic.syntax_nodes_latest`, `semantic.edges_latest`, etc.

13. (Optional but strong) ingest Hamilton cache logs to DuckDB for lineage-grade introspection (run graph)

---

## 8) Why this design matches your “max inference” philosophy better than pyiceberg

* Your “history” is captured by **repo snapshot + run_id + schema hashes + cached node versions**, rather than a second table-format stack.
* You still get robustness via:

  * deterministic spans and IDs
  * explicit crosswalks
  * health metrics gates
* You keep DuckDB in the role it’s best at for your use case: **fast local analytics + serving** over Parquet/Arrow-backed data.

---

If you want the next step to be extremely concrete, I can write you a **“minimum viable CPG schema”** (nodes/edges/symbols/occurrences + required columns + recommended types) aligned to the layered model above, and a Hamilton module skeleton that implements **Stages A + B** end-to-end (syntax + SCIP weld), which is the foundation everything else builds on.

According to a document from **December 30, 2025** (the `CPG_construction.md` you attached in this chat), the Stage A + Stage B “foundation” you want is:

* lock down the **repo snapshot contract** (so byte-accurate spans are stable),
* treat **Span** (line/col + encoding, plus **byte offsets**) as the *universal join key*,
* emit a **Syntax layer** as *tables* (`SyntaxNode`, `SyntaxEdge`) with **deterministic node IDs**, then
* ingest **SCIP** and perform the **critical weld**: `Occurrence → best matching syntax leaf`, producing `SymbolNode`, `SymbolEdge`, and `Occurrence(node_id ↔ symbol)` tables. 

Below are the two deliverables you asked for:

1. a **minimum viable CPG schema** (nodes/edges/symbols/occurrences + required columns + recommended Arrow types) aligned to that model, and
2. a **Hamilton module skeleton** that implements **Stage A + Stage B end-to-end** (syntax + SCIP weld), in a way that stays “maximally inference-driven” while still enforcing a small contract at the boundary.

---

## Minimum viable CPG schema for Stage A + Stage B

### Design invariants (these matter more than any individual column)

**Invariant 1 — spans must refer to exact file bytes.** Store both (a) line/col + encoding and (b) `byte_start/byte_end`, and maintain a deterministic `LineIndex` per file to avoid UTF-16 vs UTF-8 stitching bugs. 

**Invariant 2 — deterministic NodeId.** Use the doc’s strategy so IDs are stable and reproducible:
`node_id = hash(file_id, node_kind, start_byte, end_byte, optional_disambiguator)` 

**Invariant 3 — welding algorithm is deterministic.** Prefer exact span match; else pick smallest containing syntax node with deterministic tie-breakers. 

---

### Recommended Arrow typing conventions

To keep this robust + DuckDB-friendly:

* **IDs**

  * `file_id`: `pa.large_string()` (canonical path/URI) 
  * `node_id`: `pa.fixed_size_binary(16)` (128-bit digest; deterministic)
  * `symbol`: `pa.large_string()` (SCIP symbol string; stable identity) 

* **Spans**

  * `start_line/start_col/end_line/end_col`: `pa.int32()`
  * `start_byte/end_byte`: `pa.int64()`
  * `encoding`: `pa.string()` 

* **Flexible properties**

  * `props_json`: `pa.large_string()` (JSON)
    (This is your “maximally inference-driven” escape hatch: you don’t predeclare every property; you only standardize the join keys + minimal graph shape.)

---

## Tables

### 0) Supporting table: `repo_files_v1` (snapshot contract)

This isn’t “CPG” per se, but you need it because SCIP + parsers must agree on *exact bytes*. 

**Required columns (Arrow types):**

* `repo_root` : `pa.large_string()`
* `commit` : `pa.string()`
* `file_id` : `pa.large_string()`  (canonical path/URI)
* `rel_path` : `pa.large_string()`
* `content_hash` : `pa.fixed_size_binary(32)` (e.g., sha256)
* `byte_length` : `pa.int64()`
* `newline_mode` : `pa.string()`  (e.g., `LF`, `CRLF`) 
* `encoding` : `pa.string()` (default `utf-8`)

---

### 1) `cpg_nodes_v1` (SyntaxNode)

Matches the doc’s `SyntaxNode(node_id, file_id, kind, span, ...)`. 

**Required columns (Arrow types):**

* `node_id` : `pa.fixed_size_binary(16)`
* `file_id` : `pa.large_string()`
* `kind` : `pa.string()`
* `start_line` : `pa.int32()`
* `start_col` : `pa.int32()`
* `end_line` : `pa.int32()`
* `end_col` : `pa.int32()`
* `start_byte` : `pa.int64()`
* `end_byte` : `pa.int64()`
* `encoding` : `pa.string()`

**Strongly recommended (but optional) columns:**

* `code_snippet_hash` : `pa.fixed_size_binary(32)` (hash of the exact byte slice) 
* `parser` : `pa.dictionary(pa.int32(), pa.string())`  (e.g., `ast`, `cst`)
* `flags_json` : `pa.large_string()`
* `props_json` : `pa.large_string()`  (JSON)

**Notes**

* Keep `kind` language-agnostic where possible (“FunctionDef”, “Call”, “Attribute”, “Identifier”, etc.). The doc lists the minimum kinds you’ll want. 

---

### 2) `cpg_edges_v1` (SyntaxEdge)

Matches `SyntaxEdge(src, dst, label, order)`. 

**Required columns (Arrow types):**

* `src_node_id` : `pa.fixed_size_binary(16)`
* `dst_node_id` : `pa.fixed_size_binary(16)`
* `label` : `pa.string()`  (e.g., `AST`, `CONTAINS`, `NEXT_SIBLING`, …) 
* `order` : `pa.int32()`  (child index / deterministic ordering)

**Recommended (optional):**

* `file_id` : `pa.large_string()` (denormalization convenience)
* `props_json` : `pa.large_string()`

---

### 3) `cpg_symbols_v1` (SymbolNode)

Matches `SymbolNode(symbol_id, kind, display_name, docs_hash, …)`. 

**Required columns (Arrow types):**

* `symbol` : `pa.large_string()`  (SCIP symbol string)
* `kind` : `pa.string()`          (SCIP symbol kind normalized)
* `display_name` : `pa.large_string()`

**Recommended (optional):**

* `docs_hash` : `pa.fixed_size_binary(32)` 
* `props_json` : `pa.large_string()`

---

### 4) `cpg_symbol_edges_v1` (SymbolEdge)

Matches `SymbolEdge(symbol → symbol, label="RELATIONSHIP_*")`. 

**Required columns (Arrow types):**

* `src_symbol` : `pa.large_string()`
* `dst_symbol` : `pa.large_string()`
* `label` : `pa.string()`  (e.g., `RELATIONSHIP_IMPLEMENTATION`, etc.) 

**Recommended (optional):**

* `props_json` : `pa.large_string()`

---

### 5) `cpg_occurrences_v1` (Occurrence = weld table)

Matches `Occurrence(node_id → symbol_id, role=DEF|REF|…, occurrence_span=…)`. 

**Required columns (Arrow types):**

* `occurrence_id` : `pa.fixed_size_binary(16)` (hash of file_id + span + symbol + roles)

* `file_id` : `pa.large_string()`

* `symbol` : `pa.large_string()`

* `role` : `pa.string()`  (`DEF`, `REF`, possibly `READ/WRITE` if you model it)

* `node_id` : `pa.fixed_size_binary(16)` (nullable — weld can fail)

* `start_line` : `pa.int32()`

* `start_col` : `pa.int32()`

* `end_line` : `pa.int32()`

* `end_col` : `pa.int32()`

* `start_byte` : `pa.int64()`

* `end_byte` : `pa.int64()`

* `encoding` : `pa.string()`

**Strongly recommended (optional):**

* `enclosing_start_line/enclosing_start_col/enclosing_end_line/enclosing_end_col` : `pa.int32()` (SCIP often provides `enclosing_range`) 
* `weld_quality` : `pa.string()` (e.g., `exact`, `contained`, `fallback`, `unmatched`)
* `weld_reason` : `pa.large_string()` (debuggability)

---

## Minimal Arrow schema objects (copy/paste starting point)

This pattern also leans into Arrow’s “schema objects are immutable” behavior (you return new schema objects when attaching metadata). 

```python
# cpg/contracts.py
import pyarrow as pa

def _id16(name: str) -> pa.Field:
    return pa.field(name, pa.fixed_size_binary(16))

def _sha256(name: str) -> pa.Field:
    return pa.field(name, pa.fixed_size_binary(32))

REPO_FILES_V1 = pa.schema([
    pa.field("repo_root", pa.large_string()),
    pa.field("commit", pa.string()),
    pa.field("file_id", pa.large_string()),
    pa.field("rel_path", pa.large_string()),
    _sha256("content_hash"),
    pa.field("byte_length", pa.int64()),
    pa.field("newline_mode", pa.string()),
    pa.field("encoding", pa.string()),
]).with_metadata({"contract": "repo_files_v1"})  # attach schema metadata:contentReference[oaicite:25]{index=25}

CPG_NODES_V1 = pa.schema([
    _id16("node_id"),
    pa.field("file_id", pa.large_string()),
    pa.field("kind", pa.string()),
    pa.field("start_line", pa.int32()),
    pa.field("start_col", pa.int32()),
    pa.field("end_line", pa.int32()),
    pa.field("end_col", pa.int32()),
    pa.field("start_byte", pa.int64()),
    pa.field("end_byte", pa.int64()),
    pa.field("encoding", pa.string()),
    _sha256("code_snippet_hash"),
    pa.field("parser", pa.string()),
    pa.field("flags_json", pa.large_string()),
    pa.field("props_json", pa.large_string()),
]).with_metadata({"contract": "cpg_nodes_v1"})

CPG_EDGES_V1 = pa.schema([
    _id16("src_node_id"),
    _id16("dst_node_id"),
    pa.field("label", pa.string()),
    pa.field("order", pa.int32()),
    pa.field("file_id", pa.large_string()),
    pa.field("props_json", pa.large_string()),
]).with_metadata({"contract": "cpg_edges_v1"})

CPG_SYMBOLS_V1 = pa.schema([
    pa.field("symbol", pa.large_string()),
    pa.field("kind", pa.string()),
    pa.field("display_name", pa.large_string()),
    _sha256("docs_hash"),
    pa.field("props_json", pa.large_string()),
]).with_metadata({"contract": "cpg_symbols_v1"})

CPG_SYMBOL_EDGES_V1 = pa.schema([
    pa.field("src_symbol", pa.large_string()),
    pa.field("dst_symbol", pa.large_string()),
    pa.field("label", pa.string()),
    pa.field("props_json", pa.large_string()),
]).with_metadata({"contract": "cpg_symbol_edges_v1"})

CPG_OCCURRENCES_V1 = pa.schema([
    _id16("occurrence_id"),
    pa.field("file_id", pa.large_string()),
    pa.field("symbol", pa.large_string()),
    pa.field("role", pa.string()),
    _id16("node_id"),
    pa.field("start_line", pa.int32()),
    pa.field("start_col", pa.int32()),
    pa.field("end_line", pa.int32()),
    pa.field("end_col", pa.int32()),
    pa.field("start_byte", pa.int64()),
    pa.field("end_byte", pa.int64()),
    pa.field("encoding", pa.string()),
    pa.field("weld_quality", pa.string()),
    pa.field("weld_reason", pa.large_string()),
]).with_metadata({"contract": "cpg_occurrences_v1"})
```

**Why keep schema objects at all if you want “max inference”?**
Because you can still *infer first*, then do a light “contract enforcement” pass that:

* ensures required columns exist,
* casts to the expected types,
* reorders fields (Arrow `Table.cast()` requires names + order match). 

---

## Hamilton module skeleton implementing Stage A + Stage B end-to-end

This skeleton is explicitly aligned to:

* Stage A emits syntax nodes/edges with deterministic NodeIds 
* Stage B ingests SCIP, converts occurrences to stitchable facts, then welds occurrences to syntax leaves deterministically 

### Folder layout

```
cpg/
  contracts.py
  ids.py
  line_index.py
  arrow_utils.py

hamilton_modules/
  stage_a_syntax.py
  stage_b_scip.py
  stage_ab_driver_example.py
```

---

### `cpg/ids.py` (deterministic IDs)

```python
# cpg/ids.py
from __future__ import annotations
import hashlib

def id16(*parts: object) -> bytes:
    """
    Deterministic 128-bit ID.
    Uses blake2b(digest_size=16) so it's stable + fast and in stdlib.
    """
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        if p is None:
            h.update(b"\x00")
        elif isinstance(p, bytes):
            h.update(p)
        else:
            h.update(str(p).encode("utf-8", errors="strict"))
            h.update(b"\x1f")  # separator
    return h.digest()
```

This directly implements the doc’s NodeId strategy (hash of file_id + kind + byte span + disambiguator). 

---

### `cpg/line_index.py` (byte-accurate mapping)

```python
# cpg/line_index.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class LineIndex:
    """
    Deterministic mapping line -> byte offset.
    Lines are 0-based, columns are interpreted as *character offsets* for UTF-8 text,
    then converted to bytes.
    """
    line_start_byte: list[int]
    encoding: str = "utf-8"

    @staticmethod
    def from_bytes(b: bytes, encoding: str = "utf-8") -> "LineIndex":
        # Find byte offsets of each line start (0-based line numbers).
        starts = [0]
        i = 0
        while True:
            j = b.find(b"\n", i)
            if j == -1:
                break
            starts.append(j + 1)
            i = j + 1
        return LineIndex(starts, encoding=encoding)

    def lc_to_byte(self, line: int, col: int, file_bytes: bytes) -> int:
        # Convert (line, col) -> byte offset.
        # col is in "characters"; we encode the substring to get bytes.
        start = self.line_start_byte[line]
        line_end = self.line_start_byte[line + 1] if line + 1 < len(self.line_start_byte) else len(file_bytes)
        line_bytes = file_bytes[start:line_end]
        line_text = line_bytes.decode(self.encoding, errors="strict")
        prefix = line_text[:col].encode(self.encoding, errors="strict")
        return start + len(prefix)
```

This is the “store both line/col and byte offsets + deterministic LineIndex” requirement. 

---

### `cpg/arrow_utils.py` (infer-first, then enforce contract)

```python
# cpg/arrow_utils.py
from __future__ import annotations
import pyarrow as pa

def ensure_columns(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """
    Infer-first pattern:
      - allow extra columns
      - add missing required columns as nulls
      - cast/reorder required columns into the front
    """
    cols = {name: table.column(name) for name in table.schema.names}

    for f in schema:
        if f.name not in cols:
            cols[f.name] = pa.nulls(table.num_rows, type=f.type)

    # Rebuild in schema order first, then append extras (preserve inference-driven extras)
    ordered_names = [f.name for f in schema]
    extra_names = [n for n in table.schema.names if n not in ordered_names]
    rebuilt = pa.table({n: cols[n] for n in ordered_names + extra_names})

    # Now cast the required prefix (names+order must match for cast):contentReference[oaicite:31]{index=31}
    required_prefix = rebuilt.select(ordered_names).cast(schema, safe=True)
    out = pa.table({**{n: required_prefix[n] for n in ordered_names},
                    **{n: rebuilt[n] for n in extra_names}})

    out.validate(full=False)  # cheap correctness checks:contentReference[oaicite:32]{index=32}
    return out
```

---

## Stage A Hamilton module: `hamilton_modules/stage_a_syntax.py`

This does “Parse CST + AST and build the Syntax Graph” and emits the Syntax layer as tables. 

```python
# hamilton_modules/stage_a_syntax.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ast
import pyarrow as pa

from cpg.contracts import CPG_NODES_V1, CPG_EDGES_V1, REPO_FILES_V1
from cpg.arrow_utils import ensure_columns
from cpg.ids import id16
from cpg.line_index import LineIndex

# Optional but recommended: lossless CST parsing for exact token spans (libcst).
# import libcst


@dataclass(frozen=True)
class FileBlob:
    file_id: str         # canonical path/URI (use rel path or a URI)
    abs_path: str
    bytes_: bytes
    encoding: str = "utf-8"
    newline_mode: str = "LF"


@dataclass(frozen=True)
class SyntaxFileArtifacts:
    # Keep it all-in-one so Collect[] aggregation only needs ONE input (Hamilton caveat):contentReference[oaicite:34]{index=34}
    nodes: pa.Table
    edges: pa.Table
    # Leaf candidates used by Stage B weld (identifiers/attributes)
    leaf_index: pa.Table


# -------------------------
# Repo snapshot contract
# -------------------------

def repo_root(repo_root: str) -> str:
    return repo_root


def commit(commit: str) -> str:
    return commit


def repo_python_files(repo_root: str) -> list[str]:
    root = Path(repo_root)
    return [str(p) for p in root.rglob("*.py") if p.is_file()]


def file_blobs(repo_root: str, repo_python_files: list[str], encoding: str = "utf-8") -> Iterable[FileBlob]:
    # In production you can make this Parallelizable[FileBlob] (dynamic execution).
    # Keeping it iterable makes the skeleton runnable under default executor.
    for abs_path in repo_python_files:
        p = Path(abs_path)
        b = p.read_bytes()
        # file_id should be canonical; simplest is repo-relative path
        file_id = str(p.relative_to(Path(repo_root)))
        newline_mode = "CRLF" if b"\r\n" in b else "LF"
        yield FileBlob(file_id=file_id, abs_path=str(p), bytes_=b, encoding=encoding, newline_mode=newline_mode)


def repo_files_v1(repo_root: str, commit: str, file_blobs: Iterable[FileBlob]) -> pa.Table:
    rows = []
    for fb in file_blobs:
        import hashlib
        rows.append({
            "repo_root": repo_root,
            "commit": commit,
            "file_id": fb.file_id,
            "rel_path": fb.file_id,
            "content_hash": hashlib.sha256(fb.bytes_).digest(),
            "byte_length": len(fb.bytes_),
            "newline_mode": fb.newline_mode,
            "encoding": fb.encoding,
        })
    t = pa.Table.from_pylist(rows)
    return ensure_columns(t, REPO_FILES_V1)


# -------------------------
# AST/CST parse + syntax graph emit
# -------------------------

def syntax_file_artifacts(file_blobs: Iterable[FileBlob]) -> list[SyntaxFileArtifacts]:
    """
    MVP: do Stage A per file, return list of artifacts.
    Later: convert this into a Parallelizable/Collect block.
    """
    out: list[SyntaxFileArtifacts] = []
    for fb in file_blobs:
        li = LineIndex.from_bytes(fb.bytes_, encoding=fb.encoding)

        # Parse AST
        src_text = fb.bytes_.decode(fb.encoding, errors="strict")
        tree = ast.parse(src_text)

        nodes_rows = []
        edges_rows = []

        # Walk AST with parent relationships
        def visit(node: ast.AST, parent_id: bytes | None, edge_label: str | None, order: int | None):
            # Best effort spans from AST; convert to byte offsets via LineIndex.
            # (Python AST uses line/col; end_* available on 3.8+)
            start_line = getattr(node, "lineno", None)
            start_col  = getattr(node, "col_offset", None)
            end_line   = getattr(node, "end_lineno", None)
            end_col    = getattr(node, "end_col_offset", None)

            # Convert to 0-based for consistency (SCIP/LSP commonly 0-based; normalize everywhere)
            if start_line is None or start_col is None or end_line is None or end_col is None:
                # Unknown span (e.g., synthetic nodes) — you can still include them with null spans
                sb = eb = None
                sl = sc = el = ec = None
            else:
                sl, sc, el, ec = start_line - 1, start_col, end_line - 1, end_col
                sb = li.lc_to_byte(sl, sc, fb.bytes_)
                eb = li.lc_to_byte(el, ec, fb.bytes_)

            kind = type(node).__name__
            node_id = id16(fb.file_id, kind, sb, eb, None)  # doc strategy:contentReference[oaicite:35]{index=35}

            nodes_rows.append({
                "node_id": node_id,
                "file_id": fb.file_id,
                "kind": kind,
                "start_line": sl,
                "start_col": sc,
                "end_line": el,
                "end_col": ec,
                "start_byte": sb,
                "end_byte": eb,
                "encoding": fb.encoding,
                "code_snippet_hash": None,  # fill if sb/eb present
                "parser": "ast",
                "flags_json": None,
                "props_json": None,
            })

            if parent_id is not None:
                edges_rows.append({
                    "src_node_id": parent_id,
                    "dst_node_id": node_id,
                    "label": edge_label or "AST",
                    "order": order if order is not None else 0,
                    "file_id": fb.file_id,
                    "props_json": None,
                })

            # Recurse children deterministically
            child_i = 0
            for field_name, value in ast.iter_fields(node):
                if isinstance(value, ast.AST):
                    visit(value, node_id, field_name, child_i)
                    child_i += 1
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            visit(item, node_id, field_name, child_i)
                            child_i += 1

        visit(tree, parent_id=None, edge_label=None, order=None)

        nodes_t = ensure_columns(pa.Table.from_pylist(nodes_rows), CPG_NODES_V1)
        edges_t = ensure_columns(pa.Table.from_pylist(edges_rows), CPG_EDGES_V1)

        # Leaf index candidates for welding (identifier-ish nodes).
        # You can refine this to Name/Attribute/alias/etc.
        leaf_kinds = {"Name", "Attribute"}
        # Filter in Arrow (simple Python loop here for clarity)
        leaf_rows = []
        for row in nodes_rows:
            if row["kind"] in leaf_kinds and row["start_byte"] is not None and row["end_byte"] is not None:
                leaf_rows.append({
                    "file_id": row["file_id"],
                    "node_id": row["node_id"],
                    "kind": row["kind"],
                    "start_byte": row["start_byte"],
                    "end_byte": row["end_byte"],
                })
        leaf_index = pa.Table.from_pylist(leaf_rows)

        out.append(SyntaxFileArtifacts(nodes=nodes_t, edges=edges_t, leaf_index=leaf_index))

    return out


def cpg_nodes_v1(syntax_file_artifacts: list[SyntaxFileArtifacts]) -> pa.Table:
    tables = [a.nodes for a in syntax_file_artifacts]
    return pa.concat_tables(tables, promote=True)


def cpg_edges_v1(syntax_file_artifacts: list[SyntaxFileArtifacts]) -> pa.Table:
    tables = [a.edges for a in syntax_file_artifacts]
    return pa.concat_tables(tables, promote=True)


def syntax_leaf_index(syntax_file_artifacts: list[SyntaxFileArtifacts]) -> pa.Table:
    tables = [a.leaf_index for a in syntax_file_artifacts]
    return pa.concat_tables(tables, promote=True)
```

**Why “leaf_index”?** Because Stage B needs “best matching syntax leaf” to weld occurrences deterministically. 

---

## Stage B Hamilton module: `hamilton_modules/stage_b_scip.py`

This implements “Ingest SCIP and build the Symbol Graph” + the weld step. 

```python
# hamilton_modules/stage_b_scip.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from cpg.contracts import CPG_SYMBOLS_V1, CPG_SYMBOL_EDGES_V1, CPG_OCCURRENCES_V1
from cpg.arrow_utils import ensure_columns
from cpg.ids import id16
from cpg.line_index import LineIndex


# You must provide SCIP protobuf bindings (generated from scip.proto).
# e.g., `from scip import scip_pb2`
# This aligns with the doc: index.scip is protobuf, decode via protoc if needed.:contentReference[oaicite:38]{index=38}
def scip_index_path(scip_index_path: str) -> str:
    return scip_index_path


def scip_index_bytes(scip_index_path: str) -> bytes:
    return open(scip_index_path, "rb").read()


def scip_index(scip_index_bytes: bytes) -> Any:
    # from scip import scip_pb2
    # idx = scip_pb2.Index()
    # idx.ParseFromString(scip_index_bytes)
    # return idx
    raise NotImplementedError("Wire in scip_pb2.Index ParseFromString()")


def cpg_symbols_v1(scip_index: Any) -> pa.Table:
    rows = []
    # Pseudocode depending on SCIP bindings:
    # for sym in scip_index.symbols:
    #   rows.append({...})
    t = pa.Table.from_pylist(rows)
    return ensure_columns(t, CPG_SYMBOLS_V1)


def cpg_symbol_edges_v1(scip_index: Any) -> pa.Table:
    rows = []
    # Pseudocode:
    # for sym in scip_index.symbols:
    #   for rel in sym.relationships:
    #       rows.append({"src_symbol": sym.symbol, "dst_symbol": rel.symbol, "label": rel.relationship, ...})
    t = pa.Table.from_pylist(rows)
    return ensure_columns(t, CPG_SYMBOL_EDGES_V1)


def cpg_occurrences_v1(
    scip_index: Any,
    repo_root: str,
    syntax_leaf_index: pa.Table,
) -> pa.Table:
    """
    Convert SCIP occurrences to stitchable facts, then weld:
      - map (line,col) ranges -> byte offsets using deterministic LineIndex
      - select best matching syntax leaf (exact else smallest containing)
    """
    # Build per-file leaf span index in Python for fast interval matching.
    # leaf rows: {file_id, node_id, kind, start_byte, end_byte}
    leaf_by_file: dict[str, list[tuple[int, int, bytes, str]]] = {}
    for batch in syntax_leaf_index.to_batches():
        cols = batch.to_pydict()
        for file_id, node_id, kind, sb, eb in zip(cols["file_id"], cols["node_id"], cols["kind"], cols["start_byte"], cols["end_byte"]):
            leaf_by_file.setdefault(file_id, []).append((sb, eb, node_id, kind))

    # Sort for determinism
    for file_id in leaf_by_file:
        leaf_by_file[file_id].sort(key=lambda x: (x[0], x[1], x[3]))

    def best_leaf(file_id: str, sb: int, eb: int) -> tuple[bytes | None, str, str]:
        """
        Implements doc weld rule:
          - prefer exact match
          - else smallest containing node
          - deterministic tie-breakers
        :contentReference[oaicite:39]{index=39}
        """
        cands = leaf_by_file.get(file_id, [])
        exact = [c for c in cands if c[0] == sb and c[1] == eb]
        if exact:
            # deterministic: choose kind priority then traversal order
            exact.sort(key=lambda x: (0, x[3], x[0], x[1]))
            return exact[0][2], "exact", "exact-span"

        containing = [c for c in cands if c[0] <= sb and c[1] >= eb]
        if containing:
            # smallest containing span
            containing.sort(key=lambda x: ((x[1] - x[0]), x[3], x[0], x[1]))
            return containing[0][2], "contained", "smallest-containing"
        return None, "unmatched", "no-leaf-span"

    rows = []

    # Pseudocode iteration over scip_index.documents and occurrences.
    # Each occurrence has: range, symbol, symbol_roles, maybe enclosing_range.:contentReference[oaicite:40]{index=40}
    #
    # for doc in scip_index.documents:
    #   file_id = doc.relative_path (normalize to your canonical file_id)
    #   file_bytes = open(Path(repo_root)/file_id, "rb").read()
    #   li = LineIndex.from_bytes(file_bytes, encoding="utf-8" or scip_index.metadata.text_document_encoding)
    #   for occ in doc.occurrences:
    #       (sl, sc, el, ec) = occ.range (normalize 0-based)
    #       sb = li.lc_to_byte(sl, sc, file_bytes)
    #       eb = li.lc_to_byte(el, ec, file_bytes)
    #       node_id, quality, reason = best_leaf(file_id, sb, eb)
    #       rows.append({...})

    t = pa.Table.from_pylist(rows)
    return ensure_columns(t, CPG_OCCURRENCES_V1)
```

---

## Driver skeleton (with optional dynamic execution later)

If/when you switch to Hamilton dynamic execution for per-file parsing, the docs require enabling it explicitly: `Builder.enable_dynamic_execution(allow_experimental_mode=True)`. 

Also note the caveats you should design around (no nested blocks; only one `Collect[]` input per function), which is why I wrapped per-file outputs into a single `SyntaxFileArtifacts` object. 

```python
# hamilton_modules/stage_ab_driver_example.py
from __future__ import annotations

from hamilton import driver
# from hamilton.execution import executors  # if you enable dynamic execution

from hamilton_modules import stage_a_syntax, stage_b_scip

def build_driver():
    b = driver.Builder().with_modules(stage_a_syntax, stage_b_scip)

    # Later, when you make Stage A truly per-file parallel:
    # b = (
    #   b.enable_dynamic_execution(allow_experimental_mode=True)  # required:contentReference[oaicite:43]{index=43}
    #    .with_local_executor(executors.SynchronousLocalTaskExecutor())
    #    .with_remote_executor(executors.MultiProcessingExecutor(max_tasks=8))
    # )

    return b.build()

def run(repo_root: str, commit: str, scip_index_path: str):
    dr = build_driver()
    return dr.execute(
        final_vars=[
            "repo_files_v1",
            "cpg_nodes_v1",
            "cpg_edges_v1",
            "cpg_symbols_v1",
            "cpg_symbol_edges_v1",
            "cpg_occurrences_v1",
        ],
        inputs={
            "repo_root": repo_root,
            "commit": commit,
            "scip_index_path": scip_index_path,
        },
    )
```

---

## How this fits your “DuckDB is storage + query serving” goal

Once these Arrow tables exist, DuckDB can query Arrow objects directly in Python scope (zero-copy path), which is exactly what you want for “serve to agents via DuckDB + FastMCP” workflows. 

---

## What you should implement next (to make this truly “best in class”)

1. **Implement SCIP decoding + normalized ranges** (the `scip_pb2` bindings), and ensure you respect SCIP’s `text_document_encoding`. 

2. **Make welding measurable**: emit simple metrics like `% occurrences matched`, by file/kind, and keep `weld_quality/weld_reason` for debugging. The doc explicitly treats this weld as the critical step. 

3. **Add CST support** (LibCST) to improve identifier span precision; the doc positions CST as the “lossless anchor” for exact spans. 

4. **Optional, but strongly recommended**: upgrade Stage A to a `Parallelizable/Collect` per-file parse block when you’re ready; it’s the cleanest “incremental recompute boundary” for repo-scale analysis. (Enable it exactly as Hamilton documents; design around its caveats.) 

---

If you want, I can extend this in the same style to include the *next* north-star layer after Stage A+B: **Stage C (CFG edges on existing syntax nodes)** with a minimal `cpg_cfg_edges_v1` schema that follows the doc’s guidance (“CFG edges connect executable syntax nodes, not separate CFG-only nodes”). 

Below is a **Stage C “north-star” design** that stays faithful to the guidance in your CPG construction doc:

* **CFG is per-function and intraprocedural**, rooted at function/method bodies.
* Prefer **statement-level CFG** first (simpler, and aligns well with later PDG/CPG layers).
* **Emit CFG edges on existing syntax nodes** (statements/predicates), not separate CFG-only nodes, unless you later add basic blocks. 

I’ll give you:

1. the **minimal `cpg_cfg_edges_v1` schema** (Arrow/DuckDB-friendly),
2. the **edge semantics** (what edges you must emit for Python),
3. a **Hamilton Stage C module skeleton** (nodes + responsibilities),
4. **validation checks** (“CFG sanity”) you can bake in.

You can treat this as the contract for Stage C, while still keeping your implementation maximally inference-driven (no hand-authored schemas for the *code*; only the *output tables* have a stable contract).

---

## Stage C design choice: statement-level CFG on syntax nodes

Your doc explicitly calls out two “best-in-class” granularities and recommends connecting executable syntax nodes (statements/predicates) via CFG edges. 

For v1 I strongly recommend:

* **Statement-level CFG** where “nodes” are already-existing `SyntaxNode`s of kinds like:

  * `stmt:*` (assign, expr_stmt, return, raise, break, continue, import, with, try, etc.)
  * `expr:*` only when it is a predicate / decision point (`if_test`, `while_test`, `match_subject`, etc.)
* CFG edges are **just rows** that refer to `src_node_id` and `dst_node_id` from the syntax layer.

This keeps the “welded layers” property intact: the *same* syntax nodes participate in AST edges, CFG edges, and later DFG/CDG edges.

---

## Minimal schema: `cpg_cfg_edges_v1`

### Core requirement from the doc

The doc’s “emit” shape is essentially:

`Edge(label="CFG", src=stmt_or_predicate_node, dst=next_stmt_node, branch="T|F|case", …)` 

So the minimal table must at least support:

* `src_node_id`
* `dst_node_id`
* `branch` (at least `T`, `F`, `case`, plus a default/none)

### `cpg_cfg_edges_v1` columns

**Required columns (minimum viable):**

* `repo_id: string`
* `snapshot_id: string` (commit SHA or content hash; your “repo snapshot contract”)
* `function_node_id: int64` (the syntax node id for the function/method whose CFG this edge belongs to)
* `src_node_id: int64` (syntax node id; executable stmt or predicate)
* `dst_node_id: int64` (syntax node id; executable stmt or predicate)
* `branch: string` (nullable)

  * Suggested allowed values:

    * `NULL` / `""` for straight-line flow
    * `"T"`, `"F"` for boolean branches
    * `"case"` for match/case (optionally combined with `case_key`)
* `edge_ordinal: int32`

  * deterministic tie-breaker for multiple outgoing edges from the same `src_node_id` (so the edge set is stable across runs)

**Strongly recommended (still “v1-friendly”):**

* `edge_kind: string` (nullable but recommended)
  Use this to avoid encoding everything into `branch`. Suggested values:

  * `"NEXT"` (straight-line)
  * `"BRANCH"` (if/elif/ternary-style)
  * `"LOOP_BACK"`
  * `"BREAK"`
  * `"CONTINUE"`
  * `"EXCEPT"`
  * `"FINALLY"`
  * `"RETURN"`
  * `"RAISE"`
  * `"YIELD"` (if you model generator resumption)
* `confidence: float32` (default 1.0)
  Useful for Python where exceptional edges / implicit flows can be conservative; your doc explicitly recommends storing uncertainty as properties rather than forcing a single wrong answer. 
* `reason: string` (nullable)
  e.g., `"syntactic"`, `"implicit_exception_flow"`, `"conservative_try_edge"`, etc.

### PyArrow schema (recommended types)

Here’s a good Arrow contract that stays simple but scales:

```python
import pyarrow as pa

cpg_cfg_edges_v1 = pa.schema(
    [
        pa.field("repo_id", pa.string(), nullable=False),
        pa.field("snapshot_id", pa.string(), nullable=False),

        pa.field("function_node_id", pa.int64(), nullable=False),
        pa.field("src_node_id", pa.int64(), nullable=False),
        pa.field("dst_node_id", pa.int64(), nullable=False),

        pa.field("branch", pa.string(), nullable=True),
        pa.field("edge_kind", pa.string(), nullable=True),

        pa.field("edge_ordinal", pa.int32(), nullable=False),

        pa.field("confidence", pa.float32(), nullable=True),
        pa.field("reason", pa.string(), nullable=True),
    ],
    metadata={
        b"table": b"cpg_cfg_edges_v1",
        b"layer": b"CFG",
        b"version": b"1",
        b"contract": b"CFG edges connect executable SyntaxNodes (statements/predicates)",
    },
)
```

If you want extra compression/perf, you can dictionary-encode `edge_kind` and `branch`, but it’s not required for MVP.

### DuckDB DDL (storage/query serving)

```sql
CREATE TABLE cpg_cfg_edges_v1 (
  repo_id          VARCHAR NOT NULL,
  snapshot_id      VARCHAR NOT NULL,
  function_node_id BIGINT  NOT NULL,
  src_node_id      BIGINT  NOT NULL,
  dst_node_id      BIGINT  NOT NULL,
  branch           VARCHAR,
  edge_kind        VARCHAR,
  edge_ordinal     INTEGER NOT NULL,
  confidence       FLOAT,
  reason           VARCHAR
);
```

---

## Edge semantics for Python: what to emit (v1 rules)

The goal is: **every function has a CFG over executable syntax nodes**.

### 1) Straight-line blocks

For a sequence of statements `s1, s2, s3`:

* `s1 -> s2 (edge_kind="NEXT")`
* `s2 -> s3 (edge_kind="NEXT")`

### 2) `if / elif / else`

Model the **predicate node** (the syntax node for the test expression) as the decision point:

* `pred -> first_stmt(body) (branch="T", edge_kind="BRANCH")`
* `pred -> first_stmt(orelse) (branch="F", edge_kind="BRANCH")`

  * If no `else`, then `pred -> next_after_if (branch="F")`

Then connect all terminal statements in each arm to `next_after_if` (unless they terminate via return/raise).

### 3) `while`

Let `pred` be the loop test expression node:

* `pred -> first_stmt(body) (branch="T")`
* `pred -> next_after_loop (branch="F")`
* terminal nodes of body that don’t break/return/raise:

  * `tail(body) -> pred (edge_kind="LOOP_BACK")`
* `continue`:

  * `continue_stmt -> pred (edge_kind="CONTINUE")`
* `break`:

  * `break_stmt -> next_after_loop (edge_kind="BREAK")`

### 4) `for` (+ Python’s `for-else`)

Treat the iteration check as the predicate-like decision point (you can use the `for` statement syntax node itself if you don’t have a distinct “iter decision” node in your syntax layer):

* `for_pred -> first_stmt(body) (branch="T")`
* If there is an `else`:

  * `for_pred -> first_stmt(orelse) (branch="F")`
  * terminal nodes of orelse flow to `next_after_loop`
* If no `else`:

  * `for_pred -> next_after_loop (branch="F")`
* `break` edges go to `next_after_loop` (skipping else), as above.

### 5) `match/case` (Py 3.10+)

Let `match_subject` (or the `match` statement node) be the dispatch point:

* `subject -> first_stmt(case_i) (branch="case", edge_ordinal=i)`
* Optionally store `case_key` later (pattern text hash or normalized pattern kind) as an added column in v2.

### 6) `try/except/finally` (minimal, conservative v1)

This is where v1 often starts conservative:

* Normal flow: within `try` block, connect statements with `NEXT`.
* Exceptional flow (v1 conservative):

  * Add `EXCEPT` edges from **try entry** (or from each statement in try) to each handler entry, with `confidence < 1.0` and `reason="conservative_try_edge"`.
* Finally flow:

  * connect all normal exits of try/except blocks to finally entry (`edge_kind="FINALLY"`).
  * finally tails go to `next_after_try`.

You can refine exceptional edges later by marking which nodes can raise, but the “store uncertainty” guidance explicitly supports carrying conservative edges with confidence/reason rather than pretending precision. 

---

## Hamilton Stage C module skeleton

You asked for “same style” as earlier: this is the minimal set of Hamilton nodes that makes Stage C clean and robust, while still inference-driven.

> Key design: **CFG recompute boundary is per-function/method**, which the doc calls out as the right granularity for incremental recompute. 

### Node inventory (what each node returns)

1. `cfg_granularity` (config / constant)

* returns `"statement"` (v1)

2. `executable_syntax_nodes_v1(syntax_nodes_v1) -> pa.Table`

* filters syntax nodes to *executable units*:

  * statement nodes
  * predicate expression nodes (if/while/match tests)
* (Optional) adds a boolean `is_executable`

3. `functions_v1(syntax_nodes_v1) -> pa.Table`

* subset of syntax nodes where kind is `function` / `method`
* includes span/byte range so you can collect nodes in the body

4. `function_executable_nodes_index_v1(functions_v1, executable_syntax_nodes_v1) -> dict[int64, list[int64]]`

* per `function_node_id`, list of executable node_ids in source order
* this is the “block structure” input to actual CFG construction

5. `cfg_edges_v1(function_executable_nodes_index_v1, syntax_edges_v1, ...) -> pa.Table`

* builds per-function CFG edges using AST structure + block lists
* outputs a `pa.Table` cast to `cpg_cfg_edges_v1` schema

6. `cfg_health_v1(cfg_edges_v1, functions_v1, executable_syntax_nodes_v1) -> pa.Table`

* emits metrics: counts, % coverage, #functions failing sanity checks

### Minimal pseudo-code for the core builder

```python
def build_cfg_for_function(fn_id, stmts_in_order, ast_structure) -> list[dict]:
    edges = []
    # recursively connect blocks; each construct returns its "exit" nodes
    # emit edges {src_node_id, dst_node_id, branch, edge_kind, edge_ordinal, ...}
    return edges
```

Where “ast_structure” can be obtained either from:

* your existing AST-derived syntax edges (e.g., `CONTAINS`, `AST`), or
* a lightweight per-function AST object you cache in Hamilton.

---

## Stage C validation: “CFG sanity” checks you should enforce

Your doc explicitly recommends health metrics including **CFG sanity: single entry/exit per function and edges resolve to executable nodes**. 

For `cpg_cfg_edges_v1`, implement at least:

1. **All edges resolve to executable nodes**

* every `src_node_id` and `dst_node_id` must exist in `executable_syntax_nodes_v1`

2. **No cross-function leakage**

* `src_node_id` and `dst_node_id` must lie within the byte span of `function_node_id`

3. **Entry/exit sanity**

* compute in-degree zero executable nodes inside each function:

  * expect **exactly 1** entry (or track + report multiple)
* compute terminal nodes:

  * nodes with out-degree zero (excluding `return/raise`) should be rare; report them

4. **Determinism**

* ensure `edge_ordinal` is stable:

  * sort outgoing edges by `(edge_kind priority, branch priority, dst_start_byte)` and assign `0..n-1`

These checks will catch the “drift” class of bugs early.

---

## References you attached

* CPG construction guidance (Stage C + “emit CFG edges on existing syntax nodes”)  
* Stitching principles + health metrics (“CFG sanity… edges resolve to executable nodes”)  

---

If you want the next concrete step after this schema: I can write the **exact “executable node kind taxonomy”** I’d standardize on (a small enum-like set for Python statements/predicates), and a **reference CFG builder** for the subset (`if/while/for/return/break/continue/match`) that produces `cpg_cfg_edges_v1` deterministically.
