Below is an end-to-end “best-in-class” CPG construction pipeline that starts from **CST + AST + SCIP** and ends at a **stitched, queryable Code Property Graph** (syntax + symbol + control flow + dataflow + dependence). I’m describing it in the way production CPG stacks do it: **layered graphs, welded by stable IDs + spans + symbols**, then enriched with flow analyses and interprocedural stitching. (Joern’s spec is a good concrete reference point: a CPG is a **directed, edge-labeled, attributed multigraph** with a **layered schema**.) ([Code Property Graph][1])

---

## 0) The core idea: build *layers*, then “weld” them into a single property graph

You’re constructing (at least) these layers:

1. **Syntax layer** (AST/CST)
2. **Symbol / resolution layer** (SCIP)
3. **Control-flow layer** (CFG)
4. **Dataflow layer** (def-use / reaching defs, aliasing approximations)
5. **Dependence layer** (PDG = CDG + DDG)
6. **Interprocedural layer** (call graph + summaries + cross-file stitching)

Then you merge (“weld”) them so that **the *same* statement/expression nodes** participate in multiple edge sets (AST edges, CFG edges, dataflow edges, control-dependence edges, etc.). This is exactly the “merge classic representations into one graph” definition of CPG. ([Wikipedia][2])

---

## 1) Inputs and invariants (what you must lock down up front)

### 1.1 Repo snapshot contract

Best practice is to treat analysis as operating on an immutable snapshot:

* `repo_root` (URI) + `commit` (or content hash)
* per file: `path`, `content_hash`, `bytes`, `newline_mode`
* environment: interpreter version, dependency lock, platform

This matters because **SCIP ranges** and **parser spans** must refer to *exact* file bytes.

### 1.2 “Span” is your universal join key

Everything gets anchored to a common coordinate system:

* `FileId` (canonical path/URI)
* `Span = (start_line, start_col, end_line, end_col, encoding)`
* optionally also `byte_start, byte_end` (strongly recommended)

SCIP explicitly carries `text_document_encoding` in metadata (e.g., UTF8), and its ranges must be interpreted accordingly. ([help.sourcegraph.com][3])

> **Best practice:** store *both* line/col and byte offsets, and keep a deterministic `LineIndex` (mapping line→byte/char offsets) per file. This avoids “UTF-16 vs UTF-8 column” bugs that silently break stitching.

---

## 2) Stage A — Parse CST + AST and build the Syntax Graph

### 2.1 CST: the “lossless anchor”

Use CST primarily for:

* exact spans (including trivia/formatting),
* stable refactor tooling,
* precise token/identifier locations.

### 2.2 AST: the “semantic skeleton”

Use AST for:

* structural semantics (statements/expressions normalized),
* easier CFG construction,
* easier def-use extraction.

**Best-in-class approach:** pick one as the *primary* node inventory (often AST-like), but keep a **CST↔AST crosswalk** so every node can map back to exact source spans.

### 2.3 Emit the Syntax layer as *tables/graphs*

Create node IDs that are stable and reproducible:

**NodeId strategy (deterministic):**

```
node_id = hash(file_id, node_kind, start_byte, end_byte, (optional) disambiguator)
```

Emit:

* `SyntaxNode(node_id, file_id, kind, span, code_snippet_hash, flags…)`
* `SyntaxEdge(src=node_id, dst=node_id, label="AST|CONTAINS|NEXT_SIBLING|…", order=…)`

Minimum node kinds you’ll want (language-agnostic):

* module/file, class, function/method, parameter, local, statement kinds, expression kinds, identifier, attribute/member access, call, literal, import, return, branch predicates, etc.

> If you want to align with an existing schema, Joern’s CPG spec and docs are an explicit, layered node/edge taxonomy for this style of representation. ([Code Property Graph][1])

---

## 3) Stage B — Ingest SCIP and build the Symbol Graph

SCIP’s job in your pipeline is: **name resolution + stable symbol identity + def/ref indexing**, produced by language-aware tooling (for Python: `scip-python` is a Pyright fork focused on generating SCIP). ([GitHub][4])

### 3.1 Decode / load SCIP

Operationally, `index.scip` is protobuf; Sourcegraph documents decoding it via `protoc --decode=scip.Index scip.proto`. ([help.sourcegraph.com][3])

### 3.2 Convert SCIP “occurrences” into stitchable facts

A SCIP `Document` contains **occurrences** that have:

* a range,
* a symbol,
* symbol_roles (definition vs reference),
* often an enclosing_range (useful for scope/container stitching). ([jsDelivr][5])

### 3.3 The critical weld: Occurrence → SyntaxNode

For each SCIP occurrence:

1. locate `file_id`
2. map SCIP range → byte offsets using your `LineIndex` + encoding rules
3. find the **best matching syntax leaf** (usually an identifier / attribute node):

   * prefer exact span match
   * else choose the *smallest* syntax node whose span contains the occurrence span
   * deterministic tie-breakers (node_kind priority, then smallest span, then traversal order)

Emit:

* `SymbolNode(symbol_id, kind, display_name, docs_hash, …)`
* `SymbolEdge(symbol_id -> symbol_id, label="RELATIONSHIP_*", …)` (where available)
* `Occurrence(node_id -> symbol_id, role=DEF|REF|… , occurrence_span=…)`

This step turns SCIP from “code nav facts” into **semantic glue** that binds syntax nodes to globally stable identities.

### 3.4 Best-in-class symbol stitching outcomes

After this, you can do *cross-file* joins like:

* all references of symbol X across repo,
* def→ref edges,
* import edges resolved to symbol identity,
* method override / implementation links if present in relationships.

---

## 4) Stage C — Build per-function CFG (Control Flow Graph)

CFG construction is usually intraprocedural and rooted at function/method bodies.

### 4.1 Choose your CFG granularity

Two common “best-in-class” options:

* **Statement-level CFG**: nodes are statements/predicates (simple, great for PDG/CPG)
* **Basic-block CFG**: nodes are blocks; statements belong to blocks via `CONTAINS` (more compiler-like)

### 4.2 Emit CFG edges on *existing syntax nodes*

In CPG style, you generally want CFG edges to connect **syntax nodes representing executable units** (statements/predicates), not separate CFG-only nodes (unless you need blocks).

Emit:

* `Edge(label="CFG", src=stmt_or_predicate_node, dst=next_stmt_node, branch="T|F|case", …)`

---

## 5) Stage D — Build Dataflow: def-use chains + aliasing approximations

This is where “cause → effect” starts to feel real.

### 5.1 Extract defs/uses from AST

For each statement/expression, extract:

* definitions: assignments, parameter binding, loop targets, imports, exception binds, attribute writes, subscripts, etc.
* uses: reads of names, attributes, subscripts, call arguments, predicate reads, return values, etc.

Tie defs/uses to **SymbolNodes** whenever possible via the SCIP weld.

### 5.2 Compute reaching definitions (classic forward dataflow)

Run standard dataflow analysis over CFG to compute def→use edges.

Emit:

* `Edge(label="REACHING_DEF" or "DDG", src=def_site_node, dst=use_site_node, symbol_id=?, …)`

### 5.3 Python reality: you need an alias/points-to story

For “best-in-class” Python you won’t get perfect precision (dynamic features), but you can get strong utility with:

* type-guided resolution (Pyright inference, stubs)
* allocation-site abstraction (object created at call/literal site)
* attribute flow via (base_obj, attribute_name) “field” abstraction
* conservative fallback: if unknown, widen alias set rather than lying

> Key best practice: **store uncertainty as properties** (confidence, “dynamic/unknown”), instead of forcing a single wrong answer.

---

## 6) Stage E — Build Control Dependence (CDG) and the PDG

### 6.1 Control dependence via postdominators

Control dependence is typically computed from CFG postdominators (reverse CFG). The standard definition is: **Y is control dependent on X** if Y postdominates *a successor* of X but does not postdominate *all successors* of X. 

Emit:

* `Edge(label="CDG", src=branch_predicate_node, dst=controlled_stmt_node, branch="T|F|case", …)`

### 6.2 PDG = DDG + CDG

Now you can define the Program Dependence Graph edges as:

* **data dependence edges** (def→use, alias-mediated)
* **control dependence edges** (predicate→controlled statement)

This PDG layer is what many people colloquially mean by “cause and effect” in static analysis.

---

## 7) Stage F — Assemble the CPG (merge all layers into one property graph)

At this point your graph is a single node set with multiple edge labels:

* AST/CST structural edges (`AST`, `CONTAINS`, …)
* symbol edges (`REFERS_TO`, `DEFINES`, symbol relationships)
* control flow edges (`CFG`)
* dataflow edges (`DDG` / `REACHING_DEF`)
* control dependence edges (`CDG`)
* plus higher-level semantic edges (`CALLS`, `IMPORTS`, `INHERITS`, …)

This aligns with the formal CPG notion of a **directed, edge-labeled, attributed multigraph** and (in Joern’s spec) a **layered schema**. ([Code Property Graph][1])

---

## 8) Stage G — Interprocedural stitching (what “best-in-class” looks like)

This is the differentiator between a “cool graph” and something that supports truly advanced queries.

### 8.1 Call graph: direct + resolved + conservative

Construct call edges:

1. **Direct symbol calls** (easy): call site resolves to a known function symbol via SCIP+types
2. **Method dispatch**: use type inference + MRO + attribute resolution
3. **Dynamic fallbacks**: record `CALLS_UNKNOWN` with candidate sets if possible

Emit:

* `Edge(label="CALLS", src=call_expr_node, dst=callee_function_node_or_symbol, …)`

### 8.2 Parameter/return flow (“call wiring”)

To enable interprocedural dataflow:

* actual argument expression → formal parameter node
* return expression → call result use site

Emit:

* `Edge(label="ARG_TO_PARAM", src=arg_expr_node, dst=param_node, position=i, …)`
* `Edge(label="RET_TO_CALL", src=return_expr_node, dst=call_expr_node, …)`

### 8.3 Summaries (scalable interprocedural analysis)

Instead of inlining every callee body, build **function summaries**:

* reads/writes sets
* “taint in → taint out” transforms
* side-effect footprints (global writes, attribute writes)
* thrown exceptions, yielded values

Then compose these summaries along call edges.

> This is how you get “best-in-class” scale without exploding runtime.

---

## 9) Best-in-class stitching principles (the stuff that makes it robust)

### 9.1 Deterministic IDs + crosswalk tables

Always keep explicit crosswalks:

* `CST node ↔ AST node`
* `SCIP occurrence ↔ syntax node`
* `syntax node ↔ symbol`
* `syntax node ↔ CFG node` (if you have separate CFG nodes)
* `def/use fact ↔ symbol ↔ node`

### 9.2 Track confidence and unknowns

For Python especially:

* attach `confidence`, `reason` (static, type-inferred, heuristic, dynamic/unknown)
* store candidate sets when ambiguous

### 9.3 Incremental recompute boundary

Recompute at the smallest practical unit:

* parse: file
* CFG/DFG/CDG/PDG: function/method body
* interproc: affected callgraph neighborhood

### 9.4 Validation “health metrics”

Have automated checks like:

* % SCIP occurrences mapped to syntax nodes
* CFG sanity (single entry/exit per function, edges resolve to executable nodes)
* def-use coverage for locals/params
* symbol uniqueness and stable symbol IDs across runs

---

## 10) A concrete “construction order” that works well in practice

1. Build `LineIndex` + byte spans for every file
2. Parse CST + AST, emit syntax nodes/edges
3. Ingest SCIP, weld occurrences → syntax nodes, emit symbol graph
4. For each function: build CFG
5. For each function: extract def/use + compute DDG
6. For each function: compute postdominators → CDG
7. Union edges into PDG + CPG
8. Build call graph + call wiring
9. Add summaries + interproc flow
10. Produce derived semantic graphs (imports, inheritance, taint, slices, etc.)

---

If you want, I can also write the **“minimum viable schema”** (node/edge types + required properties) for a CPG that is *friendly to DuckDB/Arrow storage* (nodes table + edges table + property columns) while staying close to Joern’s layered CPG concepts, so your downstream querying is clean and predictable.

[1]: https://cpg.joern.io/ "Code Property Graph Specification Website | Code Property Graph Specification Website"
[2]: https://en.wikipedia.org/wiki/Code_property_graph?utm_source=chatgpt.com "Code property graph"
[3]: https://help.sourcegraph.com/hc/en-us/articles/15045932124941-Decoding-SCIP-index-file "Decoding SCIP index file – Sourcegraph Help Center"
[4]: https://github.com/sourcegraph/scip-python?utm_source=chatgpt.com "sourcegraph/scip-python: SCIP indexer for Python"
[5]: https://cdn.jsdelivr.net/npm/%40sourcegraph/scip-typescript%400.3.15/src/SnapshotTesting.ts?utm_source=chatgpt.com "https://cdn.jsdelivr.net/npm/@sourcegraph/scip-typ..."
