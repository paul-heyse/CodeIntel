
Below is the **best-in-class “scope delta” plan** I’d implement next *given your current post‑ingestion / storage‑decoupled codebase*, focused on:

1. **Producing a true Code Property Graph (CPG)** via calculations + joins
2. **Streamlining / de‑duplicating Hamilton DAG boilerplate** and shifting more work into **Polars/Arrow table ops**
3. **Ingesting + surfacing a materially broader slice of LibCST + Tree‑sitter data** (without exploding P0)

This is aligned with the staged CPG build you attached (Syntax → SCIP weld → CFG/DFG → dependence) and with your earlier “best‑in‑class ingestion + stitching” plan (P0 syntax facts now, P1 enrichment/resolution and wider tree-sitter data).

---

## Design goal the delta should satisfy

### What “CPG” should mean in your system

A CPG is not “one more table”; it’s a **layered, welded graph**:

* **Syntax layer** (node inventory + parent/child edges; spans as anchors)
* **Name binding layer** (SCIP symbol/occurrence binding welded to syntax nodes)
* **Control flow layer** (CFG)
* **Data flow layer** (DFG / def-use)
* Optionally **Control dependence / PDG** as a derived layer

Your doc’s staged approach captures this and explicitly calls out that **SyntaxNode/SyntaxEdge come first**, then SCIP welding, then CFG, etc.. It also emphasizes spans as the universal join key, and being explicit about **0-based** coordinates and **byte vs char** semantics.

### Architectural invariants to preserve (you already moved here)

* **Build produces Arrow/Parquet datasets** (PyArrow/Polars objects); it never imports storage.
* **DuckDB is a consumer** (loading datasets for serving), not a build dependency.
* **Hamilton remains the orchestrator**; Polars/Arrow remain the compute substrate.

The delta below preserves all of that.

---

## Current baseline (what you already have that we should reuse)

From the codebase you shared, you already have most of the CPG “ingredients”:

### P0 syntax facts (already exist)

Your best-in-class plan called these out explicitly as P0 outputs: `core.syntax_spans`, `core.syntax_scopes`, `core.syntax_defs`, `core.syntax_refs`, `core.syntax_calls`, `core.syntax_imports`, and you’re producing them.

### SCIP ingestion + partial resolution (already exist)

You have SCIP datasets + crosswalk outputs (e.g., occurrence/symbol xrefs). This maps to Stage B in the CPG doc (symbols + occurrences).

### Flow graphs + call/import graphs (already exist)

You already produce `graph.cfg_*`, `graph.dfg_*`, plus call graph/import graph/symbol-use style artifacts.

So the **net-new** work is mostly:

* “Make syntax a real node+edge layer (not just facts)”
* “Make the SCIP→syntax weld explicit and cheap downstream”
* “Materialize a *unified* CPG view (nodes/edges)”
* “Optionally add CDG/PDG tables”

---

## The CPG delta: what to add/change

### Delta A — Add canonical “syntax graph” tables (nodes + edges)

Your CPG construction doc makes Stage A a **first-class syntax node/edge emission step**. Today you have:

* LibCST: `core.cst_nodes` (but it does **not** currently encode enough parent/child identity to serve as an edge layer)
* Tree-sitter: `core.ts_captures` (captures are not a full parse tree)
* P0 syntax facts: spans/scopes/defs/refs/calls/imports (excellent, but not an AST/CST edge layer)

**Best-in-class move:** introduce **two canonical tables** that become the universal syntax graph surface area for everything downstream.

#### New tables

* `core.syntax_nodes`
* `core.syntax_edges`

These should be **producer-agnostic** (LibCST vs tree-sitter vs “python_ast” if you ever add it). This keeps downstream Hamilton logic generic.

#### Recommended schema (representative)

```python
# core.syntax_nodes
# PK: (repo, commit, rel_path, producer, node_id)
[
  repo, commit, rel_path,
  producer,          # "libcst" | "tree_sitter" | ...
  language,          # "python", "go", "ts", ...
  node_id,           # stable string OR uint128
  node_kind,         # normalized kind (if possible)
  raw_kind,          # exact parser kind/type
  start_byte, end_byte,
  start_line, start_col, end_line, end_col,   # 0-based
  text_preview,
  extras_json,
]

# core.syntax_edges
# PK: (repo, commit, rel_path, producer, parent_node_id, child_node_id, edge_kind, child_ordinal)
[
  repo, commit, rel_path,
  producer,
  parent_node_id,
  child_node_id,
  edge_kind,         # "AST_CHILD" | "FIELD" | "SIBLING" (keep minimal at first)
  field_name,        # optional (tree-sitter supports this better than libcst)
  child_ordinal,     # stable ordering among children
]
```

#### Why this matters

* It cleanly implements Stage A (SyntaxNode/SyntaxEdge).
* It gives you a **single** cross-language syntax substrate.
* It makes later joins stable because spans are first-class columns (and your doc stresses span semantics).

#### How to populate (LibCST)

Today your LibCST node extraction records parent *kinds*, not parent *node IDs*. To make LibCST usable for edges:

* Change the visitor to maintain a **stack of node_ids**, not just node kinds.
* Emit:

  * a row for the current node
  * an edge row from parent_id → node_id

Representative visitor pattern:

```python
class SyntaxGraphVisitor(cst.CSTVisitor):
    def __init__(self, source_index: LineIndexedSource):
        self.node_stack: list[str] = []
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self.child_counters: list[int] = []   # parallel stack for child ordinal

    def on_visit(self, node: cst.CSTNode) -> bool:
        node_id = stable_node_id_for_cst(node, source_index)
        parent_id = self.node_stack[-1] if self.node_stack else None

        # emit node row
        self.nodes.append({...})

        # emit edge row
        if parent_id is not None:
            ordinal = self.child_counters[-1]
            self.child_counters[-1] += 1
            self.edges.append({
                "parent_node_id": parent_id,
                "child_node_id": node_id,
                "edge_kind": "AST_CHILD",
                "child_ordinal": ordinal,
            })

        # push
        self.node_stack.append(node_id)
        self.child_counters.append(0)
        return True

    def on_leave(self, node: cst.CSTNode) -> None:
        self.node_stack.pop()
        self.child_counters.pop()
```

*(Note: “field_name” is hard to get losslessly from LibCST’s visitor API; the best-in-class compromise is: emit stable `child_ordinal`, and use byte-range sorting if you need determinism beyond visitor order.)*

#### How to populate (Tree-sitter)

Your existing `ts_captures` output is capture-based. For CPG you also want **full parse tree nodes/edges**, which tree-sitter supports naturally.

Add:

* `core.ts_nodes`
* `core.ts_edges`

Then either:

* (Preferred) derive canonical `core.syntax_nodes/core.syntax_edges` directly from tree-sitter parse
* or (Acceptable) keep ts_* tables and have one canonicalization step

This “full tree” ingestion is also consistent with your earlier plan to leverage tree-sitter as a rich input source, not just captures.

---

### Delta B — Make the SCIP → syntax weld explicit and cheap

Your CPG doc calls the “critical weld” the mapping between SCIP Occurrences and SyntaxNodes, and it places this immediately after syntax emission.

Right now, your world has:

* SCIP occurrences (ranges)
* syntax spans (ranges)
* but the weld is still “implicit” (teams join later, differently)

**Best-in-class move:** materialize a dedicated xref table that says:

> “this SCIP occurrence binds to this syntax node/span id, using these deterministic rules”

#### New table

* `core.scip_occurrence_syntax_xref`

Representative schema:

```python
# PK: (repo, commit, rel_path, scip_symbol, scip_occurrence_id, producer)
[
  repo, commit, rel_path,
  producer,                 # which syntax graph we're welding to (libcst/tree_sitter)
  scip_document_uri,
  scip_symbol,
  scip_occurrence_id,       # stable within document (or hash of (range, symbol, role))
  occ_start_byte, occ_end_byte,
  occ_start_line, occ_start_col, occ_end_line, occ_end_col,
  syntax_node_id,           # output of the weld
  match_kind,               # "EXACT" | "CONTAINS" | "NEAREST"
  candidate_count,
]
```

#### Deterministic match rules

Use the span semantics guidance from your doc—store bytes and 0-based line/col, and treat bytes as the authoritative join space whenever possible.

Algorithm (per file):

1. Build an interval index of syntax nodes by `(start_byte, end_byte)`
2. For each SCIP occurrence:

   * prefer exact span match
   * else smallest containing node
   * else nearest enclosing *named* node (configurable)

This is exactly where `intervaltree` becomes “best in class” (you’re already using it elsewhere).

Representative weld function:

```python
from intervaltree import IntervalTree

def weld_occurrences_to_syntax_nodes(syntax_nodes_df: pl.DataFrame,
                                    occ_df: pl.DataFrame) -> pl.DataFrame:
    # Build interval tree: [start, end) -> node_id
    tree = IntervalTree(
        (row["start_byte"], row["end_byte"], row["node_id"])
        for row in syntax_nodes_df.iter_rows(named=True)
        if row["start_byte"] is not None and row["end_byte"] is not None
    )

    out = []
    for occ in occ_df.iter_rows(named=True):
        s, e = occ["occ_start_byte"], occ["occ_end_byte"]
        candidates = sorted(tree.overlap(s, e), key=lambda iv: (iv.end - iv.begin))
        if candidates:
            chosen = candidates[0]  # smallest containing
            out.append({**occ, "syntax_node_id": chosen.data, "match_kind": "CONTAINS"})
        else:
            out.append({**occ, "syntax_node_id": None, "match_kind": "NONE"})
    return pl.DataFrame(out)
```

(You can later optimize this by grouping by `rel_path` and running the weld in parallel shards; the correctness contract stays the same.)

---

### Delta C — Add “resolved syntax fact” tables (P1), then build the CPG from them

Your earlier best-in-class plan explicitly calls out a P1 enrichment step producing resolved/enriched fact tables (e.g., `syntax_refs_resolved`, `syntax_calls_resolved`, etc.).

**Best-in-class move:** implement `syntax_enrich` now as the “bridge” between:

* P0 syntax facts (defs/refs/calls/imports)
* SCIP symbol resolution (occurrence→symbol, occurrence→syntax node)

#### New P1 tables (recommended)

* `core.syntax_defs_resolved`
* `core.syntax_refs_resolved`
* `core.syntax_calls_resolved`
* `core.syntax_imports_resolved`

Each should carry a canonical identity:

* source span (`span_id` or `syntax_node_id`)
* resolved SCIP symbol (and optionally resolved GOID, where possible)

This prevents downstream components (CPG, metrics, storage views) from re‑implementing resolution.

---

### Delta D — Materialize the unified CPG as nodes + edges tables

Now you have:

* canonical syntax graph tables
* canonical SCIP weld tables
* P0/P1 fact tables
* CFG/DFG/call/import edges

The CPG becomes a **simple assembly job**.

#### New graph outputs

* `graph.cpg_nodes`
* `graph.cpg_edges`

##### `graph.cpg_nodes` (minimal index, not full duplication)

Store:

* a stable `cpg_node_id`
* a `node_kind`
* enough pointers to join back to the underlying “source table” row

Representative:

```python
[
  repo, commit,
  cpg_node_id,         # uint128 recommended
  node_kind,           # "SYNTAX_NODE" | "SCIP_SYMBOL" | "GOID" | "CFG_BLOCK" | ...
  source_table_key,    # "core.syntax_nodes", "core.scip_symbols", ...
  source_pk_json,      # stable pk encoding
  rel_path,            # optional convenience
  start_byte, end_byte # optional convenience
]
```

##### `graph.cpg_edges` (the real payload)

Each row is a typed edge (multi-graph). Keep it narrow; properties go in `extras_json` if needed.

Representative:

```python
[
  repo, commit,
  src_cpg_node_id,
  dst_cpg_node_id,
  edge_kind,           # "AST" | "CONTAINS" | "DEFINES" | "REFERS_TO" | "CALLS" | "CFG" | "DFG" | ...
  edge_layer,          # "SYNTAX" | "SYMBOL" | "FLOW"
  rel_path,            # for local edges
  ordinal,             # for AST child ordering
  extras_json,
]
```

#### Hamilton implementation pattern

Create a new target module:

* `codeintel/build/hamilton/native/graphs/cpg.py`

and a new target function:

```python
CPG_TARGET_NAME = "cpg"
CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
CPG_EDGES_TABLE_KEY = "graph.cpg_edges"

@codeintel_target(CPG_TARGET_NAME, domain="graphs")
def cpg(
    build_env: BuildEnv,
    save_relation_table: SaveRelationTable,
    cpg_nodes: pl.LazyFrame,
    cpg_edges: pl.LazyFrame,
) -> list[ArtifactMetadata]:
    return [
        save_relation_table(RelationTableSaveSpec(
            table_key=CPG_NODES_TABLE_KEY,
            frame=cpg_nodes,
            # ... partitioning, etc.
        )),
        save_relation_table(RelationTableSaveSpec(
            table_key=CPG_EDGES_TABLE_KEY,
            frame=cpg_edges,
        )),
    ]
```

Then build `cpg_edges` as a `pl.concat` of normalized edge frames.

Example assembly (representative):

```python
@tag_dataset(table_key=CPG_EDGES_TABLE_KEY, kind="relation_table")
def cpg_edges(
    syntax_edges: pl.LazyFrame,              # core.syntax_edges normalized into cpg form
    cfg_edges: pl.LazyFrame,                 # graph.cfg_edges normalized
    dfg_edges: pl.LazyFrame,                 # graph.dfg_edges normalized
    call_edges: pl.LazyFrame,                # graph.call_graph_edges normalized
    scip_occ_weld: pl.LazyFrame,             # core.scip_occurrence_syntax_xref
) -> pl.LazyFrame:
    return pl.concat(
        [
            syntax_edges.select(["repo","commit","src","dst","edge_kind","edge_layer","rel_path","ordinal","extras_json"]),
            cfg_edges.select(...),
            dfg_edges.select(...),
            call_edges.select(...),
            scip_occ_weld.select(...),
        ],
        how="vertical_relaxed",
    )
```

---

### Delta E — Add CDG/PDG as “derived layers” (optional but best-in-class)

Your CPG doc’s stage order includes Control Dependence (CDG) and PDG as later steps.

You already have:

* CFG edges
* DFG edges (block-level)

Best-in-class next:

* `graph.cdg_edges`: compute postdominators on each function CFG, emit control-dependence edges
* `graph.pdg_edges`: union of CDG + DFG (with `edge_kind` differentiating)

This can be implemented using NetworkX algorithms or custom dominator computation; the key is: **PDG is derived** and should not complicate earlier ingestion contracts.

---

## Streamlining / efficiency delta in the Hamilton DAG

You asked specifically: “we have repeated boilerplate; can we condense into joins/table ops and use Hamilton decorators/generalized functions?”

Here are the best “ROI” refactors I’d do **without weakening correctness**:

### 1) Push “join math” into Polars SQL or expression pipelines

Where you currently have “read → Python transform → write”, you can often reduce code to:

* register frames
* execute SQL join
* return LazyFrame

Polars SQLContext is designed exactly for “multi-source SQL as just another transform” and supports registering lazy scans + pyarrow tables with lazy benefits.

Representative pattern:

```python
import polars as pl

def resolved_refs_sql(syntax_refs: pl.LazyFrame,
                      scip_occ_xref: pl.LazyFrame) -> pl.LazyFrame:
    with pl.SQLContext(
        refs=syntax_refs,
        occ=scip_occ_xref,
        eager=False,
    ) as ctx:
        return ctx.execute("""
            SELECT
              refs.*,
              occ.scip_symbol,
              occ.role
            FROM refs
            LEFT JOIN occ
              ON refs.repo = occ.repo
             AND refs.commit = occ.commit
             AND refs.rel_path = occ.rel_path
             AND refs.start_byte = occ.occ_start_byte
             AND refs.end_byte = occ.occ_end_byte
        """)
```

This *massively* reduces bespoke join code and makes the DAG easier to reason about.

### 2) Introduce a single “graph assembly utility” for CPG edge normalization

Instead of repeating:

* rename columns
* add constants
* select common set

Create one helper:

```python
def as_cpg_edges(
    lf: pl.LazyFrame,
    *,
    src_col: str,
    dst_col: str,
    edge_kind: str,
    edge_layer: str,
    extra_cols: list[str] = (),
) -> pl.LazyFrame:
    return (
        lf.with_columns([
            pl.lit(edge_kind).alias("edge_kind"),
            pl.lit(edge_layer).alias("edge_layer"),
        ])
        .rename({src_col: "src", dst_col: "dst"})
        .select(["repo","commit","src","dst","edge_kind","edge_layer",*extra_cols])
    )
```

Then the CPG assembly reads like a spec, not a pile of column plumbing.

### 3) Keep “variants” only where they truly skip heavy upstream compute

Your current compute/existing/empty pattern is justified when “existing” avoids triggering heavy dependencies.

But you can reduce boilerplate by:

* concentrating variants in **subdags** (one variant switch controls a whole family)
* or generating trivial `*_empty` frames via a shared helper (which you already do in places)

The “best-in-class” north star is: the majority of nodes are “pure relational transforms”, and only the ingestion + a handful of heavy analyzers need variants.

---

## Expanding LibCST + Tree-sitter ingestion “fully” without blowing up P0

Your earlier best-in-class doc already separated:

* P0: core syntax facts
* P1: richer/wider tables (`syntax_index_ext`, more tree-sitter, etc.)

I strongly agree with that separation.

### P0 stays: facts + canonical syntax graph

* keep P0 as:

  * `core.syntax_*` facts
  * `core.syntax_nodes/core.syntax_edges` (canonical graph substrate)

### P1 add: “wide” + “high-volume” artifacts

Examples of **P1-only** additions (best-in-class but optional):

* `core.syntax_tokens` (identifiers, keywords, operators, literals)
* `core.syntax_trivia` (comments, docstrings, whitespace)
* `core.syntax_types` (annotations, inferred types if you later add)
* `core.ts_nodes/core.ts_edges` (full parse tree)
* additional tree-sitter query packs for more languages

This preserves your *practical inference-driven* approach: P0 is always available; P1 is available when desired.

---

## Concrete “next implementation slice” (what I’d build first)

If you want the highest-leverage ordering (teams can parallelize cleanly):

### Slice 1 (unblocks everything): canonical syntax graph

1. Implement `core.syntax_nodes/core.syntax_edges`
2. Populate from:

   * LibCST for Python
   * Tree-sitter parse tree for non-Python (or at least one other language initially)

### Slice 2 (critical weld): SCIP occurrence → syntax node

3. Implement `core.scip_occurrence_syntax_xref`
4. Add deterministic tie-break rules and a conformance test (golden)

### Slice 3 (CPG assembly): `graph.cpg_nodes` + `graph.cpg_edges`

5. Materialize CPG edges as union of:

   * AST edges (syntax_edges)
   * SCIP binding edges (occurrence binds/refers)
   * call/import edges
   * CFG/DFG edges

### Slice 4 (best-in-class “full”): CDG/PDG + resolved syntax facts

6. Add `syntax_*_resolved`
7. Add `graph.cdg_edges`, `graph.pdg_edges`

---

## Why this delta is “best-in-class” for LLM-facing metadata

It gives you:

* A **single, cross-language syntax substrate** (nodes+edges), not “Python gets special handling and other langs are captures-only”
* Explicit, deterministic **SCIP welding** (no “everyone joins ranges differently”)
* A unified **property-graph export** that downstream storage/serving can consume without needing to understand every upstream table
* Preservation of your key architectural constraint: **build is Arrow/Polars-first; storage is a consumer**

And it follows the staged construction you attached: Syntax first, SCIP second, CFG later, with careful span semantics called out explicitly.

---

If you want, I can follow up with a **CPG contract page** in the same style as your suite contracts (dataset keys + PKs + required columns + span semantics) specifically for:

* `core.syntax_nodes`
* `core.syntax_edges`
* `core.scip_occurrence_syntax_xref`
* `graph.cpg_nodes`
* `graph.cpg_edges`

That contract is the single best way to eliminate ambiguity before multiple engineers start implementing in parallel.

Below is the **implementation plan (dense narrative + representative snippets)** for the **net-new “best-in-class scope delta”** you described: (1) **full Code Property Graph (CPG) construction**, (2) **Hamilton DAG streamlining** (beyond just “edge outputs”), and (3) **maximal LibCST + tree-sitter surface ingestion** integrated into the same Arrow/Polars-first pipeline.

The plan assumes your current state matches what you described: **P0 ingestion exists**, build is fully **storage-decoupled**, and **PyArrow datasets are the contract artifacts** that downstream consumers (DuckDB/storage/serving) load.

---

## 0) Design goal and invariants (so every change has a “why”)

### Goal

Produce a **stitched, queryable Code Property Graph** (syntax + symbol + control flow + dataflow + dependence + interprocedural edges) as **Arrow datasets** designed for downstream AI/LLM consumption. The CPG is built as **layered graphs welded by stable IDs + spans + symbols**—exactly the production posture described in your CPG overview. 

### Hard invariants

1. **Span is the universal join key**: everything anchors to file + span, and you store deterministic byte offsets as the “gold standard join coordinate”. Your CPG notes emphasize mapping SCIP ranges → byte offsets via a LineIndex and encoding rules before stitching. 
2. **Deterministic IDs**: nodes/edges must be reproducible across runs.
3. **No build→storage dependency**: build emits Arrow datasets + manifests; storage only consumes.

---

## 1) Net-new datasets to add for a full CPG (and how they relate)

You already have (or are close to having) “syntax facts” and SCIP ingestion. A full CPG requires you to promote those into two canonical relational artifacts:

### 1.1 Canonical “property graph” tables

You will produce at minimum:

* `cpg.nodes` (single node inventory)
* `cpg.edges` (single edge inventory)

…while *also* retaining “typed” edge tables for ergonomics/validation:

* `cpg.edges_ast`
* `cpg.edges_cfg`
* `cpg.edges_ddg` (data dependence)
* `cpg.edges_cdg` (control dependence)
* `cpg.edges_call` (call graph)
* `cpg.edges_arg_param`, `cpg.edges_ret_call` (call wiring)

Your CPG overview explicitly calls out **SyntaxNode/SyntaxEdge** as the base and then layering CFG/dataflow edges over the same underlying nodes. 

### 1.2 Crosswalk tables (these are not optional if you want robustness)

You should explicitly persist crosswalks like:

* `xref.cst_node_to_syntax_node` (if CST is used for anchors)
* `xref.ast_node_to_syntax_node`
* `xref.scip_occurrence_to_syntax_node`
* `xref.syntax_node_to_symbol`

The CPG doc is explicit: keep crosswalks so “the same statement/expression nodes participate in multiple edge sets” and stitching stays deterministic. 

---

## 2) Construction order (this matters for correctness *and* Hamilton factoring)

You want your pipeline to follow the concrete construction order outlined in the CPG overview (because it naturally defines Hamilton module boundaries and caching boundaries):

1. Build `LineIndex` + byte spans per file
2. Parse CST + AST → emit syntax nodes/edges
3. Ingest SCIP → weld occurrences → syntax nodes → emit symbol graph
4. For each function: build CFG
5. For each function: extract def/use + compute DDG
6. For each function: compute postdominators → CDG
7. Union edges into PDG + CPG
8. Build call graph + call wiring
9. Add summaries + interproc flow
10. Produce derived semantic graphs 

This is the “shape” your Hamilton DAG should reflect—i.e., **modules by stage**, and **datasets by stage outputs**.

---

## 3) Stage A — Syntax graph (AST/CST promoted into a node/edge inventory)

### 3.1 Deterministic node identity (do this once; everything else benefits)

Your CPG notes already define the canonical emission shape: `SyntaxNode(node_id, file_id, kind, span, ...)` and `SyntaxEdge(src,dst,label,order)`. Make that real and make it *stable*:

**NodeId strategy (recommended)**

* Use **(repo_snapshot_id, file_id, start_byte, end_byte, kind, disambiguator)** → hash128
* The only legitimate reason for a disambiguator is: *multiple nodes with identical spans* (rare but possible in some CST representations).

Representative snippet (hash128 + schema-stable types):

```python
import xxhash

def node_id_h128(repo_id: str, file_id: str, kind: str, start_b: int, end_b: int, disambig: int = 0) -> bytes:
    h = xxhash.xxh128()
    h.update(repo_id.encode())
    h.update(b"\x1f")
    h.update(file_id.encode())
    h.update(b"\x1f")
    h.update(kind.encode())
    h.update(b"\x1f")
    h.update(str(start_b).encode())
    h.update(b"\x1f")
    h.update(str(end_b).encode())
    h.update(b"\x1f")
    h.update(str(disambig).encode())
    return h.digest()   # 16 bytes
```

### 3.2 Node inventory source of truth (AST vs CST)

Best practice (and consistent with your CPG writeup) is:

* **AST drives semantic structure** (CFG/DFG friendliness)
* **CST anchors exact spans** (lossless evidence)

So: pick one inventory (usually AST-ish), but emit `xref.cst↔ast` to preserve evidence-grade spans.

---

## 4) Stage B — SCIP weld (symbol graph bound to syntax graph)

Your “critical weld” is precisely defined: for each SCIP occurrence, map its range to bytes via LineIndex+encoding, then match the smallest syntax leaf that contains it, preferring exact span matches, with deterministic tie-breaking. 

### 4.1 The core weld algorithm (representative, deterministic)

```python
def weld_occurrence_to_node(occ_span, syntax_leaves_by_file):
    # occ_span: (file_id, start_b, end_b)
    leaves = syntax_leaves_by_file[occ_span.file_id]

    # 1) exact match
    exact = leaves.get((occ_span.start_b, occ_span.end_b))
    if exact:
        return exact.node_id

    # 2) containment: choose smallest containing span; tie-break by kind priority then traversal order
    candidates = [
        n for n in leaves.by_interval.overlap(occ_span.start_b, occ_span.end_b)
        if n.start_b <= occ_span.start_b and n.end_b >= occ_span.end_b
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda n: (n.end_b - n.start_b, KIND_PRIORITY[n.kind], n.preorder_index))
    return candidates[0].node_id
```

### 4.2 Output tables (minimum)

From the CPG spec:

* `SymbolNode(symbol_id, kind, display_name, …)`
* `SymbolEdge(symbol_id → symbol_id, RELATIONSHIP_*)`
* `Occurrence(node_id → symbol_id, role=DEF|REF, occurrence_span=…)`

And once you have that, you unlock cross-file def→ref, import resolution, etc. 

---

## 5) Stage C/D/E — CFG, DDG, CDG on existing syntax nodes (the CPG “meat”)

### 5.1 CFG (statement-level first; blocks optional)

Your CPG doc recommends statement-level CFG as a strong default and explicitly says CFG edges should target existing executable syntax nodes (statements/predicates). 

**Implementation approach**

* Partition syntax nodes into `function bodies` (scope container).
* For each function:

  * Build a CFG over statement nodes (and predicate nodes for branches).
  * Emit `cpg.edges_cfg` with `branch` metadata.

### 5.2 Def-use facts → DDG (reaching definitions)

Follow your CPG guidance:

* Extract defs/uses from AST, bind to symbols via SCIP weld when possible.
* Run reaching defs over CFG.
* Emit `Edge(label="DDG"|"REACHING_DEF", src=def_site_node, dst=use_site_node, symbol_id=...)`. 

Key “best-in-class” posture from your doc: **store uncertainty/candidate sets instead of lying**, and attach `confidence/reason`. 

### 5.3 CDG (control dependence)

Compute postdominators; emit CDG edges from predicate nodes to controlled statements (with branch metadata). (This is standard CDG construction; your CPG overview explicitly introduces CDG at Stage E.)

---

## 6) Interprocedural: call wiring + summaries (so you don’t blow up runtime)

Your CPG notes explicitly define call wiring edges:

* `ARG_TO_PARAM` and `RET_TO_CALL` edges, enabling interprocedural flow. 

Then you scale with **function summaries** rather than inlining everything:

* reads/writes sets
* taint transforms
* side-effect footprints
* thrown/yielded values
  …and compose summaries along call edges. 

This is the correct “best-in-class” move: summaries are how you scale without exploding runtime. 

---

## 7) Tree-sitter: maximal extraction surface (and why it’s worth doing *now*)

You want tree-sitter not just for “some edges”, but as a **declarative extraction engine** where query packs are data, not code. That’s explicitly called out as an “analysis unlock” in your best-in-class doc. 

### 7.1 Locals query: cross-language scope/def/ref scaffold

Tree-sitter locals queries have a **fixed capture vocabulary** (`@local.scope`, `@local.definition`, `@local.reference`) and are explicitly intended to drive consistent scope/def/ref behavior. 

Implementation posture:

* Use locals captures to build a scope stack (interval nesting) and resolve nearest enclosing defs by text (good-enough cross-language baseline). The tree-sitter advanced doc even provides a minimal resolution skeleton. 

This gives you “P0-level” symbol scaffolding for non-Python languages even before SCIP exists.

### 7.2 Injections query: embedded language parsing with correct coordinates

Your best-in-class doc explicitly describes parsing embedded DSLs (SQL/regex/templates) using tree-sitter **included_ranges** + injection query packs so node ranges stay aligned to host bytes—this is *huge* for evidence and joins. 

Tree-sitter advanced further specifies `@injection.content` and `@injection.language` and the pattern properties that control combination/children inclusion, etc. 

This is where you should connect to **SQLGlot** (below): tree-sitter gets you the *span-anchored SQL literal*, SQLGlot gives you *semantic SQL AST and canonicalization*.

### 7.3 Make tree-sitter “best-in-class” operationally: typed wrappers + query linting

Two critical upgrades from your tree-sitter advanced notes:

1. Generate typed wrappers from `node-types.json` so you stop “field hallucination” and make extractor code safer (and LLM-assisted codegen safer). 

2. Add query-pack linting against `node-types.json` to prevent “silent zero matches” from typos and to treat grammar upgrades as breaking-change signals. 

---

## 8) LibCST: maximal useful surface (Python-specific precision engine)

Your LibCST best practices checklist is exactly the right operational discipline:

* analyze `wrapper.module`, never persist CST nodes
* store both `CodeRange` and `CodeSpan` (byte span)
* sort/dedup emitted lists; encode ambiguity as lists 

That maps directly to the “best-in-class” goals for auditability and stitching.

**Net additions to LibCST ingestion (beyond your current basics)**
Emit tables for:

* exact **definitions** and **references** (with qualified name candidates)
* **imports** (imported module, imported name, alias, relative level)
* **call sites** + **argument facts** (positional vs keyword, literal-only evaluation where safe)
* **attribute accesses** (base expr span, attr name span)
* **string literal facts** (raw bytes span, prefix, triple-quoted, f-string parts)
* docstrings, decorators, annotations, type comments

The key “why”: these become high-signal node/edge properties in the CPG (e.g., call node enriched with arg literals; attribute reads/writes become dataflow facts).

---

## 9) Hamilton streamlining: what must change *across the DAG* (not just outputs)

The core insight: **CPG construction is table algebra + per-function graph algorithms**. You will get a far smaller, more robust DAG if you stop modeling every derived column/table as bespoke nodes and instead:

1. Make the *unit of computation* a **LazyFrame** (Polars) or Arrow Table/BatchReader.
2. Use Hamilton decorators to “inline subDAGs inside a dataframe node”.

Your Hamilton notes explicitly describe that `with_columns` runs a subDAG of map ops “inside” the dataframe node and that `columns_to_pass` controls which deps come from the df. 

### 9.1 The “module factoring” you should adopt

Refactor DAG modules by **construction stage**, not by tool:

* `stage0_snapshot` → repo/files/bytes/lineindex
* `stageA_syntax` → syntax nodes/edges + xref
* `stageB_symbols` → scip symbol graph + occurrence welds
* `stageC_cfg` → cfg edges per function
* `stageD_ddg` → def/use facts + ddg edges
* `stageE_cdg` → cdg edges
* `stageF_interproc` → call graph + arg/ret wiring + summaries
* `stageZ_export` → write Arrow datasets + manifest

Each stage exports *tables as first-class Hamilton outputs* (tagged).

### 9.2 “Dataset nodes” are first-class; scalar nodes are internal only

Instead of hundreds of nodes like `foo_count`, `foo_enriched`, `foo_final`, you define:

* one node per dataset (returns LazyFrame)
* optional extracted columns as *views* (only if needed for debugging)

Representative pattern:

```python
import polars as pl
from hamilton.function_modifiers import tag
from hamilton.plugins.h_polars import with_columns  # your hamilton-polars integration

@tag(dataset="cpg.nodes", stage="cpg", kind="table")
@with_columns(columns_to_pass=["file_id", "start_byte", "end_byte", "kind"])
def cpg_nodes(
    syntax_nodes: pl.LazyFrame,            # Stage A output
    symbol_occurrences: pl.LazyFrame,      # Stage B output
) -> pl.LazyFrame:
    # keep as lazy; do not collect in build except at sinks
    return (
        syntax_nodes
        # enrich with “has_symbol”, “symbol_count”, etc.
        .join(
            symbol_occurrences.group_by("node_id").agg(pl.len().alias("occ_n")),
            on="node_id",
            how="left",
        )
        .with_columns(
            pl.col("occ_n").fill_null(0),
            (pl.col("occ_n") > 0).alias("has_symbol"),
        )
    )
```

### 9.3 Use tags + a registry so CLI targets don’t hardcode node names

Hamilton’s `Driver.list_available_variables(tag_filter=...)` supports tag filtering and is intended for building registries from tags. 

So your build CLI should:

* compile driver
* resolve “target = stage=cpg AND kind=table”
* execute those outputs

This eliminates massive “if/elif target list” code.

### 9.4 Build-time vs run-time: keep the control planes clean

Hamilton explicitly distinguishes:

* build-time DAG construction via `driver.Builder()`
* run-time execution via `Driver.execute(final_vars, inputs=..., overrides=...)`
  …and notes config changes require rebuilding the driver. 

That maps to your architecture:

* build profiles select modules/stages at build-time
* repo snapshot path, manifest path, etc. are run-time inputs

### 9.5 Output materialization: prefer Polars sinks for stability + perf

For large datasets: keep everything lazy and materialize at the boundary via sinks.
Polars streaming docs recommend native sinks for throughput and explain `sink_parquet` as out-of-core write (write bigger than RAM). 

Also: for “fanout” (write multiple datasets from one shared upstream plan), use lazy sinks + `collect_all` to get common subplan elimination. 

---

## 10) Polars SQL and SQLGlot: yes, use SQLGlot to generate dynamic SQL for Polars

### 10.1 Polars SQLContext semantics to respect

Polars documents:

* `SQLContext.execute(query, eager=...)` always runs lazily; `eager` just controls whether you get a LazyFrame or DataFrame back. 
* Polars SQL is a subset (aims to follow PostgreSQL where possible). 

So: generate SQL dynamically *only within the supported subset*, or keep your heavy lifting in Polars expressions and use SQL for convenience.

### 10.2 SQLGlot is explicitly built for programmatic query building and AST rewriting

Your SQLGlot doc explicitly shows:

* builder functions: `select().from_().where(condition().and_())`
* parse + modify queries (`parse_one(...).from_(...)`)
* AST transforms via `.transform(...)`
  …and render via `.sql()` 

### 10.3 Representative integration pattern (SQLGlot → Polars SQLContext)

```python
import polars as pl
import sqlglot
from sqlglot import exp
from sqlglot.helper import seq_get

def build_join_query(table_left: str, table_right: str, key: str, cols: list[str]) -> str:
    sel = sqlglot.select(*[exp.column(c) for c in cols]).from_(table_left)
    sel = sel.join(table_right, on=exp.EQ(this=exp.column(key), expression=exp.column(key)))
    return sel.sql(dialect="postgres")  # Polars aims for Postgres-like SQL

ctx = pl.SQLContext()
ctx.register("left", left_lf)    # LazyFrame
ctx.register("right", right_lf)

query = build_join_query("left", "right", "node_id", ["node_id", "kind"])
out_lf = ctx.execute(query)      # LazyFrame (execution is lazy) :contentReference[oaicite:35]{index=35}
```

### 10.4 Why this is *specifically useful in your pipeline*

Use SQLGlot programmatic SQL when:

* you’re building “projection/join/where/groupby” queries based on **config** (dataset keys, selected columns, feature flags)
* you want to **canonicalize** SQL strings extracted from code (embedded SQL literals) before storing them as metadata

That last bullet is a direct synergy with your tree-sitter “embedded language parsing” goal: tree-sitter gives spans and extraction, SQLGlot gives normalized SQL AST and stable rendering. 

---

## 11) Where schema inference *usually* breaks in this architecture (and how to eliminate it)

Even if you prefer inference-driven behavior, CPG work benefits from two tactical moves:

1. **Schema resolution as an explicit, stage-boundary step**, not an emergent property.

   * In Polars, `collect_schema()` exists specifically because schema resolution can be expensive but canonical. 
   * In practice: resolve schema right before writing, attach to Arrow schema metadata, and treat that as the “contract surface”.

2. **Never mix “plan-time unknown” types into primary keys**

   * CPG keys (`node_id`, `symbol_id`, `edge_id`) must be fixed-width or stable binary.
   * If inference sees Python objects, you’ll get unstable dtypes. Fix by using binary(16) for hash128 IDs and explicit int types for offsets.

---

## 12) Summary: what you implement next (minimal ambiguity, maximal leverage)

If you implement only the “highest ROI” pieces first, do them in this order:

1. **Promote SyntaxNode/SyntaxEdge** into canonical `cpg.nodes` / `cpg.edges_ast` tables (deterministic node IDs). 
2. **SCIP weld** into `occurrence(node_id → symbol_id)` and `symbol_nodes` (the semantic glue). 
3. **Per-function CFG edges** referencing existing syntax nodes. 
4. **Def/use + DDG** (store uncertainty, don’t lie). 
5. **CDG** then union into `cpg.edges` (typed edges retained).
6. **Call wiring edges + summaries** (to scale). 
7. **Hamilton refactor**: stage modules + dataset nodes + tags + registry; use `with_columns` to collapse boilerplate. 
8. **Tree-sitter locals + injections** as declarative extraction packs, with typed wrappers + query linting. 

---

If you want, I can also write a **single “CPG stage module template”** (still narrative-first, not a patch) showing exactly how I’d structure `stageC_cfg.py`, `stageD_ddg.py`, etc. as Hamilton modules using:

* tagged dataset outputs,
* Polars LazyFrame pipelines,
* and a consistent “sink boundary” (write Arrow/Parquet + manifest),

…so multiple engineers can implement different stages in parallel without diverging conventions.
Below is the **single “CPG stage module template”** I would standardize on for your repo, **fully aligned with your current native Hamilton patterns** (tagged outputs, ArrowDatasetSaver sink boundary, TargetRunRecord finalizers), but **structured so multiple engineers can build different stages in parallel without inventing new conventions**.

I’m going to show this as a **single-file pattern** that you can copy to `stageC_cfg.py`, `stageD_ddg.py`, etc., with the only differences being: (a) table keys, (b) upstream inputs, and (c) the Polars pipelines / analysis logic.

---

## Core conventions this template enforces

### 1) Every stage file is a *complete Hamilton “target slice”*

Each stage module contains **four layers** in one file, always in this order:

1. **Constants + metadata** (domain, target name, stage version, output table keys)
2. **Inputs (optional scan helpers)** that convert `DatasetRef → pl.LazyFrame` via `load_snapshot_lazyframe`
3. **Compute nodes** (pure; no I/O; return `pl.LazyFrame` or Arrow reader)
4. **Sink boundary nodes**: `@save_dataset` pass-through nodes (write Arrow/Parquet + dataset manifest)
5. **Target finalizer**: `t__<target>` producing `TargetRunRecord`

This makes stage modules independently runnable and reviewable.

### 2) Outputs are *tagged datasets* and **only saved at the sink boundary**

Compute nodes can be as many as you want; they are internal. Only the sink nodes are “contract outputs”:

* `@tag_dataset(domain=..., target=..., table_key=...)`
* `@save_dataset(context=SaverContext(...), spec=DatasetSaveSpec(...))`

This matches how your `DagCatalog` and support-node generation discover tables.

### 3) Polars is the default “table algebra”

* Anything that looks like “join/filter/group/derive columns” is done in **Polars `LazyFrame`**.
* Anything that requires AST walking / graph algorithms can be done in Python, but the *table assembly* should return either:

  * a **LazyFrame** (preferred for join-heavy outputs)
  * or a **RecordBatchReader / iterable of RecordBatch** (preferred for streaming row generation)

### 4) Use a **collect_group** per stage to avoid repeated `.collect()`

Your `ArrowDatasetSaver` already supports `collect_group` to do `pl.collect_all()` across outputs that share the same upstream scans. That is *exactly* what you want for CPG stages (multiple outputs from the same joins).

---

## The template (copy/paste into each stage module)

> Replace `stageC_cfg` with `stageD_ddg`, `stageE_cdg`, `stageF_cpg`, etc.
> Replace the table keys and the compute pipelines.

```python
"""
CPG Stage Template: <stage_name>

This module is a self-contained Hamilton target slice:
- scan upstream datasets as Polars LazyFrames (preferred)
- compute stage outputs as LazyFrame pipelines
- persist outputs via ArrowDatasetSaver (Parquet + dataset manifest)
- emit a TargetRunRecord for orchestration + lineage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from hamilton.function_modifiers import resolve_from_config

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    SaverContext,
    make_table_materializations_collector,
    save_dataset,
)
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_lazyframe, load_snapshot_tabular
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_dataset, tag_helper, tag_loader_query
from codeintel.build.tabular.types import InferableTabularInput, TabularFrame
from codeintel.core.columnar.rows import empty_reader_for_table
from codeintel.core.hamilton import tags as ht
from codeintel.build.hamilton.io.dataset_ref import DatasetRef

# -------------------------
# 0) Stage identity + output contract
# -------------------------

DOMAIN: Final[str] = "graphs"
TARGET: Final[str] = "cpg_stageC_cfg"
STAGE_VERSION: Final[str] = "v1"

# CPG table keys should live under schema="graph" to match your registry conventions.
# (Declare schemas in core.schemas.output_registry.py when you want strict contracts.)
CFG_NODES_TABLE_KEY: Final[str] = "graph.cpg_cfg_nodes"
CFG_EDGES_TABLE_KEY: Final[str] = "graph.cpg_cfg_edges"

TABLE_KEYS: Final[tuple[str, ...]] = (CFG_NODES_TABLE_KEY, CFG_EDGES_TABLE_KEY)

# Tags applied to sinks (useful for inventory queries / UI grouping)
EXTRA_TAGS: Final[dict[str, str]] = {
    ht.TAG_LAYER: "cpg",
    ht.TAG_KIND: "cfg",
    ht.TAG_VERSION: STAGE_VERSION,
}

SAVE_CONTEXT: Final[SaverContext] = SaverContext(domain=DOMAIN, target=TARGET, extra_tags=EXTRA_TAGS)

# One collect group per stage → saver can pl.collect_all() for grouped outputs.
COLLECT_GROUP: Final[str] = f"{TARGET}.{STAGE_VERSION}"


# -------------------------
# 1) Optional: dataset scan helpers (best-in-class for Polars LazyFrame pipelines)
# -------------------------
# Rationale:
# - q__* support nodes return RecordBatchReader; converting that to LazyFrame usually forces read_all().
# - for join-heavy stages, you want pl.scan_pyarrow_dataset via load_snapshot_lazyframe.

def _snapshot_id(env: BuildEnv, ref: DatasetRef) -> str:
    snap = ref.commit or env.commit
    if not snap:
        raise ValueError(f"Missing snapshot_id for {ref.table_key}")
    return snap

@tag_loader_query(domain=DOMAIN, target=TARGET, table_key="core.syntax_spans")
def lf__core__syntax_spans(env: BuildEnv, d__core__syntax_spans: DatasetRef) -> TabularFrame:
    return load_snapshot_lazyframe(
        env=env,
        table_key=d__core__syntax_spans.table_key,
        snapshot_id=_snapshot_id(env, d__core__syntax_spans),
    )

@tag_loader_query(domain=DOMAIN, target=TARGET, table_key="core.syntax_scopes")
def lf__core__syntax_scopes(env: BuildEnv, d__core__syntax_scopes: DatasetRef) -> TabularFrame:
    return load_snapshot_lazyframe(
        env=env,
        table_key=d__core__syntax_scopes.table_key,
        snapshot_id=_snapshot_id(env, d__core__syntax_scopes),
    )

@tag_loader_query(domain=DOMAIN, target=TARGET, table_key="core.syntax_calls")
def lf__core__syntax_calls(env: BuildEnv, d__core__syntax_calls: DatasetRef) -> TabularFrame:
    return load_snapshot_lazyframe(
        env=env,
        table_key=d__core__syntax_calls.table_key,
        snapshot_id=_snapshot_id(env, d__core__syntax_calls),
    )

# Add more lf__* inputs per-stage as needed (syntax_defs, syntax_refs, scip_occurrences, etc.)


# -------------------------
# 2) Variants (compute vs existing vs empty) — optional but strongly recommended
# -------------------------

def _pick_backend(
    *,
    graph_backend: str | None,
    empty_node: str,
    existing_node: str,
    compute_node: str,
    param_name: str,
):
    """
    Mirrors your existing graphs/variants.py pattern:
      graph_backend == "existing" → use *_existing
      graph_backend == "compute"  → use *_compute
      else                        → *_empty
    """
    from hamilton.function_modifiers.base import NodeTransformLifecycle
    from codeintel.build.hamilton.transforms.registry_inject import inject_from_registry

    if graph_backend == "existing":
        return inject_from_registry(param_name=param_name, node_name=existing_node)
    if graph_backend == "compute":
        return inject_from_registry(param_name=param_name, node_name=compute_node)
    return inject_from_registry(param_name=param_name, node_name=empty_node)


def _pick_cfg_nodes(graph_backend: str | None = None):
    return _pick_backend(
        graph_backend=graph_backend,
        empty_node="cpg_cfg_nodes_empty",
        existing_node="cpg_cfg_nodes_existing",
        compute_node="cpg_cfg_nodes_compute",
        param_name="nodes",
    )

def _pick_cfg_edges(graph_backend: str | None = None):
    return _pick_backend(
        graph_backend=graph_backend,
        empty_node="cpg_cfg_edges_empty",
        existing_node="cpg_cfg_edges_existing",
        compute_node="cpg_cfg_edges_compute",
        param_name="edges",
    )

@resolve_from_config(decorate_with=_pick_cfg_nodes)
def cpg_cfg_nodes(nodes: InferableTabularInput) -> InferableTabularInput:
    return nodes

@resolve_from_config(decorate_with=_pick_cfg_edges)
def cpg_cfg_edges(edges: InferableTabularInput) -> InferableTabularInput:
    return edges


# -------------------------
# 3) Compute nodes (pure; return LazyFrame if join-heavy; return Arrow stream if row-gen-heavy)
# -------------------------

@tag_helper(domain=DOMAIN, target=TARGET, extra_tags={ht.TAG_MCP_VISIBLE: "0"})
def _cfg_stmt_inventory(
    lf__core__syntax_spans: TabularFrame,
    lf__core__syntax_scopes: TabularFrame,
) -> TabularFrame:
    """
    Example: produce the *executable statement node inventory* that CFG will connect.

    Best practice for CPG:
    - CFG edges should connect *existing syntax nodes* (statement/predicate nodes)
    - so this step creates a deterministic 'cfg_node_id' anchored to span_id/scope_id
    """
    spans = lf__core__syntax_spans.select([
        "repo", "commit", "rel_path", "producer",
        "span_id", "span_kind",
        "start_line", "start_col", "end_line", "end_col",
        "start_byte", "end_byte",
    ])

    scopes = lf__core__syntax_scopes.select([
        "repo", "commit", "rel_path", "producer",
        "scope_id", "scope_kind",
        "start_line", "start_col", "end_line", "end_col",
        "parent_scope_id",
    ])

    # Minimal example: "cfg node" = subset of spans considered executable
    # Replace with your real executable kinds.
    executable = spans.filter(pl.col("span_kind").is_in([
        "stmt", "expr_stmt", "if", "for", "while", "try", "with", "return", "raise",
        "call", "assign",
    ]))

    # Join to find containing scope (simplified; real version may be span containment join or precomputed xref)
    # In best-in-class implementations you usually have a span→scope xref to avoid O(N^2) containment logic.
    stmt_nodes = executable.join(
        scopes,
        on=["repo", "commit", "rel_path", "producer"],
        how="left",
    ).with_columns([
        # Deterministic ID; prefer hash of (rel_path, span_id, scope_id) or use a stable join key.
        (pl.col("rel_path") + ":" + pl.col("span_id")).alias("cfg_node_id"),
    ])

    return stmt_nodes


def cpg_cfg_nodes_compute(_cfg_stmt_inventory: TabularFrame) -> TabularFrame:
    """
    Output table: graph.cpg_cfg_nodes
    """
    return _cfg_stmt_inventory.select([
        "repo", "commit", "rel_path", "producer",
        "cfg_node_id",
        "span_id", "scope_id",
        "span_kind",
        "start_line", "start_col", "end_line", "end_col",
        "start_byte", "end_byte",
    ])


def cpg_cfg_edges_compute(
    env: BuildEnv,
    _cfg_stmt_inventory: TabularFrame,
    lf__core__syntax_calls: TabularFrame,
) -> TabularFrame:
    """
    Output table: graph.cpg_cfg_edges

    This is intentionally a placeholder:
    - for Python you’ll often compute per-function CFG using AST (fast in Python),
      then weld block/stmt nodes back to cfg_node_id via spans.
    - but the *assembly* of edges should still be a LazyFrame pipeline:
      normalize rows → join to cfg_node_id → emit edges
    """

    # Example of producing *candidate* edges from simple ordering heuristics.
    # Replace with your real CFG builder output welded by span_id/cfg_node_id.
    inv = _cfg_stmt_inventory.select([
        "repo", "commit", "rel_path", "producer",
        "cfg_node_id", "start_byte", "end_byte",
        "start_line", "start_col",
    ])

    ordered = inv.sort(["rel_path", "start_byte"]).with_columns([
        pl.col("cfg_node_id").shift(-1).over(["repo", "commit", "rel_path", "producer"]).alias("dst_cfg_node_id"),
    ]).filter(pl.col("dst_cfg_node_id").is_not_null())

    edges = ordered.select([
        "repo", "commit", "rel_path", "producer",
        pl.col("cfg_node_id").alias("src_cfg_node_id"),
        pl.col("dst_cfg_node_id"),
        pl.lit("NEXT").alias("edge_kind"),
        pl.lit(None).cast(pl.Utf8).alias("branch"),
    ])

    return edges


# Existing + Empty sources (used by variants)

def cpg_cfg_nodes_existing(env: BuildEnv) -> InferableTabularInput:
    return load_snapshot_tabular(env=env, table_key=CFG_NODES_TABLE_KEY, snapshot_id=env.commit)

def cpg_cfg_edges_existing(env: BuildEnv) -> InferableTabularInput:
    return load_snapshot_tabular(env=env, table_key=CFG_EDGES_TABLE_KEY, snapshot_id=env.commit)

def cpg_cfg_nodes_empty(env: BuildEnv) -> InferableTabularInput:
    _ = env
    return empty_reader_for_table(CFG_NODES_TABLE_KEY)

def cpg_cfg_edges_empty(env: BuildEnv) -> InferableTabularInput:
    _ = env
    return empty_reader_for_table(CFG_EDGES_TABLE_KEY)


# -------------------------
# 4) Sink boundary (Arrow/Parquet + manifest) — ALWAYS pass-through
# -------------------------

@save_dataset(
    context=SAVE_CONTEXT,
    spec=DatasetSaveSpec(
        table_key=CFG_NODES_TABLE_KEY,
        partition_columns=("repo", "commit"),
        collect_group=COLLECT_GROUP,
        validation_profile="lenient",
    ),
)
@tag_dataset(domain=DOMAIN, target=TARGET, table_key=CFG_NODES_TABLE_KEY)
def cpg_stageC_cfg__nodes_table(cpg_cfg_nodes: InferableTabularInput) -> InferableTabularInput:
    return cpg_cfg_nodes


@save_dataset(
    context=SAVE_CONTEXT,
    spec=DatasetSaveSpec(
        table_key=CFG_EDGES_TABLE_KEY,
        partition_columns=("repo", "commit"),
        collect_group=COLLECT_GROUP,
        validation_profile="lenient",
    ),
)
@tag_dataset(domain=DOMAIN, target=TARGET, table_key=CFG_EDGES_TABLE_KEY)
def cpg_stageC_cfg__edges_table(cpg_cfg_edges: InferableTabularInput) -> InferableTabularInput:
    return cpg_cfg_edges


# Materialization collector + target finalizer

cpg_stageC_cfg__table_materializations = make_table_materializations_collector(
    domain=DOMAIN,
    target=TARGET,
    table_keys=TABLE_KEYS,
    node_name="cpg_stageC_cfg__table_materializations",
)

@codeintel_target(domain=DOMAIN, target=TARGET)
def t__cpg_stageC_cfg(
    env: BuildEnv,
    catalog: DagCatalog,
    cpg_stageC_cfg__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    context = MaterializationRecordContext(env=env, catalog=catalog, target_name=TARGET)
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=cpg_stageC_cfg__table_materializations,
    )


__all__ = [
    "TARGET",
    "CFG_NODES_TABLE_KEY",
    "CFG_EDGES_TABLE_KEY",
    "t__cpg_stageC_cfg",
]
```

---

## How you specialize this template for each CPG stage

### Stage C: `stageC_cfg.py`

* **Inputs**: syntax spans + scopes + (optionally) AST/CST crosswalk + function inventory
* **Compute**:

  * build statement/predicate node inventory (or reuse your syntax-span “executable unit” table)
  * compute CFG edges (per-function analysis) and weld to `cfg_node_id`
* **Outputs**:

  * `graph.cpg_cfg_nodes`
  * `graph.cpg_cfg_edges`

### Stage D: `stageD_ddg.py`

Keep the same skeleton; swap compute nodes.

* **Inputs**:

  * `core.syntax_defs`, `core.syntax_refs`, `core.syntax_scopes`, `graph.cpg_cfg_edges`
  * SCIP weld tables (prefer `core.scip_occurrence_span_xref` + `core.scip_symbols` / relationships)
* **Compute** (Polars-heavy):

  * normalize `defs` and `uses` keyed by `(rel_path, span_id)` (or `(cfg_node_id, symbol)`)
  * compute reaching-def / def-use edges (CFG traversal may be Python, but edge assembly stays tabular)
* **Outputs**:

  * `graph.cpg_ddg_edges` (or `graph.cpg_reaching_defs` if you want intermediate)
* **Key pattern**:

  * compute intermediate “def sites” and “use sites” tables as LazyFrames
  * join on `symbol_id` and scope visibility rules
  * weld edge endpoints to `cfg_node_id` (or your canonical syntax node id)

### Stage E: `stageE_cdg.py`

* **Inputs**: `graph.cpg_cfg_edges` + `graph.cpg_cfg_nodes`
* **Compute**:

  * compute postdominators / control dependence (algorithmic core may be Python)
  * emit `cdg_edges(src=predicate_cfg_node_id, dst=controlled_cfg_node_id, branch=...)`
* **Outputs**:

  * `graph.cpg_cdg_edges`
  * optionally `graph.cpg_pdg_edges` as `concat([ddg_edges, cdg_edges])`

### Stage F: `stageF_cpg.py`

* **Inputs**: all layers (`syntax`, `scip`, `cfg`, `ddg`, `cdg`, `call_graph`, `import_graph`)
* **Compute**:

  * normalize to a single **property graph edge table**:

    * `src_node_id`, `dst_node_id`, `edge_label`, `edge_props_json`, plus repo/commit
  * and a single **node table**:

    * `node_id`, `kind`, `span`, `symbol_id?`, `properties_json`
* **Outputs**:

  * `graph.cpg_nodes`
  * `graph.cpg_edges`

---

## Two “best-in-class” implementation details to keep everyone consistent

### A) Prefer *dataset-ref-based scanning* for join-heavy stages

The `lf__*` pattern in the template is deliberate: it avoids forcing Arrow readers into memory just to “get a LazyFrame”.

If a stage is truly row-generator heavy (e.g., AST walker producing CFG edges), then return an Arrow stream directly and skip LazyFrame. But for the “welding” joins (span ↔ scope ↔ symbol ↔ node-id), scanning Parquet datasets lazily is what keeps the build stable at scale.

### B) Put stage outputs in a single `collect_group`

If stage outputs are derived from the same upstream scan graph, set:

```python
collect_group = f"{TARGET}.{STAGE_VERSION}"
```

in every `DatasetSaveSpec` for that stage. Your saver will opportunistically `pl.collect_all()` under the hood, which prevents repeated scans and makes runs much less fragile.

---

## Quick note on SQLGlot + Polars (since it’s highly relevant for these stages)

Yes—**SQLGlot can absolutely be used to generate SQL programmatically** and then execute it via Polars’ SQL engine, *as long as you keep the dialect constraints in mind*.

The practical pattern is:

1. build an expression/query with SQLGlot,
2. compile to a SQL string in the dialect you want (often `"duckdb"` or `"sqlite"`-ish),
3. run it through `polars.SQLContext` after registering your LazyFrames.

Representative snippet:

```python
import polars as pl
import sqlglot
from sqlglot import exp

ctx = pl.SQLContext()

ctx.register("defs", lf_defs)      # LazyFrame
ctx.register("refs", lf_refs)      # LazyFrame

query = (
    exp.select(
        exp.column("repo"),
        exp.column("commit"),
        exp.column("rel_path"),
        exp.column("symbol_id"),
        exp.column("def_span_id"),
        exp.column("ref_span_id"),
    )
    .from_("defs")
    .join("refs", on=exp.and_(
        exp.column("repo", table="defs").eq(exp.column("repo", table="refs")),
        exp.column("commit", table="defs").eq(exp.column("commit", table="refs")),
        exp.column("symbol_id", table="defs").eq(exp.column("symbol_id", table="refs")),
    ))
)

sql = query.sql(dialect="duckdb")  # pick a stable dialect you standardize on

lf = ctx.execute(sql)             # returns LazyFrame
```

This gives you:

* programmatic query construction (no string concatenation),
* a single place to enforce join key conventions,
* easy reuse across stages.

(Just be disciplined about which SQL features you rely on, because Polars SQL support is not identical to DuckDB/Postgres.)

---

If you want, I can also produce the **exact specializations** of the compute section for:

* `stageD_ddg.py` (def-use edges welded to symbols + cfg nodes), and
* `stageE_cdg.py` (postdom + control dependence),
  in the same dense “module template” style (i.e., *the same skeleton, but with the real core compute pipelines filled in using your existing `core.syntax_*` + `core.scip_*` tables*).


Below are the **compute specializations** for:

* **`stageD_ddg.py`** — *intraprocedural* **Data Dependence Graph** via **reaching-definitions** (def→use edges), “welded” to **SCIP symbol identity** + **CFG node identity**, and
* **`stageE_cdg.py`** — **Control Dependence Graph** via **postdominators** (CDG edges)

…written in the same “stage module template” style you adopted, but with the real core compute filled in.

I’m grounding a couple of key assumptions in the design notes you attached:

* SCIP occurrences are meant to be welded to syntax/leaf nodes with deterministic span mapping, and then used as the “semantic glue” (stable symbol identity) for downstream graph edges. 
* Your pipeline stages explicitly include DDG and postdom→CDG as steps after CFG. 
* SCIP occurrence `symbol_roles` is a bitmask that distinguishes Definition vs Read/Write/etc, and local symbols may take the form `"local <local-id>"` (important for def/use partitioning). 

I’m **not** emitting a patch; this is intended to be dense, unambiguous “what to implement”.

---

## Stage D — `stageD_ddg.py` (def-use edges welded to symbols + CFG nodes)

### Design intent

**Goal:** produce a **Data Dependence Graph** (DDG) at *statement-level CFG* granularity, where each edge represents:

> “definition of symbol **S** at CFG node **A** reaches a **use** of **S** at CFG node **B** along the CFG”

This is the standard “reaching definitions” / def-use chain. It is also the best foundation for later PDG wiring (e.g., data dependencies + control dependencies) and interprocedural edges (call/return) if you choose to add them later.

**Why this approach fits your architecture:**

* It respects the “semantic weld” principle: use **SCIP symbols** as the identity surface, not inferred names. 
* It’s *intraprocedural* and deterministic, and can be computed per-procedure (parallelizable, cacheable, bounded memory).
* It outputs a **tabular edge relation** (Arrow/Parquet), keeping storage decoupled.

---

## Stage D inputs (assumed tables; adapt names to your actual keys)

You already have equivalents, but the DDG compute needs these *facts*:

### Required upstream datasets

1. **CFG (Stage C outputs)**

* `stageC.cfg_nodes`:

  * `proc_id` (procedure/function identity; can be a symbol or syntax node id)
  * `cfg_node_id` (statement/predicate identity; ideally a syntax node id per your design)
* `stageC.cfg_edges`:

  * `proc_id`
  * `src_cfg_node_id`, `dst_cfg_node_id`
  * (optional) `edge_kind` (T/F/exception/etc)

(Your CPG notes explicitly recommend connecting CFG edges on executable syntax nodes rather than inventing separate CFG-only nodes; DDG becomes dramatically simpler if `cfg_node_id` is already a syntax node id.) 

2. **SCIP occurrences welded to syntax nodes (Stage B outputs)**

* `core.scip_occurrences` (or equivalent):

  * `occurrence_id` (stable row id)
  * `file_id` / `doc_path`
  * `syntax_node_id` (leaf node the occurrence welded to)
  * `symbol` (SCIP symbol string)
  * `symbol_roles` (bitmask)
  * (optional) `enclosing_range` or `enclosing_syntax_node_id`

SCIP roles are bitmasks: Definition bit=1; Import=2; Write=4; Read=8 (as described in your SCIP doc). 

3. **Leaf→CFG node mapping** (one of the below)

* Prefer: `core.xref_occurrence_to_cfg_node`

  * `occurrence_id` → `cfg_node_id` (+ `proc_id`), **precomputed** by span/ancestor rules.
* Or: `core.syntax_parent_edges` + `core.syntax_nodes` with spans so we can climb ancestors / do containment mapping.

**Best-in-class recommendation:** treat this mapping as a first-class dataset; DDG should *not* re-infer containment every time. (Your own notes emphasize preserving explicit crosswalks and deterministic welds.) 

---

## Stage D outputs

I recommend emitting **three datasets** (even if you only “serve” ddg_edges), because it makes debugging and downstream composition vastly easier:

1. `stageD.ddg_defs`
   **PK:** `(proc_id, def_site_id)`
   Columns:

* `proc_id`
* `def_site_id` (stable id for this write/definition occurrence)
* `cfg_node_id` (where write occurs)
* `symbol` (SCIP symbol)
* `occurrence_id` (source occurrence)
* (optional) `is_definition` (roles&DEF), `is_write` (roles&WRITE)

2. `stageD.ddg_uses`
   **PK:** `(proc_id, use_site_id)`
   Columns:

* `proc_id`
* `use_site_id`
* `cfg_node_id`
* `symbol`
* `occurrence_id`
* `is_read`

3. `stageD.ddg_edges`
   **PK:** `(proc_id, def_site_id, use_site_id)` (or include dst cfg node too)
   Columns:

* `proc_id`
* `src_cfg_node_id` (def site node)
* `dst_cfg_node_id` (use site node)
* `symbol`
* `def_site_id`, `use_site_id`
* (optional) `edge_kind="DDG_REACHES_DEF"`

---

## Core compute: reaching-definitions in a relational style

### The key trick (to avoid enormous KILL sets)

Don’t model KILL as “all defs of the symbol except these”; that explodes.
Instead:

* Maintain a membership table of “reaching definitions” keyed by `(cfg_node_id, symbol, def_site_id)`
* When propagating IN→OUT, **filter incoming defs by symbol** if a node defines that symbol.

This turns KILL into a cheap anti-join against `(cfg_node_id, symbol)`.

---

## Representative `stageD_ddg.py` compute skeleton (dense, minimal, implementable)

```python
# stageD_ddg.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import polars as pl

# Hamilton
from hamilton.function_modifiers import tag

# --- SCIP symbol role bits (align to your constants module)
SCIP_ROLE_DEFINITION = 1
SCIP_ROLE_IMPORT     = 2
SCIP_ROLE_WRITE      = 4
SCIP_ROLE_READ       = 8


def _stable_u64(*parts: str) -> int:
    """
    Deterministic 64-bit id from strings; keep identical across runs.
    Replace with your canonical hashing util (xxhash/blake3/etc).
    """
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "little", signed=False)


# ---------------------------
# 1) Build def/use fact tables
# ---------------------------

@tag(dataset="stageD.ddg_defs")
def ddg_defs_df(
    core_scip_occurrences_df: pl.DataFrame,
    core_xref_occurrence_to_cfg_node_df: pl.DataFrame,  # occurrence_id -> (proc_id, cfg_node_id)
) -> pl.DataFrame:
    """
    Extract write/definition occurrences as 'def sites'.
    Statement-level: treat any write as a def site for reaching-defs.
    """
    occ = core_scip_occurrences_df.select([
        "occurrence_id", "symbol", "symbol_roles"
    ])

    xref = core_xref_occurrence_to_cfg_node_df.select([
        "occurrence_id", "proc_id", "cfg_node_id"
    ])

    df = (
        occ.join(xref, on="occurrence_id", how="inner")
           .with_columns([
               ((pl.col("symbol_roles") & SCIP_ROLE_DEFINITION) != 0).alias("is_definition"),
               ((pl.col("symbol_roles") & SCIP_ROLE_WRITE) != 0).alias("is_write"),
           ])
           .filter(pl.col("is_definition") | pl.col("is_write"))
           .with_columns([
               pl.struct(["proc_id", "occurrence_id", "symbol"]).map_elements(
                   lambda s: _stable_u64(str(s["proc_id"]), str(s["occurrence_id"]), str(s["symbol"])),
                   return_dtype=pl.UInt64,
               ).alias("def_site_id")
           ])
           .select(["proc_id", "def_site_id", "cfg_node_id", "symbol", "occurrence_id", "is_definition", "is_write"])
           .unique(subset=["proc_id", "def_site_id"])
    )
    return df


@tag(dataset="stageD.ddg_uses")
def ddg_uses_df(
    core_scip_occurrences_df: pl.DataFrame,
    core_xref_occurrence_to_cfg_node_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Extract read occurrences as 'use sites'.
    """
    occ = core_scip_occurrences_df.select([
        "occurrence_id", "symbol", "symbol_roles"
    ])
    xref = core_xref_occurrence_to_cfg_node_df.select([
        "occurrence_id", "proc_id", "cfg_node_id"
    ])

    df = (
        occ.join(xref, on="occurrence_id", how="inner")
           .with_columns([
               ((pl.col("symbol_roles") & SCIP_ROLE_READ) != 0).alias("is_read"),
           ])
           .filter(pl.col("is_read"))
           .with_columns([
               pl.struct(["proc_id", "occurrence_id", "symbol"]).map_elements(
                   lambda s: _stable_u64(str(s["proc_id"]), str(s["occurrence_id"]), str(s["symbol"])),
                   return_dtype=pl.UInt64,
               ).alias("use_site_id")
           ])
           .select(["proc_id", "use_site_id", "cfg_node_id", "symbol", "occurrence_id", "is_read"])
           .unique(subset=["proc_id", "use_site_id"])
    )
    return df


# -----------------------------------------
# 2) Reaching-definitions fixpoint per proc
# -----------------------------------------

def _reaching_defs_for_proc(
    cfg_edges_proc: pl.DataFrame,     # src_cfg_node_id, dst_cfg_node_id
    defs_proc: pl.DataFrame,          # cfg_node_id, symbol, def_site_id
) -> pl.DataFrame:
    """
    Returns IN sets as rows: (cfg_node_id, symbol, def_site_id)
    where membership means def_site_id reaches *before* executing cfg_node_id.

    Implementation: iterate OUT until stable; derive IN from OUT at convergence.
    KILL is implemented as "drop incoming defs whose symbol is defined in this node".
    """
    edges = cfg_edges_proc.select([
        pl.col("src_cfg_node_id").alias("src"),
        pl.col("dst_cfg_node_id").alias("dst"),
    ])

    gen = defs_proc.select([
        pl.col("cfg_node_id"),
        pl.col("symbol"),
        pl.col("def_site_id"),
    ])

    kill_symbols = gen.select(["cfg_node_id", "symbol"]).unique()

    # OUT membership rows: definitions reaching AFTER the node executes
    out = gen.clone()

    # Iterative fixpoint
    # NOTE: Use deterministic termination: compare row counts + hash of sorted rows.
    def _fingerprint(df: pl.DataFrame) -> int:
        # stable-ish fingerprint: sort + hash a string representation of key cols
        key = df.sort(["cfg_node_id", "symbol", "def_site_id"])
        # avoid to_string() on huge frames; you can use pyarrow.compute.hash if preferred
        return hash(tuple(key.select(["cfg_node_id", "symbol", "def_site_id"]).iter_rows()))

    prev_fp = None
    for _iter in range(1000):  # hard cap to avoid infinite loops on unexpected graphs
        # IN[n] = union OUT[p] for p -> n
        in_rows = (
            out.join(edges, left_on="cfg_node_id", right_on="src", how="inner")
               .select([
                   pl.col("dst").alias("cfg_node_id"),
                   "symbol",
                   "def_site_id",
               ])
               .unique(subset=["cfg_node_id", "symbol", "def_site_id"])
        )

        # Remove incoming defs for symbols that are defined at the node
        survivors = (
            in_rows.join(kill_symbols, on=["cfg_node_id", "symbol"], how="anti")
        )

        new_out = (
            pl.concat([survivors, gen], how="vertical")
              .unique(subset=["cfg_node_id", "symbol", "def_site_id"])
        )

        fp = _fingerprint(new_out)
        if fp == prev_fp:
            out = new_out
            break
        out = new_out
        prev_fp = fp
    else:
        # If you hit this, CFG is pathological or iteration cap too low.
        raise RuntimeError("reaching-definitions did not converge")

    # Final IN derived from converged OUT
    in_final = (
        out.join(edges, left_on="cfg_node_id", right_on="src", how="inner")
           .select([
               pl.col("dst").alias("cfg_node_id"),
               "symbol",
               "def_site_id",
           ])
           .unique(subset=["cfg_node_id", "symbol", "def_site_id"])
    )
    return in_final


# ----------------------------------------
# 3) Join IN sets to uses => def-use edges
# ----------------------------------------

@tag(dataset="stageD.ddg_edges")
def ddg_edges_df(
    stageC_cfg_edges_df: pl.DataFrame,  # proc_id, src_cfg_node_id, dst_cfg_node_id
    ddg_defs_df: pl.DataFrame,
    ddg_uses_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Produces def-use edges (DDG) per procedure.
    """
    # Pre-index defs by id so we can recover src cfg node from def_site_id
    defs_by_id = ddg_defs_df.select([
        "proc_id", "def_site_id", pl.col("cfg_node_id").alias("def_cfg_node_id"), "symbol"
    ])

    proc_ids = (
        stageC_cfg_edges_df.select("proc_id").unique().to_series().to_list()
    )

    out_edges: list[pl.DataFrame] = []

    for proc_id in proc_ids:
        edges_proc = stageC_cfg_edges_df.filter(pl.col("proc_id") == proc_id).select([
            "src_cfg_node_id", "dst_cfg_node_id"
        ])
        defs_proc = ddg_defs_df.filter(pl.col("proc_id") == proc_id).select([
            "cfg_node_id", "symbol", "def_site_id"
        ])
        uses_proc = ddg_uses_df.filter(pl.col("proc_id") == proc_id).select([
            "use_site_id", "cfg_node_id", "symbol"
        ])

        if defs_proc.height == 0 or uses_proc.height == 0 or edges_proc.height == 0:
            continue

        in_final = _reaching_defs_for_proc(edges_proc, defs_proc)

        # For each use at node N of symbol S, attach all reaching defs (N,S,def_site_id)
        du = (
            uses_proc.join(in_final, on=["cfg_node_id", "symbol"], how="inner")
                     .join(
                         defs_by_id.filter(pl.col("proc_id") == proc_id),
                         on=["def_site_id", "symbol"], how="left"
                     )
                     .with_columns([
                         pl.lit(proc_id).alias("proc_id"),
                         pl.col("cfg_node_id").alias("dst_cfg_node_id"),
                         pl.col("def_cfg_node_id").alias("src_cfg_node_id"),
                         pl.lit("DDG_REACHES_DEF").alias("edge_kind"),
                     ])
                     .select([
                         "proc_id",
                         "src_cfg_node_id",
                         "dst_cfg_node_id",
                         "symbol",
                         "def_site_id",
                         "use_site_id",
                         "edge_kind",
                     ])
                     .unique(subset=["proc_id", "def_site_id", "use_site_id"])
        )
        out_edges.append(du)

    if not out_edges:
        return pl.DataFrame(schema={
            "proc_id": pl.Utf8,
            "src_cfg_node_id": pl.Int64,
            "dst_cfg_node_id": pl.Int64,
            "symbol": pl.Utf8,
            "def_site_id": pl.UInt64,
            "use_site_id": pl.UInt64,
            "edge_kind": pl.Utf8,
        })

    return pl.concat(out_edges, how="vertical")
```

### Why this is “best-in-class” for your current system

* It is **maximally inference-driven** *within the correct boundaries*:
  you infer def/use classification from SCIP’s role bitmask (read/write/definition) rather than re-parsing assignment semantics. 
* It stays faithful to your design: SCIP is semantic glue; CFG is the control substrate; DDG is derived as a pure function. 
* It makes KILL cheap and deterministic.

### Important refinement notes (for correctness)

1. **Statement-level semantics caveat**
   This computes def-use at “statement CFG node” granularity, and assumes “all uses happen before defs” *within the statement*. That is a safe over-approx for many languages and matches your statement-level CFG choice, but it will not model Python’s intra-expression evaluation order perfectly (e.g., walrus). If you want that, you’ll eventually need **expression-level micro CFG** or SSA conversion.

2. **Local vs global symbols**
   You likely want a config gate to focus on intraprocedural/local defs first (locals look like `"local <local-id>"` per your parser). 
   Add:

* `symbol.startswith("local ")` filter for P0 DDG
* then extend later to fields/attributes with aliasing rules

---

## Stage E — `stageE_cdg.py` (postdom + control dependence)

### Design intent

**Goal:** build **Control Dependence Graph edges** from the CFG, per procedure.

CDG is used to complete PDG/CPG: if statement B is control-dependent on branch statement A, you want a CDG edge A→B.

Your design doc explicitly calls out postdominators as the step for CDG. 

### Best-in-class compute strategy for Python + Polars architecture

* Represent each procedure CFG as a compact adjacency list.
* Compute postdominators with a **bitset fixpoint** (Python `int` bitmasks): extremely fast, deterministic, and doesn’t require heavy graph libs.
* Produce CDG edges using the classic set-difference rule:

  * for each CFG edge (A→B), for each node X in `postdom(B) - postdom(A)`, emit a control dependence edge **A→X** (optionally annotated with the successor B that witnesses the dependence).

This is (a) standard, (b) stable, (c) easy to reason about for engineers.

---

## Stage E inputs (assumed)

* `stageC.cfg_nodes`: `proc_id`, `cfg_node_id`
* `stageC.cfg_edges`: `proc_id`, `src_cfg_node_id`, `dst_cfg_node_id`, (optional `edge_kind`)

---

## Representative `stageE_cdg.py` compute specialization

```python
# stageE_cdg.py
from __future__ import annotations

from typing import Dict, List, Tuple

import polars as pl
from hamilton.function_modifiers import tag


def _bit_iter(mask: int):
    """Yield set bit positions from a Python int bitmask."""
    while mask:
        lsb = mask & -mask
        i = (lsb.bit_length() - 1)
        yield i
        mask ^= lsb


def _compute_postdom_bitsets(
    node_ids: List[int],
    succ: Dict[int, List[int]],
    exit_id: int,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Returns:
      postdom[node_id] -> bitmask of postdominators (including node)
      idx[node_id] -> dense index used for bit positions
    """
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    ALL = (1 << n) - 1

    exit_bit = 1 << idx[exit_id]

    post = {nid: ALL for nid in node_ids}
    post[exit_id] = exit_bit

    # Fixpoint: postdom(n) = {n} U intersection(postdom(s) for s in succ(n))
    for _ in range(1000):
        changed = False
        for nid in node_ids:
            if nid == exit_id:
                continue
            succs = succ.get(nid, [])
            if not succs:
                # If CFG has a dead-end node, treat it as flowing to exit (synthetic edge).
                succs = [exit_id]

            inter = ALL
            for s in succs:
                inter &= post[s]
            new_mask = (1 << idx[nid]) | inter

            if new_mask != post[nid]:
                post[nid] = new_mask
                changed = True
        if not changed:
            break
    else:
        raise RuntimeError("postdom did not converge")

    return post, idx


def _compute_cdg_edges_for_proc(
    cfg_nodes_proc: pl.DataFrame,   # cfg_node_id
    cfg_edges_proc: pl.DataFrame,   # src_cfg_node_id, dst_cfg_node_id, (optional edge_kind)
) -> pl.DataFrame:
    """
    Implements:
      For each CFG edge (A->B), for each X in postdom(B) - postdom(A): emit CDG edge A->X.
    """
    node_ids = cfg_nodes_proc.get_column("cfg_node_id").to_list()
    if not node_ids:
        return pl.DataFrame()

    # Choose a single exit:
    # - Prefer explicit 'exit' node if you have one in cfg_nodes
    # - Else synthesize: pick a node with out_degree=0; if multiple, connect them to a synthetic exit
    edges = cfg_edges_proc.select(["src_cfg_node_id", "dst_cfg_node_id"])

    outdeg = (
        edges.group_by("src_cfg_node_id").len().rename({"len": "outdeg"})
    )
    nodes_df = cfg_nodes_proc.join(outdeg, left_on="cfg_node_id", right_on="src_cfg_node_id", how="left").with_columns([
        pl.col("outdeg").fill_null(0)
    ])

    exits = nodes_df.filter(pl.col("outdeg") == 0).get_column("cfg_node_id").to_list()

    if len(exits) == 1:
        exit_id = exits[0]
        edges_aug = edges
        node_ids_aug = node_ids
    else:
        # Synthetic exit node id: pick a reserved negative id or a high sentinel.
        # Use something deterministic per proc; here we just use -(max+1).
        synthetic_exit = -(max(node_ids) + 1)
        exit_id = synthetic_exit
        node_ids_aug = node_ids + [synthetic_exit]
        extra_edges = pl.DataFrame({
            "src_cfg_node_id": exits,
            "dst_cfg_node_id": [synthetic_exit] * len(exits),
        })
        edges_aug = pl.concat([edges, extra_edges], how="vertical")

    # Build succ adjacency for bitset algorithm
    succ: Dict[int, List[int]] = {}
    for src, dst in edges_aug.iter_rows():
        succ.setdefault(src, []).append(dst)

    post, idx = _compute_postdom_bitsets(node_ids_aug, succ, exit_id)

    # Compute CDG edges
    rows = []
    for src, dst in edges_aug.iter_rows():
        pd_dst = post[dst]
        pd_src = post[src]
        diff = pd_dst & (~pd_src)
        if diff == 0:
            continue
        for x_i in _bit_iter(diff):
            # map bit index back to cfg_node_id
            # build inverse idx once (tiny per proc)
            pass

    inv_idx = {v: k for k, v in idx.items()}

    for src, dst in edges_aug.iter_rows():
        pd_dst = post[dst]
        pd_src = post[src]
        diff = pd_dst & (~pd_src)
        if diff == 0:
            continue
        for x_i in _bit_iter(diff):
            x = inv_idx[x_i]
            # skip synthetic exit in outputs
            if x == exit_id:
                continue
            rows.append((src, x, dst))

    if not rows:
        return pl.DataFrame(schema={
            "src_cfg_node_id": pl.Int64,
            "dst_cfg_node_id": pl.Int64,
            "via_succ_cfg_node_id": pl.Int64,
        })

    return pl.DataFrame(rows, schema=["src_cfg_node_id", "dst_cfg_node_id", "via_succ_cfg_node_id"])


@tag(dataset="stageE.cdg_edges")
def cdg_edges_df(
    stageC_cfg_nodes_df: pl.DataFrame,  # proc_id, cfg_node_id
    stageC_cfg_edges_df: pl.DataFrame,  # proc_id, src_cfg_node_id, dst_cfg_node_id, ...
) -> pl.DataFrame:
    proc_ids = stageC_cfg_nodes_df.select("proc_id").unique().to_series().to_list()
    out: list[pl.DataFrame] = []

    for proc_id in proc_ids:
        nodes_proc = stageC_cfg_nodes_df.filter(pl.col("proc_id") == proc_id).select(["cfg_node_id"])
        edges_proc = stageC_cfg_edges_df.filter(pl.col("proc_id") == proc_id).select([
            "src_cfg_node_id", "dst_cfg_node_id"
        ])

        if nodes_proc.height == 0 or edges_proc.height == 0:
            continue

        df = _compute_cdg_edges_for_proc(nodes_proc, edges_proc)
        if df.height == 0:
            continue

        df = df.with_columns([
            pl.lit(proc_id).alias("proc_id"),
            pl.lit("CDG_CONTROL_DEP").alias("edge_kind"),
        ]).select([
            "proc_id",
            "src_cfg_node_id",
            "dst_cfg_node_id",
            "via_succ_cfg_node_id",
            "edge_kind",
        ])

        out.append(df)

    if not out:
        return pl.DataFrame(schema={
            "proc_id": pl.Utf8,
            "src_cfg_node_id": pl.Int64,
            "dst_cfg_node_id": pl.Int64,
            "via_succ_cfg_node_id": pl.Int64,
            "edge_kind": pl.Utf8,
        })

    return pl.concat(out, how="vertical").unique(subset=[
        "proc_id", "src_cfg_node_id", "dst_cfg_node_id", "via_succ_cfg_node_id"
    ])
```

### Notes / best-in-class choices embedded here

1. **Synthetic exit normalization**
   Postdom definitions assume a single exit. We synthesize one when multiple real exits exist. This is standard and makes the computation stable.

2. **“via successor” annotation**
   Keeping `via_succ_cfg_node_id` is *high value* for explainability (why is B control-dependent on A?) and for downstream PDG/CPG tooling.

3. **Bitset approach**
   This is substantially faster than Python set-of-nodes postdom, and avoids bringing in heavier graph libs while still being rigorous.

---

## What changes elsewhere in the Hamilton DAG are implied

Because you explicitly asked: “it’s presumably not just as simple as specifying those edge outputs” — correct. To make Stage D and E “drop in” cleanly and avoid brittleness, you need these **supporting DAG conventions**:

### 1) Ensure Stage C emits procedure identity consistently

Both DDG and CDG require a stable `proc_id`:

* If `proc_id` is a SCIP symbol for the function/method definition, that’s ideal (semantic stable identity).
* If it’s a syntax node id for the function body, that’s also acceptable, but then you must carry `proc_symbol` somewhere for joining later.

### 2) Make the weld datasets first-class

DDG depends on:

* welded occurrences (`occurrence_id → symbol + symbol_roles + syntax_node_id`)
* and a deterministic mapping from occurrence → cfg node

So either:

* Stage C emits `core.xref_occurrence_to_cfg_node`, OR
* Stage D consumes `core.syntax_parent`/`span` and computes it once and emits it as a dataset (then cached, reused)

Your CPG notes explicitly emphasize deterministic welds and explicit crosswalks; DDG/CDG are downstream consumers and should never be responsible for “guessing containment” on the fly. 

### 3) Enforce “CFG nodes are executable syntax nodes” (or provide map)

If `cfg_node_id` is *not* directly a syntax node id, you must provide:

* `stageC.xref_cfg_node_to_syntax_node` (or inverse)
  or you will have to do span containment joins repeatedly (slow, bug-prone).

### 4) Adopt a standard “sink boundary” node per dataset

Both modules should end with a sink like:

* `stageD_ddg_edges_manifest_path`
* `stageE_cdg_edges_manifest_path`

that:

* casts schema
* writes Parquet dataset
* writes manifest JSON

…and returns the manifest path to be included in the suite manifest (consistent with what you already implemented in bootstrap/seed workflow).

---

## Alternate approaches (when you want “even more best-in-class”)

### A) SSA conversion instead of reaching-defs

If you want a DDG that is:

* *exact* within procedure, and
* easier to query (“each use has exactly one reaching def”)

you can convert each proc to SSA, then DDG edges become trivial (def dominates use along SSA). This requires dominator tree + phi insertion. It’s more complex but very powerful.

### B) Use SCIP read/write roles more aggressively

SCIP encodes read/write/definition in a standardized way. 
You can:

* treat writes as defs
* treat reads as uses
* treat imports as “def-like” edges into module scope (optional)

This is already how the above is designed.

### C) Improve Python statement semantics in P1

If you want to fix tricky cases (walrus, comprehension scoping, etc.), you’ll eventually need AST/CST semantics (LibCST) to refine within-statement ordering. For now, statement-level DDG is the correct P0.

---

## Required file link citations

You asked me previously to include these as explicit file citations; included here as references to the uploaded docs:

*
*
*

If you want, the next drill-down after these two modules (still narrative-first) is the **Stage F “call wiring”** step from your CPG doc: connect `CALL` sites to callee entry + parameter flow (`ARG_TO_PARAM`, `RET_TO_CALL`). That’s where DDG and CDG start becoming truly “property graph complete” for LLM consumption.

## Stage F: Call wiring (interprocedural edges) — dense implementation plan + core snippets

### Design goal (why Stage F exists)

Stages C–E give you **intraprocedural** structure (CFG), **data flow** (DDG), and **control dependence** (CDG). A “property-graph complete” CPG for LLM consumption needs the **interprocedural seam**:

1. **CALL → CALLEE entry** (control transfer / call graph)
2. **ARG_TO_PARAM** (actual-in → formal-in value flow)
3. **RET_TO_CALL** (formal-out → actual-out value flow)

Stage F should be the *only* stage that “crosses” function boundaries, and it should do so in a way that:

* maximally leverages **SCIP** for *resolution* (symbol identity + def/ref roles),
* leverages **Tree-sitter/LibCST** for *shape* (call syntax, args, receiver, parameter list),
* leverages **Polars LazyFrame** for *bulk joins/mapping* (fast, predictable, streamable sinks),
* and stays consistent with your “**build produces Arrow datasets; storage consumes**” boundary.

---

## 0) Preconditions (minimal upstream deltas you may need)

Stage F becomes straightforward if upstream tables already exist with stable IDs. If you’re missing any of these, add them as *thin* extraction outputs (no heavy analysis) in your existing syntax/SCIP ingestion and/or Stage C/D.

### Required inputs (dataset-level, Arrow/Parquet)

I’ll refer to generic dataset keys; map to your exact `core.syntax_*`, `core.scip_*`, and `cpg.*` names.

#### A) Call sites (syntax-derived)

`core.syntax_calls`

* `call_id` (stable, deterministic)
* `doc_id` (stable file/document key)
* `call_start_b`, `call_end_b` (byte offsets; 0-based)
* `callee_start_b`, `callee_end_b` (span of the “function expression” part, e.g. `foo.bar` in `foo.bar(x)`)
* `receiver_start_b`, `receiver_end_b` (nullable; span for `foo` in `foo.bar(x)`; for receiver binding)
* `call_node_id` (CPG node for the call expression; used for edges)

#### B) Call arguments (syntax-derived)

`core.syntax_call_args`

* `call_id`
* `arg_ordinal` (0-based)
* `arg_kind` ∈ `{positional, keyword, starargs, kwargs}`
* `arg_name` (nullable; keyword name)
* `arg_expr_node_id` (value-producing expression node id from your CFG/DDG node space)
* (optional) `arg_start_b`, `arg_end_b` (debug/trace)

#### C) Function formal parameters (syntax-derived)

`core.syntax_func_params`

* `callee_def_node_id` **or** `callee_symbol` (pick one primary join key; I recommend `callee_def_node_id` once resolved)
* `param_ordinal` (0-based in function signature order)
* `param_name`
* `param_kind` (Python needs: `posonly`, `pos_or_kw`, `kwonly`, `varargs`, `varkw`)
* `param_node_id` (node representing the parameter variable)

#### D) SCIP occurrences & symbol roles (SCIP-derived)

`core.scip_occurrences`

* `doc_id`
* `occ_start_b`, `occ_end_b`
* `symbol`
* `symbol_roles` (bitmask; use Definition bit to separate defs from refs)

`core.scip_definitions` (or derived view)

* `symbol`
* `def_doc_id`, `def_start_b`, `def_end_b`
* `callee_def_node_id` (node id for the function definition) **and** ideally `callee_entry_node_id` (CFG entry node)

> Stage F should *not* infer “what is a function definition” from scratch. It should consume your existing symbol→definition mapping.

#### E) Return summary node (DDG-derived)

`cpg.func_return_summary`

* `callee_def_node_id`
* `return_node_id` (single summary per callee; Stage D is the right place to create it by welding all `return expr` values into one “formal-out” node)

---

## 1) Stage F outputs (what Stage F writes)

I recommend producing **three edge datasets + one diagnostic dataset** (all Arrow/Parquet + manifest):

1. `cpg.call_targets`
   *(call_id → resolved callee symbol/def, with confidence/metadata)*

2. `cpg.edges_calls`
   `CALLS` edges: `call_node_id → callee_entry_node_id`

3. `cpg.edges_arg_to_param`
   `ARG_TO_PARAM` edges: `arg_expr_node_id → param_node_id` (plus ordinal/name metadata)

4. `cpg.edges_ret_to_call`
   `RET_TO_CALL` edges: `return_node_id → call_node_id`

These four datasets are the “interprocedural seam”. Everything downstream can just `UNION ALL` edges.

---

## 2) Target resolution strategy (SCIP-first, syntax-scoped)

### Key idea

A call site gives you `callee_span` (byte offsets). SCIP occurrences give you `(occ_start_b, occ_end_b, symbol, roles)` per document. Stage F resolves targets by **finding occurrences that overlap the callee span** and choosing the “best” candidate symbol. Use `symbol_roles` to downrank definitions/import-only occurrences; prefer references.

### Why you don’t want a pure Polars join here

Polars doesn’t have a native “interval join” operator; implementing overlap joins via cross-joins + filters explodes. Stage F is the right place to use `intervaltree/sortedcontainers` as the “span join accelerator” (while still keeping everything else in Polars).

---

## 3) Core algorithm: `call_targets` (intervaltree-backed span join)

### Output schema

`cpg.call_targets`

* `call_id`
* `call_node_id`
* `callee_symbol` (nullable)
* `callee_def_node_id` (nullable)
* `callee_entry_node_id` (nullable)
* `resolution_kind` (e.g. `scip_overlap_best`, `scip_none`, `ambiguous`)
* `confidence` (float32)
* (optional) `candidate_count`, `candidate_symbols` (list<str> for debugging)

### Representative implementation snippet (span mapping + ranking)

```python
from __future__ import annotations

from dataclasses import dataclass
from intervaltree import Interval, IntervalTree

import polars as pl

DEFINITION_BIT = 1  # SymbolRole.Definition in SCIP bitmask:contentReference[oaicite:3]{index=3}

@dataclass(frozen=True)
class OccRec:
    start: int
    end: int
    symbol: str
    roles: int

def _pick_best_symbol(overlaps: list[OccRec], callee_end: int) -> tuple[str | None, float, str]:
    """
    Heuristic ranking:
      - prefer refs (not definitions)
      - prefer shorter spans (more specific token)
      - prefer symbol whose start is closest to callee_end (tends to pick 'bar' in 'foo.bar')
    """
    if not overlaps:
        return None, 0.0, "scip_none"

    def score(o: OccRec) -> tuple[int, int, int]:
        is_def = 1 if (o.roles & DEFINITION_BIT) else 0
        span_len = o.end - o.start
        dist_to_end = abs(callee_end - o.end)
        # lower is better
        return (is_def, span_len, dist_to_end)

    best = min(overlaps, key=score)
    # confidence is intentionally coarse but stable:
    # 1.0 if non-def and unique-ish; degrade if definition or many overlaps.
    conf = 1.0
    if best.roles & DEFINITION_BIT:
        conf *= 0.4
    if len(overlaps) > 3:
        conf *= 0.7
    return best.symbol, conf, "scip_overlap_best"

def resolve_call_targets(
    syntax_calls: pl.DataFrame,         # call_id, doc_id, callee_start_b, callee_end_b, call_node_id
    scip_occurrences: pl.DataFrame,     # doc_id, occ_start_b, occ_end_b, symbol, symbol_roles
) -> pl.DataFrame:
    out_rows: list[dict] = []

    # group per doc to keep trees small and deterministic
    calls_by_doc = syntax_calls.partition_by("doc_id", as_dict=True)
    occs_by_doc = scip_occurrences.partition_by("doc_id", as_dict=True)

    for doc_id, calls_df in calls_by_doc.items():
        occ_df = occs_by_doc.get(doc_id)
        if occ_df is None or occ_df.height == 0:
            for row in calls_df.iter_rows(named=True):
                out_rows.append({
                    "call_id": row["call_id"],
                    "call_node_id": row["call_node_id"],
                    "callee_symbol": None,
                    "resolution_kind": "scip_none",
                    "confidence": 0.0,
                })
            continue

        tree = IntervalTree()
        for o in occ_df.iter_rows(named=True):
            sym = o["symbol"]
            if not sym:
                continue
            tree.add(Interval(o["occ_start_b"], o["occ_end_b"], OccRec(o["occ_start_b"], o["occ_end_b"], sym, o["symbol_roles"])))

        for row in calls_df.iter_rows(named=True):
            a, b = row["callee_start_b"], row["callee_end_b"]
            overlaps = [iv.data for iv in tree.overlap(a, b)]
            sym, conf, kind = _pick_best_symbol(overlaps, b)
            out_rows.append({
                "call_id": row["call_id"],
                "call_node_id": row["call_node_id"],
                "callee_symbol": sym,
                "resolution_kind": kind,
                "confidence": conf,
                "candidate_count": len(overlaps),
            })

    return pl.DataFrame(out_rows)
```

### Join to definition + entry node

Immediately after `callee_symbol` is produced, switch back to Polars joins:

```python
call_targets = (
    call_targets_lf
    .join(scip_definitions_lf, on="callee_symbol", how="left")   # adds callee_def_node_id, callee_entry_node_id
    # optionally filter to project-local defs here
)
```

---

## 4) `CALLS` edges (easy once targets are resolved)

**Semantics:** “This call site may invoke that callee entry.”

Schema: `cpg.edges_calls`

* `src_node_id` = `call_node_id`
* `dst_node_id` = `callee_entry_node_id`
* `edge_type` = `"CALLS"`
* `call_id`
* `confidence`

Polars:

```python
edges_calls = (
    call_targets
    .filter(pl.col("callee_entry_node_id").is_not_null())
    .select([
        pl.col("call_node_id").alias("src_node_id"),
        pl.col("callee_entry_node_id").alias("dst_node_id"),
        pl.lit("CALLS").alias("edge_type"),
        "call_id",
        "confidence",
    ])
)
```

---

## 5) `ARG_TO_PARAM` edges (the “real” work)

### Critical design choice: do argument mapping **after** callee resolution

You need a callee identity (`callee_def_node_id`) to look up the **formal parameter table**. That makes mapping deterministic and avoids “guessing the signature”.

### Mapping rules (Python-centric, but genericizable)

Treat the call boundary as an *adapter* between:

* actual argument stream (`core.syntax_call_args`)
* formal parameter stream (`core.syntax_func_params`)

Then implement the language rules as table ops:

1. **Receiver binding (method calls)**
   If you have a receiver span and can materialize `receiver_expr_node_id`, map it to the first parameter *if* the callee signature begins with an instance param (usually `self`).

   * This is the single most important “quality jump” for Python CPGs.
   * If you cannot reliably classify instance vs staticmethod: still emit receiver→param0 with lower confidence, and allow downstream consumers to ignore low-confidence edges.

2. **Positional arguments → positional parameters**

3. **Keyword arguments → matching name**

4. `*args` flows into `varargs` param if present

5. `**kwargs` flows into `varkw` param if present

6. Overflows/invalid calls: still emit edges into varargs/varkw if possible; otherwise drop with a diagnostic row.

### Representative Polars LazyFrame pipeline (positional + keyword + star)

This is written to be *pure table ops* (no per-row python).

```python
def build_arg_to_param_edges(
    call_targets: pl.LazyFrame,         # call_id, callee_def_node_id, confidence
    call_args: pl.LazyFrame,            # call_id, arg_ordinal, arg_kind, arg_name, arg_expr_node_id
    func_params: pl.LazyFrame,          # callee_def_node_id, param_ordinal, param_kind, param_name, param_node_id
) -> pl.LazyFrame:
    # Attach callee_def_node_id to each arg row
    args = call_args.join(
        call_targets.select(["call_id", "callee_def_node_id", "confidence"]),
        on="call_id",
        how="left",
    ).filter(pl.col("callee_def_node_id").is_not_null())

    params = func_params.select([
        "callee_def_node_id", "param_ordinal", "param_kind", "param_name", "param_node_id"
    ])

    # --- positional args (excluding *args) ---
    pos_args = (
        args
        .filter((pl.col("arg_kind") == "positional"))
        .select(["call_id", "callee_def_node_id", "arg_ordinal", "arg_expr_node_id", "confidence"])
    )

    # Match positional args to non-vararg params by ordinal (first pass)
    non_variadic_params = params.filter(~pl.col("param_kind").is_in(["varargs", "varkw"]))

    pos_direct = (
        pos_args
        .join(
            non_variadic_params,
            on=["callee_def_node_id", ("arg_ordinal", "param_ordinal")],
            how="inner",
        )
        .select([
            pl.col("arg_expr_node_id").alias("src_node_id"),
            pl.col("param_node_id").alias("dst_node_id"),
            pl.lit("ARG_TO_PARAM").alias("edge_type"),
            "call_id",
            pl.col("arg_ordinal").alias("arg_ordinal"),
            pl.col("param_ordinal").alias("param_ordinal"),
            pl.col("confidence").alias("confidence"),
        ])
    )

    # Overflow positionals go to varargs if present
    varargs_param = params.filter(pl.col("param_kind") == "varargs").select([
        "callee_def_node_id", "param_node_id"
    ])

    pos_overflow = (
        pos_args
        .join(non_variadic_params, on="callee_def_node_id", how="left")
        .group_by(["call_id", "callee_def_node_id"])
        .agg([
            pl.max("arg_ordinal").alias("max_arg"),
            pl.max("param_ordinal").alias("max_param"),
        ])
        .filter(pl.col("max_arg") > pl.col("max_param"))
        .join(varargs_param, on="callee_def_node_id", how="inner")
        .join(pos_args, on=["call_id", "callee_def_node_id"], how="inner")
        .filter(pl.col("arg_ordinal") > pl.col("max_param"))
        .select([
            pl.col("arg_expr_node_id").alias("src_node_id"),
            pl.col("param_node_id").alias("dst_node_id"),
            pl.lit("ARG_TO_PARAM").alias("edge_type"),
            "call_id",
            pl.col("arg_ordinal"),
            pl.lit(None).cast(pl.Int32).alias("param_ordinal"),
            (pl.col("confidence") * pl.lit(0.7)).alias("confidence"),
        ])
    )

    # --- keyword args ---
    kw_args = args.filter(pl.col("arg_kind") == "keyword").select([
        "call_id", "callee_def_node_id", "arg_name", "arg_expr_node_id", "confidence"
    ])

    kw_direct = (
        kw_args
        .join(params, left_on=["callee_def_node_id", "arg_name"], right_on=["callee_def_node_id", "param_name"], how="inner")
        .select([
            pl.col("arg_expr_node_id").alias("src_node_id"),
            pl.col("param_node_id").alias("dst_node_id"),
            pl.lit("ARG_TO_PARAM").alias("edge_type"),
            "call_id",
            pl.lit(None).cast(pl.Int32).alias("arg_ordinal"),
            pl.col("param_ordinal").alias("param_ordinal"),
            "confidence",
        ])
    )

    # Unmatched keywords go to **kwargs if present
    varkw_param = params.filter(pl.col("param_kind") == "varkw").select([
        "callee_def_node_id", "param_node_id"
    ])

    kw_overflow = (
        kw_args
        .join(params, left_on=["callee_def_node_id", "arg_name"], right_on=["callee_def_node_id", "param_name"], how="left")
        .filter(pl.col("param_node_id").is_null())
        .join(varkw_param, on="callee_def_node_id", how="inner")
        .select([
            pl.col("arg_expr_node_id").alias("src_node_id"),
            pl.col("param_node_id").alias("dst_node_id"),
            pl.lit("ARG_TO_PARAM").alias("edge_type"),
            "call_id",
            pl.lit(None).cast(pl.Int32).alias("arg_ordinal"),
            pl.lit(None).cast(pl.Int32).alias("param_ordinal"),
            (pl.col("confidence") * pl.lit(0.6)).alias("confidence"),
        ])
    )

    return pl.concat([pos_direct, pos_overflow, kw_direct, kw_overflow], how="vertical")
```

### Receiver binding (method calls)

If you have `receiver_expr_node_id` in `core.syntax_calls` (recommended), then:

* Join `call_targets` → `callee_def_node_id`
* Get `param_ordinal==0` node id for that callee
* Emit `ARG_TO_PARAM` edge from receiver node → param0 with moderate confidence

This is extremely high ROI for Python.

---

## 6) `RET_TO_CALL` edges (formal-out → actual-out)

Assuming you already have `cpg.func_return_summary` from Stage D:

```python
edges_ret_to_call = (
    call_targets
    .join(func_return_summary, on="callee_def_node_id", how="inner")
    .select([
        pl.col("return_node_id").alias("src_node_id"),
        pl.col("call_node_id").alias("dst_node_id"),
        pl.lit("RET_TO_CALL").alias("edge_type"),
        "call_id",
        (pl.col("confidence") * pl.lit(0.9)).alias("confidence"),
    ])
)
```

This gives LLMs the “this call produces a value that originates in that function’s returns” hook, which is often more useful than any single intraprocedural DDG fact.

---

## 7) Hamilton module template (Stage F)

This matches the “stage module template” pattern you’ve been using: **tagged outputs**, **LazyFrame compute**, **sink boundary**.

> Note: tags are the lever that makes downstream orchestration/reflection work cleanly.

```python
# stageF_call_wiring.py
from __future__ import annotations

import polars as pl
from hamilton.function_modifiers import tag

# ---- Inputs: assume your loader layer already yields LazyFrames for these dataset keys ----

@tag(dataset="cpg.call_targets", stage="F", kind="diagnostic")
def cpg_call_targets(
    core_syntax_calls: pl.LazyFrame,
    core_scip_occurrences: pl.LazyFrame,
    core_scip_definitions: pl.LazyFrame,
) -> pl.LazyFrame:
    # materialize minimal columns for intervaltree phase
    calls_df = core_syntax_calls.select(["call_id","doc_id","callee_start_b","callee_end_b","call_node_id"]).collect()
    occs_df  = core_scip_occurrences.select(["doc_id","occ_start_b","occ_end_b","symbol","symbol_roles"]).collect()

    targets_df = resolve_call_targets(calls_df, occs_df)  # intervaltree-backed
    targets_lf = targets_df.lazy()

    # join to def/entry mapping
    return (
        targets_lf
        .join(core_scip_definitions.select(["symbol","callee_def_node_id","callee_entry_node_id"]),
              left_on="callee_symbol", right_on="symbol", how="left")
        .drop(["symbol"])
    )

@tag(dataset="cpg.edges_calls", stage="F", kind="edge")
def cpg_edges_calls(cpg_call_targets: pl.LazyFrame) -> pl.LazyFrame:
    return (
        cpg_call_targets
        .filter(pl.col("callee_entry_node_id").is_not_null())
        .select([
            pl.col("call_node_id").alias("src_node_id"),
            pl.col("callee_entry_node_id").alias("dst_node_id"),
            pl.lit("CALLS").alias("edge_type"),
            "call_id",
            "confidence",
        ])
    )

@tag(dataset="cpg.edges_arg_to_param", stage="F", kind="edge")
def cpg_edges_arg_to_param(
    cpg_call_targets: pl.LazyFrame,
    core_syntax_call_args: pl.LazyFrame,
    core_syntax_func_params: pl.LazyFrame,
) -> pl.LazyFrame:
    return build_arg_to_param_edges(
        call_targets=cpg_call_targets,
        call_args=core_syntax_call_args,
        func_params=core_syntax_func_params,
    )

@tag(dataset="cpg.edges_ret_to_call", stage="F", kind="edge")
def cpg_edges_ret_to_call(
    cpg_call_targets: pl.LazyFrame,
    cpg_func_return_summary: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        cpg_call_targets
        .join(cpg_func_return_summary, on="callee_def_node_id", how="inner")
        .select([
            pl.col("return_node_id").alias("src_node_id"),
            pl.col("call_node_id").alias("dst_node_id"),
            pl.lit("RET_TO_CALL").alias("edge_type"),
            "call_id",
            (pl.col("confidence") * pl.lit(0.9)).alias("confidence"),
        ])
    )
```

### Sink boundary (Arrow/Parquet + manifest)

Use Polars **native sinks** for scale. Polars’ docs emphasize that sinks are the preferred high-throughput path, and you can execute multiple sinks together (`collect_all`) to avoid recomputation when writing multiple outputs from the same pipeline.

In your stage runner, treat Stage F like other stages:

* compute these LazyFrames,
* write `*.parquet` (or IPC) + dataset manifests,
* no storage imports.

---

## 8) Practical “best-in-class” refinements (worth doing early)

### A) Preserve ambiguity without poisoning the graph

When multiple candidate callee symbols overlap the callee span, don’t discard them. Emit:

* one **primary** target edge (`CALLS`, `confidence` high),
* plus optional `CALLS` edges for secondary candidates with lower confidence.

This is invaluable for Python where static resolution is imperfect, and it lets LLM consumers decide if they want “high precision only” or “high recall”.

### B) Store *why* you resolved a call

Add `resolution_kind` and `candidate_count` to `cpg.call_targets`. In debugging sessions, this is the first table you look at.

### C) Receiver mapping improves everything

Even a heuristic “receiver → param0” makes the DDG across calls far more usable for “what flows into this method?” questions.

---

## 9) Addendum: SQLGlot + Polars SQL (yes, it can be programmatic)

If you’re using Polars’ `SQLContext` (or generating SQL strings for some join-heavy operations), SQLGlot is a clean way to build/transform SQL programmatically instead of string concatenation. SQLGlot supports constructing SQL ASTs via builder APIs and then emitting SQL via `.sql()`.

A minimal pattern:

```python
from sqlglot import select, exp

def build_edges_sql(edges_table: str) -> str:
    # SELECT src_node_id, dst_node_id FROM edges_table WHERE edge_type = 'CALLS'
    q = (
        select("src_node_id", "dst_node_id")
        .from_(edges_table)
        .where(exp.column("edge_type").eq(exp.Literal.string("CALLS")))
    )
    return q.sql()
```

Then feed that SQL into Polars SQLContext (or wherever you currently evaluate SQL). This keeps query generation:

* composable,
* inspectable (AST),
* safer than ad-hoc string formatting.

---

If you want the next drill-down after this Stage F plan: the most useful is a **single end-to-end “call wiring golden test”** for 3–5 tiny Python files (method call + free function call + kwargs + starargs) where the expected `call_targets`, `ARG_TO_PARAM`, and `RET_TO_CALL` rows are asserted with your Arrow comparison harness. That’s the fastest way to keep Stage F deterministic while multiple engineers add language features in parallel.

Below is a **single end-to-end Stage F “call wiring golden test” fixture** that is intentionally tiny, but exercises the exact semantics you called out:

* **free function call**
* **instance method call (implicit `self`)**
* **`@classmethod` call (implicit `cls`)**
* **`@staticmethod` call (no implicit receiver)**
* **kwargs**
* **`*starargs`** (tuple literal expansion)
* **`**kwargs`** (dict literal expansion; both “matches named param” and “falls into `**kw` bag”)

It’s written to be **deterministic** and **contract-driven** using **span keys** (`path:start_byte:end_byte`) so you can assert byte-for-byte equality using your Arrow harness.

---

## 1) Fixture layout

Recommend committing this under something like:

```
tests/fixtures/call_wiring_golden/
  README.md                      # optional, but nice for humans
  corpus/                        # the “repo” you index
    pkg/__init__.py
    pkg/lib.py
    pkg/methods.py
    pkg/client.py
  expected/
    cpg.call_targets.parquet
    cpg.arg_to_param.parquet
    cpg.ret_to_call.parquet
```

**Important test hygiene choices:**

* The corpus is **ASCII-only** → byte offsets match char offsets; no ambiguity.
* Use **LF line endings** only (CRLF will shift byte offsets and invalidate the golden tables).
* The corpus includes one extra call site in `pkg/methods.py` (`Greeter()`), but the golden expectations **filter to call sites in `pkg/client.py`** via `call_span_key.startswith("pkg/client.py:")`. That keeps the fixture realistic (module globals exist) without exploding the expected rows.

---

## 2) Identity + span key rule

This golden test assumes the Stage F outputs identify nodes/edges using a canonical **span key**:

```text
span_key := f"{path}:{start_byte}:{end_byte}"
```

Where:

* `start_byte` / `end_byte` are **0-based byte offsets in UTF-8**
* `end_byte` is **exclusive**
* `path` is the repo-relative path using `/` separators

This is the same “span-as-primary-key” discipline you’ve been converging on across the suite: it’s stable, joins well, and is language-agnostic.

---

## 3) The 4 tiny Python files (exact corpus)

You can either commit these as files, or (better) have the test **write them verbatim** into a temp dir to eliminate copy/paste drift.

### `pkg/__init__.py`

```python
from __future__ import annotations
```

### `pkg/lib.py`

```python
from __future__ import annotations

def add(a: int, b: int) -> int:
    return a + b

def combine(a: int, b: int = 0, *rest: int, **kw: int) -> int:
    return a + b
```

### `pkg/methods.py`

```python
from __future__ import annotations

class Greeter:
    def greet(self, name: str, punct: str = "!") -> str:
        return self.prefix + name + punct

    @staticmethod
    def join(a: str, b: str) -> str:
        return a + b

    @classmethod
    def make(cls, prefix: str) -> str:
        return prefix

greeter_instance = Greeter()
greeter_instance.prefix = "Hi, "
```

### `pkg/client.py`

```python
from __future__ import annotations

from pkg.lib import add, combine
from pkg.methods import Greeter, greeter_instance

def run():
    x = add(1, 2)
    y = add(a=10, b=20)
    z = combine(*(1, 2, 3))
    w = combine(1, **{"b": 2})
    u = combine(1, **{"x": 4})
    g = Greeter.make(prefix="Hi, ")
    s = greeter_instance.greet(name="Bob", punct="?")
    t = Greeter.join("a", "b")
    return x, y, z, w, u, g, s, t
```

### Pinned digests (optional but recommended)

If you want a “fail fast if files changed” guard (highly recommended because spans are byte-based):

```text
pkg/__init__.py  sha256=5384bfdb2df380b6557cc7a71d16891415bccaa87699406e236f752c6415389f
pkg/lib.py       sha256=4b2d6c7c832f186a933a36940c92f01b10f86604f78945661881613158180c65
pkg/methods.py   sha256=f0d2209e52f0661a3396d93a5f2a9826c42e6e2379b20e451d28c7fadfb261c1
pkg/client.py    sha256=dda98698a68dcd8f27dd93edb8181f6f46ca5bc82379ccecec4e3f76e0d1e1db
```

---

## 4) Expected outputs (golden rows)

These are the **expected rows after Stage F** for **call sites in `pkg/client.py` only**.

### 4.1 `cpg.call_targets`

**PK:** `(call_span_key, callee_qname)`

Each call in `pkg/client.py` resolves to exactly one local callee.

```text
call_span_key              | callee_qname                  | callee_def_span_key         | resolution | confidence
-------------------------- | ---------------------------- | --------------------------- | ---------- | ----------
pkg/client.py:139:148      | pkg.lib.add                  | pkg/lib.py:40:43            | local_def  | 1.0
pkg/client.py:157:172      | pkg.lib.add                  | pkg/lib.py:40:43            | local_def  | 1.0
pkg/client.py:181:200      | pkg.lib.combine              | pkg/lib.py:90:97            | local_def  | 1.0
pkg/client.py:209:231      | pkg.lib.combine              | pkg/lib.py:90:97            | local_def  | 1.0
pkg/client.py:240:262      | pkg.lib.combine              | pkg/lib.py:90:97            | local_def  | 1.0
pkg/client.py:271:298      | pkg.methods.Greeter.make     | pkg/methods.py:253:257      | local_def  | 1.0
pkg/client.py:307:352      | pkg.methods.Greeter.greet    | pkg/methods.py:59:64        | local_def  | 1.0
pkg/client.py:361:383      | pkg.methods.Greeter.join     | pkg/methods.py:177:181      | local_def  | 1.0
```

---

### 4.2 `cpg.arg_to_param`

**PK:** `(call_span_key, arg_span_key, param_span_key)`

Contract semantics asserted by this table:

* receiver-to-first-param for **instance methods** (`self`) and **classmethods** (`cls`)
* **no receiver edge** for `@staticmethod`
* `*(1,2,3)` expands to three positional arg nodes mapping to `a`, `b`, `*rest`
* `**{"b":2}` maps to named param `b`
* `**{"x":4}` maps to the `**kw` bag param (`kw_key="x"`)

```text
call_span_key              | callee_qname               | arg_span_key              | arg_kind   | arg_name | param_name | param_span_key           | kw_key
-------------------------- | -------------------------- | ------------------------- | ---------- | -------- | ---------- | ------------------------ | ------
pkg/client.py:139:148      | pkg.lib.add                | pkg/client.py:143:144     | positional | None     | a          | pkg/lib.py:44:45         | None
pkg/client.py:139:148      | pkg.lib.add                | pkg/client.py:146:147     | positional | None     | b          | pkg/lib.py:52:53         | None

pkg/client.py:157:172      | pkg.lib.add                | pkg/client.py:163:165     | keyword    | a        | a          | pkg/lib.py:44:45         | None
pkg/client.py:157:172      | pkg.lib.add                | pkg/client.py:169:171     | keyword    | b        | b          | pkg/lib.py:52:53         | None

pkg/client.py:181:200      | pkg.lib.combine            | pkg/client.py:191:192     | positional | None     | a          | pkg/lib.py:98:99         | None
pkg/client.py:181:200      | pkg.lib.combine            | pkg/client.py:194:195     | positional | None     | b          | pkg/lib.py:106:107       | None
pkg/client.py:181:200      | pkg.lib.combine            | pkg/client.py:197:198     | positional | None     | rest       | pkg/lib.py:119:123       | None

pkg/client.py:209:231      | pkg.lib.combine            | pkg/client.py:217:218     | positional | None     | a          | pkg/lib.py:98:99         | None
pkg/client.py:209:231      | pkg.lib.combine            | pkg/client.py:228:229     | kwstar     | b        | b          | pkg/lib.py:106:107       | None

pkg/client.py:240:262      | pkg.lib.combine            | pkg/client.py:248:249     | positional | None     | a          | pkg/lib.py:98:99         | None
pkg/client.py:240:262      | pkg.lib.combine            | pkg/client.py:259:260     | kwstar     | x        | kw         | pkg/lib.py:132:134       | x

pkg/client.py:271:298      | pkg.methods.Greeter.make   | pkg/client.py:271:278     | receiver   | None     | cls        | pkg/methods.py:258:261   | None
pkg/client.py:271:298      | pkg.methods.Greeter.make   | pkg/client.py:291:297     | keyword    | prefix    | prefix     | pkg/methods.py:263:269   | None

pkg/client.py:307:352      | pkg.methods.Greeter.greet  | pkg/client.py:307:323     | receiver   | None     | self       | pkg/methods.py:65:69     | None
pkg/client.py:307:352      | pkg.methods.Greeter.greet  | pkg/client.py:335:340     | keyword    | name      | name       | pkg/methods.py:71:75     | None
pkg/client.py:307:352      | pkg.methods.Greeter.greet  | pkg/client.py:348:351     | keyword    | punct     | punct      | pkg/methods.py:82:87     | None

pkg/client.py:361:383      | pkg.methods.Greeter.join   | pkg/client.py:374:377     | positional | None     | a          | pkg/methods.py:182:183   | None
pkg/client.py:361:383      | pkg.methods.Greeter.join   | pkg/client.py:379:382     | positional | None     | b          | pkg/methods.py:190:191   | None
```

---

### 4.3 `cpg.ret_to_call`

**PK:** `(call_span_key, callee_qname, ret_expr_span_key)` (or `(call_span_key, ret_expr_span_key)` if you guarantee 1 target)

You’re asserting the “return value flows to call expression result” edge. Here we model it as the returned expression span → call expression span.

```text
call_span_key              | callee_qname                | ret_expr_span_key          | kind
-------------------------- | --------------------------- | -------------------------- | ----------
pkg/client.py:139:148      | pkg.lib.add                 | pkg/lib.py:79:84           | RET_TO_CALL
pkg/client.py:157:172      | pkg.lib.add                 | pkg/lib.py:79:84           | RET_TO_CALL
pkg/client.py:181:200      | pkg.lib.combine             | pkg/lib.py:160:165         | RET_TO_CALL
pkg/client.py:209:231      | pkg.lib.combine             | pkg/lib.py:160:165         | RET_TO_CALL
pkg/client.py:240:262      | pkg.lib.combine             | pkg/lib.py:160:165         | RET_TO_CALL
pkg/client.py:271:298      | pkg.methods.Greeter.make    | pkg/methods.py:299:305     | RET_TO_CALL
pkg/client.py:307:352      | pkg.methods.Greeter.greet   | pkg/methods.py:123:149     | RET_TO_CALL
pkg/client.py:361:383      | pkg.methods.Greeter.join    | pkg/methods.py:221:226     | RET_TO_CALL
```

---

## 5) Minimal pytest harness (Arrow-centric, schema/PK-driven)

This is a compact test that:

1. writes the corpus into a temp directory (so bytes are exact),
2. runs your pipeline through Stage F,
3. loads the produced Parquet/Arrow tables,
4. filters to `pkg/client.py:*` call sites,
5. projects the contract columns,
6. sorts by PK from schema metadata (with fallback),
7. asserts equality.

### 5.1 Fixture writer (prevents whitespace/line-ending drift)

```python
from __future__ import annotations

from pathlib import Path

FIXTURE_FILES: dict[str, str] = {
    "pkg/__init__.py": "from __future__ import annotations\n",
    "pkg/lib.py": """from __future__ import annotations

def add(a: int, b: int) -> int:
    return a + b

def combine(a: int, b: int = 0, *rest: int, **kw: int) -> int:
    return a + b
""",
    "pkg/methods.py": """from __future__ import annotations

class Greeter:
    def greet(self, name: str, punct: str = "!") -> str:
        return self.prefix + name + punct

    @staticmethod
    def join(a: str, b: str) -> str:
        return a + b

    @classmethod
    def make(cls, prefix: str) -> str:
        return prefix

greeter_instance = Greeter()
greeter_instance.prefix = "Hi, "
""",
    "pkg/client.py": """from __future__ import annotations

from pkg.lib import add, combine
from pkg.methods import Greeter, greeter_instance

def run():
    x = add(1, 2)
    y = add(a=10, b=20)
    z = combine(*(1, 2, 3))
    w = combine(1, **{"b": 2})
    u = combine(1, **{"x": 4})
    g = Greeter.make(prefix="Hi, ")
    s = greeter_instance.greet(name="Bob", punct="?")
    t = Greeter.join("a", "b")
    return x, y, z, w, u, g, s, t
""",
}

def write_fixture_repo(root: Path) -> None:
    for rel, content in FIXTURE_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        # ensure LF and UTF-8
        p.write_text(content.replace("\r\n", "\n"), encoding="utf-8")
```

### 5.2 Arrow compare helper (PK from schema metadata)

```python
from __future__ import annotations

import pyarrow.compute as pc
import pyarrow.parquet as pq

def pk_cols_from_schema(table, fallback: list[str]) -> list[str]:
    md = table.schema.metadata or {}
    raw = md.get(b"codeintel.pk")
    if raw:
        return raw.decode("utf-8").split(",")
    return fallback

def normalize(table, *, pk_fallback: list[str], cols: list[str], call_prefix: str):
    # filter to call sites in client.py
    mask = pc.starts_with(table["call_span_key"], call_prefix)
    table = table.filter(mask)

    # project contract columns
    table = table.select(cols)

    # sort by PK
    pk_cols = pk_cols_from_schema(table, pk_fallback)
    sort_keys = [(c, "ascending") for c in pk_cols]
    return table.sort_by(sort_keys)

def assert_arrow_equal(expected_path, actual_path, *, pk_fallback, cols, call_prefix="pkg/client.py:"):
    exp = pq.read_table(expected_path)
    act = pq.read_table(actual_path)

    exp_n = normalize(exp, pk_fallback=pk_fallback, cols=cols, call_prefix=call_prefix)
    act_n = normalize(act, pk_fallback=pk_fallback, cols=cols, call_prefix=call_prefix)

    # strict equality: schema + row values
    assert exp_n.schema == act_n.schema, (exp_n.schema, act_n.schema)
    assert exp_n.equals(act_n), "Arrow table mismatch"
```

### 5.3 The actual Stage F golden test

You’ll wire this to **your build CLI** (or your Hamilton driver invocation). Pseudocode shape:

```python
from pathlib import Path
import subprocess

def run_build_stageF(*, repo_root: Path, out_dir: Path) -> None:
    """
    Replace this with however you run:
      - P0 bootstrap/index_suite
      - stageF call wiring
    """
    subprocess.check_call([
        "python", "-m", "codeintel_build",
        "bootstrap", "index_suite",
        "--root", str(repo_root),
        "--out", str(out_dir),
    ])
    subprocess.check_call([
        "python", "-m", "codeintel_build",
        "cpg", "stageF_call_wiring",
        "--seed-suite-manifest", str(out_dir / "bootstrap" / "index_suite.json"),
        "--out", str(out_dir),
    ])

def test_stageF_call_wiring_golden(tmp_path: Path):
    repo_root = tmp_path / "repo"
    out_dir = tmp_path / "build_out"
    write_fixture_repo(repo_root)
    run_build_stageF(repo_root=repo_root, out_dir=out_dir)

    # contract columns (keep this *minimal*; avoid churn)
    assert_arrow_equal(
        expected_path="tests/fixtures/call_wiring_golden/expected/cpg.call_targets.parquet",
        actual_path=out_dir / "cpg" / "call_targets.parquet",
        pk_fallback=["call_span_key","callee_qname"],
        cols=["call_span_key","callee_qname","callee_def_span_key","resolution","confidence"],
    )
    assert_arrow_equal(
        expected_path="tests/fixtures/call_wiring_golden/expected/cpg.arg_to_param.parquet",
        actual_path=out_dir / "cpg" / "arg_to_param.parquet",
        pk_fallback=["call_span_key","arg_span_key","param_span_key"],
        cols=["call_span_key","callee_qname","arg_span_key","arg_kind","arg_name","param_name","param_span_key","kw_key"],
    )
    assert_arrow_equal(
        expected_path="tests/fixtures/call_wiring_golden/expected/cpg.ret_to_call.parquet",
        actual_path=out_dir / "cpg" / "ret_to_call.parquet",
        pk_fallback=["call_span_key","callee_qname","ret_expr_span_key"],
        cols=["call_span_key","callee_qname","ret_expr_span_key","kind"],
    )
```

---

## 6) How to generate the `expected/*.parquet` once (one-time bootstrap)

You can generate the expected Parquet files from the row lists above in a small helper script, using Arrow schemas with PK metadata. Skeleton:

```python
import pyarrow as pa
import pyarrow.parquet as pq

def write_expected(path, rows, schema: pa.Schema):
    t = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(t, path)

call_targets_schema = pa.schema([
    ("call_span_key", pa.string()),
    ("callee_qname", pa.string()),
    ("callee_def_span_key", pa.string()),
    ("resolution", pa.string()),
    ("confidence", pa.float64()),
], metadata={b"codeintel.pk": b"call_span_key,callee_qname"})

# repeat for arg_to_param and ret_to_call
```

You generate them once, commit them, and then Stage F is locked down.

---

## 7) What this golden test *forces* engineers to implement correctly

Even if people build Stage F in parallel, this fixture prevents subtle incompatibilities:

* correct receiver semantics (instance vs classmethod vs staticmethod)
* correct `*` expansion when it’s a tuple literal (the cheapest “best-in-class” win)
* correct `**` expansion when it’s a dict literal with string keys
* correct “unknown keyword → `**kw` bag” flow (`kw_key` is carried)
* stable span-based PKs that can be joined deterministically later in the CPG

---

If you want, I can also provide a **second micro-fixture** that targets the next hardest Python call features (pos-only params, kw-only params, defaulted params, and overload-ish patterns) while still keeping expected rows small and deterministic. 
Below is the **second micro‑fixture** (“hard mode”) for Stage‑F call wiring, targeting **positional‑only params**, **keyword‑only params**, **defaulted params**, and **typing.overload (“overload‑ish”)** patterns — while keeping the expected outputs **small, deterministic, and diffable**.

The key design choice that makes this fixture practical: **we compute call spans by searching for a unique ASCII call substring**, so we can derive stable `call_id = f"{relpath}:{byte_start}:{byte_end}"` without needing LibCST range plumbing or AST byte accounting in the golden spec.

---

## Micro‑fixture: `call_wiring_hard_py`

### 1) Fixture file layout

```
tests/fixtures/call_wiring_hard_py/
  pyrightconfig.json
  hard_calls/
    __init__.py
    defs.py
    uses.py
  expected/
    call_wiring_hard_spec.py   # declarative spec → build expected Arrow tables in-test
```

### 2) `pyrightconfig.json` (to keep SCIP/Pyright behavior deterministic)

This ensures Pyright understands pos‑only syntax and stays within the fixture package.

```json
{
  "pythonVersion": "3.11",
  "typeCheckingMode": "basic",
  "include": ["hard_calls"]
}
```

(If you’re indexing via `scip-python`, it uses Pyright under the hood; you want this config to avoid “ambient repo config” affecting the fixture.) 

---

## 3) Fixture source: `hard_calls/defs.py`

This file is intentionally tiny but exercises the binding edge cases.

```python
from __future__ import annotations

from typing import overload, Union


def f_pos_kw(a: int, b: int, /, c: int = 3, *, d: int, e: int = 5) -> int:
    # a,b are positional-only; c is pos-or-kw (defaulted);
    # d is required kw-only; e is defaulted kw-only.
    return a + b + c + d + e


def g_kwonly(*, x: int, y: int = 2) -> int:
    # pure kw-only + default
    return x * y


@overload
def parse_num(x: int, /) -> int: ...


@overload
def parse_num(x: str, /) -> int: ...


def parse_num(x: Union[int, str], /) -> int:
    # runtime implementation (should be the call target)
    return int(x)


class C:
    def method(self, a: int, /, b: int = 2, *, c: int, d: int = 4) -> int:
        # pos-only on method (self and a), defaulted b, required kw-only c, defaulted kw-only d
        return a + b + c + d
```

**Overload-specific expectation:** `parse_num` produces *multiple defs* (2 overload stubs + 1 implementation). A best-in-class call targeter should link calls to the **runtime implementation def** (and optionally record “resolved overload signature” as an annotation), rather than emitting 3 competing call targets.

Also note: SCIP symbol strings are verbose and encode tool/language/project and symbol descriptors; this is normal and why it’s useful to also carry a **stable derived qualname** in your tables for tests and joins. 

---

## 4) Fixture source: `hard_calls/uses.py`

Every call expression is a **unique ASCII substring** (so we can locate its byte span by `bytes.find()` deterministically).

```python
from __future__ import annotations

from .defs import C, f_pos_kw, g_kwonly, parse_num


def run() -> None:
    r1 = f_pos_kw(1, 2, d=4)
    r2 = f_pos_kw(1, 2, 30, d=40, e=50)
    r3 = f_pos_kw(1, 2, c=30, d=40)

    r4 = g_kwonly(x=10)

    r5 = parse_num(7)
    r6 = parse_num("8")

    obj = C()
    r7 = obj.method(1, c=3)
    r8 = obj.method(1, 22, c=33, d=44)

    _ = (r1, r2, r3, r4, r5, r6, r7, r8)
```

---

# Golden expectations (declarative spec → expected tables)

You want a single declarative file that drives expected rows for:

* `stageF.call_targets`
* `stageF.arg_to_param`
* `stageF.ret_to_call`

…and can be turned into Arrow tables in-test.

## 5) `expected/call_wiring_hard_spec.py`

This spec is intentionally **not** coupled to your internal node IDs; it only relies on:

* `doc_relpath`
* `call_text` (unique substring in that doc)
* `callee_qualname` (stable; recommended)
* the Python argument→parameter binding (the real thing under test)

```python
# tests/fixtures/call_wiring_hard_py/expected/call_wiring_hard_spec.py

CALLS = [
    # ---- f_pos_kw(a,b,/ c=3, *, d, e=5) ----
    {
        "label": "r1",
        "doc_relpath": "hard_calls/uses.py",
        "call_text": "f_pos_kw(1, 2, d=4)",
        "callee_qualname": "hard_calls.defs:f_pos_kw",
        "callee_kind": "function",
        "args": [
            # site_arg_pos follows evaluation order for this fixture:
            # positional args then keyword args
            {"site_arg_pos": 0, "arg_kind": "positional", "arg_name": None, "param_pos": 0, "param_name": "a", "param_kind": "posonly"},
            {"site_arg_pos": 1, "arg_kind": "positional", "arg_name": None, "param_pos": 1, "param_name": "b", "param_kind": "posonly"},
            {"site_arg_pos": 2, "arg_kind": "keyword",    "arg_name": "d",  "param_pos": 3, "param_name": "d", "param_kind": "kwonly"},
            # c and e are defaulted -> no ARG_TO_PARAM row in this fixture’s expectation
        ],
    },
    {
        "label": "r2",
        "doc_relpath": "hard_calls/uses.py",
        "call_text": "f_pos_kw(1, 2, 30, d=40, e=50)",
        "callee_qualname": "hard_calls.defs:f_pos_kw",
        "callee_kind": "function",
        "args": [
            {"site_arg_pos": 0, "arg_kind": "positional", "arg_name": None, "param_pos": 0, "param_name": "a", "param_kind": "posonly"},
            {"site_arg_pos": 1, "arg_kind": "positional", "arg_name": None, "param_pos": 1, "param_name": "b", "param_kind": "posonly"},
            {"site_arg_pos": 2, "arg_kind": "positional", "arg_name": None, "param_pos": 2, "param_name": "c", "param_kind": "pos_or_kw"},
            {"site_arg_pos": 3, "arg_kind": "keyword",    "arg_name": "d",  "param_pos": 3, "param_name": "d", "param_kind": "kwonly"},
            {"site_arg_pos": 4, "arg_kind": "keyword",    "arg_name": "e",  "param_pos": 4, "param_name": "e", "param_kind": "kwonly"},
        ],
    },
    {
        "label": "r3",
        "doc_relpath": "hard_calls/uses.py",
        "call_text": "f_pos_kw(1, 2, c=30, d=40)",
        "callee_qualname": "hard_calls.defs:f_pos_kw",
        "callee_kind": "function",
        "args": [
            {"site_arg_pos": 0, "arg_kind": "positional", "arg_name": None, "param_pos": 0, "param_name": "a", "param_kind": "posonly"},
            {"site_arg_pos": 1, "arg_kind": "positional", "arg_name": None, "param_pos": 1, "param_name": "b", "param_kind": "posonly"},
            {"site_arg_pos": 2, "arg_kind": "keyword",    "arg_name": "c",  "param_pos": 2, "param_name": "c", "param_kind": "pos_or_kw"},
            {"site_arg_pos": 3, "arg_kind": "keyword",    "arg_name": "d",  "param_pos": 3, "param_name": "d", "param_kind": "kwonly"},
        ],
    },

    # ---- g_kwonly(*, x, y=2) ----
    {
        "label": "r4",
        "doc_relpath": "hard_calls/uses.py",
        "call_text": "g_kwonly(x=10)",
        "callee_qualname": "hard_calls.defs:g_kwonly",
        "callee_kind": "function",
        "args": [
            {"site_arg_pos": 0, "arg_kind": "keyword", "arg_name": "x", "param_pos": 0, "param_name": "x", "param_kind": "kwonly"},
            # y defaulted -> no row
        ],
    },

    # ---- parse_num overload set + implementation ----
    # Expectation: call target is the IMPLEMENTATION def, not the overload stubs.
    {
        "label": "r5",
        "doc_relpath": "hard_calls/uses.py",
        "call_text": "parse_num(7)",
        "callee_qualname": "hard_calls.defs:parse_num",
        "callee_kind": "function",
        "args": [
            {"site_arg_pos": 0, "arg_kind": "positional", "arg_name": None, "param_pos": 0, "param_name": "x", "param_kind": "posonly"},
        ],
        "notes": {"overload_resolution": "int"},
    },
    {
        "label": "r6",
        "doc_relpath": "hard_calls/uses.py",
        "call_text": "parse_num(\"8\")",
        "callee_qualname": "hard_calls.defs:parse_num",
        "callee_kind": "function",
        "args": [
            {"site_arg_pos": 0, "arg_kind": "positional", "arg_name": None, "param_pos": 0, "param_name": "x", "param_kind": "posonly"},
        ],
        "notes": {"overload_resolution": "str"},
    },

    # ---- obj.method(self,a,/ b=2, *, c, d=4) ----
    # Best-in-class expectation: include receiver->self binding as a first-class arg mapping.
    {
        "label": "r7",
        "doc_relpath": "hard_calls/uses.py",
        "call_text": "obj.method(1, c=3)",
        "callee_qualname": "hard_calls.defs:C.method",
        "callee_kind": "method",
        "args": [
            {"site_arg_pos": 0, "arg_kind": "receiver",   "arg_name": None, "param_pos": 0, "param_name": "self", "param_kind": "posonly"},
            {"site_arg_pos": 1, "arg_kind": "positional", "arg_name": None, "param_pos": 1, "param_name": "a",    "param_kind": "posonly"},
            {"site_arg_pos": 2, "arg_kind": "keyword",    "arg_name": "c",  "param_pos": 3, "param_name": "c",    "param_kind": "kwonly"},
            # b and d defaulted -> no rows
        ],
    },
    {
        "label": "r8",
        "doc_relpath": "hard_calls/uses.py",
        "call_text": "obj.method(1, 22, c=33, d=44)",
        "callee_qualname": "hard_calls.defs:C.method",
        "callee_kind": "method",
        "args": [
            {"site_arg_pos": 0, "arg_kind": "receiver",   "arg_name": None, "param_pos": 0, "param_name": "self", "param_kind": "posonly"},
            {"site_arg_pos": 1, "arg_kind": "positional", "arg_name": None, "param_pos": 1, "param_name": "a",    "param_kind": "posonly"},
            {"site_arg_pos": 2, "arg_kind": "positional", "arg_name": None, "param_pos": 2, "param_name": "b",    "param_kind": "pos_or_kw"},
            {"site_arg_pos": 3, "arg_kind": "keyword",    "arg_name": "c",  "param_pos": 3, "param_name": "c",    "param_kind": "kwonly"},
            {"site_arg_pos": 4, "arg_kind": "keyword",    "arg_name": "d",  "param_pos": 4, "param_name": "d",    "param_kind": "kwonly"},
        ],
    },
]
```

### Why this spec is “best-in-class useful”

* It tests *binding semantics* rather than surface syntax.
* It forces you to handle **receiver/self** as a real input edge (critical for dataflow).
* It forces you to implement the **overload stub vs implementation** distinction (or you’ll get duplicate targets).
* It keeps expectations minimal: defaulted params are tested *implicitly* (binder correctness), without requiring you to represent “default flows” in Stage‑F.

---

# Turning the spec into expected Arrow tables (in-test)

You said you already have an Arrow comparison harness that:

1. reads Arrow
2. checks schema
3. sorts by PK (from schema metadata)
4. compares rows

Here’s the missing bridge: **build expected tables from CALLS** and compute `call_id` spans by byte-search.

## 6) Span derivation helper (ASCII‑safe, byte‑based)

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

def find_unique_utf8_span(path: Path, needle: str) -> tuple[int, int]:
    data = path.read_bytes()
    n = needle.encode("utf-8")
    i = data.find(n)
    if i < 0:
        raise AssertionError(f"needle not found in {path}: {needle!r}")
    j = data.find(n, i + 1)
    if j != -1:
        raise AssertionError(f"needle not unique in {path}: {needle!r}")
    return i, i + len(n)

def span_call_id(doc_relpath: str, b0: int, b1: int) -> str:
    # Align to your canonical span-key rule. If you hash spans in prod, hash here too.
    return f"{doc_relpath}:{b0}:{b1}"
```

## 7) Expected tables builder (PyArrow-first, Polars optional)

This produces **three** expected tables:

* `call_targets`
* `arg_to_param`
* `ret_to_call`

```python
from __future__ import annotations

from pathlib import Path
import pyarrow as pa

def build_expected_tables(fixture_root: Path, calls: list[dict]) -> dict[str, pa.Table]:
    rows_call_targets: list[dict] = []
    rows_arg_to_param: list[dict] = []
    rows_ret_to_call: list[dict] = []

    for c in calls:
        doc_relpath = c["doc_relpath"]
        doc_path = fixture_root / doc_relpath
        b0, b1 = find_unique_utf8_span(doc_path, c["call_text"])
        call_id = span_call_id(doc_relpath, b0, b1)

        # 1) call_targets: 1 target per call in this fixture (by design)
        rows_call_targets.append(
            {
                "call_id": call_id,
                "doc_relpath": doc_relpath,
                "call_b0": b0,
                "call_b1": b1,
                "callee_qualname": c["callee_qualname"],
                "callee_kind": c["callee_kind"],
                "dispatch": "static",
                "confidence": 1.0,
                # optionally: "callee_scip_symbol": None (fill later if you assert it)
                # optionally: "callee_def_role": "implementation" for overload cases
            }
        )

        # 2) ARG_TO_PARAM edges
        for a in c["args"]:
            rows_arg_to_param.append(
                {
                    "call_id": call_id,
                    "site_arg_pos": a["site_arg_pos"],     # includes receiver at 0 for methods
                    "arg_kind": a["arg_kind"],             # positional|keyword|receiver
                    "arg_name": a["arg_name"],             # only set for keyword args
                    "callee_qualname": c["callee_qualname"],
                    "param_pos": a["param_pos"],
                    "param_name": a["param_name"],
                    "param_kind": a["param_kind"],         # posonly|pos_or_kw|kwonly
                }
            )

        # 3) RET_TO_CALL: 1 per call
        rows_ret_to_call.append(
            {
                "call_id": call_id,
                "callee_qualname": c["callee_qualname"],
                "edge_kind": "RET_TO_CALL",
            }
        )

    # You will likely want to enforce your real Arrow schemas here (contracts),
    # including schema.metadata[b"codeintel.pk"].
    # For fixture purposes: table-from-pylist is fine; cast later to the contract schema.
    return {
        "call_targets": pa.Table.from_pylist(rows_call_targets),
        "arg_to_param": pa.Table.from_pylist(rows_arg_to_param),
        "ret_to_call": pa.Table.from_pylist(rows_ret_to_call),
    }
```

### How to integrate with your “PK-from-schema metadata” harness

* Load the **actual** output Arrow schema first.
* Cast/align expected tables to actual schema (including metadata), then compare.

Conceptually:

```python
actual = pa.ipc.open_file(actual_path).read_all()
expected = expected_tbl.cast(actual.schema).replace_schema_metadata(actual.schema.metadata)
assert_table_equal_sorted_by_pk(actual, expected)  # your harness
```

This keeps the fixture generator **agnostic** to contract evolution while still enforcing exactness at compare time.

---

# What this fixture will catch immediately (the point)

1. **Positional-only enforcement in binder**

   * `f_pos_kw` and `parse_num` both have `/`. If you accidentally treat `a,b` or `x` as keywordable, you’ll either:

     * incorrectly accept a keyword mapping, or
     * produce inconsistent `param_kind` annotations.

2. **Kw-only routing**

   * `d` and `c` must be routed via keyword binding in `f_pos_kw` cases.
   * `g_kwonly(x=10)` verifies a pure kw-only signature.

3. **Default skipping without misalignment**

   * `r1` omits `c` and `e`, `r4` omits `y`, `r7` omits `b` and `d`.
     The binder must still map later keyword args correctly and not shift parameter positions.

4. **Overload stubs vs implementation**
   If you naïvely treat every `def parse_num` as a target, you’ll likely emit:

   * 2 overload targets + 1 implementation target per call.
     Best-in-class: pick **implementation** as `call_target`, optionally attach `resolved_overload` metadata (not required for this fixture to pass).

5. **Receiver/self mapping**
   If you skip the receiver edge, your graph loses crucial dataflow and “who is `self`?” context. This fixture forces a decision and gives you a clean expected outcome.

---

# Optional: SCIP snapshot debugging for this fixture

If you want to make SCIP regressions reviewable, the SCIP CLI supports producing a **snapshot directory** with inline reference/definition annotations that diff well in PRs. 
Use `--comment-syntax="#"` so Python files remain visually sane. 

---



