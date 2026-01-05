Below is a concrete **“CPG vNext schema expansion”** that is **additive + backward-compatible**:

* **Keep** `graph.cpg_nodes` and `graph.cpg_edges` schemas exactly as-is.
* **Add** new `node_kind` / `edge_kind` values (safe because those are `VARCHAR`).
* **Add** new **property tables** (typed, query-friendly) and **coverage tables** (so “how good is this build?” is measurable).
* **Maintain build/storage decoupling**: everything here is produced in `src/codeintel/build` as **Arrow tables / Parquet datasets**, and DuckDB ingestion remains a **separate concern** (e.g., DuckDB can read Arrow/Parquet without the DAG depending on a DB instance).

This design explicitly maps to the upstream products you already have in the DAG lineage (syntax facts, syntax_enrich resolved facts, syntax_augment weld/xrefs, tree-sitter structural tables, inspect annotations, bytecode exception tables, etc.).

---

## 1) What stays stable

### Stable base graph tables

* `graph.cpg_nodes`
* `graph.cpg_edges`

Your lineage doc already shows `cpg` consumes AST + syntax + SCIP + GOIDs + flow graphs + symtable/bytecode + inspect overlays + tree-sitter tokenization.

**Rule:** vNext must not change those tables’ columns or primary keys—only add rows (new kinds) and add new sibling tables.

---

## 2) New node kinds and their upstream sources

These are **new rows in `graph.cpg_nodes`** with `source_table_key` pointing at upstream tables and `source_pk_json` holding the upstream PK.

### A. Syntax-facts-as-nodes (high leverage)

Stage 2 already emits these fact tables: `core.syntax_spans/scopes/defs/refs/calls/call_args/func_params/imports`.

Add node kinds:

1. `SYNTAX_SPAN`

* **source_table_key:** `core.syntax_spans`
* **PK in `source_pk_json`:** `{repo, commit, rel_path, producer, span_id}`
* Purpose: canonical, reusable anchor unit across defs/refs/calls/imports (and later token/TS welding).

2. `SYNTAX_SCOPE`

* **source_table_key:** `core.syntax_scopes`
* **PK:** `{repo, commit, rel_path, producer, scope_id}`
* Purpose: syntax-level lexical scope model (complementary to symtable scopes).

3. `SYNTAX_DEF_FACT`

* **source_table_key:** `core.syntax_defs_resolved` (preferred) or `core.syntax_defs`
* **PK:** `{repo, commit, rel_path, producer, def_id}`
* Purpose: a first-class “definition occurrence” node, carrying resolution metadata.

4. `SYNTAX_REF_FACT`

* **source_table_key:** `core.syntax_refs_resolved` or `core.syntax_refs`
* **PK:** `{repo, commit, rel_path, producer, ref_id}`

5. `SYNTAX_CALL_FACT`

* **source_table_key:** `core.syntax_calls_resolved` or `core.syntax_calls`
* **PK:** `{repo, commit, rel_path, producer, call_id}`

6. `SYNTAX_IMPORT_FACT`

* **source_table_key:** `core.syntax_imports_resolved` or `core.syntax_imports`
* **PK:** `{repo, commit, rel_path, producer, import_id}`

7. `SYNTAX_CALL_ARG_FACT`

* **source_table_key:** `core.syntax_call_args`
* **PK:** `{repo, commit, rel_path, producer, call_id, arg_ordinal}`

8. `SYNTAX_FUNC_PARAM_FACT`

* **source_table_key:** `core.syntax_func_params`
* **PK:** `{repo, commit, rel_path, producer, def_id, param_ordinal}`

Why this matters:

* Today, many of these facts are “only” rows in `core.*`. Making them nodes gives you a **uniform place** to attach edges + properties + coverage and enables consistent graph traversals like “callsite → callee resolution → runtime object → signature param”.

### B. Tree-sitter structural and tag nodes (optional but very powerful)

Stage 2 already emits `core.ts_nodes`, `core.ts_edges`, `core.ts_captures`, plus tokens/trivia/errors.

Add node kinds:

9. `TS_NODE`

* **source_table_key:** `core.ts_nodes`
* **PK:** `{repo, commit, rel_path, language, node_id}`

10. `TS_CAPTURE`

* **source_table_key:** `core.ts_captures`
* **PK:** `{repo, commit, rel_path, language, query_pack, capture_name, start_byte, end_byte, node_type}`
* Purpose: “semantic tags” from query packs (decorators, docstring literal spans, SQL strings, etc.).

11. `TS_PARSE_ERROR` (optional, but useful for quality/debug)

* **source_table_key:** `core.ts_parse_errors`
* **PK:** `{repo, commit, rel_path, language, start_byte, end_byte, error_type}`

### C. Runtime/overlay nodes

Stage 2 already emits docstrings + inspect annotations + bytecode exception tables.

Add node kinds:

12. `DOCSTRING`

* **source_table_key:** `core.docstrings`
* **PK suggestion:** `{repo, commit, rel_path, module, qualname, kind, lineno, end_lineno}`
* (Docstrings don’t need byte spans to be useful; line spans + qualname are enough.)

13. `INSPECT_ANNOTATION_KV`

* **source_table_key:** `core.py_inspect_annotations_kv`
* **PK:** `{repo, commit, object_id, key}`

14. `BC_EXCEPTION_ENTRY`

* **source_table_key:** `core.py_bc_exception_table`
* **PK:** `{repo, commit, rel_path, exc_entry_id}`

---

## 3) New edge kinds and how they map upstream

These are **additional rows in `graph.cpg_edges`**.

### A. Syntax fact wiring edges

Derived primarily from Stage 2 `core.syntax_*` + Stage 3 resolved enrich tables.

**Edges:**

1. `HAS_SPAN` (layer: `SYNTAX`)

* `SYNTAX_*_FACT  →  SYNTAX_SPAN`
* Source fields: `span_id`

2. `IN_SCOPE` (layer: `SYNTAX`)

* `SYNTAX_*_FACT  →  SYNTAX_SCOPE`
* Source fields: `scope_id`

3. `ANCHORS_SYNTAX_NODE` (layer: `SYNTAX`)

* `SYNTAX_*_FACT  →  SYNTAX_NODE`
* Source fields:

  * defs/refs/calls_resolved provide `syntax_node_id` (when present)
  * calls_resolved provide `call_node_id`
  * call_args provide `arg_expr_node_id`
  * func_params provide `param_node_id`

4. `ARG_OF` (layer: `SYNTAX`)

* `SYNTAX_CALL_ARG_FACT  →  SYNTAX_CALL_FACT`
* `ordinal = arg_ordinal`

5. `PARAM_OF` (layer: `SYNTAX`)

* `SYNTAX_FUNC_PARAM_FACT  →  SYNTAX_DEF_FACT`
* `ordinal = param_ordinal`

6. `RESOLVES_TO_SCIP_SYMBOL` (layer: `SYMBOL`)

* `SYNTAX_{DEF,REF,CALL,IMPORT}_FACT  →  SCIP_SYMBOL`
* Source: `core.syntax_*_resolved.scip_symbol` (and `match_kind`, `candidate_count` go into `extras_json`)

7. `RESOLVES_TO_GOID` (layer: `SYMBOL`)

* `SYNTAX_{DEF,REF,CALL,IMPORT}_FACT  →  GOID`
* Source: `core.syntax_*_resolved.goid_h128` (and/or existing GOID/SCIP crosswalks)

This is explicitly aligned with your “call_wiring” inputs/outputs: `core.syntax_calls`, `core.syntax_call_args`, `core.syntax_func_params`, plus resolved defs and SCIP span xrefs.

### B. Tree-sitter structural + weld edges

Stage 3 `syntax_augment` already emits the weld mapping (`core.ts_syntax_node_xref`) and coverage (`core.ts_weld_coverage`).

**Edges:**

8. `TS_CHILD` (layer: `TS`)

* `TS_NODE  →  TS_NODE`
* Source: `core.ts_edges`
* `ordinal = child_ordinal`
* `extras_json`: include `field_name`, `field_id`

9. `WELDS_TO_SYNTAX` (layer: `TS`)

* `TS_NODE  →  SYNTAX_NODE`
* Source: `core.ts_syntax_node_xref`
* Put `match_kind`, `candidate_count` into edge `extras_json`

10. `CAPTURED_AS` (layer: `TS`)

* `TS_CAPTURE  →  TS_NODE` (or `→ SYNTAX_NODE` if you join through weld)
* Source: `core.ts_captures` (+ optional matching logic)

### C. Docstrings and inspect overlay edges

Docstrings exist in Stage 2 (`core.docstrings`).
Inspect annotations exist in Stage 2 (`core.py_inspect_annotations_kv`).

**Edges:**

11. `HAS_DOCSTRING` (layer: `DOC`)

* `GOID  →  DOCSTRING`
* Join key: `core.goid_crosswalk` provides module+qualname for GOID alignment (from the `goids` target).

12. `ANNOTATED_WITH` (layer: `INSPECT`)

* `INSPECT_OBJECT  →  INSPECT_ANNOTATION_KV`
* Source: `core.py_inspect_annotations_kv.object_id`

### D. Bytecode exception-flow edges

Bytecode exception tables exist in Stage 2 (`core.py_bc_exception_table`).

**Edges:**

13. `EXC_HANDLER_TARGET` (layer: `FLOW`)

* `BC_EXCEPTION_ENTRY  →  BC_INSTR`
* Join: exception `target_offset` → instruction row with matching `offset` within same `code_unit_id`

14. `EXC_PROTECTS_RANGE` (layer: `FLOW`)

* `BC_EXCEPTION_ENTRY  →  BC_INSTR` (range endpoints)
* Either:

  * emit 2 edges (`EXC_RANGE_START`, `EXC_RANGE_END`), or
  * store as properties in a property table (recommended) to avoid weird semantics.

---

## 4) New property tables

These tables are the “query-friendly layer” that make the base graph usable without everyone re-implementing joins.

### Naming convention

All live in schema `graph`, and are purely additive:

* `graph.cpgx_*` (recommended prefix) or `graph.cpg_*` if you prefer.

I’ll use `cpgx_` below to avoid collision/confusion with existing `graph.cpg_*` tables like call wiring outputs.

---

### 4.1 Syntax fact property tables (plus CPG IDs)

These tables are direct “core table + CPG ids + FK ids” projections.

#### A) `graph.cpgx_syntax_spans`

**Columns**

* `repo`, `commit`, `rel_path`, `producer`, `span_id`, … (all `core.syntax_spans` columns)
* `cpg_span_id` (DECIMAL(38,0)) — the `SYNTAX_SPAN` node’s id

**Derived from**

* `core.syntax_spans` (Stage 2)

#### B) `graph.cpgx_syntax_scopes`

Same pattern:

* core columns + `cpg_scope_id`

#### C) `graph.cpgx_syntax_defs`

**Columns**

* all columns from `core.syntax_defs_resolved` (or `core.syntax_defs`)
* `cpg_def_fact_id`
* `cpg_scope_id` (FK to span/scope nodes)
* `cpg_span_id`
* `cpg_syntax_node_id` (if present)
* `cpg_scip_symbol_id` (if resolved)
* `cpg_goid_id` (if resolved)

**Derived from**

* `core.syntax_defs_resolved` (Stage 3)
* plus lookups to:

  * `core.syntax_spans`, `core.syntax_scopes` (Stage 2)
  * `core.scip_symbol_information` / `core.goids` (already CPG inputs)

Repeat analogous tables:

* `graph.cpgx_syntax_refs`
* `graph.cpgx_syntax_calls`
* `graph.cpgx_syntax_imports`
* `graph.cpgx_syntax_call_args`
* `graph.cpgx_syntax_func_params`

Why these are worth it:

* They become the canonical “semantic facts API” for downstream analytics, instead of every consumer rejoining `core.*` tables back to `graph.cpg_nodes` via JSON PKs.

---

### 4.2 Tree-sitter structural and tag property tables

#### D) `graph.cpgx_ts_nodes`

* all columns from `core.ts_nodes`
* `cpg_ts_node_id`

#### E) `graph.cpgx_ts_edges`

**Columns**

* `repo`, `commit`, `rel_path`, `language`
* `parent_cpg_ts_node_id`
* `child_cpg_ts_node_id`
* `field_id`, `field_name`
* `child_ordinal`

**Derived from**

* `core.ts_edges`

#### F) `graph.cpgx_ts_weld`

**Columns**

* all columns from `core.ts_syntax_node_xref`
* plus:

  * `cpg_ts_node_id`
  * `cpg_syntax_node_id`

**Derived from**

* `core.ts_syntax_node_xref` from `syntax_augment`

#### G) `graph.cpgx_ts_captures`

**Columns**

* all columns from `core.ts_captures`
* `cpg_ts_capture_id`
* optional join outputs:

  * `cpg_ts_node_id` (if you match capture span to a TS node)
  * `cpg_syntax_node_id` (if you weld → syntax)

**Derived from**

* `core.ts_captures` (Stage 2)

#### H) `graph.cpgx_ts_parse_errors`

* all columns from `core.ts_parse_errors`
* plus `cpg_ts_parse_error_id` (optional)

---

### 4.3 Docstring + inspect annotation property tables

#### I) `graph.cpgx_docstrings`

* all columns from `core.docstrings`
* plus `cpg_docstring_id`
* plus `cpg_goid_id` if you align via `core.goid_crosswalk`

Upstream docstrings are already part of the pipeline.

#### J) `graph.cpgx_inspect_annotations_kv`

* all columns from `core.py_inspect_annotations_kv`
* plus `cpg_inspect_annotation_id`
* plus `cpg_inspect_object_id` (FK to `INSPECT_OBJECT` node)

Inspect annotations are explicitly in your Stage 2 outputs.

---

### 4.4 Bytecode exception property table

#### K) `graph.cpgx_bc_exception_entries`

**Columns**

* all columns from `core.py_bc_exception_table`
* plus:

  * `cpg_exc_entry_id`
  * `cpg_code_unit_id` (if you represent code units as nodes already)
  * `start_instr_id`, `end_instr_id`, `target_instr_id` (resolved via join to `core.py_bc_instructions`)
  * `cpg_start_instr_id`, `cpg_end_instr_id`, `cpg_target_instr_id`

Bytecode exception table is a first-class Stage 2 output.

---

### 4.5 One “crosswalk to rule them all” table (high ROI)

#### L) `graph.cpgx_symbol_xref`

Purpose: make it trivial to answer “what is this thing?” across SCIP / GOID / syntax facts / inspect objects.

**Columns (suggested)**

* `repo`, `commit`
* `scip_symbol`, `cpg_scip_symbol_id`
* `goid_h128`, `cpg_goid_id`
* `module_name`, `qualname`
* `inspect_object_id`, `cpg_inspect_object_id`
* `def_id`, `cpg_def_fact_id` (nullable)
* `ref_id`, `cpg_ref_fact_id` (nullable)
* `call_id`, `cpg_call_fact_id` (nullable)
* `confidence` (DOUBLE)
* `extras_json` (e.g., why/how matched)

**Derived from**

* `core.scip_symbol_goid_xref`, `core.scip_occurrence_*_xref` (Stage 3)
* `core.goid_crosswalk` (Stage 4)
* `core.syntax_*_resolved` (Stage 3)
* `core.py_inspect_objects` (Stage 2)

---

## 5) New coverage tables

You already have the idea of CPG quality reporting downstream (`analytics.py_cpg_quality_report`).
vNext makes **coverage first-class at the graph layer** so you can gate builds and debug regressions earlier.

### A) `graph.cpgx_coverage_ts_weld`

Per `(repo, commit, rel_path, producer, language)`:

* `ts_node_count`
* `mapped_to_syntax_count`
* `coverage_ratio`
* `unmatched_syntax_count` (optional)
* `match_kind_breakdown` (STRUCT/MAP)

**Derived from**

* `core.ts_weld_coverage`
* `core.ts_syntax_node_xref`

### B) `graph.cpgx_coverage_syntax_resolution`

Per `(repo, commit, rel_path, producer)`:

* `defs_total`, `defs_resolved_to_symbol`, `defs_resolved_to_goid`
* `refs_total`, `refs_resolved_to_symbol`, …
* `calls_total`, `calls_resolved_to_symbol`, …
* breakdown by `match_kind` / `candidate_count` quantiles

**Derived from**

* `core.syntax_*_resolved` tables

### C) `graph.cpgx_coverage_call_wiring`

Per `(repo, commit, rel_path, producer)`:

* `call_count`
* `call_targets_count`
* `arg_to_param_edges_count`
* `ret_to_call_edges_count`
* optionally: `pct_calls_wired`

Your call wiring dependencies are explicitly enumerated in the lineage doc, making this straightforward and attributable.

### D) `graph.cpgx_coverage_bytecode_exceptions`

Per `(repo, commit, rel_path, code_unit_id)`:

* `exception_entry_count`
* `exception_entries_with_resolved_target_instr`
* `pct_resolved`
* `parse_failures` (if join fails)

**Derived from**

* `core.py_bc_exception_table`, `core.py_bc_instructions`

### E) `graph.cpgx_coverage_inspect_annotations`

Per `(repo, commit, mode)`:

* `inspect_objects`
* `objects_with_annotations`
* `annotation_kv_rows`
* `pct_objects_annotated`

**Derived from**

* `core.py_inspect_objects`, `core.py_inspect_annotations_kv`

---

## 6) Hamilton DAG implementation blueprint

### A. Make this a separate target: `cpg_vnext` (recommended)

Keep `cpg` lean and stable, then add a **new** graph target (e.g., `cpg_vnext` or `cpg_enrich`) that depends on:

* `graph.cpg_nodes`, `graph.cpg_edges` (baseline)
* plus additional upstream tables (syntax facts, resolved tables, tree-sitter structural tables, inspect annotations, bytecode exception table, etc.).

This is the best balance of:

* backward compatibility,
* runtime control (you can toggle it on/off),
* and keeping your “canonical CPG” definition crisp.

### B. Outputs are Arrow datasets, not DuckDB writes

Implement vNext tables using the same **table-backed target** pattern you already use for `cpg` / `call_wiring` (dataset save specs partitioned by `(repo, commit)`).

This preserves the separation you want:

* `src/codeintel/build`: computes + saves Parquet datasets
* `src/codeintel/storage`: optional ingestion/query layer

DuckDB can read Arrow or Parquet directly (so the storage layer can remain thin and purely “adapter code”).

### C. Don’t add new tight coupling to `src/codeintel/storage`

Concretely:

* The **build target** should return `pyarrow.Table` (or `RecordBatchReader`) for each new output.
* The “load into DuckDB” step (if you want it) should live in storage/CLI tooling and use DuckDB’s ability to consume Parquet/Arrow externally.

---

## 7) Why this makes the final CPG “richer” in practice

This vNext expansion gives you capabilities that most CPGs either don’t have or can’t unify:

1. **Span-native semantic facts** (defs/refs/calls/imports) as graph nodes
   → enables uniform “occurrence → resolved symbol/goid → runtime object” traversals.

2. **Tree-sitter structural alignment** with coverage metrics
   → better cross-language portability and query-pack derived tags.

3. **Docstrings and runtime annotations become first-class**
   → you can answer “what does this function promise?” / “what are the param types?” in-graph, not by re-parsing text.

4. **Exception table becomes explicit flow structure**
   → improves correctness of control-flow reasoning (and future security analyses).

5. **Coverage tables let you gate regressions**
   → you can treat mapping quality as a measurable contract, not vibes.

---

If you want, I can also write this up as an “implementation PR plan” (exact new `TableSchema` entries to add to the registry + the `graph_targets.py` spec block for `cpg_vnext` + a suggested `CPGVNextOptions` config shape), but the above is the full concrete schema expansion and lineage mapping.

According to a document from **January 5, 2026**, your current Hamilton pipeline already has all the upstream “planes” needed to support a *much* richer, still-backward-compatible CPG: AST/CST/syntax, SCIP resolution + xrefs, GOIDs + CFG/DFG/CDG/PDG, symtable, bytecode, inspect overlays, and tree-sitter tokenization. The lineage doc explicitly lists these as inputs to `cpg` and `call_wiring`, and it also calls out `syntax_augment` tables like `core.ts_syntax_node_xref` / `core.ts_weld_coverage` that are perfect for higher-fidelity anchoring. 

Below is what you asked for:

1. a **concrete “CPG vNext schema expansion”** (backward-compatible: keep `graph.cpg_nodes` / `graph.cpg_edges` stable; add new node/edge kinds plus new property + coverage tables), **explicitly mapped** to the upstream tables you already produce; and
2. an **implementation PR plan**: the exact new `TableSchema` registry entries + the exact `graph_targets.py` block for `cpg_vnext` + a suggested `CPGVNextOptions` config shape — **without introducing any `build -> storage` dependency**.

---

## CPG vNext schema expansion

### Backward-compatibility rules

* **Do not change** the schema (columns/types) of:

  * `graph.cpg_nodes`
  * `graph.cpg_edges`
* You *may* add:

  * new `node_kind` values and new `edge_kind` values (string vocabulary expansion)
  * **new tables** that join to the stable tables via `cpg_node_id` (and optionally a stable `cpg_edge_id` mapping table)
  * new “coverage/quality” tables

This matches your earlier goal and is aligned with your current lineage: the base CPG is already built from many planes and tables, but you’re not “surfacing” those joins as first-class outputs yet.

---

## Vocabulary expansion: node_kinds and edge_kinds

This is the “cheap” richness upgrade because it doesn’t require schema changes — only additional rows (or reclassification) in `graph.cpg_nodes` / `graph.cpg_edges`.

### Node kinds to add (vNext)

These correspond directly to upstream planes you already materialize:

**Syntax / AST plane**

* `MODULE`, `IMPORT`, `IMPORT_FROM`
* `CLASS`, `FUNCTION`, `METHOD`, `LAMBDA`
* `PARAM`, `ARG`, `CALLSITE`
* `NAME`, `ATTRIBUTE`, `SUBSCRIPT`
* `LITERAL`, `FSTRING`, `DICT`, `LIST`, `SET`, `TUPLE`
* `ASSIGN`, `AUG_ASSIGN`, `ANN_ASSIGN`, `RETURN`, `YIELD`, `RAISE`
* `IF`, `FOR`, `WHILE`, `TRY`, `EXCEPT`, `WITH`, `MATCH` (if you support it)
* `DECORATOR`

**SCIP / symbol plane**

* `SYMBOL` (SCIP symbol identity node)
* `OCCURRENCE` (optional: occurrence anchor node)
* `DIAGNOSTIC` (optional)

**GOID / graph plane**

* `GOID_FUNCTION`
* `CFG_BLOCK` (you already do this)
* (optional) `CFG_EDGE`, `DFG_EDGE` as reified nodes if you want edge-properties via node-properties

**symtable plane**

* `SCOPE`, `BINDING`, `SYMBOL_REF` (you already have many of these per your implementation status doc)

**bytecode plane**

* `BC_CODE_UNIT`, `BC_INSTR`, `BC_BLOCK` (already in your status list)

**inspect plane**

* `INSPECT_OBJECT`, `INSPECT_SIGNATURE`, `INSPECT_SIGNATURE_PARAM` (already in your status list)

### Edge kinds to add (vNext)

**Structure**

* `CONTAINS` (module -> top-level defs; function -> body nodes)
* `AST_PARENT_OF` / `AST_CHILD_OF` (if you want explicit tree edges)

**Binding / naming**

* `DEFINES`, `REFERS_TO`
* `BINDS_DEF`, `BINDS_USE` (you already list these)
* `USES_BINDING`, `DEFINES_BINDING`

**Calls**

* `CALLS`
* `ARGUMENT_OF` / `PARAM_OF`
* `ARG_TO_PARAM` (you already have `graph.cpg_edges_arg_to_param` in call_wiring; surface it in the main CPG layer too if desired)

**Flow**

* `CFG_NEXT`, `CFG_TRUE`, `CFG_FALSE`, `CFG_EXC`
* `DFG_REACHES` (or `REACHES`), `DEF_USE`

**Types**

* `HAS_TYPE`, `RETURNS_TYPE`, `PARAM_HAS_TYPE`

**Inspect overlays**

* `WRAPS`, `DECORATES`
* `HAS_SIGNATURE`, `HAS_PARAM`
* `INSPECT_ANCHORS_AST`, `INSPECT_SYMBOL` (already in your doc’s “Completed” list)

---

# New vNext tables

The key idea: **don’t widen `graph.cpg_nodes` / `graph.cpg_edges`.** Keep them minimal. Put richness in *joinable property tables*.

## 1) `graph.cpg_node_spans`

**Goal:** canonical source coordinates for every node in line/col space (plus bytes), so users don’t have to decode/guess spans.

**Primary join:** `(repo, commit, cpg_node_id)` → `graph.cpg_nodes.id`

**Lineage inputs**

* `graph.cpg_nodes` (bytes + rel_path already present)
* `core.file_line_index` (byte→line/col mapping)

## 2) `graph.cpg_node_xrefs`

**Goal:** *first-class anchoring* to upstream IDs: syntax node IDs, AST node IDs, tree-sitter IDs, SCIP symbol/occurrence IDs, GOIDs, symtable scopes, bytecode instruction IDs, inspect object IDs, etc.

**Primary join:** `(repo, commit, cpg_node_id)`

**Lineage inputs (by xref kind)**

* `core.scip_occurrence_span_xref`, `core.scip_occurrence_syntax_xref`, `core.scip_symbol_goid_xref`
* `core.ts_syntax_node_xref`, `core.ts_weld_coverage` (from `syntax_augment`)
* `core.py_sym_*` (symtable plane) + `core.py_bc_*` (bytecode plane) + `core.py_inspect_*` (inspect plane) are explicitly enumerated as CPG inputs in the lineage doc

## 3) `graph.cpg_edge_keys` (new stable edge id mapping)

**Goal:** give edges a stable ID *without changing `graph.cpg_edges`*.

* `graph.cpg_edges` stays as-is.
* `graph.cpg_edge_keys` assigns `cpg_edge_id = stable_hash(repo, commit, src, dst, kind, layer, rel_path, ordinal)`.

This enables edge property tables and lets you reference edges from other outputs cleanly.

## 4) `graph.cpg_node_kv` (typed K/V properties)

**Goal:** extensible, queryable node properties without “schema churn”.

Examples of properties to emit:

* `name`, `qualname`, `module`, `is_async`, `literal_value_repr`
* `scip_symbol`, `scip_symbol_kind`, `scip_role`
* `type_annotation`, `type_inferred`, `type_runtime`
* bytecode facts: `opcode`, `oparg`, `stack_effect`
* inspect facts: `signature_str`, `unwrap_depth`, `mro_json`

**Lineage sources**

* `core.syntax_nodes_augmented` / `core.syntax_edges_augmented` (for enriched syntax facts)
* `core.scip_symbol_information` / relationships
* `core.py_inspect_annotations_kv` + signatures/params
* `core.py_bc_instructions` + compiler metadata
* `core.py_sym_bindings` / resolution edges
  (all already upstream and already part of CPG’s enumerated inputs)

## 5) `graph.cpg_edge_kv` (typed K/V properties)

**Goal:** store edge metadata without bloating `graph.cpg_edges.extras_json`.

Examples:

* `dataflow_var`, `defuse_event_kind`, `slot_kind`
* `callsite_confidence`, `resolution_source`
* `arg_ordinal`, `param_ordinal`, `mapping_kind`

**Join:** `(repo, commit, cpg_edge_id)` → `graph.cpg_edge_keys.cpg_edge_id`

## 6) Coverage tables (analytics)

Your implementation-status doc explicitly calls out the need to “track anchor coverage metrics … and extend coverage metrics beyond instruction span anchoring”. That’s exactly what vNext coverage tables should do. 

### `analytics.cpg_coverage_files`

Per file: how much of the CPG is anchored and through which planes.

### `analytics.cpg_coverage_summary`

Run/repo/commit level rollup (similar spirit to `analytics.py_cpg_quality_report`, but generalized beyond Python-only bytecode/inspect metrics).

---

# Implementation PR plan

Everything below is structured so that **`src/codeintel/build` stays independent of `src/codeintel/storage`**:

* build produces Arrow datasets (tables) via existing dataset savers
* storage may ingest them later, but build never imports storage

---

## PR 1 — Add TableSchema entries (registry)

**File:** `src/codeintel/core/schemas/output_registry.py`

### 1) Add new TableSchema tuple

Place this near the existing `CPG_OVERRIDE_TABLES` block (same file section).

```python
# --- CPG vNext additional outputs (property + coverage tables) ---

CPG_VNEXT_OVERRIDE_TABLES: tuple[TableSchema, ...] = (
    TableSchema(
        schema="graph",
        name="cpg_node_spans",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("cpg_node_id", "DECIMAL(38,0)", nullable=False),
            Column("rel_path", "VARCHAR"),
            Column("start_byte", "BIGINT"),
            Column("end_byte", "BIGINT"),
            Column("start_line", "INTEGER"),
            Column("start_col", "INTEGER"),
            Column("end_line", "INTEGER"),
            Column("end_col", "INTEGER"),
            Column("span_origin", "VARCHAR"),
            Column("confidence", "DOUBLE"),
            Column("extras_json", "BLOB"),
        ],
        primary_key=("repo", "commit", "cpg_node_id"),
        indexes=(
            Index("idx_graph_cpg_node_spans_repo_commit", ("repo", "commit")),
            Index("idx_graph_cpg_node_spans_rel_path", ("rel_path",)),
        ),
        description="Canonical span (bytes + line/col) for each CPG node.",
    ),
    TableSchema(
        schema="graph",
        name="cpg_node_xrefs",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("cpg_node_id", "DECIMAL(38,0)", nullable=False),
            Column("xref_kind", "VARCHAR", nullable=False),
            Column("xref_value", "VARCHAR", nullable=False),
            Column("origin_table_key", "VARCHAR"),
            Column("origin_pk_json", "BLOB"),
            Column("confidence", "DOUBLE"),
            Column("extras_json", "BLOB"),
        ],
        primary_key=("repo", "commit", "cpg_node_id", "xref_kind", "xref_value"),
        indexes=(
            Index("idx_graph_cpg_node_xrefs_repo_commit", ("repo", "commit")),
            Index("idx_graph_cpg_node_xrefs_kind_value", ("xref_kind", "xref_value")),
        ),
        description="Cross-references from CPG nodes to upstream IDs and external symbols (SCIP, syntax, AST, TS, GOIDs, symtable, bytecode, inspect).",
    ),
    TableSchema(
        schema="graph",
        name="cpg_edge_keys",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("cpg_edge_id", "DECIMAL(38,0)", nullable=False),
            Column("src_cpg_node_id", "DECIMAL(38,0)", nullable=False),
            Column("dst_cpg_node_id", "DECIMAL(38,0)", nullable=False),
            Column("edge_kind", "VARCHAR", nullable=False),
            Column("edge_layer", "VARCHAR", nullable=False),
            Column("rel_path", "VARCHAR"),
            Column("ordinal", "INTEGER"),
            Column("extras_json", "BLOB"),
        ],
        primary_key=("repo", "commit", "cpg_edge_id"),
        indexes=(
            Index("idx_graph_cpg_edge_keys_repo_commit", ("repo", "commit")),
            Index("idx_graph_cpg_edge_keys_src", ("src_cpg_node_id",)),
            Index("idx_graph_cpg_edge_keys_dst", ("dst_cpg_node_id",)),
        ),
        description="Stable edge IDs for CPG edges without modifying graph.cpg_edges.",
    ),
    TableSchema(
        schema="graph",
        name="cpg_node_kv",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("cpg_node_id", "DECIMAL(38,0)", nullable=False),
            Column("prop_key", "VARCHAR", nullable=False),
            Column("ordinal", "INTEGER", nullable=False),
            Column("value_type", "VARCHAR", nullable=False),
            Column("value_str", "VARCHAR"),
            Column("value_i64", "BIGINT"),
            Column("value_f64", "DOUBLE"),
            Column("value_bool", "BOOLEAN"),
            Column("value_json", "BLOB"),
            Column("value_blob", "BLOB"),
            Column("provenance", "VARCHAR"),
            Column("confidence", "DOUBLE"),
            Column("extras_json", "BLOB"),
        ],
        primary_key=("repo", "commit", "cpg_node_id", "prop_key", "ordinal"),
        indexes=(
            Index("idx_graph_cpg_node_kv_repo_commit", ("repo", "commit")),
            Index("idx_graph_cpg_node_kv_key", ("prop_key",)),
        ),
        description="Typed key/value properties for CPG nodes (extensible).",
    ),
    TableSchema(
        schema="graph",
        name="cpg_edge_kv",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("cpg_edge_id", "DECIMAL(38,0)", nullable=False),
            Column("prop_key", "VARCHAR", nullable=False),
            Column("ordinal", "INTEGER", nullable=False),
            Column("value_type", "VARCHAR", nullable=False),
            Column("value_str", "VARCHAR"),
            Column("value_i64", "BIGINT"),
            Column("value_f64", "DOUBLE"),
            Column("value_bool", "BOOLEAN"),
            Column("value_json", "BLOB"),
            Column("value_blob", "BLOB"),
            Column("provenance", "VARCHAR"),
            Column("confidence", "DOUBLE"),
            Column("extras_json", "BLOB"),
        ],
        primary_key=("repo", "commit", "cpg_edge_id", "prop_key", "ordinal"),
        indexes=(
            Index("idx_graph_cpg_edge_kv_repo_commit", ("repo", "commit")),
            Index("idx_graph_cpg_edge_kv_key", ("prop_key",)),
        ),
        description="Typed key/value properties for CPG edges (joins via graph.cpg_edge_keys).",
    ),
    TableSchema(
        schema="analytics",
        name="cpg_coverage_files",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("rel_path", "VARCHAR", nullable=False),
            Column("cpg_node_count", "INTEGER"),
            Column("cpg_edge_count", "INTEGER"),
            Column("nodes_with_span_count", "INTEGER"),
            Column("nodes_with_span_rate", "DOUBLE"),
            Column("nodes_with_scip_count", "INTEGER"),
            Column("nodes_with_syntax_count", "INTEGER"),
            Column("nodes_with_ast_count", "INTEGER"),
            Column("nodes_with_ts_count", "INTEGER"),
            Column("nodes_with_goid_count", "INTEGER"),
            Column("nodes_with_symtable_count", "INTEGER"),
            Column("nodes_with_bytecode_count", "INTEGER"),
            Column("nodes_with_inspect_count", "INTEGER"),
            Column("created_at", "TIMESTAMP"),
        ],
        primary_key=("repo", "commit", "rel_path"),
        indexes=(Index("idx_analytics_cpg_coverage_files_repo_commit", ("repo", "commit")),),
        description="Per-file anchoring/coverage metrics for CPG vNext joins across planes.",
    ),
    TableSchema(
        schema="analytics",
        name="cpg_coverage_summary",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("run_id", "VARCHAR", nullable=False),
            Column("cpg_node_count", "INTEGER"),
            Column("cpg_edge_count", "INTEGER"),
            Column("nodes_with_span_rate", "DOUBLE"),
            Column("nodes_with_scip_rate", "DOUBLE"),
            Column("nodes_with_ts_rate", "DOUBLE"),
            Column("nodes_with_bytecode_rate", "DOUBLE"),
            Column("nodes_with_inspect_rate", "DOUBLE"),
            Column("created_at", "TIMESTAMP"),
        ],
        primary_key=("repo", "commit", "run_id"),
        indexes=(Index("idx_analytics_cpg_coverage_summary_run_id", ("run_id",)),),
        description="Run-level rollup of CPG vNext anchoring/coverage.",
    ),
)
```

### 2) Wire it into the registry

Still in `output_registry.py`:

* Add `*CPG_VNEXT_OVERRIDE_TABLES` into `_all_output_tables()`.
* Extend `_GRAPH_OVERRIDE_TABLE_KEYS` to include these (graph ones).
* Add coverage keys to `NON_INFERABLE_OUTPUT_KEYS` (like `analytics.py_cpg_quality_report` already is).

Concrete patch points:

```python
def _all_output_tables() -> tuple[TableSchema, ...]:
    return (
        ...
        *CPG_OVERRIDE_TABLES,
        *CPG_VNEXT_OVERRIDE_TABLES,
        ...
    )
```

```python
_GRAPH_OVERRIDE_TABLE_KEYS: frozenset[str] = frozenset(
    (
        ...
        *tuple(t.table_key for t in CPG_OVERRIDE_TABLES),
        *tuple(t.table_key for t in CPG_VNEXT_OVERRIDE_TABLES if t.schema == "graph"),
        ...
    )
)
```

```python
NON_INFERABLE_OUTPUT_KEYS: frozenset[str] = frozenset(
    (
        ...
        "analytics.cpg_coverage_files",
        "analytics.cpg_coverage_summary",
        ...
    )
).union(_GRAPH_OVERRIDE_TABLE_KEYS)
```

---

## PR 2 — Add CPGVNextOptions (no storage dependency)

**File:** `src/codeintel/build/hamilton/native/options/graphs.py`

Add:

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class CPGVNextOptions:
    # Whether to emit each vNext table
    enable_node_spans: bool = True
    enable_node_xrefs: bool = True
    enable_edge_keys: bool = True
    enable_node_kv: bool = True
    enable_edge_kv: bool = True
    enable_coverage_tables: bool = True

    # Span selection policy
    span_precedence: tuple[str, ...] = (
        "syntax",
        "ast",
        "ts",
        "scip",
        "bytecode",
        "inspect",
    )

    # Guardrails
    max_xrefs_per_node: int = 32
    max_props_per_node: int = 128
    max_props_per_edge: int = 64

    # Whether to include heavier properties
    include_bytecode_props: bool = True
    include_inspect_props: bool = True
    include_symtable_props: bool = True
    include_scip_props: bool = True
```

Then in a new module (recommended):

**File:** `src/codeintel/build/hamilton/native/graphs/cpg_vnext/options.py`

```python
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.options.graphs import CPGVNextOptions

CPG_VNEXT_TARGET_NAME = "cpg_vnext"

def cpg_vnext__options(env: BuildEnv) -> CPGVNextOptions:
    # Resolve from env.config (toml) with safe defaults.
    # IMPORTANT: keep it pure; no storage access.
    cfg = getattr(env.config, "cpg_vnext", None)
    if cfg is None:
        return CPGVNextOptions()
    return CPGVNextOptions(**cfg.model_dump())
```

(Exact config plumbing depends on your `BuildConfig` shape, but this keeps the dependency direction correct.)

---

## PR 3 — Implement the new DAG nodes (build-only)

**New package:** `src/codeintel/build/hamilton/native/graphs/cpg_vnext/`

Suggested files:

* `__init__.py`
* `spans.py`
* `xrefs.py`
* `edge_keys.py`
* `kv.py`
* `coverage.py`
* `options.py` (from PR2)

**Key constraint:** these modules must only use:

* `codeintel.build.*` utilities (`tabular_to_table`, `align_table_to_contract`, etc.)
* `codeintel.core.schemas.*` (schema provider)
* upstream table inputs (already in build)

…and **must not import** anything from `codeintel.storage.*`.

### Minimal node signatures (Hamilton nodes)

* `cpg_node_spans(env, cpg_nodes, file_line_index, cpg_vnext__options) -> InferableTabularInput`
* `cpg_node_xrefs(env, cpg_nodes, scip_occurrence_span_xref, scip_occurrence_syntax_xref, scip_symbol_goid_xref, ts_syntax_node_xref, ... , cpg_vnext__options) -> ...`
* `cpg_edge_keys(env, cpg_edges, cpg_vnext__options) -> ...`
* `cpg_node_kv(env, cpg_nodes, syntax_nodes_augmented, scip_symbol_information, py_bc_instructions, py_inspect_annotations_kv, ... , cpg_vnext__options) -> ...`
* `cpg_edge_kv(env, cpg_edge_keys, cpg_edges_calls, cpg_edges_arg_to_param, py_bc_defuse_events, ... , cpg_vnext__options) -> ...`
* `cpg_coverage_files(env, cpg_nodes, cpg_edges, cpg_node_spans, cpg_node_xrefs, cpg_vnext__options) -> ...`
* `cpg_coverage_summary(env, cpg_coverage_files, cpg_vnext__options) -> ...`

The lineage doc already enumerates where the data comes from, so these nodes are mostly “surface joins + normalization”, not brand new extraction. 

---

## PR 4 — Add the new target spec in graph_targets.py (exact block)

**File:** `src/codeintel/build/hamilton/native/graphs/graph_targets.py`

### 1) Add imports

```python
from codeintel.build.hamilton.native.graphs.cpg_vnext import (
    CPG_VNEXT_TARGET_NAME,
    CPG_NODE_SPANS_TABLE_KEY,
    CPG_NODE_XREFS_TABLE_KEY,
    CPG_EDGE_KEYS_TABLE_KEY,
    CPG_NODE_KV_TABLE_KEY,
    CPG_EDGE_KV_TABLE_KEY,
    CPG_COVERAGE_FILES_TABLE_KEY,
    CPG_COVERAGE_SUMMARY_TABLE_KEY,
)
```

### 2) Add the target spec block

```python
_CPG_VNEXT_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="graphs",
    target_name=CPG_VNEXT_TARGET_NAME,
    description=(
        "CPG vNext: backward-compatible enrichment of graph.cpg_nodes/graph.cpg_edges "
        "via joinable property tables (spans, xrefs, kv) + coverage outputs."
    ),
    spec_version="1",
    tables=(
        TableTargetTableSpec(
            name="cpg_node_spans",
            base_node="cpg_node_spans",
            save_spec=DatasetSaveSpec(
                table_key=CPG_NODE_SPANS_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
        ),
        TableTargetTableSpec(
            name="cpg_node_xrefs",
            base_node="cpg_node_xrefs",
            save_spec=DatasetSaveSpec(
                table_key=CPG_NODE_XREFS_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
        ),
        TableTargetTableSpec(
            name="cpg_edge_keys",
            base_node="cpg_edge_keys",
            save_spec=DatasetSaveSpec(
                table_key=CPG_EDGE_KEYS_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
        ),
        TableTargetTableSpec(
            name="cpg_node_kv",
            base_node="cpg_node_kv",
            save_spec=DatasetSaveSpec(
                table_key=CPG_NODE_KV_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
        ),
        TableTargetTableSpec(
            name="cpg_edge_kv",
            base_node="cpg_edge_kv",
            save_spec=DatasetSaveSpec(
                table_key=CPG_EDGE_KV_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
        ),
        TableTargetTableSpec(
            name="cpg_coverage_files",
            base_node="cpg_coverage_files",
            save_spec=DatasetSaveSpec(
                table_key=CPG_COVERAGE_FILES_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
        ),
        TableTargetTableSpec(
            name="cpg_coverage_summary",
            base_node="cpg_coverage_summary",
            save_spec=DatasetSaveSpec(
                table_key=CPG_COVERAGE_SUMMARY_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
        ),
    ),
    # Recommend depending on cpg + the “plane alignment” targets you explicitly listed
    # as providing xrefs + line mapping.
    dependencies=(
        CPG_TARGET_NAME,
        "scip_resolution",
        "syntax_augment",
        "file_line_index",
    ),
)
```

### 3) Register it

At the bottom where `TABLE_TARGETS` is declared:

```python
TABLE_TARGETS: tuple[TableTargetSpec, ...] = (
    ...
    _CPG_TABLE_TARGET_SPEC,
    _CPG_VNEXT_TABLE_TARGET_SPEC,
    ...
)
```

> **Note on duplicate outputs:** In this plan, `cpg_vnext` is additive — it produces *new* tables only. It depends on `cpg` for `graph.cpg_nodes`/`graph.cpg_edges`, keeping those stable and avoiding duplicate contract outputs.

---

## PR 5 — Add `cpg_vnext` module exports + table key constants

**New file:** `src/codeintel/build/hamilton/native/graphs/cpg_vnext/__init__.py`

```python
CPG_VNEXT_TARGET_NAME = "cpg_vnext"

CPG_NODE_SPANS_TABLE_KEY = "graph.cpg_node_spans"
CPG_NODE_XREFS_TABLE_KEY = "graph.cpg_node_xrefs"
CPG_EDGE_KEYS_TABLE_KEY  = "graph.cpg_edge_keys"
CPG_NODE_KV_TABLE_KEY    = "graph.cpg_node_kv"
CPG_EDGE_KV_TABLE_KEY    = "graph.cpg_edge_kv"
CPG_COVERAGE_FILES_TABLE_KEY   = "analytics.cpg_coverage_files"
CPG_COVERAGE_SUMMARY_TABLE_KEY = "analytics.cpg_coverage_summary"

from .options import cpg_vnext__options
from .spans import cpg_node_spans
from .xrefs import cpg_node_xrefs
from .edge_keys import cpg_edge_keys
from .kv import cpg_node_kv, cpg_edge_kv
from .coverage import cpg_coverage_files, cpg_coverage_summary
```

Then update:

**File:** `src/codeintel/build/hamilton/native/graphs/__init__.py`

Add:

```python
from .cpg_vnext import *  # or explicit exports
```

---

## PR 6 — Tests + invariants (what to actually assert)

You’ll get a lot of robustness by adding a few “contract-ish” invariants:

### Schema/contract validation

* every produced vNext table aligns to `TableSchema` (you already have the validation machinery)

### Referential invariants

* `graph.cpg_node_spans.cpg_node_id ⊆ graph.cpg_nodes.id`
* `graph.cpg_node_xrefs.cpg_node_id ⊆ graph.cpg_nodes.id`
* `graph.cpg_edge_keys` rows correspond 1:1 to `graph.cpg_edges` rows for the same `(repo, commit)` partition (or at least `>=` if you filter some layers)

### Coverage sanity checks

* per file: `nodes_with_span_count <= cpg_node_count`, and rates are in `[0,1]`

### Decoupling invariant (import boundary)

A simple import test that fails if anything in `src/codeintel/build/**` imports `codeintel.storage`.

---

# Why this is “best-in-class” for your architecture

* **Streamlined:** you keep the base graph stable and minimal; you add “richness” via join tables.
* **Robust:** coverage tables make missing anchors and plane gaps *visible* (and your own doc flags coverage tracking as a remaining checklist item).
* **Extensible:** new properties require no schema changes beyond adding new `prop_key` rows.
* **Decoupled:** build outputs Arrow datasets; storage/DuckDB ingestion remains downstream and separate.

---

## Source docs referenced

* Hamilton DAG lineage and upstream table list: 
* Python AST/symtable/dis/inspect CPG implementation status + explicit need for anchor coverage metrics: 

---

If you want one more layer of concreteness, I can also draft the **exact vNext compute logic** for each new node (e.g., `cpg_node_xrefs` precedence rules + how to cap cardinality deterministically + how to compute `cpg_edge_id` in a stable way using your existing stable-id helpers) — still strictly within `src/codeintel/build` and still without touching `src/codeintel/storage`.

Below is the “one more layer of concreteness” you asked for: **exact vNext compute logic** (nodes + edges + the new `cpg_node_xrefs` precedence/capping rules + a stable `cpg_edge_id`), written so it can live **entirely in `src/codeintel/build`** and **never import `src/codeintel/storage`**.

I’m going to assume your already-proposed vNext expansion includes (at minimum) these new *additions*:

* **New node kinds** added into **`graph.cpg_nodes`** (same schema; just more `node_kind`s)
* **New edge kinds** added into **`graph.cpg_edges`** (same schema; just more `edge_kind`s)
* **New tables** (property/xref/coverage) added alongside, e.g.:

  * `graph.cpg_node_xrefs`
  * `graph.cpg_edge_ids`
  * (optionally) `analytics.cpg_xref_coverage_by_file`, etc.

If your schema names differ slightly, the logic still applies 1:1.

---

## 0) Ground rules for determinism and “no storage dependency”

### Determinism invariants

You already have the right primitives in `src/codeintel/build`:

* `stable_decimal_id(payload, digest_size=16)` via `codeintel.build.graphs.assembly.ids`
* `stable_int_hash(payload, digest_size=8, modulus=2**31-1)` via the same module
* msgpack payload encoding via `codeintel.core.serialization.payload.encode_payload`

In your current CPG, `_legacy.py` wraps these as:

* `_stable_cpg_id(table_key, pk)` → DECIMAL(38,0) ID
* `_stable_ordinal(table_key, payload)` → 32-bit-ish ordinal

**vNext should reuse the same exact approach** so IDs remain stable and predictable.

### Hard separation: build vs storage

Everything below:

* takes **Arrow tables** (`pa.Table`) / “InferableTabularInput” as inputs,
* emits **Arrow tables** as outputs,
* relies on **schema contracts** + `dedupe_table_for_table()` / `align_table_to_contract()`,
* does **not** open DuckDB, does **not** call any storage gateway, does **not** import `codeintel.storage.*`.

---

## 1) Stable ID recipes (nodes + edges)

### 1.1 Node ID: always `stable_cpg_id(source_table_key, source_pk)`

Follow the existing convention exactly:

```python
def cpg_node_id_for_row(source_table_key: str, pk: dict[str, object]) -> int:
    # payload = {"table_key": source_table_key, "pk": pk}
    return stable_decimal_id({"table_key": source_table_key, "pk": dict(pk)}, digest_size=16)
```

**Rule**: the pk dict should be the *actual PK* of the source table whenever possible, so the node can be traced back.

### 1.2 Edge ID (new): `stable_cpg_id("graph.cpg_edges", edge_pk)`

Since `graph.cpg_edges` has **no `edge_id` column** and you want to keep it stable, add a new table `graph.cpg_edge_ids` and compute:

```python
def cpg_edge_id(edge_row: dict[str, object]) -> int:
    pk = {
        "repo": edge_row["repo"],
        "commit": edge_row["commit"],
        "src_cpg_node_id": edge_row["src_cpg_node_id"],
        "dst_cpg_node_id": edge_row["dst_cpg_node_id"],
        "edge_kind": edge_row["edge_kind"],
        "edge_layer": edge_row["edge_layer"],
        "ordinal": edge_row["ordinal"],
        # (optional) include rel_path if you want it “more unique”
        # "rel_path": edge_row.get("rel_path"),
    }
    return stable_decimal_id({"table_key": "graph.cpg_edges", "pk": pk}, digest_size=16)
```

That gives you a **stable** `DECIMAL(38,0)` edge ID derived from the *existing* edge PK tuple.

---

## 2) vNext node additions: exact compute logic per node kind

All of these follow the same pattern:

* Validate required columns exist
* Build `pk_values`
* Compute `cpg_node_id = stable_cpg_id(source_table_key, pk_values)`
* Emit `graph.cpg_nodes` row with:

  * `node_kind = "..."`
  * `source_table_key = "..."`
  * `source_pk_json = encode_payload(pk_values)`
  * `rel_path/start_byte/end_byte` pulled from the source row when available
  * `extras_json = encode_payload({...small, query-friendly summary...})`

### 2.1 `SYNTAX_SCOPE` nodes from `core.syntax_scopes`

**Source**: `core.syntax_scopes`
**PK**: (`repo`, `commit`, `rel_path`, `producer`, `scope_id`)
**Span**: no bytes in schema → `start_byte=None`, `end_byte=None`

**Compute logic**:

* For each scope row, emit node with:

  * extras: `scope_kind`, `start_line/start_col/end_line/end_col`, `parent_scope_id`

### 2.2 `SYNTAX_SPAN` nodes from `core.syntax_spans`

**Source**: `core.syntax_spans`
**PK**: (`repo`, `commit`, `rel_path`, `producer`, `span_id`)
**Span**: `start_byte/end_byte` present (nullable)

**Compute logic**:

* Emit node with:

  * `node_kind="SYNTAX_SPAN"`
  * extras: `span_kind`, line/col

### 2.3 `SYNTAX_DEF` nodes from `core.syntax_defs_resolved`

**Source**: `core.syntax_defs_resolved`
**PK**: (`repo`, `commit`, `rel_path`, `producer`, `def_id`)
**Span**: `start_byte/end_byte` (nullable)

**Compute logic**:

* Emit node with:

  * extras: `def_kind`, `name`, `scope_id`, `span_id`
  * plus resolution summary: `scip_symbol`, `scip_occurrence_id`, role flags, `goid_h128`, `match_kind`, `candidate_count`

### 2.4 `SYNTAX_REF` nodes from `core.syntax_refs_resolved`

Same pattern:

* extras: `ref_kind`, `name`, `scope_id`, `span_id` + resolution fields

### 2.5 `SYNTAX_CALL` nodes from `core.syntax_calls_resolved`

**PK**: (`repo`, `commit`, `rel_path`, `producer`, `call_id`)
**Span**: `start_byte/end_byte` + optional `callee_start_byte/callee_end_byte`

extras should include:

* `callee_text`, `arg_count`, `scope_id`, `span_id`, `callee_span_id`, `call_node_id`
* and resolution fields

### 2.6 `SYNTAX_IMPORT` nodes from `core.syntax_imports_resolved`

extras should include:

* `import_kind`, `module`, `name`, `alias`, `level`
* plus resolution fields

### 2.7 `SYNTAX_CALL_ARG` nodes from `core.syntax_call_args`

**PK**: (`repo`, `commit`, `rel_path`, `producer`, `call_id`, `arg_ordinal`)
extras:

* `arg_kind`, `arg_name`
* `arg_span_id`, `arg_expr_node_id`

### 2.8 `SYNTAX_PARAM` nodes from `core.syntax_func_params`

**PK**: (`repo`, `commit`, `rel_path`, `producer`, `func_def_id`, `param_ordinal`)
extras:

* `param_kind`, `param_name`
* `param_def_id`, `param_span_id`, `param_node_id`

### 2.9 `SCIP_OCCURRENCE` nodes from `core.scip_occurrence_syntax_xref` (+ role join)

This is the first “richer” node that benefits from careful logic.

**Source**: `core.scip_occurrence_syntax_xref`
**PK**: (`repo`, `commit`, `rel_path`, `producer`, `scip_occurrence_id`)
**Span**: use `occ_start_byte/occ_end_byte` if present else null

**Join to enrich roles/goid**:

* `core.scip_occurrence_span_xref` doesn’t have `scip_occurrence_id`, but it has:

  * `rel_path, scip_symbol, start_line,start_col,end_line,end_col`
* Your `scip_occurrence_syntax_xref` row has:

  * `scip_symbol` + `occ_start_line/occ_start_col/occ_end_line/occ_end_col`

So do an **exact key join**:

```python
join_key = (
  "repo","commit","rel_path","scip_symbol",
  ("occ_start_line" -> "start_line"),
  ("occ_start_col"  -> "start_col"),
  ("occ_end_line"   -> "end_line"),
  ("occ_end_col"    -> "end_col"),
)
```

If the join hits:

* enrich node extras with:

  * `roles`, `is_definition`, `is_import`, `is_write`, `is_read`, `goid_h128`, `enclosing_symbol`
    If it misses:
* keep those null (but keep `match_kind/candidate_count` from the syntax weld)

### 2.10 `TS_NODE` nodes from `core.ts_nodes`

**PK**: (`repo`, `commit`, `rel_path`, `language`, `node_id`)
**Span**: `start_byte/end_byte` required (non-null)

extras:

* `node_type`, `is_named`, `is_missing`, `is_error`, `has_error`, `grammar_id`, `kind_id`, `text_preview`

### 2.11 `TS_CAPTURE` nodes from `core.ts_captures`

**PK** is multi-column; use the **actual table PK**:

(`repo`,`commit`,`rel_path`,`language`,`query_pack`,`capture_name`,`start_byte`,`end_byte`,`node_type`)

extras:

* `query_pack`, `capture_name`, `node_type`, `text_preview`, and `extras`

### 2.12 `PY_BC_EXCEPTION_ENTRY` nodes from `core.py_bc_exception_table`

**PK**: (`repo`, `commit`, `rel_path`, `code_unit_id`, `exc_entry_index`)
**Span bytes**: not directly present, but you *can* derive:

**Derivation**:

* Build an index from `core.py_bc_instructions`:

  * key: (`repo`,`commit`,`rel_path`,`code_unit_id`,`offset`) → (`span_start_byte`,`span_end_byte`)
* For each exception entry:

  * `start_byte = instr_span_start_byte at start_offset` (if found)
  * `end_byte   = instr_span_end_byte at end_offset` (if found)

extras:

* offsets/labels/depth/lasti + whether span bytes were derived

---

## 3) vNext edge additions: exact compute logic

### 3.1 Syntax scope tree edges

From `core.syntax_scopes.parent_scope_id`:

* Edge: child_scope → parent_scope
* `edge_kind="PARENT_SCOPE"`
* `edge_layer="SYNTAX"`
* `ordinal = stable_ordinal("graph.cpg_edges_syntax_scope_parent", {"child":child_scope_id,"parent":parent_scope_id})`

### 3.2 Fact anchoring edges (defs/refs/calls/imports/args/params)

For each SYNTAX_* fact node, emit edges to the canonical anchoring objects:

#### A) Fact → SYNTAX_SCOPE

If the fact row has `scope_id`:

* `edge_kind="IN_SCOPE"`
* `edge_layer="SYNTAX"`

#### B) Fact → SYNTAX_SPAN

If it has `span_id`:

* `edge_kind="HAS_SPAN"`
* `edge_layer="SYNTAX"`

#### C) Fact → SYNTAX_NODE

If it has `syntax_node_id` (for resolved tables) or `call_node_id` / `arg_expr_node_id` / `param_node_id`:

* `edge_kind="ANCHORS"`
* `edge_layer="SYNTAX"`
* extras: `{"anchor_field":"syntax_node_id"|"call_node_id"|...,"match_kind":..., "candidate_count":...}` when available

**Deterministic ordinal**:
Use a payload that cannot vary with row ordering:

```python
ordinal = stable_ordinal(
  "graph.cpg_edges_fact_anchor",
  {"fact_table": source_table_key, "fact_pk": pk_values, "anchor": anchor_pk_values},
)
```

(You can’t hash nested dicts directly unless you canonicalize; easiest is to hash a flattened, sorted tuple or hash `encode_payload({...})`.)

### 3.3 Fact → SCIP_SYMBOL

If resolved row has `scip_symbol`:

* `edge_kind="RESOLVES_SYMBOL"`
* `edge_layer="SYMBOL"`
* extras: `scip_occurrence_id`, `scip_roles`, `match_kind`, `candidate_count`, role flags

### 3.4 Fact → GOID

If resolved row has `goid_h128`:

* `edge_kind="RESOLVES_TO"`
* `edge_layer="SYMBOL"` (or `"SEMANTIC"` if you introduce it)
* extras: `{"via":"syntax_*_resolved"}`

### 3.5 SCIP occurrence edges (explicit occurrence node)

From `core.scip_occurrence_syntax_xref`:

#### A) occurrence → symbol

* `edge_kind="OCCURRENCE_OF"`
* `edge_layer="SYMBOL"`

#### B) occurrence → syntax node (when syntax_node_id present)

* `edge_kind="OCCURS_AT"`
* `edge_layer="SYNTAX"`
* extras: `match_kind`, `candidate_count`

#### C) occurrence → goid (when goid_h128 from span_xref join present)

* `edge_kind="RESOLVES_TO"`
* `edge_layer="SYMBOL"`
* extras: `{"via":"core.scip_occurrence_span_xref"}`

### 3.6 Tree-sitter structural edges

From `core.ts_edges`:

* parent TS_NODE → child TS_NODE
* `edge_kind="TS_AST"`
* `edge_layer="SYNTAX"`
* extras: `field_name`, `field_id`, `child_ordinal`
* ordinal: prefer *the actual structural ordinal* to preserve deterministic ordering:

  * `ordinal = child_ordinal` **if** you guarantee uniqueness per (parent, child, child_ordinal)
  * else: `stable_ordinal("core.ts_edges", pk_of_ts_edge)` (safer)

### 3.7 Tree-sitter weld edges

From `core.ts_syntax_node_xref`:

* TS_NODE → SYNTAX_NODE (when `syntax_node_id` not null)
* `edge_kind="WELDS_TO"`
* `edge_layer="SYNTAX"`
* extras: `match_kind`, `candidate_count`, `producer`, `language`

---

## 4) `graph.cpg_node_xrefs`: precedence rules + deterministic cardinality caps

This is the part you explicitly called out.

### 4.1 What `cpg_node_xrefs` should represent

Treat `cpg_node_xrefs` as a **canonical crosswalk** with ranking:

* One row per `(repo, commit, cpg_node_id, xref_kind, xref_rank)`
* `xref_rank=0` is the **best** mapping
* additional ranks are “alternates” up to a cap

Recommended columns (conceptually):

* `cpg_node_id`
* `xref_kind` (e.g. `"SCIP_SYMBOL"`, `"GOID"`, `"TS_NODE"`, `"AST_NODE"`, `"PY_BINDING"`)
* `xref_rank` (0..N-1)
* `target_cpg_node_id`
* `confidence` (optional)
* `match_kind`, `candidate_count` (optional)
* `extras_json` for provenance

### 4.2 Candidate generation (where do xrefs come from?)

Generate candidates **from upstream resolved/xref tables**, not from `cpg_edges`, so you retain match metadata:

#### Candidates for `xref_kind="SCIP_SYMBOL"`

Primary candidate sources (best → worst):

1. `core.syntax_defs_resolved` (definitions)
2. `core.syntax_imports_resolved`
3. `core.syntax_calls_resolved` (callee)
4. `core.syntax_refs_resolved`
5. `core.scip_occurrence_syntax_xref` (fallback for remaining syntax nodes)

Each candidate is: `(syntax_node_id -> scip_symbol)` plus metadata.

#### Candidates for `xref_kind="GOID"`

Sources:

1. `*_resolved.goid_h128` where present (strongest because already localized)
2. else `core.scip_symbol_goid_xref` via the selected scip_symbol candidate
3. else `core.scip_occurrence_span_xref.goid_h128` via occurrence candidate

#### Candidates for `xref_kind="TS_NODE"`

From `core.ts_syntax_node_xref` **inverted** (group by `syntax_node_id`).

#### Candidates for `xref_kind="AST_NODE"`

From a **span resolver** join between `core.syntax_nodes` and `core.ast_nodes` by `rel_path` and byte spans.

(If you want this cheap: only compute AST candidates for syntax nodes that already have a symbol xref, because those are the ones typically queried.)

### 4.3 Scoring / precedence (the exact rule)

Define *three* ranking dimensions:

#### (A) Source priority

```python
SOURCE_RANK = {
  "core.syntax_defs_resolved": 0,
  "core.syntax_imports_resolved": 1,
  "core.syntax_calls_resolved": 2,
  "core.syntax_refs_resolved": 3,
  "core.scip_occurrence_syntax_xref": 4,
}
```

#### (B) Role priority (when applicable)

For symbol-ish candidates:

```python
ROLE_RANK = {
  "DEFINES": 0,
  "IMPORTS": 1,
  "WRITES": 2,
  "REFERS_TO": 3,
  # calls can be treated like refs or separate:
  "CALLS": 4,
}
```

You can derive the role from:

* the resolved-table row flags (`is_definition/is_import/is_write/is_read`)
* or from your existing scip occurrence edge-kind logic (DEFINES/IMPORTS/WRITES/REFERS_TO)

#### (C) Match quality

Use a stable rank map:

```python
MATCH_RANK = {
  "EXACT": 0,
  "BYTE_RANGE": 1,
  "ADJACENT_POINT": 2,
  "SPAN_CONTAINS": 3,
  "LINE_CONTAINS": 4,
  "NONE": 99,
  None: 99,
}
```

Then a full candidate sort key:

```python
sort_key = (
  SOURCE_RANK.get(source_table_key, 99),
  ROLE_RANK.get(role, 99),
  MATCH_RANK.get(match_kind, 99),
  (candidate_count if candidate_count is not None else 2**31-1),
  stable_int_hash({"tie": tie_payload}, digest_size=8, modulus=2**31-1),
)
```

Where `tie_payload` is something like:

* `{"target": target_cpg_node_id, "source": source_table_key, "extra": some_pk_fields}`

### 4.4 Deterministic cap per node/kind

Let `max_xrefs_per_kind_per_node` come from `CpgVNextOptions` (e.g. default 3).

Algorithm:

1. Build a candidate list for each `(repo, commit, cpg_node_id, xref_kind)`
2. Sort using the key above
3. Keep first `K`
4. Emit as `xref_rank = 0..K-1`

This is deterministic because:

* sort key is deterministic
* tiebreaker uses a stable hash over canonical payload

---

## 5) Concrete “how-to” for computing `cpg_node_xrefs` without expensive global joins

Here’s the pragmatic way to do it *within your current patterns*:

### Step 1: build small “candidate frames” (Arrow/Polars) per xref type

Example: `syntax_node -> scip_symbol` candidates from `core.syntax_defs_resolved`:

* input columns: `repo,commit,rel_path,producer,syntax_node_id,scip_symbol,match_kind,candidate_count,is_definition,is_import,is_write,is_read`
* compute:

  * `cpg_node_id` for syntax node using the same PK recipe as your existing `_syntax_node_cpg_id()`
  * `target_cpg_node_id` for scip symbol using existing symbol PK recipe
  * `source_table_key="core.syntax_defs_resolved"`
  * `role="DEFINES"` (forced for defs)
  * `match_kind/candidate_count` from row

Repeat for other resolved tables and for `scip_occurrence_syntax_xref`.

### Step 2: union candidates, group+sort+cap

Implement grouping in Polars (best), or Python dict-of-lists (ok if manageable).

Polars sketch:

* `df = pl.concat([cand_defs, cand_imports, cand_calls, cand_refs, cand_occ], how="vertical_relaxed")`
* add computed `source_rank`, `role_rank`, `match_rank`, `candidate_count_filled`, `tie_hash`
* sort by `(repo,commit,cpg_node_id,xref_kind, source_rank, role_rank, match_rank, candidate_count_filled, tie_hash)`
* groupby `(repo,commit,cpg_node_id,xref_kind)` and take `head(K)` preserving sort order
* assign `xref_rank = pl.cumcount().over(group_cols)`
* emit Arrow table

This stays fully in `build` and scales far better than Python row loops.

---

## 6) `graph.cpg_edge_ids`: exact compute logic

Inputs:

* `graph.cpg_edges` (either from the legacy `cpg_edges` or the new `cpg_vnext_edges` node)

Compute:

* For each edge row, compute `cpg_edge_id` as described in §1.2
* Emit rows:

Columns I recommend:

* `repo`, `commit`, `cpg_edge_id`
* `src_cpg_node_id`, `dst_cpg_node_id`
* `edge_kind`, `edge_layer`, `ordinal`, `rel_path`
* `extras_json` optional (usually null; or `{"pk":...}` if you want traceability)

Primary key:

* (`repo`, `commit`, `cpg_edge_id`)

---

## 7) Where this lives in `src/codeintel/build` (no storage touches)

A clean way to implement without entangling with the legacy file:

* `src/codeintel/build/hamilton/native/graphs/cpg_vnext.py` (or a package `cpg_vnext/`)

  * `cpg_vnext_nodes(...) -> graph.cpg_nodes`
  * `cpg_vnext_edges(...) -> graph.cpg_edges`
  * `cpg_vnext_node_xrefs(...) -> graph.cpg_node_xrefs`
  * `cpg_vnext_edge_ids(...) -> graph.cpg_edge_ids`
* `src/codeintel/build/hamilton/native/options/graphs.py`

  * add `@dataclass(frozen=True) class CpgVNextOptions: ...`
* `src/codeintel/build/hamilton/native/graphs/graph_targets.py`

  * add `TableTargetSpec` for `"cpg_vnext"` that materializes:

    * `graph.cpg_nodes`
    * `graph.cpg_edges`
    * `graph.cpg_node_xrefs`
    * `graph.cpg_edge_ids`
    * (optional coverage tables)

No storage imports required anywhere in that set.

---

## 8) Suggested `CpgVNextOptions` fields (so the above remains controllable)

This directly supports the cardinality caps + heavyweight node toggles:

```python
@dataclass(frozen=True)
class CpgVNextOptions:
    include_syntax_fact_nodes: bool = True        # SYNTAX_DEF/REF/CALL/IMPORT/ARG/PARAM/SPAN/SCOPE
    include_scip_occurrence_nodes: bool = True
    include_ts_nodes: bool = False               # TS_NODE/TS_CAPTURE can be large
    include_bc_exception_nodes: bool = True

    max_xrefs_per_kind_per_node: int = 3
    max_ts_xrefs_per_syntax_node: int = 2
    enable_ast_xrefs: bool = False               # can be expensive; gate it

    # If you want: cap extremely noisy joins deterministically
    max_edges_per_file_per_kind: int = 200_000
```

---

## 9) If you want, I can turn this into “drop-in code blocks” per function

What I wrote above is the *exact* logic and ranking rules, but not pasted as full-length source files.

If you want the next increment of concreteness, I can provide **copy/paste-ready** implementations for:

* `_syntax_defs_resolved_to_cpg_nodes()` (and the parallel ones)
* `_fact_anchor_edges_to_cpg()`
* `_compute_cpg_node_xrefs_polars()` with the scoring/capping implemented exactly
* `_compute_cpg_edge_ids()` vectorized

…all written to match your current helper style (`dedupe_table_for_table`, `align_table_to_contract`, msgpack payloads, `stable_decimal_id/stable_int_hash`), and still strictly confined to `src/codeintel/build`.
