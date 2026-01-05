According to a document from **January 5, 2026**, you already have all the upstream “planes” needed (AST/CST/syntax, SCIP, GOIDs, bytecode, inspect overlays, and tree-sitter), and the design constraints we’re keeping are: **deterministic stable IDs** and **hard build↔storage separation** (build produces Arrow/Parquet; storage is optional downstream ingestion/query).

Below is an explicit **CPG2 table catalog** for a *full redesign* (no compatibility constraints), organized so the **Anchor Layer is built first**, then everything composes cleanly.

I’m intentionally making this “best-in-class” in three ways:

1. **Anchors are first-class** (canonical file + byte ranges + line/col), and every plane maps into them.
2. **Nodes and edges are minimal + typed**, while plane tables carry rich structured fields (no “everyone re-joining JSON PKs”).
3. **Cross-plane mappings are explicit and ranked** (xref tables with deterministic capping/scoring), plus **coverage tables** so quality is measurable—not vibes.

---

# CPG2 target overview

## Naming and partition conventions

* **Target name:** `cpg2`
* **All output tables are Arrow tables** (saved as datasets partitioned by `(repo, commit)`) and **must not import `codeintel.storage.*`** or open DuckDB.
* **Stable IDs:** use your existing stable-id primitives (`stable_decimal_id`, `stable_int_hash`, `encode_payload`) for:

  * `node_id`, `edge_id`, `file_id`, `anchor_id` (all `DECIMAL(38,0)`).
  * deterministic `ordinal` for order-sensitive edges (child ordinal, arg ordinal, param ordinal, etc.).

## Upstream plane availability (inputs you already produce)

* **Tree-sitter plane:** `core.ts_parse_manifest`, `core.ts_nodes`, `core.ts_edges`, `core.ts_captures`, `core.ts_parse_errors`, tokens/trivia, etc.
* **LibCST plane:** `core.cst_nodes`, and (independently) LibCST drives `core.syntax_*` canonical tables.
* **Python AST plane:** `core.ast_nodes` is a primary input to GOIDs and other analyses.
* **SCIP + resolution plane:** symbol/occurrence tables + `core.scip_symbol_goid_xref`, `core.scip_occurrence_span_xref`, `core.scip_occurrence_syntax_xref`, etc.
* **Syntax augment plane:** `core.ts_syntax_node_xref`, `core.ts_weld_coverage` for welding TS→syntax (high-value anchors).
* **Inspect plane:** `core.py_inspect_objects`, signatures, params, annotations KV, etc.

---

# CPG2 output tables (topological order)

**Layer 0 – Manifest & options**

1. `graph.cpg2_manifest`

**Layer 1 – Anchor Layer (built first)**
2. `graph.cpg2_files`
3. `graph.cpg2_anchors`
4. `graph.cpg2_anchor_map_syntax_spans`
5. `graph.cpg2_anchor_map_syntax_nodes`
6. `graph.cpg2_anchor_map_ast_nodes`
7. `graph.cpg2_anchor_map_cst_nodes`
8. `graph.cpg2_anchor_map_ts_nodes`
9. `graph.cpg2_anchor_map_ts_captures`
10. `graph.cpg2_anchor_map_scip_occurrences`
11. `graph.cpg2_anchor_map_bc_instructions`
12. `graph.cpg2_anchor_map_docstrings`

**Layer 2 – Plane tables (typed “facts”, each assigns `node_id` and FKs)**
13. `graph.cpg2_syntax_scopes`
14. `graph.cpg2_syntax_spans`
15. `graph.cpg2_syntax_defs`
16. `graph.cpg2_syntax_refs`
17. `graph.cpg2_syntax_calls`
18. `graph.cpg2_syntax_imports`
19. `graph.cpg2_syntax_call_args`
20. `graph.cpg2_syntax_func_params`
21. `graph.cpg2_ts_nodes`
22. `graph.cpg2_ts_edges`
23. `graph.cpg2_ts_captures`
24. `graph.cpg2_ts_parse_errors`
25. `graph.cpg2_ts_weld`
26. `graph.cpg2_docstrings`
27. `graph.cpg2_inspect_objects`
28. `graph.cpg2_inspect_annotations_kv`
29. `graph.cpg2_bc_instructions`
30. `graph.cpg2_bc_exception_entries`
31. `graph.cpg2_scip_symbols`
32. `graph.cpg2_scip_occurrences`
33. `graph.cpg2_goids`
34. `graph.cpg2_ast_nodes`
35. `graph.cpg2_cst_nodes`

**Layer 3 – Graph assembly**
36. `graph.cpg2_nodes`
37. `graph.cpg2_edges`

**Layer 4 – Cross-plane joins and “best mapping” APIs**
38. `graph.cpg2_node_xrefs`
39. `graph.cpg2_symbol_xref`

**Layer 5 – Coverage / quality gates**
40. `analytics.cpg2_coverage_ts_weld`
41. `analytics.cpg2_coverage_syntax_resolution`
42. `analytics.cpg2_coverage_bytecode_exceptions`
43. `analytics.cpg2_coverage_inspect_annotations`
44. `analytics.cpg2_coverage_anchor_density`

This matches (and generalizes) the “richer graph” plan: syntax facts as nodes, TS structure + weld, docstrings + inspect annotations, bytecode exception modeling, and explicit coverage tables.

---

# Table catalog: TableSchema definitions and Hamilton node signatures

Below I’m writing schemas in a registry-friendly “TableSchema-like” YAML. Types are DuckDB-ish (they’ll map cleanly to Arrow). If your `TableSchema` API wants Arrow `pa.schema`, you can convert 1:1.

## Layer 0 — manifest

### 1) `graph.cpg2_manifest`

```yaml
table_key: graph.cpg2_manifest
pk: [repo, commit]
columns:
  repo: VARCHAR
  commit: VARCHAR
  cpg2_version: VARCHAR            # e.g. "2"
  build_fingerprint: VARCHAR       # deterministic hash of options + code version
  options_msgpack: BLOB            # encode_payload(CPG2Options as dict)
  created_at_utc: TIMESTAMP
```

**Hamilton node signature**

```python
def cpg2__options(env) -> "CPG2Options": ...

def cpg2_manifest(env, cpg2__options: "CPG2Options") -> pa.Table: ...
```

---

## Layer 1 — Anchor Layer

### 2) `graph.cpg2_files`

Purpose: canonical file identity across all planes.

```yaml
table_key: graph.cpg2_files
pk: [repo, commit, file_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  file_id: DECIMAL(38,0)           # stable_decimal_id({"table_key":"graph.cpg2_files","pk":{repo,commit,rel_path}})
  rel_path: VARCHAR
  language_hint: VARCHAR           # best-effort: from ts_parse_manifest, modules, file extension
  module_name: VARCHAR             # if known (python); else NULL
  is_python: BOOLEAN
```

**Hamilton node signature**

```python
def cpg2_files(
    env,
    modules: pa.Table,              # core.modules
    ts_parse_manifest: pa.Table,     # core.ts_parse_manifest
    cpg2__options: "CPG2Options",
) -> pa.Table: ...
```

(These upstream inputs exist per the lineage doc; TS manifest is in tree-sitter target output, and modules exist for GOIDs/SCIP inputs.)

---

### 3) `graph.cpg2_anchors`

Purpose: canonical “location atoms” (file + byte span + line/col), used to weld everything.

```yaml
table_key: graph.cpg2_anchors
pk: [repo, commit, anchor_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  anchor_id: DECIMAL(38,0)          # stable_decimal_id({"table_key":"graph.cpg2_anchors","pk":{file_id,start_byte,end_byte}})
  file_id: DECIMAL(38,0)            # FK → graph.cpg2_files.file_id
  start_byte: BIGINT
  end_byte: BIGINT
  start_line: INTEGER               # nullable if unmapped
  start_col: INTEGER
  end_line: INTEGER
  end_col: INTEGER
  anchor_kind: VARCHAR              # e.g. "EXACT_SPAN" | "LINE_DERIVED" | "SYNTHETIC"
  source_priority: INTEGER          # 0=best (syntax_span), 1=ts_node, 2=ast_node, ...
```

**Hamilton node signature**

```python
def cpg2_anchors(
    env,
    cpg2_files: pa.Table,
    file_line_index: pa.Table,          # core.file_line_index (byte<->line mapping)
    # anchor candidates from multiple planes:
    syntax_spans: pa.Table,             # core.syntax_spans
    syntax_nodes: pa.Table,             # core.syntax_nodes (or syntax_nodes_augmented)
    ast_nodes: pa.Table,                # core.ast_nodes
    cst_nodes: pa.Table,                # core.cst_nodes
    ts_nodes: pa.Table,                 # core.ts_nodes
    ts_captures: pa.Table,              # core.ts_captures
    scip_occurrence_span_xref: pa.Table,# core.scip_occurrence_span_xref
    py_bc_instructions: pa.Table,       # core.py_bc_instructions
    docstrings: pa.Table,               # core.docstrings
    cpg2__options: "CPG2Options",
) -> pa.Table: ...
```

Using `core.file_line_index` as the canonical byte→(line,col) mapping is already an established upstream dependency in the vNext plan and is listed in the lineage doc under Stage 3 outputs.

---

### 4–12) Anchor map tables

These are *not* “nice-to-have”—they’re how you keep the build robust and debuggable: every plane can be measured for anchorability.

#### 4) `graph.cpg2_anchor_map_syntax_spans`

```yaml
table_key: graph.cpg2_anchor_map_syntax_spans
pk: [repo, commit, rel_path, producer, span_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  span_id: BIGINT
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  start_byte: BIGINT
  end_byte: BIGINT
```

Node:

```python
def cpg2_anchor_map_syntax_spans(
    env,
    cpg2_files: pa.Table,
    cpg2_anchors: pa.Table,
    syntax_spans: pa.Table,              # core.syntax_spans
) -> pa.Table: ...
```

#### 5) `graph.cpg2_anchor_map_syntax_nodes`

```yaml
table_key: graph.cpg2_anchor_map_syntax_nodes
pk: [repo, commit, rel_path, producer, syntax_node_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  syntax_node_id: BIGINT
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  start_byte: BIGINT
  end_byte: BIGINT
```

```python
def cpg2_anchor_map_syntax_nodes(env, cpg2_files, cpg2_anchors, syntax_nodes: pa.Table) -> pa.Table: ...
```

#### 6) `graph.cpg2_anchor_map_ast_nodes`

```yaml
table_key: graph.cpg2_anchor_map_ast_nodes
pk: [repo, commit, rel_path, ast_node_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  ast_node_id: BIGINT
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  start_byte: BIGINT
  end_byte: BIGINT
```

```python
def cpg2_anchor_map_ast_nodes(env, cpg2_files, cpg2_anchors, ast_nodes: pa.Table) -> pa.Table: ...
```

#### 7) `graph.cpg2_anchor_map_cst_nodes`

```yaml
table_key: graph.cpg2_anchor_map_cst_nodes
pk: [repo, commit, rel_path, cst_node_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  cst_node_id: BIGINT
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  start_byte: BIGINT
  end_byte: BIGINT
```

```python
def cpg2_anchor_map_cst_nodes(env, cpg2_files, cpg2_anchors, cst_nodes: pa.Table) -> pa.Table: ...
```

(CST nodes exist as a direct LibCST extraction per lineage doc.)

#### 8) `graph.cpg2_anchor_map_ts_nodes`

```yaml
table_key: graph.cpg2_anchor_map_ts_nodes
pk: [repo, commit, rel_path, language, ts_node_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  language: VARCHAR
  ts_node_id: BIGINT
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  start_byte: BIGINT
  end_byte: BIGINT
```

```python
def cpg2_anchor_map_ts_nodes(env, cpg2_files, cpg2_anchors, ts_nodes: pa.Table) -> pa.Table: ...
```

#### 9) `graph.cpg2_anchor_map_ts_captures`

TS captures already have byte spans (per your earlier plan).

```yaml
table_key: graph.cpg2_anchor_map_ts_captures
pk: [repo, commit, rel_path, language, query_pack, capture_name, start_byte, end_byte, node_type]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  language: VARCHAR
  query_pack: VARCHAR
  capture_name: VARCHAR
  node_type: VARCHAR
  start_byte: BIGINT
  end_byte: BIGINT
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
```

```python
def cpg2_anchor_map_ts_captures(env, cpg2_files, cpg2_anchors, ts_captures: pa.Table) -> pa.Table: ...
```

#### 10) `graph.cpg2_anchor_map_scip_occurrences`

We prefer to use `core.scip_occurrence_span_xref` as the bridge to file/byte space (that’s exactly why it exists).

```yaml
table_key: graph.cpg2_anchor_map_scip_occurrences
pk: [repo, commit, rel_path, producer, scip_occurrence_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  scip_occurrence_id: BIGINT
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  occ_start_byte: BIGINT
  occ_end_byte: BIGINT
  match_kind: VARCHAR
  candidate_count: INTEGER
```

```python
def cpg2_anchor_map_scip_occurrences(
    env,
    cpg2_files,
    cpg2_anchors,
    scip_occurrence_span_xref: pa.Table,   # core.scip_occurrence_span_xref
) -> pa.Table: ...
```

#### 11) `graph.cpg2_anchor_map_bc_instructions`

Your vNext plan shows you can index `core.py_bc_instructions` by `(code_unit_id, offset)` to derive spans, and it strongly implies instruction spans are either present or derivable deterministically.

```yaml
table_key: graph.cpg2_anchor_map_bc_instructions
pk: [repo, commit, rel_path, code_unit_id, offset]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  code_unit_id: BIGINT
  offset: INTEGER
  instr_id: BIGINT
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  span_start_byte: BIGINT
  span_end_byte: BIGINT
  span_was_derived: BOOLEAN
```

```python
def cpg2_anchor_map_bc_instructions(
    env,
    cpg2_files,
    cpg2_anchors,
    py_bc_instructions: pa.Table,          # core.py_bc_instructions
) -> pa.Table: ...
```

#### 12) `graph.cpg2_anchor_map_docstrings`

Docstrings exist in Stage 2; anchor them using line→byte mapping where possible.

```yaml
table_key: graph.cpg2_anchor_map_docstrings
pk: [repo, commit, rel_path, module_name, qualname, kind, lineno, end_lineno]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  module_name: VARCHAR
  qualname: VARCHAR
  kind: VARCHAR
  lineno: INTEGER
  end_lineno: INTEGER
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  start_byte: BIGINT
  end_byte: BIGINT
  byte_span_available: BOOLEAN
```

```python
def cpg2_anchor_map_docstrings(
    env,
    cpg2_files,
    cpg2_anchors,
    file_line_index: pa.Table,
    docstrings: pa.Table,                  # core.docstrings
) -> pa.Table: ...
```

---

## Layer 2 — Plane tables (typed, node-rich)

All plane tables follow the same best-in-class pattern:

* Compute `node_id = stable_decimal_id({"table_key": <this_table_key>, "pk": <source pk>})`
* Attach `file_id` and `anchor_id` (from anchor maps)
* Provide FKs to other plane nodes where applicable (scope, span, syntax node, symbol/goid, etc.)
* Keep “match metadata” (`match_kind`, `candidate_count`) as typed columns, not buried in extras, because you’ll query them constantly.

### 13) `graph.cpg2_syntax_scopes`

```yaml
table_key: graph.cpg2_syntax_scopes
pk: [repo, commit, rel_path, producer, scope_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  scope_id: BIGINT
  node_id: DECIMAL(38,0)
  parent_scope_id: BIGINT
  parent_node_id: DECIMAL(38,0)      # nullable
  scope_kind: VARCHAR
  # optional location if present upstream:
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)           # nullable if scope has no byte span
```

```python
def cpg2_syntax_scopes(
    env,
    cpg2_files,
    cpg2_anchor_map_syntax_nodes: pa.Table,
    syntax_scopes: pa.Table,          # core.syntax_scopes
) -> pa.Table: ...
```

(Parent-scope relationships are explicitly part of your edge plan.)

---

### 14) `graph.cpg2_syntax_spans`

```yaml
table_key: graph.cpg2_syntax_spans
pk: [repo, commit, rel_path, producer, span_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  span_id: BIGINT
  node_id: DECIMAL(38,0)
  span_kind: VARCHAR
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  start_byte: BIGINT
  end_byte: BIGINT
```

```python
def cpg2_syntax_spans(env, cpg2_anchor_map_syntax_spans, syntax_spans: pa.Table) -> pa.Table: ...
```

(Syntax spans as canonical anchor unit is a core part of the earlier plan.)

---

### 15) `graph.cpg2_syntax_defs`

```yaml
table_key: graph.cpg2_syntax_defs
pk: [repo, commit, rel_path, producer, def_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  def_id: BIGINT
  node_id: DECIMAL(38,0)
  def_kind: VARCHAR
  name: VARCHAR
  scope_id: BIGINT
  scope_node_id: DECIMAL(38,0)
  span_id: BIGINT
  span_node_id: DECIMAL(38,0)
  syntax_node_id: BIGINT
  syntax_node_node_id: DECIMAL(38,0)
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)

  # resolution surface:
  scip_symbol: VARCHAR
  scip_symbol_node_id: DECIMAL(38,0)
  goid_h128: VARCHAR
  goid_node_id: DECIMAL(38,0)
  match_kind: VARCHAR
  candidate_count: INTEGER
```

```python
def cpg2_syntax_defs(
    env,
    cpg2_syntax_scopes,
    cpg2_syntax_spans,
    cpg2_nodes_syntax_nodes: pa.Table,      # (see below) or map table
    cpg2_scip_symbols,
    cpg2_goids,
    syntax_defs_resolved: pa.Table,         # core.syntax_defs_resolved
) -> pa.Table: ...
```

This table is the “semantic facts as first-class nodes” idea made query-friendly.

---

### 16–20) refs, calls, imports, args, params

I’m keeping these compact since they’re isomorphic to defs.

#### 16) `graph.cpg2_syntax_refs`

```yaml
table_key: graph.cpg2_syntax_refs
pk: [repo, commit, rel_path, producer, ref_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  ref_id: BIGINT
  node_id: DECIMAL(38,0)
  ref_kind: VARCHAR
  name: VARCHAR
  scope_node_id: DECIMAL(38,0)
  span_node_id: DECIMAL(38,0)
  syntax_node_node_id: DECIMAL(38,0)
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  scip_symbol_node_id: DECIMAL(38,0)
  goid_node_id: DECIMAL(38,0)
  match_kind: VARCHAR
  candidate_count: INTEGER
```

```python
def cpg2_syntax_refs(env, cpg2_syntax_scopes, cpg2_syntax_spans, cpg2_scip_symbols, cpg2_goids, syntax_refs_resolved: pa.Table) -> pa.Table: ...
```

#### 17) `graph.cpg2_syntax_calls`

```yaml
table_key: graph.cpg2_syntax_calls
pk: [repo, commit, rel_path, producer, call_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  call_id: BIGINT
  node_id: DECIMAL(38,0)
  callee_text: VARCHAR
  arg_count: INTEGER
  scope_node_id: DECIMAL(38,0)
  span_node_id: DECIMAL(38,0)
  call_syntax_node_node_id: DECIMAL(38,0)
  callee_syntax_node_node_id: DECIMAL(38,0)   # if tracked
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  scip_symbol_node_id: DECIMAL(38,0)
  goid_node_id: DECIMAL(38,0)
  match_kind: VARCHAR
  candidate_count: INTEGER
```

```python
def cpg2_syntax_calls(env, cpg2_syntax_scopes, cpg2_syntax_spans, cpg2_scip_symbols, cpg2_goids, syntax_calls_resolved: pa.Table) -> pa.Table: ...
```

#### 18) `graph.cpg2_syntax_imports`

```yaml
table_key: graph.cpg2_syntax_imports
pk: [repo, commit, rel_path, producer, import_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  import_id: BIGINT
  node_id: DECIMAL(38,0)
  import_kind: VARCHAR
  module: VARCHAR
  name: VARCHAR
  alias: VARCHAR
  level: INTEGER
  scope_node_id: DECIMAL(38,0)
  span_node_id: DECIMAL(38,0)
  syntax_node_node_id: DECIMAL(38,0)
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  scip_symbol_node_id: DECIMAL(38,0)
  goid_node_id: DECIMAL(38,0)
  match_kind: VARCHAR
  candidate_count: INTEGER
```

```python
def cpg2_syntax_imports(env, cpg2_syntax_scopes, cpg2_syntax_spans, cpg2_scip_symbols, cpg2_goids, syntax_imports_resolved: pa.Table) -> pa.Table: ...
```

#### 19) `graph.cpg2_syntax_call_args`

```yaml
table_key: graph.cpg2_syntax_call_args
pk: [repo, commit, rel_path, producer, call_id, arg_ordinal]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  call_id: BIGINT
  arg_ordinal: INTEGER
  node_id: DECIMAL(38,0)
  call_node_id: DECIMAL(38,0)
  arg_kind: VARCHAR
  arg_name: VARCHAR
  arg_span_node_id: DECIMAL(38,0)
  arg_expr_syntax_node_node_id: DECIMAL(38,0)
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
```

```python
def cpg2_syntax_call_args(env, cpg2_syntax_calls, cpg2_syntax_spans, syntax_call_args: pa.Table) -> pa.Table: ...
```

#### 20) `graph.cpg2_syntax_func_params`

```yaml
table_key: graph.cpg2_syntax_func_params
pk: [repo, commit, rel_path, producer, def_id, param_ordinal]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  def_id: BIGINT
  param_ordinal: INTEGER
  node_id: DECIMAL(38,0)
  def_node_id: DECIMAL(38,0)
  param_kind: VARCHAR
  param_name: VARCHAR
  param_span_node_id: DECIMAL(38,0)
  param_syntax_node_node_id: DECIMAL(38,0)
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
```

```python
def cpg2_syntax_func_params(env, cpg2_syntax_defs, cpg2_syntax_spans, syntax_func_params: pa.Table) -> pa.Table: ...
```

These syntax fact node kinds + relationships are exactly what your earlier scope expansion emphasized (defs/refs/calls/imports/args/params as first-class facts).

---

### 21–25) Tree-sitter tables

Tree-sitter structural nodes/edges/captures/errors exist, and `syntax_augment` emits the weld mapping + coverage.

#### 21) `graph.cpg2_ts_nodes`

```yaml
table_key: graph.cpg2_ts_nodes
pk: [repo, commit, rel_path, language, ts_node_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  language: VARCHAR
  ts_node_id: BIGINT
  node_id: DECIMAL(38,0)
  node_type: VARCHAR
  is_named: BOOLEAN
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
```

```python
def cpg2_ts_nodes(env, cpg2_anchor_map_ts_nodes, ts_nodes: pa.Table) -> pa.Table: ...
```

#### 22) `graph.cpg2_ts_edges`

```yaml
table_key: graph.cpg2_ts_edges
pk: [repo, commit, rel_path, language, parent_ts_node_id, child_ts_node_id, child_ordinal]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  language: VARCHAR
  parent_ts_node_id: BIGINT
  child_ts_node_id: BIGINT
  parent_node_id: DECIMAL(38,0)
  child_node_id: DECIMAL(38,0)
  field_id: INTEGER
  field_name: VARCHAR
  child_ordinal: INTEGER
```

```python
def cpg2_ts_edges(env, cpg2_ts_nodes, ts_edges: pa.Table) -> pa.Table: ...
```

(Edge kind `TS_CHILD` / `TS_AST` with deterministic ordinal was explicitly called out.)

#### 23) `graph.cpg2_ts_captures`

```yaml
table_key: graph.cpg2_ts_captures
pk: [repo, commit, rel_path, language, query_pack, capture_name, start_byte, end_byte, node_type]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  language: VARCHAR
  query_pack: VARCHAR
  capture_name: VARCHAR
  node_type: VARCHAR
  start_byte: BIGINT
  end_byte: BIGINT
  node_id: DECIMAL(38,0)
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  ts_node_node_id: DECIMAL(38,0)      # nullable if capture not attached
  syntax_node_node_id: DECIMAL(38,0)  # nullable if welded
  text_preview: VARCHAR               # optional, capped
```

```python
def cpg2_ts_captures(env, cpg2_anchor_map_ts_captures, cpg2_ts_nodes, cpg2_ts_weld, ts_captures: pa.Table) -> pa.Table: ...
```

#### 24) `graph.cpg2_ts_parse_errors`

```yaml
table_key: graph.cpg2_ts_parse_errors
pk: [repo, commit, rel_path, language, start_byte, end_byte, error_type]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  language: VARCHAR
  start_byte: BIGINT
  end_byte: BIGINT
  error_type: VARCHAR
  node_id: DECIMAL(38,0)
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
```

```python
def cpg2_ts_parse_errors(env, cpg2_anchors, cpg2_files, ts_parse_errors: pa.Table) -> pa.Table: ...
```

#### 25) `graph.cpg2_ts_weld`

```yaml
table_key: graph.cpg2_ts_weld
pk: [repo, commit, rel_path, producer, language, ts_node_id, syntax_node_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  language: VARCHAR
  ts_node_id: BIGINT
  syntax_node_id: BIGINT
  ts_node_node_id: DECIMAL(38,0)
  syntax_node_node_id: DECIMAL(38,0)
  match_kind: VARCHAR
  candidate_count: INTEGER
```

```python
def cpg2_ts_weld(env, cpg2_ts_nodes, syntax_nodes: pa.Table, ts_syntax_node_xref: pa.Table) -> pa.Table: ...
```

(TS→syntax weld edges and coverage are explicitly in the earlier scope expansion.)

---

### 26–28) Docstrings + inspect overlay

Docstrings and inspect annotation KV are Stage 2 outputs, and the earlier scope expansion adds explicit edges for both (GOID→DOCSTRING and INSPECT_OBJECT→ANNOTATION_KV).

#### 26) `graph.cpg2_docstrings`

```yaml
table_key: graph.cpg2_docstrings
pk: [repo, commit, rel_path, module_name, qualname, kind, lineno, end_lineno]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  module_name: VARCHAR
  qualname: VARCHAR
  kind: VARCHAR
  lineno: INTEGER
  end_lineno: INTEGER
  node_id: DECIMAL(38,0)
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)           # nullable if byte span unavailable
  goid_node_id: DECIMAL(38,0)        # nullable (join via goid_crosswalk)
  doc_text: VARCHAR                  # optional, maybe truncated
```

```python
def cpg2_docstrings(
    env,
    cpg2_anchor_map_docstrings,
    goid_crosswalk: pa.Table,         # core.goid_crosswalk
    goids: pa.Table,                  # core.goids
    docstrings: pa.Table,             # core.docstrings
) -> pa.Table: ...
```

(Using `core.goid_crosswalk` as the alignment key was explicit in the plan.)

#### 27) `graph.cpg2_inspect_objects`

```yaml
table_key: graph.cpg2_inspect_objects
pk: [repo, commit, object_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  object_id: BIGINT
  node_id: DECIMAL(38,0)
  object_kind: VARCHAR
  module_name: VARCHAR
  qualname: VARCHAR
  goid_node_id: DECIMAL(38,0)        # nullable (optional crosswalk)
```

```python
def cpg2_inspect_objects(env, py_inspect_objects: pa.Table, goid_crosswalk: pa.Table, cpg2_goids: pa.Table) -> pa.Table: ...
```

(Inspect tables are explicitly enumerated as outputs.)

#### 28) `graph.cpg2_inspect_annotations_kv`

```yaml
table_key: graph.cpg2_inspect_annotations_kv
pk: [repo, commit, object_id, key]
columns:
  repo: VARCHAR
  commit: VARCHAR
  object_id: BIGINT
  key: VARCHAR
  node_id: DECIMAL(38,0)
  object_node_id: DECIMAL(38,0)      # FK → cpg2_inspect_objects.node_id
  annotation_repr: VARCHAR           # serialized/pretty string form
  annotation_kind: VARCHAR           # e.g. "type" | "param" | "return" | "var"
```

```python
def cpg2_inspect_annotations_kv(
    env,
    cpg2_inspect_objects,
    py_inspect_annotations_kv: pa.Table,    # core.py_inspect_annotations_kv
) -> pa.Table: ...
```

(Edge `ANNOTATED_WITH` from INSPECT_OBJECT→INSPECT_ANNOTATION_KV was explicit.)

---

### 29–30) Bytecode plane (instructions + exception entries)

Exception tables exist, and you already described the join from exception `target_offset` → instruction with matching offset in same code unit. This is high-value flow structure.

#### 29) `graph.cpg2_bc_instructions`

```yaml
table_key: graph.cpg2_bc_instructions
pk: [repo, commit, rel_path, code_unit_id, offset]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  code_unit_id: BIGINT
  offset: INTEGER
  instr_id: BIGINT
  node_id: DECIMAL(38,0)
  opname: VARCHAR
  arg: INTEGER
  argrepr: VARCHAR
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
```

```python
def cpg2_bc_instructions(env, cpg2_anchor_map_bc_instructions, py_bc_instructions: pa.Table) -> pa.Table: ...
```

#### 30) `graph.cpg2_bc_exception_entries`

```yaml
table_key: graph.cpg2_bc_exception_entries
pk: [repo, commit, rel_path, code_unit_id, exc_entry_index]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  code_unit_id: BIGINT
  exc_entry_index: INTEGER
  node_id: DECIMAL(38,0)

  start_offset: INTEGER
  end_offset: INTEGER
  target_offset: INTEGER
  depth: INTEGER
  lasti: INTEGER

  start_instr_node_id: DECIMAL(38,0)     # nullable if join fails
  end_instr_node_id: DECIMAL(38,0)
  target_instr_node_id: DECIMAL(38,0)

  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)               # derived from start/end instr spans where possible
  join_succeeded: BOOLEAN
```

```python
def cpg2_bc_exception_entries(
    env,
    cpg2_bc_instructions,
    py_bc_exception_table: pa.Table,      # core.py_bc_exception_table
) -> pa.Table: ...
```

(The join + edge plan was explicit; also the plan suggests storing range endpoints as properties if you want to avoid awkward semantics.)

---

### 31–33) SCIP symbols/occurrences + GOIDs

SCIP tables + resolution xrefs exist, and GOIDs + crosswalk exist (Stage 4).

#### 31) `graph.cpg2_scip_symbols`

```yaml
table_key: graph.cpg2_scip_symbols
pk: [repo, commit, scip_symbol]
columns:
  repo: VARCHAR
  commit: VARCHAR
  scip_symbol: VARCHAR
  node_id: DECIMAL(38,0)
  display_name: VARCHAR
  symbol_kind: VARCHAR
  documentation: VARCHAR          # optional, possibly truncated
```

```python
def cpg2_scip_symbols(env, scip_symbol_information: pa.Table) -> pa.Table: ...
```

#### 32) `graph.cpg2_scip_occurrences`

```yaml
table_key: graph.cpg2_scip_occurrences
pk: [repo, commit, rel_path, producer, scip_occurrence_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  scip_occurrence_id: BIGINT
  node_id: DECIMAL(38,0)
  scip_symbol: VARCHAR
  scip_symbol_node_id: DECIMAL(38,0)
  role: VARCHAR                  # DEFINES/REFERS_TO/etc
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  syntax_node_node_id: DECIMAL(38,0)  # nullable via core.scip_occurrence_syntax_xref
  match_kind: VARCHAR
  candidate_count: INTEGER
```

```python
def cpg2_scip_occurrences(
    env,
    cpg2_scip_symbols,
    cpg2_anchor_map_scip_occurrences,
    scip_occurrence_syntax_xref: pa.Table,   # core.scip_occurrence_syntax_xref
    scip_occurrences: pa.Table,              # core.scip_occurrences
) -> pa.Table: ...
```

#### 33) `graph.cpg2_goids`

```yaml
table_key: graph.cpg2_goids
pk: [repo, commit, goid_h128]
columns:
  repo: VARCHAR
  commit: VARCHAR
  goid_h128: VARCHAR
  node_id: DECIMAL(38,0)
  module_name: VARCHAR
  qualname: VARCHAR
  goid_kind: VARCHAR
```

```python
def cpg2_goids(env, goids: pa.Table) -> pa.Table: ...
```

---

### 34–35) AST and CST nodes (optional but recommended in CPG2)

These are explicitly called out as upstream planes; including them makes your CPG2 truly “multi-representation,” not just “syntax facts + TS tags.”

#### 34) `graph.cpg2_ast_nodes`

```yaml
table_key: graph.cpg2_ast_nodes
pk: [repo, commit, rel_path, ast_node_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  ast_node_id: BIGINT
  node_id: DECIMAL(38,0)
  ast_type: VARCHAR
  parent_ast_node_id: BIGINT
  parent_node_id: DECIMAL(38,0)
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
```

```python
def cpg2_ast_nodes(env, cpg2_anchor_map_ast_nodes, ast_nodes: pa.Table) -> pa.Table: ...
```

#### 35) `graph.cpg2_cst_nodes`

```yaml
table_key: graph.cpg2_cst_nodes
pk: [repo, commit, rel_path, cst_node_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  cst_node_id: BIGINT
  node_id: DECIMAL(38,0)
  cst_type: VARCHAR
  parent_cst_node_id: BIGINT
  parent_node_id: DECIMAL(38,0)
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
```

```python
def cpg2_cst_nodes(env, cpg2_anchor_map_cst_nodes, cst_nodes: pa.Table) -> pa.Table: ...
```

---

## Layer 3 — Graph assembly tables

This is where everything becomes a property graph.

### 36) `graph.cpg2_nodes`

This is a *union-index* over all node-bearing plane tables. It makes traversal/query consistent.

```yaml
table_key: graph.cpg2_nodes
pk: [repo, commit, node_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  node_id: DECIMAL(38,0)
  node_kind: VARCHAR              # e.g. SYNTAX_DEF | TS_NODE | SCIP_SYMBOL | GOID | BC_INSTR | ANCHOR | ...
  file_id: DECIMAL(38,0)
  anchor_id: DECIMAL(38,0)
  display_name: VARCHAR
  language: VARCHAR
  origin_table_key: VARCHAR
  origin_pk_msgpack: BLOB         # encode_payload(pk dict)
```

```python
def cpg2_nodes(
    env,
    cpg2_files,
    cpg2_anchors,
    cpg2_syntax_scopes,
    cpg2_syntax_spans,
    cpg2_syntax_defs,
    cpg2_syntax_refs,
    cpg2_syntax_calls,
    cpg2_syntax_imports,
    cpg2_syntax_call_args,
    cpg2_syntax_func_params,
    cpg2_ts_nodes,
    cpg2_ts_captures,
    cpg2_ts_parse_errors,
    cpg2_docstrings,
    cpg2_inspect_objects,
    cpg2_inspect_annotations_kv,
    cpg2_bc_instructions,
    cpg2_bc_exception_entries,
    cpg2_scip_symbols,
    cpg2_scip_occurrences,
    cpg2_goids,
    cpg2_ast_nodes,
    cpg2_cst_nodes,
) -> pa.Table: ...
```

---

### 37) `graph.cpg2_edges`

This is the main semantic connectivity table.

```yaml
table_key: graph.cpg2_edges
pk: [repo, commit, edge_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  edge_id: DECIMAL(38,0)           # stable_decimal_id({"table_key":"graph.cpg2_edges","pk":{src,dst,kind,layer,ordinal}})
  src_node_id: DECIMAL(38,0)
  dst_node_id: DECIMAL(38,0)
  edge_kind: VARCHAR               # IN_SCOPE | HAS_SPAN | ANCHORS | RESOLVES_SYMBOL | TS_CHILD | WELDS_TO | ...
  edge_layer: VARCHAR              # SYNTAX | SYMBOL | TS | DOC | INSPECT | FLOW | ...
  ordinal: INTEGER
  match_kind: VARCHAR              # nullable
  candidate_count: INTEGER         # nullable
  confidence: DOUBLE               # nullable
  via_table_key: VARCHAR           # provenance (e.g. core.ts_edges)
  via_pk_msgpack: BLOB             # encode_payload(upstream pk dict)
```

**Hamilton node signature**

```python
def cpg2_edges(
    env,
    cpg2_nodes: pa.Table,
    # plane tables used to generate edges deterministically:
    cpg2_syntax_scopes: pa.Table,
    cpg2_syntax_defs: pa.Table,
    cpg2_syntax_refs: pa.Table,
    cpg2_syntax_calls: pa.Table,
    cpg2_syntax_imports: pa.Table,
    cpg2_syntax_call_args: pa.Table,
    cpg2_syntax_func_params: pa.Table,
    cpg2_scip_occurrences: pa.Table,
    cpg2_ts_edges: pa.Table,
    cpg2_ts_weld: pa.Table,
    cpg2_ts_captures: pa.Table,
    cpg2_docstrings: pa.Table,
    cpg2_inspect_annotations_kv: pa.Table,
    cpg2_bc_exception_entries: pa.Table,
    cpg2__options: "CPG2Options",
) -> pa.Table: ...
```

Edge families are directly based on the scope expansion you already detailed:

* Syntax scope tree edges (`PARENT_SCOPE`)
* Fact anchoring edges (`IN_SCOPE`, `HAS_SPAN`, `ANCHORS`, `ARG_OF`, `PARAM_OF`)
* Resolution edges (`RESOLVES_SYMBOL`, `RESOLVES_TO`/GOID)
* TS structural edges + weld (`TS_CHILD`, `WELDS_TO_SYNTAX`)
* Docstrings + inspect overlay edges
* Bytecode exception-flow edges (`EXC_HANDLER_TARGET`, range endpoints as properties)

---

## Layer 4 — Cross-plane “best mapping” tables

### 38) `graph.cpg2_node_xrefs`

This is the ranked mapping API: “for this node, what’s the best SCIP symbol / GOID / TS node / AST node / …?”

```yaml
table_key: graph.cpg2_node_xrefs
pk: [repo, commit, node_id, xref_kind, xref_rank]
columns:
  repo: VARCHAR
  commit: VARCHAR
  node_id: DECIMAL(38,0)
  xref_kind: VARCHAR              # SCIP_SYMBOL | GOID | TS_NODE | AST_NODE | CST_NODE | SYNTAX_NODE | INSPECT_OBJECT | ...
  xref_rank: INTEGER              # 0 best
  target_node_id: DECIMAL(38,0)
  confidence: DOUBLE
  match_kind: VARCHAR
  candidate_count: INTEGER
  via_table_key: VARCHAR
  via_pk_msgpack: BLOB
```

```python
def cpg2_node_xrefs(
    env,
    cpg2_nodes: pa.Table,
    cpg2_edges: pa.Table,
    # (optional) include resolved sources directly for richer ranking features:
    syntax_defs_resolved: pa.Table,
    syntax_refs_resolved: pa.Table,
    syntax_calls_resolved: pa.Table,
    syntax_imports_resolved: pa.Table,
    ts_syntax_node_xref: pa.Table,
    scip_occurrence_syntax_xref: pa.Table,
    scip_symbol_goid_xref: pa.Table,
    cpg2__options: "CPG2Options",
) -> pa.Table: ...
```

This is the generalized version of the earlier “xref precedence + deterministic capping” section, but now in the CPG2 redesign. (You already laid out the need for ranked xrefs + deterministic caps.)

---

### 39) `graph.cpg2_symbol_xref`

This is the “what is this thing?” crosswalk between SCIP, GOID, inspect objects, and syntax fact nodes (high ROI).

```yaml
table_key: graph.cpg2_symbol_xref
pk: [repo, commit, scip_symbol, goid_h128, inspect_object_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  scip_symbol: VARCHAR
  scip_symbol_node_id: DECIMAL(38,0)
  goid_h128: VARCHAR
  goid_node_id: DECIMAL(38,0)
  inspect_object_id: BIGINT
  inspect_object_node_id: DECIMAL(38,0)

  # optional backlinks to “where observed”
  def_node_id: DECIMAL(38,0)
  ref_node_id: DECIMAL(38,0)
  call_node_id: DECIMAL(38,0)

  confidence: DOUBLE
  extras_msgpack: BLOB
```

```python
def cpg2_symbol_xref(
    env,
    cpg2_scip_symbols,
    cpg2_goids,
    cpg2_inspect_objects,
    cpg2_syntax_defs,
    cpg2_syntax_refs,
    cpg2_syntax_calls,
    scip_symbol_goid_xref: pa.Table,
    goid_crosswalk: pa.Table,
    cpg2__options: "CPG2Options",
) -> pa.Table: ...
```

This mirrors the earlier proposed “symbol_xref” table shape and inputs.

---

## Layer 5 — Coverage / quality tables

These are explicitly part of the scope expansion (TS weld coverage, syntax resolution coverage, bytecode exception resolution coverage, inspect annotation coverage).

### 40) `analytics.cpg2_coverage_ts_weld`

```yaml
table_key: analytics.cpg2_coverage_ts_weld
pk: [repo, commit, rel_path, producer, language]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  language: VARCHAR
  ts_node_count: BIGINT
  mapped_to_syntax_count: BIGINT
  coverage_ratio: DOUBLE
  match_kind_breakdown_msgpack: BLOB
```

```python
def cpg2_coverage_ts_weld(env, ts_weld_coverage: pa.Table, ts_syntax_node_xref: pa.Table) -> pa.Table: ...
```

### 41) `analytics.cpg2_coverage_syntax_resolution`

```yaml
table_key: analytics.cpg2_coverage_syntax_resolution
pk: [repo, commit, rel_path, producer]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  producer: VARCHAR
  defs_total: BIGINT
  defs_resolved_to_symbol: BIGINT
  defs_resolved_to_goid: BIGINT
  refs_total: BIGINT
  refs_resolved_to_symbol: BIGINT
  refs_resolved_to_goid: BIGINT
  calls_total: BIGINT
  calls_resolved_to_symbol: BIGINT
  calls_resolved_to_goid: BIGINT
  match_kind_quantiles_msgpack: BLOB
```

```python
def cpg2_coverage_syntax_resolution(env, syntax_defs_resolved, syntax_refs_resolved, syntax_calls_resolved, syntax_imports_resolved) -> pa.Table: ...
```

### 42) `analytics.cpg2_coverage_bytecode_exceptions`

```yaml
table_key: analytics.cpg2_coverage_bytecode_exceptions
pk: [repo, commit, rel_path, code_unit_id]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  code_unit_id: BIGINT
  exception_entry_count: BIGINT
  entries_with_resolved_target_instr: BIGINT
  pct_resolved: DOUBLE
  parse_failures: BIGINT
```

```python
def cpg2_coverage_bytecode_exceptions(env, py_bc_exception_table: pa.Table, py_bc_instructions: pa.Table) -> pa.Table: ...
```

### 43) `analytics.cpg2_coverage_inspect_annotations`

```yaml
table_key: analytics.cpg2_coverage_inspect_annotations
pk: [repo, commit, mode]
columns:
  repo: VARCHAR
  commit: VARCHAR
  mode: VARCHAR
  inspect_objects: BIGINT
  objects_with_annotations: BIGINT
  annotation_kv_rows: BIGINT
  pct_objects_annotated: DOUBLE
```

```python
def cpg2_coverage_inspect_annotations(env, py_inspect_objects: pa.Table, py_inspect_annotations_kv: pa.Table) -> pa.Table: ...
```

### 44) `analytics.cpg2_coverage_anchor_density`

This one is new (but extremely useful): “how anchorable is each plane per file?”

```yaml
table_key: analytics.cpg2_coverage_anchor_density
pk: [repo, commit, rel_path, plane]
columns:
  repo: VARCHAR
  commit: VARCHAR
  rel_path: VARCHAR
  plane: VARCHAR                 # SYNTAX_SPANS | SYNTAX_NODES | TS_NODES | AST_NODES | CST_NODES | SCIP_OCC | BC_INSTR | DOCSTRINGS
  total_rows: BIGINT
  rows_with_anchor: BIGINT
  pct_anchored: DOUBLE
```

```python
def cpg2_coverage_anchor_density(
    env,
    cpg2_anchor_map_syntax_spans,
    cpg2_anchor_map_syntax_nodes,
    cpg2_anchor_map_ts_nodes,
    cpg2_anchor_map_ast_nodes,
    cpg2_anchor_map_cst_nodes,
    cpg2_anchor_map_scip_occurrences,
    cpg2_anchor_map_bc_instructions,
    cpg2_anchor_map_docstrings,
) -> pa.Table: ...
```

---

# Hamilton module layout (build-only) and node organization

To make this implementation clean, I’d structure:

* `graphs/cpg2/options.py` → `cpg2__options`
* `graphs/cpg2/anchors.py` → `cpg2_files`, `cpg2_anchors`, all `cpg2_anchor_map_*`
* `graphs/cpg2/planes/syntax.py` → all `cpg2_syntax_*`
* `graphs/cpg2/planes/ts.py` → all `cpg2_ts_*`
* `graphs/cpg2/planes/inspect.py` → `cpg2_docstrings`, `cpg2_inspect_*`
* `graphs/cpg2/planes/bytecode.py` → `cpg2_bc_*`
* `graphs/cpg2/planes/symbols.py` → `cpg2_scip_*`, `cpg2_goids`
* `graphs/cpg2/planes/ast_cst.py` → `cpg2_ast_nodes`, `cpg2_cst_nodes`
* `graphs/cpg2/graph.py` → `cpg2_nodes`, `cpg2_edges`
* `graphs/cpg2/xrefs.py` → `cpg2_node_xrefs`, `cpg2_symbol_xref`
* `graphs/cpg2/coverage.py` → `cpg2_coverage_*`

Everything here stays strictly in build and is consistent with the hard separation requirement (Arrow in/out, no DuckDB in build).

---

According to a document from **January 5, 2026**, the “best” way to enrich your CPG is to treat the graph outputs as **Arrow datasets** produced by `src/codeintel/build` (Hamilton), and keep DuckDB loading/querying as an optional adapter in `src/codeintel/storage`—i.e., keep the dependency direction clean and one-way. 

You also asked me *not* to assume what CFG/DFG/CDG/PDG tables you “already likely have,” so I reviewed the repo you provided in `CodeIntel_CPG_implemented.zip`. Concretely:

* You **do** currently define these graph outputs (schema `graph`) in `src/codeintel/core/schemas/output_registry.py`:

  * `graph.cfg_blocks`, `graph.cfg_edges`
  * `graph.dfg_edges`
  * `graph.cdg_edges`
  * `graph.pdg_edges`
* And you compute them via:

  * `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py` (AST-based CFG + reaching-defs DFG)
  * `src/codeintel/build/hamilton/native/graphs/cdg.py` (CDG via postdominators)
  * `src/codeintel/build/hamilton/native/graphs/pdg.py` (PDG union)
* Separately, you already ingest a **much stronger “bytecode flow plane”** in `core.*`:

  * `core.py_bc_blocks`, `core.py_bc_cfg_edges`, `core.py_bc_exception_table`
  * `core.py_bc_defuse_events`, `core.py_bc_instructions`, `core.py_bc_code_units`

Given your new directive (“we’re still designing; optimize for best-in-class; compatibility not required”), my recommendation is:

## Redesign principle for CFG/DFG/CDG/PDG in CPG2

### Make bytecode the canonical *flow IR*, and make source mappings first-class

Python AST/CST/Tree-sitter are excellent for:

* anchors/spans
* lexical structure, names, scopes
* “semantic facts” (defs/refs/calls/imports)
* query tags and surface structure

…but they are **not** the best source of truth for control-flow and def-use in Python, because:

* Python’s compilation choices (short-circuiting, exception edges, finally, etc.) are captured precisely in bytecode + exception tables.
* Your current AST-CFG builder is necessarily heuristic (and it currently treats `try` as “just visit children,” i.e., it does not encode exception flow).
* Your current DFG is block-level reaching definitions over `ast.Name` loads/stores, which is too lossy for best-in-class PDG/DDG.

So the “best-in-class” redesign is:

1. **CFG2**: derived primarily from `core.py_bc_blocks` + `core.py_bc_cfg_edges` (+ exception table already baked into those edges).
2. **DFG2**: derived from `core.py_bc_defuse_events` + CFG2 (and refined by symtable/syntax facts for spaces/bindings).
3. **CDG2**: derived from CFG2 postdominators (block-level) and optionally *lifted* to instruction/span level.
4. **PDG2**: build a *canonical* PDG at instruction/event granularity, then provide **folded projections** to source spans/statements for queryability.

Everything remains **build-only**: Hamilton nodes read upstream `core.*` Arrow tables and output new `graph.*` Arrow tables. No DuckDB imports, no `storage` imports. This stays aligned with your separation constraint. 

---

# CPG2 flow tables: concrete catalog (CFG2/DFG2/CDG2/PDG2)

Below is the **explicit “table catalog”** for the flow plane, written in the style of your existing `output_registry.py` TableSchema definitions.

I’m intentionally **not** reusing your current `graph.cfg_*`/`graph.dfg_*` schemas, because you explicitly said compatibility is not required. The new plane is `*2`-suffixed, canonical, bytecode-driven, and span-mappable.

---

## 1) `graph.cfg2_procs`

A normalized procedure/code-unit inventory for flow analysis and joins.

**Why:** every downstream flow table should key by a stable “procedure” identifier, even if GOID mapping is partial.

**Derived from:** `core.py_bc_code_units` (+ optional mapping to `core.goids` via span alignment using `core.file_line_index`).

```python
CFG2_PROCS_TABLE = TableSchema(
    schema="graph",
    name="cfg2_procs",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),

        # Canonical identity for flow
        Column("proc_id", "DECIMAL(38,0)", nullable=False),

        # Bytecode identity
        Column("code_unit_id", "VARCHAR", nullable=False),
        Column("parent_code_unit_id", "VARCHAR"),

        # Optional semantic identity (best-effort)
        Column("goid_h128", "DECIMAL(38,0)"),

        # Helpful labels
        Column("qualpath", "VARCHAR"),
        Column("co_name", "VARCHAR"),
        Column("co_qualname", "VARCHAR"),
        Column("kind", "VARCHAR"),

        # Spans (byte) to support anchoring / mapping
        Column("span_start_byte", "BIGINT"),
        Column("span_end_byte", "BIGINT"),

        Column("extras_json", "BLOB"),
    ],
    primary_key=("repo", "commit", "proc_id"),
    indexes=(
        Index("idx_graph_cfg2_procs_path", ("rel_path",)),
        Index("idx_graph_cfg2_procs_code_unit", ("code_unit_id",)),
        Index("idx_graph_cfg2_procs_goid", ("goid_h128",)),
    ),
    description="Canonical procedure inventory for flow analysis (bytecode-first, GOID-mappable).",
)
```

**Stable ID rule (recommended):**

* `proc_id = stable_decimal_id({"t":"cfg2_proc","repo":repo,"commit":commit,"rel_path":rel_path,"code_unit_id":code_unit_id})`

---

## 2) `graph.cfg2_blocks`

Canonical CFG block nodes (bytecode blocks), span-aware.

**Derived from:** `core.py_bc_blocks` joined to `graph.cfg2_procs`.

```python
CFG2_BLOCKS_TABLE = TableSchema(
    schema="graph",
    name="cfg2_blocks",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),

        Column("proc_id", "DECIMAL(38,0)", nullable=False),
        Column("code_unit_id", "VARCHAR", nullable=False),

        Column("block_id", "DECIMAL(38,0)", nullable=False),
        Column("block_key", "VARCHAR", nullable=False),  # original core.py_bc_blocks.block_id
        Column("block_index", "INTEGER"),               # optional: stable ordering within proc
        Column("kind", "VARCHAR"),                      # from core (entry/exit/handler/etc)

        # Bytecode offsets
        Column("start_offset", "INTEGER"),
        Column("end_offset", "INTEGER"),
        Column("start_label", "VARCHAR"),

        # Instruction range (index-based) for fast joins
        Column("first_instr_index", "INTEGER"),
        Column("last_instr_index", "INTEGER"),

        # Source anchoring
        Column("span_start_byte", "BIGINT"),
        Column("span_end_byte", "BIGINT"),

        # Graph metrics
        Column("in_degree", "INTEGER", nullable=False),
        Column("out_degree", "INTEGER", nullable=False),

        Column("extras_json", "BLOB"),
    ],
    primary_key=("repo", "commit", "proc_id", "block_id"),
    indexes=(
        Index("idx_graph_cfg2_blocks_proc", ("proc_id",)),
        Index("idx_graph_cfg2_blocks_code_unit", ("code_unit_id",)),
    ),
    description="Bytecode-derived CFG blocks (canonical flow block nodes).",
)
```

**Stable ID rule (recommended):**

* `block_id = stable_decimal_id({"t":"cfg2_block","proc_id":proc_id,"block_key":block_key})`

---

## 3) `graph.cfg2_edges`

Canonical CFG edges between bytecode blocks, including exception edges.

**Derived from:** `core.py_bc_cfg_edges` joined to `graph.cfg2_blocks` (for numeric ids).

```python
CFG2_EDGES_TABLE = TableSchema(
    schema="graph",
    name="cfg2_edges",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),

        Column("proc_id", "DECIMAL(38,0)", nullable=False),
        Column("code_unit_id", "VARCHAR", nullable=False),

        Column("edge_id", "DECIMAL(38,0)", nullable=False),

        Column("src_block_id", "DECIMAL(38,0)", nullable=False),
        Column("dst_block_id", "DECIMAL(38,0)", nullable=False),

        # Normalized kinds (your canonical vocabulary)
        Column("kind", "VARCHAR", nullable=False),

        # Branch/exception metadata from core.py_bc_cfg_edges
        Column("cond_instr_id", "VARCHAR"),
        Column("exc_entry_index", "INTEGER"),

        # Deterministic ordering for multi-edges
        Column("ordinal", "INTEGER", nullable=False),

        Column("extras_json", "BLOB"),
    ],
    primary_key=("repo", "commit", "proc_id", "edge_id"),
    indexes=(
        Index("idx_graph_cfg2_edges_proc", ("proc_id",)),
        Index("idx_graph_cfg2_edges_src", ("src_block_id",)),
        Index("idx_graph_cfg2_edges_dst", ("dst_block_id",)),
    ),
    description="Bytecode-derived CFG edges (canonical, exception-aware).",
)
```

**Notes on “kind” normalization**

* Map `core.py_bc_cfg_edges.kind` into a small canonical set, e.g.:

  * `FALLTHROUGH`, `BR_TRUE`, `BR_FALSE`, `JUMP`, `RETURN`, `RAISE`, `EXC_HANDLER`, `FINALLY`, `BACKEDGE`, …
* Preserve raw/original info in `extras_json`.

---

## 4) `graph.cfg2_block_instrs`

Explicit ordered membership of instructions in blocks.

**Why:** essential for instruction-level lifting and PDG folding without “LIST columns” or JSON blobs.

**Derived from:** `core.py_bc_instructions` + `core.py_bc_blocks` (using `first_instr_index/last_instr_index`) + `graph.cfg2_blocks`.

```python
CFG2_BLOCK_INSTRS_TABLE = TableSchema(
    schema="graph",
    name="cfg2_block_instrs",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),

        Column("proc_id", "DECIMAL(38,0)", nullable=False),
        Column("block_id", "DECIMAL(38,0)", nullable=False),

        Column("instr_index", "INTEGER", nullable=False),
        Column("instr_id", "VARCHAR", nullable=False),

        Column("ordinal", "INTEGER", nullable=False),
    ],
    primary_key=("repo", "commit", "proc_id", "block_id", "instr_index"),
    indexes=(Index("idx_graph_cfg2_block_instrs_instr", ("instr_id",)),),
    description="Block→instruction ordered membership for lifting CFG/DFG to instruction granularity.",
)
```

---

## 5) `graph.dfg2_events`

Normalized def/use events with block + instruction + span context.

**Derived from:** `core.py_bc_defuse_events` + join to `core.py_bc_instructions` (for spans) + join to `graph.cfg2_block_instrs` (for block membership).

```python
DFG2_EVENTS_TABLE = TableSchema(
    schema="graph",
    name="dfg2_events",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),

        Column("proc_id", "DECIMAL(38,0)", nullable=False),
        Column("code_unit_id", "VARCHAR", nullable=False),

        Column("event_id", "VARCHAR", nullable=False),
        Column("instr_id", "VARCHAR", nullable=False),
        Column("instr_index", "INTEGER"),

        # Attach to CFG blocks (optional but strongly recommended)
        Column("block_id", "DECIMAL(38,0)"),

        # Event classification
        Column("event_kind", "VARCHAR", nullable=False),  # DEF / USE / KILL / etc
        Column("space", "VARCHAR"),                       # LOCAL / GLOBAL / CELL / FREE / ATTR / ...
        Column("name", "VARCHAR"),                        # symbol name where applicable

        # Span anchoring (from instruction span; later fold to source spans)
        Column("span_start_byte", "BIGINT"),
        Column("span_end_byte", "BIGINT"),

        Column("confidence", "DOUBLE"),
        Column("extras_json", "BLOB"),
    ],
    primary_key=("repo", "commit", "proc_id", "event_id"),
    indexes=(
        Index("idx_graph_dfg2_events_proc", ("proc_id",)),
        Index("idx_graph_dfg2_events_instr", ("instr_id",)),
        Index("idx_graph_dfg2_events_space_name", ("space", "name")),
    ),
    description="Bytecode def/use events normalized for DFG construction and lifting to spans.",
)
```

---

## 6) `graph.dfg2_edges`

Canonical def→use reachability edges at event granularity.

**Derived from:** `graph.dfg2_events` + `graph.cfg2_edges` (+ optional symtable/syntax refinements).

```python
DFG2_EDGES_TABLE = TableSchema(
    schema="graph",
    name="dfg2_edges",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),

        Column("proc_id", "DECIMAL(38,0)", nullable=False),
        Column("code_unit_id", "VARCHAR", nullable=False),

        Column("edge_id", "DECIMAL(38,0)", nullable=False),

        Column("src_event_id", "VARCHAR", nullable=False),  # DEF
        Column("dst_event_id", "VARCHAR", nullable=False),  # USE

        Column("space", "VARCHAR"),
        Column("name", "VARCHAR"),

        Column("kind", "VARCHAR", nullable=False),          # DEF_USE / MAY_DEF_USE / PARAM_USE / ...
        Column("via_phi", "BOOLEAN", nullable=False),
        Column("phi_group_id", "DECIMAL(38,0)"),            # optional, if you materialize φ groups

        Column("confidence", "DOUBLE"),
        Column("ordinal", "INTEGER", nullable=False),

        Column("extras_json", "BLOB"),
    ],
    primary_key=("repo", "commit", "proc_id", "edge_id"),
    indexes=(
        Index("idx_graph_dfg2_edges_proc", ("proc_id",)),
        Index("idx_graph_dfg2_edges_src", ("src_event_id",)),
        Index("idx_graph_dfg2_edges_dst", ("dst_event_id",)),
    ),
    description="Def→use dataflow edges between def/use events (bytecode-first).",
)
```

---

## 7) `graph.cdg2_edges`

Control dependence edges derived from CFG2 postdominators.

**Derived from:** `graph.cfg2_edges` + `graph.cfg2_blocks`.

```python
CDG2_EDGES_TABLE = TableSchema(
    schema="graph",
    name="cdg2_edges",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),

        Column("proc_id", "DECIMAL(38,0)", nullable=False),
        Column("code_unit_id", "VARCHAR", nullable=False),

        Column("edge_id", "DECIMAL(38,0)", nullable=False),

        Column("controller_block_id", "DECIMAL(38,0)", nullable=False),
        Column("dependent_block_id", "DECIMAL(38,0)", nullable=False),

        # Preserve your current “via successor” idea, but in numeric ids
        Column("via_succ_block_id", "DECIMAL(38,0)", nullable=False),

        # Canonical kinds
        Column("kind", "VARCHAR", nullable=False),      # CONTROL / EXCEPTION_CONTROL / ...

        # If you can attach a predicate instruction (recommended)
        Column("predicate_instr_id", "VARCHAR"),

        Column("ordinal", "INTEGER", nullable=False),
        Column("extras_json", "BLOB"),
    ],
    primary_key=("repo", "commit", "proc_id", "edge_id"),
    indexes=(
        Index("idx_graph_cdg2_edges_proc", ("proc_id",)),
        Index("idx_graph_cdg2_edges_controller", ("controller_block_id",)),
        Index("idx_graph_cdg2_edges_dependent", ("dependent_block_id",)),
    ),
    description="Control dependence edges derived from CFG2 postdominators (bytecode CFG).",
)
```

---

## 8) `graph.pdg2_edges_instr`

Program dependence edges at *instruction/event* granularity (canonical PDG).

**Derived from:** `graph.dfg2_edges` + `graph.cdg2_edges` lifted to instruction/event points.

```python
PDG2_EDGES_INSTR_TABLE = TableSchema(
    schema="graph",
    name="pdg2_edges_instr",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),

        Column("proc_id", "DECIMAL(38,0)", nullable=False),

        Column("edge_id", "DECIMAL(38,0)", nullable=False),

        # PDG endpoints are “program points”; in v1 keep it concrete:
        # - data deps: def-event -> use-event
        # - control deps: controller-block -> dependent-block (or predicate instr -> instr)
        Column("src_kind", "VARCHAR", nullable=False),   # EVENT / BLOCK / INSTR
        Column("src_id", "VARCHAR", nullable=False),     # event_id or instr_id or stringified numeric block_id
        Column("dst_kind", "VARCHAR", nullable=False),
        Column("dst_id", "VARCHAR", nullable=False),

        Column("layer", "VARCHAR", nullable=False),      # DATA / CONTROL
        Column("kind", "VARCHAR", nullable=False),       # e.g. DEF_USE / CONTROL

        # Optional dataflow payload
        Column("space", "VARCHAR"),
        Column("name", "VARCHAR"),

        Column("confidence", "DOUBLE"),
        Column("ordinal", "INTEGER", nullable=False),
        Column("extras_json", "BLOB"),
    ],
    primary_key=("repo", "commit", "proc_id", "edge_id"),
    indexes=(
        Index("idx_graph_pdg2_edges_instr_proc", ("proc_id",)),
        Index("idx_graph_pdg2_edges_instr_layer", ("layer",)),
    ),
    description="Canonical PDG edges at instruction/event granularity (union of DFG2 + CDG2).",
)
```

> If you’d rather keep IDs strongly typed, replace `(src_kind, src_id)` with nullable columns:
>
> * `src_event_id`, `src_block_id`, `src_instr_id`, same for dst.
>   This is usually nicer in DuckDB/SQL later.

---

## 9) `graph.pdg2_edges_span`

Folded PDG for queryability: span→span dependencies.

**Why:** most graph queries want “this statement depends on that statement,” not raw instruction/event ids.

**Derived from:** `graph.pdg2_edges_instr` + span resolution that maps each instr/event/block to a canonical **source span** (via `core.py_bc_instructions.span_*` and your syntax/AST spans).

```python
PDG2_EDGES_SPAN_TABLE = TableSchema(
    schema="graph",
    name="pdg2_edges_span",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),

        Column("proc_id", "DECIMAL(38,0)", nullable=False),

        Column("edge_id", "DECIMAL(38,0)", nullable=False),

        # Canonical source anchors (byte spans)
        Column("src_start_byte", "BIGINT", nullable=False),
        Column("src_end_byte", "BIGINT", nullable=False),
        Column("dst_start_byte", "BIGINT", nullable=False),
        Column("dst_end_byte", "BIGINT", nullable=False),

        Column("layer", "VARCHAR", nullable=False),     # DATA / CONTROL
        Column("kind", "VARCHAR", nullable=False),

        Column("space", "VARCHAR"),
        Column("name", "VARCHAR"),

        # Aggregation metadata
        Column("supporting_edge_count", "INTEGER", nullable=False),
        Column("confidence", "DOUBLE"),
        Column("ordinal", "INTEGER", nullable=False),
        Column("extras_json", "BLOB"),
    ],
    primary_key=("repo", "commit", "proc_id", "edge_id"),
    indexes=(Index("idx_graph_pdg2_edges_span_proc", ("proc_id",)),),
    description="Folded PDG edges between source spans (query-friendly projection).",
)
```

This table is the one that will make downstream use feel “best-in-class.”

---

# Hamilton node signatures and dependency order

This is the **clean Hamilton ordering** so the flow plane composes and stays decoupled:

### Anchor-ish prerequisites (already in your pipeline)

* `core.py_bc_code_units`
* `core.py_bc_blocks`
* `core.py_bc_cfg_edges`
* `core.py_bc_instructions`
* `core.py_bc_defuse_events`
* (optional but recommended for GOID mapping) `core.goids`, `core.file_line_index`

### Flow plane nodes

```python
# 1) Procedures
def cfg2_procs(
    env: BuildEnv,
    q__core__py_bc_code_units: InferableTabularInput,
    q__core__goids: InferableTabularInput,
    q__core__file_line_index: InferableTabularInput,
) -> InferableTabularInput: ...

# 2) CFG blocks/edges
def cfg2_blocks(
    env: BuildEnv,
    cfg2_procs: InferableTabularInput,
    q__core__py_bc_blocks: InferableTabularInput,
) -> InferableTabularInput: ...

def cfg2_edges(
    env: BuildEnv,
    cfg2_procs: InferableTabularInput,
    cfg2_blocks: InferableTabularInput,
    q__core__py_bc_cfg_edges: InferableTabularInput,
) -> InferableTabularInput: ...

def cfg2_block_instrs(
    env: BuildEnv,
    cfg2_blocks: InferableTabularInput,
    q__core__py_bc_instructions: InferableTabularInput,
) -> InferableTabularInput: ...

# 3) DFG events/edges
def dfg2_events(
    env: BuildEnv,
    cfg2_procs: InferableTabularInput,
    cfg2_block_instrs: InferableTabularInput,
    q__core__py_bc_defuse_events: InferableTabularInput,
    q__core__py_bc_instructions: InferableTabularInput,
) -> InferableTabularInput: ...

def dfg2_edges(
    env: BuildEnv,
    cfg2_edges: InferableTabularInput,
    dfg2_events: InferableTabularInput,
) -> InferableTabularInput: ...

# 4) CDG
def cdg2_edges(
    env: BuildEnv,
    cfg2_edges: InferableTabularInput,
) -> InferableTabularInput: ...

# 5) PDG
def pdg2_edges_instr(
    env: BuildEnv,
    dfg2_edges: InferableTabularInput,
    cdg2_edges: InferableTabularInput,
) -> InferableTabularInput: ...

def pdg2_edges_span(
    env: BuildEnv,
    pdg2_edges_instr: InferableTabularInput,
    q__core__py_bc_instructions: InferableTabularInput,
    q__core__syntax_spans: InferableTabularInput,
    # optionally: q__core__syntax_nodes / q__core__ast_nodes to improve folding fidelity
) -> InferableTabularInput: ...
```

All of these nodes live in `src/codeintel/build/hamilton/native/graphs/flow2/` (or similar) and only use:

* `pyarrow` / `pyarrow.compute`
* existing build utilities (aligning, joining, stable ids)
* `codeintel.core.*` schemas/types

No storage imports. This stays consistent with the constraint highlighted in the improvement doc. 

---

# How this aligns to (and improves upon) your current CFG/DFG/CDG/PDG outputs

Your current outputs are:

* CFG: `graph.cfg_blocks`, `graph.cfg_edges` (AST-block-based)
* DFG: `graph.dfg_edges` (block-level reaching defs)
* CDG: `graph.cdg_edges` (computed over the above CFG)
* PDG: `graph.pdg_edges` (union)

These are a fine starting point, but they’re structurally limited for “best-in-class” because they:

* don’t encode exception edges in the CFG from real Python execution semantics,
* compute dataflow at *block granularity* over `ast.Name` only,
* lose instruction-level and event-level precision needed for serious slicing/taint/dep queries.

The CFG2/DFG2/CDG2/PDG2 catalog above fixes this by:

* using the bytecode CFG that already incorporates exception-table behavior,
* using explicit def/use events as the basis for DFG,
* producing a canonical PDG at event/instruction granularity and a folded span projection.

---

# Integration with your CPG2 assembly

Once these tables exist, “CPG2 graph assembly” becomes straightforward:

* Either:

  * emit `CFG_NEXT/CFG_TRUE/CFG_FALSE/CFG_EXC` edges in `cpg2_edges` from `cfg2_edges`
  * emit `DEF_USE` edges from `dfg2_edges`
  * emit `CONTROL_DEP` edges from `cdg2_edges`
  * emit `PDG_*` edges from `pdg2_edges_span` (for query-friendly endpoints)

* Or keep them as separate plane tables and let downstream consumers join as needed.

(Your earlier enrichment plan already calls out “FLOW” edge kinds for CFG/DFG/exception structure. This flow2 catalog is the “serious” version of that layer.) 

---

## References you attached

* Hamilton lineage doc: 
* Scope expansion / compute improvement plan:  

---

If you want, I can now do the next “most concrete” step for this flow plane (still build-only): **draft the exact compute logic** for:

* `dfg2_edges` (event-level reaching defs with deterministic φ-grouping),
* `cdg2_edges` (postdom with a fast bitset impl rather than NetworkX),
* `pdg2_edges_span` (SpanResolver folding rules + deterministic aggregation/ordinals),

…and I can shape it to reuse your existing stable-id helpers (`stable_decimal_id`) and Arrow join utilities so it drops cleanly into `src/codeintel/build` without touching `storage`.

Below is the “next-most-concrete” step for the **flow plane** in your **CPG2 redesign**, written so it can be implemented **entirely inside `src/codeintel/build`** (no `src/codeintel/storage` imports, no DuckDB assumptions), while reusing your existing:

* `stable_decimal_id()` (from `codeintel.build.graphs.assembly.ids`)
* Arrow-first helpers (`tabular_to_table`, `align_table_to_contract`, `dedupe_table_for_table`, `arrow_join_tables`, etc.)
* `SpanResolver` (from `codeintel.core.intervals.span_resolver`)

I’m going to describe the **exact compute logic** for:

1. `graph.dfg2_edges` — **event-level reaching defs** with **deterministic φ-grouping (SSA-like)**
2. `graph.cdg2_edges` — **control dependence** via **postdominators** using a **fast bitset**
3. `graph.pdg2_edges_span` — span-level PDG edges using **SpanResolver folding rules** + **deterministic aggregation/ordinals**

I’ll keep the description **implementation-ready**, including recommended function signatures, deterministic ordering rules, and stable ID payloads.

---

# A. `graph.dfg2_edges`: event-level reaching defs with deterministic φ-grouping

## A1) Goal

Build a **near-SSA** DFG over bytecode def/use events:

* Each variable has **at most one current reaching def** at any program point (after φ insertion).
* Join points introduce **φ defs** deterministically:

  * one φ per `(code_unit_id, block_id, var_key)` when needed
* Each `USE` gets exactly **one** incoming def edge (or one synthetic `UNDEF`/`EXTERNAL`/`PARAM` def), which is how we **cap cardinality deterministically**.

This is the most important shift from your current reaching-defs logic (which emits edges from *all* reaching defs).

---

## A2) Inputs (Arrow tables)

Recommended to source from your existing core tables (build-only):

### Required

* `core.py_bc_blocks`
  Columns needed: `repo, commit, rel_path, code_unit_id, block_id, first_instr_index, last_instr_index, start_offset, end_offset, anchor_span_start_byte, anchor_span_end_byte`
* `core.py_bc_cfg_edges`
  Columns needed: `repo, commit, rel_path, code_unit_id, src_block_id, dst_block_id, kind, cond_instr_id, exc_entry_index`
* `core.py_bc_defuse_events`
  Columns needed: `repo, commit, rel_path, code_unit_id, instr_id, instr_index, event_kind, space, name, confidence`
* `core.py_bc_instructions` *(needed for exception precision and better anchoring; still build-only)*
  Columns needed: `repo, commit, rel_path, code_unit_id, instr_id, instr_index, offset, start_offset, end_offset, span_start_byte, span_end_byte`
* `core.py_bc_exception_table` *(optional but strongly recommended for exception-edge precision)*
  Columns needed: `repo, commit, rel_path, code_unit_id, exc_entry_index, start_offset, end_offset, target_offset`

### Strongly recommended for correct parameter + scope semantics

* `core.py_sym_scopes`
* `core.py_sym_bindings`
* `core.py_sym_resolution_edges`
* `core.py_sym_function_partitions`

Why: bytecode does not explicitly “DEF” parameters; without symtable (or code object arg metadata), many parameter `LOAD_FAST` reads will appear uninitialized.

---

## A3) Output schema (recommended)

**Table**: `graph.dfg2_edges`

Minimum columns (I’d do this exact shape):

* `repo, commit, rel_path`
* `code_unit_id`
* `edge_id` (DECIMAL / BIGINT-compatible int; produced by `stable_decimal_id`)
* `src_event_id` (DECIMAL/BIGINT int)
* `dst_event_id` (DECIMAL/BIGINT int)
* `edge_kind` (e.g. `REACHES`, `PHI_IN`)
* `var_key` (VARCHAR) — stable, deterministic representation of the variable key
* `ordinal` (INTEGER) — deterministic input position for φ edges; 0 for reaches edges
* `confidence` (DOUBLE) — propagated from the USE/DEF event or φ rule
* `extras_json` (STRUCT/JSON/BLOB depending on your conventions) — include `space`, `name`, `binding_id`, and match metadata if you want

> **Important:** This table references φ events and synthetic defs, so you should also emit those event IDs into your event/node plane (even if that’s a separate table like `graph.flow2_events`). The edge computation below will generate them deterministically.

---

## A4) Deterministic variable identity (`var_key`)

You need a **single stable string** for “the thing whose value flows”.

Recommended `var_key` format:

* For locals/globals/frees/names:
  `"{space}:{name}:{binding_id_or_dash}"`

Example: `local:x:py_sym_bind__<id>` or `global:os:-`

* For attribute/subscript: **do not** pretend it’s the same variable as locals. Use separate namespaces:

  * `attr:<attr_name>:<base_token?>`
  * `subscr:<index_token?>:<base_token?>`

If you don’t have base/index identity yet, do:

* `attr:<attr_name>:-`
* `subscr:-:-`

…but keep the prefix so your schema can later become more precise without a breaking re-encode.

---

## A5) Deterministic event IDs (defs, uses, φ, synthetic)

Convert all event-like nodes to a unified numeric ID space with `stable_decimal_id`.

### A5.1) Bytecode def/use event ID

Even though `core.py_bc_defuse_events` already has `event_id` (string), for CPG2 flow it’s cleaner to hash a canonical payload:

```python
event_node_id = stable_decimal_id({
  "t": "flow2_event",
  "repo": repo, "commit": commit, "rel_path": rel_path,
  "code_unit_id": code_unit_id,
  "instr_id": instr_id,
  "instr_index": instr_index,
  "event_kind": event_kind,   # DEF/USE/KILL
  "space": space,
  "name": name,
})
```

### A5.2) φ event ID

Deterministic and *independent of iteration order*:

```python
phi_event_id = stable_decimal_id({
  "t": "flow2_phi",
  "repo": repo, "commit": commit, "rel_path": rel_path,
  "code_unit_id": code_unit_id,
  "block_id": block_id,
  "var_key": var_key,
})
```

### A5.3) Parameter synthetic defs

Seed “initial defs” for parameters:

```python
param_def_id = stable_decimal_id({
  "t": "flow2_param_def",
  "repo": repo, "commit": commit, "rel_path": rel_path,
  "code_unit_id": code_unit_id,
  "var_key": var_key,
})
```

### A5.4) External / unknown defs

For globals, builtins, closure frees, etc.:

```python
external_def_id = stable_decimal_id({
  "t": "flow2_external_def",
  "repo": repo, "commit": commit, "rel_path": rel_path,
  "code_unit_id": code_unit_id,
  "var_key": var_key,
})
```

### A5.5) Uninitialized sentinel

For local reads before any def/param seed:

```python
undef_def_id = stable_decimal_id({
  "t": "flow2_undef",
  "repo": repo, "commit": commit, "rel_path": rel_path,
  "code_unit_id": code_unit_id,
  "var_key": var_key,
})
```

---

## A6) Deterministic CFG normalization (per code unit)

For each `code_unit_id`:

1. **Blocks**: sort deterministically
   Recommended sort key:

   * `(first_instr_index, start_offset, block_id)`

2. Build:

   * `block_idx[block_id] -> 0..N-1`
   * `preds[block_id] -> list[pred_block_id]` sorted by `(block_idx[pred], pred_block_id)`
   * `succ_edges[src_block_id] -> list[Edge]` sorted by:
     `(block_idx[dst], kind, exc_entry_index or -1, cond_instr_id or "")`

3. Compute reachability from entry:

   * entry block = smallest block by sort order (or explicit `kind=="entry"` if you define it)
   * BFS/DFS over successors
   * drop unreachable blocks from analysis (deterministic)

---

## A7) Exception-edge precision (edge-specific out state)

This is where you can be **meaningfully “more best-in-class”** than typical block-level analyses, while staying in bytecode land.

**Problem:** exception edges should not “see” defs that occur after the protected range ends.

**Solution:** produce **edge-specific out states** for exception edges:

* For normal edges (`FALLTHROUGH`, `BRANCH`, `JUMP`): propagate block’s `out_state_normal`
* For exception edges (`EXCEPTION` with `exc_entry_index`):

  * join to `core.py_bc_exception_table` to get `end_offset`
  * determine the **cutoff instruction index** within the source block:

    * last `instr_index` whose `offset < end_offset` (use `core.py_bc_instructions`)
  * propagate `out_state_at_cutoff` (state after processing DEF/KILL events up to that cutoff)

This is still conservative (good for recall), but avoids the worst unsoundness (defs entirely outside protected range reaching the handler).

---

## A8) The φ-grouped SSA-like fixpoint

We compute per code unit:

* `in_def[block_id][var_key] -> def_event_id`  (single ID)
* `out_def_normal[block_id][var_key] -> def_event_id`
* `out_def_exc[(src_block_id, edge_id)][var_key] -> def_event_id`  (only for exception edges)
* `phi_needed[(block_id, var_key)] -> bool`
* `phi_inputs[(block_id, var_key)] -> list[(pred_block_id, incoming_def_id)]` (sorted)

### A8.1) Incoming def for a predecessor edge

For a particular predecessor `p -> b` via edge `e`:

* if `e.kind != "EXCEPTION"`: incoming = `out_def_normal[p].get(var_key)`
* else: incoming = `out_def_exc[(p,e.edge_id)].get(var_key)`

### A8.2) φ decision rule (deterministic)

At block `b` with `k = len(preds[b])`:

For each `var_key` appearing in any incoming state:

* collect incoming def IDs (may be None)
* normalize None to `undef_def_id` **only if space is local/name/free**, or to `external_def_id` if global/attr/subscr (your choice; I recommend this behavior)
* if number of distinct incoming IDs > 1 and `k >= 2`:

  * `in_def[b][var_key] = phi_event_id(b,var_key)`
  * record `phi_inputs[(b,var_key)] = sorted(preds with their incoming IDs)`
* else:

  * `in_def[b][var_key] = the single incoming ID` (or sentinel if none)

### A8.3) Transfer through block

Simulate events in block order (instr_index asc; tie-break by stable event_node_id):

State is `cur[var_key] -> def_event_id`

* start: `cur = in_def[b].copy()`
* for each event in this block:

  * if `event_kind == "USE"`: no state change
  * if `event_kind == "DEF"`: `cur[var_key] = def_event_id(event)`
  * if `event_kind == "KILL"`:

    * treat delete as a **def** to the delete event itself (recommended), i.e.
      `cur[var_key] = kill_event_id(event)`
    * this makes downstream uses flow from the deletion site (useful!)

End:

* `out_def_normal[b] = cur`

Additionally, for exception cutoffs:

* while simulating, whenever you pass a cutoff instr_index, snapshot `cur` into `out_def_exc[(b, edge_id)]`

### A8.4) Convergence loop

Iterate blocks in **reverse postorder** (or simple forward order works; reverse postorder is faster):

Stop when:

* `out_def_normal` and all `out_def_exc` stop changing
* and φ decisions stop changing

Because we collapse joins into φ defs, this converges quickly.

---

## A9) Edge emission pass (after convergence)

Once the fixpoint is stable:

### A9.1) Emit φ input edges (`PHI_IN`)

For each `(block_id, var_key)` where φ exists:

Let `phi_id = phi_event_id(block_id,var_key)`

For each `(pred_block_id, incoming_def_id)` in `phi_inputs[(block_id,var_key)]`:

* `edge_kind = "PHI_IN"`
* `ordinal = pred_ordinal` where pred_ordinal is index of pred in `sorted(preds[block_id])`
* edge_id payload:

```python
edge_id = stable_decimal_id({
  "t": "dfg2_edge",
  "kind": "PHI_IN",
  "repo": repo, "commit": commit,
  "code_unit_id": code_unit_id,
  "phi_id": phi_id,
  "pred": pred_block_id,
  "src": incoming_def_id,
})
```

### A9.2) Emit reaches edges (`REACHES`)

Simulate each block again with final `in_def`:

For a `USE` event with event_node_id `use_id`:

* find `src_def_id = cur.get(var_key, undef/external sentinel)`
* emit:

```python
edge_id = stable_decimal_id({
  "t": "dfg2_edge",
  "kind": "REACHES",
  "repo": repo, "commit": commit,
  "code_unit_id": code_unit_id,
  "src": src_def_id,
  "dst": use_id,
  "var_key": var_key,
})
```

* `ordinal = 0` (since exactly one def per use under SSA)
* `confidence = min(def_confidence, use_confidence)` if you track it, else use `use_confidence`

---

## A10) Hamilton node signature (build-only)

Put the algorithm in a pure compute module (e.g. `src/codeintel/build/graphs/compute/dfg2.py`), and wrap with a Hamilton node in `src/codeintel/build/hamilton/native/graphs/dfg2.py`.

Suggested Hamilton node signature:

```python
def dfg2_edges(
    q__core__py_bc_blocks: InferableTabularInput,
    q__core__py_bc_cfg_edges: InferableTabularInput,
    q__core__py_bc_defuse_events: InferableTabularInput,
    q__core__py_bc_instructions: InferableTabularInput,
    q__core__py_bc_exception_table: InferableTabularInput,
    q__core__py_sym_scopes: InferableTabularInput,
    q__core__py_sym_bindings: InferableTabularInput,
    q__core__py_sym_resolution_edges: InferableTabularInput,
    q__core__py_sym_function_partitions: InferableTabularInput,
) -> InferableTabularInput:
    ...
```

Return a `pa.Table` and finalize with:

* `dedupe_table_for_table("graph.dfg2_edges", table)`
* `align_table_to_contract("graph.dfg2_edges", table)`

No storage dependencies.

---

# B. `graph.cdg2_edges`: postdom-based control dependence with fast bitsets

You already have a bitset-style CDG implementation in `build/hamilton/native/graphs/cdg.py`. This is the **CPG2 version** tuned for:

* bytecode CFG (`core.py_bc_*`)
* synthetic exit node handling
* deterministic edge IDs and ordinals
* explicit `via_succ_block_id` (critical for explaining *why* the dependence exists)

---

## B1) Inputs

* `core.py_bc_blocks`
* `core.py_bc_cfg_edges`

Optional (recommended):

* `core.py_bc_instructions` (only if you want to attach predicate spans later without re-join)

---

## B2) Output schema (recommended)

**Table**: `graph.cdg2_edges`

Columns:

* `repo, commit, rel_path`
* `code_unit_id`
* `edge_id` (stable_decimal_id)
* `src_block_id` (controller)
* `dst_block_id` (controlled)
* `via_succ_block_id` (the successor edge witnessing the dependence)
* `via_edge_kind` (e.g. `BRANCH`, `FALLTHROUGH`, `EXCEPTION`, `JUMP`)
* `cond_instr_id` (from CFG edge if available)
* `ordinal` (deterministic; described below)
* `extras_json` (optional: exc_entry_index, etc.)

---

## B3) Deterministic CFG indexing

For each code unit:

* index blocks in deterministic order:

  * `(first_instr_index, start_offset, block_id)`

Build:

* `succs[idx] -> list[int]`
* Keep a parallel list of outgoing CFG edges so you can map `(src_idx, dst_idx)` back to `via_edge_kind / cond_instr_id`.

---

## B4) Add synthetic exit node (if needed)

Let `exit_nodes = {i | succs[i] is empty}`.

If `len(exit_nodes) == 1`, use it.

If `len(exit_nodes) > 1`:

* create synthetic index `exit_idx = N`
* add `succs[i].append(exit_idx)` for all `i in exit_nodes`
* (do *not* add to original CFG edge list; it’s synthetic)

This makes postdom well-defined.

---

## B5) Postdom computation using int bitsets

Represent a set of postdominators for node `i` as a Python `int` bitmask.

* `bit(i) = 1 << i`
* `ALL = (1 << (N+1)) - 1`

Initialize:

* `post[exit_idx] = bit(exit_idx)`
* for all other nodes `i`: `post[i] = ALL`

Iterate to fixpoint:

* for `i != exit_idx`:

  * if `succs[i]` empty: treat succs as `[exit_idx]`
  * `new = bit(i) | AND(post[s] for s in succs[i])`
  * if new != post[i]: update

This is fast and avoids NetworkX.

---

## B6) Control dependence extraction

Use the standard “postdom difference” method (Ferrante-style):

For each CFG edge `src -> succ`:

* `diff = post[succ] & ~post[src]`
* each node `n` in `diff` is control-dependent on `src` via successor `succ`
* emit `src_block -> n_block` with `via_succ_block_id = succ_block`

Iterate bits in `diff` deterministically:

* convert `diff` to indices by repeated `lsb = diff & -diff` trick
* or precompute bit positions

Skip:

* `n == exit_idx`

---

## B7) Deterministic edge ID and ordinal

### Edge ID

```python
edge_id = stable_decimal_id({
  "t": "cdg2_edge",
  "repo": repo, "commit": commit,
  "code_unit_id": code_unit_id,
  "src": src_block_id,
  "dst": dst_block_id,
  "via": via_succ_block_id,
})
```

### Ordinal

You can make ordinal both meaningful and deterministic:

* For each `src_block_id`, compute its successor list sorted by `block_idx[succ]`.
* `ordinal = index_of(via_succ_block_id in sorted_succs)`

That gives a stable “which successor caused it” ordering.

---

## B8) Hamilton node signature

```python
def cdg2_edges(
    q__core__py_bc_blocks: InferableTabularInput,
    q__core__py_bc_cfg_edges: InferableTabularInput,
) -> InferableTabularInput:
    ...
```

Finalize with `dedupe_table_for_table` + `align_table_to_contract` as usual.

---

# C. `graph.pdg2_edges_span`: SpanResolver folding + deterministic aggregation/ordinals

## C1) Goal

Produce a **span-level PDG** for UI/analytics/search:

* Input: low-level DFG2 + CDG2 edges (event/block level)
* Output: edges between **canonical code anchors** (“spans”)
* Must be:

  * deterministic
  * compressing (aggregation)
  * explainable (store folding provenance + counts)

---

## C2) Inputs

At minimum:

* `graph.dfg2_edges`
* `graph.cdg2_edges`
* `core.py_bc_defuse_events`
* `core.py_bc_instructions`
* `core.py_bc_blocks`
* `core.syntax_spans`
* `core.ts_nodes` *(optional fallback; recommended)*

---

## C3) Output schema (recommended)

**Table**: `graph.pdg2_edges_span`

Columns:

* `repo, commit, rel_path`
* `edge_id` (stable_decimal_id)
* `src_anchor_id` (stable_decimal_id)
* `dst_anchor_id` (stable_decimal_id)
* `edge_kind` (`DATA`, `CONTROL`)
* `ordinal` (deterministic)
* `src_anchor_kind` (e.g. `SYNTAX_SPAN`, `TS_NODE`, `BC_INSTR`, `RAW_BYTE`)
* `dst_anchor_kind`
* `src_start_byte, src_end_byte, dst_start_byte, dst_end_byte` (BIGINT) *(optional but very useful)*
* `extras_json`:

  * aggregation counts
  * folded-from metadata (match_kind, candidate_count)
  * (for DATA) truncated var list summary

---

## C4) SpanResolver folding rules (the critical part)

### Build 3 resolvers (tiered)

**Tier 1: SYNTAX spans**

* payload = a lightweight anchor record:

  * `("SYNTAX_SPAN", producer, span_id, start_byte, end_byte, span_kind)`
* Insert spans in **deterministic priority order**:

  1. producer priority (recommend): `libcst > ast > tree_sitter > other`
  2. span_kind priority (you define; recommend identifiers/expr/stmt before module/file)
  3. `(start_byte, end_byte, span_id)`

This matters because `SpanResolver` tie-breaks by (span length, insertion order).

**Tier 2: TS nodes (fallback)**

* payload = `("TS_NODE", language, node_id, start_byte, end_byte, node_type)`
* Filter to `is_named=True`, `is_error=False`, `has_error=False` (optional)

**Tier 3: RAW byte spans**

* if neither resolves, anchor directly on the byte interval

---

## C5) Anchor ID (canonical, producer-independent)

Even if you resolve to a syntax span, it’s best to use an **anchor_id that is independent of the producer**, so multiple producers can collapse naturally.

Use:

```python
anchor_id = stable_decimal_id({
  "t": "anchor_span",
  "repo": repo, "commit": commit, "rel_path": rel_path,
  "start_byte": start_byte,
  "end_byte": end_byte,
})
```

Store producer/span_id/node_id in `extras_json` for provenance.

---

## C6) Mapping endpoints to byte spans

### For DFG2 edges

* `src_event_id` / `dst_event_id` refer to flow events
* you need `instr_id` for each event to get `span_start_byte/span_end_byte`

Two approaches:

1. If `dfg2_edges` already carries `src_instr_id/dst_instr_id`, use those.
2. Otherwise:

   * join `dfg2_edges` → `core.py_bc_defuse_events` (to recover instr_id for that event payload)
   * join → `core.py_bc_instructions` (to recover bytes)

In build-only Arrow land, use `arrow_join_tables()` with validated join keys.

### For CDG2 edges

* Preferred controlling anchor:

  * if `cond_instr_id` exists: use that instruction’s span
  * else: use `src_block.anchor_span_*`
* Controlled anchor:

  * use `dst_block.anchor_span_*`

---

## C7) Folding procedure per endpoint

Given `(rel_path, start_byte, end_byte)`:

1. Try syntax resolver:

   * `match = syntax_resolver.resolve(rel_path, start_byte, end_byte, allow_adjacent_point=True)`
   * If found: anchor_kind = `SYNTAX_SPAN`, anchor bytes = matched span bytes
2. Else try TS resolver similarly
3. Else: anchor_kind=`RAW_BYTE`, bytes=original

Return:

* `anchor_id`
* `anchor_kind`
* `match_kind`, `candidate_count` for coverage

---

## C8) Deterministic aggregation rules

You want a span PDG that’s compact but still informative.

### C8.1) Construct “raw span edges”

Emit raw rows from each underlying edge:

* DATA edge from `dfg2_edges`:

  * `(src_anchor_id, dst_anchor_id, edge_kind="DATA", var_key)`
* CONTROL edge from `cdg2_edges`:

  * `(src_anchor_id, dst_anchor_id, edge_kind="CONTROL", via_succ_block_id)` (optional in extras)

### C8.2) Aggregate

Group by:

* `(repo, commit, rel_path, src_anchor_id, dst_anchor_id, edge_kind)`

Aggregate deterministically:

* `edge_count = count(*)`
* For DATA:

  * collect `var_key` into a **sorted unique list**
  * apply deterministic cap `K` (e.g. 25):

    * `var_keys_sorted = sorted(set(var_key))`
    * `var_keys_topk = var_keys_sorted[:K]`
    * store `var_key_total = len(var_keys_sorted)`, `var_key_truncated = max(0, total-K)`
* For CONTROL:

  * optionally collect `via_edge_kind` counts (`BRANCH/FALLTHROUGH/EXCEPTION`) into a small dict

### C8.3) Deterministic ordinal

Since aggregation collapses most multiplicity, you can keep ordinal simple and stable:

* `ordinal = 0` for `CONTROL`
* `ordinal = 1` for `DATA`

Or, if you want lexical determinism:

* `ordinal = 0` for `DATA`, `1` for `CONTROL` — just be consistent.

### C8.4) Stable `edge_id`

```python
edge_id = stable_decimal_id({
  "t": "pdg2_span_edge",
  "repo": repo, "commit": commit,
  "rel_path": rel_path,
  "src": src_anchor_id,
  "dst": dst_anchor_id,
  "edge_kind": edge_kind,
})
```

---

## C9) Hamilton node signature

```python
def pdg2_edges_span(
    q__graph__dfg2_edges: InferableTabularInput,
    q__graph__cdg2_edges: InferableTabularInput,
    q__core__py_bc_instructions: InferableTabularInput,
    q__core__py_bc_blocks: InferableTabularInput,
    q__core__syntax_spans: InferableTabularInput,
    q__core__ts_nodes: InferableTabularInput,
) -> InferableTabularInput:
    ...
```

Then:

* build resolvers (syntax + ts)
* map endpoints
* aggregate deterministically
* `dedupe_table_for_table("graph.pdg2_edges_span", ...)`
* `align_table_to_contract("graph.pdg2_edges_span", ...)`

Still build-only.

---

# D. Practical implementation layout in `src/codeintel/build`

To keep this clean and extensible:

## Pure compute (no Hamilton, no schema service)

* `src/codeintel/build/graphs/compute/dfg2.py`
* `src/codeintel/build/graphs/compute/cdg2.py`
* `src/codeintel/build/graphs/compute/pdg2_span.py`

These accept `pa.Table`s and return `pa.Table`s (and optionally intermediate dicts in Python).

## Hamilton wrappers (thin)

* `src/codeintel/build/hamilton/native/graphs/dfg2.py`
* `src/codeintel/build/hamilton/native/graphs/cdg2.py`
* `src/codeintel/build/hamilton/native/graphs/pdg2_span.py`

They:

* convert inputs with `tabular_to_table`
* call compute functions
* align & dedupe

**No imports from `codeintel.storage.*` anywhere in these new modules.**

---

# E. What you get immediately from this redesign

* **DFG2** becomes linear-ish in event count, not combinatorial at joins.
* **φ edges** preserve multi-path provenance without blowing up `REACHES`.
* **CDG2** is fast and explainable (`via_succ_block_id` + cond info).
* **PDG2 span** becomes a stable, compact surface for:

  * UI graph overlays
  * “what depends on what” queries
  * downstream embeddings / ranking features

---

If you want, I can now take *this exact compute logic* and translate it into **literal code skeletons** for each module (with the exact Arrow column selections, joins via `arrow_join_tables`, and the specific internal Python structures I’d use to keep it fast). That would still be strictly build-only and would be written to match your existing style (`align_table_to_contract`, `dedupe_table_for_table`, etc.).

