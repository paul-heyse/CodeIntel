# Tree-sitter integration plan (Python-only pipeline)

This plan integrates tree-sitter as a first-class, tightly coupled layer in the
existing Python pipeline (LibCST + CPython AST + SCIP), without changing the
canonical ownership of `core.syntax_nodes`/`core.syntax_edges`. The goal is to
add the data and capabilities tree-sitter provides that LibCST and AST do not,
while keeping the pipeline deterministic and feed-forward.

## 1. Goals and non-goals

Goals
- Error-tolerant parsing and diagnostics so files still yield usable structure
  when LibCST cannot parse.
- Incremental, change-driven indexing for future watch-mode or incremental
  builds (changed ranges + scoped queries).
- Query-pack extraction at scale (structural facts, tokens, trivia, error nodes)
  in a declarative form that is easy to extend.
- Additional syntax metadata that is not present in LibCST or AST:
  - `is_error`, `is_missing`, `has_error`
  - `grammar_id`, `kind_id`, `field_id`, `field_name`
  - `parse_state`, `next_parse_state`
  - `node_id` (tree-local) for short-lived incremental caches
- Tight integration with existing canonical tables so CPG and downstream tools
  use the same node ids and spans.

Non-goals
- Replacing LibCST or AST as the canonical syntax graph for Python.
- Multi-language injection support (we only target Python).
- Long-term persistence of tree-sitter parse trees (they are not serializable
  in a stable format).

## 2. Architecture at a glance

Current pipeline (simplified)
- LibCST -> `core.syntax_nodes`, `core.syntax_edges`, `core.syntax_*` facts
- AST -> AST facts merged into `core.syntax_nodes.extras_json`
- SCIP -> occurrence weld -> `core.scip_occurrence_*`

Integrated pipeline (proposed)
1. `syntax_index` (LibCST + AST) produces canonical syntax graph.
2. `tree_sitter_index` produces:
   - full tree-sitter CST nodes/edges
   - error/missing nodes
   - tokens and trivia
   - query-pack captures
3. `syntax_augment` stage welds tree-sitter outputs to the canonical
   `core.syntax_nodes` using spans, and:
   - attaches tree-sitter metadata to `core.syntax_nodes.extras_json`
   - fills canonical nodes/edges for files where LibCST failed
4. CPG assembly uses canonical `core.syntax_nodes` and can optionally consume
   tree-sitter tokens/trivia and error diagnostics.

## 3. Data model extensions

Keep existing tables:
- `core.ts_captures`
- `core.ts_parse_errors`
- `core.parse_manifest` (producer = `tree_sitter`)

Add new tables to capture data missing from LibCST/AST.

### 3.1 Tree-sitter nodes and edges

`core.ts_nodes`
- PK: `(repo, commit, rel_path, language, node_id)`
- Required columns:
  - `repo`, `commit`, `rel_path`, `language`
  - `node_id` (stable hash of `rel_path + start_byte + end_byte + node_type +
    grammar_id + field_id + child_ordinal`)
  - `node_type`, `grammar_id`, `kind_id`
  - `is_named`, `is_missing`, `is_error`, `has_error`
  - `start_byte`, `end_byte`, `start_row`, `start_col`, `end_row`, `end_col`
  - `parse_state`, `next_parse_state`
  - `text_preview`, `extras_json`

`core.ts_edges`
- PK: `(repo, commit, rel_path, language, parent_node_id, child_node_id,
  child_ordinal)`
- Required columns:
  - `repo`, `commit`, `rel_path`, `language`
  - `parent_node_id`, `child_node_id`
  - `field_id`, `field_name`
  - `child_ordinal`

### 3.2 Tokens and trivia

`core.ts_tokens`
- PK: `(repo, commit, rel_path, language, token_id)`
- Required columns:
  - `token_id` (stable hash of `rel_path + start_byte + end_byte + token_kind`)
  - `token_kind` (identifier, keyword, operator, literal, punctuation)
  - `node_type` (tree-sitter node type)
  - `start_byte`, `end_byte`, `start_row`, `start_col`, `end_row`, `end_col`
  - `text_preview`, `extras_json`

`core.ts_trivia`
- PK: `(repo, commit, rel_path, language, trivia_id)`
- Required columns:
  - `trivia_id` (stable hash of `rel_path + start_byte + end_byte + trivia_kind`)
  - `trivia_kind` (comment, whitespace, newline, indent)
  - `start_byte`, `end_byte`, `start_row`, `start_col`, `end_row`, `end_col`
  - `text_preview`, `extras_json`

### 3.3 Language metadata

`core.ts_language_metadata`
- PK: `(language, abi_version, semantic_version)`
- Columns:
  - `language`, `abi_version`, `semantic_version`
  - `node_kind_count`, `field_count`, `parse_state_count`
  - `created_at`

### 3.4 Crosswalk to canonical syntax nodes

Option A (preferred): `core.ts_syntax_node_xref`
- PK: `(repo, commit, rel_path, language, ts_node_id, producer)`
- Columns:
  - `repo`, `commit`, `rel_path`, `language`
  - `producer` (canonical syntax producer, usually `libcst`)
  - `ts_node_id`, `syntax_node_id`
  - `match_kind` (EXACT, CONTAINS, OVERLAP, POINT)
  - `candidate_count`

Option B: store per-node array in `core.syntax_nodes.extras_json.ts_nodes[]`
with `{ts_node_id, ts_node_type, start_byte, end_byte, match_kind}`.
Use Option A if you want fast joins.

## 4. Extraction logic (tree-sitter indexer)

### 4.1 Full CST traversal

Extend `codeintel.ingestion.tree_sitter.runner.run_tree_sitter` to:
- Traverse the full tree with `TreeCursor` (no list allocations).
- Emit `core.ts_nodes` and `core.ts_edges`.
- Record `field_id` and `field_name` for edges.
- Capture `parse_state` and `next_parse_state` on nodes.

### 4.2 Error and missing nodes

Keep `core.ts_parse_errors` but enrich `extras_json` with:
- `node_type` (error node type)
- `has_error` (ancestor info if useful)
- `parse_state` (for diagnostics)

### 4.3 Token and trivia extraction

Use query packs to emit tokens and trivia:
- Add `tokens.scm` and `trivia.scm` packs under
  `src/codeintel/ingestion/tree_sitter/packs/python/`.
- Use `QueryCursor` with `match_limit` and `set_byte_range` when available.

### 4.4 Query pack linting

Before running packs, lint each query:
- Reject non-local patterns unless explicitly allowed.
- Require rooted patterns by default.
- Track per-pack `pattern_count`, `capture_count`.

Store query pack metadata in `core.ts_captures.extras`:
- `pattern_index`, `capture_index`
- `field_name` (if capture is field-specific)
- `query_hash` or `query_version`

## 5. Canonical integration with LibCST and AST

### 5.1 Span-based weld (tree-sitter -> syntax_nodes)

Algorithm (per file):
1. Build interval index on `core.syntax_nodes` by `(start_byte, end_byte)`.
2. For each tree-sitter node:
   - Prefer exact span match.
   - Else smallest containing node.
   - Else overlap fallback.
3. Record weld in `core.ts_syntax_node_xref` (or in `extras_json`).

### 5.2 Error-tolerant fallback

If LibCST fails for a file (parse_manifest parse_ok=false):
- Use tree-sitter nodes/edges to produce canonical `core.syntax_nodes` and
  `core.syntax_edges` for that file with `producer=tree_sitter`.
- Emit a parse manifest row for `producer=tree_sitter` and set `parse_ok=true`
  if tree-sitter could parse.

### 5.3 AST and LibCST remain primary

LibCST continues to:
- define canonical syntax nodes/edges for Python
- provide byte-accurate spans and trivia
AST continues to:
- provide semantic facts and CFG/DFG inputs
Tree-sitter supplements and backfills only.

## 6. Incremental indexing (change-driven, future-ready)

Phase-in plan for incremental usage:
1. Maintain a per-file in-memory tree cache during a run.
2. When a file changes, apply `Tree.edit` and reparse using `old_tree`.
3. Use `changed_ranges` to scope query packs and token/trivia extraction.
4. Expose `core.ts_changed_ranges` (optional) for observability:
   - `rel_path`, `start_byte`, `end_byte`, `start_row`, `end_row`

This can plug into the existing module inventory and file_state machinery once
incremental builds are supported.

## 7. Pipeline wiring changes

### 7.1 Ingestion target updates

`codeintel.build.hamilton.native.ingestion.tree_sitter`
- Add outputs for `core.ts_nodes`, `core.ts_edges`, `core.ts_tokens`,
  `core.ts_trivia`, `core.ts_language_metadata`, `core.ts_syntax_node_xref`.

`codeintel.ingestion.compute.tree_sitter_index`
- Extend buffers to collect new tables.
- Add TreeCursor traversal and query pack execution with range limits.

### 7.2 New integration stage

Add `syntax_augment` (or extend `syntax_index`):
- Inputs:
  - `core.syntax_nodes`, `core.syntax_edges`
  - `core.ts_nodes`, `core.ts_edges`, `core.ts_parse_errors`
- Outputs:
  - `core.ts_syntax_node_xref` (or extras_json merge)
  - Fallback canonical syntax nodes/edges for LibCST failures

### 7.3 Registry updates

Update:
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/schemas/generated_rows/core.py`
- `src/codeintel/core/registry/dag_output_inventory.yaml`
- `src/codeintel/build/hamilton/native/ingestion/tree_sitter.py`

## 8. Downstream usage and CPG integration

- `graph.cpg_nodes` remains based on canonical `core.syntax_nodes`.
- If desired, add optional node properties sourced from tree-sitter:
  - `has_error`, `is_missing`, `grammar_id`, `field_name`, `parse_state`
- Use `core.ts_tokens` and `core.ts_trivia` for:
  - advanced query engine token searches
  - indexing doc comments / inline comments
  - lightweight syntax-level filters in storage queries

## 9. Validation and acceptance criteria

1. Every Python file yields:
   - `core.parse_manifest` row for LibCST and tree-sitter.
   - `core.ts_parse_errors` entries when tree-sitter detects errors.
2. If LibCST fails, tree-sitter still yields canonical `core.syntax_nodes` and
   `core.syntax_edges` for that file (producer=tree_sitter).
3. Weld coverage:
   - At least 95 percent of tree-sitter nodes map to a syntax node by span.
4. Query packs:
   - All packs pass linting (rooted, not non-local unless allowed).
   - `match_limit` never exceeded on baseline repo scan.

## 10. Implementation phases (suggested order)

Phase 0 - Query pack foundation
- Add `tokens.scm` and `trivia.scm` packs for Python.
- Add query pack linting and metadata in captures.

Phase 1 - Full CST output
- Emit `core.ts_nodes` and `core.ts_edges`.
- Add `core.ts_language_metadata`.

Phase 2 - Canonical weld
- Add `core.ts_syntax_node_xref`.
- Merge tree-sitter metadata into `core.syntax_nodes.extras_json`.

Phase 3 - Fallback canonicalization
- Use tree-sitter nodes/edges when LibCST fails.

Phase 4 - Incremental indexing
- Add in-memory tree cache and `changed_ranges` scoped queries.

Phase 5 - CPG consumption
- Optional: surface tree-sitter node flags and token/trivia data into CPG
  node properties or analytics views.

## 11. Config toggles

Introduce optional knobs (defaults shown):
- `tree_sitter.enabled = true`
- `tree_sitter.emit_nodes_edges = true`
- `tree_sitter.emit_tokens = true`
- `tree_sitter.emit_trivia = true`
- `tree_sitter.enable_incremental = false`
- `tree_sitter.match_limit = 10000`
- `tree_sitter.allow_non_local_patterns = false`

These keep the pipeline adaptable without changing the canonical contracts.

