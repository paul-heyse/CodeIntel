Yes — if you treat LibCST / SCIP / tree-sitter / `ast` as more than “dataset generators”, they unlock a whole second layer of **static “program facts” and “semantic indexes”** that are *not* just graph edges. The big design shift is: **materialize canonical indexes + summaries first**, then let CFG/DFG/lineage be *views* over those facts.

Below is a concrete catalog of additional *syntax + symbol–based* static analyses you can add (with emphasis on **high-certainty** outputs), and exactly what the advanced capabilities of each library buy you.

---

## 1) Canonical cross-reference and symbol facts (beyond edges)

### What you can add

**A. Definition/reference/import/read/write classification at token-span resolution**

* For every identifier occurrence: is it a **definition**, **import**, **read**, **write**?
* This yields extremely high-certainty “causal” facts like:

  * “This function *must* write to global `X`.”
  * “This module *must* import `foo.bar` under alias `baz`.”
  * “This statement *must* read `df` and *must* write `df2`.”

### What unlocks it

* **SCIP** exposes an `Occurrence` with a `symbol` string, a `symbol_roles` bitmask, and a source `range`; the bitmask explicitly includes **Definition**, **Import**, **Write**, **Read**, etc. (so you can derive def/ref/read/write without heuristics).
* **LibCST** exposes `ExpressionContextProvider` returning Load/Store/Del-style context (similar intent as `ast.expr_context`, but available in CST workflows).

**Design note:** this becomes a *non-graph* “fact table” like `occurrence_fact(symbol_id, role_mask, span_id, file_id, container_id)` that everything else joins against.

---

## 2) Repo-wide name binding + qualified naming (turn “names” into stable identities)

### What you can add

**B. Qualified-name and fully-qualified-name attribution for every expression**

* You can attach:

  * “This `Name` node resolves to qualified name(s) …”
  * “This module is actually `package.subpackage.module`”
* This is foundational for high-certainty causal claims because it turns “`foo`” into **which `foo`**.

### What unlocks it

* **LibCST** explicitly provides `QualifiedNameProvider` (semantic qualified names) and `FullyQualifiedNameProvider` (repo-aware, including relative-import handling and module FQN stored on `Module`).
* **LibCST** also calls out index suites that include “definitions”, “references”, “imports”, “call-sites”, and “types” as first-class “index suite” outputs (exactly the kind of non-graph analysis you’re asking about).

**Why this matters for “high certainty”:** you can mark any relationship as MUST only when the qualified identity is unambiguous, and otherwise downgrade to MAY.

---

## 3) Symbol relationship analysis (OOP semantics without “graph metrics”)

### What you can add

**C. Override / implementation / inheritance relationship facts**

* Examples of causal questions you can answer more confidently:

  * “Method `C.m` *overrides* `Base.m`.”
  * “This class *extends* that class.”
  * “This method is part of an interface/ABC implementation chain.”

### What unlocks it

* **SCIP’s** `SymbolInformation` is explicitly described as carrying documentation and **relationships** such as “class extends another” or “method overrides parent method.”

This is *not* a “complexity metric” — it’s semantic structure you can use to improve dispatch reasoning, slicing, and provenance.

---

## 4) Incremental / change-driven static analysis (not just for performance)

### What you can add

**D. “Change impact” analysis and incremental re-indexing**

* Given two versions of a file, compute:

  * what structural regions changed,
  * which indexes must be recomputed,
  * which prior facts remain valid.

### What unlocks it

* **tree-sitter** provides `changed_ranges` as an “ancestor-path diff” notion: outside those byte ranges, the ancestor chain is identical; you can safely cache facts outside changed regions and only rerun relevant query packs within lifted containers.

This enables “state of the art” workflows like: *edit → minimal invalidation → recompute only affected facts → recompute only affected higher-level analyses.*

---

## 5) Query-driven, rule-based extraction as a first-class analysis product

### What you can add

**E. Deterministic pattern extraction with rule metadata**

* Rather than hard-coding AST walks for every library/framework pattern, you can define “rules” as tree-sitter queries and emit:

  * `pattern_match_fact(rule_id, capture_set, span_id, attributes…)`
* This is ideal for:

  * framework hooks (FastAPI routes, Hamilton nodes, etc.),
  * known data APIs (pandas/Polars/DuckDB calls),
  * banned constructs (`eval`, dynamic import patterns),
  * “this call is a join” / “this call is a SQL execution site”.

### What unlocks it

* tree-sitter query execution encourages match-by-match extraction (`matches`) when you need relational coherence between captures (definition node + name + body), and query packs can be windowed and bounded for safety.
* tree-sitter supports **pattern settings** (via `#set!`) that you can retrieve programmatically using `Query.pattern_settings(pattern_index)`, which is perfect for attaching structured metadata to a match (“this capture is SQL”, “language=sql”, etc.).

This is an *analysis unlock*: your “static analysis rules” become **data** (query packs) instead of only code.

---

## 6) Embedded-language parsing (SQL/regex/templates) with correct source coordinates

### What you can add

**F. High-certainty extraction + parsing of embedded DSLs**
Examples:

* SQL strings passed to `duckdb.sql(...)`, `con.execute(...)`, `.read_sql(...)`
* Regex literals
* HTML/Jinja templates embedded in Python strings
* GraphQL strings, etc.

Instead of treating these as opaque strings, you can:

1. locate them deterministically,
2. parse them with the right grammar, and
3. preserve host-file coordinates for every embedded token.

### What unlocks it

* tree-sitter’s recommended approach is parsing embedded languages over the host bytes using `included_ranges`, producing an embedded tree whose node ranges still line up with the full host document (key for evidence + joins).
* This is designed to work with injection query packs that provide `injection.language`, `injection.combined`, and `injection.include-children` semantics (so your extraction is robust and not “string slicing hacks”).

**This is one of the biggest “beyond edges” unlocks** for your join/lineage goals: it turns “stringly-typed SQL” into a real parse tree with evidence-grade spans.

---

## 7) Canonicalization, semantic diffing, and stable evidence (AST-native, not metrics)

### What you can add

**G. Structural equivalence + canonical rendering**

* Use cases:

  * semantic diff that ignores whitespace/formatting,
  * detect refactors vs real logic changes,
  * stable fingerprinting of expressions/blocks,
  * normalization before downstream analyses.

### What unlocks it

* Python `ast.unparse()` is explicitly documented as producing code that yields an **equivalent AST if parsed back** (not identical source), i.e., it’s a canonicalization tool, not a pretty-printer guarantee.
* Python 3.14+ adds `ast.compare(a, b, compare_attributes=...)` for first-class structural equality (ignoring offsets by default, or requiring them if needed).

**H. “Evidence extraction” with precise spans**

* Extract the exact source substring corresponding to a node (for auditability / explainability).
* This is crucial if you’re aiming for high-certainty “causal” claims: you can always attach the literal code evidence.

What unlocks it:

* `ast.get_source_segment` returns the exact substring only when full span info is present (`lineno`, `end_lineno`, `col_offset`, `end_col_offset`).
* LibCST gives you `PositionProvider` and `ByteSpanPositionProvider` for span attribution in CST space.

---

## 8) Constant and contract extraction (turn syntax into “configuration facts”)

### What you can add

**I. Literal-only constant evaluation for schemas, column lists, join keys, etc.**

* Extract “declared” values from code with high certainty *when they are literals/containers* (lists of column names, dict schemas, configuration flags, etc.).

### What unlocks it

* `ast.literal_eval` evaluates only literal/container structures (not arbitrary expressions), and the guide explicitly recommends using it on a subtree you already control (parse once, evaluate that node).

(And the same guide correctly warns it’s **not safe on untrusted input**; for your offline indexing pipeline, you still treat it as bounded-but-not-zero-risk.)

This is *hugely relevant* to your join/lineage work because many real-world pipelines express join keys and column projections as literal lists/dicts.

---

## 9) Scaling these analyses without turning the system into a tangle

### What you can add

**J. A formal “index suite” architecture**
Instead of “we compute a bunch of edges”, the design becomes:

* `parse_manifest` / `syntax_errors`
* `definitions_index`
* `references_index`
* `imports_index`
* `call_sites_index`
* `type_index` (optional)
* `embedded_dsl_index` (tree-sitter injections)
* `constants_index` (ast.literal_eval)
* `doc_index` (docstrings + signature docs)
* `relationship_index` (overrides/extends)

LibCST basically hands you this mental model: it explicitly enumerates those index suites as first-class outputs.

**K. Repo-scale caching and parallelism**

* If you start doing repo-wide qualified naming + indexing, you want deterministic caching semantics.
* LibCST’s `FullRepoManager` patterns include precomputing a `resolve_cache` before forking multiprocessing workers (so each worker doesn’t redo resolution work).

---

## What’s “state of the art” here (for *high-certainty* relationships)?

A best-in-class design in 2025 for “high certainty causal relationships” in a dynamic language like Python usually looks like:

1. **Materialize authoritative, minimal, stable facts** (occurrences with roles, qualified names, spans, import aliases, doc/signature payloads, override relationships).

   * SCIP gives roles + symbol identity.
   * LibCST gives qualified naming + CST-level contexts/spans.

2. **Treat anything uncertain as explicitly MAY with provenance**, never silently “upgrade” it to MUST.

   * Example: call target resolution is MUST only if symbol resolution is unambiguous; otherwise it’s MAY with candidates.

3. **Use tree-sitter query packs + injections** as the “rule layer” for scalable, multi-language extraction (including embedded DSLs), and use incremental parsing to avoid recomputing the world on every change.

4. **Use AST canonicalization/diff** to stabilize results across formatting churn and to detect real semantic changes early (great for test harnesses and regression prevention).

---

## A practical “next 5 tables” recommendation (beyond edges)

If you want a concrete, minimal set of **non-edge** artifacts to add that will immediately expand what you can do:

1. `occurrence_fact` (from SCIP): `(symbol_id, role_mask, file_id, span_id, container_id)`
2. `qualified_name_fact` (from LibCST): `(span_id, qualified_name[], fully_qualified_name[], resolution_source)`
3. `pattern_match_fact` (from tree-sitter queries): `(rule_id, capture_map, span_id, rule_settings)`
4. `embedded_dsl_fact` (from tree-sitter injections + included_ranges): `(host_span_id, language, embedded_span_id, embedded_tree_hash)`
5. `constant_fact` (from Python AST): `(span_id, const_kind, value_json, is_literal_only=true)`

Once you have these, many “advanced analyses” become *predictable joins* and group-bys rather than bespoke traversals.

---

If you want, I can take your current Phase5 outputs (as you have them) and propose a **delta plan** specifically for these *non-edge* index tables: which ones you already implicitly compute, which ones you’re missing, and the cleanest producers (SCIP vs LibCST vs tree-sitter vs `ast`) for each.


Below is a **Phase5 “non-edge index tables” delta plan**: what you already have as first-class tables, what you’re **already computing implicitly** (but hiding inside JSON blobs / edge tables / intermediate logic), what you’re **missing**, and the **cleanest producer** for each (SCIP vs LibCST vs tree-sitter vs ast).

I’m intentionally treating “index tables” as: **stable, query-friendly fact tables** that let you compute *many* advanced analyses as straightforward joins—rather than re-running bespoke visitors or decoding JSON repeatedly.

---

## Legend

* **HAVE** = already materialized as a Phase5 output table
* **IMPLICIT** = Phase5 already computes the info, but it’s embedded inside JSON fields, edge tables, or transient compute state (so it’s hard to join/reuse/debug)
* **MISSING** = not currently computed/persisted in Phase5
* **Primary producer** = the “best source of truth” among:

  * **SCIP** (scip-python): semantic symbols/occurrences/roles/relationships
  * **LibCST**: exact syntax + stable spans + pattern extraction via metadata/matchers
  * **tree-sitter**: cross-language syntax + query packs (locals/injections) for scalable pattern extraction
  * **python `ast`**: structural Python semantics + fallback parsing

---

## A. Your current Phase5 non-edge index tables (KEEP)

These are already solid “golden index” building blocks—keep them, and use them as join anchors.

### Repo / file / module identity

* **`core.modules` (HAVE)**
  Producer: *filesystem discovery* (not one of the 4 libs)
  Use: which files exist, language, module path, etc.

* **`core.file_state` (HAVE)**
  Producer: *filesystem*
  Use: content hash, mtime, size (incremental re-index gating)

* **`core.repo_map` (HAVE)**
  Producer: *filesystem / module mapping*
  Use: repo/module inventory snapshot

### Syntax & doc extraction

* **`core.ast_nodes` + `core.ast_metrics` (HAVE)**
  Producer: **python `ast`**
  Use: module/class/function defs, decorator ranges, basic structure/metrics

* **`core.cst_nodes` (HAVE)**
  Producer: **LibCST**
  Use: your “wide net” capture of many syntactic nodes with spans/snippets/parents/qnames

* **`core.docstrings` (HAVE)**
  Producer: (looks like Griffe-based extraction)
  Use: rich doc metadata (params/returns/raises/examples)

### Semantic symbol index

* **`core.scip_symbols`, `core.scip_occurrences`, `core.scip_symbol_information`, `core.scip_symbol_relationships`, `core.scip_diagnostics`, `core.scip_external_symbols`, `core.scip_module_state` (HAVE)**
  Producer: **SCIP / scip-python**
  Use: symbol graph + occurrences (+ roles bitmask) + semantic relationships

### Stable entity IDs

* **`core.goids` + `core.goid_crosswalk` (HAVE)**
  Producer: (your GOID builder; joins across AST/SCIP/CST identities)
  Use: the *best* “definition identity” table you have—absolutely keep.

---

## B. Index tables you already compute implicitly (MATERIALIZE)

This is the biggest immediate ROI: you already did the hard work, but the results are trapped in JSON and edge tables.

### B1) Callsite facts and resolution attempts

**What’s implicit today**

* `graph.call_graph_edges` contains **callsite location** + `evidence_json` (callee name, attr chain, strategy/confidence, maybe SCIP candidates).

**Delta**

* ✅ **Add `core.call_sites` (IMPLICIT → MATERIALIZE)**
  **Primary producer:** **LibCST** (fallback: `ast`)
  Why LibCST: it’s built for accurate positions and can do “index extraction” passes efficiently using `MetadataWrapper` + `PositionProvider`.
  Suggested columns:

  * `repo, commit, rel_path`
  * `callsite_id` (stable hash of file+range+callee form)
  * `enclosing_function_goid_h128` (nullable)
  * `start_line, start_col, end_line, end_col`
  * `callee_expr_kind` (Name/Attribute/Subscript/Call/etc)
  * `callee_base_name`, `attr_chain_json`
  * `arg_count`, `kwarg_names_json`
  * `raw_text_preview` (optional)

* ✅ **Add `core.call_resolution_attempts` (IMPLICIT → MATERIALIZE)**
  **Primary producer:** **LibCST + SCIP**
  One callsite can have multiple attempts (local table, import aliases, instance-method heuristic, SCIP-based, etc.). Persist them as rows:

  * `callsite_id`
  * `attempt_rank`
  * `resolved_goid_h128` (nullable)
  * `resolved_scip_symbol` (nullable)
  * `resolved_via` (strategy)
  * `confidence`
  * `evidence_json` (tiny, strategy-specific)

* Then treat **`graph.call_graph_edges`** as a *derived view*:

  * `call_sites` filtered to “best resolution where resolved_goid_h128 not null”.

**Why this is “best in class”**

* You can debug/iterate resolution logic without rewriting the call graph table.
* Other analyses (external dependency calls, config flow, ORM usage, etc.) can reuse `core.call_sites` directly.

> LibCST matchers make these collection passes clean and maintainable (pattern-based extraction of `Call()` inside `FunctionDef()`, etc.).

---

### B2) CFG statement stream normalization

**What’s implicit today**

* `graph.cfg_blocks.stmts_json` is the statement sequence per basic block.

**Delta**

* ✅ **Add `graph.cfg_statements` (IMPLICIT → MATERIALIZE)**
  **Primary producer:** **python `ast`** (because your CFG builder is AST-driven)
  Purpose: make “what statements are in this block?” a real join, not JSON parsing.
  Suggested columns:

  * `function_goid_h128, block_id, stmt_idx`
  * `stmt_kind` (Assign/Call/If/Return/Raise/…)
  * `start_line, start_col, end_line, end_col`
  * `text_preview` (optional)
  * `ast_node_key` (optional stable pointer)

This table becomes the bridge between CFG/DFG and “syntax facts” tables like calls/assignments/raises.

---

### B3) Config flow call chains

**What’s implicit today**

* `analytics.config_data_flow.call_chain_json` is a serialized chain (and you likely compute additional internal representation while building the flow).

**Delta**

* ✅ **Add `analytics.call_chains` (IMPLICIT → MATERIALIZE)**
  **Primary producer:** **SCIP + your call resolution** (but stored from the config flow pass)
  Suggested columns:

  * `call_chain_id` (already exists)
  * `chain_len`
  * `goid_chain_json` (or normalize further into a child table)
  * `root_entrypoint_goid` (if you have it)

* ✅ **Add `analytics.call_chain_steps` (optional but ideal)**

  * `call_chain_id, step_idx`
  * `caller_goid_h128, callee_goid_h128`
  * `callsite_id` (join to `core.call_sites`)

---

### B4) Normalize “evidence JSON” across analytics outputs

You have multiple tables that embed “evidence” as JSON (not just call graph):

* `analytics.external_dependency_calls.evidence_json`
* `analytics.function_effects.effects_json`
* `analytics.config_data_flow.evidence_json`
* (and others)

**Delta**

* ✅ **Add `analytics.evidence_items` (IMPLICIT → MATERIALIZE)**
  **Primary producer:** depends on source:

  * call evidence: **LibCST + SCIP**
  * effect evidence: **LibCST/ast (+ SCIP for resolved symbols)**
    Suggested columns:
  * `evidence_id`
  * `owner_table`, `owner_pk_json` (or typed columns per owner)
  * `rel_path`, `start_line`, `start_col`, `end_line`, `end_col`
  * `evidence_kind` (call/import/attribute/string_literal/etc)
  * `payload_json` (small)

This turns evidence into something you can **join, slice, and dedupe** across analyses.

---

## C. High-leverage index tables you are missing (ADD)

These are the “next tier” that dramatically increases the number of **high-certainty causal** analyses you can do *without inventing new edge types*.

### C1) Import statements + import bindings

Even if you have an import graph edge table, you still want the *raw import facts*.

* ✅ **Add `core.import_statements` (MISSING)**
  **Primary producer:** **LibCST**
  Why: exact syntax, aliasing, relative import level, star imports, multi-import statements.

* ✅ **Add `core.import_bindings` (MISSING)**
  **Primary producer:** **LibCST** (join with SCIP if you want symbol IDs)
  Each imported name produces bindings:

  * `bound_name` (local)
  * `imported_module`
  * `imported_qualname` / `imported_symbol` (if resolvable)
  * `asname`
  * span + `enclosing_scope_goid` (or module)

This table is a *massive* stabilizer for call resolution, reference resolution, and data lineage through imported helpers.

---

### C2) Name occurrences (token-level identifiers with spans)

You already have SCIP occurrences (which are semantic), and CST nodes (which are structural). You’re missing the “name token stream” with exact syntax spans.

* ✅ **Add `core.name_occurrences` (MISSING)**
  **Primary producer:** **LibCST**
  Use: join with SCIP occurrences by span to map syntax-level names to semantic symbols.

LibCST is particularly good here because metadata is node-identity keyed and designed for building indexers; it explicitly supports pull (`resolve`) and push (`METADATA_DEPENDENCIES`) access patterns for metadata like positions.

---

### C3) Assignments and binding sites

For *causal* analyses (dataflow), you want explicit binding facts.

* ✅ **Add `core.assignments` (MISSING)**
  **Primary producer:** **LibCST** (fallback: `ast`)
  Suggested columns:

  * `assignment_id`
  * `enclosing_function_goid_h128` (nullable for module level)
  * `lhs_kind` (Name/Attribute/Subscript/Tuple/List/…)
  * `lhs_names_json` (for destructuring)
  * `rhs_kind`
  * span fields
  * `is_augassign`, `is_annotated_assign`

* ✅ **Add `core.binding_events` (MISSING)**
  **Primary producer:** **LibCST**
  Normalize “this name is bound here” events:

  * defs from assignment targets, function params, `for` targets, `with ... as`, `except ... as`, imports, class/function defs, etc.

This table is the backbone for:

* precise **def-site** indexing
* local dataflow without needing a full DFG in every scenario

---

### C4) Attribute & subscript access facts

These enable high-certainty statements like:

* “function F reads config object X”

* “function F writes `obj.field`”

* “function F uses `df['col']` (column lineage)”

* ✅ **Add `core.attribute_accesses` (MISSING)**
  **Primary producer:** **LibCST**
  Columns:

  * `access_id`
  * `base_expr_kind`
  * `base_name` (if simple)
  * `attr_name`
  * `is_write_context` (if you track context)
  * span + enclosing GOID

* ✅ **Add `core.subscript_accesses` (MISSING)**
  **Primary producer:** **LibCST** (AST can help classify literal keys)
  Columns:

  * `base_expr`
  * `index_expr_kind`
  * `index_literal_value` (nullable)
  * span + enclosing GOID

---

### C5) Literal strings (for SQL, config keys, resource IDs)

You will keep finding “semantic facts embedded in strings”.

* ✅ **Add `core.string_literals` (MISSING)**
  **Primary producer:** **LibCST** (because you want raw string form and span)
  Optional enrichment by `ast`: attempt safe evaluation for constant folding.

Once you have this, you can layer:

* SQL extraction
* regex extraction
* file path usage
* environment variable key usage
* feature flag key usage

---

## D. Where tree-sitter becomes uniquely valuable (ADD, cross-language + embedded languages)

If you stay Python-only, LibCST + SCIP already cover most of what you want. Tree-sitter becomes *uniquely* valuable in two situations:

### D1) Cross-language “good enough” scope/def/ref scaffolding

Tree-sitter’s **locals query pack** has fixed capture semantics (`@local.scope`, `@local.definition`, `@local.reference`, `@ignore`) designed to support consistent scope/def/ref tracking.

* ✅ **Add `core.ts_scopes`, `core.ts_definitions`, `core.ts_references` (MISSING)**
  **Primary producer:** **tree-sitter**
  Use: for languages where you *don’t* have SCIP (or don’t trust the index), you still get:

  * a scope stack
  * definition table per scope
  * resolved reference stream via nearest enclosing def-by-text (best-effort)

This is not a replacement for SCIP, but it’s a **practical, scalable fallback** to widen language coverage.

### D2) Embedded language extraction (SQL-in-Python, regex-in-strings, etc.)

Tree-sitter’s **injections query pack** explicitly models “this node’s contents should be re-parsed as another language” via `@injection.content` and `@injection.language` (plus properties like `injection.language`, `injection.combined`, etc.).

* ✅ **Add `core.embedded_segments` (MISSING)**
  **Primary producer:** **tree-sitter**
  Columns:

  * `host_rel_path`, `host_span`
  * `embedded_language`
  * `embedded_text` (or hash + storage pointer)
  * `segment_id`

This is the cleanest path to “blank page, best-in-class” analysis of:

* SQL lineage from Python strings
* templated SQL in f-strings
* DSLs embedded in config blocks

---

## E. SCIP splits you should materialize as first-class index tables

SCIP occurrences already have:

* `symbol`
* `symbol_roles` bitmask (definition/reference/import/write/read/etc.)
* `range` in document

Right now you store occurrences as a single table. For causal analyses, it’s worth materializing role-filtered tables/views.

* ✅ **Add `core.symbol_def_occurrences` (IMPLICIT → MATERIALIZE)**
  Producer: **SCIP**
  Definition occurrences are explicitly encoded in `symbol_roles` (Definition bit).

* ✅ **Add `core.symbol_ref_occurrences` (IMPLICIT → MATERIALIZE)**
  Producer: **SCIP**

* ✅ **Add `core.symbol_read_occurrences`, `core.symbol_write_occurrences`, `core.symbol_import_occurrences` (IMPLICIT → MATERIALIZE)**
  Producer: **SCIP**
  These role splits are extremely helpful to increase certainty (read vs write is “causal”, not “metric”).

These can be materialized as:

* either physical tables, or
* views in DuckDB (often enough), depending on performance needs.

---

## F. “Cleanest producer” cheat sheet (rules of thumb)

### Use **SCIP** when you need:

* definitions vs references vs reads/writes/imports (roles bitmask)
* symbol identity stable across the repo
* cross-file semantic linking (relationships)

### Use **LibCST** when you need:

* exact source spans / stable text association
* syntactic detail (imports, decorators, calls, args, keywords)
* fast “index extraction” via metadata (`resolve`, `METADATA_DEPENDENCIES`)
* maintainable pattern extractors (matchers) for specific constructs

### Use **tree-sitter** when you need:

* cross-language parsing + uniform query-based extraction
* locals/injections query packs (scope/def/ref scaffolding; embedded languages)

### Use **python `ast`** when you need:

* cheap structural parsing
* fallback when CST parse fails
* semantics-oriented transforms (control-flow-friendly structure)

---

## Summary: the delta plan in one prioritized list

### P0 (do these first)

1. `core.call_sites` + `core.call_resolution_attempts` (materialize what call graph already computes)
2. `graph.cfg_statements` (normalize `stmts_json`)
3. `core.import_statements` + `core.import_bindings`
4. `core.name_occurrences`
5. `core.assignments` + `core.binding_events`
6. `core.attribute_accesses` + `core.subscript_accesses`

### P1 (unlock embedded lineage + richer causal facts)

7. `core.string_literals`
8. SCIP role-split tables/views: defs/refs/reads/writes/imports
9. `analytics.evidence_items` (normalize evidence JSON across analytics tables)

### P2 (cross-language expansion)

10. tree-sitter locals tables: `core.ts_scopes/definitions/references`
11. tree-sitter injections → `core.embedded_segments`

---

If you want, I can take this one step further and write the **exact schemas (columns + keys + recommended indexes)** for the P0/P1 tables in a Phase5-consistent style (including how each should join to `core.goid_crosswalk`, `core.scip_occurrences`, and your existing function/module identity tables), plus a “minimum viable extraction pass” outline for each producer (LibCST visitor + metadata deps, SCIP role filter, tree-sitter query pack runner).


Below is a **Phase5-consistent, concrete** set of **P0 (foundational)** and **P1 (enriching)** *non-edge index tables* that (a) make your analyses far less brittle, (b) let you leverage **LibCST metadata**, **SCIP resolution**, and **tree-sitter query packs** cleanly, and (c) join predictably to:

* `core.goid_crosswalk`
* `core.scip_occurrences`
* your existing identity tables (`core.goids`, `core.modules`, etc.)

I’m writing these in the same “`TableSchema(...)` shape” as `src/codeintel/core/schemas/output_registry.py`.

---

## Coordinate convention (critical for “high-certainty joins”)

To avoid the current “mixed base” problem (GOIDs/AST often 1-based lines vs SCIP 0-based lines vs LibCST 1-based lines vs tree-sitter 0-based points), these new tables assume:

* `start_line`, `start_col`, `end_line`, `end_col` are **0-based**
* ranges are **half-open** (end is exclusive) — aligns with LSP/SCIP expectations

**Bridging to existing tables (Phase5 today):**

* `core.scip_occurrences` already fits (0-based).
* `core.goids.start_line/end_line` and `core.goid_crosswalk.start_line/end_line` are currently **1-based line-only** (no cols). For containment joins, use:

  * `goid_start_line_0 = goid_start_line_1 - 1`
  * `goid_end_line_0 = goid_end_line_1 - 1` (and treat end as inclusive vs exclusive carefully)
* If you want maximum cleanliness: add a view `core.v_goids_0based` and `core.v_goid_crosswalk_0based` (or migrate the producers).

---

# P0 tables (must-have foundations)

These are the “small number of predictable joins” tables that everything else (CFG/DFG/PDG/slicing/join-lineage) can build on.

---

## P0.1 `core.parse_manifest`

**Purpose:** repo-wide *parse observability* (what parsed, how, why it failed). This becomes the gating input for downstream extraction passes and makes your pipeline debuggable.

```python
TableSchema(
    schema="core",
    name="parse_manifest",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("language", "VARCHAR", nullable=False),

        Column("parser", "VARCHAR", nullable=False),  # libcst | tree_sitter | ast | scip
        Column("ok", "BOOLEAN", nullable=False),

        Column("error_class", "VARCHAR"),
        Column("error_message", "VARCHAR"),
        Column("error_line", "INTEGER"),
        Column("error_col", "INTEGER"),

        # LibCST fidelity fields (also ok for Python-only)
        Column("encoding", "VARCHAR"),
        Column("default_indent", "VARCHAR"),
        Column("default_newline", "VARCHAR"),
        Column("has_trailing_newline", "BOOLEAN"),

        # Tree-sitter fidelity fields
        Column("grammar_name", "VARCHAR"),
        Column("grammar_semver", "VARCHAR"),
        Column("grammar_abi", "INTEGER"),

        Column("tool_version", "VARCHAR"),
        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "rel_path", "parser"),
    indexes=(
        Index("idx_core_parse_manifest_repo_commit", ("repo", "commit")),
        Index("idx_core_parse_manifest_path", ("rel_path",)),
        Index("idx_core_parse_manifest_ok", ("ok",)),
    ),
    description="Per-file parse status and fidelity metadata across parsers.",
)
```

**Joins**

* To modules: `rel_path = core.modules.path` (and `repo/commit` if you enforce them).
* Not usually joined to `goid_crosswalk` / `scip_occurrences` directly; it gates the tables that do.

---

## P0.2 `core.syntax_scopes`

**Purpose:** a *lexical scope index* (module/class/function/lambda/comprehension) with stable IDs. This is the backbone for high-certainty “name binding vs name use”.

```python
TableSchema(
    schema="core",
    name="syntax_scopes",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("language", "VARCHAR", nullable=False),
        Column("producer", "VARCHAR", nullable=False),  # libcst | tree_sitter

        Column("scope_id", "VARCHAR", nullable=False),
        Column("scope_kind", "VARCHAR", nullable=False),  # module|class|function|lambda|comprehension|...
        Column("scope_name", "VARCHAR"),
        Column("qualname", "VARCHAR"),
        Column("parent_scope_id", "VARCHAR"),

        # 0-based half-open
        Column("start_line", "INTEGER", nullable=False),
        Column("start_col", "INTEGER", nullable=False),
        Column("end_line", "INTEGER", nullable=False),
        Column("end_col", "INTEGER", nullable=False),

        # Optional linkage into your identity system
        Column("goid_h128", "DECIMAL(38,0)"),
        Column("goid_urn", "VARCHAR"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "rel_path", "producer", "scope_id"),
    indexes=(
        Index("idx_core_syntax_scopes_repo_commit_path", ("repo", "commit", "rel_path")),
        Index("idx_core_syntax_scopes_parent", ("parent_scope_id",)),
        Index("idx_core_syntax_scopes_goid", ("goid_h128",)),
    ),
    description="Lexical scope tree with stable scope IDs; supports binding/use disambiguation.",
)
```

**Joins**

* To function identity: `goid_h128 -> core.goids.goid_h128`
* To goid_crosswalk: `core.goids.urn = core.goid_crosswalk.goid` (join via goids)
* To module identity: `rel_path = core.goids.rel_path AND core.goids.kind='module'` (or via `core.modules`)

---

## P0.3 `core.syntax_bindings`

**Purpose:** every *binding event* that introduces/updates a name in a scope (assignments, imports, parameters, `with ... as`, `except ... as`, `for` targets, walrus, etc.).

```python
TableSchema(
    schema="core",
    name="syntax_bindings",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("language", "VARCHAR", nullable=False),
        Column("producer", "VARCHAR", nullable=False),  # libcst | tree_sitter

        Column("scope_id", "VARCHAR", nullable=False),
        Column("function_goid_h128", "DECIMAL(38,0)"),

        Column("binding_kind", "VARCHAR", nullable=False),
        # examples: assign|ann_assign|aug_assign|param|import|with_as|except_as|for_target|walrus|match_as|...

        Column("name", "VARCHAR", nullable=False),

        # span of the bound NAME token (0-based half-open)
        Column("name_start_line", "INTEGER", nullable=False),
        Column("name_start_col", "INTEGER", nullable=False),
        Column("name_end_line", "INTEGER", nullable=False),
        Column("name_end_col", "INTEGER", nullable=False),

        # span of the whole binding statement/pattern
        Column("stmt_start_line", "INTEGER", nullable=False),
        Column("stmt_start_col", "INTEGER", nullable=False),
        Column("stmt_end_line", "INTEGER", nullable=False),
        Column("stmt_end_col", "INTEGER", nullable=False),

        # optional: span of RHS/value (if present)
        Column("value_start_line", "INTEGER"),
        Column("value_start_col", "INTEGER"),
        Column("value_end_line", "INTEGER"),
        Column("value_end_col", "INTEGER"),

        Column("value_text_preview", "VARCHAR"),
        Column("extra_json", "JSON"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=(
        "repo",
        "commit",
        "rel_path",
        "producer",
        "name_start_line",
        "name_start_col",
        "binding_kind",
    ),
    indexes=(
        Index("idx_core_syntax_bindings_repo_commit_path", ("repo", "commit", "rel_path")),
        Index("idx_core_syntax_bindings_scope_name", ("scope_id", "name")),
        Index("idx_core_syntax_bindings_fn", ("function_goid_h128",)),
        Index("idx_core_syntax_bindings_name", ("name",)),
    ),
    description="Name binding events (assign/import/params/etc.) with scope context and spans.",
)
```

**Joins**

* To scopes: `(repo,commit,rel_path,producer,scope_id) -> core.syntax_scopes`
* To function identity:

  * direct: `function_goid_h128 -> core.goids`
  * fallback containment: `rel_path + stmt_start_line` within `core.goids` function range (with 0↔1 based adjustment)
* To `core.goid_crosswalk`: through `core.goids.urn`
* To SCIP: later via `core.scip_occurrences` overlap on the `name_*` span (see P0.6)

---

## P0.4 `core.syntax_name_uses`

**Purpose:** every name *use-site* (read/write/del/call-target name, etc.) with scope context. This is the other half of def-use.

```python
TableSchema(
    schema="core",
    name="syntax_name_uses",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("language", "VARCHAR", nullable=False),
        Column("producer", "VARCHAR", nullable=False),  # libcst | tree_sitter

        Column("scope_id", "VARCHAR", nullable=False),
        Column("function_goid_h128", "DECIMAL(38,0)"),

        Column("use_kind", "VARCHAR", nullable=False),  # read|write|del|call|type|import|...
        Column("name", "VARCHAR", nullable=False),

        # span of the NAME token
        Column("start_line", "INTEGER", nullable=False),
        Column("start_col", "INTEGER", nullable=False),
        Column("end_line", "INTEGER", nullable=False),
        Column("end_col", "INTEGER", nullable=False),

        # Filled by SCIP enrichment when available
        Column("scip_symbol", "VARCHAR"),
        Column("scip_roles", "INTEGER"),

        # Filled by GOID resolution (via P0.7)
        Column("resolved_goid_h128", "DECIMAL(38,0)"),
        Column("resolution_method", "VARCHAR"),
        Column("resolution_confidence", "DOUBLE"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=(
        "repo",
        "commit",
        "rel_path",
        "producer",
        "start_line",
        "start_col",
        "use_kind",
    ),
    indexes=(
        Index("idx_core_syntax_name_uses_repo_commit_path", ("repo", "commit", "rel_path")),
        Index("idx_core_syntax_name_uses_scope_name", ("scope_id", "name")),
        Index("idx_core_syntax_name_uses_fn", ("function_goid_h128",)),
        Index("idx_core_syntax_name_uses_scip_symbol", ("scip_symbol",)),
        Index("idx_core_syntax_name_uses_resolved_goid", ("resolved_goid_h128",)),
    ),
    description="Identifier uses with lexical scope, optional SCIP symbol, and optional GOID resolution.",
)
```

**Joins**

* To `core.scip_occurrences` (exact hit, best-case):

  ```sql
  ON o.repo=u.repo AND o.commit=u.commit AND o.rel_path=u.rel_path
  AND o.start_line=u.start_line AND o.start_col=u.start_col
  ```

  then copy `o.symbol -> u.scip_symbol`, `o.roles -> u.scip_roles`
* To GOIDs (internal resolution): `u.scip_symbol -> core.scip_symbol_goid_xref`
* To goid_crosswalk: `resolved_goid_h128 -> core.goids -> core.goid_crosswalk`

---

## P0.5 `core.import_facts`

**Purpose:** normalized import statements (for import graph correctness, alias resolution, and data lineage “where did this symbol come from?”).

```python
TableSchema(
    schema="core",
    name="import_facts",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("language", "VARCHAR", nullable=False),
        Column("producer", "VARCHAR", nullable=False),  # libcst | tree_sitter

        Column("scope_id", "VARCHAR", nullable=False),
        Column("function_goid_h128", "DECIMAL(38,0)"),

        Column("import_kind", "VARCHAR", nullable=False),  # import|from
        Column("module", "VARCHAR", nullable=False),       # e.g. "pkg.sub"
        Column("name", "VARCHAR", nullable=False),         # e.g. "x" or "*" (for star)
        Column("alias", "VARCHAR", nullable=False),        # "" if none
        Column("level", "INTEGER", nullable=False),        # 0 if absolute
        Column("is_star", "BOOLEAN", nullable=False),

        Column("stmt_start_line", "INTEGER", nullable=False),
        Column("stmt_start_col", "INTEGER", nullable=False),
        Column("stmt_end_line", "INTEGER", nullable=False),
        Column("stmt_end_col", "INTEGER", nullable=False),

        # optional span of the imported name token
        Column("name_start_line", "INTEGER"),
        Column("name_start_col", "INTEGER"),
        Column("name_end_line", "INTEGER"),
        Column("name_end_col", "INTEGER"),

        # optional SCIP (import site occurrence may exist depending on indexer)
        Column("scip_symbol", "VARCHAR"),
        Column("scip_roles", "INTEGER"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=(
        "repo",
        "commit",
        "rel_path",
        "producer",
        "stmt_start_line",
        "stmt_start_col",
        "import_kind",
        "module",
        "name",
        "alias",
        "level",
    ),
    indexes=(
        Index("idx_core_import_facts_repo_commit_path", ("repo", "commit", "rel_path")),
        Index("idx_core_import_facts_module", ("module",)),
        Index("idx_core_import_facts_alias", ("alias",)),
        Index("idx_core_import_facts_fn", ("function_goid_h128",)),
        Index("idx_core_import_facts_scip_symbol", ("scip_symbol",)),
    ),
    description="Normalized import facts with aliasing and scope context.",
)
```

**Joins**

* To import graph edges: this is the *raw truth*; `graph.import_graph_edges` should be derived from it (instead of reverse-engineering imports from other artifacts).
* To GOID resolution: if you can map imported symbol occurrences to SCIP symbols, you can resolve to internal GOIDs via `core.scip_symbol_goid_xref`.

---

## P0.6 `core.call_sites`

**Purpose:** call expressions as first-class facts (even when unresolved). This is what you need for data-operation detection (joins/merges), argument analysis, and “causal” relationships like “A calls B with keys X,Y”.

```python
TableSchema(
    schema="core",
    name="call_sites",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("language", "VARCHAR", nullable=False),
        Column("producer", "VARCHAR", nullable=False),  # libcst | tree_sitter

        Column("scope_id", "VARCHAR", nullable=False),
        Column("function_goid_h128", "DECIMAL(38,0)"),

        # span of the entire call expression
        Column("call_start_line", "INTEGER", nullable=False),
        Column("call_start_col", "INTEGER", nullable=False),
        Column("call_end_line", "INTEGER", nullable=False),
        Column("call_end_col", "INTEGER", nullable=False),

        Column("callee_kind", "VARCHAR", nullable=False),   # name|attribute|subscript|...
        Column("callee_text", "VARCHAR"),
        Column("callee_chain_json", "JSON"),                # e.g. ["df","merge"] or ["pkg","fn"]

        # optional: span of callee "name token" (best join to SCIP)
        Column("callee_start_line", "INTEGER"),
        Column("callee_start_col", "INTEGER"),
        Column("callee_end_line", "INTEGER"),
        Column("callee_end_col", "INTEGER"),

        Column("arg_count", "INTEGER", nullable=False),
        Column("kwarg_names", "JSON"),
        Column("has_star_args", "BOOLEAN", nullable=False),
        Column("has_star_kwargs", "BOOLEAN", nullable=False),
        Column("args_preview_json", "JSON"),

        # SCIP + GOID resolution
        Column("callee_scip_symbol", "VARCHAR"),
        Column("callee_scip_roles", "INTEGER"),
        Column("resolved_callee_goid_h128", "DECIMAL(38,0)"),
        Column("resolution_method", "VARCHAR"),
        Column("resolution_confidence", "DOUBLE"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=(
        "repo",
        "commit",
        "rel_path",
        "producer",
        "call_start_line",
        "call_start_col",
        "call_end_line",
        "call_end_col",
    ),
    indexes=(
        Index("idx_core_call_sites_repo_commit_path", ("repo", "commit", "rel_path")),
        Index("idx_core_call_sites_fn", ("function_goid_h128",)),
        Index("idx_core_call_sites_callee_scip", ("callee_scip_symbol",)),
        Index("idx_core_call_sites_resolved_callee", ("resolved_callee_goid_h128",)),
    ),
    description="Call expressions with callee chain, argument summary, and optional resolution.",
)
```

**Joins**

* To SCIP occurrences (callee token join):

  ```sql
  ON o.repo=c.repo AND o.commit=c.commit AND o.rel_path=c.rel_path
  AND o.start_line=c.callee_start_line AND o.start_col=c.callee_start_col
  ```
* To GOID resolution:

  * `callee_scip_symbol -> core.scip_symbol_goid_xref.scip_symbol`
* To call graph edges:

  * `graph.call_graph_edges` becomes a *projection* of `core.call_sites` (only rows where `resolved_callee_goid_h128 IS NOT NULL`) + evidence fields

---

## P0.7 `core.scip_symbol_goid_xref`

**Purpose:** the canonical bridge: **SCIP symbol → internal GOID** (and def location). This is what unlocks “resolved references” and makes cross-language joins possible.

```python
TableSchema(
    schema="core",
    name="scip_symbol_goid_xref",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),

        Column("scip_symbol", "VARCHAR", nullable=False),

        # Definition site from SCIP (0-based)
        Column("def_rel_path", "VARCHAR", nullable=False),
        Column("def_start_line", "INTEGER", nullable=False),
        Column("def_start_col", "INTEGER", nullable=False),
        Column("def_end_line", "INTEGER", nullable=False),
        Column("def_end_col", "INTEGER", nullable=False),

        # Matched internal entity (nullable for external symbols)
        Column("goid_h128", "DECIMAL(38,0)"),
        Column("goid_urn", "VARCHAR"),
        Column("goid_kind", "VARCHAR"),
        Column("qualname", "VARCHAR"),
        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "scip_symbol"),
    indexes=(
        Index("idx_core_scip_symbol_goid_repo_commit", ("repo", "commit")),
        Index("idx_core_scip_symbol_goid_path", ("def_rel_path",)),
        Index("idx_core_scip_symbol_goid_goid", ("goid_h128",)),
    ),
    description="Crosswalk from SCIP symbols to internal GOIDs via definition-site matching.",
)
```

**Joins**

* To SCIP: `scip_symbol` joins to `core.scip_symbols` / `core.scip_symbol_information` / `core.scip_occurrences`
* To GOIDs: `goid_h128 -> core.goids`
* To goid_crosswalk: via `core.goids.urn = core.goid_crosswalk.goid`

---

# P1 tables (high leverage enrichments)

These are “state-of-the-art enablers” for PDG-ish reasoning, slicing, and join-lineage precision—without adding speculative edges.

---

## P1.1 `core.call_arguments`

**Purpose:** argument-level facts (positional vs keyword vs splat) — essential for detecting join keys, dataset names, and parameter flows.

```python
TableSchema(
    schema="core",
    name="call_arguments",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("language", "VARCHAR", nullable=False),
        Column("producer", "VARCHAR", nullable=False),

        Column("call_start_line", "INTEGER", nullable=False),
        Column("call_start_col", "INTEGER", nullable=False),
        Column("call_end_line", "INTEGER", nullable=False),
        Column("call_end_col", "INTEGER", nullable=False),

        Column("arg_idx", "INTEGER", nullable=False),
        Column("arg_kind", "VARCHAR", nullable=False),  # positional|keyword|star|double_star
        Column("arg_name", "VARCHAR", nullable=False),  # "" if not keyword

        Column("arg_start_line", "INTEGER"),
        Column("arg_start_col", "INTEGER"),
        Column("arg_end_line", "INTEGER"),
        Column("arg_end_col", "INTEGER"),

        Column("value_text_preview", "VARCHAR"),
        Column("value_kind", "VARCHAR"),  # name|attr|literal|string|call|lambda|...
        Column("extra_json", "JSON"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=(
        "repo","commit","rel_path","producer",
        "call_start_line","call_start_col","call_end_line","call_end_col",
        "arg_idx",
    ),
    indexes=(
        Index("idx_core_call_args_callsite", ("repo","commit","rel_path","call_start_line","call_start_col")),
        Index("idx_core_call_args_name", ("arg_name",)),
    ),
    description="One row per call argument; enables parameter mapping and join-lineage extraction.",
)
```

**Joins**

* To `core.call_sites`: join on callsite PK columns
* To SCIP occurrences: if `value_kind=name|attr`, you can optionally add `value_scip_symbol` later using span matching

---

## P1.2 `core.function_parameters`

**Purpose:** parameter model for each function/method, enabling argument→parameter mapping and higher-certainty dataflow.

```python
TableSchema(
    schema="core",
    name="function_parameters",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),

        Column("param_idx", "INTEGER", nullable=False),
        Column("param_name", "VARCHAR", nullable=False),
        Column("param_kind", "VARCHAR", nullable=False),  # posonly|pos_or_kw|varargs|kwonly|varkw
        Column("has_default", "BOOLEAN", nullable=False),
        Column("default_text_preview", "VARCHAR"),
        Column("annotation_text", "VARCHAR"),

        Column("start_line", "INTEGER"),
        Column("start_col", "INTEGER"),
        Column("end_line", "INTEGER"),
        Column("end_col", "INTEGER"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "function_goid_h128", "param_idx"),
    indexes=(
        Index("idx_core_fn_params_goid", ("function_goid_h128",)),
        Index("idx_core_fn_params_name", ("param_name",)),
        Index("idx_core_fn_params_path", ("rel_path",)),
    ),
    description="Function signature parameters (name/kind/default/annotation) keyed by function GOID.",
)
```

**Joins**

* To function identity: `function_goid_h128 -> core.goids`
* To crosswalk: `core.goids.urn -> core.goid_crosswalk.goid`

---

## P1.3 `core.attribute_accesses`

**Purpose:** member access facts separate from calls (read/write/del). Helps causal reasoning like “writes df.columns” vs “calls df.merge”.

```python
TableSchema(
    schema="core",
    name="attribute_accesses",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("language", "VARCHAR", nullable=False),
        Column("producer", "VARCHAR", nullable=False),

        Column("scope_id", "VARCHAR", nullable=False),
        Column("function_goid_h128", "DECIMAL(38,0)"),

        Column("access_kind", "VARCHAR", nullable=False),  # read|write|del
        Column("base_text_preview", "VARCHAR"),
        Column("attr_name", "VARCHAR", nullable=False),

        # full expr span
        Column("expr_start_line", "INTEGER", nullable=False),
        Column("expr_start_col", "INTEGER", nullable=False),
        Column("expr_end_line", "INTEGER", nullable=False),
        Column("expr_end_col", "INTEGER", nullable=False),

        # attr token span (best for SCIP join)
        Column("attr_start_line", "INTEGER"),
        Column("attr_start_col", "INTEGER"),
        Column("attr_end_line", "INTEGER"),
        Column("attr_end_col", "INTEGER"),

        Column("attr_scip_symbol", "VARCHAR"),
        Column("attr_scip_roles", "INTEGER"),
        Column("resolved_attr_goid_h128", "DECIMAL(38,0)"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo","commit","rel_path","producer","expr_start_line","expr_start_col","expr_end_line","expr_end_col"),
    indexes=(
        Index("idx_core_attr_access_repo_commit_path", ("repo", "commit", "rel_path")),
        Index("idx_core_attr_access_fn", ("function_goid_h128",)),
        Index("idx_core_attr_access_attr", ("attr_name",)),
        Index("idx_core_attr_access_scip", ("attr_scip_symbol",)),
    ),
    description="Attribute/member access facts with optional SCIP and GOID resolution.",
)
```

---

## P1.4 `core.type_annotation_facts`

**Purpose:** typed facts (param/return/AnnAssign/type comments) used for higher-certainty resolution and “what does this symbol represent?”

```python
TableSchema(
    schema="core",
    name="type_annotation_facts",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("language", "VARCHAR", nullable=False),
        Column("producer", "VARCHAR", nullable=False),

        Column("owner_kind", "VARCHAR", nullable=False),  # param|return|ann_assign|type_comment|cast|...
        Column("owner_function_goid_h128", "DECIMAL(38,0)"),
        Column("owner_name", "VARCHAR"),

        Column("ann_start_line", "INTEGER", nullable=False),
        Column("ann_start_col", "INTEGER", nullable=False),
        Column("ann_end_line", "INTEGER", nullable=False),
        Column("ann_end_col", "INTEGER", nullable=False),

        Column("annotation_text", "VARCHAR"),
        Column("annotation_json", "JSON"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo","commit","rel_path","producer","owner_kind","ann_start_line","ann_start_col","ann_end_line","ann_end_col"),
    indexes=(
        Index("idx_core_type_ann_repo_commit_path", ("repo","commit","rel_path")),
        Index("idx_core_type_ann_owner_fn", ("owner_function_goid_h128",)),
        Index("idx_core_type_ann_owner_kind", ("owner_kind",)),
    ),
    description="Extracted type annotation facts (signature + AnnAssign + related typed constructs).",
)
```

---

## P1.5 `core.string_literal_facts`

**Purpose:** string literals (and optionally classification) for SQL lineage, dataset names, file paths, etc.

```python
TableSchema(
    schema="core",
    name="string_literal_facts",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("language", "VARCHAR", nullable=False),
        Column("producer", "VARCHAR", nullable=False),

        Column("scope_id", "VARCHAR", nullable=False),
        Column("function_goid_h128", "DECIMAL(38,0)"),

        Column("start_line", "INTEGER", nullable=False),
        Column("start_col", "INTEGER", nullable=False),
        Column("end_line", "INTEGER", nullable=False),
        Column("end_col", "INTEGER", nullable=False),

        Column("quote_kind", "VARCHAR"),
        Column("is_fstring", "BOOLEAN"),
        Column("value_preview", "VARCHAR"),
        Column("value_hash", "VARCHAR"),

        Column("classification", "VARCHAR"),  # sql|path|url|json|regex|other (optional)
        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo","commit","rel_path","producer","start_line","start_col","end_line","end_col"),
    indexes=(
        Index("idx_core_string_lits_repo_commit_path", ("repo","commit","rel_path")),
        Index("idx_core_string_lits_fn", ("function_goid_h128",)),
        Index("idx_core_string_lits_hash", ("value_hash",)),
        Index("idx_core_string_lits_class", ("classification",)),
    ),
    description="String literal facts with optional classification (e.g., SQL, file path).",
)
```

---

## P1.6 `core.decorator_facts`

**Purpose:** decorator applications (often *very* causal in Python: DI, routing, caching, retries, transactional boundaries).

```python
TableSchema(
    schema="core",
    name="decorator_facts",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("language", "VARCHAR", nullable=False),
        Column("producer", "VARCHAR", nullable=False),

        Column("owner_goid_h128", "DECIMAL(38,0)", nullable=False),
        Column("owner_kind", "VARCHAR", nullable=False),  # function|class|method

        Column("decorator_idx", "INTEGER", nullable=False),
        Column("decorator_text", "VARCHAR"),

        Column("start_line", "INTEGER", nullable=False),
        Column("start_col", "INTEGER", nullable=False),
        Column("end_line", "INTEGER", nullable=False),
        Column("end_col", "INTEGER", nullable=False),

        Column("scip_symbol", "VARCHAR"),
        Column("resolved_goid_h128", "DECIMAL(38,0)"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo","commit","owner_goid_h128","decorator_idx"),
    indexes=(
        Index("idx_core_decorators_owner", ("owner_goid_h128",)),
        Index("idx_core_decorators_repo_commit", ("repo","commit")),
        Index("idx_core_decorators_scip", ("scip_symbol",)),
    ),
    description="Decorator applications on functions/classes with optional SCIP/GOID resolution.",
)
```

---

# Minimum viable extraction passes (by producer)

This is the **fastest clean implementation path** that still gives you “best-in-class” leverage.

---

## A) LibCST visitor + metadata deps (Python)

### Metadata providers to enable high-certainty facts

Use `MetadataWrapper(module).resolve(...)` with at least:

* `PositionProvider` (exact spans)
* `ParentNodeProvider` (context classification: read vs write vs call-target)
* `QualifiedNameProvider` (stable qualnames)
* `ScopeProvider` (lexical scope IDs + binding resolution)
* (Optional but valuable) `ExpressionContextProvider`-style logic implemented via parent inspection if you don’t rely on a provider

### Minimal pass outline (single traversal)

For each Python file:

1. **Parse bytes-first**

   * write `core.parse_manifest(parser="libcst")`
2. **Compute a stable `scope_id`** for each scope node
   Recommended: hash of `(rel_path, scope_kind, start_line, start_col, end_line, end_col)`
3. Emit:

   * `core.syntax_scopes` for Module / ClassDef / FunctionDef / Lambda / Comprehension
   * `core.import_facts` for Import / ImportFrom (flatten aliases)
   * `core.syntax_bindings` for:

     * assignment targets (Assign/AnnAssign/AugAssign)
     * `for` targets
     * `with ... as`
     * `except ... as`
     * params (FunctionDef params)
     * imports (alias binds a local name)
   * `core.call_sites` (+ optionally `core.call_arguments`) for Call nodes
   * `core.syntax_name_uses` for Name nodes:

     * classify `use_kind` by parent:

       * Name in Store context ⇒ `write`
       * Name in Del context ⇒ `del`
       * Name in Call.func position ⇒ `call`
       * else ⇒ `read`
   * `core.attribute_accesses` for Attribute nodes (read/write/del from parent)
   * P1 extras:

     * `core.function_parameters` from FunctionDef params
     * `core.type_annotation_facts` from Annotation nodes
     * `core.string_literal_facts` from string nodes
     * `core.decorator_facts` from decorators on defs

**Normalization step:** convert LibCST’s 1-based line positions to **0-based** before writing.

---

## B) SCIP role filter (symbol truth + resolution)

You already ingest `core.scip_occurrences`, `core.scip_symbols`, etc. The “advanced leverage” is turning those into a *resolution layer*.

### Minimal pass outline

1. **Select definition occurrences** from `core.scip_occurrences`

   * filter rows where `roles` indicates “definition” (bitmask)
2. **Match definition-sites to GOIDs** to build `core.scip_symbol_goid_xref`
   Practical matching (P0 good enough, high certainty for functions/classes/modules):

   * join on:

     * `scip.def_rel_path = goids.rel_path`
     * `scip.def_start_line_0 + 1 = goids.start_line` (until you unify bases)
     * `goids.kind in ('module','class','function','method')`
   * store the result as xref
3. **Enrich syntax tables with SCIP** (optional to persist; you can also do views)

   * `core.syntax_name_uses`:

     * join by exact span to `core.scip_occurrences`
     * fill `scip_symbol`, `scip_roles`
     * then join to xref to fill `resolved_goid_h128` + `resolution_method="scip_def_site"`
   * `core.call_sites`:

     * join callee token span to `core.scip_occurrences`
     * map `callee_scip_symbol -> resolved_callee_goid_h128` via xref

This is the step that turns “syntax facts” into **high-certainty causal links**.

---

## C) tree-sitter query pack runner (multi-language scale)

Tree-sitter is best used as: **parse → run a curated query pack → emit captures** into the *same normalized tables* as Python.

### Minimal pass outline

1. For each non-Python file (or for all files if you want parity):

   * parse bytes with the language grammar
   * write `core.parse_manifest(parser="tree_sitter")` including grammar ABI/semver
2. Run query packs (one per language) that capture:

   * `function_definition`, `class_definition`  → `core.syntax_scopes` (+ possible `goid_h128` later)
   * `import_statement` / `import_from`         → `core.import_facts`
   * `call_expression`                          → `core.call_sites` (+ `core.call_arguments` if extracted)
   * `identifier` in usage contexts             → `core.syntax_name_uses`
   * assignment/declaration patterns            → `core.syntax_bindings`
   * member access nodes                        → `core.attribute_accesses`
   * string literal nodes                       → `core.string_literal_facts`
3. Normalize spans:

   * tree-sitter points are already **0-based row/column**, so it naturally matches the canonical convention
4. If SCIP exists for that language:

   * do the same span join enrichment as in section B

---

# How these tables join (quick “recipes”)

## Join any syntax fact → enclosing function (GOID)

Preferred (when you store `function_goid_h128`): direct join to `core.goids`.

Fallback (if not stored yet): containment join (with base conversion).

## Join syntax fact → SCIP symbol

Span equality join to `core.scip_occurrences` (exact match is the highest-certainty mode).

## Join SCIP symbol → internal GOID

Join to `core.scip_symbol_goid_xref` (P0.7). This becomes your canonical “resolution service in SQL”.

## Join internal GOID → crosswalk

`core.goids.urn = core.goid_crosswalk.goid`

---

# Why this is “state of the art” for high-certainty (without “graph metrics”)

These tables let you compute *causal relationships* as **deterministic relational derivations**:

* **Def-use chains** (scope-aware) from `syntax_bindings` + `syntax_name_uses`
* **Call causality** (callee + argument semantics) from `call_sites` + `call_arguments` + `function_parameters`
* **Import causality** (alias, provenance) from `import_facts`
* **Decorator causality** (framework behavior) from `decorator_facts`
* **String-literal causality** (SQL/query/data source) from `string_literal_facts`

…and then edges (CFG/DFG/PDG) become *downstream projections* instead of the “source of truth”.

---

If you want the next increment, I can also propose the **exact Hamilton target boundaries** for each table (e.g., `t__syntax_index__extract`, `t__scip_resolution__xref`, `t__syntax_enrich__scip`, etc.) in a way that aligns with your current `build/hamilton/native/*` patterns and keeps expensive enrichments feature-flagged.


Below is a concrete “next increment” plan: **exact Hamilton target boundaries** (anchor + tool/compute/helper nodes) for a **Phase5-consistent set of non-edge index tables**, with **expensive enrichments isolated into feature-flag targets** and naming aligned to your existing `build/hamilton/native/*` style.

I’m going to assume the same P0/P1 index-table set we’ve been discussing (syntax/span/defs/refs/calls/imports + SCIP↔GOID↔span crosswalks + optional enrichments + optional tree-sitter query pack). If you want to adjust table names, you can do that mechanically—the boundary pattern stays identical.

---

## Conventions to match your Phase5 Hamilton style

These mirror what you already do in:

* `native/ingestion/extraction_targets.py` (AST/CST/docstrings)
* `native/graphs/graph_targets.py` (GOIDs + crosswalk)
* `native/graphs/call_graph.py` / `native/graphs/cfg_dfg.py` (marker outputs + tool extract + anchor)

**Naming**

* **Target anchor**: `t__<target>` decorated with `@codeintel_target(domain=..., target=...)`
* **Tool node(s)**: `t__<target>__<step>` decorated with `@tag_tool(domain=..., target=...)`
* **Row materializers**: `<target>__<table>_rows` decorated with `SaveToObjectMetadataDecorator([DuckDBRowsSaver], table_key=...)` and `@tag_compute(...)`
* **Materialization collectors** (if you use `record_from_duckdb_materializations` pattern): `<target>__materializations` returns `{table_key: m__...}`

**Feature-flag rule**

* Anything “expensive” should be a **separate target** (so build specs can include/exclude it), not a conditional branch inside a P0 target that downstream depends on.

---

## The target set (P0/P1) and what each owns

### P0 targets (should be “always on” for best-in-class causal analyses)

1. **`syntax_index`** (domain: `ingestion`)
   Produces the *canonical syntax fact tables* from **LibCST (primary)** and **stdlib `ast` (secondary)**.

2. **`scip_resolution`** (domain: `graphs`)
   Produces *crosswalks* that make SCIP usable as a “resolver” for syntax-level facts:

   * SCIP symbol ↔ GOID
   * SCIP occurrence ↔ span_id (and optionally ↔ GOID)

### P1 targets (feature-flagged; no P0 target depends on them)

3. **`syntax_enrich`** (domain: `graphs`)
   Joins syntax facts + SCIP resolution to produce “resolved” versions (call targets, ref targets, etc.).

4. **`tree_sitter_index`** (domain: `ingestion`)
   Runs **tree-sitter query packs** to extract *language-agnostic* syntax facts (mostly for cross-language lineage).

5. **`syntax_index_ext`** (domain: `ingestion`)
   Any “wide” tables that explode row counts or require deep metadata passes (tokens, normalized expr text, literal inventories, etc.).

---

## Table → owner target mapping (with exact boundary node names)

### P0: Syntax fact tables (owned by `t__syntax_index`)

These are the tables that turn “dozens of brittle joins” into a few predictable ones.

| Table key             | Owner target   | Tool boundary              | Materializing compute node(s)             |
| --------------------- | -------------- | -------------------------- | ----------------------------------------- |
| `core.syntax_spans`   | `syntax_index` | `t__syntax_index__extract` | `syntax_index__core__syntax_spans_rows`   |
| `core.syntax_scopes`  | `syntax_index` | `t__syntax_index__extract` | `syntax_index__core__syntax_scopes_rows`  |
| `core.syntax_defs`    | `syntax_index` | `t__syntax_index__extract` | `syntax_index__core__syntax_defs_rows`    |
| `core.syntax_refs`    | `syntax_index` | `t__syntax_index__extract` | `syntax_index__core__syntax_refs_rows`    |
| `core.syntax_calls`   | `syntax_index` | `t__syntax_index__extract` | `syntax_index__core__syntax_calls_rows`   |
| `core.syntax_imports` | `syntax_index` | `t__syntax_index__extract` | `syntax_index__core__syntax_imports_rows` |

> **Why these are one target:** they share the same LibCST parse + metadata providers; splitting them forces re-parse or caching complexity.

---

### P0: SCIP resolution crosswalk tables (owned by `t__scip_resolution`)

| Table key                   | Owner target      | Tool boundary              | Materializing compute node(s)                      |
| --------------------------- | ----------------- | -------------------------- | -------------------------------------------------- |
| `core.scip_symbol_xref`     | `scip_resolution` | `t__scip_resolution__xref` | `scip_resolution__core__scip_symbol_xref_rows`     |
| `core.scip_occurrence_xref` | `scip_resolution` | `t__scip_resolution__xref` | `scip_resolution__core__scip_occurrence_xref_rows` |

> This is the “make SCIP usable for causal joins” step: it produces deterministic join keys so downstream analyses don’t guess.

---

### P1: Resolved/enriched syntax fact tables (owned by `t__syntax_enrich`)

| Table key                      | Owner target    | Tool boundary            | Materializing compute node(s)                       |
| ------------------------------ | --------------- | ------------------------ | --------------------------------------------------- |
| `core.syntax_refs_resolved`    | `syntax_enrich` | `t__syntax_enrich__scip` | `syntax_enrich__core__syntax_refs_resolved_rows`    |
| `core.syntax_calls_resolved`   | `syntax_enrich` | `t__syntax_enrich__scip` | `syntax_enrich__core__syntax_calls_resolved_rows`   |
| `core.syntax_imports_resolved` | `syntax_enrich` | `t__syntax_enrich__scip` | `syntax_enrich__core__syntax_imports_resolved_rows` |

> **Feature-flag justification:** these joins can be big (especially `calls_resolved`) and can require heuristics. Keep them optional until you’re happy with correctness.

---

### P1: Tree-sitter query-pack tables (owned by `t__tree_sitter_index`)

| Table key              | Owner target        | Tool boundary                   | Materializing compute node(s)                   |
| ---------------------- | ------------------- | ------------------------------- | ----------------------------------------------- |
| `core.ts_captures`     | `tree_sitter_index` | `t__tree_sitter_index__extract` | `tree_sitter_index__core__ts_captures_rows`     |
| `core.ts_parse_errors` | `tree_sitter_index` | `t__tree_sitter_index__extract` | `tree_sitter_index__core__ts_parse_errors_rows` |

> Tree-sitter’s big unlock is **repeatable query packs** across languages. Treat this as an optional “multi-language intake layer.”

---

### P1: Extended syntax indexing tables (owned by `t__syntax_index_ext`)

| Table key               | Owner target       | Tool boundary                  | Materializing compute node(s)                   |
| ----------------------- | ------------------ | ------------------------------ | ----------------------------------------------- |
| `core.syntax_tokens`    | `syntax_index_ext` | `t__syntax_index_ext__extract` | `syntax_index_ext__core__syntax_tokens_rows`    |
| `core.syntax_literals`  | `syntax_index_ext` | `t__syntax_index_ext__extract` | `syntax_index_ext__core__syntax_literals_rows`  |
| `core.syntax_types`     | `syntax_index_ext` | `t__syntax_index_ext__extract` | `syntax_index_ext__core__syntax_types_rows`     |
| `core.syntax_expr_norm` | `syntax_index_ext` | `t__syntax_index_ext__extract` | `syntax_index_ext__core__syntax_expr_norm_rows` |

> These tend to be row-explosive and/or require extra passes; don’t make core builds pay for them.

---

## Exact Hamilton boundary skeletons per target (drop-in patterns)

### 1) `syntax_index` (domain `ingestion`)

**File**: `src/codeintel/build/hamilton/native/ingestion/syntax_index_targets.py`

**Dependencies** (hard):

* `t__modules` (for module list + change sets / scope filtering)

**Nodes**

* `@tag_tool(domain="ingestion", target="syntax_index")`

  * `def t__syntax_index__extract(env, graph, t__modules, module_records, ...) -> SyntaxIndexExtractResult`
  * **Output**: a payload with `{table_key: row_tuples}` plus an `ExecutionResult`

* `@SaveToObjectMetadataDecorator(... table_key="core.syntax_spans" ...)`

  * `def syntax_index__core__syntax_spans_rows(t__syntax_index__extract) -> rows|None`

* repeat for `syntax_scopes`, `syntax_defs`, `syntax_refs`, `syntax_calls`, `syntax_imports`

* `def syntax_index__materializations(m__core__syntax_spans, ... ) -> dict[str, MaterializationMetadata]`

* `@codeintel_target(domain="ingestion", target="syntax_index", spec=TargetSpecDescriptor(...CPU...))`

  * `def t__syntax_index(env, graph, t__syntax_index__extract, syntax_index__materializations, syntax_index__hash_options) -> TargetRunRecord`

**Why this boundary is “clean”**

* Everything inside is a *single LibCST parse pipeline*.
* Downstream graphs/analytics stop parsing source files repeatedly.

---

### 2) `scip_resolution` (domain `graphs`)

**File**: `src/codeintel/build/hamilton/native/graphs/scip_resolution_targets.py`

**Dependencies** (hard):

* `t__scip` (SCIP tables exist)
* `t__goids` (GOIDs exist)

**Nodes**

* `@tag_tool(domain="graphs", target="scip_resolution")`

  * `def t__scip_resolution__xref(env, q__core__scip_occurrences, q__core__scip_symbol_information, q__core__goids, q__core__goid_crosswalk, t__scip, t__goids) -> ScipResolutionXrefResult`

* `@SaveToObjectMetadataDecorator(... table_key="core.scip_symbol_xref" ...)`

  * `def scip_resolution__core__scip_symbol_xref_rows(t__scip_resolution__xref) -> rows|None`

* `@SaveToObjectMetadataDecorator(... table_key="core.scip_occurrence_xref" ...)`

  * `def scip_resolution__core__scip_occurrence_xref_rows(t__scip_resolution__xref) -> rows|None`

* `scip_resolution__materializations(...) -> dict[...]`

* `@codeintel_target(domain="graphs", target="scip_resolution")`

  * `def t__scip_resolution(...) -> TargetRunRecord`

**Why graphs-domain?**

* It’s “identity & resolution,” similar in spirit to `goids` (which is already in `graphs` and writes `core.goids` + `core.goid_crosswalk`).

---

### 3) `syntax_enrich` (domain `graphs`) — feature flagged

**File**: `src/codeintel/build/hamilton/native/graphs/syntax_enrich_targets.py`

**Dependencies**

* hard: `t__syntax_index`, `t__scip_resolution`
* optional: `t__goids` if you want to attach GOID-only enrichments

**Nodes**

* `@tag_tool(domain="graphs", target="syntax_enrich")`

  * `def t__syntax_enrich__scip(env, q__core__syntax_calls, q__core__syntax_refs, q__core__scip_occurrence_xref, q__core__scip_symbol_xref, t__syntax_index, t__scip_resolution) -> SyntaxEnrichResult`

* Materializers:

  * `syntax_enrich__core__syntax_calls_resolved_rows(...)`
  * `syntax_enrich__core__syntax_refs_resolved_rows(...)`
  * `syntax_enrich__core__syntax_imports_resolved_rows(...)`

* Anchor: `t__syntax_enrich`

**Feature flag mechanism**

* Make the target excluded by default in your buildspec / default plan.
* If you additionally want a runtime flag, implement `SyntaxEnrichOptions(enabled: bool=False)` and have `t__syntax_enrich__scip` return `ExecutionResult.skip(...)` when disabled.

---

### 4) `tree_sitter_index` (domain `ingestion`) — feature flagged

**File**: `src/codeintel/build/hamilton/native/ingestion/tree_sitter_index_targets.py`

**Dependencies**

* hard: `t__modules`
* optional: `t__repo_map` if you want language detection from repo map conventions

**Nodes**

* `@tag_tool(domain="ingestion", target="tree_sitter_index")`

  * `def t__tree_sitter_index__extract(env, t__modules, module_records, ...) -> TreeSitterExtractResult`

* Materializers:

  * `tree_sitter_index__core__ts_captures_rows(...)`
  * `tree_sitter_index__core__ts_parse_errors_rows(...)`

* Anchor: `t__tree_sitter_index`

**Why isolated?**

* Tree-sitter adds a runtime dependency footprint and can be CPU-heavy.
* It’s also “optional value” until you’re truly going cross-language.

---

### 5) `syntax_index_ext` (domain `ingestion`) — feature flagged

**File**: `src/codeintel/build/hamilton/native/ingestion/syntax_index_ext_targets.py`

**Dependencies**

* hard: `t__syntax_index` (so you can reuse span_ids/kinds and only add more detail)
* optional: `t__typing` if you want to fuse Pyright outputs later

**Nodes**

* `@tag_tool(domain="ingestion", target="syntax_index_ext")`

  * `def t__syntax_index_ext__extract(...) -> SyntaxIndexExtResult`

* Materializers:

  * `syntax_index_ext__core__syntax_tokens_rows`
  * `syntax_index_ext__core__syntax_literals_rows`
  * `syntax_index_ext__core__syntax_types_rows`
  * `syntax_index_ext__core__syntax_expr_norm_rows`

* Anchor: `t__syntax_index_ext`

---

## How these boundaries simplify/refactor existing targets (optional but highly recommended)

Once these tables exist, several of your current graph targets can stop parsing source repeatedly:

* **`call_graph`** can consume:

  * `core.syntax_calls` (or `core.syntax_calls_resolved` when enabled)
  * plus `core.goids` / `core.goid_crosswalk` for node identity
    Instead of re-running LibCST/AST walks inside `graphs/call_graph.py`.

* **`import_graph`** can consume:

  * `core.syntax_imports` (and optionally `imports_resolved`)
    Instead of re-parsing imports.

* **CFG/DFG** can use:

  * `core.syntax_spans` + scope info as stable anchors for block boundaries / statement ranges
    (Even if the underlying CFG algorithm still parses AST, your joins become deterministic.)

This is the main payoff of having explicit boundaries: **parsing happens once**, everything else becomes **joins over golden tables**.

---

## Minimal “target inclusion” recommendation (so feature flags don’t break the world)

A practical default build set for “best causal certainty” (without optional explosion):

**Default ON (P0)**

* `modules`
* `scip`
* `goids`
* `syntax_index`
* `scip_resolution`
* plus your existing `cfg/dfg/call_graph/...` as desired

**Default OFF (P1)**

* `syntax_enrich`
* `tree_sitter_index`
* `syntax_index_ext`

This ensures:

* P0 analyses have deterministic join keys
* expensive enrichments don’t become hidden prerequisites

---

If you want, I can take this one more step and write:

* the **exact `TargetSpecDescriptor`** values (resources/execution) for each new target (mirroring how you do CPU-heavy AST/SCIP),
* the **exact `options` dataclasses** to implement the feature flags cleanly (`enabled`, `scope_paths`, `max_files`, etc.),
* and a **“refactor checklist”**: which existing targets (`call_graph`, `import_graph`, specific analytics targets) should switch to reading these golden tables first.
