According to a document from **January 5, 2026**, you already have the right *architectural constraint* for getting to “best-in-class” organization: **treat build as a pure, Arrow-in/Arrow-out computation layer, and keep DuckDB/storage as an optional downstream adapter**—no `build -> storage` imports in the CPG compute path.

Below is the most conservative, end-to-end re-organization I’d recommend for **AST + (LibCST→syntax) + Tree-sitter + SCIP ingest → CPG output** that will make the Hamilton computations **more robust, faster, and clearer** *without changing your existing output tables* (`graph.cpg_nodes`, `graph.cpg_edges`)—but with an internal structure that’s “best-in-class” for extensibility.

---

## 1) What’s currently holding you back (based on `src.zip`)

### A. CPG assembly is monolithic and row-loop heavy

Your `src/codeintel/build/hamilton/native/graphs/cpg/_legacy.py` does the right *things* functionally, but its shape causes technical debt:

* It mixes **ID policy**, **plane mapping**, **node emission**, **edge emission**, **joining**, **fallback logic**, and **extras serialization** in one place.
* Many conversions are `for row in table_rows(...)` / `to_pylist()` + `Table.from_pylist(rows)` patterns. This is the most common source of:

  * slow runtime (Python object churn),
  * memory blowups (materializing huge Python lists),
  * unclear execution boundaries (hard to reason about “where time goes”).

### B. IDs are recomputed repeatedly

Edges recompute node IDs by reconstructing PK payloads again and again, rather than joining through a single “anchor map”. That’s pure overhead and is a major reason the CPG stage feels slow and hard to refactor.

### C. “Plane” boundaries exist conceptually but not structurally

You already have distinct upstream planes (syntax/AST, tree-sitter, SCIP resolution, symtable, bytecode, CFG/DFG/CDG/PDG, inspect). Your own doc emphasizes exploiting these planes and adding coverage gates. 
But in the CPG compute layer, those planes are not expressed as isolated modules with clear contracts.

---

## 2) The best-in-class organization: “Plane-first → Anchor maps → Join-based assembly”

This is the core idea:

### **Phase 1: Make each upstream plane produce stable “anchors”**

For every upstream “entity table” that becomes a CPG node (syntax_node, ast_node, scip_symbol, goid, cfg_block, bc_instr, inspect_object, ts_token, …), compute **exactly once**:

* `cpg_node_id = stable_decimal_id({"table_key": ..., "pk": ...})`
* `source_pk_json = encode_payload(pk)`
* Minimal span fields: `rel_path`, `start_byte`, `end_byte` when available

This becomes an **anchor map** (a small table with PK columns + `cpg_node_id`) used by edges and other nodes.

### **Phase 2: Build node tables by projection, not by re-serialization**

Instead of “row dict → pylist → from_pylist”, build nodes by:

* selecting columns,
* adding constant columns,
* attaching `cpg_node_id` and `source_pk_json` from the anchor maps,
* only serializing `extras_json` where truly needed (and preferably *only* for node kinds that benefit).

### **Phase 3: Build edges via Arrow joins against anchor maps**

Edges should almost never loop rows. They should:

* join the edge source table to **left-anchor-map** to get `src_cpg_node_id`
* join to **right-anchor-map** to get `dst_cpg_node_id`
* attach constant edge_kind/layer, compute ordinal deterministically
* dedupe + contract align

This one change (anchor maps + join-based edge emission) is the single biggest improvement you can make for **speed** and **clarity** while staying conservative.

---

## 3) Proposed end-to-end DAG shape (ingest → normalize → resolve → graph assembly)

Below is a “best-in-class” *logical* pipeline. You already have most of the ingredients; this is mainly about organizing compute boundaries:

### Layer 0 — Repo & file plan

* `core.modules`, `core.file_state`, `core.repo_map`
  Purpose: stable module inventory + file digests for incremental work.

### Layer 1 — Parse & extract planes (tool outputs)

* Python AST → `core.ast_nodes`, `core.ast_metrics`
* LibCST/syntax index → `core.syntax_nodes`, `core.syntax_edges`, `core.syntax_*` facts
* Tree-sitter → `core.ts_nodes`, `core.ts_edges`, `core.ts_tokens`, `core.ts_trivia`, captures, parse errors
* SCIP → `core.scip_*` tables

### Layer 2 — Canonicalization & indexing

* `core.file_line_index` (byte→(line,col) map; used everywhere for robust span mapping)
* “normalize” transforms: standardize `rel_path`, integer casts, null filtering, etc.

### Layer 3 — Cross-plane resolution

* `core.goids`, `core.goid_crosswalk`
* `core.scip_occurrence_span_xref`, `core.scip_occurrence_syntax_xref`, `core.scip_symbol_goid_xref`
* `core.syntax_defs_resolved`, `core.syntax_refs_resolved`, `core.syntax_calls_resolved`, `core.syntax_imports_resolved`
* tree-sitter weld alignment `core.ts_syntax_node_xref`, coverage tables

### Layer 4 — Flow graphs

* `graph.cfg_blocks`, `graph.cfg_edges`
* `graph.dfg_edges`
* `graph.cdg_edges`
* `graph.pdg_edges`

### Layer 5 — CPG assembly (this is what we restructure)

* **Anchor maps**
* **Plane node emission**
* **Plane edge emission**
* Union → `graph.cpg_nodes`, `graph.cpg_edges`

This matches the “planes + coverage gating” philosophy in your own doc. 

---

## 4) Concrete package layout I recommend inside `src/codeintel/build/hamilton/native`

This is the conservative refactor that preserves outputs and target names but makes the codebase clean:

### Keep:

* `src/codeintel/build/hamilton/native/ingestion/*` (ingest + resolution)
* `src/codeintel/build/hamilton/native/graphs/*` (graph targets)

### Refactor CPG into a package with explicit “planes”

Create:

```
src/codeintel/build/hamilton/native/graphs/cpg2/
  __init__.py
  options.py          # reuse existing CpgOptions; just organize
  ids.py              # stable id + ordinal helpers (single source of truth)
  anchors.py          # anchor map builders for each source table

  planes/
    __init__.py
    syntax.py         # syntax nodes/edges + syntax facts
    ast.py            # ast nodes as CPG nodes (minimal extras)
    scip.py           # scip symbol nodes + occurrence edges if desired
    goids.py          # goid nodes + containment edges
    treesitter.py     # ts nodes/tokens/trivia nodes + edges
    py_sym.py         # symtable scopes/bindings + edges
    bytecode.py       # bc units/instr/blocks + CFG edges
    flow.py           # cfg/dfg/cdg/pdg edges emission into CPG edge format
    inspect.py        # inspect objects/signatures + overlay edges

  assemble.py         # concatenate/dedupe/align for final cpg_nodes/cpg_edges
```

### Hamilton discovery guardrails (new)

Native module discovery imports every **package** under `native/graphs`. That means any callables
exported from `cpg2/__init__.py` become Hamilton nodes by default. To avoid collisions and
unintended nodes:

* keep `cpg2/__init__.py` minimal (docstring only, no re-exports),
* if internal frames must be discoverable, name them with `cpg2_nodes__*` /
  `cpg2_edges__*` prefixes and only re-export from `cpg/` explicitly.
* never define `cpg_nodes` or `cpg_edges` inside `cpg2`.

Then change `src/codeintel/build/hamilton/native/graphs/cpg/__init__.py` to **re-export only the
stable DAG surface** from `cpg2` (or slowly migrate pieces), so the target spec remains stable.

### CPG2 naming + aggregators (new)

Adopt a strict naming convention inside `cpg2`:

* `cpg2_nodes__*` for node-frame emitters.
* `cpg2_edges__*` for edge-frame emitters.
* no unprefixed callables that Hamilton might treat as nodes.

Add explicit aggregator nodes:

* `cpg2_nodes__frames`: pure concat/dedupe/align of node frames.
* `cpg2_edges__frames`: pure concat/dedupe/align of edge frames.

Then make `cpg.cpg_nodes` and `cpg.cpg_edges` depend **only** on those aggregators, so
assembly is centralized and diagnostics are easy to attach.

---

## 5) The anchor-map pattern in detail (the “speed + clarity” unlock)

### 5.0 Anchor registry (identity + lookup keys) (new)

Create an explicit registry table (constant) in `cpg2/anchors.py` that lists:

* identity keys per source table (the full PK used for `stable_cpg_id`)
* lookup keys per edge family (the minimal columns available in edge tables)

This prevents drift and keeps joins deterministic across planes.

### 5.1 Anchor map contract (internal, not a saved table)

For each source table that becomes CPG nodes, define a small internal Arrow table:

**Example: syntax node anchor map**
Columns:

* `repo`, `commit`, `rel_path`, `producer`, `node_id`  (the actual PK parts)
* `cpg_node_id`
* `source_pk_json` (optional here; nodes may need it)

You compute this once. Everything else joins to it.

### 5.2 Node emission becomes “table projection + computed columns”

For `graph.cpg_nodes` you still need:

* `node_kind`
* `source_table_key`
* `source_pk_json`
* `rel_path`, `start_byte`, `end_byte`
* `extras_json` (only where valuable)

But now you produce it like:

* start from the upstream table (already Arrow)
* project columns
* compute `cpg_node_id` + `source_pk_json` arrays column-wise (no dict-per-row)
* append constant columns (`node_kind`, `source_table_key`)
* attach extras via *one* serializer pass per table (or skip if low value)

### 5.3 Edge emission becomes “2 joins + constant columns”

**Example: syntax AST edges**
Inputs:

* `core.syntax_edges` with parent/child IDs
* `anchor_syntax_nodes` mapping

Compute:

* join to anchor on parent key → `src_cpg_node_id`
* join to anchor on child key → `dst_cpg_node_id`
* append `edge_kind="AST"`, `edge_layer="SYNTAX"`
* `ordinal = child_ordinal` (already present)
* `extras_json = NULL`
* dedupe + align

No row loops.

### 5.4 Canonical column normalization at anchor boundaries

To prevent recurring schema mismatches (e.g., `goid_h128` vs `function_goid_h128`,
`string` vs `string_view`), add a small normalization step per anchor map:

* Normalize column names to a single canonical vocabulary (e.g., `goid_h128` as the
  core GOID column; `function_goid_h128` only for function-scoped analytics tables).
* Cast Arrow types to the canonical contract once at the anchor boundary.
* When a table uses legacy names, alias to canonical columns before joins.

This makes every downstream join deterministic and avoids late-stage failures from
schema drift.

---

## 6) Determinism + robustness rules (make the pipeline “production-grade” even during redesign)

Your doc calls out determinism + “no storage dependency” as ground rules.
So I’d codify these as *explicit invariants* in the CPG layer:

### A. Deterministic IDs and ordinals

* Node IDs: always `stable_decimal_id({"table_key": source_table_key, "pk": pk_dict})`
* Edge ordinals:

  * Prefer a **natural ordinal** column from the upstream table (child_ordinal, instruction_offset ordering, arg_index, etc.)
  * Otherwise: compute `stable_int_hash({...}) % ORDINAL_MOD` from a canonical payload

### B. Plane outputs are total functions (never crash the graph)

Every plane emitter should follow:

* If required input columns are missing → return `empty_table_for_table(key)`
* If joins fail validation → fall back to a safe mode (left join + filter invalid IDs)
* Always return a valid Arrow table aligned to the contract

### C. Coverage outputs (cheap, high-value)

Even if you’re “conservative”, adding **coverage reporting** is one of the easiest improvements that immediately increases robustness. Your doc explicitly highlights coverage tables as a way to gate regressions. 
At minimum, add internal (or analytics) outputs like:

* percent of edges whose src/dst resolved to known anchors
* per-plane row counts
* per-file weld/resolution coverage (you already have some upstream tables for this)

### D. Full-run invariants (no seeded datasets, no partial recompute)

These invariants match your current build policy:

* Every run recomputes from the repo source of truth; cached nodes are **proof**
  of equivalence, not control flow.
* Seeded datasets are not used in the compute path; any snapshot loads are
  strictly for diagnostics or downstream consumption.
* If a cached node is missing, the DAG computes it rather than failing due to
  “missing inputs.”

### E. Contract alignment is diagnostics, not execution control

Contracts should be targets, not blockers:

* If a table doesn’t align with its contract, emit a diagnostic artifact and
  proceed with the actual output.
* Ensure diagnostics are written even on failure so the run is debuggable.

Recommended outputs under `build/diagnostics`:

* `cpg_anchor_resolution.json` (per-plane join success/coverage)
* `cpg_plane_row_counts.json` (rows emitted per plane, per table)
* `cpg_contract_mismatches.json` (schema drift + column/type deltas)
* `cpg_join_drop_rates.json` (rows dropped due to missing anchors)

---

## 7) Execution speed improvements that don’t require redesigning outputs

These are the “easy additions” that usually cut runtime a lot:

### 1) Eliminate `Table.from_pylist(rows)` in hot paths

Replace with:

* build per-column Python lists (still loops, but far less overhead), then
* `pa.Table.from_arrays([...], names=[...])`

This is especially impactful in:

* CPG node assembly
* CPG edge assembly
* any “xref building” logic that currently appends dict rows

### 2) Partition-by-`rel_path` for heavy span/resolve logic

Anything that:

* uses `SpanResolver`, or
* does per-file indexing,
  should run per file.

Hamilton already supports dynamic fan-out (`Parallelizable` + `Collect`) in your ingestion domain; use the same for:

* tree-sitter weld/xref creation
* syntax augmentation that matches spans
* any per-file call argument wiring
* (optionally) flow graphs per-function

### 3) Join early, project early

You’re already doing this in places (e.g., selecting needed columns before join). Apply it everywhere in CPG assembly:

* `table.select([...])` before joins and before extras encoding
* avoid carrying fat `extras_json` blobs through joins unless needed

### 4) Prefer Arrow joins over Python dict indexes unless truly necessary

If you need lookup tables:

* either build a small anchor map and join,
* or dictionary-encode join keys to reduce memory.

### 5) Prefer Arrow compute primitives for transforms

For all hot paths:

* Use `pyarrow.compute` for casts, filters, and derived columns.
* Build arrays column-wise and avoid `to_pylist()` except for truly tiny tables.
* Normalize with `pc.cast`/`pc.if_else` so types are consistent before joins.

---

## 8) A conservative PR plan to get there without breaking everything

This sequence keeps you “working” at every step, but moves you to the clean architecture quickly.

### Phase 1 status (completed)

Delivered the core scaffolding and the first anchor-join plane:

* `cpg2/ids.py`, `cpg2/anchors.py`, `cpg2/assemble.py`, and `cpg2/planes/syntax.py`.
* Syntax nodes + syntax edges now use anchor maps and joins (no row loops).
* CPG diagnostics are emitted under `build/diagnostics` for syntax coverage and drop rates.
* `cpg_edges` now accepts `env` to allow diagnostics emission from edge assembly.

Additional completed work since the initial scaffolding:

* Added `cpg2` plane modules for AST, tree-sitter, SCIP, GOIDs, flow, link, call wiring,
  bytecode, symtable, and inspect.
* `_legacy.py` now delegates node/edge assembly for these planes to `cpg2` (keeping DAG names).
* Contract alignment failures now emit a non-blocking JSONL log under
  `build/diagnostics/contract_alignment_failures.jsonl`.

### Integration note (new)

The remaining phases should **reuse the existing `cpg2` scaffolding** rather than introducing
new top-level modules that duplicate `build/graphs/assembly` or `build/tabular` helpers.
Specifically:

* Keep ID/payload helpers in `cpg2/ids.py` and `cpg2/anchors.py`.
* Use existing Arrow-first helpers from `src/codeintel/build/graphs/assembly` and
  `src/codeintel/build/tabular` rather than creating a new `cpg2/arrow.py` or `cpg2/payloads.py`.
* Add a **lookup anchor registry** in `cpg2/anchors.py` that defines identity keys
  vs lookup keys per plane (e.g., `(repo,commit,node_id)` for syntax lookups) with a
  deterministic tie-break policy when collisions exist.
* Keep `cpg2/__init__.py` free of Hamilton node exports; discovery should only see the `cpg/`
  surface unless you explicitly opt into prefixed `cpg2_*` nodes.
* Use `cpg2_nodes__*` / `cpg2_edges__*` prefixed emitters and keep the aggregation
  in `cpg2_nodes__frames` / `cpg2_edges__frames`.

### PR 1 — Extract policies into dedicated modules (no behavior change)

* Create `cpg2/ids.py` with:

  * `cpg_node_id(table_key, pk_dict)`
  * `cpg_source_pk_json(pk_dict)`
  * `stable_edge_ordinal(table_key, payload)`
* Create `cpg2/anchors.py` with anchor-map builders for the big tables.
* Add a small canonicalization helper (column renames + type casts) that every
  anchor map applies before joins.

Update `_legacy.py` to call those helpers (still produces exact same outputs).

Status: **Completed** (scaffolding + syntax wiring).

### PR 2 — Convert one plane fully to anchor+join (big win fast)

Do syntax plane first:

* `cpg_nodes_from_syntax_nodes` uses anchor map
* `cpg_edges_from_syntax_edges` becomes two joins (no row loops)

This will prove the approach and usually yields immediate speedups.

Status: **Completed**.

### PR 3 — Migrate remaining planes incrementally

Move one file at a time out of `_legacy.py` into `cpg2/planes/*`:

* **goids + scip (first)**: align GOID/SCIP naming, add anchor maps, and push join-based edges.
* **flow edges (second)**: cfg/dfg/cdg join through CFG anchors; record drop rates.
* **bytecode + inspect (third)**: anchors for instructions/objects/signatures; keep extras payloads.
* **symtable + treesitter (fourth)**: anchors for scopes/bindings/ts nodes; emit coverage stats.

At the end, `_legacy.py` becomes either:

* thin wrappers calling `cpg2`, or
* deleted entirely.

Status: **Completed** (AST, tree-sitter, SCIP, GOIDs, flow, link, call wiring,
bytecode, symtable, inspect, overlays) via delegation; overlay diagnostics now wired.

### PR 4 — Add coverage counters + invariants tests

* “No storage import” test for `build/**` (your doc explicitly calls this out as an invariant you want).
* referential checks:

  * every `src_cpg_node_id/dst_cpg_node_id` exists in `cpg_nodes`
  * ordinals are non-null
  * per-plane anchor resolution rates
* Write diagnostics to the concrete files listed in §6E, and ensure they are
  flushed even on run failure.

Status: **Partially complete** (CPG-specific diagnostics emitted; edge integrity counts added;
contract alignment failures and contract mismatches now logged under
`build/diagnostics/contract_alignment_failures.jsonl` and
`build/diagnostics/cpg_contract_mismatches.json`; storage-import allowlist test added; remaining
referential invariants still pending).

---

## 9) Module-level checklist (short, traceable)

Use this to track the exact files touched for each planned change.

- [x] **Anchor map + ID policies**: `src/codeintel/build/hamilton/native/graphs/cpg2/ids.py`,
  `src/codeintel/build/hamilton/native/graphs/cpg2/anchors.py`
- [x] **Canonical column normalization (scaffold)**: `src/codeintel/build/hamilton/native/graphs/cpg2/anchors.py`,
  `src/codeintel/build/hamilton/native/graphs/cpg2/ids.py` (apply to all planes next)
- [x] **Syntax plane migration (anchor + join)**: `src/codeintel/build/hamilton/native/graphs/cpg2/planes/syntax.py`,
  `src/codeintel/build/hamilton/native/graphs/cpg/_legacy.py`
- [x] **Plane-by-plane extraction**: `src/codeintel/build/hamilton/native/graphs/cpg2/planes/{ast,scip,goids,treesitter,py_sym,bytecode,flow,inspect,link,call_wiring,overlays_*}.py`,
  `src/codeintel/build/hamilton/native/graphs/cpg/_legacy.py`
- [x] **Final assembly union/dedupe**: `src/codeintel/build/hamilton/native/graphs/cpg2/assemble.py`
- [x] **Diagnostics outputs (coverage + drop rates)**: `src/codeintel/build/hamilton/native/graphs/cpg2/assemble.py`,
  `src/codeintel/build/hamilton/native/graphs/cpg/_legacy.py` (syntax + core plane edges)
- [x] **Anchor resolution + join drop-rate reporting (all planes)**:
  `src/codeintel/build/hamilton/native/graphs/cpg2/anchors.py`,
  `src/codeintel/build/hamilton/native/graphs/cpg2/planes/*`
- [x] **Contract mismatch reporting (non-blocking)**: `src/codeintel/core/schemas/resolution.py`,
  `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
- [ ] **Hamilton discovery guardrails**: keep `cpg2/__init__.py` minimal, avoid unprefixed
  node names in `cpg2`, and ensure only `cpg/` exposes `cpg_nodes`/`cpg_edges`.
- [ ] **Prefixed node naming + aggregators**: adopt `cpg2_nodes__*` / `cpg2_edges__*` emitters
  and explicit `cpg2_nodes__frames` / `cpg2_edges__frames` aggregators.
- [ ] **Anchor registry table**: add identity + lookup key registry to `cpg2/anchors.py`.
- [ ] **Re-export targets to preserve DAG names**: `src/codeintel/build/hamilton/native/graphs/cpg/__init__.py`
- [ ] **Acceptance check**: `native_module_paths()` loads without duplicate node names and
  `cpg.cpg_nodes`/`cpg.cpg_edges` depend only on `cpg2_nodes__frames` / `cpg2_edges__frames`.

---

## 10) Why this is “best-in-class” for Hamilton specifically

Hamilton is strongest when:

* nodes are small, composable, and cacheable,
* expensive steps are isolated (so caching can prove equivalence in full runs),
* dynamic fan-out is used where natural partitions exist (files, functions).

The **plane-first + anchor-map** organization is basically the “Hamilton-native” way to build something like a CPG:

* the DAG mirrors your conceptual model,
* each plane is optional and testable,
* performance is dominated by Arrow joins (fast) rather than Python dict churn (slow),
* and you preserve the clean build/storage boundary your doc emphasizes.

---

## Phase 2 acceptance checklist (per plane)

Use this to gate the next migration phase. Each plane should emit diagnostics under
`build/diagnostics` and meet the drop-rate thresholds below. Treat thresholds as
targets; if a threshold is exceeded, keep the plane join-built but record the
deviation and follow up with improved anchors or lookup keys.

- **Syntax plane (`planes/syntax.py`)**
  - Diagnostics: `cpg_anchor_resolution.json`, `cpg_join_drop_rates.json`
  - Expected: `syntax_edges.drop_rate <= 0.5%`, `syntax_nodes.resolved >= 99%`
- **Symbol plane (`planes/symbol.py`)**
  - Diagnostics: `cpg_anchor_resolution.json`, `cpg_join_drop_rates.json`
  - Expected: `symbol_edges.drop_rate <= 2%`, fallback SpanResolver coverage recorded
- **Link plane (`planes/link.py`)**
  - Diagnostics: `cpg_join_drop_rates.json`
  - Expected: `call_graph.drop_rate <= 1%`, `import_graph.drop_rate <= 1%`
- **Flow plane (`planes/flow.py`)**
  - Diagnostics: `cpg_join_drop_rates.json`
  - Expected: `cfg/dfg/cdg drop_rate <= 1%` after cfg anchor normalization
- **Call wiring plane (`planes/call_wiring.py`)**
  - Diagnostics: `cpg_join_drop_rates.json`
  - Expected: `call_wiring drop_rate <= 2%` (syntax+cfg anchor joins)
- **Overlays — symtable (`planes/overlays_symtable.py`)**
  - Diagnostics: `cpg_join_drop_rates.json`
  - Expected: `scope/resolution drop_rate <= 1%`; namespace binding joins recorded
- **Overlays — bytecode (`planes/overlays_bytecode.py`)**
  - Diagnostics: `cpg_join_drop_rates.json`
  - Expected: algorithmic edges record match rates; joins drop_rate <= 5%
- **Overlays — inspect (`planes/overlays_inspect.py`)**
  - Diagnostics: `cpg_join_drop_rates.json`
  - Expected: join-built edges drop_rate <= 5%; heuristic edges record match rates

If any plane fails thresholds:
1) tighten anchor lookups (add/normalize lookup keys)
2) re-run, and record before/after in `cpg_anchor_resolution.json`

---

If you want, I can take your current `graph.cpg_nodes` / `graph.cpg_edges` implementation and **map it plane-by-plane into the exact `cpg2/` module skeleton above**, including:

* the specific anchor map keys for each upstream table (what columns to use),
* which edges should be join-built vs. algorithm-built,
* and which current `_legacy.py` functions move to which plane file.

That would stay conservative (no schema changes), but you’d end up with a CPG pipeline that’s dramatically easier to extend and optimize.


# addendum - feedback on specific planes #

Below is a **conservative, no-schema-change** refactor plan that takes your *current* `graph.cpg_nodes` / `graph.cpg_edges` implementation (today: almost everything living in `src/codeintel/build/hamilton/native/graphs/cpg/_legacy.py`) and maps it **plane-by-plane** into a **CPG2-style module layout** that is:

* easier to reason about (each “plane” owns a coherent slice),
* faster (compute anchor IDs once; prefer Arrow joins over Python dict-index loops),
* more robust (centralized ID rules + centralized anchor lookups),
* still **build-only** (no dependency on `src/codeintel/storage`).

I’ll give you:

1. the **exact plane layout** (files + responsibilities),
2. the **anchor map keys** (what columns define identity, what columns are used for lookups),
3. **join-built vs algorithm-built** decisions per edge family, and
4. a **function migration map**: which `_legacy.py` functions move where.

---

## 0) What “plane-by-plane” means for your current CPG

Your current CPG assembly has a natural decomposition already:

### Nodes (“inventory plane”)

`cpg_nodes()` concatenates “node inventories” from:

* `core.syntax_nodes`, `core.ast_nodes`, `core.scip_symbol_information`, `core.goids`
* Python overlays: `core.py_sym_scopes`, `core.py_sym_bindings`, `core.py_bc_*`, `core.py_inspect_*`
* tree-sitter tokens/trivia: `core.ts_tokens`, `core.ts_trivia`
* graph inventories: `graph.cfg_blocks`, `graph.import_modules`

### Edges (“relation planes”)

`cpg_edges()` concatenates:

* **syntax**: AST edges from `core.syntax_edges`
* **symbol**: SCIP occurrence edges + symbol relationships + symbol->GOID
* **link**: call graph edges + import graph edges
* **flow**: CFG/DFG/CDG edges
* **interproc call wiring**: `graph.cpg_edges_calls`, `graph.cpg_edges_arg_to_param`, `graph.cpg_edges_ret_to_call`
* **overlays** (gated by ingestion options + CPG options): symtable, bytecode, inspect

This maps cleanly to CPG2 planes.

---

## 1) Proposed CPG2 conservative module skeleton (no schema changes)

Create a new package:

```
src/codeintel/build/hamilton/native/graphs/cpg2/
  __init__.py
  ids.py
  anchors.py
  assemble.py

  planes/
    __init__.py
    syntax.py               # syntax nodes + syntax edges (anchor + join)
    symbol.py               # SCIP occurrence + rels + symbol->goid
    link.py                 # call graph + import graph
    flow.py                 # cfg/dfg/cdg -> cpg
    call_wiring.py          # call wiring -> cpg
    nodes.py                # graph.cpg_nodes assembly
    overlays_symtable.py
    overlays_bytecode.py
    overlays_inspect.py
```

### Conservative integration

Keep your existing `cpg/` package as a shim for now:

* `src/codeintel/build/hamilton/native/graphs/cpg/nodes.py` imports from `cpg2/planes/nodes.py`
* `.../cpg/edges.py` imports from `cpg2/assemble.py`
* `.../cpg/ids.py` imports from `cpg2/ids.py`
* `.../cpg/bytecode.py` imports from `cpg2/planes/overlays_bytecode.py`
* `.../cpg/inspect_overlay.py` imports from `cpg2/planes/overlays_inspect.py`

So **graph targets + table contracts remain unchanged**, while the internal organization becomes “CPG2”.

**Implementation note:** avoid adding new “payloads/arrow/contracts” modules inside `cpg2`.
Reuse the existing helpers in `src/codeintel/build/graphs/assembly` and
`src/codeintel/build/tabular` to keep the diff conservative and avoid duplication.

---

## 2) Anchor map keys: “identity keys” vs “lookup keys”

This is the single most important structural improvement: **compute anchor IDs once** and reuse via joins.

**Implementation note (applies to all anchors):** normalize join key types at the anchor boundary
(`string_view` → `string`, decimal casts) to avoid schema drift and join failures later.

### 2.1 Canonical identity keys (these define `stable_cpg_id(table_key, pk)`)

These match your current `_legacy.py` behavior exactly:

| Node kind                 | source_table_key                   | identity key columns (`pk_values`)                                  |
| ------------------------- | ---------------------------------- | ------------------------------------------------------------------- |
| `SYNTAX_NODE`             | `core.syntax_nodes`                | `repo, commit, rel_path, producer, node_id`                         |
| `AST_NODE`                | `core.ast_nodes`                   | `hash` *(repo/commit come from env in node rows but not in the pk)* |
| `SCIP_SYMBOL`             | `core.scip_symbol_information`     | `repo, commit, symbol`                                              |
| `GOID`                    | `core.goids`                       | `goid_h128`                                                         |
| `CFG_BLOCK`               | `graph.cfg_blocks`                 | `function_goid_h128, block_idx`                                     |
| `MODULE`                  | `graph.import_modules`             | `repo, commit, module`                                              |
| `TS_TOKEN`                | `core.ts_tokens`                   | `repo, commit, rel_path, language, token_id`                        |
| `TS_TRIVIA`               | `core.ts_trivia`                   | `repo, commit, rel_path, language, trivia_id`                       |
| `SCOPE`                   | `core.py_sym_scopes`               | `repo, commit, rel_path, scope_id`                                  |
| `BINDING`                 | `core.py_sym_bindings`             | `repo, commit, rel_path, binding_id`                                |
| `BC_CODE_UNIT`            | `core.py_bc_code_units`            | `repo, commit, rel_path, code_unit_id`                              |
| `BC_INSTR`                | `core.py_bc_instructions`          | `repo, commit, rel_path, code_unit_id, instr_id`                    |
| `BC_BLOCK`                | `core.py_bc_blocks`                | `repo, commit, rel_path, block_id`                                  |
| `INSPECT_OBJECT`          | `core.py_inspect_objects`          | `repo, commit, object_id`                                           |
| `INSPECT_SIGNATURE`       | `core.py_inspect_signatures`       | `repo, commit, signature_id`                                        |
| `INSPECT_SIGNATURE_PARAM` | `core.py_inspect_signature_params` | `repo, commit, signature_id, param_index`                           |

Put this in `cpg2/anchors.py` as a single registry constant (even if only used implicitly).

### 2.2 Lookup keys (the keys edges actually have on hand)

Edges very often **do not have the full identity pk**. Example: call wiring edges know `call_node_id` but not always `producer`/`rel_path`.

So you want a second registry: **how to look up an anchor row** given an edge row.

Here are the lookups I recommend **aligned to your current tables**:

#### Syntax nodes

* **Lookup key:** `(repo, commit, node_id)`
* **Reason:** many producers/edges only carry `node_id` (your current `_syntax_node_index()` uses exactly this).

So build an internal anchor table:

`cpg2_anchor_syntax_node_ids` columns:

* `repo, commit, node_id`
* `rel_path, producer` *(to disambiguate if needed later)*
* `cpg_node_id`

Implementation note (robustness): if multiple `(repo,commit,node_id)` exist across producers,
pick deterministically (producer preference or “max span coverage”). Centralize the tie-break
rule inside `cpg2/anchors.py` so all planes inherit the same behavior.

#### CFG blocks

* CFG/DFG/CDG edges carry block ids like `"GOID:blockN"`, i.e. globally unique.
* **Lookup key:** `(block_id)` *(safe; `graph.cfg_blocks.block_id` is derived as `f"{function_goid}:block{idx}"`)*

So build:

`cpg2_anchor_cfg_block_ids` columns:

* `block_id`
* `function_goid_h128, block_idx`
* `repo, commit` (join from `core.goids` on `function_goid_h128`)
* `rel_path` (from `cfg_blocks.file_path`)
* `cpg_node_id`
* **Canonicalize** types before joins (`string_view` → `string`, decimal casts).

#### GOIDs

* **Lookup key:** `(goid_h128)` -> `cpg_node_id`

#### Modules

* **Lookup key:** `(repo, commit, module)` -> `cpg_node_id`

#### AST nodes

* **Lookup key:** `(hash)` -> `cpg_node_id`
* plus keep `repo/commit/path/start_byte/end_byte/lineno/end_lineno/node_type/qualname`
  available for algorithmic overlays to avoid cross-snapshot ambiguity.

#### Python symtable anchors

* Scopes lookup: `(repo, commit, rel_path, scope_id)`
* Bindings lookup: `(repo, commit, rel_path, binding_id)`
* For namespace edges you often need “binding by (scope_id, name)”: build a helper index table keyed by:

  * `(repo, commit, rel_path, scope_id, name)` -> `binding_id, cpg_node_id`

#### Bytecode anchors

* Instruction lookup: `(repo, commit, rel_path, code_unit_id, instr_id)` -> `cpg_node_id`
* Block lookup: `(repo, commit, rel_path, block_id)` -> `cpg_node_id`
* Code unit lookup: `(repo, commit, rel_path, code_unit_id)` -> `cpg_node_id`

#### Inspect anchors

* Object lookup: `(repo, commit, object_id)` -> `cpg_node_id`
* Signature lookup: `(repo, commit, signature_id)` -> `cpg_node_id`
* Param lookup: `(repo, commit, signature_id, param_index)` -> `cpg_node_id`

---

## 3) Which edges become join-built vs algorithm-built

A clean CPG2 refactor makes this explicit per plane.

### Join-built (Arrow joins + column transforms)

These should be implemented as **join-built** in CPG2 because the edge sources are already keyed:

#### Plane: `planes/syntax.py`

* `core.syntax_edges` → `graph.cpg_edges`
* Join parent/child node IDs via `cpg2_anchor_syntax_node_ids` (or compute IDs directly if the full identity key is present).
* Output: `edge_kind="AST"`, `edge_layer="SYNTAX"`, `ordinal=child_ordinal`

#### Plane: `planes/symbol.py`

* `core.scip_symbol_relationships` → `edge_kind=relationship_kind`, `edge_layer="SYMBOL"` (direct map)
* `core.scip_symbol_goid_xref` → `edge_kind="RESOLVES_TO"`, `edge_layer="SYMBOL"` (direct map)
* `core.scip_occurrence_*_xref`: *partly join-built* (see below)

#### Plane: `planes/link.py`

* `graph.call_graph_edges` → `edge_kind="CALLS"`, `edge_layer="FLOW"` (direct mapping + stable ordinal)
* `graph.import_graph_edges` → `edge_kind="IMPORTS"`, `edge_layer="SYMBOL"` (direct mapping + extras)

#### Plane: `planes/flow.py`

* `graph.cfg_edges`, `graph.dfg_edges`, `graph.cdg_edges`:

  * join src/dst via `cpg2_anchor_cfg_block_ids` on `src_block_id`, `dst_block_id`
  * emit `edge_kind="CFG"/"DFG"/"CDG"` (or your current kind mapping) + ordinals

#### Plane: `planes/call_wiring.py`

* `graph.cpg_edges_calls`:

  * `call_node_id` join → syntax anchor
  * `callee_entry_block_id` join → cfg anchor
* `graph.cpg_edges_arg_to_param`:

  * join both ends via syntax anchor
* `graph.cpg_edges_ret_to_call`:

  * `exit_block_id` join → cfg anchor
  * `call_node_id` join → syntax anchor

#### Overlay: `planes/overlays_symtable.py` (mostly join-built)

* `core.py_sym_scope_edges` (scope→scope)
* `core.py_sym_resolution_edges` (binding→binding)
* `core.py_sym_namespace_edges` (binding→scope) **join-built** by joining namespace edges to bindings via `(repo,commit,rel_path,scope_id,name)` → `binding_id`

---

### Algorithm-built (SpanResolver, simulation, graph algorithms, heuristics)

These remain algorithmic, but their *inputs* should still come from anchor tables.

#### SCIP occurrences

Plane: `planes/symbol.py`

* `core.scip_occurrence_syntax_xref` and fallback `core.scip_occurrence_span_xref`
* Needs:

  * role precedence rules,
  * fallback selection (SpanResolver),
  * de-duplication.
* Still: you can join “what you can” first, then only fallback rows go through SpanResolver.

#### Bytecode overlay edges

Overlay: `planes/overlays_bytecode.py`
These are inherently algorithmic in your current design:

* instruction ↔ AST anchor selection (span/line match)
* instruction ↔ callsite matching
* stack simulation
* memory edges from def/use events
* reaching-defs (`enable_reaches`)

#### Inspect overlay

Overlay: `planes/overlays_inspect.py`
Heuristic components:

* arg-to-param mapping (signature + call arg semantics)
* inspect → AST anchor selection via file path + line spans
* runtime state mapping (frame→instruction inference)

---

## 4) Plane-by-plane mapping of today’s CPG into CPG2

Below I list each plane, what it owns, the anchor keys it relies on, and **which `_legacy.py` functions move there**.

---

## Plane A: IDs + core helpers (reuse existing assembly utilities)

### `cpg2/ids.py`

Moves from `_legacy.py`:

* `_stable_int_hash`
* `_stable_cpg_id`
* `_stable_ordinal`
* `_stable_cpg_id_from_row`
* `_stable_ordinal_from_row`
* `_instruction_cpg_id`
* `_binding_cpg_id`
* `_ast_cpg_id`
* `_syntax_node_cpg_id`
* `_inspect_signature_param_cpg_id`

Public wrappers to keep:

* `stable_cpg_id`
* `instruction_cpg_id`

**Do not create new `payloads.py` or `arrow.py` modules.** Reuse the existing helpers in:

* `src/codeintel/build/graphs/assembly/*`
* `src/codeintel/build/tabular/*`

This keeps the refactor conservative and avoids duplicating stable helper logic.

---

## Plane B: Anchor inventory builders (internal reusable lookup tables)

Each of these is a **pure “make IDs once”** helper. They can be Hamilton nodes (internal) or private functions.
Start centralized in `cpg2/anchors.py`; split into submodules only if a plane grows too large.

### `cpg2/anchors.py` (initially centralized, split later if needed)

Moves:

* `_syntax_node_keys` *(maybe keep as helper)*
* `_syntax_node_index` *(replaced by join-built anchor table; keep only if needed)*

New in CPG2:

* `cpg2_anchor_syntax_node_ids`: `(repo, commit, node_id) -> cpg_node_id (+rel_path, producer, span)`

### `cpg2/anchors.py` (cfg block anchors)

Moves:

* `_cfg_block_index`
* `_block_id_index`

New in CPG2:

* `cpg2_anchor_cfg_block_ids`: `(block_id) -> cpg_node_id (+function_goid_h128, block_idx, repo, commit, rel_path)`

### `cpg2/anchors.py` (ast anchor helpers)

Moves:

* `_ast_nodes_by_path`
* `_normalize_path`
* `_best_source_path`
* `_ast_span_for_source`
* `_select_ast_anchor_for_source`
* `_ast_anchor_candidates_by_span`
* `_ast_anchor_candidates_by_line`
* `_select_ast_anchor`

(These are shared by inspect overlay + bytecode overlay; keeping them centralized is a big win.)

### `cpg2/anchors.py` (symtable anchors)

Moves:

* `_scope_qualname_from_qualpath`
* `_expected_scope_type`
* `_scope_candidates`
* `_span_length`
* `_span_contains`
* `_select_scope_by_span`
* `_select_scope_by_lineno`
* `_select_scope_for_unit`
* `_build_code_unit_scope_map`
* `_binding_payload_from_row`
* `_build_binding_index`
* `_build_resolution_map`
* `_resolve_binding_for_event`
* `_event_var_key`

(These are currently scattered in `_legacy.py` but are conceptually “symtable anchor + matching helpers”.)

### `cpg2/anchors.py` (bytecode anchors)

Moves helper parsing for def/use & blocks:

* `_assign_events_to_blocks`
* `_block_gen_kill`
* `_merge_def_maps`
* `_apply_gen_kill`
* `_compute_reaching_defs`
* `_parse_defuse_event_row`
* `_group_defuse_events`
* `_parse_block_row`
* `_group_blocks`
* `_parse_cfg_edge_row`
* `_group_cfg_edges`

### `cpg2/anchors.py` (inspect anchors)

Moves:

* `_inspect_full_qualname`
* `_inspect_status_ok`

(and keep small “inspect indexing” helpers here)

---

## Plane C: Nodes plane (graph.cpg_nodes)

### `cpg2/planes/nodes.py`

Moves *exactly* these node constructors:

* `_syntax_nodes_to_cpg`
* `_ast_nodes_to_cpg`
* `_scip_symbols_to_cpg`
* `_goids_to_cpg`
* `_cfg_blocks_to_cpg`
* `_import_modules_to_cpg`
* `_ts_tokens_to_cpg`
* `_ts_trivia_to_cpg`
* `_py_sym_scopes_to_cpg`
* `_py_sym_bindings_to_cpg`
* `_py_bc_code_units_to_cpg`
* `_py_bc_instructions_to_cpg`
* `_py_bc_blocks_to_cpg`
* `_py_inspect_objects_to_cpg`
* `_py_inspect_signatures_to_cpg`
* `_py_inspect_signature_params_to_cpg`

Moves the Hamilton input bundlers (or replaces with cleaner per-plane bundlers):

* `cpg_nodes__syntax_inputs`
* `cpg_nodes__py_inputs`
* `cpg_nodes__inspect_inputs`
* `cpg_nodes__core_inputs`
* `cpg_nodes__graph_inputs`
* `cpg_nodes__inputs`
* `_core_lazyframes`
* `_graph_lazyframes`
* `cpg_nodes`

**CPG2 improvement while staying conservative:**
Emit internal anchor tables from `cpg2/anchors.py` (the lookup tables described above).
They are not “schema changes” because they are not exported targets; they are internal DAG nodes.

---

## Plane D: Syntax edge plane

### `cpg2/planes/syntax.py`

Moves:

* `_syntax_edges_to_cpg`

Refactor change (no schema change):
replace per-row stable ID recomputation with joins to `cpg2_anchor_syntax_node_ids` (or at least centralize the “node_id → full pk” lookup in one place).

---

## Plane E: Symbol edge plane (SCIP)

### `cpg2/planes/symbol.py`

Moves:

* `_occurrence_role_resolvers`
* `_occurrence_fallback_rows`
* `_occurrence_roles`
* `_occurrence_span_index`
* `_occurrence_joined_rows`
* `_apply_occurrence_resolvers`
* `_scip_occurrence_edges_to_cpg`
* `_scip_symbol_relationships_to_cpg`
* `_scip_symbol_goid_edges_to_cpg`

CPG2 improvement (still conservative):

* Make the occurrence pipeline explicitly two-stage:

  1. join-built where exact keys match,
  2. algorithm-built fallback via SpanResolver.
* Emit an internal “occurrence coverage” debug table if helpful (still build-only).

---

## Plane F: Link edge plane

### `cpg2/planes/link.py`

Moves:

* `_call_graph_edges_to_cpg`
* `_import_graph_edges_to_cpg`

These are almost pure mapping transforms already.

---

## Plane G: Flow edge plane (CFG/DFG/CDG)

### `cpg2/planes/flow.py`

Moves:

* `_cfg_edges_to_cpg`
* `_dfg_edges_to_cpg`
* `_dfg_edge_row`
* `_cdg_edges_to_cpg`
* `_cdg_edge_row`

CPG2 improvement (still conservative):

* rewrite these to be join-built:

  * join `src_block_id` and `dst_block_id` to `cpg2_anchor_cfg_block_ids` to get `src_cpg_node_id`/`dst_cpg_node_id`
  * stop using `_cfg_block_index()` dicts in these planes

This is a big speed + clarity win.

---

## Plane H: Call wiring plane

### `cpg2/planes/call_wiring.py`

Moves:

* `_call_wiring_calls_to_cpg`
* `_call_wiring_arg_to_param_to_cpg`
* `_call_wiring_ret_to_call_to_cpg`

CPG2 improvement (still conservative):

* make this join-built as well:

  * `call_node_id` → syntax anchor
  * `callee_entry_block_id` / `exit_block_id` → cfg anchor

---

## Plane I: Overlay — symtable

### `cpg2/planes/overlays_symtable.py`

Moves:

* `_py_sym_scope_edges_to_cpg`
* `_py_sym_namespace_edges_to_cpg`
* `_namespace_edge_row`
* `_py_sym_binding_edges_to_cpg`
* `_py_sym_resolution_edges_to_cpg`
* `_py_sym_resolution_edge_row`
* `_py_sym_binding_symbol_edges_to_cpg`
* `_scope_qualname_index`
* `_symbol_display_index`
* `_binding_symbol_edge_rows`
* `_ast_event_kind`
* `_ast_binding_name`
* `_ast_event_row`
* `_scope_for_ast_event`
* `_ast_binding_context_for_event`
* `_binding_context_from_info`
* `_ast_binding_edge_row`
* `_scopes_by_path`
* `_ast_binding_edges_to_cpg`

Join-built vs algorithm-built:

* scope/resolution/binding edges: join-built
* namespace edges: join-built (via binding lookup join)
* ast-binding edges: algorithm-built (heuristics)

---

## Plane J: Overlay — bytecode

### `cpg2/planes/overlays_bytecode.py`

Moves:

* `_py_bc_instruction_ast_edges_to_cpg`
* `_py_bc_callsite_edges_to_cpg`
* `_py_bc_callsite_symbol_edges_to_cpg`
* `_py_bc_cfg_edges_to_cpg`
* `_py_bc_defuse_binding_edges_to_cpg`
* `_py_bc_memory_edges_to_cpg`
* `_py_bc_stack_edges_to_cpg`
* `_py_bc_reaches_edges_to_cpg`

And *all* supporting bytecode helpers currently in `_legacy.py`:

* callsite matching helpers (`_is_call_op`, `_select_syntax_call`, etc.)
* symbol matching helpers (`_display_name_variants`, `_callsite_symbol_matches`, etc.)
* stack simulation helpers (`_stack_effect_net`, `_stack_push_from_*`, etc.)
* reaches helpers (`_emit_reaches_edges`, `_reaches_context_from_rows`, etc.)

This is its own world — keeping it isolated prevents the rest of the CPG from being “infected” by bytecode complexity.

---

## Plane K: Overlay — inspect

### `cpg2/planes/overlays_inspect.py`

Moves:

* `_inspect_arg_to_param_edges_to_cpg` and all its internal helpers:

  * `_callee_qname_priority`, `_call_callee_candidates`, `_assign_args_to_params`, etc.
* `_py_inspect_signature_edges_to_cpg`
* `_inspect_to_ast_edges_to_cpg` and its helper indexers:

  * `_inspect_ast_indices`, `_inspect_sources_by_object`, `_inspect_ast_edges_for_source`, etc.
* `_inspect_to_scip_edges_to_cpg`
* `_py_inspect_class_mro_edges_to_cpg`
* `_py_inspect_class_attr_edges_to_cpg` and helpers
* `_py_inspect_runtime_state_edges_to_cpg` and helpers
* `_py_inspect_unwrap_edges_to_cpg`

---

## Plane L: Edge assembly + overlay gating (graph.cpg_edges)

### `cpg2/assemble.py`

Moves:

* `cpg_edge_symbol_inputs`
* `cpg_edge_flow_inputs`
* `cpg_edge_link_inputs`
* `cpg_edge_call_wiring_inputs`
* `cpg_edge_syntax_node_inputs`
* all `cpg_edge_overlay_*_inputs` bundle functions
* `cpg_edge_core_inputs`
* `_overlay_frames`
* `cpg_edges`

Refactor change (no schema change):

* `cpg_edges()` becomes a simple concatenation of plane outputs:

  * `cpg2_edges_syntax`
  * `cpg2_edges_symbol`
  * `cpg2_edges_link`
  * `cpg2_edges_flow`
  * `cpg2_edges_call_wiring`
  * overlays (gated)

Each plane returns a properly contracted `pa.Table` already aligned to `_CPG_EDGE_COLUMNS`.

---

## 5) Quick “edge strategy matrix” for your current edge families

| Edge family      | Current source(s)                               | CPG2 plane                    | Strategy                                          |
| ---------------- | ----------------------------------------------- | ----------------------------- | ------------------------------------------------- |
| Syntax AST edges | `core.syntax_edges`                             | `planes/syntax.py`      | **Join-built**                                    |
| SCIP occurrence  | `core.scip_occurrence_*_xref`                   | `planes/symbol.py`      | **Hybrid:** join-built + fallback SpanResolver    |
| SCIP symbol rels | `core.scip_symbol_relationships`                | `planes/symbol.py`      | **Direct map**                                    |
| SCIP symbol→GOID | `core.scip_symbol_goid_xref`                    | `planes/symbol.py`      | **Direct map / join to anchors optional**         |
| Call graph       | `graph.call_graph_edges`                        | `planes/link.py`        | **Direct map**                                    |
| Import graph     | `graph.import_graph_edges`                      | `planes/link.py`        | **Direct map**                                    |
| CFG/DFG/CDG      | `graph.cfg_edges/dfg_edges/cdg_edges`           | `planes/flow.py`        | **Join-built** (via cfg anchor map)               |
| Call wiring      | `graph.cpg_edges_*`                             | `planes/call_wiring.py` | **Join-built**                                    |
| Symtable overlay | `core.py_sym_*`                                 | `planes/overlays_symtable.py`  | Mostly **join-built**; AST-binding is algorithmic |
| Bytecode overlay | `core.py_bc_*` + syntax calls + AST nodes       | `planes/overlays_bytecode.py`  | Mostly **algorithm-built**                        |
| Inspect overlay  | `core.py_inspect_*` + syntax calls + AST + SCIP | `planes/overlays_inspect.py`   | Mixed; several heuristic/algorithmic              |

---

## 6) Why this solves your “legacy code tripping me up” problem (without redesigning semantics)

This refactor doesn’t require you to change a single output schema or even change the high-level `cpg_nodes()` / `cpg_edges()` interface.

What it does is:

* **centralize identity rules** (`cpg2/ids.py`),
* **compute reusable anchor maps once** (`cpg2/anchors.py`),
* **make each plane responsible for one kind of reasoning**, and
* make join vs algorithm decisions explicit and isolated.

So the next time you add a new edge kind or improve a resolver, you touch **one plane**, not a 7,800-line `_legacy.py`.

---

## 7) If you want the “next most concrete” step

I can take *this* mapping and turn it into a **literal file-by-file rewrite plan** with:

* the exact new function names to introduce (Hamilton node signatures),
* which modules you need to add to the target module list (or how to import them through `cpg/__init__.py` so Hamilton discovers them),
* and a mechanical “move these functions verbatim first, then replace dict-index lookups with joins” checklist.

Just say the word and I’ll draft that PR plan in the same “plane-by-plane” format.
