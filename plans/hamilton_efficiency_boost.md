Yes — there are *very* plausible ways to get dramatically faster than 11 minutes **without** multi-core, but whether you can hit **~30 seconds for a full rebuild** depends on *what you’re computing*.

From reviewing the structure of `src_revised.zip`, the big story is:

* You are doing **a lot of work per file** (AST + LibCST + symtable + bytecode + (possibly) SCIP + graph assembly).
* Several parts of the pipeline currently do **redundant parsing** and **Python-level row-by-row loops** on large tables (especially via `to_pylist()` and per-row stable hashing/JSON encoding), which can easily dominate runtime even with NetworkX disabled.

Below is a concrete, conservative-but-high-impact performance plan that stays in your Hamilton approach and doesn’t require multi-core complexity.

---

## Can you get to ~30 seconds?

### Full rebuild, full richness (AST + LibCST scopes + bytecode + SCIP + CPG)

On a “large-ish” repo, **single-core ~30s is usually not realistic** if you insist on:

* LibCST parse + `ScopeProvider`/`QualifiedNameProvider` for every file,
* bytecode compilation + instruction/block extraction for every file,
* and SCIP indexing.

Those are fundamentally CPU-heavy steps and (for LibCST) mostly Python-level.

### But you *can* get:

* **Big constant-factor wins** (often **3×–10×**) by removing redundant work and eliminating Python row loops.
* **Incremental runs** in the **tens-of-seconds** range by making caching/incrementality “actually bite” (even on one core).

So: **30 seconds is plausible for incremental runs or a “fast profile.”**
For full rebuild, you can still likely get from **11 minutes → low single-digit minutes** with engineering fixes; potentially better if your current hotspot is per-row hashing/serialization.

---

## The biggest concrete bottlenecks I see in your current structure

### 1) Redundant AST parsing (this is a *huge* fixable tax)

Right now, per file, you parse Python multiple times across different ingestion steps:

* `codeintel/ingestion/compute/ast_extract.py` → `ast.parse(...)`
* `codeintel/ingestion/compute/docstrings_extract.py` → `ast.parse(...)`
* `codeintel/ingestion/compute/symtable_extract.py` → `ast.parse(...)` (plus `symtable.symtable(...)`)
* `codeintel/ingestion/compute/cst_extract.py` → **if** `SyntaxIndexOptions.emit_ast_nodes=True`, you call `collect_ast_nodes(...)` which parses AST again
* `codeintel/ingestion/compute/dis_extract.py` → `compile(source, ...)` which parses again (unless you compile from an AST object)

This can easily turn “AST work” into 3–5× what it needs to be.

---

### 2) `to_pylist()` on big tables + per-row hashing/JSON encoding

Even with NetworkX disabled, you still have major Python hot loops, for example:

* `codeintel/build/hamilton/native/ingestion/syntax_augment.py`

  * builds a span index via `SpanResolver` by iterating `syntax_nodes.to_pylist()`
  * then iterates `ts_nodes.to_pylist()` doing per-row matching
* `codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`

  * computes `ordinal` via `cpg_edge_ordinal(...)` using a Python loop over `.to_pylist()`
  * and computes `extras_json` via `_payload_bytes(...)` again row-by-row

And your stable ID helpers are **expensive per row**:

* `stable_int_hash()` / `stable_decimal_id()` uses `json.dumps(sort_keys=True)` + `hashlib.blake2b(...)` per row.

If you have *hundreds of thousands* or *millions* of edge rows, this alone can take minutes.

---

### 3) IntervalTree-based span matching applied “globally”

`SpanResolver` (IntervalTree-backed) is great when you need overlap/containment matching, but it’s expensive to build at the scale of “every syntax node in the repo,” especially when most xrefs are exact-span matches.

---

## Best-in-class speed plan (still conservative, still Hamilton)

### Phase A — “Immediate wins” (minimal logic changes, big returns)

#### A1) Stop parsing AST multiple times

**Do this first.** It’s the cleanest, highest-confidence speedup.

**Option 1 (smallest change):**

* Set `SyntaxIndexOptions.emit_ast_nodes = False` so `cst_extract.py` stops calling `collect_ast_nodes(...)`.

  * If you need AST facts for syntax augmentation, do the merge downstream using the already-built `core.ast_nodes` table.

That alone can remove a full AST pass per file.

**Option 2 (best practice, still conservative):**
Create a single “python frontend” ingestion step that:

* reads bytes once
* decodes once
* parses AST once
* compiles bytecode from the AST (so compile doesn’t re-parse)

…and then downstream steps *consume that shared parsed artifact* rather than re-parsing.

You don’t need to change your output schema to do this; you can introduce an internal Hamilton node like:

* `py_frontend__artifacts -> Dict[rel_path, PyFrontendArtifact]`

  * `PyFrontendArtifact(ast_tree, source_text, source_bytes, encoding, line_index, compiled_code_obj, ...)`

Then:

* `ast_extract` consumes `ast_tree`
* `docstrings_extract` consumes `ast_tree`
* `symtable_extract` consumes `ast_tree` (and `source_text` for `symtable.symtable`)
* `dis_extract` consumes `compiled_code_obj` (or compiles from `ast_tree`)

This keeps your DAG clear while removing redundant heavy work.

---

#### A2) Enforce “no `to_pylist()` on large tables” as a rule

A practical best-in-class rule:

> Any table that can exceed ~50k rows must not be converted with `to_pylist()` in the hot path.

Instead:

* operate in Arrow/Polars/DuckDB (vectorized),
* or iterate `RecordBatch`es and work columnar.

Even without multi-core, moving “Python loops over rows” to “vectorized compute” is often the difference between minutes and seconds.

---

#### A3) Replace per-row stable hashing + JSON encoding with a vectorized ID strategy

Your current stable ID path (`json.dumps + blake2b`) is robust but **way too slow** at edge-scale.

Best-in-class approach for throughput:

* Use a **vectorized 64-bit hash** over selected key columns for ordinals / edge IDs.
* Only compute heavy “payload JSON blobs” when truly needed (often: never in the core CPG).

Pragmatically:

* Keep `extras_json` as `NULL` or minimal for flow edges unless you have a consumer that truly needs it.
* Or store structured extras in side tables, not per-edge payload blobs.

If you *must* keep deterministic hashing semantics, use a fast canonical encoding (e.g., `orjson` with sort keys) and hash bytes — but still avoid doing it per-row in Python.

---

### Phase B — “Big algorithmic wins” (still conservative, but refactors logic shape)

#### B1) Rebuild `syntax_augment` around an exact-span join first

In `syntax_augment.py`, most TS nodes that correspond to LibCST nodes will match on:

* `(rel_path, start_byte, end_byte)`

So the best-in-class approach is:

1. **Exact match join** between TS nodes and syntax nodes on those keys (fast, vectorized).
2. For the *unmatched remainder only*:

   * do interval matching (SpanResolver / binary search) per file.

This typically collapses an expensive “interval resolver over millions of nodes” into a cheap join + small fallback.

Also: `_failure_paths()` and `_producer_by_path()` should be computed via columnar filtering/group-by (not `to_pylist()`).

---

#### B2) Make span resolution structures “lazy” and per-file

Your `SpanResolver` currently builds an IntervalTree on every `add_span(...)`.

Best-in-class pattern:

* keep an exact map always (cheap)
* only build the interval structure **if** you actually need overlap/containment matching *for that file*.

This is a huge constant-factor win when most matches are exact.

---

### Phase C — “If you truly want 30 seconds” (you need a fast profile)

If your goal is to serve an LLM agent via FastMCP, the best-in-class system is usually **progressive enrichment**:

1. **Fast pass (repo-wide, seconds):**

   * module inventory
   * tree-sitter structural captures (defs/imports/calls/refs, tokens optional)
   * minimal symbol index (maybe SCIP if already available / incremental)
   * lightweight CPG nodes/edges (syntax + symbol links, import/call edges)

2. **Deep pass (on-demand / targeted):**

   * LibCST scopes + trivia + formatting fidelity
   * bytecode CFG/DFG/def-use
   * expensive graph-derived edges (PDG/CDG-style) only for files/functions in focus

This is how you get “interactive latency” without multi-core and without compromising richness *when you actually need it*.

---

## Concrete recommendations for *your* codebase (what I’d change first)

### 1) Turn off AST merge inside LibCST syntax extraction

In `SyntaxIndexOptions` (`codeintel/build/hamilton/native/options/ingestion.py`):

* Set `emit_ast_nodes=False` in your config/profile **if `core.ast_nodes` is already produced elsewhere**.

Why: `codeintel/ingestion/compute/cst_extract.py` currently builds AST nodes inside `_extract_module_syntax(...)` when `emit_ast_nodes` is enabled — that’s an extra AST parse pass per file.

---

### 2) Collapse AST/docstrings/symtable/bytecode into a shared per-file frontend artifact

Even if you keep them as separate targets, introduce **one upstream node** that produces:

* `source_bytes`, `source_text`, `encoding`, `LineIndexedSource`
* `ast_tree` (parsed once)
* `compiled_code` (compiled once from AST tree)

Then rewrite:

* `DocstringsExtractStep` to accept `ast_tree` instead of parsing
* `SymtableExtractStep` to accept `ast_tree` for anchors and `source_text` for symtable
* `DisExtractStep` to compile from `ast_tree` (or consume `compiled_code`)

This is the single most “best-in-class engineering” move you can make without changing outputs.

---

### 3) Rewrite the hot `to_pylist()` sections in graph assembly (especially flow plane)

Specifically in `codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`:

* Replace the Python list comprehension loops that compute ordinals and extras.

Instead:

* derive ordinals in a columnar way (hash over key columns)
* keep `extras_json` either constant/NULL or computed via joins to a small lookup table

This is *very* likely to be one of your largest remaining CPU sinks once parsing is fixed.

---

### 4) Re-architect `syntax_augment` to avoid building IntervalTrees for everything

In `codeintel/build/hamilton/native/ingestion/syntax_augment.py`:

* do an exact span join first
* only build SpanResolver for files and nodes that didn’t match

And ideally:

* modify `SpanResolver` to support “exact-only mode” (skip IntervalTree adds).

---

### 5) Don’t pay “full persistence cost” for intermediate tables unless you need it

If your run writes many large tables and builds manifests/observations for each, I/O can become a major share of wall time.

A very pragmatic “speed mode”:

* persist only `graph.cpg_nodes` / `graph.cpg_edges` (+ a few essential lineage tables)
* keep everything else ephemeral in-memory for that run

This isn’t multi-core; it’s just avoiding unnecessary disk work.

---

## The fastest path to clarity: identify *your* top 3 hotspots

You already have a structure that tracks target durations via `TargetRunRecord.duration_ms` (`codeintel/build/hamilton/run_records.py`).

Best-in-class debugging pattern:

* print a “top N slowest targets” summary at the end of a run, including:

  * duration
  * rows emitted
  * bytes written (if known)

That immediately tells you whether the 11 minutes is:

* parsing dominated (LibCST / SCIP / bytecode)
* or row-processing dominated (`to_pylist` + hashing + payload encoding)
* or I/O dominated (parquet writes + manifests)

---

## Bottom line

* **No**, you are not hitting an unavoidable fundamental limit at 11 minutes — there are clear engineering issues that can plausibly yield **multi‑X speedups**.
* **Yes**, there *are* fundamental limits if you demand “full richness everywhere” on one core — which is why best-in-class systems use **progressive enrichment** + caching.
* If you want the highest ROI without multi-core:

  1. **Unify AST parse / compile** (stop doing it 3–5 times per file)
  2. **Eliminate `to_pylist()` + per-row hashing** in graph assembly
  3. **Exact-span join first; interval fallback second** in `syntax_augment`

If you want, I can go one level more concrete and give you a **surgical “performance PR plan”** that is specifically ordered by impact, with:

* the exact new shared “py_frontend artifact” node signatures,
* the exact places to route existing extraction logic to reuse the artifact,
* and a targeted rewrite sketch for `syntax_augment` + `cpg2/planes/flow.py` to eliminate the row-wise hashing/encoding.

Below is a **surgical, impact-ordered “performance PR plan”** that keeps things **build-only** (no `codeintel.storage` imports) and stays aligned with your existing Stage structure (AST/CST/SCIP ingest → `syntax_augment` → CPG2 planes). I’m going to be very explicit about **(a)** the *new shared `py_frontend` artifact node signatures*, **(b)** *exact routing points* where your existing extraction logic should reuse it, and **(c)** a *targeted rewrite sketch* for `syntax_augment` + `cpg2/planes/flow.py` to eliminate the current **row-wise hashing/encoding** hotspots.

I’ll also answer the **pyarrow/polars** and **msgspec** questions at the end.

---

## Why you’re stuck at ~11 minutes right now

From reviewing your current implementation patterns, the runtime is dominated by three avoidable classes of work:

1. **Repeated source I/O + repeated decoding**
   Multiple targets read the same `.py` bytes repeatedly (file_state/hash, line index, AST, CST/syntax_index, docstrings, symtable, bytecode, tree-sitter). Even if OS caches help, you’re still paying Python overhead, decode costs, and allocator churn.

2. **Repeated parsing / compilation**

   * LibCST parsing is expensive; if both `cst` and `syntax_index` run independently, you parse twice.
   * Docstrings extraction parses an AST even if you already parsed it for `ast`.
   * Symtable and bytecode often compile the same source separately.

3. **Row-wise Python loops over Arrow tables** (the really brutal part)

   * `syntax_augment` converts large Arrow tables to Python (`.to_pylist()`) and does per-row span matching + dict mutation.
   * CPG2 Flow plane computes ordinals and payloads using **per-row hashing + per-row msgpack encoding**.

Those last two are the biggest “why it feels impossible to get to 30 seconds” culprits. Your parsing cost is real, but your **Python-level per-row work** is what’s crushing throughput.

---

## Performance PR plan (ordered by impact)

### PR 1 — Add a shared `py_frontend` artifact and route *all ingestion* through it

**Goal:** Read/decode each module’s source **once** per run, and allow downstream extractors to reuse the same cached bytes/text/line-index (and optionally AST/CST/tree-sitter/code object).

This PR is your single highest ROI step because it immediately removes duplicated I/O and makes later PRs (deduped parsing + vectorized augmentation) easy.

#### 1A) New build-only artifact type

Create a small build-only “frontend service” object. Think of it as a caching *source/parse provider*, not a storage layer.

```python
# src/codeintel/build/hamilton/native/ingestion/py_frontend.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable

@dataclass(frozen=True)
class PyFrontendOptions:
    # source caching
    cache_bytes: bool = True
    cache_text: bool = True
    cache_line_index: bool = True

    # parse caching (start conservative; turn on as PR2 lands)
    cache_ast: bool = False
    cache_cst: bool = False
    cache_tree_sitter: bool = False
    cache_codeobj: bool = False
    cache_symtable: bool = False

    # safety
    decode_errors: str = "replace"

    # memory bound knobs
    max_cache_entries: int = 4096
    max_cache_bytes: int = 512 * 1024 * 1024  # 512MB

class PyFrontend:
    """Build-only service: repo_root + LRU caches + helpers."""

    def __init__(self, repo_root, opts: PyFrontendOptions): ...
    def get_bytes(self, rel_path: str) -> bytes: ...
    def get_text(self, rel_path: str) -> str: ...
    def get_line_index(self, rel_path: str): ...  # your line-index type

    # optional parse providers (PR2 enables these)
    def get_ast(self, rel_path: str): ...
    def get_cst(self, rel_path: str): ...
    def get_tree_sitter(self, rel_path: str): ...
    def get_codeobj(self, rel_path: str): ...
    def get_symtable(self, rel_path: str): ...
```

#### 1B) Exact new Hamilton nodes (signatures)

```python
# src/codeintel/build/hamilton/native/ingestion/py_frontend.py

def py_frontend__options(env) -> PyFrontendOptions:
    ...

def py_frontend(env, py_frontend__options: PyFrontendOptions) -> PyFrontend:
    """Shared artifact node: constructed once per DAG run."""
    ...

def py_frontend__python_module_records(
    t__modules,
    module_records: tuple,  # tuple[ModuleRecord, ...]
) -> tuple:
    """Filter to python modules only (endswith .py, language==python, etc)."""
    ...

def py_frontend__rel_paths(
    py_frontend__python_module_records: tuple,
) -> tuple[str, ...]:
    """Convenience for targets that currently depend on q__core__modules."""
    ...
```

#### 1C) Exact routing points (where to reuse it)

Update these run nodes to accept `py_frontend: PyFrontend` and/or reuse its cached discovery:

* `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`

  * `t__ast__run(..., py_frontend: PyFrontend, py_frontend__python_module_records, ...)`
  * `t__cst__run(..., py_frontend: PyFrontend, ...)`
  * `t__syntax_index__run(..., py_frontend: PyFrontend, ...)`
  * `t__docstrings__run(..., py_frontend: PyFrontend, ...)`
  * `t__symtable__run(..., py_frontend: PyFrontend, ...)`
  * `t__bytecode__run(..., py_frontend: PyFrontend, ...)`
  * `t__inspect__run(..., py_frontend: PyFrontend, ...)` (for import resolution / module loading helpers)
* `src/codeintel/build/hamilton/native/ingestion/tree_sitter.py`

  * `t__tree_sitter_index__run(..., py_frontend: PyFrontend, ...)`
* `src/codeintel/build/hamilton/native/ingestion/file_line_index.py`

  * Rewrite to stop reading files directly; instead consume `py_frontend__rel_paths` + `py_frontend.get_line_index(rel_path)`.

This is still fully consistent with your separation goal: build produces datasets; storage remains optional downstream. Your own earlier design notes explicitly call out that build outputs should be Arrow datasets and remain decoupled from DuckDB/storage. 

---

### PR 2 — Collapse duplicate parsing/compilation by sharing “run once” results

**Goal:** Ensure you don’t parse/compile the same file multiple times in the same run.

This PR is strictly about DAG wiring, not algorithms.

#### 2A) Unify `cst` + `syntax_index` to one LibCST pass

Your lineage/design already treats `syntax_index` as the stage that emits the structural syntax tables; `syntax_augment` then builds the weld mapping + coverage. 

**Do this:**

* Introduce a shared internal node:

```python
def _t__cst_bundle__run(
    env,
    py_frontend: PyFrontend,
    py_frontend__python_module_records,
    catalog,
    ...
) -> ToolResultBundle:
    """Runs LibCST parsing once, returns all tables needed by both targets."""
```

* `t__cst__save` pulls only `core.cst_nodes` out of the bundle.
* `t__syntax_index__save` pulls `core.parse_manifest`, `core.syntax_*`, etc, out of the same bundle.

**Result:** One LibCST parse per file, not two.

#### 2B) Unify `ast` + `docstrings` to one AST parse

Same approach:

* `_t__ast_bundle__run(...)` produces both `core.ast_nodes` and `core.docstrings` from one `ast.parse()`.

#### 2C) Unify `bytecode` + `symtable` compilation paths (optional but high ROI)

Even if you can’t share CPython internals perfectly, you can share:

* source text,
* filename normalization,
* compilation flags,
* and (often) the compiled code object across:

  * `dis.get_instructions(codeobj)`
  * exception table extraction
  * other bytecode tables

This interacts with your bytecode/symtable/inspect tables, which your earlier notes treat as first-class Stage 2 outputs. 

---

### PR 3 — Rewrite `syntax_augment` to be “join-first + sparse fallback,” eliminating `.to_pylist()` hot paths

Your own stage mapping calls out that `syntax_augment` already emits:

* `core.ts_syntax_node_xref`
* `core.ts_weld_coverage` 

Right now, your implementation pays a huge Python tax by converting entire tables to Python and doing row-wise matching + dict mutation.

#### 3A) New structure: exact-join first, fuzzy fallback second

**Step 1: exact join on byte spans** (fast, columnar)

* Build keyed projections:

  * `syntax_nodes_key = (repo, commit, rel_path, start_byte, end_byte) -> syntax_node_id`
  * `ts_nodes_key = (repo, commit, rel_path, start_byte, end_byte) -> ts_node_id`

* Do `arrow_join_tables(ts_nodes_key, syntax_nodes_key, join_keys=[repo,commit,rel_path,start_byte,end_byte], how="left")`

This yields:

* matched rows: `syntax_node_id != null` with `match_kind="EXACT"`
* unmatched rows: `syntax_node_id == null` → handle later

**Step 2: fuzzy fallback only for unmatched** (small)

Only for the `unmatched_ts_nodes` subset:

* group by `rel_path`
* build a SpanResolver once per file from the *syntax nodes for that file* (not global)
* resolve (BYTE_RANGE / ADJACENT_POINT / etc)

This keeps Python loops proportional to “hard cases,” not proportional to the entire repo.

#### 3B) Stop mutating `extras_json` on `syntax_nodes` row-by-row

Instead of:

* converting `syntax_nodes` to Python dicts,
* attaching `ts_nodes` lists into `extras_json` per syntax node,

do this columnar:

1. Join xref rows with `core.ts_nodes` to get payload columns (node_type, field_name, etc).
2. Build a **struct payload column** (Arrow struct).
3. `group_by(syntax_node_id).aggregate(list(payload_struct))`
4. Left-join that aggregated list back onto `syntax_nodes` as a new column (e.g. `ts_payloads`).

If you truly need `extras_json` to contain that payload, do the final “encode to msgpack” at **export time** (or behind an option), not during augmentation.

#### 3C) Compute weld coverage purely by grouping, not loops

Coverage table is naturally:

* `ts_node_count`
* `mapped_to_syntax_count`
* `coverage_ratio`
  grouped by `(repo, commit, rel_path, producer, language)`.

That is a group-by aggregate, not a Python loop.

---

### PR 4 — Rewrite CPG2 Flow plane to eliminate row-wise hashing *and* row-wise msgpack encoding

Your current Flow plane does the right joins, but pays a lot of CPU on:

* per-row `stable_int_hash(...)` for ordinals
* per-row `encode_payload({...})` for `extras_json`

Your earlier design notes already point toward using Polars for scalable grouping/sorting when needed.  And Hamilton supports column-subDAG execution over Polars via `with_columns`, which is exactly the right style for “compute these derived columns efficiently inside the frame engine.” 

#### 4A) Replace `cpg_edge_ordinal(...)` with deterministic **sort+cumcount** (no hashing)

**Principle:** You do not need a cryptographic hash to get deterministic ordinals.
If you sort by a stable key set, then `cumcount` (or row_number) is deterministic.

Example for CFG edges:

* Define a stable sort key:

  * `(repo, commit, function_goid_h128, src_cpg_node_id, dst_cpg_node_id, edge_kind, cfg_edge_kind, rel_path)`
* Sort by that key
* `ordinal = cumcount().over(group=(repo,commit,src_cpg_node_id,dst_cpg_node_id,edge_kind))`

This is **fast**, **deterministic**, and avoids row-wise hashing entirely.

**Implementation sketch (Polars window op):**

* Convert Arrow→Polars (zero-copy)
* `sort(...)`
* `with_columns(pl.cum_count().over([...group cols...]).alias("ordinal"))`
* Convert back

#### 4B) Stop encoding `extras_json` inside the plane

Instead, keep *typed columns* for plane-specific extras:

* CFG edges:

  * keep `cfg_edge_kind` as a column
* DFG edges:

  * keep `src_var`, `dst_var`, `via_phi`, `use_kind` as columns
* CDG edges:

  * keep `via_succ_block_id`, `via_edge_kind` as columns

Then:

* only at *final export / API boundary* (FastMCP output shaping), optionally pack these into msgpack `extras_json`.

This single change typically drops Flow plane runtime by a huge factor because msgpack encoding is still Python-level work.

**If you must preserve `extras_json` in the persisted table**, add a config:

* `FlowPlaneOptions.emit_extras_json: bool = False` (default false for full-repo runs)
* set true only for small runs/tests.

#### 4C) Also fix the hidden big one: Anchor-map ID generation

Anywhere you compute node IDs via per-row hashing (e.g. `stable_decimal_id`), you will burn CPU.

For anchor maps and IDs, prefer:

* **sort unique keys**
* assign **dense sequential IDs**
* (optionally) compute a stable “external id” later if you truly need it

This is dramatically faster than hashing JSON per row.

---

### PR 5 — Add “fast path” toggles and a small profiler summary (no multicore required)

This PR is small but helps you keep speed regressions from creeping back in.

Add:

* `--fast` mode that:

  * disables extras packing
  * disables the expensive fuzzy-weld fallback except for unmatched files above a threshold
  * turns on cached frontend
* a per-target time + row-count summary

This aligns with your earlier emphasis on coverage/quality reporting being critical to robustness/debugging. 

---

## Targeted rewrite sketch: `syntax_augment` (what it becomes)

Here’s the “shape” I recommend for the file after PR3—still the same conceptual stage, but fast:

### New nodes / functions inside `syntax_augment.py`

1. `syntax_augment__fallback_paths(parse_manifest) -> pa.Array[str]`
   Columnar filter on parse failures.

2. `syntax_augment__syntax_nodes(...) -> pa.Table`
   Apply fallback once, using Arrow filters/concats (you already do this part well).

3. `ts_syntax_node_xref__exact(ts_nodes, syntax_nodes) -> pa.Table`
   Exact join on `(repo,commit,rel_path,start_byte,end_byte)`; label `match_kind="EXACT"`.

4. `ts_syntax_node_xref__fuzzy(unmatched_ts_nodes, syntax_nodes) -> pa.Table`
   Only unmatched rows; grouped per file; SpanResolver.

5. `ts_syntax_node_xref(ts_syntax_node_xref__exact, ts_syntax_node_xref__fuzzy) -> pa.Table`
   Union + align to contract.

6. `ts_payloads_by_syntax_node(ts_syntax_node_xref, ts_nodes) -> pa.Table`
   Group-by list aggregation producing:

   * `syntax_node_id`
   * `ts_payloads: list<struct<...>>`

7. `syntax_nodes_augmented(syntax_nodes, ts_payloads_by_syntax_node) -> pa.Table`
   Join `ts_payloads` onto syntax nodes (no dict mutation, no msgpack encoding here).

8. `ts_weld_coverage(ts_syntax_node_xref, ts_nodes) -> pa.Table`
   Pure group-by counts and ratio.

This preserves what Stage 3 is supposed to produce (xref + coverage) while eliminating the current Python-row bottlenecks. 

---

## Targeted rewrite sketch: `cpg2/planes/flow.py` (what it becomes)

### Replace the “hash+encode per row” with “polars window + typed extras”

For each of `cfg2_edges`, `dfg2_edges`, `cdg2_edges`:

1. Keep your join pattern (Arrow joins are good).
2. Convert to Polars once for:

   * deterministic sort
   * ordinal computation by `cum_count().over(...)`
3. Return Arrow with:

   * required CPG edge columns
   * typed extras columns
4. Only encode payload bytes if an option says to.

This directly follows the approach your own earlier notes suggested: do scalable group/sort logic in Polars rather than Python loops.  Hamilton’s `with_columns` pattern is also designed for exactly this kind of “column-subDAG executed inside the dataframe engine.” 

---

## PyArrow vs Polars: where you should lean harder (and where not)

### Use PyArrow for

* joins (`Table.join`)
* filters/masks
* projections / column selection
* partitioned dataset writes

This is already your “Arrow-first” direction and it’s correct.

### Use Polars *surgically* for

* **window functions** (row_number / cumcount / rank)
* **group-by list aggregation**
* **large sorts**
* **dedupe + stable ordering**

These are exactly the operations you currently emulate with Python loops and hashing.

**Important note:** Polars may use multiple threads internally. That’s not “you implementing multicore complexity,” but if you want to keep it truly single-threaded, you can cap it via env/config. Either way, the bigger win is eliminating Python loops.

### Don’t use Polars for

* span/interval matching (that remains a custom algorithm problem)
* parsing/AST/CST/tree-sitter work

---

## msgspec: yes, but only if you stop decoding/encoding in the hot path

You’re already using msgspec msgpack payloads in places (good). The key is: **treat extras as opaque bytes until you absolutely need them**.

### Best-in-class recommendation for `extras_json`

1. In build tables, prefer:

   * typed columns (struct/list/primitive) for “frequent” fields
   * *or* `extras_msgpack: binary` if truly schema-less

2. Do **not** msgpack-encode per row during graph construction.
   Encode only:

   * at export time (FastMCP response shaping)
   * or in a small “finalize” node gated behind an option

### “Keep in msgspec form, decode only when needed”

Yes—if your `extras_json` column is already `binary` msgpack bytes:

* you can pass it through your pipeline without decoding
* only your FastMCP tool output code decodes it for the subset of nodes/edges the LLM requests

If you want an even cleaner pattern:

* define a `msgspec.Struct` type for each payload shape (CFG extras, DFG extras, etc.)
* at export time, decode into those structs
* this gives faster decode + validation than untyped dicts

But again: **the main speed win is “don’t encode/decode at build time.”**

---

## One last reality check: can you get to ~30 seconds single-core?

If your repo is truly large, parsing + tree-sitter + LibCST + AST + compilation may still be a few minutes on one core.

However, the PRs above should unlock **multiplicative** improvements because you’re currently paying for:

* duplicate parses,
* huge Python loops over Arrow tables,
* and per-row hashing/encoding.

For many codebases, **dropping from ~11 minutes to ~1–3 minutes** single-core is realistic after:

* frontend reuse,
* deduped parsing,
* join-first augmentation,
* and eliminating per-row hashing/encoding.

Getting to ~30 seconds usually requires at least one of:

* partial/incremental rebuilds (only changed files)
* or letting Polars/Arrow use multiple threads
* or narrowing the extraction set (fast mode)

But you can absolutely get *dramatically* faster without you hand-implementing multicore.

---

## References from your attached design notes

* Stage 3 `syntax_augment` emits the TS→syntax weld mapping and weld coverage tables (the outputs we’re optimizing around). 
* The build-vs-storage decoupling principle (build produces Arrow datasets; storage/DuckDB is downstream/optional). 
* Hamilton’s `with_columns` execution model supports running a column-subDAG inside dataframe engines (including Polars), which is ideal for the Flow-plane rewrite. 
* Earlier Polars sketch for scalable grouping/sorting/capping candidates (same mechanics we’re applying to ordinals/aggregation). 

---

If you want, I can take PR3 + PR4 and write them as a **literal edit checklist** against your current files (like “replace node X with X_exact + X_fuzzy + X_union,” “these are the join keys,” “these columns become typed extras”), but the plan above is the clean, best-in-class route to “dramatically faster without multicore complexity.”

Below is the “literal edit checklist” for the two performance PRs we were discussing (PR3 + PR4), written **against your current `src_revised.zip` layout** and staying **build/ingestion-only** (no `src/codeintel/storage` dependencies introduced).

I’m going to be very direct and mechanical here—this is meant to be something you can implement as a sequence of commits without re-arguing architecture each time.

---

## PR3 (Highest ROI): Introduce a shared LibCST “frontend artifact” run node and route `cst` + `syntax_index` through it

### Why this is the best first cut

In your current code, **`CstExtractStep` is executed twice across the same module set**:

* `t__cst__run()` executes `CstExtractStep(... emit_ast_nodes=False ...)`
* `t__syntax_index__run()` executes `CstExtractStep(... emit_ast_nodes=options.emit_ast_nodes ...)`

Both runs re-read + re-parse + re-walk the repo with LibCST, which is usually one of the most expensive single-thread costs. Collapsing these into one pass is often a “minutes → tens of seconds” type win all by itself on large repos (depending on repo size and LibCST workload).

### PR3 deliverable

Add a shared run node that executes LibCST once and returns the full result bundle, then make both target run nodes **slice** that output instead of executing LibCST themselves.

---

### PR3.1 — Edit `codeintel/build/hamilton/native/ingestion/extraction_targets.py`

#### A) Add a new combined tool output dataclass

Add this near your other `ToolStepOutput` dataclasses (around `CstToolOutput` / `SyntaxIndexToolOutput`):

* **New dataclass:** `CstSyntaxIndexToolOutput(ToolStepOutput)`
* Fields should cover the union of what you need for **both** targets:

  * `cst_rows: pa.Table`
  * `parse_manifest_rows: pa.Table`
  * `syntax_spans_rows: pa.Table`
  * `syntax_nodes_rows: pa.Table`
  * `syntax_edges_rows: pa.Table`
  * `syntax_scopes_rows: pa.Table`
  * `syntax_defs_rows: pa.Table`
  * `syntax_refs_rows: pa.Table`
  * `syntax_calls_rows: pa.Table`
  * `syntax_call_args_rows: pa.Table`
  * `syntax_func_params_rows: pa.Table`
  * `syntax_imports_rows: pa.Table`
  * plus row counts for each (mirror what `SyntaxIndexToolOutput` does, and add `cst_row_count`)

This is just a container—no schema changes, no storage.

#### B) Add a new shared run node (the actual “artifact producer”)

Add **one** new Hamilton node function:

**New node signature (exact shape you should implement):**

```python
def py_frontend__cst_syntax_index__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> CstSyntaxIndexToolOutput:
    ...
```

**Mechanical implementation checklist:**

* Copy the `_module_inventory_precheck(...)` pattern and warning merge behavior you already use in `t__cst__run` / `t__syntax_index__run`.
* Create one `ToolRunContext` (pick a stable name; I recommend `target_name="py_frontend"` so logs/metrics aren’t confusing).
* Instantiate `FilesystemDiscoveryAdapter(env.snapshot.repo_root)` once.
* Load **both** option objects, because you currently allow separate configs:

  * `options_cst = load_target_options(... target_name=CST_TARGET_NAME, options_type=SyntaxIndexOptions)`
  * `options_syntax = load_target_options(... target_name=SYNTAX_INDEX_TARGET_NAME, options_type=SyntaxIndexOptions)`
* Merge options deterministically (this is important so behavior is stable):

  * `batch_size = max(options_cst.batch_size, options_syntax.batch_size)` (max usually wins for throughput)
  * `emit_ast_nodes = bool(options_syntax.emit_ast_nodes)` (cst target never required it; syntax_index owns that knob)
* Execute **one** `CstExtractStep`:

  * `step = CstExtractStep(discovery=discovery, emit_ast_nodes=emit_ast_nodes, batch_size=batch_size)`
  * `extract_result = step.execute(module_records, repo=env.snapshot.repo, commit=env.snapshot.commit)`
* Return `CstSyntaxIndexToolOutput(...)` populated from `extract_result.*_reader` and row counts.

#### C) Rewrite `t__cst__run` to slice the shared run output

**Replace the body** of `t__cst__run` with a wrapper around the shared node.

**New signature:**

```python
def t__cst__run(
    py_frontend__cst_syntax_index__run: CstSyntaxIndexToolOutput,
) -> CstToolOutput:
```

**Wrapper logic:**

* Take `result = py_frontend__cst_syntax_index__run.result`
* Return `CstToolOutput(result=result, rows=<combined.cst_rows>, row_count=<combined.cst_row_count>)`
* Keep your warnings logging loop (or move it to the shared node, but do it once).

#### D) Rewrite `t__syntax_index__run` to slice the shared run output

Same idea:

**New signature:**

```python
def t__syntax_index__run(
    py_frontend__cst_syntax_index__run: CstSyntaxIndexToolOutput,
) -> SyntaxIndexToolOutput:
```

Populate `SyntaxIndexToolOutput` from the combined output fields.

#### E) Do **not** change your attach_tool_target_template calls

At the bottom of the file you already have:

* `_CST_TARGET_SPEC` attached to `t__cst__run`
* `_SYNTAX_INDEX_TARGET_SPEC` attached to `t__syntax_index__run`

Those remain correct—only the internals change.

#### F) Optional but recommended: keep the shared node discoverable

Hamilton discovery in your project should still see `py_frontend__cst_syntax_index__run` because it’s a top-level function. If you have any module filtering based on `__all__`, add it to `__all__` in this file.

---

### PR3.2 — What you should NOT do (to keep this surgical)

* Do not modify `codeintel/ingestion/compute/cst_extract.py` yet.
* Do not touch schema registries or storage.
* Do not introduce new tables.

This PR should be a clean “duplicate work removal” change.

---

## PR4: Remove the biggest remaining Python hotspots (row-wise `to_pylist()` + JSON hashing/encoding)

This PR is specifically aimed at the two places that are currently “Python-loop heavy” and scale terribly with repo size:

1. `codeintel/build/hamilton/native/ingestion/syntax_augment.py`
2. `codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`

### The pattern we’re eliminating

Anything like:

* `table.to_pylist()` over large tables
* Per-row `json.dumps(...)` hashing (`stable_int_hash`, `stable_decimal_id`)
* Per-row dict construction for extras, then `encode_payload(...)`

Instead:

* Iterate **record batches**, not whole tables
* Compute ordinals/extras in **batch loops**
* Replace JSON hashing with **msgspec msgpack hashing** (still deterministic, far less overhead)

---

## PR4.A — Syntax augment rewrite: replace `xref_rows + nodes_rows = to_pylist()` with a 3-pass xref + join-built augmentation

### PR4.A.1 — Edit `codeintel/build/hamilton/native/options/ingestion.py`

Extend `SyntaxAugmentOptions` (currently only 2 flags) to let you turn off the heaviest part when you don’t need it.

Add:

* `augment_ts_extras: bool = False`
  (meaning: don’t embed ts payloads into `core.syntax_nodes_augmented.extras_json` unless explicitly requested)
* `xref_mode: Literal["exact_only", "exact_then_fuzzy"] = "exact_then_fuzzy"`
  (so you can force a fast path)
* `xref_limit_to_paths: Literal["fallback_only", "all"] = "fallback_only"`
  (defaulting to fallback-only is a huge win on healthy repos)

This is “build config only,” no storage changes.

### PR4.A.2 — Edit `codeintel/build/hamilton/native/ingestion/syntax_augment.py`

#### A) Split the xref computation into three nodes

Right now `syntax_augment__frames` always does:

* build SpanResolver index from syntax nodes using `to_pylist()`
* iterate `ts_nodes.to_pylist()` to create `xref_rows`

Replace that with:

1. **Exact pass (join-built):**

   * **New node:** `syntax_augment__ts_syntax_node_xref__exact`
   * Join keys (use these exactly):

     * `["repo", "commit", "rel_path", "start_byte", "end_byte"]`
   * Output columns:

     * required contract cols for `core.ts_syntax_node_xref`:

       * `repo, commit, rel_path, language, producer, ts_node_id, syntax_node_id, match_kind, candidate_count`
   * `match_kind = "EXACT"` when match found else `"NONE"`
   * `candidate_count = 1` if match found else `0`

2. **Fuzzy pass (algorithm-built, but only for unmatched):**

   * **New node:** `syntax_augment__ts_syntax_node_xref__fuzzy`
   * Input: `ts_nodes` filtered to rows where exact match failed
   * Build SpanResolver **per rel_path**, but do it from **batch iteration** (no whole-table `.to_pylist()`):

     * iterate `syntax_nodes.to_batches(...)`
     * for each row in batch, add span to resolver keyed by `rel_path`
   * For fuzzy matching, only emit a row when you actually matched something.
   * `match_kind` should reflect your resolver result kind.
   * `candidate_count` from resolver result.

3. **Union pass:**

   * **New node:** `syntax_augment__ts_syntax_node_xref`
   * Union exact + fuzzy with precedence:

     * Prefer EXACT if both exist for the same `ts_node_id`
   * Deduplicate on `ts_node_id` (and keep the chosen one)

This directly matches your “X_exact + X_fuzzy + X_union” request.

#### B) Make extras augmentation optional and join-built

Today you do:

* `nodes_rows = syntax_nodes.to_pylist()`
* mutate `extras_json` dict
* rebuild table via `table_for_rows`

Replace with:

* If `syntax_augment__options.augment_ts_extras` is **False**:

  * do **not** touch `extras_json` at all
  * just return canonical syntax_nodes as `core.syntax_nodes_augmented` (aligned/deduped)

* If `augment_ts_extras` is **True**:

  1. Join `ts_nodes` with `ts_syntax_node_xref` (filtered to match_kind != "NONE") on `ts_node_id`
  2. Project the TS payload columns you currently serialize into extras:

     * `ts_node_id, ts_node_type, start_byte, end_byte, …, match_kind`
  3. Aggregate per `syntax_node_id` into a deterministic list:

     * sort key: `(start_byte, end_byte, ts_node_id)`
  4. Encode *one blob per syntax node*:

     * extras payload should be `{ "ts_nodes": [...] }`
     * do **not** decode old extras unless required:

       * fast path: if original `extras_json` is null → set it to encoded ts payload
       * slow path: if original `extras_json` not null → decode + merge + re-encode (rare)

This flips the work from “per syntax row mutation” to “join → group → encode per group”.

#### C) Rewrite weld coverage to be group-by + join (no Python dict loops)

Replace `_weld_coverage_table`’s Python loops with:

* group-by counts on `ts_nodes` per `(repo, commit, rel_path, language)`
* group-by counts on `xref` filtered to matched rows
* join and compute ratio vectorized

#### D) Delete (or stop calling) the worst offenders

Once the above is in place, these become dead weight (or only used in the fuzzy node):

* `_ts_node_index`
* `_payloads_by_syntax_node`
* `_apply_ts_payloads`
* `nodes_rows = syntax_nodes.to_pylist()` block in `syntax_augment__frames`

---

## PR4.B — Flow plane rewrite: eliminate per-row hashing/encoding in `cpg2/planes/flow.py`

### PR4.B.1 — Add msgspec-based fast hashing primitives

#### A) Edit `codeintel/build/graphs/assembly/ids.py`

Add **new** functions (don’t overwrite the old ones unless you’re comfortable changing IDs globally):

* `stable_int_hash_msgpack(parts: object, *, digest_size: int, modulus: int) -> int`
* `stable_decimal_id_msgpack(parts: object, *, digest_size: int = 16) -> int`

Implementation rule:

* Use `encode_payload(...)` to produce bytes (msgspec msgpack), then `hashlib.blake2b` on those bytes
* No `json.dumps(...)`, no `default=str`, no dict key sorting cost

This is deterministic and *dramatically* faster than JSON serialization.

#### B) Edit `codeintel/build/hamilton/native/graphs/cpg2/ids.py`

Add a new ordinal helper that avoids dict construction:

* `def cpg_edge_ordinal_parts(table_key: str, parts: tuple[object, ...]) -> int:`

Where `parts` is a stable tuple in a fixed column order.

Example:

* CFG ordinal parts tuple:

  * `(function_goid_h128, src_block_id, dst_block_id, cfg_edge_kind)`
* DFG ordinal parts tuple:

  * `(function_goid_h128, src_block_id, dst_block_id, src_var, dst_var)`
* CDG ordinal parts tuple:

  * `(function_goid_h128, src_block_id, dst_block_id, via_succ_block_id)`

Then hash:

* `stable_int_hash_msgpack((table_key, parts), digest_size=8, modulus=ORDINAL_MOD)`

This preserves “stable deterministic ordinal” semantics while removing JSON/dict overhead.

---

### PR4.B.2 — Rewrite `codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`

#### A) Replace every `.to_pylist()` loop that computes ordinals/extras

You have three hotspots:

* `cpg2_edges__cfg_edges`: builds `ordinals = [cpg_edge_ordinal(... row.get ... ) for row in joined.select(...).to_pylist()]`
* `cpg2_edges__dfg_edges`: same pattern, plus `_payload_bytes(...)` per row
* `cpg2_edges__cdg_edges`: same pattern, plus `edge_kinds = [row.get(...)]`

**Replace all of them** with batch-wise builders.

#### B) Add internal “batch builders” in `flow.py`

Add helpers (names can vary; these are the functions you want conceptually):

1. `_build_ordinals(table_key: str, table: pa.Table, cols: list[str]) -> pa.Array`

* Iterate `for batch in table.to_batches(max_chunksize=DEFAULT_ARROW_BATCH_SIZE):`
* For the batch:

  * pull each required column once (`col = batch.column(i).to_pylist()`)
  * build `parts_tuple` per row using zipped lists
  * call `cpg_edge_ordinal_parts(table_key, parts_tuple)`
* Append to a python list of ints
* Return `pa.array(ints, type=pa.int64())`

2. `_build_extras_cfg(batch)`, `_build_extras_dfg(batch)`, `_build_extras_cdg(batch) -> list[bytes | None]`

* Define **msgspec.Struct** payloads locally (or in a shared module) to avoid dict creation:

  * `CfgEdgeExtras(cfg_edge_kind: str | None)`
  * `DfgEdgeExtras(src_var: str | None, dst_var: str | None, edge_kind: str | None, via_phi: bool | None, use_kind: str | None)`
  * `CdgEdgeExtras(via_succ_block_id: str | None, via_edge_kind: str | None)`
* Encode via `msgspec.msgpack.encode(struct_instance)` directly (faster than dict → sanitize → encode)

3. Replace CDG edge_kind fill with vectorized compute
   Instead of:

```python
edge_kinds = [row.get("edge_kind") or "CDG" for row in joined.select(["edge_kind"]).to_pylist()]
```

use a null-fill:

* `pc.fill_null(joined["edge_kind"], "CDG")` (or `if_else(is_valid, edge_kind, "CDG")`)

#### C) Apply the helpers in each edge node

Mechanically:

* In `cpg2_edges__cfg_edges`:

  * `joined = joined.append_column("ordinal", _build_ordinals("graph.cfg_edges", joined, [...]))`
  * `joined = joined.append_column("extras_json", pa.array(extras_list, type=pa.binary()))`
  * no `.to_pylist()` anywhere

* In `cpg2_edges__dfg_edges`:

  * same, with DFG-specific parts + extras struct

* In `cpg2_edges__cdg_edges`:

  * same, with CDG-specific parts + extras struct
  * vectorized edge_kind fill

This is the “rewrite sketch” that actually removes the scaling poison.

---

## Where to use more PyArrow / Polars (and where not to)

### PyArrow opportunities you’re currently leaving on the table

1. **Group-by coverage and counts**

   * Replace Python dict-based counting (like `_weld_coverage_table`) with Arrow group-by aggregations.

2. **Null-fill / coalesce / if_else**

   * You already use `pc.if_else` in `_coalesce_rel_path`; extend that pattern everywhere you currently do Python-side `or "CDG"`.

3. **Batch iteration instead of full materialization**

   * Whenever you “must” do Python work, do it as `RecordBatch` loops, not `Table.to_pylist()`.

### Polars opportunities (worth it in exactly one place)

Polars is most valuable where you need:

* group-by aggregations that produce **list outputs**
* deterministic sorting inside groups

That maps perfectly to:

* “payload list per `syntax_node_id`” (ts payload aggregation) in `syntax_augment`

If you choose to use polars for that step:

* Convert Arrow → Polars with your existing helpers (`table_to_frame`)
* Do `group_by("syntax_node_id").agg(...)` building list columns
* Convert back to Arrow

If you truly want to stay “single-core” at the OS level, note: polars may use multithreading internally. If that’s a concern, you can cap it via environment, but most teams accept “library-internal parallelism” because it doesn’t add pipeline complexity.

---

## msgspec: is there value beyond what you’re doing?

### You are *already* getting the biggest msgspec win

Your `encode_payload()` stores extras as **msgpack bytes**. That is effectively “keeping it in msgspec form” already:

* You don’t pay JSON costs on disk
* You only decode when you call `decode_payload(...)`

### Where msgspec can help you *a lot more* (and should be part of PR4)

1. **Replace JSON hashing with msgpack hashing**

   * This is the single cleanest way to make `stable_*` hashing cheaper without giving up determinism.

2. **Typed extras without dict allocations**

   * Use `msgspec.Struct` for edge extras payloads (CFG/DFG/CDG), and encode directly.
   * This removes:

     * dict creation
     * sanitize recursion cost on mappings
     * (often) a chunk of GC pressure

### About “deserialize only if needed, otherwise keep msgspec form”

Best-in-class approach for your pipeline:

* Keep extras as `bytes` (msgpack) *throughout* the pipeline whenever possible
* Only decode:

  * at API boundaries (FastMCP / LLM agent request)
  * or when you truly must merge/inspect

In `syntax_augment`, your current algorithm *forces* a decode/merge because it mutates nested extras maps. The performance-friendly redesign is:

* **avoid merging** most of the time (fast path)
* or **add a new side-table** for TS payloads keyed by `syntax_node_id` (best overall, but would be a schema expansion)

Given your request to keep things conservative right now, PR4.A’s “fast path (extras null → set extras) / slow path decode only if non-null” is the pragmatic sweet spot.

---

## Quick “do this first” ordering inside PR3 + PR4

If you want the fastest path to a big runtime drop without changing semantics too much:

1. **PR3**: unify `CstExtractStep` execution for `cst` + `syntax_index` (single LibCST pass)
2. **PR4.B**: flow plane batch ordinals/extras + msgspec hashing (kills a major scaling hotspot)
3. **PR4.A**: syntax_augment xref exact/fuzzy/union + make ts extras optional + coverage group-by

---

