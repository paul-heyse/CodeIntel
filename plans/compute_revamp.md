If you’re willing to make breaking changes, the “best-in-class PyArrow” end state I’d aim for in CodeIntel is:

**Every derived dataset is produced by a small number of fused, columnar plans (scan → project → filter → join → aggregate → order), expressed in Arrow compute / expressions (not Python loops), executed by Acero when possible, and finalized by a single strict/tolerant contract+QA gate.**

That gets you (a) max throughput (C++ kernels + multithreading), (b) fewer intermediate materializations, and (c) one place to enforce correctness + produce diagnostics.

Below are the core changes + patterns I’d adopt.

---

## 1) Treat “Arrow expressions + Acero” as the default execution model

### Why

Acero is a **streaming execution engine** that runs a declared plan of compute nodes (scan/filter/project/hash_join/aggregate/etc.) and evaluates `pyarrow.compute` expressions efficiently in batches. It’s experimental, but it’s exactly the path to “fastest possible Arrow-native” execution. ([Apache Arrow][1])

### Pattern: build a plan, don’t hand-write per-node loops

```python
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.acero as acero

def build_edges_plan(dataset_path: str) -> pa.Table:
    dset = ds.dataset(dataset_path, format="parquet")

    # 1) Push down projection + filter at scan time when possible
    scan = acero.Declaration(
        "scan",
        acero.ScanNodeOptions(
            dset,
            columns={
                "src_id": ds.field("src_id"),
                "callee_ids": ds.field("callee_ids"),  # list<...>
                "kind": ds.field("kind"),
            },
            filter=(ds.field("kind") == "call"),
        ),
    )

    # 2) Project (compute new columns) using compute expressions
    # (For more complex cases, you'll add project/filter/join/aggregate nodes here.)

    # 3) Collect
    return scan.to_table(use_threads=True)

edges = build_edges_plan("/path/to/parquet_dir")
```

Notes:

* `ScanNodeOptions` can apply projection/filter pushdown (and can preserve implicit ordering if you need determinism) ([Apache Arrow][2])
* `Declaration.to_table(use_threads=...)` controls CPU threading for the plan ([Apache Arrow][3])

---

## 2) Make dataset scanning do more of the work (projection/filter as early as possible)

For parquet-backed steps, push down *everything* that can be expressed as expressions at scan time. The dataset `Scanner` API supports projecting columns **or computed expressions** and filtering by an expression (with pushdown when supported). ([Apache Arrow][4])

This is the highest ROI “free speed” because you avoid reading / materializing columns you’ll drop anyway.

---

## 3) Replace Python row loops with Arrow-native “explode list” (huge for edge builders)

A ton of graph/edge building becomes trivial if you store nested relationships as `list<T>` columns and explode them with:

* `pc.list_flatten(lists)` → the concatenated values
* `pc.list_parent_indices(lists)` → which parent row each flattened value came from

Both are Arrow compute kernels. ([Apache Arrow][5])

### Pattern: explode `src_id + list<dst_id>` into an edges table

```python
import pyarrow as pa
import pyarrow.compute as pc

def explode_edges(src_ids: pa.Array, dst_lists: pa.Array) -> pa.Table:
    # parent_idx: for each emitted dst value, which source row did it come from?
    parent_idx = pc.list_parent_indices(dst_lists)          # uint32/uint64
    dst_flat   = pc.list_flatten(dst_lists)                 # dst_id values

    src_rep    = pc.take(src_ids, parent_idx)               # repeat src_id to match dst_flat

    return pa.Table.from_arrays([src_rep, dst_flat], names=["src_id", "dst_id"])
```

This single pattern can delete a *lot* of bespoke “iterate rows → append dicts” code in `call_wiring`, `cpg2`, etc.

---

## 4) Canonicalize “dedupe + determinism” using group_by + hash_first (Arrow-native)

Instead of custom dedupe logic, use grouped aggregation with `first`/`hash_first` for each non-key column (Arrow supports the `hash_first` family for grouped aggregations). ([Apache Arrow][6])

Also: `Table.group_by(use_threads=True)` can produce unstable ordering; turn it off when you need deterministic outputs. ([Apache Arrow][7])

### Pattern: “keep first row by key” (drop duplicates)

```python
import pyarrow as pa

def dedupe_keep_first(t: pa.Table, keys: list[str]) -> pa.Table:
    non_keys = [c for c in t.column_names if c not in keys]
    aggs = [(c, "first") for c in non_keys]  # "first" == hash_first per docs
    # use_threads=False favors determinism (and can still be very fast if upstream is heavy)
    return t.group_by(keys, use_threads=False).aggregate(aggs)
```

If you want “best of both worlds” (fast + deterministic):

1. sort once by `[keys..., tie_breaker...]`
2. group_by(keys, use_threads=True).aggregate(first…)

---

## 5) Collapse “extras_json” into typed structs early (faster + higher data quality)

JSON strings are expensive and brittle. If you’re optimizing for throughput and fault tolerance, store “extras” as a `struct<...>` (or `map<...>`) and only serialize to JSON at the boundary (export/UI).

Arrow can build struct columns with `pc.make_struct(...)`, and you can later `table.flatten()` or extract fields with `pc.struct_field`. ([Apache Arrow][8])

### Pattern: typed extras struct

```python
import pyarrow.compute as pc

extras = pc.make_struct(
    pc.field("repo_id"),
    pc.field("parse_version"),
    field_names=["repo_id", "parse_version"],
)
# Then add extras as a column in a project step / or table.append_column(...)
```

This tends to shrink code sprawl too: instead of “build dict → json.dumps per row”, you keep it columnar.

---

## 6) Force “fast, predictable kernels” by normalizing chunks + threads up front

Two practical knobs that matter for throughput:

### (a) Combine tiny chunks early

Many small chunks = more overhead per kernel call. If memory isn’t a concern, combine. ([Apache Arrow][7])

```python
t = t.combine_chunks()
```

### (b) Own Arrow’s CPU & I/O thread pools explicitly

Arrow uses global thread pools for CPU work and I/O; you can set both. ([Apache Arrow][9])

```python
import pyarrow as pa
pa.set_cpu_count(32)       # CPU compute pool :contentReference[oaicite:10]{index=10}
pa.set_io_thread_count(32) # dataset scan / IO pool :contentReference[oaicite:11]{index=11}
```

(You can also control defaults via `OMP_NUM_THREADS` / `OMP_THREAD_LIMIT` as Arrow documents.) ([Apache Arrow][9])

---

## 7) Make “fault tolerance” a first-class output: strict mode + tolerant mode

For a production-grade, error-resilient pipeline:

* **Strict mode**: fail fast on contract violations / invalid casts / impossible invariants.
* **Tolerant mode**: *never throw* inside transform steps; instead:

  * compute an `is_valid` mask (columnar)
  * split output into `(good_table, error_table)` where error_table includes `row_id` + `error_code` + key fields

To make that easy, I’d introduce a single “finalize” gate that every Hamilton target uses:

* schema align / cast
* invariant checks (vectorized)
* dedupe policy
* emit alignment + error artifacts

This is where you get “fault tolerant” without polluting every node with ad hoc checks.

---

## 8) Practical “best-in-class” architecture change in your repo

If you *really* want to go all-in:

1. **Create an internal Arrow DSL** (tiny wrapper) that returns either:

   * an Acero `Declaration` (preferred), or
   * a `pa.Table` fallback for ops Acero can’t express

2. Make Hamilton nodes mostly do:

   * declare plan (pure)
   * execute plan
   * finalize(strict/tolerant)

3. Move all compute idioms (explode list, dedupe, safe divide/coalesce, struct extras, invariant validators) into one module, so “calc sprawl” stops expanding.

Acero gives you scan/project/filter/hash_join/aggregate/order primitives. ([Apache Arrow][10])
That set covers most of what build pipelines do if you model the intermediate data correctly (lists/structs instead of Python objects / JSON blobs).

---

### One optional “escape hatch” (still Arrow-native)

If you ever hit a ceiling with complex relational transforms, **DataFusion (Rust)** can execute Arrow plans/SQL with zero-copy interchange. It’s not “pyarrow.compute”, but it’s still Arrow memory and can be extremely fast. ([Apache DataFusion][11])

---

If you want one very actionable next step: pick *one* expensive graph builder (e.g., call edges) and refactor it to the **“list explode + Acero plan + finalize gate”** pattern. That single migration usually reveals ~80% of the repetitive helper surface you’ll want to standardize.

[1]: https://arrow.apache.org/docs/python/api/acero.html?utm_source=chatgpt.com "Acero - Streaming Execution Engine — Apache Arrow v22.0.0"
[2]: https://arrow.apache.org/docs/python/generated/pyarrow.acero.ScanNodeOptions.html?utm_source=chatgpt.com "pyarrow.acero.ScanNodeOptions — Apache Arrow v22.0.0"
[3]: https://arrow.apache.org/docs/python/generated/pyarrow.acero.Declaration.html?utm_source=chatgpt.com "pyarrow.acero.Declaration — Apache Arrow v22.0.0"
[4]: https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html?utm_source=chatgpt.com "pyarrow.dataset.Scanner — Apache Arrow v22.0.0"
[5]: https://arrow.apache.org/docs/python/generated/pyarrow.compute.list_flatten.html?utm_source=chatgpt.com "pyarrow.compute.list_flatten — Apache Arrow v22.0.0"
[6]: https://arrow.apache.org/docs/python/compute.html "Compute Functions — Apache Arrow v22.0.0"
[7]: https://arrow.apache.org/docs/python/generated/pyarrow.Table.html?utm_source=chatgpt.com "pyarrow.Table — Apache Arrow v22.0.0"
[8]: https://arrow.apache.org/docs/6.0/python/generated/pyarrow.compute.make_struct.html?utm_source=chatgpt.com "pyarrow.compute.make_struct — Apache Arrow v6.0.1"
[9]: https://arrow.apache.org/docs/python/generated/pyarrow.cpu_count.html?utm_source=chatgpt.com "pyarrow.cpu_count — Apache Arrow v22.0.0"
[10]: https://arrow.apache.org/docs/cpp/acero/user_guide.html?utm_source=chatgpt.com "Acero User’s Guide — Apache Arrow v22.0.0"
[11]: https://datafusion.apache.org/python/?utm_source=chatgpt.com "DataFusion in Python — Apache Arrow DataFusion documentation"


Below is a concrete “**list explode + Acero plan + finalize gate**” implementation plan for **CALLS edges** (today: `src/codeintel/build/hamilton/native/graphs/call_wiring.py::cpg_edges_calls`), with representative code patterns that fit your repo’s style.

The key shift is: **stop building `graph.cpg_edges_calls` via Python row loops**, and instead:

1. represent call targets as **one row per call** with a **`list<struct>` candidates column**
2. **explode** that list using Arrow compute (`list_flatten` + `list_parent_indices`)
3. do the heavy relational work (filter/join/project/order) via **Acero** execution plans
4. run a **single finalize gate** that handles alignment, dedupe, invariants, and “tolerant vs strict” behavior.

Acero is designed exactly for “express compute as a streaming execution plan” and supports nodes like `table_source`, `filter`, `project`, `hashjoin`, `aggregate`, etc.

---

## Phase 0 — Define the target “best-in-class” shape

### A) New internal intermediate (recommended): `graph.cpg_call_candidates`

One row per callsite, with a list of candidates:

* Call identity: `repo`, `commit`, `rel_path`, `call_id`, `call_node_id`
* Call metadata: `callee_symbol`, `call_kind`, `augop`, …
* **`candidates`: list<struct<...>>** where struct contains:

  * `callee_def_id`, `callee_goid_h128`
  * `target_role`, `binding_kind`, `origin`, `resolution_kind`
  * `confidence`
  * optionally `extras` (but see note below)

This will become the canonical thing you explode for:

* CALL edges
* RET edges
* arg→param wiring (later)

### B) Keep `graph.cpg_edges_calls` output (but change how it’s produced)

Output columns can remain compatible, but for speed I’d strongly prefer you **stop building per-row encoded `extras_json`**. Two options:

* **Option 1 (fastest + cleanest):** make metadata first-class columns (no encoding), and have `extras_json` be nullable or removed.
* **Option 2 (compatible):** keep `extras_json` but set it to the already-existing candidate extras (don’t re-encode per edge).

(Your `cpg2` plane currently does `decode_payload(row["extras_json"])`; if you change semantics, update that consumer accordingly.)

---

## Phase 1 — Add a reusable list-explode helper (Arrow compute)

Create a helper that explodes a `list<struct>` column into a flat table using:

* `pyarrow.compute.list_flatten`
* `pyarrow.compute.list_parent_indices`
* `pyarrow.compute.struct_field`

**New file:** `src/codeintel/build/tabular/explode_ops.py`

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import pyarrow as pa
import pyarrow.compute as pc


def explode_list_struct(
    table: pa.Table,
    *,
    list_col: str,
    parent_cols: Sequence[str],
    struct_fields: Mapping[str, str],
) -> pa.Table:
    """
    Explode a list<struct<...>> column into a row-per-element table.

    Parameters
    ----------
    table:
        Input table with list<struct> column.
    list_col:
        Name of the list column to explode.
    parent_cols:
        Columns to repeat for each exploded element.
    struct_fields:
        Mapping of struct field name -> output column name.

    Returns
    -------
    pa.Table
        Table with repeated parent columns + extracted struct fields.
    """
    lists = table[list_col]  # ChunkedArray ok
    parent_idx = pc.list_parent_indices(lists)  # indices of source row per emitted element
    flat_struct = pc.list_flatten(lists)        # the struct elements, flattened

    cols: dict[str, pa.Array] = {}
    for name in parent_cols:
        cols[name] = pc.take(table[name], parent_idx)

    for field_name, out_name in struct_fields.items():
        cols[out_name] = pc.struct_field(flat_struct, field_name)

    return pa.table(cols)
```

This helper is your reusable “explode primitive” you’ll use across the DAG.

---

## Phase 2 — Add a thin Acero exec helper (plan builder)

Acero nodes you’ll use for this refactor:

* `table_source` (`TableSourceNodeOptions`)
* `filter` (`FilterNodeOptions`)
* `project` (`ProjectNodeOptions`)
* `hashjoin` (`HashJoinNodeOptions`)
* optionally `order_by` (`OrderByNodeOptions`)

**New file:** `src/codeintel/build/tabular/acero_ops.py`

```python
from __future__ import annotations

import pyarrow as pa
import pyarrow.acero as acero
import pyarrow.compute as pc


def table_source(table: pa.Table) -> acero.Declaration:
    return acero.Declaration("table_source", acero.TableSourceNodeOptions(table))


def to_table(decl: acero.Declaration, *, use_threads: bool = True) -> pa.Table:
    return decl.to_table(use_threads=use_threads)  # Declaration.to_table 
```

(Keep this small—your real leverage is standardizing how you express plans.)

---

## Phase 3 — Build `graph.cpg_call_candidates` (one row per call)

You already construct per-candidate dict rows via `_call_target_record(...)` (fields include `callee_goid_h128`, `binding_kind`, `target_role`, etc.). In `cpg_call_targets`, you currently turn those into a large table and join to CFG blocks.

Instead, **stop early** and group candidates into a list per call.

### Minimal change strategy

Add a new node in `call_wiring.py`:

```python
CPG_CALL_CANDIDATES_TABLE_KEY = "graph.cpg_call_candidates"
```

**New node (sketch):** `cpg_call_candidates(...) -> InferableTabularInput`

* reuse existing resolution path to get `explicit_rows + implicit_rows`
* group by call PK
* store candidates as list-of-struct dicts

Representative grouping pattern:

```python
from collections import defaultdict

_CALL_KEY = ("repo", "commit", "rel_path", "call_id", "call_node_id")

_CANDIDATE_FIELDS = (
    "callee_symbol",
    "callee_def_id",
    "callee_def_node_id",
    "callee_goid_h128",
    "target_role",
    "binding_kind",
    "origin",
    "call_kind",
    "augop",
    "resolution_kind",
    "confidence",
    "candidate_count",
    "extras_json",
)

def _group_candidates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, object]] = {}
    for r in rows:
        key = tuple(r.get(k) for k in _CALL_KEY)
        g = grouped.get(key)
        if g is None:
            g = {k: r.get(k) for k in _CALL_KEY}
            # add call-level fields once
            g["callee_symbol"] = r.get("callee_symbol")
            g["call_kind"] = r.get("call_kind")
            g["augop"] = r.get("augop")
            g["candidates"] = []
            grouped[key] = g

        cand = {k: r.get(k) for k in _CANDIDATE_FIELDS if k not in _CALL_KEY}
        g["candidates"].append(cand)

    return list(grouped.values())
```

Then:

```python
candidate_rows = _group_candidates(all_rows)
table = table_for_rows(CPG_CALL_CANDIDATES_TABLE_KEY, candidate_rows)[0]
return _table_to_reader(CPG_CALL_CANDIDATES_TABLE_KEY, table)
```

**Important “best-in-class” note:** if you want maximum speed + fewer schema surprises, define an explicit schema for `graph.cpg_call_candidates` (so `candidates` is a true `list<struct<...>>` not “whatever inference produces”). This makes explode + join far more predictable.

---

## Phase 4 — Refactor `cpg_edges_calls` to: explode → Acero join/filter/project → finalize

### What it does

1. Convert candidates to a table and explode candidates list.
2. Build `entry_blocks` table from CFG blocks (you already have `_entry_blocks`).
3. Use Acero `hashjoin` to attach `callee_entry_block_id`.
4. Filter unresolved edges (`entry_block_id is valid`).
5. Project only the output columns + constant edge_kind.
6. Finalize.

### Representative `cpg_edges_calls` (sketch)

```python
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.acero as acero

from codeintel.build.tabular.explode_ops import explode_list_struct
from codeintel.build.tabular.acero_ops import table_source, to_table
from codeintel.build.tabular.dedupe_ops import dedupe_table_for_table
from codeintel.build.tabular.arrow_ops import align_table_to_contract
from codeintel.build.tabular.conversion import tabular_to_table
# reuse your existing _entry_blocks + _cast_table_column in call_wiring.py

def cpg_edges_calls(
    cpg_call_candidates: InferableTabularInput,
    q__graph__cfg_blocks: InferableTabularInput,
) -> InferableTabularInput:
    candidates = tabular_to_table(cpg_call_candidates)
    if candidates.num_rows == 0:
        return empty_reader(CPG_CALL_EDGES_TABLE_KEY)

    # 1) Explode candidates list<struct> into per-candidate rows
    exploded = explode_list_struct(
        candidates,
        list_col="candidates",
        parent_cols=["repo", "commit", "call_id", "call_node_id"],
        struct_fields={
            "callee_goid_h128": "callee_goid_h128",
            "confidence": "confidence",
            # include metadata columns instead of encoding extras_json if you can
            "binding_kind": "binding_kind",
            "target_role": "target_role",
            "call_kind": "call_kind",
            "origin": "origin",
            "augop": "augop",
            "extras_json": "extras_json",  # optional
        },
    )

    if exploded.num_rows == 0:
        return empty_reader(CPG_CALL_EDGES_TABLE_KEY)

    # 2) Entry block mapping table
    cfg_blocks = tabular_to_table(q__graph__cfg_blocks)
    entry = _entry_blocks(cfg_blocks)
    entry = _cast_table_column(entry, "function_goid_h128", _GOID_ARROW_TYPE)
    entry = _cast_table_column(entry, "entry_block_id", _BLOCK_ID_ARROW_TYPE)

    # 3) Acero plan: join + filter + project
    left = acero.Declaration("table_source", acero.TableSourceNodeOptions(exploded))
    right = acero.Declaration("table_source", acero.TableSourceNodeOptions(entry))

    joined = acero.Declaration(
        "hashjoin",
        acero.HashJoinNodeOptions(
            join_type="left outer",
            left_keys=["callee_goid_h128"],
            right_keys=["function_goid_h128"],
            left_output=[
                "repo", "commit", "call_id", "call_node_id",
                "confidence", "binding_kind", "target_role", "call_kind", "origin", "augop",
                "extras_json",
            ],
            right_output=["entry_block_id"],
        ),
        inputs=[left, right],
    )

    # keep only resolved targets
    resolved = acero.Declaration(
        "filter",
        acero.FilterNodeOptions(pc.is_valid(pc.field("entry_block_id"))),
        inputs=[joined],
    )

    projected = acero.Declaration(
        "project",
        acero.ProjectNodeOptions(
            expressions=[
                pc.field("repo"),
                pc.field("commit"),
                pc.field("call_id"),
                pc.field("call_node_id"),
                pc.field("entry_block_id"),
                pc.scalar("CALLS"),
                pc.field("confidence"),
                pc.field("binding_kind"),
                pc.field("target_role"),
                pc.field("call_kind"),
                pc.field("origin"),
                pc.field("augop"),
                pc.field("extras_json"),
            ],
            names=[
                "repo",
                "commit",
                "call_id",
                "call_node_id",
                "callee_entry_block_id",
                "edge_kind",
                "confidence",
                "binding_kind",
                "target_role",
                "call_kind",
                "origin",
                "augop",
                "extras_json",
            ],
        ),
        inputs=[resolved],
    )

    edges = projected.to_table(use_threads=True)

    # 4) Finalize gate (align + dedupe)
    edges = align_table_to_contract(
        CPG_CALL_EDGES_TABLE_KEY,
        edges,
        target_name=CALL_WIRING_TARGET_NAME,
        reporter=emit_alignment_report,
        extras_policy=None,
    )
    edges = dedupe_table_for_table(CPG_CALL_EDGES_TABLE_KEY, edges)
    return edges
```

This is the core pattern. The expensive bits are now C++ kernels + Acero exec nodes.

**Why this works well**

* Explode is pure Arrow compute (fast) using standard kernels.
* Join/filter/project is Acero streaming exec plan (fast).
* No Python per-row loops.
* You can add an `order_by` node for deterministic ordering if needed (at the cost of a full materializing sort).

---

## Phase 5 — Turn “finalize gate” into a shared helper (so every table does it the same way)

You’re already repeating:

* empty handling
* align-to-contract + reporter
* dedupe-by-PK
* sometimes “preference sort” then dedupe

Make it one helper that every output table calls.

**New file:** `src/codeintel/build/tabular/finalize_ops.py`

```python
from __future__ import annotations

from collections.abc import Sequence
import pyarrow as pa

from codeintel.build.tabular.arrow_ops import align_table_to_contract
from codeintel.build.tabular.dedupe_ops import dedupe_table_for_table
from codeintel.build.tabular.compute_columns import empty_table_for_table


def finalize_table(
    table_key: str,
    table: pa.Table,
    *,
    target_name: str,
    prefer_columns: Sequence[str] | None = None,
    reporter=None,
) -> pa.Table:
    if table is None or table.num_rows == 0:
        return empty_table_for_table(table_key)

    # Optional: table = table.combine_chunks() if you want to trade memory for speed
    table = align_table_to_contract(
        table_key,
        table,
        target_name=target_name,
        reporter=reporter,
        extras_policy=None,
    )
    table = dedupe_table_for_table(table_key, table, prefer_columns=prefer_columns)
    return table
```

Then your node ends become:

```python
return finalize_table(
    CPG_CALL_EDGES_TABLE_KEY,
    edges,
    target_name=CALL_WIRING_TARGET_NAME,
    reporter=emit_alignment_report,
)
```

---

## Phase 6 — Wire it into the Hamilton target (minimal, concrete edits)

### A) Add the new table key + node to the call_wiring target

In `src/codeintel/build/hamilton/native/graphs/graph_targets.py`:

* add a `TableTargetTableContext` for `graph.cpg_call_candidates` if you want it materialized
* or keep it “internal only” and don’t save it (your call)

### B) Update call_wiring module exports

* Add `CPG_CALL_CANDIDATES_TABLE_KEY`
* Add `cpg_call_candidates(...)`
* Update `cpg_edges_calls(...)` signature to depend on `cpg_call_candidates` (and `cfg_blocks`)

---

## Optional “fault tolerant” extension (high value)

Instead of silently dropping unresolved joins, produce a diagnostics side table:

* `graph.cpg_edges_calls_unresolved` with columns:

  * call keys + candidate metadata + reason code (`missing_entry_block`)
* In strict mode: raise if unresolved ratio > threshold
* In tolerant mode: persist unresolved table + continue

This makes failures observable without killing throughput.

---

## The shortest path to land this (2 PRs)

**PR 1**

* Add `explode_list_struct`
* Add minimal `acero_ops`
* Add `finalize_table`
* Refactor `cpg_edges_calls` to use `cpg_call_candidates` (even if candidates is built in the same module first)

**PR 2**

* Add real `cpg_call_candidates` node (one row per call, list<struct> candidates)
* Update `graph_targets.py` to include it (optional)
* Switch `cpg_edges_calls` to use it
* Update `cpg2/planes/call_wiring.py` if you changed extras semantics

---

If you want, I can also sketch the **exact** `cpg_call_candidates` schema (as a `TableSchema` override / Arrow schema) so the `candidates` field is *guaranteed* to be `list<struct<...>>` (this is the single biggest thing that keeps explode + Acero join predictable and bug-resistant).


Below is a concrete, *repo-native* sketch for `graph.cpg_call_candidates` that **guarantees** the `candidates` column is a **`LIST(STRUCT(...))`** at the contract layer (TableSchema), and therefore at the Arrow contract layer via your existing `arrow_gen` machinery.

I’m giving you **both**:

1. the **TableSchema override** you’d add to `src/codeintel/core/schemas/output_registry.py`, and
2. the corresponding **explicit `pa.Schema`** you can use as a mental model + for tests (and as an optional “hard override” if you ever want to bypass SQL type parsing).

---

## 1) TableSchema override (canonical contract)

### Add these type constants near the other `*_STRUCT` constants in `output_registry.py`

```python
# src/codeintel/core/schemas/output_registry.py

CPG_CALL_CANDIDATE_STRUCT = (
    "STRUCT("
    "callee_def_id VARCHAR, "
    "callee_def_node_id VARCHAR, "
    "callee_goid_h128 DECIMAL(38,0), "
    "target_role VARCHAR, "
    "binding_kind VARCHAR, "
    "origin VARCHAR, "
    "resolution_kind VARCHAR, "
    "confidence DOUBLE"
    ")"
)

CPG_CALL_CANDIDATES_LIST = f"LIST({CPG_CALL_CANDIDATE_STRUCT})"
```

### Add the table to `CALL_WIRING_OVERRIDE_TABLES`

Put it **before** `cpg_call_targets` (or after—either is fine; this is an internal intermediate).

```python
# src/codeintel/core/schemas/output_registry.py

CALL_WIRING_OVERRIDE_TABLES: tuple[TableSchema, ...] = (
    TableSchema(
        schema="graph",
        name="cpg_call_candidates",
        columns=[
            *REPO_COMMIT_COLS,
            Column("rel_path", "VARCHAR", nullable=False),
            Column("call_id", "VARCHAR", nullable=False),
            Column("call_node_id", "VARCHAR"),
            Column("callee_symbol", "VARCHAR"),
            Column("call_kind", "VARCHAR"),
            Column("augop", "VARCHAR"),
            Column("candidate_count", "INTEGER", nullable=False),
            # This is *call-level* extras (same “extras_json” you already carry on call targets).
            Column("extras_json", "BLOB"),
            # The key part: guaranteed LIST(STRUCT(...)) and non-null (store [] not NULL).
            Column("candidates", CPG_CALL_CANDIDATES_LIST, nullable=False),
        ],
        primary_key=("repo", "commit", "rel_path", "call_id"),
        indexes=(
            Index("idx_graph_cpg_call_candidates_repo_commit", ("repo", "commit")),
            Index("idx_graph_cpg_call_candidates_call_id", ("call_id",)),
            Index("idx_graph_cpg_call_candidates_rel_path", ("rel_path",)),
        ),
        description=(
            "Per-callsite candidate callee resolutions. "
            "candidates is LIST(STRUCT(...)) to support Arrow-native explode + join."
        ),
    ),

    # ...existing:
    TableSchema(... name="cpg_call_targets", ...),
    TableSchema(... name="cpg_edges_calls", ...),
    ...
)
```

### Why this alone “locks in” the type

Your existing `codeintel.core.schemas.arrow_gen` path parses DuckDB complex types (`LIST`, `STRUCT`, etc.) via SQLGlot and emits the matching Arrow types. So once this TableSchema exists, any of:

* `table_for_rows("graph.cpg_call_candidates", rows)`
* `align_table_to_contract("graph.cpg_call_candidates", table, ...)`

will force `candidates` to be a real `list<struct<...>>` in Arrow, rather than “inferred Python list-of-dicts”.

---

## 2) The equivalent Arrow schema (for clarity + tests)

This is what your contract will effectively become (modulo metadata fields):

```python
import pyarrow as pa

GOID = pa.decimal128(38, 0)

CANDIDATE_STRUCT = pa.struct([
    pa.field("callee_def_id", pa.string()),
    pa.field("callee_def_node_id", pa.string()),
    pa.field("callee_goid_h128", GOID),
    pa.field("target_role", pa.string()),
    pa.field("binding_kind", pa.string()),
    pa.field("origin", pa.string()),
    pa.field("resolution_kind", pa.string()),
    pa.field("confidence", pa.float64()),
])

CPG_CALL_CANDIDATES_ARROW_SCHEMA = pa.schema([
    pa.field("repo", pa.string(), nullable=False),
    pa.field("commit", pa.string(), nullable=False),
    pa.field("rel_path", pa.string(), nullable=False),
    pa.field("call_id", pa.string(), nullable=False),
    pa.field("call_node_id", pa.string()),
    pa.field("callee_symbol", pa.string()),
    pa.field("call_kind", pa.string()),
    pa.field("augop", pa.string()),
    pa.field("candidate_count", pa.int32(), nullable=False),
    pa.field("extras_json", pa.binary()),
    pa.field("candidates", pa.list_(CANDIDATE_STRUCT), nullable=False),
])
```

---

## 3) Two small “gotchas” to bake into the contract rules

### A) Make `candidates` **never NULL**

Set `nullable=False` (as above) and in producers always emit:

* `[]` when no candidates
* never `None`

This avoids `is_null(candidates)` branches and makes explode logic simpler.

### B) Keep `extras_json` at **call-level**, not per candidate

Right now `_call_edge_extras()` treats `extras_json` as call extras and adds call metadata + decoded call extras into per-edge payload. Keeping it call-level avoids duplicating the same blob N times inside the list.

---

## 4) A minimal test you can add to enforce the guarantee

```python
def test_cpg_call_candidates_schema_has_list_struct(schema_service):
    schema = schema_service.require_table_schema("graph.cpg_call_candidates")
    arrow = schema_service.get_arrow_schema("graph.cpg_call_candidates")
    assert arrow is not None

    field = arrow.field("candidates")
    assert pa.types.is_list(field.type)
    assert pa.types.is_struct(field.type.value_type)

    struct_type = field.type.value_type
    assert struct_type.get_field_index("callee_goid_h128") >= 0
    assert struct_type.field("callee_goid_h128").type == pa.decimal128(38, 0)
```

(Where `schema_service` is whatever fixture you already use to configure `SchemaService` in tests.)

---

## Optional “even more best-in-class” variant (if you’re willing to break goid type)

If you want maximum join throughput, consider switching `callee_goid_h128` (and all goids) from `DECIMAL(38,0)` to a **16-byte binary key**:

* TableSchema uses `BLOB`
* Arrow uses `pa.binary(16)` or `pa.fixed_size_binary(16)` (fixed-size is ideal)

That typically reduces casting + speeds hashing/joining. But it’s a bigger project-wide change, so I’d only do it once you’ve landed the call-candidates pipeline and confirmed it’s the main hot path.

---

If you want the next increment: I can show the **producer-side shape** for `graph.cpg_call_candidates` so it naturally emits a *typed* `list<struct>` (without relying on “Python dict inference”), using your existing `table_for_rows()`/`ColumnarBatchCollector` patterns.
