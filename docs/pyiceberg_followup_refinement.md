I pulled the repo apart and focused specifically on the end‑to‑end “Hamilton → Arrow/Polars → PyIceberg → DuckDB serving” path. The big picture is: you *can* keep the same functional intent, but your current implementation is doing too many things in too many places (and some of those things are “nice-to-have” side effects that can currently fail the whole run). The result is exactly what you’re feeling: brittle execution and tests that are painful to maintain.

Below is a rigorous, “best‑in‑class” redesign that keeps your constraints (Hamilton DAG, Arrow/Polars compute, PyIceberg as initial sink, DuckDB as serving engine), but makes the solution far more unified and robust.

---

## What you have today (as implemented)

### Write path (build / Hamilton)

* `IcebergDatasetSaver` (`codeintel/build/hamilton/materializers/iceberg_saver.py`) does **everything**:

  * derives a `TableSchema` + Arrow contract schema
  * aligns Arrow batches to the contract
  * validates
  * ensures/updates the Iceberg table
  * writes data files via a custom `_IcebergWriter`
  * updates snapshot refs
  * refreshes an Iceberg metadata cache inside DuckDB
  * optionally appends tombstones
  * computes & persists Iceberg statistics

### Read/serve path

* DuckDB serving does **not** read Iceberg directly. Instead it:

  * uses PyIceberg `table.scan()` → `DataScan.to_arrow_batch_reader()`
  * then loads that reader into DuckDB via `con.from_arrow(reader)` (`duckdb_relation_builder._scan_iceberg`)
* Polars serving reads via `DataScan.to_polars()` and then converts to lazy.

This creates multiple “bridges” (Iceberg→Arrow→DuckDB, Iceberg→Polars eager→lazy), plus a metadata cache that’s updated in more than one place.

---

## Why it feels fragile (the concrete culprits)

### 1) Your materializer mixes “commit” with “side effects”

In `_write_to_iceberg()` you do:

* commit write
* then update refs
* then refresh metadata cache **without protective error handling**
* then tombstones
* then statistics & persist

Any failure after the Iceberg commit can still fail the Hamilton run, leaving you with:

* data written successfully to Iceberg
* but a failed build run and/or partial metadata side effects

That is a classic robustness killer and also a classic test killer.

### 2) Serving reads Iceberg through an Arrow stream instead of letting DuckDB scan Iceberg

Right now DuckDB is acting as a *compute engine on top of Arrow streams*, not as a lakehouse engine reading Iceberg.
That tends to be:

* slower / more memory sensitive
* harder to reason about pushdown behavior
* harder to test deterministically (because you depend on PyIceberg scan semantics + Arrow reader behavior + DuckDB ingestion)

### 3) There are two scanning stacks (PyIceberg scan → DuckDB, and PyIceberg scan → Polars)

So you have to maintain and test:

* projection/filter/order semantics in both
* schema alignment in both (or accept divergence)
* tombstone handling in both

### 4) Your “Iceberg metadata cache” is tightly coupled into the build path

Refreshing it in the write transaction path makes tests hard:

* tests must create DuckDB schemas/meta tables correctly
* tests become integration-heavy even when you want unit coverage

---

## The single biggest architectural improvement

### Let DuckDB read Iceberg directly via DuckDB’s `iceberg` extension

DuckDB’s Iceberg extension supports scanning Iceberg tables directly using `iceberg_scan()` and supports snapshot selection via `snapshot_from_id` / `snapshot_from_timestamp`. ([DuckDB][1])

That means your “serving” engine can be:

**DuckDB → iceberg_scan(path, snapshot_from_id=…) → normal SQL execution**

instead of:

**PyIceberg → Arrow reader → DuckDB from_arrow(reader) → normal SQL execution**

DuckDB also exposes `iceberg_metadata()` and `iceberg_snapshots()` for metadata introspection. ([DuckDB][1])

And if you decide to move to a real catalog later, DuckDB supports attaching Iceberg REST catalogs. ([DuckDB][2])

This aligns extremely well with your constraint “PyIceberg is initial recipient before DuckDB”:

* PyIceberg remains the writer / committer
* DuckDB becomes the query engine that reads Iceberg natively at serving time

---

## Proposed best‑in‑class architecture

### Canonical boundary: Arrow streaming

Make **Arrow `RecordBatchReader`** the canonical interchange at the “storage boundary”:

* Hamilton nodes can return Polars/Arrow/etc
* materializers must normalize to Arrow reader
* storage writers accept Arrow reader

You’re close to this already (`to_record_batch_reader`), but the *rest of the system* should treat the reader as the stable contract boundary.

### Split responsibilities into 4 components

#### A) Contract & schema layer (pure, testable)

* Determine `TableSchema` & `ArrowSchemaMetadata`
* Build contract Arrow schema
* Provide schema diff/evolution plan
* **No IO** here

#### B) Iceberg table management (small, focused)

* ensure namespace/table
* update properties/schema/partition spec

#### C) Iceberg write/commit (single responsibility)

* write files
* commit snapshot (append/overwrite)
* return snapshot id + commit metadata

#### D) Post-commit hooks (best effort)

* update snapshot refs (if you keep refs)
* refresh DuckDB cache (if you keep cache)
* append tombstones (if enabled)
* compute/persist stats

**Key change:** hooks must not be allowed to fail the build by default.

---

## Concrete refactor plan (high impact, minimal disruption)

### Step 1 — Replace DuckDB’s Iceberg read path with `iceberg_scan`

Modify `codeintel/serving/semantic/duckdb_relation_builder.py`:

* Today `_scan_iceberg()`:

  * loads Iceberg table via PyIceberg
  * creates `DataScan`
  * converts to Arrow
  * `con.from_arrow(reader)`

* Replace with:

  * resolve snapshot id (you already do this)
  * resolve Iceberg table root path or metadata json path
  * `con.sql("SELECT * FROM iceberg_scan('…', snapshot_from_id=…, allow_moved_paths=true)")`

DuckDB documents `iceberg_scan()` (including snapshot parameters and `allow_moved_paths`). ([DuckDB][1])

**Practical note:** you will need to ensure DuckDB loads the `iceberg` extension for serving connections (`INSTALL iceberg; LOAD iceberg;`). DuckDB describes install/load behavior. ([DuckDB][1])

This change alone:

* deletes an entire failure class (Arrow stream ingestion issues)
* simplifies pushdown reasoning
* makes join queries more natural (DuckDB can optimize across scans)

### Step 2 — Make `_write_to_iceberg` “commit-only” + move the rest to hooks

In `iceberg_saver._write_to_iceberg()`:

Today, after commit you do:

* `_update_snapshot_refs`
* `refresh_iceberg_metadata_cache`
* tombstones
* stats + persist

Refactor to:

1. Commit data + return `(snapshot_id, maybe_minimal_stats)`
2. Run hooks inside separate `try/except` blocks, and record warnings (or structured log) instead of failing the run.

This is one of the most important robustness fixes.

### Step 3 — Make metadata cache refresh lazy (serving-time), not build-time

You already have a serving-time refresh path in `serving/db/manager.py::_refresh_iceberg_cache`.

Lean into that:

* remove build-time cache refresh (or make it optional)
* tests that don’t care about cache shouldn’t have to set it up

### Step 4 — Unify scan semantics across DuckDB and Polars (optional but recommended)

Right now Polars reads from PyIceberg scans; DuckDB will read from `iceberg_scan`.
That’s *already* much better than today, but you’ll still have two scan stacks.

A clean unification is:

* for Polars queries: use DuckDB query → Arrow → Polars when needed
* or use a shared “IcebergScanPlan” that can produce:

  * DuckDB scan relation via `iceberg_scan`
  * Polars LazyFrame via `scan_parquet(file_list)` (if you want pure Polars)

Given your stated goal (“singular, integrated, streamlined”), I’d strongly prefer:

* **DuckDB as the only serving execution engine**, with Polars used for build/compute, not serving.
* If you need Polars execution for some workloads, make it a controlled fallback with very explicit surface area.

---

## Specific code-level issues worth fixing even if you change nothing else

### 1) “Side effects can fail the whole run” (high severity)

Wrap these calls in `try/except` with logging + structured event recording:

* `refresh_iceberg_metadata_cache(...)`
* `_maybe_append_tombstones(...)`
* `iceberg_stats_for_table(...)`
* `persist_iceberg_statistics(...)`

Right now a transient DuckDB metadata issue can fail a successful Iceberg commit.

### 2) `IcebergStreamAdapter.to_arrow_batch_reader` ignores `batch_size`

In `core/iceberg/stream.py`, `to_arrow_batch_reader(self, *, batch_size)` doesn’t use `batch_size`.
If PyIceberg doesn’t support batch sizing there, that’s fine, but then:

* either remove the parameter
* or document it as advisory/no-op
* or enforce batch size downstream by re-batching the reader

This sort of mismatch causes confusing perf + test behavior.

### 3) Your DuckDB catalog schema name `"iceberg"` may collide conceptually with DuckDB’s Iceberg extension

Not necessarily a runtime collision, but it’s a human/debugging trap once you load the `iceberg` extension.
Consider renaming your internal schema from `iceberg` → `iceberg_catalog` or `iceberg_meta`.

---

## How to implement DuckDB `iceberg_scan` cleanly in your code

Here’s the shape I’d recommend (conceptually), implemented as a tiny service module:

### `codeintel/storage/iceberg/duckdb_scan.py` (new)

Responsibilities:

* ensure DuckDB iceberg extension is loaded (or raise a clear error)
* build a relation for a given `table_key` and `snapshot_id`

Pseudo-implementation sketch:

```python
def iceberg_relation(
    con: duckdb.DuckDBPyConnection,
    *,
    table_root_or_metadata: str,
    snapshot_id: int | None,
) -> duckdb.DuckDBPyRelation:
    con.execute("LOAD iceberg")  # or require_extension wrapper
    path = table_root_or_metadata.replace("'", "''")
    sql = f"SELECT * FROM iceberg_scan('{path}', allow_moved_paths=true"
    if snapshot_id is not None:
        sql += f", snapshot_from_id={int(snapshot_id)}"
    sql += ")"
    return con.sql(sql)
```

The `iceberg_scan` function and snapshot parameters are documented by DuckDB. ([DuckDB][1])

If later you move to a REST catalog, DuckDB supports attaching REST catalogs and querying tables through the attached catalog (and supports metadata functions). ([DuckDB][2])

---

## Testing strategy that will stop hurting

### 1) Stop requiring “build-time metadata cache correctness” for most tests

If you implement “commit-only + hooks”, then:

* unit tests can validate schema/contract logic without any DuckDB meta setup
* integration tests can explicitly enable hooks and validate cache/stats/tombstones

### 2) Add a focused “Iceberg writer contract test suite”

These tests should only validate:

* append produces readable rows
* overwrite(table) replaces fully
* overwrite(snapshot) replaces partitions correctly
* schema evolution: add column, read old + new snapshots

Everything else (tracking tables, UI stats, etc.) should be separate tests.

### 3) Add a focused “Serving scan test suite”

Once you switch to DuckDB’s `iceberg_scan`, serving tests become:

* “Given Iceberg table path + snapshot id, does DuckDB return expected result?”
  DuckDB documents snapshot selection and scanning behavior. ([DuckDB][1])

---

## Optional but very strong “future-proofing” move: REST Catalog alignment

DuckDB’s docs explicitly show a workflow where PyIceberg manages schema and DuckDB reads/writes via an attached Iceberg catalog. ([DuckDB][3])

Even if you don’t adopt REST immediately, designing your internal interfaces as if “catalog attachment exists” will pay off:

* your serving engine won’t care whether the table is reached by `iceberg_scan(path)` or `attached_catalog.schema.table`
* your tests can run in “path mode” while prod can move to “catalog mode”

---

## Summary of what I would do first (highest ROI order)

1. **Switch DuckDB serving from Arrow ingestion to DuckDB `iceberg_scan()`**. This is the biggest simplifier and robustness win. ([DuckDB][1])
2. **Make Iceberg materialization commit-only; move cache/tombstones/stats into post-commit hooks with error isolation.**
3. **Move metadata cache refresh out of the build path** (or make it optional) and keep it as a serving-time refresh.
4. **Consolidate scanning semantics** (ideally DuckDB-only for serving).
5. Clean up smaller correctness/perf mismatches (batch sizing no-ops, schema naming collisions, etc.).

---

If you want, I can also propose a concrete “diff plan” across the exact files in your repo (new module layout + which functions move where + which existing functions get deleted) so an engineer can implement it quickly. The key is: you don’t need to abandon Hamilton, Arrow/Polars, or PyIceberg—you mainly need to (a) stop doing post-commit side effects inline, and (b) stop streaming Iceberg into DuckDB through Arrow when DuckDB can scan Iceberg directly.

[1]: https://duckdb.org/docs/stable/core_extensions/iceberg/overview.html "Iceberg Extension – DuckDB"
[2]: https://duckdb.org/docs/stable/core_extensions/iceberg/iceberg_rest_catalogs.html "Iceberg REST Catalogs – DuckDB"
[3]: https://duckdb.org/docs/stable/lakehouse_formats.html "Lakehouse Formats – DuckDB"


Below is a concrete, repo-specific **diff plan** (PR-sized chunks) that an engineer can execute quickly. I’ve focused on (1) immediately fixing the robustness/test pain you’re feeling (notably the **layering violations**), and (2) setting you up for a **more unified Iceberg-first + DuckDB-serving + Arrow/Polars** architecture without doing a risky “big bang” rewrite.

---

## PR 1 — Fix the two biggest architecture breaks (build → serving imports)

### Goal

Make `tests/test_layering_serving_imports.py` pass again and remove the “middle layer imports serving” coupling that makes everything brittle.

### 1) Move `ServingSnapshotPointer` to a shared, non-serving module

**Problem:** `src/codeintel/build/serving/publisher.py` imports `codeintel.serving.db.pointer.ServingSnapshotPointer`, which violates the layering rule (build should not import serving).

#### Diff

* **ADD**: `src/codeintel/core/serving_snapshot_pointer.py`

  * Move the entire `ServingSnapshotPointer` dataclass (and `load()`/`to_json()` helpers) from:

    * `src/codeintel/serving/db/pointer.py`
  * Keep the class name identical to avoid churn.

* **EDIT**: `src/codeintel/serving/db/pointer.py`

  * Replace contents with a thin re-export:

    * `from codeintel.core.serving_snapshot_pointer import ServingSnapshotPointer`
    * `__all__ = ["ServingSnapshotPointer"]`
  * This keeps *all existing serving imports working*.

* **EDIT**: `src/codeintel/build/serving/publisher.py`

  * Change import:

    * **from** `codeintel.serving.db.pointer import ServingSnapshotPointer`
    * **to** `codeintel.core.serving_snapshot_pointer import ServingSnapshotPointer`

✅ Result: build no longer imports serving, without breaking serving/tests that still import `codeintel.serving.db.pointer`.

---

### 2) Remove build’s dependency on `serving.semantic.iceberg_scans`

**Problem:** `src/codeintel/build/hamilton/native/patterns/loaders.py` imports from `codeintel.serving.semantic.iceberg_scans`, which is exactly what the layering test forbids.

#### Diff

* **ADD**: `src/codeintel/core/iceberg/ref_scans.py`

  * Move the *generic* Iceberg “ref scan” primitives out of serving and into core:

    * `class IcebergScanError`
    * `@dataclass IcebergScanResult`
    * `@dataclass IcebergRefScanRequest`
    * `iceberg_table_exists()`
    * `resolve_iceberg_ref_for_identity()`
    * `resolve_iceberg_ref()`  *(this is generic; it only uses run_id/commit/settings)*
    * `iceberg_scan_for_ref()`
    * `resolve_iceberg_snapshot_id()` *(optional but recommended—DuckDB engine uses it and it’s not “serving-specific”)*
    * private helpers: `_resolve_snapshot_id()`, `_selected_fields()`

  These currently live in:

  * `src/codeintel/serving/semantic/iceberg_scans.py`

* **EDIT**: `src/codeintel/serving/semantic/iceberg_scans.py`

  * Keep **serving-specific filter translation** here (because it depends on serving `FilterSpec`, allowed ops, etc.).
  * Remove the generic ref-scan code listed above and instead:

    * import from `codeintel.core.iceberg.ref_scans`
    * re-export in `__all__` so existing serving code/tests don’t break.

  Concretely:

  * Serving keeps:

    * `IcebergFilterResult`
    * `IcebergScanRequest`
    * `required_scan_fields()`
    * `iceberg_row_filter_from_filters()`
    * `iceberg_scan_for_query()`
    * `_expression_for_filter()` and friends (the op/type-specific filter compilation)

* **EDIT**: `src/codeintel/build/hamilton/native/patterns/loaders.py`

  * Change imports:

    * **from** `codeintel.serving.semantic.iceberg_scans import IcebergRefScanRequest, iceberg_scan_for_ref, iceberg_table_exists, resolve_iceberg_ref_for_identity`
    * **to** `codeintel.core.iceberg.ref_scans import IcebergRefScanRequest, iceberg_scan_for_ref, iceberg_table_exists, resolve_iceberg_ref_for_identity`

✅ Result: build no longer imports serving; serving keeps the public API stable via re-exports.

---

### “Done” checklist for PR 1

* [ ] `build/*` contains **zero** `codeintel.serving.*` imports
* [ ] `tests/test_layering_serving_imports.py` passes
* [ ] Serving code compiles unchanged because the original modules re-export the moved symbols

---

## PR 2 — Make Iceberg reads/writes a single integrated subsystem (stop scattering logic)

### Goal

Right now Iceberg logic is split across:

* build materializer (write path),
* serving semantic scans (read path),
* and “core/storage Iceberg helpers” (metadata/catalog utilities)

This PR introduces a single “Iceberg I/O” surface that both build and serving call, so you don’t keep rewriting/duplicating edge-case handling.

### Diff

* **ADD**: `src/codeintel/storage/iceberg/io.py` (new)

  * New “service” object that centralizes:

    * `exists(table_key)`
    * `load_table(table_key)`
    * `scan_by_ref(table_key, ref, *, row_filter, selected_fields, batch_size) -> IcebergScanResult`
    * `scan_for_query(...) -> IcebergScanResult` *(optional: can remain in serving until PR 3 if you want minimal churn)*

  Internally this just composes:

  * `IcebergCatalogProvider`
  * `IcebergSettings`
  * `io_options`
  * and the scan-plan dataclasses you already have (`IcebergScanPlan`)

* **EDIT**: `src/codeintel/core/iceberg/ref_scans.py`

  * Make it thin wrappers around `storage.iceberg.io.IcebergIO` (or whichever name you choose).
  * Keep `core.*` as “shared API / stable imports”; actual I/O lives in `storage.*`.

* **EDIT**: `src/codeintel/serving/semantic/duckdb_relation_builder.py`

  * Replace direct calls to `iceberg_scan_for_query()` with calls into the Iceberg I/O service
  * This consolidates error handling and plan/metrics capture.

* **EDIT**: `src/codeintel/serving/semantic/engines/polars_engine.py`

  * Same: source Iceberg scans via the centralized service.

* **EDIT**: `src/codeintel/build/hamilton/native/patterns/loaders.py`

  * Use the same centralized Iceberg I/O rather than calling scan helpers directly.

✅ Result: There is now **one** place to fix Iceberg read robustness (ref resolution, io_options, snapshot selection, scan plan capture), and both build + serving benefit.

---

## PR 3 — Break up `IcebergDatasetSaver` into coherent modules (massive maintainability win)

### Goal

`src/codeintel/build/hamilton/materializers/iceberg_saver.py` is doing too many jobs:

* schema inference + contract alignment
* table create/evolve
* file sizing + write task batching
* partition derivation
* delete handling
* tombstones
* snapshot refs
* stats + observation persistence
* cache refresh

That makes tests hard to target and makes robustness improvements risky.

### New module layout (concrete)

**ADD directory:** `src/codeintel/build/hamilton/materializers/iceberg/`

**MOVE (extract) code from** `src/codeintel/build/hamilton/materializers/iceberg_saver.py` into:

1. `src/codeintel/build/hamilton/materializers/iceberg/context.py`

   * Move:

     * `_MaterializeContext`
     * `_IcebergPlan`
     * `_ValidationSetup`
     * `_ValidationOutcome`

2. `src/codeintel/build/hamilton/materializers/iceberg/planning.py`

   * Move:

     * `_build_plan`
     * `_load_inferred_settings`
     * `_build_write_settings`
     * `_extras_policy_for_table`
     * `_name_mapping_digest`

3. `src/codeintel/build/hamilton/materializers/iceberg/table_ops.py`

   * Move:

     * `_ensure_namespace`
     * `_ensure_table`
     * `_table_properties`
     * `_apply_table_updates`
     * `_apply_schema_update`
     * `_apply_table_properties`
     * `_apply_partition_update`
     * `_snapshot_properties`

4. `src/codeintel/build/hamilton/materializers/iceberg/writer.py`

   * Move:

     * `_IcebergWriter`
     * `_IcebergWriteContext`
     * `_build_write_context`
     * `_append_reader_batches`
     * `_flush_batches`
     * `_write_data_files`
     * `_WriteTaskContext`
     * `_write_task_context`
     * `_write_tasks_for_table`
     * `_target_file_size`
     * `_task_counter`

5. `src/codeintel/build/hamilton/materializers/iceberg/partitioning.py`

   * Move:

     * `_TablePartition`
     * `_determine_partitions`
     * `_partition_predicate`
     * `_get_field_from_arrow_table`
     * `_compute_field`, `_resolve_compute_fn`, `_compute_is_null`, `_compute_equal`, `_compute_and`
     * `_compute_struct_field`

6. `src/codeintel/build/hamilton/materializers/iceberg/delete_ops.py`

   * Move:

     * `_append_producer`
     * `_delete_filter`
     * `_delete_matching_files`

7. `src/codeintel/build/hamilton/materializers/iceberg/tombstones.py`

   * Move:

     * `_maybe_append_tombstones`
     * `_ensure_tombstone_table`
     * `_tombstone_table_key`
     * `_tombstone_table_schema`
     * `_merge_tombstone_stats`
     * `_tombstone_diff_reader`
     * `_append_tombstones`

8. `src/codeintel/build/hamilton/materializers/iceberg/validation.py`

   * Move:

     * `_prepare_validation`
     * `_build_observed_reader`
     * `_finalize_validation`
     * `_record_batch_reader_for_data`

9. `src/codeintel/build/hamilton/materializers/iceberg/metadata.py`

   * Move:

     * `_minimal_iceberg_stats`
     * `_update_snapshot_refs`

10. `src/codeintel/build/hamilton/materializers/iceberg/observations.py`

* Move:

  * `_persist_observation_if_ready`

11. `src/codeintel/build/hamilton/materializers/iceberg/utils.py`

* Move:

  * `_coerce_int`
  * `_coerce_bool`
  * `_coerce_tuple`

### What stays in `iceberg_saver.py`

* Keep the **public** `IcebergDatasetSaver(DataSaver)` class.
* Keep a single orchestrator function (or two):

  * `_materialize_iceberg()` which becomes mostly “call into modules in the right order”.

### Deletes

* You should be able to **delete** most private helpers from `iceberg_saver.py` after extraction (they live in the new modules).
* Optionally keep a few private wrappers for backward compatibility if tests import internals (unlikely, but check).

✅ Result: you can write tests against `writer.py` (file sizing and batch flush), `partitioning.py`, `tombstones.py`, etc. without booting the whole Hamilton materializer.

---

## PR 4 — Make the system “Iceberg-first” in one place, not five

### Goal

Right now Iceberg enablement/enforcement appears in multiple places (guardrails, serving engines, build loader). This leads to “it works in serving but not in build” drift.

### Diff

* **ADD**: `src/codeintel/storage/table_routing.py` (new)

  * A single function:

    * `resolve_table_backend(table_key, iceberg_settings) -> Literal["iceberg", "duckdb"]`
  * It should be the *only* place that interprets:

    * `read_enabled`
    * `read_fallback_enabled`
    * `enforced_table_prefixes`
    * table existence checks

* **EDIT**:

  * `src/codeintel/build/hamilton/native/patterns/loaders.py`
  * `src/codeintel/serving/semantic/duckdb_relation_builder.py`
  * `src/codeintel/serving/semantic/engines/polars_engine.py`
  * Replace their local “if iceberg enabled and table exists…” logic with the centralized router.

✅ Result: “where does this table come from?” becomes deterministic and testable.

---

## Follow-up actions checklist (high leverage)

### A) Add a single end-to-end “golden path” test

Create one integration test that:

1. builds a tiny Arrow/Polars dataset,
2. writes to Iceberg via `IcebergDatasetSaver`,
3. reads it back via the unified Iceberg I/O scan,
4. executes a simple serving query via DuckDB engine.

This becomes your “if this breaks, the system is broken” test.

Suggested location:

* **ADD**: `tests/integration/test_iceberg_to_duckdb_serving_roundtrip.py`

### B) Tighten the tabular boundary (optional, but big robustness win)

Right now `TabularInput` in `core/columnar/tabular_adapter.py` includes `| object`, which makes type safety and debugging much harder.

A robust move is:

* introduce `StrictTabularInput` (no `object`)
* keep `TabularInput` for backward compatibility
* gradually migrate call sites to strict types so invalid inputs fail fast.

### C) Enforce “no build artifacts in repo”

Your zip contains lots of `__pycache__/` and `.pyc` files. Make sure they’re not actually being committed.

* add/update `.gitignore`
* add a CI check that fails if `**/__pycache__/**` or `**/*.pyc` appears in git status

### D) Document the architecture in one page

Add:

* **ADD**: `docs/architecture/lakehouse.md`

Include:

* “Iceberg is system-of-record”
* “DuckDB is query engine (and metadata cache), not canonical storage”
* “Arrow RecordBatchReader is the boundary format”
* “Hamilton materializers own all persistence”

### E) Operational runbook

Add a short runbook for:

* snapshot refs creation/cleanup
* expiring old snapshots
* rebuilding metadata cache tables
* what to do when scan fallback triggers

---

If you want, I can also write this as a literal implementation checklist per file (ordered edits with “copy/paste this block here, delete lines X–Y there”), but the structure above is already aligned to your exact modules and should be straightforward for an engineer to execute.


Below is a **literal, per-file implementation checklist** (ordered edits, “copy this here / delete that there”) that turns the current “everything-in-`iceberg_saver.py`” approach into a **single, clean Iceberg storage layer**:

* **Table management** (create/evolve partition spec/properties/schema)
* **Writing** (pyarrow → parquet tasks → snapshot commit)
* **Orchestration** (ensure table → write → refs → cache refresh → stats → tombstones)

…and leaves the Hamilton materializer responsible only for **planning + validation + observation**, which is where it belongs.

I
---

## 1) New file: `src/codeintel/storage/iceberg/writer.py`

### Goal

Move **all low-level write mechanics** (bin-pack → write_file → append/overwrite → delete matching files) out of the Hamilton materializer.

### Do this

1. **Create** `src/codeintel/storage/iceberg/writer.py`.

2. In the **original** `iceberg_saver.py`, locate the writer section starting at:

* `class _IcebergWriter:` (in your zip snapshot it begins around **line ~953**)

3. **Cut** the entire writer block through the last delete/overwrite helper:

* from `class _IcebergWriter` through `_delete_matching_files(...)`
* in the zip snapshot, that’s roughly **lines 953–1414**

4. **Paste** into `writer.py`.

5. Apply these mechanical edits inside the pasted code:

* Rename `_IcebergWriter` → `IcebergWriter`
* Change constructor signature:

  * from `plan: _IcebergPlan` → `table_schema: TableSchema`
* Replace `_write_policy` logic:

  * from `self._plan.table_schema.write_policy` → `self._table_schema.write_policy`

6. Ensure imports at top of `writer.py` include:

* `from codeintel.core.schemas.primitives import TableSchema, TableWritePolicy`

7. Add:

* `__all__ = ["IcebergWriter"]`

### Result

`IcebergWriter` becomes the only “thing” that knows about:

* `_append_producer`
* `_build_write_context`
* `_append_reader_batches`
* partitioned/unpartitioned write logic
* snapshot overwrite deletion logic

---

## 2) New file: `src/codeintel/storage/iceberg/table_manager.py`

### Goal

Move **table creation/evolution** concerns out of the materializer:

* `create_table(...)`
* `transaction().update_schema(...)`
* `set_properties(...)`
* partition spec and sort order derivation from tags

### Do this

1. **Create** `src/codeintel/storage/iceberg/table_manager.py`.

2. In the **original** `iceberg_saver.py`, cut these parts and paste them into `table_manager.py`:

#### A) Constants + tag keys

Cut these constant definitions:

* `_DEFAULT_PARTITION_COLUMNS`
* `_PARTITION_*` tag keys
* `_SORT_*` tag keys

In your zip snapshot those sit around **lines ~168–176** (and nearby).

#### B) Table ensure/update logic

Cut these functions:

* `_ensure_table(...)`
* `_ensure_namespace(...)`
* `_table_properties(...)`
* `_apply_table_updates(...)`
* `_apply_schema_update(...)`
* `_apply_table_properties(...)`
* `_apply_partition_update(...)`

In the zip snapshot: roughly **lines 814–951**.

#### C) Partition + sort spec + tag parsing

Cut these functions:

* `_resolve_partition_columns(...)`
* `_validate_partition_columns(...)`
* `_partition_spec(...)`
* `_partition_columns_from_tags(...)`
* `_sort_order(...)`
* `_tag_list(...)`
* `_tag_order_list(...)`
* `_coerce_order_item(...)`
* `_coerce_order_list(...)`
* `_parse_order_item(...)`

In the zip snapshot: roughly **lines 1415–1598**.

3. In `table_manager.py`, wrap the dependencies into a single plan object:

Add:

```python
@dataclass(frozen=True, slots=True)
class IcebergTablePlan:
    table_schema: TableSchema
    iceberg_bundle: IcebergSchemaBundle
    extras_policy: ExtrasPolicy
    write_settings: Mapping[str, object]
```

4. Replace the old `_ensure_table(...)` with a public function:

```python
def ensure_table(
    *,
    catalog: Catalog,
    identifier: tuple[str, ...],
    plan: IcebergTablePlan,
    partition_columns: tuple[str, ...],
    tag_sets: Sequence[Mapping[str, object]],
    settings: IcebergSettings,
) -> Table:
    ...
```

5. Make sure `ensure_table(...)`:

* resolves fallback partition cols using `resolve_partition_columns(...)`
* calls `catalog.create_table(...)` with:

  * `schema=plan.iceberg_bundle.schema`
  * `partition_spec=_partition_spec(...)`
  * `sort_order=_sort_order(...)`
  * `properties=_table_properties(...)`

6. Export:

* `__all__ = ["IcebergTablePlan", "ensure_table", "apply_table_updates", "resolve_partition_columns"]`

---

## 3) New file: `src/codeintel/storage/iceberg/tombstones.py`

### Goal

Make tombstones a **storage concern**, not a Hamilton materializer concern.

### Do this

1. **Create** `src/codeintel/storage/iceberg/tombstones.py`.

2. In original `iceberg_saver.py`, cut and paste the tombstone section:

* `_maybe_append_tombstones(...)`
* `_append_tombstones(...)`
* `_merge_tombstone_stats(...)`
* `_tombstone_table_key(...)`
* `_tombstone_table_schema(...)`
* `_ensure_tombstone_table(...)`
* `_tombstone_diff_reader(...)`

Zip snapshot: roughly **lines 1600–1779**.

3. Inside the moved code, replace the internal writer usage:

* Instead of calling the old internal `_build_write_context/_append_reader_batches` directly,
  use the new writer:

```python
writer = IcebergWriter(
    table=tombstone_table,
    table_schema=tombstone_schema,
    snapshot_properties=snapshot_properties,
)
writer.write(deleted_reader)
```

4. Keep `merge_tombstone_stats(...)` public.

5. Export:

* `__all__ = ["maybe_append_tombstones", "ensure_tombstone_table", "tombstone_table_key", "tombstone_table_schema", "merge_tombstone_stats"]`

---

## 4) New file: `src/codeintel/storage/iceberg/materialize.py`

### Goal

Create exactly **one entry point** for “write this dataset to Iceberg” that:

* loads catalog
* ensures/updates table
* writes (append/overwrite)
* updates snapshot refs
* refreshes metadata cache
* optional tombstones
* stats + puffin persistence

### Do this

1. **Create** `src/codeintel/storage/iceberg/materialize.py`.

2. In original `iceberg_saver.py`, cut these helper blocks and move them here:

* `_write_to_iceberg(...)`
* `_minimal_iceberg_stats(...)`
* `_update_snapshot_refs(...)`

Zip snapshot: roughly **lines 677–788**.

3. Rewrite `_write_to_iceberg(...)` as a public orchestrator:

```python
@dataclass(frozen=True, slots=True)
class IcebergMaterializeResult:
    snapshot_id: int | None
    stats: IcebergStatsPayload | None


def materialize_dataset_to_iceberg(
    *,
    table_key: str,
    plan: IcebergTablePlan,
    reader: pa.RecordBatchReader,
    snapshot_properties: Mapping[str, str],
    settings: IcebergSettings,
    tag_sets: Sequence[Mapping[str, object]],
    requested_partition_columns: tuple[str, ...],
    gateway: object | None = None,
    tombstones_enabled: bool = False,
    commit: str | None = None,
    run_id: str | None = None,
) -> IcebergMaterializeResult:
    ...
```

4. Inside `materialize_dataset_to_iceberg(...)`, implement this exact order:

* `provider = IcebergCatalogProvider(settings)`
* `catalog = provider.load()`
* `identifier = provider.resolve_identifier(table_key)`
* `partition_columns = resolve_partition_columns(plan.table_schema, requested_partition_columns)`
* `table = ensure_table(catalog=..., identifier=..., plan=..., partition_columns=..., tag_sets=..., settings=settings)`
* `previous_snapshot_id = table.metadata.current_snapshot_id`
* `IcebergWriter(table=table, table_schema=plan.table_schema, snapshot_properties=snapshot_properties).write(reader)`
* `table.refresh(); snapshot_id = table.metadata.current_snapshot_id`
* `_update_snapshot_refs(table=table, snapshot_id=snapshot_id, commit=commit, run_id=run_id, table_key=table_key)`
* if `gateway` present: `refresh_iceberg_metadata_cache(...)`
* if `tombstones_enabled` and `gateway.con` exists: call `maybe_append_tombstones(...)`
* `iceberg_stats_for_table(...)` with fallback `_minimal_iceberg_stats(...)`
* `merge_tombstone_stats(...)` if applicable
* `persist_iceberg_statistics(...)` if stats present

5. Export:

* `__all__ = ["IcebergMaterializeResult", "materialize_dataset_to_iceberg"]`

---

## 5) Edit: `src/codeintel/storage/iceberg/__init__.py`

### Goal

Make `codeintel.storage.iceberg` the “blessed” import surface for storage operations.

### Do this

1. Keep existing exports.
2. Add exports for:

* `IcebergWriter`
* `IcebergTablePlan`
* `IcebergMaterializeResult`
* `materialize_dataset_to_iceberg`
* (optional) tombstone result types

---

## 6) Edit: `src/codeintel/build/hamilton/materializers/iceberg_saver.py`

### Goal

Make the Hamilton materializer **thin**: plan → validation → observation → call storage orchestrator.

### Do this

#### A) Add imports

Add:

```python
from codeintel.storage.iceberg.materialize import materialize_dataset_to_iceberg
from codeintel.storage.iceberg.table_manager import IcebergTablePlan
```

#### B) Replace the write call site

In `_materialize_iceberg(...)`, replace:

```python
snapshot_id, iceberg_stats = _write_to_iceberg(...)
```

with this block (copy/paste):

```python
snapshot_properties = _snapshot_properties(ctx=ctx, plan=plan)

run_id = None
if ctx.env.execution_context is not None:
    run_id_value = ctx.env.execution_context.run.run_id
    if run_id_value:
        run_id = run_id_value

tag_sets = schema_tag_sets_for_table(catalog=ctx.catalog, table_key=ctx.table_key)

result = materialize_dataset_to_iceberg(
    table_key=ctx.table_key,
    plan=IcebergTablePlan(
        table_schema=plan.table_schema,
        iceberg_bundle=plan.iceberg_bundle,
        extras_policy=plan.extras_policy,
        write_settings=plan.write_settings,
    ),
    reader=reader,
    snapshot_properties=snapshot_properties,
    settings=ctx.settings_view.build.iceberg,
    tag_sets=tag_sets,
    requested_partition_columns=ctx.partition_columns,
    gateway=ctx.env.gateway,
    tombstones_enabled=ctx.settings_view.build.iceberg.tombstones_enabled,
    commit=ctx.env.commit,
    run_id=run_id,
)
snapshot_id, iceberg_stats = result.snapshot_id, result.stats
```

#### C) Delete now-moved code blocks

In the **original** `iceberg_saver.py` (zip snapshot line numbers):

1. Delete **lines ~677–788**:

* `_write_to_iceberg`
* `_minimal_iceberg_stats`
* `_update_snapshot_refs`

2. Delete **lines ~814–1779**:

* `_ensure_table` and all table update helpers
* writer class and all its helper functions
* partition/sort tag parsing helpers
* tombstone helpers

Keep:

* `_snapshot_properties(...)`
* `_persist_observation_if_ready(...)`

#### D) Cleanup imports (after the move)

Remove imports that were only used by the deleted blocks (typical ones):

* `IcebergCatalogProvider`
* pyiceberg table/partition/sorting/expressions helpers
* `refresh_iceberg_metadata_cache`, `iceberg_stats_for_table`, `persist_iceberg_statistics` (now called inside storage orchestrator)
* `parse_table_key` (only used by tombstone helpers)
* `pyarrow.compute as pc` (writer only)

Then run:

* `ruff check --fix src/codeintel/build/hamilton/materializers/iceberg_saver.py`

---

## 7) Follow-up actions to lock robustness in (high leverage)

These are not strictly required for the refactor, but they’ll address the “tests are painful / robustness is low” complaint directly:

### A) Add a single “Iceberg write contract” integration test

Create:

* `tests/storage/iceberg/test_materialize_roundtrip.py`

Test:

1. Build a tiny Polars DF → adapt to RecordBatchReader
2. Call `materialize_dataset_to_iceberg(...)`
3. Read back from Iceberg via `table.scan().to_arrow_batch_reader()`
4. Validate:

   * schema matches (including ids / mapping digest if you store it)
   * row counts
   * overwrite behavior
   * tombstones when enabled

This gives you end-to-end confidence with one test.

### B) Introduce a `GatewayProtocol`

Right now a lot of code takes a “gateway-like” object. Define a small protocol:

* `.con`
* `.schemas.*` (if needed)
  So storage-layer code doesn’t depend on the full build env.

### C) Centralize the repeated “coerce int/bool/list” helpers

Your repo has multiple `_coerce_int` variants across modules. Consolidate into:

* `codeintel/core/utils/coerce.py`
  and import everywhere. This reduces subtle drift (and test churn).

---



