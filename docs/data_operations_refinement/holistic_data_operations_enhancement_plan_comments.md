
After unzipping and reviewing the consolidated repo (especially `src/codeintel/storage/*`, `src/codeintel/build/exports/*`, and `src/codeintel/serving/*`) alongside your enhancement plan, the plan’s “repo reality” gaps are exactly the right ones to pull forward—exports buffering, upsert-from-expression pandas fallback, hand-maintained DDL typing, split extension loading, and lack of typed parameterization .

What I’d add/change to make this genuinely “best-in-class” is mostly about **operational correctness under failure + long-running streaming**, **closing the loop between build artifacts and serving runtime**, and **making session/bootstrap/config a first-class product** (not an implementation detail).

Below are the additions/adjustments I recommend to your plan (with concrete integration points based on the current code).

---

## 1) Add “atomic snapshot writes” as a first-class requirement (transaction boundaries)

### Why this matters in *your* repo

Right now, snapshot-scoped replace semantics in `Warehouse.materialize_table()` / `materialize_dataframe()` do:

1. `delete_for_snapshot(...)`
2. write/insert/upsert
3. record asset

…but it is **not wrapped in a transaction**, so a failure between (1) and (2) can permanently delete that snapshot’s data.

DuckDB auto-commits each statement by default (“single-statement transactions”), and you need explicit `BEGIN/COMMIT` (or `conn.begin()/commit()`) to group statements atomically .

### What to add to the plan

**Add a new Opportunity (or pull into Phase 1) called: *Transactional Materialization + Snapshot Safety*.**

**Implementation sketch**

* Add `StorageGateway.transaction()` (or `DuckDBSession.transaction()`) context manager:

  * `begin()`
  * `yield`
  * `commit()` / `rollback()` on exception
* Wrap:

  * `Warehouse.materialize_table()`
  * `Warehouse.materialize_dataframe()`
  * any multi-statement upsert path where you stage temp tables + insert
  * (optionally) schema bootstrap operations that need to be all-or-nothing

**Tests to add**

* “replace snapshot + forced error mid-write → snapshot rows still exist afterward”
* “replace snapshot success → all rows replaced”

### Why it’s “best-in-class”

This is the difference between “works in happy path” and “safe under failures,” which is the bar for production data ops.

---

## 2) Update the streaming plan to explicitly cover connection lifecycle, cancellation, and pool isolation

Your plan correctly calls out that streaming is currently buffering via `list(...)` and `fetchall()`  and that tests should assert you don’t do that .

### What’s missing for best-in-class streaming

Once you make exports *truly* streaming, you’ll hold a DuckDB connection open for the duration of the stream. That creates 3 new requirements the plan should state explicitly:

1. **Cancellation-aware generators**
   If the client disconnects, you must stop producing batches and close the result/connection.

2. **Pool starvation prevention**
   If your streaming endpoint uses the same read pool as interactive queries, a couple of long exports can exhaust the pool.

3. **Backpressure + bounded memory**
   Generators should naturally backpressure, but only if you aren’t pre-buffering, and only if your batch size is bounded.

### What to add/change in the plan

Add a sub-section under “Arrow-Native Pipeline + True Streaming Exports”:

**“Streaming connection lifecycle + pool isolation”**

* Introduce a dedicated export pool:

  * `ReadPoolWarehouse` for interactive queries
  * `ExportPoolWarehouse` (smaller concurrency, maybe bigger memory/temp dir, maybe separate timeout policy)
* Implement generators that:

  * acquire the warehouse/connection *inside the generator*
  * `try/finally` release it no matter how the stream ends
  * optionally check disconnect signal (FastAPI exposes disconnect/cancellation signals) and break early

This complements the plan’s existing concerns about temp object hygiene and misconfiguration risk  , but makes it explicit for streaming.

---

## 3) Extend the Arrow-first/streaming effort to build exports (not only serving)

The plan is scoped “storage + build + serving,” but the concrete streaming focus is mostly on serving endpoints .

In the consolidated repo, build exports still have at least one high-risk path:

* JSONL export materializes into pandas and then into Python dict records (OOM risk on large tables/views).

### What to add to the plan

Add a bullet under “Repo Review Addendum” or Phase 1:

**“Build exports: eliminate pandas materialization”**

* For JSONL:

  * iterate `to_pyarrow_batches()` (constant memory) 
  * write NDJSON per batch
* For Parquet:

  * you already prefer relation-level `write_parquet` when available (good)
  * remove/mitigate the remaining fallback path(s) that load full `.df()` when `write_parquet` isn’t available

This keeps build and serving consistent with the same “Arrow-native” standard the plan sets .

---

## 4) Add “capability stamping” to metadata: versions, extensions, config, and build/runtime identity

You already have strong metadata tables for pipeline runs/steps and dataset registries. What’s missing (and very valuable operationally) is the ability to answer:

* Which DuckDB version built this snapshot?
* Which extensions were loaded?
* What connection config (threads/memory/temp dir) was used?
* Which Ibis/SQLGlot versions compiled the views?

This is crucial because SQLGlot minor releases can be incompatible, and Ibis compilation depends on it (your plan already treats this as “compiler upgrade gates”) .

### What to add to the plan

Add a Phase 2 item (or Phase 1 if you want stronger reproducibility):

**“Metadata: runtime/build environment stamping”**

* New table like `metadata.environment` (or extend `metadata.pipeline_runs.extra JSON`):

  * duckdb version
  * ibis version
  * sqlglot version
  * loaded extensions list
  * key DuckDB settings (threads, memory_limit, temp_directory, extension_directory)
* Write it:

  * at build start/end
  * at serving startup (for the currently mounted DB)
* Expose it in a health endpoint for debugging.

This pairs naturally with your session bootstrap standardization goal .

---

## 5) Make “DuckDB tuning profiles” explicit (build vs serving)

Your connection layer already supports config dicts and init SQL, but a best-in-class backbone makes **profiles** a first-class concept:

* **Build profile:** maximize throughput (threads, memory_limit, controlled temp_directory spill)
* **Serving profile:** predictable latencies and bounded resource usage (lower memory_limit, maybe fewer threads)

DuckDB supports tuning these via config/SET statements (threads, memory_limit, temp_directory, etc.) .

### What to add/change in plan

Under “Centralize Session Bootstrap” add:

**“Tuning Profiles + defaults”**

* `DuckDBTuningProfile(build|serving|test)`
* applied inside the single bootstrap surface (before anything else)
* logged/stamped (see #4)
* validated by startup health check (plan already alludes to startup health checks for LOAD/INSTALL policy) 

This makes performance and stability reproducible across environments.

---

## 6) Strengthen raw SQL perimeter governance with SQLGlot strictness knobs

Your plan already pushes “Ibis-first scoping + boundary validation for raw SQL” and perimeter validation. I’d tighten it further by making SQLGlot parsing/transpilation **fail-fast** rather than “best effort.”

SQLGlot supports configuring unsupported behavior to raise instead of warn, which is exactly what you want at a safety boundary .

### What to add to the plan

Under governance:

* Enforce “SELECT-only” for any user-provided/externally sourced SQL.
* Parse with explicit dialect.
* If transpiling, set `unsupported_level=RAISE` (or equivalent) so unsupported constructs are rejected.

This complements your existing recommendation to prefer `Table.sql`/`Backend.sql` over `raw_sql` so you retain downstream lineage/validation/canonicalization .

---

## 7) Evolve the “dataflow graph” from static contract edges to derived lineage for views

You currently build dependency ordering for views by compiling SQL and parsing to extract dependencies. Your plan moves toward an Ibis→SQLGlot AST hook so you can do lineage without string parsing round-trips  and explicitly proposes table/column lineage extraction from SQLGlot ASTs .

### Best-in-class upgrade

Use that same lineage extraction to:

* populate `metadata.dataset_dataflow_edges` for views based on actual query dependencies
* compare derived lineage vs contract-declared lineage and alert on mismatches (optional strict mode)

This gives you:

* reliable provenance for docs/UX
* early warning when a view starts depending on new base tables
* better targeted invalidation for caching

---

## 8) Add a dual-mode “templating strategy”: Ibis params for correctness + DuckDB DB-API params for hot paths

Your plan’s `QueryTemplate` idea is great and aligns with Ibis’s typed parameterization approach .

For best-in-class serving performance, I’d explicitly add a **second mode** for the hottest routes:

* Use Ibis `params={...}` for typed IR templates and for AST/fingerprint generation.
* Optionally render *stable SQL text* and execute with DuckDB DB-API parameters (`?`, `$name`, etc.) for repeated execution/plan reuse where it actually matters. DuckDB supports parameterized queries in the Python DB-API , and your Ibis/SQLGlot playbook explicitly calls out this split as the clean way to get strict “same SQL, different bindings” semantics .

This is not required everywhere, but it’s a “best-in-class” knob for the endpoints that become traffic-heavy.

---

## 9) Reorder your roadmap slightly: move “transaction safety” into the top 5

Your plan’s “Implement First” list is excellent .

I would add one more item into that top tier:

* **Atomic snapshot writes (transaction boundaries)** (new item)

  * because it’s correctness/safety, not “nice-to-have performance”

So Phase 1 becomes:

1. True Streaming Exports 
2. Eliminate Pandas Write Fallbacks 
3. Centralize Session Bootstrap 
4. Typed Query Inputs 
5. Schema Round-Trips for DDL 
6. **NEW: Transactional Materialization + Snapshot Safety**

---

## 10) Add two very practical test categories

Your testing strategy section is strong . Two extra “best-in-class” test categories I’d add:

### A) Transactional semantics tests

* replace snapshot rollback behavior (see #1)

### B) Streaming lifecycle tests

* generator releases connection back to pool even on:

  * client disconnect simulation
  * mid-stream exception
* export pool isolation: long stream does not block short query paths

These are the “production-hardening” tests that prevent regressions once you have real users.

---

# If I had to summarize the recommended plan delta

Your plan already nails the big integration ideas: use Ibis params, use SQLGlot AST access for lineage/fingerprints, Arrow-native streaming, deterministic session bootstrap .

To push it into truly best-in-class territory, add:

* **Atomicity guarantees for snapshot-scoped writes** (transactions) 
* **Streaming lifecycle + pool isolation** (so “true streaming” doesn’t create operational incidents)
* **Apply the Arrow-first principle to build exports too** 
* **Capability stamping + tuning profiles** (reproducibility + predictable performance) 
* **Stricter SQL perimeter checks** (SQLGlot fail-fast) 
* **Derived lineage feeding your metadata graph** 

If you want, I can translate the above into a concrete patch plan (module-by-module changes + minimal PR sequence) that fits your existing abstractions (`StorageGateway`, `DuckDBSession`, `IbisGateway`, `Warehouse`, and the serving pools) without large refactors.

# concrete patch plan #

Below is a **concrete patch plan** that merges (a) your *Holistic Data Operations Enhancement Plan* priorities (streaming exports, eliminate pandas fallbacks, centralize bootstrap, typed inputs, schema round-trips, etc.)  with (b) the additional “make it production-grade” adjustments that fall out of the current repo structure (especially: **connection lifetime safety for streaming**, deterministic extension policy, and a “fast lane” for Arrow/replacement scans). It’s explicitly shaped to fit the abstractions you called out: **StorageGateway, DuckDBSession, IbisGateway, Warehouse, and serving pools**, without big refactors.

---

## Target end-state (what “best-in-class” looks like in this repo)

### 1) One deterministic bootstrap surface for DuckDB

Right now, extension/session bootstrap is split across connect-time behaviors and ad hoc loads, which your plan flags as a gap.  The end-state is:

* **DuckDBSession becomes the single place** that:

  * applies core PRAGMAs / settings
  * runs init SQL (secrets, repo paths, cache tuning)
  * loads extensions with an explicit policy (`INSTALL+LOAD` in build, `LOAD-only` in serving)
  * validates extension availability via a health check / startup probe

This directly addresses the “centralize session bootstrap” priority. 
And it ensures you don’t rely on DuckDB’s default “auto behaviors” (which can be surprising in production): DuckDB notes that `autoinstall_known_extensions` and `autoload_known_extensions` are enabled by default. 

---

### 2) Arrow-first query/export execution (constant memory, true streaming)

Your plan correctly calls out that “streaming” paths currently buffer via `fetchall()` and `list(...)`.  The end-state is:

* A **single Arrow-batch iterator** as the primitive: `expr.to_pyarrow_batches(chunk_size=...)` 
* NDJSON & Arrow IPC built from those batches (time-to-first-byte, constant memory)
* Parquet export uses **spool-to-disk** (Parquet needs a footer; true streaming is not practical), but still constant memory.

---

### 3) No pandas fallback in *any* write path that can be large

Your plan flags `expr.to_pandas()` in upsert-from-expression as a major OOM risk, and Python-tuple normalization as a throughput/overhead issue. 

End-state:

* **Upsert-from-expression** becomes: `INSERT ... ON CONFLICT ... SELECT ...` (no pandas) 
* **Large DataFrame / Arrow writes** become: `con.register(...) + INSERT ... SELECT ...` (replacement scan fast lane) 
* Small writes keep the current safe/tuple path (still useful for tiny frames, easier debugging).

---

### 4) Ibis ↔ SQLGlot “query intelligence layer” is first-class

Two key building blocks:

1. **SQLGlot AST hook from Ibis** (for lineage, fingerprints, diffs): `con.compiler.to_sqlglot(expr, ...)` 
2. Treat Ibis/SQLGlot bumps as compiler upgrades with golden SQL snapshots. 

This aligns with the plan’s overall “three-layer integration model.”  

---

## Module-by-module patch plan (what changes where)

### A) `src/codeintel/storage/backend/duckdb_session.py`

**Make DuckDBSession real and central.**

Add/extend:

* `DuckDBSession.open()` and `open_reader()` should:

  * call a single `_bootstrap_connection(con, cfg)` that:

    * sets deterministic config (`autoinstall_known_extensions`, `autoload_known_extensions`) based on environment/policy 
    * runs init SQL (already implemented here, but currently unused by most call sites)
    * loads extensions under a policy (see connection.py)
    * attaches history DB (existing in `connect`)
* Add a small “capabilities” check helper:

  * `DuckDBSession.check_required_extensions(required: list[str]) -> None`
  * used by serving startup health checks (fail fast if `LOAD-only` is missing an installed extension)

**Why:** Implements your “deterministic capability bootstrap” goal. 

---

### B) `src/codeintel/storage/gateway/connection.py`

**Convert `connect()` into a low-level primitive (and move policy into config/session).**

Changes:

* Replace `_load_duckdb_extensions_from_env(con)` with `_load_extensions(con, *, extensions: list[str], policy: ExtensionPolicy)`

  * `policy = INSTALL_AND_LOAD | LOAD_ONLY`
  * serving uses `LOAD_ONLY` (and fails clearly if not installed)
* Make auto behaviors explicit:

  * pass `config={"autoinstall_known_extensions": False, "autoload_known_extensions": False, ...}` for serving
  * allow build to leave defaults or set explicitly to `True` depending on your tolerance (I recommend explicit in both for determinism)
* Ensure `connect()` is called only from DuckDBSession (or at least that DuckDBSession is the preferred entry point)

---

### C) `src/codeintel/storage/gateway/factory.py`

**Use DuckDBSession everywhere the “full gateway” is opened.**

* In `open_gateway(path, *, config)`:

  * instantiate `DuckDBSession(config, path)` and use `session.open()` instead of calling `connect()` directly
* Keep the rest of the gateway wiring (IbisGateway + PolicyBackend) the same.

---

### D) `src/codeintel/storage/gateway/pool.py`

**Fix streaming correctness and unify bootstrap for pooled connections.**

* In `ReadPoolWarehouse._open()`:

  * use `DuckDBSession(cfg, path).open_reader()` instead of `connect(...)`
* Add `PoolConfig.threads` default logic improvement (optional but valuable):

  * if not specified, derive from `os.cpu_count()` and pool size to avoid oversubscribing

---

### E) `src/codeintel/storage/ibis_adapter.py` (IbisGateway)

This becomes your “query intelligence + Arrow execution + fast-lane writes” hub.

Add:

1. **Arrow execution primitives**

* `fetch_arrow(expr) -> pa.Table`
* `fetch_arrow_batches(expr, *, chunk_size=10_000) -> pa.RecordBatchReader` 

2. **AST + fingerprints**

* `to_sqlglot(expr, *, params=None, limit=None) -> exp.Expression` 
* `extract_table_lineage(expr)` and `extract_column_lineage(expr)` 
* `canonicalize(...)` + `query_fingerprint(...)` 

3. **Write fast lanes**

* `write_arrow(table_key, arrow_table, *, mode="append|replace|upsert")`

  * implement via `con.register(temp_name, arrow_table)` then `INSERT ... SELECT ...`, cleanup via `unregister` in `finally` 
* Update `_write_dataframe`:

  * if row_count >= threshold: convert to Arrow once and route to `write_arrow`
  * else keep current tuple/`executemany` path

4. **Remove pandas fallback for upsert-from-expression**

* Update `_write_ibis_expression(..., mode="upsert")`:

  * compile expression to SQL (or use SQLGlot AST)
  * call new PolicyBackend method `upsert_select(...)` (see below)
  * **delete** the `expr.to_pandas()` fallback except maybe behind a debug flag

This is explicitly one of your “implement first” priorities. 

---

### F) `src/codeintel/storage/duckdb_policy_backend.py`

Two main improvements:

1. **Upsert-from-select builder**
   Add a sibling to `insert_select(...)`:

* `upsert_select(*, table_schema: TableSchema, select_sql: str, update_columns: list[str] | None = None) -> str`

  * Build `INSERT INTO schema.table (cols...) <select_ast> ON CONFLICT(pk...) DO UPDATE SET ...`
  * This is the “no pandas” upsert path. 

2. **DDL generation round-trip (reduce maintenance tax)**
   Replace `_column_type_to_sqlglot()` and `_build_column_def()` (your plan calls this out as “hand-maintained tax”). 

* Introduce a helper:

  * `table_schema_to_ibis_schema(TableSchema) -> ibis.Schema`
  * then use Ibis schema’s `to_sqlglot_column_defs(dialect="duckdb")` 
* Use those column defs to build `CREATE TABLE ...`

If you hit any type edge cases (TIMESTAMPTZ/JSON/DECIMAL), keep a *very small* compatibility map in that helper. The goal is to move the logic into one place, not eliminate every mapping overnight.

---

### G) `src/codeintel/storage/warehouse.py`

Add an execution-time **temp object manager** to support:

* large IN-list staging
* Arrow staging temp views
* any future “scoped query” behavior

Shape:

* `@contextmanager def temp_objects(self) -> Iterator[TempObjectManager]: ...`
* `TempObjectManager.register_arrow(name, arrow_table)` / `create_temp_table_from_memtable(...)`
* ensures cleanup even on exceptions/cancellation

Your plan explicitly warns that staging needs lifecycle management and shouldn’t live in “pure query builder” code. 

---

### H) `src/codeintel/serving/semantic/kernel.py`

**Replace `fetchall()` export with Arrow-batch streaming.**

Add methods:

* `iter_export_batches(payload, *, chunk_size) -> pa.RecordBatchReader`
* `iter_export_rows(payload, *, chunk_size) -> Iterator[dict]` (implemented by converting each batch)

Remove/stop using:

* `.fetchall()` for exports (it’s the root buffering problem)

Also add lightweight telemetry hooks (duration, query hash, rows returned) as per plan. 

---

### I) `src/codeintel/serving/http/routes/v1/export.py`

This is the **most important serving correctness change**:

✅ **Do not borrow a pooled Warehouse outside the generator** once you stream.
If you stream from a generator but release the connection before iteration finishes, you’ll get correctness and concurrency bugs.

So change export routes to:

* Return a `StreamingResponse` whose **body iterator creates and closes** the warehouse inside the generator.

Implement:

* `def ndjson_body_iterator():`

  * `with manager.borrow() as warehouse:`

    * build kernel
    * iterate Arrow batches → yield NDJSON lines
* Similar for Arrow IPC stream
* Parquet: `with manager.borrow()...` write parquet to temp file, then return `FileResponse` + background cleanup

Your plan requires “remove list buffering and stream generator output directly.” 

---

### J) `src/codeintel/serving/http/streaming.py`

Add helpers to standardize streaming:

* `iter_ndjson_from_batches(reader) -> Iterator[bytes]`
* `iter_arrow_ipc_stream(reader) -> Iterator[bytes]`
* `file_response_with_cleanup(path)` (Parquet spool)

---

### K) `src/codeintel/serving/settings.py`

Add knobs:

* `export_batch_size` (default 10_000)
* `export_parquet_temp_dir`
* optional: `duckdb_extension_policy="load_only"` for serving (or just read from StorageConfig / env)

---

### L) `src/codeintel/serving/semantic/query_builder.py`

Two-layer approach:

**Layer 1 (now):** contract-driven validation (fast fail)

* validate operator compatibility using inventory column types (your plan) 
* keep current safe literal injection for small lists

**Layer 2 (next):** typed template support

* introduce QueryTemplate (see below) and gradually migrate high-value paths

---

### M) New module: `src/codeintel/serving/semantic/templates.py`

Implement the `QueryTemplate` system exactly as in your plan (with minor adjustments for your style). 
This is the foundation for:

* stable query shapes
* param-aware SQL compilation and AST analysis
* better caching/fingerprints

Ibis parameter binding mechanics are clear and consistent: `params={ScalarParamExpr: value}`. 

---

### N) New module: `src/codeintel/serving/semantic/in_list.py` (execution-time staging)

Implement the 2–3 tier strategy described in your plan (small `.isin`, large memtable + semi-join). 
But: implement it as an **execution-time context manager** (so the temp table exists during execution), consistent with the plan’s warning. 

---

### O) Tests

Add tests exactly aligned with your plan’s “Testing Strategy.” 

Key adds:

* export endpoints return StreamingResponse backed by generators (no pre-materialization)
* upsert-from-expression executes without pandas materialization
* DDL round-trip stability
* IN-list staging cleans up temp tables even on exceptions/cancel

---

## Minimal PR sequence (small, safe, mergeable)

### PR 1 — Centralize DuckDB bootstrap behind DuckDBSession

**Files**

* `storage/backend/duckdb_session.py`
* `storage/gateway/connection.py`
* `storage/gateway/factory.py`
* `storage/gateway/pool.py`

**Outcome**

* All gateways/pools open via DuckDBSession
* Extension policy is explicit (`INSTALL+LOAD` vs `LOAD-only`)
* Serving disables autoinstall/autoload unless explicitly enabled 

**Tests**

* new unit test: serving read-only connection does not attempt INSTALL
* startup “required extensions present” check (can be a simple `LOAD <ext>` test)

---

### PR 2 — IbisGateway: SQLGlot AST hook + fingerprints + lineage helpers

**Files**

* `storage/ibis_adapter.py` (+ small helper module if you prefer)

**Outcome**

* `to_sqlglot`, `canonicalize`, `query_fingerprint`, `extract_*_lineage` implemented 

**Tests**

* golden AST/SQL snapshots for 2–3 representative expressions (this becomes the start of your “compiler upgrade gate”) 

---

### PR 3 — Arrow-batch export primitive in SemanticKernel (no fetchall)

**Files**

* `serving/semantic/kernel.py`
* `storage/ibis_adapter.py` (add `fetch_arrow_batches` if not already)

**Outcome**

* Kernel can produce Arrow RecordBatchReader via `to_pyarrow_batches()` 
* No `fetchall()` in export paths

**Tests**

* unit test: exporting uses batch reader (can be asserted by mocking)

---

### PR 4 — Fix export endpoint streaming *correctly* (connection lifetime-safe)

**Files**

* `serving/http/routes/v1/export.py`
* `serving/http/streaming.py`

**Outcome**

* NDJSON endpoint streams without `list(...)` materialization 
* Warehouse borrowing happens inside the generator (no premature pool release)

**Tests**

* regression test that route returns StreamingResponse and does not build a list
* cancellation test (best-effort): ensure generator cleanup releases connection

---

### PR 5 — Arrow IPC streaming + Parquet spool-to-disk

**Files**

* `serving/http/routes/v1/export.py`
* `serving/http/streaming.py`
* `serving/settings.py`

**Outcome**

* Arrow IPC stream export (true streaming)
* Parquet export via temp file + FileResponse cleanup

**Tests**

* Arrow response content-type and non-empty body
* Parquet file created + cleaned up

---

### PR 6 — Eliminate pandas fallback: implement `upsert_select`

**Files**

* `storage/duckdb_policy_backend.py`
* `storage/ibis_adapter.py`

**Outcome**

* `IbisGateway._write_ibis_expression(mode="upsert")` uses `INSERT ... ON CONFLICT ... SELECT ...` 
* pandas fallback removed or gated behind an explicit env flag for emergency rollback

**Tests**

* upsert-from-expression test that fails if `to_pandas()` is called
* semantic equivalence test comparing old vs new for small sample

---

### PR 7 — Bulk writes fast lane via Arrow/replacement scans

**Files**

* `storage/ibis_adapter.py`
* `storage/warehouse.py` (optional helper)
* maybe `storage/duckdb_policy_backend.py` (if you want an insert-from-relation builder)

**Outcome**

* Large DataFrame writes avoid tuple loops 
* `write_arrow(...)` exists and is used above threshold 

**Tests**

* large-ish dataframe insert uses register path
* cleanup: `unregister` always happens

---

### PR 8 — Schema/DDL round-trip via Ibis Schema → SQLGlot ColumnDefs

**Files**

* `storage/duckdb_policy_backend.py`
* small new helper module for `TableSchema -> ibis.Schema`

**Outcome**

* Remove/retire `_column_type_to_sqlglot()` / `_build_column_def()` maintenance tax 
* DDL output remains stable/parseable

**Tests**

* DDL snapshot tests for several TableSchemas (including DECIMAL, JSON, TIMESTAMPTZ)

---

### PR 9 — Contract-driven filter validation + execution-time IN-list staging

**Files**

* `serving/semantic/query_builder.py`
* `serving/semantic/in_list.py` (new)
* `storage/warehouse.py` (temp object manager)

**Outcome**

* Invalid operator/type combos are rejected before DuckDB 
* Large IN lists are staged via memtable+semi-join at execution time 

**Tests**

* invalid filters error early
* temp tables cleaned up on exception

---

### PR 10 — Observability alignment + compiler upgrade gate

**Files**

* `serving/semantic/kernel.py`
* `storage/warehouse.py`
* `tests/...` golden SQL snapshots

**Outcome**

* lightweight telemetry (duration, rows, query hash, schema hash) 
* golden SQL snapshots + execution validation for compiler bumps 

---

## A few “best-in-class” additions I’d include (small, high impact)

1. **Export streaming correctness > performance**
   The subtle but critical improvement is the connection-lifetime-safe generator structure. This is the #1 thing that prevents “it works in tests but fails under concurrency.”

2. **Explicit extension policy + startup check**
   Because serving must default to `LOAD-only` (plan mitigation). 
   Make the failure mode obvious at boot, not at first query.

3. **Compiler upgrade discipline**
   The Ibis/SQLGlot coupling means changes can silently alter SQL semantics, so the golden snapshot suite should be treated like a compiler test suite. 

---

If you want, I can turn this into a “PR checklist doc” you can drop into the repo (e.g., `docs/data_ops_patch_plan.md`) with:

* exact function signatures,
* acceptance criteria per PR,
* and a test matrix mapping each risk to a test (matching your risk table + testing strategy). 
