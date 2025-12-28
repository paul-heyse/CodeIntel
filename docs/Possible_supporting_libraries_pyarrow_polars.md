Below is a **feature-level catalog** for each library, scoped to what materially matters when your “native plane” is **PyArrow ↔ Polars** (Arrow buffers, Arrow C interfaces, Arrow datasets/scanners/record-batch streams, and Polars eager/lazy execution).

---

## Delta Lake (Delta protocol) in an Arrow/Polars stack

### A) delta-rs / `deltalake` (Python, non-Spark) — table state, reads, writes, DML, maintenance

**Table identity + storage backends**

* Load a table state: `from deltalake import DeltaTable; dt = DeltaTable(uri_or_path, storage_options=..., version=...)` (supports version pinning / time travel) and later `dt.load_version(v)` / `dt.load_with_datetime("...")`. ([delta-io.github.io][1])
* Cloud/object-store auth is passed via `storage_options` (and also env-var based) and delta-rs documents supported URI schemes for S3/Azure/GCS. ([delta-io.github.io][1])
* Catalog-backed loading: `DeltaTable.from_data_catalog(data_catalog=..., database_name=..., table_name=...)` supports at least AWS Glue + Databricks Unity Catalog in delta-rs docs. ([delta-io.github.io][1])
* Arrow filesystem customization: delta-rs can read bulk data through any `pyarrow.fs.FileSystem` (common pattern: wrap with `pyarrow.fs.SubTreeFileSystem`). ([delta-io.github.io][1])

**Metadata / introspection (high leverage for “table-aware” engines)**

* Table metadata: `dt.metadata()` exposes table id/name/description/partition columns/created_time/config map (e.g. `delta.appendOnly`). ([delta-io.github.io][1])
* Schema surfaces: `dt.schema()` returns Delta schema; `dt.schema().to_pyarrow()` returns Arrow schema; plus JSON round-tripping via `Schema.json()` / `Schema.from_json(...)`. ([delta-io.github.io][1])
* Provenance/history: `dt.history()` returns operation log entries (when written by a writer that records it; retention governed by `delta.logRetentionDuration`). ([delta-io.github.io][1])
* File-state introspection: `dt.get_add_actions(flatten=True)` yields a dataframe-like view with file stats (records/min/max/null counts), and can be queried at past versions. ([delta-io.github.io][1])

**Arrow-native reads (what makes it fit an Arrow/Polars design)**

* Materialize to Arrow: `dt.to_pyarrow_table(partitions=[...], columns=[...])` supports partition pruning + projection. ([delta-io.github.io][1])
* Materialize to Arrow Dataset: `dt.to_pyarrow_dataset(filesystem=...)` (lets you drive `ds.scanner(...).to_reader()` / `to_batches()` in pure Arrow for streaming). ([delta-io.github.io][1])
* “Large table” strategy is explicitly called out: partition filters, column projection, and batch-by-batch reading paths are described as first-class concerns. ([delta-io.github.io][1])

**Writes (append/overwrite)**

* Main writer: `from deltalake import write_deltalake; write_deltalake(target_uri, data, mode=..., overwrite_schema=..., partition_by=...)`.

  * `data` accepts **Pandas**, **PyArrow Table**, or an **iterator of PyArrow RecordBatch** (the iterator path is the key “streaming ingest” seam in an Arrow stack). ([delta-io.github.io][1])
  * Modes: `mode="overwrite"` / `mode="append"`; by default errors if table exists; schema mismatch raises `ValueError` unless `overwrite_schema=True`. ([delta-io.github.io][1])
* Targeted partition overwrite: `write_deltalake(table_or_uri, df, mode="overwrite", partition_filters=[("y","=","b")])` overwrites exactly one partition (idempotent insert/replace for that partition). ([delta-io.github.io][1])

**In-place DML (row-level update/delete)**

* Update rows: `dt.update(updates={col: "expr", ...}, predicate="...")` where updates/predicate are strings. Crucially: delta-rs notes these strings are **parsed into Apache DataFusion expressions** (meaning DataFusion’s expression grammar/semantics matters for correctness). ([delta-io.github.io][1])
* Delete rows: `dt.delete(predicate="...")` rewrites or removes affected files depending on whether the predicate matches full partitions vs rows within files; it updates table state but does not delete physical files until vacuum. ([delta-io.github.io][1])
* Merge/upsert: delta-rs exposes a merge API in its Python API reference: `dt.merge(source, predicate, source_alias=..., target_alias=..., error_on_type_mismatch=...)` as the entrypoint to construct merge semantics. ([delta-io.github.io][2])

**Maintenance (the operational “must-know” for production-like usage)**

* Vacuum: `dt.vacuum(dry_run=True|False)` defaults to **dry-run** (lists files) to prevent accidental deletion; vacuum can break time travel beyond the retained window. ([delta-io.github.io][1])
* Optimize/compaction: `dt.optimize.compact()` does bin-packing (small-file compaction) and produces remove-actions but does not delete storage objects; you vacuum to reclaim. It can fail under certain concurrent file-removal writes. ([delta-io.github.io][1])
* Z-order: `dt.optimize.z_order(["colA","colB"])` exists in delta-rs optimizer to improve data skipping for multi-column filters. ([delta-io.github.io][1])

### B) Polars’ Delta IO (thin layer over delta-rs, but important ergonomics)

**Read**

* Eager read: `pl.read_delta(source, version=..., columns=..., storage_options=...)` reads a Delta table (path/URI) and can pin version; latest if omitted. ([Polars][3])
* Lazy read: Polars exposes `pl.scan_delta(source, version=..., ...)` for lazy pipelines (listed in Polars IO index). ([Polars][4])

**Write**

* `df.write_delta(target, mode="error|append|overwrite|ignore|merge", overwrite_schema=..., storage_options=..., delta_write_options=...)`.

  * `mode="merge"` returns a `TableMerger` object (merge/upsert workflow). ([Polars][5])
  * `overwrite_schema` is deprecated in favor of `delta_write_options={"schema_mode":"overwrite"}`. ([Polars][5])
  * Type constraints: Polars docs note `Null` and `Time` aren’t supported by the Delta protocol and error; `Categorical` is converted to strings on write; and writing non-nullable Delta columns requires providing a custom PyArrow schema via `delta_write_options`. ([Polars][6])

---

## Apache Iceberg in an Arrow/Polars stack

### A) Core Iceberg semantics worth designing around (even if you use PyIceberg)

* **Snapshots are fundamental** (basis for reader isolation + time travel) and Iceberg has lifecycle management (expire snapshots) to control metadata size/cost. ([Apache Iceberg][7])
* Branching/tagging are first-class in the table metadata model (mutable branches, immutable tags). ([Apache Iceberg][7])
* Delete files are part of the format: position deletes + equality deletes exist alongside data files (this matters for “does my reader support deletes?”). PyIceberg’s inspection output explicitly distinguishes `content=0` data vs `1` position deletes vs `2` equality deletes. ([PyIceberg][8])

### B) PyIceberg (Python) — catalog-centric + Arrow-first scans + Polars integration

**Catalog + table loading**

* Catalog-centric loading: `from pyiceberg.catalog import load_catalog; catalog = load_catalog(name, **props)`; configs can live in `.pyiceberg.yaml` and support multiple catalogs (e.g., Hive/REST). ([PyIceberg][8])
* `catalog.load_table("ns.table")` returns a mutable `Table` object (read/write/alter). Static metadata loading is described as read-only. ([PyIceberg][8])

**Create table (schema, partition spec, sort order)**

* Create with Iceberg schema objects: `catalog.create_table(identifier=..., schema=..., partition_spec=..., sort_order=...)`. ([PyIceberg][8])
* Create with Arrow schema: `schema = pa.schema([...]); catalog.create_table(identifier=..., schema=schema)`. ([PyIceberg][8])
* Transactional creation helper: `with catalog.create_table_transaction(identifier=..., schema=...) as txn: ... txn.update_schema(); txn.update_spec(); txn.set_properties(...)`. ([PyIceberg][8])

**Write + row-level operations**

* Append: `tbl.append(pa_table, snapshot_properties={...})`. ([PyIceberg][8])
* Overwrite: `tbl.overwrite(pa_table, overwrite_filter=..., snapshot_properties=..., case_sensitive=True, branch=...)` (Iceberg “overwrite” can map to delete/overwrite/append behaviors depending on whether files can be dropped vs rewritten). ([PyIceberg][9])
* Delete rows: `tbl.delete(delete_filter=..., snapshot_properties=..., branch=...)`. ([PyIceberg][9])
* Dynamic partition overwrite: `tbl.dynamic_partition_overwrite(pa_table, snapshot_properties=..., branch=...)` replaces partitions detected from the input. ([PyIceberg][9])
* Upsert: `tbl.upsert(pa_table)` with identifier fields; PyIceberg docs show it computing inserted/updated rows and ignoring duplicates. ([PyIceberg][8])

**Scan planning (the “hidden superpower”: file-level planning + engine handoff)**

* Construct a scan: `scan = table.scan(row_filter=..., selected_fields=(...), limit=..., snapshot_id=...)`. ([PyIceberg][8])
* Low-level planning: `scan.plan_files()` returns file tasks (lets you own execution in Arrow/Polars/DuckDB/etc). PyIceberg explicitly frames this as a low-level API where “engine filters the file itself” vs the higher-level conversion helpers. ([PyIceberg][8])
* Arrow outputs:

  * `scan.to_arrow()` returns `pyarrow.Table`. ([PyIceberg][8])
  * `scan.to_arrow_batch_reader()` returns a `pyarrow.RecordBatchReader` (streaming). ([PyIceberg][8])

**Polars integration (two distinct surfaces)**

* “Use Iceberg scan API, analyze in Polars”: `table.scan(...).to_polars()` → Polars **DataFrame** (PyIceberg drives file selection + filtering). ([PyIceberg][8])
* “Use Polars for lazy filtering/retrieval”: `table.to_polars()` → Polars **LazyFrame** rooted in the Iceberg table. ([PyIceberg][8])

**Inspection / metadata tables (often underused, huge leverage)**

* Snapshots: `table.inspect.snapshots()` ([PyIceberg][8])
* Partitions: `table.inspect.partitions()` includes record/file counts and delete counts. ([PyIceberg][8])
* Entries/manifests/refs/history/files/metadata log:

  * `table.inspect.entries()`, `table.inspect.manifests()`, `table.inspect.refs()`, `table.inspect.history()`, `table.inspect.files()`, `table.inspect.metadata_log_entries()`. ([PyIceberg][8])
* Time travel for inspection: metadata inspection can accept `snapshot_id=...` (PyIceberg notes time travel supported on all metadata tables except `snapshots` and `refs`). ([PyIceberg][8])
* Delete/data file split: `table.inspect.data_files()` vs `table.inspect.delete_files()` is explicitly documented. ([PyIceberg][8])

**“Add files” (no rewrite)**

* `tbl.add_files(file_paths=[...], snapshot_properties=..., check_duplicate_files=True)` commits existing Parquet files as Iceberg data files without rewriting, producing a new snapshot and manifests. ([PyIceberg][8])
* Name mapping / field IDs nuance: if Parquet files lack field IDs, the table needs a name mapping; PyIceberg can auto-create one based on current schema if missing. ([PyIceberg][8])

**Schema evolution (full API surface)**

* Entry: `with table.update_schema() as update: ...` or via transaction: `with table.transaction() as tx: with tx.update_schema() as update: ...`. ([PyIceberg][8])
* Operations: `update.union_by_name(schema)`, `update.add_column(...)`, `update.rename_column(...)`, `update.move_first/move_after/move_before(...)`, `update.update_column(...)`, `update.delete_column(...)` (delete requires `allow_incompatible_changes=True`). ([PyIceberg][8])

**Partition evolution (hidden partitioning done properly)**

* Entry: `with table.update_spec() as update: ...` (also available under transaction). ([PyIceberg][8])
* Operations: `update.add_field(field, transform, name)`, `update.identity(field)`, `update.remove_field(name)`, `update.rename_field(old, new)`; examples use `BucketTransform`, `DayTransform`. ([PyIceberg][8])

**Sort order updates**

* Entry: `with table.update_sort_order() as update: ...` and only additive (cannot delete/modify existing). ([PyIceberg][8])
* Ops: `update.asc(field, transform, null_order)` / `update.desc(...)`. ([PyIceberg][8])

**Table + snapshot properties**

* Table properties via transaction: `with table.transaction() as tx: tx.set_properties(...); tx.remove_properties(...)` or fluent `table.transaction().set_properties(...).commit_transaction()`. ([PyIceberg][8])
* Snapshot properties at write-time: `tbl.append(df, snapshot_properties={...})` / `tbl.overwrite(df, snapshot_properties={...})`. ([PyIceberg][8])

**Branching + tagging (snapshot refs)**

* Manage snapshots: `table.manage_snapshots().create_tag(...).commit()`, `.remove_tag(...)`, `.create_branch(...)`, `.remove_branch(...)`; retention knobs include `max_ref_age_ms`, `max_snapshot_age_ms`, `min_snapshots_to_keep`. ([PyIceberg][8])
* Tags are immutable snapshot refs; branches are mutable refs. ([PyIceberg][8])

**Maintenance**

* PyIceberg exposes `table.maintenance.expire_snapshots()` with `.older_than(datetime)` / `.by_id(id)` and context-manager patterns. ([PyIceberg][8])
* “Remove orphan files” exists as a core Iceberg maintenance concept, but official Iceberg maintenance docs warn that too-short retention risks corrupting the table by deleting in-progress files (default interval is 3 days). ([Apache Iceberg][10])

---

## ConnectorX (DB → Arrow/Polars ingest) — what you might miss

**Core API surface**

* Primary entrypoint: `connectorx.read_sql(conn, query, return_type="polars|arrow|arrow_stream|pandas|modin|dask", protocol="binary", partition_on=..., partition_range=..., partition_num=...)`. ([SFU Database][11])
* `conn` can be a URI string *or* a dict of named DB URIs (multi-source querying model). ([SFU Database][11])
* `query` can be a single SQL string or an explicit list of partitioned SQL queries (manual partitioning when auto-splitting isn’t viable). ([SFU Database][11])
* `return_type="arrow_stream"` is the underused “stream batches instead of one big frame” mode (useful for backpressure pipelines). ([SFU Database][11])

**Parallelism and partition planning**

* Partition-based parallelism is a first-class design: ConnectorX explicitly positions itself around zero-copy + partition parallelism. ([Docs.rs][12])
* Tuning guidance (from ConnectorX packaging docs): `partition_num` should often match logical cores; `partition_on` ideally numeric and evenly distributed; indexes can affect performance depending on DB. ([PyPI][13])

**Polars integration surface**

* Polars front door: `pl.read_database_uri(query, uri, partition_on=..., partition_range=..., partition_num=..., protocol=...)` (ConnectorX is the default engine and these kwargs are annotated as ConnectorX-specific). ([Polars][14])
* Polars docs explicitly note ConnectorX is Rust-based and stores data in Arrow format enabling zero-copy to Polars. ([Polars][15])

**Known sharp edges you’ll want in a design doc**

* ConnectorX return-type compatibility can break across versions (e.g., Polars issue reports a failure tied to ConnectorX removing `"arrow2"` as a `return_type` value). ([GitHub][16])
* Timestamp range hazards can appear if a DB returns timestamps interpreted at nanosecond precision (reported “out of range DateTime” with very large dates). ([GitHub][17])

---

## ADBC (Arrow Database Connectivity) — Arrow-native DBAPI, metadata, ingestion, progress

**Two-layer API design (important for choosing abstraction level)**

* Low-level: `adbc_driver_manager` is a close mapping to the C API; high-level: `adbc_driver_manager.dbapi` provides a DBAPI (PEP 249) interface and explicitly requires PyArrow. ([Apache Arrow][18])
* ADBC docs note: if **PyArrow or Polars** are installed, ADBC provides a DBAPI-style API. ([Apache Arrow][19])

**Prepared statements + parameter typing**

* ADBC exposes parameter schema introspection: `Statement.get_parameter_schema()` (documented as returning an Arrow schema describing ordinal/named parameters and types; unknown types may be NA/NullType). ([Apache Arrow][20])

**Bulk ingestion (DB writes without row-wise DBAPI)**

* Driver manager exposes experimental ingest configuration keys:

  * Mode: `adbc_driver_manager.INGEST_MODE` / `INGEST_OPTION_MODE_{CREATE,APPEND}`
  * Target: `INGEST_TARGET_TABLE`, `INGEST_TARGET_CATALOG`, `INGEST_TARGET_DB_SCHEMA`
  * Temporary ingest: `INGEST_TEMPORARY`
  * Plus statement progress key: `PROGRESS = 'adbc.statement.exec.progress'` ([Apache Arrow][21])
* Net effect: a write path that is conceptually “Arrow stream → DB table” rather than “rows → cursor.executemany”.

**Partitioned execution + progress**

* ADBC driver manager includes “incremental execution on ExecutePartitions” (notable for long-running analytics queries) and a progress retrieval option. ([Apache Arrow][21])

**Flight SQL driver (ADBC as the client library for Flight SQL servers)**

* ADBC Flight SQL driver is documented as providing access to any Flight SQL compatible endpoint; docs list Python availability. ([Apache Arrow][22])
* Typical Python usage pattern in the ecosystem is `import adbc_driver_flightsql.dbapi; conn = adbc_driver_flightsql.dbapi.connect(...)` (example shown in third-party Flight SQL integrations). ([Deephaven][23])

---

## Arrow-over-ODBC (`arrow-odbc`) — “ODBC in, Arrow batches out” (and back)

**Read path (ODBC → Arrow batches → PyArrow/Polars)**

* High-level read: `arrow_odbc.read_arrow_batches_from_odbc(query=..., connection_string=..., batch_size=..., user=..., password=..., ...) -> BatchReader`. ([Arrow ODBC Documentation][24])
* `BatchReader` is an iterator over Arrow batches, and can be converted to a PyArrow stream: `BatchReader.into_pyarrow_record_batch_reader()` (transfers ownership; leaves the original reader empty). ([Arrow ODBC Documentation][24])
* Multiple result sets (stored procedures / multi-statement queries): `BatchReader.more_results(...) -> bool` advances to the next result set without requiring you to exhaust the current one. ([Arrow ODBC Documentation][24])

**Performance knobs (surprisingly deep)**

* Batch sizing and memory limits: `more_results(batch_size=..., max_bytes_per_batch=..., max_text_size=..., max_binary_size=..., ...)`. ([Arrow ODBC Documentation][24])
* Concurrency knob: `fetch_concurrently=True|False` trades memory for speed by allocating an additional transit buffer and fetching the next batch on a dedicated thread while converting the prior batch to Arrow arrays (docs note near ~2× memory when enabled). ([Arrow ODBC Documentation][24])

**Write path (Arrow → ODBC)**

* The Python module surface lists DB write helpers: `arrow_odbc.from_table_to_db(...)`, `arrow_odbc.insert_into_table(...)`, and a `BatchWriter` type. ([Arrow ODBC Documentation][24])

**Connection plumbing**

* Explicit connect helper: `arrow_odbc.connect(...)` and global pooling hook: `arrow_odbc.enable_odbc_connection_pooling()`. ([Arrow ODBC Documentation][24])

---

## Arrow Flight SQL — protocol features that matter for Arrow/Polars systems

**What it is**

* Flight SQL is a protocol to interact with SQL databases using Arrow in-memory format + Flight RPC; clients fetch both query results and metadata as Arrow. ([Apache Arrow][25])

**Metadata as Arrow (often underused, but huge for agentic systems)**

* Metadata commands (catalogs, db schemas, FK references, etc.) are invoked via `GetFlightInfo` / `GetSchema`, then results are retrieved via `DoGet` as Arrow data with fixed schemas. ([Apache Arrow][25])

**Query execution + prepared statements**

* Flight SQL defines commands for executing queries and managing prepared statements (create/close). ([Apache Arrow][25])
* Parameter binding: `CommandPreparedStatementQuery` can be used with `DoPut` to bind parameters; spec describes servers optionally returning an updated handle (enables stateless servers by encoding bound values into the handle). ([Apache Arrow][25])

**Updates + ingestion**

* Update/ingest commands (e.g., `CommandStatementUpdate`, `CommandStatementIngest`) return results via a `DoPutUpdateResult` encoded in Flight `PutResult.app_metadata` after consuming the full stream. ([Apache Arrow][25])

**Capability discovery (design-critical in heterogeneous ecosystems)**

* The `SqlInfo` capability surface includes whether the server supports SQL, Substrait, the Flight SQL transaction endpoints, query cancellation, bulk ingestion, etc. (e.g., `FLIGHT_SQL_SERVER_SUBSTRAIT`, `FLIGHT_SQL_SERVER_TRANSACTION`, `FLIGHT_SQL_SERVER_CANCEL`, `FLIGHT_SQL_SERVER_BULK_INGESTION`). ([Apache Arrow][25])

**How you typically consume it from Python today**

* Most Arrow-native Python clients go through ADBC Flight SQL (`adbc-driver-flightsql`), which is explicitly documented as a driver that can talk to any Flight SQL endpoint. ([Apache Arrow][22])

---

## DataFusion (Python bindings) — Arrow-native query engine + interop glue

**Execution engine primitives**

* `datafusion.SessionContext(...)` is documented as having a query optimizer, physical planner, and multi-threaded execution engine. ([arrow.staged.apache.org][26])
* SQL entrypoint: `ctx.sql(query) -> DataFrame` (plan corresponds to SQL). ([arrow.staged.apache.org][26])
* Partition execution streaming: `ctx.execute(plan, part)` returns a stream of record batches. ([arrow.staged.apache.org][26])

**Arrow/Polars ingestion**

* Arrow import in user guide: `SessionContext.from_arrow(...)` accepts any Python object implementing `__arrow_c_stream__` or `__arrow_c_array__` (Arrow C interfaces), including `pyarrow.Table`, `RecordBatch`, and `RecordBatchReader`. ([Apache DataFusion][27])
* API surface also includes `SessionContext.from_arrow_table(data, name=None)` and `SessionContext.from_polars(data, name=None)` per the generated docs. ([arrow.staged.apache.org][26])

**IO surfaces (DataFusion as an Arrow-native scan engine)**

* Reads: `ctx.read_parquet(path, parquet_pruning=True, skip_metadata=True, schema=...)`, plus `read_csv/read_json/read_avro`. ([arrow.staged.apache.org][26])
* Registration (for SQL referencing): `ctx.register_parquet(name, path, parquet_pruning=..., skip_metadata=...)`, `ctx.register_dataset(name, dataset)`, and `ctx.register_record_batches(name, partitions)`. ([arrow.staged.apache.org][26])
* Object store plumbing: `ctx.register_object_store(scheme, store, host=None)` (useful if you want custom S3/GCS/etc object store implementations). ([arrow.staged.apache.org][26])

**UDF/UDAF hooks**

* Register UDF/UDAF: `ctx.register_udf(udf)` / `ctx.register_udaf(udaf)`. ([arrow.staged.apache.org][26])

**Less-obvious capabilities (worth designing for)**

* DataFusion PyPI feature list calls out:

  * exchanging data with dataframe libs supporting PyArrow,
  * **Substrait plan** serialize/deserialize,
  * and **experimental SQL transpilation** into DataFrame calls (including Polars). ([PyPI][28])
* DataFusion is also a hidden dependency boundary in delta-rs: delta-rs states it parses update predicates/expressions into DataFusion expressions. ([delta-io.github.io][1])

---

## Narwhals — dataframe-agnostic layer that plays well with Polars + PyArrow

**Core mental model**

* Wrap a native object: `nw.from_native(native_df_or_lf_or_series, ...)`, compute using a supported subset of Polars-like API, then unwrap with `nw.to_native(...)`. ([Narwhals Dev][29])
* Supports wrapping DataFrame/LazyFrame across multiple backends, explicitly including PyArrow + Polars in docs/README. ([GitHub][30])

**Laziness preservation**

* Narwhals explicitly guarantees: if you start from a lazy backend, calculations stay lazy; it “will never automatically trigger computation without you asking”. ([Narwhals Dev][29])
* Force eager-only behavior: `nw.from_native(obj, eager_only=True)` (documented as the knob when you want eager semantics). ([Narwhals Dev][29])

**Extension hooks (for integrating custom dataframe-like objects)**

* `nw.from_native(...)` can accept objects implementing `__narwhals_dataframe__`, `__narwhals_lazyframe__`, or `__narwhals_series__` (this is the “plug in your own frame type” seam). ([Narwhals Dev][31])

**Utility surface that matters in mixed ecosystems**

* Namespace detection: `nw.get_native_namespace(obj)` (lets libraries branch without type-checking every backend). ([Narwhals Dev][31])
* Index/dtype adapters (particularly relevant if you must interop with pandas-ish inputs while remaining Polars-first):

  * `nw.maybe_align_index(df, s)`, `nw.maybe_get_index(obj)`, `nw.maybe_reset_index(df)`, `nw.maybe_set_index(df, ...)`
  * dtype hygiene helpers like `nw.maybe_convert_dtypes(...)` ([Narwhals Dev][31])

---

### If you want one extra “design payoff” output

I can translate the above into a **capability matrix keyed by pipeline stage** (ingest → storage/table semantics → query/compute → transport → compatibility), where each cell lists:

* the exact Python entrypoints,
* what Arrow object types they consume/emit (`pa.Table`, `pa.RecordBatchReader`, Polars `DataFrame`/`LazyFrame`),
* and the operational obligations (maintenance/retention, schema evolution constraints, delete-file support, etc.).

[1]: https://delta-io.github.io/delta-rs/python/usage.html "Usage — delta-rs  documentation"
[2]: https://delta-io.github.io/delta-rs/python/api_reference.html?utm_source=chatgpt.com "API Reference — delta-rs documentation"
[3]: https://docs.pola.rs/py-polars/html/reference/api/polars.read_delta.html?utm_source=chatgpt.com "polars.read_delta — Polars documentation"
[4]: https://docs.pola.rs/api/python/version/0.19/reference/io.html?utm_source=chatgpt.com "Input/output — Polars documentation"
[5]: https://docs.pola.rs/py-polars/html/reference/api/polars.DataFrame.write_delta.html?utm_source=chatgpt.com "polars.DataFrame.write_delta — Polars documentation"
[6]: https://docs.pola.rs/py-polars/html/reference/dataframe/index.html?utm_source=chatgpt.com "DataFrame — Polars documentation"
[7]: https://iceberg.apache.org/docs/1.8.1/branching/?utm_source=chatgpt.com "Branching and Tagging - Apache Iceberg™"
[8]: https://py.iceberg.apache.org/api/ "API - PyIceberg"
[9]: https://py.iceberg.apache.org/reference/pyiceberg/table/ "table - PyIceberg"
[10]: https://iceberg.apache.org/docs/latest/maintenance/?utm_source=chatgpt.com "Maintenance - Apache Iceberg™"
[11]: https://sfu-db.github.io/connector-x/api.html "Basic usage — ConnectorX"
[12]: https://docs.rs/connectorx?utm_source=chatgpt.com "connectorx - Rust"
[13]: https://pypi.org/project/connectorx/0.2.3/?utm_source=chatgpt.com "connectorx 0.2.3"
[14]: https://docs.pola.rs/py-polars/html/reference/api/polars.read_database_uri.html?utm_source=chatgpt.com "polars.read_database_uri — Polars documentation"
[15]: https://docs.pola.rs/user-guide/io/database/?utm_source=chatgpt.com "Databases - Polars user guide"
[16]: https://github.com/pola-rs/polars/issues/21274?utm_source=chatgpt.com "read_database_uri fails with \"ValueError: arrow2\" because ..."
[17]: https://github.com/pola-rs/polars/issues/16768?utm_source=chatgpt.com "read_database_uri panics for dates beyond 2262.04.11"
[18]: https://arrow.apache.org/adbc/0.1.0/python/api/adbc_driver_manager.html?utm_source=chatgpt.com "adbc_driver_manager - ADBC 0.1.0 documentation"
[19]: https://arrow.apache.org/adbc/main/python/driver_manager.html?utm_source=chatgpt.com "Driver Manager - ADBC 22 (dev) documentation - Apache Arrow"
[20]: https://arrow.apache.org/adbc/0.4.0/python/api/adbc_driver_manager.html?utm_source=chatgpt.com "adbc_driver_manager - ADBC 0.4.0 documentation"
[21]: https://arrow.apache.org/adbc/main/python/api/adbc_driver_manager.html?utm_source=chatgpt.com "adbc_driver_manager - ADBC 22 (dev) documentation"
[22]: https://arrow.apache.org/adbc/0.5.1/driver/flight_sql.html?utm_source=chatgpt.com "Flight SQL Driver - ADBC 0.5.1 documentation"
[23]: https://deephaven.io/core/0.40.1/docs/how-to-guides/flight-sql/?utm_source=chatgpt.com "Connecting to a Deephaven Flight SQL server"
[24]: https://arrow-odbc.readthedocs.io/en/stable/arrow_odbc.html "arrow_odbc package — arrow-odbc 9.3.0 documentation"
[25]: https://arrow.apache.org/docs/format/FlightSql.html "Arrow Flight SQL — Apache Arrow v22.0.0"
[26]: https://arrow.staged.apache.org/datafusion-python/generated/datafusion.SessionContext.html "datafusion.SessionContext — Apache Arrow DataFusion  documentation"
[27]: https://datafusion.apache.org/python/user-guide/io/arrow.html "Arrow — Apache Arrow DataFusion  documentation"
[28]: https://pypi.org/project/datafusion/31.0.0/?utm_source=chatgpt.com "DataFusion in Python"
[29]: https://narwhals-dev.github.io/narwhals/basics/dataframe/ "DataFrame - Narwhals"
[30]: https://github.com/narwhals-dev/narwhals "GitHub - narwhals-dev/narwhals: Lightweight and extensible compatibility layer between dataframe libraries!"
[31]: https://narwhals-dev.github.io/narwhals/api-reference/narwhals/ "Top-level functions - Narwhals"
