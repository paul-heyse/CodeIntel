
According to a document from December 27, 2025, DuckDB can interoperate very tightly with both Polars and PyArrow (including replacement scans over `pyarrow.Dataset` / `pyarrow.Scanner` and streaming Arrow readers), which makes it feasible to *reposition DuckDB as an optional edge tool* while making **PyArrow + Polars the default data plane** for your system.

Below is a comprehensive assessment of the repo (from `CodeIntel-pre_dataops_revamp.zip`) and a concrete, breaking refactor proposal to maximize PyArrow + Polars usage, replace most SQL, and *prefer Polars/PyArrow over DuckDB* (while still allowing DuckDB where it’s genuinely the best tool).

---

## What you have today (repo assessment)

### 1) DuckDB is the “center of gravity”

The `storage/` layer (gateway, warehouse, policy backend, view materialization, repositories) is DuckDB-centric. The most important abstractions are typed around:

* `duckdb.DuckDBPyConnection`
* `duckdb.DuckDBPyRelation`
* SQL text (including a large amount of embedded SQL)

This becomes a forcing function: even when you *already* compute in Polars, the “rest of the world” expects DuckDB relations and SQL.

### 2) Polars is already used for *some* of the best parts

A bunch of your Hamilton-native analytics tables are already written in Polars style (expressions / lazy pipelines), e.g.:

* `build/hamilton/native/analytics/tables_functions.py` constructs `pl.LazyFrame` transforms and only uses DuckDB at the very end for persistence.
* `build/hamilton/transforms/tabular_steps.py` is explicitly set up to run transforms on Polars (and even supports pandas).

This is a strong signal the codebase is ready to go “Polars-first”; the *main thing holding you back* is storage + serving being “DuckDB-first”.

### 3) SQL is heavily used in “serving” and “views”

The biggest “SQL gravity wells” are:

* `serving/semantic/*`: the semantic query path compiles to SQL via Ibis and executes in DuckDB.
* `storage/views/*` and especially `storage/views/view_sql_map.json`: you have ~37 curated views represented as SQL strings plus tag metadata.

That view map is *exactly* the kind of artifact that becomes painful over time:

* hard to refactor safely
* hard to unit test at the expression level
* hard to share subexpressions
* encourages “just add another SQL view”

### 4) There’s still pandas in critical paths (validation)

In `storage/repositories/base.py`, you convert Arrow → pandas → Pandera validate → Arrow. That’s a triple tax:

* unnecessary materialization
* unnecessary copies
* pandas-only semantics bleeding into a columnar system

Given your stated goal, this is one of the highest ROI cleanups.

### 5) Many analytics modules run SQL and then do Python loops

Examples like `analytics/compute/coverage/compute.py`, `analytics/entrypoints/core.py`, and several dependency/graph extractors do:

* SQL query to fetch rows
* Python loop to aggregate

These are prime candidates to become:

* **Polars groupby/agg pipelines** (fast, maintainable)
* **PyArrow dataset scanner → batches** for streaming + controlled memory where needed

---

## Design goal

### Make this true:

1. **Arrow is the contract boundary** (schemas, batch interchange, serialization)
2. **Polars LazyFrame is the transformation IR** (instead of SQL)
3. **Parquet (Arrow dataset) is the primary persistent store**
4. **DuckDB becomes optional**, used only for:

   * full-text search indexing / specialized SQL-only features (if you keep them)
   * ad hoc debugging
   * temporary compatibility during migration

This aligns perfectly with the PyArrow “dataset control plane” idea: treat partitioned parquet + scanners as the core IO primitive and avoid accidental full-materialization (`scanner.to_table()` is explicitly the footgun).

---

## The big refactor: Arrow Dataset Store + Polars Compute

### A) Replace “Warehouse” with a `DatasetStore` (Parquet/Arrow-first)

**New core abstraction** (conceptually replaces `Warehouse`, most of `StorageGateway`, and large parts of `Repository`):

```python
@dataclass(frozen=True)
class SnapshotKey:
    repo: str
    commit: str

@dataclass(frozen=True)
class TableKey:
    namespace: str   # core, graph, analytics, docs, metadata
    name: str        # goids, modules, call_graph_edges, ...

class DatasetStore(Protocol):
    def scan(self, key: TableKey, snapshot: SnapshotKey | None, *,
             columns: list[str] | None = None,
             predicate: Any | None = None) -> pl.LazyFrame: ...

    def write(self, key: TableKey, snapshot: SnapshotKey, data: pl.LazyFrame | pa.Table, *,
              mode: Literal["overwrite_snapshot","append"]="overwrite_snapshot") -> None: ...

    def schema(self, key: TableKey) -> pa.Schema: ...
```

#### Storage layout recommendation

Use **Hive partitioning** so both PyArrow and Polars can prune efficiently:

```
lake/
  core/goids/repo=<repo>/commit=<commit>/part-*.parquet
  core/modules/repo=<repo>/commit=<commit>/part-*.parquet
  graph/call_graph_edges/repo=<repo>/commit=<commit>/part-*.parquet
  ...
```

PyArrow natively supports dataset partitioning and `write_dataset` with partitioning rules (including hive flavor).

#### Scanner-based reads (to avoid OOM)

Build scans as Arrow datasets + scanners, with:

* projection (columns)
* filter expressions
* batch sizing

This is exactly what `pyarrow.dataset.Scanner` is for, and it supports tuning `batch_size`, `batch_readahead`, and `fragment_readahead` for IO/perf tradeoffs.

Also: if you do use filters, `dataset.get_fragments(filter=...)` enables fragment pruning before scanning (key for large stores).

### B) Make Polars the primary “query engine” (no SQL strings)

#### Use lazy scans and native sinks

Your default posture should be:

* `pl.scan_parquet(...)` for reads
* Polars lazy transforms
* `sink_parquet(...)` for writes (instead of collecting into Python and writing manually)

Native sinks are explicitly recommended for performance (rather than collecting batches and writing yourself).

Also, Polars can inject row indices *at scan time* (important because adding a row index later can block predicate/projection pushdown).

### C) Stop persisting “views as SQL”; persist “views as Polars functions”

Right now, `view_sql_map.json` is essentially your semantic model.

**Replace it with a `ViewSpec` registry**:

```python
@dataclass(frozen=True)
class ViewSpec:
    name: str
    deps: tuple[TableKey | str, ...]       # base tables or other views
    tags: dict[str, Any]                   # keep your semantic tags!
    build: Callable[[DatasetStore, SnapshotKey], pl.LazyFrame]
```

Benefits:

* each view is unit-testable as a pure transformation
* you can share subexpressions in Python
* you don’t need SQLGlot lineage extraction for most cases (deps are explicit)
* you can materialize views to parquet as part of a build

### D) Replace semantic serving SQL (Ibis) with Polars plan compilation

The current flow in `serving/semantic/*` is:

`FilterSpec/QueryPlan` → Ibis expr → SQL → DuckDB → result conversion

**New flow**:

`FilterSpec/QueryPlan` → Polars Expr/LazyFrame plan → collect/sink → Arrow/Polars output

You already have a structured filter spec. Compile it to safe `pl.Expr` with an allowlist of operations/columns.

#### Bonus: better async behavior in FastAPI

Polars supports async collection (`collect_async`) and even a background collection handle (unstable API) which can fit better with serving workloads.

---

## Concrete, high-ROI code design changes (by subsystem)

### 1) Ingestion: replace DuckDB storage adapter with Parquet adapter

Current: `ingestion/adapters/duckdb_storage.py` clears snapshot via SQL deletes and inserts rows.

New: `ingestion/adapters/parquet_storage.py`:

* “clear snapshot” = delete `lake/<table>/repo=.../commit=.../`
* write new parquet files for each ingested table

Because your ingestion already behaves like “overwrite snapshot”, Parquet is a natural match.

If you still need schema drift handling across batches/files, unify Arrow schemas at the boundary (PyArrow’s `unify_schemas` supports strict-ish and permissive modes).

### 2) Build/Hamilton: introduce a Parquet materializer and make it default

You have `DuckDBRelationSaver` today.

Add:

* `PolarsParquetSaver` (preferred)
* `ArrowDatasetSaver` (if you want dataset-style multi-file output)

Use Polars native sinks where possible. (They’re built for exactly this.)

Also consider `collect_all()` when you need to materialize multiple outputs that share subplans (reduces duplicated scanning/compute).

### 3) Views: delete `view_sql_map.json` long term

Migration strategy:

* keep it temporarily for equivalence testing
* but shift the “source of truth” to Polars view specs
* optionally auto-generate a SQL view for debugging only (not production)

### 4) Serving semantic kernel: turn “tables” into LazyFrames

Implement a view/table loader:

```python
def lf(store: DatasetStore, key_or_view: str | TableKey, snap: SnapshotKey) -> pl.LazyFrame:
    ...
```

Then compilation is mechanical:

* `select` -> `.select([...])`
* `filter` -> `.filter(expr)`
* `order_by` -> `.sort([...])`
* `limit/offset` -> `.slice(offset, limit)`

And you can preserve your existing metadata tags:

* `semantic_default_order_by`
* `semantic_primary_key`
* `semantic_joins`
* exportability flags, etc.

### 5) Validation: remove pandas from the data plane

Replace:

Arrow → pandas → pandera → Arrow

With either:

* Arrow schema validation (fast, minimal dependencies)
* **Pandera-on-Polars** (if you want richer constraints) without pandas

Even if you keep Pandera, the key is: **validate `pl.DataFrame` or Arrow**, not pandas.

### 6) Analytics modules: replace “SQL + Python loop” with Polars aggregations

A big class of these can become:

* scan minimal columns
* groupby/agg
* (optional) join to enrich

This yields huge maintainability wins and typically performance wins.

#### Example rewrite pattern: range-join heavy SQL (coverage)

Your `analytics/compute/coverage/compute.py` does a join on `coverage.line BETWEEN goid.start_line AND goid.end_line`.

Instead of a literal range join, you can compute cumulative counts per file and do two **as-of joins** (end and start-1), then subtract. That reproduces the SQL semantics *without* exploding rows and stays in Polars.

That kind of rewrite is exactly the sort of “breaking but better” refactor you can now afford.

### 7) Graph extractors: load edge tables columnarly, not via SQL strings

Modules like `graphs/engine/views.py` can become:

* `pl.scan_parquet(edges_path).filter(repo/commit predicate).select(["src", "dst", "kind", ...])`
* `collect()` to a small in-memory DF only at the very end to feed NetworkX

This keeps most of the work (filter/projection) in a columnar engine.

---

## DuckDB: what to keep it for (and how to keep it safely)

Even in a Polars/PyArrow-first world, DuckDB can still be valuable for:

1. **Full-text search index**, if you want to keep your current approach.
2. **Compatibility mode** for a few queries you haven’t ported yet.

If you keep DuckDB as an auxiliary tool, prefer Arrow-based integration instead of SQL strings:

* replacement scans can consume `pyarrow.Dataset` / `pyarrow.Scanner` directly
* streaming results via `fetch_arrow_reader()` avoids materializing whole results at once

This keeps DuckDB at the boundary and prevents it from re-becoming the “center.”

---

## A pragmatic migration plan (fast, safe, and you can do it solo)

### Phase 1: Introduce the Parquet `DatasetStore` and dual-write

* Build still writes to DuckDB (so nothing breaks)
* Build also writes parquet datasets (new “lake”)
* Add schema validation and row-count checks between DuckDB tables and parquet tables

Polars `collect_all()` is useful when writing multiple artifacts from shared subplans (e.g., a view used in multiple exports).

### Phase 2: Switch read paths (serving + analytics) to parquet + polars

* semantic kernel now reads from store and compiles to Polars
* analytics modules read via Polars
* keep DuckDB only for search (if needed)

### Phase 3: Delete SQL view system + most Ibis/SQLGlot usage

* remove `storage/views/*` SQL materialization
* remove Ibis query builder path
* keep SQLGlot only if you still need parsing for user-provided SQL (otherwise remove)

### Phase 4: Clean contracts + validation

* codify Arrow schemas per table/view
* enforce at write time
* store schema metadata tags (contract/version) in Arrow schema metadata if you want stable provenance

---

## Polars-specific “advanced” practices worth adopting immediately

1. **Row index injection during scan**
   If you ever need stable row IDs, prefer scan-time row index creation (pushdown-friendly) instead of adding it later.

2. **Use native sinks instead of manual batch writing**
   For parquet outputs, `sink_parquet` is the right default (often faster and simpler than hand-rolled batch writers).

3. **Do not persist serialized LazyFrames**
   Polars serialization isn’t guaranteed stable across versions—persist parquet/IPC instead.

4. **Async collection for serving**
   If your FastAPI endpoints are async-heavy, consider `collect_async` to avoid blocking patterns.

---

## What I would change first (highest impact)

If you want the “80/20” ordering:

1. **Introduce Parquet DatasetStore** (this enables everything else)
2. **Replace semantic serving SQL path with Polars compiler**
3. **Replace SQL view registry with Polars `ViewSpec` registry**
4. **Replace ingestion storage adapter with parquet writer**
5. **Remove pandas from validation**
6. **Port the biggest SQL analytics queries next** (coverage, dependency extraction, hotspots)

This sequence rapidly moves the repo’s center of gravity away from DuckDB/SQL and toward Polars/PyArrow.

---

## If you want, I can go one level deeper

I can propose a concrete `DatasetStore` implementation (path layout, write modes, atomic snapshot swaps, schema enforcement) and a skeleton Polars semantic compiler that mirrors your existing `QueryPlan` / `FilterSpec` surface area—so you can start cutting code immediately.

But the core conclusion is: **your codebase is already half-way architected for Polars** (Hamilton tabular transforms, Arrow outputs, etc.). The biggest unlock is to stop treating DuckDB as the canonical store/engine and instead make **Parquet/Arrow datasets + Polars LazyFrame** the canonical reality.

Below are two “drop-in” building blocks you can paste into your repo right now:

1. a **concrete, file-backed `DatasetStore`** with:

* a clear **path layout**
* **write modes** (`replace`, `append`)
* **atomic snapshot publish** via `os.replace()`
* **schema enforcement** using your existing `TableSchema -> pa.Schema` renderer (`arrow_schema_from_table_schema`)

2. a **skeleton Polars semantic compiler** that consumes your existing:

* `SemanticQueryPlan` (your “QueryPlan” surface)
* `FilterSpec` (your filter surface)
  …and compiles them into a `polars.LazyFrame` (no SQL).

I’m writing these to match your code style (frozen dataclasses, `slots=True`, `__all__`, `split_table_key`, etc.), and to dovetail with how you already do atomic pointer swaps in `ServingSnapshotPointer`.

---

## 1) Concrete DatasetStore

### File layout (what this code implements)

Given a store root like:

```
<store_root>/
  current.json
  snapshots/
    <commit>/
      <run_id>/
        artifacts/
          schema_manifest.json
          semantic_registry.json
          buildspec.json
          ...
        datasets/
          <schema>/
            <table>/
              _schema.arrow
              _manifest.json
              part-000000.parquet
              part-000001.parquet
        _SUCCESS
```

Publishing a snapshot is **atomic**:

* write everything into a **staging dir**
* `os.replace(staging_dir, final_dir)`
* `os.replace(tmp_current.json, current.json)`

Readers that have already opened the old snapshot continue safely; new readers see the new pointer.

---

### `src/codeintel/storage/datasets/store.py`

```python
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from codeintel.config.primitives import SnapshotRef
from codeintel.core.schemas.arrow_gen import arrow_schema_from_table_schema
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.storage.helpers.table_key import split_table_key, validate_table_key

if TYPE_CHECKING:
    import polars as pl

WriteMode = Literal["replace", "append"]
SchemaEnforcementMode = Literal["strict", "coerce", "off"]


class DatasetStoreError(RuntimeError):
    """Raised when DatasetStore operations fail."""


@dataclass(frozen=True, slots=True)
class ParquetWriteOptions:
    """Centralized Parquet writing options (tune once, reuse everywhere)."""

    compression: str = "zstd"
    write_statistics: bool = True
    use_dictionary: bool = True
    row_group_size: int | None = None


@dataclass(frozen=True, slots=True)
class DatasetStorePointer:
    """Pointer to the currently active *file-backed* snapshot.

    This is the file-store analogue of ServingSnapshotPointer, designed to be
    updated atomically via os.replace().
    """

    snapshot_root: Path
    repo: str
    commit: str
    run_id: str
    published_at: datetime
    semantic_layer_version: str | None = None

    @classmethod
    def load(cls, path: Path) -> "DatasetStorePointer":
        raw = json.loads(path.read_text(encoding="utf-8"))
        published_at = datetime.fromisoformat(raw["published_at"])
        return cls(
            snapshot_root=Path(raw["snapshot_root"]).resolve(),
            repo=raw["repo"],
            commit=raw["commit"],
            run_id=raw["run_id"],
            published_at=published_at,
            semantic_layer_version=raw.get("semantic_layer_version"),
        )

    def to_json(self) -> str:
        payload = {
            "snapshot_root": str(self.snapshot_root),
            "repo": self.repo,
            "commit": self.commit,
            "run_id": self.run_id,
            "published_at": self.published_at.isoformat(),
            "semantic_layer_version": self.semantic_layer_version,
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_parquet(table: pa.Table, path: Path, *, options: ParquetWriteOptions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        table,
        tmp,
        compression=options.compression,
        use_dictionary=options.use_dictionary,
        write_statistics=options.write_statistics,
        row_group_size=options.row_group_size,
    )
    os.replace(tmp, path)


def _coerce_to_arrow_table(obj: object) -> pa.Table:
    # Keep this forgiving: allow pa.Table, polars DF, polars LF (collected).
    if isinstance(obj, pa.Table):
        return obj
    # Polars is optional at import-time for this module (only used if caller passes it).
    to_arrow = getattr(obj, "to_arrow", None)
    if callable(to_arrow):
        return to_arrow()
    collect = getattr(obj, "collect", None)
    if callable(collect):
        collected = collect()
        to_arrow2 = getattr(collected, "to_arrow", None)
        if callable(to_arrow2):
            return to_arrow2()
    raise TypeError(f"Unsupported dataset payload type: {type(obj)!r}")


def _enforce_schema(
    table: pa.Table,
    *,
    expected: pa.Schema,
    mode: SchemaEnforcementMode,
) -> pa.Table:
    if mode == "off":
        return table

    incoming_names = set(table.column_names)
    expected_names = [f.name for f in expected]
    expected_set = set(expected_names)

    missing = [name for name in expected_names if name not in incoming_names]
    extra = sorted(incoming_names - expected_set)

    if mode == "strict":
        if missing:
            raise DatasetStoreError(f"Missing required columns: {missing}")
        if extra:
            raise DatasetStoreError(f"Unexpected extra columns: {extra}")

    # coerce mode: drop extras, add missing nullable columns as nulls
    if mode == "coerce":
        if extra:
            table = table.select([c for c in table.column_names if c in expected_set])

        if missing:
            for name in missing:
                field = expected.field(name)
                if not field.nullable:
                    raise DatasetStoreError(
                        f"Missing non-nullable column in coerce mode: {name}"
                    )
                table = table.append_column(name, pa.nulls(table.num_rows, type=field.type))

    # Reorder + cast to expected schema (preserves your field metadata from expected)
    # Note: cast will raise if unsafe conversions occur.
    try:
        # Ensure columns are in expected order (and only expected columns).
        table = table.select(expected_names)
        table = table.cast(expected)
    except Exception as exc:
        raise DatasetStoreError(f"Failed to cast table to expected schema: {exc}") from exc

    return table


@dataclass(frozen=True, slots=True)
class DatasetStore:
    """A snapshotting Parquet dataset store with atomic publish."""

    root: Path
    parquet_options: ParquetWriteOptions = ParquetWriteOptions()

    def __post_init__(self) -> None:
        if not self.root.is_absolute():
            object.__setattr__(self, "root", self.root.resolve())
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def pointer_path(self) -> Path:
        return self.root / "current.json"

    @property
    def snapshots_dir(self) -> Path:
        return self.root / "snapshots"

    def load_pointer(self) -> DatasetStorePointer:
        if not self.pointer_path.exists():
            raise DatasetStoreError(f"DatasetStore pointer not found: {self.pointer_path}")
        return DatasetStorePointer.load(self.pointer_path)

    def current_snapshot_root(self) -> Path:
        return self.load_pointer().snapshot_root

    def snapshot_root(self, *, commit: str, run_id: str) -> Path:
        return self.snapshots_dir / commit / run_id

    def begin_snapshot(
        self,
        *,
        snapshot: SnapshotRef,
        run_id: str,
        semantic_layer_version: str | None = None,
        schema_enforcement: SchemaEnforcementMode = "strict",
    ) -> "SnapshotWriter":
        return SnapshotWriter(
            store=self,
            snapshot=snapshot,
            run_id=run_id,
            semantic_layer_version=semantic_layer_version,
            schema_enforcement=schema_enforcement,
        )

    def dataset_dir(
        self,
        *,
        table_key: str,
        snapshot_root: Path | None = None,
    ) -> Path:
        validate_table_key(table_key)
        schema, name = split_table_key(table_key)
        root = snapshot_root or self.current_snapshot_root()
        return root / "datasets" / schema / name

    def exists(self, *, table_key: str, snapshot_root: Path | None = None) -> bool:
        d = self.dataset_dir(table_key=table_key, snapshot_root=snapshot_root)
        return d.exists() and any(d.glob("part-*.parquet"))

    def scan_polars(self, *, table_key: str, snapshot_root: Path | None = None) -> "pl.LazyFrame":
        # Import locally so DatasetStore stays usable in non-polars contexts.
        import polars as pl

        d = self.dataset_dir(table_key=table_key, snapshot_root=snapshot_root)
        pattern = str(d / "part-*.parquet")
        return pl.scan_parquet(pattern)


@dataclass(slots=True)
class SnapshotWriter:
    """Write datasets + artifacts into a staged snapshot, then publish atomically."""

    store: DatasetStore
    snapshot: SnapshotRef
    run_id: str
    semantic_layer_version: str | None
    schema_enforcement: SchemaEnforcementMode = "strict"

    _stage_root: Path = field(init=False, repr=False)
    _final_root: Path = field(init=False, repr=False)
    _dataset_parts: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    _dataset_rows: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        final_root = self.store.snapshot_root(commit=self.snapshot.commit, run_id=self.run_id)
        stage_root = final_root.with_name(final_root.name + ".staging")
        self._final_root = final_root
        self._stage_root = stage_root

        if self._stage_root.exists():
            shutil.rmtree(self._stage_root)
        self._stage_root.mkdir(parents=True, exist_ok=True)

        (self._stage_root / "datasets").mkdir(parents=True, exist_ok=True)
        (self._stage_root / "artifacts").mkdir(parents=True, exist_ok=True)

    @property
    def stage_root(self) -> Path:
        return self._stage_root

    @property
    def final_root(self) -> Path:
        return self._final_root

    def write_artifact(self, *, name: str, content: str) -> Path:
        if not name or "/" in name or "\\" in name:
            raise DatasetStoreError(f"Invalid artifact name: {name!r}")
        path = self._stage_root / "artifacts" / name
        _atomic_write_text(path, content)
        return path

    def copy_artifact(self, *, name: str, src: Path) -> Path:
        if not src.exists():
            raise DatasetStoreError(f"Artifact source does not exist: {src}")
        if not name or "/" in name or "\\" in name:
            raise DatasetStoreError(f"Invalid artifact name: {name!r}")
        dst = self._stage_root / "artifacts" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return dst

    def write_dataset(
        self,
        *,
        contract: DatasetContract,
        data: object,  # pa.Table | pl.DataFrame | pl.LazyFrame
        mode: WriteMode = "replace",
        part_basename: str = "part",
    ) -> dict[str, object]:
        """Write a dataset according to its contract schema (Arrow metadata preserved)."""
        if contract.schema is None:
            raise DatasetStoreError(f"Cannot write view-backed contract as dataset: {contract.table_key}")

        table_key = contract.table_key
        validate_table_key(table_key)
        schema_name, table_name = split_table_key(table_key)

        expected_schema = arrow_schema_from_table_schema(table_schema=contract.schema)
        table = _coerce_to_arrow_table(data)
        table = _enforce_schema(table, expected=expected_schema, mode=self.schema_enforcement)

        ds_dir = self._stage_root / "datasets" / schema_name / table_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        # replace mode clears staged dataset dir
        if mode == "replace":
            for p in ds_dir.glob("part-*.parquet"):
                p.unlink(missing_ok=True)
            for p in (ds_dir / "_manifest.json", ds_dir / "_schema.arrow"):
                if p.exists():
                    p.unlink(missing_ok=True)
            self._dataset_parts[table_key] = []
            self._dataset_rows[table_key] = 0

        # Persist schema for fast introspection + enforcement on read if desired
        schema_path = ds_dir / "_schema.arrow"
        schema_bytes = expected_schema.serialize().to_pybytes()
        _atomic_write_text(schema_path, schema_bytes.decode("latin1"))  # simple byte round-trip
        # Note: latin1 encoding is a pragmatic way to round-trip bytes through text safely.
        # If you prefer, store as .bin and write bytes directly.

        # Pick next part index
        parts = self._dataset_parts.setdefault(table_key, [])
        idx = len(parts)
        filename = f"{part_basename}-{idx:06d}.parquet"
        out_path = ds_dir / filename

        _atomic_write_parquet(table, out_path, options=self.store.parquet_options)

        parts.append(filename)
        self._dataset_rows[table_key] = self._dataset_rows.get(table_key, 0) + table.num_rows

        manifest = {
            "table_key": table_key,
            "schema": schema_name,
            "name": table_name,
            "parts": list(parts),
            "rows": int(self._dataset_rows[table_key]),
            "written_at": datetime.now(tz=UTC).isoformat(),
            "schema_enforcement": self.schema_enforcement,
        }
        _atomic_write_text(ds_dir / "_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))

        return {"table_key": table_key, "rows_written": int(table.num_rows), "parts": list(parts)}

    def publish(self) -> DatasetStorePointer:
        """Atomically publish staged snapshot as the store's current snapshot."""
        # 1) finalize snapshot dir
        self._stage_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self._stage_root / "_SUCCESS", "ok\n")

        self._final_root.parent.mkdir(parents=True, exist_ok=True)
        if self._final_root.exists():
            raise DatasetStoreError(f"Final snapshot dir already exists: {self._final_root}")

        os.replace(self._stage_root, self._final_root)

        # 2) atomically swap pointer
        pointer = DatasetStorePointer(
            snapshot_root=self._final_root,
            repo=self.snapshot.repo,
            commit=self.snapshot.commit,
            run_id=self.run_id,
            published_at=datetime.now(tz=UTC),
            semantic_layer_version=self.semantic_layer_version,
        )
        _atomic_write_text(self.store.pointer_path, pointer.to_json() + "\n")
        return pointer

    def abort(self) -> None:
        """Best-effort cleanup of staged snapshot."""
        if self._stage_root.exists():
            shutil.rmtree(self._stage_root, ignore_errors=True)


__all__ = [
    "DatasetStore",
    "DatasetStoreError",
    "DatasetStorePointer",
    "ParquetWriteOptions",
    "SchemaEnforcementMode",
    "SnapshotWriter",
    "WriteMode",
]
```

**Notes you may want to tweak immediately**

* I used a pragmatic text round-trip for `_schema.arrow` (`latin1`) to keep the helper `_atomic_write_text` simple. If you prefer, change `_schema.arrow` to `_schema.bin` and write bytes directly with an atomic binary write helper.
* The writer currently writes one Parquet file per call. In practice you’ll call `write_dataset()` once per dataset per build step (replace mode), or multiple times if you want append-style incremental generation.

---

## 2) Skeleton Polars semantic compiler

This compiles your existing semantic plan surface:

* `SemanticQueryPlan` (from `codeintel.serving.semantic.query_builder`)
* `FilterSpec` (from `codeintel.serving.semantic.models`)
  …into a Polars `LazyFrame` that reads from the `DatasetStore` Parquet snapshot.

### `src/codeintel/serving/semantic/polars_compiler.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import polars as pl

from codeintel.core.schemas.primitives import ColumnType
from codeintel.serving.semantic.filter_ops import allowed_ops_for_column_type
from codeintel.serving.semantic.models import FilterSpec
from codeintel.serving.semantic.query_builder import SemanticQueryPlan
from codeintel.storage.datasets.store import DatasetStore

if TYPE_CHECKING:
    from collections.abc import Mapping


class PolarsQueryBuilderError(ValueError):
    """Raised when semantic Polars query construction fails."""


def _validate_pagination(*, limit: int, offset: int) -> None:
    if limit < 0:
        raise PolarsQueryBuilderError("limit must be >= 0")
    if offset < 0:
        raise PolarsQueryBuilderError("offset must be >= 0")


def _require_allowed_column(*, column: str, allowed_columns: frozenset[str], ctx: str) -> None:
    if column not in allowed_columns:
        raise PolarsQueryBuilderError(f"Unknown {ctx} column: {column}")


def _build_string_predicate(*, col: str, op: str, value: object) -> pl.Expr:
    if not isinstance(value, str):
        raise PolarsQueryBuilderError(f"{op} operator requires string value")
    if op == "contains":
        # literal=True => treat input as a literal substring, not a regex
        return pl.col(col).cast(pl.Utf8).str.contains(value, literal=True)
    if op == "startswith":
        return pl.col(col).cast(pl.Utf8).str.starts_with(value)
    raise PolarsQueryBuilderError(f"Unsupported string operator: {op}")


def _build_in_predicate(*, col: str, value: object, column_type: ColumnType | None) -> pl.Expr:
    if not isinstance(value, list):
        raise PolarsQueryBuilderError("IN operator requires list value")
    if column_type == "JSON":
        raise PolarsQueryBuilderError("IN operator is not supported for JSON columns")

    # Polars will handle list/Series membership; for very large IN lists you can
    # later swap this to a semi-join strategy if needed.
    return pl.col(col).is_in(value)


def _build_simple_predicate(*, col: str, op: str, value: object, column_type: ColumnType | None) -> pl.Expr:
    if op in {"lt", "lte", "gt", "gte"} and column_type == "VARCHAR":
        raise PolarsQueryBuilderError(f"Operator {op} is not supported for string columns")

    left = pl.col(col)
    if op == "eq":
        return left == value
    if op == "ne":
        return left != value
    if op == "lt":
        return left < value
    if op == "lte":
        return left <= value
    if op == "gt":
        return left > value
    if op == "gte":
        return left >= value

    raise PolarsQueryBuilderError(f"Unsupported operator: {op}")


def _build_predicate(
    *,
    filter_spec: FilterSpec,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> pl.Expr:
    _require_allowed_column(column=filter_spec.column, allowed_columns=allowed_columns, ctx="filter")

    col = filter_spec.column
    op = filter_spec.op
    value = filter_spec.value

    column_type = None
    if column_types is not None:
        column_type = column_types.get(col)

    allowed_ops = allowed_ops_for_column_type(column_type)
    if op not in allowed_ops:
        raise PolarsQueryBuilderError(
            f"Operator {op} is not supported for column type {column_type or 'UNKNOWN'}"
        )

    if op in {"contains", "startswith"}:
        return _build_string_predicate(col=col, op=op, value=value)

    if op == "in":
        return _build_in_predicate(col=col, value=value, column_type=column_type)

    return _build_simple_predicate(col=col, op=op, value=value, column_type=column_type)


def _build_sort_spec(*, order_by: list[str], allowed_columns: frozenset[str]) -> tuple[list[str], list[bool]]:
    cols: list[str] = []
    desc: list[bool] = []
    for item in order_by:
        descending = item.startswith("-")
        col = item[1:] if descending else item
        _require_allowed_column(column=col, allowed_columns=allowed_columns, ctx="order_by")
        cols.append(col)
        desc.append(descending)
    return cols, desc


@dataclass(frozen=True, slots=True)
class PolarsCompiledQuery:
    """Compiled Polars query artifact (rough analogue to BoundQuery for SQL)."""

    lazyframe: pl.LazyFrame

    def explain(self, *, optimized: bool = True) -> str:
        # Polars supports explain() on LazyFrame
        return self.lazyframe.explain(optimized=optimized)


def build_polars_query(
    *,
    store: DatasetStore,
    plan: SemanticQueryPlan,
    column_types: Mapping[str, ColumnType] | None = None,
    snapshot_root: "pl.Expr | None" = None,  # unused placeholder to show extension seam
) -> PolarsCompiledQuery:
    """Compile a SemanticQueryPlan into a Polars LazyFrame (no SQL)."""
    _validate_pagination(limit=plan.limit, offset=plan.offset)

    # Validate select columns
    for col in plan.columns:
        if col not in plan.allowed_columns:
            raise PolarsQueryBuilderError(f"Unknown column: {col}")

    lf = store.scan_polars(table_key=plan.table_key)

    # Filters
    if plan.filters:
        preds = [
            _build_predicate(
                filter_spec=f,
                allowed_columns=plan.allowed_columns,
                column_types=column_types,
            )
            for f in plan.filters
        ]
        combined = preds[0]
        for p in preds[1:]:
            combined = combined & p
        lf = lf.filter(combined)

    # Order
    if plan.order_by:
        cols, desc = _build_sort_spec(order_by=plan.order_by, allowed_columns=plan.allowed_columns)
        lf = lf.sort(by=cols, descending=desc)

    # Project
    lf = lf.select([pl.col(c) for c in plan.columns])

    # Pagination (offset+limit)
    lf = lf.slice(plan.offset, plan.limit)

    return PolarsCompiledQuery(lazyframe=lf)


__all__ = [
    "PolarsCompiledQuery",
    "PolarsQueryBuilderError",
    "build_polars_query",
]
```

### Why this mirrors your existing surface area cleanly

* Same `SemanticQueryPlan` dataclass
* Same `FilterSpec` model and `allowed_ops_for_column_type()` rules
* Same semantics for `order_by`, `limit`, `offset`
* Produces a reusable compiled artifact (`PolarsCompiledQuery`) analogous to “compiled SQL + temp tables”, except the artifact is a `LazyFrame`

---

## Minimal integration sketch in ServingKernel (so you can start cutting immediately)

Once you have Parquet snapshots being published, you can wire serving to Polars with a *small* executor swap:

```python
# inside codeintel/serving/semantic/kernel.py (conceptual patch)

from codeintel.serving.semantic.polars_compiler import build_polars_query
from codeintel.storage.datasets.store import DatasetStore

# in __init__ of ServingSemanticKernel:
self._dataset_store = DatasetStore(root=self.settings.dataset_store_root)

def _execute_semantic_plan(...):
    compiled = build_polars_query(
        store=self._dataset_store,
        plan=plan,
        column_types=column_types,
    )
    df = compiled.lazyframe.collect()
    rows = df.to_dicts()
    # compiled_sql no longer exists; you can return compiled.explain() or None
    return rows, compiled.explain()
```

You’ll likely also update your response model fields:

* replace `sql_fingerprint` with `plan_fingerprint` (hash of `compiled.explain()`), or keep `sql_fingerprint=None`.

---

## References


