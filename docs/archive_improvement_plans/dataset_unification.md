
# Context, high level architecture, and high level implementation details #

---

## 0. Context and Problem Statement

The current CodeIntel storage + export architecture has grown powerful but brittle because **dataset metadata is scattered across multiple modules**:

* **Logical schemas**: `config/schemas/tables.py` (`TABLE_SCHEMAS`, `TableSchema`)
* **Ingestion + SQL column lists**: `config/schemas/ingestion_sql.py`, `config/schemas/sql_builder.py`
* **Row shapes + serializers**: `storage/rows.py` (TypedDicts + `*_to_tuple` functions)
* **Docs views and view SQL**: `storage/views.py` (all `docs.v_*` definitions)
* **Dataset → filename mappings**: `pipeline/export/datasets.py` (`JSONL_DATASETS`, `PARQUET_DATASETS`)
* **Dataset registries and boundaries**: `storage/gateway.py`, `serving/http/datasets.py`, `serving/mcp/query_service.py`

Any time a dataset is introduced or renamed (e.g. `analytics.graph_metrics_modules_ext`), we typically have to touch:

* `TABLE_SCHEMAS`
* one or more ingestion SQL helpers
* `storage/rows.py`
* export filename maps
* view definitions
* and sometimes query-service code

This multiplies the risk of drift and makes “what datasets exist?” or “how do I export X?” harder to answer—both for humans and for LLM agents.

At the same time, DuckDB is already the *center* of our analytics universe but is underused as:

* A **catalog** of datasets and their properties
* The **canonical home** of common operations (e.g. “fetch rows for dataset X”)

We want to fix this by moving to a **DuckDB-centric dataset architecture** while preserving our Python types, docstrings, and architecture tests.

---

## 1. High-Level Objectives

### 1.1. Primary goals

1. **Single source of truth for datasets**

   * For each dataset, have *one* canonical mapping of:

     * `table_key` (DuckDB table/view, e.g. `"analytics.function_profile"` or `"docs.v_function_profile"`)
     * `name` (short dataset name, e.g. `"function_profile"`)
     * `is_view` (base table vs docs view)
     * Optional default filenames (`*.jsonl`, `*.parquet`)

2. **DuckDB as the catalog and operations engine**

   * A **catalog table** in DuckDB (`metadata.datasets`) contains dataset-level metadata.
   * Table macros (e.g. `metadata.dataset_rows`) provide canonical operations *inside* DuckDB.

3. **Python as a typed mirror, not the source of truth**

   * A Python `Dataset` model and registry (`storage/datasets.py`) mirrors the DB catalog.
   * Existing `TABLE_SCHEMAS` and `storage/rows` remain, and Python continues to enforce types, docstrings, and architecture rules—but now *layered* on top of DuckDB metadata.

4. **Exports as thin wrappers**

   * `pipeline/export/export_jsonl.py` and `export_parquet.py` become thin wrappers that:

     * Ask the **gateway** for dataset metadata (from DuckDB),
     * Use DuckDB relational / `COPY` APIs for actual I/O,
     * Write manifests driven by the same dataset catalog.

5. **Safer, more agent-friendly public surface**

   * Query and export entrypoints can say “give me dataset `function_profile`” instead of “run this arbitrary SQL”.
   * It’s trivial to introspect all datasets (and their schemas, filenames) from **inside DuckDB** and from Python.

### 1.2. Non-goals / out of scope (for this change set)

* We are **not** redesigning the content of individual tables (schemas, columns) beyond what’s needed to wire metadata.
* We are **not** rewriting all ingestion paths; they will adopt the registry incrementally.
* We are **not** changing the semantics of existing views (`docs.v_*`), just how they are discovered and exported.
* We are **not** yet migrating everything to DuckDB `COPY`—we’ll keep existing relation-based writes where they’re stable, and can later consolidate them.

---

## 2. Target Architecture: Components and Responsibilities

### 2.1. DuckDB side

**Schemas & tables:**

* `core.*`, `graph.*`, `analytics.*` – all existing base tables from `TABLE_SCHEMAS`.
* `docs.*` – all derived views defined in `storage/views.py` (unchanged content).

**New metadata schema:**

* `metadata.datasets` (new table):

  ```sql
  CREATE SCHEMA IF NOT EXISTS metadata;

  CREATE TABLE IF NOT EXISTS metadata.datasets (
      table_key        TEXT PRIMARY KEY,  -- e.g. 'analytics.function_profile'
      name             TEXT NOT NULL,     -- e.g. 'function_profile'
      is_view          BOOLEAN NOT NULL,  -- TRUE for docs.* views, FALSE for base tables
      jsonl_filename   TEXT,              -- optional default JSONL filename
      parquet_filename TEXT               -- optional default Parquet filename
  );
  ```

  * This is the canonical catalog of datasets.
  * Populated by a Python bootstrap helper using existing metadata. 

**New table macro(s):**

* `metadata.dataset_rows(table_key TEXT, row_limit BIGINT := 100, row_offset BIGINT := 0)`:

  ```sql
  CREATE OR REPLACE MACRO metadata.dataset_rows(
      table_key  TEXT,
      row_limit  BIGINT := 100,
      row_offset BIGINT := 0
  ) AS TABLE
  SELECT *
  FROM query_table(table_key)
  LIMIT row_limit OFFSET row_offset;
  ```

  * Python resolves `dataset_name -> table_key` via the registry and passes `table_key` into the macro.
  * This keeps the macro free of lateral joins/subqueries while still working for tables and `docs.*` views.

---

### 2.2. Python side: metadata bootstrap, registry, and gateway

**Bootstrap helper** (`storage/metadata_bootstrap.py`):

* Responsibilities:

  * Ensure `metadata` schema and `metadata.datasets` table exist.
  * Apply `metadata.dataset_rows` macro.
  * Populate or refresh `metadata.datasets` from:

    * `TABLE_SCHEMAS` (all known base tables),
    * `DOCS_VIEWS` (all docs views),
    * previous JSONL/PARQUET mapping (from `pipeline/export/datasets.py`).

* Core function:

  ```python
  def bootstrap_metadata_datasets(con: DuckDBPyConnection) -> None:
      apply_metadata_ddl(con)
      # upsert base tables from TABLE_SCHEMAS
      # upsert docs views from DOCS_VIEWS
  ```

  * Uses `INSERT ... ON CONFLICT(table_key) DO UPDATE` so it is idempotent and safe to run multiple times. 

**Dataset mirror + registry** (`storage/datasets.py`):

* Defines:

  ```python
  @dataclass(frozen=True)
  class Dataset:
      table_key: str                   # 'analytics.function_profile'
      name: str                        # 'function_profile'
      schema: TableSchema | None       # None for views
      row_binding: RowBinding | None   # TypedDict + to_tuple
      jsonl_filename: str | None
      parquet_filename: str | None
      is_view: bool
  ```

* Row bindings are Python-only:

  ```python
  ROW_BINDINGS_BY_TABLE_KEY = {
      "analytics.coverage_lines": RowBinding(
          row_type=CoverageLineRow,
          to_tuple=coverage_line_to_tuple,
      ),
      ...
  }
  ```

* `load_dataset_registry(con)` reads `metadata.datasets` and returns:

  ```python
  @dataclass(frozen=True)
  class DatasetRegistry:
      by_name: Mapping[str, Dataset]          # 'function_profile' -> Dataset
      by_table_key: Mapping[str, Dataset]     # 'analytics.function_profile' -> Dataset
      jsonl_datasets: Mapping[str, str]       # table_key -> jsonl filename
      parquet_datasets: Mapping[str, str]     # table_key -> parquet filename
  ```

  * `schema` is set from `TABLE_SCHEMAS` for base tables when available.
  * If a base table is listed in `metadata.datasets` but missing from `TABLE_SCHEMAS`, we set `schema=None` and rely on existing schema-alignment checks to catch drift.

**Storage gateway integration** (`storage/gateway.py`):

* `open_gateway(config)` now:

  1. Connects to DuckDB (`_connect(config)`, which already applies schemas and views).

  2. Calls `bootstrap_metadata_datasets(con)` for non-read-only configs.

  3. Calls `build_dataset_registry(con)` which wraps `load_dataset_registry(con)` into a small gateway-specific `DatasetRegistry`:

     ```python
     @dataclass(frozen=True)
     class GatewayDatasetRegistry:
         mapping: Mapping[str, str]        # dataset_name -> table_key
         tables: tuple[str, ...]
         views: tuple[str, ...]
         meta: Mapping[str, Dataset] | None
         jsonl_mapping: Mapping[str, str] | None
         parquet_mapping: Mapping[str, str] | None
     ```

  4. Returns `_DuckDBGateway(config=config, datasets=registry, con=con)`.

* This registry is what downstream layers use to resolve dataset names → tables, and to discover filenames for exports.

---

### 2.3. Serving and export layers

**Serving (MCP / HTTP) uses the macro:**

* `serving/mcp/query_service.DuckDBQueryService.read_dataset_rows` changes from “inspect table, special-case docs views, apply limits in Python” to:

  ```python
  table_key = self.gateway.datasets.mapping.get(dataset_name)
  if table_key is None:
      raise errors.invalid_argument(...)

  relation = self.con.sql(
      "SELECT * FROM metadata.dataset_rows(?, ?, ?)",
      [table_key, applied_limit, applied_offset],
  )
  rows = relation.fetchall()
  cols = [desc[0] for desc in relation.description]
  ...
  ```

* This allows us to delete `DOCS_VIEW_QUERIES` and other special casing: “dataset” is now a DuckDB concept.

**Export pipeline uses dataset registry:**

* `pipeline/export/export_jsonl.py` and `export_parquet.py`:

  * No longer import `pipeline/export/datasets.JSONL_DATASETS` / `PARQUET_DATASETS`.

  * Instead, they ask the gateway:

    ```python
    dataset_mapping = gateway.datasets.mapping          # dataset_name -> table_key
    jsonl_mapping = gateway.datasets.jsonl_mapping or {}
    parquet_mapping = gateway.datasets.parquet_mapping or {}
    ```

  * Selection logic:

    ```python
    selected = {
        name: table
        for name, table in dataset_mapping.items()
        if table in jsonl_mapping  # or parquet_mapping
    } if datasets is None else { ... resolve explicitly ... }
    ```

  * Export each dataset by:

    ```python
    filename = jsonl_mapping.get(table_name, f"{dataset_name}.jsonl")
    path = output_dir / filename
    export_jsonl_for_table(gateway, table_name, path)
    ```

  * Manifest generation still calls `write_dataset_manifest(…, jsonl_mapping, parquet_mapping, selected=[...])`, except the mapping now comes **from DuckDB via the registry**, not from static dicts in Python.

---

## 3. Key Behavioral Guarantees

1. **Single catalog for all dataset-aware features**

   * Anything that needs to know “what datasets exist?” or “how do I export dataset X?” goes through `metadata.datasets` → `storage.datasets` → `StorageGateway.datasets`.

2. **Views and tables treated uniformly**

   * The same registry handles both base tables (`core.*`, `analytics.*`, `graph.*`) and `docs.*` views.
   * The dataset macro uses `query_table(table_key)` so views require no special logic in Python.

3. **Incremental, backward-compatible rollout**

   * We keep `TABLE_SCHEMAS` and `storage/views` as they are; `bootstrap_metadata_datasets` is additive.
   * Existing architecture tests and `http/datasets` helpers remain valid but can be slowly refocused to assert consistency with the DuckDB-backed registry.

4. **Drift detection**

   * If `TABLE_SCHEMAS` and `metadata.datasets` diverge (e.g. table added in DuckDB without a `TableSchema`), we’ll see:

     * `Dataset.schema` set to `None`, which can be caught by tests.
     * Potential failures in schema-alignment tests that compare DuckDB `information_schema` to `TABLE_SCHEMAS`.

5. **LLM / agent friendliness**

   * The dataset catalog is small, explicit, and introspectable from both Python and DuckDB.
   * Query-service entrypoints operate in terms of dataset names and slices, not arbitrary SQL snippets.

---

## 4. Implementation Notes and Edge Cases (filling in gaps)

### 4.1. When and how bootstrap runs

* **Non-read-only** gateways:

  * `open_gateway(config)` runs `bootstrap_metadata_datasets(con)` on every initialization.
  * This is safe: the bootstrap uses idempotent upserts.

* **Read-only** gateways:

  * `bootstrap_metadata_datasets` is **not** called when `config.read_only` is `True`.
  * Assumption: the DB already has up-to-date `metadata.datasets` (e.g. written by an earlier write-capable run).

### 4.2. Handling of missing schema entries

* When reading registry:

  ```python
  if not ds.is_view:
      schema = TABLE_SCHEMAS.get(table_key)
      if schema is None:
          # drift: table is in metadata.datasets but not in TABLE_SCHEMAS
          schema = None
  ```

* Impact:

  * Dataset is still visible to exports and query-service (because DuckDB knows about it).
  * But schema-aware operations that rely on `TableSchema` (e.g. some SQL builders) will see `schema=None` and can either:

    * Skip that dataset, or
    * Raise a descriptive error.

### 4.3. Adding a new dataset going forward

For the LLM agent, the “happy path” to add a new dataset is:

1. Add its **schema** in `config/schemas/tables.py` (`TABLE_SCHEMAS`).
2. Add row model & serializer (if needed) in `storage/rows.py` and register it in `ROW_BINDINGS_BY_TABLE_KEY`.
3. Optionally, assign JSONL/PARQUET filenames in `bootstrap_metadata_datasets` (or via a dedicated CLI to maintain `metadata.datasets`).
4. Run CodeIntel with a write-capable gateway once; `bootstrap_metadata_datasets` will insert/update the row.
5. All other layers (serving, export) see it automatically via the registry.

### 4.4. Tests to add around this

To keep things tight:

* **Metadata roundtrip test**:

  * Create a transient DB, run `_connect`, `create_all_views`, `bootstrap_metadata_datasets`.
  * Call `load_dataset_registry`.
  * Assert:

    * For a known dataset (e.g. `function_profile`), `Dataset.table_key` matches a row in `metadata.datasets`.
    * `jsonl_datasets` / `parquet_datasets` contain the expected table keys and filenames.

* **Export coverage test**:

  * Insert a row into `metadata.datasets` with non-null filenames for a fake table and ensure `export_all_jsonl` picks it up (or, if using real tables, assert that actual tables are exported).

* **Drift detection test**:

  * Temporarily add a fake row in `metadata.datasets` for a non-existent table, reload registry, and assert that:

    * `Dataset.schema is None`,
    * Tests that assert alignment between `TABLE_SCHEMAS` and DuckDB information_schema fail as expected.

---


# Detailed code #


Let’s make this concrete.

I’ll give you three drop-in pieces:

1. **SQL DDL** for `metadata.datasets` + a DuckDB table macro `metadata.dataset_rows`.
2. A **Python bootstrap helper** that fills `metadata.datasets` using your existing `TABLE_SCHEMAS`, `JSONL_DATASETS`, and `DOCS_VIEWS`.
3. An updated **`storage/datasets.py`** that *reads from DuckDB*, and a small patch to `DuckDBQueryService.read_dataset_rows` to use the new macro.

Everything below is aligned to the current repo layout you’ve got under `config/`, `storage/`, `pipeline/`, and `serving/` as extracted from the zip bundles.

---

## 1. DuckDB DDL: `metadata.datasets` + `metadata.dataset_rows` macro

You can keep this as a `.sql` file or embed the strings in Python and run them via `apply_all_schemas(extra_ddl=...)`.

### 1.1. Schema + table

```sql
-- 1. Ensure the metadata schema exists
CREATE SCHEMA IF NOT EXISTS metadata;

-- 2. Central dataset catalog
CREATE TABLE IF NOT EXISTS metadata.datasets (
    table_key        TEXT PRIMARY KEY,  -- e.g. 'analytics.function_profile'
    name             TEXT NOT NULL,     -- e.g. 'function_profile'
    is_view          BOOLEAN NOT NULL,  -- TRUE for docs.* views, FALSE for base tables
    jsonl_filename   TEXT,              -- e.g. 'function_profile.jsonl'
    parquet_filename TEXT               -- e.g. 'function_profile.parquet'
);
```

That’s the DB-side mirror of what you previously had in Python as:

* `table_key`, `name`, `is_view` from `TABLE_SCHEMAS` + `DOCS_VIEWS`
* filenames from `pipeline/export/datasets.JSONL_DATASETS` / `PARQUET_DATASETS`.

### 1.2. Table macro: `metadata.dataset_rows`

This macro is your canonical “read a dataset by table key” operation inside DuckDB:

```sql
CREATE OR REPLACE MACRO metadata.dataset_rows(
    table_key  TEXT,
    row_limit  BIGINT := 100,
    row_offset BIGINT := 0
) AS TABLE
SELECT *
FROM query_table(table_key)
LIMIT row_limit OFFSET row_offset;
```

Key points:

* Python resolves `dataset_name` → `table_key` via `metadata.datasets` and passes `table_key` into the macro.
* It calls `query_table(table_key)` so it works for both **tables and views**, including your `docs.*` views (no special-casing in Python).
* It accepts `limit` and `offset` so you can line it up with your `BackendLimits` clamping logic.

You can optionally also define a scalar helper for “does this dataset exist?”:

```sql
CREATE OR REPLACE MACRO metadata.dataset_exists(ds_name TEXT) AS
    EXISTS (
        SELECT 1
        FROM metadata.datasets d
        WHERE d.name = ds_name
    );
```

You don’t *have* to use that (you can still check via Python), but it’s nice for DB-side validations.

---

## 2. Python bootstrap helper: populate `metadata.datasets`

You only need this for migration / synchronization: given your existing Python mappings, populate or refresh the DuckDB table.

A good home for this is a small module like `storage/metadata_bootstrap.py` (or `storage/datasets_bootstrap.py`).

```python
# src/codeintel/storage/metadata_bootstrap.py

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

from duckdb import DuckDBPyConnection

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.pipeline.export.datasets import JSONL_DATASETS, PARQUET_DATASETS
from codeintel.storage.gateway import DOCS_VIEWS


METADATA_SCHEMA_DDL: tuple[str, ...] = (
    # schema + table
    """
    CREATE SCHEMA IF NOT EXISTS metadata;
    """,
    """
    CREATE TABLE IF NOT EXISTS metadata.datasets (
        table_key        TEXT PRIMARY KEY,
        name             TEXT NOT NULL,
        is_view          BOOLEAN NOT NULL,
        jsonl_filename   TEXT,
        parquet_filename TEXT
    );
    """,
    # dataset_rows macro
    """
    CREATE OR REPLACE MACRO metadata.dataset_rows(
        table_key  TEXT,
        row_limit  BIGINT := 100,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT *
    FROM query_table(table_key)
    LIMIT row_limit OFFSET row_offset;
    """,
)


def apply_metadata_ddl(con: DuckDBPyConnection) -> None:
    """Create metadata schema, table, and macros."""
    for stmt in METADATA_SCHEMA_DDL:
        con.execute(stmt)


def _upsert_dataset_row(
    con: DuckDBPyConnection,
    *,
    table_key: str,
    name: str,
    is_view: bool,
    jsonl_filename: str | None,
    parquet_filename: str | None,
) -> None:
    con.execute(
        """
        INSERT INTO metadata.datasets (table_key, name, is_view, jsonl_filename, parquet_filename)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(table_key) DO UPDATE SET
            name             = excluded.name,
            is_view          = excluded.is_view,
            jsonl_filename   = excluded.jsonl_filename,
            parquet_filename = excluded.parquet_filename;
        """,
        [table_key, name, is_view, jsonl_filename, parquet_filename],
    )


def bootstrap_metadata_datasets(con: DuckDBPyConnection) -> None:
    """
    Populate metadata.datasets from TABLE_SCHEMAS, JSONL_DATASETS, PARQUET_DATASETS, and DOCS_VIEWS.

    Safe to run repeatedly; uses INSERT ... ON CONFLICT for idempotent upserts.
    """
    apply_metadata_ddl(con)

    # Base tables from TABLE_SCHEMAS
    for table_key in sorted(TABLE_SCHEMAS.keys()):
        # table_key like "analytics.function_profile" or "graph.call_graph_edges"
        _, name = table_key.split(".", maxsplit=1)
        jsonl = JSONL_DATASETS.get(table_key)
        parquet = PARQUET_DATASETS.get(table_key)
        _upsert_dataset_row(
            con,
            table_key=table_key,
            name=name,
            is_view=False,
            jsonl_filename=jsonl,
            parquet_filename=parquet,
        )

    # docs.* views (may overlap with TABLE_SCHEMAS by name, but we mark as views)
    for view_key in DOCS_VIEWS:
        _, name = view_key.split(".", maxsplit=1)
        jsonl = JSONL_DATASETS.get(view_key)
        parquet = PARQUET_DATASETS.get(view_key)
        _upsert_dataset_row(
            con,
            table_key=view_key,
            name=name,
            is_view=True,
            jsonl_filename=jsonl,
            parquet_filename=parquet,
        )
```

Usage patterns:

* From your **gateway** initialization:

  ```python
  from codeintel.storage.metadata_bootstrap import bootstrap_metadata_datasets

  class StorageGateway:
      def __init__(...):
          ...
          apply_all_schemas(self.con)    # existing
          create_all_views(self.con)     # existing
          bootstrap_metadata_datasets(self.con)  # new
          ...
  ```

* Or from a **CLI** command (e.g. `codeintel storage init-metadata`) if you want explicit control.

---

## 3. Updated `storage/datasets.py`: read from DuckDB

Now we let DuckDB be the source of truth for:

* `table_key`, `name`, `is_view`
* `jsonl_filename`, `parquet_filename`

and we only layer **Python-only** concerns on top:

* Row bindings (TypedDict + `*_to_tuple`).
* `TableSchema` for base tables.

Here is an updated `src/codeintel/storage/datasets.py` that does exactly that.

```python
"""Dataset metadata registry backed by DuckDB's metadata.datasets table.

This module mirrors the metadata.datasets catalog table into typed Python
structures so that storage, serving, and export layers share a single source of
truth while still enjoying type-checking and helpers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Type, TypedDict

from duckdb import DuckDBPyConnection

from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema
from codeintel.storage import rows as row_models


RowToTuple = Callable[[object], tuple[object, ...]]
RowDictType = Type[TypedDict]


@dataclass(frozen=True)
class RowBinding:
    """Connect a DuckDB table key to a TypedDict row model and its serializer."""

    row_type: RowDictType
    to_tuple: RowToTuple


@dataclass(frozen=True)
class Dataset:
    """Metadata describing a logical dataset backed by a DuckDB table or view."""

    # Fully qualified DuckDB key, e.g. "analytics.function_metrics" or
    # "docs.v_function_profile".
    table_key: str

    # Short dataset name exposed to callers (e.g. "function_metrics",
    # "v_function_profile").
    name: str

    # Schema for base tables; None for views.
    schema: TableSchema | None

    # Optional row binding for inserts/updates.
    row_binding: RowBinding | None = None

    # Default filenames for Document Output exports, when applicable.
    jsonl_filename: str | None = None
    parquet_filename: str | None = None

    # True when this dataset is a docs.* view instead of a base table.
    is_view: bool = False


@dataclass(frozen=True)
class DatasetRegistry:
    """In-memory view of metadata.datasets plus Python row bindings."""

    by_name: Mapping[str, Dataset]
    by_table_key: Mapping[str, Dataset]
    jsonl_datasets: Mapping[str, str]     # table_key -> filename
    parquet_datasets: Mapping[str, str]   # table_key -> filename

    @property
    def all_datasets(self) -> tuple[str, ...]:
        """Return all dataset names."""
        return tuple(self.by_name.keys())

    def resolve_table_key(self, name: str) -> str:
        """Resolve dataset name into fully qualified table/view key."""
        ds = self.by_name.get(name)
        if ds is None:
            message = f"Unknown dataset: {name}"
            raise KeyError(message)
        return ds.table_key


# ---------------------------------------------------------------------------
# Row bindings: TypedDict + row->tuple helpers from storage.rows
# ---------------------------------------------------------------------------

ROW_BINDINGS_BY_TABLE_KEY: dict[str, RowBinding] = {
    # analytics.*
    "analytics.coverage_lines": RowBinding(
        row_type=row_models.CoverageLineRow,
        to_tuple=row_models.coverage_line_to_tuple,
    ),
    "analytics.config_values": RowBinding(
        row_type=row_models.ConfigValueRow,
        to_tuple=row_models.config_value_to_tuple,
    ),
    "analytics.typedness": RowBinding(
        row_type=row_models.TypednessRow,
        to_tuple=row_models.typedness_row_to_tuple,
    ),
    "analytics.static_diagnostics": RowBinding(
        row_type=row_models.StaticDiagnosticRow,
        to_tuple=row_models.static_diagnostic_to_tuple,
    ),
    "analytics.function_validation": RowBinding(
        row_type=row_models.FunctionValidationRow,
        to_tuple=row_models.function_validation_row_to_tuple,
    ),
    "analytics.graph_validation": RowBinding(
        row_type=row_models.GraphValidationRow,
        to_tuple=row_models.graph_validation_to_tuple,
    ),
    "analytics.hotspots": RowBinding(
        row_type=row_models.HotspotRow,
        to_tuple=row_models.hotspot_row_to_tuple,
    ),
    "analytics.test_catalog": RowBinding(
        row_type=row_models.TestCatalogRowModel,
        to_tuple=row_models.test_catalog_row_to_tuple,
    ),
    "analytics.test_coverage_edges": RowBinding(
        row_type=row_models.TestCoverageEdgeRow,
        to_tuple=row_models.test_coverage_edge_to_tuple,
    ),
    # core.*
    "core.docstrings": RowBinding(
        row_type=row_models.DocstringRow,
        to_tuple=row_models.docstring_row_to_tuple,
    ),
    "core.goids": RowBinding(
        row_type=row_models.GoidRow,
        to_tuple=row_models.goid_to_tuple,
    ),
    "core.goid_crosswalk": RowBinding(
        row_type=row_models.GoidCrosswalkRow,
        to_tuple=row_models.goid_crosswalk_to_tuple,
    ),
    # graph.*
    "graph.call_graph_nodes": RowBinding(
        row_type=row_models.CallGraphNodeRow,
        to_tuple=row_models.call_graph_node_to_tuple,
    ),
    "graph.call_graph_edges": RowBinding(
        row_type=row_models.CallGraphEdgeRow,
        to_tuple=row_models.call_graph_edge_to_tuple,
    ),
    "graph.import_graph_edges": RowBinding(
        row_type=row_models.ImportEdgeRow,
        to_tuple=row_models.import_edge_to_tuple,
    ),
    "graph.import_modules": RowBinding(
        row_type=row_models.ImportModuleRow,
        to_tuple=row_models.import_module_to_tuple,
    ),
    "graph.cfg_blocks": RowBinding(
        row_type=row_models.CFGBlockRow,
        to_tuple=row_models.cfg_block_to_tuple,
    ),
    "graph.cfg_edges": RowBinding(
        row_type=row_models.CFGEdgeRow,
        to_tuple=row_models.cfg_edge_to_tuple,
    ),
    "graph.dfg_edges": RowBinding(
        row_type=row_models.DFGEdgeRow,
        to_tuple=row_models.dfg_edge_to_tuple,
    ),
    "graph.symbol_use_edges": RowBinding(
        row_type=row_models.SymbolUseRow,
        to_tuple=row_models.symbol_use_to_tuple,
    ),
}


# ---------------------------------------------------------------------------
# Registry construction from metadata.datasets
# ---------------------------------------------------------------------------

def load_dataset_registry(con: DuckDBPyConnection) -> DatasetRegistry:
    """
    Load dataset metadata from DuckDB's metadata.datasets table.

    This assumes codeintel.storage.metadata_bootstrap.bootstrap_metadata_datasets()
    has been run at least once on this database.
    """
    rows = con.execute(
        """
        SELECT table_key, name, is_view, jsonl_filename, parquet_filename
        FROM metadata.datasets
        ORDER BY table_key
        """
    ).fetchall()

    by_name: dict[str, Dataset] = {}
    by_table: dict[str, Dataset] = {}
    jsonl_map: dict[str, str] = {}
    parquet_map: dict[str, str] = {}

    for table_key, name, is_view, jsonl_filename, parquet_filename in rows:
        schema: TableSchema | None
        if is_view:
            schema = None
        else:
            schema = TABLE_SCHEMAS.get(table_key)
            if schema is None:
                # Drift: metadata.datasets lists a base table that TABLE_SCHEMAS doesn't know about.
                # Let assert_schema_alignment catch this elsewhere; we just warn in logs if needed.
                # For now, we simply leave schema=None.
                schema = None

        row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(table_key)
        ds = Dataset(
            table_key=table_key,
            name=name,
            schema=schema,
            row_binding=row_binding,
            jsonl_filename=jsonl_filename,
            parquet_filename=parquet_filename,
            is_view=bool(is_view),
        )

        by_name[name] = ds
        by_table[table_key] = ds
        if jsonl_filename:
            jsonl_map[table_key] = jsonl_filename
        if parquet_filename:
            parquet_map[table_key] = parquet_filename

    return DatasetRegistry(
        by_name=by_name,
        by_table_key=by_table,
        jsonl_datasets=jsonl_map,
        parquet_datasets=parquet_map,
    )


def dataset_for_name(registry: DatasetRegistry, name: str) -> Dataset:
    ds = registry.by_name.get(name)
    if ds is None:
        message = f"Unknown dataset name: {name}"
        raise KeyError(message)
    return ds


def dataset_for_table(registry: DatasetRegistry, table_key: str) -> Dataset:
    ds = registry.by_table_key.get(table_key)
    if ds is None:
        message = f"Unknown dataset table key: {table_key}"
        raise KeyError(message)
    return ds
```

How you plug this into `StorageGateway`:

* In `storage/gateway.py`, instead of building `DatasetRegistry` purely from `TABLE_SCHEMAS` + `DOCS_VIEWS`, you can:

  ```python
  from codeintel.storage.datasets import load_dataset_registry

  class StorageGateway:
      def __init__(...):
          ...
          apply_all_schemas(self.con)
          create_all_views(self.con)
          bootstrap_metadata_datasets(self.con)  # from metadata_bootstrap
          self.datasets = load_dataset_registry(self.con)
  ```

* And you can expose `self.datasets.jsonl_datasets` / `self.datasets.parquet_datasets` to `pipeline/export/*` as needed instead of importing static dicts.

---

## 4. Wiring `metadata.dataset_rows` into `DuckDBQueryService.read_dataset_rows`

Finally, here’s how to plug the macro into your existing MCP query service.

The current implementation (simplified) looks like this in `serving/mcp/query_service.py`:

```python
table = self.gateway.datasets.mapping.get(dataset_name)
if not table:
    message = f"Unknown dataset: {dataset_name}"
    raise errors.invalid_argument(message)

...

relation: DuckDBRelation
if table in DOCS_VIEW_QUERIES:
    relation = self.con.sql(DOCS_VIEW_QUERIES[table])
else:
    relation = self.con.table(table)
relation = relation.limit(limit_clamp.applied, offset_clamp.applied)
rows = relation.fetchall()
cols = [desc[0] for desc in relation.description]
mapped = [{col: row[idx] for idx, col in enumerate(cols)} for row in rows]
...
return DatasetRowsResponse(...)
```

With the `metadata.datasets` table and `metadata.dataset_rows` macro in place, you can simplify it to:

```python
from codeintel.storage.datasets import DatasetRegistry  # optional, if you want to reuse it

...

def read_dataset_rows(
    self,
    *,
    dataset_name: str,
    limit: int = 100,
    offset: int = 0,
) -> DatasetRowsResponse:
    """
    Read dataset rows with clamping and messaging, backed by metadata.dataset_rows().
    """

    table_key = self.gateway.datasets.mapping.get(dataset_name)
    if table_key is None:
        message = f"Unknown dataset: {dataset_name}"
        raise errors.invalid_argument(message)

    limit_clamp = clamp_limit_value(
        limit,
        default=limit,
        max_limit=self.limits.max_rows_per_call,
    )
    offset_clamp = clamp_offset_value(offset)
    meta = ResponseMeta(
        requested_limit=limit,
        applied_limit=limit_clamp.applied,
        requested_offset=offset,
        applied_offset=offset_clamp.applied,
        messages=[*limit_clamp.messages, *offset_clamp.messages],
    )

    if limit_clamp.applied <= 0:
        return DatasetRowsResponse(
            dataset=dataset_name,
            limit=limit_clamp.applied,
            offset=offset_clamp.applied,
            rows=[],
            meta=meta,
        )

    # Use the DuckDB macro instead of bespoke table/View logic.
    relation: DuckDBRelation = self.con.sql(
        "SELECT * FROM metadata.dataset_rows(?, ?, ?)",
        [table_key, limit_clamp.applied, offset_clamp.applied],
    )
    rows = relation.fetchall()
    cols = [desc[0] for desc in relation.description]
    mapped = [{col: row[idx] for idx, col in enumerate(cols)} for row in rows]

    meta.truncated = (
        limit_clamp.applied > 0 and len(mapped) == limit_clamp.applied
    )
    if not mapped:
        meta.messages.append(
            Message(
                code="dataset_empty",
                severity="info",
                detail="Dataset returned no rows for the requested slice.",
                context={"dataset": dataset_name, "offset": offset_clamp.applied},
            )
        )

    return DatasetRowsResponse(
        dataset=dataset_name,
        limit=limit_clamp.applied,
        offset=offset_clamp.applied,
        rows=[ViewRow.model_validate(r) for r in mapped],
        meta=meta,
    )
```

You can now **delete** `DOCS_VIEW_QUERIES` from `serving/mcp/query_service.py` entirely, because the macro covers both base tables and docs views via `metadata.datasets` + `query_table`.

---

If you’d like, next step we can:

Update storage.gateway.build_dataset_registry to delegate entirely to load_dataset_registry, and

Refactor pipeline/export/export_jsonl.py / export_parquet.py to use COPY + DatasetRegistry instead of static dicts, so exports become a thin wrapper around DuckDB.


# Detailed code of followup integrations #

Awesome, let’s finish wiring this through so your exports are 100% DuckDB-centric and driven off the unified dataset catalog.

I’ll assume you already have (or will add) these pieces from the previous steps:

* `metadata.datasets` table + `metadata.dataset_rows` macro in DuckDB. 
* `codeintel.storage.metadata_bootstrap.bootstrap_metadata_datasets(con)` that creates the metadata schema/table/macro and upserts rows from `TABLE_SCHEMAS` + `DOCS_VIEWS` + old JSONL/PARQUET mappings. 
* `codeintel.storage.datasets.load_dataset_registry(con)` returning a small struct with:

  * `by_name: Mapping[str, Dataset]`
  * `jsonl_datasets: Mapping[str, str]`
  * `parquet_datasets: Mapping[str, str]`

Now we’ll:

1. Update `storage.gateway.build_dataset_registry` + `open_gateway` to delegate fully to DuckDB + `load_dataset_registry`.
2. Refactor `pipeline/export/export_jsonl.py` and `export_parquet.py` to use `gateway.datasets.*` instead of the static `JSONL_DATASETS` / `PARQUET_DATASETS` dicts.

---

## 1. Update `storage.gateway` to use `load_dataset_registry`

### 1.1. Imports

In **`src/codeintel/storage/gateway.py`**, add these imports near the top:

```python
from codeintel.storage.datasets import Dataset, load_dataset_registry
from codeintel.storage.metadata_bootstrap import bootstrap_metadata_datasets
```

*(Adjust relative vs absolute imports if you prefer, but this matches your existing style.)* 

### 1.2. Extend `DatasetRegistry` to carry filenames + metadata

Find your current `DatasetRegistry`:

```python
class DatasetRegistry:
    """Track known table and view dataset names."""

    mapping: Mapping[str, str]
    tables: tuple[str, ...]
    views: tuple[str, ...]
```

Replace it with:

```python
class DatasetRegistry:
    """Track known table and view dataset names and export metadata."""

    # dataset_name -> fully qualified table/view key (e.g. "analytics.function_profile")
    mapping: Mapping[str, str]

    # dataset names backed by base tables
    tables: tuple[str, ...]

    # dataset names backed by docs.* views
    views: tuple[str, ...]

    # Optional richer metadata for each dataset (mirrors metadata.datasets).
    # Keys are dataset names, values are Dataset objects from storage.datasets.
    meta: Mapping[str, Dataset] | None = None

    # Export filename mappings (table/view key -> filename)
    jsonl_mapping: Mapping[str, str] | None = None
    parquet_mapping: Mapping[str, str] | None = None

    @property
    def all_datasets(self) -> tuple[str, ...]:
        """
        Return all registered dataset identifiers.

        Returns
        -------
        tuple[str, ...]
            Combined table and view names.
        """
        return self.tables + self.views

    def resolve(self, name: str) -> str:
        """
        Return a validated dataset name.

        Parameters
        ----------
        name
            Dataset identifier to validate.

        Returns
        -------
        str
            Fully qualified dataset name.

        Raises
        ------
        KeyError
            If the dataset name is unknown.
        """
        if name not in self.mapping:
            message = f"Unknown dataset: {name}"
            raise KeyError(message)
        return self.mapping[name]
```

Existing callers (which only use `.mapping`) continue to work, and you now have extra fields that exports can use. 

### 1.3. Rewrite `build_dataset_registry` to read from DuckDB

Replace your existing:

```python
def build_dataset_registry(*, include_views: bool = True) -> DatasetRegistry:
    """
    Build a dataset registry from known tables and docs views.
    ...
    table_keys = tuple(sorted(TABLE_SCHEMAS.keys()))
    view_keys = DOCS_VIEWS if include_views else ()
    mapping = {_dataset_name(key): key for key in table_keys}
    mapping.update({_dataset_name(key): key for key in view_keys})
    table_names = tuple(_dataset_name(key) for key in table_keys)
    view_names = tuple(_dataset_name(key) for key in view_keys)
    return DatasetRegistry(mapping=mapping, tables=table_names, views=view_names)
```

with a new, DuckDB-backed version:

```python
def build_dataset_registry(
    con: DuckDBConnection,
    *,
    include_views: bool = True,
) -> DatasetRegistry:
    """
    Build a dataset registry from DuckDB's metadata.datasets catalog.

    Parameters
    ----------
    con
        Open DuckDB connection with schemas/views/metadata applied.
    include_views
        When True, include docs.* views alongside base tables.

    Returns
    -------
    DatasetRegistry
        Registry containing dataset name -> table/view mappings and export metadata.
    """
    # Load rich Dataset objects from storage.datasets
    ds_registry = load_dataset_registry(con)

    # dataset_name -> table/view key mapping
    mapping: dict[str, str] = {
        name: ds.table_key for name, ds in ds_registry.by_name.items()
    }

    # Names of datasets by base-table vs view
    table_names = tuple(
        name for name, ds in ds_registry.by_name.items() if not ds.is_view
    )
    view_names = (
        tuple(name for name, ds in ds_registry.by_name.items() if ds.is_view)
        if include_views
        else tuple()
    )

    return DatasetRegistry(
        mapping=mapping,
        tables=table_names,
        views=view_names,
        meta=ds_registry.by_name,
        jsonl_mapping=ds_registry.jsonl_datasets,
        parquet_mapping=ds_registry.parquet_datasets,
    )
```

Now **all dataset name ↔ table/view ↔ filename logic flows from:**

`metadata.datasets` → `load_dataset_registry` → `build_dataset_registry` → `gateway.datasets`. 

### 1.4. Update `open_gateway` to bootstrap metadata + registry

Your current `open_gateway`:

```python
def open_gateway(config: StorageConfig) -> StorageGateway:
    ...
    datasets = build_dataset_registry()
    con = _connect(config)
    return _DuckDBGateway(config=config, datasets=datasets, con=con)
```

Change it to:

```python
def open_gateway(config: StorageConfig) -> StorageGateway:
    """
    Create a StorageGateway bound to a DuckDB database.

    Parameters
    ----------
    config
        Storage configuration describing connection options.

    Returns
    -------
    StorageGateway
        Gateway exposing typed accessors and dataset registry.
    """
    # 1) Connect and apply schemas/views/validation
    con = _connect(config)

    # 2) Ensure metadata.datasets and macros exist & are up to date
    if not config.read_only:
        bootstrap_metadata_datasets(con)

    # 3) Build dataset registry from DuckDB catalog
    datasets = build_dataset_registry(con)

    # 4) Return concrete gateway
    return _DuckDBGateway(config=config, datasets=datasets, con=con)
```

Because `_connect` already calls `apply_all_schemas` and `create_all_views` for non-read-only configs, `bootstrap_metadata_datasets` can rely on all base tables and docs views existing. 

`open_memory_gateway` and snapshot resolvers still just call `open_gateway`, so they automatically gain the new behavior.

---

## 2. Refactor `export_jsonl.py` to use `gateway.datasets` instead of static dicts

Now we swap `JSONL_DATASETS` / `PARQUET_DATASETS` for the mappings on `gateway.datasets`.

### 2.1. Remove static dataset imports

In **`src/codeintel/pipeline/export/export_jsonl.py`**, delete this line:

```python
from codeintel.pipeline.export.datasets import JSONL_DATASETS, PARQUET_DATASETS
```

Leave the rest of the imports alone.

### 2.2. `_select_dataset_tables` now takes `jsonl_mapping`

Change the function signature + body from:

```python
def _select_dataset_tables(
    dataset_mapping: Mapping[str, str],
    datasets: list[str] | None,
) -> dict[str, str]:
    if datasets is None:
        return {name: table for name, table in dataset_mapping.items() if table in JSONL_DATASETS}
    selected: dict[str, str] = {}
    for dataset_name in datasets:
        selected[dataset_name] = _resolve_dataset_table(dataset_name, dataset_mapping)
    return selected
```

to:

```python
def _select_dataset_tables(
    dataset_mapping: Mapping[str, str],
    jsonl_mapping: Mapping[str, str],
    datasets: list[str] | None,
) -> dict[str, str]:
    """
    Determine which dataset names/tables to export.

    Parameters
    ----------
    dataset_mapping
        dataset_name -> table/view key mapping from the gateway registry.
    jsonl_mapping
        table/view key -> JSONL filename mapping from the gateway registry.
    datasets
        Optional list of dataset names to export.

    Returns
    -------
    dict[str, str]
        Selected dataset_name -> table/view key mapping.
    """
    if datasets is None:
        # Export all datasets that have a configured JSONL filename.
        return {
            name: table
            for name, table in dataset_mapping.items()
            if table in jsonl_mapping
        }

    selected: dict[str, str] = {}
    for dataset_name in datasets:
        selected[dataset_name] = _resolve_dataset_table(dataset_name, dataset_mapping)
    return selected
```

### 2.3. `export_dataset_to_jsonl` uses registry filenames

Update the function to pull filenames from the registry:

```python
def export_dataset_to_jsonl(
    gateway: StorageGateway,
    dataset_name: str,
    output_dir: Path,
) -> Path:
    ...
    dataset_mapping = gateway.datasets.mapping
    jsonl_mapping = gateway.datasets.jsonl_mapping or {}

    if dataset_name not in dataset_mapping:
        message = f"Unknown dataset: {dataset_name}"
        raise ValueError(message)

    table_name = dataset_mapping[dataset_name]
    filename = jsonl_mapping.get(table_name, f"{dataset_name}.jsonl")
    output_path = output_dir / filename
    export_jsonl_for_table(gateway, table_name, output_path)
    return output_path
```

No more direct reference to `JSONL_DATASETS`.

### 2.4. `export_all_jsonl` wires everything together

In `export_all_jsonl`:

**Before:**

```python
    _validate_registry_or_raise(gateway)
    dataset_mapping = gateway.datasets.mapping
    selected = _select_dataset_tables(dataset_mapping, datasets)
    missing_tables = set(JSONL_DATASETS) - set(dataset_mapping.values())
```

**After:**

```python
    _validate_registry_or_raise(gateway)
    dataset_mapping = gateway.datasets.mapping
    jsonl_mapping = gateway.datasets.jsonl_mapping or {}
    parquet_mapping = gateway.datasets.parquet_mapping or {}

    selected = _select_dataset_tables(dataset_mapping, jsonl_mapping, datasets)
    missing_tables = set(jsonl_mapping) - set(dataset_mapping.values())
```

Then in the loop:

```python
    for dataset_name, table_name in sorted(selected.items()):
        filename = jsonl_mapping.get(table_name, f"{dataset_name}.jsonl")
        output_path = document_output_dir / filename
        ...
```

And finally, when writing the dataset manifest:

```python
    manifest_path = write_dataset_manifest(
        document_output_dir,
        dataset_mapping,
        jsonl_mapping=jsonl_mapping,
        parquet_mapping=parquet_mapping,
        selected=list(selected.keys()),
    )
```

This keeps `write_dataset_manifest` happy but removes the static JSONL/PARQUET dict usage entirely.

### 2.5. JSONL I/O: still a thin wrapper around DuckDB

Your current `export_jsonl_for_table` already:

* Normalizes a few special tables via `_normalized_relation`, then
* Calls `rel.write_json` if available (DuckDB relation API), or falls back to Pandas `to_json`.

That’s already a thin wrapper around DuckDB in the common case, so for now we leave it as-is; the main refactor here is metadata-driven filenames & selection. If you want to go all-in on `COPY`, you can later swap this body for:

```python
con.execute(
    f"COPY (SELECT * FROM {table_name}) TO ? (FORMAT JSON, ARRAY FALSE)",
    [str(output_path)],
)
```

because `table_name` comes from the dataset registry and is not user-supplied. But you don’t need that change to get the metadata unification benefits. 

---

## 3. Refactor `export_parquet.py` to use the same registry

Parquet follows the same pattern as JSONL.

### 3.1. Remove static dataset imports

In **`src/codeintel/pipeline/export/export_parquet.py`**, remove:

```python
from codeintel.pipeline.export.datasets import JSONL_DATASETS, PARQUET_DATASETS
```

### 3.2. `_select_dataset_tables` takes `parquet_mapping`

Change from:

```python
def _select_dataset_tables(
    dataset_mapping: Mapping[str, str],
    datasets: list[str] | None,
) -> dict[str, str]:
    if datasets is None:
        return {name: table for name, table in dataset_mapping.items() if table in PARQUET_DATASETS}
    selected: dict[str, str] = {}
    ...
```

to:

```python
def _select_dataset_tables(
    dataset_mapping: Mapping[str, str],
    parquet_mapping: Mapping[str, str],
    datasets: list[str] | None,
) -> dict[str, str]:
    if datasets is None:
        return {
            name: table
            for name, table in dataset_mapping.items()
            if table in parquet_mapping
        }
    selected: dict[str, str] = {}
    for dataset_name in datasets:
        selected[dataset_name] = _resolve_dataset_table(dataset_name, dataset_mapping)
    return selected
```

### 3.3. `export_dataset_to_parquet` uses registry filenames

Inside `export_dataset_to_parquet`:

```python
    dataset_mapping = gateway.datasets.mapping
    parquet_mapping = gateway.datasets.parquet_mapping or {}

    if dataset_name not in dataset_mapping:
        message = f"Unknown dataset: {dataset_name}"
        raise ValueError(message)

    table_name = dataset_mapping[dataset_name]
    filename = parquet_mapping.get(table_name, f"{dataset_name}.parquet")
    output_path = output_dir / filename
    export_parquet_for_table(gateway, table_name, output_path)
    return output_path
```

### 3.4. `export_all_parquet` uses registry mappings + manifest wiring

In `export_all_parquet`:

**Before:**

```python
    dataset_mapping = gateway.datasets.mapping
    selected = _select_dataset_tables(dataset_mapping, datasets)
    ...
    filename = PARQUET_DATASETS.get(table_name, f"{dataset_name}.parquet")
    ...
    manifest_path = write_dataset_manifest(
        document_output_dir,
        dataset_mapping,
        jsonl_mapping=JSONL_DATASETS,
        parquet_mapping=PARQUET_DATASETS,
        selected=list(selected.keys()),
    )
```

**After:**

```python
    dataset_mapping = gateway.datasets.mapping
    jsonl_mapping = gateway.datasets.jsonl_mapping or {}
    parquet_mapping = gateway.datasets.parquet_mapping or {}

    selected = _select_dataset_tables(dataset_mapping, parquet_mapping, datasets)

    written: list[Path] = []

    for dataset_name, table_name in sorted(selected.items()):
        filename = parquet_mapping.get(table_name, f"{dataset_name}.parquet")
        output_path = document_output_dir / filename
        try:
            export_parquet_for_table(gateway, table_name, output_path)
            written.append(output_path)
        except (DuckDBError, OSError, ValueError) as exc:
            ...
            # unchanged logging
```

And manifest:

```python
    manifest_path = write_dataset_manifest(
        document_output_dir,
        dataset_mapping,
        jsonl_mapping=jsonl_mapping,
        parquet_mapping=parquet_mapping,
        selected=list(selected.keys()),
    )
    written.append(manifest_path)
```

### 3.5. Parquet I/O is already DuckDB-centric

`export_parquet_for_table` already uses:

```python
rel = _normalized_relation(gateway.con, table_name)
rel.write_parquet(str(output_path))
```

Which is a DuckDB relation write, so it’s already a thin wrapper. You can leave that as-is; you’re now just driving file names and “which datasets to export” from the unified registry / `metadata.datasets`.

---

## 4. Sanity checklist

Once you wire this in, I’d sanity-check in this order:

1. **Run a small in-memory export:**

   * `open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=False)`
   * Populate a couple tables.
   * Call `export_all_jsonl` and `export_all_parquet` with no `datasets` argument.
   * Confirm the set of exported files matches what’s in `metadata.datasets` (i.e. any table/view with a non-NULL JSONL/PARQUET filename).

2. **Check the manifest content:**

   * Confirm `write_dataset_manifest` receives the same mapping as before (just now coming from `gateway.datasets.jsonl_mapping` / `parquet_mapping`).

3. **Validate registry:**

   * Your existing `validate_dataset_registry` still works; it only uses `gateway.datasets.mapping`, which we preserved.

If you’d like, next step we can add a tiny `test_dataset_registry_roundtrip` that:

* Inserts a synthetic row into `metadata.datasets`,
* Reloads the registry via `load_dataset_registry` + `build_dataset_registry`,
* And asserts that `export_all_jsonl` picks it up automatically.
