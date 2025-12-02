You already did a lot of the hard thinking with this epic; we’ll now turn it into a concrete, step‑by‑step migration plan with code sketches you can almost paste in.

I’ll assume the current codebase is exactly what’s in the zips (i.e., big `config/dataset_contract.py`, `storage/schema_generation.py`, `storage/datasets.py`, etc.). Note that `DatasetContract` *already* has a `schema_version` field and `DatasetRegistry` already surfaces it, so we’ll treat Epic 4 as:

* **Splitting the monolith into `config/datasets/` modules**
* **Tightening versioning semantics & propagation**
* **Adding an optional generator for row models + serializers**
* **Keeping everything backward‑compatible via `config/dataset_contract.py`**

---

## Phase 0 – Inventory & safety rails

Before changing anything, keep this mental map (this matches your current code):

* **`config/dataset_contract.py`** contains:

  * `ColumnType`, `Column`, `Index`, `TableSchema`, `CompositeSchema`
  * `RowToTuple`, `RowBinding`, `DatasetContract`
  * `TABLE_SCHEMAS: dict[str, TableSchema]`
  * `DATASET_CONTRACTS`, `DATASET_CONTRACTS_BY_TABLE_KEY`, plus various constants
  * Row `TypedDict`s like `FunctionProfileRowModel`, `CallGraphEdgeRow`, etc.
  * Row serializer functions like `function_profile_row_to_tuple`, `call_graph_edge_to_tuple`, etc.
  * `INSERT_SQL_BY_TABLE`, `DELETE_SQL_BY_TABLE` and helper functions.
* **Heavy dependencies** on this module from:

  * `storage` (e.g. `storage/datasets.py`, `storage/sql_helpers.py`, `storage/metadata_bootstrap.py`, `storage/contract_validation.py`)
  * `analytics` (`analytics/datasets.py`, `analytics/tests/*`)
  * `serving` (`serving/backend/datasets.py`, `serving/backend/duckdb_service.py`, `serving/services/datasets.py`)
  * Tests (e.g. `tests/config/test_dataset_contract.py`, `tests/storage/test_schema_roundtrip.py`)

**Design principle for the migration:**

> All existing imports of `codeintel.config.dataset_contract` must keep working
> through a compatibility shim, even as we move logic into `codeintel.config.datasets.*`.

---

## Phase 1 – Create `config/datasets/` package skeleton

### 1.1. Create the directory and `__init__.py`

**New files:**

```text
config/config/datasets/
    __init__.py
    primitives.py
    contracts.py
    rows.py
    sql.py
    gen_rows.py      # optional codegen script (Phase 6)
```

### 1.2. Initial `__init__.py`

Start with a simple façade; we’ll fill it later:

```python
# config/config/datasets/__init__.py

"""
Modular dataset contract definitions, schemas, row models, and SQL helpers.

This package is the new home for the contents of the legacy
`codeintel.config.dataset_contract` module. All new code should import from
`codeintel.config.datasets` instead of `codeintel.config.dataset_contract`.
"""

from . import primitives, contracts, rows, sql

# Re-export most commonly used symbols for ergonomic imports.
from .primitives import Column, ColumnType, Index, TableSchema, CompositeSchema, RowBinding, RowToTuple
from .contracts import (
    DatasetContract,
    TABLE_SCHEMAS,
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
)
from .sql import INSERT_SQL_BY_TABLE, DELETE_SQL_BY_TABLE

# NOTE: row models & serializer helpers will be re-exported after Phase 4.
```

You’ll extend this later to re‑export row models once `rows.py` is populated.

---

## Phase 2 – Move primitives (Column, TableSchema, CompositeSchema, RowBinding)

### 2.1. Extract primitive dataclasses into `primitives.py`

Cut the *type definitions only* from `dataset_contract.py` and paste into:

```python
# config/config/datasets/primitives.py

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeVar

ColumnType = Literal[
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "DOUBLE",
    "DECIMAL(38,0)",
    "VARCHAR",
    "JSON",
    "TIMESTAMP",
    "TIMESTAMPTZ",
]
COLUMN_TYPE: Final = ColumnType   # backwards-compat alias if used

@dataclass(frozen=True)
class Column:
    """Definition of a single table column."""

    name: str
    type: ColumnType
    nullable: bool = True
    description: str | None = None

@dataclass(frozen=True)
class Index:
    """Secondary index definition."""
    name: str
    columns: tuple[str, ...]
    unique: bool = False

@dataclass(frozen=True)
class TableSchema:
    """Schema definition for a DuckDB table."""

    schema: str
    name: str
    columns: list[Column]
    primary_key: tuple[str, ...] = ()
    indexes: tuple[Index, ...] = ()
    description: str | None = None

    @property
    def fq_name(self) -> str:
        return f"{self.schema}.{self.name}"

    def column_names(self) -> list[str]:
        return [col.name for col in self.columns]

# Row / binding primitives
RowToTuple = Callable[[Mapping[str, object]], tuple[object, ...]]
RowDictType = type[object]
_Column = TypeVar("_Column", bound=str)

@dataclass(frozen=True)
class CompositeSchema:
    """Declare how a profile schema is composed from source tables."""
    composed_of: tuple[str, ...]
    shared_fragments: tuple[tuple[Column, ...], ...]
    additional_columns: tuple[Column, ...]
    column_mappings: dict[str, str]
    excluded_columns: frozenset[str]

    # existing methods (e.g. _get_shared_column_names, source_column_names)
    # move here unchanged from dataset_contract

@dataclass(frozen=True)
class RowBinding:
    """Connect a DuckDB table key to a TypedDict row model and serializer."""
    row_type: RowDictType
    to_tuple: RowToTuple
```

**Edits to `config/dataset_contract.py`:**

* Remove these definitions and replace them with imports:

```python
# at top of config/config/dataset_contract.py

from codeintel.config.datasets.primitives import (
    Column,
    ColumnType,
    ColumnType as COLUMN_TYPE,  # if you used this alias
    Index,
    TableSchema,
    CompositeSchema,
    RowBinding,
    RowToTuple,
)
```

You can keep `RowDictType` as an alias locally or move it as well.

**Impact:** purely internal; test suite should still pass if we didn’t break imports.

---

## Phase 3 – Move TABLE_SCHEMAS & DatasetContract into `contracts.py`

### 3.1. Define DatasetContract (with versioning) in `contracts.py`

Move the `DatasetContract` dataclass definition into `contracts.py` and **keep all existing fields**; you already have `schema_version` and `upstream_dependencies`:

```python
# config/config/datasets/contracts.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from .primitives import TableSchema, RowBinding, CompositeSchema

@dataclass(frozen=True)
class DatasetContract:
    """Metadata describing a logical dataset backed by a DuckDB table or view."""

    table_key: str
    name: str
    schema: TableSchema | None
    row_binding: RowBinding | None = None
    json_schema_id: str | None = None
    jsonl_filename: str | None = None
    parquet_filename: str | None = None
    is_view: bool = False
    owner_package: Literal["core", "analytics", "graphs", "qa", "docs"] | None = None
    tags: frozenset[str] = frozenset()
    description: str | None = None
    family: str | None = None
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    stable_id: str | None = None
    schema_version: str | None = None
    upstream_dependencies: tuple[str, ...] = ()
    validation_profile: Literal["strict", "lenient"] = "strict"
    composition: CompositeSchema | None = None
    deprecated: bool = False
    deprecation_message: str | None = None

    # existing convenience methods (column_names(), capabilities(), require_row_binding(), etc.)
    # move here unchanged from dataset_contract.py
```

> ✅ **New**: `deprecated` and `deprecation_message` give you deprecation metadata you mentioned as “optional”.

### 3.2. Move TABLE_SCHEMAS and build DATASET_CONTRACTS here

From `dataset_contract.py`, extract:

* Column fragments (`REPO_COMMIT_COLS`, `FUNCTION_GOID_COL`, etc.)
* The big `TABLE_SCHEMAS: dict[str, TableSchema] = {...}`
* Any helper functions used to construct schemas (e.g. `_make_table`, `_add_indexes`, etc.)
* The `RowBinding` registry `_ROW_BINDINGS_BY_TABLE_KEY` if you have one.

Place them into `contracts.py` under clear sections:

```python
# config/config/datasets/contracts.py (continued)

# ---------------------------------------------------------------------------
# TABLE_SCHEMAS - All table definitions
# ---------------------------------------------------------------------------

TABLE_SCHEMAS: dict[str, TableSchema] = {
    "core.ast_nodes": TableSchema(
        schema="core",
        name="ast_nodes",
        columns=[
            # all the Column(...) literals moved verbatim
        ],
        primary_key=("repo", "commit", "node_id"),
        description="Raw AST nodes extracted from source files",
    ),
    # ...
}
```

Then define dataset contracts:

```python
# Bind row models (rows.py will provide types and serializers)
from . import rows  # circular-safe if rows.py does NOT import contracts

def _row_binding(*, row_type: type[object], to_tuple: RowToTuple) -> RowBinding:
    return RowBinding(row_type=row_type, to_tuple=to_tuple)

ROW_BINDINGS_BY_TABLE_KEY: dict[str, RowBinding] = {
    "analytics.function_profile": _row_binding(
        row_type=rows.FunctionProfileRowModel,
        to_tuple=rows.function_profile_row_to_tuple,
    ),
    # ...
}

DATASET_CONTRACTS: Final[dict[str, DatasetContract]] = {
    # keyed by dataset logical name ("function_profile", "call_graph_edges", etc.)
    "function_profile": DatasetContract(
        table_key="analytics.function_profile",
        name="function_profile",
        schema=TABLE_SCHEMAS["analytics.function_profile"],
        row_binding=ROW_BINDINGS_BY_TABLE_KEY["analytics.function_profile"],
        json_schema_id="function_profile",
        jsonl_filename="function_profile.jsonl",
        parquet_filename="function_profile.parquet",
        is_view=False,
        owner_package="analytics",
        tags=frozenset({"base_table"}),
        description="Behavior and metadata for individual functions",
        family="analytics",
        schema_version="v1",  # start somewhere
        upstream_dependencies=("function_metrics", "call_graph_edges"),
    ),
    # ...
}

DATASET_CONTRACTS_BY_TABLE_KEY: Final[dict[str, DatasetContract]] = {
    contract.table_key: contract for contract in DATASET_CONTRACTS.values()
}
```

You can keep any helper like `get_dataset_contract(name: str)` here.

### 3.3. Rewire `dataset_contract.py` to use new contracts

At the bottom of `config/config/dataset_contract.py`, replace the existing `DATASET_CONTRACTS` definitions with imports:

```python
# config/config/dataset_contract.py

from codeintel.config.datasets.contracts import (
    DatasetContract,
    TABLE_SCHEMAS,
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
)

# For backwards compatibility, re-export these names
__all__ = [
    "Column",
    "ColumnType",
    "Index",
    "TableSchema",
    "CompositeSchema",
    "RowBinding",
    "DatasetContract",
    "TABLE_SCHEMAS",
    "DATASET_CONTRACTS",
    "DATASET_CONTRACTS_BY_TABLE_KEY",
    # plus many more once rows/sql are moved
]
```

You can expand `__all__` later as you move rows and SQL.

---

## Phase 4 – Move row `TypedDict`s and serializers to `rows.py`

This is the noisiest part but very mechanical.

### 4.1. Create `rows.py` with one example

```python
# config/config/datasets/rows.py

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

# Example: analytics.function_profile

class FunctionProfileRowModel(TypedDict):
    """Row shape for `analytics.function_profile` inserts."""

    function_goid_h128: int
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    module: str | None
    language: str | None
    # ... all the other fields exactly as in dataset_contract.py
    lines_deleted: int

def function_profile_row_to_tuple(row: "FunctionProfileRowModel") -> tuple[object, ...]:
    """Serialize a FunctionProfileRowModel into INSERT column order."""
    return (
        row["function_goid_h128"],
        row["urn"],
        row["repo"],
        row["commit"],
        row["rel_path"],
        row["module"],
        row["language"],
        # ... exactly matching TABLE_SCHEMAS["analytics.function_profile"].column_names()
        row["lines_deleted"],
    )
```

Then repeat for all row models currently defined in `dataset_contract.py`:

* `BehavioralCoverageRowModel`, `CallGraphEdgeRow`, `CallGraphNodeRow`, `ConfigValueRow`, `CoverageLineRow`, etc.
* All the `*_row_to_tuple`, `serialize_test_*` helpers.

### 4.2. Update `contracts.py` RowBinding registry

Make sure `contracts.py` imports row models only from `.rows`:

```python
from . import rows

ROW_BINDINGS_BY_TABLE_KEY: dict[str, RowBinding] = {
    "analytics.function_profile": _row_binding(
        row_type=rows.FunctionProfileRowModel,
        to_tuple=rows.function_profile_row_to_tuple,
    ),
    "analytics.function_metrics": _row_binding(
        row_type=rows.FunctionMetricsRow,
        to_tuple=rows.function_metrics_row_to_tuple,
    ),
    # etc
}
```

### 4.3. Re‑export row models through `datasets.__init__` and `dataset_contract.py`

In `config/config/datasets/__init__.py`:

```python
from .rows import (
    FunctionProfileRowModel,
    function_profile_row_to_tuple,
    BehavioralCoverageRowModel,
    # ... all row models and serializers
)
```

In `config/config/dataset_contract.py`:

```python
from codeintel.config.datasets.rows import (
    FunctionProfileRowModel,
    function_profile_row_to_tuple,
    BehavioralCoverageRowModel,
    # ...
)
```

You *don’t* need `__all__` to be exhaustive as long as star imports are rare, but for clarity you can populate it programmatically:

```python
from codeintel.config.datasets import rows as _rows

__all__ += [name for name in dir(_rows) if not name.startswith("_")]
```

### 4.4. Why this order avoids cycles

* `rows.py` does not import `DatasetContract` or `TABLE_SCHEMAS`.
* `contracts.py` imports `rows` for `RowBinding`.
* `dataset_contract.py` imports both; nothing imports `dataset_contract` from inside these.

So imports remain acyclic.

---

## Phase 5 – Move SQL helpers into `sql.py`

### 5.1. Create `sql.py`

Take the `_build_insert_sql`, `_build_insert_sql_by_table`, `_build_delete_sql`, `_build_delete_sql_by_table`, and `INSERT_SQL_BY_TABLE`, `DELETE_SQL_BY_TABLE` definitions from `dataset_contract.py` and put them into:

```python
# config/config/datasets/sql.py

from __future__ import annotations

from typing import Final

from .contracts import TABLE_SCHEMAS

def build_insert_sql(table_key: str) -> str:
    """
    Generate an INSERT SQL statement from the TableSchema.

    Raises
    ------
    ValueError
        If no schema is defined for the table key.
    """
    schema = TABLE_SCHEMAS.get(table_key)
    if schema is None:
        msg = f"No schema defined for table {table_key}"
        raise ValueError(msg)

    col_names = [col.name for col in schema.columns]
    cols_str = ", ".join(col_names)
    placeholders = ", ".join("?" * len(col_names))
    return f"INSERT INTO {table_key} ({cols_str}) VALUES ({placeholders})"  # noqa: S608

def _build_insert_sql_by_table() -> dict[str, str]:
    result: dict[str, str] = {}
    for table_key, schema in TABLE_SCHEMAS.items():
        if table_key.startswith("docs."):
            # docs.* views are read-only
            continue
        result[table_key] = build_insert_sql(table_key)
    return result

def _build_delete_sql(table_key: str) -> str | None:
    schema = TABLE_SCHEMAS.get(table_key)
    if schema is None:
        return None
    col_names = [col.name for col in schema.columns]
    if "repo" in col_names and "commit" in col_names:
        return f"DELETE FROM {table_key} WHERE repo = ? AND commit = ?"  # noqa: S608
    return None

def _build_delete_sql_by_table() -> dict[str, str]:
    result: dict[str, str] = {}
    for table_key in TABLE_SCHEMAS:
        if table_key.startswith("docs."):
            continue
        sql = _build_delete_sql(table_key)
        if sql is not None:
            result[table_key] = sql
    return result

INSERT_SQL_BY_TABLE: Final[dict[str, str]] = _build_insert_sql_by_table()
DELETE_SQL_BY_TABLE: Final[dict[str, str]] = _build_delete_sql_by_table()
```

### 5.2. Update `dataset_contract.py` and consumers

In `dataset_contract.py`:

```python
from codeintel.config.datasets.sql import (
    INSERT_SQL_BY_TABLE,
    DELETE_SQL_BY_TABLE,
    build_insert_sql,
)
```

In modules like `storage/sql_helpers.py` you can **either**:

* Keep importing from `codeintel.config.dataset_contract` (compat shim), or
* Move to the new home:

```python
from codeintel.config.datasets.sql import INSERT_SQL_BY_TABLE, DELETE_SQL_BY_TABLE
```

Same for any tests/analytics modules that rely on insert/delete SQL from contracts; they can move to `codeintel.config.datasets.sql` over time.

---

## Phase 6 – Tighten versioning semantics

You already have:

* `schema_version` on `DatasetContract`
* `storage/datasets.DatasetRegistry` storing and exposing it

We’ll now:

1. **Store version in `metadata.datasets`**
2. **Surface version in catalogs/exports**
3. **Use it in schema generation exports**

### 6.1. Modify metadata DDL in `storage/metadata_bootstrap.py`

Right now `metadata.datasets` is:

```sql
CREATE TABLE IF NOT EXISTS metadata.datasets (
    table_key        TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    is_view          BOOLEAN NOT NULL,
    jsonl_filename   TEXT,
    parquet_filename TEXT,
    family           TEXT,
    description      TEXT
);
```

Change to:

```sql
CREATE TABLE IF NOT EXISTS metadata.datasets (
    table_key        TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    is_view          BOOLEAN NOT NULL,
    jsonl_filename   TEXT,
    parquet_filename TEXT,
    family           TEXT,
    description      TEXT,
    schema_version   TEXT
);
```

And add an ALTER for existing DBs:

```sql
ALTER TABLE metadata.datasets ADD COLUMN IF NOT EXISTS schema_version TEXT;
```

Then, wherever you insert rows into `metadata.datasets` from `DATASET_CONTRACTS` (there’s a bootstrap function in this file that loops over contracts), update the insert to include `schema_version`:

```python
for contract in DATASET_CONTRACTS.values():
    con.execute(
        """
        INSERT OR REPLACE INTO metadata.datasets(
            table_key, name, is_view, jsonl_filename,
            parquet_filename, family, description, schema_version
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            contract.table_key,
            contract.name,
            contract.is_view,
            contract.jsonl_filename,
            contract.parquet_filename,
            contract.family,
            contract.description,
            contract.schema_version,
        ],
    )
```

### 6.2. Surface version in dataset descriptions

In `storage/datasets.describe_dataset`, you already have:

```python
return {
    "name": ds.name,
    "table_key": ds.table_key,
    "is_view": ds.is_view,
    "schema_columns": (...),
    # ...
}
```

Add:

```python
        "schema_version": ds.schema_version,
        "deprecated": ds.deprecated,
        "deprecation_message": ds.deprecation_message,
```

This is already partially there in the current code (I see `schema_version` used), but verify that all call sites (e.g. `storage/catalog.py`, `tests/storage/test_catalog_describe.py`) expect it and test for it.

### 6.3. Use version in JSON Schema generation (optional but nice)

In `storage/schema_generation.py`, instead of:

```python
schema_id = f"https://schemas.codeintel.dev/export/{contract.json_schema_id}.json"
```

You can incorporate version:

```python
version = contract.schema_version or "v1"
schema_id = f"https://schemas.codeintel.dev/export/{version}/{contract.json_schema_id}.json"
path = output_dir / f"{contract.json_schema_id}.{version}.json"
```

And adjust tests that look for `call_graph_edges.json` to accept `call_graph_edges.v1.json` (or keep the old name if you don’t want to break anything yet).

---

## Phase 7 – Add optional codegen for row models & serializers

You said “partly generated”; here’s a pragmatic way:

> Table schemas stay hand‑written; row models + serializers can be generated from `TABLE_SCHEMAS` plus a small type mapping, with overrides for tricky cases.

### 7.1. Simple generator script `gen_rows.py`

Create `config/config/datasets/gen_rows.py`:

```python
# config/config/datasets/gen_rows.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .contracts import TABLE_SCHEMAS
from .primitives import Column, TableSchema

@dataclass
class TypeMapping:
    duckdb_type: str
    python_type: str  # as source code string, e.g. "int", "str | None"

# base mapping (nullable handled separately)
BASE_TYPE_MAP: dict[str, str] = {
    "SMALLINT": "int",
    "INTEGER": "int",
    "BIGINT": "int",
    "DOUBLE": "float",
    "DECIMAL(38,0)": "int",
    "VARCHAR": "str",
    "JSON": "object",
    "TIMESTAMP": "datetime",
    "TIMESTAMPTZ": "datetime",
}

# per-column overrides by fully qualified "schema.table.column"
COLUMN_TYPE_OVERRIDES: dict[str, str] = {
    # e.g. "analytics.function_profile.slow_test_threshold_ms": "float",
}

def _python_type_for_column(table: TableSchema, col: Column) -> str:
    key = f"{table.schema}.{table.name}.{col.name}"
    if key in COLUMN_TYPE_OVERRIDES:
        return COLUMN_TYPE_OVERRIDES[key]
    base = BASE_TYPE_MAP[col.type]
    return f"{base} | None" if col.nullable else base

def _generate_typeddict(table_key: str, table: TableSchema) -> str:
    class_name = _row_class_name(table_key)
    lines: list[str] = []
    lines.append(f"class {class_name}(TypedDict):")
    lines.append(f'    """Row shape for `{table_key}` inserts."""')
    lines.append("")
    for col in table.columns:
        py_type = _python_type_for_column(table, col)
        lines.append(f"    {col.name}: {py_type}")
    lines.append("")
    return "\n".join(lines)

def _row_class_name(table_key: str) -> str:
    schema, name = table_key.split(".", 1)
    parts = [p.capitalize() for p in name.split("_")]
    return "".join(parts) + "Row"

def _generate_serializer(table_key: str, table: TableSchema) -> str:
    class_name = _row_class_name(table_key)
    func_name = f"{name}_row_to_tuple".replace(".", "_")  # e.g. function_profile_row_to_tuple
    col_names = [c.name for c in table.columns]
    lines: list[str] = []
    lines.append(f"def {func_name}(row: \"{class_name}\") -> tuple[object, ...]:")
    lines.append(f'    """Serialize a {class_name} into INSERT column order."""')
    lines.append("    return (")
    for col in col_names:
        lines.append(f'        row["{col}"],')
    lines.append("    )")
    lines.append("")
    return "\n".join(lines)

def render_rows_module(table_keys: Iterable[str]) -> str:
    lines: list[str] = []
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from datetime import datetime")
    lines.append("from typing import TypedDict")
    lines.append("")
    for table_key in table_keys:
        table = TABLE_SCHEMAS[table_key]
        lines.append(_generate_typeddict(table_key, table))
        lines.append(_generate_serializer(table_key, table))
    return "\n".join(lines)

def main() -> None:
    # choose which tables to generate rows for; could be all non-view tables
    table_keys = [
        key for key, schema in TABLE_SCHEMAS.items()
        if not key.startswith("docs.")
    ]
    content = render_rows_module(table_keys)
    target = Path(__file__).with_name("rows_generated.py")
    target.write_text(content, encoding="utf-8")

if __name__ == "__main__":
    main()
```

You can then:

* Either **use `rows_generated.py` as a source** and import from it in `rows.py`, or
* Let the generator overwrite `rows.py` itself once you trust the mapping.

A pattern I like:

```python
# rows.py
# 1. Hand-written special cases / overrides
# 2. `from .rows_generated import *`
```

That lets you keep special rows (like tests or docs) more precise or richer than the generated ones.

### 7.2. Hook generator into your workflow (optional)

* Add a simple `make` task: `make gen-dataset-rows` running `python -m codeintel.config.datasets.gen_rows`.
* If you want, add a pre‑commit hook that checks `rows.py` matches the generator output.

---

## Phase 8 – Tests & layering updates

### 8.1. Update imports to prefer `codeintel.config.datasets`

In tests that currently import from `codeintel.config.dataset_contract`, you can gradually move them to:

```python
from codeintel.config.datasets import (
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
    FUNCTION_PROFILE_COLUMNS,
    INSERT_SQL_BY_TABLE,
    # ...
)
```

But thanks to the shim, you don’t *have* to change everything immediately.

Concretely, check and adjust:

* `tests/config/test_dataset_contract.py`
* `tests/storage/test_schema_roundtrip.py`
* `tests/storage/test_sql_helpers.py`
* `tests/config/test_composite_schemas.py`
* Any tests that import row models or constants from `dataset_contract`.

### 8.2. Add tests for versioning semantics

1. **Dataset contracts have versions**

   In `tests/config/test_dataset_contract.py` (or a new `test_datasets_contracts.py`):

   ```python
   from codeintel.config.datasets import DATASET_CONTRACTS

   def test_all_datasets_have_schema_version() -> None:
       missing = [name for name, ds in DATASET_CONTRACTS.items() if ds.schema_version is None]
       assert not missing, f"Datasets missing schema_version: {missing}"
   ```

   (If you want some datasets unversioned, adjust the rule.)

2. **Metadata includes version**

   In `tests/storage/test_catalog_describe.py` or similar, assert `schema_version` is present:

   ```python
   def test_catalog_describe_includes_schema_version(con, registry) -> None:
       entries = describe_catalog(registry, con)
       for entry in entries:
           assert "schema_version" in entry
   ```

3. **JSON schema generation includes version in id/filename (if you changed it)**

   Update `tests/storage/test_schema_generation.py` to match your new `schema_id` and file naming scheme (`call_graph_edges.v1.json`, etc.), or keep old names if you stayed compatible.

### 8.3. Update layering checks

In `config/layering_checks.py`, add the new package to the layer map:

```python
LAYER_FOR_PREFIX: Final = {
    "codeintel.config.primitives": "core",
    "codeintel.config.datasets": "schema",  # new
    "codeintel.config.dataset_contract": "schema",  # legacy shim
    # ...
}
```

And update `ALLOWED` so that:

* `schema` can be used by `core`, `analytics`, `storage`, `serving` (if that’s your existing rule).
* `schema` itself only depends on `core`/stdlib (and not on `analytics`, `storage`, etc.).

---

## How to roll it out safely

If you want to minimize risk:

1. **Phase 1–3** (primitives + contracts) first. Run tests.
2. **Phase 4–5** (rows + SQL) next, but keep `dataset_contract.py` re‑exporting everything. Run tests.
3. **Phase 6** (versioning into metadata + schema generation) – update DDL, metadata bootstrap, and tests.
4. **Phase 7** (codegen) after you’re comfortable with the new layout; start with writing `rows_generated.py` but *don’t* auto‑overwrite `rows.py` until you’ve diffed them.

Once that’s done, `config/dataset_contract.py` is just a backwards‑compat shim and `codeintel.config.datasets.*` becomes the real single source of truth – modular, versioned, and ready for generation.
