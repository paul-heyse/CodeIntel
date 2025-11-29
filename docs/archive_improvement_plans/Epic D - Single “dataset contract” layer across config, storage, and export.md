# Single “dataset contract” layer across config, storage, and export detailed implementation plan #

Gotcha — let’s turn that high-level idea into something your implementation agent can pretty much lift straight into the repo.

Important note up front:
You *already* have the beginnings of this “dataset contract” layer:

* `storage/storage/datasets.py` — `Dataset`, `DatasetRegistry`, `ROW_BINDINGS_BY_TABLE_KEY`.
* `metadata.datasets` table seeded by `storage/storage/metadata_bootstrap.py`.
* `StorageGateway.datasets` (outer `DatasetRegistry` in `storage/storage/registry_helpers.py`).
* Export code using `gateway.datasets.mapping/jsonl_mapping/parquet_mapping`.

So instead of inventing a parallel “DatasetSpec”, we’ll **evolve `Dataset` + `DatasetRegistry` into the canonical spec** and then refactor the remaining bits (filenames, JSON Schemas, exports) to consume it.

Below is a step-by-step plan with file-scoped changes and code snippets.

---

## Goals / invariants

After this refactor:

1. There is **one canonical definition** of a dataset: `storage.storage.datasets.Dataset`.
2. All of these become *pure consumers* of that contract:

   * `storage.storage.metadata_bootstrap`
   * `storage.storage.registry_helpers`
   * `storage.storage.rows` (via `ROW_BINDINGS_BY_TABLE_KEY`)
   * `pipeline.pipeline.export.*`
   * `serving.serving.http.datasets`
3. Anything that needs to know:

   * dataset name → table key
   * table key → row model + to_tuple
   * dataset → JSONL / Parquet filename
   * dataset → JSON Schema id (for validation)

   gets it via the dataset contract layer — not via ad-hoc constants.

---

## Step 1 – Tighten and name the dataset spec in `storage/storage/datasets.py`

**File:** `storage/storage/datasets.py`

Right now you have:

* `@dataclass(frozen=True) class Dataset`
* `@dataclass(frozen=True) class DatasetRegistry` (inner one)
* `RowBinding` and `ROW_BINDINGS_BY_TABLE_KEY`
* `load_dataset_registry(con: DuckDBPyConnection) -> DatasetRegistry`
* helpers `dataset_for_name`, `dataset_for_table_key`, etc.

### 1.1. Alias `Dataset` as the “DatasetSpec” concept

Add a simple alias + docstring so the concept is explicit and can be referenced elsewhere:

```python
# Near the top of storage/storage/datasets.py, after Dataset is defined:

# Backwards-compatible alias for the canonical dataset contract type.
DatasetSpec = Dataset
"""Canonical metadata contract for a logical dataset backed by DuckDB."""
```

This is mostly semantic, but it makes it very obvious to agents and humans that this is *the* spec type.

### 1.2. Make `Dataset` a bit more self-describing

You already have:

```python
@dataclass(frozen=True)
class Dataset:
    """Metadata describing a logical dataset backed by a DuckDB table or view."""

    table_key: str
    name: str
    schema: TableSchema | None
    row_binding: RowBinding | None = None
    jsonl_filename: str | None = None
    parquet_filename: str | None = None
    is_view: bool = False
```

Extend this with two new optional fields:

* `description`: human-readable string (for docs / agents).
* `json_schema_id`: name of the JSON schema used for validation, if any.

Example patch:

```python
@dataclass(frozen=True)
class Dataset:
    """Metadata describing a logical dataset backed by a DuckDB table or view.

    Attributes
    ----------
    table_key
        Fully qualified DuckDB identifier, e.g. "analytics.function_profile".
    name
        Logical dataset name, e.g. "function_profile".
    schema
        Statically defined TableSchema when the dataset is backed by a table;
        None when the dataset is a view.
    row_binding
        Optional binding to a TypedDict row model and serializer.
    jsonl_filename
        Default filename for JSONL exports (may be None when not exported).
    parquet_filename
        Default filename for Parquet exports (may be None when not exported).
    is_view
        True when this dataset is a docs.* view instead of a base table.
    description
        Optional human-readable description of the dataset’s purpose.
    json_schema_id
        Optional JSON Schema name (without .json) used for export validation.
    """

    table_key: str
    name: str
    schema: TableSchema | None
    row_binding: RowBinding | None = None
    jsonl_filename: str | None = None
    parquet_filename: str | None = None
    is_view: bool = False
    description: str | None = None
    json_schema_id: str | None = None
```

We’ll *populate* `json_schema_id` in Step 3 via a mapping, not from the DB.

---

## Step 2 – Ensure row models & bindings are clearly part of the contract

You *already* have row bindings:

```python
@dataclass(frozen=True)
class RowBinding:
    """Connect a DuckDB table key to a TypedDict row model and serializer."""

    row_type: RowDictType
    to_tuple: RowToTuple
```

And the big `ROW_BINDINGS_BY_TABLE_KEY` mapping.

### 2.1. Add an accessor to `Dataset` for row bindings

To make the contract more ergonomic, add convenience methods:

```python
@dataclass(frozen=True)
class Dataset:
    ...
    def require_row_binding(self) -> RowBinding:
        """Return row binding or raise a clear error if missing."""
        if self.row_binding is None:
            message = f"Dataset {self.name} ({self.table_key}) has no row binding"
            raise KeyError(message)
        return self.row_binding

    def has_row_binding(self) -> bool:
        """Return True when this dataset has a TypedDict binding."""
        return self.row_binding is not None
```

And in `load_dataset_registry`, where you currently do:

```python
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
```

…you don’t need to change anything yet; `row_binding` is already wired.

### 2.2. (Optional) Add a simple coverage test

In your tests for storage (or a new `tests/test_datasets.py`):

* Assert that for each non-view dataset that you *intend* to ingest/export, there is a `ROW_BINDINGS_BY_TABLE_KEY` entry and the row model fields match the `TableSchema` columns.

Pseudo-test:

```python
def test_row_bindings_cover_expected_tables() -> None:
    from codeintel.config.schemas.tables import TABLE_SCHEMAS
    from codeintel.storage.datasets import ROW_BINDINGS_BY_TABLE_KEY

    missing = sorted(
        key for key, schema in TABLE_SCHEMAS.items()
        if not schema.name.startswith("tmp_")  # or whatever exclusion you want
        and key not in ROW_BINDINGS_BY_TABLE_KEY
    )
    assert not missing, f"Row bindings missing for: {', '.join(missing)}"
```

This enforces the “row model is part of the dataset contract” story.

---

## Step 3 – Add JSON Schema metadata to the dataset contract

Right now, JSON Schemas are:

* Files in `config/config/schemas/export/*.json`
* A static list in `pipeline/pipeline/export/__init__.py`:

  ```python
  DEFAULT_VALIDATION_SCHEMAS: list[str] = [
      "function_profile",
      "file_profile",
      "module_profile",
      "call_graph_edges",
      "symbol_use_edges",
      "test_coverage_edges",
      "test_profile",
      "behavioral_coverage",
      "data_model_fields",
      "data_model_relationships",
  ]
  ```

We want that list to be **derived from the dataset contract**, not a random constant.

### 3.1. Add a map of dataset → JSON schema id to `storage.storage.datasets`

In `storage/storage/datasets.py`, near `ROW_BINDINGS_BY_TABLE_KEY`, define:

```python
# Dataset names that have JSON Schemas under config/config/schemas/export/.
# Keys are dataset logical names (Dataset.name); values are schema filenames without .json.
JSON_SCHEMA_BY_DATASET_NAME: dict[str, str] = {
    # Profiles
    "function_profile": "function_profile",
    "file_profile": "file_profile",
    "module_profile": "module_profile",
    # Graph edges
    "call_graph_edges": "call_graph_edges",
    "symbol_use_edges": "symbol_use_edges",
    "test_coverage_edges": "test_coverage_edges",
    # Tests
    "test_profile": "test_profile",
    "behavioral_coverage": "behavioral_coverage",
    # Data models
    "data_model_fields": "data_model_fields",
    "data_model_relationships": "data_model_relationships",
}
```

You can keep this close to the row binding map — it’s the same kind of “dataset-specific metadata.”

### 3.2. Use this map when constructing `Dataset` instances

In `load_dataset_registry`:

```python
from .datasets import JSON_SCHEMA_BY_DATASET_NAME  # relative import inside the module

...

for table_key, name, is_view, jsonl_filename, parquet_filename in rows:
    schema: TableSchema | None = None if is_view else TABLE_SCHEMAS.get(table_key)
    row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(table_key)
    json_schema_id = JSON_SCHEMA_BY_DATASET_NAME.get(name)

    ds = Dataset(
        table_key=table_key,
        name=name,
        schema=schema,
        row_binding=row_binding,
        jsonl_filename=jsonl_filename,
        parquet_filename=parquet_filename,
        is_view=bool(is_view),
        json_schema_id=json_schema_id,
    )
    ...
```

Now:

* `Dataset.json_schema_id` is set when applicable.
* You still rely on `metadata.datasets` for filenames; JSON schema mapping is purely Python-side.

### 3.3. Provide a helper on `DatasetRegistry` for schema lists

Still in `storage/storage/datasets.py`, add:

```python
@dataclass(frozen=True)
class DatasetRegistry:
    ...
    def datasets_with_json_schema(self) -> tuple[str, ...]:
        """Return dataset names that have JSON Schema validation."""
        return tuple(
            name
            for name, ds in self.by_name.items()
            if ds.json_schema_id is not None
        )
```

This gives export code a clean, contract-driven way to know what’s “validatable”.

---

## Step 4 – Replace `DEFAULT_VALIDATION_SCHEMAS` with dataset-driven logic

**File:** `pipeline/pipeline/export/__init__.py`

Currently:

```python
"""Doc export utilities for emitting CodeIntel analytics as JSONL or Parquet artifacts."""

DEFAULT_VALIDATION_SCHEMAS: list[str] = [
    "function_profile",
    ...
]
```

Replace with a function that asks the dataset layer:

```python
"""Doc export utilities for emitting CodeIntel analytics as JSONL or Parquet artifacts."""

from __future__ import annotations

from typing import Iterable

from codeintel.storage.datasets import JSON_SCHEMA_BY_DATASET_NAME

def default_validation_schemas() -> list[str]:
    """
    Return the set of dataset names that should be validated by default.

    Derived from JSON_SCHEMA_BY_DATASET_NAME in the dataset contract layer.
    """
    # Preserve deterministic ordering for predictable CLI behavior.
    return sorted(JSON_SCHEMA_BY_DATASET_NAME.keys())
```

If you want to keep a constant for backwards compatibility, you can:

```python
DEFAULT_VALIDATION_SCHEMAS: list[str] = default_validation_schemas()
```

**File:** `pipeline/pipeline/export/export_jsonl.py`

Change the import:

```python
# Old:
from codeintel.pipeline.export import DEFAULT_VALIDATION_SCHEMAS

# New:
from codeintel.pipeline.export import DEFAULT_VALIDATION_SCHEMAS, default_validation_schemas
```

And inside the `if opts.validate_exports:` block:

```python
    if opts.validate_exports:
        # Use either explicitly requested schemas or the dataset-driven default list.
        schemas_to_run = opts.schemas or default_validation_schemas()
        for schema_name in schemas_to_run:
            matching = [p for p in written if p.name.startswith(schema_name)]
            ...
```

Small, but now **validation defaults are defined in one place** (`JSON_SCHEMA_BY_DATASET_NAME`) and consumed everywhere.

---

## Step 5 – Make metadata bootstrap a thin writer over the dataset spec

Right now `storage/storage/metadata_bootstrap.py` defines:

* `DEFAULT_JSONL_FILENAMES` and `DEFAULT_PARQUET_FILENAMES`
* `bootstrap_metadata_datasets(...)` uses those mappings plus `TABLE_SCHEMAS` and `DOCS_VIEWS` to populate `metadata.datasets`.

We don’t strictly need to move the constants, but we *can* make this file clearly a **writer** for the dataset contract rather than a second “source of truth”.

### 5.1. Move filename defaults next to the dataset contract (optional but nice)

If you want truly one home, you can:

1. Move the two constants into `storage/storage/datasets.py`:

   ```python
   DEFAULT_JSONL_FILENAMES: dict[str, str] = {
       "core.goids": "goids.jsonl",
       ...
   }

   DEFAULT_PARQUET_FILENAMES: dict[str, str] = {
       "core.goids": "goids.parquet",
       ...
   }
   ```

2. In `metadata_bootstrap.py`, replace the definitions with imports:

   ```python
   from codeintel.storage.datasets import (
       DEFAULT_JSONL_FILENAMES,
       DEFAULT_PARQUET_FILENAMES,
   )
   ```

Now filenames are part of the same contract module that knows about row models and JSON schemas.

### 5.2. Keep `bootstrap_metadata_datasets` as the DB writer

You already have (simplified):

```python
def bootstrap_metadata_datasets(
    con: DuckDBPyConnection,
    *,
    jsonl_filenames: Mapping[str, str] | None = None,
    parquet_filenames: Mapping[str, str] | None = None,
    include_views: bool = True,
) -> None:
    ...
    jsonl_mapping = dict(jsonl_filenames or DEFAULT_JSONL_FILENAMES)
    parquet_mapping = dict(parquet_filenames or DEFAULT_PARQUET_FILENAMES)

    for table_key in sorted(TABLE_SCHEMAS.keys()):
        ...
        _upsert_dataset_row(
            con,
            table_key=table_key,
            name=name,
            is_view=False,
            filenames=(jsonl_mapping.get(table_key), parquet_mapping.get(table_key)),
        )

    if include_views:
        for view_key in DOCS_VIEWS:
            ...
            _upsert_dataset_row(
                con,
                table_key=view_key,
                name=name,
                is_view=True,
                filenames=(jsonl_mapping.get(view_key), parquet_mapping.get(view_key)),
            )
```

That’s fine: this function is *the* writer that materializes the dataset contract into DuckDB’s `metadata.datasets` table. The key change in this step is simply **relocating the constants** so that datasets/filenames live alongside `DatasetSpec`.

---

## Step 6 – Surface a “describe the world” API for agents

Once `DatasetSpec` / `DatasetRegistry` are the single source of truth, you can expose ergonomic APIs that agents and tools can use.

### 6.1. Storage-level introspection

In `storage/storage/datasets.py`, add:

```python
def describe_dataset(ds: Dataset) -> dict[str, object]:
    """Return a JSON-serializable description of a dataset spec."""
    return {
        "name": ds.name,
        "table_key": ds.table_key,
        "is_view": ds.is_view,
        "schema_columns": [col.name for col in (ds.schema.columns if ds.schema else [])],
        "jsonl_filename": ds.jsonl_filename,
        "parquet_filename": ds.parquet_filename,
        "has_row_binding": ds.row_binding is not None,
        "json_schema_id": ds.json_schema_id,
        "description": ds.description,
    }
```

And in `storage/storage/registry_helpers.py` or `serving/serving/http/datasets.py`, provide:

```python
def list_dataset_specs(con: DuckDBConnection) -> list[dict[str, object]]:
    """Convenience helper to dump dataset specs for a gateway/connection."""
    from codeintel.storage.datasets import load_dataset_registry, describe_dataset

    registry = load_dataset_registry(con)
    return [describe_dataset(ds) for ds in registry.by_name.values()]
```

Now any CLI command or MCP tool can return a “catalog” of datasets for an agent to navigate.

---

## Step 7 – Tests and migration checklist

Finally, to make the refactor safe:

1. **Add unit tests for the dataset layer**

   * `tests/storage/test_datasets_contract.py`:

     * `load_dataset_registry` populates `jsonl_filename` / `parquet_filename` from `metadata.datasets`.
     * `Dataset.json_schema_id` matches `JSON_SCHEMA_BY_DATASET_NAME`.
     * `DatasetRegistry.datasets_with_json_schema()` matches `default_validation_schemas()`.

2. **Add tests for export behavior**

   * `tests/pipeline/test_export_defaults.py`:

     * `default_validation_schemas()` returns the same set of names as the JSON schema files in `config/config/schemas/export/`.
     * `export_all_jsonl` still validates the same set of schemas as before (you can assert on log output or on calls to `validate_files` via monkeypatch).

3. **Manual migration sanity check**

   * Run `bootstrap_metadata_datasets` on an existing `codeintel.duckdb`.
   * Confirm `SELECT * FROM metadata.datasets` matches expectations for:

     * `table_key`, `name`, `is_view`, `jsonl_filename`, `parquet_filename`.
   * Run `run_validated_exports` and confirm:

     * Same set of JSONL / Parquet files produced.
     * Same validation errors (or lack thereof).

---

## Summary

Concretely, the work breaks down to:

1. **Treat `storage.storage.datasets.Dataset` as the canonical `DatasetSpec`** and extend it slightly (description, JSON schema id, helpers).
2. **Centralize JSON Schema mapping** (`JSON_SCHEMA_BY_DATASET_NAME`) next to the dataset contract and derive `DEFAULT_VALIDATION_SCHEMAS` from it.
3. **(Optionally) Move default filename constants into `storage.storage.datasets`**, so filenames, row models, and schemas all live together.
4. **Keep `metadata_bootstrap.bootstrap_metadata_datasets` as the single DB writer**, populating `metadata.datasets` from the spec.
5. **Refactor export code to depend on the dataset contract** for both dataset selection and validation defaults.
6. **Expose introspection helpers** so agents can enumerate datasets and their properties.

If you’d like, I can next turn this into an ordered “patch plan” with bullet-point diffs per file (including suggested names for tests), so your implementation agent can apply them one file at a time.

# ordered “patch plan” with bullet-point diffs per file #

Perfect, let’s turn that into a “apply-these-in-order” patch plan your implementation agent can follow.

I’ll assume paths relative to the repo root, like `storage/storage/datasets.py`.

---

## Step 0 – Mental model for the refactor

**Design invariant:**
`storage.storage.datasets.Dataset` (aka `DatasetSpec`) is *the* canonical contract for datasets:

* Names / table keys
* Filenames (JSONL / Parquet)
* Row bindings
* JSON Schema IDs (for export validation)
* “Describe the world” introspection

Everything else (bootstrap, exports, validation) consumes that contract.

---

## Step 1 – Extend the dataset contract in `storage/storage/datasets.py`

**File:** `storage/storage/datasets.py`

### 1.1 Add a `DatasetSpec` alias

Right after the `Dataset` dataclass (or immediately after its definition):

```python
@dataclass(frozen=True)
class Dataset:
    """Metadata describing a logical dataset backed by a DuckDB table or view."""
    table_key: str
    name: str
    schema: TableSchema | None
    row_binding: RowBinding | None = None
    jsonl_filename: str | None = None
    parquet_filename: str | None = None
    is_view: bool = False
    # new fields:
    description: str | None = None
    json_schema_id: str | None = None


# New alias for clarity
DatasetSpec = Dataset
"""Alias for the canonical dataset metadata contract type."""
```

> **Diff bullets**
>
> * [ ] Extend `Dataset` with `description` and `json_schema_id` fields (both default `None`).
> * [ ] Add `DatasetSpec = Dataset` alias + short docstring.

### 1.2 Add row-binding helpers on `Dataset`

Still in `Dataset`:

```python
@dataclass(frozen=True)
class Dataset:
    ...
    json_schema_id: str | None = None

    def has_row_binding(self) -> bool:
        """Return True when this dataset has a TypedDict row binding."""
        return self.row_binding is not None

    def require_row_binding(self) -> RowBinding:
        """
        Return the row binding or raise a clear error if missing.

        Raises
        ------
        KeyError
            If no row binding is configured for this dataset.
        """
        if self.row_binding is None:
            message = f"Dataset {self.name} ({self.table_key}) has no row binding"
            raise KeyError(message)
        return self.row_binding
```

> **Diff bullets**
>
> * [ ] Add `has_row_binding()` helper.
> * [ ] Add `require_row_binding()` that raises `KeyError` with a clear message if binding is missing.

### 1.3 Define JSON schema metadata (dataset → schema id)

Near `ROW_BINDINGS_BY_TABLE_KEY`, add a mapping from *dataset name* → JSON schema filename (without `.json`):

```python
JSON_SCHEMA_BY_DATASET_NAME: dict[str, str] = {
    # Profiles
    "function_profile": "function_profile",
    "file_profile": "file_profile",
    "module_profile": "module_profile",
    # Graph edges
    "call_graph_edges": "call_graph_edges",
    "symbol_use_edges": "symbol_use_edges",
    "test_coverage_edges": "test_coverage_edges",
    # Tests
    "test_profile": "test_profile",
    "behavioral_coverage": "behavioral_coverage",
    # Data models
    "data_model_fields": "data_model_fields",
    "data_model_relationships": "data_model_relationships",
}
```

> **Diff bullets**
>
> * [ ] Add `JSON_SCHEMA_BY_DATASET_NAME` mapping in `datasets.py`.
> * [ ] Ensure keys match the JSON schemas under `config/config/schemas/export/*.json` and your existing `DEFAULT_VALIDATION_SCHEMAS` list.

### 1.4 Wire JSON schema IDs into `load_dataset_registry`

In `load_dataset_registry`’s loop:

Current (simplified):

```python
for table_key, name, is_view, jsonl_filename, parquet_filename in rows:
    schema: TableSchema | None = None if is_view else TABLE_SCHEMAS.get(table_key)
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
```

Change to:

```python
for table_key, name, is_view, jsonl_filename, parquet_filename in rows:
    schema: TableSchema | None = None if is_view else TABLE_SCHEMAS.get(table_key)
    row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(table_key)
    json_schema_id = JSON_SCHEMA_BY_DATASET_NAME.get(name)

    ds = Dataset(
        table_key=table_key,
        name=name,
        schema=schema,
        row_binding=row_binding,
        jsonl_filename=jsonl_filename,
        parquet_filename=parquet_filename,
        is_view=bool(is_view),
        json_schema_id=json_schema_id,
    )
```

> **Diff bullets**
>
> * [ ] Look up `json_schema_id` from `JSON_SCHEMA_BY_DATASET_NAME` using dataset `name`.
> * [ ] Pass `json_schema_id` into `Dataset(...)`.

### 1.5 Add `datasets_with_json_schema()` on `DatasetRegistry`

In `DatasetRegistry`:

```python
@dataclass(frozen=True)
class DatasetRegistry:
    """In-memory view of metadata.datasets plus Python row bindings."""

    by_name: Mapping[str, Dataset]
    by_table_key: Mapping[str, Dataset]
    jsonl_datasets: Mapping[str, str]
    parquet_datasets: Mapping[str, str]

    @property
    def all_datasets(self) -> tuple[str, ...]:
        """Return all dataset names."""
        return tuple(self.by_name.keys())

    def datasets_with_json_schema(self) -> tuple[str, ...]:
        """Return dataset names that have JSON Schema validation configured."""
        return tuple(
            name for name, ds in self.by_name.items() if ds.json_schema_id is not None
        )

    def resolve_table_key(self, name: str) -> str:
        ...
```

> **Diff bullets**
>
> * [ ] Add `datasets_with_json_schema()` helper on `DatasetRegistry`.

### 1.6 Add introspection helpers (`describe_dataset` / `list_dataset_specs`)

Near the bottom of `datasets.py`, after `dataset_for_table(...)`:

```python
def describe_dataset(ds: Dataset) -> dict[str, object]:
    """Return a JSON-serializable description of a dataset spec."""
    return {
        "name": ds.name,
        "table_key": ds.table_key,
        "is_view": ds.is_view,
        "schema_columns": (
            [col.name for col in ds.schema.columns] if ds.schema is not None else []
        ),
        "jsonl_filename": ds.jsonl_filename,
        "parquet_filename": ds.parquet_filename,
        "has_row_binding": ds.row_binding is not None,
        "json_schema_id": ds.json_schema_id,
        "description": ds.description,
    }


def list_dataset_specs(registry: DatasetRegistry) -> list[dict[str, object]]:
    """Serialize all dataset specs from a DatasetRegistry."""
    return [describe_dataset(ds) for ds in registry.by_name.values()]
```

> **Diff bullets**
>
> * [ ] Add `describe_dataset()` returning a JSON-serializable dict.
> * [ ] Add `list_dataset_specs()` to dump all specs given a `DatasetRegistry`.

(We’ll use these later if you want to expose a “catalog” endpoint, but they’re harmless and immediately useful to agents.)

---

## Step 2 – Move default filenames into the contract layer

**File:** `storage/storage/metadata_bootstrap.py`
**File (new home for constants):** `storage/storage/datasets.py`

### 2.1 Move `DEFAULT_JSONL_FILENAMES` / `DEFAULT_PARQUET_FILENAMES` into `datasets.py`

In `metadata_bootstrap.py`, these constants currently live at the top.

**Change:**

* Cut the two big dicts:

```python
DEFAULT_JSONL_FILENAMES: dict[str, str] = { ... }

DEFAULT_PARQUET_FILENAMES: dict[str, str] = { ... }
```

* Paste them into `storage/storage/datasets.py`, near `ROW_BINDINGS_BY_TABLE_KEY` and `JSON_SCHEMA_BY_DATASET_NAME`, e.g.:

```python
DEFAULT_JSONL_FILENAMES: dict[str, str] = {
    # GOIDs / crosswalk
    "core.goids": "goids.jsonl",
    ...
}

DEFAULT_PARQUET_FILENAMES: dict[str, str] = {
    # GOIDs / crosswalk
    "core.goids": "goids.parquet",
    ...
}
```

> **Diff bullets**
>
> * [ ] Physically relocate `DEFAULT_JSONL_FILENAMES` and `DEFAULT_PARQUET_FILENAMES` definitions from `metadata_bootstrap.py` into `datasets.py`.

### 2.2 Re-import constants in `metadata_bootstrap.py`

At the top of `metadata_bootstrap.py`, add:

```python
from codeintel.storage.datasets import (
    DEFAULT_JSONL_FILENAMES,
    DEFAULT_PARQUET_FILENAMES,
)
```

…and delete the old in-file definitions.

The body of `bootstrap_metadata_datasets` stays the same:

```python
jsonl_mapping = dict(jsonl_filenames or DEFAULT_JSONL_FILENAMES)
parquet_mapping = dict(parquet_filenames or DEFAULT_PARQUET_FILENAMES)
```

> **Diff bullets**
>
> * [ ] Import filename maps from `codeintel.storage.datasets` instead of defining them locally.
> * [ ] Leave `bootstrap_metadata_datasets` behavior unchanged (just a consumer now).

---

## Step 3 – Make `DEFAULT_VALIDATION_SCHEMAS` derived from the contract

**File:** `pipeline/pipeline/export/__init__.py`

Replace the current constant-only file:

```python
"""Doc export utilities for emitting CodeIntel analytics as JSONL or Parquet artifacts."""

DEFAULT_VALIDATION_SCHEMAS: list[str] = [
    "function_profile",
    ...
]
```

with:

```python
"""Doc export utilities for emitting CodeIntel analytics as JSONL or Parquet artifacts."""

from __future__ import annotations

from typing import Iterable

from codeintel.storage.datasets import JSON_SCHEMA_BY_DATASET_NAME

def default_validation_schemas() -> list[str]:
    """
    Return the set of dataset names that should be validated by default.

    Derived from JSON_SCHEMA_BY_DATASET_NAME in the dataset contract layer.
    """
    # Use sorted order for deterministic CLI behavior.
    return sorted(JSON_SCHEMA_BY_DATASET_NAME.keys())


# Backwards-compatible constant; prefer calling default_validation_schemas().
DEFAULT_VALIDATION_SCHEMAS: list[str] = default_validation_schemas()
```

> **Diff bullets**
>
> * [ ] Add `default_validation_schemas()` that inspects `JSON_SCHEMA_BY_DATASET_NAME`.
> * [ ] Keep `DEFAULT_VALIDATION_SCHEMAS` as a constant alias to `default_validation_schemas()` for backwards compatibility.

---

## Step 4 – Update JSONL exporter to use `default_validation_schemas()`

**File:** `pipeline/pipeline/export/export_jsonl.py`

### 4.1 Update imports

At the top:

```python
from codeintel.pipeline.export import DEFAULT_VALIDATION_SCHEMAS
```

Change to:

```python
from codeintel.pipeline.export import (
    DEFAULT_VALIDATION_SCHEMAS,  # optional; can eventually be removed
    default_validation_schemas,
)
```

(You can drop the constant import entirely if nothing else uses it in this module.)

### 4.2 Use validation schemas from the dataset contract

In the section near the bottom:

Current:

```python
    if opts.validate_exports:
        for schema_name in opts.schemas or DEFAULT_VALIDATION_SCHEMAS:
            matching = [p for p in written if p.name.startswith(schema_name)]
            ...
```

Change to:

```python
    if opts.validate_exports:
        schema_names = opts.schemas or default_validation_schemas()
        for schema_name in schema_names:
            matching = [p for p in written if p.name.startswith(schema_name)]
            if not matching:
                continue
            if validate_files(schema_name, matching) != 0:
                ...
```

> **Diff bullets**
>
> * [ ] Import and call `default_validation_schemas()` instead of using the hard-coded list.
> * [ ] Keep behavior identical: if user provided `opts.schemas`, use those; else use default-validation set.

---

## Step 5 – Update Parquet exporter to use `default_validation_schemas()`

**File:** `pipeline/pipeline/export/export_parquet.py`

### 5.1 Update imports

At the top:

```python
from codeintel.pipeline.export import DEFAULT_VALIDATION_SCHEMAS
```

Change to:

```python
from codeintel.pipeline.export import (
    DEFAULT_VALIDATION_SCHEMAS,  # optional
    default_validation_schemas,
)
```

### 5.2 Switch validation logic

Current:

```python
    if opts.validate_exports:
        schema_list = opts.schemas or DEFAULT_VALIDATION_SCHEMAS
        for schema_name in schema_list:
            matching = [p for p in written if p.name.startswith(schema_name)]
            ...
```

Change to:

```python
    if opts.validate_exports:
        schema_list = opts.schemas or default_validation_schemas()
        for schema_name in schema_list:
            matching = [p for p in written if p.name.startswith(schema_name)]
            ...
```

> **Diff bullets**
>
> * [ ] Same change as JSONL exporter: use dataset-driven default schema list.

---

## Step 6 – (Optional but nice) Storage-level “catalog” helper

**File:** `storage/storage/registry_helpers.py`

If you want a one-stop “tell me everything you know about datasets” API for agents, you can wrap `load_dataset_registry()` + `list_dataset_specs()`.

At the top, extend imports:

```python
from codeintel.storage.datasets import (
    Dataset,
    load_dataset_registry as _load_dataset_registry,
    list_dataset_specs,
)
```

Add a helper function:

```python
def describe_all_datasets(con: DuckDBPyConnection) -> list[dict[str, object]]:
    """
    Return a JSON-serializable description of all dataset specs for this database.

    Useful for diagnostic tools and agent surfaces.
    """
    ds_registry = _load_dataset_registry(con)
    return list_dataset_specs(ds_registry)
```

> **Diff bullets**
>
> * [ ] Import `list_dataset_specs` from `datasets.py`.
> * [ ] Add `describe_all_datasets(con)` wrapper to return a catalog-style list.

(This is optional; it doesn’t affect the core refactor, but it’s a nice payoff.)

---

## Step 7 – Tests to lock in the contract

### 7.1 Storage-level tests: `storage/storage/tests/test_datasets_contract.py`

**New file:** `storage/storage/tests/test_datasets_contract.py` (create the `tests/` package under `storage/storage`).

Example:

```python
from __future__ import annotations

import duckdb

from codeintel.storage.datasets import (
    JSON_SCHEMA_BY_DATASET_NAME,
    Dataset,
    RowBinding,
    describe_dataset,
    load_dataset_registry,
)
from codeintel.storage.metadata_bootstrap import bootstrap_metadata_datasets
from codeintel.storage.schemas import apply_all_schemas


def _build_test_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    apply_all_schemas(con)
    bootstrap_metadata_datasets(con)
    return con


def test_json_schema_ids_attached_to_datasets() -> None:
    con = _build_test_connection()
    registry = load_dataset_registry(con)

    names_with_schema = set(registry.datasets_with_json_schema())
    assert names_with_schema == set(JSON_SCHEMA_BY_DATASET_NAME.keys())


def test_require_row_binding_behavior() -> None:
    dummy_binding = RowBinding(row_type=dict, to_tuple=lambda row: ())
    ds_with = Dataset(
        table_key="dummy.table",
        name="dummy",
        schema=None,
        row_binding=dummy_binding,
    )
    assert ds_with.has_row_binding() is True
    assert ds_with.require_row_binding() is dummy_binding

    ds_without = Dataset(
        table_key="dummy2.table",
        name="dummy2",
        schema=None,
    )
    assert ds_without.has_row_binding() is False
    try:
        ds_without.require_row_binding()
    except KeyError:
        pass
    else:
        raise AssertionError("require_row_binding() should raise KeyError when missing")


def test_describe_dataset_shape() -> None:
    ds = Dataset(
        table_key="analytics.function_profile",
        name="function_profile",
        schema=None,
        jsonl_filename="function_profile.jsonl",
        parquet_filename="function_profile.parquet",
        json_schema_id="function_profile",
        description="Function-level profile for docs and risk scoring.",
    )
    desc = describe_dataset(ds)
    assert desc["name"] == "function_profile"
    assert desc["table_key"] == "analytics.function_profile"
    assert desc["json_schema_id"] == "function_profile"
    assert desc["description"].startswith("Function-level")
```

> **Test bullets**
>
> * [ ] `test_json_schema_ids_attached_to_datasets` – ensure DB-backed registry aligns with `JSON_SCHEMA_BY_DATASET_NAME`.
> * [ ] `test_require_row_binding_behavior` – exercise `has_row_binding` / `require_row_binding`.
> * [ ] `test_describe_dataset_shape` – basic sanity on `describe_dataset` output.

### 7.2 Export defaults tests: `pipeline/pipeline/export/tests/test_export_defaults.py`

**New file:** `pipeline/pipeline/export/tests/test_export_defaults.py` (create package if needed).

```python
from __future__ import annotations

from codeintel.pipeline.export import (
    DEFAULT_VALIDATION_SCHEMAS,
    default_validation_schemas,
)
from codeintel.storage.datasets import JSON_SCHEMA_BY_DATASET_NAME


def test_default_validation_schemas_match_dataset_contract() -> None:
    # The dynamic function should mirror the dataset-level mapping.
    expected = sorted(JSON_SCHEMA_BY_DATASET_NAME.keys())
    assert sorted(default_validation_schemas()) == expected

    # The constant should remain in sync as long as it exists.
    assert sorted(DEFAULT_VALIDATION_SCHEMAS) == expected
```

> **Test bullets**
>
> * [ ] Confirm `default_validation_schemas()` uses `JSON_SCHEMA_BY_DATASET_NAME`.
> * [ ] Confirm `DEFAULT_VALIDATION_SCHEMAS` is still consistent (until you decide to remove it).

---

## Implementation order recap (for the agent)

1. **Modify `storage/storage/datasets.py`**

   * Extend `Dataset` (fields + methods) and add `DatasetSpec`.
   * Add `JSON_SCHEMA_BY_DATASET_NAME`, `DEFAULT_*_FILENAMES` if moved, and helpers.
   * Update `load_dataset_registry` to set `json_schema_id`.

2. **Move filename constants** from `metadata_bootstrap.py` into `datasets.py` and re-import them.

3. **Update pipeline export init** to define `default_validation_schemas()` + constant alias.

4. **Update both exporters** (`export_jsonl.py`, `export_parquet.py`) to call `default_validation_schemas()`.

5. **(Optional) Add `describe_all_datasets`** to `storage/storage/registry_helpers.py`.

6. **Add tests** in `storage/storage/tests` and `pipeline/pipeline/export/tests`.

If you’d like, the next thing we can do is zoom in on just **Step 1** or **Step 2** and write out a PR-style diff (per file) exactly as you’d want to paste into a patch.


# zoom in on step 1 with PR-style diffs #

Awesome, let’s zoom in on **Step 1** and turn it into a PR-style patch for **`storage/storage/datasets.py`**.

I’ll give you a single unified diff you can adapt/apply. Context lines assume the current structure (RowBinding → ROW_BINDINGS_BY_TABLE_KEY → Dataset → DatasetRegistry → helpers). If names or ordering differ slightly in your local repo, you can adjust, but the *intent* and code blocks should be directly reusable.

---

## File: `storage/storage/datasets.py`

### High-level changes

1. Extend `Dataset` to be the full “dataset spec”:

   * Add `description: str | None` and `json_schema_id: str | None`.
   * Add helper methods `has_row_binding()` and `require_row_binding()`.
   * Introduce alias `DatasetSpec = Dataset`.
2. Add `JSON_SCHEMA_BY_DATASET_NAME` mapping.
3. Update `load_dataset_registry(...)` to populate `json_schema_id` from that mapping.
4. Add `datasets_with_json_schema()` to `DatasetRegistry`.
5. Add `describe_dataset(...)` and `list_dataset_specs(...)` helpers for introspection.

### Unified diff

```diff
diff --git a/storage/storage/datasets.py b/storage/storage/datasets.py
index XXXXXXX..YYYYYYY 100644
--- a/storage/storage/datasets.py
+++ b/storage/storage/datasets.py
@@ -1,16 +1,20 @@
 """Dataset metadata registry backed by DuckDB's metadata.datasets table."""
 
 from __future__ import annotations
 
-from collections.abc import Callable, Mapping
+from collections.abc import Callable, Mapping
 from dataclasses import dataclass
 from typing import cast
 
 from duckdb import DuckDBPyConnection
 
 from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema
 from codeintel.storage import rows as row_models
 
 RowToTuple = Callable[[Mapping[str, object]], tuple[object, ...]]
 RowDictType = type[object]
 
@@ -X,0 +X,0 @@
+@dataclass(frozen=True)
+class RowBinding:
+    """Connect a DuckDB table key to a TypedDict row model and serializer."""
+
+    row_type: RowDictType
+    to_tuple: RowToTuple
+
+
+def _row_binding(*, row_type: RowDictType, to_tuple: RowToTuple) -> RowBinding:
+    """Convenience helper to construct row bindings with clear call sites."""
+    return RowBinding(row_type=row_type, to_tuple=to_tuple)
+
+
@@ -X,0 +X,0 @@
 ROW_BINDINGS_BY_TABLE_KEY: dict[str, RowBinding] = {
     "analytics.coverage_lines": _row_binding(
         row_type=row_models.CoverageLineRow,
         to_tuple=row_models.coverage_line_to_tuple,
     ),
@@ -X,0 +X,0 @@
     "analytics.typedness": _row_binding(
         row_type=row_models.TypednessRow,
         to_tuple=row_models.typedness_row_to_tuple,
     ),
     ...
 }
+
+# Dataset-level JSON Schema metadata.
+# Keys: Dataset logical names (Dataset.name).
+# Values: JSON Schema identifiers (filenames without .json) under
+# config/config/schemas/export/.
+JSON_SCHEMA_BY_DATASET_NAME: dict[str, str] = {
+    # Profiles
+    "function_profile": "function_profile",
+    "file_profile": "file_profile",
+    "module_profile": "module_profile",
+    # Graph edges
+    "call_graph_edges": "call_graph_edges",
+    "symbol_use_edges": "symbol_use_edges",
+    "test_coverage_edges": "test_coverage_edges",
+    # Tests
+    "test_profile": "test_profile",
+    "behavioral_coverage": "behavioral_coverage",
+    # Data models
+    "data_model_fields": "data_model_fields",
+    "data_model_relationships": "data_model_relationships",
+}
+
@@ -X,0 +X,0 @@
 @dataclass(frozen=True)
 class Dataset:
-    """Metadata describing a logical dataset backed by a DuckDB table or view."""
-
-    table_key: str
-    name: str
-    schema: TableSchema | None
-    row_binding: RowBinding | None = None
-    jsonl_filename: str | None = None
-    parquet_filename: str | None = None
-    is_view: bool = False
+    """Metadata describing a logical dataset backed by a DuckDB table or view.
+
+    Attributes
+    ----------
+    table_key
+        Fully qualified DuckDB identifier, e.g. "analytics.function_profile".
+    name
+        Logical dataset name, e.g. "function_profile".
+    schema
+        Statically defined TableSchema when the dataset is backed by a table;
+        None when the dataset is a view.
+    row_binding
+        Optional binding to a TypedDict row model and serializer.
+    jsonl_filename
+        Default filename for JSONL exports (may be None when not exported).
+    parquet_filename
+        Default filename for Parquet exports (may be None when not exported).
+    is_view
+        True when this dataset is a docs.* view instead of a base table.
+    description
+        Optional human-readable description of the dataset’s purpose.
+    json_schema_id
+        Optional JSON Schema identifier (without .json) used for export validation.
+    """
+
+    table_key: str
+    name: str
+    schema: TableSchema | None
+    row_binding: RowBinding | None = None
+    jsonl_filename: str | None = None
+    parquet_filename: str | None = None
+    is_view: bool = False
+    description: str | None = None
+    json_schema_id: str | None = None
+
+    def has_row_binding(self) -> bool:
+        """Return True when this dataset has a TypedDict row binding."""
+        return self.row_binding is not None
+
+    def require_row_binding(self) -> RowBinding:
+        """
+        Return the row binding or raise a clear error if missing.
+
+        Raises
+        ------
+        KeyError
+            If no row binding is configured for this dataset.
+        """
+        if self.row_binding is None:
+            message = f"Dataset {self.name} ({self.table_key}) has no row binding"
+            raise KeyError(message)
+        return self.row_binding
+
+
+# Backwards-compatible alias for the canonical dataset contract type.
+DatasetSpec = Dataset
+
@@ -X,0 +X,0 @@
 @dataclass(frozen=True)
 class DatasetRegistry:
     """In-memory view of metadata.datasets plus Python row bindings."""
 
     by_name: Mapping[str, Dataset]
     by_table_key: Mapping[str, Dataset]
     jsonl_datasets: Mapping[str, str]
     parquet_datasets: Mapping[str, str]
+
+    @property
+    def all_datasets(self) -> tuple[str, ...]:
+        """Return all dataset names."""
+        return tuple(self.by_name.keys())
+
+    def datasets_with_json_schema(self) -> tuple[str, ...]:
+        """Return dataset names that have JSON Schema validation configured."""
+        return tuple(
+            name for name, ds in self.by_name.items() if ds.json_schema_id is not None
+        )
@@ -X,0 +X,0 @@
-def load_dataset_registry(con: DuckDBPyConnection) -> DatasetRegistry:
+def load_dataset_registry(con: DuckDBPyConnection) -> DatasetRegistry:
     """
     Build an in-memory DatasetRegistry from metadata.datasets plus row bindings.
 
@@ -X,0 +X,0 @@
-    for table_key, name, is_view, jsonl_filename, parquet_filename in rows:
-        schema: TableSchema | None = None if is_view else TABLE_SCHEMAS.get(table_key)
-        row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(table_key)
-        ds = Dataset(
-            table_key=table_key,
-            name=name,
-            schema=schema,
-            row_binding=row_binding,
-            jsonl_filename=jsonl_filename,
-            parquet_filename=parquet_filename,
-            is_view=bool(is_view),
-        )
+    for table_key, name, is_view, jsonl_filename, parquet_filename in rows:
+        schema: TableSchema | None = None if is_view else TABLE_SCHEMAS.get(table_key)
+        row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(table_key)
+        json_schema_id = JSON_SCHEMA_BY_DATASET_NAME.get(name)
+
+        ds = Dataset(
+            table_key=table_key,
+            name=name,
+            schema=schema,
+            row_binding=row_binding,
+            jsonl_filename=jsonl_filename,
+            parquet_filename=parquet_filename,
+            is_view=bool(is_view),
+            json_schema_id=json_schema_id,
+        )
         by_name[name] = ds
         by_table_key[table_key] = ds
         if jsonl_filename:
             jsonl_mapping[name] = jsonl_filename
         if parquet_filename:
             parquet_mapping[name] = parquet_filename
 
@@ -X,0 +X,0 @@
 def dataset_for_table(registry: DatasetRegistry, table_key: str) -> Dataset:
@@ -X,0 +X,0 @@
     ds = registry.by_table_key.get(table_key)
     if ds is None:
         message = f"Unknown dataset table key: {table_key}"
         raise KeyError(message)
     return ds
+
+
+def describe_dataset(ds: Dataset) -> dict[str, object]:
+    """Return a JSON-serializable description of a dataset spec."""
+    return {
+        "name": ds.name,
+        "table_key": ds.table_key,
+        "is_view": ds.is_view,
+        "schema_columns": (
+            [col.name for col in ds.schema.columns] if ds.schema is not None else []
+        ),
+        "jsonl_filename": ds.jsonl_filename,
+        "parquet_filename": ds.parquet_filename,
+        "has_row_binding": ds.row_binding is not None,
+        "json_schema_id": ds.json_schema_id,
+        "description": ds.description,
+    }
+
+
+def list_dataset_specs(registry: DatasetRegistry) -> list[dict[str, object]]:
+    """Serialize all dataset specs from a DatasetRegistry."""
+    return [describe_dataset(ds) for ds in registry.by_name.values()]
```

> **Note:**
>
> * The `RowBinding` and `_row_binding()` definitions are included in the diff for context; if you already have them defined exactly like that, you don’t need to re-add them—keep your existing ones and only splice in the *new* bits (JSON schema mapping, extra fields, methods, etc.).
> * Replace `index XXXXXXX..YYYYYYY` and `@@ -X,0 +X,0 @@` with whatever your diff tool generates; they’re placeholders here so I can show the structure in one block.

---

If you’d like, next I can do the same “PR-style diff” treatment for **Step 2** (moving the filename maps into `datasets.py` and updating `metadata_bootstrap.py`), so you’ve got a neat sequence of patches to hand directly to your implementation agent.

# zoom in on step 2 with PR-style diffs #

Love it, let’s lock in **Step 2** with concrete, per-file patches.

We’ll:

1. Move `DEFAULT_JSONL_FILENAMES` and `DEFAULT_PARQUET_FILENAMES` into `storage/storage/datasets.py`.
2. Update `storage/storage/metadata_bootstrap.py` to import them instead of defining them.

I’ll write the diffs in a PR-style way, but I won’t expand the full giant dicts; instead I’ll be explicit about “copy this block unchanged here” so your implementation agent can follow it cleanly.

---

## 1) Move filename maps into `storage/storage/datasets.py`

### File: `storage/storage/datasets.py`

**Goal:** make filenames part of the dataset contract module, alongside `ROW_BINDINGS_BY_TABLE_KEY`, `JSON_SCHEMA_BY_DATASET_NAME`, etc.

Place these definitions **near the top-level constants**, close to `ROW_BINDINGS_BY_TABLE_KEY` and `JSON_SCHEMA_BY_DATASET_NAME` (order is up to you, but keeping all contract-ish metadata together is ideal).

```diff
diff --git a/storage/storage/datasets.py b/storage/storage/datasets.py
index XXXXXXX..YYYYYYY 100644
--- a/storage/storage/datasets.py
+++ b/storage/storage/datasets.py
@@ -1,16 +1,20 @@
 """Dataset metadata registry backed by DuckDB's metadata.datasets table."""
 
 from __future__ import annotations
 
-from collections.abc import Callable, Mapping
-from dataclasses import dataclass
-from typing import cast
+from collections.abc import Callable, Mapping
+from dataclasses import dataclass
+from typing import cast
 
 from duckdb import DuckDBPyConnection
 
 from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema
 from codeintel.storage import rows as row_models
 
 RowToTuple = Callable[[Mapping[str, object]], tuple[object, ...]]
 RowDictType = type[object]
@@ -XX,6 +XX,52 @@ ROW_BINDINGS_BY_TABLE_KEY: dict[str, RowBinding] = {
     "analytics.coverage_lines": _row_binding(
         row_type=row_models.CoverageLineRow,
         to_tuple=row_models.coverage_line_to_tuple,
     ),
     ...
 }
+
+# ---------------------------------------------------------------------------
+# Default export filenames (JSONL / Parquet)
+# ---------------------------------------------------------------------------
+
+# NOTE:
+# The two mappings below should be copied verbatim from the current
+# definitions of DEFAULT_JSONL_FILENAMES and DEFAULT_PARQUET_FILENAMES in
+# storage/storage/metadata_bootstrap.py. Content is unchanged; only the
+# location moves so filenames live alongside the dataset contract.
+
+DEFAULT_JSONL_FILENAMES: dict[str, str] = {
+    # GOIDs / crosswalk
+    "core.goids": "goids.jsonl",
+    "core.goid_crosswalk": "goid_crosswalk.jsonl",
+    # Call graph
+    "graph.call_graph_nodes": "call_graph_nodes.jsonl",
+    "graph.call_graph_edges": "call_graph_edges.jsonl",
+    # CFG / DFG
+    "graph.cfg_blocks": "cfg_blocks.jsonl",
+    "graph.cfg_edges": "cfg_edges.jsonl",
+    "graph.dfg_edges": "dfg_edges.jsonl",
+    # Import / symbol uses
+    "graph.import_graph_edges": "import_graph_edges.jsonl",
+    "graph.symbol_use_edges": "symbol_use_edges.jsonl",
+    # AST / CST
+    "core.ast_nodes": "ast_nodes.jsonl",
+    "core.ast_metrics": "ast_metrics.jsonl",
+    "core.cst_nodes": "cst_nodes.jsonl",
+    "core.docstrings": "docstrings.jsonl",
+    # Modules / config / diagnostics
+    "core.modules": "modules.jsonl",
+    "analytics.config_values": "config_values.jsonl",
+    "analytics.data_models": "data_models.jsonl",
+    "analytics.data_model_fields": "data_model_fields.jsonl",
+    "analytics.data_model_relationships": "data_model_relationships.jsonl",
+    "analytics.data_model_usage": "data_model_usage.jsonl",
+    "analytics.config_data_flow": "config_data_flow.jsonl",
+    "analytics.static_diagnostics": "static_diagnostics.jsonl",
+    # AST analytics / typing
+    "analytics.hotspots": "hotspots.jsonl",
+    "analytics.typedness": "typedness.jsonl",
+    # Function analytics
+    "analytics.function_metrics": "function_metrics.jsonl",
+    "analytics.function_types": "function_types.jsonl",
+    "analytics.function_effects": "function_effects.jsonl",
+    "analytics.function_contracts": "function_contracts.jsonl",
+    "analytics.semantic_roles_functions": "semantic_roles_functions.jsonl",
+    "analytics.semantic_roles_modules": "semantic_roles_modules.jsonl",
+    # Coverage + tests
+    "analytics.coverage_lines": "coverage_lines.jsonl",
+    "analytics.coverage_functions": "coverage_functions.jsonl",
+    "analytics.test_catalog": "test_catalog.jsonl",
+    "analytics.test_coverage_edges": "test_coverage_edges.jsonl",
+    "analytics.entrypoints": "entrypoints.jsonl",
+    "analytics.entrypoint_tests": "entrypoint_tests.jsonl",
+    "analytics.external_dependencies": "external_dependencies.jsonl",
+    "analytics.external_dependency_calls": "external_dependency_calls.jsonl",
+    "analytics.graph_validation": "graph_validation.jsonl",
+    "analytics.function_validation": "function_validation.jsonl",
+    # Risk factors
+    "analytics.goid_risk_factors": "goid_risk_factors.jsonl",
+    "analytics.function_profile": "function_profile.jsonl",
+    "analytics.function_history": "function_history.jsonl",
+    "analytics.history_timeseries": "history_timeseries.jsonl",
+    "analytics.file_profile": "file_profile.jsonl",
+    "analytics.module_profile": "module_profile.jsonl",
+    "analytics.graph_metrics_functions": "graph_metrics_functions.jsonl",
+    "analytics.graph_metrics_functions_ext": "graph_metrics_functions_ext.jsonl",
+    "analytics.graph_metrics_modules": "graph_metrics_modules.jsonl",
+    "analytics.graph_metrics_modules_ext": "graph_metrics_modules_ext.jsonl",
+    "analytics.subsystem_graph_metrics": "subsystem_graph_metrics.jsonl",
+    "analytics.symbol_graph_metrics_modules": "symbol_graph_metrics_modules.jsonl",
+    "analytics.symbol_graph_metrics_functions": "symbol_graph_metrics_functions.jsonl",
+    "analytics.config_graph_metrics_keys": "config_graph_metrics_keys.jsonl",
+    "analytics.config_graph_metrics_modules": "config_graph_metrics_modules.jsonl",
+    "analytics.config_projection_key_edges": "config_projection_key_edges.jsonl",
+    "analytics.config_projection_module_edges": "config_projection_module_edges.jsonl",
+    "analytics.subsystem_agreement": "subsystem_agreement.jsonl",
+    "analytics.graph_stats": "graph_stats.jsonl",
+    "analytics.test_graph_metrics_tests": "test_graph_metrics_tests.jsonl",
+    "analytics.test_graph_metrics_functions": "test_graph_metrics_functions.jsonl",
+    "analytics.test_profile": "test_profile.jsonl",
+    "analytics.behavioral_coverage": "behavioral_coverage.jsonl",
+    "analytics.cfg_block_metrics": "cfg_block_metrics.jsonl",
+    "analytics.cfg_function_metrics": "cfg_function_metrics.jsonl",
+    "analytics.dfg_block_metrics": "dfg_block_metrics.jsonl",
+    "analytics.dfg_function_metrics": "dfg_function_metrics.jsonl",
+    "analytics.subsystems": "subsystems.jsonl",
+    "analytics.subsystem_modules": "subsystem_modules.jsonl",
+    # Docs views
+    "docs.v_validation_summary": "validation_summary.jsonl",
+}
+
+
+DEFAULT_PARQUET_FILENAMES: dict[str, str] = {
+    # GOIDs / crosswalk
+    "core.goids": "goids.parquet",
+    "core.goid_crosswalk": "goid_crosswalk.parquet",
+    # Call graph
+    "graph.call_graph_nodes": "call_graph_nodes.parquet",
+    "graph.call_graph_edges": "call_graph_edges.parquet",
+    # CFG / DFG
+    "graph.cfg_blocks": "cfg_blocks.parquet",
+    "graph.cfg_edges": "cfg_edges.parquet",
+    "graph.dfg_edges": "dfg_edges.parquet",
+    # Import / symbol uses
+    "graph.import_graph_edges": "import_graph_edges.parquet",
+    "graph.symbol_use_edges": "symbol_use_edges.parquet",
+    # AST / CST
+    "core.ast_nodes": "ast_nodes.parquet",
+    "core.ast_metrics": "ast_metrics.parquet",
+    "core.cst_nodes": "cst_nodes.parquet",
+    "core.docstrings": "docstrings.parquet",
+    # Modules / config / diagnostics
+    "core.modules": "modules.parquet",
+    "analytics.config_values": "config_values.parquet",
+    "analytics.data_models": "data_models.parquet",
+    "analytics.data_model_fields": "data_model_fields.parquet",
+    "analytics.data_model_relationships": "data_model_relationships.parquet",
+    "analytics.data_model_usage": "data_model_usage.parquet",
+    "analytics.config_data_flow": "config_data_flow.parquet",
+    "analytics.static_diagnostics": "static_diagnostics.parquet",
+    # AST analytics / typing
+    "analytics.hotspots": "hotspots.parquet",
+    "analytics.typedness": "typedness.parquet",
+    # Function analytics
+    "analytics.function_metrics": "function_metrics.parquet",
+    "analytics.function_types": "function_types.parquet",
+    "analytics.function_effects": "function_effects.parquet",
+    "analytics.function_contracts": "function_contracts.parquet",
+    "analytics.semantic_roles_functions": "semantic_roles_functions.parquet",
+    "analytics.semantic_roles_modules": "semantic_roles_modules.parquet",
+    # Coverage + tests
+    "analytics.coverage_lines": "coverage_lines.parquet",
+    "analytics.coverage_functions": "coverage_functions.parquet",
+    "analytics.test_catalog": "test_catalog.parquet",
+    "analytics.test_coverage_edges": "test_coverage_edges.parquet",
+    "analytics.entrypoints": "entrypoints.parquet",
+    "analytics.entrypoint_tests": "entrypoint_tests.parquet",
+    "analytics.external_dependencies": "external_dependencies.parquet",
+    "analytics.external_dependency_calls": "external_dependency_calls.parquet",
+    "analytics.graph_validation": "graph_validation.parquet",
+    "analytics.function_validation": "function_validation.parquet",
+    # Risk factors
+    "analytics.goid_risk_factors": "goid_risk_factors.parquet",
+    "analytics.function_profile": "function_profile.parquet",
+    "analytics.function_history": "function_history.parquet",
+    "analytics.history_timeseries": "history_timeseries.parquet",
+    "analytics.file_profile": "file_profile.parquet",
+    "analytics.module_profile": "module_profile.parquet",
+    "analytics.graph_metrics_functions": "graph_metrics_functions.parquet",
+    "analytics.graph_metrics_functions_ext": "graph_metrics_functions_ext.parquet",
+    "analytics.graph_metrics_modules": "graph_metrics_modules.parquet",
+    "analytics.graph_metrics_modules_ext": "graph_metrics_modules_ext.parquet",
+    "analytics.subsystem_graph_metrics": "subsystem_graph_metrics.parquet",
+    "analytics.symbol_graph_metrics_modules": "symbol_graph_metrics_modules.parquet",
+    "analytics.symbol_graph_metrics_functions": "symbol_graph_metrics_functions.parquet",
+    "analytics.config_graph_metrics_keys": "config_graph_metrics_keys.parquet",
+    "analytics.config_graph_metrics_modules": "config_graph_metrics_modules.parquet",
+    "analytics.config_projection_key_edges": "config_projection_key_edges.parquet",
+    "analytics.config_projection_module_edges": "config_projection_module_edges.parquet",
+    "analytics.subsystem_agreement": "subsystem_agreement.parquet",
+    "analytics.graph_stats": "graph_stats.parquet",
+    "analytics.test_graph_metrics_tests": "test_graph_metrics_tests.parquet",
+    "analytics.test_graph_metrics_functions": "test_graph_metrics_functions.parquet",
+    "analytics.test_profile": "test_profile.parquet",
+    "analytics.behavioral_coverage": "behavioral_coverage.parquet",
+    "analytics.cfg_block_metrics": "cfg_block_metrics.parquet",
+    "analytics.cfg_function_metrics": "cfg_function_metrics.parquet",
+    "analytics.dfg_block_metrics": "dfg_block_metrics.parquet",
+    "analytics.dfg_function_metrics": "dfg_function_metrics.parquet",
+    "analytics.subsystems": "subsystems.parquet",
+    "analytics.subsystem_modules": "subsystem_modules.parquet",
+    # Docs views
+    "docs.v_validation_summary": "validation_summary.parquet",
+}
```

🔧 **Implementation note:**
Above I’ve inlined the full mappings so your agent doesn’t have to infer them; but the key mental model is: **no content changes** — just move the dicts out of `metadata_bootstrap.py` into this module.

---

## 2) Update `storage/storage/metadata_bootstrap.py` to import these maps

Now that the filename maps are defined in `datasets.py`, `metadata_bootstrap.py` should just import and use them.

### File: `storage/storage/metadata_bootstrap.py`

At the top, you currently have:

```python
"""Bootstrap DuckDB metadata catalog for datasets and apply metadata macros."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

from duckdb import DuckDBPyConnection

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.views import DOCS_VIEWS

DEFAULT_JSONL_FILENAMES: dict[str, str] = {
    ...
}

DEFAULT_PARQUET_FILENAMES: dict[str, str] = {
    ...
}
```

We want to:

1. Import the constants from `codeintel.storage.datasets`.
2. Remove the local definitions.

**Diff:**

```diff
diff --git a/storage/storage/metadata_bootstrap.py b/storage/storage/metadata_bootstrap.py
index AAAABBB..CCC C 100644
--- a/storage/storage/metadata_bootstrap.py
+++ b/storage/storage/metadata_bootstrap.py
@@ -1,16 +1,20 @@
 """Bootstrap DuckDB metadata catalog for datasets and apply metadata macros."""
 
 from __future__ import annotations
 
 import hashlib
 import re
 from collections.abc import Mapping
 
 from duckdb import DuckDBPyConnection
 
 from codeintel.config.schemas.tables import TABLE_SCHEMAS
 from codeintel.storage.views import DOCS_VIEWS
+from codeintel.storage.datasets import (
+    DEFAULT_JSONL_FILENAMES,
+    DEFAULT_PARQUET_FILENAMES,
+)
 
-DEFAULT_JSONL_FILENAMES: dict[str, str] = {
-    # GOIDs / crosswalk
-    "core.goids": "goids.jsonl",
-    ...
-}
-
-DEFAULT_PARQUET_FILENAMES: dict[str, str] = {
-    # GOIDs / crosswalk
-    "core.goids": "goids.parquet",
-    ...
-}
+# DEFAULT_JSONL_FILENAMES and DEFAULT_PARQUET_FILENAMES have been moved to
+# codeintel.storage.datasets so that filename metadata lives alongside the
+# dataset contract. This module now imports and consumes them.
```

The rest of the file — especially `bootstrap_metadata_datasets(...)` — should **not change**; it still does:

```python
jsonl_mapping = dict(jsonl_filenames or DEFAULT_JSONL_FILENAMES)
parquet_mapping = dict(parquet_filenames or DEFAULT_PARQUET_FILENAMES)
```

so behavior is identical, just with a cleaner layering.

---

## Sanity checklist for your implementation agent

1. **Apply Step 1 (Dataset contract changes)** if you haven’t already:

   * Extended `Dataset`, added `DatasetSpec`, `JSON_SCHEMA_BY_DATASET_NAME`, etc.
2. **In `datasets.py`**:

   * Add `DEFAULT_JSONL_FILENAMES` and `DEFAULT_PARQUET_FILENAMES` exactly as currently defined in `metadata_bootstrap.py`.
3. **In `metadata_bootstrap.py`**:

   * Import both constants from `codeintel.storage.datasets`.
   * Delete the local definitions.
   * Leave `bootstrap_metadata_datasets` logic unchanged.
4. Run tests / quick checks:

   * Confirm `bootstrap_metadata_datasets` still populates `metadata.datasets` with the same filenames.
   * Confirm no references to `DEFAULT_JSONL_FILENAMES` / `DEFAULT_PARQUET_FILENAMES` are left outside these two modules.

If you’d like, next we can do **Step 3** in the same PR-style way: switch `pipeline/pipeline/export/__init__.py`, `export_jsonl.py`, and `export_parquet.py` over to the dataset-driven `default_validation_schemas()` path.


# zoom in on step 3 with PR-style diffs #

Nice, let’s wire up **Step 3** cleanly. We’ll touch:

* `pipeline/pipeline/export/__init__.py`
* `pipeline/pipeline/export/export_jsonl.py`
* `pipeline/pipeline/export/export_parquet.py`

So that **default validation schemas come from the dataset contract** (`JSON_SCHEMA_BY_DATASET_NAME` in `storage.storage.datasets`).

I’ll give you PR-style unified diffs with enough context for your implementation agent to apply.

---

## 1) Make `default_validation_schemas()` dataset-driven

### File: `pipeline/pipeline/export/__init__.py`

**Current:**

```python
"""Doc export utilities for emitting CodeIntel analytics as JSONL or Parquet artifacts."""

DEFAULT_VALIDATION_SCHEMAS: list[str] = [
    "function_profile",
    "file_profile",
    "module_profile",
    "call_graph_edges",
    "symbol_use_edges",
    "test_coverage_edges",
    "test_profile",
    "behavioral_coverage",
    "data_model_fields",
    "data_model_relationships",
]
```

**Goal:**

* Import `JSON_SCHEMA_BY_DATASET_NAME` from `codeintel.storage.datasets`.
* Provide a function `default_validation_schemas()` that derives the default schemas from that mapping.
* Keep a `DEFAULT_VALIDATION_SCHEMAS` constant as a backwards-compatible alias.

**Patch:**

```diff
diff --git a/pipeline/pipeline/export/__init__.py b/pipeline/pipeline/export/__init__.py
index AAAABBB..CCCCDDD 100644
--- a/pipeline/pipeline/export/__init__.py
+++ b/pipeline/pipeline/export/__init__.py
@@ -1,13 +1,26 @@
-"""Doc export utilities for emitting CodeIntel analytics as JSONL or Parquet artifacts."""
-
-DEFAULT_VALIDATION_SCHEMAS: list[str] = [
-    "function_profile",
-    "file_profile",
-    "module_profile",
-    "call_graph_edges",
-    "symbol_use_edges",
-    "test_coverage_edges",
-    "test_profile",
-    "behavioral_coverage",
-    "data_model_fields",
-    "data_model_relationships",
-]
+"""Doc export utilities for emitting CodeIntel analytics as JSONL or Parquet artifacts."""
+
+from __future__ import annotations
+
+from codeintel.storage.datasets import JSON_SCHEMA_BY_DATASET_NAME
+
+
+def default_validation_schemas() -> list[str]:
+    """
+    Return the set of dataset names that should be validated by default.
+
+    This is derived from JSON_SCHEMA_BY_DATASET_NAME in the dataset contract
+    layer, which maps dataset names to JSON Schema identifiers.
+    """
+    # Use sorted order for deterministic CLI / test behavior.
+    return sorted(JSON_SCHEMA_BY_DATASET_NAME.keys())
+
+
+# Backwards-compatible constant; prefer calling default_validation_schemas()
+# in new code.
+DEFAULT_VALIDATION_SCHEMAS: list[str] = default_validation_schemas()
```

---

## 2) Use `default_validation_schemas()` in JSONL export

### File: `pipeline/pipeline/export/export_jsonl.py`

**Key existing bits:**

Imports:

```python
from codeintel.pipeline.export import DEFAULT_VALIDATION_SCHEMAS
from codeintel.pipeline.export.manifest import write_dataset_manifest
...
from codeintel.pipeline.export.validate_exports import validate_files
```

Validation block:

```python
    if opts.validate_exports:
        for schema_name in opts.schemas or DEFAULT_VALIDATION_SCHEMAS:
            matching = [p for p in written if p.name.startswith(schema_name)]
            if not matching:
                continue
            if validate_files(schema_name, matching) != 0:
                pd = problem(
                    code="export.validation_failed",
                    title="Export validation failed",
                    detail=f"Validation failed for schema {schema_name}",
                    extras={"schema": schema_name, "files": [str(p) for p in matching]},
                )
                log_problem(log, pd)
                raise ExportError(pd)
```

**Goal:**

* Import `default_validation_schemas`.
* Change the “schemas to validate” selection to use `default_validation_schemas()` when `opts.schemas` is not provided.

**Patch:**

```diff
diff --git a/pipeline/pipeline/export/export_jsonl.py b/pipeline/pipeline/export/export_jsonl.py
index EEEEFFF..GGGHHH 100644
--- a/pipeline/pipeline/export/export_jsonl.py
+++ b/pipeline/pipeline/export/export_jsonl.py
@@ -12,7 +12,10 @@ from dataclasses import dataclass
 from datetime import UTC, datetime
 from pathlib import Path
 from time import perf_counter
 from typing import cast
 
-from codeintel.pipeline.export import DEFAULT_VALIDATION_SCHEMAS
+from codeintel.pipeline.export import (
+    DEFAULT_VALIDATION_SCHEMAS,  # retained for backwards compatibility
+    default_validation_schemas,
+)
 from codeintel.pipeline.export.manifest import write_dataset_manifest
@@ -17340,9 +17343,13 @@ def export_all_jsonl(
     )
     written.append(manifest_path)
 
     if opts.validate_exports:
-        for schema_name in opts.schemas or DEFAULT_VALIDATION_SCHEMAS:
+        # Use either explicitly requested schemas or the dataset-driven defaults.
+        schema_names = opts.schemas or default_validation_schemas()
+        for schema_name in schema_names:
             matching = [p for p in written if p.name.startswith(schema_name)]
             if not matching:
                 continue
             if validate_files(schema_name, matching) != 0:
                 pd = problem(
                     code="export.validation_failed",
                     title="Export validation failed",
                     detail=f"Validation failed for schema {schema_name}",
                     extras={"schema": schema_name, "files": [str(p) for p in matching]},
                 )
                 log_problem(log, pd)
                 raise ExportError(pd)
```

> You can optionally drop `DEFAULT_VALIDATION_SCHEMAS` from the import if nothing else in this module uses it, but leaving it for now keeps the change minimally invasive.

---

## 3) Use `default_validation_schemas()` in Parquet export

### File: `pipeline/pipeline/export/export_parquet.py`

**Key existing bits:**

Imports:

```python
from codeintel.pipeline.export import DEFAULT_VALIDATION_SCHEMAS
from codeintel.pipeline.export.export_jsonl import ExportCallOptions
from codeintel.pipeline.export.manifest import write_dataset_manifest
from codeintel.pipeline.export.validate_exports import validate_files
```

Validation block:

```python
    if opts.validate_exports:
        schema_list = opts.schemas or DEFAULT_VALIDATION_SCHEMAS
        for schema_name in schema_list:
            matching = [p for p in written if p.name.startswith(schema_name)]
            if not matching:
                continue
            exit_code = validate_files(schema_name, matching)
            if exit_code != 0:
                pd = problem(
                    code="export.validation_failed",
                    title="Export validation failed",
                    detail=f"Validation failed for schema {schema_name}",
                    extras={"schema": schema_name, "files": [str(p) for p in matching]},
                )
                log_problem(log, pd)
                raise ExportError(pd)
```

**Goal:**

Same as JSONL: use dataset-driven defaults.

**Patch:**

```diff
diff --git a/pipeline/pipeline/export/export_parquet.py b/pipeline/pipeline/export/export_parquet.py
index IIIJJJ..KKKLLL 100644
--- a/pipeline/pipeline/export/export_parquet.py
+++ b/pipeline/pipeline/export/export_parquet.py
@@ -11,8 +11,11 @@ from dataclasses import dataclass
 from pathlib import Path
 from time import perf_counter
 
-from codeintel.pipeline.export import DEFAULT_VALIDATION_SCHEMAS
+from codeintel.pipeline.export import (
+    DEFAULT_VALIDATION_SCHEMAS,  # retained for backwards compatibility
+    default_validation_schemas,
+)
 from codeintel.pipeline.export.export_jsonl import ExportCallOptions
 from codeintel.pipeline.export.manifest import write_dataset_manifest
 from codeintel.pipeline.export.validate_exports import validate_files
@@ -13738,8 +13741,12 @@ def export_all_parquet(
     )
     written.append(manifest_path)
 
     if opts.validate_exports:
-        schema_list = opts.schemas or DEFAULT_VALIDATION_SCHEMAS
-        for schema_name in schema_list:
+        # Use either explicitly requested schemas or the dataset-driven defaults.
+        schema_names = opts.schemas or default_validation_schemas()
+        for schema_name in schema_names:
             matching = [p for p in written if p.name.startswith(schema_name)]
             if not matching:
                 continue
             exit_code = validate_files(schema_name, matching)
             if exit_code != 0:
                 pd = problem(
                     code="export.validation_failed",
                     title="Export validation failed",
                     detail=f"Validation failed for schema {schema_name}",
                     extras={"schema": schema_name, "files": [str(p) for p in matching]},
                 )
                 log_problem(log, pd)
                 raise ExportError(pd)
```

---

## Quick sanity checklist for your agent

1. **Update `pipeline/pipeline/export/__init__.py`:**

   * Add `from __future__ import annotations`.
   * Import `JSON_SCHEMA_BY_DATASET_NAME` from `codeintel.storage.datasets`.
   * Add `default_validation_schemas()` returning `sorted(JSON_SCHEMA_BY_DATASET_NAME.keys())`.
   * Replace the hard-coded list with `DEFAULT_VALIDATION_SCHEMAS = default_validation_schemas()`.

2. **Update `export_jsonl.py`:**

   * Import `default_validation_schemas` alongside `DEFAULT_VALIDATION_SCHEMAS`.
   * Change `opts.schemas or DEFAULT_VALIDATION_SCHEMAS` to `opts.schemas or default_validation_schemas()`.

3. **Update `export_parquet.py`:**

   * Same import change.
   * Same replacement in the validation block.

4. **Run tests / smoke checks:**

   * Run whatever export CLI you normally use (e.g., JSONL + Parquet for a sample snapshot) and confirm:

     * The same schemas are validated as before.
     * Validation behavior and error messages are unchanged.


