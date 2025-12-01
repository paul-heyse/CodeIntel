Here’s a concrete, step-by-step implementation plan for **Epic 1 – Single `DatasetContract` SSoT** across **config, storage, and serving**, with code sketches you can drop in and adapt.

I’ll structure it as:

1. New `config.dataset_contract` module (canonical contract + registry).
2. Refactor `storage.datasets` to be a DB-backed view of those contracts.
3. Refactor `metadata_bootstrap`, `contract_validation`, `conformance`, `schema_generation`.
4. Refactor `serving.backend.datasets` and `serving.registry`.
5. Tests & migration checklist.

---

## 1. New SSoT module: `codeintel.config.dataset_contract`

### 1.1. Create `config/dataset_contract.py`

Goal: one place that knows, for *every* dataset:

* logical name
* DuckDB table/view key
* `TableSchema` (when it’s a table)
* TypedDict row model + serializer
* JSON Schema id
* default filenames
* dependencies, ownership, tags, etc.

**New module skeleton:**

```python
# src/codeintel/config/dataset_contract.py
"""Single source of truth for dataset contracts (tables + docs views)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema
from codeintel.storage import rows as row_models
from codeintel.storage.views import DERIVED_DOCS_VIEWS

RowToTuple = Callable[[Mapping[str, object]], tuple[object, ...]]
RowDictType = type[object]


@dataclass(frozen=True)
class RowBinding:
    """Connect a DuckDB table key to a TypedDict row model and serializer."""

    row_type: RowDictType
    to_tuple: RowToTuple


@dataclass(frozen=True)
class DatasetContract:
    """
    Canonical contract for a logical dataset backed by a DuckDB table or docs view.

    This type is the single source of truth for dataset metadata across storage,
    export, and serving.
    """

    # Identity + physical location
    name: str                     # "function_profile" / "call_graph_edges"
    table_key: str                # "analytics.function_profile", "graph.call_graph_edges"
    schema: TableSchema | None    # None for docs.* views

    # Row model + serializers
    row_binding: RowBinding | None

    # Export + validation
    json_schema_id: str | None
    jsonl_filename: str | None
    parquet_filename: str | None

    # Topology / classification
    is_view: bool
    owner_package: Literal["core", "analytics", "graphs", "qa", "docs"] | None
    tags: frozenset[str]

    # Stewardship + SLAs
    description: str | None
    family: str | None
    owner: str | None
    freshness_sla: str | None
    retention_policy: str | None
    stable_id: str | None
    schema_version: str | None

    # Upstream dependency graph
    upstream_dependencies: tuple[str, ...]
    validation_profile: Literal["strict", "lenient"]

    def capabilities(self) -> dict[str, bool]:
        """
        Return capability flags derived from the contract attributes.

        These are surfaced directly through MCP and HTTP meta surfaces.
        """
        docs_view = self.table_key.startswith("docs.")
        read_only = self.is_view or docs_view or "read_only" in self.tags
        return {
            "can_validate": self.json_schema_id is not None,
            "can_export_jsonl": self.jsonl_filename is not None,
            "can_export_parquet": self.parquet_filename is not None,
            "has_row_binding": self.row_binding is not None,
            "is_view": self.is_view,
            "docs_view": docs_view,
            "read_only": read_only,
        }
```

### 1.2. Move row bindings & static metadata into this module

Today they live in `storage.datasets` as:

* `ROW_BINDINGS_BY_TABLE_KEY`
* `JSON_SCHEMA_BY_DATASET_NAME`
* `DESCRIPTION_BY_DATASET_NAME`, `OWNER_BY_DATASET_NAME`, `FRESHNESS_BY_DATASET_NAME`, …
* `DEPENDENCIES_BY_DATASET_NAME`, `SCHEMA_VERSION_BY_DATASET_NAME`, `VALIDATION_PROFILE_BY_DATASET_NAME`
* `DEFAULT_JSONL_FILENAMES`, `DEFAULT_PARQUET_FILENAMES`

**Do:**

1. **Cut** these definitions out of `storage/datasets.py`.
2. **Paste** them into `config/dataset_contract.py`, above the `DatasetContract` builder logic.
3. Keep them named the same, but conceptually treat them as “private feedstock” for building `DatasetContract` objects.

Example (shortened) inside `config/dataset_contract.py`:

```python
# JSON Schema mapping
JSON_SCHEMA_BY_DATASET_NAME: Final[dict[str, str]] = {
    # Profiles
    "function_profile": "function_profile",
    "file_profile": "file_profile",
    "module_profile": "module_profile",
    # Graph edges
    "call_graph_edges": "call_graph_edges",
    "symbol_use_edges": "symbol_use_edges",
    "test_coverage_edges": "test_coverage_edges",
    # ...
}

DESCRIPTION_BY_DATASET_NAME: Final[dict[str, str]] = {
    "function_profile": "Function-level profile combining metrics, risk, and topology.",
    "file_profile": "File-level profile with coverage, hotspots, and ownership signals.",
    # ...
}

# Same idea for OWNER_BY_DATASET_NAME, FRESHNESS_BY_DATASET_NAME, etc.

def _metadata_for_name(name: str) -> dict[str, object]:
    return {
        "description": DESCRIPTION_BY_DATASET_NAME.get(name),
        "owner": OWNER_BY_DATASET_NAME.get(name),
        "freshness_sla": FRESHNESS_BY_DATASET_NAME.get(name),
        "retention_policy": RETENTION_BY_DATASET_NAME.get(name),
        "upstream_dependencies": DEPENDENCIES_BY_DATASET_NAME.get(name, ()),
        "stable_id": STABLE_ID_BY_DATASET_NAME.get(name, name),
        "schema_version": SCHEMA_VERSION_BY_DATASET_NAME.get(name, "1"),
        "validation_profile": VALIDATION_PROFILE_BY_DATASET_NAME.get(name, "strict"),
    }
```

And:

```python
def _row_binding(
    row_type: RowDictType,
    to_tuple: RowToTuple,
) -> RowBinding:
    return RowBinding(row_type=row_type, to_tuple=to_tuple)

ROW_BINDINGS_BY_TABLE_KEY: Final[dict[str, RowBinding]] = {
    "analytics.function_profile": _row_binding(
        row_type=row_models.FunctionProfileRow,
        to_tuple=row_models.function_profile_row_to_tuple,
    ),
    "analytics.file_profile": _row_binding(
        row_type=row_models.FileProfileRow,
        to_tuple=row_models.file_profile_row_to_tuple,
    ),
    # ... all existing entries copied over
}
```

(You don’t need to invent new bindings; just move them.)

### 1.3. Build `DATASET_CONTRACTS` from feedstock

Now add **one builder function** that uses:

* `TABLE_SCHEMAS` for base tables
* `DERIVED_DOCS_VIEWS` for docs views that should appear as datasets
* the metadata maps + row bindings you just moved

```python
def _owner_package_for_prefix(prefix: str) -> Literal["core", "analytics", "graphs", "qa", "docs"] | None:
    if prefix == "core":
        return "core"
    if prefix == "analytics":
        return "analytics"
    if prefix in {"graph", "cfg"}:
        return "graphs"
    if prefix == "docs":
        return "docs"
    return None


def _build_contracts() -> dict[str, DatasetContract]:
    contracts: dict[str, DatasetContract] = {}

    # Base tables from TABLE_SCHEMAS
    for table_key, schema in TABLE_SCHEMAS.items():
        if table_key.startswith("tmp_"):
            continue
        schema_prefix, name = table_key.split(".", maxsplit=1)
        meta = _metadata_for_name(name)
        row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(table_key)
        json_schema_id = JSON_SCHEMA_BY_DATASET_NAME.get(name)
        jsonl_filename = DEFAULT_JSONL_FILENAMES.get(table_key)
        parquet_filename = DEFAULT_PARQUET_FILENAMES.get(table_key)
        owner_pkg = _owner_package_for_prefix(schema_prefix)
        family = schema_prefix

        contracts[name] = DatasetContract(
            name=name,
            table_key=table_key,
            schema=schema,
            row_binding=row_binding,
            json_schema_id=json_schema_id,
            jsonl_filename=jsonl_filename,
            parquet_filename=parquet_filename,
            is_view=False,
            owner_package=owner_pkg,
            tags=frozenset({"base_table"}),
            description=meta["description"],
            family=family,
            owner=meta["owner"],
            freshness_sla=meta["freshness_sla"],
            retention_policy=meta["retention_policy"],
            stable_id=meta["stable_id"],
            schema_version=meta["schema_version"],
            upstream_dependencies=meta["upstream_dependencies"],
            validation_profile=meta["validation_profile"],  # "strict" / "lenient"
        )

    # docs.* views that should be treated as datasets
    for view_key in DERIVED_DOCS_VIEWS:
        schema_prefix, view_name = view_key.split(".", maxsplit=1)
        meta = _metadata_for_name(view_name)
        row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(view_key)
        json_schema_id = JSON_SCHEMA_BY_DATASET_NAME.get(view_name)
        jsonl_filename = DEFAULT_JSONL_FILENAMES.get(view_key)
        parquet_filename = DEFAULT_PARQUET_FILENAMES.get(view_key)
        owner_pkg = _owner_package_for_prefix(schema_prefix)
        family = schema_prefix

        contracts[view_name] = DatasetContract(
            name=view_name,
            table_key=view_key,
            schema=None,  # views don’t have a TableSchema
            row_binding=row_binding,
            json_schema_id=json_schema_id,
            jsonl_filename=jsonl_filename,
            parquet_filename=parquet_filename,
            is_view=True,
            owner_package=owner_pkg,
            tags=frozenset({"docs_view", "read_only"}),
            description=meta["description"],
            family=family,
            owner=meta["owner"],
            freshness_sla=meta["freshness_sla"],
            retention_policy=meta["retention_policy"],
            stable_id=meta["stable_id"],
            schema_version=meta["schema_version"],
            upstream_dependencies=meta["upstream_dependencies"],
            validation_profile=meta["validation_profile"],
        )

    return contracts


DATASET_CONTRACTS: Final[dict[str, DatasetContract]] = _build_contracts()
DATASET_CONTRACTS_BY_TABLE_KEY: Final[dict[str, DatasetContract]] = {
    c.table_key: c for c in DATASET_CONTRACTS.values()
}
```

### 1.4. Provide derived constant views from contracts

Instead of hand-maintaining maps, derive them from `DATASET_CONTRACTS` and keep the *names* for backwards compatibility:

```python
JSON_SCHEMA_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: c.json_schema_id
    for name, c in DATASET_CONTRACTS.items()
    if c.json_schema_id is not None
}

DEFAULT_JSONL_FILENAMES: Final[dict[str, str]] = {
    c.table_key: c.jsonl_filename
    for c in DATASET_CONTRACTS.values()
    if c.jsonl_filename is not None
}

DEFAULT_PARQUET_FILENAMES: Final[dict[str, str]] = {
    c.table_key: c.parquet_filename
    for c in DATASET_CONTRACTS.values()
    if c.parquet_filename is not None
}

DEPENDENCIES_BY_DATASET_NAME: Final[dict[str, tuple[str, ...]]] = {
    name: c.upstream_dependencies
    for name, c in DATASET_CONTRACTS.items()
    if c.upstream_dependencies
}
```

This is the critical “SSoT” step: **edit the contract once, everything else updates.**

---

## 2. Refactor `storage.datasets` to use `DatasetContract`

Now that contracts live in config, `storage.datasets` becomes a DB-backed view over them.

### 2.1. Replace local dataclasses with aliases

At the top of `storage/datasets.py`, change imports:

```python
# OLD
from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema
from codeintel.storage import rows as row_models

# NEW
from codeintel.config.dataset_contract import (
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
    DEFAULT_JSONL_FILENAMES,
    DEFAULT_PARQUET_FILENAMES,
    JSON_SCHEMA_BY_DATASET_NAME,
    DatasetContract,
    RowBinding,
)
from codeintel.config.schemas.tables import TableSchema
```

Then **remove** the local `RowBinding` and `Dataset` definitions and instead alias:

```python
# Backwards-compatible aliases
Dataset = DatasetContract
DatasetSpec = DatasetContract
```

Keep `DatasetRegistry` as the DB view type:

```python
@dataclass(frozen=True)
class DatasetRegistry:
    """In-memory view of metadata.datasets plus Python row bindings."""

    by_name: Mapping[str, Dataset]
    by_table_key: Mapping[str, Dataset]
    jsonl_datasets: Mapping[str, str]
    parquet_datasets: Mapping[str, str]
    # ... methods unchanged (all just use Dataset fields)
```

`describe_dataset(ds: Dataset)` at the bottom of the file is already compatible with `DatasetContract` and can remain as-is.

### 2.2. Rewrite `load_dataset_registry` to hydrate from contracts

Right now `load_dataset_registry` recomputes all metadata from TABLE_SCHEMAS + the various maps.

Change it so it *joins* DB rows with `DATASET_CONTRACTS_BY_TABLE_KEY`:

```python
from typing import cast

from duckdb import DuckDBPyConnection
from codeintel.config.dataset_contract import DATASET_CONTRACTS_BY_TABLE_KEY


def load_dataset_registry(con: DuckDBPyConnection) -> DatasetRegistry:
    """
    Load dataset metadata from DuckDB's metadata.datasets table and
    hydrate it with DatasetContract SSoT information.
    """
    rows = con.execute(
        """
        SELECT
            table_key,
            name,
            is_view,
            jsonl_filename,
            parquet_filename,
            family,
            description
        FROM metadata.datasets
        ORDER BY table_key
        """
    ).fetchall()

    by_name: dict[str, Dataset] = {}
    by_table: dict[str, Dataset] = {}
    jsonl_map: dict[str, str] = {}
    parquet_map: dict[str, str] = {}

    for (
        table_key,
        name,
        is_view,
        jsonl_filename,
        parquet_filename,
        db_family,
        db_description,
    ) in rows:
        base = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
        if base is None:
            # Option 1: strict
            # raise KeyError(f"metadata.datasets row {table_key} has no DatasetContract")
            # Option 2: lenient logging – up to you
            continue

        # Prefer DB overrides when present, otherwise fall back to contract
        family = cast("str | None", db_family or base.family)
        description = cast("str | None", db_description or base.description)

        ds = Dataset(
            name=name,                                # should match base.name
            table_key=table_key,
            schema=base.schema,
            row_binding=base.row_binding,
            json_schema_id=base.json_schema_id,
            jsonl_filename=jsonl_filename or base.jsonl_filename,
            parquet_filename=parquet_filename or base.parquet_filename,
            is_view=bool(is_view),
            owner_package=base.owner_package,
            tags=base.tags,
            description=description,
            family=family,
            owner=base.owner,
            freshness_sla=base.freshness_sla,
            retention_policy=base.retention_policy,
            stable_id=base.stable_id,
            schema_version=base.schema_version,
            upstream_dependencies=base.upstream_dependencies,
            validation_profile=base.validation_profile,
        )

        by_name[name] = ds
        by_table[table_key] = ds
        if ds.jsonl_filename:
            jsonl_map[table_key] = ds.jsonl_filename
        if ds.parquet_filename:
            parquet_map[table_key] = ds.parquet_filename

    return DatasetRegistry(
        by_name=by_name,
        by_table_key=by_table,
        jsonl_datasets=jsonl_map,
        parquet_datasets=parquet_map,
    )
```

This is where the “DB-backed materialization of contracts” actually happens.

### 2.3. Keep helper functions but rely on contracts

Functions like:

* `dataset_for_table`
* `dataset_for_name`
* `list_dataset_specs`
* `build_dataset_dependency_graph`
* `describe_dataset`

are all fine as-is; they operate on `Dataset` instances, which are now hydrated `DatasetContract`s.

---

## 3. Refactor storage validation & bootstrap to use contracts

### 3.1. `storage/metadata_bootstrap.py`

Right now, `bootstrap_metadata_datasets` loops over `TABLE_SCHEMAS` and `DERIVED_DOCS_VIEWS` and pulls metadata from maps.

Change it to iterate **contracts**:

```python
from codeintel.config.dataset_contract import DATASET_CONTRACTS

def bootstrap_metadata_datasets(
    con: DuckDBPyConnection,
    *,
    jsonl_filenames: Mapping[str, str] | None = None,
    parquet_filenames: Mapping[str, str] | None = None,
    include_views: bool = True,
) -> None:
    """
    Populate metadata.datasets from DatasetContract SSoT.

    Safe to run repeatedly; uses idempotent upserts to refresh filenames and view flags.
    """
    if include_views:
        create_all_views(con)
    _assert_macro_coverage()
    apply_metadata_ddl(con)

    # Optional overrides
    jsonl_mapping = dict(jsonl_filenames or {})
    parquet_mapping = dict(parquet_filenames or {})

    for name, contract in sorted(DATASET_CONTRACTS.items(), key=lambda item: item[1].table_key):
        if contract.is_view and not include_views:
            continue

        table_key = contract.table_key
        schema_prefix, _ = table_key.split(".", maxsplit=1)

        jsonl_filename = jsonl_mapping.get(table_key) or contract.jsonl_filename
        parquet_filename = parquet_mapping.get(table_key) or contract.parquet_filename

        _upsert_dataset_row(
            con,
            _DatasetUpsert(
                table_key=table_key,
                name=contract.name,
                is_view=contract.is_view,
                jsonl_filename=jsonl_filename,
                parquet_filename=parquet_filename,
                family=contract.family or schema_prefix,
                description=contract.description,
            ),
        )
```

That eliminates all duplication between metadata_bootstrap and the contract.

### 3.2. `storage/contract_validation.py`

Change imports:

```python
# OLD
from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.datasets import (
    DEPENDENCIES_BY_DATASET_NAME,
    JSON_SCHEMA_BY_DATASET_NAME,
    DatasetRegistry,
    build_dataset_dependency_graph,
    load_dataset_registry,
)

# NEW
from codeintel.config.dataset_contract import (
    DATASET_CONTRACTS,
    JSON_SCHEMA_BY_DATASET_NAME,
)
from codeintel.storage.datasets import (
    DatasetRegistry,
    build_dataset_dependency_graph,
    load_dataset_registry,
)
from codeintel.config.schemas.tables import TABLE_SCHEMAS
```

Then rewrite the validation helpers to lean on contracts:

* **JSON Schema coverage**:

```python
BINDING_REQUIRED_DATASETS: set[str] = {
    name
    for name, contract in DATASET_CONTRACTS.items()
    if contract.json_schema_id is not None
    and name not in {"data_model_fields", "data_model_relationships"}
}
```

* **Dependency alignment**:

Instead of referencing `DEPENDENCIES_BY_DATASET_NAME`, compare `build_dataset_dependency_graph(registry)` directly to `contract.upstream_dependencies` for each dataset:

```python
def _validate_dependency_graph(registry: DatasetRegistry) -> list[str]:
    issues: list[str] = []
    actual_graph = build_dataset_dependency_graph(registry)

    for name, contract in DATASET_CONTRACTS.items():
        expected = set(contract.upstream_dependencies)
        actual = set(actual_graph.get(name, ()))
        if expected != actual:
            issues.append(
                f"Dataset {name} dependency mismatch: expected {sorted(expected)}, "
                f"got {sorted(actual)}"
            )

    return issues
```

* **Schema alignment** can stay similar, but you can optionally cross-check that for every non-view contract, `TABLE_SCHEMAS[contract.table_key]` is present and matches `contract.schema` (if you decide to generate `TABLE_SCHEMAS` from contracts later).

The overall `collect_contract_issues` + `validate_contract_or_raise` flow stays the same; they just rely on `DATASET_CONTRACTS`.

### 3.3. `storage/conformance.py`

You mostly just need to keep imports in sync:

```python
# keep using _schema_path and collect_contract_issues from contract_validation
from codeintel.storage.contract_validation import (
    _schema_path,
    collect_contract_issues,
)
```

Because `contract_validation` now uses `DATASET_CONTRACTS`, your conformance validation automatically becomes contract-driven.

### 3.4. `storage/schema_generation.py`

No structural change needed; it already uses `DatasetRegistry` and row models. The only subtle improvement: you can explicitly ignore datasets with `contract.json_schema_id is None` when generating new JSON Schemas.

---

## 4. Refactor serving to consume contracts instead of TABLE_SCHEMAS

### 4.1. `serving/backend/datasets.py`

Right now, `build_dataset_registry` constructs a name→table mapping by walking `TABLE_SCHEMAS` and `DOCS_VIEWS`.

Replace that with contract-driven logic:

```python
from typing import Literal

from codeintel.config.dataset_contract import DATASET_CONTRACTS
from codeintel.serving.backend.pagination import BackendLimits

def build_dataset_registry(
    *, include_docs_views: Literal["include", "exclude"] = "include"
) -> dict[str, str]:
    """
    Build deterministic dataset registry from DatasetContract SSoT.

    Returns
    -------
    dict[str, str]
        Mapping of dataset name to fully qualified table/view name.
    """
    registry: OrderedDict[str, str] = OrderedDict()
    for name, contract in sorted(DATASET_CONTRACTS.items(), key=lambda item: item[1].table_key):
        if contract.is_view and include_docs_views == "exclude":
            continue
        registry[name] = contract.table_key
    return dict(registry)
```

Replace `describe_dataset(name: str, table: str)` to prefer contracts:

```python
from codeintel.config.dataset_contract import DATASET_CONTRACTS_BY_TABLE_KEY

def describe_dataset(name: str, table: str) -> str:
    """
    Produce a human-friendly description for a dataset/table.
    """
    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table)
    if contract is None:
        return f"{name}: {table}"
    column_names = (
        contract.schema.column_names()[:PREVIEW_COLUMN_COUNT]
        if contract.schema is not None
        else []
    )
    cols = ", ".join(column_names)
    extra = "" if not contract.schema or len(contract.schema.columns) <= PREVIEW_COLUMN_COUNT else "..."
    return f"{name}: {table} ({cols}{extra})"
```

Everything else in this module (macro validation, registry/limits composition) can stay the same, just powered by the new registry.

### 4.2. `serving/registry.py`

`build_dataset_meta` already consumes `DatasetSpecDescriptor` from `QueryService.dataset_specs()`, which in turn is built from `list_dataset_specs(registry)` → `describe_dataset(ds)` (now contract-driven).

You can optionally **enrich** `DatasetMeta` with contract info that’s now available:

* Add `owner`, `freshness_sla`, `retention_policy`, `tags` to `DatasetMeta`.
* Map them from the `DatasetSpecDescriptor` fields you already have (`owner`, `freshness_sla`, `retention_policy`, `capabilities`, `validation_profile`).

Example extension:

```python
@dataclass(frozen=True)
class DatasetMeta:
    """Dataset metadata enriched with serving limits and flags."""

    id: str
    name: str
    table_key: str
    description: str
    schema_version: str | None
    family: str | None
    is_docs_view: bool
    is_read_only: bool
    default_limit: int
    max_limit: int
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    validation_profile: str | None = None


def build_dataset_meta(service: QueryService, limits: BackendLimits) -> list[DatasetMeta]:
    specs: list[DatasetSpecDescriptor] = service.dataset_specs()
    metas: list[DatasetMeta] = []

    for spec in specs:
        family = getattr(spec, "family", None)
        is_docs_view = bool(family == "docs" or spec.table_key.startswith("docs."))
        capabilities = getattr(spec, "capabilities", {}) or {}
        is_read_only = bool(capabilities.get("read_only", False))
        description = spec.description or f"{spec.name} ({spec.table_key})"
        metas.append(
            DatasetMeta(
                id=spec.name,
                name=spec.name,
                table_key=spec.table_key,
                description=description,
                schema_version=spec.schema_version,
                family=family,
                is_docs_view=is_docs_view,
                is_read_only=is_read_only,
                default_limit=limits.default_limit,
                max_limit=limits.max_rows_per_call,
                owner=spec.owner,
                freshness_sla=spec.freshness_sla,
                retention_policy=spec.retention_policy,
                validation_profile=spec.validation_profile,
            )
        )

    return metas
```

No change needed to operation specs here; they already refer to datasets by name.

---

## 5. Tests & migration checklist

### 5.1. Update imports / aliases

* `tests/storage/test_datasets_contract.py`:

  * You can keep importing `Dataset`, `RowBinding`, `JSON_SCHEMA_BY_DATASET_NAME`, `describe_dataset` from `codeintel.storage.datasets` because you aliased them.
* `tests/serving/test_dataset_specs.py`:

  * `DEFAULT_JSONL_FILENAMES` is now derived from contracts; existing assertions should still hold.
* `tests/docs_export/test_export_defaults.py`:

  * `JSON_SCHEMA_BY_DATASET_NAME` now lives in `config.dataset_contract` but is re-exported from `storage.datasets`; either update imports or keep aliases for B/C.

### 5.2. Add new tests for the contract SSoT

Add a new test file `tests/config/test_dataset_contract.py`:

* **Contract/DDL alignment:**

```python
from codeintel.config.dataset_contract import DATASET_CONTRACTS, DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.config.schemas.tables import TABLE_SCHEMAS

def test_contracts_cover_all_tables() -> None:
    for table_key in TABLE_SCHEMAS:
        if table_key.startswith("tmp_"):
            continue
        assert table_key in DATASET_CONTRACTS_BY_TABLE_KEY, table_key

def test_contract_schema_matches_tables() -> None:
    for table_key, schema in TABLE_SCHEMAS.items():
        if table_key.startswith("tmp_"):
            continue
        contract = DATASET_CONTRACTS_BY_TABLE_KEY[table_key]
        assert contract.schema is schema
```

* **Derived maps consistency:**

```python
from codeintel.config.dataset_contract import (
    DATASET_CONTRACTS,
    JSON_SCHEMA_BY_DATASET_NAME,
    DEFAULT_JSONL_FILENAMES,
)

def test_json_schema_map_matches_contracts() -> None:
    expected = {
        name: c.json_schema_id
        for name, c in DATASET_CONTRACTS.items()
        if c.json_schema_id is not None
    }
    assert JSON_SCHEMA_BY_DATASET_NAME == expected

def test_default_jsonl_filenames_match_contracts() -> None:
    expected = {
        c.table_key: c.jsonl_filename
        for c in DATASET_CONTRACTS.values()
        if c.jsonl_filename is not None
    }
    assert DEFAULT_JSONL_FILENAMES == expected
```

### 5.3. Run validation in CI

* Ensure `storage.contract_validation.validate_contract_or_raise` still runs in your CI bootstrap, but now it’s exercising the **contract module**.
* Add a simple unit test that `collect_contract_issues` returns an empty list against an in-memory DB bootstrapped via `metadata_bootstrap.bootstrap_metadata_datasets`.

---

## Mental model after Epic 1

After you implement this:

* **config/dataset_contract** is the *only* place you define what a dataset is.
* `storage.metadata_bootstrap` and `storage.datasets.load_dataset_registry` are purely **materialization and validation layers**.
* `pipeline.export`, `storage.conformance`, `serving.backend.datasets`, and `serving.registry` all consume **contract-derived views** (constants or registry objects), never hand-assembled combinations of `TABLE_SCHEMAS` + ad-hoc maps.

If you’d like, next we can turn this into an ordered patch plan (file-by-file diff ordering) or zoom into one sub-slice (for example, just the `metadata_bootstrap` rewiring) and make it fully plug-and-play for an implementation agent.

# ordered patch plan #

Here’s an ordered patch plan you can hand straight to an implementation agent. It’s designed so that:

* Each patch has a clear goal.
* Changes are grouped by files.
* Behavior stays stable until late in the sequence (most changes are “refactor, not feature”).

I’ll assume file roots like `src/codeintel/...`.

---

## Patch 0 — Pre-checks (no code changes)

**Goal:** Make sure you know where things are referenced so nothing surprises you.

1. Grep for dataset metadata usage (just to have a mental map):

   * `storage/datasets.py`
   * `storage/metadata_bootstrap.py`
   * `storage/contract_validation.py`
   * `storage/conformance.py`
   * `storage/schema_generation.py`
   * `docs_export/*` (JSON Schemas / filenames)
   * `serving/backend/datasets.py`
   * `serving/registry.py`
   * any `DEFAULT_JSONL_FILENAMES`, `DEFAULT_PARQUET_FILENAMES`, `JSON_SCHEMA_BY_DATASET_NAME`, `DEPENDENCIES_BY_DATASET_NAME` usages.

No code to change here, just grounding.

---

## Patch 1 — Create `config.dataset_contract` and move constants there

**Goal:** Introduce the new **SSoT module** and move all dataset metadata constants into it, while preserving behavior via re-exports from `storage.datasets`.

### Files touched

* **NEW** `src/codeintel/config/dataset_contract.py`
* **EDIT** `src/codeintel/storage/datasets.py`
* (optional) `src/codeintel/config/__init__.py`

### 1.1. Create `config/dataset_contract.py`

Create a new module with:

* `RowBinding` dataclass
* `DatasetContract` dataclass
* **all** the metadata maps currently defined in `storage/datasets.py`:

  * `ROW_BINDINGS_BY_TABLE_KEY`
  * `JSON_SCHEMA_BY_DATASET_NAME`
  * `DESCRIPTION_BY_DATASET_NAME`
  * `OWNER_BY_DATASET_NAME`
  * `FRESHNESS_BY_DATASET_NAME`
  * `RETENTION_BY_DATASET_NAME`
  * `STABLE_ID_BY_DATASET_NAME`
  * `SCHEMA_VERSION_BY_DATASET_NAME`
  * `VALIDATION_PROFILE_BY_DATASET_NAME`
  * `DEPENDENCIES_BY_DATASET_NAME`
  * `DEFAULT_JSONL_FILENAMES`
  * `DEFAULT_PARQUET_FILENAMES`

Plus helper functions like `_metadata_for_name`.

Rough structure:

```python
# src/codeintel/config/dataset_contract.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Mapping, Callable

from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema
from codeintel.storage import rows as row_models
from codeintel.storage.views import DERIVED_DOCS_VIEWS

RowToTuple = Callable[[Mapping[str, object]], tuple[object, ...]]
RowDictType = type[object]


@dataclass(frozen=True)
class RowBinding:
    row_type: RowDictType
    to_tuple: RowToTuple


@dataclass(frozen=True)
class DatasetContract:
    # identity + physical
    name: str
    table_key: str
    schema: TableSchema | None
    row_binding: RowBinding | None

    # export / validation
    json_schema_id: str | None
    jsonl_filename: str | ocs/Serving_Epic_5.mdNone
    parquet_filename: str | None

    # topology / classification
    is_view: bool
    owner_package: Literal["core", "analytics", "graphs", "qa", "docs"] | None
    tags: frozenset[str]

    # stewardship
    description: str | None
    family: str | None
    owner: str | None
    freshness_sla: str | None
    retention_policy: str | None
    stable_id: str | None
    schema_version: str | None

    # dependencies
    upstream_dependencies: tuple[str, ...]
    validation_profile: Literal["strict", "lenient"]

    def capabilities(self) -> dict[str, bool]:
        ...
```

Then **physically move** the constant definitions from `storage/datasets.py` into this file and adjust imports accordingly (e.g. row models from `storage.rows`).

### 1.2. Build `DATASET_CONTRACTS` (not used yet)

In the same module, add:

* `_owner_package_for_prefix(prefix: str) -> Literal[...] | None`
* `_metadata_for_name(name: str) -> dict[...]` (using the moved maps)
* `_build_contracts() -> dict[str, DatasetContract]` that:

  * Iterates `TABLE_SCHEMAS` for base tables.
  * Iterates `DERIVED_DOCS_VIEWS` for docs views.
  * Joins in JSON Schema IDs, filenames, row bindings, metadata.

Then:

```python
DATASET_CONTRACTS: Final[dict[str, DatasetContract]] = _build_contracts()
DATASET_CONTRACTS_BY_TABLE_KEY: Final[dict[str, DatasetContract]] = {
    c.table_key: c for c in DATASET_CONTRACTS.values()
}

# Derived maps (for backwards compatibility)
JSON_SCHEMA_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: c.json_schema_id
    for name, c in DATASET_CONTRACTS.items()
    if c.json_schema_id is not None
}

DEFAULT_JSONL_FILENAMES: Final[dict[str, str]] = {
    c.table_key: c.jsonl_filename
    for c in DATASET_CONTRACTS.values()
    if c.jsonl_filename is not None
}

DEFAULT_PARQUET_FILENAMES: Final[dict[str, str]] = {
    c.table_key: c.parquet_filename
    for c in DATASET_CONTRACTS.values()
    if c.parquet_filename is not None
}

DEPENDENCIES_BY_DATASET_NAME: Final[dict[str, tuple[str, ...]]] = {
    name: c.upstream_dependencies
    for name, c in DATASET_CONTRACTS.items()
    if c.upstream_dependencies
}
```

### 1.3. Re-export constants from `storage.datasets` for B/C

In `storage/datasets.py`, **delete** the moved constant definitions and instead:

```python
# src/codeintel/storage/datasets.py

from codeintel.config.dataset_contract import (
    RowBinding,
    DatasetContract,
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
    JSON_SCHEMA_BY_DATASET_NAME,
    DEFAULT_JSONL_FILENAMES,
    DEFAULT_PARQUET_FILENAMES,
    DEPENDENCIES_BY_DATASET_NAME,
    # any other maps tests rely on
)

# Backwards-compatible aliases if other modules import these names from storage.datasets
Dataset = DatasetContract
DatasetSpec = DatasetContract
```

For now, **do not** change `DatasetRegistry` or `load_dataset_registry` logic – this patch should be behavior-neutral.

### 1.4. Optional: export from `config.__init__`

If you like, add:

```python
# src/codeintel/config/__init__.py
from .dataset_contract import DatasetContract, DATASET_CONTRACTS
```

**Checkpoint:** tests should still pass; behavior unchanged, just definitions moved.

---

## Patch 2 — Make `DatasetRegistry` a view over `DatasetContract`s

**Goal:** `storage.datasets.DatasetRegistry` becomes a hydrated view over `DATASET_CONTRACTS` + `metadata.datasets` rather than its own parallel universe.

### Files touched

* `src/codeintel/storage/datasets.py`

### 2.1. Alias Dataset types explicitly

At the top of `storage/datasets.py`, make it clear that the “dataset” concept is now the contract:

```python
from codeintel.config.dataset_contract import DatasetContract

Dataset = DatasetContract
DatasetSpec = DatasetContract
```

Ensure any local `@dataclass class Dataset` definition is removed.

### 2.2. Rewrite `DatasetRegistry` to hold `DatasetContract`

Confirm:

```python
@dataclass(frozen=True)
class DatasetRegistry:
    by_name: Mapping[str, Dataset]         # DatasetContract
    by_table_key: Mapping[str, Dataset]
    jsonl_datasets: Mapping[str, str]
    parquet_datasets: Mapping[str, str]
```

No behavior change yet.

### 2.3. Rewrite `load_dataset_registry` to hydrate from contracts

Replace its internal construction logic with an implementation that:

* Executes the existing `SELECT ... FROM metadata.datasets`.
* For each row, looks up `base = DATASET_CONTRACTS_BY_TABLE_KEY[table_key]`.
* Builds a new `Dataset` (which is `DatasetContract`) by “overlaying” any DB overrides onto the base contract.

Rough sketch (from previous answer):

```python
from codeintel.config.dataset_contract import DATASET_CONTRACTS_BY_TABLE_KEY

def load_dataset_registry(con: DuckDBPyConnection) -> DatasetRegistry:
    rows = con.execute(...).fetchall()

    by_name: dict[str, Dataset] = {}
    by_table: dict[str, Dataset] = {}
    jsonl_map: dict[str, str] = {}
    parquet_map: dict[str, str] = {}

    for table_key, name, is_view, jsonl_filename, parquet_filename, db_family, db_description in rows:
        base = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
        if base is None:
            # optional: raise or log
            continue

        family = db_family or base.family
        description = db_description or base.description

        ds = Dataset(
            name=name,
            table_key=table_key,
            schema=base.schema,
            row_binding=base.row_binding,
            json_schema_id=base.json_schema_id,
            jsonl_filename=jsonl_filename or base.jsonl_filename,
            parquet_filename=parquet_filename or base.parquet_filename,
            is_view=bool(is_view),
            owner_package=base.owner_package,
            tags=base.tags,
            description=description,
            family=family,
            owner=base.owner,
            freshness_sla=base.freshness_sla,
            retention_policy=base.retention_policy,
            stable_id=base.stable_id,
            schema_version=base.schema_version,
            upstream_dependencies=base.upstream_dependencies,
            validation_profile=base.validation_profile,
        )

        by_name[name] = ds
        by_table[table_key] = ds
        if ds.jsonl_filename:
            jsonl_map[table_key] = ds.jsonl_filename
        if ds.parquet_filename:
            parquet_map[table_key] = ds.parquet_filename

    return DatasetRegistry(
        by_name=by_name,
        by_table_key=by_table,
        jsonl_datasets=jsonl_map,
        parquet_datasets=parquet_map,
    )
```

### 2.4. Ensure helper functions still work

Functions like:

* `dataset_for_name`
* `dataset_for_table`
* `list_dataset_specs`
* `build_dataset_dependency_graph`
* `describe_dataset`

should now be operating on `DatasetContract` instances; usually no change needed except type hints.

**Checkpoint:** run storage-focused tests (`tests/storage/test_datasets_*.py` etc.) and any metadata bootstrap tests.

---

## Patch 3 — Make metadata bootstrap contract-driven

**Goal:** `storage.metadata_bootstrap` builds `metadata.datasets` from `DATASET_CONTRACTS` instead of re-deriving everything manually.

### Files touched

* `src/codeintel/storage/metadata_bootstrap.py`

### 3.1. Import contracts

At the top:

```python
from codeintel.config.dataset_contract import DATASET_CONTRACTS
```

### 3.2. Rewrite `bootstrap_metadata_datasets`

Instead of iterating `TABLE_SCHEMAS` + `DERIVED_DOCS_VIEWS` and using many maps, change to:

```python
def bootstrap_metadata_datasets(
    con: DuckDBPyConnection,
    *,
    jsonl_filenames: Mapping[str, str] | None = None,
    parquet_filenames: Mapping[str, str] | None = None,
    include_views: bool = True,
) -> None:
    if include_views:
        create_all_views(con)
    _assert_macro_coverage()
    apply_metadata_ddl(con)

    jsonl_override = dict(jsonl_filenames or {})
    parquet_override = dict(parquet_filenames or {})

    for name, contract in sorted(DATASET_CONTRACTS.items(), key=lambda item: item[1].table_key):
        if contract.is_view and not include_views:
            continue

        table_key = contract.table_key
        schema_prefix, _ = table_key.split(".", 1)

        jsonl_filename = jsonl_override.get(table_key) or contract.jsonl_filename
        parquet_filename = parquet_override.get(table_key) or contract.parquet_filename

        _upsert_dataset_row(
            con,
            _DatasetUpsert(
                table_key=table_key,
                name=contract.name,
                is_view=contract.is_view,
                jsonl_filename=jsonl_filename,
                parquet_filename=parquet_filename,
                family=contract.family or schema_prefix,
                description=contract.description,
            ),
        )
```

All references to `DEFAULT_JSONL_FILENAMES`, `DEFAULT_PARQUET_FILENAMES`, etc. vanish here—they’re baked into the contract.

**Checkpoint:** run whatever tests validate `metadata.datasets` (and maybe a quick manual `SELECT * FROM metadata.datasets` in a dev session).

---

## Patch 4 — Contract-driven validation & conformance

**Goal:** `storage.contract_validation` and `storage.conformance` validate against `DATASET_CONTRACTS` instead of local maps.

### Files touched

* `src/codeintel/storage/contract_validation.py`
* `src/codeintel/storage/conformance.py`

### 4.1. Update imports in `contract_validation.py`

Replace imports like:

```python
from codeintel.storage.datasets import (
    DEPENDENCIES_BY_DATASET_NAME,
    JSON_SCHEMA_BY_DATASET_NAME,
    load_dataset_registry,
    build_dataset_dependency_graph,
)
from codeintel.config.schemas.tables import TABLE_SCHEMAS
```

with:

```python
from codeintel.config.dataset_contract import (
    DATASET_CONTRACTS,
    JSON_SCHEMA_BY_DATASET_NAME,
)
from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.datasets import (
    load_dataset_registry,
    build_dataset_dependency_graph,
    DatasetRegistry,
)
```

### 4.2. Make JSON Schema coverage contract-driven

Wherever you compute “datasets that should have JSON Schema”, derive from contracts:

```python
SCHEMA_REQUIRED_DATASETS: set[str] = {
    name
    for name, contract in DATASET_CONTRACTS.items()
    if contract.json_schema_id is not None
}
```

Use this set to check that:

* the JSON files exist on disk;
* the mapping `JSON_SCHEMA_BY_DATASET_NAME` matches the contracts.

### 4.3. Align dependency graph with `upstream_dependencies`

Add a helper:

```python
def _validate_dependency_graph(registry: DatasetRegistry) -> list[str]:
    issues: list[str] = []
    graph = build_dataset_dependency_graph(registry)

    for name, contract in DATASET_CONTRACTS.items():
        expected = set(contract.upstream_dependencies)
        actual = set(graph.get(name, ()))
        if expected != actual:
            issues.append(
                f"Dataset {name} dependency mismatch: expected {sorted(expected)}, "
                f"got {sorted(actual)}"
            )
    return issues
```

And ensure `collect_contract_issues` includes this check.

### 4.4. Keep schema alignment checks, but tie them back to contracts

Optionally add:

```python
def _validate_schemas_match_contracts() -> list[str]:
    issues: list[str] = []
    for table_key, schema in TABLE_SCHEMAS.items():
        if table_key.startswith("tmp_"):
            continue
        contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
        if contract is None:
            issues.append(f"No DatasetContract for table {table_key}")
        elif contract.schema is not schema:
            issues.append(f"Schema mismatch for {table_key}")
    return issues
```

Include that in `collect_contract_issues`.

### 4.5. `conformance.py` uses new validation

`conformance.py` mostly calls `collect_contract_issues` and `_schema_path` – as long as those are still exported, you likely only need to:

* Update any imports that referenced removed constants.
* Confirm that its error messages still make sense (they now talk about contracts, not ad-hoc maps).

**Checkpoint:** run tests under `tests/storage/test_contract_validation.py` and any conformance tests.

---

## Patch 5 — Optional: tighten `schema_generation` to contracts

**Goal:** Ensure `storage.schema_generation` doesn’t drift from contracts.

### Files touched

* `src/codeintel/storage/schema_generation.py`

You don’t have to change much here; it already works via `DatasetRegistry`. Two quick improvements:

1. When generating JSON Schemas, filter datasets by `contract.json_schema_id is not None` (via `DATASET_CONTRACTS`) instead of hand-curated lists.
2. If there’s any hand-maintained “dataset list”, replace it with a comprehension over `DATASET_CONTRACTS`.

**Checkpoint:** any schema-generation tests & docs export tests.

---

## Patch 6 — Serving backend datasets: contract-driven registry

**Goal:** `serving.backend.datasets` uses `DATASET_CONTRACTS` for `/meta/datasets` instead of `TABLE_SCHEMAS` / `DOCS_VIEWS`.

### Files touched

* `src/codeintel/serving/backend/datasets.py`

### 6.1. Import contracts

At the top:

```python
from codeintel.config.dataset_contract import DATASET_CONTRACTS, DATASET_CONTRACTS_BY_TABLE_KEY
```

### 6.2. Rewrite `build_dataset_registry` to use contracts

Replace any logic that walks `TABLE_SCHEMAS` / `DOCS_VIEWS` with:

```python
from collections import OrderedDict

def build_dataset_registry(
    *, include_docs_views: bool = True,
) -> dict[str, str]:
    registry: OrderedDict[str, str] = OrderedDict()
    for name, contract in sorted(DATASET_CONTRACTS.items(), key=lambda item: item[1].table_key):
        if contract.is_view and not include_docs_views:
            continue
        registry[name] = contract.table_key
    return dict(registry)
```

### 6.3. Make `describe_dataset` contract-aware

If you have:

```python
def describe_dataset(name: str, table: str) -> str:
    ...
```

update it to look up the contract:

```python
def describe_dataset(name: str, table: str) -> str:
    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table)
    if contract is None or contract.schema is None:
        return f"{name}: {table}"
    column_names = contract.schema.column_names()[:PREVIEW_COLUMN_COUNT]
    cols = ", ".join(column_names)
    extra = "" if len(contract.schema.columns) <= PREVIEW_COLUMN_COUNT else "..."
    return f"{name}: {table} ({cols}{extra})"
```

Everything else (macro coverage, limits) stays the same.

**Checkpoint:** run serving/backend dataset meta tests and a quick manual hit to `/meta/datasets` in dev.

---

## Patch 7 — Enrich `serving.registry` with contract metadata

**Goal:** `DatasetMeta` surfaces owner/freshness/validation info derived from contracts (via `DatasetSpecDescriptor`).

### Files touched

* `src/codeintel/serving/registry.py`

### 7.1. Extend `DatasetMeta`

Add optional fields:

```python
@dataclass(frozen=True)
class DatasetMeta:
    id: str
    name: str
    table_key: str
    description: str
    schema_version: str | None
    family: str | None
    is_docs_view: bool
    is_read_only: bool
    default_limit: int
    max_limit: int
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    validation_profile: str | None = None
```

### 7.2. Populate from `DatasetSpecDescriptor`

Adjust `build_dataset_meta` to pull fields from descriptors:

```python
def build_dataset_meta(service: QueryService, limits: BackendLimits) -> list[DatasetMeta]:
    specs = service.dataset_specs()
    metas: list[DatasetMeta] = []

    for spec in specs:
        family = getattr(spec, "family", None)
        caps = getattr(spec, "capabilities", {}) or {}
        is_docs_view = bool(family == "docs" or spec.table_key.startswith("docs."))
        is_read_only = bool(caps.get("read_only", False))
        description = spec.description or f"{spec.name} ({spec.table_key})"

        metas.append(
            DatasetMeta(
                id=spec.name,
                name=spec.name,
                table_key=spec.table_key,
                description=description,
                schema_version=spec.schema_version,
                family=family,
                is_docs_view=is_docs_view,
                is_read_only=is_read_only,
                default_limit=limits.default_limit,
                max_limit=limits.max_rows_per_call,
                owner=spec.owner,
                freshness_sla=spec.freshness_sla,
                retention_policy=spec.retention_policy,
                validation_profile=spec.validation_profile,
            )
        )

    return metas
```

If `DatasetSpecDescriptor` doesn’t yet expose `owner`, `freshness_sla`, etc., add them there as read-only fields populated from the underlying `DatasetContract` inside `QueryService.dataset_specs()` (which already uses `describe_dataset(ds)` and `capabilities()`).

**Checkpoint:** HTTP `/meta/datasets` & MCP meta responses still serialize, and now with richer fields.

---

## Patch 8 — Tests, layering, and cleanup

**Goal:** Make the new structure robust and verified; remove any dead code.

### Files touched

* `tests/config/test_dataset_contract.py` (new)
* `tests/storage/...` (updated)
* `tests/serving/...` (updated)
* `src/codeintel/config/layering_checks.py` (if needed)
* any remaining references to moved constants

### 8.1. Add tests for `config.dataset_contract`

Create `tests/config/test_dataset_contract.py` with:

* All `TABLE_SCHEMAS` (except `tmp_*`) have a corresponding contract:

  ```python
  def test_all_tables_have_contracts() -> None:
      from codeintel.config.dataset_contract import DATASET_CONTRACTS_BY_TABLE_KEY
      from codeintel.config.schemas.tables import TABLE_SCHEMAS

      for table_key in TABLE_SCHEMAS:
          if table_key.startswith("tmp_"):
              continue
          assert table_key in DATASET_CONTRACTS_BY_TABLE_KEY
  ```

* Derived maps equal comprehensions over `DATASET_CONTRACTS`:

  ```python
  def test_json_schema_map_matches_contracts() -> None:
      from codeintel.config.dataset_contract import (
          DATASET_CONTRACTS,
          JSON_SCHEMA_BY_DATASET_NAME,
      )

      expected = {
          name: c.json_schema_id
          for name, c in DATASET_CONTRACTS.items()
          if c.json_schema_id is not None
      }

      assert JSON_SCHEMA_BY_DATASET_NAME == expected
  ```

* Simple smoke test for `capabilities()` shape.

### 8.2. Update existing tests to use new locations

Where tests import constants from `codeintel.storage.datasets`, keep that working via re-exports. Optionally:

* Add a few tests that import `JSON_SCHEMA_BY_DATASET_NAME` directly from `config.dataset_contract` to encourage callers to move over time.

### 8.3. Layering checker

If your `config/layering_checks.py` tracks per-module imports, make sure:

* `config.dataset_contract` → `storage.rows` and `storage.views` is allowed (core→core).
* `storage.datasets` → `config.dataset_contract` is still allowed (core→core).
* No new domain/app imports leaked into the core.

If you want to keep config “above” storage, you can later move row bindings (or at least their wiring) back into storage and have `DatasetContract` just refer to a `row_binding_id` string instead of `RowBinding`; but for now, just mark the new imports as allowed.

### 8.4. Cleanup stragglers

Search for:

* `DEFAULT_JSONL_FILENAMES`
* `DEFAULT_PARQUET_FILENAMES`
* `JSON_SCHEMA_BY_DATASET_NAME`
* `DEPENDENCIES_BY_DATASET_NAME`

and confirm:

* They are either imported from `config.dataset_contract` (preferred) or from `storage.datasets` (re-export).
* There are no “shadow” copies of these maps anywhere else.

At the very end, if everything passes, you can optionally **delete** the re-exports from `storage.datasets` and force all callers to import from `config.dataset_contract` — but that’s optional and can be a separate tidy-up commit.

---

If you’d like, I can now zoom in on **one** of these patches (e.g. Patch 2: `DatasetRegistry` → view over `DatasetContract`) and write it as a literal, file-by-file diff you can paste into an implementation agent.

# file by file diff deep dive on Patch 2: `DatasetRegistry` → view over `DatasetContract`#

Here’s a focused “Patch 2” diff for **`src/codeintel/storage/datasets.py`**, assuming:

* You’ve already done **Patch 1** and created `codeintel.config.dataset_contract` with:

  * `DatasetContract`
  * `DATASET_CONTRACTS_BY_TABLE_KEY`
* The signature of `DatasetContract` matches the existing `Dataset` dataclass fields.

This patch does **two things only**:

1. Makes `load_dataset_registry` hydrate from `DATASET_CONTRACTS_BY_TABLE_KEY` (the SSoT).
2. Updates imports to pull in `DATASET_CONTRACTS_BY_TABLE_KEY` and drop `TABLE_SCHEMAS`.

You can hand this to an implementation agent as-is.

---

## File: `src/codeintel/storage/datasets.py`

### 1) Imports: drop `TABLE_SCHEMAS`, add `DATASET_CONTRACTS_BY_TABLE_KEY`

```diff
diff --git a/src/codeintel/storage/datasets.py b/src/codeintel/storage/datasets.py
--- a/src/codeintel/storage/datasets.py
+++ b/src/codeintel/storage/datasets.py
@@ -8,9 +8,11 @@ from typing import Literal, cast
 
 from duckdb import DuckDBPyConnection
 
-from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema
+from codeintel.config.dataset_contract import DATASET_CONTRACTS_BY_TABLE_KEY
+from codeintel.config.schemas.tables import TableSchema
 from codeintel.storage import rows as row_models
```

> Note: we still import `TableSchema` for the type annotation on `Dataset.schema`, but `TABLE_SCHEMAS` is no longer needed in this module after we rewrite `load_dataset_registry`.

---

### 2) `load_dataset_registry`: hydrate from `DatasetContract` SSoT

Replace the body of `load_dataset_registry` with the version below.

**Before** (for reference, you don’t need to paste this, just locate it):

```python
def load_dataset_registry(con: DuckDBPyConnection) -> DatasetRegistry:
    """
    Load dataset metadata from DuckDB's metadata.datasets table.

    Assumes metadata_bootstrap.bootstrap_metadata_datasets() has run on this database.

    Returns
    -------
    DatasetRegistry
        Registry containing dataset metadata mirrored from DuckDB.
    """
    rows = con.execute(
        """
        SELECT
            table_key,
            name,
            is_view,
            jsonl_filename,
            parquet_filename,
            family,
            description
        FROM metadata.datasets
        ORDER BY table_key
        """
    ).fetchall()

    by_name: dict[str, Dataset] = {}
    by_table: dict[str, Dataset] = {}
    jsonl_map: dict[str, str] = {}
    parquet_map: dict[str, str] = {}

    for (
        table_key,
        name,
        is_view,
        jsonl_filename,
        parquet_filename,
        db_family,
        db_description,
    ) in rows:
        schema: TableSchema | None = None if is_view else TABLE_SCHEMAS.get(table_key)
        row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(table_key)
        json_schema_id = JSON_SCHEMA_BY_DATASET_NAME.get(name)
        meta = _metadata_for_name(name)
        inferred_family = table_key.split(".", maxsplit=1)[0] if "." in table_key else None
        family = db_family if db_family is not None else inferred_family
        description = db_description if db_description is not None else meta["description"]
        ds = Dataset(
            table_key=table_key,
            name=name,
            schema=schema,
            row_binding=row_binding,
            jsonl_filename=jsonl_filename,
            parquet_filename=parquet_filename,
            is_view=bool(is_view),
            json_schema_id=json_schema_id,
            description=cast("str | None", description),
            family=family,
            owner=cast("str | None", meta["owner"]),
            freshness_sla=cast("str | None", meta["freshness_sla"]),
            retention_policy=cast("str | None", meta["retention_policy"]),
            stable_id=cast("str | None", meta["stable_id"]),
            schema_version=cast("str | None", meta["schema_version"]),
            upstream_dependencies=cast("tuple[str, ...]", meta["upstream_dependencies"]),
            validation_profile=cast("Literal['strict', 'lenient']", meta["validation_profile"]),
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
```

**After** (this is what you actually paste in):

```diff
@@ def load_dataset_registry(con: DuckDBPyConnection) -> DatasetRegistry:
-    rows = con.execute(
-        """
-        SELECT
-            table_key,
-            name,
-            is_view,
-            jsonl_filename,
-            parquet_filename,
-            family,
-            description
-        FROM metadata.datasets
-        ORDER BY table_key
-        """
-    ).fetchall()
-
-    by_name: dict[str, Dataset] = {}
-    by_table: dict[str, Dataset] = {}
-    jsonl_map: dict[str, str] = {}
-    parquet_map: dict[str, str] = {}
-
-    for (
-        table_key,
-        name,
-        is_view,
-        jsonl_filename,
-        parquet_filename,
-        db_family,
-        db_description,
-    ) in rows:
-        schema: TableSchema | None = None if is_view else TABLE_SCHEMAS.get(table_key)
-        row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(table_key)
-        json_schema_id = JSON_SCHEMA_BY_DATASET_NAME.get(name)
-        meta = _metadata_for_name(name)
-        inferred_family = table_key.split(".", maxsplit=1)[0] if "." in table_key else None
-        family = db_family if db_family is not None else inferred_family
-        description = db_description if db_description is not None else meta["description"]
-        ds = Dataset(
-            table_key=table_key,
-            name=name,
-            schema=schema,
-            row_binding=row_binding,
-            jsonl_filename=jsonl_filename,
-            parquet_filename=parquet_filename,
-            is_view=bool(is_view),
-            json_schema_id=json_schema_id,
-            description=cast("str | None", description),
-            family=family,
-            owner=cast("str | None", meta["owner"]),
-            freshness_sla=cast("str | None", meta["freshness_sla"]),
-            retention_policy=cast("str | None", meta["retention_policy"]),
-            stable_id=cast("str | None", meta["stable_id"]),
-            schema_version=cast("str | None", meta["schema_version"]),
-            upstream_dependencies=cast("tuple[str, ...]", meta["upstream_dependencies"]),
-            validation_profile=cast("Literal['strict', 'lenient']", meta["validation_profile"]),
-        )
-        by_name[name] = ds
-        by_table[table_key] = ds
-        if jsonl_filename:
-            jsonl_map[table_key] = jsonl_filename
-        if parquet_filename:
-            parquet_map[table_key] = parquet_filename
-
-    return DatasetRegistry(
-        by_name=by_name,
-        by_table_key=by_table,
-        jsonl_datasets=jsonl_map,
-        parquet_datasets=parquet_map,
-    )
+    rows = con.execute(
+        """
+        SELECT
+            table_key,
+            name,
+            is_view,
+            jsonl_filename,
+            parquet_filename,
+            family,
+            description
+        FROM metadata.datasets
+        ORDER BY table_key
+        """
+    ).fetchall()
+
+    by_name: dict[str, Dataset] = {}
+    by_table: dict[str, Dataset] = {}
+    jsonl_map: dict[str, str] = {}
+    parquet_map: dict[str, str] = {}
+
+    for (
+        table_key,
+        name,
+        is_view,
+        jsonl_filename,
+        parquet_filename,
+        db_family,
+        db_description,
+    ) in rows:
+        base = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
+        if base is None:
+            msg = f"metadata.datasets row {table_key} has no DatasetContract"
+            raise KeyError(msg)
+
+        # Prefer DB overrides when present; fall back to the contract.
+        inferred_family = table_key.split(".", maxsplit=1)[0] if "." in table_key else None
+        family = (
+            db_family
+            if db_family is not None
+            else base.family
+            if base.family is not None
+            else inferred_family
+        )
+        description = (
+            db_description
+            if db_description is not None
+            else base.description
+        )
+        effective_jsonl = jsonl_filename or base.jsonl_filename
+        effective_parquet = parquet_filename or base.parquet_filename
+
+        ds = Dataset(
+            table_key=table_key,
+            name=name,
+            schema=base.schema,
+            row_binding=base.row_binding,
+            jsonl_filename=effective_jsonl,
+            parquet_filename=effective_parquet,
+            is_view=bool(is_view),
+            json_schema_id=base.json_schema_id,
+            description=cast("str | None", description),
+            family=family,
+            owner=base.owner,
+            freshness_sla=base.freshness_sla,
+            retention_policy=base.retention_policy,
+            stable_id=base.stable_id,
+            schema_version=base.schema_version,
+            upstream_dependencies=base.upstream_dependencies,
+            validation_profile=base.validation_profile,
+        )
+        by_name[name] = ds
+        by_table[table_key] = ds
+        if ds.jsonl_filename:
+            jsonl_map[table_key] = ds.jsonl_filename
+        if ds.parquet_filename:
+            parquet_map[table_key] = ds.parquet_filename
+
+    return DatasetRegistry(
+        by_name=by_name,
+        by_table_key=by_table,
+        jsonl_datasets=jsonl_map,
+        parquet_datasets=parquet_map,
+    )
```

---

### What this patch achieves (in practice)

* **Single SSoT**: The runtime registry is now explicitly hydrated from `DATASET_CONTRACTS_BY_TABLE_KEY`, so any change to a `DatasetContract` automatically reflects in:

  * `DatasetRegistry`
  * downstream callers (`dataset_specs`, serving meta, export, etc.).
* **DB-as-materialization**: DuckDB’s `metadata.datasets` becomes an override layer for a few fields (`family`, `description`, filenames) instead of the primary source of truth.
* **Fewer moving parts**: The `load_dataset_registry` function no longer needs to understand:

  * row bindings (`ROW_BINDINGS_BY_TABLE_KEY`)
  * JSON Schema bindings (`JSON_SCHEMA_BY_DATASET_NAME`)
  * dependency maps, etc.
    all of which now live in the contract module.

