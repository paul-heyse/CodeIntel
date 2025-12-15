"""Dataset metadata registry backed by DuckDB's metadata.datasets table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from codeintel.build.schemas import get_contract_for_table_key
from codeintel.core.schemas.contract_primitives import DatasetContract

if TYPE_CHECKING:
    from collections.abc import Mapping

    from duckdb import DuckDBPyConnection


__all__ = [
    "DatasetRegistry",
    "build_dataset_dependency_graph",
    "dataset_for_name",
    "dataset_for_table",
    "describe_all_datasets",
    "describe_dataset",
    "list_dataset_specs",
    "load_dataset_registry",
]


@dataclass(frozen=True)
class DatasetRegistry:
    """In-memory view of metadata.datasets plus Python row bindings."""

    by_name: Mapping[str, DatasetContract]
    by_table_key: Mapping[str, DatasetContract]
    jsonl_datasets: Mapping[str, str]
    parquet_datasets: Mapping[str, str]

    @property
    def all_datasets(self) -> tuple[str, ...]:
        """Return all dataset names.

        Returns
        -------
        tuple[str, ...]
            Dataset identifiers present in the registry.
        """
        return tuple(self.by_name.keys())

    def datasets_with_json_schema(self) -> tuple[str, ...]:
        """Return dataset names that have JSON Schema validation configured.

        Returns
        -------
        tuple[str, ...]
            Dataset names with JSON Schema bindings.
        """
        return tuple(name for name, ds in self.by_name.items() if ds.json_schema_id is not None)

    def dataset_dependencies(self) -> dict[str, tuple[str, ...]]:
        """Return upstream dependencies for each dataset.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Mapping of dataset name to upstream dependencies.
        """
        return {
            name: ds.upstream_dependencies
            for name, ds in self.by_name.items()
            if ds.upstream_dependencies
        }

    def docs_dataset_names(self) -> tuple[str, ...]:
        """Return dataset names backed by docs.* views.

        Returns
        -------
        tuple[str, ...]
            Dataset names for docs schema views.
        """
        return tuple(
            name
            for name, ds in self.by_name.items()
            if ds.is_view and ds.table_key.startswith("docs.")
        )

    def resolve_table_key(self, name: str) -> str:
        """Resolve dataset name into fully qualified table or view key.

        Returns
        -------
        str
            Fully qualified table or view identifier.

        Raises
        ------
        KeyError
            If the dataset is unknown.
        """
        ds = self.by_name.get(name)
        if ds is None:
            message = f"Unknown dataset: {name}"
            raise KeyError(message)
        return ds.table_key

    @property
    def mapping(self) -> Mapping[str, str]:
        """Return name -> table_key mapping for compatibility.

        Returns
        -------
        Mapping[str, str]
            Mapping from dataset name to fully qualified table key.
        """
        return {name: ds.table_key for name, ds in self.by_name.items()}

    @property
    def tables(self) -> tuple[str, ...]:
        """Return table dataset names (non-views).

        Returns
        -------
        tuple[str, ...]
            Dataset names that are base tables (not views).
        """
        return tuple(name for name, ds in self.by_name.items() if not ds.is_view)

    @property
    def views(self) -> tuple[str, ...]:
        """Return view dataset names.

        Returns
        -------
        tuple[str, ...]
            Dataset names that are views.
        """
        return tuple(name for name, ds in self.by_name.items() if ds.is_view)

    @property
    def meta(self) -> Mapping[str, DatasetContract]:
        """Return name -> contract mapping (alias for by_name).

        Returns
        -------
        Mapping[str, DatasetContract]
            Mapping from dataset name to DatasetContract.
        """
        return self.by_name

    @property
    def jsonl_mapping(self) -> Mapping[str, str]:
        """Return jsonl_datasets (alias for compatibility).

        Returns
        -------
        Mapping[str, str]
            Mapping from table key to JSONL filename.
        """
        return self.jsonl_datasets

    @property
    def parquet_mapping(self) -> Mapping[str, str]:
        """Return parquet_datasets (alias for compatibility).

        Returns
        -------
        Mapping[str, str]
            Mapping from table key to Parquet filename.
        """
        return self.parquet_datasets

    def table_for_name(self, name: str) -> str:
        """Return table_key for dataset name (alias for resolve_table_key).

        Delegates to :meth:`resolve_table_key` for actual resolution.

        Parameters
        ----------
        name
            Dataset name to resolve.

        Returns
        -------
        str
            Fully qualified table key.
        """
        return self.resolve_table_key(name)


def load_dataset_registry(con: DuckDBPyConnection) -> DatasetRegistry:
    """Load dataset metadata from DuckDB's metadata.datasets table.

    Hydrates DatasetContracts by merging database metadata with contract
    defaults from the contract provider.

    Returns
    -------
    DatasetRegistry
        Registry constructed from the database and contract defaults.

    Raises
    ------
    KeyError
        If a metadata row lacks a corresponding DatasetContract.
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

    by_name: dict[str, DatasetContract] = {}
    by_table: dict[str, DatasetContract] = {}
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
        try:
            base = get_contract_for_table_key(table_key)
        except KeyError:
            msg = f"metadata.datasets row {table_key} has no DatasetContract"
            raise KeyError(msg) from None

        inferred_family = table_key.split(".", maxsplit=1)[0] if "." in table_key else None
        family = (
            db_family
            if db_family is not None
            else base.family
            if base.family is not None
            else inferred_family
        )
        description = db_description if db_description is not None else base.description
        effective_jsonl = jsonl_filename or base.jsonl_filename
        effective_parquet = parquet_filename or base.parquet_filename

        ds = DatasetContract(
            table_key=table_key,
            name=name,
            schema=base.schema,
            row_binding=base.row_binding,
            json_schema_id=base.json_schema_id,
            jsonl_filename=effective_jsonl,
            parquet_filename=effective_parquet,
            is_view=bool(is_view),
            owner_package=base.owner_package,
            tags=base.tags,
            description=cast("str | None", description),
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


def dataset_for_name(registry: DatasetRegistry, name: str) -> DatasetContract:
    """Return dataset metadata for a dataset name.

    Returns
    -------
    Dataset
        Dataset resolved from the registry.

    Raises
    ------
    KeyError
        If the dataset name is not present.
    """
    ds = registry.by_name.get(name)
    if ds is None:
        message = f"Unknown dataset name: {name}"
        raise KeyError(message)
    return ds


def dataset_for_table(registry: DatasetRegistry, table_key: str) -> DatasetContract:
    """Return dataset metadata for a fully qualified table or view key.

    Returns
    -------
    Dataset
        Dataset resolved from the registry.

    Raises
    ------
    KeyError
        If the table key is not present.
    """
    ds = registry.by_table_key.get(table_key)
    if ds is None:
        message = f"Unknown dataset table key: {table_key}"
        raise KeyError(message)
    return ds


def describe_dataset(ds: DatasetContract) -> dict[str, object]:
    """Return a JSON-serializable description of a dataset spec.

    Returns
    -------
    dict[str, object]
        Dataset description derived from the registry entry.
    """
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
        "family": ds.family,
        "owner": ds.owner,
        "freshness_sla": ds.freshness_sla,
        "retention_policy": ds.retention_policy,
        "stable_id": ds.stable_id,
        "schema_version": ds.schema_version,
        "upstream_dependencies": list(ds.upstream_dependencies),
        "validation_profile": ds.validation_profile,
        "capabilities": ds.capabilities(),
    }


def list_dataset_specs(registry: DatasetRegistry) -> list[dict[str, object]]:
    """Serialize all dataset specs from a DatasetRegistry.

    Returns
    -------
    list[dict[str, object]]
        JSON-friendly dataset spec mappings.
    """
    return [describe_dataset(ds) for ds in registry.by_name.values()]


def build_dataset_dependency_graph(registry: DatasetRegistry) -> dict[str, tuple[str, ...]]:
    """Construct a dependency graph mapping dataset -> upstream datasets.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Dependency mapping keyed by dataset name.
    """
    return registry.dataset_dependencies()


def describe_all_datasets(con: DuckDBPyConnection) -> list[dict[str, object]]:
    """Return a JSON-serializable description of all dataset specs for a database.

    Parameters
    ----------
    con
        Active DuckDB connection with metadata tables initialized.

    Returns
    -------
    list[dict[str, object]]
        Dataset descriptions derived from the active connection.
    """
    registry = load_dataset_registry(con)
    return list_dataset_specs(registry)
