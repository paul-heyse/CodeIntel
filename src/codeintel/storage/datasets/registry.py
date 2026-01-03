"""Dataset metadata registry backed by DuckDB's metadata.datasets table."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from sqlglot import exp

from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.sqlglot_tools import render_sql_duckdb, table_expr_from_ref
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.contracts.provider import get_contract_for_table_key
from codeintel.storage.datasets.manifests import load_dataset_manifest
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.query_results import iter_tuples_from_arrow_reader

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from duckdb import DuckDBPyConnection

    from codeintel.core.manifests import ArrowDatasetManifest

__all__ = [
    "DatasetRegistry",
    "attach_dataset_manifests",
    "build_dataset_dependency_graph",
    "dataset_for_name",
    "dataset_for_table",
    "describe_all_datasets",
    "describe_dataset",
    "list_dataset_specs",
    "load_dataset_manifests_for_snapshot",
    "load_dataset_registry",
]


@dataclass(frozen=True)
class DatasetRegistry:
    """In-memory view of metadata.datasets plus Python row bindings."""

    by_name: Mapping[str, DatasetContract]
    by_table_key: Mapping[str, DatasetContract]
    jsonl_datasets: Mapping[str, str]
    parquet_datasets: Mapping[str, str]
    dataset_root_dir: Path | None = None
    dataset_manifests: Mapping[str, ArrowDatasetManifest] = field(default_factory=dict)

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

    def dataset_manifest_for_table(self, table_key: str) -> ArrowDatasetManifest | None:
        """Return the Arrow dataset manifest for a table key, if present.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        ArrowDatasetManifest | None
            Manifest for the table key when available.
        """
        return self.dataset_manifests.get(table_key)

    def with_dataset_root(self, dataset_root_dir: Path | None) -> DatasetRegistry:
        """Return a new registry with dataset root configured.

        Parameters
        ----------
        dataset_root_dir
            Root directory for Arrow dataset snapshots, when available.

        Returns
        -------
        DatasetRegistry
            New registry with dataset root metadata attached.
        """
        return DatasetRegistry(
            by_name=self.by_name,
            by_table_key=self.by_table_key,
            jsonl_datasets=self.jsonl_datasets,
            parquet_datasets=self.parquet_datasets,
            dataset_root_dir=dataset_root_dir,
            dataset_manifests=self.dataset_manifests,
        )

    def with_dataset_manifests(
        self,
        dataset_manifests: Mapping[str, ArrowDatasetManifest],
    ) -> DatasetRegistry:
        """Return a new registry with dataset manifest bindings.

        Parameters
        ----------
        dataset_manifests
            Mapping of table_key to Arrow dataset manifest metadata.

        Returns
        -------
        DatasetRegistry
            New registry with dataset manifest metadata attached.
        """
        return DatasetRegistry(
            by_name=self.by_name,
            by_table_key=self.by_table_key,
            jsonl_datasets=self.jsonl_datasets,
            parquet_datasets=self.parquet_datasets,
            dataset_root_dir=self.dataset_root_dir,
            dataset_manifests=dict(dataset_manifests),
        )


def load_dataset_registry(
    con: DuckDBPyConnection,
    *,
    dataset_root_dir: Path | None = None,
    dataset_manifests: Mapping[str, ArrowDatasetManifest] | None = None,
) -> DatasetRegistry:
    """Load dataset metadata from DuckDB's metadata.datasets table.

    Hydrates DatasetContracts by merging database metadata with contract
    defaults from the contract provider.

    Parameters
    ----------
    con
        DuckDB connection used to read metadata datasets.
    dataset_root_dir
        Optional Arrow dataset root path for downstream consumers.
    dataset_manifests
        Optional mapping of table_key to Arrow dataset manifests.

    Returns
    -------
    DatasetRegistry
        Registry constructed from the database and contract defaults.

    Raises
    ------
    KeyError
        If a metadata row lacks a corresponding DatasetContract.
    """
    table_ref = meta_table_ref("metadata.datasets")
    table_expr = table_expr_from_ref(table_ref)
    query = (
        exp.select(
            exp.column("table_key"),
            exp.column("name"),
            exp.column("is_view"),
            exp.column("jsonl_filename"),
            exp.column("parquet_filename"),
            exp.column("family"),
            exp.column("description"),
        )
        .from_(table_expr)
        .order_by(exp.Ordered(this=exp.column("table_key")))
    )
    reader = con.execute(render_sql_duckdb(query)).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)

    try:
        return _registry_from_rows(
            iter_tuples_from_arrow_reader(reader),
            dataset_root_dir=dataset_root_dir,
            dataset_manifests=dataset_manifests,
        )
    except KeyError as exc:
        msg = str(exc)
        raise KeyError(msg) from exc


def _registry_from_rows(
    rows: Iterable[tuple[object, ...]],
    *,
    dataset_root_dir: Path | None,
    dataset_manifests: Mapping[str, ArrowDatasetManifest] | None,
) -> DatasetRegistry:
    by_name: dict[str, DatasetContract] = {}
    by_table: dict[str, DatasetContract] = {}
    jsonl_map: dict[str, str] = {}
    parquet_map: dict[str, str] = {}

    for row in rows:
        ds = _dataset_contract_from_row(row)
        by_name[ds.name] = ds
        by_table[ds.table_key] = ds
        if ds.jsonl_filename:
            jsonl_map[ds.table_key] = ds.jsonl_filename
        if ds.parquet_filename:
            parquet_map[ds.table_key] = ds.parquet_filename

    return DatasetRegistry(
        by_name=by_name,
        by_table_key=by_table,
        jsonl_datasets=jsonl_map,
        parquet_datasets=parquet_map,
        dataset_root_dir=dataset_root_dir,
        dataset_manifests=dict(dataset_manifests or {}),
    )


def _dataset_contract_from_row(row: tuple[object, ...]) -> DatasetContract:
    (
        table_key,
        name,
        is_view,
        jsonl_filename,
        parquet_filename,
        db_family,
        db_description,
    ) = row
    try:
        base = get_contract_for_table_key(cast("str", table_key))
    except KeyError:
        msg = f"metadata.datasets row {table_key} has no DatasetContract"
        raise KeyError(msg) from None

    inferred_family = split_table_key(cast("str", table_key))[0] if "." in str(table_key) else None
    family = (
        db_family
        if db_family is not None
        else base.family
        if base.family is not None
        else inferred_family
    )
    description = db_description if db_description is not None else base.description
    effective_jsonl = cast("str | None", jsonl_filename) or base.jsonl_filename
    effective_parquet = cast("str | None", parquet_filename) or base.parquet_filename

    return DatasetContract(
        table_key=cast("str", table_key),
        name=cast("str", name),
        schema=base.schema,
        row_binding=base.row_binding,
        json_schema_id=base.json_schema_id,
        jsonl_filename=effective_jsonl,
        parquet_filename=effective_parquet,
        is_view=bool(is_view),
        owner_package=base.owner_package,
        tags=base.tags,
        description=cast("str | None", description),
        family=cast("str | None", family),
        owner=base.owner,
        freshness_sla=base.freshness_sla,
        retention_policy=base.retention_policy,
        stable_id=base.stable_id,
        schema_version=base.schema_version,
        upstream_dependencies=base.upstream_dependencies,
        validation_profile=base.validation_profile,
    )


def attach_dataset_manifests(
    registry: DatasetRegistry,
    *,
    dataset_root_dir: Path | None,
    dataset_manifests: Mapping[str, ArrowDatasetManifest] | None = None,
) -> DatasetRegistry:
    """Return a registry augmented with Arrow dataset manifest metadata.

    Parameters
    ----------
    registry
        Base dataset registry to augment.
    dataset_root_dir
        Root directory for Arrow dataset snapshots, when available.
    dataset_manifests
        Mapping of table_key to Arrow dataset manifest metadata.

    Returns
    -------
    DatasetRegistry
        Registry augmented with dataset manifest metadata.
    """
    return DatasetRegistry(
        by_name=registry.by_name,
        by_table_key=registry.by_table_key,
        jsonl_datasets=registry.jsonl_datasets,
        parquet_datasets=registry.parquet_datasets,
        dataset_root_dir=dataset_root_dir,
        dataset_manifests=dict(dataset_manifests or {}),
    )


def load_dataset_manifests_for_snapshot(
    registry: DatasetRegistry,
    *,
    dataset_root_dir: Path,
    snapshot_id: str,
) -> dict[str, ArrowDatasetManifest]:
    """Load dataset manifests for a snapshot from the dataset root.

    Parameters
    ----------
    registry
        Dataset registry with table keys and contracts.
    dataset_root_dir
        Root directory for Arrow dataset snapshots.
    snapshot_id
        Snapshot identifier for the manifests.

    Returns
    -------
    dict[str, ArrowDatasetManifest]
        Mapping of table_key to manifest payloads that were found on disk.
    """
    manifests: dict[str, ArrowDatasetManifest] = {}
    if not dataset_root_dir.is_dir():
        return manifests
    for table_key, dataset in registry.by_table_key.items():
        if dataset.is_view:
            continue
        manifest = load_dataset_manifest(
            dataset_root=dataset_root_dir,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
        if manifest is not None:
            manifests[table_key] = manifest
    return manifests


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
