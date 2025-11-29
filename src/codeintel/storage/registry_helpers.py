"""Dataset registry helpers shared across storage components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from duckdb import DuckDBPyConnection

from codeintel.storage.datasets import (
    Dataset,
    list_dataset_specs,
)
from codeintel.storage.datasets import (
    load_dataset_registry as _load_dataset_registry,
)

__all__ = ["DatasetRegistry", "build_dataset_registry", "describe_all_datasets"]


@dataclass(frozen=True)
class DatasetRegistry:
    """Track known table and view dataset names and export metadata."""

    mapping: Mapping[str, str]
    tables: tuple[str, ...]
    views: tuple[str, ...]
    meta: Mapping[str, Dataset] | None = None
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

    def table_for_name(self, name: str) -> str:
        """
        Return a validated dataset table or view key.

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


def build_dataset_registry(
    con: DuckDBPyConnection,
    *,
    include_views: bool = True,
) -> DatasetRegistry:
    """
    Build a dataset registry from DuckDB's metadata.datasets catalog.

    Parameters
    ----------
    con
        Active DuckDB connection with metadata tables initialized.
    include_views
        When True, include docs views alongside base tables.

    Returns
    -------
    DatasetRegistry
        Registry containing dataset metadata and export filenames.
    """
    ds_registry = _load_dataset_registry(con)
    mapping: dict[str, str] = {name: ds.table_key for name, ds in ds_registry.by_name.items()}
    table_names = tuple(name for name, ds in ds_registry.by_name.items() if not ds.is_view)
    view_names = (
        tuple(name for name, ds in ds_registry.by_name.items() if ds.is_view)
        if include_views
        else ()
    )
    return DatasetRegistry(
        mapping=mapping,
        tables=table_names,
        views=view_names,
        meta=ds_registry.by_name,
        jsonl_mapping=ds_registry.jsonl_datasets,
        parquet_mapping=ds_registry.parquet_datasets,
    )


def describe_all_datasets(con: DuckDBPyConnection) -> list[dict[str, object]]:
    """
    Return a JSON-serializable description of all dataset specs for this database.

    Returns
    -------
    list[dict[str, object]]
        Dataset descriptions derived from the active connection.
    """
    ds_registry = _load_dataset_registry(con)
    return list_dataset_specs(ds_registry)
