"""PyArrow Parquet adapters for Hamilton cache format support."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from hamilton import registry
from hamilton.io.data_adapters import DataLoader, DataSaver


@dataclass(frozen=True, slots=True)
class PyArrowParquetSaver(DataSaver):
    """Persist a PyArrow table to a Parquet file for cache storage."""

    path: str

    @classmethod
    def applicable_types(cls) -> Collection[type]:
        """Return the supported input types for Parquet caching.

        Returns
        -------
        Collection[type]
            Supported input types for the Parquet cache adapter.
        """
        return [pa.Table]

    @classmethod
    def name(cls) -> str:
        """Return the registry name for the Parquet cache adapter.

        Returns
        -------
        str
            Registry key for the Parquet cache adapter.
        """
        return "parquet"

    def save_data(self, data: pa.Table) -> dict[str, Any]:
        """Write a table to Parquet and return cache metadata.

        Returns
        -------
        dict[str, Any]
            Parquet cache metadata payload.
        """
        pq.write_table(data, self.path)
        return {
            "path": self.path,
            "format": "parquet",
            "rows": data.num_rows,
            "columns": data.num_columns,
        }


@dataclass(frozen=True, slots=True)
class PyArrowParquetLoader(DataLoader):
    """Load a PyArrow table from a Parquet cache file."""

    path: str

    @classmethod
    def applicable_types(cls) -> Collection[type]:
        """Return the supported output types for Parquet caching.

        Returns
        -------
        Collection[type]
            Supported output types for the Parquet cache adapter.
        """
        return [pa.Table]

    @classmethod
    def name(cls) -> str:
        """Return the registry name for the Parquet cache adapter.

        Returns
        -------
        str
            Registry key for the Parquet cache adapter.
        """
        return "parquet"

    def load_data(self, type_: type | None) -> tuple[pa.Table, dict[str, Any]]:
        """Read a Parquet file into a PyArrow table.

        Returns
        -------
        tuple[pa.Table, dict[str, Any]]
            Table and Parquet cache metadata payload.
        """
        _ = type_
        table = pq.read_table(self.path)
        return (
            table,
            {
                "path": self.path,
                "format": "parquet",
                "rows": table.num_rows,
                "columns": table.num_columns,
            },
        )


def register_arrow_parquet_cache_adapters() -> None:
    """Register PyArrow Parquet adapters for Hamilton cache format support."""
    savers = registry.SAVER_REGISTRY.get(PyArrowParquetSaver.name(), [])
    loaders = registry.LOADER_REGISTRY.get(PyArrowParquetLoader.name(), [])
    if PyArrowParquetSaver in savers and PyArrowParquetLoader in loaders:
        return
    registry.register_adapter(PyArrowParquetSaver)
    registry.register_adapter(PyArrowParquetLoader)


__all__ = [
    "PyArrowParquetLoader",
    "PyArrowParquetSaver",
    "register_arrow_parquet_cache_adapters",
]
