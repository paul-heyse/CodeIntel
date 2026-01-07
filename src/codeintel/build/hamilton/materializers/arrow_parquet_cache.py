"""PyArrow Parquet adapters for Hamilton cache format support."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from hamilton import registry
from hamilton.io.data_adapters import DataLoader, DataSaver

from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE


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


_DICTIONARY_ENCODINGS = frozenset({"PLAIN_DICTIONARY", "RLE_DICTIONARY"})


def _has_dictionary_encoding(column: pq.ColumnChunkMetaData) -> bool:
    encodings = getattr(column, "encodings", None)
    if not encodings:
        return False
    return any(str(encoding).upper() in _DICTIONARY_ENCODINGS for encoding in encodings)


def _is_dictionary_candidate(data_type: pa.DataType) -> bool:
    return pa.types.is_string(data_type) or pa.types.is_large_string(data_type)


def _dictionary_columns_for_path(path: str) -> tuple[str, ...] | None:
    try:
        parquet_file = pq.ParquetFile(path)
    except (OSError, ValueError, pa.ArrowInvalid):
        return None
    schema = parquet_file.schema_arrow
    column_names = set(schema.names)
    dictionary_columns: set[str] = set()
    metadata = parquet_file.metadata
    if metadata is not None:
        for group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(group_index)
            for column_index in range(row_group.num_columns):
                column = row_group.column(column_index)
                if not _has_dictionary_encoding(column):
                    continue
                column_name = column.path_in_schema
                if column_name not in column_names:
                    continue
                field = schema.field(column_name)
                if _is_dictionary_candidate(field.type):
                    dictionary_columns.add(column_name)
    if not dictionary_columns:
        dictionary_columns = {
            field.name for field in schema if _is_dictionary_candidate(field.type)
        }
    if not dictionary_columns:
        return None
    return tuple(sorted(dictionary_columns))


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
        dictionary_columns = _dictionary_columns_for_path(self.path)
        read_dictionary = list(dictionary_columns) if dictionary_columns else None
        try:
            parquet_file = pq.ParquetFile(
                self.path,
                memory_map=True,
                pre_buffer=True,
                read_dictionary=read_dictionary,
            )
            batches = parquet_file.iter_batches(
                batch_size=DEFAULT_ARROW_BATCH_SIZE,
                use_threads=True,
            )
            table = pa.Table.from_batches(batches, schema=parquet_file.schema_arrow)
        except (OSError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
            table = pq.read_table(self.path, read_dictionary=read_dictionary or False)
        if dictionary_columns:
            table = table.unify_dictionaries()
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
