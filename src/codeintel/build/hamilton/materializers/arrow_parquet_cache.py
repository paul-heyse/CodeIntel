"""PyArrow Parquet adapters for Hamilton cache format support."""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq
from hamilton import registry
from hamilton.io.data_adapters import DataLoader, DataSaver

from codeintel.core.columnar.finalize_ops import finalize_spec_for_table, finalize_table
from codeintel.core.columnar.kernels import SortKey
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.datasets.parquet_metadata import read_parquet_metadata, read_parquet_schema
from codeintel.core.schemas.arrow_polars import table_schema_from_arrow_schema
from codeintel.core.schemas.primitives import resolve_stable_sort_keys

LOG = logging.getLogger(__name__)
ORDER_ASC: Final = "ascending"


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
        finalized = _finalize_cache_table(data)
        pq.write_table(finalized, self.path)
        return {
            "path": self.path,
            "format": "parquet",
            "rows": finalized.num_rows,
            "columns": finalized.num_columns,
        }


_DICTIONARY_ENCODINGS = frozenset({"PLAIN_DICTIONARY", "RLE_DICTIONARY"})


def _finalize_cache_table(table: pa.Table) -> pa.Table:
    try:
        table_schema = table_schema_from_arrow_schema(arrow_schema=table.schema)
    except (TypeError, ValueError) as exc:
        LOG.debug("Cache finalize skipped: %s", exc)
        return table
    stable_sort_keys = resolve_stable_sort_keys(table_schema)
    order_by = _order_by_for_keys(stable_sort_keys)
    result = finalize_table(
        table,
        spec=finalize_spec_for_table(
            table_schema.table_key,
            mode="tolerant",
            order_by=order_by,
        ),
    )
    if result.errors.num_rows:
        LOG.warning(
            "Parquet cache finalize produced %d error rows for %s",
            result.errors.num_rows,
            table_schema.table_key,
        )
    return result.good


def _order_by_for_keys(
    stable_sort_keys: tuple[str, ...] | None,
) -> tuple[SortKey, ...]:
    if not stable_sort_keys:
        return ()
    return tuple((key, ORDER_ASC) for key in stable_sort_keys)


def _has_dictionary_encoding(column: pq.ColumnChunkMetaData) -> bool:
    encodings = getattr(column, "encodings", None)
    if not encodings:
        return False
    return any(str(encoding).upper() in _DICTIONARY_ENCODINGS for encoding in encodings)


def _is_dictionary_candidate(data_type: pa.DataType) -> bool:
    return pa.types.is_string(data_type) or pa.types.is_large_string(data_type)


def _dictionary_columns_for_path(path: Path) -> tuple[str, ...] | None:
    schema = read_parquet_schema(path)
    if schema is None:
        return None
    column_names = set(schema.names)
    dictionary_columns: set[str] = set()
    metadata = read_parquet_metadata(path)
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
        dictionary_columns = _dictionary_columns_for_path(Path(self.path))
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
