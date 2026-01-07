"""Helpers for parquet-backed dataset snapshots in tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow as pa

from codeintel.core.columnar.rows import ColumnarRows, columnar_row_count
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.datasets.arrow_store import scan_dataset_reader, write_dataset
from codeintel.storage.datasets.scanning import DatasetScanOptions
from tests._helpers.columnar_streams import table_for_rows


def write_snapshot_rows(
    dataset_root: Path,
    *,
    table_key: str,
    snapshot_id: str,
    rows: Sequence[Mapping[str, object]],
    allow_empty: bool = False,
) -> None:
    """Write row mappings into a snapshot parquet dataset."""
    if not rows and not allow_empty:
        return
    table = table_for_rows(table_key, rows)
    write_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        data=table,
    )


def write_snapshot_rows_raw(
    dataset_root: Path,
    *,
    table_key: str,
    snapshot_id: str,
    rows: Sequence[Mapping[str, object]],
    schema: pa.Schema | None = None,
) -> None:
    """Write row mappings to a parquet dataset without contract alignment.

    Raises
    ------
    ValueError
        If the dataset is empty and no schema is provided.
    """
    if not rows and schema is None:
        msg = f"Schema required for empty parquet dataset: {table_key}"
        raise ValueError(msg)
    table = pa.Table.from_pylist(list(rows), schema=schema)
    write_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        data=table,
    )


def rows_from_columnar_rows(rows: ColumnarRows) -> list[dict[str, object]]:
    """Expand columnar rows into row mappings.

    Returns
    -------
    list[dict[str, object]]
        Row mappings aligned to column order.
    """
    columns = list(rows.keys())
    count = columnar_row_count(rows)
    return [{name: rows[name][idx] for name in columns} for idx in range(count)]


def read_snapshot_rows(
    dataset_root: Path,
    *,
    table_key: str,
    snapshot_id: str,
    columns: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Read snapshot rows into a list of dictionaries.

    Returns
    -------
    list[dict[str, object]]
        Row mappings from the dataset snapshot.
    """
    options = DatasetScanOptions(
        batch_size=DEFAULT_ARROW_BATCH_SIZE,
        columns=tuple(columns) if columns is not None else None,
    )
    reader = scan_dataset_reader(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=options,
    )
    table = pa.Table.from_batches(list(reader), schema=reader.schema)
    return list(table.to_pylist())


__all__ = [
    "read_snapshot_rows",
    "rows_from_columnar_rows",
    "write_snapshot_rows",
    "write_snapshot_rows_raw",
]
