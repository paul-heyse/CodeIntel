"""Contract alignment helpers for Arrow-first graph assembly."""

from __future__ import annotations

from typing import Unpack

import pyarrow as pa

from codeintel.build.tabular.arrow_ops import (
    AlignmentOptions,
    AlignmentOverrides,
)
from codeintel.build.tabular.arrow_ops import (
    align_reader_to_contract as _align_reader_to_contract,
)
from codeintel.build.tabular.arrow_ops import (
    align_table_to_contract as _align_table_to_contract,
)
from codeintel.core.columnar.rows import empty_table_for_table


def align_reader_to_contract(
    table_key: str,
    reader: pa.RecordBatchReader,
    *,
    options: AlignmentOptions | None = None,
    **overrides: Unpack[AlignmentOverrides],
) -> pa.RecordBatchReader:
    """Align a RecordBatchReader to the contract schema for the table key.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader aligned to the contract schema.
    """
    return _align_reader_to_contract(
        table_key,
        reader,
        options=options,
        **overrides,
    )


def align_table_to_contract(
    table_key: str,
    table: pa.Table,
    *,
    options: AlignmentOptions | None = None,
    **overrides: Unpack[AlignmentOverrides],
) -> pa.Table:
    """Align a Table to the contract schema for the table key.

    Returns
    -------
    pyarrow.Table
        Table aligned to the contract schema.
    """
    return _align_table_to_contract(
        table_key,
        table,
        options=options,
        **overrides,
    )


def empty_contract_reader(table_key: str) -> pa.Table:
    """Return an empty table aligned to the table contract.

    Returns
    -------
    pyarrow.Table
        Empty table with the contract schema.
    """
    return empty_table_for_table(table_key)


__all__ = [
    "align_reader_to_contract",
    "align_table_to_contract",
    "empty_contract_reader",
]
