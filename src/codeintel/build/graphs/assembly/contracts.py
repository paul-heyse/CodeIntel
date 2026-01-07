"""Contract alignment helpers for Arrow-first graph assembly."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.contracts.types import ContractPolicy
from codeintel.build.tabular.arrow_ops import (
    align_reader_to_contract as _align_reader_to_contract,
)
from codeintel.build.tabular.arrow_ops import (
    align_table_to_contract as _align_table_to_contract,
)
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.schemas.arrow_gen import ExtrasPolicy


def align_reader_to_contract(
    table_key: str,
    reader: pa.RecordBatchReader,
    *,
    target_name: str | None = None,
    policy: ContractPolicy | None = None,
    extras_policy: ExtrasPolicy | None = None,
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
        target_name=target_name,
        policy=policy,
        extras_policy=extras_policy,
    )


def align_table_to_contract(
    table_key: str,
    table: pa.Table,
    *,
    target_name: str | None = None,
    policy: ContractPolicy | None = None,
    extras_policy: ExtrasPolicy | None = None,
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
        target_name=target_name,
        policy=policy,
        extras_policy=extras_policy,
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
