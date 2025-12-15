"""Dataset schema primitives and column utilities.

This package provides primitive types for defining table schemas and
utilities for column-based operations.

For schema and contract access, prefer the build-owned providers:

- Schemas: ``codeintel.build.schemas.get_schema_provider()``
- Contracts: ``codeintel.build.schemas.get_contract_for_table_key()``
- Row bindings: ``codeintel.build.schemas.get_row_binding()``
- Row types: ``codeintel.core.schemas.generated_types``
- Dataflow graph: ``codeintel.config.datasets.dataflow.build_contract_dataflow_graph()``
"""

from __future__ import annotations

import importlib

from codeintel.config.datasets.columns import load_columns_by_table, serialize_row
from codeintel.config.datasets.primitives import Column, ColumnType, Index, TableSchema
from codeintel.core.schemas.contract_primitives import DatasetContract, RowBinding


def get_row_bindings() -> dict[str, RowBinding]:
    """Return schema-generated row bindings for all declared table schemas.

    Returns
    -------
    dict[str, RowBinding]
        Mapping from table_key to row binding.
    """
    mod = importlib.import_module("codeintel.config.datasets.contracts")
    return mod.get_row_bindings()


def get_table_schemas() -> dict[str, TableSchema]:
    """Return declared table schemas keyed by table key.

    Returns
    -------
    dict[str, TableSchema]
        Mapping from table_key to TableSchema.
    """
    mod = importlib.import_module("codeintel.config.datasets.contracts")
    return mod.get_table_schemas()


__all__ = [
    "Column",
    "ColumnType",
    "DatasetContract",
    "Index",
    "RowBinding",
    "TableSchema",
    "get_row_bindings",
    "get_table_schemas",
    "load_columns_by_table",
    "serialize_row",
]
