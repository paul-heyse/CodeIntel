"""Deprecated shim for legacy dataset helpers.

This package used to be the source of truth for:
- Table schemas (TABLE_SCHEMAS)
- Dataset contracts and row bindings
- Hand-maintained TypedDict row models and serializers

Hamilton Consolidation moved schema and contract authority to build-owned providers.
New code should prefer:
- Schemas: ``codeintel.build.schemas.get_schema_provider()``
- Contracts: ``codeintel.build.schemas.get_contract_for_table_key()``
- Row bindings: ``codeintel.build.schemas.get_row_binding()``
- Row types: ``codeintel.core.schemas.generated_types``

Notes
-----
This module intentionally avoids eager imports of legacy helpers to prevent
circular imports (notably between schema declarations and dataset shims).
Consumers should import the concrete submodules directly when possible.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from codeintel.config.datasets.columns import load_columns_by_table, serialize_row
from codeintel.config.datasets.primitives import Column, ColumnType, Index, TableSchema
from codeintel.core.schemas.contract_primitives import DatasetContract, RowBinding

if TYPE_CHECKING:
    from codeintel.config.datasets.dataflow import DataflowEdge, DataflowNode


def build_contract_dataflow_graph() -> tuple[list[DataflowNode], list[DataflowEdge]]:
    """Return the dataset lineage graph derived from dataset contracts.

    Returns
    -------
    tuple[list[DataflowNode], list[DataflowEdge]]
        Nodes and edges describing datasets, views, and composition relationships.
    """
    mod = importlib.import_module("codeintel.config.datasets.dataflow")
    return mod.build_contract_dataflow_graph()


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
    "build_contract_dataflow_graph",
    "get_row_bindings",
    "get_table_schemas",
    "load_columns_by_table",
    "serialize_row",
]
