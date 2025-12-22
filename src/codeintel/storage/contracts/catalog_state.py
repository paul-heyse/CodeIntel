"""In-memory canonical contract catalog state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.schemas.contract_primitives import DatasetContract

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema


@dataclass(slots=True)
class _ContractCatalogState:
    catalog: dict[str, DatasetContract] | None = None


_CONTRACT_CATALOG_STATE = _ContractCatalogState()


def set_contract_catalog(contracts: Mapping[str, DatasetContract] | None) -> None:
    """Set the canonical contract catalog mapping."""
    _CONTRACT_CATALOG_STATE.catalog = dict(contracts) if contracts is not None else None


def get_contract_catalog() -> dict[str, DatasetContract] | None:
    """Return the current contract catalog mapping, if loaded.

    Returns
    -------
    dict[str, DatasetContract] | None
        Current contract mapping when loaded; otherwise None.
    """
    return _CONTRACT_CATALOG_STATE.catalog


def contract_catalog_table_schemas() -> dict[str, TableSchema]:
    """Return table schemas from the loaded contract catalog.

    Returns
    -------
    dict[str, TableSchema]
        Mapping of table keys to table schemas.
    """
    catalog = _CONTRACT_CATALOG_STATE.catalog
    if catalog is None:
        return {}
    return {
        table_key: contract.schema
        for table_key, contract in catalog.items()
        if contract.schema is not None
    }


__all__ = [
    "contract_catalog_table_schemas",
    "get_contract_catalog",
    "set_contract_catalog",
]
