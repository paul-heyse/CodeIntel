"""Contract helper utilities for tests."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.core.schemas.contract_factory import is_docs_view
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.storage.helpers.table_key import split_table_key
from tests._helpers.db import count_rows


@dataclass(frozen=True)
class ContractCtx:
    """Context for contract validation tests."""

    gateway: object
    repo: str
    commit: str


def contract_for_keys(*table_keys: str) -> dict[str, DatasetContract]:
    """Build minimal DatasetContract entries for the supplied table keys.

    Returns
    -------
    dict[str, DatasetContract]
        Mapping of table key to minimal dataset contract.
    """
    contracts: dict[str, DatasetContract] = {}
    for table_key in table_keys:
        _, name = split_table_key(table_key)
        contracts[table_key] = DatasetContract(
            table_key=table_key,
            name=name,
            schema=None,
            is_view=is_docs_view(table_key),
        )
    return contracts


__all__ = [
    "ContractCtx",
    "contract_for_keys",
    "count_rows",
]
