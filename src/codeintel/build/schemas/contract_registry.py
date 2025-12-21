"""Canonical contract registry backed by cached catalogs."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.catalogs.canonical import load_contract_catalog
from codeintel.core.schemas.contract_primitives import DatasetContract

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


@lru_cache(maxsize=1)
def get_dag_free_contracts_by_table_key() -> Mapping[str, DatasetContract]:
    """Return canonical dataset contracts keyed by table_key.

    Returns
    -------
    Mapping[str, DatasetContract]
        Mapping of table_key to DatasetContract for CLI-friendly enumeration.
    """
    return load_contract_catalog()


def iter_dag_free_contracts() -> Iterable[DatasetContract]:
    """Iterate DAG-free dataset contracts.

    Returns
    -------
    Iterable[DatasetContract]
        Iterable of DAG-free dataset contracts.
    """
    return get_dag_free_contracts_by_table_key().values()


def iter_dag_free_contracts_by_table_key() -> Iterable[tuple[str, DatasetContract]]:
    """Iterate DAG-free dataset contracts as (table_key, contract) pairs.

    Returns
    -------
    Iterable[tuple[str, DatasetContract]]
        Iterable of table key and contract pairs.
    """
    return get_dag_free_contracts_by_table_key().items()


def clear_dag_free_contract_cache() -> None:
    """Clear cached DAG-free contract registry state."""
    get_dag_free_contracts_by_table_key.cache_clear()


__all__ = [
    "clear_dag_free_contract_cache",
    "get_dag_free_contracts_by_table_key",
    "iter_dag_free_contracts",
    "iter_dag_free_contracts_by_table_key",
]
