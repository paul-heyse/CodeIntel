"""Dataset contract provider (storage-owned).

This module provides the canonical dataset contract interface used by the
storage layer. It is intentionally independent of `codeintel.build.*`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.config.datasets.composites import get_composite_schemas
from codeintel.core.schemas.contract_factory import (
    build_dataset_contract,
    is_docs_view,
)
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.contract_serde import contract_from_json_obj
from codeintel.core.schemas.row_models import row_binding_for_table_schema
from codeintel.core.schemas.service import SchemaService
from codeintel.core.singleton import SingletonHolder
from codeintel.storage.contracts.catalog_state import (
    get_contract_catalog,
)
from codeintel.storage.contracts.catalog_state import (
    set_contract_catalog as _set_contract_catalog,
)
from codeintel.storage.contracts.schema_provider import get_schema_provider
from codeintel.storage.metadata.catalogs import load_latest_canonical_catalog_from_connection
from codeintel.storage.views.inventory import discover_derived_docs_views

if TYPE_CHECKING:
    from collections.abc import Iterable

    from duckdb import DuckDBPyConnection

    from codeintel.config.datasets.primitives import CompositeSchema


_CONTRACT_CATALOG_KIND = "dataset_contracts"


def is_view(table_key: str) -> bool:
    """Return True when the table key represents a docs view.

    Returns
    -------
    bool
        True when the table key maps to a docs view.
    """
    return is_docs_view(table_key)


def _get_composition_for_table_key(table_key: str) -> CompositeSchema | None:
    return get_composite_schemas().get(table_key)


def _attach_row_binding(contract: DatasetContract) -> DatasetContract:
    if contract.schema is None:
        return contract
    binding = row_binding_for_table_schema(table_schema=contract.schema)
    return replace(contract, row_binding=binding)


def _contracts_from_payload(payload: Mapping[str, object]) -> dict[str, DatasetContract]:
    contracts_raw = payload.get("contracts")
    if not isinstance(contracts_raw, Mapping):
        return {}
    contracts: dict[str, DatasetContract] = {}
    for table_key, contract_obj in contracts_raw.items():
        if not isinstance(table_key, str) or not isinstance(contract_obj, Mapping):
            continue
        contracts[table_key] = _attach_row_binding(contract_from_json_obj(contract_obj))
    return contracts


def set_contract_catalog(contracts: Mapping[str, DatasetContract] | None) -> None:
    """Set the canonical contract catalog mapping."""
    _set_contract_catalog(contracts)
    get_contract_for_table_key.cache_clear()


def load_contract_catalog_from_connection(con: DuckDBPyConnection) -> None:
    """Load canonical contracts from metadata and store in-memory."""
    entry = load_latest_canonical_catalog_from_connection(
        con,
        catalog_kind=_CONTRACT_CATALOG_KIND,
    )
    if entry is None:
        set_contract_catalog(None)
        return
    contracts = _contracts_from_payload(entry.payload)
    if contracts:
        set_contract_catalog(contracts)
    else:
        set_contract_catalog(None)


@lru_cache(maxsize=1)
def _schema_service() -> SchemaService:
    return SchemaService(table_provider=get_schema_provider())


@lru_cache(maxsize=512)
def get_contract_for_table_key(table_key: str) -> DatasetContract:
    """Return the DatasetContract for a table or view.

    Parameters
    ----------
    table_key
        Fully qualified key (schema.table).

    Returns
    -------
    DatasetContract
        Contract describing the dataset or view.

    Raises
    ------
    KeyError
        Raised when the key is unknown to the schema provider and is not treated as a view.
    """
    catalog = get_contract_catalog()
    if catalog is not None:
        contract = catalog.get(table_key)
        if contract is None:
            msg = f"Unknown table key: {table_key}"
            raise KeyError(msg)
        return contract
    is_view = is_docs_view(table_key)
    schema = get_schema_provider().get_table_schema(table_key)
    if schema is None and not is_view:
        msg = f"Unknown table key: {table_key}"
        raise KeyError(msg)
    composition = _get_composition_for_table_key(table_key)
    return build_dataset_contract(
        table_key=table_key,
        schema_service=_schema_service(),
        overrides=None,
        composition=composition,
        is_view_override=is_view,
    )


def iter_contracts() -> Iterable[DatasetContract]:
    """Iterate all known dataset contracts.

    Yields
    ------
    DatasetContract
        Each known dataset contract.
    """
    catalog = get_contract_catalog()
    if catalog is not None:
        yield from catalog.values()
        return
    provider = get_schema_provider()
    seen: set[str] = set()

    for schema in provider.iter_table_schemas():
        table_key = schema.table_key
        if table_key in seen:
            continue
        seen.add(table_key)
        yield get_contract_for_table_key(table_key)

    for view_key in discover_derived_docs_views():
        if view_key in seen:
            continue
        seen.add(view_key)
        yield get_contract_for_table_key(view_key)


def iter_contracts_by_table_key() -> Iterable[tuple[str, DatasetContract]]:
    """Iterate all known contracts as (table_key, contract) pairs.

    Yields
    ------
    tuple[str, DatasetContract]
        Each (table_key, contract) pair.
    """
    for contract in iter_contracts():
        yield contract.table_key, contract


def clear_contract_cache() -> None:
    """Clear the contract cache (for testing)."""
    set_contract_catalog(None)


class ContractProvider:
    """Lazy provider for dataset contracts and related lookups."""

    @property
    def json_schema_by_dataset_name(self) -> dict[str, str]:
        """Return mapping from dataset name to JSON schema id.

        Returns
        -------
        dict[str, str]
            Mapping from dataset name to json_schema_id for datasets that define one.
        """
        return {
            contract.name: contract.json_schema_id
            for contract in iter_contracts()
            if contract.json_schema_id is not None
        }

    @staticmethod
    def get_contract_for_table_key(table_key: str) -> DatasetContract:
        """Return the contract for a specific table key.

        Parameters
        ----------
        table_key
            Fully qualified table or view key.

        Returns
        -------
        DatasetContract
            Contract describing the dataset or view.
        """
        return get_contract_for_table_key(table_key)


class _ContractProviderHolder(SingletonHolder["ContractProvider"]):
    """Thread-safe singleton holder for ContractProvider."""


def get_contract_provider() -> ContractProvider:
    """Return the singleton contract provider instance.

    Returns
    -------
    ContractProvider
        Singleton provider instance.
    """
    return _ContractProviderHolder.get(ContractProvider)


__all__ = [
    "ContractProvider",
    "clear_contract_cache",
    "get_contract_for_table_key",
    "get_contract_provider",
    "is_view",
    "iter_contracts",
    "iter_contracts_by_table_key",
    "load_contract_catalog_from_connection",
    "set_contract_catalog",
]
