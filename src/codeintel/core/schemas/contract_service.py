"""Canonical ContractService access for dataset contracts."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.contracts import OutputContract
    from codeintel.build.schemas.contract_service import (
        ContractResolutionSettings,
        DatasetContractOverrides,
    )
    from codeintel.core.schemas.contract_primitives import DatasetContract


class ContractService(Protocol):
    """Protocol for contract resolution services."""

    def get_contract_for_table_key(self, table_key: str) -> DatasetContract:
        """Return the dataset contract for a table key."""
        ...

    def iter_contracts(self) -> Iterable[DatasetContract]:
        """Iterate all dataset contracts."""
        ...

    def iter_contracts_by_table_key(self) -> Iterable[tuple[str, DatasetContract]]:
        """Iterate dataset contracts as (table_key, contract) pairs."""
        ...


def get_contract_service(
    *,
    settings: ContractResolutionSettings | None = None,
) -> ContractService:
    """Return the canonical ContractService implementation.

    Returns
    -------
    ContractService
        Contract service configured for the supplied resolution settings.
    """
    factory = cast(
        "Callable[..., ContractService]",
        _load_handler("get_contract_service"),
    )
    if settings is None:
        return factory()
    return factory(settings=settings)


def get_enriched_contract_service(
    *,
    settings: ContractResolutionSettings | None = None,
) -> ContractService:
    """Return the enriched ContractService implementation.

    Returns
    -------
    ContractService
        Contract service configured for enriched contract resolution.
    """
    factory = cast(
        "Callable[..., ContractService]",
        _load_handler("get_enriched_contract_service"),
    )
    if settings is None:
        return factory()
    return factory(settings=settings)


def is_view(table_key: str) -> bool:
    """Return True if the table key represents a docs view.

    Returns
    -------
    bool
        True when the table key resolves to a docs view.
    """
    handler = cast("Callable[[str], bool]", _load_handler("is_view"))
    return bool(handler(table_key))


def overrides_from_output_contract(
    contract: OutputContract,
    *,
    table_key: str,
) -> DatasetContractOverrides:
    """Return DatasetContract overrides sourced from an OutputContract.

    Returns
    -------
    DatasetContractOverrides
        Contract overrides derived from the output contract.
    """
    handler = cast(
        "Callable[..., DatasetContractOverrides]",
        _load_handler("overrides_from_output_contract"),
    )
    return handler(contract, table_key=table_key)


def get_contract_for_table_key(
    table_key: str,
    *,
    settings: ContractResolutionSettings | None = None,
) -> DatasetContract:
    """Return the dataset contract for a table key.

    Returns
    -------
    DatasetContract
        Contract resolved for the requested table key.
    """
    handler = cast(
        "Callable[..., DatasetContract]",
        _load_handler("get_contract_for_table_key"),
    )
    return handler(table_key, settings=settings)


def column_order_for_table_key(table_key: str) -> tuple[str, ...]:
    """Return the column ordering for a table key when available.

    Returns
    -------
    tuple[str, ...]
        Ordered column names from the contract schema.
    """
    handler = cast(
        "Callable[[str], tuple[str, ...]]",
        _load_handler("column_order_for_table_key"),
    )
    return handler(table_key)


def iter_contracts(
    *,
    settings: ContractResolutionSettings | None = None,
) -> Iterable[DatasetContract]:
    """Iterate all dataset contracts.

    Returns
    -------
    Iterable[DatasetContract]
        Dataset contracts resolved for the requested settings.
    """
    handler = cast(
        "Callable[..., Iterable[DatasetContract]]",
        _load_handler("iter_contracts"),
    )
    return handler(settings=settings)


def iter_contracts_by_table_key(
    *,
    settings: ContractResolutionSettings | None = None,
) -> Iterable[tuple[str, DatasetContract]]:
    """Iterate dataset contracts as (table_key, contract) pairs.

    Returns
    -------
    Iterable[tuple[str, DatasetContract]]
        Dataset contracts indexed by table key.
    """
    handler = cast(
        "Callable[..., Iterable[tuple[str, DatasetContract]]]",
        _load_handler("iter_contracts_by_table_key"),
    )
    return handler(settings=settings)


def _load_handler(name: str) -> Callable[..., object]:
    module = importlib.import_module("codeintel.build.schemas.contract_service")
    return cast("Callable[..., object]", getattr(module, name))


__all__ = [
    "ContractService",
    "column_order_for_table_key",
    "get_contract_for_table_key",
    "get_contract_service",
    "get_enriched_contract_service",
    "is_view",
    "iter_contracts",
    "iter_contracts_by_table_key",
    "overrides_from_output_contract",
]
