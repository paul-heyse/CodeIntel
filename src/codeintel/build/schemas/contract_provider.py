"""Compatibility wrappers for dataset contract access.

This module preserves the legacy contract_provider API but delegates all logic
to the canonical ContractService. New code should import from
``codeintel.build.schemas.contract_service`` directly.
"""

from __future__ import annotations

from collections.abc import Iterable

from codeintel.build.schemas.contract_service import (
    clear_contract_cache,
    get_contract_for_table_key,
    get_contract_service,
    is_view,
    iter_contracts,
    iter_contracts_by_table_key,
)
from codeintel.core.schemas.contract_primitives import DatasetContract

__all__ = [
    "DatasetContract",
    "clear_contract_cache",
    "get_contract_for_table_key",
    "get_contract_service",
    "is_view",
    "iter_contracts",
    "iter_contracts_by_table_key",
]


def iter_contracts_legacy() -> Iterable[DatasetContract]:
    """Iterate dataset contracts (legacy alias).

    Returns
    -------
    Iterable[DatasetContract]
        Dataset contracts from the canonical ContractService.
    """
    return iter_contracts()
