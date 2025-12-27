"""Build-owned contract catalog compilation and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.contract_serde import contract_to_json_obj
from codeintel.core.schemas.contract_service import get_enriched_contract_service
from codeintel.storage.contracts.provider import load_contract_catalog_from_connection
from codeintel.storage.metadata.catalogs import build_catalog_entry, upsert_canonical_catalog

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.storage.gateway.protocol import StorageGateway


_CONTRACT_CATALOG_KIND = "dataset_contracts"


@dataclass(frozen=True, slots=True)
class ContractCatalogResult:
    """Summary of a contract catalog persistence operation."""

    catalog_kind: str
    catalog_hash: str
    contract_count: int


def _build_contract_payload() -> dict[str, object]:
    service = get_enriched_contract_service()
    contracts = {contract.table_key: contract for contract in service.iter_contracts()}
    return {
        "version": 1,
        "contracts": {
            table_key: contract_to_json_obj(contract) for table_key, contract in contracts.items()
        },
    }


def persist_contract_catalog(
    gateway: StorageGateway,
    *,
    inputs: Mapping[str, object] | None = None,
) -> ContractCatalogResult:
    """Persist the canonical dataset contract catalog.

    Parameters
    ----------
    gateway
        Storage gateway for persistence.
    inputs
        Optional inputs metadata stored alongside the canonical catalog entry.

    Returns
    -------
    ContractCatalogResult
        Summary of the persisted catalog entry.
    """
    if gateway.config.read_only:
        msg = "Cannot persist contract catalog into a read-only storage gateway"
        raise RuntimeError(msg)

    payload = _build_contract_payload()
    catalog_hash = fingerprint(payload)
    entry = build_catalog_entry(
        catalog_kind=_CONTRACT_CATALOG_KIND,
        catalog_hash=catalog_hash,
        payload=payload,
        inputs=dict(inputs) if inputs is not None else None,
    )
    upsert_canonical_catalog(gateway, entry)
    load_contract_catalog_from_connection(gateway.con)

    contracts_raw = payload.get("contracts")
    contract_count = len(contracts_raw) if isinstance(contracts_raw, dict) else 0

    return ContractCatalogResult(
        catalog_kind=_CONTRACT_CATALOG_KIND,
        catalog_hash=catalog_hash,
        contract_count=contract_count,
    )


__all__ = [
    "ContractCatalogResult",
    "persist_contract_catalog",
]
