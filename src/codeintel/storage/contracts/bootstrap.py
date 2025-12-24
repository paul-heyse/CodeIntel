"""Bootstrap canonical contract catalogs for storage gateways."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.contract_serde import contract_to_json_obj
from codeintel.core.schemas.contract_service import get_enriched_contract_service
from codeintel.storage.contracts.provider import load_contract_catalog_from_connection
from codeintel.storage.metadata import build_catalog_entry, upsert_canonical_catalog

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.gateway.config import StorageConfig

_CONTRACT_CATALOG_KIND = "dataset_contracts"


def _contracts_payload(contracts: Mapping[str, DatasetContract]) -> dict[str, object]:
    return {
        "version": 1,
        "contracts": {key: contract_to_json_obj(contract) for key, contract in contracts.items()},
    }


@dataclass(frozen=True, slots=True)
class _CatalogGateway:
    con: DuckDBPyConnection


def bootstrap_contract_catalog(con: DuckDBPyConnection, *, config: StorageConfig) -> None:
    """Persist the canonical contract catalog into metadata and load it in memory."""
    service = get_enriched_contract_service()
    contracts = {contract.table_key: contract for contract in service.iter_contracts()}
    payload = _contracts_payload(contracts)
    catalog_hash = fingerprint(payload)
    entry = build_catalog_entry(
        catalog_kind=_CONTRACT_CATALOG_KIND,
        catalog_hash=catalog_hash,
        payload=payload,
        inputs={"source": "storage_bootstrap", "read_only": config.read_only},
    )
    upsert_canonical_catalog(_CatalogGateway(con=con), entry)
    load_contract_catalog_from_connection(con)


__all__ = ["bootstrap_contract_catalog"]
