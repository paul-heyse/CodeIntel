"""Build-owned contract catalog compilation and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.schemas.contract_service import (
    ContractService,
    get_enriched_contract_service,
)
from codeintel.core.duckdb_types import DuckDBConnection
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.contract_serde import contract_to_json_obj
from codeintel.storage.contracts.provider import load_contract_catalog_from_connection
from codeintel.storage.metadata.catalogs import build_catalog_entry, upsert_canonical_catalog

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.gateway import BuildGateway


_CONTRACT_CATALOG_KIND = "dataset_contracts"


@dataclass(frozen=True, slots=True)
class ContractCatalogResult:
    """Summary of a contract catalog persistence operation."""

    catalog_kind: str
    catalog_hash: str
    contract_count: int


@dataclass(frozen=True, slots=True)
class _CatalogConnectionGateway:
    con: DuckDBConnection


def _resolve_contract_service() -> ContractService:
    return get_enriched_contract_service()


def build_contract_catalog_payload() -> dict[str, object]:
    """Build the canonical dataset contract catalog payload.

    Returns
    -------
    dict[str, object]
        JSON-serializable contract catalog payload.
    """
    service = _resolve_contract_service()
    contracts = {contract.table_key: contract for contract in service.iter_contracts()}
    return {
        "version": 1,
        "contracts": {
            table_key: contract_to_json_obj(contract) for table_key, contract in contracts.items()
        },
    }


def persist_contract_catalog_to_connection(
    con: DuckDBConnection,
    *,
    inputs: Mapping[str, object] | None = None,
) -> ContractCatalogResult:
    """Persist the canonical dataset contract catalog using a DuckDB connection.

    Parameters
    ----------
    con
        DuckDB connection for catalog persistence.
    inputs
        Optional inputs metadata stored alongside the canonical catalog entry.

    Returns
    -------
    ContractCatalogResult
        Summary of the persisted catalog entry.
    """
    payload = build_contract_catalog_payload()
    catalog_hash = fingerprint(payload)
    entry = build_catalog_entry(
        catalog_kind=_CONTRACT_CATALOG_KIND,
        catalog_hash=catalog_hash,
        payload=payload,
        inputs=dict(inputs) if inputs is not None else None,
    )
    upsert_canonical_catalog(_CatalogConnectionGateway(con), entry)

    contracts_raw = payload.get("contracts")
    contract_count = len(contracts_raw) if isinstance(contracts_raw, dict) else 0

    return ContractCatalogResult(
        catalog_kind=_CONTRACT_CATALOG_KIND,
        catalog_hash=catalog_hash,
        contract_count=contract_count,
    )


def persist_contract_catalog(
    gateway: BuildGateway,
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

    Raises
    ------
    RuntimeError
        If the gateway is read-only.
    """
    if gateway.config.read_only:
        msg = "Cannot persist contract catalog into a read-only storage gateway"
        raise RuntimeError(msg)

    result = persist_contract_catalog_to_connection(
        gateway.con,
        inputs=inputs,
    )
    load_contract_catalog_from_connection(gateway.con)
    return result


__all__ = [
    "ContractCatalogResult",
    "build_contract_catalog_payload",
    "persist_contract_catalog",
    "persist_contract_catalog_to_connection",
]
