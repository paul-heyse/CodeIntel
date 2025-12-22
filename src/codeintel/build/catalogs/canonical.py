"""Canonical catalog generation and caching for contracts and targets."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.build.catalogs.hashing import compute_global_catalog_hash
from codeintel.build.catalogs.target_serde import (
    output_target_from_json_obj,
    output_target_to_json_obj,
)
from codeintel.build.target_metadata import get_target_metadata_service
from codeintel.core.schemas.contract_serde import (
    contract_from_json_obj,
    contract_to_json_obj,
)
from codeintel.core.schemas.row_models import row_binding_for_table_schema
from codeintel.storage.metadata import (
    build_catalog_entry,
    load_canonical_catalog,
    upsert_canonical_catalog,
)

if TYPE_CHECKING:
    from codeintel.build.schemas.contract_service import ContractService
    from codeintel.build.schemas.service import SchemaService
    from codeintel.build.targets import OutputTarget
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.gateway import StorageGateway


CONTRACT_CATALOG_KIND = "dataset_contracts"
TARGET_CATALOG_KIND = "output_targets"


def _should_persist_catalog(gateway: StorageGateway) -> bool:
    return not gateway.config.read_only


def _get_schema_service() -> SchemaService:
    module = importlib.import_module("codeintel.build.schemas.service")
    service_factory = cast("Callable[[], SchemaService]", module.get_schema_service)
    return service_factory()


def _attach_row_binding(contract: DatasetContract) -> DatasetContract:
    if contract.schema is None:
        return contract
    schema_service = _get_schema_service()
    binding = schema_service.get_row_binding(contract.table_key)
    if binding is None:
        binding = row_binding_for_table_schema(table_schema=contract.schema)
    return replace(contract, row_binding=binding)


def _contracts_payload(contracts: Mapping[str, DatasetContract]) -> dict[str, object]:
    return {
        "version": 1,
        "contracts": {key: contract_to_json_obj(contract) for key, contract in contracts.items()},
    }


def _targets_payload(targets: Mapping[str, OutputTarget]) -> dict[str, object]:
    return {
        "version": 1,
        "targets": {name: output_target_to_json_obj(target) for name, target in targets.items()},
    }


def _get_contract_service() -> Callable[[], ContractService]:
    module = importlib.import_module("codeintel.build.schemas.contract_service")
    return cast(
        "Callable[[], ContractService]",
        module.get_enriched_contract_service,
    )


def _build_contract_catalog() -> dict[str, DatasetContract]:
    service_factory = _get_contract_service()
    service = service_factory()
    contracts = {contract.table_key: contract for contract in service.iter_contracts()}
    return {key: _attach_row_binding(contract) for key, contract in contracts.items()}


def _build_target_catalog() -> dict[str, OutputTarget]:
    service = get_target_metadata_service()
    targets = {target.name: target for target in service.system.graph.all_targets}
    return dict(sorted(targets.items(), key=lambda item: item[0]))


def load_contract_catalog(
    *,
    gateway: StorageGateway | None = None,
    root: Path | None = None,
) -> dict[str, DatasetContract]:
    """Load or build the canonical DatasetContract catalog.

    Returns
    -------
    dict[str, DatasetContract]
        Mapping of table keys to dataset contracts.
    """
    catalog_hash, inputs = compute_global_catalog_hash(root)
    entry = (
        load_canonical_catalog(
            gateway,
            catalog_kind=CONTRACT_CATALOG_KIND,
            catalog_hash=catalog_hash,
        )
        if gateway is not None
        else None
    )
    if entry is None:
        contracts = _build_contract_catalog()
        payload = _contracts_payload(contracts)
        if gateway is not None and _should_persist_catalog(gateway):
            upsert_canonical_catalog(
                gateway,
                build_catalog_entry(
                    catalog_kind=CONTRACT_CATALOG_KIND,
                    catalog_hash=catalog_hash,
                    payload=payload,
                    inputs=inputs.to_dict(),
                ),
            )
        return contracts

    payload_raw = entry.payload.get("contracts")
    if not isinstance(payload_raw, Mapping):
        return _build_contract_catalog()
    contracts: dict[str, DatasetContract] = {}
    for table_key, contract_obj in payload_raw.items():
        if not isinstance(table_key, str) or not isinstance(contract_obj, Mapping):
            continue
        contracts[table_key] = _attach_row_binding(contract_from_json_obj(contract_obj))
    return contracts


def load_target_catalog(
    *,
    gateway: StorageGateway | None = None,
    root: Path | None = None,
) -> dict[str, OutputTarget]:
    """Load or build the canonical OutputTarget catalog.

    Returns
    -------
    dict[str, OutputTarget]
        Mapping of target names to output targets.
    """
    catalog_hash, inputs = compute_global_catalog_hash(root)
    entry = (
        load_canonical_catalog(
            gateway,
            catalog_kind=TARGET_CATALOG_KIND,
            catalog_hash=catalog_hash,
        )
        if gateway is not None
        else None
    )
    if entry is None:
        targets = _build_target_catalog()
        payload = _targets_payload(targets)
        if gateway is not None and _should_persist_catalog(gateway):
            upsert_canonical_catalog(
                gateway,
                build_catalog_entry(
                    catalog_kind=TARGET_CATALOG_KIND,
                    catalog_hash=catalog_hash,
                    payload=payload,
                    inputs=inputs.to_dict(),
                ),
            )
        return targets

    payload_raw = entry.payload.get("targets")
    if not isinstance(payload_raw, Mapping):
        return _build_target_catalog()
    targets: dict[str, OutputTarget] = {}
    for name, target_obj in payload_raw.items():
        if not isinstance(name, str) or not isinstance(target_obj, Mapping):
            continue
        targets[name] = output_target_from_json_obj(target_obj)
    return targets


__all__ = [
    "CONTRACT_CATALOG_KIND",
    "TARGET_CATALOG_KIND",
    "load_contract_catalog",
    "load_target_catalog",
]
