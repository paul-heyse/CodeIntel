"""Build-owned contract catalog compilation."""

from __future__ import annotations

from codeintel.build.schemas.contract_service import (
    ContractService,
    get_enriched_contract_service,
)
from codeintel.core.schemas.contract_factory import is_docs_view
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.contract_serde import (
    contract_payload_from_contract,
    contract_payload_to_json_obj,
)
from codeintel.core.schemas.resolution import SchemaDerivationProvider
from codeintel.core.views.inventory import discover_derived_docs_views

_DECLARED_SOURCE_KIND = "declared_source"


def _resolve_contract_service() -> ContractService:
    return get_enriched_contract_service()


def _view_table_keys(service: ContractService) -> set[str]:
    if service.tag_query is None:
        return set()
    return set(discover_derived_docs_views(tag_query=service.tag_query))


def _declared_external_table_keys(
    service: ContractService,
    *,
    output_table_keys: set[str],
) -> set[str]:
    provider = service.schema_service.table_provider
    declared: set[str] = set()
    if isinstance(provider, SchemaDerivationProvider):
        for schema in provider.iter_table_schemas():
            table_key = schema.table_key
            derivation = provider.derivation(table_key)
            if derivation is None or derivation.source_kind != _DECLARED_SOURCE_KIND:
                continue
            declared.add(table_key)
        return declared
    for schema in provider.iter_table_schemas():
        declared.add(schema.table_key)
    declared.difference_update(output_table_keys)
    return declared


def _contract_table_keys(
    service: ContractService,
    *,
    include_views: bool,
) -> tuple[str, ...]:
    output_table_keys = set(service.target_metadata.all_table_keys())
    declared_table_keys = _declared_external_table_keys(
        service,
        output_table_keys=output_table_keys,
    )
    if not include_views:
        output_table_keys = {key for key in output_table_keys if not is_docs_view(key)}
        declared_table_keys = {key for key in declared_table_keys if not is_docs_view(key)}
    contract_keys = output_table_keys | declared_table_keys
    if include_views:
        contract_keys |= _view_table_keys(service)
    return tuple(sorted(contract_keys))


def build_contract_catalog_payload(*, include_views: bool = True) -> dict[str, object]:
    """Build the canonical dataset contract catalog payload.

    Parameters
    ----------
    include_views
        Whether to include derived docs views in the catalog.

    Returns
    -------
    dict[str, object]
        JSON-serializable contract catalog payload.
    """
    service = _resolve_contract_service()
    contracts: dict[str, DatasetContract] = {}
    for table_key in _contract_table_keys(service, include_views=include_views):
        try:
            contract = service.get_dataset_contract(table_key)
        except KeyError:
            continue
        contracts[table_key] = contract
    return {
        "version": 1,
        "contracts": {
            table_key: contract_payload_to_json_obj(contract_payload_from_contract(contract))
            for table_key, contract in contracts.items()
        },
    }


__all__ = [
    "build_contract_catalog_payload",
]
