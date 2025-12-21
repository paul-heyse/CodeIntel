"""DAG-free contract registry for CLI and lightweight tooling."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.contracts import is_placeholder_schema
from codeintel.build.schemas.contract_service import overrides_from_output_contract
from codeintel.build.target_catalog import load_target_specs
from codeintel.build.target_inventory import get_output_inventory
from codeintel.config.datasets.composites import get_composite_schemas
from codeintel.core.schemas.contract_factory import build_dataset_contract
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.declared import source_declared_schema_provider
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.core.schemas.service import SchemaService
from codeintel.storage.views.inventory import discover_derived_docs_views

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.targets import OutputTarget
    from codeintel.core.schemas.primitives import TableSchema


def _targets_by_table_key(targets: Iterable[OutputTarget]) -> dict[str, OutputTarget]:
    by_table_key: dict[str, OutputTarget] = {}
    for target in targets:
        for table_key in target.contract.table_keys:
            if table_key in by_table_key:
                msg = f"Duplicate table key in target specs: {table_key}"
                raise ValueError(msg)
            by_table_key[table_key] = target
    return by_table_key


def _dag_free_schema_service() -> SchemaService:
    output_inventory = get_output_inventory()
    source_provider = source_declared_schema_provider(
        exclude_table_keys=output_inventory.all_dataset_keys
    )
    schemas: dict[str, TableSchema] = {
        schema.table_key: schema for schema in source_provider.iter_table_schemas()
    }
    for target in load_target_specs():
        for table_schema in target.contract.tables:
            if is_placeholder_schema(table_schema):
                continue
            schemas[table_schema.table_key] = table_schema
    return SchemaService(table_provider=MappingSchemaProvider(schemas))


@lru_cache(maxsize=1)
def get_dag_free_contracts_by_table_key() -> Mapping[str, DatasetContract]:
    """Return DAG-free dataset contracts keyed by table_key.

    Returns
    -------
    Mapping[str, DatasetContract]
        Mapping of table_key to DatasetContract for CLI-friendly enumeration.
    """
    targets = load_target_specs()
    targets_by_table_key = _targets_by_table_key(targets)
    schema_service = _dag_free_schema_service()
    derived_views = set(discover_derived_docs_views())
    compositions = get_composite_schemas()

    table_keys = {schema.table_key for schema in schema_service.table_provider.iter_table_schemas()}
    table_keys.update(targets_by_table_key)
    table_keys.update(derived_views)

    contracts: dict[str, DatasetContract] = {}
    for table_key in sorted(table_keys):
        target = targets_by_table_key.get(table_key)
        overrides = (
            overrides_from_output_contract(target.contract, table_key=table_key)
            if target is not None
            else None
        )
        composition = compositions.get(table_key)
        is_view_override = table_key in derived_views
        contracts[table_key] = build_dataset_contract(
            table_key=table_key,
            schema_service=schema_service,
            overrides=overrides,
            composition=composition,
            is_view_override=is_view_override if is_view_override else None,
        )
    return contracts


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
