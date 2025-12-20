"""Contract service for dataset and output contract resolution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Protocol

from codeintel.build.schemas.service import get_schema_service
from codeintel.build.table_keys import split_table_key
from codeintel.build.target_metadata import TargetMetadataProvider, get_target_metadata_provider
from codeintel.config.datasets.composites import get_composite_schemas
from codeintel.core.schemas.contract_policy import (
    default_json_schema_id,
    default_jsonl_filename,
    default_parquet_filename,
)
from codeintel.core.schemas.contract_primitives import DatasetContract, RowBinding
from codeintel.core.schemas.declared import (
    declared_schema_provider as core_declared_schema_provider,
)
from codeintel.core.schemas.service import SchemaService
from codeintel.storage.views.inventory import discover_derived_docs_views

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.contracts import OutputContract
    from codeintel.build.targets import OutputTarget
    from codeintel.config.datasets.primitives import CompositeSchema
    from codeintel.core.schemas.primitives import TableSchema

__all__ = [
    "ContractProvider",
    "ContractResolutionSettings",
    "ContractService",
    "SchemaContractService",
    "clear_contract_cache",
    "column_order_for_table_key",
    "get_contract_for_table_key",
    "get_contract_provider",
    "get_contract_service",
    "get_enriched_contract_service",
    "is_view",
    "iter_contracts",
    "iter_contracts_by_table_key",
]


def _get_composition_for_table_key(table_key: str) -> CompositeSchema | None:
    """Return composite schema metadata for a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    CompositeSchema | None
        Composition metadata when available.
    """
    return get_composite_schemas().get(table_key)


@lru_cache(maxsize=1)
def _schema_only_service() -> SchemaService:
    return SchemaService(table_provider=core_declared_schema_provider())


def _get_row_binding_safe(service: SchemaService, table_key: str) -> RowBinding | None:
    """Try to get a row binding, returning None on failure.

    Parameters
    ----------
    service
        SchemaService instance.
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    RowBinding | None
        The row binding if available, None otherwise.
    """
    try:
        generated = service.get_row_binding(table_key)
        if generated is None:
            return None
        return RowBinding(
            row_type=generated.row_model,
            to_tuple=generated.serializer,
        )
    except KeyError:
        return None


def is_view(table_key: str) -> bool:
    """Return True if the table key represents a docs view.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    bool
        True when this is a docs view.
    """
    return table_key.startswith("docs.v_")


def _owner_package_from_prefix(
    schema_prefix: str,
) -> Literal["core", "analytics", "graphs", "qa", "docs"] | None:
    mapping: dict[str, Literal["core", "analytics", "graphs", "qa", "docs"]] = {
        "core": "core",
        "analytics": "analytics",
        "graph": "graphs",
        "docs": "docs",
        "qa": "qa",
    }
    return mapping.get(schema_prefix)


def _extract_indexed_metadata(
    contract: OutputContract,
    table_key: str,
    metadata_tuple: tuple[str, ...],
) -> str | None:
    """Extract metadata value for a specific table from indexed tuples.

    Parameters
    ----------
    contract
        OutputContract containing metadata.
    table_key
        Table key to resolve.
    metadata_tuple
        Tuple of metadata entries aligned with contract table order.

    Returns
    -------
    str | None
        Metadata value if present.
    """
    if not metadata_tuple:
        return None
    table_keys = contract.table_keys
    try:
        idx = table_keys.index(table_key)
        if idx < len(metadata_tuple):
            return metadata_tuple[idx]
    except (ValueError, IndexError):
        return None
    return None


class ContractProvider(Protocol):
    """Protocol for dataset contract providers."""

    def get_contract_for_table_key(self, table_key: str) -> DatasetContract:
        """Return a dataset contract for a table key."""
        ...

    def iter_contracts(self) -> Iterable[DatasetContract]:
        """Iterate dataset contracts."""
        ...

    def iter_contracts_by_table_key(self) -> Iterable[tuple[str, DatasetContract]]:
        """Iterate dataset contracts as (table_key, contract) pairs."""
        ...


@dataclass(frozen=True, slots=True)
class ContractResolutionSettings:
    """Settings controlling contract resolution behavior."""

    include_target_metadata: bool = False


@dataclass(frozen=True, slots=True)
class SchemaContractService:
    """Resolve dataset contracts without target metadata."""

    schema_service: SchemaService

    def get_dataset_contract(self, table_key: str) -> DatasetContract:
        """Return the DatasetContract for a table key.

        Returns
        -------
        DatasetContract
            Dataset contract for the table key.

        Raises
        ------
        KeyError
            Raised when the table key is unknown.
        """
        if is_view(table_key):
            return _derive_view_contract(service=self.schema_service, view_key=table_key)

        schema = self.schema_service.table_provider.get_table_schema(table_key)
        if schema is not None:
            return _derive_contract_from_schema(
                service=self.schema_service,
                table_key=table_key,
                schema=schema,
            )

        msg = f"Unknown table key: {table_key}"
        raise KeyError(msg)

    def iter_dataset_contracts(self) -> Iterable[DatasetContract]:
        """Iterate all known dataset contracts.

        Yields
        ------
        DatasetContract
            Dataset contract entries known to the schema provider.
        """
        seen: set[str] = set()
        for schema in self.schema_service.table_provider.iter_table_schemas():
            table_key = schema.table_key
            if table_key in seen:
                continue
            seen.add(table_key)
            try:
                yield self.get_dataset_contract(table_key)
            except KeyError:
                continue

        for view_key in discover_derived_docs_views():
            if view_key in seen:
                continue
            seen.add(view_key)
            try:
                yield self.get_dataset_contract(view_key)
            except KeyError:
                continue

    def iter_dataset_contracts_by_table_key(self) -> Iterable[tuple[str, DatasetContract]]:
        """Iterate dataset contracts as (table_key, contract) pairs.

        Yields
        ------
        tuple[str, DatasetContract]
            Table key and contract pairs.
        """
        for contract in self.iter_dataset_contracts():
            yield contract.table_key, contract

    def get_contract_for_table_key(self, table_key: str) -> DatasetContract:
        """Return dataset contract for a table key.

        Returns
        -------
        DatasetContract
            Dataset contract for the table key.
        """
        return self.get_dataset_contract(table_key)

    def iter_contracts(self) -> Iterable[DatasetContract]:
        """Iterate dataset contracts.

        Returns
        -------
        Iterable[DatasetContract]
            Iterable of dataset contracts.
        """
        return self.iter_dataset_contracts()

    def iter_contracts_by_table_key(self) -> Iterable[tuple[str, DatasetContract]]:
        """Iterate dataset contracts as (table_key, contract) pairs.

        Returns
        -------
        Iterable[tuple[str, DatasetContract]]
            Iterable of table key and contract pairs.
        """
        return self.iter_dataset_contracts_by_table_key()


@dataclass(frozen=True, slots=True)
class ContractService:
    """Resolve dataset and output contracts with target metadata."""

    schema_service: SchemaService
    target_metadata: TargetMetadataProvider

    def get_output_contract(self, target_name: str) -> OutputContract | None:
        """Return the OutputContract for a target.

        Returns
        -------
        OutputContract | None
            Output contract if available, otherwise None.
        """
        target = self.target_metadata.get_target(target_name)
        return target.contract if target is not None else None

    def get_dataset_contract(self, table_key: str) -> DatasetContract:
        """Return the DatasetContract for a table key.

        Returns
        -------
        DatasetContract
            Dataset contract for the table key.

        Raises
        ------
        KeyError
            Raised when the table key is unknown.
        """
        if is_view(table_key):
            return _derive_view_contract(service=self.schema_service, view_key=table_key)

        schema = self.schema_service.table_provider.get_table_schema(table_key)
        target = self.target_metadata.target_for_table_key(table_key)
        if target is not None:
            return _derive_contract_from_target(
                service=self.schema_service,
                table_key=table_key,
                target=target,
                schema=schema,
            )

        if schema is not None:
            return _derive_contract_from_schema(
                service=self.schema_service,
                table_key=table_key,
                schema=schema,
            )

        msg = f"Unknown table key: {table_key}"
        raise KeyError(msg)

    def iter_dataset_contracts(self) -> Iterable[DatasetContract]:
        """Iterate all known dataset contracts.

        Yields
        ------
        DatasetContract
            Dataset contract entries known to the schema provider.
        """
        seen: set[str] = set()
        for schema in self.schema_service.table_provider.iter_table_schemas():
            table_key = schema.table_key
            if table_key in seen:
                continue
            seen.add(table_key)
            try:
                yield self.get_dataset_contract(table_key)
            except KeyError:
                continue

        for view_key in discover_derived_docs_views():
            if view_key in seen:
                continue
            seen.add(view_key)
            try:
                yield self.get_dataset_contract(view_key)
            except KeyError:
                continue

    def iter_dataset_contracts_by_table_key(self) -> Iterable[tuple[str, DatasetContract]]:
        """Iterate dataset contracts as (table_key, contract) pairs.

        Yields
        ------
        tuple[str, DatasetContract]
            Table key and contract pairs.
        """
        for contract in self.iter_dataset_contracts():
            yield contract.table_key, contract

    def get_contract_for_table_key(self, table_key: str) -> DatasetContract:
        """Return dataset contract for a table key.

        Returns
        -------
        DatasetContract
            Dataset contract for the table key.
        """
        return self.get_dataset_contract(table_key)

    def iter_contracts(self) -> Iterable[DatasetContract]:
        """Iterate dataset contracts.

        Returns
        -------
        Iterable[DatasetContract]
            Iterable of dataset contracts.
        """
        return self.iter_dataset_contracts()

    def iter_contracts_by_table_key(self) -> Iterable[tuple[str, DatasetContract]]:
        """Iterate dataset contracts as (table_key, contract) pairs.

        Returns
        -------
        Iterable[tuple[str, DatasetContract]]
            Iterable of table key and contract pairs.
        """
        return self.iter_dataset_contracts_by_table_key()


def _derive_contract_from_target(
    *,
    service: SchemaService,
    table_key: str,
    target: OutputTarget,
    schema: TableSchema | None,
) -> DatasetContract:
    schema_prefix, table_name = split_table_key(table_key)
    contract = target.contract
    row_binding = _get_row_binding_safe(service, table_key)

    json_schema_id = _extract_indexed_metadata(contract, table_key, contract.json_schema_ids)
    if json_schema_id is None:
        json_schema_id = default_json_schema_id(table_key=table_key, schema=schema)

    jsonl_filename = _extract_indexed_metadata(contract, table_key, contract.jsonl_filenames)
    if jsonl_filename is None:
        jsonl_filename = default_jsonl_filename(table_key=table_key, schema=schema)

    parquet_filename = _extract_indexed_metadata(contract, table_key, contract.parquet_filenames)
    if parquet_filename is None:
        parquet_filename = default_parquet_filename(table_key=table_key, schema=schema)

    description = contract.description
    if description is None and schema is not None:
        description = schema.description

    composition = _get_composition_for_table_key(table_key)

    return DatasetContract(
        table_key=table_key,
        name=table_name,
        schema=schema,
        row_binding=row_binding,
        json_schema_id=json_schema_id,
        jsonl_filename=jsonl_filename,
        parquet_filename=parquet_filename,
        is_view=False,
        owner_package=_owner_package_from_prefix(schema_prefix),
        tags=contract.tags | frozenset({"base_table"}),
        description=description,
        family=contract.family or schema_prefix,
        owner=contract.owner,
        freshness_sla=contract.freshness_sla,
        retention_policy=contract.retention_policy,
        upstream_dependencies=contract.upstream_dependencies,
        validation_profile=contract.validation_profile,
        composition=composition,
    )


def _derive_contract_from_schema(
    *,
    service: SchemaService,
    table_key: str,
    schema: TableSchema | None,
) -> DatasetContract:
    schema_prefix, table_name = split_table_key(table_key)
    row_binding = _get_row_binding_safe(service, table_key)
    description = schema.description if schema is not None else None
    composition = _get_composition_for_table_key(table_key)
    json_schema_id = default_json_schema_id(table_key=table_key, schema=schema)
    jsonl_filename = default_jsonl_filename(table_key=table_key, schema=schema)
    parquet_filename = default_parquet_filename(table_key=table_key, schema=schema)

    return DatasetContract(
        table_key=table_key,
        name=table_name,
        schema=schema,
        row_binding=row_binding,
        json_schema_id=json_schema_id,
        jsonl_filename=jsonl_filename,
        parquet_filename=parquet_filename,
        is_view=False,
        owner_package=_owner_package_from_prefix(schema_prefix),
        tags=frozenset({"base_table"}),
        description=description,
        family=schema_prefix,
        owner=None,
        freshness_sla=None,
        retention_policy=None,
        upstream_dependencies=(),
        validation_profile="strict",
        composition=composition,
    )


def _derive_view_contract(*, service: SchemaService, view_key: str) -> DatasetContract:
    schema_prefix, view_name = split_table_key(view_key)
    schema = service.table_provider.get_table_schema(view_key)
    row_binding = _get_row_binding_safe(service, view_key)
    description = schema.description if schema is not None else None
    composition = _get_composition_for_table_key(view_key)
    json_schema_id = default_json_schema_id(table_key=view_key, schema=schema)
    jsonl_filename = default_jsonl_filename(table_key=view_key, schema=schema)
    parquet_filename = default_parquet_filename(table_key=view_key, schema=schema)

    return DatasetContract(
        table_key=view_key,
        name=view_name,
        schema=schema,
        row_binding=row_binding,
        json_schema_id=json_schema_id,
        jsonl_filename=jsonl_filename,
        parquet_filename=parquet_filename,
        is_view=True,
        owner_package=_owner_package_from_prefix(schema_prefix),
        tags=frozenset({"docs_view", "read_only"}),
        description=description,
        family=schema_prefix,
        owner=None,
        freshness_sla=None,
        retention_policy=None,
        upstream_dependencies=(),
        validation_profile="strict",
        composition=composition,
    )


@lru_cache(maxsize=1)
def get_contract_service() -> SchemaContractService:
    """Return the schema-only contract service instance.

    Returns
    -------
    SchemaContractService
        Schema-only contract service.
    """
    return SchemaContractService(schema_service=_schema_only_service())


@lru_cache(maxsize=1)
def get_enriched_contract_service() -> ContractService:
    """Return the contract service that includes target metadata.

    Returns
    -------
    ContractService
        Contract service with target metadata.
    """
    return ContractService(
        schema_service=get_schema_service(),
        target_metadata=get_target_metadata_provider(),
    )


def get_contract_provider(
    settings: ContractResolutionSettings | None = None,
) -> ContractProvider:
    """Return a contract provider based on resolution settings.

    Returns
    -------
    ContractProvider
        Contract provider configured for the requested resolution mode.
    """
    if settings is not None and settings.include_target_metadata:
        return get_enriched_contract_service()
    return get_contract_service()


@lru_cache(maxsize=256)
def _get_schema_contract_for_table_key(table_key: str) -> DatasetContract:
    return get_contract_service().get_contract_for_table_key(table_key)


@lru_cache(maxsize=256)
def _get_enriched_contract_for_table_key(table_key: str) -> DatasetContract:
    return get_enriched_contract_service().get_contract_for_table_key(table_key)


def get_contract_for_table_key(
    table_key: str,
    *,
    settings: ContractResolutionSettings | None = None,
) -> DatasetContract:
    """Return a dataset contract for a table key.

    Returns
    -------
    DatasetContract
        Dataset contract for the table key.
    """
    if settings is not None and settings.include_target_metadata:
        return _get_enriched_contract_for_table_key(table_key)
    return _get_schema_contract_for_table_key(table_key)


def iter_contracts(
    *,
    settings: ContractResolutionSettings | None = None,
) -> Iterable[DatasetContract]:
    """Iterate dataset contracts based on resolution settings.

    Returns
    -------
    Iterable[DatasetContract]
        Iterable of dataset contracts.
    """
    return get_contract_provider(settings).iter_contracts()


def iter_contracts_by_table_key(
    *,
    settings: ContractResolutionSettings | None = None,
) -> Iterable[tuple[str, DatasetContract]]:
    """Iterate dataset contracts as (table_key, contract) pairs.

    Returns
    -------
    Iterable[tuple[str, DatasetContract]]
        Iterable of table key and contract pairs.
    """
    return get_contract_provider(settings).iter_contracts_by_table_key()


def clear_contract_cache() -> None:
    """Clear cached dataset contracts."""
    _get_enriched_contract_for_table_key.cache_clear()
    _get_schema_contract_for_table_key.cache_clear()
    get_enriched_contract_service.cache_clear()
    get_contract_service.cache_clear()
    _schema_only_service.cache_clear()


def column_order_for_table_key(table_key: str) -> tuple[str, ...]:
    """Return column order for a table key based on the dataset contract.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    tuple[str, ...]
        Ordered column names, or empty tuple when schema is unavailable.
    """
    schema = get_schema_service().get_table_schema(table_key)
    if schema is None:
        return ()
    return tuple(column.name for column in schema.columns)
