"""Contract service for dataset and output contract resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

from codeintel.build.catalogs.canonical import load_contract_catalog
from codeintel.build.output_inventory import OutputInventory
from codeintel.build.schemas.provider_declared import declared_schema_provider_for_inventory
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.target_inventory import get_output_inventory
from codeintel.build.target_metadata import TargetMetadataProvider, get_target_metadata_provider
from codeintel.config.datasets.composites import get_composite_schemas
from codeintel.core.schemas.contract_factory import (
    DatasetContractOverrides,
    build_dataset_contract,
    is_docs_view,
)
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.service import SchemaService
from codeintel.storage.views.inventory import discover_derived_docs_views

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.contracts import OutputContract
    from codeintel.config.datasets.primitives import CompositeSchema

__all__ = [
    "ContractProvider",
    "ContractResolutionMode",
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
    "overrides_from_output_contract",
]


def is_view(table_key: str) -> bool:
    """Return True if the table key represents a docs view.

    Returns
    -------
    bool
        True when the table key maps to a docs view.
    """
    return is_docs_view(table_key)


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
def _declared_only_service() -> SchemaService:
    inventory = get_output_inventory()
    provider = declared_schema_provider_for_inventory(inventory)
    return SchemaService(table_provider=provider)


def _declared_only_service_for_inventory(inventory: OutputInventory) -> SchemaService:
    return SchemaService(table_provider=declared_schema_provider_for_inventory(inventory))


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


def overrides_from_output_contract(
    contract: OutputContract,
    *,
    table_key: str,
) -> DatasetContractOverrides:
    """
    Build dataset contract overrides from an output contract and table key.

    Extended Summary
    ----------------
    This helper pulls per-table metadata out of an OutputContract to seed
    DatasetContract derivation. It aligns JSON schema IDs and export filenames
    by table position, so downstream contract builders can override defaults
    without re-parsing target metadata. It is used by ContractService when
    enriching schema-derived contracts with target-specific metadata.

    Parameters
    ----------
    contract : OutputContract
        Output contract providing table metadata for a build target.
    table_key : str
        Fully qualified table key (schema.table) to resolve metadata for.

    Returns
    -------
    DatasetContractOverrides
        Override values sourced from the output contract. Per-table fields are
        None when the table key is not present in the contract.

    Notes
    -----
    - Time complexity is O(n) for the table key lookup in the contract table
      list; memory overhead is constant.
    - No I/O, caching, or global state is touched; the function is pure.

    Examples
    --------
    >>> from codeintel.build.contracts import OutputContract
    >>> from codeintel.core.schemas.primitives import Column, TableSchema
    >>> contract = OutputContract(
    ...     tables=(
    ...         TableSchema(
    ...             schema="core",
    ...             name="symbols",
    ...             columns=[Column("name", "VARCHAR")],
    ...         ),
    ...     ),
    ...     json_schema_ids=("core.symbols.v1",),
    ...     jsonl_filenames=("symbols.jsonl",),
    ...     parquet_filenames=("symbols.parquet",),
    ...     owner="core-team",
    ... )
    >>> overrides = overrides_from_output_contract(contract, table_key="core.symbols")
    >>> overrides.json_schema_id
    'core.symbols.v1'

    >>> missing = overrides_from_output_contract(contract, table_key="core.missing")
    >>> missing.json_schema_id is None
    True
    """
    json_schema_id = _extract_indexed_metadata(contract, table_key, contract.json_schema_ids)
    jsonl_filename = _extract_indexed_metadata(contract, table_key, contract.jsonl_filenames)
    parquet_filename = _extract_indexed_metadata(contract, table_key, contract.parquet_filenames)
    return DatasetContractOverrides(
        json_schema_id=json_schema_id,
        jsonl_filename=jsonl_filename,
        parquet_filename=parquet_filename,
        owner=contract.owner,
        description=contract.description,
        family=contract.family,
        freshness_sla=contract.freshness_sla,
        retention_policy=contract.retention_policy,
        upstream_dependencies=contract.upstream_dependencies,
        tags=contract.tags,
        validation_profile=contract.validation_profile,
    )


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


class ContractResolutionMode(Enum):
    """Contract resolution mode."""

    FULL = "full"
    DECLARED_ONLY = "declared_only"


@dataclass(frozen=True, slots=True)
class ContractResolutionSettings:
    """Settings controlling contract resolution behavior."""

    mode: ContractResolutionMode = ContractResolutionMode.FULL
    target_metadata_provider: TargetMetadataProvider | None = None
    output_inventory: OutputInventory | None = None


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
        is_view = is_docs_view(table_key)
        schema = self.schema_service.table_provider.get_table_schema(table_key)
        if schema is None and not is_view:
            msg = f"Unknown table key: {table_key}"
            raise KeyError(msg)
        composition = _get_composition_for_table_key(table_key)
        return build_dataset_contract(
            table_key=table_key,
            schema_service=self.schema_service,
            overrides=None,
            composition=composition,
            is_view_override=is_view,
        )

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
        is_view = is_docs_view(table_key)
        schema = self.schema_service.table_provider.get_table_schema(table_key)
        target = self.target_metadata.target_for_table_key(table_key)
        if schema is None and target is None and not is_view:
            msg = f"Unknown table key: {table_key}"
            raise KeyError(msg)
        overrides = (
            overrides_from_output_contract(target.contract, table_key=table_key) if target else None
        )
        composition = _get_composition_for_table_key(table_key)
        return build_dataset_contract(
            table_key=table_key,
            schema_service=self.schema_service,
            overrides=overrides,
            composition=composition,
            is_view_override=is_view,
        )

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
class CatalogContractProvider:
    """Contract provider backed by the canonical catalog."""

    contracts: Mapping[str, DatasetContract]

    def get_contract_for_table_key(self, table_key: str) -> DatasetContract:
        contract = self.contracts.get(table_key)
        if contract is None:
            msg = f"Unknown table key: {table_key}"
            raise KeyError(msg)
        return contract

    def iter_contracts(self) -> Iterable[DatasetContract]:
        return self.contracts.values()

    def iter_contracts_by_table_key(self) -> Iterable[tuple[str, DatasetContract]]:
        return self.contracts.items()


@lru_cache(maxsize=1)
def get_contract_service() -> SchemaContractService:
    """Return the declared-only contract service instance.

    Returns
    -------
    SchemaContractService
        Declared-only contract service.
    """
    return SchemaContractService(schema_service=_declared_only_service())


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

    Raises
    ------
    ValueError
        Raised when the resolution mode is unsupported.
    """
    if settings is None:
        return CatalogContractProvider(contracts=_get_catalog_contracts())
    mode = settings.mode
    if mode is ContractResolutionMode.FULL:
        if settings.target_metadata_provider is not None:
            return ContractService(
                schema_service=get_schema_service(),
                target_metadata=settings.target_metadata_provider,
            )
        return CatalogContractProvider(contracts=_get_catalog_contracts())
    if mode is ContractResolutionMode.DECLARED_ONLY:
        if settings.output_inventory is not None:
            return SchemaContractService(
                schema_service=_declared_only_service_for_inventory(settings.output_inventory)
            )
        return get_contract_service()
    msg = f"Unsupported contract resolution mode: {mode}"
    raise ValueError(msg)


@lru_cache(maxsize=256)
def _get_schema_contract_for_table_key(table_key: str) -> DatasetContract:
    return get_contract_service().get_contract_for_table_key(table_key)


@lru_cache(maxsize=256)
def _get_enriched_contract_for_table_key(table_key: str) -> DatasetContract:
    return get_enriched_contract_service().get_contract_for_table_key(table_key)


@lru_cache(maxsize=1)
def _get_catalog_contracts() -> Mapping[str, DatasetContract]:
    return load_contract_catalog()


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

    Raises
    ------
    ValueError
        Raised when the resolution mode is unsupported.
    """
    if settings is None:
        provider = CatalogContractProvider(contracts=_get_catalog_contracts())
        return provider.get_contract_for_table_key(table_key)
    mode = settings.mode
    if mode is ContractResolutionMode.FULL:
        if settings.target_metadata_provider is not None:
            service = ContractService(
                schema_service=get_schema_service(),
                target_metadata=settings.target_metadata_provider,
            )
            return service.get_contract_for_table_key(table_key)
        provider = CatalogContractProvider(contracts=_get_catalog_contracts())
        return provider.get_contract_for_table_key(table_key)
    if mode is ContractResolutionMode.DECLARED_ONLY:
        if settings.output_inventory is not None:
            service = SchemaContractService(
                schema_service=_declared_only_service_for_inventory(settings.output_inventory)
            )
            return service.get_contract_for_table_key(table_key)
        return _get_schema_contract_for_table_key(table_key)
    msg = f"Unsupported contract resolution mode: {mode}"
    raise ValueError(msg)


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
    _get_catalog_contracts.cache_clear()
    _get_enriched_contract_for_table_key.cache_clear()
    _get_schema_contract_for_table_key.cache_clear()
    get_enriched_contract_service.cache_clear()
    get_contract_service.cache_clear()
    _declared_only_service.cache_clear()


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
