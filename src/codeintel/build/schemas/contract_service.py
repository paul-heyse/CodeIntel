"""Contract service for dataset contract resolution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, cast

from codeintel.config.datasets.composites import get_composite_schemas
from codeintel.core.imports.lazy import lazy_getattr
from codeintel.core.schemas.contract_factory import (
    DatasetContractOverrides,
    build_dataset_contract,
    is_docs_view,
)
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.declared import declared_schema_provider
from codeintel.core.schemas.provider import SchemaProvider
from codeintel.core.schemas.resolution import SchemaDerivationProvider
from codeintel.core.schemas.service import SchemaService
from codeintel.core.validation.profiles import ValidationProfile, normalize_validation_profile
from codeintel.core.views.inventory import discover_derived_docs_views

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.hamilton.dag_catalog import OutputDescriptor
    from codeintel.build.target_metadata import TargetMetadataProvider
    from codeintel.config.datasets.primitives import CompositeSchema
    from codeintel.core.hamilton.tag_query import TagQuery
    from codeintel.runtime.runtime_bundle import RuntimeBundle

__all__ = [
    "ContractProvider",
    "ContractResolutionMode",
    "ContractResolutionSettings",
    "ContractService",
    "clear_contract_cache",
    "column_order_for_table_key",
    "configure_contract_service",
    "get_contract_for_table_key",
    "get_contract_provider",
    "get_contract_service",
    "get_enriched_contract_service",
    "is_view",
    "iter_contracts",
    "iter_contracts_by_table_key",
    "overrides_from_output_descriptor",
]


def _schema_service() -> SchemaService:
    get_service = cast(
        "Callable[[], SchemaService]",
        lazy_getattr("codeintel.build.schemas.service", "get_schema_service"),
    )
    return get_service()


def _target_metadata_provider(*, runtime: RuntimeBundle) -> TargetMetadataProvider:
    get_provider = cast(
        "Callable[..., TargetMetadataProvider]",
        lazy_getattr("codeintel.build.target_metadata", "get_target_metadata_provider"),
    )
    return get_provider(runtime=runtime)


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


def _optional_tag(tags: Mapping[str, object], key: str) -> str | None:
    value = tags.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        msg = f"Invalid tag value for {key}: {value!r}"
        raise ValueError(msg)
    return value


def _non_inferable_schema_service(schema_service: SchemaService) -> SchemaService:
    provider = _non_inferable_provider(schema_service.table_provider)
    return SchemaService(table_provider=provider)


def _non_inferable_provider(_provider: SchemaProvider) -> SchemaProvider:
    return declared_schema_provider()


def _declared_external_table_keys(
    schema_provider: SchemaProvider,
    *,
    output_table_keys: set[str],
) -> set[str]:
    declared: set[str] = set()
    if isinstance(schema_provider, SchemaDerivationProvider):
        for schema in schema_provider.iter_table_schemas():
            table_key = schema.table_key
            derivation = schema_provider.derivation(table_key)
            if derivation is None or derivation.source_kind != "declared_source":
                continue
            declared.add(table_key)
        return declared
    for schema in schema_provider.iter_table_schemas():
        declared.add(schema.table_key)
    declared.difference_update(output_table_keys)
    return declared


def overrides_from_output_descriptor(output: OutputDescriptor) -> DatasetContractOverrides:
    """Build dataset contract overrides from saver output tags.

    Parameters
    ----------
    output
        Output descriptor carrying saver tags.

    Returns
    -------
    DatasetContractOverrides
        Parsed overrides derived from output tags.

    Raises
    ------
    ValueError
        If a tag has an invalid value or an unknown validation profile is provided.
    """
    tags = output.tags
    json_schema_id = _optional_tag(tags, "ci.json_schema_id")
    jsonl_filename = _optional_tag(tags, "ci.jsonl_filename")
    parquet_filename = _optional_tag(tags, "ci.parquet_filename")
    owner = _optional_tag(tags, "ci.dataset_owner")
    validation_profile_raw = _optional_tag(tags, "ci.validation_profile")
    try:
        validation_profile: ValidationProfile = normalize_validation_profile(
            validation_profile_raw,
            default="strict",
        )
    except ValueError as exc:
        msg = f"Invalid validation profile tag: {validation_profile_raw!r}"
        raise ValueError(msg) from exc
    return DatasetContractOverrides(
        json_schema_id=json_schema_id,
        jsonl_filename=jsonl_filename,
        parquet_filename=parquet_filename,
        owner=owner,
        validation_profile=validation_profile,
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


@dataclass(frozen=True, slots=True)
class ContractResolutionSettings:
    """Settings controlling contract resolution behavior."""

    mode: ContractResolutionMode = ContractResolutionMode.FULL
    target_metadata_provider: TargetMetadataProvider | None = None


@dataclass(frozen=True, slots=True)
class ContractService:
    """Resolve dataset contracts with target metadata."""

    schema_service: SchemaService
    target_metadata: TargetMetadataProvider
    tag_query: TagQuery | None = None

    def get_dataset_contract(
        self,
        table_key: str,
        *,
        schema_service: SchemaService | None = None,
    ) -> DatasetContract:
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
        active_schema_service = schema_service or self.schema_service
        schema = active_schema_service.table_provider.get_table_schema(table_key)
        output = self.target_metadata.output_for_table_key(table_key)
        if schema is None and output is None and not is_view:
            msg = f"Unknown table key: {table_key}"
            raise KeyError(msg)
        overrides = overrides_from_output_descriptor(output) if output else None
        composition = _get_composition_for_table_key(table_key)
        return build_dataset_contract(
            table_key=table_key,
            schema_service=active_schema_service,
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
        schema_provider = self.schema_service.table_provider
        output_table_keys = set(self.target_metadata.all_table_keys())
        declared_table_keys = _declared_external_table_keys(
            schema_provider,
            output_table_keys=output_table_keys,
        )
        view_keys = (
            set(discover_derived_docs_views(tag_query=self.tag_query))
            if self.tag_query is not None
            else set()
        )
        all_keys = output_table_keys | declared_table_keys | view_keys
        for table_key in sorted(all_keys):
            try:
                yield self.get_dataset_contract(table_key)
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


@dataclass(slots=True)
class _ContractServiceState:
    service: ContractService | None = None
    fingerprint: str | None = None


_CONTRACT_SERVICE_STATE = _ContractServiceState()


def configure_contract_service(*, runtime: RuntimeBundle) -> ContractService:
    """Configure the contract service for a runtime bundle.

    Parameters
    ----------
    runtime
        Runtime bundle providing catalog metadata and tag query.

    Returns
    -------
    ContractService
        Configured contract service.
    """
    state = _CONTRACT_SERVICE_STATE
    if runtime.fingerprint == state.fingerprint and state.service is not None:
        return state.service
    service = ContractService(
        schema_service=_schema_service(),
        target_metadata=_target_metadata_provider(runtime=runtime),
        tag_query=runtime.tag_query,
    )
    state.service = service
    state.fingerprint = runtime.fingerprint
    return service


def get_contract_service() -> ContractService:
    """Return the canonical contract service instance.

    Returns
    -------
    ContractService
        Contract service configured for full contract resolution.

    Raises
    ------
    RuntimeError
        If the ContractService has not been configured.
    """
    service = _CONTRACT_SERVICE_STATE.service
    if service is None:
        msg = "ContractService has not been configured"
        raise RuntimeError(msg)
    return service


def get_enriched_contract_service() -> ContractService:
    """Return the contract service that includes target metadata.

    Returns
    -------
    ContractService
        Contract service with target metadata.
    """
    return get_contract_service()


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
        return get_enriched_contract_service()
    mode = settings.mode
    if mode is ContractResolutionMode.FULL:
        if settings.target_metadata_provider is not None:
            return ContractService(
                schema_service=_schema_service(),
                target_metadata=settings.target_metadata_provider,
            )
        return get_enriched_contract_service()
    msg = f"Unsupported contract resolution mode: {mode}"
    raise ValueError(msg)


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

    Raises
    ------
    ValueError
        Raised when the resolution mode is unsupported.
    """
    if settings is None:
        return get_enriched_contract_service().get_contract_for_table_key(table_key)
    mode = settings.mode
    if mode is ContractResolutionMode.FULL:
        if settings.target_metadata_provider is not None:
            service = ContractService(
                schema_service=_schema_service(),
                target_metadata=settings.target_metadata_provider,
            )
            return service.get_contract_for_table_key(table_key)
        return get_enriched_contract_service().get_contract_for_table_key(table_key)
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
    _get_enriched_contract_for_table_key.cache_clear()
    _CONTRACT_SERVICE_STATE.service = None
    _CONTRACT_SERVICE_STATE.fingerprint = None


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
    schema = _schema_service().get_table_schema(table_key)
    if schema is None:
        return ()
    return tuple(column.name for column in schema.columns)
