"""Contract service for dataset and output contract resolution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from codeintel.build.schemas.service import get_schema_service
from codeintel.build.table_keys import parse_table_key, split_table_key
from codeintel.build.target_metadata import get_target_metadata_service
from codeintel.config.datasets.composites import get_composite_schemas
from codeintel.core.schemas.contract_primitives import DatasetContract, RowBinding
from codeintel.core.singleton import SingletonHolder
from codeintel.storage.views.inventory import discover_derived_docs_views

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.contracts import OutputContract
    from codeintel.build.target_metadata import TargetMetadataService
    from codeintel.build.targets import OutputTarget
    from codeintel.config.datasets.primitives import CompositeSchema
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.service import SchemaService

__all__ = [
    "ContractService",
    "clear_contract_cache",
    "column_order_for_table_key",
    "get_contract_for_table_key",
    "get_contract_service",
    "is_view",
    "iter_contracts",
    "iter_contracts_by_table_key",
]


_NON_EXPORTABLE_CORE_TABLES: frozenset[str] = frozenset(
    {
        "file_state",
        "ingest_runs",
        "repo_map",
        "scip_occurrences",
        "scip_symbols",
        "test_results",
        "test_summary",
    }
)

_NON_EXPORTABLE_ANALYTICS_TABLES: frozenset[str] = frozenset({"tags_index"})


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


def _table_name_from_key(table_key: str) -> str:
    """Return the dataset/table name part of a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    str
        Table name portion of the key.
    """
    parsed = parse_table_key(table_key)
    return parsed.name


def _exportable_by_default(table_key: str) -> bool:
    """Return True if a dataset should be exported by default.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    bool
        True when the table is exportable by default.
    """
    if "." not in table_key:
        return False

    schema_prefix, table_name = table_key.split(".", maxsplit=1)

    if schema_prefix == "build":
        return False

    if schema_prefix == "core":
        return table_name not in _NON_EXPORTABLE_CORE_TABLES

    if schema_prefix == "graph":
        return not (table_name == "import_modules" or table_name.startswith("v_"))

    if schema_prefix == "analytics":
        is_internal_metrics_ext = table_name.endswith("_metrics_ext") and table_name.startswith(
            ("cfg_", "dfg_")
        )
        return (
            table_name not in _NON_EXPORTABLE_ANALYTICS_TABLES
            and not table_name.endswith("_cache")
            and not is_internal_metrics_ext
        )

    return True


def _default_export_filename(
    table_key: str,
    *,
    kind: Literal["jsonl", "parquet"],
) -> str:
    """Return the deterministic export filename for a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    kind
        Export kind ("jsonl" or "parquet").

    Returns
    -------
    str
        Default export filename for the dataset.
    """
    name = _table_name_from_key(table_key)
    return f"{name}.{kind}"


def _default_json_schema_id(*, table_key: str, schema: TableSchema | None) -> str | None:
    """Return deterministic JSON Schema ID for a dataset.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    schema
        TableSchema for the dataset when available.

    Returns
    -------
    str | None
        JSON Schema ID when exportable, else None.
    """
    if schema is None:
        return None
    if not _exportable_by_default(table_key):
        return None
    return _table_name_from_key(table_key)


def _default_jsonl_filename(*, table_key: str, schema: TableSchema | None) -> str | None:
    """Return deterministic JSONL filename for a dataset.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    schema
        TableSchema for the dataset when available.

    Returns
    -------
    str | None
        Default JSONL filename when exportable, else None.
    """
    if schema is None:
        return None
    if not _exportable_by_default(table_key):
        return None
    return _default_export_filename(table_key, kind="jsonl")


def _default_parquet_filename(*, table_key: str, schema: TableSchema | None) -> str | None:
    """Return deterministic Parquet filename for a dataset.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    schema
        TableSchema for the dataset when available.

    Returns
    -------
    str | None
        Default Parquet filename when exportable, else None.
    """
    if schema is None:
        return None
    if not _exportable_by_default(table_key):
        return None
    return _default_export_filename(table_key, kind="parquet")


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


@dataclass(frozen=True, slots=True)
class ContractService:
    """Resolve dataset and output contracts from build metadata."""

    schema_service: SchemaService
    target_metadata: TargetMetadataService

    def get_output_contract(self, target_name: str) -> OutputContract | None:
        """Return the OutputContract for a target.

        Parameters
        ----------
        target_name
            Target name to resolve.

        Returns
        -------
        OutputContract | None
            OutputContract when the target exists.
        """
        target = self.target_metadata.get_target(target_name)
        return target.contract if target is not None else None

    def get_dataset_contract(self, table_key: str) -> DatasetContract:
        """Return the DatasetContract for a table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        DatasetContract
            Derived dataset contract.

        Raises
        ------
        KeyError
            If the table key is unknown to all schema sources.
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
            Derived dataset contract for each known table or view.
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
            Table key and dataset contract pair.
        """
        for contract in self.iter_dataset_contracts():
            yield contract.table_key, contract


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
        json_schema_id = _default_json_schema_id(table_key=table_key, schema=schema)

    jsonl_filename = _extract_indexed_metadata(contract, table_key, contract.jsonl_filenames)
    if jsonl_filename is None:
        jsonl_filename = _default_jsonl_filename(table_key=table_key, schema=schema)

    parquet_filename = _extract_indexed_metadata(contract, table_key, contract.parquet_filenames)
    if parquet_filename is None:
        parquet_filename = _default_parquet_filename(table_key=table_key, schema=schema)

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
    json_schema_id = _default_json_schema_id(table_key=table_key, schema=schema)
    jsonl_filename = _default_jsonl_filename(table_key=table_key, schema=schema)
    parquet_filename = _default_parquet_filename(table_key=table_key, schema=schema)

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
    json_schema_id = _default_json_schema_id(table_key=view_key, schema=schema)
    jsonl_filename = _default_jsonl_filename(table_key=view_key, schema=schema)
    parquet_filename = _default_parquet_filename(table_key=view_key, schema=schema)

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


class _ContractServiceHolder(SingletonHolder["ContractService"]):
    """Singleton holder for ContractService."""


def get_contract_service() -> ContractService:
    """Return the singleton ContractService instance.

    Returns
    -------
    ContractService
        ContractService instance.
    """
    return _ContractServiceHolder.get(
        lambda: ContractService(
            schema_service=get_schema_service(),
            target_metadata=get_target_metadata_service(),
        )
    )


@lru_cache(maxsize=256)
def get_contract_for_table_key(table_key: str) -> DatasetContract:
    """Return a dataset contract for a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    DatasetContract
        Dataset contract for the table key.
    """
    return get_contract_service().get_dataset_contract(table_key)


def iter_contracts() -> Iterable[DatasetContract]:
    """Iterate all dataset contracts known to the contract service.

    Returns
    -------
    Iterable[DatasetContract]
        Dataset contracts.
    """
    return get_contract_service().iter_dataset_contracts()


def iter_contracts_by_table_key() -> Iterable[tuple[str, DatasetContract]]:
    """Iterate dataset contracts as (table_key, contract) pairs.

    Returns
    -------
    Iterable[tuple[str, DatasetContract]]
        Table key to contract pairs.
    """
    return get_contract_service().iter_dataset_contracts_by_table_key()


def clear_contract_cache() -> None:
    """Clear cached dataset contracts."""
    get_contract_for_table_key.cache_clear()
    _ContractServiceHolder.reset()


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
    contract = get_contract_for_table_key(table_key)
    schema = contract.schema
    if schema is None:
        return ()
    return tuple(column.name for column in schema.columns)
