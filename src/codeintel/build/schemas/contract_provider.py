"""Contract provider for deriving DatasetContract from build targets.

This module provides the canonical way to derive DatasetContract instances
from OutputTarget metadata, eliminating the need for manually maintained
contract dictionaries.

The contract provider:
- Derives contracts from OutputTarget metadata when a target produces the table
- Falls back to schema-only contracts for tables without targets
- Handles docs views via Hamilton tag discovery

Examples
--------
>>> from codeintel.build.schemas import is_view, get_contract_for_table_key
>>> is_view("docs.v_function_profile")
True
>>> contract = get_contract_for_table_key("analytics.function_metrics")
>>> contract.table_key
'analytics.function_metrics'
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal, cast

from codeintel.build.schemas.registry import get_schema_provider
from codeintel.core.providers.base import LazyProvider
from codeintel.core.schemas.contract_primitives import DatasetContract, RowBinding
from codeintel.core.singleton import SingletonHolder
from codeintel.storage.views.inventory import discover_derived_docs_views

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType

    from codeintel.build.contracts import OutputContract
    from codeintel.build.targets import OutputTarget
    from codeintel.config.datasets.primitives import CompositeSchema
    from codeintel.core.schemas.primitives import TableSchema


# -----------------------------------------------------------------------------
# Lazy Module Providers
#
# These providers break circular dependencies by deferring imports until first
# access. Using LazyProvider with importlib eliminates global statements
# (PLW0603) and function-level imports (PLC0415).
# -----------------------------------------------------------------------------

import importlib


def _load_module(name: str) -> ModuleType:
    """Load a module by name using importlib.

    Parameters
    ----------
    name
        Fully qualified module name.

    Returns
    -------
    ModuleType
        The loaded module.
    """
    return importlib.import_module(name)


# Module-level lazy providers (no global statements needed)
_registry_provider: LazyProvider[ModuleType] = LazyProvider(
    lambda: _load_module("codeintel.build.registry"),
    name="registry_module",
)
_row_registry_provider: LazyProvider[ModuleType] = LazyProvider(
    lambda: _load_module("codeintel.build.schemas.row_registry"),
    name="row_registry_module",
)
_composites_provider: LazyProvider[ModuleType] = LazyProvider(
    lambda: _load_module("codeintel.config.datasets.composites"),
    name="composites_module",
)


def _registry_module() -> ModuleType:
    """Get build registry module lazily.

    Returns
    -------
    ModuleType
        The codeintel.build.registry module.
    """
    return _registry_provider.get()


def _row_registry_module() -> ModuleType:
    """Get row registry module lazily.

    Returns
    -------
    ModuleType
        The codeintel.build.schemas.row_registry module.
    """
    return _row_registry_provider.get()


def _composites_module() -> ModuleType:
    """Get composites module lazily.

    Returns
    -------
    ModuleType
        The codeintel.config.datasets.composites module.
    """
    return _composites_provider.get()


def _get_composition_for_table_key(table_key: str) -> CompositeSchema | None:
    """Get the CompositeSchema for a table key if it exists.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    CompositeSchema | None
        The composition metadata if this is a profile table, None otherwise.
    """
    composites_mod = _composites_module()
    composite_schemas = composites_mod.get_composite_schemas()
    return composite_schemas.get(table_key)


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


def _table_name_from_key(table_key: str) -> str:
    """Return the dataset/table name part of a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    str
        Table name portion of the key. If no schema prefix is present, returns the full key.
    """
    return table_key.split(".", maxsplit=1)[1] if "." in table_key else table_key


def _exportable_by_default(table_key: str) -> bool:
    """Return True if a dataset should be exported by default.

    This replaces legacy hand-maintained filename maps with deterministic rules. In the long-term,
    export intent should be declared in the Hamilton graph (tags/metadata), but for now we
    keep the default export surface aligned with the current build schema taxonomy.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    bool
        True if the dataset should have default export filenames, False otherwise.
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
        Export kind.

    Returns
    -------
    str
        Default export filename.
    """
    return f"{_table_name_from_key(table_key)}.{kind}"


def _default_json_schema_id(*, table_key: str, schema: TableSchema | None) -> str | None:
    """Return deterministic JSON schema id for a dataset.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    schema
        TableSchema for the dataset when available.

    Returns
    -------
    str | None
        JSON schema id when the dataset has a schema and supports validation, else None.
    """
    if schema is None:
        return None
    if "." not in table_key:
        return None
    schema_prefix = table_key.split(".", maxsplit=1)[0]
    if schema_prefix == "build":
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
        Default JSONL filename when the dataset is exportable, else None.
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
        Default Parquet filename when the dataset is exportable, else None.
    """
    if schema is None:
        return None
    if not _exportable_by_default(table_key):
        return None
    return _default_export_filename(table_key, kind="parquet")


def _get_row_binding_safe(table_key: str) -> RowBinding | None:
    """Try to get a row binding, returning None on failure.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    RowBinding | None
        The row binding if available, None otherwise.
    """
    row_registry = _row_registry_module()
    try:
        generated_binding = row_registry.get_row_binding(table_key)
        return RowBinding(
            row_type=generated_binding.row_model,
            to_tuple=generated_binding.serializer,
        )
    except KeyError:
        return None


def _find_producing_target(table_key: str) -> OutputTarget | None:
    """Find the target that produces a given table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    OutputTarget | None
        The target that produces this table, or None if no target found.
    """
    registry = _registry_module()
    graph = registry.get_target_graph()
    for target in graph.all_targets:
        if table_key in target.contract.table_keys:
            return cast("OutputTarget", target)
    return None


def is_view(table_key: str) -> bool:
    """Return True if the table key represents a view.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    bool
        True if this is a view (docs.v_* or in DERIVED_DOCS_VIEWS).

    Examples
    --------
    >>> is_view("docs.v_function_profile")
    True
    >>> is_view("analytics.function_metrics")
    False
    """
    return table_key.startswith("docs.v_")


def _owner_package_from_prefix(
    schema_prefix: str,
) -> Literal["core", "analytics", "graphs", "qa", "docs"] | None:
    """Derive owner package from schema prefix.

    Parameters
    ----------
    schema_prefix
        The schema portion of a table key (e.g., "analytics").

    Returns
    -------
    Literal["core", "analytics", "graphs", "qa", "docs"] | None
        The owner package if derivable, None otherwise.
    """
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
        The OutputContract containing metadata.
    table_key
        The table key to find.
    metadata_tuple
        The metadata tuple to index into.

    Returns
    -------
    str | None
        The metadata value if found, None otherwise.
    """
    if not metadata_tuple:
        return None
    table_keys = contract.table_keys
    try:
        idx = table_keys.index(table_key)
        if idx < len(metadata_tuple):
            return metadata_tuple[idx]
    except (ValueError, IndexError):
        pass
    return None


def _derive_contract_from_target(
    table_key: str,
    target: OutputTarget,
    schema: TableSchema | None,
) -> DatasetContract:
    """Derive a DatasetContract from an OutputTarget.

    Parameters
    ----------
    table_key
        Fully qualified table key.
    target
        The OutputTarget that produces this table.
    schema
        The TableSchema for this table, if available.

    Returns
    -------
    DatasetContract
        Derived contract combining target metadata and schema.
    """
    schema_prefix, table_name = table_key.split(".", maxsplit=1)
    contract = target.contract

    row_binding = _get_row_binding_safe(table_key)

    # Prefer target-declared metadata, fall back to deterministic defaults.
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
    table_key: str,
    schema: TableSchema | None,
) -> DatasetContract:
    """Derive a DatasetContract from a TableSchema when no target produces it.

    Parameters
    ----------
    table_key
        Fully qualified table key.
    schema
        The TableSchema for this table, if available.

    Returns
    -------
    DatasetContract
        Minimal contract derived from schema only.
    """
    schema_prefix, table_name = table_key.split(".", maxsplit=1)
    row_binding = _get_row_binding_safe(table_key)
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


def _derive_view_contract(view_key: str) -> DatasetContract:
    """Derive a DatasetContract for a view.

    Parameters
    ----------
    view_key
        Fully qualified view key (e.g., "docs.v_function_profile").

    Returns
    -------
    DatasetContract
        Contract for the view with is_view=True.
    """
    schema_prefix, view_name = view_key.split(".", maxsplit=1)
    provider = get_schema_provider()
    schema = provider.get_table_schema(view_key)
    row_binding = _get_row_binding_safe(view_key)
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


@lru_cache(maxsize=256)
def get_contract_for_table_key(table_key: str) -> DatasetContract:
    """Derive a DatasetContract from the target that produces a table key.

    This function provides the canonical way to obtain dataset contracts.
    It derives contracts from OutputTarget metadata when available, falling
    back to schema-only contracts for tables without targets.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    DatasetContract
        Derived contract combining target metadata and schema.

    Raises
    ------
    KeyError
        If the table key is not found in schemas or views.

    Examples
    --------
    >>> contract = get_contract_for_table_key("analytics.function_metrics")
    >>> contract.table_key
    'analytics.function_metrics'
    """
    # Handle views first
    if is_view(table_key):
        return _derive_view_contract(table_key)

    # Get schema from provider
    provider = get_schema_provider()
    schema = provider.get_table_schema(table_key)

    # Find producing target
    target = _find_producing_target(table_key)

    if target is not None:
        return _derive_contract_from_target(table_key, target, schema)

    if schema is not None:
        return _derive_contract_from_schema(table_key, schema)

    msg = f"Unknown table key: {table_key}"
    raise KeyError(msg)


def iter_contracts() -> Iterable[DatasetContract]:
    """Iterate all known dataset contracts.

    Yields contracts for all tables known to the schema provider
    and all docs views discovered from view builders.

    Yields
    ------
    DatasetContract
        Each known dataset contract.

    Examples
    --------
    >>> contracts = list(iter_contracts())
    >>> len(contracts) > 0
    True
    """
    provider = get_schema_provider()
    seen: set[str] = set()

    # Yield contracts for all known tables
    for schema in provider.iter_table_schemas():
        table_key = schema.table_key
        if table_key not in seen:
            seen.add(table_key)
            try:
                yield get_contract_for_table_key(table_key)
            except KeyError:
                continue

    # Yield contracts for all discovered docs views
    for view_key in discover_derived_docs_views():
        if view_key not in seen:
            seen.add(view_key)
            try:
                yield get_contract_for_table_key(view_key)
            except KeyError:
                continue


def iter_contracts_by_table_key() -> Iterable[tuple[str, DatasetContract]]:
    """Iterate all known contracts as (table_key, contract) pairs.

    Yields
    ------
    tuple[str, DatasetContract]
        Pairs of (table_key, contract) for all known datasets.
    """
    for contract in iter_contracts():
        yield contract.table_key, contract


def clear_contract_cache() -> None:
    """Clear the contract cache (for testing)."""
    get_contract_for_table_key.cache_clear()


class ContractProvider:
    """Lazy provider for dataset contracts and related lookups.

    This class provides convenient access to contract collections without
    requiring module-level computation at import time.

    Attributes
    ----------
    json_schema_by_dataset_name
        Mapping from dataset name to JSON schema ID for datasets with schemas.
    """

    @property
    def json_schema_by_dataset_name(self) -> dict[str, str]:
        """Return mapping from dataset name to JSON schema ID.

        Returns
        -------
        dict[str, str]
            Dataset name to JSON schema ID mapping.
        """
        return {
            contract.name: contract.json_schema_id
            for contract in iter_contracts()
            if contract.json_schema_id is not None
        }

    @staticmethod
    def get_contract_for_table_key(table_key: str) -> DatasetContract:
        """Get contract for a specific table key.

        Parameters
        ----------
        table_key
            The table key to look up.

        Returns
        -------
        DatasetContract
            The contract for this table key.
        """
        return get_contract_for_table_key(table_key)


class _ContractProviderHolder(SingletonHolder["ContractProvider"]):
    """Thread-safe singleton holder for ContractProvider."""


def get_contract_provider() -> ContractProvider:
    """Get the singleton contract provider instance.

    Returns
    -------
    ContractProvider
        The contract provider with lazy lookups for contracts.

    Examples
    --------
    >>> provider = get_contract_provider()
    >>> "function_metrics" in provider.json_schema_by_dataset_name
    True
    """
    return _ContractProviderHolder.get(ContractProvider)


__all__ = [
    "ContractProvider",
    "clear_contract_cache",
    "get_contract_for_table_key",
    "get_contract_provider",
    "is_view",
    "iter_contracts",
    "iter_contracts_by_table_key",
]
