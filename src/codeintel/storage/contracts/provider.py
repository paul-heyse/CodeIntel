"""Dataset contract provider (storage-owned).

This module provides the canonical dataset contract interface used by the
storage layer. It is intentionally independent of `codeintel.build.*`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from codeintel.config.datasets.composites import get_composite_schemas
from codeintel.config.datasets.contracts import get_row_bindings
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.singleton import SingletonHolder
from codeintel.storage.contracts.schema_provider import get_schema_provider
from codeintel.storage.views.inventory import discover_derived_docs_views

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.config.datasets.primitives import CompositeSchema
    from codeintel.core.schemas.contract_primitives import RowBinding
    from codeintel.core.schemas.primitives import TableSchema


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
    return table_key.split(".", maxsplit=1)[1] if "." in table_key else table_key


def _exportable_by_default(table_key: str) -> bool:
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
    return f"{_table_name_from_key(table_key)}.{kind}"


def _default_json_schema_id(*, table_key: str, schema: TableSchema | None) -> str | None:
    if schema is None or "." not in table_key:
        return None
    schema_prefix = table_key.split(".", maxsplit=1)[0]
    if schema_prefix == "build":
        return None
    return _table_name_from_key(table_key)


def _default_jsonl_filename(*, table_key: str, schema: TableSchema | None) -> str | None:
    if schema is None or not _exportable_by_default(table_key):
        return None
    return _default_export_filename(table_key, kind="jsonl")


def _default_parquet_filename(*, table_key: str, schema: TableSchema | None) -> str | None:
    if schema is None or not _exportable_by_default(table_key):
        return None
    return _default_export_filename(table_key, kind="parquet")


def is_view(table_key: str) -> bool:
    """Return True when the table key represents a docs view.

    Parameters
    ----------
    table_key
        Fully qualified table or view key.

    Returns
    -------
    bool
        True when the key is treated as a view.
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


def _get_row_binding_safe(table_key: str) -> RowBinding | None:
    return get_row_bindings().get(table_key)


def _get_composition_for_table_key(table_key: str) -> CompositeSchema | None:
    return get_composite_schemas().get(table_key)


def _derive_contract_from_schema(table_key: str, schema: TableSchema | None) -> DatasetContract:
    schema_prefix, table_name = table_key.split(".", maxsplit=1)
    row_binding = _get_row_binding_safe(table_key)
    composition = _get_composition_for_table_key(table_key)
    return DatasetContract(
        table_key=table_key,
        name=table_name,
        schema=schema,
        row_binding=row_binding,
        json_schema_id=_default_json_schema_id(table_key=table_key, schema=schema),
        jsonl_filename=_default_jsonl_filename(table_key=table_key, schema=schema),
        parquet_filename=_default_parquet_filename(table_key=table_key, schema=schema),
        is_view=False,
        owner_package=_owner_package_from_prefix(schema_prefix),
        tags=frozenset({"base_table"}),
        description=schema.description if schema is not None else None,
        family=schema_prefix,
        validation_profile="strict",
        composition=composition,
    )


def _derive_view_contract(view_key: str) -> DatasetContract:
    schema_prefix, view_name = view_key.split(".", maxsplit=1)
    provider = get_schema_provider()
    schema = provider.get_table_schema(view_key)
    row_binding = _get_row_binding_safe(view_key)
    composition = _get_composition_for_table_key(view_key)
    return DatasetContract(
        table_key=view_key,
        name=view_name,
        schema=schema,
        row_binding=row_binding,
        json_schema_id=_default_json_schema_id(table_key=view_key, schema=schema),
        jsonl_filename=_default_jsonl_filename(table_key=view_key, schema=schema),
        parquet_filename=_default_parquet_filename(table_key=view_key, schema=schema),
        is_view=True,
        owner_package=_owner_package_from_prefix(schema_prefix),
        tags=frozenset({"docs_view", "read_only"}),
        description=schema.description if schema is not None else None,
        family=schema_prefix,
        validation_profile="strict",
        composition=composition,
    )


@lru_cache(maxsize=256)
def get_contract_for_table_key(table_key: str) -> DatasetContract:
    """Return the DatasetContract for a table or view.

    Parameters
    ----------
    table_key
        Fully qualified key (schema.table).

    Returns
    -------
    DatasetContract
        Contract describing the dataset or view.

    Raises
    ------
    KeyError
        Raised when the key is unknown to the schema provider and is not treated as a view.
    """
    if is_view(table_key):
        return _derive_view_contract(table_key)

    schema = get_schema_provider().get_table_schema(table_key)
    if schema is not None:
        return _derive_contract_from_schema(table_key, schema)

    msg = f"Unknown table key: {table_key}"
    raise KeyError(msg)


def iter_contracts() -> Iterable[DatasetContract]:
    """Iterate all known dataset contracts.

    Yields
    ------
    DatasetContract
        Each known dataset contract.
    """
    provider = get_schema_provider()
    seen: set[str] = set()

    for schema in provider.iter_table_schemas():
        table_key = schema.table_key
        if table_key in seen:
            continue
        seen.add(table_key)
        yield get_contract_for_table_key(table_key)

    for view_key in discover_derived_docs_views():
        if view_key in seen:
            continue
        seen.add(view_key)
        yield get_contract_for_table_key(view_key)


def iter_contracts_by_table_key() -> Iterable[tuple[str, DatasetContract]]:
    """Iterate all known contracts as (table_key, contract) pairs.

    Yields
    ------
    tuple[str, DatasetContract]
        Each (table_key, contract) pair.
    """
    for contract in iter_contracts():
        yield contract.table_key, contract


def clear_contract_cache() -> None:
    """Clear the contract cache (for testing)."""
    get_contract_for_table_key.cache_clear()


class ContractProvider:
    """Lazy provider for dataset contracts and related lookups."""

    @property
    def json_schema_by_dataset_name(self) -> dict[str, str]:
        """Return mapping from dataset name to JSON schema id.

        Returns
        -------
        dict[str, str]
            Mapping from dataset name to json_schema_id for datasets that define one.
        """
        return {
            contract.name: contract.json_schema_id
            for contract in iter_contracts()
            if contract.json_schema_id is not None
        }

    @staticmethod
    def get_contract_for_table_key(table_key: str) -> DatasetContract:
        """Return the contract for a specific table key.

        Parameters
        ----------
        table_key
            Fully qualified table or view key.

        Returns
        -------
        DatasetContract
            Contract describing the dataset or view.
        """
        return get_contract_for_table_key(table_key)


class _ContractProviderHolder(SingletonHolder["ContractProvider"]):
    """Thread-safe singleton holder for ContractProvider."""


def get_contract_provider() -> ContractProvider:
    """Return the singleton contract provider instance.

    Returns
    -------
    ContractProvider
        Singleton provider instance.
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
