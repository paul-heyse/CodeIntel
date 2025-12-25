"""Validation helpers for the dataset contract."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.core.schemas.contract_validation import (
    ContractRegistry,
    TableColumnsLookup,
    build_contract_registry,
)
from codeintel.core.schemas.contract_validation import (
    collect_contract_issues as collect_contract_issues_core,
)
from codeintel.core.schemas.contract_validation import (
    validate_contract_or_raise as validate_contract_or_raise_core,
)
from codeintel.storage.contracts.provider import iter_contracts
from codeintel.storage.datasets.registry import (
    load_dataset_registry,
)
from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


@lru_cache(maxsize=1)
def _contract_registry() -> ContractRegistry:
    """Return cached contract registry.

    Returns
    -------
    ContractRegistry
        Cached contract registry instance.
    """
    return build_contract_registry(iter_contracts())


@lru_cache(maxsize=1)
def get_binding_required_datasets() -> frozenset[str]:
    """Return set of dataset names that require row bindings.

    This function is lazily evaluated and cached to avoid circular imports
    at module load time.

    Returns
    -------
    frozenset[str]
        Dataset names with JSON schema IDs (excluding data model tables).
    """
    return frozenset(
        contract.name
        for contract in _contract_registry().by_name.values()
        if contract.json_schema_id is not None
        and contract.name not in {"data_model_fields", "data_model_relationships"}
    )


__all__ = [
    "collect_contract_issues",
    "collect_contract_issues_lenient",
    "get_binding_required_datasets",
    "validate_contract_or_raise",
]


def _table_columns_lookup(con: DuckDBPyConnection, *, missing_ok: bool) -> TableColumnsLookup:
    def _lookup(table_key: str) -> list[str] | None:
        schema_name, table_name = split_table_key(table_key)
        info = con.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            ORDER BY ordinal_position
            """,
            [schema_name, table_name],
        ).fetchall()
        if not info and missing_ok:
            return None
        return [row[0] for row in info]

    return _lookup


def collect_contract_issues(
    con: DuckDBPyConnection,
    *,
    include_views: bool = True,
    missing_ok: bool = False,
) -> list[str]:
    """Collect contract inconsistencies for the active database.

    JSON schema validation is performed using generated schemas from TableSchema
    definitions, not file-based schemas (removed in PR-73).

    Returns
    -------
    list[str]
        Human-readable list of problems. Empty when the contract is healthy.
    """
    registry = load_dataset_registry(con)
    contracts = _contract_registry()
    return collect_contract_issues_core(
        registry,
        contracts_by_table_key=contracts.by_table_key,
        contracts_by_name=contracts.by_name,
        include_views=include_views,
        table_columns_lookup=_table_columns_lookup(con, missing_ok=missing_ok),
    )


def collect_contract_issues_lenient(
    con: DuckDBPyConnection,
    *,
    include_views: bool = True,
) -> list[str]:
    """Collect contract issues while ignoring missing tables.

    Returns
    -------
    list[str]
        Human-readable list of problems. Empty when the contract is healthy.
    """
    return collect_contract_issues(con, include_views=include_views, missing_ok=True)


def validate_contract_or_raise(
    con: DuckDBPyConnection,
    *,
    include_views: bool = True,
) -> None:
    """Validate dataset contract and raise on any issues."""
    registry = load_dataset_registry(con)
    contracts = _contract_registry()
    validate_contract_or_raise_core(
        registry,
        contracts_by_table_key=contracts.by_table_key,
        contracts_by_name=contracts.by_name,
        include_views=include_views,
        table_columns_lookup=_table_columns_lookup(con, missing_ok=False),
    )
