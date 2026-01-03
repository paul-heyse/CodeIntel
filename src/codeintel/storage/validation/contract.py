"""Validation helpers for the dataset contract."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

import pyarrow as pa

from codeintel.core.validation.schema_constraints import (
    schema_errors,
    schema_metadata_errors,
)
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.contracts.provider import iter_contracts
from codeintel.storage.datasets.registry import load_dataset_registry
from codeintel.storage.duckdb_types import DuckDBCatalogException, DuckDBError

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from codeintel.storage.duckdb_types import DuckDBRelation


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
        for contract in iter_contracts()
        if contract.json_schema_id is not None
        and contract.name not in {"data_model_fields", "data_model_relationships"}
    )


__all__ = [
    "clear_contract_validation_cache",
    "collect_contract_issues",
    "collect_contract_issues_lenient",
    "get_binding_required_datasets",
    "validate_contract_or_raise",
]


def clear_contract_validation_cache() -> None:
    """Clear cached contract validation lookups."""
    get_binding_required_datasets.cache_clear()


def _arrow_schema_for_table(
    con: DuckDBPyConnection,
    *,
    table_key: str,
) -> pa.Schema | None:
    try:
        relation = con.table(table_key)
    except (DuckDBCatalogException, DuckDBError):
        return None
    limited: object = relation
    limiter = getattr(relation, "limit", None)
    if callable(limiter):
        try:
            limited = limiter(0)
        except TypeError:
            limited = relation
    relation_to_read = (
        limited if callable(getattr(limited, "fetch_record_batch", None)) else relation
    )
    reader = cast("DuckDBRelation", relation_to_read).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    return reader.schema


def collect_contract_issues(
    con: DuckDBPyConnection,
    *,
    include_views: bool = True,
    missing_ok: bool = False,
) -> list[str]:
    """Collect contract inconsistencies for the active database.

    Returns
    -------
    list[str]
        Human-readable list of problems. Empty when the contract is healthy.
    """
    registry = load_dataset_registry(con)
    issues: list[str] = []
    for contract in registry.by_table_key.values():
        if contract.is_view and not include_views:
            continue
        table_schema = contract.schema
        if table_schema is None:
            continue
        arrow_schema = _arrow_schema_for_table(con, table_key=contract.table_key)
        if arrow_schema is None:
            if missing_ok:
                continue
            issues.append(f"{contract.table_key}: missing table")
            continue
        errors = schema_errors(table_schema, arrow_schema)
        errors.extend(schema_metadata_errors(arrow_schema))
        issues.extend(f"{contract.table_key}: {error}" for error in errors)
    return issues


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
    """Validate dataset contract and raise on any issues.

    Raises
    ------
    ValueError
        Raised when contract validation reports any issues.
    """
    issues = collect_contract_issues(con, include_views=include_views, missing_ok=False)
    if not issues:
        return
    message = "Contract validation failed:\n" + "\n".join(f"- {issue}" for issue in issues)
    raise ValueError(message)
