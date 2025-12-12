"""Dataset registry helpers shared by all DuckDB-backed backends."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Literal

from codeintel.config.datasets import get_dataset_contracts, get_dataset_contracts_by_table_key
from codeintel.serving.backend.pagination import BackendLimits

if TYPE_CHECKING:
    from codeintel.storage.gateway import DuckDBConnection, StorageGateway

DOCS_VIEWS = {
    name: contract.table_key
    for name, contract in get_dataset_contracts().items()
    if contract.is_view
}


def _normalize_type(value: str) -> str:
    normalized = value.upper()
    if normalized in {"TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"}:
        return "TIMESTAMP WITH TIME ZONE"
    if normalized.startswith("DECIMAL"):
        return "DECIMAL"
    return normalized


PREVIEW_COLUMN_COUNT = 5


def build_dataset_registry(
    *, include_docs_views: Literal["include", "exclude"] = "include"
) -> dict[str, str]:
    """
    Build deterministic dataset registry.

    Returns
    -------
    dict[str, str]
        Mapping of dataset name to fully qualified table/view name.
    """
    registry: OrderedDict[str, str] = OrderedDict()
    for name, contract in sorted(
        get_dataset_contracts().items(), key=lambda item: item[1].table_key
    ):
        if include_docs_views == "exclude" and contract.is_view:
            continue
        registry[name] = contract.table_key
    return dict(registry)


def build_registry_and_limits(
    cfg: object, *, include_docs_views: Literal["include", "exclude"] = "include"
) -> tuple[dict[str, str], BackendLimits]:
    """
    Return a dataset registry and backend limits derived from configuration.

    Parameters
    ----------
    cfg:
        Configuration object exposing default_limit and max_rows_per_call.
    include_docs_views:
        Whether to include docs views in the registry.

    Returns
    -------
    tuple[dict[str, str], BackendLimits]
        Registry mapping and backend limits built from the configuration.
    """
    registry = build_dataset_registry(include_docs_views=include_docs_views)
    limits = BackendLimits.from_config(cfg)
    return registry, limits


def describe_dataset(name: str, table: str) -> str:
    """
    Produce a human-friendly description for a dataset/table.

    Returns
    -------
    str
        Description string including a column preview when available.
    """
    contract = get_dataset_contracts_by_table_key().get(table)
    if contract is None or contract.schema is None:
        return f"{name}: {table}"
    column_names = contract.schema.column_names()[:PREVIEW_COLUMN_COUNT]
    extra = "" if len(contract.schema.columns) <= PREVIEW_COLUMN_COUNT else "..."
    return f"{name}: {table} ({', '.join(column_names)}{extra})"


def _collect_dataset_registry_issues(
    con: DuckDBConnection, dataset_mapping: dict[str, str]
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    mismatched: list[str] = []

    for dataset_name, table in sorted(dataset_mapping.items()):
        if "." not in table:
            missing.append(f"{dataset_name} ({table})")
            continue
        schema_name, table_name = table.split(".", maxsplit=1)
        exists = con.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = ? AND table_name = ?
            LIMIT 1
            """,
            [schema_name, table_name],
        ).fetchone()
        if exists is None:
            missing.append(f"{dataset_name} ({table})")
            continue

        contract = get_dataset_contracts_by_table_key().get(table)
        if contract is None:
            missing.append(f"{dataset_name} ({table})")
            continue

        expected_schema = contract.schema
        if expected_schema is None:
            continue

        rows = con.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            ORDER BY ordinal_position
            """,
            [schema_name, table_name],
        ).fetchall()
        actual = [
            (
                str(col_name).lower(),
                _normalize_type(str(col_type)),
                str(nullable).upper() == "YES",
            )
            for col_name, col_type, nullable in rows
        ]
        expected = [
            (col.name.lower(), _normalize_type(col.type), col.nullable)
            for col in expected_schema.columns
        ]
        if actual != expected:
            mismatched.append(table)
            continue
    return missing, mismatched


def validate_dataset_registry(gateway: StorageGateway) -> None:
    """
    Validate that registered datasets exist and match expected schemas.

    Parameters
    ----------
    gateway
        StorageGateway providing the connection and dataset registry.

    Raises
    ------
    ValueError
        When required tables/views are missing or mismatched.
    """
    con = gateway.con
    dataset_mapping = dict(gateway.datasets.mapping)
    missing, mismatched = _collect_dataset_registry_issues(con, dataset_mapping)

    if missing or mismatched:
        parts: list[str] = []
        if missing:
            parts.append(f"missing tables/views: {', '.join(missing)}")
        if mismatched:
            parts.append(f"schema mismatches: {', '.join(mismatched)}")
        message = "Dataset registry validation failed; " + " | ".join(parts)
        raise ValueError(message)
