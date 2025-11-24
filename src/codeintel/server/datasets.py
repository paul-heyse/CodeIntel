"""Dataset registry helpers shared by server and MCP backends."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Literal

import duckdb

from codeintel.config.schemas.tables import TABLE_SCHEMAS

if TYPE_CHECKING:
    from codeintel.mcp.query_service import BackendLimits

DOCS_VIEWS = {
    "v_function_summary": "docs.v_function_summary",
    "v_call_graph_enriched": "docs.v_call_graph_enriched",
    "v_test_to_function": "docs.v_test_to_function",
    "v_file_summary": "docs.v_file_summary",
    "v_function_profile": "docs.v_function_profile",
    "v_file_profile": "docs.v_file_profile",
    "v_module_profile": "docs.v_module_profile",
    "v_function_architecture": "docs.v_function_architecture",
    "v_module_architecture": "docs.v_module_architecture",
    "v_subsystem_summary": "docs.v_subsystem_summary",
    "v_module_with_subsystem": "docs.v_module_with_subsystem",
}


def _dataset_name(table_key: str) -> str:
    """
    Derive dataset name from a fully qualified table key.

    Returns
    -------
    str
        Dataset-safe name derived from the table key.
    """
    _, name = table_key.split(".", maxsplit=1)
    return name


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
    for table_key in sorted(TABLE_SCHEMAS):
        name = _dataset_name(table_key)
        if name not in registry:
            registry[name] = table_key
    if include_docs_views == "include":
        for name, table in DOCS_VIEWS.items():
            if name not in registry:
                registry[name] = table
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
    from codeintel.mcp.query_service import BackendLimits  # Local import to avoid circular deps.

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
    schema = TABLE_SCHEMAS.get(table)
    if schema is None:
        return f"DuckDB table/view {table}"
    column_names = ", ".join(col.name for col in schema.columns[:PREVIEW_COLUMN_COUNT])
    extra = "" if len(schema.columns) <= PREVIEW_COLUMN_COUNT else "..."
    return f"{name}: {table} ({column_names}{extra})"


def validate_dataset_registry(
    con: duckdb.DuckDBPyConnection, dataset_tables: dict[str, str]
) -> None:
    """
    Validate that registered datasets exist and match expected schemas.

    Raises
    ------
    ValueError
        When required tables/views are missing or mismatched.
    """
    missing: list[str] = []
    mismatched: list[str] = []

    for dataset_name, table in sorted(dataset_tables.items()):
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

        expected_schema = TABLE_SCHEMAS.get(table)
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
            (str(col_name).lower(), str(col_type).upper(), str(nullable).upper() == "YES")
            for col_name, col_type, nullable in rows
        ]
        expected = [
            (col.name.lower(), col.type.upper(), col.nullable) for col in expected_schema.columns
        ]
        if actual != expected:
            mismatched.append(table)

    if missing or mismatched:
        parts: list[str] = []
        if missing:
            parts.append(f"missing tables/views: {', '.join(missing)}")
        if mismatched:
            parts.append(f"schema mismatches: {', '.join(mismatched)}")
        message = "Dataset registry validation failed; " + " | ".join(parts)
        raise ValueError(message)
