"""Factories for building shared query services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import duckdb

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.config.serving_models import ServingConfig, verify_db_identity
from codeintel.mcp.query_service import DuckDBQueryService
from codeintel.server.datasets import build_dataset_registry, describe_dataset
from codeintel.services.query_service import (
    BackendLimits,
    HttpQueryService,
    LocalQueryService,
    ServiceObservability,
)


def _split_table_identifier(table: str) -> tuple[str, str]:
    """
    Split a schema-qualified table name into schema and table components.

    Parameters
    ----------
    table:
        Schema-qualified table name (e.g., "core.ast_nodes").

    Returns
    -------
    tuple[str, str]
        Schema name and table name.

    Raises
    ------
    ValueError
        When the identifier is not schema-qualified.
    """
    if "." not in table:
        message = f"Table identifier must be schema-qualified: {table}"
        raise ValueError(message)
    schema_name, table_name = table.split(".", maxsplit=1)
    return schema_name, table_name


def validate_dataset_registry(
    con: duckdb.DuckDBPyConnection, dataset_tables: dict[str, str]
) -> None:
    """
    Validate that registered datasets exist and match expected schemas.

    Parameters
    ----------
    con:
        Open DuckDB connection.
    dataset_tables:
        Mapping of dataset name to fully qualified table/view name.

    Raises
    ------
    ValueError
        When required tables/views are missing or schemas do not match.
    """
    missing: list[str] = []
    mismatched: list[str] = []

    for dataset_name, table in sorted(dataset_tables.items()):
        schema_name, table_name = _split_table_identifier(table)
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
            (col.name.lower(), col.type.upper(), col.nullable)
            for col in expected_schema.columns
        ]
        if actual != expected:
            actual_desc = ", ".join(
                f"{name}:{col_type}:{'null' if is_nullable else 'notnull'}"
                for name, col_type, is_nullable in actual
            )
            expected_desc = ", ".join(
                f"{name}:{col_type}:{'null' if is_nullable else 'notnull'}"
                for name, col_type, is_nullable in expected
            )
            mismatched.append(f"{table} expected [{expected_desc}] found [{actual_desc}]")

    if missing or mismatched:
        parts: list[str] = []
        if missing:
            parts.append(f"missing tables/views: {', '.join(missing)}")
        if mismatched:
            parts.append(f"schema mismatches: {', '.join(mismatched)}")
        message = "Dataset registry validation failed; " + " | ".join(parts)
        raise ValueError(message)


@dataclass
class DatasetRegistryOptions:
    """Options controlling dataset registry composition and validation."""

    tables: dict[str, str] | None = None
    describe_fn: Callable[[str, str], str] = describe_dataset
    validate: bool = True


def build_local_query_service(
    con: duckdb.DuckDBPyConnection,
    cfg: ServingConfig,
    *,
    query: DuckDBQueryService,
    registry: DatasetRegistryOptions | None = None,
    observability: ServiceObservability | None = None,
) -> LocalQueryService:
    """
    Construct a LocalQueryService with identity verification and dataset registry.

    Parameters
    ----------
    con:
        Open DuckDB connection.
    cfg:
        Serving configuration describing repo/commit and limits.
    query:
        Pre-constructed DuckDBQueryService to attach to the LocalQueryService.
    registry:
        Dataset registry options including validation behavior.
    observability:
        Optional observability configuration for structured logging.

    Returns
    -------
    LocalQueryService
        Service bound to the provided DuckDB connection.
    """
    verify_db_identity(con, cfg)
    opts = registry or DatasetRegistryOptions()
    tables = opts.tables if opts.tables is not None else build_dataset_registry()
    if opts.validate and tables:
        validate_dataset_registry(con, tables)
    return LocalQueryService(
        query=query,
        dataset_tables=tables,
        describe_dataset_fn=opts.describe_fn,
        observability=observability,
    )


def build_http_query_service(
    request_json: Callable[[str, dict[str, object]], object],
    *,
    limits: BackendLimits,
    observability: ServiceObservability | None = None,
) -> HttpQueryService:
    """
    Construct an HttpQueryService for remote API delegation.

    Parameters
    ----------
    request_json:
        Callable that performs HTTP GETs and returns decoded JSON.
    limits:
        BackendLimits instance controlling clamping.
    observability:
        Optional observability configuration for structured logging.

    Returns
    -------
    HttpQueryService
        Service wrapper for remote transport.
    """
    return HttpQueryService(
        request_json=request_json,
        limits=limits,
        observability=observability,
    )
