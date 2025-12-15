"""DuckDB schema definitions for the CodeIntel metadata warehouse.

This module provides thin wrapper functions for schema management that delegate
to `DuckDBPolicyBackend` for all DDL generation. The policy backend generates
DDL from the canonical schema provider, making the DAG the single source of truth.

All DDL is now generated via the schema provider and policy backend.
Legacy string-based DDL constants have been removed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.build.schemas import (
    get_schema_provider,
    is_view,
    iter_contracts,
)
from codeintel.storage.constants import SCHEMAS
from codeintel.storage.gateway.minimal import MinimalStorageGateway

if TYPE_CHECKING:
    from collections.abc import Iterable

    from duckdb import DuckDBPyConnection

    from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

log = logging.getLogger(__name__)

__all__ = [
    "SCHEMAS",
    "apply_all_schemas",
    "assert_schema_alignment",
    "create_schemas",
    "ensure_schemas_preserve",
]


def _get_policy_backend(con: DuckDBPyConnection) -> DuckDBPolicyBackend:
    """Create a policy backend wrapper for a raw connection.

    Parameters
    ----------
    con
        DuckDB connection.

    Returns
    -------
    DuckDBPolicyBackend
        Policy backend instance wrapping the connection.
    """
    provider = get_schema_provider()
    return MinimalStorageGateway(con, schema_provider=provider).policy


def create_schemas(con: DuckDBPyConnection) -> None:
    """Ensure logical schemas (core, graph, analytics, docs) exist.

    Parameters
    ----------
    con
        DuckDB connection.
    """
    for schema in SCHEMAS:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")


def apply_all_schemas(
    con: DuckDBPyConnection,
    extra_ddl: Iterable[str] | None = None,
) -> None:
    """Create all known tables in the current DuckDB database.

    Call this once at startup before running any pipeline steps that
    insert into these tables. This function generates all DDL from
    dataset contracts via the DuckDBPolicyBackend.

    Parameters
    ----------
    con
        DuckDB connection.
    extra_ddl
        Optional additional DDL statements to execute.
    """
    backend = _get_policy_backend(con)
    backend.ensure_all_schemas(drop_existing=True, extra_ddl=extra_ddl)


def ensure_schemas_preserve(
    con: DuckDBPyConnection,
    extra_ddl: Iterable[str] | None = None,
) -> None:
    """Ensure schemas and tables exist without dropping existing data.

    Create missing tables and indexes using IF NOT EXISTS; existing tables are
    left untouched. Use assert_schema_alignment separately to detect drift.

    Parameters
    ----------
    con
        DuckDB connection.
    extra_ddl
        Optional additional DDL statements to execute.
    """
    backend = _get_policy_backend(con)
    backend.ensure_schemas_preserve(extra_ddl=extra_ddl)


def assert_schema_alignment(
    con: DuckDBPyConnection,
    *,
    include_views: bool = True,
    strict: bool = True,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Validate that the live DuckDB schema matches the schema provider definitions.

    Parameters
    ----------
    con
        DuckDB connection.
    strict
        If True, raise RuntimeError on schema drift.
    include_views
        When False, ignore view contracts during alignment checks.
    logger
        Optional logger for error messages.

    Returns
    -------
    list[str]
        Human-readable drift messages; empty when aligned.

    Raises
    ------
    RuntimeError
        If strict is True and schema drift is detected.
    """
    provider = get_schema_provider()
    issues: list[str] = []

    for contract in iter_contracts():
        table_key = contract.table_key
        if is_view(table_key) and not include_views:
            continue
        table = provider.get_table_schema(table_key)
        if table is None:
            continue
        rows = con.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            ORDER BY ordinal_position
            """,
            [table.schema, table.name],
        ).fetchall()
        actual = [row[0] for row in rows]
        expected = table.column_names()
        if actual != expected:
            issues.append(f"{table.fq_name}: expected {expected} got {actual}")

    if issues:
        message = "; ".join(issues)
        logref = logger or log
        logref.error("Schema drift detected: %s", message)
        if strict:
            error_message = f"Schema drift detected: {message}"
            raise RuntimeError(error_message)
    return issues
