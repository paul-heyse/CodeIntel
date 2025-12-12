"""Helpers for docs view tests (indexes, seeding, profiling DB creation)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb

from codeintel.storage.metadata import bootstrap_metadata_datasets
from codeintel.storage.schema import apply_all_schemas
from codeintel.storage.views import create_all_views

if TYPE_CHECKING:
    from pathlib import Path

    from duckdb import DuckDBPyConnection


def list_indexes(con: DuckDBPyConnection, *, schema: str, table: str) -> set[str]:
    """
    Return index names for a table.

    Returns
    -------
    set[str]
        Index names defined for the table.
    """
    rows = con.execute(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE schema_name = ? AND table_name = ?
        """,
        [schema, table],
    ).fetchall()
    return {str(row[0]) for row in rows}


def seed_subsystem(con: DuckDBPyConnection, *, overrides: dict[str, object] | None = None) -> None:
    """
    Seed analytics.subsystems with a single subsystem row.

    Parameters
    ----------
    con
        Active DuckDB connection.
    overrides
        Optional overrides for the default subsystem payload.
    """
    base: dict[str, object] = {
        "repo": "demo/repo",
        "commit": "deadbeef",
        "subsystem_id": "subsysdemo",
        "name": "Subsystem Demo",
        "description": "demo subsystem",
        "module_count": 1,
        "function_count": 1,
    }
    if overrides:
        base.update(overrides)
    con.execute(
        """
        INSERT OR REPLACE INTO analytics.subsystems (
            repo, commit, subsystem_id, name, description,
            module_count, modules_json, entrypoints_json,
            internal_edge_count, external_edge_count, fan_in, fan_out,
            function_count, avg_risk_score, max_risk_score,
            high_risk_function_count, risk_level, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, '[]', '[]', ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [
            base["repo"],
            base["commit"],
            base["subsystem_id"],
            base["name"],
            base["description"],
            base["module_count"],
            0,
            0,
            0,
            0,
            base["function_count"],
            0.1,
            0.2,
            0,
            "low",
        ],
    )


def create_bootstrapped_docs_db(db_path: Path) -> None:
    """Create a file-backed DuckDB with schemas, views, and metadata bootstrapped."""
    con = duckdb.connect(":memory:")
    try:
        con.execute(f"ATTACH DATABASE '{db_path}' AS test_db (STORAGE_VERSION 'v1.4.0')")
        con.execute("USE test_db")
        apply_all_schemas(con)
        create_all_views(con)
        bootstrap_metadata_datasets(con, include_views=True)
    finally:
        con.close()


__all__ = [
    "create_bootstrapped_docs_db",
    "list_indexes",
    "seed_subsystem",
]
