"""Shared DuckDB test harness utilities."""

from __future__ import annotations

from typing import Final

import duckdb

from codeintel.storage.gateway import StorageGateway, open_memory_gateway

DEFAULT_REPO: Final = "demo/repo"
DEFAULT_COMMIT: Final = "deadbeef"


def make_memory_gateway(
    *,
    apply_schema: bool = True,
    ensure_views: bool = False,
    validate_schema: bool = True,
) -> StorageGateway:
    """
    Create an in-memory StorageGateway for tests.

    Parameters
    ----------
    apply_schema
        When True, applies all project schemas.
    ensure_views
        When True, creates docs views after schema application.
    validate_schema
        When True, validates schema alignment after setup.

    Returns
    -------
    StorageGateway
        Gateway backed by an in-memory DuckDB connection.
    """
    return open_memory_gateway(
        apply_schema=apply_schema,
        ensure_views=ensure_views,
        validate_schema=validate_schema,
    )


def seed_repo_identity(
    con: duckdb.DuckDBPyConnection,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
) -> None:
    """Seed core.repo_map with a single repo/commit identity row."""
    con.execute("CREATE SCHEMA IF NOT EXISTS core;")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS core.repo_map (
            repo TEXT,
            commit TEXT,
            modules JSON,
            overlays JSON,
            generated_at TIMESTAMP
        );
        """
    )
    con.execute("DELETE FROM core.repo_map;")
    con.execute(
        "INSERT INTO core.repo_map VALUES (?, ?, '{}', '{}', CURRENT_TIMESTAMP)",
        [repo, commit],
    )


def seed_modules(
    con: duckdb.DuckDBPyConnection,
    modules: list[tuple[str, str, str, str]],
) -> None:
    """
    Insert module rows into core.modules.

    Parameters
    ----------
    con
        DuckDB connection to write into.
    modules
        Tuples of (module, path, repo, commit).
    """
    con.execute("CREATE SCHEMA IF NOT EXISTS core;")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS core.modules (
            module TEXT,
            path TEXT,
            repo TEXT,
            commit TEXT,
            language TEXT,
            tags JSON,
            owners JSON
        );
        """
    )
    con.executemany(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, 'python', '[]', '[]')
        """,
        modules,
    )
