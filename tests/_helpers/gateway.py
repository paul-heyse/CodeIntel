"""Shared helpers for isolated gateway/DuckDB test setup."""

from __future__ import annotations

from pathlib import Path

from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway
from codeintel.storage.gateway import open_memory_gateway as _open_memory_gateway


def open_fresh_duckdb(db_path: Path) -> StorageGateway:
    """
    Return a fresh DuckDB connection for tests.

    Returns
    -------
    StorageGateway
        Open gateway (caller must close).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=True,
        validate_schema=True,
    )
    return open_gateway(cfg)


def seed_tables(gateway: StorageGateway, ddl: list[str]) -> None:
    """Apply defensive DDL statements (DROP/CREATE) to avoid cross-test conflicts."""
    for stmt in ddl:
        gateway.con.execute(stmt)


def open_ingestion_gateway(
    *,
    apply_schema: bool = True,
    ensure_views: bool = False,
    validate_schema: bool = True,
    strict_schema: bool = True,
) -> StorageGateway:
    """
    Return an in-memory gateway prepped for ingestion runners.

    Parameters mirror `open_memory_gateway`; schema application is enabled by default so
    ingestion steps can write tables without extra setup.

    Returns
    -------
    StorageGateway
        Gateway configured for ingestion tests.
    """
    effective_ensure_views = ensure_views or strict_schema
    effective_validate_schema = validate_schema or strict_schema
    return _open_memory_gateway(
        apply_schema=apply_schema,
        ensure_views=effective_ensure_views,
        validate_schema=effective_validate_schema,
    )
