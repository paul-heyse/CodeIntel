"""Ephemeral DuckDB gateway utilities.

This module provides helpers for in-memory connections that support schema
compilation and inference workflows while reusing DuckDBSession bootstrapping.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.storage.backend import DuckDBSession
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.minimal import MinimalStorageGateway

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.core.schemas.provider import SchemaProvider


@contextmanager
def ephemeral_gateway(*, schema_provider: SchemaProvider) -> Iterator[MinimalStorageGateway]:
    """Yield an in-memory MinimalStorageGateway wired with a SchemaProvider.

    Parameters
    ----------
    schema_provider
        Schema provider used for DDL and column-order enforcement.

    Yields
    ------
    MinimalStorageGateway
        In-memory gateway suitable for schema compilation/inference.
    """
    cfg = StorageConfig(
        db_path=Path(":memory:"),
        read_only=False,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
    )
    session = DuckDBSession(cfg)
    con = session.open()
    gateway = MinimalStorageGateway(con, schema_provider=schema_provider)
    try:
        yield gateway
    finally:
        gateway.close()


__all__ = ["ephemeral_gateway"]
