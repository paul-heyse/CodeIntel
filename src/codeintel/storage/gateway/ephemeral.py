"""Ephemeral DuckDB gateway utilities.

This module is the single place outside tests where we import DuckDB directly.
It provides helpers for in-memory connections that support schema compilation
and inference workflows.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import duckdb

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
    con = duckdb.connect(":memory:")
    gateway = MinimalStorageGateway(con, schema_provider=schema_provider)
    try:
        yield gateway
    finally:
        gateway.close()


__all__ = ["ephemeral_gateway"]
