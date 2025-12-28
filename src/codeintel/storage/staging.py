"""Lifecycle-safe staging helpers for DuckDB-backed workflows.

This module provides small utilities for temporarily registering in-memory data
structures (e.g., Polars DataFrames or Arrow tables) with a DuckDB connection
and guaranteeing cleanup.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from codeintel.core.execution.ids import new_uuid_hex

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def registered_temp_relation(
    con: object,
    obj: object,
    *,
    prefix: str = "ci_tmp_",
) -> Iterator[str]:
    """Register an object on a DuckDB connection and unregister it on exit.

    Parameters
    ----------
    con
        DuckDB connection object exposing `register(name, obj)` and
        `unregister(name)` methods.
    obj
        Object to register (e.g., Polars DataFrame, pyarrow Table).
    prefix
        Prefix used to generate a unique registered name.

    Yields
    ------
    str
        Registered relation name.

    Raises
    ------
    TypeError
        If the connection does not support `register`/`unregister`.
    """
    register = getattr(con, "register", None)
    unregister = getattr(con, "unregister", None)
    if not callable(register) or not callable(unregister):
        msg = "DuckDB connection must support register(name, obj) and unregister(name)"
        raise TypeError(msg)

    safe_prefix = prefix.strip() or "ci_tmp_"
    name = f"{safe_prefix}{new_uuid_hex()}"
    register(name, obj)
    try:
        yield name
    finally:
        unregister(name)


__all__ = ["registered_temp_relation"]
