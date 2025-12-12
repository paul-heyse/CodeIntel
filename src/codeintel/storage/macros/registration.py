"""Deprecated ingest macro registration helpers (no-op stubs).

Macros are retired; this module remains only to surface deprecation warnings for any
lingering imports during the transition to ibis/policy-backend.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from warnings import warn

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

_MACRO_CACHE: dict[int, set[str]] = {}
_MACRO_LOCK = threading.RLock()

warn(
    "codeintel.storage.macros.registration is deprecated; ingest macros are retired.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "assert_ingest_macros_present",
    "clear_macro_cache_for_connection",
    "ensure_ingest_macros",
    "list_ingest_macros",
]


def _registered_macros(con: DuckDBPyConnection) -> set[str]:
    rows = con.execute(
        """
        SELECT schema_name, function_name
        FROM duckdb_functions()
        WHERE function_type IN ('macro', 'table_macro')
        """
    ).fetchall()
    names: set[str] = set()
    for schema_name, function_name in rows:
        if isinstance(function_name, str):
            fn_lower = function_name.lower()
            names.add(fn_lower)
            if isinstance(schema_name, str) and schema_name:
                names.add(f"{schema_name.lower()}.{fn_lower}")

            parts = fn_lower.split(".")
            if len(parts) > 1:
                schema_part, fn_part = parts[-2], parts[-1]
                names.add(f"{schema_part}.{fn_part}")
                names.add(fn_part)
    return names


def ensure_ingest_macros(con: DuckDBPyConnection) -> None:
    """Deprecate ingest macro registration (no-op)."""
    warn(
        "ensure_ingest_macros is deprecated and now a no-op; macros are retired.",
        DeprecationWarning,
        stacklevel=2,
    )
    _ = con


def clear_macro_cache_for_connection(con_or_key: DuckDBPyConnection | int) -> None:
    """Deprecate macro cache clearing (no-op)."""
    warn(
        "clear_macro_cache_for_connection is deprecated; macro caching is unused.",
        DeprecationWarning,
        stacklevel=2,
    )
    cache_key = con_or_key if isinstance(con_or_key, int) else id(con_or_key)
    _MACRO_CACHE.pop(cache_key, None)


def list_ingest_macros(con: DuckDBPyConnection) -> set[str]:
    """Deprecate macro listing; always returns empty set.

    Returns
    -------
    set[str]
        Empty set (macros are retired).
    """
    warn(
        "list_ingest_macros is deprecated; macros are retired.",
        DeprecationWarning,
        stacklevel=2,
    )
    _ = con
    return set()


def assert_ingest_macros_present(con: DuckDBPyConnection) -> None:
    """Deprecate macro validation (no-op)."""
    warn(
        "assert_ingest_macros_present is deprecated; macros are retired.",
        DeprecationWarning,
        stacklevel=2,
    )
    _ = con
