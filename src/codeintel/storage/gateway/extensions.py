"""DuckDB extension loading helpers.

This module centralizes extension parsing/validation and load/install behavior.
Feature modules should not ad-hoc INSTALL/LOAD extensions with bespoke rules.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

from codeintel.storage.gateway.protocol import DuckDBError

__all__ = [
    "DuckDBExecutor",
    "load_extensions_from_env",
    "parse_extensions_env",
    "require_extension",
]


class DuckDBExecutor(Protocol):
    """DuckDB execution protocol for extension load/install."""

    def execute(self, query: str, parameters: object | None = None) -> object:
        """Execute a SQL statement on the underlying connection."""
        ...


_EXTENSION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_EXTENSIONS_ENV = "CODEINTEL_DUCKDB_EXTENSIONS"


def _validate_extension_name(extension: str) -> None:
    if _EXTENSION_NAME_PATTERN.fullmatch(extension) is None:
        message = f"Invalid DuckDB extension name in {_EXTENSIONS_ENV}: {extension!r}"
        raise ValueError(message)


def parse_extensions_env(*, env_var: str = _EXTENSIONS_ENV) -> tuple[str, ...]:
    """Parse a comma-delimited DuckDB extension list from the environment.

    Parameters
    ----------
    env_var
        Environment variable name containing the comma-delimited extension list.

    Returns
    -------
    tuple[str, ...]
        Parsed extension names (trimmed), or an empty tuple when unset.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return ()
    return tuple(ext.strip() for ext in raw.split(",") if ext.strip())


def load_extensions_from_env(con: DuckDBExecutor, *, allow_install: bool) -> None:
    """Install/load DuckDB extensions listed in `CODEINTEL_DUCKDB_EXTENSIONS`.

    Parameters
    ----------
    con
        Active DuckDB connection.
    allow_install
        When True, attempt `INSTALL` before `LOAD`. For read-only connections this
        should be False to avoid unexpected network or disk mutations.
    """
    for extension in parse_extensions_env():
        _validate_extension_name(extension)
        if allow_install:
            con.execute(f"INSTALL {extension}")
        con.execute(f"LOAD {extension}")


def require_extension(con: DuckDBExecutor, extension: str, *, allow_install: bool) -> None:
    """Ensure a DuckDB extension is available and loaded.

    Parameters
    ----------
    con
        Active DuckDB connection.
    extension
        DuckDB extension name (alphanumeric/underscore).
    allow_install
        When True, attempt `INSTALL` before `LOAD`. Use False for read-only paths.

    Raises
    ------
    RuntimeError
        If the extension cannot be installed/loaded.
    """
    _validate_extension_name(extension)
    try:
        if allow_install:
            con.execute(f"INSTALL {extension}")
        con.execute(f"LOAD {extension}")
    except DuckDBError as exc:
        message = (
            f"DuckDB extension {extension!r} is required "
            f"(set {_EXTENSIONS_ENV}={extension})"
        )
        raise RuntimeError(message) from exc
