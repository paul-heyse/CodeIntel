"""DuckDB extension loading helpers.

This module centralizes extension parsing/validation and load/install behavior.
Feature modules should not ad-hoc INSTALL/LOAD extensions with bespoke rules.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import DuckDBConnection

__all__ = [
    "load_extensions_from_env",
    "parse_extensions_env",
]

_EXTENSION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_EXTENSIONS_ENV = "CODEINTEL_DUCKDB_EXTENSIONS"


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


def load_extensions_from_env(con: DuckDBConnection, *, allow_install: bool) -> None:
    """Install/load DuckDB extensions listed in `CODEINTEL_DUCKDB_EXTENSIONS`.

    Parameters
    ----------
    con
        Active DuckDB connection.
    allow_install
        When True, attempt `INSTALL` before `LOAD`. For read-only connections this
        should be False to avoid unexpected network or disk mutations.

    Raises
    ------
    ValueError
        If an extension name is invalid.
    """
    for extension in parse_extensions_env():
        if _EXTENSION_NAME_PATTERN.fullmatch(extension) is None:
            message = f"Invalid DuckDB extension name in {_EXTENSIONS_ENV}: {extension!r}"
            raise ValueError(message)
        if allow_install:
            con.execute(f"INSTALL {extension}")
        con.execute(f"LOAD {extension}")
