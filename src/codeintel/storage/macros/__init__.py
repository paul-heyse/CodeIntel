"""Macro management utilities for DuckDB.

This package provides utilities for managing DuckDB macros used for data
ingestion and normalization:

- macros.generation: Render normalized macro DDL from table schemas
- macros.registration: Registration and cache management

Macro DDL definitions and mappings remain in metadata_bootstrap.py as they
are tightly coupled with the bootstrap process.
"""

from __future__ import annotations

from codeintel.storage.macros.generation import (
    DEFAULT_LIMIT,
    RenderedMacro,
    render_macro,
)
from codeintel.storage.macros.registration import (
    assert_ingest_macros_present,
    clear_macro_cache_for_connection,
    ensure_ingest_macros,
    list_ingest_macros,
)

__all__ = [
    "DEFAULT_LIMIT",
    "RenderedMacro",
    "assert_ingest_macros_present",
    "clear_macro_cache_for_connection",
    "ensure_ingest_macros",
    "list_ingest_macros",
    "render_macro",
]
