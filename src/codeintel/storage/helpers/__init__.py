"""Storage helper utilities.

This package provides various helper functions for DuckDB operations.

Submodules
----------
helpers.db
    Bulk row insertion via `macro_insert_rows()` - the canonical method
    for inserting data into DuckDB tables. Used internally by accessor classes.

helpers.json
    JSON encode/decode helpers for DuckDB column values. Handles the various
    forms DuckDB returns JSON data (string, dict, list, None).

helpers.profiling
    Docs view profiling utilities. Imports from gateway, so must be imported
    directly from the submodule.

helpers.module_index
    Module metadata helpers. Imports from ingestion, so must be imported
    directly from the submodule.

Note
----
Only db and json helpers are re-exported here to avoid circular imports.
Import profiling and module_index directly from their submodules:

    from codeintel.storage.helpers.profiling import run_profile
    from codeintel.storage.helpers.module_index import load_module_map

For row count operations, use `codeintel.storage.validation.data_checks`.
"""

from __future__ import annotations

from codeintel.storage.errors import DUCKDB_ERRORS
from codeintel.storage.helpers.db import macro_insert_rows
from codeintel.storage.helpers.json import (
    decode_json,
    decode_json_dict,
    decode_json_list,
    encode_json_compact,
)

__all__ = [
    "DUCKDB_ERRORS",
    "decode_json",
    "decode_json_dict",
    "decode_json_list",
    "encode_json_compact",
    "macro_insert_rows",
]
