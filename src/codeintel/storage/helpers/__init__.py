"""Storage helper utilities.

This package provides various helper functions for DuckDB operations.

Submodules
----------
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
Only json helpers are re-exported here to avoid circular imports.
Import profiling and module_index directly from their submodules:

    from codeintel.storage.helpers.profiling import run_profile
    from codeintel.storage.helpers.module_index import load_module_map

For row count operations, use `codeintel.storage.validation.data_checks`.
"""

from __future__ import annotations

from codeintel.storage.helpers.json import (
    decode_json,
    decode_json_dict,
    decode_json_list,
    deserialize_str_tuple,
    encode_json_compact,
    serialize_str_sequence,
)
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.helpers.time import utc_now

__all__ = [
    "decode_json",
    "decode_json_dict",
    "decode_json_list",
    "deserialize_str_tuple",
    "encode_json_compact",
    "serialize_str_sequence",
    "split_table_key",
    "utc_now",
]
