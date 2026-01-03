"""Storage helper utilities.

This package provides various helper functions for DuckDB operations.

Submodules
----------
helpers.json
    JSON encode/decode helpers for DuckDB column values. Handles the various
    forms DuckDB returns JSON data (string, dict, list, None).

helpers.module_index
    Module metadata helpers. Imports from ingestion, so must be imported
    directly from the submodule.

Note
----
Only json helpers are re-exported here to avoid circular imports.
Import module_index directly from its submodule:

    from codeintel.storage.helpers.module_index import load_module_map

For row count operations, use `codeintel.storage.queries.safe`.

For profiling view plans, use `codeintel.storage.warehouse.Warehouse.profile_views`.
"""

from __future__ import annotations

from codeintel.core.serialization.json import (
    decode_json,
    decode_json_dict,
    decode_json_list,
    deserialize_str_tuple,
    encode_json_compact,
    serialize_str_sequence,
)
from codeintel.storage.helpers.table_key import (
    TableKey,
    TableKeyValidationError,
    is_valid_table_key,
    parse_table_key,
    split_table_key,
    validate_table_key,
)

__all__ = [
    "TableKey",
    "TableKeyValidationError",
    "decode_json",
    "decode_json_dict",
    "decode_json_list",
    "deserialize_str_tuple",
    "encode_json_compact",
    "is_valid_table_key",
    "parse_table_key",
    "serialize_str_sequence",
    "split_table_key",
    "validate_table_key",
]
