"""Nested Arrow helper re-exports for build pipelines."""

from __future__ import annotations

from codeintel.core.columnar.nested_ops import (
    deep_cast_array,
    deep_cast_table_to_contract,
    is_allowed_promotion,
    make_extras_kv_map,
    make_extras_struct,
    unify_schemas_with_contract_first,
)

__all__ = [
    "deep_cast_array",
    "deep_cast_table_to_contract",
    "is_allowed_promotion",
    "make_extras_kv_map",
    "make_extras_struct",
    "unify_schemas_with_contract_first",
]
