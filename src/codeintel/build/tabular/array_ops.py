"""Shared Arrow array helpers for compute-heavy operations."""

from __future__ import annotations

from codeintel.core.columnar.normalization import (
    normalize_array as ensure_array,
)
from codeintel.core.columnar.normalization import (
    normalize_array_for_compute,
    normalize_table_for_compute,
)
from codeintel.core.columnar.set_ops import index_in, take_by_key, value_set_array
from codeintel.core.columnar.type_normalization import (
    normalize_binary_view_array,
    normalize_string_view_array,
)

__all__ = [
    "ensure_array",
    "index_in",
    "normalize_array_for_compute",
    "normalize_binary_view_array",
    "normalize_string_view_array",
    "normalize_table_for_compute",
    "take_by_key",
    "value_set_array",
]
