"""Arrow compute kernel helpers for tabular pipelines."""

from __future__ import annotations

from codeintel.core.columnar.kernels import (
    SortKey,
    case_when,
    coalesce,
    hash_struct_goid,
    hash_struct_ordinal,
    indices_nonzero,
    list_element,
    list_slice,
    list_value_length,
    regex_match,
    regex_replace,
    replace_with_mask,
    safe_cast,
    safe_divide,
    stable_sort_indices,
    stable_sort_table,
    struct_field,
)

__all__ = [
    "SortKey",
    "case_when",
    "coalesce",
    "hash_struct_goid",
    "hash_struct_ordinal",
    "indices_nonzero",
    "list_element",
    "list_slice",
    "list_value_length",
    "regex_match",
    "regex_replace",
    "replace_with_mask",
    "safe_cast",
    "safe_divide",
    "stable_sort_indices",
    "stable_sort_table",
    "struct_field",
]
