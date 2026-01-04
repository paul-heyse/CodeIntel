"""Graph-specific Arrow compute filters."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import (
    and_kleene,
    is_valid_mask,
    kind_is_function_or_method,
    language_is_python_or_null,
    non_empty_string_mask,
)


def filter_python_modules(modules_table: pa.Table) -> pa.Table:
    """Filter core.modules to Python entries with valid path/module values.

    Returns
    -------
    pa.Table
        Filtered table or the original on failure.
    """
    if modules_table.num_rows == 0:
        return modules_table
    columns = set(modules_table.column_names)
    if not {"path", "module"}.issubset(columns):
        return modules_table
    try:
        path_mask = non_empty_string_mask(modules_table.column("path"))
        module_mask = non_empty_string_mask(modules_table.column("module"))
        mask = and_kleene(path_mask, module_mask)
        if "language" in columns:
            language_mask = language_is_python_or_null(modules_table.column("language"))
            mask = and_kleene(mask, language_mask)
        return safe_filter(modules_table, mask)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return modules_table


def filter_modules_with_language(modules_table: pa.Table) -> pa.Table:
    """Filter core.modules to entries with non-empty path/module/language strings.

    Returns
    -------
    pa.Table
        Filtered table or the original on failure.
    """
    if modules_table.num_rows == 0:
        return modules_table
    columns = set(modules_table.column_names)
    if not {"path", "module", "language"}.issubset(columns):
        return modules_table
    try:
        path_mask = non_empty_string_mask(modules_table.column("path"))
        module_mask = non_empty_string_mask(modules_table.column("module"))
        language_mask = non_empty_string_mask(modules_table.column("language"))
        mask = and_kleene(path_mask, module_mask)
        mask = and_kleene(mask, language_mask)
        return safe_filter(modules_table, mask)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return modules_table


def filter_python_goids(goids_table: pa.Table) -> pa.Table:
    """Filter core.goids to Python function/method entries with valid identifiers.

    Returns
    -------
    pa.Table
        Filtered table or the original on failure.
    """
    if goids_table.num_rows == 0:
        return goids_table
    columns = set(goids_table.column_names)
    required = {"kind", "rel_path", "goid_h128"}
    if not required.issubset(columns):
        return goids_table
    try:
        kind_mask = kind_is_function_or_method(goids_table.column("kind"))
        path_mask = non_empty_string_mask(goids_table.column("rel_path"))
        goid_mask = is_valid_mask(goids_table.column("goid_h128"))
        mask = and_kleene(kind_mask, path_mask)
        mask = and_kleene(mask, goid_mask)
        if "language" in columns:
            language_mask = language_is_python_or_null(goids_table.column("language"))
            mask = and_kleene(mask, language_mask)
        return safe_filter(goids_table, mask)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return goids_table


def filter_symbol_occurrences(occurrences_table: pa.Table) -> pa.Table:
    """Filter SCIP occurrences to rows with valid symbol/path/line values.

    Returns
    -------
    pa.Table
        Filtered table or the original on failure.
    """
    if occurrences_table.num_rows == 0:
        return occurrences_table
    required = {"symbol", "rel_path", "start_line"}
    if not required.issubset(set(occurrences_table.column_names)):
        return occurrences_table
    try:
        symbol_mask = non_empty_string_mask(occurrences_table.column("symbol"))
        path_mask = non_empty_string_mask(occurrences_table.column("rel_path"))
        line_mask = is_valid_mask(occurrences_table.column("start_line"))
        mask = and_kleene(symbol_mask, path_mask)
        mask = and_kleene(mask, line_mask)
        return safe_filter(occurrences_table, mask)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return occurrences_table


def filter_goids_with_spans(goids_table: pa.Table) -> pa.Table:
    """Filter GOID rows to entries with valid span identifiers.

    Returns
    -------
    pa.Table
        Filtered table or the original on failure.
    """
    if goids_table.num_rows == 0:
        return goids_table
    required = {"rel_path", "goid_h128", "start_line"}
    if not required.issubset(set(goids_table.column_names)):
        return goids_table
    try:
        path_mask = non_empty_string_mask(goids_table.column("rel_path"))
        goid_mask = is_valid_mask(goids_table.column("goid_h128"))
        line_mask = is_valid_mask(goids_table.column("start_line"))
        mask = and_kleene(path_mask, goid_mask)
        mask = and_kleene(mask, line_mask)
        return safe_filter(goids_table, mask)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return goids_table


__all__ = [
    "filter_goids_with_spans",
    "filter_modules_with_language",
    "filter_python_goids",
    "filter_python_modules",
    "filter_symbol_occurrences",
]
