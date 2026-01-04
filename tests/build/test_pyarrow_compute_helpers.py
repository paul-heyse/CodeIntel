"""Tests for shared PyArrow compute helpers."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.hamilton.native.graphs.compute_filters import (
    filter_function_ast_nodes,
    filter_goids_with_spans,
    filter_modules_with_language,
    filter_python_goids,
    filter_python_modules,
    filter_symbol_occurrences,
)
from codeintel.build.tabular.compute_columns import append_constant_columns, empty_table
from codeintel.build.tabular.compute_masks import (
    kind_is_function_or_method,
    language_is_python_or_null,
    node_type_is_function,
    non_empty_string_mask,
)
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_empty_table_schema() -> None:
    """empty_table should preserve column names on empty outputs."""
    table = empty_table(["col_a", "col_b"])

    expect_equal(table.column_names, ["col_a", "col_b"])
    expect_equal(table.num_rows, 0)


def test_append_constant_columns_adds_missing() -> None:
    """append_constant_columns should add missing constant columns."""
    table = pa.table({"col_a": [1, 2]})

    updated = append_constant_columns(table, {"col_a": "ignored", "edge_kind": "DFG"})

    expect_equal(updated.column_names, ["col_a", "edge_kind"])
    expect_equal(updated.column("edge_kind").to_pylist(), ["DFG", "DFG"])


def test_non_empty_string_mask() -> None:
    """non_empty_string_mask should ignore empty or null strings."""
    values = pa.array(["value", "", None])

    mask = non_empty_string_mask(values)

    expect_equal(mask.to_pylist(), [True, False, False])


def test_language_is_python_or_null() -> None:
    """language_is_python_or_null should accept Python or NULL markers."""
    values = pa.array(["python", None, "go"])

    mask = language_is_python_or_null(values)

    expect_equal(mask.to_pylist(), [True, True, False])


def test_kind_is_function_or_method() -> None:
    """kind_is_function_or_method should match function and method kinds."""
    values = pa.array(["function", "method", "class"])

    mask = kind_is_function_or_method(values)

    expect_equal(mask.to_pylist(), [True, True, False])


def test_node_type_is_function() -> None:
    """node_type_is_function should match function/async function nodes."""
    values = pa.array(["FunctionDef", "AsyncFunctionDef", "ClassDef"])

    mask = node_type_is_function(values)

    expect_equal(mask.to_pylist(), [True, True, False])


def test_filter_python_modules() -> None:
    """filter_python_modules should keep Python module rows only."""
    table = pa.Table.from_pylist(
        [
            {"path": "a.py", "module": "a", "language": "python"},
            {"path": "", "module": "b", "language": "python"},
            {"path": "c.py", "module": "c", "language": "go"},
            {"path": "d.py", "module": "d", "language": None},
        ]
    )

    filtered = filter_python_modules(table)

    expect_equal(filtered.column("path").to_pylist(), ["a.py", "d.py"])


def test_filter_modules_with_language() -> None:
    """filter_modules_with_language should require a non-empty language."""
    table = pa.Table.from_pylist(
        [
            {"path": "a.py", "module": "a", "language": "python"},
            {"path": "b.py", "module": "b", "language": None},
        ]
    )

    filtered = filter_modules_with_language(table)

    expect_equal(filtered.column("module").to_pylist(), ["a"])


def test_filter_python_goids() -> None:
    """filter_python_goids should keep Python function/method rows."""
    table = pa.Table.from_pylist(
        [
            {"kind": "function", "rel_path": "a.py", "goid_h128": 1, "language": "python"},
            {"kind": "class", "rel_path": "a.py", "goid_h128": 2, "language": "python"},
            {"kind": "method", "rel_path": "", "goid_h128": 3, "language": "python"},
            {"kind": "function", "rel_path": "b.py", "goid_h128": None, "language": None},
            {"kind": "method", "rel_path": "c.py", "goid_h128": 4, "language": None},
        ]
    )

    filtered = filter_python_goids(table)

    expect_equal(filtered.column("goid_h128").to_pylist(), [1, 4])


def test_filter_symbol_occurrences() -> None:
    """filter_symbol_occurrences should keep valid occurrence rows."""
    table = pa.Table.from_pylist(
        [
            {"symbol": "sym", "rel_path": "a.py", "start_line": 1},
            {"symbol": "", "rel_path": "b.py", "start_line": 2},
            {"symbol": "sym", "rel_path": "c.py", "start_line": None},
        ]
    )

    filtered = filter_symbol_occurrences(table)

    expect_equal(filtered.column("rel_path").to_pylist(), ["a.py"])


def test_filter_goids_with_spans() -> None:
    """filter_goids_with_spans should keep valid span rows."""
    table = pa.Table.from_pylist(
        [
            {"rel_path": "a.py", "goid_h128": 1, "start_line": 0},
            {"rel_path": "b.py", "goid_h128": None, "start_line": 2},
            {"rel_path": "", "goid_h128": 3, "start_line": 3},
        ]
    )

    filtered = filter_goids_with_spans(table)

    expect_equal(filtered.column("goid_h128").to_pylist(), [1])


def test_filter_function_ast_nodes() -> None:
    """filter_function_ast_nodes should keep function nodes with valid metadata."""
    table = pa.Table.from_pylist(
        [
            {"path": "a.py", "node_type": "FunctionDef", "name": "func", "lineno": 1},
            {"path": "b.py", "node_type": "ClassDef", "name": "Cls", "lineno": 2},
            {"path": "", "node_type": "AsyncFunctionDef", "name": "run", "lineno": 3},
            {"path": "c.py", "node_type": "AsyncFunctionDef", "name": "", "lineno": 4},
        ]
    )

    filtered = filter_function_ast_nodes(table)

    expect_equal(filtered.column("path").to_pylist(), ["a.py"])
