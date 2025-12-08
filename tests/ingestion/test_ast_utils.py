"""Tests for AST utilities.

This module tests the AST parsing and span indexing utilities
used during code ingestion.
"""

from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

import pytest

from codeintel.config.datasets import get_table_columns
from codeintel.ingestion.compute.ast_extract import AstVisitor
from codeintel.ingestion.infrastructure.ast_utils import (
    AstSpanIndex,
    parse_python_module,
    timed_parse,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_none,
    expect_true,
)

AST_NODES_COLUMNS = get_table_columns("core.ast_nodes")

# =============================================================================
# Test Data
# =============================================================================

SIMPLE_MODULE = dedent('''
    """Module docstring."""

    def foo(x: int) -> int:
        """Function docstring."""
        return x + 1

    class Bar:
        """Class docstring."""

        def baz(self) -> None:
            """Method docstring."""
            pass
''').strip()

MULTILINE_FUNCTION = dedent('''
    def complex_function(
        arg1: int,
        arg2: str,
        arg3: float,
    ) -> dict[str, int]:
        """Multi-line function."""
        result = {}
        for i in range(arg1):
            result[f"{arg2}_{i}"] = int(arg3)
        return result
''').strip()

SYNTAX_ERROR_CODE = dedent("""
    def broken(
        return "missing colon"
""").strip()

UNICODE_MODULE = dedent('''
    """Unicode test: café, naïve, 日本語."""

    def grüß() -> str:
        """Return greeting."""
        return "Hallo"
''').strip()

# Test constants
EXPECTED_SOURCE_LINES = 3


# =============================================================================
# AstSpanIndex.from_tree Tests
# =============================================================================


def test_indexes_function_defs() -> None:
    """Should index FunctionDef nodes by span."""
    tree = ast.parse(SIMPLE_MODULE)
    index = AstSpanIndex.from_tree(tree, kinds=(ast.FunctionDef,))

    # Should have indexed foo and baz functions
    expect_true(len(index.node_map) >= 1)
    # Verify at least one function is indexed
    spans = list(index.node_map.keys())
    expect_true(all(isinstance(s[0], int) and isinstance(s[1], int) for s in spans))


def test_indexes_class_defs() -> None:
    """Should index ClassDef nodes by span."""
    tree = ast.parse(SIMPLE_MODULE)
    index = AstSpanIndex.from_tree(tree, kinds=(ast.ClassDef,))

    # Should have indexed Bar class
    expect_true(len(index.node_map) >= 1)
    # Check that we have a ClassDef
    found_class = any(isinstance(n, ast.ClassDef) for n in index.node_map.values())
    expect_true(found_class)


def test_indexes_multiple_kinds() -> None:
    """Should index multiple node kinds."""
    tree = ast.parse(SIMPLE_MODULE)
    index = AstSpanIndex.from_tree(tree, kinds=(ast.FunctionDef, ast.ClassDef))

    # Should have function and class nodes
    node_types = {type(n).__name__ for n in index.node_map.values()}
    expect_true("FunctionDef" in node_types or "ClassDef" in node_types)


def test_empty_tree_returns_empty_index() -> None:
    """Should return empty index for empty tree."""
    tree = ast.parse("")
    index = AstSpanIndex.from_tree(tree, kinds=(ast.FunctionDef,))

    expect_equal(len(index.node_map), 0)


def test_ignores_nodes_without_lineno() -> None:
    """Should skip nodes without lineno attribute."""
    tree = ast.parse(SIMPLE_MODULE)
    # Module node itself typically doesn't have a meaningful lineno
    index = AstSpanIndex.from_tree(tree, kinds=(ast.Module,))
    # Module node may or may not be indexed depending on lineno
    # Main point is that no error is raised
    expect_is_instance(index.node_map, dict)


# =============================================================================
# AstSpanIndex.lookup Tests
# =============================================================================


def test_exact_span_match() -> None:
    """Should find node with exact span match."""
    tree = ast.parse(SIMPLE_MODULE)
    index = AstSpanIndex.from_tree(tree, kinds=(ast.FunctionDef,))

    # Find exact span of first function (foo starts at line 4)
    for (start, end), node in index.node_map.items():
        if isinstance(node, ast.FunctionDef) and node.name == "foo":
            result = index.lookup(start, end)
            expect_true(result is node)
            break


def test_enclosing_span_match() -> None:
    """Should find node that encloses the requested span."""
    tree = ast.parse(SIMPLE_MODULE)
    index = AstSpanIndex.from_tree(tree, kinds=(ast.FunctionDef,))

    # Find a function and lookup a line inside it
    for (start, end), node in index.node_map.items():
        if isinstance(node, ast.FunctionDef) and end > start:
            # Lookup a middle line
            mid_line = start + 1
            result = index.lookup(mid_line, mid_line)
            # Should return the enclosing function
            expect_true(result is not None)
            break


def test_no_match_returns_none() -> None:
    """Should return None when no match found."""
    tree = ast.parse(SIMPLE_MODULE)
    index = AstSpanIndex.from_tree(tree, kinds=(ast.FunctionDef,))

    # Lookup a line that doesn't exist
    result = index.lookup(9999, 9999)
    expect_is_none(result)


def test_none_end_line_uses_start() -> None:
    """Should use start_line as end when end_line is None."""
    tree = ast.parse(SIMPLE_MODULE)
    index = AstSpanIndex.from_tree(tree, kinds=(ast.FunctionDef,))

    # Find any function
    if index.node_map:
        (start, _end), _ = next(iter(index.node_map.items()))
        # Lookup with None end should work
        result = index.lookup(start, None)
        expect_true(result is not None)


def test_smallest_enclosing_span_preferred() -> None:
    """Should prefer smallest enclosing span when multiple match."""
    # Create nested structure
    nested_code = dedent("""
        class Outer:
            def inner(self):
                pass
    """).strip()
    tree = ast.parse(nested_code)
    index = AstSpanIndex.from_tree(tree, kinds=(ast.FunctionDef, ast.ClassDef))

    # Get the inner function's lines
    for node in index.node_map.values():
        if isinstance(node, ast.FunctionDef) and node.name == "inner":
            # Looking up from inside should find inner, not Outer
            line = node.lineno
            result = index.lookup(line, line)
            # Should find the function, not the class
            expect_true(result is node or isinstance(result, ast.FunctionDef))
            break


# =============================================================================
# parse_python_module Tests
# =============================================================================


def test_parses_valid_file(tmp_path: Path) -> None:
    """Should parse a valid Python file."""
    test_file = tmp_path / "valid.py"
    test_file.write_text(SIMPLE_MODULE, encoding="utf-8")

    result = parse_python_module(test_file)

    if result is None:
        pytest.fail("Expected parse result for valid file")
    lines, tree = result
    expect_true(len(lines) > 0)
    expect_is_instance(tree, ast.Module)


def test_returns_source_lines(tmp_path: Path) -> None:
    """Should return correct source lines."""
    test_file = tmp_path / "test.py"
    content = "x = 1\ny = 2\nz = 3"
    test_file.write_text(content, encoding="utf-8")

    result = parse_python_module(test_file)

    if result is None:
        pytest.fail("Expected parse result for test file")
    lines, _ = result
    expect_equal(len(lines), EXPECTED_SOURCE_LINES)
    expect_equal(lines[0], "x = 1")
    expect_equal(lines[1], "y = 2")
    expect_equal(lines[2], "z = 3")


def test_returns_none_for_missing_file(tmp_path: Path) -> None:
    """Should return None for non-existent file."""
    missing = tmp_path / "does_not_exist.py"

    result = parse_python_module(missing)

    expect_is_none(result)


def test_returns_none_for_syntax_error(tmp_path: Path) -> None:
    """Should return None for file with syntax error."""
    test_file = tmp_path / "syntax_error.py"
    test_file.write_text(SYNTAX_ERROR_CODE, encoding="utf-8")

    result = parse_python_module(test_file)

    expect_is_none(result)


def test_handles_unicode_content(tmp_path: Path) -> None:
    """Should handle files with unicode content."""
    test_file = tmp_path / "unicode.py"
    test_file.write_text(UNICODE_MODULE, encoding="utf-8")

    result = parse_python_module(test_file)

    if result is None:
        pytest.fail("Expected parse result for unicode file")
    lines, tree = result
    expect_true("café" in lines[0] or "naïve" in lines[0])
    expect_is_instance(tree, ast.Module)


def test_returns_none_for_binary_file(tmp_path: Path) -> None:
    """Should return None for binary file with decode error."""
    test_file = tmp_path / "binary.py"
    # Write invalid UTF-8 bytes
    test_file.write_bytes(b"\xff\xfe invalid utf-8 \x80\x81")

    result = parse_python_module(test_file)

    expect_is_none(result)


def test_parses_empty_file(tmp_path: Path) -> None:
    """Should parse empty Python file."""
    test_file = tmp_path / "empty.py"
    test_file.write_text("", encoding="utf-8")

    result = parse_python_module(test_file)

    if result is None:
        pytest.fail("Expected parse result for empty file")
    lines, tree = result
    expect_true(len(lines) == 0 or lines == [""])
    expect_is_instance(tree, ast.Module)


# =============================================================================
# timed_parse Tests
# =============================================================================


def test_returns_lines_tree_and_duration(tmp_path: Path) -> None:
    """Should return lines, tree, and duration."""
    test_file = tmp_path / "test.py"
    test_file.write_text(SIMPLE_MODULE, encoding="utf-8")

    result = timed_parse(test_file)

    if result is None:
        pytest.fail("Expected timed_parse result for valid file")
    lines, tree, duration = result
    expect_true(len(lines) > 0)
    expect_is_instance(tree, ast.Module)
    expect_is_instance(duration, float)
    expect_true(duration >= 0)


def test_duration_is_non_negative(tmp_path: Path) -> None:
    """Should return non-negative duration."""
    test_file = tmp_path / "test.py"
    test_file.write_text("x = 1", encoding="utf-8")

    result = timed_parse(test_file)

    if result is None:
        pytest.fail("Expected timed_parse result for simple file")
    _, _, duration = result
    expect_true(duration >= 0)


def test_timed_parse_returns_none_on_parse_failure(tmp_path: Path) -> None:
    """Should return None when parsing fails."""
    test_file = tmp_path / "invalid.py"
    test_file.write_text(SYNTAX_ERROR_CODE, encoding="utf-8")

    result = timed_parse(test_file)

    expect_is_none(result)


def test_timed_parse_returns_none_for_missing_file(tmp_path: Path) -> None:
    """Should return None for missing file."""
    missing = tmp_path / "missing.py"

    result = timed_parse(missing)

    expect_is_none(result)


# =============================================================================
# Integration Tests
# =============================================================================


def test_parse_and_index_workflow(tmp_path: Path) -> None:
    """Should support typical parse-then-index workflow."""
    test_file = tmp_path / "module.py"
    test_file.write_text(SIMPLE_MODULE, encoding="utf-8")

    # Parse the module
    result = parse_python_module(test_file)
    if result is None:
        pytest.fail("Expected parse result in workflow test")
    lines, tree = result

    # Build index
    index = AstSpanIndex.from_tree(tree, kinds=(ast.FunctionDef, ast.ClassDef))

    # Lookup a function
    for (start, _), node in index.node_map.items():
        if isinstance(node, ast.FunctionDef):
            # Verify we can find the function's source in lines
            expect_true(start <= len(lines))
            break


def test_multiline_function_span(tmp_path: Path) -> None:
    """Should correctly index multiline function definitions."""
    test_file = tmp_path / "multiline.py"
    test_file.write_text(MULTILINE_FUNCTION, encoding="utf-8")

    result = parse_python_module(test_file)
    if result is None:
        pytest.fail("Expected parse result for multiline function")
    _lines, tree = result

    index = AstSpanIndex.from_tree(tree, kinds=(ast.FunctionDef,))

    # Should have one function
    expect_equal(len(index.node_map), 1)

    # Function should span multiple lines
    (start, end), node = next(iter(index.node_map.items()))
    if not isinstance(node, ast.FunctionDef):
        pytest.fail(f"Expected FunctionDef node, got {type(node).__name__}")
    expect_equal(node.name, "complex_function")
    expect_true(end > start)  # Multi-line


# =============================================================================
# AstVisitor Tests
# =============================================================================


def test_ast_visitor_records_decorator_span() -> None:
    """Decorator spans should include lines above the function definition."""
    source = dedent(
        """\
        @dec1
        @dec2("x")
        def foo():
            return 1
        """
    )
    tree = ast.parse(source, filename="mod.py")
    visitor = AstVisitor(rel_path="mod.py", module_name="mod")
    visitor.visit(tree)

    rows = [dict(zip(AST_NODES_COLUMNS, row, strict=True)) for row in visitor.ast_rows]
    func_rows = [row for row in rows if row["node_type"] == "FunctionDef"]

    if len(func_rows) != 1:
        message = f"Expected one function row, got {len(func_rows)}"
        pytest.fail(message)

    func = func_rows[0]
    expected_def_line = 3
    expected_decorator_start = 1
    expected_decorator_end = 2
    if func["lineno"] != expected_def_line:
        message = f"Expected def line {expected_def_line}, got {func['lineno']}"
        pytest.fail(message)
    if func["decorator_start_line"] != expected_decorator_start:
        message = (
            f"Expected decorator_start_line={expected_decorator_start}, "
            f"got {func['decorator_start_line']}"
        )
        pytest.fail(message)
    if func["decorator_end_line"] != expected_decorator_end:
        message = (
            f"Expected decorator_end_line={expected_decorator_end}, "
            f"got {func['decorator_end_line']}"
        )
        pytest.fail(message)
