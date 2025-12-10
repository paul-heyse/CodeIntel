"""Tests for AST utilities.

This module tests the AST parsing and span indexing utilities
used during code ingestion.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

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
from tests._helpers.ingestion_samples import (
    DECORATED_FUNCTION,
    MULTILINE_FUNCTION,
    NESTED_CLASS_FUNCTION,
    SIMPLE_MODULE,
    SYNTAX_ERROR_CODE,
    UNICODE_MODULE,
)

AST_NODES_COLUMNS = get_table_columns("core.ast_nodes")

# Test constants
EXPECTED_SOURCE_LINES = 3
INDEX_CASES = [
    (("functions", (ast.FunctionDef,), ast.FunctionDef), "function defs"),
    (("classes", (ast.ClassDef,), ast.ClassDef), "class defs"),
    (("both", (ast.FunctionDef, ast.ClassDef), (ast.FunctionDef, ast.ClassDef)), "mixed defs"),
]


# =============================================================================
# AstSpanIndex.from_tree Tests
# =============================================================================


def _expect_index_has_kind(index: AstSpanIndex, kinds: tuple[type[ast.AST], ...]) -> None:
    expect_true(index.node_map)
    expect_true(
        any(isinstance(node, kinds) for node in index.node_map.values()),
        message="Expected at least one node of requested kinds",
    )


def _build_index_for_functions() -> AstSpanIndex:
    tree = ast.parse(SIMPLE_MODULE)
    return AstSpanIndex.from_tree(tree, kinds=(ast.FunctionDef,))


def _first_function_span(index: AstSpanIndex) -> tuple[tuple[int, int], ast.FunctionDef]:
    for span, node in index.node_map.items():
        if isinstance(node, ast.FunctionDef):
            return span, node
    message = "Expected at least one FunctionDef in index"
    raise AssertionError(message)


@pytest.mark.parametrize(
    ("case_kinds", "label"),
    [(c[0][1], c[1]) for c in INDEX_CASES],
)
def test_indexes_requested_kinds(case_kinds: tuple[type[ast.AST], ...], label: str) -> None:
    """Should index requested node kinds by span."""
    tree = ast.parse(SIMPLE_MODULE)
    index = AstSpanIndex.from_tree(tree, kinds=case_kinds)

    _expect_index_has_kind(index, case_kinds)


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


LookupSpan = tuple[int, int | None]
LookupCase = tuple[str, Callable[[LookupSpan], LookupSpan], str]

LOOKUP_CASES: tuple[LookupCase, ...] = (
    ("exact_span_match", lambda span: span, "exact"),
    ("enclosing_span_match", lambda span: (span[0] + 1, span[0] + 1), "enclosing"),
    ("no_match_returns_none", lambda _span: (9999, 9999), "missing"),
    ("none_end_line_uses_start", lambda span: (span[0], None), "none_end"),
)


@pytest.mark.parametrize(
    ("_case_name", "build_span", "mode"),
    LOOKUP_CASES,
)
def test_lookup_cases(
    _case_name: str, build_span: Callable[[LookupSpan], LookupSpan], mode: str
) -> None:
    """Lookup should handle exact, enclosing, missing, and None end spans."""
    index = _build_index_for_functions()
    (span_start, span_end), func = _first_function_span(index)

    start, end = build_span((span_start, span_end))

    result = index.lookup(start, end)

    if mode == "exact":
        expect_true(result is func)
    elif mode == "enclosing":
        expect_true(result is not None)
    elif mode == "missing":
        expect_is_none(result)
    else:
        expect_true(result is not None)


def test_smallest_enclosing_span_preferred() -> None:
    """Should prefer smallest enclosing span when multiple match."""
    tree = ast.parse(NESTED_CLASS_FUNCTION)
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
    source = DECORATED_FUNCTION
    tree = ast.parse(source, filename="mod.py")
    visitor = AstVisitor(rel_path="mod.py", module_name="mod")
    visitor.visit(tree)

    rows = [dict(zip(AST_NODES_COLUMNS, row, strict=True)) for row in visitor.ast_rows]
    func_rows = [row for row in rows if row["node_type"] == "FunctionDef"]

    expect_equal(len(func_rows), 1)
    func = func_rows[0]
    expect_equal(func["lineno"], 3)
    expect_equal(func["decorator_start_line"], 1)
    expect_equal(func["decorator_end_line"], 2)
