"""Validate AST visitor captures decorator spans."""

from __future__ import annotations

import ast
from textwrap import dedent

import pytest

from codeintel.config.schemas.ingestion_sql import AST_NODES_COLUMNS
from codeintel.ingestion.py_ast_extract import AstVisitor


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
