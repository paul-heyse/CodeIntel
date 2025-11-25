"""Unit tests for analytics AST utilities."""

from __future__ import annotations

import ast

import pytest

from codeintel.analytics import ast_utils


def test_call_name_resolves_attribute_chain() -> None:
    """call_name should stitch dotted attribute chains."""
    node = ast.parse("pkg.module.func()", mode="eval").body
    if ast_utils.call_name(node) != "pkg.module.func":
        pytest.fail("Failed to resolve nested attribute chain")


def test_resolve_call_target_uses_alias_map() -> None:
    """resolve_call_target applies alias mapping to the library root."""
    node = ast.parse("alias()", mode="eval").body
    if not isinstance(node, ast.Call):
        pytest.fail("Expected parsed expression to be an ast.Call")
    target = ast_utils.resolve_call_target(node.func, {"alias": "library"})
    if target.library != "library":
        pytest.fail("Library alias should be rewritten")
    if target.attribute != "alias" or target.base != "alias":
        pytest.fail("Attribute/base names were not preserved")


def test_literal_value_coercions() -> None:
    """literal_value should coerce nested literals."""
    node = ast.parse("[-1, (1, 2), {'a': True}]", mode="eval").body
    value = ast_utils.literal_value(node)
    if value != [-1, (1, 2), {"a": True}]:
        pytest.fail("Nested literal coercion failed")


def test_safe_unparse_survives_invalid_nodes() -> None:
    """safe_unparse should never raise and should handle valid nodes."""
    if ast_utils.safe_unparse(ast.AST()):
        pytest.fail("Invalid node should yield empty string")
    parsed = ast_utils.safe_unparse(ast.parse("x + 1", mode="eval").body)
    if parsed != "x + 1":
        pytest.fail("Valid nodes should unparse to their source")


def test_snippet_from_lines_includes_range() -> None:
    """snippet_from_lines should honor the provided line span."""
    lines = ["first", " second", "third"]
    snippet = ast_utils.snippet_from_lines(lines, lineno=2, end_lineno=3)
    if snippet != " second\nthird":
        pytest.fail("Snippet did not include the expected lines")
