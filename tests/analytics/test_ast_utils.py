"""Unit tests for analytics AST utilities."""

from __future__ import annotations

import ast

import pytest

from codeintel.analytics.utilities import (
    call_name,
    literal_bool,
    literal_int_sequence,
    literal_value,
    resolve_call_target,
    safe_unparse,
    snippet_from_lines,
)


def test_call_name_resolves_attribute_chain() -> None:
    """call_name should stitch dotted attribute chains."""
    node = ast.parse("pkg.module.func()", mode="eval").body
    if call_name(node) != "pkg.module.func":
        pytest.fail("Failed to resolve nested attribute chain")


def test_resolve_call_target_uses_alias_map() -> None:
    """resolve_call_target applies alias mapping to the library root."""
    node = ast.parse("alias()", mode="eval").body
    if not isinstance(node, ast.Call):
        pytest.fail("Expected parsed expression to be an ast.Call")
    target = resolve_call_target(node.func, {"alias": "library"})
    if target.library != "library":
        pytest.fail("Library alias should be rewritten")
    if target.attribute != "alias" or target.base != "alias":
        pytest.fail("Attribute/base names were not preserved")


def test_literal_value_coercions() -> None:
    """literal_value should coerce nested literals."""
    node = ast.parse("[-1, (1, 2), {'a': True}]", mode="eval").body
    value = literal_value(node)
    if value != [-1, (1, 2), {"a": True}]:
        pytest.fail("Nested literal coercion failed")


def test_safe_unparse_survives_invalid_nodes() -> None:
    """safe_unparse should never raise and should handle valid nodes."""
    if safe_unparse(ast.AST()):
        pytest.fail("Invalid node should yield empty string")
    parsed = safe_unparse(ast.parse("x + 1", mode="eval").body)
    if parsed != "x + 1":
        pytest.fail("Valid nodes should unparse to their source")


def test_snippet_from_lines_includes_range() -> None:
    """snippet_from_lines should honor the provided line span."""
    lines = ["first", " second", "third"]
    snippet = snippet_from_lines(lines, lineno=2, end_lineno=3)
    if snippet != " second\nthird":
        pytest.fail("Snippet did not include the expected lines")


def test_literal_bool_extracts_boolean_values() -> None:
    """literal_bool should return booleans for literal values."""
    true_node = ast.parse("True", mode="eval").body
    false_node = ast.parse("False", mode="eval").body
    non_bool = ast.parse("1", mode="eval").body

    if literal_bool(true_node) is not True:
        pytest.fail("Expected literal_bool to extract True")
    if literal_bool(false_node) is not False:
        pytest.fail("Expected literal_bool to extract False")
    if literal_bool(non_bool) is not None:
        pytest.fail("Expected non-bool to return None")


def test_literal_int_sequence_extracts_ints() -> None:
    """literal_int_sequence should return ints for list/tuple literals."""
    list_node = ast.parse("[1, 2]", mode="eval").body
    tuple_node = ast.parse("(3, 4)", mode="eval").body
    mixed_node = ast.parse("[1, 'a']", mode="eval").body
    non_sequence = ast.parse("42", mode="eval").body

    if literal_int_sequence(list_node) != [1, 2]:
        pytest.fail("Expected list of ints from list literal")
    if literal_int_sequence(tuple_node) != [3, 4]:
        pytest.fail("Expected list of ints from tuple literal")
    if literal_int_sequence(mixed_node) is not None:
        pytest.fail("Expected None for mixed-type sequence")
    if literal_int_sequence(non_sequence) is not None:
        pytest.fail("Expected None for non-sequence literal")
