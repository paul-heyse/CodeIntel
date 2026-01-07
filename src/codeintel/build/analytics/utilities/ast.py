"""Shared AST utilities for analytics pipelines."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.serialization.payload import decode_payload

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


@dataclass(frozen=True)
class CallTarget:
    """Resolved call target split into library, attribute, and base name."""

    library: str | None
    attribute: str | None
    base: str | None


@dataclass(frozen=True, slots=True)
class RowDecoder:
    """Decode msgspec-backed payload columns from row mappings."""

    columns: Sequence[str]

    def decode(self, row: Mapping[str, object]) -> dict[str, object]:
        """Return a copy of the row with decoded payload columns.

        Returns
        -------
        dict[str, object]
            Row mapping with requested columns decoded from payloads.
        """
        decoded = dict(row)
        for name in self.columns:
            decoded[name] = decode_payload(row.get(name))
        return decoded


def call_name(node: ast.AST | None) -> str | None:
    """
    Return dotted name for Name/Attribute chains, or None when unknown.

    Returns
    -------
    str | None
        Dotted path for names/attributes, otherwise None.
    """
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return call_name(node.func)
    return None


def resolve_call_target(func: ast.AST, alias_map: Mapping[str, str] | None = None) -> CallTarget:
    """
    Resolve a call target into (library, attribute, base).

    Parameters
    ----------
    func : ast.AST
        AST node representing the callable being resolved.
    alias_map : dict[str, str] | None
        Optional mapping of local alias to library root for import rewrites.

    Returns
    -------
    CallTarget
        Structured target with resolved library root, attribute, and base name.
    """
    alias_map = alias_map or {}
    base_name = _base_name(func)
    library = alias_map.get(base_name, base_name) if base_name is not None else None
    attr = None
    base = None
    if isinstance(func, ast.Attribute):
        attr = func.attr
        base = call_name(func.value)
    elif isinstance(func, ast.Name):
        attr = func.id
        base = func.id
    return CallTarget(library=library, attribute=attr, base=base)


def literal_value(node: ast.AST | None) -> object:
    """
    Extract a Python literal value from an AST node, falling back to None.

    Returns
    -------
    object
        Parsed literal value or None when the node is not a literal.
    """
    if node is None:
        return None
    result: object | None = None
    if isinstance(node, ast.Constant):
        result = node.value
    elif (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
    ):
        value = node.operand.value
        if isinstance(value, (int, float)):
            result = -value
    elif isinstance(node, ast.List):
        result = [literal_value(elt) for elt in node.elts]
    elif isinstance(node, ast.Tuple):
        result = tuple(literal_value(elt) for elt in node.elts)
    elif isinstance(node, ast.Dict):
        result = {
            literal_value(k): literal_value(v) for k, v in zip(node.keys, node.values, strict=True)
        }
    return result


def literal_str(node: ast.AST | None) -> str | None:
    """
    Extract string literal content when available.

    Returns
    -------
    str | None
        String value when the node is a literal string.
    """
    value = literal_value(node)
    return str(value) if isinstance(value, str) else None


def literal_int(node: ast.AST | None) -> int | None:
    """
    Extract integer literal content when available.

    Parameters
    ----------
    node : ast.AST | None
        AST node representing the candidate literal.

    Returns
    -------
    int | None
        Integer value when the node is an int literal (including bool), otherwise None.
    """
    value = literal_value(node)
    return int(value) if isinstance(value, int) else None


def literal_bool(node: ast.AST | None) -> bool | None:
    """
    Extract boolean literal content when available.

    Parameters
    ----------
    node : ast.AST | None
        AST node representing the candidate literal.

    Returns
    -------
    bool | None
        Boolean value when the node is a bool literal, otherwise None.
    """
    value = literal_value(node)
    return bool(value) if isinstance(value, bool) else None


def literal_int_sequence(node: ast.AST | None) -> list[int] | None:
    """
    Extract a sequence of integer literals when available.

    Parameters
    ----------
    node : ast.AST | None
        AST node that may wrap a list or tuple literal.

    Returns
    -------
    list[int] | None
        Integer sequence when every element is an int literal; otherwise None.
    """
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    ints: list[int] = []
    for elt in node.elts:
        value = literal_int(elt)
        if value is None:
            return None
        ints.append(value)
    return ints


def safe_unparse(node: ast.AST | None) -> str:
    """
    Best-effort unparse that never raises.

    Returns
    -------
    str
        Unparsed source or empty string on failure.
    """
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except (ValueError, TypeError, AttributeError, RecursionError):
        return ""


def snippet_from_lines(lines: Iterable[str], lineno: int | None, end_lineno: int | None) -> str:
    """
    Return a trimmed snippet from lines using 1-based line numbers.

    Returns
    -------
    str
        Joined snippet; empty string on failure.
    """
    if lineno is None:
        return ""
    start_index = max(lineno - 1, 0)
    end_index = end_lineno if end_lineno is not None else lineno
    sequence: Sequence[str]
    if isinstance(lines, Sequence):
        sequence = lines
    else:
        try:
            sequence = tuple(lines)
        except (TypeError, ValueError):
            return ""
    try:
        slice_lines = sequence[start_index:end_index]
    except (TypeError, IndexError):
        return ""
    return "\n".join(line.rstrip("\n") for line in slice_lines)


def _base_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        dotted = call_name(node)
        return dotted.split(".")[0] if dotted else None
    return None


__all__ = [
    "CallTarget",
    "RowDecoder",
    "call_name",
    "literal_bool",
    "literal_int",
    "literal_int_sequence",
    "literal_str",
    "literal_value",
    "resolve_call_target",
    "safe_unparse",
    "snippet_from_lines",
]
