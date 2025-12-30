"""Shared AST parsing and literal extraction helpers."""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def parse_python_module(path: Path) -> tuple[list[str], ast.AST] | None:
    """Parse a Python module into lines and an AST.

    Returns
    -------
    tuple[list[str], ast.AST] | None
        Lines and parsed AST when successful; None when the file is missing or invalid.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError):
        return None

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return None

    return source.splitlines(), tree


def timed_parse(path: Path) -> tuple[list[str], ast.AST, float] | None:
    """Parse a Python file and return lines, AST, and duration seconds.

    Returns
    -------
    tuple[list[str], ast.AST, float] | None
        Parsed lines, AST, and elapsed seconds; None when parsing fails.
    """
    start = time.perf_counter()
    parsed = parse_python_module(path)
    if parsed is None:
        return None
    lines, tree = parsed
    return lines, tree, time.perf_counter() - start


def literal_value(node: ast.AST | None) -> object:
    """Extract a Python literal value from an AST node, falling back to None.

    Returns
    -------
    object
        Extracted literal value, or None when not a literal.
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
    """Extract string literal content when available.

    Returns
    -------
    str | None
        String literal value, or None when not a string literal.
    """
    value = literal_value(node)
    return str(value) if isinstance(value, str) else None


def literal_int(node: ast.AST | None) -> int | None:
    """Extract integer literal content when available.

    Returns
    -------
    int | None
        Integer literal value, or None when not an int literal.
    """
    value = literal_value(node)
    return int(value) if isinstance(value, int) else None


def literal_bool(node: ast.AST | None) -> bool | None:
    """Extract boolean literal content when available.

    Returns
    -------
    bool | None
        Boolean literal value, or None when not a bool literal.
    """
    value = literal_value(node)
    return bool(value) if isinstance(value, bool) else None


def literal_int_sequence(node: ast.AST | None) -> list[int] | None:
    """Extract a sequence of integer literals when available.

    Returns
    -------
    list[int] | None
        Integer literal sequence, or None when parsing fails.
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
    """Best-effort unparse that never raises.

    Returns
    -------
    str
        Unparsed source, or an empty string on failure.
    """
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except (ValueError, TypeError, AttributeError, RecursionError):
        return ""


def snippet_from_lines(lines: Iterable[str], lineno: int | None, end_lineno: int | None) -> str:
    """Return a trimmed snippet from lines using 1-based line numbers.

    Returns
    -------
    str
        Trimmed snippet, or an empty string when no snippet is available.
    """
    if lineno is None:
        return ""
    start_index = max(lineno - 1, 0)
    end_index = end_lineno if end_lineno is not None else lineno
    if isinstance(lines, list):
        sequence = lines
    else:
        try:
            sequence = list(lines)
        except (TypeError, ValueError):
            return ""
    try:
        slice_lines = sequence[start_index:end_index]
    except (TypeError, IndexError):
        return ""
    return "\n".join(line.rstrip("\n") for line in slice_lines)


__all__ = [
    "literal_bool",
    "literal_int",
    "literal_int_sequence",
    "literal_str",
    "literal_value",
    "parse_python_module",
    "safe_unparse",
    "snippet_from_lines",
    "timed_parse",
]
