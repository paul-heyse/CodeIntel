"""Shared AST utilities for analytics pipelines."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.parsing.ast_utils import (
    literal_bool,
    literal_int,
    literal_int_sequence,
    literal_str,
    literal_value,
    safe_unparse,
    snippet_from_lines,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class CallTarget:
    """Resolved call target split into library, attribute, and base name."""

    library: str | None
    attribute: str | None
    base: str | None


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


def _base_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        dotted = call_name(node)
        return dotted.split(".")[0] if dotted else None
    return None


__all__ = [
    "CallTarget",
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
