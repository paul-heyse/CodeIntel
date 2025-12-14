"""Shared AST utilities for capture and span lookup.

This module provides utilities for parsing Python modules and working with ASTs.
For span indexing, import ``AstSpanIndex`` directly from ``codeintel.core.parsing``.
"""

from __future__ import annotations

import ast
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def parse_python_module(path: Path) -> tuple[list[str], ast.AST] | None:
    """
    Parse a Python module into an AST, returning source lines and the tree.

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
    """
    Parse a Python file and return lines, AST, and duration seconds.

    Returns
    -------
    tuple[list[str], ast.AST, float] | None
        Lines, AST, and parse duration; None on failure.
    """
    start = time.perf_counter()
    parsed = parse_python_module(path)
    if parsed is None:
        return None
    lines, tree = parsed
    return lines, tree, time.perf_counter() - start
