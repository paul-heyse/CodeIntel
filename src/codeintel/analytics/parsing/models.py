"""Shared parsing dataclasses for analytics subsystems."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeintel.ingestion.ast_utils import AstSpanIndex


@dataclass(frozen=True)
class SourceSpan:
    """Source span in (path, [start_line, end_line], [start_col, end_col])."""

    path: Path
    start_line: int
    start_col: int
    end_line: int
    end_col: int


@dataclass(frozen=True)
class ParsedFunction:
    """
    Language-agnostic parsed function representation consumed by analytics.

    Attributes mirror the generic shape expected by span resolution and
    validation, while keeping the AST payload flexible for language-specific
    nodes.
    """

    path: Path
    qualname: str
    function_goid_h128: int | None
    span: SourceSpan
    ast: Any
    docstring: str | None
    param_annotations: Mapping[str, Any]
    return_annotation: Any | None
    param_any_flags: Mapping[str, bool]
    return_is_any: bool


@dataclass(frozen=True)
class ParsedModule:
    """Parsed module contents and extracted functions."""

    path: Path
    source: str
    lines: Sequence[str]
    module_ast: ast.AST
    span_index: AstSpanIndex
    functions: Sequence[ParsedFunction]
