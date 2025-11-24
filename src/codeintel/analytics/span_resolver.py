"""Helpers for resolving AST spans to function nodes."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Literal

from codeintel.ingestion.ast_utils import AstSpanIndex


@dataclass(frozen=True)
class SpanResolution:
    """Outcome of an AST span lookup."""

    node: ast.AST | None
    reason: Literal["ok", "missing_index", "missing_span"]
    detail: str | None = None


def resolve_span(
    index: AstSpanIndex | None,
    start_line: int,
    end_line: int,
) -> SpanResolution:
    """
    Resolve an AST node for a span, returning structured diagnostics.

    Parameters
    ----------
    index:
        Span index built for the parsed file.
    start_line:
        1-based starting line for the GOID span.
    end_line:
        1-based ending line for the GOID span.

    Returns
    -------
    SpanResolution
        Resolution outcome with the matched node when available.
    """
    if index is None:
        return SpanResolution(node=None, reason="missing_index", detail="No span index available")

    node = index.lookup(start_line, end_line)
    if node is None:
        detail = f"Span {start_line}-{end_line}"
        return SpanResolution(node=None, reason="missing_span", detail=detail)

    return SpanResolution(node=node, reason="ok", detail=None)
