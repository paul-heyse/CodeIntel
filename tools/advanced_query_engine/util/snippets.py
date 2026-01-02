"""Snippet extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass

from tools.advanced_query_engine.contracts import EvidenceSnippet, Span
from tools.advanced_query_engine.util.line_index import LineIndex


@dataclass(frozen=True)
class SnippetConfig:
    """Configuration for snippet extraction."""

    before_lines: int = 1
    after_lines: int = 1
    max_excerpt_bytes: int = 400


@dataclass(frozen=True)
class SnippetRequest:
    """Inputs for snippet extraction."""

    source: bytes
    span: Span
    config: SnippetConfig
    line_index: LineIndex | None = None


def build_snippet(request: SnippetRequest) -> EvidenceSnippet:
    """Build a snippet with context lines for a byte span.

    Parameters
    ----------
    request:
        Snippet request payload.

    Returns
    -------
    EvidenceSnippet
        Evidence snippet with contextual lines.
    """
    index = request.line_index or LineIndex.build(request.source)
    start = request.span.start_byte
    end = request.span.end_byte
    span_with_lines = Span(
        path=request.span.path,
        start_byte=start,
        end_byte=end,
        **index.span_to_range(start, end),
    )
    excerpt_bytes = request.source[start:end]
    if len(excerpt_bytes) > request.config.max_excerpt_bytes:
        excerpt_bytes = excerpt_bytes[: request.config.max_excerpt_bytes] + b"..."
    excerpt = excerpt_bytes.decode("utf-8", errors="replace")
    start_line, _ = index.line_col(start)
    end_line, _ = index.line_col(end)
    before: list[str] = []
    after: list[str] = []
    for line_no in range(max(1, start_line - request.config.before_lines), start_line):
        begin = index.line_start_byte(line_no)
        finish = index.line_start_byte(line_no + 1)
        text = request.source[begin:finish].decode("utf-8", errors="replace")
        before.append(text.rstrip("\n"))
    for line_no in range(end_line + 1, end_line + request.config.after_lines + 1):
        if line_no > len(index.line_starts):
            break
        begin = index.line_start_byte(line_no)
        finish = index.line_start_byte(line_no + 1)
        text = request.source[begin:finish].decode("utf-8", errors="replace")
        after.append(text.rstrip("\n"))
    return EvidenceSnippet(
        span=span_with_lines, text=excerpt, context_before=before, context_after=after
    )


__all__ = ["SnippetConfig", "SnippetRequest", "build_snippet"]
