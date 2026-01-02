from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .line_index import LineIndex


@dataclass(frozen=True)
class Evidence:
    span: dict
    excerpt: str
    context: dict


def build_evidence(src: bytes, path: str, start: int, end: int, *, before_lines: int = 1, after_lines: int = 1, max_excerpt_bytes: int = 400) -> Evidence:
    li = LineIndex.build(src)
    span = {"path": path, "start_byte": start, "end_byte": end, **li.span_to_range(start, end)}
    excerpt_bytes = src[start:end]
    if len(excerpt_bytes) > max_excerpt_bytes:
        excerpt_bytes = excerpt_bytes[:max_excerpt_bytes] + b"..."  # not reversible; display-only
    excerpt = excerpt_bytes.decode("utf-8", errors="replace")
    context = li.extract_lines_around_span(start, end, before=before_lines, after=after_lines)
    return Evidence(span=span, excerpt=excerpt, context=context)
