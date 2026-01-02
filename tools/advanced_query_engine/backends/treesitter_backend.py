"""Tree-sitter backend for structural query packs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import (
    LANGUAGE_VERSION,
    MIN_COMPATIBLE_LANGUAGE_VERSION,
    Language,
    Parser,
    Query,
    QueryCursor,
    Tree,
)
from tree_sitter_language_pack import get_language

from tools.advanced_query_engine.contracts import Span
from tools.advanced_query_engine.util.line_index import LineIndex


@dataclass(frozen=True)
class TreeSitterCapture:
    """Capture record from a tree-sitter query."""

    pack_id: str
    capture_name: str
    node_type: str
    span: Span
    text_preview: str | None


@dataclass(frozen=True)
class TreeSitterResult:
    """Result of tree-sitter query execution."""

    captures: list[TreeSitterCapture]
    parse_ok: bool
    warnings: list[str]


@dataclass(frozen=True)
class TreeSitterQuery:
    """Single tree-sitter query pack."""

    pack_id: str
    query_text: str


@dataclass(frozen=True)
class TreeSitterRequest:
    """Request for executing tree-sitter query packs."""

    language: str
    source_bytes: bytes
    path: str
    queries: Iterable[TreeSitterQuery]
    match_limit: int
    preview_limit: int
    parsed: TreeSitterParsed | None = None


@dataclass(frozen=True)
class TreeSitterParsed:
    """Cached parse data for a source file."""

    tree: Tree
    line_index: LineIndex


_LANGUAGE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "python": (".py", ".pyi"),
}


def language_for_path(path: Path) -> str | None:
    """Return language name for a given file path suffix.

    Returns
    -------
    str | None
        Language identifier when supported.
    """
    suffix = path.suffix.lower()
    for language, extensions in _LANGUAGE_EXTENSIONS.items():
        if suffix in extensions:
            return language
    return None


def _assert_language_abi(language: Language) -> None:
    if not (MIN_COMPATIBLE_LANGUAGE_VERSION <= language.abi_version <= LANGUAGE_VERSION):
        msg = (
            "Tree-sitter language ABI not supported: "
            f"{language.abi_version} "
            f"(expected {MIN_COMPATIBLE_LANGUAGE_VERSION}-{LANGUAGE_VERSION})"
        )
        raise RuntimeError(msg)


def load_language(language: str) -> Language:
    """Load a tree-sitter language with ABI checks.

    Returns
    -------
    Language
        Loaded tree-sitter language.
    """
    ts_language = get_language(language)
    _assert_language_abi(ts_language)
    return ts_language


def load_parser(language: str) -> Parser:
    """Return a parser configured for the given language.

    Returns
    -------
    Parser
        Configured parser.
    """
    return Parser(load_language(language))


def parse_tree_sitter_source(language: str, source_bytes: bytes) -> TreeSitterParsed:
    """Parse source bytes into a cached tree-sitter representation.

    Returns
    -------
    TreeSitterParsed
        Parsed tree and line index.
    """
    parser = load_parser(language)
    tree = parser.parse(source_bytes)
    line_index = LineIndex.build(source_bytes)
    return TreeSitterParsed(tree=tree, line_index=line_index)


def run_query_packs(request: TreeSitterRequest) -> TreeSitterResult:
    """Run multiple query packs and aggregate captures.

    Returns
    -------
    TreeSitterResult
        Aggregated query results.
    """
    combined: list[TreeSitterCapture] = []
    warnings: list[str] = []
    parse_ok = True
    parsed = request.parsed or parse_tree_sitter_source(request.language, request.source_bytes)
    for query in request.queries:
        result = _run_query_text(parsed, request, query)
        combined.extend(result.captures)
        warnings.extend(result.warnings)
        if not result.parse_ok:
            parse_ok = False
    combined.sort(key=lambda item: (item.span.start_byte, item.span.end_byte, item.capture_name))
    return TreeSitterResult(captures=combined, parse_ok=parse_ok, warnings=warnings)


def _run_query_text(
    parsed: TreeSitterParsed, request: TreeSitterRequest, query: TreeSitterQuery
) -> TreeSitterResult:
    warnings: list[str] = []
    captures: list[TreeSitterCapture] = []

    try:
        ts_query = Query(parsed.tree.language, query.query_text)
    except (ValueError, RuntimeError) as exc:
        warnings.append(f"Tree-sitter query '{query.pack_id}' failed: {exc}")
        return TreeSitterResult(captures=captures, parse_ok=False, warnings=warnings)

    cursor = QueryCursor(ts_query, match_limit=request.match_limit)
    capture_map = cursor.captures(parsed.tree.root_node)
    line_index = parsed.line_index

    for capture_name, nodes in capture_map.items():
        for node in nodes:
            span = Span(
                path=request.path,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                **line_index.span_to_range(node.start_byte, node.end_byte),
            )
            text_preview = _slice_preview(
                request.source_bytes,
                start=node.start_byte,
                end=node.end_byte,
                preview_limit=request.preview_limit,
            )
            captures.append(
                TreeSitterCapture(
                    pack_id=query.pack_id,
                    capture_name=capture_name,
                    node_type=node.type,
                    span=span,
                    text_preview=text_preview,
                )
            )

    captures.sort(key=lambda item: (item.span.start_byte, item.span.end_byte, item.capture_name))
    return TreeSitterResult(captures=captures, parse_ok=True, warnings=warnings)


def _slice_preview(source: bytes, *, start: int, end: int, preview_limit: int) -> str | None:
    start = max(start, 0)
    end = max(end, start)
    if start >= len(source):
        return None
    clipped_end = min(end, len(source))
    payload = source[start:clipped_end]
    if not payload:
        return None
    return payload[:preview_limit].decode("utf-8", errors="replace")


__all__ = [
    "TreeSitterCapture",
    "TreeSitterParsed",
    "TreeSitterQuery",
    "TreeSitterRequest",
    "TreeSitterResult",
    "language_for_path",
    "load_language",
    "parse_tree_sitter_source",
    "run_query_packs",
]
