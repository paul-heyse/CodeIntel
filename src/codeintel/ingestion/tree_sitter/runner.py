"""Tree-sitter parsing and query execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tree_sitter import Query, QueryCursor

from codeintel.ingestion.tree_sitter.registry import load_language, load_parser, load_query_packs

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from tree_sitter import Node, Tree
    from tree_sitter_language_pack import SupportedLanguage


DEFAULT_MATCH_LIMIT = 10_000
DEFAULT_PREVIEW_LIMIT = 200


@dataclass(frozen=True)
class TreeSitterCapture:
    """Capture metadata emitted from a query pack."""

    query_pack: str
    capture_name: str
    node_type: str
    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    text_preview: str | None
    extras: dict[str, object] | None


@dataclass(frozen=True)
class TreeSitterParseError:
    """Parsed tree-sitter error/missing node metadata."""

    error_type: str
    message: str | None
    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    text_preview: str | None


@dataclass(frozen=True)
class TreeSitterParseResult:
    """Parse result including captures, errors, and warnings."""

    language: SupportedLanguage
    captures: tuple[TreeSitterCapture, ...]
    errors: tuple[TreeSitterParseError, ...]
    parse_ok: bool
    warnings: tuple[str, ...]


def run_tree_sitter(
    *,
    language: SupportedLanguage,
    source_bytes: bytes,
    match_limit: int = DEFAULT_MATCH_LIMIT,
    preview_limit: int = DEFAULT_PREVIEW_LIMIT,
) -> TreeSitterParseResult:
    """Parse source bytes and execute query packs for a language.

    Returns
    -------
    TreeSitterParseResult
        Parsed captures, errors, and warnings.
    """
    parser = load_parser(language)
    ts_language = load_language(language)
    tree = parser.parse(source_bytes)
    errors = tuple(_parse_errors(tree, source_bytes, preview_limit=preview_limit))
    warnings: list[str] = []

    captures: list[TreeSitterCapture] = []
    query_packs = load_query_packs(language)
    if not query_packs:
        warnings.append(f"No tree-sitter query packs registered for language '{language}'.")
    for pack in query_packs:
        try:
            query = Query(ts_language, pack.query_text)
        except (ValueError, RuntimeError) as exc:
            warnings.append(f"Tree-sitter query pack '{pack.name}' failed: {exc}")
            continue
        cursor = QueryCursor(query, match_limit=match_limit)
        for capture_name, nodes in cursor.captures(tree.root_node).items():
            captures.extend(
                [
                    _capture_from_node(
                        node,
                        source_bytes=source_bytes,
                        query_pack=pack.name,
                        capture_name=capture_name,
                        preview_limit=preview_limit,
                    )
                    for node in nodes
                ]
            )

    return TreeSitterParseResult(
        language=language,
        captures=tuple(captures),
        errors=errors,
        parse_ok=not errors,
        warnings=tuple(warnings),
    )


def _parse_errors(
    tree: Tree,
    source_bytes: bytes,
    *,
    preview_limit: int,
) -> Iterable[TreeSitterParseError]:
    for node in _iter_error_nodes(tree.root_node):
        error_type = "missing" if node.is_missing else "error"
        yield TreeSitterParseError(
            error_type=error_type,
            message=node.type,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_row=node.start_point.row,
            start_col=node.start_point.column,
            end_row=node.end_point.row,
            end_col=node.end_point.column,
            text_preview=_slice_preview(
                source_bytes,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                preview_limit=preview_limit,
            ),
        )


def _iter_error_nodes(root: Node) -> Iterator[Node]:
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_error or node.is_missing:
            yield node
        stack.extend(node.children)


def _capture_from_node(
    node: Node,
    *,
    source_bytes: bytes,
    query_pack: str,
    capture_name: str,
    preview_limit: int,
) -> TreeSitterCapture:
    return TreeSitterCapture(
        query_pack=query_pack,
        capture_name=capture_name,
        node_type=node.type,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        start_row=node.start_point.row,
        start_col=node.start_point.column,
        end_row=node.end_point.row,
        end_col=node.end_point.column,
        text_preview=_slice_preview(
            source_bytes,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            preview_limit=preview_limit,
        ),
        extras=None,
    )


def _slice_preview(
    source_bytes: bytes,
    *,
    start_byte: int,
    end_byte: int,
    preview_limit: int,
) -> str | None:
    if start_byte < 0 or end_byte <= start_byte:
        return None
    if start_byte >= len(source_bytes):
        return None
    clipped_end = min(end_byte, len(source_bytes))
    payload = source_bytes[start_byte:clipped_end]
    if not payload:
        return None
    return payload[:preview_limit].decode("utf-8", "replace")


__all__ = [
    "DEFAULT_MATCH_LIMIT",
    "DEFAULT_PREVIEW_LIMIT",
    "TreeSitterCapture",
    "TreeSitterParseError",
    "TreeSitterParseResult",
    "run_tree_sitter",
]
