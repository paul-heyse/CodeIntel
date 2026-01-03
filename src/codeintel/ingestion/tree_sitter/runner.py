"""Tree-sitter parsing and query-pack execution helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, TypeGuard

from tree_sitter import Query, QueryCursor, TreeCursor

from codeintel.core.spans import normalize_byte_span
from codeintel.ingestion.tree_sitter.registry import load_language, load_parser, load_query_packs

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tree_sitter import Language, Node, Tree
    from tree_sitter_language_pack import SupportedLanguage

    from codeintel.core.schemas.generated_rows.core import (
        TreeSitterCaptureExtras,
        TreeSitterNodeExtras,
        TreeSitterTokenExtras,
    )

_TOKEN_PREFIX = "token."
_TRIVIA_PREFIX = "trivia."
_TOKEN_LITERAL_KINDS = {"string", "number", "boolean", "none"}
_PREVIEW_LIMIT = 200
_TUPLE_PAIR_LEN = 2


class TreeSitterParserUnavailableError(TypeError):
    """Raised when the tree-sitter parser is not callable."""

    def __init__(self) -> None:
        super().__init__("Tree-sitter parser unavailable.")


class TreeSitterCursorError(RuntimeError):
    """Raised when a tree-sitter cursor is missing a node."""

    def __init__(self) -> None:
        super().__init__("Tree-sitter cursor has no current node.")


@dataclass(frozen=True, slots=True)
class TreeSitterCapture:
    """Tree-sitter query capture data."""

    query_pack: str
    capture_name: str
    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    node_type: str
    text_preview: str | None
    extras: TreeSitterCaptureExtras | None


@dataclass(frozen=True, slots=True)
class TreeSitterNode:
    """Tree-sitter CST node data."""

    node_id: str
    node_type: str
    grammar_id: int | None
    kind_id: int | None
    is_named: bool
    is_missing: bool
    is_error: bool
    has_error: bool
    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    parse_state: int | None
    next_parse_state: int | None
    text_preview: str | None
    extras_json: TreeSitterNodeExtras | None


@dataclass(frozen=True, slots=True)
class TreeSitterEdge:
    """Tree-sitter CST edge data."""

    parent_node_id: str
    child_node_id: str
    field_id: int | None
    field_name: str | None
    child_ordinal: int


@dataclass(frozen=True, slots=True)
class TreeSitterToken:
    """Token-level tree-sitter capture."""

    token_id: str
    token_kind: str
    node_type: str
    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    text_preview: str | None
    extras_json: TreeSitterTokenExtras | None


@dataclass(frozen=True, slots=True)
class TreeSitterTrivia:
    """Trivia-level tree-sitter capture."""

    trivia_id: str
    trivia_kind: str
    node_type: str
    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    text_preview: str | None
    extras_json: TreeSitterTokenExtras | None


@dataclass(frozen=True, slots=True)
class TreeSitterParseError:
    """Tree-sitter parse error record."""

    error_type: str
    message: str | None
    node_type: str | None
    has_error: bool | None
    parse_state: int | None
    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    text_preview: str | None


@dataclass(frozen=True, slots=True)
class TreeSitterChangedRange:
    """Tree-sitter changed range for incremental parsing."""

    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int


@dataclass(frozen=True, slots=True)
class TreeSitterParseResult:
    """Result bundle for a tree-sitter parse plus query captures."""

    language: str
    parse_ok: bool
    tree: Tree | None = None
    captures: list[TreeSitterCapture] = field(default_factory=list)
    nodes: list[TreeSitterNode] = field(default_factory=list)
    edges: list[TreeSitterEdge] = field(default_factory=list)
    tokens: list[TreeSitterToken] = field(default_factory=list)
    trivia: list[TreeSitterTrivia] = field(default_factory=list)
    errors: list[TreeSitterParseError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    changed_ranges: list[TreeSitterChangedRange] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TreeSitterRunOptions:
    emit_nodes_edges: bool = True
    emit_tokens: bool = True
    emit_trivia: bool = True
    match_limit: int = 10000
    allow_non_local_patterns: bool = False
    old_tree: Tree | None = None
    old_source_bytes: bytes | None = None


@dataclass(frozen=True, slots=True)
class _CompiledQueryPack:
    name: str
    query: Query
    query_text: str
    query_hash: str
    capture_index: dict[str, int]
    unrooted_patterns: tuple[int, ...]
    non_local_patterns: tuple[int, ...]
    pattern_count: int
    capture_count: int


@dataclass(frozen=True, slots=True)
class _QueryPackContext:
    source_bytes: bytes
    rel_path: str
    match_limit: int | None
    emit_tokens: bool
    emit_trivia: bool


@dataclass(frozen=True, slots=True)
class _CaptureRowContext:
    pack: _CompiledQueryPack
    context: _QueryPackContext
    language: Language


@dataclass(frozen=True, slots=True)
class _QueryPackRunContext:
    pack: _CompiledQueryPack
    root: Node
    context: _QueryPackContext
    language: Language
    changed_ranges: Sequence[TreeSitterChangedRange]


@dataclass(frozen=True, slots=True)
class _RunQueryPacksContext:
    language: SupportedLanguage
    root: Node
    source_bytes: bytes
    rel_path: str
    options: TreeSitterRunOptions
    changed_ranges: Sequence[TreeSitterChangedRange]


@dataclass(slots=True)
class _CursorFrame:
    node_id: str
    next_child_ordinal: int = 0


@dataclass(slots=True)
class _NodeProcessingContext:
    rel_path: str
    source_bytes: bytes
    emit_nodes_edges: bool
    nodes: list[TreeSitterNode]
    edges: list[TreeSitterEdge]
    errors: list[TreeSitterParseError]


@dataclass(slots=True)
class _QueryPackAccumulator:
    captures: list[TreeSitterCapture]
    tokens: list[TreeSitterToken]
    trivia: list[TreeSitterTrivia]
    warnings: list[str]
    seen_captures: set[tuple[str, int, int, str]]
    seen_tokens: set[str]
    seen_trivia: set[str]


@dataclass(frozen=True, slots=True)
class _NodeContext:
    parent_id: str | None
    field_name: str | None
    field_id: int | None
    child_ordinal: int


def _stable_id(*parts: object) -> str:
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=False)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def _text_preview(source_bytes: bytes, start_byte: int, end_byte: int) -> str | None:
    if end_byte <= start_byte:
        return None
    snippet = source_bytes[start_byte:end_byte]
    if len(snippet) > _PREVIEW_LIMIT:
        snippet = snippet[:_PREVIEW_LIMIT]
    try:
        return snippet.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return snippet.decode("utf-8", errors="replace")


def _node_points(node: Node) -> tuple[int, int, int, int]:
    start_point = getattr(node, "start_point", None)
    end_point = getattr(node, "end_point", None)
    if isinstance(start_point, tuple) and len(start_point) >= _TUPLE_PAIR_LEN:
        start_row, start_col = int(start_point[0]), int(start_point[1])
    else:
        start_row, start_col = 0, 0
    if isinstance(end_point, tuple) and len(end_point) >= _TUPLE_PAIR_LEN:
        end_row, end_col = int(end_point[0]), int(end_point[1])
    else:
        end_row, end_col = start_row, start_col
    return start_row, start_col, end_row, end_col


def _node_bool(node: Node, attr: str) -> bool:
    value = getattr(node, attr, False)
    return bool(value)


def _node_int(node: Node, attr: str) -> int | None:
    value = getattr(node, attr, None)
    if isinstance(value, int):
        return value
    return None


def _field_id(language: Language, field_name: str | None) -> int | None:
    if field_name is None:
        return None
    try:
        value = language.field_id_for_name(field_name)
    except (AttributeError, ValueError, TypeError):
        return None
    if isinstance(value, int):
        return value
    return None


def _is_tree(value: object) -> TypeGuard[Tree]:
    return hasattr(value, "root_node")


def _is_node(value: object) -> TypeGuard[Node]:
    return hasattr(value, "start_byte") and hasattr(value, "end_byte")


def _require_cursor_node(cursor: TreeCursor) -> Node:
    node = cursor.node
    if node is None:
        raise TreeSitterCursorError
    return node


def _cursor_field_name(cursor: TreeCursor) -> str | None:
    field_name = getattr(cursor, "current_field_name", None)
    if isinstance(field_name, str):
        return field_name
    if callable(field_name):
        value = field_name()
        return value if isinstance(value, str) else None
    return None


def _capture_index(query: Query) -> dict[str, int]:
    names = getattr(query, "capture_name", None)
    if not callable(names):
        capture_names = getattr(query, "capture_names", None)
        if isinstance(capture_names, Sequence) and not isinstance(capture_names, (bytes, str)):
            return {str(name): idx for idx, name in enumerate(capture_names)}
        return {}
    count = getattr(query, "capture_count", 0)
    return {str(names(i)): i for i in range(int(count))}


def _pattern_flags(query: Query) -> tuple[tuple[int, ...], tuple[int, ...]]:
    pattern_count = int(getattr(query, "pattern_count", 0))
    rooted = getattr(query, "is_pattern_rooted", None)
    non_local = getattr(query, "is_pattern_non_local", None)
    unrooted: list[int] = []
    non_local_patterns: list[int] = []
    for idx in range(pattern_count):
        if callable(rooted) and not rooted(idx):
            unrooted.append(idx)
        if callable(non_local) and non_local(idx):
            non_local_patterns.append(idx)
    return tuple(unrooted), tuple(non_local_patterns)


def _point_for_offset(source_bytes: bytes, offset: int) -> tuple[int, int]:
    if offset < 0:
        return 0, 0
    offset = min(offset, len(source_bytes))
    prefix = source_bytes[:offset]
    row = prefix.count(b"\n")
    last_newline = prefix.rfind(b"\n")
    col = offset if last_newline == -1 else offset - last_newline - 1
    return row, col


def _compute_edit(
    old_bytes: bytes,
    new_bytes: bytes,
) -> tuple[int, int, int, tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    if old_bytes == new_bytes:
        return None
    old_len = len(old_bytes)
    new_len = len(new_bytes)
    prefix = 0
    limit = min(old_len, new_len)
    while prefix < limit and old_bytes[prefix] == new_bytes[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < old_len - prefix
        and suffix < new_len - prefix
        and old_bytes[old_len - 1 - suffix] == new_bytes[new_len - 1 - suffix]
    ):
        suffix += 1
    old_end = old_len - suffix
    new_end = new_len - suffix
    start_point = _point_for_offset(old_bytes, prefix)
    old_end_point = _point_for_offset(old_bytes, old_end)
    new_end_point = _point_for_offset(new_bytes, new_end)
    return prefix, old_end, new_end, start_point, old_end_point, new_end_point


def _apply_tree_edit(old_tree: Tree, old_bytes: bytes, new_bytes: bytes) -> bool:
    edit = _compute_edit(old_bytes, new_bytes)
    if edit is None:
        return False
    start_byte, old_end_byte, new_end_byte, start_point, old_end_point, new_end_point = edit
    edit_fn = getattr(old_tree, "edit", None)
    if not callable(edit_fn):
        return False
    try:
        edit_fn(
            start_byte,
            old_end_byte,
            new_end_byte,
            start_point,
            old_end_point,
            new_end_point,
        )
    except TypeError:
        edit_fn(
            {
                "start_byte": start_byte,
                "old_end_byte": old_end_byte,
                "new_end_byte": new_end_byte,
                "start_point": start_point,
                "old_end_point": old_end_point,
                "new_end_point": new_end_point,
            }
        )
        return True
    else:
        return True


def _compile_query(language: Language, source: str) -> Query:
    try:
        return Query(language, source)
    except (TypeError, ValueError):
        query_fn = getattr(language, "query", None)
        if not callable(query_fn):
            raise
        result = query_fn(source)
        if isinstance(result, Query):
            return result
        msg = "Tree-sitter language query() did not return a Query."
        raise TypeError(msg) from None


def _compile_query_pack(
    language: SupportedLanguage,
    pack_name: str,
    source: str,
) -> _CompiledQueryPack:
    ts_language = load_language(language)
    query = _compile_query(ts_language, source)
    query_hash = hashlib.blake2b(source.encode("utf-8"), digest_size=16).hexdigest()
    capture_index = _capture_index(query)
    unrooted, non_local = _pattern_flags(query)
    pattern_count = int(getattr(query, "pattern_count", 0))
    capture_count = int(getattr(query, "capture_count", 0))
    return _CompiledQueryPack(
        name=pack_name,
        query=query,
        query_text=source,
        query_hash=query_hash,
        capture_index=capture_index,
        unrooted_patterns=unrooted,
        non_local_patterns=non_local,
        pattern_count=pattern_count,
        capture_count=capture_count,
    )


@cache
def _compiled_query_packs(language: SupportedLanguage) -> tuple[_CompiledQueryPack, ...]:
    packs = load_query_packs(language)
    return tuple(_compile_query_pack(language, pack.name, pack.query_text) for pack in packs)


def _lint_query_pack(pack: _CompiledQueryPack, *, allow_non_local: bool) -> None:
    if pack.unrooted_patterns:
        msg = f"Tree-sitter query pack {pack.name} has unrooted patterns: {pack.unrooted_patterns}"
        raise ValueError(msg)
    if pack.non_local_patterns and not allow_non_local:
        msg = (
            f"Tree-sitter query pack {pack.name} has non-local patterns: {pack.non_local_patterns}"
        )
        raise ValueError(msg)


def _make_query_cursor(query: Query, match_limit: int | None) -> QueryCursor:
    try:
        if match_limit is None:
            return QueryCursor(query)
        return QueryCursor(query, match_limit=match_limit)
    except TypeError:
        cursor = QueryCursor(query)
    if match_limit is not None:
        setter = getattr(cursor, "set_match_limit", None)
        if callable(setter):
            setter(match_limit)
        else:
            with suppress(AttributeError):
                cursor.match_limit = match_limit
    return cursor


def _set_query_cursor_byte_range(
    cursor: QueryCursor,
    start_byte: int,
    end_byte: int,
) -> bool:
    setter = getattr(cursor, "set_byte_range", None)
    if callable(setter):
        try:
            setter(start_byte, end_byte)
        except (TypeError, ValueError):
            return False
        else:
            return True
    return False


def _run_matches(cursor: QueryCursor, query: Query, node: Node) -> object:
    matches = getattr(cursor, "matches", None)
    if callable(matches):
        try:
            return matches(node)
        except TypeError:
            return matches(query, node)
    captures = getattr(cursor, "captures", None)
    if callable(captures):
        try:
            return captures(node)
        except TypeError:
            return captures(query, node)
    return []


def _capture_from_item(item: object) -> tuple[str, Node] | None:
    name = getattr(item, "name", None)
    node = getattr(item, "node", None)
    if isinstance(name, str) and _is_node(node):
        return name, node
    if isinstance(item, tuple) and len(item) >= _TUPLE_PAIR_LEN:
        first, second = item[0], item[1]
        if isinstance(first, str) and _is_node(second):
            return first, second
        if isinstance(second, str) and _is_node(first):
            return second, first
    return None


def _iter_captures(matches: object) -> Iterator[tuple[int | None, str, Node]]:
    if isinstance(matches, dict):
        yield from _iter_capture_dict(matches)
        return
    if isinstance(matches, Sequence):
        yield from _iter_capture_sequence(matches)


def _iter_capture_nodes(nodes: object) -> Iterator[Node]:
    if isinstance(nodes, Iterable) and not isinstance(nodes, (bytes, str)):
        for node in nodes:
            if _is_node(node):
                yield node
    elif _is_node(nodes):
        yield nodes


def _iter_capture_dict(matches: dict[object, object]) -> Iterator[tuple[int | None, str, Node]]:
    for name, nodes in matches.items():
        if not isinstance(name, str):
            continue
        for node in _iter_capture_nodes(nodes):
            yield None, name, node


def _match_capture_data(match: object) -> tuple[int | None, object | None]:
    pattern = getattr(match, "pattern", None)
    captures = getattr(match, "captures", None)
    if isinstance(pattern, int):
        return pattern, captures
    if isinstance(match, tuple) and len(match) == _TUPLE_PAIR_LEN:
        pattern_index, captures = match
        return pattern_index if isinstance(pattern_index, int) else None, captures
    return None, None


def _iter_capture_sequence(
    matches: Sequence[object],
) -> Iterator[tuple[int | None, str, Node]]:
    for match in matches:
        pattern_index, captures = _match_capture_data(match)
        if captures is None:
            continue
        if isinstance(captures, dict):
            for name, nodes in captures.items():
                if not isinstance(name, str):
                    continue
                for node in _iter_capture_nodes(nodes):
                    yield pattern_index, name, node
            continue
        if isinstance(captures, Iterable):
            for capture in captures:
                resolved = _capture_from_item(capture)
                if resolved is None:
                    continue
                name, node = resolved
                yield pattern_index, name, node


def _capture_field_name(node: Node) -> str | None:
    field_name = getattr(node, "field_name", None)
    if isinstance(field_name, str):
        return field_name
    parent = getattr(node, "parent", None)
    if parent is None:
        return None
    field_fn = getattr(parent, "field_name_for_child", None)
    if not callable(field_fn):
        return None
    try:
        name = field_fn(node)
    except (TypeError, ValueError):
        return None
    if isinstance(name, str):
        return name
    return None


def _token_kind(raw_kind: str) -> tuple[str, str | None]:
    if raw_kind in _TOKEN_LITERAL_KINDS:
        return "literal", raw_kind
    return raw_kind, None


def _merge_changed_ranges(
    ranges: Sequence[TreeSitterChangedRange],
) -> list[tuple[int, int]]:
    spans = sorted(
        (
            (int(entry.start_byte), int(entry.end_byte))
            for entry in ranges
            if isinstance(entry.start_byte, int) and isinstance(entry.end_byte, int)
        ),
        key=lambda item: item[0],
    )
    if not spans:
        return []
    merged: list[list[int]] = [[spans[0][0], spans[0][1]]]
    for start, end in spans[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


@dataclass(frozen=True, slots=True)
class _CaptureMetadata:
    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    node_type: str
    pattern_value: int | None
    capture_index: int | None
    field_name: str | None
    field_id: int | None


def _capture_metadata(
    *,
    pack: _CompiledQueryPack,
    capture_name: str,
    node: Node,
    pattern_index: int | None,
    language: Language,
) -> _CaptureMetadata | None:
    start_byte = int(node.start_byte)
    end_byte = int(node.end_byte)
    normalized = normalize_byte_span(start_byte, end_byte)
    if normalized is None:
        return None
    start_byte, end_byte = normalized
    start_row, start_col, end_row, end_col = _node_points(node)
    node_type = str(getattr(node, "type", ""))
    capture_index = pack.capture_index.get(capture_name)
    pattern_value = int(pattern_index) if isinstance(pattern_index, int) else None
    field_name = _capture_field_name(node)
    field_id = _field_id(language, field_name)
    return _CaptureMetadata(
        start_byte=start_byte,
        end_byte=end_byte,
        start_row=start_row,
        start_col=start_col,
        end_row=end_row,
        end_col=end_col,
        node_type=node_type,
        pattern_value=pattern_value,
        capture_index=capture_index,
        field_name=field_name,
        field_id=field_id,
    )


def _capture_row(
    *,
    pack: _CompiledQueryPack,
    capture_name: str,
    metadata: _CaptureMetadata,
    context: _QueryPackContext,
) -> TreeSitterCapture:
    extras: TreeSitterCaptureExtras = {
        "query_hash": pack.query_hash,
        "pattern_index": metadata.pattern_value,
        "capture_index": metadata.capture_index,
        "pattern_count": pack.pattern_count,
        "capture_count": pack.capture_count,
        "field_name": metadata.field_name,
        "field_id": metadata.field_id,
    }
    return TreeSitterCapture(
        query_pack=pack.name,
        capture_name=capture_name,
        start_byte=metadata.start_byte,
        end_byte=metadata.end_byte,
        start_row=metadata.start_row,
        start_col=metadata.start_col,
        end_row=metadata.end_row,
        end_col=metadata.end_col,
        node_type=metadata.node_type,
        text_preview=_text_preview(context.source_bytes, metadata.start_byte, metadata.end_byte),
        extras=extras,
    )


def _token_row(
    *,
    pack: _CompiledQueryPack,
    capture_name: str,
    metadata: _CaptureMetadata,
    context: _QueryPackContext,
) -> TreeSitterToken:
    raw_kind = capture_name[len(_TOKEN_PREFIX) :]
    token_kind, literal_kind = _token_kind(raw_kind)
    token_extras: TreeSitterTokenExtras = {
        "query_hash": pack.query_hash,
        "pattern_index": metadata.pattern_value,
        "capture_index": metadata.capture_index,
        "capture_name": capture_name,
        "pattern_count": pack.pattern_count,
        "capture_count": pack.capture_count,
        "field_name": metadata.field_name,
        "field_id": metadata.field_id,
    }
    if literal_kind is not None:
        token_extras["literal_kind"] = literal_kind
    return TreeSitterToken(
        token_id=_stable_id(
            "ts_token",
            context.rel_path,
            metadata.start_byte,
            metadata.end_byte,
            token_kind,
        ),
        token_kind=token_kind,
        node_type=metadata.node_type,
        start_byte=metadata.start_byte,
        end_byte=metadata.end_byte,
        start_row=metadata.start_row,
        start_col=metadata.start_col,
        end_row=metadata.end_row,
        end_col=metadata.end_col,
        text_preview=_text_preview(context.source_bytes, metadata.start_byte, metadata.end_byte),
        extras_json=token_extras,
    )


def _trivia_row(
    *,
    pack: _CompiledQueryPack,
    capture_name: str,
    metadata: _CaptureMetadata,
    context: _QueryPackContext,
) -> TreeSitterTrivia:
    trivia_kind = capture_name[len(_TRIVIA_PREFIX) :]
    trivia_extras: TreeSitterTokenExtras = {
        "query_hash": pack.query_hash,
        "pattern_index": metadata.pattern_value,
        "capture_index": metadata.capture_index,
        "capture_name": capture_name,
        "pattern_count": pack.pattern_count,
        "capture_count": pack.capture_count,
        "field_name": metadata.field_name,
        "field_id": metadata.field_id,
    }
    return TreeSitterTrivia(
        trivia_id=_stable_id(
            "ts_trivia",
            context.rel_path,
            metadata.start_byte,
            metadata.end_byte,
            trivia_kind,
        ),
        trivia_kind=trivia_kind,
        node_type=metadata.node_type,
        start_byte=metadata.start_byte,
        end_byte=metadata.end_byte,
        start_row=metadata.start_row,
        start_col=metadata.start_col,
        end_row=metadata.end_row,
        end_col=metadata.end_col,
        text_preview=_text_preview(context.source_bytes, metadata.start_byte, metadata.end_byte),
        extras_json=trivia_extras,
    )


def _capture_rows_for_match(
    *,
    capture_context: _CaptureRowContext,
    capture_name: str,
    node: Node,
    pattern_index: int | None,
) -> tuple[TreeSitterCapture | None, TreeSitterToken | None, TreeSitterTrivia | None]:
    metadata = _capture_metadata(
        pack=capture_context.pack,
        capture_name=capture_name,
        node=node,
        pattern_index=pattern_index,
        language=capture_context.language,
    )
    if metadata is None:
        return None, None, None
    capture = _capture_row(
        pack=capture_context.pack,
        capture_name=capture_name,
        metadata=metadata,
        context=capture_context.context,
    )
    token: TreeSitterToken | None = None
    if capture_name.startswith(_TOKEN_PREFIX) and capture_context.context.emit_tokens:
        token = _token_row(
            pack=capture_context.pack,
            capture_name=capture_name,
            metadata=metadata,
            context=capture_context.context,
        )
    trivia: TreeSitterTrivia | None = None
    if capture_name.startswith(_TRIVIA_PREFIX) and capture_context.context.emit_trivia:
        trivia = _trivia_row(
            pack=capture_context.pack,
            capture_name=capture_name,
            metadata=metadata,
            context=capture_context.context,
        )
    return capture, token, trivia


def _append_capture_rows(
    matches: object,
    *,
    capture_context: _CaptureRowContext,
    accumulator: _QueryPackAccumulator,
    dedupe: bool,
) -> None:
    for pattern_index, capture_name, node in _iter_captures(matches):
        capture, token, trivia_entry = _capture_rows_for_match(
            capture_context=capture_context,
            capture_name=capture_name,
            node=node,
            pattern_index=pattern_index,
        )
        if capture is not None:
            if dedupe:
                key = (
                    capture.capture_name,
                    capture.start_byte,
                    capture.end_byte,
                    capture.node_type,
                )
                if key in accumulator.seen_captures:
                    continue
                accumulator.seen_captures.add(key)
            accumulator.captures.append(capture)
        if token is not None:
            if dedupe and token.token_id in accumulator.seen_tokens:
                continue
            accumulator.seen_tokens.add(token.token_id)
            accumulator.tokens.append(token)
        if trivia_entry is not None:
            if dedupe and trivia_entry.trivia_id in accumulator.seen_trivia:
                continue
            accumulator.seen_trivia.add(trivia_entry.trivia_id)
            accumulator.trivia.append(trivia_entry)


def _query_pack_results(
    *,
    run_context: _QueryPackRunContext,
) -> tuple[list[TreeSitterCapture], list[TreeSitterToken], list[TreeSitterTrivia], list[str]]:
    accumulator = _QueryPackAccumulator(
        captures=[],
        tokens=[],
        trivia=[],
        warnings=[],
        seen_captures=set(),
        seen_tokens=set(),
        seen_trivia=set(),
    )
    cursor = _make_query_cursor(run_context.pack.query, run_context.context.match_limit)
    capture_context = _CaptureRowContext(
        pack=run_context.pack,
        context=run_context.context,
        language=run_context.language,
    )
    exceeded = False
    byte_ranges = _merge_changed_ranges(run_context.changed_ranges)
    for start_byte, end_byte in byte_ranges:
        if not _set_query_cursor_byte_range(cursor, start_byte, end_byte):
            byte_ranges = ()
            break
        matches = _run_matches(cursor, run_context.pack.query, run_context.root)
        exceeded = exceeded or bool(getattr(cursor, "did_exceed_match_limit", False))
        _append_capture_rows(
            matches,
            capture_context=capture_context,
            accumulator=accumulator,
            dedupe=True,
        )

    if not byte_ranges:
        matches = _run_matches(cursor, run_context.pack.query, run_context.root)
        exceeded = exceeded or bool(getattr(cursor, "did_exceed_match_limit", False))
        _append_capture_rows(
            matches,
            capture_context=capture_context,
            accumulator=accumulator,
            dedupe=False,
        )

    if exceeded:
        accumulator.warnings.append(
            f"Tree-sitter pack {run_context.pack.name} exceeded match_limit"
        )

    return (
        accumulator.captures,
        accumulator.tokens,
        accumulator.trivia,
        accumulator.warnings,
    )


def _node_error_rows(node: Node, source_bytes: bytes) -> TreeSitterParseError | None:
    is_error = _node_bool(node, "is_error")
    is_missing = _node_bool(node, "is_missing")
    if not (is_error or is_missing):
        return None
    start_byte = int(node.start_byte)
    end_byte = int(node.end_byte)
    normalized = normalize_byte_span(start_byte, end_byte)
    if normalized is None:
        return None
    start_byte, end_byte = normalized
    start_row, start_col, end_row, end_col = _node_points(node)
    error_type = "ERROR" if is_error else "MISSING"
    node_type = str(getattr(node, "type", None) or error_type)
    message = node_type
    return TreeSitterParseError(
        error_type=error_type,
        message=message,
        node_type=node_type,
        has_error=_node_bool(node, "has_error"),
        parse_state=_node_int(node, "parse_state"),
        start_byte=start_byte,
        end_byte=end_byte,
        start_row=start_row,
        start_col=start_col,
        end_row=end_row,
        end_col=end_col,
        text_preview=_text_preview(source_bytes, start_byte, end_byte),
    )


def _cursor_child_context(
    frames: list[_CursorFrame],
    *,
    cursor: TreeCursor,
    language: Language,
) -> _NodeContext:
    parent_frame = frames[-1] if frames else None
    child_ordinal = parent_frame.next_child_ordinal if parent_frame else 0
    if parent_frame is not None:
        parent_frame.next_child_ordinal += 1
    field_name = _cursor_field_name(cursor)
    field_id = _field_id(language, field_name)
    parent_id = parent_frame.node_id if parent_frame else None
    return _NodeContext(
        parent_id=parent_id,
        field_name=field_name,
        field_id=field_id,
        child_ordinal=child_ordinal,
    )


def _process_tree_node(
    node: Node,
    *,
    node_context: _NodeContext,
    context: _NodeProcessingContext,
) -> str:
    start_byte = int(node.start_byte)
    end_byte = int(node.end_byte)
    normalized = normalize_byte_span(start_byte, end_byte)
    if normalized is None:
        normalized = (start_byte, end_byte)
    start_byte, end_byte = normalized
    start_row, start_col, end_row, end_col = _node_points(node)
    node_type = str(getattr(node, "type", ""))
    grammar_id = _node_int(node, "grammar_id")
    kind_id = _node_int(node, "kind_id")
    node_id = _stable_id(
        "ts_node",
        context.rel_path,
        start_byte,
        end_byte,
        node_type,
        grammar_id,
        node_context.field_id,
        node_context.child_ordinal,
    )

    error_row = _node_error_rows(node, context.source_bytes)
    if error_row is not None:
        context.errors.append(error_row)

    if context.emit_nodes_edges:
        context.nodes.append(
            TreeSitterNode(
                node_id=node_id,
                node_type=node_type,
                grammar_id=grammar_id,
                kind_id=kind_id,
                is_named=_node_bool(node, "is_named"),
                is_missing=_node_bool(node, "is_missing"),
                is_error=_node_bool(node, "is_error"),
                has_error=_node_bool(node, "has_error"),
                start_byte=start_byte,
                end_byte=end_byte,
                start_row=start_row,
                start_col=start_col,
                end_row=end_row,
                end_col=end_col,
                parse_state=_node_int(node, "parse_state"),
                next_parse_state=_node_int(node, "next_parse_state"),
                text_preview=_text_preview(context.source_bytes, start_byte, end_byte),
                extras_json=None,
            )
        )
        if node_context.parent_id is not None:
            context.edges.append(
                TreeSitterEdge(
                    parent_node_id=node_context.parent_id,
                    child_node_id=node_id,
                    field_id=node_context.field_id,
                    field_name=node_context.field_name,
                    child_ordinal=node_context.child_ordinal,
                )
            )

    return node_id


def _advance_to_next_sibling(
    cursor: TreeCursor,
    frames: list[_CursorFrame],
    *,
    language: Language,
    context: _NodeProcessingContext,
) -> bool:
    while True:
        if not cursor.goto_parent():
            return False
        frames.pop()
        if cursor.goto_next_sibling():
            frames.pop()
            node_context = _cursor_child_context(
                frames,
                cursor=cursor,
                language=language,
            )
            sibling_id = _process_tree_node(
                _require_cursor_node(cursor),
                node_context=node_context,
                context=context,
            )
            frames.append(_CursorFrame(node_id=sibling_id))
            return True


def _collect_nodes_edges(
    *,
    tree: Tree,
    language: Language,
    rel_path: str,
    source_bytes: bytes,
    emit_nodes_edges: bool,
) -> tuple[list[TreeSitterNode], list[TreeSitterEdge], list[TreeSitterParseError]]:
    context = _NodeProcessingContext(
        rel_path=rel_path,
        source_bytes=source_bytes,
        emit_nodes_edges=emit_nodes_edges,
        nodes=[],
        edges=[],
        errors=[],
    )
    cursor: TreeCursor = tree.walk()

    root_id = _process_tree_node(
        _require_cursor_node(cursor),
        node_context=_NodeContext(
            parent_id=None,
            field_name=None,
            field_id=None,
            child_ordinal=0,
        ),
        context=context,
    )
    frames: list[_CursorFrame] = [_CursorFrame(node_id=root_id)]

    while True:
        if cursor.goto_first_child():
            node_context = _cursor_child_context(
                frames,
                cursor=cursor,
                language=language,
            )
            child_id = _process_tree_node(
                _require_cursor_node(cursor),
                node_context=node_context,
                context=context,
            )
            frames.append(_CursorFrame(node_id=child_id))
            continue

        if cursor.goto_next_sibling():
            frames.pop()
            node_context = _cursor_child_context(
                frames,
                cursor=cursor,
                language=language,
            )
            sibling_id = _process_tree_node(
                _require_cursor_node(cursor),
                node_context=node_context,
                context=context,
            )
            frames.append(_CursorFrame(node_id=sibling_id))
            continue

        if not _advance_to_next_sibling(
            cursor,
            frames,
            language=language,
            context=context,
        ):
            return context.nodes, context.edges, context.errors


def _changed_ranges(tree: Tree, old_tree: Tree | None) -> list[TreeSitterChangedRange]:
    if old_tree is None:
        return []
    ranges_fn = getattr(tree, "changed_ranges", None)
    if not callable(ranges_fn):
        return []
    ranges = ranges_fn(old_tree)
    if not isinstance(ranges, Sequence):
        return []
    changed: list[TreeSitterChangedRange] = []
    for entry in ranges:
        start_byte = getattr(entry, "start_byte", None)
        end_byte = getattr(entry, "end_byte", None)
        start_point = getattr(entry, "start_point", None)
        end_point = getattr(entry, "end_point", None)
        if not isinstance(start_byte, int) or not isinstance(end_byte, int):
            continue
        if not isinstance(start_point, tuple) or not isinstance(end_point, tuple):
            continue
        if len(start_point) < _TUPLE_PAIR_LEN or len(end_point) < _TUPLE_PAIR_LEN:
            continue
        changed.append(
            TreeSitterChangedRange(
                start_byte=int(start_byte),
                end_byte=int(end_byte),
                start_row=int(start_point[0]),
                start_col=int(start_point[1]),
                end_row=int(end_point[0]),
                end_col=int(end_point[1]),
            )
        )
    return changed


def _parse_tree(parser: object, source_bytes: bytes, old_tree: Tree | None) -> Tree | None:
    parse_fn = getattr(parser, "parse", None)
    if not callable(parse_fn):
        raise TreeSitterParserUnavailableError
    if old_tree is not None:
        try:
            result = parse_fn(source_bytes, old_tree=old_tree, encoding="utf8")
        except TypeError:
            result = parse_fn(source_bytes, old_tree=old_tree)
        return result if _is_tree(result) else None
    try:
        result = parse_fn(source_bytes, encoding="utf8")
    except TypeError:
        result = parse_fn(source_bytes)
    return result if _is_tree(result) else None


def _run_query_packs(
    *,
    run_context: _RunQueryPacksContext,
) -> tuple[
    list[TreeSitterCapture],
    list[TreeSitterToken],
    list[TreeSitterTrivia],
    list[str],
]:
    """Collect capture, token, trivia, and warning data from query packs.

    Parameters
    ----------
    run_context
        Run context for tree-sitter query execution.

    Returns
    -------
    tuple[list[TreeSitterCapture], list[TreeSitterToken], list[TreeSitterTrivia], list[str]]
        Captures, tokens, trivia, and warnings collected from all packs.
    """
    captures: list[TreeSitterCapture] = []
    tokens: list[TreeSitterToken] = []
    trivia: list[TreeSitterTrivia] = []
    warnings: list[str] = []

    packs_context = run_context
    ts_language = load_language(packs_context.language)
    for pack in _compiled_query_packs(packs_context.language):
        if pack.name == "tokens" and not packs_context.options.emit_tokens:
            continue
        if pack.name == "trivia" and not packs_context.options.emit_trivia:
            continue
        _lint_query_pack(pack, allow_non_local=packs_context.options.allow_non_local_patterns)
        context = _QueryPackContext(
            source_bytes=packs_context.source_bytes,
            rel_path=packs_context.rel_path,
            match_limit=packs_context.options.match_limit,
            emit_tokens=packs_context.options.emit_tokens,
            emit_trivia=packs_context.options.emit_trivia,
        )
        pack_context = _QueryPackRunContext(
            pack=pack,
            root=packs_context.root,
            context=context,
            language=ts_language,
            changed_ranges=packs_context.changed_ranges,
        )
        pack_captures, pack_tokens, pack_trivia, pack_warnings = _query_pack_results(
            run_context=pack_context,
        )
        captures.extend(pack_captures)
        tokens.extend(pack_tokens)
        trivia.extend(pack_trivia)
        warnings.extend(pack_warnings)

    return captures, tokens, trivia, warnings


def run_tree_sitter(
    *,
    language: SupportedLanguage,
    rel_path: str,
    source_bytes: bytes,
    options: TreeSitterRunOptions | None = None,
) -> TreeSitterParseResult:
    """Parse a source buffer and execute tree-sitter query packs.

    Returns
    -------
    TreeSitterParseResult
        Parse result including captures, nodes, edges, and errors.
    """
    resolved_options = options or TreeSitterRunOptions()
    parser = load_parser(language)
    old_tree = resolved_options.old_tree
    if old_tree is not None and resolved_options.old_source_bytes is not None:
        with suppress(AttributeError, TypeError, ValueError):
            _apply_tree_edit(old_tree, resolved_options.old_source_bytes, source_bytes)
    elif old_tree is not None:
        old_tree = None
    tree = _parse_tree(parser, source_bytes, old_tree)
    if tree is None:
        return TreeSitterParseResult(
            language=str(language),
            parse_ok=False,
            warnings=["Tree-sitter parser returned no tree"],
        )

    changed_ranges = _changed_ranges(tree, old_tree)
    nodes, edges, errors = _collect_nodes_edges(
        tree=tree,
        language=load_language(language),
        rel_path=rel_path,
        source_bytes=source_bytes,
        emit_nodes_edges=resolved_options.emit_nodes_edges,
    )
    captures, tokens, trivia, warnings = _run_query_packs(
        run_context=_RunQueryPacksContext(
            language=language,
            root=tree.root_node,
            source_bytes=source_bytes,
            rel_path=rel_path,
            options=resolved_options,
            changed_ranges=changed_ranges,
        )
    )

    return TreeSitterParseResult(
        language=str(language),
        parse_ok=True,
        tree=tree,
        captures=captures,
        nodes=nodes,
        edges=edges,
        tokens=tokens,
        trivia=trivia,
        errors=errors,
        warnings=warnings,
        changed_ranges=changed_ranges,
    )


__all__ = [
    "TreeSitterCapture",
    "TreeSitterChangedRange",
    "TreeSitterEdge",
    "TreeSitterNode",
    "TreeSitterParseError",
    "TreeSitterParseResult",
    "TreeSitterToken",
    "TreeSitterTrivia",
    "run_tree_sitter",
]
