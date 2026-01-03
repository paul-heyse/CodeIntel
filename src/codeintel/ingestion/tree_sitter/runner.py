"""Tree-sitter parsing and query-pack execution helpers."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING

from tree_sitter import Query, QueryCursor, TreeCursor

from codeintel.core.spans import normalize_byte_span
from codeintel.ingestion.tree_sitter.registry import load_language, load_parser, load_query_packs

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tree_sitter import Language, Node, Tree
    from tree_sitter_language_pack import SupportedLanguage

_TOKEN_PREFIX = "token."
_TRIVIA_PREFIX = "trivia."
_TOKEN_LITERAL_KINDS = {"string", "number", "boolean", "none"}
_PREVIEW_LIMIT = 200
_TUPLE_PAIR_LEN = 2


class TreeSitterParserUnavailableError(TypeError):
    """Raised when the tree-sitter parser is not callable."""

    def __init__(self) -> None:
        super().__init__("Tree-sitter parser unavailable.")


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
    extras: dict[str, object] | None


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
    extras_json: dict[str, object] | None


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
    extras_json: dict[str, object] | None


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
    extras_json: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class TreeSitterParseError:
    """Tree-sitter parse error record."""

    error_type: str
    message: str | None
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
    captures: list[TreeSitterCapture] = field(default_factory=list)
    nodes: list[TreeSitterNode] = field(default_factory=list)
    edges: list[TreeSitterEdge] = field(default_factory=list)
    tokens: list[TreeSitterToken] = field(default_factory=list)
    trivia: list[TreeSitterTrivia] = field(default_factory=list)
    errors: list[TreeSitterParseError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    changed_ranges: list[TreeSitterChangedRange] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _CompiledQueryPack:
    name: str
    query: Query
    query_text: str
    query_hash: str
    capture_index: dict[str, int]
    unrooted_patterns: tuple[int, ...]
    non_local_patterns: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _QueryPackContext:
    source_bytes: bytes
    rel_path: str
    match_limit: int | None
    emit_tokens: bool
    emit_trivia: bool


@dataclass(slots=True)
class _CursorFrame:
    node_id: str
    next_child_ordinal: int = 0


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


def _compile_query(language: Language, source: str) -> Query:
    try:
        return Query(language, source)
    except (TypeError, ValueError):
        query_fn = getattr(language, "query", None)
        if not callable(query_fn):
            raise
        return query_fn(source)


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
    return _CompiledQueryPack(
        name=pack_name,
        query=query,
        query_text=source,
        query_hash=query_hash,
        capture_index=capture_index,
        unrooted_patterns=unrooted,
        non_local_patterns=non_local,
    )


@cache
def _compiled_query_packs(language: SupportedLanguage) -> tuple[_CompiledQueryPack, ...]:
    packs = load_query_packs(language)
    return tuple(_compile_query_pack(language, pack.name, pack.query_text) for pack in packs)


def _lint_query_pack(pack: _CompiledQueryPack, *, allow_non_local: bool) -> None:
    if pack.unrooted_patterns:
        msg = (
            f"Tree-sitter query pack {pack.name} has unrooted patterns: "
            f"{pack.unrooted_patterns}"
        )
        raise ValueError(msg)
    if pack.non_local_patterns and not allow_non_local:
        msg = (
            f"Tree-sitter query pack {pack.name} has non-local patterns: "
            f"{pack.non_local_patterns}"
        )
        raise ValueError(msg)


def _make_query_cursor(query: Query, match_limit: int | None) -> QueryCursor:
    try:
        return QueryCursor(query, match_limit=match_limit)
    except TypeError:
        cursor = QueryCursor()
    if match_limit is not None:
        setter = getattr(cursor, "set_match_limit", None)
        if callable(setter):
            setter(match_limit)
        else:
            with suppress(AttributeError):
                cursor.match_limit = match_limit
    return cursor


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


def _is_node(value: object) -> bool:
    return hasattr(value, "start_byte") and hasattr(value, "end_byte")


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
    if hasattr(match, "pattern") and hasattr(match, "captures"):
        return int(match.pattern), match.captures
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


def _token_kind(raw_kind: str) -> tuple[str, str | None]:
    if raw_kind in _TOKEN_LITERAL_KINDS:
        return "literal", raw_kind
    return raw_kind, None


def _query_pack_results(
    *,
    pack: _CompiledQueryPack,
    root: Node,
    source_bytes: bytes,
    rel_path: str,
    match_limit: int | None,
    emit_tokens: bool,
    emit_trivia: bool,
) -> tuple[list[TreeSitterCapture], list[TreeSitterToken], list[TreeSitterTrivia], list[str]]:
    captures: list[TreeSitterCapture] = []
    tokens: list[TreeSitterToken] = []
    trivia: list[TreeSitterTrivia] = []
    warnings: list[str] = []
    cursor = _make_query_cursor(pack.query, match_limit)
    matches = _run_matches(cursor, pack.query, root)
    exceeded = getattr(cursor, "did_exceed_match_limit", False)
    if exceeded:
        warnings.append(f"Tree-sitter pack {pack.name} exceeded match_limit")

    for pattern_index, capture_name, node in _iter_captures(matches):
        start_byte = int(node.start_byte)
        end_byte = int(node.end_byte)
        normalized = normalize_byte_span(start_byte, end_byte)
        if normalized is None:
            continue
        start_byte, end_byte = normalized
        start_row, start_col, end_row, end_col = _node_points(node)
        node_type = str(getattr(node, "type", ""))
        capture_index = pack.capture_index.get(capture_name)
        pattern_value = int(pattern_index) if isinstance(pattern_index, int) else None
        extras: dict[str, object] = {
            "query_hash": pack.query_hash,
            "pattern_index": pattern_value,
            "capture_index": capture_index,
        }
        captures.append(
            TreeSitterCapture(
                query_pack=pack.name,
                capture_name=capture_name,
                start_byte=start_byte,
                end_byte=end_byte,
                start_row=start_row,
                start_col=start_col,
                end_row=end_row,
                end_col=end_col,
                node_type=node_type,
                text_preview=_text_preview(source_bytes, start_byte, end_byte),
                extras=extras,
            )
        )

        if capture_name.startswith(_TOKEN_PREFIX) and emit_tokens:
            raw_kind = capture_name[len(_TOKEN_PREFIX) :]
            token_kind, literal_kind = _token_kind(raw_kind)
            token_extras: dict[str, object] = {
                "query_hash": pack.query_hash,
                "pattern_index": pattern_value,
                "capture_index": capture_index,
                "capture_name": capture_name,
            }
            if literal_kind is not None:
                token_extras["literal_kind"] = literal_kind
            tokens.append(
                TreeSitterToken(
                    token_id=_stable_id("ts_token", rel_path, start_byte, end_byte, token_kind),
                    token_kind=token_kind,
                    node_type=node_type,
                    start_byte=start_byte,
                    end_byte=end_byte,
                    start_row=start_row,
                    start_col=start_col,
                    end_row=end_row,
                    end_col=end_col,
                    text_preview=_text_preview(source_bytes, start_byte, end_byte),
                    extras_json=token_extras,
                )
            )
        if capture_name.startswith(_TRIVIA_PREFIX) and emit_trivia:
            trivia_kind = capture_name[len(_TRIVIA_PREFIX) :]
            trivia_extras: dict[str, object] = {
                "query_hash": pack.query_hash,
                "pattern_index": pattern_value,
                "capture_index": capture_index,
                "capture_name": capture_name,
            }
            trivia.append(
                TreeSitterTrivia(
                    trivia_id=_stable_id("ts_trivia", rel_path, start_byte, end_byte, trivia_kind),
                    trivia_kind=trivia_kind,
                    node_type=node_type,
                    start_byte=start_byte,
                    end_byte=end_byte,
                    start_row=start_row,
                    start_col=start_col,
                    end_row=end_row,
                    end_col=end_col,
                    text_preview=_text_preview(source_bytes, start_byte, end_byte),
                    extras_json=trivia_extras,
                )
            )

    return captures, tokens, trivia, warnings


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
    message = str(getattr(node, "type", None) or error_type)
    return TreeSitterParseError(
        error_type=error_type,
        message=message,
        start_byte=start_byte,
        end_byte=end_byte,
        start_row=start_row,
        start_col=start_col,
        end_row=end_row,
        end_col=end_col,
        text_preview=_text_preview(source_bytes, start_byte, end_byte),
    )


def _collect_nodes_edges(
    *,
    tree: Tree,
    language: Language,
    rel_path: str,
    source_bytes: bytes,
    emit_nodes_edges: bool,
) -> tuple[list[TreeSitterNode], list[TreeSitterEdge], list[TreeSitterParseError]]:
    nodes: list[TreeSitterNode] = []
    edges: list[TreeSitterEdge] = []
    errors: list[TreeSitterParseError] = []
    cursor: TreeCursor = tree.walk()

    def _process_node(
        node: Node,
        *,
        parent_id: str | None,
        field_name: str | None,
        field_id: int | None,
        child_ordinal: int,
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
            rel_path,
            start_byte,
            end_byte,
            node_type,
            grammar_id,
            field_id,
            child_ordinal,
        )

        error_row = _node_error_rows(node, source_bytes)
        if error_row is not None:
            errors.append(error_row)

        if emit_nodes_edges:
            nodes.append(
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
                    text_preview=_text_preview(source_bytes, start_byte, end_byte),
                    extras_json=None,
                )
            )
            if parent_id is not None:
                edges.append(
                    TreeSitterEdge(
                        parent_node_id=parent_id,
                        child_node_id=node_id,
                        field_id=field_id,
                        field_name=field_name,
                        child_ordinal=child_ordinal,
                    )
                )

        return node_id

    root_node = cursor.node
    root_id = _process_node(
        root_node,
        parent_id=None,
        field_name=None,
        field_id=None,
        child_ordinal=0,
    )
    frames: list[_CursorFrame] = [_CursorFrame(node_id=root_id)]

    while True:
        if cursor.goto_first_child():
            parent_frame = frames[-1]
            child_ordinal = parent_frame.next_child_ordinal
            parent_frame.next_child_ordinal += 1
            field_name = cursor.current_field_name
            field_id = _field_id(language, field_name)
            child_id = _process_node(
                cursor.node,
                parent_id=parent_frame.node_id,
                field_name=field_name,
                field_id=field_id,
                child_ordinal=child_ordinal,
            )
            frames.append(_CursorFrame(node_id=child_id))
            continue

        if cursor.goto_next_sibling():
            frames.pop()
            parent_frame = frames[-1] if frames else None
            child_ordinal = parent_frame.next_child_ordinal if parent_frame else 0
            if parent_frame is not None:
                parent_frame.next_child_ordinal += 1
            field_name = cursor.current_field_name
            field_id = _field_id(language, field_name)
            sibling_id = _process_node(
                cursor.node,
                parent_id=parent_frame.node_id if parent_frame else None,
                field_name=field_name,
                field_id=field_id,
                child_ordinal=child_ordinal,
            )
            frames.append(_CursorFrame(node_id=sibling_id))
            continue

        while True:
            if not cursor.goto_parent():
                return nodes, edges, errors
            frames.pop()
            if cursor.goto_next_sibling():
                frames.pop()
                parent_frame = frames[-1] if frames else None
                child_ordinal = parent_frame.next_child_ordinal if parent_frame else 0
                if parent_frame is not None:
                    parent_frame.next_child_ordinal += 1
                field_name = cursor.current_field_name
                field_id = _field_id(language, field_name)
                sibling_id = _process_node(
                    cursor.node,
                    parent_id=parent_frame.node_id if parent_frame else None,
                    field_name=field_name,
                    field_id=field_id,
                    child_ordinal=child_ordinal,
                )
                frames.append(_CursorFrame(node_id=sibling_id))
                break


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
        raise TreeSitterParserUnavailableError()
    if old_tree is not None:
        try:
            return parse_fn(source_bytes, old_tree=old_tree, encoding="utf8")
        except TypeError:
            return parse_fn(source_bytes, old_tree=old_tree)
    try:
        return parse_fn(source_bytes, encoding="utf8")
    except TypeError:
        return parse_fn(source_bytes)


def run_tree_sitter(
    *,
    language: SupportedLanguage,
    rel_path: str,
    source_bytes: bytes,
    emit_nodes_edges: bool = True,
    emit_tokens: bool = True,
    emit_trivia: bool = True,
    match_limit: int = 10000,
    allow_non_local_patterns: bool = False,
    old_tree: Tree | None = None,
) -> TreeSitterParseResult:
    """Parse a source buffer and execute tree-sitter query packs.

    Returns
    -------
    TreeSitterParseResult
        Parse result including captures, nodes, edges, and errors.
    """
    parser = load_parser(language)
    tree = _parse_tree(parser, source_bytes, old_tree)
    if tree is None:
        return TreeSitterParseResult(
            language=str(language),
            parse_ok=False,
            warnings=["Tree-sitter parser returned no tree"],
        )

    ts_language = load_language(language)
    root = tree.root_node
    nodes, edges, errors = _collect_nodes_edges(
        tree=tree,
        language=ts_language,
        rel_path=rel_path,
        source_bytes=source_bytes,
        emit_nodes_edges=emit_nodes_edges,
    )

    captures: list[TreeSitterCapture] = []
    tokens: list[TreeSitterToken] = []
    trivia: list[TreeSitterTrivia] = []
    warnings: list[str] = []

    for pack in _compiled_query_packs(language):
        if pack.name == "tokens" and not emit_tokens:
            continue
        if pack.name == "trivia" and not emit_trivia:
            continue
        _lint_query_pack(pack, allow_non_local=allow_non_local_patterns)
        pack_captures, pack_tokens, pack_trivia, pack_warnings = _query_pack_results(
            pack=pack,
            root=root,
            source_bytes=source_bytes,
            rel_path=rel_path,
            match_limit=match_limit,
            emit_tokens=emit_tokens,
            emit_trivia=emit_trivia,
        )
        captures.extend(pack_captures)
        tokens.extend(pack_tokens)
        trivia.extend(pack_trivia)
        warnings.extend(pack_warnings)

    return TreeSitterParseResult(
        language=str(language),
        parse_ok=True,
        captures=captures,
        nodes=nodes,
        edges=edges,
        tokens=tokens,
        trivia=trivia,
        errors=errors,
        warnings=warnings,
        changed_ranges=_changed_ranges(tree, old_tree),
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
