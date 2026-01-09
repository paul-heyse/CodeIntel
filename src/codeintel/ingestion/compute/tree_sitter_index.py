"""Tree-sitter ingestion step for query-pack captures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pyarrow as pa

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.core.columnar.rows import (
    ColumnarBatchCollector,
    columnar_batch_collector_for_table_key,
    empty_table_for_table,
)
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.spans import normalize_byte_span
from codeintel.ingestion.compute.base import (
    BaseExtractStep,
    finalize_arrow_tables,
    persist_arrow_tables,
)
from codeintel.ingestion.context import IngestionContext, resolve_repo_commit
from codeintel.ingestion.infrastructure.cst_utils import LineIndexedSource
from codeintel.ingestion.tree_sitter.registry import language_for_path, language_metadata
from codeintel.ingestion.tree_sitter.runner import TreeSitterRunOptions, run_tree_sitter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypedDict

    from tree_sitter import Tree
    from tree_sitter_language_pack import SupportedLanguage

    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort
    from codeintel.ingestion.tree_sitter.runner import (
        TreeSitterCapture,
        TreeSitterChangedRange,
        TreeSitterEdge,
        TreeSitterNode,
        TreeSitterParseError,
        TreeSitterParseResult,
        TreeSitterToken,
        TreeSitterTrivia,
    )

    class TreeSitterParseErrorExtras(TypedDict):
        node_type: str | None
        has_error: bool | None
        parse_state: int | None


TS_PARSE_MANIFEST_TABLE_KEY = "core.ts_parse_manifest"
TS_CAPTURES_TABLE_KEY = "core.ts_captures"
TS_NODES_TABLE_KEY = "core.ts_nodes"
TS_EDGES_TABLE_KEY = "core.ts_edges"
TS_PARSE_ERRORS_TABLE_KEY = "core.ts_parse_errors"
TS_CHANGED_RANGES_TABLE_KEY = "core.ts_changed_ranges"
TS_TOKENS_TABLE_KEY = "core.ts_tokens"
TS_TRIVIA_TABLE_KEY = "core.ts_trivia"
TS_LANGUAGE_METADATA_TABLE_KEY = "core.ts_language_metadata"
TREE_SITTER_PRODUCER = "tree_sitter"
TreeSitterRow = dict[str, object]


@dataclass(frozen=True)
class TreeSitterIndexResult:
    """Result bundle for tree-sitter query execution."""

    result: ExecutionResult
    parse_manifest_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(TS_PARSE_MANIFEST_TABLE_KEY)
    )
    captures_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(TS_CAPTURES_TABLE_KEY)
    )
    nodes_rows: pa.Table = field(default_factory=lambda: empty_table_for_table(TS_NODES_TABLE_KEY))
    edges_rows: pa.Table = field(default_factory=lambda: empty_table_for_table(TS_EDGES_TABLE_KEY))
    parse_errors_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(TS_PARSE_ERRORS_TABLE_KEY)
    )
    changed_ranges_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(TS_CHANGED_RANGES_TABLE_KEY)
    )
    tokens_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(TS_TOKENS_TABLE_KEY)
    )
    trivia_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(TS_TRIVIA_TABLE_KEY)
    )
    language_metadata_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(TS_LANGUAGE_METADATA_TABLE_KEY)
    )
    parse_manifest_row_count: int = 0
    captures_row_count: int = 0
    nodes_row_count: int = 0
    edges_row_count: int = 0
    parse_errors_row_count: int = 0
    changed_ranges_row_count: int = 0
    tokens_row_count: int = 0
    trivia_row_count: int = 0
    language_metadata_row_count: int = 0


@dataclass(frozen=True, slots=True)
class _ParseManifestContext:
    repo: str
    commit: str
    rel_path: str
    source_index: LineIndexedSource


@dataclass(slots=True)
class _TreeSitterBuffers:
    parse_manifest: ColumnarBatchCollector
    captures: ColumnarBatchCollector
    nodes: ColumnarBatchCollector
    edges: ColumnarBatchCollector
    parse_errors: ColumnarBatchCollector
    changed_ranges: ColumnarBatchCollector
    tokens: ColumnarBatchCollector
    trivia: ColumnarBatchCollector
    language_metadata: ColumnarBatchCollector


@dataclass(frozen=True, slots=True)
class _RowContext:
    repo: str
    commit: str
    rel_path: str
    language: str


@dataclass(frozen=True, slots=True)
class _ProcessModuleContext:
    module: ModuleRecord
    repo: str
    commit: str
    buffers: _TreeSitterBuffers
    discovery: ModuleDiscoveryPort
    config: TreeSitterIndexConfig
    seen_languages: set[str]
    created_at: datetime
    tree_cache: dict[str, Tree]
    source_cache: dict[str, bytes]


@dataclass(frozen=True, slots=True)
class TreeSitterIndexConfig:
    emit_nodes_edges: bool
    emit_tokens: bool
    emit_trivia: bool
    emit_language_metadata: bool
    enable_incremental: bool
    match_limit: int
    allow_non_local_patterns: bool


@dataclass(frozen=True, slots=True)
class TreeSitterIndexRunOptions:
    emit_nodes_edges: bool = True
    emit_tokens: bool = True
    emit_trivia: bool = True
    emit_language_metadata: bool = True
    enable_incremental: bool = False
    match_limit: int = 10000
    allow_non_local_patterns: bool = False


def _parse_manifest_row(
    context: _ParseManifestContext,
    *,
    parse_ok: bool,
    error: TreeSitterParseError | None,
    error_kind: str | None = None,
    error_message: str | None = None,
) -> TreeSitterRow:
    error_line = None
    error_col = None
    error_snippet = None

    if error is not None:
        error_kind = error.error_type
        error_message = error.message
        error_line = error.start_row
        error_col = error.start_col
        error_snippet = context.source_index.line_snippet(error.start_row)

    return {
        "repo": context.repo,
        "commit": context.commit,
        "rel_path": context.rel_path,
        "producer": TREE_SITTER_PRODUCER,
        "parse_ok": parse_ok,
        "encoding": "utf-8",
        "default_indent": None,
        "default_newline": None,
        "has_trailing_newline": None,
        "future_imports": None,
        "parser_backend": TREE_SITTER_PRODUCER,
        "libcst_version": None,
        "error_kind": error_kind,
        "error_message": error_message,
        "error_line": error_line,
        "error_col": error_col,
        "error_snippet": error_snippet,
    }


def _language_metadata_row(
    *,
    repo: str,
    commit: str,
    language: str,
    created_at: datetime,
) -> TreeSitterRow:
    metadata = language_metadata(cast("SupportedLanguage", language))
    return {
        "repo": repo,
        "commit": commit,
        "language": str(metadata.name),
        "abi_version": metadata.abi_version,
        "semantic_version": metadata.semantic_version,
        "node_kind_count": metadata.node_kind_count,
        "field_count": metadata.field_count,
        "parse_state_count": metadata.parse_state_count,
        "created_at": created_at,
    }


def _build_buffers() -> _TreeSitterBuffers:
    return _TreeSitterBuffers(
        parse_manifest=columnar_batch_collector_for_table_key(
            TS_PARSE_MANIFEST_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
        ),
        captures=columnar_batch_collector_for_table_key(
            TS_CAPTURES_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
        ),
        nodes=columnar_batch_collector_for_table_key(
            TS_NODES_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
        ),
        edges=columnar_batch_collector_for_table_key(
            TS_EDGES_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
        ),
        parse_errors=columnar_batch_collector_for_table_key(
            TS_PARSE_ERRORS_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
        ),
        changed_ranges=columnar_batch_collector_for_table_key(
            TS_CHANGED_RANGES_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
        ),
        tokens=columnar_batch_collector_for_table_key(
            TS_TOKENS_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
        ),
        trivia=columnar_batch_collector_for_table_key(
            TS_TRIVIA_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
        ),
        language_metadata=columnar_batch_collector_for_table_key(
            TS_LANGUAGE_METADATA_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
        ),
    )


def _materialize_tree_sitter_tables(
    buffers: _TreeSitterBuffers,
) -> dict[str, pa.Table]:
    return {
        TS_PARSE_MANIFEST_TABLE_KEY: buffers.parse_manifest.to_table(),
        TS_CAPTURES_TABLE_KEY: buffers.captures.to_table(),
        TS_NODES_TABLE_KEY: buffers.nodes.to_table(),
        TS_EDGES_TABLE_KEY: buffers.edges.to_table(),
        TS_PARSE_ERRORS_TABLE_KEY: buffers.parse_errors.to_table(),
        TS_CHANGED_RANGES_TABLE_KEY: buffers.changed_ranges.to_table(),
        TS_TOKENS_TABLE_KEY: buffers.tokens.to_table(),
        TS_TRIVIA_TABLE_KEY: buffers.trivia.to_table(),
        TS_LANGUAGE_METADATA_TABLE_KEY: buffers.language_metadata.to_table(),
    }


def _capture_rows(
    context: _RowContext,
    captures: Sequence[TreeSitterCapture],
) -> list[TreeSitterRow]:
    rows: list[TreeSitterRow] = []
    for capture in captures:
        normalized = normalize_byte_span(capture.start_byte, capture.end_byte)
        if normalized is None:
            continue
        start_byte, end_byte = normalized
        rows.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "rel_path": context.rel_path,
                "language": context.language,
                "query_pack": capture.query_pack,
                "capture_name": capture.capture_name,
                "start_byte": start_byte,
                "end_byte": end_byte,
                "start_row": capture.start_row,
                "start_col": capture.start_col,
                "end_row": capture.end_row,
                "end_col": capture.end_col,
                "node_type": capture.node_type,
                "text_preview": capture.text_preview,
                "extras": capture.extras,
            }
        )
    return rows


def _node_rows(
    context: _RowContext,
    nodes: Sequence[TreeSitterNode],
) -> list[TreeSitterRow]:
    rows: list[TreeSitterRow] = []
    for node in nodes:
        normalized = normalize_byte_span(node.start_byte, node.end_byte)
        if normalized is None:
            continue
        start_byte, end_byte = normalized
        rows.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "rel_path": context.rel_path,
                "language": context.language,
                "node_id": node.node_id,
                "node_type": node.node_type,
                "grammar_id": node.grammar_id,
                "kind_id": node.kind_id,
                "is_named": node.is_named,
                "is_missing": node.is_missing,
                "is_error": node.is_error,
                "has_error": node.has_error,
                "start_byte": start_byte,
                "end_byte": end_byte,
                "start_row": node.start_row,
                "start_col": node.start_col,
                "end_row": node.end_row,
                "end_col": node.end_col,
                "parse_state": node.parse_state,
                "next_parse_state": node.next_parse_state,
                "text_preview": node.text_preview,
                "extras": node.extras,
            }
        )
    return rows


def _edge_rows(
    context: _RowContext,
    edges: Sequence[TreeSitterEdge],
) -> list[TreeSitterRow]:
    return [
        {
            "repo": context.repo,
            "commit": context.commit,
            "rel_path": context.rel_path,
            "language": context.language,
            "parent_node_id": edge.parent_node_id,
            "child_node_id": edge.child_node_id,
            "field_id": edge.field_id,
            "field_name": edge.field_name,
            "child_ordinal": edge.child_ordinal,
        }
        for edge in edges
    ]


def _parse_error_rows(
    context: _RowContext,
    errors: Sequence[TreeSitterParseError],
) -> list[TreeSitterRow]:
    rows: list[TreeSitterRow] = []
    for error in errors:
        normalized = normalize_byte_span(error.start_byte, error.end_byte)
        if normalized is None:
            continue
        start_byte, end_byte = normalized
        extras: TreeSitterParseErrorExtras = {
            "node_type": error.node_type,
            "has_error": error.has_error,
            "parse_state": error.parse_state,
        }
        rows.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "rel_path": context.rel_path,
                "language": context.language,
                "error_type": error.error_type,
                "message": error.message,
                "start_byte": start_byte,
                "end_byte": end_byte,
                "start_row": error.start_row,
                "start_col": error.start_col,
                "end_row": error.end_row,
                "end_col": error.end_col,
                "text_preview": error.text_preview,
                "extras": extras,
            }
        )
    return rows


def _changed_range_rows(
    context: _RowContext,
    ranges: Sequence[TreeSitterChangedRange],
) -> list[TreeSitterRow]:
    rows: list[TreeSitterRow] = []
    for entry in ranges:
        normalized = normalize_byte_span(entry.start_byte, entry.end_byte)
        if normalized is None:
            continue
        start_byte, end_byte = normalized
        rows.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "rel_path": context.rel_path,
                "language": context.language,
                "start_byte": start_byte,
                "end_byte": end_byte,
                "start_row": entry.start_row,
                "start_col": entry.start_col,
                "end_row": entry.end_row,
                "end_col": entry.end_col,
            }
        )
    return rows


def _token_rows(
    context: _RowContext,
    tokens: Sequence[TreeSitterToken],
) -> list[TreeSitterRow]:
    rows: list[TreeSitterRow] = []
    for token in tokens:
        normalized = normalize_byte_span(token.start_byte, token.end_byte)
        if normalized is None:
            continue
        start_byte, end_byte = normalized
        rows.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "rel_path": context.rel_path,
                "language": context.language,
                "token_id": token.token_id,
                "token_kind": token.token_kind,
                "node_type": token.node_type,
                "start_byte": start_byte,
                "end_byte": end_byte,
                "start_row": token.start_row,
                "start_col": token.start_col,
                "end_row": token.end_row,
                "end_col": token.end_col,
                "text_preview": token.text_preview,
                "extras": token.extras,
            }
        )
    return rows


def _trivia_rows(
    context: _RowContext,
    trivia: Sequence[TreeSitterTrivia],
) -> list[TreeSitterRow]:
    rows: list[TreeSitterRow] = []
    for item in trivia:
        normalized = normalize_byte_span(item.start_byte, item.end_byte)
        if normalized is None:
            continue
        start_byte, end_byte = normalized
        rows.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "rel_path": context.rel_path,
                "language": context.language,
                "trivia_id": item.trivia_id,
                "trivia_kind": item.trivia_kind,
                "node_type": item.node_type,
                "start_byte": start_byte,
                "end_byte": end_byte,
                "start_row": item.start_row,
                "start_col": item.start_col,
                "end_row": item.end_row,
                "end_col": item.end_col,
                "text_preview": item.text_preview,
                "extras": item.extras,
            }
        )
    return rows


def _append_tree_sitter_rows(
    *,
    buffers: _TreeSitterBuffers,
    module: ModuleRecord,
    parse_result: TreeSitterParseResult,
    repo: str,
    commit: str,
) -> None:
    context = _RowContext(
        repo=repo,
        commit=commit,
        rel_path=module.rel_path,
        language=parse_result.language,
    )
    buffers.captures.extend(_capture_rows(context, parse_result.captures))
    buffers.nodes.extend(_node_rows(context, parse_result.nodes))
    buffers.edges.extend(_edge_rows(context, parse_result.edges))
    buffers.parse_errors.extend(_parse_error_rows(context, parse_result.errors))
    buffers.changed_ranges.extend(_changed_range_rows(context, parse_result.changed_ranges))
    buffers.tokens.extend(_token_rows(context, parse_result.tokens))
    buffers.trivia.extend(_trivia_rows(context, parse_result.trivia))


def _module_warnings(
    *,
    module: ModuleRecord,
    parse_result: TreeSitterParseResult,
) -> list[str]:
    warnings: list[str] = []
    if parse_result.warnings:
        warnings.extend(f"{module.rel_path}: {warning}" for warning in parse_result.warnings)
    if parse_result.errors:
        warnings.append(
            f"Tree-sitter parse errors in {module.rel_path}: {len(parse_result.errors)}"
        )
    return warnings


def _process_module(context: _ProcessModuleContext) -> list[str]:
    language = language_for_path(context.module.file_path)
    if language is None:
        return []

    source_bytes = context.discovery.read_module_bytes(context.module)
    if source_bytes is None:
        return [f"Tree-sitter skipped {context.module.rel_path}: unreadable source"]
    source_text = source_bytes.decode("utf-8", errors="replace")
    source_index = LineIndexedSource(source_text, source_bytes)
    manifest_context = _ParseManifestContext(
        repo=context.repo,
        commit=context.commit,
        rel_path=context.module.rel_path,
        source_index=source_index,
    )
    try:
        old_tree = None
        old_source_bytes = None
        if context.config.enable_incremental:
            old_tree = context.tree_cache.get(context.module.rel_path)
            old_source_bytes = context.source_cache.get(context.module.rel_path)
        options = TreeSitterRunOptions(
            emit_nodes_edges=context.config.emit_nodes_edges,
            emit_tokens=context.config.emit_tokens,
            emit_trivia=context.config.emit_trivia,
            match_limit=context.config.match_limit,
            allow_non_local_patterns=context.config.allow_non_local_patterns,
            old_tree=old_tree,
            old_source_bytes=old_source_bytes,
        )
        parse_result = run_tree_sitter(
            language=language,
            rel_path=context.module.rel_path,
            source_bytes=source_bytes,
            options=options,
        )
    except (RuntimeError, ValueError) as exc:
        context.buffers.parse_manifest.append(
            _parse_manifest_row(
                manifest_context,
                parse_ok=False,
                error=None,
                error_kind=type(exc).__name__,
                error_message=str(exc),
            )
        )
        return [f"Tree-sitter failed for {context.module.rel_path}: {exc}"]

    error = parse_result.errors[0] if parse_result.errors and not parse_result.parse_ok else None
    context.buffers.parse_manifest.append(
        _parse_manifest_row(
            manifest_context,
            parse_ok=parse_result.parse_ok,
            error=error,
        )
    )
    _append_tree_sitter_rows(
        buffers=context.buffers,
        module=context.module,
        parse_result=parse_result,
        repo=context.repo,
        commit=context.commit,
    )
    if context.config.enable_incremental and parse_result.tree is not None:
        context.tree_cache[context.module.rel_path] = parse_result.tree
        context.source_cache[context.module.rel_path] = source_bytes
    if (
        context.config.emit_language_metadata
        and parse_result.language not in context.seen_languages
    ):
        context.seen_languages.add(parse_result.language)
        context.buffers.language_metadata.append(
            _language_metadata_row(
                repo=context.repo,
                commit=context.commit,
                language=parse_result.language,
                created_at=context.created_at,
            )
        )
    return _module_warnings(module=context.module, parse_result=parse_result)


class TreeSitterIndexStep(BaseExtractStep):
    """Tree-sitter extraction step with port injection."""

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
        *,
        storage: IngestStoragePort | None = None,
    ) -> None:
        """Initialize the step with discovery ports and incremental caches."""
        super().__init__(discovery)
        self._tree_cache: dict[str, Tree] = {}
        self._source_cache: dict[str, bytes] = {}
        self._storage = storage

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str | None = None,
        commit: str | None = None,
        options: TreeSitterIndexRunOptions | None = None,
        context: IngestionContext | None = None,
    ) -> TreeSitterIndexResult:
        """Execute tree-sitter parsing for supported module files.

        Returns
        -------
        TreeSitterIndexResult
            Extraction result with columnar rows.
        """
        resolved_repo, resolved_commit = resolve_repo_commit(
            context=context,
            repo=repo,
            commit=commit,
        )
        try:
            buffers = _build_buffers()
        except (KeyError, RuntimeError) as exc:
            return TreeSitterIndexResult(result=ExecutionResult.failed(str(exc)))

        resolved_options = options or TreeSitterIndexRunOptions()
        config = TreeSitterIndexConfig(
            emit_nodes_edges=resolved_options.emit_nodes_edges,
            emit_tokens=resolved_options.emit_tokens,
            emit_trivia=resolved_options.emit_trivia,
            emit_language_metadata=resolved_options.emit_language_metadata,
            enable_incremental=resolved_options.enable_incremental,
            match_limit=resolved_options.match_limit,
            allow_non_local_patterns=resolved_options.allow_non_local_patterns,
        )
        created_at = datetime.now(UTC)
        seen_languages: set[str] = set()
        warnings: list[str] = []
        if resolved_options.enable_incremental:
            warnings.append("Tree-sitter incremental parsing uses in-memory caches only.")

        for module in modules:
            warnings.extend(
                _process_module(
                    _ProcessModuleContext(
                        module=module,
                        repo=resolved_repo,
                        commit=resolved_commit,
                        buffers=buffers,
                        discovery=self._discovery,
                        config=config,
                        seen_languages=seen_languages,
                        created_at=created_at,
                        tree_cache=self._tree_cache,
                        source_cache=self._source_cache,
                    )
                )
            )

        tables = _materialize_tree_sitter_tables(buffers)
        tables, finalize_warnings = finalize_arrow_tables(tables)
        warnings.extend(finalize_warnings)
        scope = f"{resolved_repo}@{resolved_commit}"
        persist_arrow_tables(self._storage, tables, scope=scope)
        return TreeSitterIndexResult(
            result=ExecutionResult.ok(warnings=tuple(warnings)),
            parse_manifest_rows=tables[TS_PARSE_MANIFEST_TABLE_KEY],
            captures_rows=tables[TS_CAPTURES_TABLE_KEY],
            nodes_rows=tables[TS_NODES_TABLE_KEY],
            edges_rows=tables[TS_EDGES_TABLE_KEY],
            parse_errors_rows=tables[TS_PARSE_ERRORS_TABLE_KEY],
            changed_ranges_rows=tables[TS_CHANGED_RANGES_TABLE_KEY],
            tokens_rows=tables[TS_TOKENS_TABLE_KEY],
            trivia_rows=tables[TS_TRIVIA_TABLE_KEY],
            language_metadata_rows=tables[TS_LANGUAGE_METADATA_TABLE_KEY],
            parse_manifest_row_count=tables[TS_PARSE_MANIFEST_TABLE_KEY].num_rows,
            captures_row_count=tables[TS_CAPTURES_TABLE_KEY].num_rows,
            nodes_row_count=tables[TS_NODES_TABLE_KEY].num_rows,
            edges_row_count=tables[TS_EDGES_TABLE_KEY].num_rows,
            parse_errors_row_count=tables[TS_PARSE_ERRORS_TABLE_KEY].num_rows,
            changed_ranges_row_count=tables[TS_CHANGED_RANGES_TABLE_KEY].num_rows,
            tokens_row_count=tables[TS_TOKENS_TABLE_KEY].num_rows,
            trivia_row_count=tables[TS_TRIVIA_TABLE_KEY].num_rows,
            language_metadata_row_count=tables[TS_LANGUAGE_METADATA_TABLE_KEY].num_rows,
        )


__all__ = ["TreeSitterIndexResult", "TreeSitterIndexStep"]
