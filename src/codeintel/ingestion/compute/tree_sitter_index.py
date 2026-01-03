"""Tree-sitter ingestion step for query-pack captures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.core.columnar.rows import (
    ColumnarBatchCollector,
    columnar_batch_collector_for_table_key,
    empty_reader_for_table,
)
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.schemas.generated_rows.core import (
    CoreParseManifestRow,
    CoreTsCapturesRow,
    CoreTsEdgesRow,
    CoreTsLanguageMetadataRow,
    CoreTsNodesRow,
    CoreTsParseErrorsRow,
    CoreTsTokensRow,
    CoreTsTriviaRow,
)
from codeintel.core.spans import normalize_byte_span
from codeintel.ingestion.compute.base import BaseExtractStep
from codeintel.ingestion.infrastructure.cst_utils import LineIndexedSource
from codeintel.ingestion.tree_sitter.registry import language_for_path, language_metadata
from codeintel.ingestion.tree_sitter.runner import run_tree_sitter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
    from codeintel.ingestion.tree_sitter.runner import TreeSitterParseError, TreeSitterParseResult

PARSE_MANIFEST_TABLE_KEY = "core.parse_manifest"
TS_CAPTURES_TABLE_KEY = "core.ts_captures"
TS_NODES_TABLE_KEY = "core.ts_nodes"
TS_EDGES_TABLE_KEY = "core.ts_edges"
TS_PARSE_ERRORS_TABLE_KEY = "core.ts_parse_errors"
TS_TOKENS_TABLE_KEY = "core.ts_tokens"
TS_TRIVIA_TABLE_KEY = "core.ts_trivia"
TS_LANGUAGE_METADATA_TABLE_KEY = "core.ts_language_metadata"
TREE_SITTER_PRODUCER = "tree_sitter"


@dataclass(frozen=True)
class TreeSitterIndexResult:
    """Result bundle for tree-sitter query execution."""

    result: ExecutionResult
    parse_manifest_rows: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(PARSE_MANIFEST_TABLE_KEY)
    )
    captures_rows: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(TS_CAPTURES_TABLE_KEY)
    )
    nodes_rows: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(TS_NODES_TABLE_KEY)
    )
    edges_rows: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(TS_EDGES_TABLE_KEY)
    )
    parse_errors_rows: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(TS_PARSE_ERRORS_TABLE_KEY)
    )
    tokens_rows: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(TS_TOKENS_TABLE_KEY)
    )
    trivia_rows: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(TS_TRIVIA_TABLE_KEY)
    )
    language_metadata_rows: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(TS_LANGUAGE_METADATA_TABLE_KEY)
    )
    parse_manifest_row_count: int = 0
    captures_row_count: int = 0
    nodes_row_count: int = 0
    edges_row_count: int = 0
    parse_errors_row_count: int = 0
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
    tokens: ColumnarBatchCollector
    trivia: ColumnarBatchCollector
    language_metadata: ColumnarBatchCollector


@dataclass(frozen=True, slots=True)
class _TreeSitterIndexConfig:
    emit_nodes_edges: bool
    emit_tokens: bool
    emit_trivia: bool
    emit_language_metadata: bool
    enable_incremental: bool
    match_limit: int
    allow_non_local_patterns: bool


def _parse_manifest_row(
    context: _ParseManifestContext,
    *,
    parse_ok: bool,
    error: TreeSitterParseError | None,
    error_kind: str | None = None,
    error_message: str | None = None,
) -> CoreParseManifestRow:
    error_line = None
    error_col = None
    error_snippet = None

    if error is not None:
        error_kind = error.error_type
        error_message = error.message
        error_line = error.start_row
        error_col = error.start_col
        error_snippet = context.source_index.line_snippet(error.start_row)

    return CoreParseManifestRow(
        repo=context.repo,
        commit=context.commit,
        rel_path=context.rel_path,
        producer=TREE_SITTER_PRODUCER,
        parse_ok=parse_ok,
        encoding="utf-8",
        default_indent=None,
        default_newline=None,
        has_trailing_newline=None,
        future_imports=None,
        parser_backend=TREE_SITTER_PRODUCER,
        libcst_version=None,
        error_kind=error_kind,
        error_message=error_message,
        error_line=error_line,
        error_col=error_col,
        error_snippet=error_snippet,
    )


def _language_metadata_row(
    *,
    repo: str,
    commit: str,
    language: str,
    created_at: datetime,
) -> CoreTsLanguageMetadataRow:
    metadata = language_metadata(language)
    return CoreTsLanguageMetadataRow(
        repo=repo,
        commit=commit,
        language=str(metadata.name),
        abi_version=metadata.abi_version,
        semantic_version=metadata.semantic_version,
        node_kind_count=metadata.node_kind_count,
        field_count=metadata.field_count,
        parse_state_count=metadata.parse_state_count,
        created_at=created_at,
    )


def _build_buffers() -> _TreeSitterBuffers:
    return _TreeSitterBuffers(
        parse_manifest=columnar_batch_collector_for_table_key(
            PARSE_MANIFEST_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
            extras_policy="retain",
        ),
        captures=columnar_batch_collector_for_table_key(
            TS_CAPTURES_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
            extras_policy="retain",
        ),
        nodes=columnar_batch_collector_for_table_key(
            TS_NODES_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
            extras_policy="retain",
        ),
        edges=columnar_batch_collector_for_table_key(
            TS_EDGES_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
            extras_policy="retain",
        ),
        parse_errors=columnar_batch_collector_for_table_key(
            TS_PARSE_ERRORS_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
            extras_policy="retain",
        ),
        tokens=columnar_batch_collector_for_table_key(
            TS_TOKENS_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
            extras_policy="retain",
        ),
        trivia=columnar_batch_collector_for_table_key(
            TS_TRIVIA_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
            extras_policy="retain",
        ),
        language_metadata=columnar_batch_collector_for_table_key(
            TS_LANGUAGE_METADATA_TABLE_KEY,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
            extras_policy="retain",
        ),
    )


def _append_tree_sitter_rows(
    *,
    buffers: _TreeSitterBuffers,
    module: ModuleRecord,
    parse_result: TreeSitterParseResult,
    repo: str,
    commit: str,
) -> None:
    capture_rows: list[CoreTsCapturesRow] = []
    for capture in parse_result.captures:
        normalized = normalize_byte_span(capture.start_byte, capture.end_byte)
        if normalized is None:
            continue
        start_byte, end_byte = normalized
        capture_rows.append(
            CoreTsCapturesRow(
                repo=repo,
                commit=commit,
                rel_path=module.rel_path,
                language=parse_result.language,
                query_pack=capture.query_pack,
                capture_name=capture.capture_name,
                start_byte=start_byte,
                end_byte=end_byte,
                start_row=capture.start_row,
                start_col=capture.start_col,
                end_row=capture.end_row,
                end_col=capture.end_col,
                node_type=capture.node_type,
                text_preview=capture.text_preview,
                extras=capture.extras,
            )
        )
    buffers.captures.extend(capture_rows)

    node_rows: list[CoreTsNodesRow] = []
    for node in parse_result.nodes:
        normalized = normalize_byte_span(node.start_byte, node.end_byte)
        if normalized is None:
            continue
        start_byte, end_byte = normalized
        node_rows.append(
            CoreTsNodesRow(
                repo=repo,
                commit=commit,
                rel_path=module.rel_path,
                language=parse_result.language,
                node_id=node.node_id,
                node_type=node.node_type,
                grammar_id=node.grammar_id,
                kind_id=node.kind_id,
                is_named=node.is_named,
                is_missing=node.is_missing,
                is_error=node.is_error,
                has_error=node.has_error,
                start_byte=start_byte,
                end_byte=end_byte,
                start_row=node.start_row,
                start_col=node.start_col,
                end_row=node.end_row,
                end_col=node.end_col,
                parse_state=node.parse_state,
                next_parse_state=node.next_parse_state,
                text_preview=node.text_preview,
                extras_json=node.extras_json,
            )
        )
    buffers.nodes.extend(node_rows)

    edge_rows: list[CoreTsEdgesRow] = [
        CoreTsEdgesRow(
            repo=repo,
            commit=commit,
            rel_path=module.rel_path,
            language=parse_result.language,
            parent_node_id=edge.parent_node_id,
            child_node_id=edge.child_node_id,
            field_id=edge.field_id,
            field_name=edge.field_name,
            child_ordinal=edge.child_ordinal,
        )
        for edge in parse_result.edges
    ]
    buffers.edges.extend(edge_rows)

    error_rows: list[CoreTsParseErrorsRow] = []
    for error in parse_result.errors:
        normalized = normalize_byte_span(error.start_byte, error.end_byte)
        if normalized is None:
            continue
        start_byte, end_byte = normalized
        error_rows.append(
            CoreTsParseErrorsRow(
                repo=repo,
                commit=commit,
                rel_path=module.rel_path,
                language=parse_result.language,
                error_type=error.error_type,
                message=error.message,
                start_byte=start_byte,
                end_byte=end_byte,
                start_row=error.start_row,
                start_col=error.start_col,
                end_row=error.end_row,
                end_col=error.end_col,
                text_preview=error.text_preview,
            )
        )
    buffers.parse_errors.extend(error_rows)

    token_rows: list[CoreTsTokensRow] = []
    for token in parse_result.tokens:
        normalized = normalize_byte_span(token.start_byte, token.end_byte)
        if normalized is None:
            continue
        start_byte, end_byte = normalized
        token_rows.append(
            CoreTsTokensRow(
                repo=repo,
                commit=commit,
                rel_path=module.rel_path,
                language=parse_result.language,
                token_id=token.token_id,
                token_kind=token.token_kind,
                node_type=token.node_type,
                start_byte=start_byte,
                end_byte=end_byte,
                start_row=token.start_row,
                start_col=token.start_col,
                end_row=token.end_row,
                end_col=token.end_col,
                text_preview=token.text_preview,
                extras_json=token.extras_json,
            )
        )
    buffers.tokens.extend(token_rows)

    trivia_rows: list[CoreTsTriviaRow] = []
    for trivia in parse_result.trivia:
        normalized = normalize_byte_span(trivia.start_byte, trivia.end_byte)
        if normalized is None:
            continue
        start_byte, end_byte = normalized
        trivia_rows.append(
            CoreTsTriviaRow(
                repo=repo,
                commit=commit,
                rel_path=module.rel_path,
                language=parse_result.language,
                trivia_id=trivia.trivia_id,
                trivia_kind=trivia.trivia_kind,
                node_type=trivia.node_type,
                start_byte=start_byte,
                end_byte=end_byte,
                start_row=trivia.start_row,
                start_col=trivia.start_col,
                end_row=trivia.end_row,
                end_col=trivia.end_col,
                text_preview=trivia.text_preview,
                extras_json=trivia.extras_json,
            )
        )
    buffers.trivia.extend(trivia_rows)


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


def _process_module(
    *,
    module: ModuleRecord,
    repo: str,
    commit: str,
    buffers: _TreeSitterBuffers,
    discovery: ModuleDiscoveryPort,
    config: _TreeSitterIndexConfig,
    seen_languages: set[str],
    created_at: datetime,
) -> list[str]:
    language = language_for_path(module.file_path)
    if language is None:
        return []

    source_bytes = discovery.read_module_bytes(module)
    if source_bytes is None:
        return [f"Tree-sitter skipped {module.rel_path}: unreadable source"]
    source_text = source_bytes.decode("utf-8", errors="replace")
    source_index = LineIndexedSource(source_text, source_bytes)
    context = _ParseManifestContext(
        repo=repo,
        commit=commit,
        rel_path=module.rel_path,
        source_index=source_index,
    )
    try:
        parse_result = run_tree_sitter(
            language=language,
            rel_path=module.rel_path,
            source_bytes=source_bytes,
            emit_nodes_edges=config.emit_nodes_edges,
            emit_tokens=config.emit_tokens,
            emit_trivia=config.emit_trivia,
            match_limit=config.match_limit,
            allow_non_local_patterns=config.allow_non_local_patterns,
        )
    except (RuntimeError, ValueError) as exc:
        buffers.parse_manifest.append(
            _parse_manifest_row(
                context,
                parse_ok=False,
                error=None,
                error_kind=type(exc).__name__,
                error_message=str(exc),
            )
        )
        return [f"Tree-sitter failed for {module.rel_path}: {exc}"]

    error = parse_result.errors[0] if parse_result.errors and not parse_result.parse_ok else None
    buffers.parse_manifest.append(
        _parse_manifest_row(
            context,
            parse_ok=parse_result.parse_ok,
            error=error,
        )
    )
    _append_tree_sitter_rows(
        buffers=buffers,
        module=module,
        parse_result=parse_result,
        repo=repo,
        commit=commit,
    )
    if config.emit_language_metadata and parse_result.language not in seen_languages:
        seen_languages.add(parse_result.language)
        buffers.language_metadata.append(
            _language_metadata_row(
                repo=repo,
                commit=commit,
                language=parse_result.language,
                created_at=created_at,
            )
        )
    return _module_warnings(module=module, parse_result=parse_result)


class TreeSitterIndexStep(BaseExtractStep):
    """Tree-sitter extraction step with port injection."""

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
        emit_nodes_edges: bool = True,
        emit_tokens: bool = True,
        emit_trivia: bool = True,
        emit_language_metadata: bool = True,
        enable_incremental: bool = False,
        match_limit: int = 10000,
        allow_non_local_patterns: bool = False,
    ) -> TreeSitterIndexResult:
        """Execute tree-sitter parsing for supported module files.

        Returns
        -------
        TreeSitterIndexResult
            Extraction result with columnar rows.
        """
        try:
            buffers = _build_buffers()
        except (KeyError, RuntimeError) as exc:
            return TreeSitterIndexResult(result=ExecutionResult.failed(str(exc)))

        config = _TreeSitterIndexConfig(
            emit_nodes_edges=emit_nodes_edges,
            emit_tokens=emit_tokens,
            emit_trivia=emit_trivia,
            emit_language_metadata=emit_language_metadata,
            enable_incremental=enable_incremental,
            match_limit=match_limit,
            allow_non_local_patterns=allow_non_local_patterns,
        )
        created_at = datetime.now(UTC)
        seen_languages: set[str] = set()
        warnings: list[str] = []
        if enable_incremental:
            warnings.append("Tree-sitter incremental parsing is not cached; full parse used.")

        for module in modules:
            warnings.extend(
                _process_module(
                    module=module,
                    repo=repo,
                    commit=commit,
                    buffers=buffers,
                    discovery=self._discovery,
                    config=config,
                    seen_languages=seen_languages,
                    created_at=created_at,
                )
            )

        return TreeSitterIndexResult(
            result=ExecutionResult.ok(warnings=tuple(warnings)),
            parse_manifest_rows=buffers.parse_manifest.to_reader(),
            captures_rows=buffers.captures.to_reader(),
            nodes_rows=buffers.nodes.to_reader(),
            edges_rows=buffers.edges.to_reader(),
            parse_errors_rows=buffers.parse_errors.to_reader(),
            tokens_rows=buffers.tokens.to_reader(),
            trivia_rows=buffers.trivia.to_reader(),
            language_metadata_rows=buffers.language_metadata.to_reader(),
            parse_manifest_row_count=buffers.parse_manifest.row_count,
            captures_row_count=buffers.captures.row_count,
            nodes_row_count=buffers.nodes.row_count,
            edges_row_count=buffers.edges.row_count,
            parse_errors_row_count=buffers.parse_errors.row_count,
            tokens_row_count=buffers.tokens.row_count,
            trivia_row_count=buffers.trivia.row_count,
            language_metadata_row_count=buffers.language_metadata.row_count,
        )


__all__ = ["TreeSitterIndexResult", "TreeSitterIndexStep"]
