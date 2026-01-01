"""Tree-sitter ingestion step for query-pack captures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.core.columnar.rows import (
    ColumnarRowBuffer,
    ColumnarRows,
    columnar_buffer_for_table_key,
)
from codeintel.core.schemas.generated_rows.core import (
    CoreParseManifestRow,
    CoreTsCapturesRow,
    CoreTsParseErrorsRow,
)
from codeintel.ingestion.compute.base import BaseExtractStep
from codeintel.ingestion.infrastructure.cst_utils import LineIndexedSource
from codeintel.ingestion.tree_sitter.registry import language_for_path
from codeintel.ingestion.tree_sitter.runner import (
    TreeSitterParseError,
    TreeSitterParseResult,
    run_tree_sitter,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord

PARSE_MANIFEST_TABLE_KEY = "core.parse_manifest"
TS_CAPTURES_TABLE_KEY = "core.ts_captures"
TS_PARSE_ERRORS_TABLE_KEY = "core.ts_parse_errors"
TREE_SITTER_PRODUCER = "tree_sitter"


@dataclass(frozen=True)
class TreeSitterIndexResult:
    """Result bundle for tree-sitter query execution."""

    result: ExecutionResult
    parse_manifest_rows: ColumnarRows = field(default_factory=dict)
    captures_rows: ColumnarRows = field(default_factory=dict)
    parse_errors_rows: ColumnarRows = field(default_factory=dict)
    parse_manifest_row_count: int = 0
    captures_row_count: int = 0
    parse_errors_row_count: int = 0


@dataclass(frozen=True, slots=True)
class _ParseManifestContext:
    repo: str
    commit: str
    rel_path: str
    source_index: LineIndexedSource


@dataclass(slots=True)
class _TreeSitterBuffers:
    parse_manifest: ColumnarRowBuffer
    captures: ColumnarRowBuffer
    parse_errors: ColumnarRowBuffer


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
        error_kind=error_kind,
        error_message=error_message,
        error_line=error_line,
        error_col=error_col,
        error_snippet=error_snippet,
    )


def _build_buffers() -> _TreeSitterBuffers:
    return _TreeSitterBuffers(
        parse_manifest=columnar_buffer_for_table_key(PARSE_MANIFEST_TABLE_KEY),
        captures=columnar_buffer_for_table_key(TS_CAPTURES_TABLE_KEY),
        parse_errors=columnar_buffer_for_table_key(TS_PARSE_ERRORS_TABLE_KEY),
    )


def _append_tree_sitter_rows(
    *,
    buffers: _TreeSitterBuffers,
    module: ModuleRecord,
    parse_result: TreeSitterParseResult,
    repo: str,
    commit: str,
) -> None:
    buffers.captures.extend(
        [
            CoreTsCapturesRow(
                repo=repo,
                commit=commit,
                rel_path=module.rel_path,
                language=parse_result.language,
                query_pack=capture.query_pack,
                capture_name=capture.capture_name,
                start_byte=capture.start_byte,
                end_byte=capture.end_byte,
                start_row=capture.start_row,
                start_col=capture.start_col,
                end_row=capture.end_row,
                end_col=capture.end_col,
                node_type=capture.node_type,
                text_preview=capture.text_preview,
                extras=capture.extras,
            )
            for capture in parse_result.captures
        ]
    )
    buffers.parse_errors.extend(
        [
            CoreTsParseErrorsRow(
                repo=repo,
                commit=commit,
                rel_path=module.rel_path,
                language=parse_result.language,
                error_type=error.error_type,
                message=error.message,
                start_byte=error.start_byte,
                end_byte=error.end_byte,
                start_row=error.start_row,
                start_col=error.start_col,
                end_row=error.end_row,
                end_col=error.end_col,
                text_preview=error.text_preview,
            )
            for error in parse_result.errors
        ]
    )


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
) -> list[str]:
    language = language_for_path(module.file_path)
    if language is None:
        return []

    source = discovery.read_module_source(module)
    if source is None:
        return [f"Tree-sitter skipped {module.rel_path}: unreadable source"]

    source_bytes = source.encode("utf-8")
    source_index = LineIndexedSource(source, source_bytes)
    context = _ParseManifestContext(
        repo=repo,
        commit=commit,
        rel_path=module.rel_path,
        source_index=source_index,
    )
    try:
        parse_result = run_tree_sitter(language=language, source_bytes=source_bytes)
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

    buffers.parse_manifest.append(
        _parse_manifest_row(
            context,
            parse_ok=parse_result.parse_ok,
            error=parse_result.errors[0] if parse_result.errors else None,
        )
    )
    _append_tree_sitter_rows(
        buffers=buffers,
        module=module,
        parse_result=parse_result,
        repo=repo,
        commit=commit,
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

        warnings: list[str] = []

        for module in modules:
            warnings.extend(
                _process_module(
                    module=module,
                    repo=repo,
                    commit=commit,
                    buffers=buffers,
                    discovery=self._discovery,
                )
            )

        return TreeSitterIndexResult(
            result=ExecutionResult.ok(warnings=tuple(warnings)),
            parse_manifest_rows=buffers.parse_manifest.data,
            captures_rows=buffers.captures.data,
            parse_errors_rows=buffers.parse_errors.data,
            parse_manifest_row_count=buffers.parse_manifest.row_count,
            captures_row_count=buffers.captures.row_count,
            parse_errors_row_count=buffers.parse_errors.row_count,
        )


__all__ = ["TreeSitterIndexResult", "TreeSitterIndexStep"]
