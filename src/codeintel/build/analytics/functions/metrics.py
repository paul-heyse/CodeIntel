"""Derive per-function typing metadata from Python source files.

This module reads GOID metadata, walks Python ASTs to compute type annotation
statistics, and emits analytics rows for downstream consumption.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

import polars as pl

from codeintel.build.analytics.compute.functions.typedness import compute_param_stats
from codeintel.build.analytics.functions.config import (
    FunctionAnalyticsOptions,
    ProcessContext,
    ProcessState,
)
from codeintel.build.analytics.functions.parsing import parse_python_file
from codeintel.build.analytics.parsing.span_resolver import SpanResolutionError, resolve_span
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.core.parsing import SourceSpan
from codeintel.core.query_results import coerce_int, coerce_optional_int
from codeintel.core.validation.reporters import FunctionValidationReporter

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.tabular.types import InferableTabularInput
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.parsing import ParsedModule
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsFunctionTypesRow as FunctionTypesRow,
    )

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FunctionAnalyticsResult:
    """Pure analysis output for function typing plus validation."""

    types_rows: list[FunctionTypesRow]
    reporter: FunctionValidationReporter

    @property
    def types_count(self) -> int:
        """Number of function_types rows produced."""
        return len(self.types_rows)

    @property
    def validation_total(self) -> int:
        """Total validation findings recorded."""
        return self.reporter.total

    @property
    def parse_failed_count(self) -> int:
        """Count of parse_failed validation issues."""
        return self.reporter.parse_failed

    @property
    def span_not_found_count(self) -> int:
        """Count of span_not_found validation issues."""
        return self.reporter.span_not_found


@dataclass(frozen=True)
class FunctionMeta:
    """Minimal metadata for a function GOID."""

    goid: int
    urn: str
    language: str
    kind: str
    qualname: str
    start_line: int
    end_line: int
    rel_path: str


class GoidRow(TypedDict):
    """Row structure for function GOIDs pulled from tabular input."""

    goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    language: str
    kind: str
    qualname: str
    start_line: int
    end_line: int | None


def _type_row_from_node(
    meta: FunctionMeta,
    node: ast.AST,
    ctx: ProcessContext,
) -> FunctionTypesRow | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    param_stats = compute_param_stats(node)

    return {
        "function_goid_h128": meta.goid,
        "urn": meta.urn,
        "repo": ctx.snapshot.repo,
        "commit": ctx.snapshot.commit,
        "rel_path": meta.rel_path,
        "language": meta.language,
        "kind": meta.kind,
        "qualname": meta.qualname,
        "start_line": meta.start_line,
        "end_line": meta.end_line,
        "total_params": param_stats.total_params,
        "return_type": param_stats.return_type,
        "type_comment": None,
        "param_types": param_stats.param_types,
        "created_at": ctx.now,
    }


def analyze_function(
    meta: FunctionMeta,
    parsed: ParsedModule,
    ctx: ProcessContext,
) -> FunctionTypesRow | None:
    """
    Derive a typing row for a single function span.

    Returns
    -------
    FunctionTypesRow | None
        Types row when a matching AST node is found; otherwise None.
    """
    node = parsed.span_index.lookup(meta.start_line, meta.end_line)
    if node is None:
        return None
    return _type_row_from_node(meta, node, ctx)


def _get_parsed_module(rel_path: str, *, state: ProcessState) -> ParsedModule | None:
    if rel_path in state.cache:
        return state.cache[rel_path]
    abs_path = (state.snapshot.repo_root / rel_path).resolve()
    try:
        parsed = parse_python_file(abs_path)
    except (OSError, ValueError):
        state.cache[rel_path] = None
        return None
    state.cache[rel_path] = parsed
    return parsed


def _process_file_functions(
    rel_path: str,
    fun_rows: list[GoidRow],
    state: ProcessState,
) -> list[FunctionTypesRow]:
    types_rows: list[FunctionTypesRow] = []

    abs_path = (state.snapshot.repo_root / rel_path).resolve()
    parsed = _get_parsed_module(rel_path, state=state)
    if parsed is None:
        detail = f"File missing or unparsable: {abs_path}"
        for row in fun_rows:
            state.reporter.record(
                function_goid_h128=int(row["goid_h128"]),
                rel_path=row["rel_path"],
                qualname=row["qualname"],
                issue="parse_failed",
                detail=detail,
            )
        log.warning("Skipping file for function typing: %s", abs_path)
        return types_rows

    for info in fun_rows:
        meta = _meta_from_goid_row(info)
        try:
            span_result = resolve_span(
                function_goid_h128=meta.goid,
                span_index=state.span_index,
            )
        except SpanResolutionError as exc:
            state.reporter.record(
                function_goid_h128=meta.goid,
                rel_path=meta.rel_path,
                qualname=meta.qualname,
                issue="span_not_found",
                detail=str(exc),
            )
            continue

        node = parsed.span_index.lookup(
            span_result.span.start_line,
            span_result.span.end_line,
        )
        if node is None or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            state.reporter.record(
                function_goid_h128=meta.goid,
                rel_path=meta.rel_path,
                qualname=meta.qualname,
                issue="span_not_found",
                detail=(
                    f"AST node not found for span "
                    f"{span_result.span.start_line}-{span_result.span.end_line}"
                ),
            )
            continue
        row = _type_row_from_node(meta, node, state.ctx)
        if row is None:
            state.reporter.record(
                function_goid_h128=meta.goid,
                rel_path=meta.rel_path,
                qualname=meta.qualname,
                issue="span_not_found",
                detail="Span matched a non-function node",
            )
            continue
        types_rows.append(row)

    return types_rows


def build_function_analytics(
    *,
    goids_by_file: dict[str, list[GoidRow]],
    state: ProcessState,
) -> FunctionAnalyticsResult:
    """
    Build typing rows for all GOIDs using a pure orchestration path.

    Parameters
    ----------
    goids_by_file : dict[str, list[GoidRow]]
        Mapping from rel_path to GOID rows.
    state : ProcessState
        Shared parser/ctx/cache state for processing.

    Returns
    -------
    FunctionAnalyticsResult
        Aggregated types rows and validation findings.
    """
    types_rows: list[FunctionTypesRow] = []

    for rel_path, fun_rows in goids_by_file.items():
        types_rows.extend(
            _process_file_functions(
                rel_path=rel_path,
                fun_rows=fun_rows,
                state=state,
            )
        )

    return FunctionAnalyticsResult(
        types_rows=types_rows,
        reporter=state.reporter,
    )


def _load_goids_from_frame(
    goids_frame: pl.DataFrame,
    snapshot: SnapshotRef,
) -> dict[str, list[GoidRow]]:
    """Load function GOIDs from a polars frame.

    Parameters
    ----------
    goids_frame
        Tabular ``core.goids`` frame.
    snapshot
        Repository and commit identifiers.

    Returns
    -------
    dict[str, list[GoidRow]]
        GOIDs grouped by relative file path.
    """
    required = {
        "goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "language",
        "kind",
        "qualname",
        "start_line",
        "end_line",
    }
    missing = required.difference(goids_frame.columns)
    if missing:
        log.warning("core.goids is missing columns: %s", ", ".join(sorted(missing)))
        return {}

    filtered = (
        goids_frame.lazy()
        .filter(
            (pl.col("repo") == snapshot.repo)
            & (pl.col("commit") == snapshot.commit)
            & (pl.col("kind").is_in(["function", "method"]))
        )
        .select(list(required))
        .collect()
    )

    if filtered.is_empty():
        log.info("No function GOIDs found for repo=%s commit=%s", snapshot.repo, snapshot.commit)
        return {}

    goids_by_file: dict[str, list[GoidRow]] = {}
    for record in filtered.iter_rows(named=True):
        rel_path_raw = record.get("rel_path")
        rel_path = str(rel_path_raw).replace("\\", "/")
        goid_row: GoidRow = {
            "goid_h128": coerce_int(record.get("goid_h128"), ctx="goid_h128"),
            "urn": str(record.get("urn")),
            "repo": str(record.get("repo")),
            "commit": str(record.get("commit")),
            "rel_path": rel_path,
            "language": str(record.get("language")),
            "kind": str(record.get("kind")),
            "qualname": str(record.get("qualname")),
            "start_line": coerce_int(record.get("start_line"), ctx="start_line"),
            "end_line": coerce_optional_int(record.get("end_line"), ctx="end_line"),
        }
        goids_by_file.setdefault(rel_path, []).append(goid_row)
    return goids_by_file


def _meta_from_goid_row(info: GoidRow) -> FunctionMeta:
    end_line_raw = info["end_line"]
    end_line = (
        coerce_int(end_line_raw, ctx="end_line")
        if end_line_raw is not None
        else coerce_int(info["start_line"], ctx="start_line")
    )
    return FunctionMeta(
        goid=coerce_int(info["goid_h128"], ctx="goid_h128"),
        urn=str(info["urn"]),
        language=str(info["language"]),
        kind=str(info["kind"]),
        qualname=str(info["qualname"]),
        start_line=coerce_int(info["start_line"], ctx="start_line"),
        end_line=end_line,
        rel_path=str(info["rel_path"]),
    )


def _build_span_index(
    goids_by_file: dict[str, list[GoidRow]], repo_root: Path
) -> dict[int, SourceSpan]:
    span_index: dict[int, SourceSpan] = {}
    for rel_path, rows in goids_by_file.items():
        abs_path = (repo_root / rel_path).resolve()
        for row in rows:
            end_line_raw = row["end_line"]
            end_line = (
                coerce_int(end_line_raw, ctx="end_line")
                if end_line_raw is not None
                else coerce_int(row["start_line"], ctx="start_line")
            )
            span_index[coerce_int(row["goid_h128"], ctx="goid_h128")] = SourceSpan(
                path=abs_path,
                start_line=coerce_int(row["start_line"], ctx="start_line"),
                start_col=0,
                end_line=end_line,
                end_col=0,
            )
    return span_index


def _compute_from_goids(
    goids_by_file: dict[str, list[GoidRow]],
    snapshot: SnapshotRef,
    *,
    options: FunctionAnalyticsOptions | None,
) -> FunctionAnalyticsResult:
    if not goids_by_file:
        return FunctionAnalyticsResult(
            types_rows=[],
            reporter=FunctionValidationReporter(snapshot.repo, snapshot.commit),
        )

    now = datetime.now(UTC)
    ctx = ProcessContext(snapshot=snapshot, now=now)

    opts = options or FunctionAnalyticsOptions()
    reporter = opts.validation_reporter or FunctionValidationReporter(
        snapshot.repo, snapshot.commit
    )
    span_index = _build_span_index(goids_by_file, snapshot.repo_root)

    if opts.has_ast_data():
        return _build_function_analytics_from_ast_data(
            goids_by_file=goids_by_file,
            process_ctx=ctx,
            ast_data=opts,
            span_index=span_index,
            reporter=reporter,
        )

    parsed_cache: dict[str, ParsedModule | None] = {}
    state = ProcessState(
        snapshot=snapshot,
        cache=parsed_cache,
        span_index=span_index,
        reporter=reporter,
        ctx=ctx,
    )
    return build_function_analytics(goids_by_file=goids_by_file, state=state)


def compute_function_analytics_result_from_tabular(
    goids_input: InferableTabularInput,
    snapshot: SnapshotRef,
    *,
    options: FunctionAnalyticsOptions | None = None,
) -> FunctionAnalyticsResult:
    """Compute function analytics result from tabular GOID inputs.

    Parameters
    ----------
    goids_input
        Tabular input for ``core.goids``.
    snapshot
        Repository and commit identifiers.
    options
        Optional hooks for reusing parsed AST context and overriding the
        validation reporter.

    Returns
    -------
    FunctionAnalyticsResult
        Container with types_rows and validation reporter.
    """
    goids_frame = tabular_to_lazyframe(goids_input).collect()
    goids_by_file = _load_goids_from_frame(goids_frame, snapshot)
    return _compute_from_goids(goids_by_file, snapshot, options=options)


def _build_function_analytics_from_ast_data(
    *,
    goids_by_file: dict[str, list[GoidRow]],
    process_ctx: ProcessContext,
    ast_data: FunctionAnalyticsOptions,
    span_index: dict[int, SourceSpan],
    reporter: FunctionValidationReporter,
) -> FunctionAnalyticsResult:
    """Build function analytics from pre-loaded AST data.

    Parameters
    ----------
    goids_by_file
        GOIDs grouped by file path.
    process_ctx
        Processing context with config and timestamp.
    ast_data
        Options containing AST map and missing GOIDs.
    span_index
        Mapping of GOID to source span.
    reporter
        Validation reporter for issues.

    Returns
    -------
    FunctionAnalyticsResult
        Computed types rows and validation reporter.
    """
    types_rows: list[FunctionTypesRow] = []
    ast_map = ast_data.get_ast_map()
    missing_goids = ast_data.get_missing_goids()

    for fun_rows in goids_by_file.values():
        for info in fun_rows:
            meta = _meta_from_goid_row(info)
            try:
                resolve_span(function_goid_h128=meta.goid, span_index=span_index)
            except SpanResolutionError as exc:
                reporter.record(
                    function_goid_h128=meta.goid,
                    rel_path=meta.rel_path,
                    qualname=meta.qualname,
                    issue="span_not_found",
                    detail=str(exc),
                )
                continue
            ast_info = ast_map.get(meta.goid)
            if ast_info is None:
                detail = (
                    "missing AST in shared context" if meta.goid in missing_goids else "missing AST"
                )
                reporter.record(
                    function_goid_h128=meta.goid,
                    rel_path=meta.rel_path,
                    qualname=meta.qualname,
                    issue="span_not_found",
                    detail=detail,
                )
                continue
            row = _type_row_from_node(meta, ast_info.node, process_ctx)
            if row is None:
                reporter.record(
                    function_goid_h128=meta.goid,
                    rel_path=meta.rel_path,
                    qualname=meta.qualname,
                    issue="span_not_found",
                    detail="context AST resolution failed",
                )
                continue
            types_rows.append(row)

    return FunctionAnalyticsResult(
        types_rows=types_rows,
        reporter=reporter,
    )


def compute_function_analytics_result(
    goids_input: InferableTabularInput,
    snapshot: SnapshotRef,
    *,
    options: FunctionAnalyticsOptions | None = None,
) -> FunctionAnalyticsResult:
    """
    Compute pure function analytics result without persisting.

    This is the pure compute path for Hamilton DAG-visible I/O. It returns
    rows ready for materialization via SaveToDecorator/ArrowDatasetSaver.

    Parameters
    ----------
    goids_input
        Tabular input for ``core.goids``.
    snapshot
        Repository and commit identifiers.
    options
        Optional hooks for reusing parsed AST context and overriding the
        validation reporter.

    Returns
    -------
    FunctionAnalyticsResult
        Container with types_rows and validation reporter.
    """
    return compute_function_analytics_result_from_tabular(
        goids_input,
        snapshot,
        options=options,
    )


def compute_function_analytics_result_from_table(
    goids_input: InferableTabularInput,
    snapshot: SnapshotRef,
    *,
    options: FunctionAnalyticsOptions | None = None,
) -> FunctionAnalyticsResult:
    """Backward-compatible wrapper around tabular analytics computation.

    Returns
    -------
    FunctionAnalyticsResult
        Container with types_rows and validation reporter.
    """
    return compute_function_analytics_result_from_tabular(
        goids_input,
        snapshot,
        options=options,
    )
