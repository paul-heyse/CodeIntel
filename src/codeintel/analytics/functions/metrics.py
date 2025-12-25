"""Derive per-function complexity metrics and type hints from Python source files.

This module reads GOID metadata, walks Python ASTs to compute structural metrics,
and emits analytics tables used by downstream scoring and documentation tools.

Architecture
------------
This module follows the layered architecture:
- **Compute Layer**: Pure functions in `analytics.compute.functions`
- **GOID Loading**: `analytics.compute.functions.goids.FunctionGoidLoader`
- **Orchestration**: This module coordinates between layers

The public API is stable.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict, cast

import pandas as pd

from codeintel.analytics.compute.functions import (
    compute_complexity,
)
from codeintel.analytics.compute.functions.loc import compute_loc
from codeintel.analytics.compute.functions.typedness import (
    compute_param_stats,
    compute_typedness_flags,
)
from codeintel.analytics.functions.config import (
    FunctionAnalyticsOptions,
    ProcessContext,
    ProcessState,
)
from codeintel.analytics.functions.parsing import parse_python_file
from codeintel.analytics.parsing.span_resolver import SpanResolutionError, resolve_span
from codeintel.analytics.utilities.dataframe import to_records
from codeintel.core.ibis_typing import and_predicates, isin_values
from codeintel.core.parsing import SourceSpan
from codeintel.core.validation.reporters import FunctionValidationReporter
from codeintel.storage.gateway import ibis_facade

if TYPE_CHECKING:
    from pathlib import Path

    import ibis.expr.types as ir

    from codeintel.analytics.compute.functions import (
        ComplexityMetrics,
    )
    from codeintel.analytics.compute.functions.typedness import (
        ParamStats,
        TypednessFlags,
    )
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.parsing import ParsedModule
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsFunctionMetricsRow as FunctionMetricsRow,
    )
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsFunctionTypesRow as FunctionTypesRow,
    )
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FunctionAnalyticsResult:
    """Pure analysis output for function metrics/types plus validation."""

    metrics_rows: list[FunctionMetricsRow]
    types_rows: list[FunctionTypesRow]
    reporter: FunctionValidationReporter

    @property
    def metrics_count(self) -> int:
        """Number of function_metrics rows produced."""
        return len(self.metrics_rows)

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


@dataclass(frozen=True)
class FunctionDerived:
    """Derived structural flags for a function body."""

    is_async: bool
    is_generator: bool
    complexity_bucket: str
    stmt_count: int
    decorator_count: int
    has_docstring: bool
    typedness: TypednessFlags


class GoidRow(TypedDict):
    """Row structure for function GOIDs pulled from DuckDB."""

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


def _compute_loc(lines: list[str], start_line: int, end_line: int) -> tuple[int, int]:
    """Compute lines of code using the compute layer.

    Parameters
    ----------
    lines
        Source lines.
    start_line
        Start line (1-indexed).
    end_line
        End line (1-indexed).

    Returns
    -------
    tuple[int, int]
        Physical LOC and logical LOC.
    """
    loc_metrics = compute_loc(lines, start_line, end_line)
    return loc_metrics.physical, loc_metrics.logical


def _derive_function_flags(
    metrics: ComplexityMetrics,
    param_stats: ParamStats,
) -> FunctionDerived:
    typedness = compute_typedness_flags(
        total_params=param_stats.total_params,
        annotated_params=param_stats.annotated_params,
        has_return_annotation=param_stats.has_return_annotation,
    )
    return FunctionDerived(
        is_async=metrics.is_async,
        is_generator=metrics.is_generator,
        complexity_bucket=metrics.complexity_bucket,
        stmt_count=metrics.stmt_count,
        decorator_count=metrics.decorator_count,
        has_docstring=metrics.has_docstring,
        typedness=typedness,
    )


def _function_rows_from_node(
    meta: FunctionMeta,
    node: ast.AST,
    lines: list[str],
    ctx: ProcessContext,
) -> tuple[FunctionMetricsRow, FunctionTypesRow] | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    loc, logical_loc = _compute_loc(lines, meta.start_line, meta.end_line)

    param_stats = compute_param_stats(node)
    complexity = compute_complexity(node)
    derived = _derive_function_flags(complexity, param_stats)
    typedness = derived.typedness
    return_type_source = "annotation" if param_stats.has_return_annotation else "unknown"

    metrics_row: FunctionMetricsRow = {
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
        "loc": loc,
        "logical_loc": logical_loc,
        "param_count": param_stats.param_count,
        "positional_params": param_stats.positional_params,
        "keyword_only_params": param_stats.keyword_only_params,
        "has_varargs": param_stats.has_varargs,
        "has_varkw": param_stats.has_varkw,
        "is_async": derived.is_async,
        "is_generator": derived.is_generator,
        "return_count": complexity.return_count,
        "yield_count": complexity.yield_count,
        "raise_count": complexity.raise_count,
        "cyclomatic_complexity": complexity.cyclomatic,
        "max_nesting_depth": complexity.max_nesting_depth,
        "stmt_count": derived.stmt_count,
        "decorator_count": derived.decorator_count,
        "has_docstring": derived.has_docstring,
        "complexity_bucket": derived.complexity_bucket,
        "created_at": ctx.now,
    }

    types_row: FunctionTypesRow = {
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
        "annotated_params": param_stats.annotated_params,
        "unannotated_params": typedness.unannotated_params,
        "param_typed_ratio": typedness.param_typed_ratio,
        "has_return_annotation": param_stats.has_return_annotation,
        "return_type": param_stats.return_type,
        "return_type_source": return_type_source,
        "type_comment": None,
        "param_types": param_stats.param_types,
        "fully_typed": typedness.fully_typed,
        "partial_typed": typedness.partial_typed,
        "untyped": typedness.untyped,
        "typedness_bucket": typedness.typedness_bucket,
        "typedness_source": typedness.typedness_source,
        "created_at": ctx.now,
    }
    return metrics_row, types_row


def analyze_function(
    meta: FunctionMeta,
    parsed: ParsedModule,
    ctx: ProcessContext,
) -> tuple[FunctionMetricsRow, FunctionTypesRow] | None:
    """
    Derive analytics rows for a single function span.

    Returns
    -------
    tuple[FunctionMetricsRow, FunctionTypesRow] | None
        Metrics row and types row when a matching AST node is found; otherwise
        None when the span cannot be resolved.
    """
    node = parsed.span_index.lookup(meta.start_line, meta.end_line)
    if node is None:
        return None
    return _function_rows_from_node(meta, node, list(parsed.lines), ctx)


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
) -> tuple[list[FunctionMetricsRow], list[FunctionTypesRow]]:
    metrics_rows: list[FunctionMetricsRow] = []
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
        log.warning("Skipping file for function analytics: %s", abs_path)
        return metrics_rows, types_rows

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
        rows = _function_rows_from_node(meta, node, list(parsed.lines), state.ctx)
        if rows is None:
            state.reporter.record(
                function_goid_h128=meta.goid,
                rel_path=meta.rel_path,
                qualname=meta.qualname,
                issue="span_not_found",
                detail="Span matched a non-function node",
            )
            continue
        metrics_row, types_row = rows
        metrics_rows.append(metrics_row)
        types_rows.append(types_row)

    return metrics_rows, types_rows


def build_function_analytics(
    *,
    goids_by_file: dict[str, list[GoidRow]],
    state: ProcessState,
) -> FunctionAnalyticsResult:
    """
    Build analytics rows for all GOIDs using a pure orchestration path.

    Parameters
    ----------
    goids_by_file : dict[str, list[GoidRow]]
        Mapping from rel_path to GOID rows.
    state : ProcessState
        Shared parser/ctx/cache state for processing.

    Returns
    -------
    FunctionAnalyticsResult
        Aggregated metrics, types, and validation findings.
    """
    metrics_rows: list[FunctionMetricsRow] = []
    types_rows: list[FunctionTypesRow] = []

    for rel_path, fun_rows in goids_by_file.items():
        file_metrics, file_types = _process_file_functions(
            rel_path=rel_path,
            fun_rows=fun_rows,
            state=state,
        )
        metrics_rows.extend(file_metrics)
        types_rows.extend(file_types)

    return FunctionAnalyticsResult(
        metrics_rows=metrics_rows,
        types_rows=types_rows,
        reporter=state.reporter,
    )


def _load_goids_from_table(
    goids_table: ir.Table,
    snapshot: SnapshotRef,
) -> dict[str, list[GoidRow]]:
    """Load function GOIDs from an Ibis table expression.

    Parameters
    ----------
    goids_table
        Ibis table expression for ``core.goids``.
    snapshot
        Repository and commit identifiers.

    Returns
    -------
    dict[str, list[GoidRow]]
        GOIDs grouped by relative file path.
    """
    scoped = goids_table.filter(
        and_predicates(
            goids_table.repo == snapshot.repo,
            goids_table.commit == snapshot.commit,
            isin_values(goids_table.kind, ["function", "method"]),
        )
    )
    expr = scoped.select(
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
    )
    df = cast("pd.DataFrame", expr.execute())

    if df.empty:
        log.info("No function GOIDs found for repo=%s commit=%s", snapshot.repo, snapshot.commit)
        return {}

    goids_by_file: dict[str, list[GoidRow]] = {}
    for record in to_records(df):
        rel_path = str(record["rel_path"]).replace("\\", "/")
        goid_row: GoidRow = {
            "goid_h128": int(record["goid_h128"]),
            "urn": str(record["urn"]),
            "repo": str(record["repo"]),
            "commit": str(record["commit"]),
            "rel_path": rel_path,
            "language": str(record["language"]),
            "kind": str(record["kind"]),
            "qualname": str(record["qualname"]),
            "start_line": int(record["start_line"]),
            "end_line": int(record["end_line"]) if record["end_line"] is not None else None,
        }
        goids_by_file.setdefault(rel_path, []).append(goid_row)
    return goids_by_file


def _load_goids(gateway: StorageGateway, snapshot: SnapshotRef) -> dict[str, list[GoidRow]]:
    """Load function GOIDs from core.goids using Ibis.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository and commit identifiers.

    Returns
    -------
    dict[str, list[GoidRow]]
        GOIDs grouped by relative file path.
    """
    table = ibis_facade.table(gateway, "core.goids")
    return _load_goids_from_table(table, snapshot)


def _meta_from_goid_row(info: GoidRow) -> FunctionMeta:
    end_line_raw = info["end_line"]
    end_line = int(end_line_raw) if end_line_raw is not None else int(info["start_line"])
    return FunctionMeta(
        goid=int(info["goid_h128"]),
        urn=str(info["urn"]),
        language=str(info["language"]),
        kind=str(info["kind"]),
        qualname=str(info["qualname"]),
        start_line=int(info["start_line"]),
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
            end_line = int(end_line_raw) if end_line_raw is not None else int(row["start_line"])
            span_index[int(row["goid_h128"])] = SourceSpan(
                path=abs_path,
                start_line=int(row["start_line"]),
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
            metrics_rows=[],
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


def compute_function_analytics_result_from_table(
    goids_table: ir.Table,
    snapshot: SnapshotRef,
    *,
    options: FunctionAnalyticsOptions | None = None,
) -> FunctionAnalyticsResult:
    """Compute function analytics result from a GOIDs table expression.

    Parameters
    ----------
    goids_table
        Ibis table expression for ``core.goids``.
    snapshot
        Repository and commit identifiers.
    options
        Optional hooks for reusing parsed AST context and overriding the
        validation reporter.

    Returns
    -------
    FunctionAnalyticsResult
        Container with metrics_rows, types_rows, and validation reporter.
    """
    goids_by_file = _load_goids_from_table(goids_table, snapshot)
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
        Computed analytics rows and validation reporter.
    """
    metrics_rows: list[FunctionMetricsRow] = []
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
            rows = _function_rows_from_node(meta, ast_info.node, ast_info.lines, process_ctx)
            if rows is None:
                reporter.record(
                    function_goid_h128=meta.goid,
                    rel_path=meta.rel_path,
                    qualname=meta.qualname,
                    issue="span_not_found",
                    detail="context AST resolution failed",
                )
                continue
            metrics_row, types_row = rows
            metrics_rows.append(metrics_row)
            types_rows.append(types_row)

    return FunctionAnalyticsResult(
        metrics_rows=metrics_rows,
        types_rows=types_rows,
        reporter=reporter,
    )


def compute_function_analytics_result(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    options: FunctionAnalyticsOptions | None = None,
) -> FunctionAnalyticsResult:
    """
    Compute pure function analytics result without persisting.

    This is the pure compute path for Hamilton DAG-visible I/O. It returns
    rows ready for materialization via SaveToDecorator/DuckDBRowsSaver.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection with `core.goids` table.
    snapshot
        Repository and commit identifiers.
    options
        Optional hooks for reusing parsed AST context and overriding the
        validation reporter.

    Returns
    -------
    FunctionAnalyticsResult
        Container with metrics_rows, types_rows, and validation reporter.
    """
    goids_by_file = _load_goids(gateway, snapshot)
    return _compute_from_goids(goids_by_file, snapshot, options=options)
