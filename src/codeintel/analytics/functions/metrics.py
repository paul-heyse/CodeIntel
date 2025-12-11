"""Derive per-function complexity metrics and type hints from Python source files.

This module reads GOID metadata, walks Python ASTs to compute structural metrics,
and emits analytics tables used by downstream scoring and documentation tools.

Architecture
------------
This module follows the layered architecture:
- **Compute Layer**: Pure functions in `analytics.compute.functions`
- **Adapters**: Database I/O in `analytics.adapters.functions`
- **Orchestration**: This module coordinates between layers

The public API is stable.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import pandas as pd

from codeintel.analytics.adapters.base import DeleteScope
from codeintel.analytics.compute.functions import (
    ComplexityMetrics,
    compute_complexity,
)
from codeintel.analytics.compute.functions.loc import compute_loc
from codeintel.analytics.compute.functions.typedness import (
    ParamStats,
    TypednessFlags,
    compute_param_stats,
    compute_typedness_flags,
)
from codeintel.analytics.functions.config import (
    FunctionAnalyticsOptions,
    ProcessContext,
    ProcessState,
)
from codeintel.analytics.functions.parsing import parse_python_file
from codeintel.analytics.parsing.models import ParsedModule, SourceSpan
from codeintel.analytics.parsing.span_resolver import SpanResolutionError, resolve_span
from codeintel.analytics.parsing.validation import FunctionValidationReporter
from codeintel.analytics.utilities.datasets import (
    get_analytics_dataset_contract,
    insert_analytics_rows,
)
from codeintel.config import FunctionAnalyticsStepConfig
from codeintel.config.datasets import FunctionMetricsRow, FunctionTypesRow
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.pandera_schemas import validate_dataset_df
from codeintel.storage.sql.builder import ensure_schema

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
        "repo": ctx.cfg.repo,
        "commit": ctx.cfg.commit,
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
        "repo": ctx.cfg.repo,
        "commit": ctx.cfg.commit,
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
    abs_path = (state.cfg.repo_root / rel_path).resolve()
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

    abs_path = (state.cfg.repo_root / rel_path).resolve()
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


def _load_goids(
    gateway: StorageGateway, cfg: FunctionAnalyticsStepConfig
) -> dict[str, list[GoidRow]]:
    """Load function GOIDs from core.goids using Ibis.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    cfg
        Step configuration with repo and commit.

    Returns
    -------
    dict[str, list[GoidRow]]
        GOIDs grouped by relative file path.
    """
    tbl = gateway.ibis.table("core.goids")
    repo_filter = cast("Any", tbl.repo == cfg.repo)
    commit_filter = cast("Any", tbl.commit == cfg.commit)
    kind_filter = cast("Any", tbl.kind.isin(cast("Any", ["function", "method"])))
    expr = tbl.filter(repo_filter & commit_filter & kind_filter).select(
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
    df = expr.execute()

    if df.empty:
        log.info("No function GOIDs found for repo=%s commit=%s", cfg.repo, cfg.commit)
        return {}

    goids_by_file: dict[str, list[GoidRow]] = {}
    for record in df.to_dict(orient="records"):  # type: ignore[call-overload]
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


def persist_function_analytics(
    gateway: StorageGateway,
    cfg: FunctionAnalyticsStepConfig,
    result: FunctionAnalyticsResult,
) -> dict[str, int]:
    """
    Persist analytics rows and validation to DuckDB.

    Parameters
    ----------
    gateway : StorageGateway
        Storage gateway exposing the DuckDB connection.
    cfg : FunctionAnalyticsStepConfig
        Repository/commit context.
    result : FunctionAnalyticsResult
        Rows and validation to persist.

    Returns
    -------
    dict[str, int]
        Summary counts of persisted rows and validation.
    """
    con = gateway.con
    ensure_schema(con, "analytics.function_validation")
    scope = f"{cfg.repo}@{cfg.commit}"
    metrics_contract = get_analytics_dataset_contract(gateway, "analytics.function_metrics")
    types_contract = get_analytics_dataset_contract(gateway, "analytics.function_types")
    delete_scope = DeleteScope(repo=cfg.repo, commit=cfg.commit)
    metrics_rows = result.metrics_rows
    types_rows = result.types_rows

    def _validated_records(
        table_key: str, rows: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        if not rows:
            return []
        df = pd.DataFrame(rows)
        validated = validate_dataset_df(table_key, df)
        return validated.where(pd.notna(validated), None).to_dict(orient="records")

    validated_metrics = _validated_records(metrics_contract.table_key, list(metrics_rows))
    validated_types = _validated_records(types_contract.table_key, list(types_rows))
    insert_analytics_rows(
        gateway,
        metrics_contract,
        validated_metrics,
        delete_scope=delete_scope,
        scope=scope,
    )
    insert_analytics_rows(
        gateway,
        types_contract,
        validated_types,
        delete_scope=delete_scope,
        scope=scope,
    )
    result.reporter.flush(gateway)

    log.info(
        ("Function metrics/types build complete for repo=%s commit=%s: %d functions (missing=%d)"),
        cfg.repo,
        cfg.commit,
        result.metrics_count,
        result.validation_total,
    )

    return {
        "metrics_rows": result.metrics_count,
        "types_rows": result.types_count,
        "validation_total": result.validation_total,
        "validation_parse_failed": result.parse_failed_count,
        "validation_span_not_found": result.span_not_found_count,
    }


def compute_function_metrics_and_types(
    gateway: StorageGateway,
    cfg: FunctionAnalyticsStepConfig,
    *,
    options: FunctionAnalyticsOptions | None = None,
) -> dict[str, int]:
    """
    Populate function metrics and type coverage tables from GOID spans.

    Extended Summary
    ----------------
    For each function or method GOID in `core.goids`, the routine parses the
    corresponding Python file, derives structural metrics (LOC, complexity,
    nesting depth), and captures annotation coverage for parameters and return
    values. Outputs are written to `analytics.function_metrics` and
    `analytics.function_types`, enabling downstream risk scoring and typedness
    reporting.

    Parameters
    ----------
    gateway :
        StorageGateway providing the DuckDB connection with `core.goids`,
        `analytics.function_metrics`, and `analytics.function_types` tables available.
    cfg : FunctionAnalyticsStepConfig
        Repository metadata and file-system root used to locate source files.
    options : FunctionAnalyticsOptions | None
        Optional hooks for reusing parsed AST context and overriding the validation reporter.

    Notes
    -----
    - The function reads each source file once and reuses the parsed AST for all
      contained GOIDs.
    - Missing spans are recorded in `analytics.function_validation` to avoid
      silent drops; set `fail_on_missing_spans` to raise instead of warn.

    Raises
    ------
    ValueError
        If `fail_on_missing_spans` is enabled and any GOID span cannot be
        matched to an AST node or parsed file.

    Returns
    -------
    dict[str, int]
        Summary counts of emitted metrics/types and validation issues.
    """
    con = gateway.con
    ensure_schema(con, "analytics.function_metrics")
    ensure_schema(con, "analytics.function_types")
    ensure_schema(con, "analytics.function_validation")

    goids_by_file = _load_goids(gateway, cfg)
    if not goids_by_file:
        return {
            "metrics_rows": 0,
            "types_rows": 0,
            "validation_total": 0,
            "validation_parse_failed": 0,
            "validation_span_not_found": 0,
        }

    now = datetime.now(UTC)
    ctx = ProcessContext(cfg=cfg, now=now)

    opts = options or FunctionAnalyticsOptions()
    reporter = opts.validation_reporter or FunctionValidationReporter(cfg.repo, cfg.commit)
    span_index = _build_span_index(goids_by_file, cfg.repo_root)

    # Use pre-loaded AST data if available (from context or direct data)
    if opts.has_ast_data():
        result = _build_function_analytics_from_ast_data(
            goids_by_file=goids_by_file,
            process_ctx=ctx,
            ast_data=opts,
            span_index=span_index,
            reporter=reporter,
        )
    else:
        parsed_cache: dict[str, ParsedModule | None] = {}
        state = ProcessState(
            cfg=cfg,
            cache=parsed_cache,
            span_index=span_index,
            reporter=reporter,
            ctx=ctx,
        )
        result = build_function_analytics(goids_by_file=goids_by_file, state=state)

    summary = persist_function_analytics(gateway, cfg, result)
    if cfg.fail_on_missing_spans and result.validation_total:
        message = (
            f"Missing analytics for {result.validation_total} functions; "
            "see analytics.function_validation"
        )
        raise ValueError(message)

    if result.validation_total:
        log.warning(
            "Function validation gaps: parse_failed=%d span_not_found=%d",
            result.parse_failed_count,
            result.span_not_found_count,
        )
    return summary
