"""
Derive per-function complexity metrics and type hints from Python source files.

This module reads GOID metadata, walks Python ASTs to compute structural metrics,
and emits analytics tables used by downstream scoring and documentation tools.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from logging import Logger, LoggerAdapter
from typing import TypedDict

from codeintel.analytics.function_parsing import (
    FunctionParserRegistry,
    ParsedFile,
    ParsedFileLoader,
    get_parsed_file,
)
from codeintel.analytics.span_resolver import resolve_span
from codeintel.config.models import FunctionAnalyticsConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.ingestion.common import run_batch
from codeintel.models.rows import FunctionValidationRow, function_validation_row_to_tuple
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

COMPLEXITY_LOW = 5
COMPLEXITY_MEDIUM = 10
SKIP_PARAM_NAMES = {"self", "cls"}


@dataclass(frozen=True)
class FunctionAnalyticsResult:
    """Pure analysis output for function metrics/types plus validation."""

    metrics_rows: list[tuple]
    types_rows: list[tuple]
    validation: list[ValidationIssue]

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
        return len(self.validation)

    @property
    def parse_failed_count(self) -> int:
        """Count of parse_failed validation issues."""
        return sum(1 for issue in self.validation if issue.issue == "parse_failed")

    @property
    def span_not_found_count(self) -> int:
        """Count of span_not_found validation issues."""
        return sum(1 for issue in self.validation if issue.issue == "span_not_found")


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
class ParamStats:
    """Parameter and return annotation statistics for a function."""

    param_count: int
    positional_params: int
    keyword_only_params: int
    has_varargs: bool
    has_varkw: bool
    total_params: int
    annotated_params: int
    param_types: dict[str, str | None]
    has_return_annotation: bool
    return_type: str | None


@dataclass(frozen=True)
class TypednessFlags:
    """Typedness summary flags derived from annotations."""

    param_typed_ratio: float
    unannotated_params: int
    fully_typed: bool
    partial_typed: bool
    untyped: bool
    typedness_bucket: str
    typedness_source: str


@dataclass(frozen=True)
class ProcessContext:
    """Shared context for building analytics rows."""

    cfg: FunctionAnalyticsConfig
    now: datetime


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


@dataclass
class ProcessState:
    """Mutable state shared across per-file processing."""

    cfg: FunctionAnalyticsConfig
    cache: dict[str, ParsedFile | None]
    parser: ParsedFileLoader
    ctx: ProcessContext


@dataclass(frozen=True)
class ValidationIssue:
    """Validation finding for a GOID that could not be processed."""

    rel_path: str
    qualname: str
    issue: str
    detail: str | None


@dataclass
class ValidationReporter:
    """Emit structured validation counters to logging or a metrics sink."""

    emit_counter: Callable[[str, int], None] | None = None
    logger: Logger | LoggerAdapter = log

    def report(self, result: FunctionAnalyticsResult, *, scope: str) -> None:
        """
        Emit validation counters for observability.

        Parameters
        ----------
        result:
            Completed analytics result to summarize.
        scope:
            Human-readable scope identifier (e.g., repo@commit).
        """
        counts = {
            "total": result.validation_total,
            "parse_failed": result.parse_failed_count,
            "span_not_found": result.span_not_found_count,
        }
        self.logger.info(
            "METRIC function_validation scope=%s total=%d parse_failed=%d span_not_found=%d",
            scope,
            counts["total"],
            counts["parse_failed"],
            counts["span_not_found"],
            extra={"function_validation": counts},
        )
        if self.emit_counter is None:
            return
        for name, value in counts.items():
            self.emit_counter(f"function_validation.{name}", value)


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


class _FunctionStats(ast.NodeVisitor):
    """
    Collect per-function structural metrics while walking an AST node.

    The visitor records cyclomatic complexity, nesting depth, and counts of
    return, yield, and raise statements for the function under inspection.
    """

    def __init__(self) -> None:
        self.return_count = 0
        self.yield_count = 0
        self.raise_count = 0
        self.complexity = 1  # base complexity
        self.max_nesting_depth = 0
        self._depth = 0

    def _enter_block(self) -> None:
        self._depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self._depth)

    def _leave_block(self) -> None:
        self._depth -= 1

    def visit_Return(self, node: ast.Return) -> None:
        del node
        self.return_count += 1

    def visit_Yield(self, node: ast.Yield) -> None:
        del node
        self.yield_count += 1

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        del node
        self.yield_count += 1

    def visit_Raise(self, node: ast.Raise) -> None:
        del node
        self.raise_count += 1

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_Try(self, node: ast.Try) -> None:
        self.complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_With(self, node: ast.With) -> None:
        self.complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # Count each additional operand as a decision point
        self.complexity += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Do not recurse into nested functions when called on the root function node.
        # A caller should call this visitor only on the target node.
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.generic_visit(node)


def _annotation_to_str(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except (TypeError, ValueError, AttributeError):
        return getattr(node, "id", None) or type(node).__name__


def _compute_loc(lines: list[str], start_line: int, end_line: int) -> tuple[int, int]:
    loc = end_line - start_line + 1
    logical_loc = 0
    for ln in range(start_line, end_line + 1):
        if 1 <= ln <= len(lines):
            stripped = lines[ln - 1].strip()
            if stripped and not stripped.startswith("#"):
                logical_loc += 1
    return loc, logical_loc


def _complexity_bucket(cc: int) -> str:
    if cc <= COMPLEXITY_LOW:
        return "low"
    if cc <= COMPLEXITY_MEDIUM:
        return "medium"
    return "high"


def _compute_param_stats(node: ast.AST) -> ParamStats:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ParamStats(
            param_count=0,
            positional_params=0,
            keyword_only_params=0,
            has_varargs=False,
            has_varkw=False,
            total_params=0,
            annotated_params=0,
            param_types={},
            has_return_annotation=False,
            return_type=None,
        )

    args = node.args
    all_params = list(getattr(args, "posonlyargs", [])) + list(args.args) + list(args.kwonlyargs)
    if args.vararg is not None:
        all_params.append(args.vararg)
    if args.kwarg is not None:
        all_params.append(args.kwarg)

    param_count = len(all_params)
    positional_params = len(getattr(args, "posonlyargs", [])) + len(args.args)
    keyword_only_params = len(args.kwonlyargs)
    has_varargs = args.vararg is not None
    has_varkw = args.kwarg is not None

    total_params = 0
    annotated_params = 0
    param_types: dict[str, str | None] = {}

    for p in all_params:
        name = p.arg
        if name in SKIP_PARAM_NAMES:
            continue
        total_params += 1
        ann_str = _annotation_to_str(p.annotation)
        if ann_str is not None:
            annotated_params += 1
        param_types[name] = ann_str

    has_return_annotation = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
        node.returns is not None
    )
    return_type = _annotation_to_str(node.returns) if hasattr(node, "returns") else None

    return ParamStats(
        param_count=param_count,
        positional_params=positional_params,
        keyword_only_params=keyword_only_params,
        has_varargs=has_varargs,
        has_varkw=has_varkw,
        total_params=total_params,
        annotated_params=annotated_params,
        param_types=param_types,
        has_return_annotation=has_return_annotation,
        return_type=return_type,
    )


def _typedness_flags(
    *, total_params: int, annotated_params: int, has_return_annotation: bool
) -> TypednessFlags:
    param_typed_ratio = annotated_params / float(total_params) if total_params else 1.0
    unannotated_params = total_params - annotated_params
    fully_typed = total_params > 0 and annotated_params == total_params and has_return_annotation
    untyped = annotated_params == 0 and not has_return_annotation
    partial_typed = (
        not fully_typed and not untyped and (annotated_params > 0 or has_return_annotation)
    )
    typedness_bucket = "typed" if fully_typed else "partial" if partial_typed else "untyped"
    typedness_source = (
        "annotations" if (annotated_params > 0 or has_return_annotation) else "unknown"
    )
    return TypednessFlags(
        param_typed_ratio=param_typed_ratio,
        unannotated_params=unannotated_params,
        fully_typed=fully_typed,
        partial_typed=partial_typed,
        untyped=untyped,
        typedness_bucket=typedness_bucket,
        typedness_source=typedness_source,
    )


def _derive_function_flags(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    stats: _FunctionStats,
    param_stats: ParamStats,
) -> FunctionDerived:
    is_async = isinstance(node, ast.AsyncFunctionDef)
    typedness = _typedness_flags(
        total_params=param_stats.total_params,
        annotated_params=param_stats.annotated_params,
        has_return_annotation=param_stats.has_return_annotation,
    )
    return FunctionDerived(
        is_async=is_async,
        is_generator=stats.yield_count > 0,
        complexity_bucket=_complexity_bucket(stats.complexity),
        stmt_count=len(getattr(node, "body", [])),
        decorator_count=len(getattr(node, "decorator_list", [])),
        has_docstring=ast.get_docstring(node) is not None,
        typedness=typedness,
    )


def _function_rows_from_node(
    meta: FunctionMeta,
    node: ast.AST,
    lines: list[str],
    ctx: ProcessContext,
) -> tuple[tuple, tuple] | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    loc, logical_loc = _compute_loc(lines, meta.start_line, meta.end_line)

    param_stats = _compute_param_stats(node)
    stats = _FunctionStats()
    stats.visit(node)
    derived = _derive_function_flags(node, stats, param_stats)
    typedness = derived.typedness
    return_type_source = "annotation" if param_stats.has_return_annotation else "unknown"

    metrics_row = (
        meta.goid,
        meta.urn,
        ctx.cfg.repo,
        ctx.cfg.commit,
        meta.rel_path,
        meta.language,
        meta.kind,
        meta.qualname,
        meta.start_line,
        meta.end_line,
        loc,
        logical_loc,
        param_stats.param_count,
        param_stats.positional_params,
        param_stats.keyword_only_params,
        param_stats.has_varargs,
        param_stats.has_varkw,
        derived.is_async,
        derived.is_generator,
        stats.return_count,
        stats.yield_count,
        stats.raise_count,
        stats.complexity,
        stats.max_nesting_depth,
        derived.stmt_count,
        derived.decorator_count,
        derived.has_docstring,
        derived.complexity_bucket,
        ctx.now,
    )

    types_row = (
        meta.goid,
        meta.urn,
        ctx.cfg.repo,
        ctx.cfg.commit,
        meta.rel_path,
        meta.language,
        meta.kind,
        meta.qualname,
        meta.start_line,
        meta.end_line,
        param_stats.total_params,
        param_stats.annotated_params,
        typedness.unannotated_params,
        typedness.param_typed_ratio,
        param_stats.has_return_annotation,
        param_stats.return_type,
        return_type_source,
        None,  # type_comment
        param_stats.param_types,
        typedness.fully_typed,
        typedness.partial_typed,
        typedness.untyped,
        typedness.typedness_bucket,
        typedness.typedness_source,
        ctx.now,
    )
    return metrics_row, types_row


def analyze_function(
    meta: FunctionMeta,
    parsed: ParsedFile,
    ctx: ProcessContext,
) -> tuple[tuple, tuple] | None:
    """
    Derive analytics rows for a single function span.

    Returns
    -------
    tuple[tuple, tuple] | None
        Metrics row and types row when a matching AST node is found; otherwise
        None when the span cannot be resolved.
    """
    resolution = resolve_span(parsed.index, meta.start_line, meta.end_line)
    if resolution.node is None:
        return None
    return _function_rows_from_node(meta, resolution.node, parsed.lines, ctx)


def _process_file_functions(
    rel_path: str,
    fun_rows: list[GoidRow],
    state: ProcessState,
) -> tuple[list[tuple], list[tuple], list[ValidationIssue]]:
    metrics_rows: list[tuple] = []
    types_rows: list[tuple] = []
    validation: list[ValidationIssue] = []

    abs_path = (state.cfg.repo_root / rel_path).resolve()
    parsed = get_parsed_file(rel_path, abs_path, state.cache, state.parser)
    if parsed is None:
        detail = f"File missing or unparsable: {abs_path}"
        validation.extend(
            ValidationIssue(
                rel_path=rel_path,
                qualname=str(row["qualname"]),
                issue="parse_failed",
                detail=detail,
            )
            for row in fun_rows
        )
        log.warning("Skipping file for function analytics: %s", abs_path)
        return metrics_rows, types_rows, validation

    for info in fun_rows:
        meta = _meta_from_goid_row(info)
        resolution = resolve_span(parsed.index, meta.start_line, meta.end_line)
        if resolution.node is None:
            validation.append(
                ValidationIssue(
                    rel_path=meta.rel_path,
                    qualname=meta.qualname,
                    issue="span_not_found",
                    detail=resolution.detail,
                )
            )
            continue
        rows = _function_rows_from_node(meta, resolution.node, parsed.lines, state.ctx)
        if rows is None:
            validation.append(
                ValidationIssue(
                    rel_path=meta.rel_path,
                    qualname=meta.qualname,
                    issue="span_not_found",
                    detail=resolution.detail,
                )
            )
            continue
        metrics_row, types_row = rows
        metrics_rows.append(metrics_row)
        types_rows.append(types_row)

    return metrics_rows, types_rows, validation


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
    state : _ProcessState
        Shared parser/ctx/cache state for processing.

    Returns
    -------
    FunctionAnalyticsResult
        Aggregated metrics, types, and validation findings.
    """
    metrics_rows: list[tuple] = []
    types_rows: list[tuple] = []
    validation: list[ValidationIssue] = []

    for rel_path, fun_rows in goids_by_file.items():
        file_metrics, file_types, file_validation = _process_file_functions(
            rel_path=rel_path,
            fun_rows=fun_rows,
            state=state,
        )
        metrics_rows.extend(file_metrics)
        types_rows.extend(file_types)
        validation.extend(file_validation)

    return FunctionAnalyticsResult(
        metrics_rows=metrics_rows,
        types_rows=types_rows,
        validation=validation,
    )


def persist_function_analytics(
    gateway: StorageGateway,
    cfg: FunctionAnalyticsConfig,
    result: FunctionAnalyticsResult,
    *,
    created_at: datetime,
) -> dict[str, int]:
    """
    Persist analytics rows and validation to DuckDB.

    Parameters
    ----------
    gateway : StorageGateway
        Storage gateway exposing the DuckDB connection.
    cfg : FunctionAnalyticsConfig
        Repository/commit context.
    result : FunctionAnalyticsResult
        Rows and validation to persist.
    created_at : datetime
        Timestamp used for validation persistence.

    Returns
    -------
    dict[str, int]
        Summary counts of persisted rows and validation.
    """
    con = gateway.con
    ensure_schema(con, "analytics.function_validation")
    scope = f"{cfg.repo}@{cfg.commit}"
    run_batch(
        gateway,
        "analytics.function_metrics",
        result.metrics_rows,
        delete_params=[cfg.repo, cfg.commit],
        scope=scope,
    )
    run_batch(
        gateway,
        "analytics.function_types",
        result.types_rows,
        delete_params=[cfg.repo, cfg.commit],
        scope=scope,
    )
    _persist_validation(gateway, cfg, result.validation, created_at=created_at)

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
    cfg: FunctionAnalyticsConfig,
    *,
    parser: ParsedFileLoader | None = None,
    parser_registry: FunctionParserRegistry | None = None,
    validation_reporter: ValidationReporter | None = None,
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
    cfg : FunctionAnalyticsConfig
        Repository metadata and file-system root used to locate source files.
    parser : ParsedFileLoader | None
        Optional parser hook primarily for testing; registry-based selection
        should be preferred for production.
    parser_registry : FunctionParserRegistry | None
        Parser registry used to resolve the parser when a hook is not provided.
    validation_reporter : ValidationReporter | None
        Reporter used to emit structured validation counters.

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

    registry = parser_registry or FunctionParserRegistry()
    if parser is not None:
        log.warning(
            "parser override is deprecated; prefer FunctionParserKind via config or registry",
        )
    selected_parser = parser or registry.get(cfg.parser)
    now = datetime.now(UTC)
    ctx = ProcessContext(cfg=cfg, now=now)
    parsed_cache: dict[str, ParsedFile | None] = {}
    state = ProcessState(cfg=cfg, cache=parsed_cache, parser=selected_parser, ctx=ctx)

    result = build_function_analytics(goids_by_file=goids_by_file, state=state)
    summary = persist_function_analytics(gateway, cfg, result, created_at=now)
    reporter = validation_reporter or ValidationReporter(logger=log)
    reporter.report(result, scope=f"{cfg.repo}@{cfg.commit}")
    if cfg.fail_on_missing_spans and result.validation:
        message = (
            f"Missing analytics for {result.validation_total} functions; "
            "see analytics.function_validation"
        )
        raise ValueError(message)

    if result.validation:
        log.warning(
            "Function validation gaps: parse_failed=%d span_not_found=%d",
            result.parse_failed_count,
            result.span_not_found_count,
        )
    return summary


def _load_goids(gateway: StorageGateway, cfg: FunctionAnalyticsConfig) -> dict[str, list[GoidRow]]:
    df = gateway.con.execute(
        """
        SELECT
            goid_h128,
            urn,
            repo,
            commit,
            rel_path,
            language,
            kind,
            qualname,
            start_line,
            end_line
        FROM core.goids
        WHERE repo = ? AND commit = ?
          AND kind IN ('function', 'method')
        """,
        [cfg.repo, cfg.commit],
    ).fetch_df()

    if df.empty:
        log.info("No function GOIDs found for repo=%s commit=%s", cfg.repo, cfg.commit)
        return {}

    goids_by_file: dict[str, list[GoidRow]] = {}
    for _, row in df.iterrows():
        rel_path = str(row["rel_path"]).replace("\\", "/")
        goid_row: GoidRow = {
            "goid_h128": int(row["goid_h128"]),
            "urn": str(row["urn"]),
            "repo": str(row["repo"]),
            "commit": str(row["commit"]),
            "rel_path": rel_path,
            "language": str(row["language"]),
            "kind": str(row["kind"]),
            "qualname": str(row["qualname"]),
            "start_line": int(row["start_line"]),
            "end_line": int(row["end_line"]) if row["end_line"] is not None else None,
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


def _persist_validation(
    gateway: StorageGateway,
    cfg: FunctionAnalyticsConfig,
    issues: list[ValidationIssue],
    *,
    created_at: datetime,
) -> None:
    con = gateway.con
    rows = [
        function_validation_row_to_tuple(
            FunctionValidationRow(
                repo=cfg.repo,
                commit=cfg.commit,
                rel_path=issue.rel_path,
                qualname=issue.qualname,
                issue=issue.issue,
                detail=issue.detail,
                created_at=created_at,
            )
        )
        for issue in issues
    ]
    run_batch(
        gateway,
        "analytics.function_validation",
        rows,
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )
