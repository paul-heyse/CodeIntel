"""
Derive per-function complexity metrics and type hints from Python source files.

This module reads GOID metadata, walks Python ASTs to compute structural metrics,
and emits analytics tables used by downstream scoring and documentation tools.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypedDict

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.functions.config import (
    FunctionAnalyticsOptions,
    ProcessContext,
    ProcessState,
)
from codeintel.analytics.functions.parsing import (
    FunctionParserRegistry,
    ParsedFile,
    get_parsed_file,
    resolve_span,
)
from codeintel.analytics.functions.typedness import (
    ParamStats,
    TypednessFlags,
    compute_param_stats,
    compute_typedness_flags,
)
from codeintel.analytics.functions.validation import (
    ValidationIssue,
    ValidationReporter,
    persist_validation,
)
from codeintel.config.models import FunctionAnalyticsConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.ingestion.common import run_batch
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

COMPLEXITY_LOW = 5
COMPLEXITY_MEDIUM = 10


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
class _FunctionStats(ast.NodeVisitor):
    """
    Collect per-function structural metrics while walking an AST node.

    The visitor records cyclomatic complexity, nesting depth, and counts of
    return, yield, and raise statements for the function under inspection.
    """

    return_count: int = 0
    yield_count: int = 0
    raise_count: int = 0
    complexity: int = 1
    max_nesting_depth: int = 0
    _depth: int = 0

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
        self.complexity += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.generic_visit(node)


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


def _derive_function_flags(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    stats: _FunctionStats,
    param_stats: ParamStats,
) -> FunctionDerived:
    is_async = isinstance(node, ast.AsyncFunctionDef)
    typedness = compute_typedness_flags(
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

    param_stats = compute_param_stats(node)
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
        None,
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
    state : ProcessState
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


def _build_function_analytics_from_context(
    *,
    goids_by_file: dict[str, list[GoidRow]],
    process_ctx: ProcessContext,
    context: AnalyticsContext,
) -> FunctionAnalyticsResult:
    metrics_rows: list[tuple] = []
    types_rows: list[tuple] = []
    validation: list[ValidationIssue] = []
    ast_map = context.function_ast_map
    missing_goids = context.missing_function_goids

    for fun_rows in goids_by_file.values():
        for info in fun_rows:
            meta = _meta_from_goid_row(info)
            ast_info = ast_map.get(meta.goid)
            if ast_info is None:
                detail = "missing AST in shared context" if meta.goid in missing_goids else None
                validation.append(
                    ValidationIssue(
                        rel_path=meta.rel_path,
                        qualname=meta.qualname,
                        issue="span_not_found",
                        detail=detail,
                    )
                )
                continue
            rows = _function_rows_from_node(meta, ast_info.node, ast_info.lines, process_ctx)
            if rows is None:
                validation.append(
                    ValidationIssue(
                        rel_path=meta.rel_path,
                        qualname=meta.qualname,
                        issue="span_not_found",
                        detail="context AST resolution failed",
                    )
                )
                continue
            metrics_row, types_row = rows
            metrics_rows.append(metrics_row)
            types_rows.append(types_row)

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
    persist_validation(gateway, cfg, result.validation, created_at=created_at)

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
    cfg : FunctionAnalyticsConfig
        Repository metadata and file-system root used to locate source files.
    options : FunctionAnalyticsOptions | None
        Optional hooks and cached context for parser selection, validation, and reuse.

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

    if opts.context is not None:
        result = _build_function_analytics_from_context(
            goids_by_file=goids_by_file,
            process_ctx=ctx,
            context=opts.context,
        )
    else:
        registry = opts.parser_registry or FunctionParserRegistry()
        if opts.parser is not None:
            log.warning(
                "parser override is deprecated; prefer FunctionParserKind via config or registry",
            )
        selected_parser = opts.parser or registry.get(cfg.parser)
        parsed_cache: dict[str, ParsedFile | None] = {}
        state = ProcessState(cfg=cfg, cache=parsed_cache, parser=selected_parser, ctx=ctx)
        result = build_function_analytics(goids_by_file=goids_by_file, state=state)

    summary = persist_function_analytics(gateway, cfg, result, created_at=now)
    reporter = opts.validation_reporter or ValidationReporter(logger=log)
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
