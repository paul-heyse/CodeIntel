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
from pathlib import Path
from typing import TypedDict

import duckdb

from codeintel.config.models import FunctionAnalyticsConfig
from codeintel.ingestion.ast_utils import AstSpanIndex, timed_parse
from codeintel.ingestion.common import run_batch
from codeintel.ingestion.source_scanner import ScanConfig, SourceScanner

log = logging.getLogger(__name__)

COMPLEXITY_LOW = 5
COMPLEXITY_MEDIUM = 10
SKIP_PARAM_NAMES = {"self", "cls"}


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


def _node_for_span(index: AstSpanIndex, start_line: int, end_line: int) -> ast.AST | None:
    return index.lookup(start_line, end_line)


def _parse_python_file(file_path: Path) -> tuple[list[str], AstSpanIndex] | None:
    parsed = timed_parse(file_path)
    if parsed is None:
        return None
    lines, tree, _duration = parsed
    index = AstSpanIndex.from_tree(tree, (ast.FunctionDef, ast.AsyncFunctionDef))
    return lines, index


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
    info: GoidRow,
    node: ast.AST,
    lines: list[str],
    ctx: ProcessContext,
) -> tuple[tuple, tuple] | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    meta = FunctionMeta(
        goid=int(info["goid_h128"]),
        urn=info["urn"],
        language=info["language"],
        kind=info["kind"],
        qualname=info["qualname"],
        start_line=int(info["start_line"]),
        end_line=int(info["end_line"]) if info["end_line"] is not None else int(info["start_line"]),
        rel_path=info["rel_path"],
    )

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


def compute_function_metrics_and_types(
    con: duckdb.DuckDBPyConnection,
    cfg: FunctionAnalyticsConfig,
) -> None:
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
    con : duckdb.DuckDBPyConnection
        Connection with `core.goids`, `analytics.function_metrics`, and
        `analytics.function_types` tables available.
    cfg : FunctionAnalyticsConfig
        Repository metadata and file-system root used to locate source files.

    Notes
    -----
    - The function reads each source file once and reuses the parsed AST for
      all contained GOIDs.
    - It skips functions whose source files are missing or invalid, logging
      context so pipeline failures can be triaged quickly.
    """
    scanner = SourceScanner(ScanConfig(repo_root=cfg.repo_root))
    goids_by_file = _load_goids(con, cfg)
    if not goids_by_file:
        return

    metrics_rows: list[tuple] = []
    types_rows: list[tuple] = []

    now = datetime.now(UTC)

    for record in scanner.iter_files(log):
        rel_path = record.rel_path
        fun_rows = goids_by_file.get(rel_path)
        if not fun_rows:
            continue
        parsed = _parse_python_file(record.path)
        if parsed is None:
            log.warning("Skipping file for function analytics: %s", record.path)
            continue
        lines, node_map = parsed
        file_metrics, file_types = _process_functions_in_file(
            rel_path=rel_path,
            fun_rows=fun_rows,
            index=node_map,
            lines=lines,
            ctx=ProcessContext(cfg=cfg, now=now),
        )
        metrics_rows.extend(file_metrics)
        types_rows.extend(file_types)

    scope = f"{cfg.repo}@{cfg.commit}"
    run_batch(
        con,
        "analytics.function_metrics",
        metrics_rows,
        delete_params=[cfg.repo, cfg.commit],
        scope=scope,
    )
    run_batch(
        con,
        "analytics.function_types",
        types_rows,
        delete_params=[cfg.repo, cfg.commit],
        scope=scope,
    )

    log.info(
        "Function metrics/types build complete for repo=%s commit=%s: %d functions",
        cfg.repo,
        cfg.commit,
        len(metrics_rows),
    )


def _load_goids(
    con: duckdb.DuckDBPyConnection, cfg: FunctionAnalyticsConfig
) -> dict[str, list[GoidRow]]:
    df = con.execute(
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


def _process_functions_in_file(
    rel_path: str,
    fun_rows: list[GoidRow],
    index: AstSpanIndex,
    lines: list[str],
    ctx: ProcessContext,
) -> tuple[list[tuple], list[tuple]]:
    metrics_rows: list[tuple] = []
    types_rows: list[tuple] = []

    for info in fun_rows:
        start_line = int(info["start_line"])
        end_line = int(info["end_line"]) if info["end_line"] is not None else start_line
        node = _node_for_span(index, start_line, end_line)
        if node is None:
            log.debug(
                "No AST node match for function %s in %s (%s-%s)",
                info["qualname"],
                rel_path,
                start_line,
                end_line,
            )
            continue

        rows = _function_rows_from_node(info, node, lines, ctx)
        if rows is None:
            continue
        metrics_row, types_row = rows
        metrics_rows.append(metrics_row)
        types_rows.append(types_row)

    return metrics_rows, types_rows
