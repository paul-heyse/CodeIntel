"""
Derive per-function complexity metrics and type hints from Python source files.

This module reads GOID metadata, walks Python ASTs to compute structural metrics,
and emits analytics tables used by downstream scoring and documentation tools.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)


@dataclass
class FunctionAnalyticsConfig:
    """
    Configuration describing the repository snapshot used for function analytics.

    Parameters
    ----------
    repo : str
        Repository identifier stored alongside analytics rows.
    commit : str
        Commit SHA of the snapshot being analyzed.
    repo_root : Path
        Filesystem path to the repository root where source files are read.
    """

    repo: str
    commit: str
    repo_root: Path


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
        self.return_count += 1

    def visit_Yield(self, node: ast.Yield) -> None:
        self.yield_count += 1

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self.yield_count += 1

    def visit_Raise(self, node: ast.Raise) -> None:
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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Do not recurse into nested functions when called on the root function node.
        # A caller should call this visitor only on the target node.
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)


def _annotation_to_str(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)  # type: ignore[attr-defined]
    except Exception:
        return getattr(node, "id", None) or type(node).__name__


def _collect_function_nodes(tree: ast.AST) -> list[ast.AST]:
    funcs: list[ast.AST] = []

    class Collector(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            funcs.append(node)
            # do not traverse nested functions further for mapping by line span
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            funcs.append(node)
            self.generic_visit(node)

    Collector().visit(tree)
    return funcs


def _build_node_map_by_span(
    functions: list[ast.AST],
) -> dict[tuple[int, int], ast.AST]:
    result: dict[tuple[int, int], ast.AST] = {}
    for fn in functions:
        lineno = getattr(fn, "lineno", None)
        end_lineno = getattr(fn, "end_lineno", None)
        if lineno is None or end_lineno is None:
            continue
        result[int(lineno), int(end_lineno)] = fn
    return result


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

    Returns
    -------
    None
        Results are persisted to DuckDB tables.

    Notes
    -----
    - The function reads each source file once and reuses the parsed AST for
      all contained GOIDs.
    - It skips functions whose source files are missing or invalid, logging
      context so pipeline failures can be triaged quickly.
    """
    repo_root = cfg.repo_root.resolve()

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
        return

    # Group GOIDs by file to parse each file once.
    goids_by_file: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        rel_path = str(row["rel_path"]).replace("\\", "/")
        goids_by_file.setdefault(rel_path, []).append(row.to_dict())

    con.execute(
        "DELETE FROM analytics.function_metrics WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )
    con.execute(
        "DELETE FROM analytics.function_types WHERE repo = ? AND commit = ?", [cfg.repo, cfg.commit]
    )

    metrics_rows: list[tuple] = []
    types_rows: list[tuple] = []

    now = datetime.utcnow()

    for rel_path, fun_rows in goids_by_file.items():
        file_path = repo_root / rel_path
        try:
            source = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            log.warning("File not found when computing function metrics: %s", file_path)
            continue

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as exc:
            log.warning("Syntax error in %s: %s; skipping", file_path, exc)
            continue

        lines = source.splitlines()
        fn_nodes = _collect_function_nodes(tree)
        node_map = _build_node_map_by_span(fn_nodes)

        for info in fun_rows:
            start_line = int(info["start_line"])
            end_line = int(info["end_line"]) if info["end_line"] is not None else start_line
            node = node_map.get((start_line, end_line))
            if node is None:
                # Fallback: try loose containment match.
                candidates = [n for (s, e), n in node_map.items() if s <= start_line <= e]
                node = candidates[0] if candidates else None
            if node is None:
                log.debug(
                    "No AST node match for function %s in %s (%s-%s)",
                    info["qualname"],
                    rel_path,
                    start_line,
                    end_line,
                )
                continue

            goid = int(info["goid_h128"])
            urn = info["urn"]
            language = info["language"]
            kind = info["kind"]
            qualname = info["qualname"]

            # LOC and logical LOC
            loc = end_line - start_line + 1
            logical_loc = 0
            for ln in range(start_line, end_line + 1):
                if 1 <= ln <= len(lines):
                    text = lines[ln - 1]
                    stripped = text.strip()
                    if stripped and not stripped.startswith("#"):
                        logical_loc += 1

            # Parameters
            assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            args = node.args
            all_params = (
                list(getattr(args, "posonlyargs", [])) + list(args.args) + list(args.kwonlyargs)
            )
            if args.vararg is not None:
                all_params.append(args.vararg)
            if args.kwarg is not None:
                all_params.append(args.kwarg)

            param_count = len(all_params)
            positional_params = len(getattr(args, "posonlyargs", [])) + len(args.args)
            keyword_only_params = len(args.kwonlyargs)
            has_varargs = args.vararg is not None
            has_varkw = args.kwarg is not None

            # Async/generator flags
            is_async = isinstance(node, ast.AsyncFunctionDef)
            stats = _FunctionStats()
            stats.visit(node)
            is_generator = stats.yield_count > 0

            # Complexity bucket
            cc = stats.complexity
            if cc <= 5:
                complexity_bucket = "low"
            elif cc <= 10:
                complexity_bucket = "medium"
            else:
                complexity_bucket = "high"

            # Statement count & decorators
            stmt_count = len(node.body)
            decorator_count = len(node.decorator_list)
            has_docstring = ast.get_docstring(node) is not None

            # Function types / annotations
            total_params = 0
            annotated_params = 0
            param_types: dict[str, str | None] = {}

            for p in all_params:
                name = p.arg
                if name in ("self", "cls"):
                    continue
                total_params += 1
                ann_str = _annotation_to_str(p.annotation)
                if ann_str is not None:
                    annotated_params += 1
                param_types[name] = ann_str

            has_return_annotation = node.returns is not None
            return_type = _annotation_to_str(node.returns)
            return_type_source = "annotation" if has_return_annotation else "unknown"

            if total_params:
                param_typed_ratio = annotated_params / float(total_params)
            else:
                param_typed_ratio = 1.0

            unannotated_params = total_params - annotated_params
            fully_typed = (
                total_params > 0 and annotated_params == total_params and has_return_annotation
            )
            untyped = annotated_params == 0 and not has_return_annotation
            partial_typed = (
                not fully_typed and not untyped and (annotated_params > 0 or has_return_annotation)
            )

            if fully_typed:
                typedness_bucket = "typed"
            elif partial_typed:
                typedness_bucket = "partial"
            else:
                typedness_bucket = "untyped"

            typedness_source = (
                "annotations" if (annotated_params > 0 or has_return_annotation) else "unknown"
            )

            # function_metrics row
            metrics_rows.append(
                (
                    goid,
                    urn,
                    cfg.repo,
                    cfg.commit,
                    rel_path,
                    language,
                    kind,
                    qualname,
                    start_line,
                    end_line,
                    loc,
                    logical_loc,
                    param_count,
                    positional_params,
                    keyword_only_params,
                    has_varargs,
                    has_varkw,
                    is_async,
                    is_generator,
                    stats.return_count,
                    stats.yield_count,
                    stats.raise_count,
                    stats.complexity,
                    stats.max_nesting_depth,
                    stmt_count,
                    decorator_count,
                    has_docstring,
                    complexity_bucket,
                    now,
                )
            )

            # function_types row
            types_rows.append(
                (
                    goid,
                    urn,
                    cfg.repo,
                    cfg.commit,
                    rel_path,
                    language,
                    kind,
                    qualname,
                    start_line,
                    end_line,
                    total_params,
                    annotated_params,
                    unannotated_params,
                    param_typed_ratio,
                    has_return_annotation,
                    return_type,
                    return_type_source,
                    None,  # type_comment
                    param_types,
                    fully_typed,
                    partial_typed,
                    untyped,
                    typedness_bucket,
                    typedness_source,
                    now,
                )
            )

    if metrics_rows:
        con.executemany(
            """
            INSERT INTO analytics.function_metrics
              (function_goid_h128, urn, repo, commit, rel_path,
               language, kind, qualname, start_line, end_line,
               loc, logical_loc,
               param_count, positional_params, keyword_only_params,
               has_varargs, has_varkw, is_async, is_generator,
               return_count, yield_count, raise_count,
               cyclomatic_complexity, max_nesting_depth,
               stmt_count, decorator_count, has_docstring,
               complexity_bucket, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            metrics_rows,
        )

    if types_rows:
        con.executemany(
            """
            INSERT INTO analytics.function_types
              (function_goid_h128, urn, repo, commit, rel_path,
               language, kind, qualname, start_line, end_line,
               total_params, annotated_params, unannotated_params,
               param_typed_ratio, has_return_annotation,
               return_type, return_type_source, type_comment,
               param_types, fully_typed, partial_typed, untyped,
               typedness_bucket, typedness_source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            types_rows,
        )

    log.info(
        "Function metrics/types build complete for repo=%s commit=%s: %d functions",
        cfg.repo,
        cfg.commit,
        len(metrics_rows),
    )
