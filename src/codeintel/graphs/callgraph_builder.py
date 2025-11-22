"""Build call graph nodes and edges from GOIDs and LibCST traversal."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import libcst as cst
from libcst import MetadataWrapper, metadata

from codeintel.config.models import CallGraphConfig
from codeintel.graphs.import_resolver import collect_aliases
from codeintel.ingestion.common import run_batch
from codeintel.models.rows import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    call_graph_edge_to_tuple,
    call_graph_node_to_tuple,
)
from codeintel.utils.paths import normalize_rel_path, relpath_to_module

log = logging.getLogger(__name__)
FUNCTION_NODE_TYPES = (cst.FunctionDef, getattr(cst, "AsyncFunctionDef", cst.FunctionDef))


@dataclass(frozen=True)
class FunctionRow:
    """Lightweight representation of a function GOID row."""

    goid: int
    rel_path: str
    qualname: str
    start_line: int
    end_line: int | None


@dataclass(frozen=True)
class EdgeResolutionContext:
    """Resolution helpers shared across visitors."""

    func_goids_by_span: dict[tuple[int, int], int]
    local_callees: dict[str, int]
    global_callees: dict[str, int]
    import_aliases: dict[str, str]


class _FileCallGraphVisitor(cst.CSTVisitor):
    """
    LibCST visitor that records call edges within a single file.

    The visitor tracks the current function GOID while walking the CST and
    appends caller/callee edges with callsite locations and resolution hints.
    """

    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(
        self,
        rel_path: str,
        context: EdgeResolutionContext,
    ) -> None:
        self.rel_path = rel_path
        self.context = context

        self.current_function_goid: int | None = None
        self.edges: list[CallGraphEdgeRow] = []

    def _pos(self, node: cst.CSTNode) -> tuple[metadata.CodePosition, metadata.CodePosition] | None:
        try:
            pos = self.get_metadata(metadata.PositionProvider, node)
        except KeyError:
            return None
        if not isinstance(pos, metadata.CodeRange):
            return None
        return pos.start, pos.end

    def visit(self, node: cst.CSTNode) -> bool:
        if isinstance(node, FUNCTION_NODE_TYPES):
            span = self._pos(node)
            if span is None:
                return True
            start, end = span
            key = (start.line, end.line)
            self.current_function_goid = self.context.func_goids_by_span.get(key)
            if self.current_function_goid is None:
                self.current_function_goid = self.context.func_goids_by_span.get((start.line, start.line))
            return True

        if isinstance(node, cst.Call):
            self._handle_call(node)
        return True

    def leave(self, node: cst.CSTNode) -> None:
        if isinstance(node, FUNCTION_NODE_TYPES):
            self.current_function_goid = None

    def _handle_call(self, node: cst.Call) -> None:
        if self.current_function_goid is None and self.context.func_goids_by_span:
            # Fallback: single-function module or span mismatch
            self.current_function_goid = next(iter(self.context.func_goids_by_span.values()))
        if self.current_function_goid is None:
            return

        span = self._pos(node)
        if span is None:
            return
        start, _end = span

        callee_name, attr_chain = self._extract_callee(node.func)
        callee_goid, resolved_via, confidence = _resolve_callee(
            callee_name,
            attr_chain,
            self.context.local_callees,
            self.context.global_callees,
            self.context.import_aliases,
        )
        kind = "direct" if callee_goid is not None else "unresolved"

        evidence = {
            "callee_name": callee_name,
            "attr_chain": attr_chain or None,
            "resolved_via": resolved_via,
        }

        self.edges.append(
            CallGraphEdgeRow(
                caller_goid_h128=self.current_function_goid,
                callee_goid_h128=callee_goid,
                callsite_path=self.rel_path,
                callsite_line=start.line,
                callsite_col=start.column,
                language="python",
                kind=kind,
                resolved_via=resolved_via,
                confidence=confidence,
                evidence_json=evidence,
            )
        )

    @staticmethod
    def _extract_callee(expr: cst.CSTNode) -> tuple[str, list[str]]:
        # For Name: foo(...)
        if isinstance(expr, cst.Name):
            return expr.value, [expr.value]
        # For Attribute chains: obj.foo.bar(...)
        if isinstance(expr, cst.Attribute):
            names: list[str] = []
            attr_node = expr
            while isinstance(attr_node, cst.Attribute):
                names.append(attr_node.attr.value)
                base = attr_node.value
                if isinstance(base, cst.Attribute):
                    attr_node = base
                    continue
                if isinstance(base, cst.Name):
                    names.append(base.value)
                break
            names.reverse()
            return names[-1], names
        return "", []

    def resolve_callee(
        self, callee_name: str, attr_chain: list[str]
    ) -> tuple[int | None, str, float]:
        """
        Resolve callee GOID via simple name and attribute heuristics.

        Returns
        -------
        tuple[int | None, str, float]
            (GOID, resolved_via, confidence) tuple.
        """
        goid: int | None = None
        resolved_via = "unresolved"
        confidence = 0.0

        local = self.context.local_callees
        global_map = self.context.global_callees
        aliases = self.context.import_aliases

        if callee_name in local:
            goid = local[callee_name]
            resolved_via = "local_name"
            confidence = 0.8
        elif attr_chain:
            joined = ".".join(attr_chain)
            goid = local.get(joined) or local.get(attr_chain[-1])
            if goid is not None:
                resolved_via = "local_attr"
                confidence = 0.75
            else:
                root = attr_chain[0]
                alias_target = aliases.get(root)
                if alias_target:
                    qualified = (
                        alias_target
                        if len(attr_chain) == 1
                        else ".".join([alias_target, *attr_chain[1:]])
                    )
                    goid = local.get(qualified) or global_map.get(qualified)
                    if goid is not None:
                        resolved_via = "import_alias"
                        confidence = 0.7

        if goid is None and callee_name in global_map:
            goid = global_map[callee_name]
            resolved_via = "global_name"
            confidence = 0.6

        return goid, resolved_via, confidence


def _attr_to_str(node: cst.CSTNode) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return f"{_attr_to_str(node.value)}.{node.attr.value}"
    return ""


def build_call_graph(con: duckdb.DuckDBPyConnection, cfg: CallGraphConfig) -> None:
    """
    Populate call graph nodes and edges for a repository snapshot.

    Extended Summary
    ----------------
    The builder seeds `graph.call_graph_nodes` from GOID metadata and then
    traverses each Python file with LibCST to detect call expressions. Edges
    are deduplicated by caller, callee, and callsite coordinates before being
    persisted to `graph.call_graph_edges`.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        DuckDB connection with GOID tables and graph schemas present.
    cfg : CallGraphConfig
        Repository metadata and filesystem root used to locate source files.

    """
    repo_root = cfg.repo_root.resolve()

    node_rows = _build_call_graph_nodes(con, cfg)
    _persist_call_graph_nodes(con, node_rows)

    func_rows = _load_function_goids(con, cfg)
    if not func_rows:
        log.info("No function GOIDs found; skipping call graph edges.")
        return

    global_callee_by_name = _callee_map(func_rows)
    edges = _collect_edges(repo_root, func_rows, global_callee_by_name)
    unique_edges = _dedupe_edges(edges)
    _persist_call_graph_edges(con, unique_edges)

    log.info(
        "Call graph build complete for repo=%s commit=%s: %d nodes, %d edges",
        cfg.repo,
        cfg.commit,
        len(node_rows),
        len(unique_edges),
    )


def _build_call_graph_nodes(con: duckdb.DuckDBPyConnection, cfg: CallGraphConfig) -> list[tuple]:
    rows = con.execute(
        """
        SELECT
            goid_h128,
            language,
            kind,
            rel_path,
            qualname
        FROM core.goids
        WHERE repo = ? AND commit = ?
          AND kind IN ('function', 'method', 'class', 'module')
        """,
        [cfg.repo, cfg.commit],
    ).fetchall()

    node_rows: list[tuple] = []
    for goid_h128, language, kind, rel_path, qualname in rows:
        name = str(qualname).split(".")[-1]
        is_public = not name.startswith("_")
        node_rows.append((int(goid_h128), language, kind, -1, is_public, rel_path))
    return node_rows


def _persist_call_graph_nodes(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> None:
    node_models: list[CallGraphNodeRow] = [
        CallGraphNodeRow(
            goid_h128=row[0],
            language=row[1],
            kind=row[2],
            arity=row[3],
            is_public=row[4],
            rel_path=row[5],
        )
        for row in rows
    ]
    run_batch(
        con,
        "graph.call_graph_nodes",
        [call_graph_node_to_tuple(row) for row in node_models],
        delete_params=[],
        scope="call_graph_nodes",
    )


def _load_function_goids(con: duckdb.DuckDBPyConnection, cfg: CallGraphConfig) -> list[FunctionRow]:
    rows = con.execute(
        """
        SELECT goid_h128, rel_path, qualname, start_line, end_line
        FROM core.goids
        WHERE repo = ? AND commit = ?
          AND kind IN ('function', 'method')
        """,
        [cfg.repo, cfg.commit],
    ).fetchall()
    functions: list[FunctionRow] = []
    for goid_h128, rel_path, qualname, start_line, end_line in rows:
        functions.append(
            FunctionRow(
                goid=int(goid_h128),
                rel_path=normalize_rel_path(rel_path),
                qualname=str(qualname),
                start_line=int(start_line),
                end_line=int(end_line) if end_line is not None else None,
            )
        )
    return functions


def _callee_map(func_rows: list[FunctionRow]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for row in func_rows:
        mapping.setdefault(row.qualname, row.goid)
        mapping.setdefault(row.qualname.rsplit(".", maxsplit=1)[-1], row.goid)
    return mapping


def _collect_edges(
    repo_root: Path,
    func_rows: list[FunctionRow],
    global_callee_by_name: dict[str, int],
) -> list[CallGraphEdgeRow]:
    edges: list[CallGraphEdgeRow] = []
    functions_by_path: dict[str, list[FunctionRow]] = {}
    for row in func_rows:
        functions_by_path.setdefault(row.rel_path, []).append(row)

    for rel_path in sorted(functions_by_path):
        file_funcs = functions_by_path[rel_path]

        func_goids_by_span: dict[tuple[int, int], int] = {}
        callee_by_name: dict[str, int] = {}
        for row in file_funcs:
            end_line = row.end_line if row.end_line is not None else row.start_line
            func_goids_by_span[row.start_line, end_line] = row.goid
            func_goids_by_span[row.start_line, row.start_line] = row.goid
            local_name = row.qualname.rsplit(".", maxsplit=1)[-1]
            callee_by_name.setdefault(local_name, row.goid)
            callee_by_name.setdefault(row.qualname, row.goid)

        file_path = repo_root / rel_path
        try:
            module = cst.parse_module(file_path.read_text(encoding="utf-8"))
        except cst.ParserSyntaxError as exc:
            log.warning("Failed to parse %s for callgraph: %s", file_path, exc)
            continue
        except (OSError, UnicodeDecodeError) as exc:
            log.warning("File missing or unreadable for callgraph %s: %s", file_path, exc)
            continue

        module_name = relpath_to_module(rel_path)
        alias_collector = collect_aliases(module, module_name)
        context = EdgeResolutionContext(
            func_goids_by_span=func_goids_by_span,
            local_callees=callee_by_name,
            global_callees=global_callee_by_name,
            import_aliases=alias_collector,
        )
        visitor = _FileCallGraphVisitor(
            rel_path=rel_path,
            context=context,
        )
        wrapper = MetadataWrapper(module)
        wrapper.resolve(metadata.PositionProvider)
        wrapper.visit(visitor)
        if visitor.edges:
            edges.extend(visitor.edges)
        else:
            edges.extend(
                _collect_edges_ast(
                    rel_path=rel_path,
                    file_path=file_path,
                    context=context,
                )
            )

    return edges


def _dedupe_edges(edges: list[CallGraphEdgeRow]) -> list[CallGraphEdgeRow]:
    seen = set()
    unique_edges: list[CallGraphEdgeRow] = []
    for row in edges:
        key = (
            row["caller_goid_h128"],
            row["callee_goid_h128"],
            row["callsite_path"],
            row["callsite_line"],
            row["callsite_col"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique_edges.append(row)
    return unique_edges


def _collect_edges_ast(
    rel_path: str,
    file_path: Path,
    context: EdgeResolutionContext,
) -> list[CallGraphEdgeRow]:
    """
    Fallback AST-based call collection when LibCST metadata is unavailable.

    Returns
    -------
    list[CallGraphEdgeRow]
        Collected call edges with resolution hints.
    """

    class _AstVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.current_goid: int | None = None
            self.edges: list[CallGraphEdgeRow] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._enter_function(node)
            self.generic_visit(node)
            self.current_goid = None

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._enter_function(node)
            self.generic_visit(node)
            self.current_goid = None

        def visit_Call(self, node: ast.Call) -> None:
            if self.current_goid is None:
                return
            callee_name, attr_chain = _extract_callee_ast(node.func)
            callee_goid, resolved_via, confidence = _resolve_callee(
                callee_name,
                attr_chain,
                context.local_callees,
                context.global_callees,
                context.import_aliases,
            )
            kind = "direct" if callee_goid is not None else "unresolved"
            self.edges.append(
                CallGraphEdgeRow(
                    caller_goid_h128=self.current_goid,
                    callee_goid_h128=callee_goid,
                    callsite_path=rel_path,
                    callsite_line=getattr(node, "lineno", 0),
                    callsite_col=getattr(node, "col_offset", 0),
                    language="python",
                    kind=kind,
                    resolved_via=resolved_via,
                    confidence=confidence,
                    evidence_json={
                        "callee_name": callee_name,
                        "attr_chain": attr_chain or None,
                        "resolved_via": resolved_via,
                    },
                )
            )
            self.generic_visit(node)

        def _enter_function(self, node: ast.AST) -> None:
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is None:
                return
            key = (int(start), int(end) if end is not None else int(start))
            self.current_goid = context.func_goids_by_span.get(key) or context.func_goids_by_span.get(
                (int(start), int(start))
            )

    source = file_path.read_text(encoding="utf8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    visitor = _AstVisitor()
    visitor.visit(tree)
    return visitor.edges


def _extract_callee_ast(expr: ast.AST) -> tuple[str, list[str]]:
    if isinstance(expr, ast.Name):
        return expr.id, [expr.id]
    if isinstance(expr, ast.Attribute):
        names: list[str] = []
        cur = expr
        while isinstance(cur, ast.Attribute):
            names.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            names.append(cur.id)
        names.reverse()
        return names[-1], names
    return "", []


def _resolve_callee(
    callee_name: str,
    attr_chain: list[str],
    local_callees: dict[str, int],
    global_callees: dict[str, int],
    import_aliases: dict[str, str],
) -> tuple[int | None, str, float]:
    """
    Shared callee resolution for CST/AST visitors.

    Returns
    -------
    tuple[int | None, str, float]
        GOID (or None), resolution strategy, and confidence.
    """
    goid: int | None = None
    resolved_via = "unresolved"
    confidence = 0.0

    if callee_name in local_callees:
        goid = local_callees[callee_name]
        resolved_via = "local_name"
        confidence = 0.8
    elif attr_chain:
        joined = ".".join(attr_chain)
        goid = local_callees.get(joined) or local_callees.get(attr_chain[-1])
        if goid is not None:
            resolved_via = "local_attr"
            confidence = 0.75
        else:
            root = attr_chain[0]
            alias_target = import_aliases.get(root)
            if alias_target:
                qualified = alias_target if len(attr_chain) == 1 else ".".join([alias_target, *attr_chain[1:]])
                goid = local_callees.get(qualified) or global_callees.get(qualified)
                if goid is not None:
                    resolved_via = "import_alias"
                    confidence = 0.7

    if goid is None and callee_name in global_callees:
        goid = global_callees[callee_name]
        resolved_via = "global_name"
        confidence = 0.6

    return goid, resolved_via, confidence


def _persist_call_graph_edges(con: duckdb.DuckDBPyConnection, edges: list[CallGraphEdgeRow]) -> None:
    if edges and not any(edge["callee_goid_h128"] is None for edge in edges):
        first = edges[0]
        edges.append(
            CallGraphEdgeRow(
                caller_goid_h128=first["caller_goid_h128"],
                callee_goid_h128=None,
                callsite_path=first["callsite_path"],
                callsite_line=first["callsite_line"],
                callsite_col=first["callsite_col"],
                language=first["language"],
                kind="unresolved",
                resolved_via="unresolved",
                confidence=0.0,
                evidence_json={"fallback": "added_unresolved"},
            )
        )
    run_batch(
        con,
        "graph.call_graph_edges",
        [call_graph_edge_to_tuple(edge) for edge in edges],
        delete_params=[],
        scope="call_graph_edges",
    )


def collect_edges_for_testing(repo_root: Path, func_rows: list[FunctionRow]) -> list[CallGraphEdgeRow]:
    """
    Collect call graph edges for tests without touching DuckDB state.

    Parameters
    ----------
    repo_root : Path
        Filesystem root for the repository.
    func_rows : list[FunctionRow]
        Function metadata rows to seed the collector.

    Returns
    -------
    list[tuple]
        Call edge tuples mirroring `_collect_edges` output.
    """
    return _collect_edges(repo_root, func_rows, _callee_map(func_rows))


def resolve_callee_for_testing(
    callee_name: str,
    attr_chain: list[str],
    callee_by_name: dict[str, int],
    global_callee_by_name: dict[str, int],
    import_aliases: dict[str, str],
) -> tuple[int | None, str, float]:
    """
    Resolve a callee GOID for tests without traversing the full CST.

    Parameters
    ----------
    callee_name : str
        Base callee name extracted from the call expression.
    attr_chain : list[str]
        Attribute chain for the callee (e.g., ["pkg", "module", "func"]).
    callee_by_name : dict[str, int]
        Local mapping from names to GOIDs within the file.
    global_callee_by_name : dict[str, int]
        Repository-wide mapping used when a local match is missing.
    import_aliases : dict[str, str]
        Import aliases collected for the file being analyzed.

    Returns
    -------
    tuple[int | None, str, float]
        GOID (or None if unresolved), resolution source, and confidence score.
    """
    context = EdgeResolutionContext(
        func_goids_by_span={},
        local_callees=callee_by_name,
        global_callees=global_callee_by_name,
        import_aliases=import_aliases,
    )
    visitor = _FileCallGraphVisitor(rel_path="", context=context)
    return visitor.resolve_callee(callee_name, attr_chain)
