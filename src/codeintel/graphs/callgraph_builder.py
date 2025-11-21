"""Build call graph nodes and edges from GOIDs and LibCST traversal."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import libcst as cst
from libcst import MetadataWrapper, metadata

log = logging.getLogger(__name__)


@dataclass
class CallGraphConfig:
    """
    Configuration for constructing a call graph for a repo snapshot.

    Parameters
    ----------
    repo : str
        Repository slug for tagging rows.
    commit : str
        Commit SHA corresponding to the analyzed snapshot.
    repo_root : Path
        Root directory containing the repository source files.
    """

    repo: str
    commit: str
    repo_root: Path


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
        func_goids_by_span: dict[tuple[int, int], int],
        callee_by_name: dict[str, int],
    ) -> None:
        self.rel_path = rel_path
        self.func_goids_by_span = func_goids_by_span
        self.callee_by_name = callee_by_name

        self.current_function_goid: int | None = None
        self.edges: list[tuple] = []

    def _pos(self, node: cst.CSTNode):
        pos = self.get_metadata(metadata.PositionProvider, node, None)
        if pos is None:
            return None
        return pos.start, pos.end

    # Track current function context ------------------------

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        span = self._pos(node)
        if span is None:
            return
        start, end = span
        key = (start.line, end.line)
        self.current_function_goid = self.func_goids_by_span.get(key)

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.current_function_goid = None

    def visit_AsyncFunctionDef(self, node: cst.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def leave_AsyncFunctionDef(self, node: cst.AsyncFunctionDef) -> None:
        self.leave_FunctionDef(node)

    # Calls -------------------------------------------------

    def visit_Call(self, node: cst.Call) -> None:
        if self.current_function_goid is None:
            return

        span = self._pos(node)
        if span is None:
            return
        start, _end = span

        callee_name, attr_chain = self._extract_callee(node.func)
        callee_goid = self.callee_by_name.get(callee_name)

        if callee_goid is not None:
            kind = "direct"
            resolved_via = "local_name"
            confidence = 0.8
        else:
            kind = "unresolved"
            resolved_via = "unresolved"
            confidence = 0.0

        evidence = {
            "callee_name": callee_name,
            "attr_chain": attr_chain or None,
        }

        self.edges.append(
            (
                self.current_function_goid,
                callee_goid,
                self.rel_path,
                start.line,
                start.column,
                "python",
                kind,
                resolved_via,
                confidence,
                json.dumps(evidence),
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
            cur = expr
            while isinstance(cur, cst.Attribute):
                names.append(cur.attr.value)
                cur = cur.value
            if isinstance(cur, cst.Name):
                names.append(cur.value)
            names.reverse()
            return names[-1], names
        return "", []


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

    Returns
    -------
    None
        Results are written directly into DuckDB tables.

    Notes
    -----
    The builder intentionally resolves callees by local function name only,
    which favors recall over precision. More advanced resolution can be added
    later using import graphs or symbol tables.
    """
    repo_root = cfg.repo_root.resolve()

    # 1) Build call_graph_nodes from GOIDs (functions/methods/classes)
    df_nodes = con.execute(
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
    ).fetch_df()

    con.execute("DELETE FROM graph.call_graph_nodes")

    node_rows: list[tuple] = []

    if not df_nodes.empty:
        for _, row in df_nodes.iterrows():
            goid = int(row["goid_h128"])
            language = row["language"]
            kind = row["kind"]
            rel_path = row["rel_path"]
            qualname = row["qualname"]
            name = str(qualname).split(".")[-1]
            is_public = not name.startswith("_")
            # arity is left as -1 (unknown) in this simple builder
            node_rows.append((goid, language, kind, -1, is_public, rel_path))

        con.executemany(
            """
            INSERT INTO graph.call_graph_nodes
              (goid_h128, language, kind, arity, is_public, rel_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            node_rows,
        )

    # 2) Build call_graph_edges via LibCST per file
    con.execute("DELETE FROM graph.call_graph_edges")

    df_files = con.execute(
        """
        SELECT DISTINCT rel_path
        FROM core.goids
        WHERE repo = ? AND commit = ?
          AND kind IN ('function', 'method')
        """,
        [cfg.repo, cfg.commit],
    ).fetch_df()

    if df_files.empty:
        log.info("No function GOIDs found; skipping call graph edges.")
        return

    all_edges: list[tuple] = []

    for rel_path in df_files["rel_path"]:
        rel_path_str = str(rel_path).replace("\\", "/")
        file_path = repo_root / rel_path_str

        # GOIDs for functions in this file
        df_funcs = con.execute(
            """
            SELECT goid_h128, start_line, end_line, qualname
            FROM core.goids
            WHERE repo = ? AND commit = ? AND rel_path = ?
              AND kind IN ('function', 'method')
            """,
            [cfg.repo, cfg.commit, rel_path_str],
        ).fetch_df()
        if df_funcs.empty:
            continue

        func_goids_by_span: dict[tuple[int, int], int] = {}
        callee_by_name: dict[str, int] = {}

        for _, row in df_funcs.iterrows():
            goid = int(row["goid_h128"])
            start_line = int(row["start_line"])
            end_line = int(row["end_line"]) if row["end_line"] is not None else start_line
            func_goids_by_span[start_line, end_line] = goid
            local_name = str(row["qualname"]).split(".")[-1]
            callee_by_name.setdefault(local_name, goid)

        try:
            source = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            log.warning("File missing for callgraph: %s", file_path)
            continue

        try:
            module = cst.parse_module(source)
        except Exception as exc:
            log.exception("Failed to parse %s for callgraph: %s", file_path, exc)
            continue

        wrapper = MetadataWrapper(module)
        visitor = _FileCallGraphVisitor(
            rel_path=rel_path_str,
            func_goids_by_span=func_goids_by_span,
            callee_by_name=callee_by_name,
        )
        wrapper.visit(visitor)
        all_edges.extend(visitor.edges)

    # Deduplicate edges per (caller, callee, line, col)
    seen = set()
    unique_edges: list[tuple] = []
    for row in all_edges:
        key = (row[0], row[1], row[2], row[3], row[4])
        if key in seen:
            continue
        seen.add(key)
        unique_edges.append(row)

    if unique_edges:
        con.executemany(
            """
            INSERT INTO graph.call_graph_edges
              (caller_goid_h128, callee_goid_h128,
               callsite_path, callsite_line, callsite_col,
               language, kind, resolved_via, confidence, evidence_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            unique_edges,
        )

    log.info(
        "Call graph build complete for repo=%s commit=%s: %d nodes, %d edges",
        cfg.repo,
        cfg.commit,
        len(node_rows),
        len(unique_edges),
    )
