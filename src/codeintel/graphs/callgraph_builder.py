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
from codeintel.graphs import symbol_uses
from codeintel.graphs.function_catalog import FunctionCatalog, FunctionMeta, load_function_catalog
from codeintel.graphs.function_index import FunctionSpan, FunctionSpanIndex
from codeintel.graphs.import_resolver import collect_aliases
from codeintel.graphs.validation import run_graph_validations
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
class EdgeResolutionContext:
    """Resolution helpers shared across visitors."""

    repo: str
    commit: str
    function_index: FunctionSpanIndex
    local_callees: dict[str, int]
    global_callees: dict[str, int]
    import_aliases: dict[str, str]
    scip_candidates_by_use_path: dict[str, tuple[str, ...]]
    def_goids_by_path: dict[str, int]


@dataclass(frozen=True)
class CallGraphRunScope:
    """Identifies the repository snapshot and filesystem root."""

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
            self.current_function_goid = self.context.function_index.lookup(
                self.rel_path, start.line, end.line
            )
            return True

        if isinstance(node, cst.Call):
            self._handle_call(node)
        return True

    def leave(self, node: cst.CSTNode) -> None:
        if isinstance(node, FUNCTION_NODE_TYPES):
            self.current_function_goid = None

    def _handle_call(self, node: cst.Call) -> None:
        if self.current_function_goid is None:
            # Fallback: single-function module or span mismatch
            spans = self.context.function_index.spans_for_path(self.rel_path)
            if spans:
                self.current_function_goid = spans[0].goid
        if self.current_function_goid is None:
            return

        span = self._pos(node)
        if span is None:
            return
        start, _end = span

        callee_name, attr_chain = self._extract_callee(node.func)
        callee_goid, resolved_via, confidence = resolve_callee(
            callee_name,
            attr_chain,
            self.context.local_callees,
            self.context.global_callees,
            self.context.import_aliases,
        )
        scip_paths = self.context.scip_candidates_by_use_path.get(self.rel_path)
        if callee_goid is None and scip_paths:
            callee_goid, resolved_via, confidence = _resolve_via_scip(
                scip_paths, self.context.def_goids_by_path
            )
        kind = "direct" if callee_goid is not None else "unresolved"

        evidence = {
            "callee_name": callee_name,
            "attr_chain": attr_chain or None,
            "resolved_via": resolved_via,
        }
        if callee_goid is None and scip_paths:
            evidence["scip_candidates"] = list(scip_paths)

        self.edges.append(
            CallGraphEdgeRow(
                repo=self.context.repo,
                commit=self.context.commit,
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
        Resolve callee GOID via shared heuristics and optional SCIP fallback.

        Returns
        -------
        tuple[int | None, str, float]
            (GOID, resolved_via, confidence) tuple.
        """
        goid, resolved_via, confidence = resolve_callee(
            callee_name,
            attr_chain,
            self.context.local_callees,
            self.context.global_callees,
            self.context.import_aliases,
        )
        if goid is not None:
            return goid, resolved_via, confidence

        scip_paths = self.context.scip_candidates_by_use_path.get(self.rel_path)
        if scip_paths:
            return _resolve_via_scip(scip_paths, self.context.def_goids_by_path)
        return goid, resolved_via, confidence


class _AstCallGraphVisitor(ast.NodeVisitor):
    """AST fallback visitor mirroring the CST call collector."""

    def __init__(self, rel_path: str, context: EdgeResolutionContext) -> None:
        self.rel_path = rel_path
        self.context = context
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
        callee_goid, resolved_via, confidence = resolve_callee(
            callee_name,
            attr_chain,
            self.context.local_callees,
            self.context.global_callees,
            self.context.import_aliases,
        )
        scip_paths = self.context.scip_candidates_by_use_path.get(self.rel_path)
        if callee_goid is None and scip_paths:
            callee_goid, resolved_via, confidence = _resolve_via_scip(
                scip_paths, self.context.def_goids_by_path
            )
        kind = "direct" if callee_goid is not None else "unresolved"
        evidence = {
            "callee_name": callee_name,
            "attr_chain": attr_chain or None,
            "resolved_via": resolved_via,
        }
        if callee_goid is None and scip_paths:
            evidence["scip_candidates"] = list(scip_paths)
        self.edges.append(
            CallGraphEdgeRow(
                repo=self.context.repo,
                commit=self.context.commit,
                caller_goid_h128=self.current_goid,
                callee_goid_h128=callee_goid,
                callsite_path=self.rel_path,
                callsite_line=getattr(node, "lineno", 0),
                callsite_col=getattr(node, "col_offset", 0),
                language="python",
                kind=kind,
                resolved_via=resolved_via,
                confidence=confidence,
                evidence_json=evidence,
            )
        )
        self.generic_visit(node)

    def _enter_function(self, node: ast.AST) -> None:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None:
            return
        self.current_goid = self.context.function_index.lookup(
            self.rel_path,
            int(start),
            int(end) if end is not None else int(start),
        )


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

    catalog = load_function_catalog(con, repo=cfg.repo, commit=cfg.commit)
    func_rows = catalog.function_spans
    if not func_rows:
        log.info("No function GOIDs found; skipping call graph edges.")
        return

    global_callee_by_name = _callee_map(func_rows)
    scip_candidates_by_use = _load_scip_candidates(con, repo_root)
    def_goids_by_path = _load_def_goid_map(con, repo=cfg.repo, commit=cfg.commit)
    scope = CallGraphRunScope(repo=cfg.repo, commit=cfg.commit, repo_root=repo_root)
    edges = _collect_edges(
        catalog,
        scope,
        global_callee_by_name,
        scip_candidates_by_use,
        def_goids_by_path,
    )
    unique_edges = _dedupe_edges(edges)
    _persist_call_graph_edges(con, unique_edges, cfg.repo, cfg.commit)
    run_graph_validations(con, repo=cfg.repo, commit=cfg.commit, logger=log)

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


def _callee_map(func_rows: list[FunctionSpan]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for row in func_rows:
        mapping.setdefault(row.qualname, row.goid)
        mapping.setdefault(row.qualname.rsplit(".", maxsplit=1)[-1], row.goid)
    return mapping


def _collect_edges(
    catalog: FunctionCatalog,
    scope: CallGraphRunScope,
    global_callee_by_name: dict[str, int],
    scip_candidates_by_use: dict[str, tuple[str, ...]],
    def_goids_by_path: dict[str, int],
) -> list[CallGraphEdgeRow]:
    edges: list[CallGraphEdgeRow] = []
    function_index = catalog.function_index
    functions_by_path = catalog.functions_by_path

    for rel_path in sorted(functions_by_path):
        callee_by_name = function_index.local_name_map(rel_path)

        file_path = scope.repo_root / rel_path
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
            repo=scope.repo,
            commit=scope.commit,
            function_index=function_index,
            local_callees=callee_by_name,
            global_callees=global_callee_by_name,
            import_aliases=alias_collector,
            scip_candidates_by_use_path=scip_candidates_by_use,
            def_goids_by_path=def_goids_by_path,
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
            row["repo"],
            row["commit"],
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
    source = file_path.read_text(encoding="utf8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    visitor = _AstCallGraphVisitor(rel_path, context)
    visitor.visit(tree)
    return visitor.edges


def _extract_callee_ast(expr: ast.AST) -> tuple[str, list[str]]:
    if isinstance(expr, ast.Name):
        return expr.id, [expr.id]
    if isinstance(expr, ast.Attribute):
        names = _flatten_attribute(expr)
        if names:
            return names[0], names
    return "", []


def _flatten_attribute(expr: ast.Attribute) -> list[str]:
    names: list[str] = []
    current: ast.AST = expr
    while isinstance(current, ast.Attribute):
        names.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        names.append(current.id)
        names.reverse()
        return names
    return []


def resolve_callee(
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
                qualified = (
                    alias_target
                    if len(attr_chain) == 1
                    else ".".join([alias_target, *attr_chain[1:]])
                )
                goid = local_callees.get(qualified) or global_callees.get(qualified)
                if goid is not None:
                    resolved_via = "import_alias"
                    confidence = 0.7

    if goid is None and callee_name in global_callees:
        goid = global_callees[callee_name]
        resolved_via = "global_name"
        confidence = 0.6
    elif goid is None and attr_chain:
        qualified = ".".join(attr_chain)
        goid = global_callees.get(qualified) or global_callees.get(attr_chain[-1])
        if goid is not None:
            resolved_via = "global_name"
            confidence = 0.6

    return goid, resolved_via, confidence


def _resolve_via_scip(
    candidate_def_paths: tuple[str, ...],
    def_goids_by_path: dict[str, int],
) -> tuple[int | None, str, float]:
    """
    Attempt to resolve a callee using SCIP definition paths.

    Returns
    -------
    tuple[int | None, str, float]
        GOID when found, resolution source label, and confidence.
    """
    for def_path in candidate_def_paths:
        goid = def_goids_by_path.get(normalize_rel_path(def_path))
        if goid is not None:
            return goid, "scip_def_path", 0.55
    return None, "unresolved", 0.0


def _load_scip_candidates(
    con: duckdb.DuckDBPyConnection, repo_root: Path
) -> dict[str, tuple[str, ...]]:
    """
    Map `use_path` -> candidate `def_path` values from symbol_use_edges.

    The result is used to enrich unresolved call edges with SCIP-derived
    evidence so downstream consumers can correlate callsites to symbols even
    when GOID resolution fails.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Mapping from use_path to possible definition paths.
    """
    rows: list[tuple[str | None, str | None]]
    try:
        rows = con.execute("SELECT def_path, use_path FROM graph.symbol_use_edges").fetchall()
    except duckdb.Error:
        rows = []

    mapping: dict[str, set[str]] = {}
    for def_path, use_path in rows:
        if def_path is None or use_path is None:
            continue
        use_norm = normalize_rel_path(str(use_path))
        mapping.setdefault(use_norm, set()).add(normalize_rel_path(str(def_path)))

    if not mapping:
        scip_path = symbol_uses.default_scip_json_path(repo_root, None)
        docs = symbol_uses.load_scip_documents(scip_path) if scip_path is not None else None
        if docs:
            def_map = symbol_uses.build_def_map(docs)
            mapping = symbol_uses.build_use_def_mapping(docs, def_map)

    return {path: tuple(sorted(defs)) for path, defs in mapping.items()}


def _load_def_goid_map(
    con: duckdb.DuckDBPyConnection, *, repo: str, commit: str
) -> dict[str, int]:
    """
    Map definition paths to GOIDs for a repo snapshot.

    Returns
    -------
    dict[str, int]
        Mapping of normalized definition path to GOID.
    """
    try:
        rows = con.execute(
            """
            SELECT gc.file_path, g.goid_h128
            FROM core.goid_crosswalk gc
            JOIN core.goids g
              ON g.urn = gc.goid
            WHERE g.repo = ? AND g.commit = ?
            """,
            [repo, commit],
        ).fetchall()
    except duckdb.Error:
        return {}

    mapping: dict[str, int] = {}
    for file_path, goid in rows:
        if file_path is None or goid is None:
            continue
        mapping[normalize_rel_path(str(file_path))] = int(goid)
    return mapping


def _catalog_from_spans(spans: list[FunctionSpan]) -> FunctionCatalog:
    """
    Build a lightweight catalog from spans for test helpers.

    Returns
    -------
    FunctionCatalog
        Catalog containing the provided spans with empty module mapping.
    """
    metas = [
        FunctionMeta(
            goid=span.goid,
            urn="",
            rel_path=span.rel_path,
            qualname=span.qualname,
            start_line=span.start_line,
            end_line=span.end_line if span.end_line is not None else span.start_line,
        )
        for span in spans
    ]
    return FunctionCatalog(functions=metas, module_by_path={})


def _persist_call_graph_edges(
    con: duckdb.DuckDBPyConnection, edges: list[CallGraphEdgeRow], repo: str, commit: str
) -> None:
    run_batch(
        con,
        "graph.call_graph_edges",
        [call_graph_edge_to_tuple(edge) for edge in edges],
        delete_params=[repo, commit],
        scope="call_graph_edges",
    )
    unresolved = sum(1 for edge in edges if edge["callee_goid_h128"] is None)
    log.info(
        "Call graph edges persisted: %d total (%d unresolved)",
        len(edges),
        unresolved,
    )


def collect_edges_for_testing(
    repo_root: Path,
    func_rows: list[FunctionSpan],
    *,
    repo: str = "test_repo",
    commit: str = "test_commit",
) -> list[CallGraphEdgeRow]:
    """
    Collect call graph edges for tests without touching DuckDB state.

    Parameters
    ----------
    repo_root : Path
        Filesystem root for the repository.
    func_rows : list[FunctionSpan]
        Function metadata rows to seed the collector.
    repo : str, optional
        Repository identifier to attach to emitted edges.
    commit : str, optional
        Commit hash to attach to emitted edges.

    Returns
    -------
    list[tuple]
        Call edge tuples mirroring `_collect_edges` output.
    """
    catalog = _catalog_from_spans(func_rows)
    scope = CallGraphRunScope(repo=repo, commit=commit, repo_root=repo_root)
    return _collect_edges(
        catalog,
        scope,
        _callee_map(func_rows),
        {},
        {},
    )


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
    goid, resolved_via, confidence = resolve_callee(
        callee_name,
        attr_chain,
        callee_by_name,
        global_callee_by_name,
        import_aliases,
    )
    return goid, resolved_via, confidence
