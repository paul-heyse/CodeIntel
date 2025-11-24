"""Build call graph nodes and edges from GOIDs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import libcst as cst

from codeintel.config.models import CallGraphConfig
from codeintel.graphs import call_ast, call_cst, call_persist, symbol_uses
from codeintel.graphs.call_context import EdgeResolutionContext
from codeintel.graphs.call_resolution import resolve_callee
from codeintel.graphs.function_catalog import FunctionCatalog, FunctionMeta, load_function_catalog
from codeintel.graphs.function_index import FunctionSpan
from codeintel.graphs.import_resolver import collect_aliases
from codeintel.graphs.validation import run_graph_validations
from codeintel.ingestion.common import run_batch
from codeintel.models.rows import CallGraphEdgeRow, CallGraphNodeRow, call_graph_node_to_tuple
from codeintel.utils.paths import normalize_rel_path, relpath_to_module

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CallGraphRunScope:
    """Identifies the repository snapshot and filesystem root."""

    repo: str
    commit: str
    repo_root: Path


def build_call_graph(con: duckdb.DuckDBPyConnection, cfg: CallGraphConfig) -> None:
    """Populate call graph nodes and edges for a repository snapshot."""
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
        catalog, scope, global_callee_by_name, scip_candidates_by_use, def_goids_by_path
    )
    unique_edges = call_persist.dedupe_edges(edges)
    call_persist.persist_call_graph_edges(con, unique_edges, cfg.repo, cfg.commit)
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
        cst_edges = call_cst.collect_edges_cst(rel_path=rel_path, module=module, context=context)
        if cst_edges:
            edges.extend(cst_edges)
        else:
            edges.extend(
                call_ast.collect_edges_ast(
                    rel_path=rel_path,
                    file_path=file_path,
                    context=context,
                )
            )

    return edges


def _load_scip_candidates(
    con: duckdb.DuckDBPyConnection, repo_root: Path
) -> dict[str, tuple[str, ...]]:
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


def _load_def_goid_map(con: duckdb.DuckDBPyConnection, *, repo: str, commit: str) -> dict[str, int]:
    try:
        rows = con.execute(
            """
            SELECT gc.file_path, g.goid_h128
            FROM core.goid_crosswalk gc
            JOIN core.goids g
              ON g.urn = gc.goid
             AND gc.repo = g.repo
             AND gc.commit = g.commit
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


def collect_edges_for_testing(
    repo_root: Path,
    func_rows: list[FunctionSpan],
    *,
    repo: str = "test_repo",
    commit: str = "test_commit",
) -> list[CallGraphEdgeRow]:
    """
    Collect call graph edges for tests without touching DuckDB state.

    Returns
    -------
    list[CallGraphEdgeRow]
        Collected edges for the provided spans.
    """
    catalog = _catalog_from_spans(func_rows)
    scope = CallGraphRunScope(repo=repo, commit=commit, repo_root=repo_root)
    return _collect_edges(catalog, scope, _callee_map(func_rows), {}, {})


def resolve_callee_for_testing(
    callee_name: str,
    attr_chain: list[str],
    callee_by_name: dict[str, int],
    global_callee_by_name: dict[str, int],
    import_aliases: dict[str, str],
) -> tuple[int | None, str, float]:
    """
    Resolve a callee GOID for tests without traversing the full CST.

    Returns
    -------
    tuple[int | None, str, float]
        Resolution result matching the production resolver.
    """
    resolution = resolve_callee(
        callee_name,
        attr_chain,
        callee_by_name,
        global_callee_by_name,
        import_aliases,
    )
    return resolution.callee_goid, resolution.resolved_via, resolution.confidence
