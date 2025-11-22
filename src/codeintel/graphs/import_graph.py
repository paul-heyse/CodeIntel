"""Construct module-level import graphs from LibCST parsing."""

from __future__ import annotations

import logging
from collections import defaultdict

import duckdb
import libcst as cst

from codeintel.config.models import ImportGraphConfig
from codeintel.graphs.import_resolver import collect_import_edges

log = logging.getLogger(__name__)


def _tarjan_scc(graph: dict[str, set[str]]) -> dict[str, int]:
    """
    Compute strongly connected components using Tarjan's algorithm.

    Parameters
    ----------
    graph : dict[str, set[str]]
        Adjacency list mapping modules to their imported modules.

    Returns
    -------
    dict[str, int]
        Mapping of module name to component identifier.
    """
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    comp_id_by_node: dict[str, int] = {}
    comp_counter = 0

    def strongconnect(v: str) -> None:
        nonlocal index, comp_counter
        indices[v] = index
        lowlinks[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, ()):
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            # Root of an SCC
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp_id_by_node[w] = comp_counter
                if w == v:
                    break
            comp_counter += 1

    for v in graph:
        if v not in indices:
            strongconnect(v)

    return comp_id_by_node


def build_import_graph(con: duckdb.DuckDBPyConnection, cfg: ImportGraphConfig) -> None:
    """
    Populate `graph.import_graph_edges` from LibCST import analysis.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection to the DuckDB instance seeded with `core.modules`.
    cfg : ImportGraphConfig
        Repository context and filesystem root.

    Notes
    -----
    The collector resolves relative imports conservatively to the current
    package. Strongly connected components are computed to identify cycles.
    """
    repo_root = cfg.repo_root.resolve()

    df_modules = con.execute(
        """
        SELECT module, path
        FROM core.modules
        WHERE repo = ? AND commit = ? AND language = 'python'
        """,
        [cfg.repo, cfg.commit],
    ).fetch_df()

    if df_modules.empty:
        log.info("No modules found in core.modules; skipping import graph.")
        return

    # Collect raw edges
    raw_edges: set[tuple[str, str]] = set()
    for _, row in df_modules.iterrows():
        module_name = str(row["module"])
        rel_path = str(row["path"]).replace("\\", "/")
        file_path = repo_root / rel_path

        try:
            source = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            log.warning("File missing for import graph: %s", file_path)
            continue

        try:
            module = cst.parse_module(source)
        except Exception:
            log.exception("Failed to parse %s for import graph", file_path)
            continue

        raw_edges.update(collect_import_edges(module_name, module))

    # Build fan-out / fan-in / SCCs
    if not raw_edges:
        con.execute("DELETE FROM graph.import_graph_edges")
        log.info("No imports found; import graph edges cleared.")
        return

    graph: dict[str, set[str]] = defaultdict(set)
    for src, dst in raw_edges:
        graph[src].add(dst)

    scc = _tarjan_scc(graph)

    fan_out: dict[str, int] = defaultdict(int)
    fan_in: dict[str, int] = defaultdict(int)
    for src, dst in raw_edges:
        fan_out[src] += 1
        fan_in[dst] += 1

    rows: list[tuple] = []
    for src, dst in sorted(raw_edges):
        rows.append(
            (
                src,
                dst,
                fan_out.get(src, 0),
                fan_in.get(dst, 0),
                scc.get(src, -1),
            )
        )

    con.execute("DELETE FROM graph.import_graph_edges")
    con.executemany(
        """
        INSERT INTO graph.import_graph_edges
          (src_module, dst_module, src_fan_out, dst_fan_in, cycle_group)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )

    log.info(
        "Import graph build complete for repo=%s commit=%s: %d edges, %d modules",
        cfg.repo,
        cfg.commit,
        len(rows),
        len(graph),
    )
