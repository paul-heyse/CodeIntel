"""Shared graph seeding helpers for analytics tests."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import networkx as nx

from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql.builder import ensure_schema
from tests._helpers.builders import (
    ConfigValueRow,
    GoidRow,
    ModuleRow,
    SubsystemModuleRow,
    SymbolUseEdgeRow,
    insert_rows,
)
from tests._helpers.fakes.graph_runtime import (
    CountingGraphEngineAdapter,
    GraphEngineAdapter,
    build_graph_engine_double,
)
from tests._helpers.fakes.graph_runtime import (
    GraphRuntimeDouble as GraphStubEngine,
)
from tests._helpers.fakes.networkx_graphs import (
    DEFAULT_CHAIN_LENGTH,
    DEFAULT_CYCLE_SIZE,
    DEFAULT_SPOKES,
    chain_graph,
    cyclic_graph,
    star_graph,
)
from tests._helpers.repo import write_canonical_repo


@dataclass
class GraphFixtures:
    """Bundled graph fixtures for analytics graph tests."""

    call_graph: nx.DiGraph
    import_graph: nx.DiGraph
    config_graph: nx.Graph
    symbol_module_graph: nx.Graph
    symbol_function_graph: nx.Graph
    cfg_graph: nx.DiGraph | None = None


def call_chain_graph(length: int = DEFAULT_CHAIN_LENGTH) -> nx.DiGraph:
    """Build a call graph with a simple chain topology.

    Returns
    -------
    nx.DiGraph
        Directed chain graph.
    """
    return chain_graph(length)


def call_star_graph(spokes: int = DEFAULT_SPOKES, *, inward: bool = False) -> nx.DiGraph:
    """Build a call graph with a star topology.

    Returns
    -------
    nx.DiGraph
        Directed star graph.
    """
    return star_graph(spokes, inward=inward)


def import_cycle_graph(size: int = DEFAULT_CYCLE_SIZE) -> nx.DiGraph:
    """Build an import graph with a directed cycle.

    Returns
    -------
    nx.DiGraph
        Directed cycle graph.
    """
    return cyclic_graph(size)


def symbol_star_graph(spokes: int = DEFAULT_SPOKES) -> nx.Graph:
    """Build a symbol graph with a star topology (undirected).

    Returns
    -------
    nx.Graph
        Undirected star graph.
    """
    return nx.Graph(star_graph(spokes, inward=False))


def standard_graph_fixtures(
    *,
    chain_length: int = DEFAULT_CHAIN_LENGTH,
    cycle_size: int = DEFAULT_CYCLE_SIZE,
    star_spokes: int = DEFAULT_SPOKES,
) -> GraphFixtures:
    """Build a consistent set of graph fixtures for tests.

    Returns
    -------
    GraphFixtures
        Fixture bundle with call/import/symbol/config graphs.
    """
    return GraphFixtures(
        call_graph=call_chain_graph(chain_length),
        import_graph=import_cycle_graph(cycle_size),
        config_graph=nx.Graph(),
        symbol_module_graph=symbol_star_graph(star_spokes),
        symbol_function_graph=symbol_star_graph(star_spokes),
        cfg_graph=nx.DiGraph(),
    )


def build_source_files(repo_root: Path) -> dict[str, Path]:
    """Write sample modules and return their paths keyed by module name.

    Parameters
    ----------
    repo_root
        Repository root used for writing the sample package.

    Returns
    -------
    dict[str, Path]
        Mapping of module name to written file path.
    """
    canonical = write_canonical_repo(repo_root)
    return {module: repo_root / rel_path for module, rel_path in canonical.module_paths.items()}


def _function_node(
    tree: ast.AST,
    target: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef:
    target_name = target.rsplit(".", maxsplit=1)[-1]
    for node in ast.walk(tree):
        if (
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            )
            and node.name == target_name
        ):
            return node
    message = f"Function {target} not found"
    raise ValueError(message)


def build_ast_map(
    paths: Mapping[str, Path],
    goids: Mapping[str, int],
    repo_root: Path,
    *,
    target_names: Mapping[str, str | Sequence[str]] | None = None,
) -> dict[int, FunctionAst]:
    """
    Build FunctionAst mapping for known targets.

    Parameters
    ----------
    paths
        Mapping of module names to file paths.
    goids
        GOID mapping keyed by target name.
    repo_root
        Repository root for computing relative paths.
    target_names
        Optional override for target names per module.

    Returns
    -------
    dict[int, FunctionAst]
        Mapping of GOID to parsed function/class AST metadata.

    Raises
    ------
    ValueError
        If a requested target cannot be found in the provided source or GOID mapping.
    """
    ast_by_goid: dict[int, FunctionAst] = {}
    target_lookup = target_names or {
        "pkg.api": "api_handler",
        "pkg.service": "process",
        "pkg.utils": "calc",
        "pkg.mod_a": "func_a",
        "pkg.mod_b": "func_b",
        "pkg.mod_c": "func_c",
        "pkg.util": ("helper", "util_func", "func_b"),
    }
    for module, path in paths.items():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        targets_raw = target_lookup[module]
        targets = (targets_raw,) if isinstance(targets_raw, str) else tuple(targets_raw)
        for target in targets:
            goid = goids.get(target)
            if goid is None:
                message = f"Function {target} not found"
                raise ValueError(message)
            func_node = _function_node(tree, target)
            start_line = getattr(func_node, "lineno", 0)
            end_line = getattr(func_node, "end_lineno", start_line)
            ast_by_goid[goid] = FunctionAst(
                goid=goid,
                rel_path=path.relative_to(repo_root).as_posix(),
                qualname=target,
                start_line=start_line,
                end_line=end_line,
                node=func_node,
                lines=list(source.splitlines()),
            )
    return ast_by_goid


def insert_modules(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    paths: Mapping[str, Path],
) -> None:
    """Insert module rows for provided paths."""
    ensure_schema(gateway.con, "core.modules")
    rows = [
        ModuleRow(
            module=module,
            path=path.relative_to(snapshot.repo_root).as_posix(),
            repo=snapshot.repo,
            commit=snapshot.commit,
        )
        for module, path in paths.items()
    ]
    insert_rows(gateway, rows)


def insert_goids(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    ast_by_goid: Mapping[int, FunctionAst],
    *,
    now: datetime,
) -> None:
    """Insert GOID rows for provided FunctionAst map."""
    ensure_schema(gateway.con, "core.goids")
    insert_rows(
        gateway,
        [
            GoidRow(
                goid_h128=func_ast.goid,
                urn=f"urn:{func_ast.qualname}",
                repo=snapshot.repo,
                commit=snapshot.commit,
                rel_path=func_ast.rel_path,
                kind="class" if isinstance(func_ast.node, ast.ClassDef) else "function",
                qualname=func_ast.qualname,
                start_line=func_ast.start_line,
                end_line=func_ast.end_line,
                created_at=now,
            )
            for func_ast in ast_by_goid.values()
        ],
    )


def insert_config_values(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    goids: Mapping[str, int],
    ast_by_goid: Mapping[int, FunctionAst],
) -> None:
    """Seed config_values rows for API_TOKEN and FEATURE_FLAG."""
    ensure_schema(gateway.con, "analytics.config_values")
    insert_rows(
        gateway,
        [
            ConfigValueRow(
                repo=snapshot.repo,
                commit=snapshot.commit,
                config_path="config/settings.yml",
                format="yaml",
                key="API_TOKEN",
                reference_paths=[
                    ast_by_goid[goids["api_handler"]].rel_path,
                    ast_by_goid[goids["process"]].rel_path,
                ],
                reference_modules=["pkg.api", "pkg.service"],
                reference_count=2,
            ),
            ConfigValueRow(
                repo=snapshot.repo,
                commit=snapshot.commit,
                config_path="config/settings.yml",
                format="yaml",
                key="FEATURE_FLAG",
                reference_paths=[ast_by_goid[goids["process"]].rel_path],
                reference_modules=["pkg.service"],
                reference_count=1,
            ),
        ],
    )


def insert_entrypoints(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    goids: Mapping[str, int],
    ast_by_goid: Mapping[int, FunctionAst],
    *,
    now: datetime,
) -> None:
    """Seed analytics.entrypoints with a single FastAPI handler."""
    ensure_schema(gateway.con, "analytics.entrypoints")
    gateway.con.execute(
        """
        INSERT INTO analytics.entrypoints (
            repo, commit, entrypoint_id, kind, framework,
            handler_goid_h128, handler_urn, handler_rel_path, handler_module,
            handler_qualname, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot.repo,
            snapshot.commit,
            "api.get_items",
            "http",
            "fastapi",
            goids["api_handler"],
            "urn:pkg.api.api_handler",
            ast_by_goid[goids["api_handler"]].rel_path,
            "pkg.api",
            "api_handler",
            now,
        ),
    )


def insert_subsystems(gateway: StorageGateway, snapshot: SnapshotRef) -> None:
    """Seed subsystem-module memberships."""
    ensure_schema(gateway.con, "analytics.subsystem_modules")
    insert_rows(
        gateway,
        [
            SubsystemModuleRow(
                repo=snapshot.repo,
                commit=snapshot.commit,
                subsystem_id="api",
                module="pkg.api",
                role="edge",
            ),
            SubsystemModuleRow(
                repo=snapshot.repo,
                commit=snapshot.commit,
                subsystem_id="core",
                module="pkg.service",
                role="service",
            ),
            SubsystemModuleRow(
                repo=snapshot.repo,
                commit=snapshot.commit,
                subsystem_id="core",
                module="pkg.utils",
                role="library",
            ),
        ],
    )


def insert_symbol_edges(
    gateway: StorageGateway,
    goids: Mapping[str, int],
    ast_by_goid: Mapping[int, FunctionAst],
) -> None:
    """Seed symbol use edges between api/service/utils."""
    ensure_schema(gateway.con, "graph.symbol_use_edges")
    insert_rows(
        gateway,
        [
            SymbolUseEdgeRow(
                symbol="pkg.mod_b.func_b",
                def_path=ast_by_goid[goids["func_b"]].rel_path,
                use_path=ast_by_goid[goids["func_a"]].rel_path,
                same_file=False,
                same_module=False,
            ),
            SymbolUseEdgeRow(
                symbol="pkg.mod_c.func_c",
                def_path=ast_by_goid[goids["func_c"]].rel_path,
                use_path=ast_by_goid[goids["func_b"]].rel_path,
                same_file=False,
                same_module=False,
            ),
        ],
    )


def build_sample_graphs(goids: Mapping[str, int]) -> GraphFixtures:
    """Construct sample graphs used across integration tests.

    Parameters
    ----------
    goids
        GOID mapping keyed by target name.

    Returns
    -------
    GraphFixtures
        Collection of seeded graph objects keyed by purpose.
    """
    call_graph = nx.DiGraph()
    call_graph.add_edge(goids["func_a"], goids["func_b"], weight=1.0)
    call_graph.add_edge(goids["func_b"], goids["func_c"], weight=1.0)
    call_graph.add_edge(goids["func_a"], goids["func_c"], weight=0.5)

    import_graph = nx.DiGraph()
    import_graph.add_edge("pkg.mod_a", "pkg.mod_b", weight=2.0)
    import_graph.add_edge("pkg.mod_b", "pkg.mod_c", weight=1.0)
    import_graph.add_edge("pkg.mod_c", "pkg.mod_a", weight=0.5)

    config_graph = nx.Graph()
    config_graph.add_node(("config_key", "API_TOKEN"), bipartite=0)
    config_graph.add_node(("config_key", "FEATURE_FLAG"), bipartite=0)
    config_graph.add_node(("module", "pkg.mod_a"), bipartite=1)
    config_graph.add_node(("module", "pkg.mod_b"), bipartite=1)
    config_graph.add_edge(("config_key", "API_TOKEN"), ("module", "pkg.mod_a"), weight=1.0)
    config_graph.add_edge(("config_key", "API_TOKEN"), ("module", "pkg.mod_b"), weight=1.0)
    config_graph.add_edge(("config_key", "FEATURE_FLAG"), ("module", "pkg.mod_b"), weight=2.0)

    symbol_module_graph = nx.Graph()
    symbol_module_graph.add_edge("pkg.mod_a", "pkg.mod_b", weight=3.0)
    symbol_module_graph.add_edge("pkg.mod_b", "pkg.mod_c", weight=1.0)

    symbol_function_graph = nx.Graph()
    symbol_function_graph.add_edge(goids["func_a"], goids["func_b"], weight=1.5)
    symbol_function_graph.add_edge(goids["func_b"], goids["func_c"], weight=1.0)

    return GraphFixtures(
        call_graph=call_graph,
        import_graph=import_graph,
        config_graph=config_graph,
        symbol_module_graph=symbol_module_graph,
        symbol_function_graph=symbol_function_graph,
    )


def build_module_map(
    ast_by_goid: Mapping[int, FunctionAst],
    goid_to_module: Mapping[int, str],
) -> dict[str, str]:
    """Map relative paths to module names for given GOIDs.

    Parameters
    ----------
    ast_by_goid
        Mapping of GOID to FunctionAst metadata.
    goid_to_module
        Mapping of GOID to module name.

    Returns
    -------
    dict[str, str]
        Mapping of relative source paths to module names.
    """
    module_map: dict[str, str] = {}
    for goid, module in goid_to_module.items():
        ast_obj = ast_by_goid[goid]
        module_map[ast_obj.rel_path] = module
    return module_map


__all__ = [
    "CountingGraphEngineAdapter",
    "GraphEngineAdapter",
    "GraphFixtures",
    "GraphStubEngine",
    "build_ast_map",
    "build_graph_engine_double",
    "build_module_map",
    "build_sample_graphs",
    "build_source_files",
    "insert_config_values",
    "insert_entrypoints",
    "insert_goids",
    "insert_modules",
    "insert_subsystems",
    "insert_symbol_edges",
]
