"""Shared graph seeding helpers for analytics tests."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.core.catalog import FunctionCatalog
from tests._helpers.builders import (
    ConfigValueRow,
    ModuleRow,
    SubsystemModuleRow,
    SymbolEdgeOptions,
    insert_rows,
    insert_symbol_use_edges,
    make_symbol_use_edge_row,
)
from tests._helpers.catalogs import seed_goids_for_snapshot
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
from tests._helpers.repo import (
    GOID_FUNC_A,
    GOID_FUNC_B,
    GOID_FUNC_C,
    GOID_HELPER,
    MOD_A_FQN,
    MOD_A_PATH,
    MOD_B_FQN,
    MOD_B_PATH,
    MOD_C_FQN,
    MOD_C_PATH,
    MOD_UTIL_FQN,
    MOD_UTIL_PATH,
    write_canonical_repo,
)
from tests._helpers.rows import function_meta
from tests._helpers.seeds import AST_METRICS_PACK, CORE_PACK

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime
    from pathlib import Path

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.context import TestContext


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


def call_graph_fixture(edges: Sequence[tuple[str, str]] | None = None) -> nx.DiGraph:
    """Create a small call graph for tests, defaulting to a simple chain.

    Returns
    -------
    nx.DiGraph
        Directed call graph containing the provided edges.
    """
    if edges is None:
        edges = [("func_a", "func_b"), ("func_b", "func_c")]
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


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


def build_canonical_ast_lookup(repo_root: Path) -> dict[int, FunctionAst]:
    """Build FunctionAst lookup for the canonical sample repo.

    Returns
    -------
    dict[int, FunctionAst]
        Mapping of GOID to parsed AST metadata.
    """
    paths = {
        MOD_A_FQN: repo_root / MOD_A_PATH,
        MOD_B_FQN: repo_root / MOD_B_PATH,
        MOD_C_FQN: repo_root / MOD_C_PATH,
        MOD_UTIL_FQN: repo_root / MOD_UTIL_PATH,
    }
    goids = {
        "func_a": GOID_FUNC_A,
        "func_b": GOID_FUNC_B,
        "func_c": GOID_FUNC_C,
        "helper": GOID_HELPER,
    }
    target_names = {
        MOD_A_FQN: "func_a",
        MOD_B_FQN: "func_b",
        MOD_C_FQN: "func_c",
        MOD_UTIL_FQN: "helper",
    }
    return build_ast_map(paths, goids, repo_root, target_names=target_names)


def canonical_ast_map(ctx: TestContext) -> dict[int, FunctionAst]:
    """Return canonical AST map for a context already seeded with AST metrics.

    Returns
    -------
    dict[int, FunctionAst]
        Mapping from GOID to parsed FunctionAst.
    """
    return build_canonical_ast_lookup(ctx.repo_root)


@dataclass(frozen=True)
class CanonicalAstArtifacts:
    """Bundle canonical FunctionCatalog plus AST map for analytics tests."""

    catalog: FunctionCatalog
    ast_map: dict[int, FunctionAst]


def canonical_ast_artifacts(ctx: TestContext) -> CanonicalAstArtifacts:
    """Ensure core/AST packs are applied and return catalog + AST map.

    Returns
    -------
    CanonicalAstArtifacts
        Bundled catalog and AST map for canonical fixtures.
    """
    canonical = write_canonical_repo(ctx.repo_root)
    ctx.require(CORE_PACK, AST_METRICS_PACK)
    functions = [
        function_meta(
            goid=meta.goid,
            rel_path=meta.rel_path,
            qualname=meta.qualname,
            snapshot=(ctx.repo, ctx.commit),
            line_span=(meta.start_line, meta.end_line),
        )
        for meta in canonical.functions.values()
    ]
    module_by_path = {path: module for module, path in canonical.module_paths.items()}
    catalog = FunctionCatalog(functions=functions, module_by_path=module_by_path)
    return CanonicalAstArtifacts(catalog=catalog, ast_map=canonical_ast_map(ctx))


def _function_node(
    tree: ast.AST,
    target: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef:
    """Find a function/class node in an AST by fully qualified name suffix.

    Returns
    -------
    ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        Matching function or class node.

    Raises
    ------
    ValueError
        If no matching node is found.
    """
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
    """Insert GOID rows for provided FunctionAst map using catalog-based seeding."""
    _ = now
    kinds = {
        func_ast.goid: "class" if isinstance(func_ast.node, ast.ClassDef) else "function"
        for func_ast in ast_by_goid.values()
    }
    functions = [
        function_meta(
            goid=func_ast.goid,
            rel_path=func_ast.rel_path,
            qualname=func_ast.qualname,
            snapshot=(snapshot.repo, snapshot.commit),
            line_span=(func_ast.start_line, func_ast.end_line),
        )
        for func_ast in ast_by_goid.values()
    ]
    catalog = FunctionCatalog(functions=functions, module_by_path={})
    seed_goids_for_snapshot(gateway, snapshot, catalog, kinds=kinds)


def insert_config_values(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    goids: Mapping[str, int],
    ast_by_goid: Mapping[int, FunctionAst],
) -> None:
    """Seed config_values rows for API_TOKEN and FEATURE_FLAG."""
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
                    ast_by_goid[goids["func_a"]].rel_path,
                    ast_by_goid[goids["func_b"]].rel_path,
                ],
                reference_modules=["pkg.mod_a", "pkg.mod_b"],
                reference_count=2,
            ),
            ConfigValueRow(
                repo=snapshot.repo,
                commit=snapshot.commit,
                config_path="config/settings.yml",
                format="yaml",
                key="FEATURE_FLAG",
                reference_paths=[ast_by_goid[goids["func_b"]].rel_path],
                reference_modules=["pkg.mod_b"],
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
    gateway.policy.ensure_schemas_preserve()
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
            goids["func_a"],
            "urn:pkg.mod_a.func_a",
            ast_by_goid[goids["func_a"]].rel_path,
            "pkg.mod_a",
            "func_a",
            now,
        ),
    )


def insert_subsystems(gateway: StorageGateway, snapshot: SnapshotRef) -> None:
    """Seed subsystem-module memberships."""
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
    insert_symbol_use_edges(
        gateway,
        [
            make_symbol_use_edge_row(
                "pkg.mod_b.func_b",
                ast_by_goid[goids["func_b"]].rel_path,
                ast_by_goid[goids["func_a"]].rel_path,
                options=SymbolEdgeOptions(
                    same_file=False,
                    same_module=False,
                ),
            ),
            make_symbol_use_edge_row(
                "pkg.mod_c.func_c",
                ast_by_goid[goids["func_c"]].rel_path,
                ast_by_goid[goids["func_b"]].rel_path,
                options=SymbolEdgeOptions(
                    same_file=False,
                    same_module=False,
                ),
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
    call_graph = nx.relabel_nodes(
        chain_graph(3),
        {"A": goids["func_a"], "B": goids["func_b"], "C": goids["func_c"]},
    )
    call_graph.add_edge(goids["func_a"], goids["func_c"], weight=0.5)
    call_graph_weights = dict.fromkeys(call_graph.edges, 1.0)
    call_graph_weights[goids["func_a"], goids["func_c"]] = 0.5
    nx.set_edge_attributes(call_graph, call_graph_weights, "weight")

    import_graph = nx.relabel_nodes(
        cyclic_graph(3),
        {"A": "pkg.mod_a", "B": "pkg.mod_b", "C": "pkg.mod_c"},
    )
    import_weights = dict.fromkeys(import_graph.edges, 1.0)
    nx.set_edge_attributes(import_graph, import_weights, "weight")

    config_graph = nx.Graph()
    config_graph.add_node(("config_key", "API_TOKEN"), bipartite=0)
    config_graph.add_node(("config_key", "FEATURE_FLAG"), bipartite=0)
    config_graph.add_node(("module", "pkg.mod_a"), bipartite=1)
    config_graph.add_node(("module", "pkg.mod_b"), bipartite=1)
    config_graph.add_edge(("config_key", "API_TOKEN"), ("module", "pkg.mod_a"), weight=1.0)
    config_graph.add_edge(("config_key", "API_TOKEN"), ("module", "pkg.mod_b"), weight=1.0)
    config_graph.add_edge(("config_key", "FEATURE_FLAG"), ("module", "pkg.mod_b"), weight=2.0)

    symbol_module_graph = nx.Graph(star_graph(2))
    symbol_module_graph = nx.relabel_nodes(
        symbol_module_graph,
        {
            "hub": "pkg.mod_a",
            "spoke1": "pkg.mod_b",
            "spoke2": "pkg.mod_c",
        },
    )
    nx.set_edge_attributes(
        symbol_module_graph,
        dict.fromkeys(symbol_module_graph.edges, 1.0),
        "weight",
    )

    symbol_function_graph = nx.Graph(star_graph(2))
    symbol_function_graph = nx.relabel_nodes(
        symbol_function_graph,
        {
            "hub": goids["func_a"],
            "spoke1": goids["func_b"],
            "spoke2": goids["func_c"],
        },
    )
    nx.set_edge_attributes(
        symbol_function_graph,
        dict.fromkeys(symbol_function_graph.edges, 1.0),
        "weight",
    )

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
    "CanonicalAstArtifacts",
    "CountingGraphEngineAdapter",
    "GraphEngineAdapter",
    "GraphFixtures",
    "GraphStubEngine",
    "build_ast_map",
    "build_canonical_ast_lookup",
    "build_graph_engine_double",
    "build_module_map",
    "build_sample_graphs",
    "build_source_files",
    "call_graph_fixture",
    "canonical_ast_artifacts",
    "canonical_ast_map",
    "insert_config_values",
    "insert_entrypoints",
    "insert_goids",
    "insert_modules",
    "insert_subsystems",
    "insert_symbol_edges",
]
