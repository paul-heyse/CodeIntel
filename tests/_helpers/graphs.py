"""Shared graph seeding helpers for analytics tests."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeVar, cast

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

_GraphT = TypeVar("_GraphT", nx.Graph, nx.DiGraph)


@dataclass
class GraphFixtures:
    """Bundled graph fixtures for analytics graph tests."""

    call_graph: nx.DiGraph
    import_graph: nx.DiGraph
    config_graph: nx.Graph
    symbol_module_graph: nx.Graph
    symbol_function_graph: nx.Graph


@dataclass
class GraphStubEngine:
    """Minimal GraphEngine implementation backed by seeded graphs."""

    gateway: StorageGateway
    snapshot: SnapshotRef
    call_graph_obj: nx.DiGraph | None = None
    import_graph_obj: nx.DiGraph | None = None
    symbol_module_graph_obj: nx.Graph | None = None
    symbol_function_graph_obj: nx.Graph | None = None
    config_bipartite_obj: nx.Graph | None = None
    test_function_bipartite_obj: nx.Graph | None = None
    copy_graphs: bool = True

    @classmethod
    def from_fixtures(
        cls,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        fixtures: GraphFixtures,
        *,
        copy_graphs: bool = True,
    ) -> GraphStubEngine:
        """
        Construct a stub engine from bundled fixtures.

        Parameters
        ----------
        gateway
            Storage gateway for analytics operations.
        snapshot
            Snapshot reference bound to the engine.
        fixtures
            Seeded graph fixtures.
        copy_graphs
            Whether to return defensive copies of seeded graphs.

        Returns
        -------
        GraphStubEngine
            Stub configured with the provided graphs.
        """
        return cls(
            gateway=gateway,
            snapshot=snapshot,
            call_graph_obj=fixtures.call_graph,
            import_graph_obj=fixtures.import_graph,
            symbol_module_graph_obj=fixtures.symbol_module_graph,
            symbol_function_graph_obj=fixtures.symbol_function_graph,
            config_bipartite_obj=fixtures.config_graph,
            test_function_bipartite_obj=nx.Graph(),
            copy_graphs=copy_graphs,
        )

    @property
    def use_gpu(self) -> bool:
        """
        Prefer CPU execution.

        Returns
        -------
        bool
            ``False`` to keep execution on CPU.
        """
        return False

    def call_graph(self) -> nx.DiGraph:
        """Return a copy of the seeded call graph.

        Returns
        -------
        nx.DiGraph
            Seeded call graph for the snapshot.
        """
        graph = self.call_graph_obj or nx.DiGraph()
        return self._clone(graph)

    def load_call_graph(self) -> nx.DiGraph:
        """Alias for call_graph.

        Returns
        -------
        nx.DiGraph
            Seeded call graph for the snapshot.
        """
        return self.call_graph()

    def import_graph(self) -> nx.DiGraph:
        """Return a copy of the seeded import graph.

        Returns
        -------
        nx.DiGraph
            Seeded import graph for the snapshot.
        """
        graph = self.import_graph_obj or nx.DiGraph()
        return self._clone(graph)

    def load_import_graph(self) -> nx.DiGraph:
        """Alias for import_graph.

        Returns
        -------
        nx.DiGraph
            Seeded import graph for the snapshot.
        """
        return self.import_graph()

    def symbol_module_graph(self) -> nx.Graph:
        """Return a copy of the seeded symbol-module graph.

        Returns
        -------
        nx.Graph
            Seeded symbol-module coupling graph.
        """
        graph = self.symbol_module_graph_obj or nx.Graph()
        return self._clone(graph)

    def load_symbol_module_graph(self) -> nx.Graph:
        """Alias for symbol_module_graph.

        Returns
        -------
        nx.Graph
            Seeded symbol-module coupling graph.
        """
        return self.symbol_module_graph()

    def symbol_function_graph(self) -> nx.Graph:
        """Return a copy of the seeded symbol-function graph.

        Returns
        -------
        nx.Graph
            Seeded symbol-function coupling graph.
        """
        graph = self.symbol_function_graph_obj or nx.Graph()
        return self._clone(graph)

    def load_symbol_function_graph(self) -> nx.Graph:
        """Alias for symbol_function_graph.

        Returns
        -------
        nx.Graph
            Seeded symbol-function coupling graph.
        """
        return self.symbol_function_graph()

    def config_module_bipartite(self) -> nx.Graph:
        """Return a copy of the seeded config bipartite graph.

        Returns
        -------
        nx.Graph
            Seeded config-module bipartite graph.
        """
        graph = self.config_bipartite_obj or nx.Graph()
        return self._clone(graph)

    def load_config_module_bipartite(self) -> nx.Graph:
        """Alias for config_module_bipartite.

        Returns
        -------
        nx.Graph
            Seeded config-module bipartite graph.
        """
        return self.config_module_bipartite()

    def test_function_bipartite(self) -> nx.Graph:
        """Return a copy of the seeded test-function bipartite graph.

        Returns
        -------
        nx.Graph
            Seeded test-function bipartite graph.
        """
        graph = self.test_function_bipartite_obj or nx.Graph()
        return self._clone(graph)

    def load_test_function_bipartite(self) -> nx.Graph:
        """Alias for test_function_bipartite.

        Returns
        -------
        nx.Graph
            Seeded test-function bipartite graph.
        """
        return self.test_function_bipartite()

    def _clone(self, graph: _GraphT) -> _GraphT:
        """Copy graphs when requested to isolate mutations in tests.

        Returns
        -------
        _GraphT
            Cloned graph when `copy_graphs` is enabled.
        """
        if self.copy_graphs:
            return cast("_GraphT", graph.copy())
        return graph


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
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    api_path = pkg_dir / "api.py"
    service_path = pkg_dir / "service.py"
    utils_path = pkg_dir / "utils.py"

    api_path.write_text(
        "\n".join(
            [
                "import os",
                "from pkg import service",
                "",
                "def api_handler(limit: int, factor: int = 2) -> int:",
                '    token = os.getenv("API_TOKEN")',
                "    if not token:",
                '        token = "fallback"',
                "    if limit > 0:",
                "        service.process(limit, token)",
                "    return limit * factor",
            ]
        ),
        encoding="utf-8",
    )
    service_path.write_text(
        "\n".join(
            [
                "import os",
                "from pkg import utils",
                "",
                "def process(limit: int, token: str | None = None) -> int:",
                '    token_env = os.getenv("API_TOKEN")',
                '    flag = os.environ.get("FEATURE_FLAG")',
                "    result = utils.calc(limit)",
                "    if token_env:",
                "        result += 1",
                "    if flag:",
                "        return result + 1",
                "    return result",
            ]
        ),
        encoding="utf-8",
    )
    utils_path.write_text(
        "def calc(value: int) -> int:\n    return value * 2\n",
        encoding="utf-8",
    )
    return {
        "pkg.api": api_path,
        "pkg.service": service_path,
        "pkg.utils": utils_path,
    }


def _function_node(
    tree: ast.AST,
    target: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    target_name = target.rsplit(".", maxsplit=1)[-1]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name:
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
    """
    ast_by_goid: dict[int, FunctionAst] = {}
    target_lookup = target_names or {
        "pkg.api": "api_handler",
        "pkg.service": "process",
        "pkg.utils": "calc",
    }
    for module, path in paths.items():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        targets_raw = target_lookup[module]
        targets = (targets_raw,) if isinstance(targets_raw, str) else tuple(targets_raw)
        for target in targets:
            goid = goids[target]
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
                kind="function",
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
                symbol="pkg.service.process",
                def_path=ast_by_goid[goids["process"]].rel_path,
                use_path=ast_by_goid[goids["api_handler"]].rel_path,
                same_file=False,
                same_module=False,
            ),
            SymbolUseEdgeRow(
                symbol="pkg.utils.calc",
                def_path=ast_by_goid[goids["calc"]].rel_path,
                use_path=ast_by_goid[goids["process"]].rel_path,
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
    call_graph.add_edge(goids["api_handler"], goids["process"], weight=1.0)
    call_graph.add_edge(goids["process"], goids["calc"], weight=1.0)
    call_graph.add_edge(goids["api_handler"], goids["calc"], weight=0.5)

    import_graph = nx.DiGraph()
    import_graph.add_edge("pkg.api", "pkg.service", weight=2.0)
    import_graph.add_edge("pkg.service", "pkg.utils", weight=1.0)
    import_graph.add_edge("pkg.utils", "pkg.api", weight=0.5)

    config_graph = nx.Graph()
    config_graph.add_node(("config_key", "API_TOKEN"), bipartite=0)
    config_graph.add_node(("config_key", "FEATURE_FLAG"), bipartite=0)
    config_graph.add_node(("module", "pkg.api"), bipartite=1)
    config_graph.add_node(("module", "pkg.service"), bipartite=1)
    config_graph.add_edge(("config_key", "API_TOKEN"), ("module", "pkg.api"), weight=1.0)
    config_graph.add_edge(("config_key", "API_TOKEN"), ("module", "pkg.service"), weight=1.0)
    config_graph.add_edge(("config_key", "FEATURE_FLAG"), ("module", "pkg.service"), weight=2.0)

    symbol_module_graph = nx.Graph()
    symbol_module_graph.add_edge("pkg.api", "pkg.service", weight=3.0)
    symbol_module_graph.add_edge("pkg.service", "pkg.utils", weight=1.0)

    symbol_function_graph = nx.Graph()
    symbol_function_graph.add_edge(goids["api_handler"], goids["process"], weight=1.5)
    symbol_function_graph.add_edge(goids["process"], goids["calc"], weight=1.0)

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
    "GraphFixtures",
    "GraphStubEngine",
    "build_ast_map",
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
