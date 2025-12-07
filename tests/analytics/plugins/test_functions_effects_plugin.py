"""Integration test for FunctionEffectsPlugin."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from codeintel.analytics.plugins.functions.effects import FunctionEffectsPlugin
from codeintel.analytics.runtime.graph import GraphRuntime, GraphRuntimeOptions
from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogService
from tests._helpers.context import create_test_context
from tests._helpers.fakes.graph_engine import StubGraphEngine
from tests._helpers.plugin_execution import PluginTestContext, execute_target_plugin
from tests._helpers.rows import function_meta


def _seed_effect_sources(repo_root: Path) -> None:
    """Write a module with a direct side effect and a transitive caller."""
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "mod.py").write_text(
        "\n".join(
            [
                "GLOBAL_STATE = 0",
                "",
                "def helper(value: int) -> int:",
                "    global GLOBAL_STATE",
                "    GLOBAL_STATE = value",
                "    return value * 2",
                "",
                "def main(value: int) -> int:",
                "    return helper(value) + 1",
            ]
        ),
        encoding="utf-8",
    )


def _make_catalog(repo: str, commit: str) -> FunctionCatalogService:
    """Construct a catalog aligned with the seeded module.

    Returns
    -------
    FunctionCatalogService
        Catalog provider that mirrors the seeded functions.
    """
    functions = [
        function_meta(
            goid=7001,
            rel_path="mod.py",
            qualname="helper",
            snapshot=(repo, commit),
            line_span=(3, 6),
        ),
        function_meta(
            goid=7002,
            rel_path="mod.py",
            qualname="main",
            snapshot=(repo, commit),
            line_span=(8, 9),
        ),
    ]
    catalog = FunctionCatalog(functions=functions, module_by_path={"mod.py": "mod"})
    return FunctionCatalogService(catalog)


def _call_graph() -> nx.DiGraph:
    """Return a simple call graph where main calls helper.

    Returns
    -------
    nx.DiGraph
        Directed graph with an edge from main to helper.
    """
    graph = nx.DiGraph()
    graph.add_edge(7002, 7001)
    return graph


def test_function_effects_plugin_detects_transitive_effects(tmp_path: Path) -> None:
    """FunctionEffectsPlugin should mark direct and transitive side effects."""
    ctx = create_test_context(tmp_path)
    _seed_effect_sources(ctx.repo_root)
    catalog_provider = _make_catalog(ctx.repo, ctx.commit)

    engine = StubGraphEngine(
        gateway=ctx.gateway,
        snapshot=ctx.snapshot,
        call_graph=_call_graph(),
    )
    runtime = GraphRuntime(
        options=GraphRuntimeOptions(snapshot=ctx.snapshot),
        engine=engine,
    )
    plugin_ctx = PluginTestContext(
        gateway=ctx.gateway,
        snapshot=ctx.snapshot,
        paths=ctx.build_paths,
    )
    plugin_ctx.resources.catalog = catalog_provider
    plugin_ctx.resources.graph_runtime = runtime

    result = execute_target_plugin(FunctionEffectsPlugin(), plugin_ctx)
    assert result.success

    helper_row = ctx.query(
        """
        SELECT is_pure, modifies_globals, has_transitive_effects
        FROM analytics.function_effects
        WHERE function_goid_h128 = ?
        """,
        [7001],
    )[0]
    assert helper_row.is_pure is False
    assert helper_row.modifies_globals is True
    assert helper_row.has_transitive_effects is False

    main_row = ctx.query(
        """
        SELECT is_pure, has_transitive_effects
        FROM analytics.function_effects
        WHERE function_goid_h128 = ?
        """,
        [7002],
    )[0]
    assert main_row.is_pure is False
    assert main_row.has_transitive_effects is True

    ctx.close()
