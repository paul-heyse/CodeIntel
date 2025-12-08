"""Integration tests for function AST feature and contract plugins."""

from __future__ import annotations

from pathlib import Path

from codeintel.analytics.plugins.functions.ast_features import FunctionAstFeaturesPlugin
from codeintel.analytics.plugins.functions.contracts import FunctionContractsPlugin
from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogService
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.plugin_execution import execute_target_plugin
from tests._helpers.rows import function_meta
from tests.analytics.conftest import PluginTestHarness


def _seed_function_sources(repo_root: Path) -> None:
    """Write simple source files to satisfy AST loading."""
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "main.py").write_text(
        "\n".join(
            [
                "def main(value: int) -> int:",
                "    if value <= 0:",
                "        raise ValueError('bad')",
                "    return value * 2",
                "",
                "def helper(text: str) -> str:",
                "    return text.upper()",
            ]
        ),
        encoding="utf-8",
    )


def _make_catalog(ctx_repo: str, ctx_commit: str) -> FunctionCatalogService:
    """Construct a catalog provider aligned to written sources.

    Returns
    -------
    FunctionCatalogService
        Catalog provider for the seeded functions.
    """
    functions = [
        function_meta(
            goid=9001,
            rel_path="main.py",
            qualname="main",
            snapshot=(ctx_repo, ctx_commit),
            line_span=(1, 5),
        ),
        function_meta(
            goid=9002,
            rel_path="main.py",
            qualname="helper",
            snapshot=(ctx_repo, ctx_commit),
            line_span=(7, 8),
        ),
    ]
    catalog = FunctionCatalog(functions=functions, module_by_path={"main.py": "main"})
    return FunctionCatalogService(catalog)


def test_function_ast_features_plugin(plugin_harness: PluginTestHarness) -> None:
    """FunctionAstFeaturesPlugin should persist feature rows for catalog functions."""
    _seed_function_sources(plugin_harness.ctx.repo_root)
    catalog_provider = _make_catalog(plugin_harness.ctx.repo, plugin_harness.ctx.commit)

    plugin_harness.plugin_ctx.resources.catalog = catalog_provider

    result = execute_target_plugin(FunctionAstFeaturesPlugin(), plugin_harness.plugin_ctx)
    expect_true(result.success)
    expect_equal(plugin_harness.ctx.query_count("analytics.function_ast_features"), 2)


def test_function_contracts_plugin(plugin_harness: PluginTestHarness) -> None:
    """FunctionContractsPlugin should derive contracts from AST-loaded functions."""
    _seed_function_sources(plugin_harness.ctx.repo_root)
    catalog_provider = _make_catalog(plugin_harness.ctx.repo, plugin_harness.ctx.commit)

    plugin_harness.plugin_ctx.resources.catalog = catalog_provider

    result = execute_target_plugin(FunctionContractsPlugin(), plugin_harness.plugin_ctx)
    expect_true(result.success)
    expect_true(plugin_harness.ctx.query_count("analytics.function_contracts") >= 1)
