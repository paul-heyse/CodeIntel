"""Integration tests for function AST feature and contract plugins."""

from __future__ import annotations

from pathlib import Path

from codeintel.analytics.plugins.functions.ast_features import FunctionAstFeaturesPlugin
from codeintel.analytics.plugins.functions.contracts import FunctionContractsPlugin
from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogService, FunctionMeta
from tests._helpers.context import create_test_context
from tests._helpers.plugin_execution import PluginTestContext, execute_target_plugin


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
    functions: list[FunctionMeta] = [
        FunctionMeta(
            goid=9001,
            urn=f"urn:{ctx_repo}:{ctx_commit}:main.py#main",
            rel_path="main.py",
            qualname="main",
            start_line=1,
            end_line=5,
        ),
        FunctionMeta(
            goid=9002,
            urn=f"urn:{ctx_repo}:{ctx_commit}:main.py#helper",
            rel_path="main.py",
            qualname="helper",
            start_line=7,
            end_line=8,
        ),
    ]
    catalog = FunctionCatalog(functions=functions, module_by_path={"main.py": "main"})
    return FunctionCatalogService(catalog)


def test_function_ast_features_plugin(tmp_path: Path) -> None:
    """FunctionAstFeaturesPlugin should persist feature rows for catalog functions."""
    ctx = create_test_context(tmp_path)
    _seed_function_sources(ctx.repo_root)
    catalog_provider = _make_catalog(ctx.repo, ctx.commit)

    plugin_ctx = PluginTestContext(
        gateway=ctx.gateway,
        snapshot=ctx.snapshot,
        paths=ctx.build_paths,
    )
    plugin_ctx.resources.catalog = catalog_provider

    result = execute_target_plugin(FunctionAstFeaturesPlugin(), plugin_ctx)
    assert result.success
    assert ctx.query_count("analytics.function_ast_features") == 2

    ctx.close()


def test_function_contracts_plugin(tmp_path: Path) -> None:
    """FunctionContractsPlugin should derive contracts from AST-loaded functions."""
    ctx = create_test_context(tmp_path)
    _seed_function_sources(ctx.repo_root)
    catalog_provider = _make_catalog(ctx.repo, ctx.commit)

    plugin_ctx = PluginTestContext(
        gateway=ctx.gateway,
        snapshot=ctx.snapshot,
        paths=ctx.build_paths,
    )
    plugin_ctx.resources.catalog = catalog_provider

    result = execute_target_plugin(FunctionContractsPlugin(), plugin_ctx)
    assert result.success
    assert ctx.query_count("analytics.function_contracts") >= 1

    ctx.close()
