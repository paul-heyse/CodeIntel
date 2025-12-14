"""Integration tests for function AST feature and contract plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.plugins.analytics.functions.ast_features import FunctionAstFeaturesPlugin
from codeintel.build.plugins.analytics.functions.contracts import FunctionContractsPlugin
from codeintel.core.catalog import CatalogService, FunctionCatalog
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.catalogs import ensure_catalog_with_goids
from tests._helpers.fakes.contexts import TargetResourceOverrides
from tests._helpers.harnesses import plugin_harness_with_packs
from tests._helpers.rows import function_meta
from tests._helpers.seeds import CORE_PACK

if TYPE_CHECKING:
    from pathlib import Path


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


def _make_catalog(ctx_repo: str, ctx_commit: str) -> CatalogService:
    """Construct a catalog provider aligned to written sources.

    Returns
    -------
    CatalogService
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
    return CatalogService(catalog)


def test_function_ast_features_plugin(tmp_path: Path) -> None:
    """FunctionAstFeaturesPlugin should persist feature rows for catalog functions."""
    with plugin_harness_with_packs(tmp_path, CORE_PACK) as harness:
        _seed_function_sources(harness.ctx.repo_root)
        catalog_provider = _make_catalog(harness.ctx.repo, harness.ctx.commit)
        ensure_catalog_with_goids(harness.ctx, catalog_provider)

        resources = TargetResourceOverrides(catalog=catalog_provider)
        result = harness.execute_plugin(FunctionAstFeaturesPlugin(), resources=resources)
        expect_true(result.success)
        expect_equal(harness.ctx.query_count("analytics.function_ast_features"), 2)


def test_function_contracts_plugin(tmp_path: Path) -> None:
    """FunctionContractsPlugin should derive contracts from AST-loaded functions."""
    with plugin_harness_with_packs(tmp_path, CORE_PACK) as harness:
        _seed_function_sources(harness.ctx.repo_root)
        catalog_provider = _make_catalog(harness.ctx.repo, harness.ctx.commit)
        ensure_catalog_with_goids(harness.ctx, catalog_provider)

        resources = TargetResourceOverrides(catalog=catalog_provider)
        result = harness.execute_plugin(FunctionContractsPlugin(), resources=resources)
        expect_true(result.success)
        expect_true(harness.ctx.query_count("analytics.function_contracts") >= 1)
