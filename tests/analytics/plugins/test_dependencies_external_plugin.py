"""Integration test for ExternalDepsPlugin."""

from __future__ import annotations

from pathlib import Path

from codeintel.analytics.plugins.dependencies.external import ExternalDepsPlugin
from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogService
from tests._helpers.context import create_test_context
from tests._helpers.plugin_execution import PluginTestContext, execute_target_plugin
from tests._helpers.rows import function_meta


def _seed_dependency_sources(ctx_repo_root: Path) -> None:
    """Write simple modules that import an external library."""
    pkg_dir = ctx_repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    config_dir = ctx_repo_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "dependency_patterns.yml").write_text(
        'libs:\n  requests:\n    patterns:\n      - mode: ["read"]\n        method: "get"\n',
        encoding="utf-8",
    )
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "client.py").write_text(
        "\n".join(
            [
                "import requests",
                "",
                "def fetch(url: str) -> int:",
                "    response = requests.get(url)",
                "    return response.status_code",
            ]
        ),
        encoding="utf-8",
    )
    (pkg_dir / "worker.py").write_text(
        "\n".join(
            [
                "from pkg.client import fetch",
                "",
                "def run() -> int:",
                "    return fetch('http://example.com')",
            ]
        ),
        encoding="utf-8",
    )


def _catalog_for_dependencies(repo: str, commit: str) -> FunctionCatalogService:
    """Build a catalog with spans for the dependency functions.

    Returns
    -------
    FunctionCatalogService
        Catalog provider with spans for seeded functions.
    """
    functions = [
        function_meta(
            goid=8001,
            rel_path="pkg/client.py",
            qualname="fetch",
            snapshot=(repo, commit),
            line_span=(3, 5),
        ),
        function_meta(
            goid=8002,
            rel_path="pkg/worker.py",
            qualname="run",
            snapshot=(repo, commit),
            line_span=(3, 4),
        ),
    ]
    catalog = FunctionCatalog(
        functions=functions,
        module_by_path={
            "pkg/client.py": "pkg.client",
            "pkg/worker.py": "pkg.worker",
        },
    )
    return FunctionCatalogService(catalog)


def test_external_deps_plugin_builds_dependency_rows(tmp_path: Path) -> None:
    """ExternalDepsPlugin should populate dependency tables from imports."""
    ctx = create_test_context(tmp_path)
    _seed_dependency_sources(ctx.repo_root)
    catalog_provider = _catalog_for_dependencies(ctx.repo, ctx.commit)

    plugin_ctx = PluginTestContext(
        gateway=ctx.gateway,
        snapshot=ctx.snapshot,
        paths=ctx.build_paths,
    )
    plugin_ctx.resources.catalog = catalog_provider

    result = execute_target_plugin(ExternalDepsPlugin(), plugin_ctx)
    assert result.success

    calls_count = ctx.query_count("analytics.external_dependency_calls")
    deps_count = ctx.query_count("analytics.external_dependencies")
    assert calls_count >= 1
    assert deps_count >= 1

    ctx.close()
