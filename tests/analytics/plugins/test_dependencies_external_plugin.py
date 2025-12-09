"""Integration test for ExternalDepsPlugin."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.plugins.dependencies.external import ExternalDepsPlugin
from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogService
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.assertions.logging_assertions import assert_logged
from tests._helpers.fakes.contexts import TargetResourceOverrides
from tests._helpers.rows import function_meta
from tests.analytics.conftest import PluginTestHarness


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


def test_external_deps_plugin_builds_dependency_rows(plugin_harness: PluginTestHarness) -> None:
    """ExternalDepsPlugin should populate dependency tables from imports."""
    _seed_dependency_sources(plugin_harness.ctx.repo_root)
    catalog_provider = _catalog_for_dependencies(plugin_harness.ctx.repo, plugin_harness.ctx.commit)

    resources = TargetResourceOverrides(catalog=catalog_provider)
    result = plugin_harness.execute_plugin(ExternalDepsPlugin(), resources=resources)
    expect_true(result.success)

    calls_count = plugin_harness.ctx.query_count("analytics.external_dependency_calls")
    deps_count = plugin_harness.ctx.query_count("analytics.external_dependencies")
    expect_true(calls_count >= 1)
    expect_true(deps_count >= 1)


def test_external_deps_plugin_logs_when_patterns_missing(
    plugin_harness: PluginTestHarness, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing dependency patterns should warn and skip inserts."""
    caplog.set_level("WARNING")
    catalog_provider = _catalog_for_dependencies(plugin_harness.ctx.repo, plugin_harness.ctx.commit)
    resources = TargetResourceOverrides(catalog=catalog_provider)

    result = plugin_harness.execute_plugin(ExternalDepsPlugin(), resources=resources)

    expect_true(result.success)
    assert_logged(caplog.records, level="WARNING", containing="Dependency patterns file not found")
    calls_count = plugin_harness.ctx.query_count("analytics.external_dependency_calls")
    deps_count = plugin_harness.ctx.query_count("analytics.external_dependencies")
    expect_equal(calls_count, 0)
    expect_equal(deps_count, 0)


def test_external_deps_plugin_logs_on_invalid_patterns(
    plugin_harness: PluginTestHarness, caplog: pytest.LogCaptureFixture
) -> None:
    """Invalid dependency patterns YAML should log and skip inserts."""
    caplog.set_level("WARNING")
    config_dir = plugin_harness.ctx.repo_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "dependency_patterns.yml").write_text("libs: - not_a_mapping", encoding="utf-8")
    catalog_provider = _catalog_for_dependencies(plugin_harness.ctx.repo, plugin_harness.ctx.commit)
    resources = TargetResourceOverrides(catalog=catalog_provider)

    result = plugin_harness.execute_plugin(ExternalDepsPlugin(), resources=resources)

    expect_true(result.success)
    assert_logged(caplog.records, level="WARNING", containing="Failed to parse dependency patterns")
    assert_logged(
        caplog.records, level="WARNING", containing="No dependency patterns loaded; skipping"
    )
    calls_count = plugin_harness.ctx.query_count("analytics.external_dependency_calls")
    deps_count = plugin_harness.ctx.query_count("analytics.external_dependencies")
    expect_equal(calls_count, 0)
    expect_equal(deps_count, 0)
