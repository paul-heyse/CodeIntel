"""Integration test for SemanticRolesPlugin."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.analytics.plugins.semantic_roles.compute import SemanticRolesPlugin
from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogService
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.builders import insert_rows
from tests._helpers.context import TestContext
from tests._helpers.plugin_execution import execute_target_plugin
from tests._helpers.rows import function_meta, function_metrics_row, module_row
from tests.analytics.conftest import PluginTestHarness

MIN_ROLE_CONFIDENCE = 0.5


def _seed_test_module(repo_root: Path) -> None:
    """Write a simple test module."""
    tests_dir = repo_root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_sample.py").write_text(
        "\n".join(
            [
                "def test_example() -> None:",
                "    return None",
            ]
        ),
        encoding="utf-8",
    )


def _make_catalog(repo: str, commit: str) -> FunctionCatalogService:
    """Construct a catalog aligned with the seeded test module.

    Returns
    -------
    FunctionCatalogService
        Catalog provider for the test function.
    """
    functions = [
        function_meta(
            goid=7101,
            rel_path="tests/test_sample.py",
            qualname="test_example",
            snapshot=(repo, commit),
            line_span=(1, 2),
        )
    ]
    catalog = FunctionCatalog(
        functions=functions,
        module_by_path={"tests/test_sample.py": "tests.test_sample"},
    )
    return FunctionCatalogService(catalog)


def _insert_function_metrics(ctx: TestContext, created_at: datetime) -> None:
    """Insert minimal function metrics for the test function."""
    insert_rows(
        ctx.gateway,
        [
            function_metrics_row(
                goid=7101,
                rel_path="tests/test_sample.py",
                qualname="test_example",
                snapshot=(ctx.repo, ctx.commit),
                metrics={"start_line": 1, "end_line": 2, "created_at": created_at},
            )
        ],
    )


def _insert_module_row(ctx: TestContext) -> None:
    """Insert module metadata for the test module."""
    insert_rows(
        ctx.gateway,
        [
            module_row(
                module="tests.test_sample",
                path="tests/test_sample.py",
                snapshot=(ctx.repo, ctx.commit),
            )
        ],
    )


def test_semantic_roles_plugin_classifies_tests(plugin_harness: PluginTestHarness) -> None:
    """SemanticRolesPlugin should classify test functions by path/name heuristics."""
    _seed_test_module(plugin_harness.ctx.repo_root)
    catalog_provider = _make_catalog(plugin_harness.ctx.repo, plugin_harness.ctx.commit)

    now = datetime.now(tz=UTC)
    _insert_function_metrics(plugin_harness.ctx, now)
    _insert_module_row(plugin_harness.ctx)

    plugin_harness.plugin_ctx.resources.catalog = catalog_provider

    result = execute_target_plugin(SemanticRolesPlugin(), plugin_harness.plugin_ctx)
    expect_true(result.success)

    role_row = plugin_harness.ctx.query(
        """
        SELECT role, role_confidence
        FROM analytics.semantic_roles_functions
        WHERE function_goid_h128 = ?
        """,
        [7101],
    )[0]
    expect_equal(role_row.role, "test")
    confidence = role_row.role_confidence
    if not isinstance(confidence, float):
        pytest.fail(f"Expected float role_confidence, got {type(confidence)}")
    expect_true(confidence > MIN_ROLE_CONFIDENCE)

    module_rows = plugin_harness.ctx.query(
        """
        SELECT role
        FROM analytics.semantic_roles_modules
        WHERE module = ?
        """,
        ["tests.test_sample"],
    )
    expect_true(module_rows, message="expected semantic role entry for module")
