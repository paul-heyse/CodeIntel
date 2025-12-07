"""Integration test for SemanticRolesPlugin."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.analytics.plugins.semantic_roles.compute import SemanticRolesPlugin
from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogService
from tests._helpers.builders import insert_rows
from tests._helpers.context import TestContext, create_test_context
from tests._helpers.plugin_execution import PluginTestContext, execute_target_plugin
from tests._helpers.rows import function_meta, function_metrics_row, module_row


def _seed_test_module(repo_root: Path) -> None:
    """Write a simple test module."""
    tests_dir = repo_root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_sample.py").write_text(
        "\n".join(
            [
                "def test_example() -> None:",
                "    assert 1 == 1",
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


def test_semantic_roles_plugin_classifies_tests(tmp_path: Path) -> None:
    """SemanticRolesPlugin should classify test functions by path/name heuristics."""
    ctx = create_test_context(tmp_path)
    _seed_test_module(ctx.repo_root)
    catalog_provider = _make_catalog(ctx.repo, ctx.commit)

    now = datetime.now(tz=UTC)
    _insert_function_metrics(ctx, now)
    _insert_module_row(ctx)

    plugin_ctx = PluginTestContext(
        gateway=ctx.gateway,
        snapshot=ctx.snapshot,
        paths=ctx.build_paths,
    )
    plugin_ctx.resources.catalog = catalog_provider

    result = execute_target_plugin(SemanticRolesPlugin(), plugin_ctx)
    assert result.success

    role_row = ctx.query(
        """
        SELECT role, role_confidence
        FROM analytics.semantic_roles_functions
        WHERE function_goid_h128 = ?
        """,
        [7101],
    )[0]
    assert role_row.role == "test"
    confidence = role_row.role_confidence
    if not isinstance(confidence, float):
        pytest.fail(f"Expected float role_confidence, got {type(confidence)}")
    assert confidence > 0.5

    module_rows = ctx.query(
        """
        SELECT role
        FROM analytics.semantic_roles_modules
        WHERE module = ?
        """,
        ["tests.test_sample"],
    )
    assert module_rows, "expected semantic role entry for module"

    ctx.close()
