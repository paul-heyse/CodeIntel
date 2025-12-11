"""Integration test for SemanticRolesPlugin."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.plugins.semantic_roles.compute import SemanticRolesPlugin
from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogService
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.builders import insert_rows
from tests._helpers.catalogs import ensure_catalog_with_goids
from tests._helpers.fakes.contexts import TargetResourceOverrides
from tests._helpers.graphs import canonical_ast_artifacts
from tests._helpers.harnesses import plugin_harness_with_packs
from tests._helpers.rows import function_meta, function_metrics_row, module_row
from tests._helpers.seeds import CORE_PACK

if TYPE_CHECKING:
    from pathlib import Path

    from tests._helpers.context import TestContext

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


def _catalog_with_tests(ctx: TestContext) -> FunctionCatalogService:
    """Construct a catalog using canonical AST artifacts plus a test function.

    Returns
    -------
    FunctionCatalogService
        Catalog service combining canonical artifacts with a sample test function.
    """
    artifacts = canonical_ast_artifacts(ctx)
    test_function = function_meta(
        goid=7101,
        rel_path="tests/test_sample.py",
        qualname="test_example",
        snapshot=(ctx.repo, ctx.commit),
        line_span=(1, 2),
    )
    module_by_path = dict(artifacts.catalog.module_by_path)
    module_by_path["tests/test_sample.py"] = "tests.test_sample"
    catalog = FunctionCatalog(functions=[test_function], module_by_path=module_by_path)
    return FunctionCatalogService(catalog)


def _apply_catalog(ctx: TestContext, catalog_provider: FunctionCatalogService) -> None:
    """Ensure GOIDs are seeded for the provided catalog."""
    ensure_catalog_with_goids(ctx, catalog_provider)


def _insert_function_metrics(ctx: TestContext, created_at: datetime, goid: int) -> None:
    """Insert minimal function metrics for the test function."""
    insert_rows(
        ctx.gateway,
        [
            function_metrics_row(
                goid=goid,
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
    with plugin_harness_with_packs(tmp_path, CORE_PACK) as harness:
        _seed_test_module(harness.ctx.repo_root)
        catalog_provider = _catalog_with_tests(harness.ctx)
        _apply_catalog(harness.ctx, catalog_provider)

        now = datetime.now(tz=UTC)
        _insert_function_metrics(harness.ctx, now, goid=7101)
        _insert_module_row(harness.ctx)

        resources = TargetResourceOverrides(catalog=catalog_provider)
        result = harness.execute_plugin(SemanticRolesPlugin(), resources=resources)
        expect_true(result.success)

        role_row = harness.ctx.query(
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

        module_rows = harness.ctx.query(
            """
            SELECT role
            FROM analytics.semantic_roles_modules
            WHERE module = ?
            """,
            ["tests.test_sample"],
        )
        expect_true(module_rows, message="expected semantic role entry for module")
