"""Tests for FunctionRepository."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from codeintel.storage.repositories.functions import FunctionRepository
from codeintel.storage.warehouse import Warehouse
from tests._helpers.assertions import (
    expect_empty,
    expect_equal,
    expect_in,
    expect_is_none,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.fixtures.rows import (
    FunctionValidationRow,
    function_metrics_row,
    function_profile_row,
    insert_rows,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.context import TestContext


TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
VALIDATION_GOID_ALPHA = 900_001
VALIDATION_GOID_BETA = 900_002


def test_resolve_function_goid_passthrough(fresh_gateway: StorageGateway) -> None:
    """Verify resolve_function_goid returns passthrough when goid_h128 provided."""
    repo = FunctionRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    result = repo.resolve_function_goid(goid_h128=12345)

    expected_goid = 12345
    expect_equal(result, expected_goid)


def test_resolve_function_goid_returns_none_when_no_identifiers(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify resolve_function_goid returns None when no identifiers provided."""
    repo = FunctionRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    result = repo.resolve_function_goid()

    expect_is_none(result)


def test_get_function_summary_by_goid_returns_none_when_not_found(
    docs_views_inferred_gateway: StorageGateway,
) -> None:
    """Verify get_function_summary_by_goid returns None when no match."""
    repo = FunctionRepository(
        gateway=docs_views_inferred_gateway,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
    )

    result = repo.get_function_summary_by_goid(99999)

    expect_is_none(result)


def test_list_function_summaries_for_file_returns_empty_when_no_match(
    docs_views_inferred_gateway: StorageGateway,
) -> None:
    """Verify list_function_summaries_for_file returns empty list when no match."""
    repo = FunctionRepository(
        gateway=docs_views_inferred_gateway,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
    )

    result = repo.list_function_summaries_for_file("nonexistent.py")

    expect_empty(result)


def test_list_function_validation_filters_by_goid(metrics_ctx: TestContext) -> None:
    """Verify list_function_validation filters by GOID and orders by newest first."""
    rows = [
        FunctionValidationRow(
            repo=metrics_ctx.repo,
            commit=metrics_ctx.commit,
            function_goid_h128=VALIDATION_GOID_ALPHA,
            rel_path="src/alpha.py",
            qualname="alpha",
            issue="parse_failed",
            detail="old",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
        FunctionValidationRow(
            repo=metrics_ctx.repo,
            commit=metrics_ctx.commit,
            function_goid_h128=VALIDATION_GOID_ALPHA,
            rel_path="src/alpha.py",
            qualname="alpha",
            issue="span_not_found",
            detail="new",
            created_at=datetime(2024, 1, 2, tzinfo=UTC),
        ),
        FunctionValidationRow(
            repo=metrics_ctx.repo,
            commit=metrics_ctx.commit,
            function_goid_h128=VALIDATION_GOID_BETA,
            rel_path="src/beta.py",
            qualname="beta",
            issue="unknown_function",
            detail="other",
            created_at=datetime(2024, 1, 3, tzinfo=UTC),
        ),
    ]
    insert_rows(metrics_ctx.gateway, rows)

    repo = FunctionRepository(
        gateway=metrics_ctx.gateway,
        repo=metrics_ctx.repo,
        commit=metrics_ctx.commit,
    )

    results = repo.list_function_validation(goid_h128=VALIDATION_GOID_ALPHA)

    expect_equal(len(results), 2)
    expect_equal(results[0]["detail"], "new")
    expect_true(
        all(row["function_goid_h128"] == VALIDATION_GOID_ALPHA for row in results),
        message="results should include only the requested GOID",
    )


def test_list_high_risk_functions_returns_empty_when_no_match(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify list_high_risk_functions returns empty list when no data."""
    repo = FunctionRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    result = repo.list_high_risk_functions(min_risk=0.0, limit=10, tested_only=False)

    expect_empty(result)


def test_list_high_risk_functions_with_tested_only_filter(
    metrics_ctx: TestContext,
) -> None:
    """Verify list_high_risk_functions applies tested_only filter."""
    untested_goid = 999_001
    insert_rows(
        metrics_ctx.gateway,
        [
            function_metrics_row(
                goid=untested_goid,
                rel_path="test.py",
                qualname="untested_fn",
                snapshot=(metrics_ctx.repo, metrics_ctx.commit),
                metrics={"complexity_bucket": "high", "cyclomatic_complexity": 5},
            )
        ],
    )
    warehouse = Warehouse(metrics_ctx.gateway)
    warehouse.materialize_mappings(
        table_key="analytics.function_profile",
        rows=[
            function_profile_row(
                goid=Decimal(untested_goid),
                repo=metrics_ctx.repo,
                commit=metrics_ctx.commit,
                rel_path="test.py",
                qualname="untested_fn",
                tested=False,
                risk_score=9.0,
                risk_level="high",
            ),
            function_profile_row(
                goid=Decimal(untested_goid + 1),
                repo=metrics_ctx.repo,
                commit=metrics_ctx.commit,
                rel_path="test.py",
                qualname="tested_fn",
                tested=True,
                risk_score=5.0,
                risk_level="medium",
            ),
        ],
    )

    repo = FunctionRepository(
        gateway=metrics_ctx.gateway,
        repo=metrics_ctx.repo,
        commit=metrics_ctx.commit,
    )

    tested_only_result = repo.list_high_risk_functions(min_risk=0.0, limit=10, tested_only=True)
    all_result = repo.list_high_risk_functions(min_risk=0.0, limit=10, tested_only=False)

    expect_true(
        any(bool(row.get("tested")) for row in tested_only_result),
        message="tested_only should include tested functions",
    )
    expect_is_not_none(tested_only_result[0]["tested"])

    expect_true(
        any(row["function_goid_h128"] == untested_goid for row in all_result),
        message="all_result should include untested_fn",
    )
    expect_true(
        all(row["function_goid_h128"] != untested_goid for row in tested_only_result),
        message="tested_only should exclude untested_fn",
    )


def test_get_function_profile_returns_none_when_not_found(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_function_profile returns None when no match."""
    repo = FunctionRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    result = repo.get_function_profile(99999)

    expect_is_none(result)


def test_get_function_profile_returns_row(metrics_ctx: TestContext) -> None:
    """Verify get_function_profile returns row when found."""
    warehouse = Warehouse(metrics_ctx.gateway)
    warehouse.materialize_mappings(
        "analytics.function_profile",
        [
            function_profile_row(
                goid=Decimal(1),
                qualname="test_fn",
                rel_path="test.py",
                repo=metrics_ctx.repo,
                commit=metrics_ctx.commit,
                doc_short="Test function",
            )
        ],
    )

    repo = FunctionRepository(
        gateway=metrics_ctx.gateway,
        repo=metrics_ctx.repo,
        commit=metrics_ctx.commit,
    )

    result = repo.get_function_profile(1)

    expect_is_not_none(result, message="Expected function profile row to exist.")
    if result is not None:
        expect_equal(result["qualname"], "test_fn")


def test_get_function_architecture_returns_none_when_not_found(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_function_architecture returns None when no match."""
    repo = FunctionRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    result = repo.get_function_architecture(99999)

    expect_is_none(result)


def test_list_function_goids_returns_empty_when_no_data(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify list_function_goids returns empty list when no functions."""
    repo = FunctionRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    result = repo.list_function_goids()

    expect_empty(result)


def test_function_repository_with_docs_export(
    docs_export_gateway: TestContext,
) -> None:
    """Verify FunctionRepository works with full docs export gateway."""
    repo = FunctionRepository(
        gateway=docs_export_gateway.gateway,
        repo=docs_export_gateway.repo,
        commit=docs_export_gateway.commit,
    )

    goid = expect_is_not_none(repo.resolve_function_goid(urn="urn:foo"))

    summary = repo.get_function_summary_by_goid(goid)
    expect_is_not_none(summary)
    if summary is not None:
        expect_equal(summary["qualname"], "pkg.foo:func")

    goids = repo.list_function_goids()
    if goid is not None:
        expect_in(goid, goids)
