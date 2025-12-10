"""Tests for FunctionRepository."""

from __future__ import annotations

from datetime import UTC, datetime

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.gateway.insert_helpers import insert_rows as insert_mapping_rows
from codeintel.storage.repositories.functions import FunctionRepository
from tests._helpers import ProvisionedGateway
from tests._helpers.assertions import (
    expect_empty,
    expect_equal,
    expect_in,
    expect_is_none,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.builders import RiskFactorRow, insert_rows
from tests._helpers.context import TestContext
from tests._helpers.rows import function_metrics_row, function_profile_row


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
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_function_summary_by_goid returns None when no match."""
    repo = FunctionRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    result = repo.get_function_summary_by_goid(99999)

    expect_is_none(result)


def test_list_function_summaries_for_file_returns_empty_when_no_match(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify list_function_summaries_for_file returns empty list when no match."""
    repo = FunctionRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    result = repo.list_function_summaries_for_file("nonexistent.py")

    expect_empty(result)


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
    now = datetime.now(tz=UTC)
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
    insert_rows(
        metrics_ctx.gateway,
        [
            RiskFactorRow(
                function_goid_h128=untested_goid,
                urn=f"urn:{metrics_ctx.repo}:{metrics_ctx.commit}:test.py#untested_fn",
                repo=metrics_ctx.repo,
                commit=metrics_ctx.commit,
                rel_path="test.py",
                language="python",
                kind="function",
                qualname="untested_fn",
                loc=12,
                logical_loc=10,
                cyclomatic_complexity=5,
                complexity_bucket="high",
                typedness_bucket="low",
                typedness_source="inferred",
                hotspot_score=0.5,
                file_typed_ratio=0.5,
                static_error_count=1,
                has_static_errors=True,
                executable_lines=2,
                covered_lines=0,
                coverage_ratio=0.0,
                tested=False,
                test_count=0,
                failing_test_count=0,
                last_test_status="unknown",
                risk_score=0.9,
                risk_level="critical",
                tags="[]",
                owners="[]",
                created_at=now,
            )
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
    insert_mapping_rows(
        metrics_ctx.con,
        "analytics.function_profile",
        [
            function_profile_row(
                goid=1,
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
    docs_export_gateway: ProvisionedGateway,
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
