"""Tests for FunctionRepository."""

from __future__ import annotations

from datetime import UTC, datetime

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.repositories.functions import FunctionRepository
from tests._helpers import ProvisionedGateway
from tests._helpers.assertions import (
    expect_empty,
    expect_equal,
    expect_in,
    expect_is_none,
    expect_is_not_none,
    expect_length,
)


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
    fresh_gateway: StorageGateway,
) -> None:
    """Verify list_high_risk_functions applies tested_only filter."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    con.execute(
        """
        INSERT INTO analytics.goid_risk_factors (
            repo, commit, function_goid_h128, urn, rel_path, qualname,
            risk_score, risk_level, coverage_ratio, tested,
            complexity_bucket, typedness_bucket, hotspot_score, created_at
        ) VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "test/repo",
            "abc123",
            1,
            "urn:tested_fn",
            "test.py",
            "tested_fn",
            0.8,
            "high",
            0.5,
            True,
            "medium",
            "high",
            0.3,
            now,
            "test/repo",
            "abc123",
            2,
            "urn:untested_fn",
            "test.py",
            "untested_fn",
            0.9,
            "critical",
            0.0,
            False,
            "high",
            "low",
            0.5,
            now,
        ],
    )

    repo = FunctionRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    tested_only_result = repo.list_high_risk_functions(min_risk=0.0, limit=10, tested_only=True)
    all_result = repo.list_high_risk_functions(min_risk=0.0, limit=10, tested_only=False)

    expect_length(tested_only_result, 1)
    expect_is_not_none(tested_only_result[0]["tested"])

    expected_all_count = 2
    expect_length(all_result, expected_all_count)


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


def test_get_function_profile_returns_row(fresh_gateway: StorageGateway) -> None:
    """Verify get_function_profile returns row when found."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    con.execute(
        """
        INSERT INTO analytics.function_profile (
            repo, commit, function_goid_h128, urn, rel_path, qualname,
            module, language, loc, cyclomatic_complexity, complexity_bucket,
            doc_short, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "test/repo",
            "abc123",
            1,
            "urn:test_fn",
            "test.py",
            "test_fn",
            "test_mod",
            "python",
            10,
            2,
            "low",
            "Test function",
            now,
        ],
    )

    repo = FunctionRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
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
