"""Unit tests for plugin contract helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from codeintel.analytics.datasets import (
    get_analytics_dataset_contract,
    insert_analytics_rows,
)
from codeintel.analytics.graphs.contracts import (
    ContractChecker,
    assert_columns_present,
    assert_not_null_fraction,
    assert_table_exists,
    assert_table_not_empty,
    columns_present_checker,
    not_null_fraction_checker,
    table_exists_checker,
    table_not_empty_checker,
)
from codeintel.analytics.rows.graph_metrics import FunctionGraphMetricsRow
from codeintel.storage.gateway import StorageGateway, open_memory_gateway


def _gateway_with_function_metrics(repo: str, commit: str) -> StorageGateway:
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    now = datetime.now(tz=UTC)
    contract = get_analytics_dataset_contract(gateway, "analytics.graph_metrics_functions")
    insert_analytics_rows(
        gateway,
        contract,
        [
            FunctionGraphMetricsRow(
                repo=repo,
                commit=commit,
                function_goid_h128=1,
                call_fan_in=2,
                call_fan_out=3,
                call_in_degree=2,
                call_out_degree=3,
                call_pagerank=0.5,
                call_betweenness=0.1,
                call_closeness=0.2,
                call_cycle_member=False,
                call_cycle_id=0,
                call_layer=1,
                created_at=now,
            )
        ],
    )
    return gateway


def test_assert_table_not_empty_passes_for_seeded_gateway() -> None:
    """Contract should pass when table contains rows."""
    repo = "demo/repo"
    commit = "deadbeef"
    gateway = _gateway_with_function_metrics(repo, commit)
    result = assert_table_not_empty(
        gateway,
        table="analytics.graph_metrics_functions",
        repo=repo,
        commit=commit,
    )
    if result.status != "passed":
        pytest.fail("Expected contract to pass when rows exist")


def test_assert_table_not_empty_fails_for_missing_rows() -> None:
    """Contract should fail when no rows are present."""
    repo = "demo/repo"
    commit = "deadbeef"
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    result = assert_table_not_empty(
        gateway,
        table="analytics.graph_metrics_functions",
        repo=repo,
        commit=commit,
    )
    if result.status != "failed":
        pytest.fail("Expected contract to fail when no rows are present")
    if "graph_metrics_functions" not in (result.message or ""):
        pytest.fail("Expected failure message to reference table name")


def test_table_exists_and_columns_present() -> None:
    """Helpers should validate table presence and required columns."""
    repo = "demo/repo"
    commit = "deadbeef"
    gateway = _gateway_with_function_metrics(repo, commit)
    exists_result = assert_table_exists(gateway, table="analytics.graph_metrics_functions")
    if exists_result.status != "passed":
        pytest.fail("Expected table existence contract to pass")
    columns_result = assert_columns_present(
        gateway,
        table="analytics.graph_metrics_functions",
        expected_columns={"repo", "commit", "pagerank"},
    )
    if columns_result.status != "passed":
        pytest.fail("Expected columns-present contract to pass")


def test_not_null_fraction_checker_flags_missing_values() -> None:
    """Non-null fraction helper should fail when data are missing."""
    repo = "demo/repo"
    commit = "deadbeef"
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    now = datetime.now(tz=UTC)
    contract = get_analytics_dataset_contract(gateway, "analytics.graph_metrics_functions")
    insert_analytics_rows(
        gateway,
        contract,
        [
            FunctionGraphMetricsRow(
                repo=repo,
                commit=commit,
                function_goid_h128=1,
                call_fan_in=2,
                call_fan_out=3,
                call_in_degree=2,
                call_out_degree=3,
                call_pagerank=None,
                call_betweenness=None,
                call_closeness=None,
                call_cycle_member=False,
                call_cycle_id=0,
                call_layer=1,
                created_at=now,
            )
        ],
    )
    result = assert_not_null_fraction(
        gateway,
        table="analytics.graph_metrics_functions",
        column="pagerank",
        repo=repo,
        commit=commit,
        min_fraction=0.5,
    )
    if result.status != "failed":
        pytest.fail("Expected not-null fraction contract to fail")


def test_checker_builders_wrap_helpers() -> None:
    """Builder helpers should adapt to ContractChecker signature."""
    repo = "demo/repo"
    commit = "deadbeef"
    gateway = _gateway_with_function_metrics(repo, commit)

    @dataclass
    class DummyCtx:
        gateway: StorageGateway
        repo: str
        commit: str

    ctx = DummyCtx(gateway=gateway, repo=repo, commit=commit)
    checkers: tuple[ContractChecker, ...] = (
        table_not_empty_checker("analytics.graph_metrics_functions"),
        table_exists_checker("analytics.graph_metrics_functions"),
        columns_present_checker(
            "analytics.graph_metrics_functions",
            expected_columns={"repo", "commit"},
        ),
        not_null_fraction_checker(
            "analytics.graph_metrics_functions",
            column="pagerank",
            min_fraction=0.5,
        ),
    )
    for checker in checkers:
        result = checker(ctx)
        if result.status != "passed":
            pytest.fail(f"Expected checker to pass: {result}")
