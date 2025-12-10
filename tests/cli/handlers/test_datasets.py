"""Tests for datasets handlers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from codeintel.cli.config.model import CliConfig
from codeintel.cli.handlers.context import HandlerContext
from codeintel.cli.handlers.datasets import (
    DatasetDiffResult,
    DatasetLintResult,
    DatasetsListResult,
    DatasetSnapshotResult,
    datasets_diff_handler,
    datasets_lint_handler,
    datasets_list_handler,
    datasets_snapshot_handler,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)


def _make_mock_context(params: dict[str, Any]) -> HandlerContext:
    """Create a HandlerContext for testing.

    Parameters
    ----------
    params
        Parameters to include in the context.

    Returns
    -------
    HandlerContext
        Test context with provided params.
    """
    mock_config = MagicMock(spec=CliConfig)
    return HandlerContext(
        config=mock_config,
        operation_id="datasets.test",
        _params=params,
    )


def test_datasets_list_result_to_dict() -> None:
    """Verify DatasetsListResult.to_dict returns correct structure."""
    result = DatasetsListResult(
        datasets=[
            {
                "name": "test_dataset",
                "table_key": "test.table",
                "category": "",
                "description": "A test dataset",
            }
        ],
        count=1,
    )

    data = result.to_dict()

    expect_equal(data["count"], 1)
    datasets = data["datasets"]
    expect_true(isinstance(datasets, list))
    if isinstance(datasets, list):
        expect_equal(len(datasets), 1)
        expect_equal(datasets[0]["name"], "test_dataset")


def test_dataset_lint_result_to_dict() -> None:
    """Verify DatasetLintResult.to_dict returns correct structure."""
    result = DatasetLintResult(
        passed=True,
        issue_count=0,
        issues=[],
    )

    data = result.to_dict()

    expect_true(data["passed"])
    expect_equal(data["issue_count"], 0)
    expect_equal(data["issues"], [])


def test_dataset_lint_result_with_issues_to_dict() -> None:
    """Verify DatasetLintResult.to_dict with issues returns correct structure."""
    result = DatasetLintResult(
        passed=False,
        issue_count=2,
        issues=["Missing column: id", "Invalid type for column: name"],
    )

    data = result.to_dict()

    expect_true(not data["passed"])
    expect_equal(data["issue_count"], 2)
    issues = data["issues"]
    expect_true(isinstance(issues, list))
    if isinstance(issues, list):
        expect_equal(len(issues), 2)


def test_dataset_snapshot_result_to_dict() -> None:
    """Verify DatasetSnapshotResult.to_dict returns correct structure."""
    result = DatasetSnapshotResult(
        output_path="build/snapshot.json",
        datasets_count=5,
    )

    data = result.to_dict()

    expect_equal(data["output_path"], "build/snapshot.json")
    expect_equal(data["datasets_count"], 5)


def test_dataset_diff_result_to_dict() -> None:
    """Verify DatasetDiffResult.to_dict returns correct structure."""
    result = DatasetDiffResult(
        added=["new_dataset"],
        removed=["old_dataset"],
        changed=[],
        has_differences=True,
    )

    data = result.to_dict()

    expect_equal(data["added"], ["new_dataset"])
    expect_equal(data["removed"], ["old_dataset"])
    expect_equal(data["changed"], [])
    expect_true(data["has_differences"])


def test_dataset_diff_result_no_differences() -> None:
    """Verify DatasetDiffResult.to_dict when no differences."""
    result = DatasetDiffResult(
        added=[],
        removed=[],
        changed=[],
        has_differences=False,
    )

    data = result.to_dict()

    expect_true(not data["has_differences"])
    expect_equal(data["added"], [])
    expect_equal(data["removed"], [])


@patch("codeintel.cli.handlers.datasets._build_runtime_from_ctx")
@patch("codeintel.cli.handlers.datasets.get_dataset_contracts_by_table_key")
def test_datasets_list_handler_success(
    mock_get_contracts: MagicMock,
    mock_build_runtime: MagicMock,
) -> None:
    """Verify datasets_list_handler returns datasets successfully."""
    # Setup mocks
    mock_contract = MagicMock()
    mock_contract.name = "test_dataset"
    mock_contract.table_key = "test.table"
    mock_contract.description = "A test dataset"
    mock_get_contracts.return_value = {"test.table": mock_contract}

    mock_runtime = MagicMock()
    mock_runtime.gateway = MagicMock()
    mock_build_runtime.return_value = mock_runtime

    ctx = _make_mock_context({})

    result = datasets_list_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    data = result.data
    if data is not None:
        expect_equal(data.count, 1)
        expect_equal(data.datasets[0]["name"], "test_dataset")


@patch("codeintel.cli.handlers.datasets._build_runtime_from_ctx")
@patch("codeintel.cli.handlers.datasets.get_dataset_contracts_by_table_key")
def test_datasets_list_handler_empty(
    mock_get_contracts: MagicMock,
    mock_build_runtime: MagicMock,
) -> None:
    """Verify datasets_list_handler handles empty results."""
    mock_get_contracts.return_value = {}

    mock_runtime = MagicMock()
    mock_runtime.gateway = MagicMock()
    mock_build_runtime.return_value = mock_runtime

    ctx = _make_mock_context({})

    result = datasets_list_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.count, 0)
        expect_equal(data.datasets, [])


@patch("codeintel.cli.handlers.datasets._build_runtime_from_ctx")
@patch("codeintel.cli.handlers.datasets.collect_contract_issues")
def test_datasets_lint_handler_success(
    mock_collect_issues: MagicMock,
    mock_build_runtime: MagicMock,
) -> None:
    """Verify datasets_lint_handler returns success when no issues."""
    mock_collect_issues.return_value = []

    mock_runtime = MagicMock()
    mock_runtime.gateway = MagicMock()
    mock_runtime.gateway.con = MagicMock()
    mock_build_runtime.return_value = mock_runtime

    ctx = _make_mock_context({})

    result = datasets_lint_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_true(data.passed)
        expect_equal(data.issue_count, 0)


@patch("codeintel.cli.handlers.datasets._build_runtime_from_ctx")
@patch("codeintel.cli.handlers.datasets.collect_contract_issues")
def test_datasets_lint_handler_with_issues(
    mock_collect_issues: MagicMock,
    mock_build_runtime: MagicMock,
) -> None:
    """Verify datasets_lint_handler returns issues when found."""
    mock_collect_issues.return_value = ["Issue 1", "Issue 2"]

    mock_runtime = MagicMock()
    mock_runtime.gateway = MagicMock()
    mock_runtime.gateway.con = MagicMock()
    mock_build_runtime.return_value = mock_runtime

    ctx = _make_mock_context({})

    result = datasets_lint_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_true(not data.passed)
        expect_equal(data.issue_count, 2)


def test_datasets_snapshot_handler_missing_output() -> None:
    """Verify datasets_snapshot_handler fails without output parameter."""
    ctx = _make_mock_context({})

    result = datasets_snapshot_handler(ctx)

    expect_true(not result.success)
    error = result.error
    if error is not None:
        expect_equal(error.type, "urn:codeintel:datasets:missing-param")


@patch("codeintel.cli.handlers.datasets.get_dataset_contracts_by_table_key")
def test_datasets_snapshot_handler_success(
    mock_get_contracts: MagicMock,
    tmp_path: Path,
) -> None:
    """Verify datasets_snapshot_handler writes snapshot file."""
    mock_contract = MagicMock()
    mock_contract.name = "test_dataset"
    mock_contract.table_key = "test.table"
    mock_contract.description = "Test"
    mock_get_contracts.return_value = {"test.table": mock_contract}

    output_path = tmp_path / "snapshot.json"
    ctx = _make_mock_context({"output": str(output_path)})

    result = datasets_snapshot_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.datasets_count, 1)
    expect_true(output_path.exists())

    # Verify file content
    content = json.loads(output_path.read_text(encoding="utf-8"))
    expect_equal(len(content), 1)
    expect_equal(content[0]["name"], "test_dataset")


def test_datasets_diff_handler_missing_baseline() -> None:
    """Verify datasets_diff_handler fails without baseline_path parameter."""
    ctx = _make_mock_context({})

    result = datasets_diff_handler(ctx)

    expect_true(not result.success)
    error = result.error
    if error is not None:
        expect_equal(error.type, "urn:codeintel:datasets:missing-param")


def test_datasets_diff_handler_baseline_not_found(tmp_path: Path) -> None:
    """Verify datasets_diff_handler fails when baseline file not found."""
    ctx = _make_mock_context({"baseline_path": str(tmp_path / "nonexistent.json")})

    result = datasets_diff_handler(ctx)

    expect_true(not result.success)
    error = result.error
    if error is not None:
        expect_equal(error.type, "urn:codeintel:datasets:file-not-found")


@patch("codeintel.cli.handlers.datasets.get_dataset_contracts_by_table_key")
def test_datasets_diff_handler_success(
    mock_get_contracts: MagicMock,
    tmp_path: Path,
) -> None:
    """Verify datasets_diff_handler computes differences correctly."""
    # Current datasets
    mock_contract = MagicMock()
    mock_contract.name = "new_dataset"
    mock_get_contracts.return_value = {"new.table": mock_contract}

    # Baseline with old dataset
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps([{"name": "old_dataset", "table_key": "old.table"}]),
        encoding="utf-8",
    )

    ctx = _make_mock_context({"baseline_path": str(baseline_path)})

    result = datasets_diff_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_true(data.has_differences)
        expect_equal(data.added, ["new_dataset"])
        expect_equal(data.removed, ["old_dataset"])


@patch("codeintel.cli.handlers.datasets.get_dataset_contracts_by_table_key")
def test_datasets_diff_handler_no_differences(
    mock_get_contracts: MagicMock,
    tmp_path: Path,
) -> None:
    """Verify datasets_diff_handler reports no differences when same."""
    # Current datasets
    mock_contract = MagicMock()
    mock_contract.name = "same_dataset"
    mock_get_contracts.return_value = {"same.table": mock_contract}

    # Baseline with same dataset
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps([{"name": "same_dataset", "table_key": "same.table"}]),
        encoding="utf-8",
    )

    ctx = _make_mock_context({"baseline_path": str(baseline_path)})

    result = datasets_diff_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_true(not data.has_differences)
        expect_equal(data.added, [])
        expect_equal(data.removed, [])
