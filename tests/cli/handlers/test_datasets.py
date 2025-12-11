"""Tests for datasets handlers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from codeintel.cli.handlers.datasets import (
    DatasetDiffResult,
    DatasetLintResult,
    DatasetListResult,
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
from tests.cli.handlers.conftest import DatasetHandlerHarness


def test_datasets_list_result_to_dict() -> None:
    """Verify DatasetsListResult.to_dict returns correct structure."""
    result = DatasetListResult(
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


def test_datasets_list_handler_success(
    dataset_handler_harness_fixture: DatasetHandlerHarness,
) -> None:
    """Verify datasets_list_handler returns datasets successfully."""
    deps = dataset_handler_harness_fixture.deps

    with dataset_handler_harness_fixture.command_context({}) as ctx:
        result = datasets_list_handler(ctx, deps=deps)

    expect_true(result.success)
    expect_is_not_none(result.data)
    data = result.data
    if data is not None:
        expect_equal(data.count, 1)
        expect_equal(data.datasets[0]["name"], "test_dataset")


def test_datasets_lint_handler_success(
    dataset_handler_harness_fixture: DatasetHandlerHarness,
) -> None:
    """Verify datasets_lint_handler returns success when no issues."""
    deps = dataset_handler_harness_fixture.deps

    with dataset_handler_harness_fixture.command_context({}) as ctx:
        result = datasets_lint_handler(ctx, deps=deps)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_true(data.passed)
        expect_equal(data.issue_count, 0)


def test_datasets_lint_handler_with_issues(
    dataset_handler_harness_fixture: DatasetHandlerHarness,
) -> None:
    """Verify datasets_lint_handler returns issues when found."""
    deps = replace(
        dataset_handler_harness_fixture.deps,
        issue_collector=lambda _con: ["Issue 1", "Issue 2"],
    )

    with dataset_handler_harness_fixture.command_context({}) as ctx:
        result = datasets_lint_handler(ctx, deps=deps)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_true(not data.passed)
        expect_equal(data.issue_count, 2)


def test_datasets_snapshot_handler_missing_output(
    dataset_handler_harness_fixture: DatasetHandlerHarness,
) -> None:
    """Verify datasets_snapshot_handler fails without output parameter."""
    with dataset_handler_harness_fixture.command_context({}) as ctx:
        result = datasets_snapshot_handler(ctx)

    expect_true(not result.success)
    error = result.error
    if error is not None:
        expect_equal(error.type, "urn:codeintel:cli:validation:missing-required")


def test_datasets_snapshot_handler_success(
    tmp_path: Path, dataset_handler_harness_fixture: DatasetHandlerHarness
) -> None:
    """Verify datasets_snapshot_handler writes snapshot file."""
    deps = dataset_handler_harness_fixture.deps

    output_path = tmp_path / "snapshot.json"
    with dataset_handler_harness_fixture.command_context({"output": str(output_path)}) as ctx:
        result = datasets_snapshot_handler(ctx, deps=deps)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.datasets_count, 1)
    expect_true(output_path.exists())

    # Verify file content
    content = json.loads(output_path.read_text(encoding="utf-8"))
    expect_equal(len(content), 1)
    expect_equal(content[0]["name"], "test_dataset")


def test_datasets_diff_handler_missing_baseline(
    dataset_handler_harness_fixture: DatasetHandlerHarness,
) -> None:
    """Verify datasets_diff_handler fails without baseline_path parameter."""
    deps = dataset_handler_harness_fixture.deps

    with dataset_handler_harness_fixture.command_context({}) as ctx:
        result = datasets_diff_handler(ctx, deps=deps)

    expect_true(not result.success)
    error = result.error
    if error is not None:
        expect_equal(error.type, "urn:codeintel:cli:validation:missing-required")


def test_datasets_diff_handler_baseline_not_found(
    tmp_path: Path, dataset_handler_harness_fixture: DatasetHandlerHarness
) -> None:
    """Verify datasets_diff_handler fails when baseline file not found."""
    deps = dataset_handler_harness_fixture.deps

    with dataset_handler_harness_fixture.command_context(
        {"baseline_path": str(tmp_path / "nonexistent.json")}
    ) as ctx:
        result = datasets_diff_handler(ctx, deps=deps)

    expect_true(not result.success)
    error = result.error
    if error is not None:
        expect_equal(error.type, "urn:codeintel:datasets:file-not-found")


def test_datasets_diff_handler_success(
    tmp_path: Path, dataset_handler_harness_fixture: DatasetHandlerHarness
) -> None:
    """Verify datasets_diff_handler computes differences correctly."""
    deps = dataset_handler_harness_fixture.deps

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps([{"name": "missing", "table_key": "missing.table"}]),
        encoding="utf-8",
    )

    with dataset_handler_harness_fixture.command_context(
        {"baseline_path": str(baseline_path)}
    ) as ctx:
        result = datasets_diff_handler(ctx, deps=deps)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_true(data.has_differences)
        expect_true(any(name for name in data.added))
        expect_equal(data.removed, ["missing"])


def test_datasets_diff_handler_no_differences(
    tmp_path: Path,
    dataset_handler_harness_fixture: DatasetHandlerHarness,
) -> None:
    """Verify datasets_diff_handler reports no differences when same."""
    deps = dataset_handler_harness_fixture.deps

    contract = deps.contracts_provider()["test_dataset"]
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps([{"name": contract.name, "table_key": contract.table_key}]),
        encoding="utf-8",
    )

    with dataset_handler_harness_fixture.command_context(
        {"baseline_path": str(baseline_path)}
    ) as ctx:
        result = datasets_diff_handler(ctx, deps=deps)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_true(not data.has_differences)
        expect_equal(data.added, [])
        expect_equal(data.removed, [])
