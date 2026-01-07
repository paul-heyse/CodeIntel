"""Tests for datasets handlers."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pyarrow as pa

from codeintel.cli.core.result_types import TabularResult
from codeintel.cli.core.results import SerializableResult
from codeintel.cli.handlers.datasets import (
    DatasetDependencies,
    DatasetDiffResult,
    DatasetLintResult,
    DatasetSnapshotResult,
    datasets_diff_handler,
    datasets_lint_handler,
    datasets_list_handler,
    datasets_migrate_parquet_handler,
    datasets_snapshot_handler,
)
from codeintel.core.columnar.stream import stream_from_table
from codeintel.core.datasets.manifests import dataset_manifest_path
from codeintel.core.query_results import records_from_arrow_reader
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.cli.handlers.conftest import DatasetHandlerHarness


def _result_to_dict(result: object) -> dict[str, object]:
    return cast("SerializableResult", result).to_dict()


def test_datasets_list_result_to_dict() -> None:
    """Verify TabularResult.to_dict returns correct structure."""
    items = [
        {
            "name": "test_dataset",
            "table_key": "test.table",
            "description": "A test dataset",
            "capabilities": {"docs_view": False},
        }
    ]
    table = pa.Table.from_pylist(items)
    result = TabularResult(stream=stream_from_table(table))

    data = _result_to_dict(result)

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

    data = _result_to_dict(result)

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

    data = _result_to_dict(result)

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

    data = _result_to_dict(result)

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

    data = _result_to_dict(result)

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

    data = _result_to_dict(result)

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
        expect_true(isinstance(data, TabularResult))
        if isinstance(data, TabularResult):
            reader = data.stream.to_reader(batch_size=DEFAULT_ARROW_BATCH_SIZE)
            records = records_from_arrow_reader(reader)
            expect_equal(len(records), 1)
            expect_equal(records[0]["name"], "test_dataset")
            expect_equal(data.metadata.get("count"), 1)


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
        expect_equal(error.type, "urn:codeintel:validation/missing-required")


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
        expect_equal(error.type, "urn:codeintel:validation/missing-required")


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
    baseline_path = tmp_path / "baseline.json"

    with dataset_handler_harness_fixture.command_context(
        {"baseline_path": str(baseline_path)}
    ) as ctx:
        contract = deps.list_datasets(docs_view="include", read_only="include")[0]
        baseline_path.write_text(
            json.dumps([{"name": contract.name, "table_key": contract.table_key}]),
            encoding="utf-8",
        )
        result = datasets_diff_handler(ctx, deps=deps)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_true(not data.has_differences)
        expect_equal(data.added, [])
        expect_equal(data.removed, [])


def _test_contract(table_name: str) -> DatasetContract:
    table_schema = TableSchema(
        schema="test",
        name=table_name,
        columns=[
            Column(name="repo", type="VARCHAR", nullable=False),
            Column(name="commit", type="VARCHAR", nullable=False),
            Column(name="value", type="BIGINT", nullable=False),
        ],
        primary_key=(),
    )
    return DatasetContract(
        table_key=table_schema.table_key,
        name=table_name,
        schema=table_schema,
        owner_package="core",
    )


def _seed_migration_table(
    gateway: StorageGateway,
    *,
    table_schema: TableSchema,
) -> None:
    gateway.policy.create_schema_if_not_exists(table_schema.schema)
    gateway.con.execute(f"DROP TABLE IF EXISTS {table_schema.schema}.{table_schema.name}")
    gateway.con.execute(
        f"""
        CREATE TABLE {table_schema.schema}.{table_schema.name} (
            repo VARCHAR,
            commit VARCHAR,
            value BIGINT
        )
        """
    )
    gateway.con.execute(
        f"INSERT INTO {table_schema.schema}.{table_schema.name} VALUES (?, ?, ?)",
        ["repo-1", "commit-1", 1],
    )


def test_datasets_migrate_parquet_handler_writes_manifest(
    tmp_path: Path, dataset_handler_harness_fixture: DatasetHandlerHarness
) -> None:
    """Verify parquet migration writes a dataset manifest."""
    contract = _test_contract("migrate_metrics")
    table_schema = cast("TableSchema", contract.schema)
    _seed_migration_table(
        dataset_handler_harness_fixture.ctx.gateway,
        table_schema=table_schema,
    )
    dataset_root_dir = tmp_path / "datasets"
    params = {
        "dataset_root_dir": str(dataset_root_dir),
        "snapshot_id": "snap-1",
        "table_keys": [contract.table_key],
        "overwrite": True,
    }
    deps = DatasetDependencies(
        list_datasets=lambda **_kwargs: [contract],
        issue_collector=lambda _con: [],
    )

    with dataset_handler_harness_fixture.command_context(params) as ctx:
        result = datasets_migrate_parquet_handler(ctx, deps=deps)

    expect_true(result.success)
    expect_is_not_none(result.data)
    manifest_path = dataset_manifest_path(
        dataset_root=dataset_root_dir,
        table_key=contract.table_key,
        snapshot_id="snap-1",
    )
    expect_true(manifest_path.is_file())
    if result.data is not None and result.data.details is not None:
        expect_equal(result.data.details.get("exported"), [contract.table_key])
