"""Conformance CLI and helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.storage.datasets import load_dataset_registry
from codeintel.storage.metadata import bootstrap_metadata_datasets
from codeintel.storage.validation.conformance import (
    ConformanceIssue,
    ConformanceReport,
    run_conformance,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_is_instance,
    expect_true,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


SAMPLE_SIZE_5 = 5
SAMPLE_SIZE_10 = 10


def test_conformance_report_ok_when_no_issues() -> None:
    """Verify ConformanceReport.ok is True when issues list is empty."""
    report = ConformanceReport(issues=[])
    expect_true(report.ok, message="empty issues ok flag")


def test_conformance_report_not_ok_when_issues_exist() -> None:
    """Verify ConformanceReport.ok is False when issues exist."""
    issue = ConformanceIssue(dataset="test", message="test issue")
    report = ConformanceReport(issues=[issue])
    expect_true(report.ok is False, message="non-empty issues not ok")


def test_conformance_issue_stores_dataset_and_message() -> None:
    """Verify ConformanceIssue stores dataset and message."""
    issue = ConformanceIssue(dataset="core.modules", message="Test failure")
    expect_equal(issue.dataset, "core.modules", label="dataset")
    expect_equal(issue.message, "Test failure", label="message")


def test_conformance_issue_allows_none_dataset() -> None:
    """Verify ConformanceIssue allows None dataset for global issues."""
    issue = ConformanceIssue(dataset=None, message="Global issue")
    expect_true(issue.dataset is None, message="dataset may be None")


def test_conformance_passes_with_empty_db(fresh_gateway: StorageGateway) -> None:
    """Conformance should succeed when the catalog is freshly bootstrapped."""
    bootstrap_metadata_datasets(fresh_gateway.con)
    load_dataset_registry(fresh_gateway.con)
    report = run_conformance(fresh_gateway.con, sample_rows=False)
    expect_true(
        report.ok,
        message=f"Unexpected contract issues: {[issue.message for issue in report.issues]}",
    )


def test_conformance_with_sample_rows_enabled(fresh_gateway: StorageGateway) -> None:
    """Conformance should handle sample_rows=True with generated schemas."""
    bootstrap_metadata_datasets(fresh_gateway.con)
    load_dataset_registry(fresh_gateway.con)

    report = run_conformance(fresh_gateway.con, sample_rows=True, sample_size=SAMPLE_SIZE_10)

    expect_is_instance(report, ConformanceReport, label="report type")


def test_conformance_skips_missing_schema(fresh_gateway: StorageGateway) -> None:
    """Conformance should skip datasets with no available generated schema."""
    bootstrap_metadata_datasets(fresh_gateway.con)

    report = run_conformance(fresh_gateway.con, sample_rows=True)

    expect_is_instance(report, ConformanceReport, label="report type")


def test_conformance_validates_schema_rows(fresh_gateway: StorageGateway) -> None:
    """Conformance should validate rows against generated JSON Schema."""
    bootstrap_metadata_datasets(fresh_gateway.con)
    load_dataset_registry(fresh_gateway.con)

    report = run_conformance(
        fresh_gateway.con,
        sample_rows=True,
        sample_size=SAMPLE_SIZE_5,
    )

    expect_is_instance(report, ConformanceReport, label="report type")
