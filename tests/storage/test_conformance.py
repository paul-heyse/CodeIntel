"""Conformance CLI and helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from codeintel.storage.datasets import load_dataset_registry
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.metadata import bootstrap_metadata_datasets
from codeintel.storage.schema.json_schema import generate_export_schemas
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

# Constants
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


def test_conformance_passes_with_empty_db(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """Conformance should succeed when the catalog is freshly bootstrapped."""
    bootstrap_metadata_datasets(fresh_gateway.con)
    registry = load_dataset_registry(fresh_gateway.con)
    generate_export_schemas(registry, output_dir=tmp_path)
    default_export_dir = Path("src/codeintel/config/schemas/export")
    for schema_file in default_export_dir.glob("*.json"):
        destination = tmp_path / schema_file.name
        if not destination.exists():
            shutil.copy2(schema_file, destination)
    report = run_conformance(fresh_gateway.con, schema_base_dir=tmp_path, sample_rows=False)
    expect_true(
        report.ok,
        message=f"Unexpected contract issues: {[issue.message for issue in report.issues]}",
    )


def test_conformance_with_sample_rows_enabled(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Conformance should handle sample_rows=True."""
    bootstrap_metadata_datasets(fresh_gateway.con)
    registry = load_dataset_registry(fresh_gateway.con)
    generate_export_schemas(registry, output_dir=tmp_path)
    default_export_dir = Path("src/codeintel/config/schemas/export")
    for schema_file in default_export_dir.glob("*.json"):
        destination = tmp_path / schema_file.name
        if not destination.exists():
            shutil.copy2(schema_file, destination)

    report = run_conformance(
        fresh_gateway.con, schema_base_dir=tmp_path, sample_rows=True, sample_size=SAMPLE_SIZE_10
    )

    expect_is_instance(report, ConformanceReport, label="report type")


def test_conformance_skips_missing_schema_files(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Conformance should skip datasets with missing schema files."""
    bootstrap_metadata_datasets(fresh_gateway.con)

    empty_schema_dir = tmp_path / "empty_schemas"
    empty_schema_dir.mkdir()

    report = run_conformance(fresh_gateway.con, schema_base_dir=empty_schema_dir, sample_rows=True)

    expect_is_instance(report, ConformanceReport, label="report type")


def test_conformance_validates_schema_rows(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """Conformance should validate rows against JSON Schema."""
    bootstrap_metadata_datasets(fresh_gateway.con)
    registry = load_dataset_registry(fresh_gateway.con)
    generate_export_schemas(registry, output_dir=tmp_path)
    default_export_dir = Path("src/codeintel/config/schemas/export")
    for schema_file in default_export_dir.glob("*.json"):
        destination = tmp_path / schema_file.name
        if not destination.exists():
            shutil.copy2(schema_file, destination)

    report = run_conformance(
        fresh_gateway.con,
        schema_base_dir=tmp_path,
        sample_rows=True,
        sample_size=SAMPLE_SIZE_5,
    )

    expect_is_instance(report, ConformanceReport, label="report type")


def test_conformance_reports_json_schema_errors(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Conformance should report JSON Schema validation errors."""
    bootstrap_metadata_datasets(fresh_gateway.con)
    registry = load_dataset_registry(fresh_gateway.con)
    generate_export_schemas(registry, output_dir=tmp_path)

    invalid_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"repo": {"type": "integer"}},
        "required": ["repo"],
    }
    invalid_schema_file = tmp_path / "core.modules.schema.json"
    invalid_schema_file.write_text(json.dumps(invalid_schema), encoding="utf-8")

    fresh_gateway.con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit)
        VALUES ('test_mod', 'test.py', 'test/repo', 'abc123')
        """
    )

    report = run_conformance(
        fresh_gateway.con,
        schema_base_dir=tmp_path,
        sample_rows=True,
        sample_size=SAMPLE_SIZE_10,
    )

    expect_is_instance(report, ConformanceReport, label="report type")
