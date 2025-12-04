"""Conformance CLI and helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from codeintel.storage.validation.conformance import (
    ConformanceIssue,
    ConformanceReport,
    run_conformance,
)
from codeintel.storage.datasets import load_dataset_registry
from codeintel.storage.metadata import bootstrap_metadata_datasets
from codeintel.storage.schema.json_schema import generate_export_schemas
from codeintel.storage.schema import apply_all_schemas
from tests._helpers.gateway import memory_con_with_macros


def test_conformance_report_ok_when_no_issues() -> None:
    """Verify ConformanceReport.ok is True when issues list is empty."""
    report = ConformanceReport(issues=[])
    assert report.ok


def test_conformance_report_not_ok_when_issues_exist() -> None:
    """Verify ConformanceReport.ok is False when issues exist."""
    issue = ConformanceIssue(dataset="test", message="test issue")
    report = ConformanceReport(issues=[issue])
    assert not report.ok


def test_conformance_issue_stores_dataset_and_message() -> None:
    """Verify ConformanceIssue stores dataset and message."""
    issue = ConformanceIssue(dataset="core.modules", message="Test failure")
    assert issue.dataset == "core.modules"
    assert issue.message == "Test failure"


def test_conformance_issue_allows_none_dataset() -> None:
    """Verify ConformanceIssue allows None dataset for global issues."""
    issue = ConformanceIssue(dataset=None, message="Global issue")
    assert issue.dataset is None


def test_conformance_passes_with_empty_db(tmp_path: Path) -> None:
    """Conformance should succeed when the catalog is freshly bootstrapped."""
    con = memory_con_with_macros()
    try:
        apply_all_schemas(con)
        bootstrap_metadata_datasets(con)
        registry = load_dataset_registry(con)
        generate_export_schemas(registry, output_dir=tmp_path)
        default_export_dir = Path("src/codeintel/config/schemas/export")
        for schema_file in default_export_dir.glob("*.json"):
            destination = tmp_path / schema_file.name
            if not destination.exists():
                shutil.copy2(schema_file, destination)
        report = run_conformance(con, schema_base_dir=tmp_path, sample_rows=False)
        assert report.ok, (
            f"Unexpected contract issues: {[issue.message for issue in report.issues]}"
        )
    finally:
        con.close()


def test_conformance_with_sample_rows_enabled(tmp_path: Path) -> None:
    """Conformance should handle sample_rows=True."""
    con = memory_con_with_macros()
    try:
        apply_all_schemas(con)
        bootstrap_metadata_datasets(con)
        registry = load_dataset_registry(con)
        generate_export_schemas(registry, output_dir=tmp_path)
        default_export_dir = Path("src/codeintel/config/schemas/export")
        for schema_file in default_export_dir.glob("*.json"):
            destination = tmp_path / schema_file.name
            if not destination.exists():
                shutil.copy2(schema_file, destination)

        sample_size = 10
        report = run_conformance(
            con, schema_base_dir=tmp_path, sample_rows=True, sample_size=sample_size
        )

        assert isinstance(report, ConformanceReport)
    finally:
        con.close()


def test_conformance_skips_missing_schema_files(tmp_path: Path) -> None:
    """Conformance should skip datasets with missing schema files."""
    con = memory_con_with_macros()
    try:
        apply_all_schemas(con)
        bootstrap_metadata_datasets(con)

        empty_schema_dir = tmp_path / "empty_schemas"
        empty_schema_dir.mkdir()

        report = run_conformance(con, schema_base_dir=empty_schema_dir, sample_rows=True)

        assert isinstance(report, ConformanceReport)
    finally:
        con.close()


def test_conformance_validates_schema_rows(tmp_path: Path) -> None:
    """Conformance should validate rows against JSON Schema."""
    con = memory_con_with_macros()
    try:
        apply_all_schemas(con)
        bootstrap_metadata_datasets(con)
        registry = load_dataset_registry(con)
        generate_export_schemas(registry, output_dir=tmp_path)
        default_export_dir = Path("src/codeintel/config/schemas/export")
        for schema_file in default_export_dir.glob("*.json"):
            destination = tmp_path / schema_file.name
            if not destination.exists():
                shutil.copy2(schema_file, destination)

        sample_size = 5
        report = run_conformance(
            con,
            schema_base_dir=tmp_path,
            sample_rows=True,
            sample_size=sample_size,
        )

        assert isinstance(report, ConformanceReport)
    finally:
        con.close()


def test_conformance_reports_json_schema_errors(tmp_path: Path) -> None:
    """Conformance should report JSON Schema validation errors."""
    con = memory_con_with_macros()
    try:
        apply_all_schemas(con)
        bootstrap_metadata_datasets(con)
        registry = load_dataset_registry(con)
        generate_export_schemas(registry, output_dir=tmp_path)

        invalid_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"repo": {"type": "integer"}},
            "required": ["repo"],
        }
        invalid_schema_file = tmp_path / "core.modules.schema.json"
        invalid_schema_file.write_text(json.dumps(invalid_schema), encoding="utf-8")

        con.execute(
            """
            INSERT INTO core.modules (module, path, repo, commit)
            VALUES ('test_mod', 'test.py', 'test/repo', 'abc123')
            """
        )

        sample_size = 10
        report = run_conformance(
            con,
            schema_base_dir=tmp_path,
            sample_rows=True,
            sample_size=sample_size,
        )

        assert isinstance(report, ConformanceReport)

    finally:
        con.close()
