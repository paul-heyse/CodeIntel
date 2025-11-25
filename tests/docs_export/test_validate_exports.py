"""Tests for validate_exports tooling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet
from codeintel.docs_export.validate_exports import main
from codeintel.services.errors import ExportError
from codeintel.storage.gateway import open_memory_gateway
from tests._helpers.fixtures import seed_docs_export_invalid_profile


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_validate_jsonl_happy_path(tmp_path: Path) -> None:
    """Validator should return 0 for conforming JSONL rows."""
    data_path = tmp_path / "edges.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "repo": "r",
                "commit": "c",
                "caller_goid_h128": 1,
                "callee_goid_h128": 2,
                "callsite_path": "a.py",
                "language": "python",
            }
        ],
    )
    exit_code = main(["--schema", "call_graph_edges", str(data_path)])
    if exit_code != 0:
        message = f"Expected success, got exit code {exit_code}"
        pytest.fail(message)


def test_validate_jsonl_failure(tmp_path: Path) -> None:
    """Validator should fail when required fields are missing."""
    data_path = tmp_path / "edges.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "commit": "c",
                "caller_goid_h128": 1,
                "callee_goid_h128": 2,
            }
        ],
    )
    exit_code = main(["--schema", "call_graph_edges", str(data_path)])
    if exit_code == 0:
        message = "Expected validation failure for missing repo field"
        pytest.fail(message)


def test_export_raises_on_validation_failure(tmp_path: Path) -> None:
    """Export functions should raise ExportError when schema validation fails."""
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    seed_docs_export_invalid_profile(
        gateway,
        repo="r",
        commit="c",
        null_commit=True,
    )

    output_dir = tmp_path / "out"
    with pytest.raises(ExportError):
        export_all_parquet(
            gateway,
            output_dir,
            validate_exports=True,
            schemas=["function_profile"],
        )


def test_export_logs_problem_detail_on_validation_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Validation failures should log ProblemDetails and raise ExportError."""
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    seed_docs_export_invalid_profile(
        gateway,
        repo="r",
        commit="c",
        null_commit=True,
        drop_commit_column=True,
    )

    output_dir = tmp_path / "out_jsonl"
    caplog.set_level("ERROR")
    with pytest.raises(ExportError):
        export_all_jsonl(
            gateway,
            output_dir,
            validate_exports=True,
            schemas=["function_profile"],
        )
    error_logs = [rec for rec in caplog.records if "export.validation_failed" in rec.getMessage()]
    if not error_logs:
        pytest.fail("Expected ProblemDetail log entry for validation failure")
