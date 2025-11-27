"""Tests for validate_exports tooling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeintel.pipeline.export.export_jsonl import ExportCallOptions, export_all_jsonl
from codeintel.pipeline.export.export_parquet import export_all_parquet
from codeintel.pipeline.export.validate_exports import main
from codeintel.serving.services.errors import ExportError
from tests._helpers.fixtures import (
    GatewayOptions,
    ProvisioningConfig,
    provisioned_gateway,
    seed_docs_export_invalid_profile,
)


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


def test_docs_export_views_exist(tmp_path: Path) -> None:
    """Docs export strict provisioning should expose canonical views."""
    with provisioned_gateway(
        tmp_path / "repo", config=ProvisioningConfig(run_ingestion=False)
    ) as ctx:
        ctx.gateway.con.execute("SELECT * FROM docs.v_symbol_module_graph LIMIT 0")


def test_export_raises_on_validation_failure(tmp_path: Path) -> None:
    """Export functions should raise ExportError when schema validation fails."""
    with provisioned_gateway(
        tmp_path / "repo",
        config=ProvisioningConfig(
            run_ingestion=False,
            gateway_options=GatewayOptions(strict_schema=False, validate_schema=False),
        ),
    ) as ctx:
        seed_docs_export_invalid_profile(
            ctx.gateway,
            repo=ctx.repo,
            commit=ctx.commit,
            null_commit=True,
        )
        output_dir = tmp_path / "out"
        with pytest.raises(ExportError):
            export_all_parquet(
                ctx.gateway,
                output_dir,
                options=ExportCallOptions(validate_exports=True, schemas=["function_profile"]),
            )


def test_export_logs_problem_detail_on_validation_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Validation failures should log ProblemDetails and raise ExportError."""
    with provisioned_gateway(
        tmp_path / "repo",
        config=ProvisioningConfig(
            run_ingestion=False,
            gateway_options=GatewayOptions(strict_schema=False, validate_schema=False),
        ),
    ) as ctx:
        seed_docs_export_invalid_profile(
            ctx.gateway,
            repo=ctx.repo,
            commit=ctx.commit,
            null_commit=True,
            drop_commit_column=True,
        )
        output_dir = tmp_path / "out_jsonl"
        caplog.set_level("ERROR")
        with pytest.raises(ExportError):
            export_all_jsonl(
                ctx.gateway,
                output_dir,
                options=ExportCallOptions(validate_exports=True, schemas=["function_profile"]),
            )
        error_logs = [
            rec for rec in caplog.records if "export.validation_failed" in rec.getMessage()
        ]
        if not error_logs:
            pytest.fail("Expected ProblemDetail log entry for validation failure")
