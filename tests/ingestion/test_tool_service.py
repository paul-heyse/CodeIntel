"""Tests for ToolService parsing and orchestration."""

from __future__ import annotations

import asyncio
import json
import pytest
from pathlib import Path

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_service import ToolService
from tests._helpers.fakes import FakeToolRunner


def test_tool_service_pyright_parses_errors(tmp_path: Path) -> None:
    """ToolService aggregates pyright diagnostics per file."""
    repo_root = tmp_path
    target = repo_root / "pkg" / "mod.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def foo(x: int) -> int:\n    return x\n", encoding="utf8")
    payload = json.dumps(
        {
            "generalDiagnostics": [
                {"severity": "error", "file": str(target)},
            ]
        }
    )
    runner = FakeToolRunner(tmp_path, payloads={"pyright": payload})
    service = ToolService(runner, ToolsConfig())

    errors = asyncio.run(service.run_pyright(repo_root))

    if errors != {"pkg/mod.py": 1}:
        pytest.fail(f"Unexpected pyright errors: {errors}")


def test_tool_service_pyrefly_parses_errors(tmp_path: Path) -> None:
    """ToolService aggregates pyrefly diagnostics per file."""
    repo_root = tmp_path
    target = repo_root / "pkg" / "mod.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def foo(x):\n    return x\n", encoding="utf8")
    payload = {
        "errors": [
            {"severity": "error", "path": str(target)},
            {"severity": "warning", "path": str(target)},
        ]
    }
    runner = FakeToolRunner(tmp_path, payloads={"pyrefly_json": payload})
    service = ToolService(runner, ToolsConfig())

    errors = asyncio.run(service.run_pyrefly(repo_root))

    if errors != {"pkg/mod.py": 1}:
        pytest.fail(f"Unexpected pyrefly errors: {errors}")


def test_tool_service_coverage_reports(tmp_path: Path) -> None:
    """ToolService normalizes coverage.json payloads."""
    repo_root = tmp_path
    coverage_file = repo_root / ".coverage"
    coverage_file.write_text("", encoding="utf8")
    target = repo_root / "pkg" / "mod.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('hi')\n", encoding="utf8")
    coverage_payload = {
        "files": {
            str(target): {
                "executed_lines": [1],
                "missing_lines": [2],
            }
        }
    }
    runner = FakeToolRunner(tmp_path, payloads={"coverage_json": coverage_payload})
    service = ToolService(runner, ToolsConfig())

    reports = asyncio.run(
        service.run_coverage_json(
            repo_root,
            coverage_file=coverage_file,
            output_path=tmp_path / "coverage.json",
        )
    )

    if len(reports) != 1:
        pytest.fail(f"Expected single coverage report, got {len(reports)}")
    report = reports[0]
    if report.rel_path != "pkg/mod.py":
        pytest.fail(f"Unexpected rel_path {report.rel_path}")
    if report.executed_lines != {1} or report.missing_lines != {2}:
        pytest.fail(f"Unexpected coverage sets: {report.executed_lines}, {report.missing_lines}")
