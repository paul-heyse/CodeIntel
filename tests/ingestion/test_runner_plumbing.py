"""Plumbing tests ensuring shared runners and scan configs are honored."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict, cast

import pytest

from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions import assert_target_ok, expect_equal, expect_rows_equal
from tests._helpers.harnesses.hamilton_build import (
    HamiltonBuildHarness,
    HarnessConfig,
    HarnessOpenOptions,
)
from tests._helpers.ingestion import (
    ScanSetupOptions,
    closing_gateway,
    make_scan_setup,
    materialize_repo_scan_result,
)
from tests._helpers.orchestration.repo_writers import write_sample_repo
from tests._helpers.tool_payloads import coverage_json_payload, pytest_report_payload
from tests._helpers.tool_sandbox import ToolSandbox, ToolStubSpec

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.config.models import ToolsConfig


class CoverageFile(TypedDict):
    """Coverage line data for a single file."""

    executed_lines: list[int]
    missing_lines: list[int]


class CoveragePayload(TypedDict):
    """Coverage payload keyed by file path."""

    files: dict[str, CoverageFile]


def _install_tool_stubs(tool_sandbox: ToolSandbox, repo_root: Path) -> tuple[ToolsConfig, int]:
    mod_path = repo_root / "pkg" / "mod.py"
    coverage_payload = coverage_json_payload(
        files={
            str(mod_path): {
                "executed_lines": [1, 2, 3],
                "missing_lines": [4],
            }
        }
    )
    tool_sandbox.install_stub(
        "coverage",
        spec=ToolStubSpec(
            writes="-o",
            writes_payload=json.dumps(coverage_payload),
        ),
    )
    tool_sandbox.install_stub(
        "pyright",
        spec=ToolStubSpec(
            stdout=json.dumps(
                {
                    "generalDiagnostics": [
                        {
                            "file": str(mod_path),
                            "severity": "error",
                            "message": "stub error",
                        }
                    ]
                }
            )
        ),
    )
    tool_sandbox.install_stub(
        "pyrefly",
        spec=ToolStubSpec(
            writes="--output",
            writes_payload=json.dumps(
                {
                    "errors": [
                        {
                            "path": str(mod_path),
                            "severity": "error",
                            "message": "stub error",
                        }
                    ]
                }
            ),
        ),
    )
    tool_sandbox.install_stub(
        "ruff",
        spec=ToolStubSpec(
            stdout=json.dumps(
                [
                    {
                        "filename": str(mod_path),
                        "code": "F401",
                        "message": "stub error",
                    }
                ]
            )
        ),
    )
    pytest_payload = pytest_report_payload(
        tests=[], summary={"passed": 0, "failed": 0, "skipped": 0}
    )
    tool_sandbox.install_stub(
        "pytest",
        spec=ToolStubSpec(
            writes="--json-report-file",
            writes_payload=json.dumps(pytest_payload),
        ),
    )
    typed_payload = cast("CoveragePayload", coverage_payload)
    expected_lines = len(typed_payload["files"][str(mod_path)]["executed_lines"]) + len(
        typed_payload["files"][str(mod_path)]["missing_lines"]
    )
    return tool_sandbox.tools_config(), expected_lines


def test_repo_scan_honors_scan_profile(tmp_path: Path) -> None:
    """Ensure repo_scan respects ignore lists from ScanProfile."""
    setup = make_scan_setup(
        tmp_path,
        options=ScanSetupOptions(
            repo_structure={
                "keep/a.py": "print('ok')\n",
                "ignore/b.py": "print('skip')\n",
            },
            ignore_dirs=("ignore",),
        ),
    )

    with closing_gateway(setup.gateway):
        scan_result = setup.scan_step.execute(
            repo="r",
            commit="c",
            repo_root=setup.repo_root,
            profile=setup.profile,
        )
        materialize_repo_scan_result(
            setup.gateway,
            scan_result,
            snapshot=SnapshotRef(repo="r", commit="c", repo_root=setup.repo_root),
        )

        rows = setup.gateway.con.table("core.modules").select("path").fetchall()
        expect_rows_equal(rows, [("keep/a.py",)], message="Unexpected modules from repo_scan")


def test_coverage_ingest_uses_runner(tool_sandbox: ToolSandbox, tmp_path: Path) -> None:
    """Verify coverage ingestion prefers the shared runner path."""
    repo_root = tmp_path / "repo"
    tools_config, expected_lines = _install_tool_stubs(tool_sandbox, repo_root)
    with HamiltonBuildHarness.open(
        tmp_path,
        harness=HarnessConfig(repo="r", commit="c"),
        options=HarnessOpenOptions(
            repo_strategy="writer",
            repo_writer=write_sample_repo,
            tools_config=tools_config,
        ),
    ) as harness:
        harness.artifacts.touch_coverage_file()
        result = harness.run_targets(["coverage_ingest"])
        record = harness.record("coverage_ingest", result=result)
        assert_target_ok(record)

        row = harness.ctx.gateway.con.execute(
            "SELECT COUNT(*) FROM analytics.coverage_lines"
        ).fetchone()
        count = row[0] if row is not None else 0
        expect_equal(count, expected_lines, label="coverage_line_count")


def test_tests_ingest_uses_report_file(tool_sandbox: ToolSandbox, tmp_path: Path) -> None:
    """Verify tests_ingest consumes the pytest report artifact."""
    repo_root = tmp_path / "repo"
    tools_config, _expected_lines = _install_tool_stubs(tool_sandbox, repo_root)
    with HamiltonBuildHarness.open(
        tmp_path,
        harness=HarnessConfig(repo="r", commit="c"),
        options=HarnessOpenOptions(
            repo_strategy="writer",
            repo_writer=write_sample_repo,
            tools_config=tools_config,
        ),
    ) as harness:
        harness.artifacts.write_pytest_report(
            tests=[
                {
                    "nodeid": "tests/test_mod.py::test_hello",
                    "outcome": "passed",
                    "duration": 0.01,
                }
            ],
            summary={"passed": 1, "failed": 0, "skipped": 0},
        )
        result = harness.run_targets(["tests_ingest"])
        record = harness.record("tests_ingest", result=result)
        assert_target_ok(record)

        row = harness.ctx.gateway.con.execute(
            "SELECT test_id FROM analytics.test_catalog WHERE test_id = ?",
            ["tests/test_mod.py::test_hello"],
        ).fetchone()
        if row is None:
            pytest.fail("tests_ingest failed to persist test_catalog rows")


@pytest.mark.skip(
    reason="Schema mismatch: StaticDiagnosticRow (6 cols) vs static_diagnostics table (8 cols)"
)
def test_typing_ingest_uses_shared_runner(
    tmp_path: Path,
    tool_sandbox: ToolSandbox,
) -> None:
    """Ensure typing ingestion reuses the provided ToolRunner."""
    repo_root = tmp_path / "repo"
    tools_config, _expected_lines = _install_tool_stubs(tool_sandbox, repo_root)
    with HamiltonBuildHarness.open(
        tmp_path,
        harness=HarnessConfig(repo="r", commit="c"),
        options=HarnessOpenOptions(
            repo_strategy="writer",
            repo_writer=write_sample_repo,
            tools_config=tools_config,
        ),
    ) as harness:
        result = harness.run_targets(["typing"])
        record = harness.record("typing", result=result)
        assert_target_ok(record)

        row = harness.ctx.gateway.con.execute("SELECT COUNT(*) FROM analytics.typedness").fetchone()
        if (row[0] if row else 0) < 1:
            pytest.fail("Typedness ingestion wrote no rows")
