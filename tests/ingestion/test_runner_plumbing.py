"""Plumbing tests ensuring shared runners and scan configs are honored."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

from codeintel.config.models import CoverageIngestConfig, RepoScanConfig, TypingIngestConfig
from codeintel.ingestion.coverage_ingest import ingest_coverage_lines
from codeintel.ingestion.repo_scan import ingest_repo
from codeintel.ingestion.source_scanner import ScanConfig
from codeintel.ingestion.tool_runner import ToolName, ToolResult, ToolRunner
from codeintel.ingestion.typing_ingest import ingest_typing_signals
from codeintel.storage.schemas import apply_all_schemas


class _FakeRunner(ToolRunner):
    def __init__(self, cache_dir: Path, payloads: dict[str, Any]) -> None:
        super().__init__(cache_dir=cache_dir)
        self.payloads = payloads
        self.calls: list[tuple[ToolName, list[str]]] = []
        self.last_cwd: Path | None = None

    def run(
        self,
        tool: ToolName,
        args: list[str],
        *,
        cwd: Path | None = None,
        output_path: Path | None = None,
    ) -> ToolResult:
        self.calls.append((tool, args))
        self.last_cwd = cwd
        if tool == "pyrefly" and output_path is not None:
            output_path.write_text('{"errors": []}', encoding="utf8")
        stdout = self.payloads.get(tool, "")
        if tool == "coverage" and output_path is not None:
            json_payload = self.payloads.get("json", {})
            output_path.write_text(json.dumps(json_payload), encoding="utf8")
        return ToolResult(
            tool=tool,
            returncode=0,
            stdout=stdout,
            stderr="",
            output_path=output_path,
            duration_s=0.0,
        )

    def load_json(self, path: Path) -> dict[str, object] | None:
        if path.is_file():
            return self.payloads.get("json", {})
        return self.payloads.get("json", {})


def _setup_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    apply_all_schemas(con)
    return con


def test_repo_scan_honors_scan_config(tmp_path: Path) -> None:
    """Ensure repo_scan respects ignore lists from ScanConfig."""
    repo_root = tmp_path / "repo"
    keep_dir = repo_root / "keep"
    ignore_dir = repo_root / "ignore"
    keep_dir.mkdir(parents=True, exist_ok=True)
    ignore_dir.mkdir(parents=True, exist_ok=True)
    (keep_dir / "a.py").write_text("print('ok')\n", encoding="utf8")
    (ignore_dir / "b.py").write_text("print('skip')\n", encoding="utf8")

    con = _setup_db()
    cfg = RepoScanConfig.from_paths(repo_root=repo_root, repo="r", commit="c")
    scan_cfg = ScanConfig(repo_root=repo_root, ignore_dirs=("ignore",), include_patterns=("*.py",))
    ingest_repo(con=con, cfg=cfg, scan_config=scan_cfg)

    rows = con.execute("SELECT path FROM core.modules").fetchall()
    if rows != [("keep/a.py",)]:
        pytest.fail(f"Unexpected modules: {rows}")


def test_coverage_ingest_uses_runner(tmp_path: Path) -> None:
    """Verify coverage ingestion prefers the shared runner path."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    file_path = repo_root / "a.py"
    file_path.write_text("print('hi')\n", encoding="utf8")
    coverage_file = repo_root / ".coverage"
    coverage_file.write_text("", encoding="utf8")

    payload = {
        "json": {
            "files": {
                str(file_path.resolve()): {
                    "executed_lines": [1],
                    "missing_lines": [],
                }
            }
        }
    }
    runner = _FakeRunner(cache_dir=tmp_path / "cache", payloads=payload)
    con = _setup_db()
    cfg = CoverageIngestConfig.from_paths(
        repo_root=repo_root, repo="r", commit="c", coverage_file=coverage_file
    )
    ingest_coverage_lines(con=con, cfg=cfg, runner=runner)

    if not any(call[0] == "coverage" for call in runner.calls):
        pytest.fail("Expected coverage tool to be invoked via runner")
    row = con.execute("SELECT COUNT(*) FROM analytics.coverage_lines").fetchone()
    count = row[0] if row is not None else 0
    if count != 1:
        pytest.fail(f"Expected 1 coverage row, got {count}")


def test_typing_ingest_uses_shared_runner(tmp_path: Path) -> None:
    """Ensure typing ingestion reuses the provided ToolRunner."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    file_path = repo_root / "m.py"
    file_path.write_text("def f(x: int) -> int:\n    return x\n", encoding="utf8")

    pyright_stdout = '{"generalDiagnostics": []}'
    payloads = {"pyright": pyright_stdout, "json": {"errors": []}, "ruff": "[]"}
    runner = _FakeRunner(cache_dir=tmp_path / "cache", payloads=payloads)

    con = _setup_db()
    cfg = TypingIngestConfig.from_paths(repo_root=repo_root, repo="r", commit="c")
    ingest_typing_signals(con=con, cfg=cfg, runner=runner)

    tools_called = {tool for tool, _ in runner.calls}
    if not {"pyrefly", "pyright", "ruff"} <= tools_called:
        pytest.fail(f"Expected pyrefly/pyright/ruff calls, saw {tools_called}")
    row = con.execute("SELECT COUNT(*) FROM analytics.typedness").fetchone()
    typedness_rows = row[0] if row is not None else 0
    if typedness_rows < 1:
        pytest.fail("Typedness ingestion wrote no rows")
