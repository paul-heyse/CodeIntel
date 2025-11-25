"""High-level façade around ToolRunner for external CLI integrations."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
import asyncio
from dataclasses import dataclass
import asyncio
from pathlib import Path
from typing import Any

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
)
from codeintel.utils.paths import normalize_rel_path, repo_relpath

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoverageFileReport:
    """Normalized coverage summary for a single file."""

    rel_path: str
    executed_lines: set[int]
    missing_lines: set[int]


class ToolService:
    """Orchestrate external tooling and parse outputs for ingestion modules."""

    def __init__(self, runner: ToolRunner, tools_config: ToolsConfig | None = None) -> None:
        self.runner = runner
        self.tools_config = tools_config or runner.tools_config

    async def run_pyright(self, repo_root: Path) -> Mapping[str, int]:
        """Run pyright and return error counts keyed by repo-relative path."""
        try:
            result = await self.runner.run_async(
                ToolName.PYRIGHT,
                ["--outputjson", str(repo_root)],
                cwd=repo_root,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError:
            log.warning("pyright binary not found; treating all files as 0 errors")
            return {}

        if result.returncode not in (0, 1):
            raise ToolExecutionError(result)
        return _parse_pyright_errors(result.stdout, repo_root)

    async def run_pyrefly(self, repo_root: Path) -> Mapping[str, int]:
        """Run pyrefly and return error counts keyed by repo-relative path."""
        output_path = self.runner.cache_dir / "pyrefly.json"
        args = [
            "check",
            str(repo_root),
            "--output-format",
            "json",
            "--output",
            str(output_path),
            "--summary",
            "none",
            "--count-errors=0",
        ]
        try:
            result = await self.runner.run_async(
                ToolName.PYREFLY,
                args,
                cwd=repo_root,
                output_path=output_path,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError:
            log.warning("pyrefly binary not found; treating all files as 0 errors")
            output_path.unlink(missing_ok=True)
            return {}

        if result.returncode != 0 and not output_path.is_file():
            log.warning(
                "pyrefly exited with code %s and produced no output; stdout=%s stderr=%s",
                result.returncode,
                result.stdout.strip(),
                result.stderr.strip(),
            )
            output_path.unlink(missing_ok=True)
            return {}

        payload = ToolRunner.load_json(output_path) or {}
        output_path.unlink(missing_ok=True)
        return _parse_pyrefly_errors(payload, repo_root)

    async def run_ruff(self, repo_root: Path) -> Mapping[str, int]:
        """Run ruff and return lint error counts keyed by repo-relative path."""
        try:
            result = await self.runner.run_async(
                ToolName.RUFF,
                ["check", str(repo_root), "--output-format", "json"],
                cwd=repo_root,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError:
            log.warning("ruff binary not found; treating all files as 0 errors")
            return {}

        if result.returncode not in (0, 1):
            raise ToolExecutionError(result)
        return _parse_ruff_errors(result.stdout, repo_root)

    async def run_coverage_json(
        self,
        repo_root: Path,
        *,
        coverage_file: Path | None = None,
        output_path: Path | None = None,
    ) -> list[CoverageFileReport]:
        """Run coverage json export and return normalized file reports."""
        target_output = output_path or (self.runner.cache_dir / "coverage.json")
        target_output.parent.mkdir(parents=True, exist_ok=True)
        args = ["json", "--quiet", f"--output={target_output}"]
        data_file = coverage_file or self.tools_config.coverage_file
        if data_file is not None:
            args.append(f"--data-file={data_file}")
        result = await self.runner.run_async(
            ToolName.COVERAGE,
            args,
            cwd=repo_root,
            output_path=target_output,
            timeout_s=self.tools_config.default_timeout_s,
        )
        if not result.ok or not target_output.is_file():
            raise ToolExecutionError(result)
        payload = ToolRunner.load_json(target_output) or {}
        target_output.unlink(missing_ok=True)
        return _parse_coverage_payload(payload, repo_root)

    async def run_pytest_report(
        self,
        repo_root: Path,
        *,
        json_report_path: Path,
    ) -> bool:
        """Generate a pytest JSON report when missing. Returns True when executed."""
        if json_report_path.is_file():
            return False

        json_report_path.parent.mkdir(parents=True, exist_ok=True)
        result = await self.runner.run_async(
            ToolName.PYTEST,
            [
                "--json-report",
                f"--json-report-file={json_report_path}",
            ],
            cwd=repo_root,
            output_path=json_report_path,
            timeout_s=self.tools_config.default_timeout_s,
        )
        if not result.ok:
            raise ToolExecutionError(result)
        if not json_report_path.is_file():
            raise ToolExecutionError(result)
        return True

    async def run_scip_full(
        self,
        repo_root: Path,
        *,
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
    ) -> None:
        """Run scip-python for a full index and export to JSON."""
        await self._run_scip_python(
            repo_root,
            output_scip=output_scip,
            target_dir=target_dir,
            target_only=None,
        )
        await self._run_scip_print(output_scip, output_json)

    async def run_scip_shard(
        self,
        repo_root: Path,
        *,
        rel_paths: Sequence[str],
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
    ) -> None:
        """Run scip-python for a subset of files and export to JSON."""
        await self._run_scip_python(
            repo_root,
            output_scip=output_scip,
            target_dir=target_dir,
            target_only=rel_paths,
        )
        await self._run_scip_print(output_scip, output_json)

    async def _run_scip_python(
        self,
        repo_root: Path,
        *,
        output_scip: Path,
        target_dir: Path | None,
        target_only: Sequence[str] | None,
    ) -> None:
        target_base = target_dir or (
            repo_root / "src" if (repo_root / "src").is_dir() else repo_root
        )
        output_scip.parent.mkdir(parents=True, exist_ok=True)
        args: list[str] = ["index", str(target_base), "--output", str(output_scip)]
        for rel_path in target_only or ():
            args.extend(["--target-only", rel_path])
        result = await self.runner.run_async(
            ToolName.SCIP_PYTHON,
            args,
            cwd=repo_root,
            output_path=output_scip,
            timeout_s=self.tools_config.default_timeout_s,
        )
        if not result.ok:
            raise ToolExecutionError(result)

    async def _run_scip_print(self, scip_path: Path, output_json: Path) -> None:
        args = ["print", "--json", str(scip_path)]
        output_json.parent.mkdir(parents=True, exist_ok=True)
        result = await self.runner.run_async(
            ToolName.SCIP,
            args,
            cwd=scip_path.parent,
            output_path=output_json,
            timeout_s=self.tools_config.default_timeout_s,
        )
        if not result.ok:
            raise ToolExecutionError(result)
        output_json.write_text(result.stdout or "", encoding="utf8")


def _parse_pyrefly_errors(payload: Mapping[str, Any], repo_root: Path) -> dict[str, int]:
    errors_field = payload.get("errors") if isinstance(payload, Mapping) else None
    errors: Iterable[Mapping[str, Any]] = errors_field if isinstance(errors_field, list) else []
    counts: dict[str, int] = {}
    for diag in errors:
        if diag.get("severity") != "error":
            continue
        file_name = diag.get("path")
        if not file_name:
            continue
        rel_path = _safe_relpath(repo_root, Path(str(file_name)))
        if rel_path is None:
            continue
        counts[rel_path] = counts.get(rel_path, 0) + 1
    return counts


def _parse_pyright_errors(stdout: str, repo_root: Path) -> dict[str, int]:
    try:
        payload = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError as exc:
        log.warning("Failed to parse pyright JSON output: %s", exc)
        return {}

    diagnostics = payload.get("generalDiagnostics") if isinstance(payload, dict) else None
    if not isinstance(diagnostics, list):
        return {}

    counts: dict[str, int] = {}
    for diag in diagnostics:
        if not isinstance(diag, Mapping):
            continue
        if diag.get("severity") != "error":
            continue
        file_name = diag.get("file")
        if not file_name:
            continue
        rel_path = _safe_relpath(repo_root, Path(str(file_name)))
        if rel_path is None:
            continue
        counts[rel_path] = counts.get(rel_path, 0) + 1
    return counts


def _parse_ruff_errors(stdout: str, repo_root: Path) -> dict[str, int]:
    try:
        payload = json.loads(stdout) if stdout else []
    except json.JSONDecodeError as exc:
        log.warning("Failed to parse ruff JSON output: %s", exc)
        return {}
    if not isinstance(payload, list):
        return {}

    counts: dict[str, int] = {}
    for diag in payload:
        if not isinstance(diag, Mapping):
            continue
        file_name = diag.get("filename")
        if not file_name:
            continue
        rel_path = _safe_relpath(repo_root, Path(str(file_name)))
        if rel_path is None:
            continue
        counts[rel_path] = counts.get(rel_path, 0) + 1
    return counts


def _parse_coverage_payload(
    payload: Mapping[str, Any],
    repo_root: Path,
) -> list[CoverageFileReport]:
    files = payload.get("files") if isinstance(payload, Mapping) else None
    if not isinstance(files, Mapping):
        return []

    reports: list[CoverageFileReport] = []
    for file_name, data in files.items():
        if not isinstance(data, Mapping):
            continue
        executed = {int(line) for line in data.get("executed_lines", []) if isinstance(line, int)}
        missing = {int(line) for line in data.get("missing_lines", []) if isinstance(line, int)}
        rel_path = _safe_relpath(repo_root, Path(str(file_name)))
        if rel_path is None:
            continue
        reports.append(
            CoverageFileReport(
                rel_path=rel_path,
                executed_lines=executed,
                missing_lines=missing,
            )
        )
    return reports


def _safe_relpath(repo_root: Path, file_path: Path) -> str | None:
    try:
        return normalize_rel_path(repo_relpath(repo_root, file_path))
    except ValueError:
        return None
