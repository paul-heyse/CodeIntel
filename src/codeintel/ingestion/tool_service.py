"""High-level façade around ToolRunner for external CLI integrations."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anyio import to_thread

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
)
from codeintel.utils.paths import normalize_rel_path, repo_relpath

log = logging.getLogger(__name__)


def _mkdir_parents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _unlink_missing(path: Path) -> None:
    path.unlink(missing_ok=True)


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf8")


def _path_is_file(path: Path) -> bool:
    return path.is_file()


def _resolve_target_base(repo_root: Path, target_dir: Path | None) -> Path:
    if target_dir is not None:
        return target_dir
    src_dir = repo_root / "src"
    return src_dir if src_dir.is_dir() else repo_root


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
        """
        Run pyright and return error counts keyed by repo-relative path.

        Parameters
        ----------
        repo_root
            Repository root supplied to the pyright invocation.

        Returns
        -------
        Mapping[str, int]
            Mapping from relative file paths to error counts.

        Raises
        ------
        ToolExecutionError
            Raised when pyright exits with an unexpected status.
        """
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

        if result.returncode not in {0, 1}:
            raise ToolExecutionError(result)
        return _parse_pyright_errors(result.stdout, repo_root)

    async def run_pyrefly(self, repo_root: Path) -> Mapping[str, int]:
        """
        Run pyrefly and return error counts keyed by repo-relative path.

        Parameters
        ----------
        repo_root
            Repository root supplied to the pyrefly invocation.

        Returns
        -------
        Mapping[str, int]
            Mapping from relative file paths to error counts.
        """
        output_path = self.runner.cache_dir / "pyrefly.json"
        await to_thread.run_sync(_mkdir_parents, output_path.parent)
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
            await to_thread.run_sync(_unlink_missing, output_path)
            return {}

        output_exists = await to_thread.run_sync(_path_is_file, output_path)
        if result.returncode != 0 and not output_exists:
            log.warning(
                "pyrefly exited with code %s and produced no output; stdout=%s stderr=%s",
                result.returncode,
                result.stdout.strip(),
                result.stderr.strip(),
            )
            await to_thread.run_sync(_unlink_missing, output_path)
            return {}

        payload = await to_thread.run_sync(ToolRunner.load_json, output_path) or {}
        await to_thread.run_sync(_unlink_missing, output_path)
        return _parse_pyrefly_errors(payload, repo_root)

    async def run_ruff(self, repo_root: Path) -> Mapping[str, int]:
        """
        Run ruff and return lint error counts keyed by repo-relative path.

        Parameters
        ----------
        repo_root
            Repository root supplied to the ruff invocation.

        Returns
        -------
        Mapping[str, int]
            Mapping from relative file paths to lint error counts.

        Raises
        ------
        ToolExecutionError
            Raised when ruff exits with an unexpected status.
        """
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

        if result.returncode not in {0, 1}:
            raise ToolExecutionError(result)
        return _parse_ruff_errors(result.stdout, repo_root)

    async def run_coverage_json(
        self,
        repo_root: Path,
        *,
        coverage_file: Path | None = None,
        output_path: Path | None = None,
    ) -> list[CoverageFileReport]:
        """
        Run coverage json export and return normalized file reports.

        Parameters
        ----------
        repo_root
            Repository root supplied to the coverage invocation.
        coverage_file
            Optional explicit .coverage path to read from.
        output_path
            Optional path where the JSON output should be written.

        Returns
        -------
        list[CoverageFileReport]
            Normalized coverage summaries grouped per file.

        Raises
        ------
        ToolExecutionError
            Raised when coverage CLI execution fails or produces no output.
        """
        target_output = output_path or (self.runner.cache_dir / "coverage.json")
        await to_thread.run_sync(_mkdir_parents, target_output.parent)
        args = ["json", "--quiet", "-o", str(target_output)]
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
        if not result.ok or not await to_thread.run_sync(_path_is_file, target_output):
            raise ToolExecutionError(result)
        payload = await to_thread.run_sync(ToolRunner.load_json, target_output) or {}
        await to_thread.run_sync(_unlink_missing, target_output)
        return _parse_coverage_payload(payload, repo_root)

    async def run_pytest_report(
        self,
        repo_root: Path,
        *,
        json_report_path: Path,
    ) -> bool:
        """
        Generate a pytest JSON report when missing.

        Parameters
        ----------
        repo_root
            Repository root passed to the pytest invocation.
        json_report_path
            Output path for the pytest JSON report.

        Returns
        -------
        bool
            True when pytest was executed to produce the report, False when reused.

        Raises
        ------
        ToolExecutionError
            Raised when pytest execution fails or does not create a report.
        """
        if await to_thread.run_sync(_path_is_file, json_report_path):
            return False

        await to_thread.run_sync(_mkdir_parents, json_report_path.parent)
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
        if not await to_thread.run_sync(_path_is_file, json_report_path):
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
        target_base = await to_thread.run_sync(_resolve_target_base, repo_root, target_dir)
        await to_thread.run_sync(_mkdir_parents, output_scip.parent)
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
        await to_thread.run_sync(_mkdir_parents, output_json.parent)
        result = await self.runner.run_async(
            ToolName.SCIP,
            args,
            cwd=scip_path.parent,
            output_path=output_json,
            timeout_s=self.tools_config.default_timeout_s,
        )
        if not result.ok:
            raise ToolExecutionError(result)
        await to_thread.run_sync(_write_text, output_json, result.stdout or "")


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
        candidate = file_path if file_path.is_absolute() else repo_root / file_path
        return normalize_rel_path(repo_relpath(repo_root, candidate))
    except ValueError:
        return None
