"""Coverage plugin for the ingestion tool runtime."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anyio import to_thread

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.paths import normalize_rel_path, repo_relpath
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
)
from codeintel.ingestion.tools.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)
from codeintel.ingestion.tools.results import CoverageReport

log = logging.getLogger(__name__)


def _safe_relpath(repo_root: Path, file_path: Path) -> str | None:
    """
    Safely compute repository-relative path.

    Parameters
    ----------
    repo_root
        Repository root path.
    file_path
        Absolute or relative file path.

    Returns
    -------
    str | None
        Normalized relative path or None on failure.
    """
    try:
        candidate = file_path if file_path.is_absolute() else repo_root / file_path
        return normalize_rel_path(repo_relpath(repo_root, candidate))
    except ValueError:
        return None


def _parse_coverage_json(
    payload: Mapping[str, Any],
    repo_root: Path,
    json_path: Path | None = None,
) -> CoverageReport:
    """
    Parse coverage.py JSON output into a CoverageReport.

    Parameters
    ----------
    payload
        Parsed JSON from coverage json output.
    repo_root
        Repository root for path normalization.
    json_path
        Path to the JSON file for reference.

    Returns
    -------
    CoverageReport
        Parsed coverage data per file.
    """
    files_data = payload.get("files") if isinstance(payload, Mapping) else None
    if not isinstance(files_data, Mapping):
        return CoverageReport.empty()

    reports: list[tuple[str, set[int], set[int]]] = []

    for file_name, data in files_data.items():
        if not isinstance(data, Mapping):
            continue

        executed = {
            int(line)
            for line in data.get("executed_lines", [])
            if isinstance(line, int)
        }
        missing = {
            int(line)
            for line in data.get("missing_lines", [])
            if isinstance(line, int)
        }

        rel_path = _safe_relpath(repo_root, Path(str(file_name)))
        if rel_path is None:
            continue

        reports.append((rel_path, executed, missing))

    return CoverageReport.from_file_reports(reports, json_path=json_path)


@dataclass
class CoveragePlugin(ToolPlugin):
    """Plugin for running coverage json and parsing the output."""

    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = field(
        default_factory=lambda: ToolPluginMetadata(
            name="coverage",
            produces_artifacts=("coverage_json",),
            consumes_configs=("coverage_bin",),
            datasets=("analytics.coverage_lines",),
        )
    )

    async def run(
        self,
        *,
        repo_root: Path,
        **kwargs: object,
    ) -> ToolPluginResult:
        """
        Run coverage CLI and return parsed coverage report.

        Parameters
        ----------
        repo_root
            Repository root passed to the CLI via cwd.
        **kwargs
            Expected keys: coverage_file (Path | None), output_path (Path).

        Returns
        -------
        ToolPluginResult
            Normalized execution result with parsed CoverageReport.

        Raises
        ------
        TypeError
            Raised when required keyword arguments are missing or of wrong type.
        """
        coverage_file_obj = kwargs.get("coverage_file")
        output_path_obj = kwargs.get("output_path")

        if not isinstance(output_path_obj, Path):
            message = "coverage plugin requires an output_path of type Path"
            raise TypeError(message)
        if coverage_file_obj is not None and not isinstance(coverage_file_obj, Path):
            message = "coverage plugin requires coverage_file to be Path or None"
            raise TypeError(message)

        output_path = output_path_obj
        coverage_file = coverage_file_obj
        await to_thread.run_sync(lambda: output_path.parent.mkdir(parents=True, exist_ok=True))

        args = ["json", "--quiet", "-o", str(output_path)]
        if coverage_file is not None:
            args.append(f"--data-file={coverage_file}")

        try:
            result = await self.runner.run_async(
                ToolName.COVERAGE,
                args,
                cwd=repo_root,
                output_path=output_path,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError as exc:
            log.warning("coverage binary not found; skipping coverage ingestion")
            return ToolPluginResult(
                tool=ToolName.COVERAGE,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
                parsed=CoverageReport.empty(),
            )
        except ToolExecutionError as exc:
            return ToolPluginResult(
                tool=ToolName.COVERAGE,
                status=ToolStatus.ERROR,
                artifacts={"coverage_json": output_path},
                run=exc.result,
                error=exc,
                parsed=CoverageReport.empty(),
            )

        # Parse the JSON output file
        parsed = CoverageReport.empty()

        def _is_file() -> bool:
            return output_path.is_file()

        output_exists = await to_thread.run_sync(_is_file)
        if output_exists:

            def _load_and_parse() -> CoverageReport:
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                    return _parse_coverage_json(payload, repo_root, output_path)
                except (OSError, json.JSONDecodeError) as exc:
                    log.warning("Failed to parse coverage JSON: %s", exc)
                    return CoverageReport.empty()

            parsed = await to_thread.run_sync(_load_and_parse)

        status = ToolStatus.OK if result.ok else ToolStatus.ERROR
        artifacts = {"coverage_json": output_path}

        return ToolPluginResult(
            tool=result.tool,
            status=status,
            artifacts=artifacts,
            run=result,
            error=None if status is ToolStatus.OK else ToolExecutionError(result),
            parsed=parsed,
        )
