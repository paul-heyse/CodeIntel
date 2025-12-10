"""Fake tool runner and service implementations for testing.

This module provides fake implementations of ToolRunner and ToolService
for tests that need deterministic tool behavior without running real tools.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anyio import to_thread

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
    ToolRunResult,
)
from codeintel.ingestion.engine.results import CoverageReport, ScipIndexResult
from codeintel.ingestion.engine.service import ToolService
from tests._helpers.records import CallRecorder, ToolRunCall


def _mkdir_parents(path: Path) -> None:
    """Create parent directories for a path."""
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, payload: str) -> None:
    """Write text content to a file."""
    path.write_text(payload, encoding="utf8")


class FakeToolRunner(ToolRunner):
    """ToolRunner stub returning canned payloads."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        payloads: dict[str, Any] | None = None,
        on_run: Callable[[ToolName, list[str]], None] | None = None,
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        self.payloads = payloads or {}
        self.calls: CallRecorder[ToolRunCall] = CallRecorder()
        self.on_run = on_run

    async def run_async(
        self,
        tool: ToolName | str,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        output_path: Path | None = None,
        timeout_s: float | None = None,
    ) -> ToolRunResult:
        """
        Execute a tool invocation with canned outputs.

        Parameters
        ----------
        tool
            Tool name or enum.
        args
            Arguments to pass to the tool.
        cwd
            Working directory (ignored).
        output_path
            Optional output path for results.
        timeout_s
            Timeout in seconds (ignored).

        Returns
        -------
        ToolRunResult
            Structured result capturing stdout/stderr and codes.
        """
        _ = cwd
        _ = timeout_s
        tool_enum = tool if isinstance(tool, ToolName) else ToolName(str(tool))
        args_list = list(args)
        self.calls.record(
            ToolRunCall(
                tool=tool_enum.value,
                args=args_list,
                cwd=cwd or self.cache_dir,
                timeout_ms=None if timeout_s is None else int(timeout_s * 1000),
                env=None,
            )
        )
        if self.on_run is not None:
            self.on_run(tool_enum, args_list)
        payload_stdout = self.payloads.get(tool_enum.value, "")
        if output_path is not None and tool_enum in {ToolName.COVERAGE, ToolName.PYREFLY}:
            json_payload = self.payloads.get(
                f"{tool_enum.value}_json",
                self.payloads.get("json", {}),
            )
            await to_thread.run_sync(_mkdir_parents, output_path.parent)
            await to_thread.run_sync(_write_text, output_path, json.dumps(json_payload))
        return ToolRunResult(
            tool=tool_enum,
            args=tuple(args_list),
            returncode=0,
            stdout=str(payload_stdout),
            stderr="",
            output_path=output_path,
            duration_s=0.0,
        )


@dataclass(frozen=True)
class FakeScipResult:
    """SCIP result stand-in mirroring dataclass fields."""

    status: str = "success"
    index_scip: Path | None = None
    index_json: Path | None = None
    reason: str | None = None


def write_dummy_scip_files(base_dir: Path, *, index_content: str = "[]") -> tuple[Path, Path]:
    """
    Create minimal SCIP artifacts for tests.

    Parameters
    ----------
    base_dir
        Base directory for SCIP files.
    index_content
        Content for the JSON index file.

    Returns
    -------
    tuple[Path, Path]
        Paths to index.scip and index.scip.json.
    """
    scip_dir = base_dir / "scip"
    scip_dir.mkdir(parents=True, exist_ok=True)
    index_scip = scip_dir / "index.scip"
    index_json = scip_dir / "index.scip.json"
    index_scip.write_text("scip-binary", encoding="utf8")
    index_json.write_text(index_content, encoding="utf8")
    return index_scip, index_json


class PresetRunner(ToolRunner):
    """ToolRunner that returns preset results without invoking subprocesses."""

    def __init__(self, result: ToolRunResult | Exception) -> None:
        """Initialize with preset result."""
        self._result = result
        super().__init__(tools_config=ToolsConfig.default(), cache_dir=Path("build/.tool_cache"))

    async def run_async(
        self,
        tool: ToolName | str,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        output_path: Path | None = None,
        timeout_s: float | None = None,
    ) -> ToolRunResult:
        """Return a preset ToolRunResult or raise the configured exception.

        Returns
        -------
        ToolRunResult
            The configured result when no error is raised.

        Raises
        ------
        ToolNotFoundError
            When initialized with a ToolNotFoundError.
        ToolExecutionError
            When initialized with a generic exception.
        """
        del tool, args, cwd, timeout_s
        if isinstance(self._result, ToolNotFoundError):
            raise ToolNotFoundError(self._result.tool, self._result.configured_path)
        if isinstance(self._result, Exception):
            raise ToolExecutionError(
                make_tool_run_result(
                    ToolName.PYRIGHT,
                    options=ToolRunOptions(
                        returncode=1,
                        stderr="dummy error",
                        output_path=output_path,
                        duration_s=0.1,
                    ),
                )
            ) from self._result
        return self._result


@dataclass(frozen=True)
class ToolRunOptions:
    """Configuration for a fake ToolRunResult."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    output_path: Path | None = None
    duration_s: float = 0.0


def make_tool_run_result(
    tool: ToolName | str,
    *,
    args: Sequence[str] | None = None,
    options: ToolRunOptions | None = None,
) -> ToolRunResult:
    """Build a ToolRunResult with sensible defaults for tests.

    Returns
    -------
    ToolRunResult
        Structured result populated with the provided options.
    """
    tool_enum = tool if isinstance(tool, ToolName) else ToolName(str(tool))
    opts = options or ToolRunOptions()
    return ToolRunResult(
        tool=tool_enum,
        args=tuple(args or ()),
        returncode=opts.returncode,
        stdout=opts.stdout,
        stderr=opts.stderr,
        output_path=opts.output_path,
        duration_s=opts.duration_s,
    )


def make_scip_index_result(base_dir: Path, *, index_content: str = "[]") -> ScipIndexResult:
    """Create SCIP artifacts and a corresponding ScipIndexResult.

    Returns
    -------
    ScipIndexResult
        Result pointing to the generated SCIP artifacts.
    """
    index_scip, index_json = write_dummy_scip_files(base_dir, index_content=index_content)
    return ScipIndexResult.from_json_documents(
        [],
        index_scip_path=index_scip,
        index_json_path=index_json,
    )


@dataclass
class FakeToolServiceConfig:
    """Configuration for FakeToolService behavior.

    Attributes
    ----------
    pyright_errors : dict[str, int]
        Mapping of file paths to error counts for pyright.
    pyrefly_errors : dict[str, int]
        Mapping of file paths to error counts for pyrefly.
    ruff_errors : dict[str, int]
        Mapping of file paths to error counts for ruff.
    coverage_report : CoverageReport | None
        Coverage report to return, or None for empty.
    scip_result : ScipIndexResult | None
        SCIP result to return, or None for empty.
    pytest_success : bool
        Whether pytest should succeed.
    raise_on_pyright : Exception | None
        Exception to raise on pyright calls.
    raise_on_pyrefly : Exception | None
        Exception to raise on pyrefly calls.
    raise_on_ruff : Exception | None
        Exception to raise on ruff calls.
    raise_on_coverage : Exception | None
        Exception to raise on coverage calls.
    raise_on_scip : Exception | None
        Exception to raise on scip calls.
    raise_on_pytest : Exception | None
        Exception to raise on pytest calls.
    """

    pyright_errors: dict[str, int] = field(default_factory=dict)
    pyrefly_errors: dict[str, int] = field(default_factory=dict)
    ruff_errors: dict[str, int] = field(default_factory=dict)
    coverage_report: CoverageReport | None = None
    scip_result: ScipIndexResult | None = None
    pytest_success: bool = True
    raise_on_pyright: Exception | None = None
    raise_on_pyrefly: Exception | None = None
    raise_on_ruff: Exception | None = None
    raise_on_coverage: Exception | None = None
    raise_on_scip: Exception | None = None
    raise_on_pytest: Exception | None = None


class FakeToolService(ToolService):
    """ToolService subclass with deterministic, configurable results.

    This fake extends the real ToolService with configurable responses,
    enabling tests to verify tool integration behavior without running real
    external tools. It inherits from ToolService for full type compatibility.

    Parameters
    ----------
    config : FakeToolServiceConfig | None
        Configuration for fake behavior. Defaults to empty/success responses.
    cache_dir : Path | None
        Cache directory for the underlying FakeToolRunner.

    Attributes
    ----------
    fake_config : FakeToolServiceConfig
        Current configuration.
    calls : CallRecorder[ToolRunCall]
        Log of method calls for verification.
    """

    def __init__(
        self, config: FakeToolServiceConfig | None = None, cache_dir: Path | None = None
    ) -> None:
        """Initialize with optional configuration."""
        effective_cache = cache_dir or Path(tempfile.gettempdir()) / "fake_tool_cache"
        fake_runner = FakeToolRunner(cache_dir=effective_cache)
        super().__init__(fake_runner)
        self.fake_config = config or FakeToolServiceConfig()
        self.calls: CallRecorder[ToolRunCall] = CallRecorder()

    async def run_pyright(self, repo_root: Path) -> dict[str, int]:
        """Run pyright and return configured error counts.

        Parameters
        ----------
        repo_root
            Repository root (logged but not used).

        Returns
        -------
        dict[str, int]
            Configured pyright errors.
        """
        self.calls.record(
            ToolRunCall(tool="pyright", args=[], cwd=repo_root, timeout_ms=None, env=None)
        )
        if self.fake_config.raise_on_pyright is not None:
            raise self.fake_config.raise_on_pyright
        return dict(self.fake_config.pyright_errors)

    async def run_pyrefly(self, repo_root: Path) -> dict[str, int]:
        """Run pyrefly and return configured error counts.

        Parameters
        ----------
        repo_root
            Repository root (logged but not used).

        Returns
        -------
        dict[str, int]
            Configured pyrefly errors.
        """
        self.calls.record(
            ToolRunCall(tool="pyrefly", args=[], cwd=repo_root, timeout_ms=None, env=None)
        )
        if self.fake_config.raise_on_pyrefly is not None:
            raise self.fake_config.raise_on_pyrefly
        return dict(self.fake_config.pyrefly_errors)

    async def run_ruff(self, repo_root: Path) -> dict[str, int]:
        """Run ruff and return configured error counts.

        Parameters
        ----------
        repo_root
            Repository root (logged but not used).

        Returns
        -------
        dict[str, int]
            Configured ruff errors.
        """
        self.calls.record(
            ToolRunCall(tool="ruff", args=[], cwd=repo_root, timeout_ms=None, env=None)
        )
        if self.fake_config.raise_on_ruff is not None:
            raise self.fake_config.raise_on_ruff
        return dict(self.fake_config.ruff_errors)

    async def run_coverage_report(
        self,
        repo_root: Path,
        *,
        coverage_file: Path | None = None,
        output_path: Path | None = None,
    ) -> CoverageReport:
        """Run coverage and return configured report.

        Parameters
        ----------
        repo_root
            Repository root (logged but not used).
        coverage_file
            Coverage file path (logged but not used).
        output_path
            Output path (logged but not used).

        Returns
        -------
        CoverageReport
            Configured coverage report.
        """
        args: list[str] = []
        if coverage_file is not None:
            args.append(str(coverage_file))
        if output_path is not None:
            args.append(str(output_path))
        self.calls.record(
            ToolRunCall(
                tool="coverage",
                args=args,
                cwd=repo_root,
                timeout_ms=None,
                env=None,
            )
        )
        if self.fake_config.raise_on_coverage is not None:
            raise self.fake_config.raise_on_coverage
        return self.fake_config.coverage_report or CoverageReport.empty()

    async def run_scip_full(
        self,
        repo_root: Path,
        *,
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
    ) -> ScipIndexResult:
        """Run full SCIP indexing and return configured result.

        Parameters
        ----------
        repo_root
            Repository root (logged but not used).
        output_scip
            Output SCIP file path (logged but not used).
        output_json
            Output JSON file path (logged but not used).
        target_dir
            Target directory (logged but not used).

        Returns
        -------
        ScipIndexResult
            Configured SCIP result.
        """
        args = [str(output_scip), str(output_json)]
        if target_dir is not None:
            args.append(str(target_dir))
        self.calls.record(
            ToolRunCall(
                tool="scip_full",
                args=args,
                cwd=repo_root,
                timeout_ms=None,
                env=None,
            )
        )
        if self.fake_config.raise_on_scip is not None:
            raise self.fake_config.raise_on_scip
        return self.fake_config.scip_result or ScipIndexResult.empty()

    async def run_scip_shard(
        self,
        repo_root: Path,
        *,
        rel_paths: list[str],
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
    ) -> ScipIndexResult:
        """Run SCIP indexing for a shard and return configured result.

        Parameters
        ----------
        repo_root
            Repository root (logged but not used).
        rel_paths
            Relative paths to index (logged but not used).
        output_scip
            Output SCIP file path (logged but not used).
        output_json
            Output JSON file path (logged but not used).
        target_dir
            Target directory (logged but not used).

        Returns
        -------
        ScipIndexResult
            Configured SCIP result.
        """
        args = [str(output_scip), str(output_json), *rel_paths]
        if target_dir is not None:
            args.append(str(target_dir))
        self.calls.record(
            ToolRunCall(
                tool="scip_shard",
                args=args,
                cwd=repo_root,
                timeout_ms=None,
                env=None,
            )
        )
        if self.fake_config.raise_on_scip is not None:
            raise self.fake_config.raise_on_scip
        return self.fake_config.scip_result or ScipIndexResult.empty()

    async def run_pytest_report(
        self,
        repo_root: Path,
        *,
        json_report_path: Path,
    ) -> bool:
        """Run pytest and return configured success.

        Parameters
        ----------
        repo_root
            Repository root (logged but not used).
        json_report_path
            JSON report path (logged but not used).

        Returns
        -------
        bool
            Configured pytest success status.
        """
        self.calls.record(
            ToolRunCall(
                tool="pytest_report",
                args=[str(json_report_path)],
                cwd=repo_root,
                timeout_ms=None,
                env=None,
            )
        )
        if self.fake_config.raise_on_pytest is not None:
            raise self.fake_config.raise_on_pytest
        return self.fake_config.pytest_success


def make_success_tool_service(
    *,
    coverage_report: CoverageReport | None = None,
) -> FakeToolService:
    """Create a FakeToolService configured for successful tool runs.

    Returns
    -------
    FakeToolService
        Service with deterministic success responses.
    """
    return FakeToolService(
        FakeToolServiceConfig(
            pyright_errors={"mod.py": 2, "other.py": 0},
            pyrefly_errors={"mod.py": 1},
            ruff_errors={"style.py": 3},
            coverage_report=coverage_report
            or CoverageReport.from_file_reports(
                [
                    ("mod.py", {1, 2, 3}, {4, 5}),
                ]
            ),
            pytest_success=True,
        )
    )


def make_failing_tool_service() -> FakeToolService:
    """Create a FakeToolService configured to raise errors on all tools.

    Returns
    -------
    FakeToolService
        Service configured to raise exceptions for every tool invocation.
    """
    config = FakeToolServiceConfig(
        raise_on_pyright=RuntimeError("pyright failed"),
        raise_on_pyrefly=RuntimeError("pyrefly failed"),
        raise_on_ruff=OSError("ruff failed"),
        raise_on_coverage=ValueError("coverage failed"),
        raise_on_scip=RuntimeError("SCIP failed"),
        raise_on_pytest=RuntimeError("pytest failed"),
    )
    return FakeToolService(config)


__all__ = [
    "FakeScipResult",
    "FakeToolRunner",
    "FakeToolService",
    "FakeToolServiceConfig",
    "PresetRunner",
    "ToolRunOptions",
    "make_failing_tool_service",
    "make_scip_index_result",
    "make_success_tool_service",
    "make_tool_run_result",
    "write_dummy_scip_files",
]
