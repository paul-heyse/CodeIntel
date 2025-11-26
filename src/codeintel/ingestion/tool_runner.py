"""Structured runners for external tools with caching and typed results."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from asyncio.subprocess import PIPE
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from codeintel.config.models import ToolsConfig

log = logging.getLogger(__name__)


class ToolName(StrEnum):
    """Supported external tools invoked by the ingestion pipeline."""

    PYRIGHT = "pyright"
    PYREFLY = "pyrefly"
    COVERAGE = "coverage"
    GIT = "git"
    RUFF = "ruff"
    PYTEST = "pytest"
    SCIP_PYTHON = "scip-python"
    SCIP = "scip"


@dataclass(frozen=True)
class ToolRunResult:
    """Structured output from a tool invocation."""

    tool: ToolName
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    output_path: Path | None = None

    @property
    def ok(self) -> bool:
        """Return True when the tool completed successfully."""
        return self.returncode == 0


class ToolNotFoundError(RuntimeError):
    """Raised when a configured tool cannot be resolved on the host."""

    def __init__(self, tool: ToolName, configured_path: str) -> None:
        message = f"Tool {tool.value} not found (configured as {configured_path!r})"
        super().__init__(message)
        self.tool = tool
        self.configured_path = configured_path


class ToolExecutionError(RuntimeError):
    """Raised when a tool invocation fails irrecoverably (e.g., timeout)."""

    def __init__(self, result: ToolRunResult) -> None:
        message = (
            f"Tool {result.tool.value} failed (code={result.returncode})\n"
            f"Args: {result.args}\n"
            f"stderr: {result.stderr.strip()}"
        )
        super().__init__(message)
        self.result = result


class ToolRunner:
    """Run external tools with optional caching and environment overrides."""

    def __init__(
        self,
        *,
        tools_config: ToolsConfig | None = None,
        cache_dir: Path | None = None,
        base_env: Mapping[str, str] | None = None,
    ) -> None:
        self.tools_config = tools_config or ToolsConfig.model_validate({})
        self.cache_dir = (cache_dir or Path("build") / ".tool_cache").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_env = dict(base_env or {})

    @staticmethod
    def _coerce_tool(tool: ToolName | str) -> ToolName:
        if isinstance(tool, ToolName):
            return tool
        try:
            return ToolName(tool)
        except ValueError as exc:
            message = f"Unknown tool {tool!r}"
            raise ValueError(message) from exc

    def _resolve_executable(self, tool: ToolName) -> str:
        configured = self.tools_config.resolve_path(tool)
        candidate_path = Path(configured)
        if candidate_path.is_file():
            return str(candidate_path)
        discovered = shutil.which(configured)
        if discovered is None:
            raise ToolNotFoundError(tool, configured)
        return discovered

    def _build_command(
        self,
        tool: ToolName,
        args: Sequence[str],
        *,
        executable: str | None = None,
    ) -> list[str]:
        resolved = executable or self._resolve_executable(tool)
        if args and args[0] in {tool.value, resolved, self.tools_config.resolve_path(tool)}:
            cmd_args = list(args[1:])
        else:
            cmd_args = list(args)
        return [resolved, *cmd_args]

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
        Execute a tool asynchronously and capture stdout/stderr.

        Parameters
        ----------
        tool
            Tool identifier to invoke.
        args
            Argument vector (with or without the executable name).
        cwd
            Optional working directory.
        output_path
            Optional path expected to be written by the tool.
        timeout_s
            Optional timeout in seconds.

        Returns
        -------
        ToolRunResult
            Structured process result including stdout, stderr, and exit code.

        Raises
        ------
        ToolNotFoundError
            When the configured tool executable cannot be located.
        ToolExecutionError
            When the subprocess fails unexpectedly (for example, due to timeout).
        """
        tool_enum = self._coerce_tool(tool)
        try:
            executable = self._resolve_executable(tool_enum)
        except ToolNotFoundError as exc:
            raise ToolNotFoundError(exc.tool, exc.configured_path) from exc
        cmd = self._build_command(tool_enum, args, executable=executable)
        env = self.tools_config.build_env(tool_enum, base_env=self.base_env)
        start_ts = time.perf_counter()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd) if cwd is not None else None,
            stdout=PIPE,
            stderr=PIPE,
            env=env if env else None,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            duration = time.perf_counter() - start_ts
            result = ToolRunResult(
                tool=tool_enum,
                args=tuple(cmd[1:]),
                returncode=proc.returncode or 1,
                stdout="",
                stderr="timed out",
                duration_s=duration,
                output_path=output_path,
            )
            raise ToolExecutionError(result) from exc

        duration = time.perf_counter() - start_ts
        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        return ToolRunResult(
            tool=tool_enum,
            args=tuple(cmd[1:]),
            returncode=proc.returncode if proc.returncode is not None else 1,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
            output_path=output_path,
        )

    def run(
        self,
        tool: ToolName | str,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        output_path: Path | None = None,
        timeout_s: float | None = None,
    ) -> ToolRunResult:
        """
        Execute a tool synchronously.

        Returns
        -------
        ToolRunResult
            Structured result from :meth:`run_async`.
        """
        return asyncio.run(
            self.run_async(
                tool,
                args,
                cwd=cwd,
                output_path=output_path,
                timeout_s=timeout_s,
            )
        )

    @staticmethod
    def load_json(path: Path) -> dict[str, object] | None:
        """
        Load JSON from a path if it exists and is non-empty.

        Returns
        -------
        dict[str, object] | None
            Parsed JSON payload as a mapping, or None when missing/empty.
        """
        if not path.is_file() or path.stat().st_size == 0:
            return None
        return json.loads(path.read_text(encoding="utf8"))


# Backwards compatibility alias for older tests/imports.
ToolResult = ToolRunResult
