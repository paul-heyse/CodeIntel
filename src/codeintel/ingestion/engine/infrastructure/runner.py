"""Structured runners for external tools with caching and typed results.

Note
----
This module provides domain-specific result types for tool execution.
For pipeline run and step tracking persistence, see `codeintel.storage.tracking`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import shutil
import time
from asyncio.subprocess import PIPE
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.models import ToolsConfig
from codeintel.core.tools import ToolName

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

log = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class ToolRunOptions:
    """Execution options for a tool invocation."""

    cwd: Path | None = None
    output_path: Path | None = None
    timeout_s: float | None = None
    env: Mapping[str, str] | None = None


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
        options: ToolRunOptions | None = None,
    ) -> ToolRunResult:
        """
        Execute a tool asynchronously and capture stdout/stderr.

        Parameters
        ----------
        tool
            Tool identifier to invoke.
        args
            Argument vector (with or without the executable name).
        options
            Execution options such as working directory, output path, timeout, and env overrides.

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
        run_options = options or ToolRunOptions()
        tool_enum = self._coerce_tool(tool)
        try:
            executable = self._resolve_executable(tool_enum)
        except ToolNotFoundError as exc:
            raise ToolNotFoundError(exc.tool, exc.configured_path) from exc
        cmd = self._build_command(tool_enum, args, executable=executable)
        tool_env = self.tools_config.build_env(tool_enum, base_env=self.base_env)
        if run_options.env:
            merged = dict(tool_env or {})
            merged.update(run_options.env)
            tool_env = merged
        start_ts = time.perf_counter()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(run_options.cwd) if run_options.cwd is not None else None,
            stdout=PIPE,
            stderr=PIPE,
            env=tool_env if tool_env else None,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(),
                timeout=run_options.timeout_s,
            )
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
                output_path=run_options.output_path,
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
            output_path=run_options.output_path,
        )

    def run(
        self,
        tool: ToolName | str,
        args: Sequence[str],
        *,
        options: ToolRunOptions | None = None,
    ) -> ToolRunResult:
        """
        Execute a tool synchronously.

        Returns
        -------
        ToolRunResult
            Structured result from :meth:`run_async`.
        """
        run_options = options or ToolRunOptions()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.run_async(
                        tool,
                        args,
                        options=run_options,
                    ),
                )
                return future.result()
        else:
            return asyncio.run(
                self.run_async(
                    tool,
                    args,
                    options=run_options,
                )
            )
