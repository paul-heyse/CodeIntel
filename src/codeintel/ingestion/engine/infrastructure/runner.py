"""Structured runners for external tools with caching and typed results.

Note
----
This module provides domain-specific result types for tool execution.
For pipeline run and step tracking persistence, see `codeintel.storage.tracking`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
import time
from asyncio.subprocess import PIPE
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.models import ToolsConfig
from codeintel.core.tools import ToolName
from codeintel.core.tools.resolver import ToolResolveConfig, resolve_tool
from codeintel.observability.runtime_registry import (
    mark_subprocess_exited,
    register_subprocess,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

log = logging.getLogger(__name__)
_TOOL_CALL_LOG_ENV = "CODEINTEL_TOOL_CALL_LOG"
_TOOL_CALL_LOG_LOCK = threading.Lock()


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
    progress_interval_s: float | None = None
    log_prefix: str | None = None
    stream_output: bool = False


@dataclass(frozen=True)
class ToolSpec:
    """Declarative contract for tool keyword arguments."""

    required_kwargs: tuple[str, ...] = ()
    optional_kwargs: tuple[str, ...] = ()
    allow_extra_kwargs: bool = False

    def validate_kwargs(
        self,
        kwargs: Mapping[str, object],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return missing and unexpected kwargs for this spec.

        Returns
        -------
        tuple[tuple[str, ...], tuple[str, ...]]
            Missing required kwargs and unexpected kwargs, respectively.
        """
        missing = tuple(name for name in self.required_kwargs if name not in kwargs)
        if self.allow_extra_kwargs:
            return missing, ()
        allowed = set(self.required_kwargs) | set(self.optional_kwargs)
        extra = tuple(sorted(name for name in kwargs if name not in allowed))
        return missing, extra


@dataclass(frozen=True)
class ToolCallRecord:
    """Captured context for a tool invocation."""

    tool: ToolName
    args: Sequence[str]
    options: ToolRunOptions
    env: Mapping[str, str] | None
    result: ToolRunResult
    started_at: datetime


class ToolNotFoundError(RuntimeError):
    """Raised when a configured tool cannot be resolved on the host."""

    def __init__(self, tool: ToolName, configured_path: str) -> None:
        message = f"Tool {tool.value} not found (configured as {configured_path!r})"
        super().__init__(message)
        self.tool = tool
        self.configured_path = configured_path


class ToolSpecError(ValueError):
    """Raised when tool invocation arguments violate a ToolSpec."""

    def __init__(
        self,
        tool_name: str,
        *,
        missing: tuple[str, ...],
        extra: tuple[str, ...],
    ) -> None:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"unexpected={sorted(extra)}")
        detail_text = ", ".join(details) if details else "invalid arguments"
        message = f"Tool {tool_name} invocation invalid: {detail_text}"
        super().__init__(message)
        self.tool_name = tool_name
        self.missing = missing
        self.extra = extra


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
        resolve_cfg = ToolResolveConfig.from_env()
        resolution = resolve_tool(tool, config=self.tools_config, resolve_cfg=resolve_cfg)
        if resolution.resolved is None:
            raise ToolNotFoundError(tool, resolution.configured)
        return str(resolution.resolved)

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
            cmd = self._build_command(
                tool_enum,
                args,
                executable=self._resolve_executable(tool_enum),
            )
        except ToolNotFoundError as exc:
            raise ToolNotFoundError(exc.tool, exc.configured_path) from exc
        tool_env = self.tools_config.build_env(tool_enum, base_env=self.base_env)
        if run_options.env:
            tool_env = {**(tool_env or {}), **run_options.env}
        started_at = datetime.now(UTC)
        start_ts = time.perf_counter()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(run_options.cwd) if run_options.cwd is not None else None,
            stdout=PIPE,
            stderr=PIPE,
            env=tool_env if tool_env else None,
        )
        register_subprocess(pid=proc.pid, command=Path(cmd[0]).name)
        log_prefix = run_options.log_prefix or tool_enum.value
        heartbeat_task: asyncio.Task[None] | None = None
        if run_options.progress_interval_s:
            heartbeat_task = asyncio.create_task(
                _tool_heartbeat(
                    proc,
                    started_at=start_ts,
                    interval_s=run_options.progress_interval_s,
                    log_prefix=log_prefix,
                    output_path=run_options.output_path,
                )
            )
        return_code: int | None = None
        try:
            if run_options.stream_output:
                stdout_b, stderr_b = await _stream_process(
                    proc,
                    timeout_s=run_options.timeout_s,
                    log_prefix=log_prefix,
                )
            else:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=run_options.timeout_s,
                )
        except TimeoutError as exc:
            proc.kill()
            with suppress(asyncio.CancelledError):
                await proc.communicate()
            return_code = proc.returncode or 1
            result = ToolRunResult(
                tool=tool_enum,
                args=tuple(cmd[1:]),
                returncode=return_code,
                stdout="",
                stderr="timed out",
                duration_s=time.perf_counter() - start_ts,
                output_path=run_options.output_path,
            )
            record = ToolCallRecord(
                tool=tool_enum,
                args=tuple(cmd[1:]),
                options=run_options,
                env=tool_env,
                result=result,
                started_at=started_at,
            )
            _record_tool_call(record)
            raise ToolExecutionError(result) from exc
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat_task
            if return_code is None:
                return_code = proc.returncode if proc.returncode is not None else 1
            mark_subprocess_exited(pid=proc.pid, exit_code=return_code)

        return_code = proc.returncode if proc.returncode is not None else 1
        result = ToolRunResult(
            tool=tool_enum,
            args=tuple(cmd[1:]),
            returncode=return_code,
            stdout=stdout_b.decode(errors="replace"),
            stderr=stderr_b.decode(errors="replace"),
            duration_s=time.perf_counter() - start_ts,
            output_path=run_options.output_path,
        )
        record = ToolCallRecord(
            tool=tool_enum,
            args=tuple(cmd[1:]),
            options=run_options,
            env=tool_env,
            result=result,
            started_at=started_at,
        )
        _record_tool_call(record)
        return result

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


async def _tool_heartbeat(
    proc: asyncio.subprocess.Process,
    *,
    started_at: float,
    interval_s: float,
    log_prefix: str,
    output_path: Path | None,
) -> None:
    while proc.returncode is None:
        await asyncio.sleep(interval_s)
        if proc.returncode is not None:
            return
        elapsed = time.perf_counter() - started_at
        output_bytes = None
        if output_path is not None:
            output_bytes = await asyncio.to_thread(_output_path_size, output_path)
        if output_bytes is None:
            log.info("%s still running (elapsed=%.1fs)", log_prefix, elapsed)
        else:
            log.info(
                "%s still running (elapsed=%.1fs, output_bytes=%d)",
                log_prefix,
                elapsed,
                output_bytes,
            )


async def _stream_process(
    proc: asyncio.subprocess.Process,
    *,
    timeout_s: float | None,
    log_prefix: str,
) -> tuple[bytes, bytes]:
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    async def _read_stream(
        stream: asyncio.StreamReader | None,
        *,
        label: str,
        chunks: list[bytes],
    ) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                return
            chunks.append(line)
            log.info("%s[%s] %s", log_prefix, label, line.decode(errors="replace").rstrip())

    stdout_task = asyncio.create_task(
        _read_stream(proc.stdout, label="stdout", chunks=stdout_chunks)
    )
    stderr_task = asyncio.create_task(
        _read_stream(proc.stderr, label="stderr", chunks=stderr_chunks)
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout_s)
    finally:
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    return b"".join(stdout_chunks), b"".join(stderr_chunks)


def _output_path_size(path: Path) -> int | None:
    if not path.exists():
        return None
    return path.stat().st_size


def _record_tool_call(record: ToolCallRecord) -> None:
    path = os.environ.get(_TOOL_CALL_LOG_ENV, "").strip()
    if not path:
        return
    output_size_bytes = None
    if record.result.output_path is not None and record.result.output_path.exists():
        output_size_bytes = record.result.output_path.stat().st_size
    payload = {
        "tool": record.tool.value,
        "argv": list(record.args),
        "cwd": str(record.options.cwd) if record.options.cwd is not None else None,
        "env_keys": sorted(record.env.keys()) if record.env else [],
        "started_at": record.started_at.isoformat(),
        "duration_s": record.result.duration_s,
        "returncode": record.result.returncode,
        "version": _extract_version(record.args, record.result.stdout),
        "output_size_bytes": output_size_bytes,
        "stdout_tail": _tail_text(record.result.stdout),
        "stderr_tail": _tail_text(record.result.stderr),
    }
    _append_json_line(Path(path), payload)


def _extract_version(args: Sequence[str], stdout: str) -> str | None:
    if "--version" in args or "-V" in args:
        return stdout.strip().splitlines()[0][:200] if stdout else ""
    return None


def _tail_text(text: str, *, max_chars: int = 2000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _append_json_line(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=True)
    with _TOOL_CALL_LOG_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
